# autonomous_agency/app/agents/prospect_researcher.py
import asyncio
import random
import json
import subprocess
import os
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
import datetime

from sqlalchemy.ext.asyncio import AsyncSession
import httpx # Use the client from agent_utils
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Corrected relative imports for package structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api
except ImportError:
    print("[ProspectResearcher] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MAX_PROSPECTS_PER_CYCLE = settings.MAX_PROSPECTS_PER_CYCLE or 20
OPEN_DEEP_RESEARCH_TIMEOUT = settings.OPEN_DEEP_RESEARCH_TIMEOUT or 300
EMAIL_VALIDATION_ENABLED = getattr(settings, 'EMAIL_VALIDATION_ENABLED', False) # Control via setting
EMAIL_VALIDATION_API_URL = getattr(settings, 'EMAIL_VALIDATION_API_URL', None) # e.g., "https://api.examplevalidator.com/v1/validate"
EMAIL_VALIDATION_API_KEY = getattr(settings, 'EMAIL_VALIDATION_API_KEY', None)
ODR_FOR_PROSPECT_DETAILS = getattr(settings, 'ODR_FOR_PROSPECT_DETAILS', False) # Control via setting

# --- Helper Functions ---

async def infer_pain_points_llm(client: httpx.AsyncClient, company_name: str, company_description: Optional[str] = None, odr_context: Optional[str] = None) -> Optional[str]:
    """Uses LLM to infer potential pain points based on company info and optional ODR context."""
    context = f"Company: {company_name}"
    if company_description: context += f"\nDescription: {company_description}"
    if odr_context: context += f"\n\nAdditional Context from Deep Research:\n{odr_context[:1500]}" # Add ODR context if available

    prompt = f"""
    Analyze the following company information:
    {context}

    Infer 1-2 specific, high-probability business pain points this company might face related to data analysis, market intelligence, competitive research, or strategic reporting, suitable for pitching Acumenis AI reports ($499/$999).
    Focus on needs addressable by rapid, AI-driven research. Be concise and action-oriented. Frame them as potential needs. Avoid generic statements. If no clear pain point can be inferred, respond with "None".

    Example format:
    - Need for faster competitor benchmarking in a dynamic market.
    - Requirement for data-driven validation of new feature ideas.
    - Difficulty synthesizing competitor insights from disparate sources.
    """
    inference_result = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest")
    if inference_result and isinstance(inference_result.get("raw_inference"), str):
        inferred_text = inference_result["raw_inference"].strip()
        if inferred_text.lower() != "none" and len(inferred_text) > 5:
            logger.info(f"[ProspectResearcher] Inferred pain points for {company_name}: {inferred_text}")
            return inferred_text
    logger.info(f"[ProspectResearcher] Could not infer pain points for {company_name}.")
    return None

async def find_contact_llm_guess(client: httpx.AsyncClient, company_name: str, website: Optional[str]) -> Optional[str]:
    """Attempts to find a contact email using LLM pattern guessing."""
    if not website:
        logger.info(f"[ProspectResearcher] Cannot guess email for {company_name} without website.")
        return None

    try:
        parsed_url = urlparse(website)
        domain = parsed_url.netloc or parsed_url.path
        domain = domain.replace("www.", "")
        if not domain or '.' not in domain: # Basic domain check
             raise ValueError("Invalid domain extracted")
    except Exception as e:
         logger.warning(f"[ProspectResearcher] Could not extract valid domain from website '{website}': {e}")
         return None

    prompt = f"""
    Given the company name "{company_name}" and domain "{domain}", suggest the single most likely professional contact email address suitable for B2B outreach regarding market research services.
    Consider common patterns: info@, contact@, marketing@, sales@, hello@, press@, partnerships@, [firstname].[lastname]@, [f].[lastname]@.
    Prioritize generic functional addresses (marketing, sales, info) unless a specific contact strategy is implied. Avoid making up names.
    Respond ONLY with the single most likely email address, or "None" if no reasonable guess can be made.
    """
    logger.info(f"[ProspectResearcher] Attempting LLM email guess for {company_name} ({domain})...")
    llm_response = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest")

    if llm_response and isinstance(llm_response.get("raw_inference"), str):
        guessed_email = llm_response["raw_inference"].strip()
        # Basic validation
        if "@" in guessed_email and "." in guessed_email.split("@")[-1] and guessed_email.lower() != "none":
            logger.info(f"[ProspectResearcher] LLM guessed email: {guessed_email} for {company_name}")
            return guessed_email
        else:
             logger.info(f"[ProspectResearcher] LLM did not provide a valid email guess for {company_name}. Response: {guessed_email}")
    else:
         logger.warning(f"[ProspectResearcher] LLM email guessing failed for {company_name}.")
    return None

async def validate_email_api(client: httpx.AsyncClient, email: str) -> bool:
    """Validates an email using a configured external API (placeholder)."""
    if not EMAIL_VALIDATION_ENABLED or not EMAIL_VALIDATION_API_URL or not email:
        return True # Skip validation if disabled or no email

    logger.info(f"[ProspectResearcher] Validating email: {email}")
    headers = {"Authorization": f"Bearer {EMAIL_VALIDATION_API_KEY}"} if EMAIL_VALIDATION_API_KEY else {}
    params = {"email": email} # Common parameter name

    try:
        # Use a separate client for validation API if needed, or reuse the main one
        async with httpx.AsyncClient(timeout=15.0) as validation_client:
            response = await validation_client.get(EMAIL_VALIDATION_API_URL, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            # --- Adapt parsing based on the specific API's response structure ---
            is_valid = result.get("is_valid") or result.get("status") == "valid" or result.get("deliverable")
            # --- End Adapt ---
            logger.info(f"Validation result for {email}: {'Valid' if is_valid else 'Invalid'}")
            return bool(is_valid)
    except Exception as e:
        logger.error(f"Email validation API call failed for {email}: {e}")
        return True # Default to true (assume valid) on API failure to avoid discarding potentially good leads

async def run_opendeepresearch_query(query: str, depth: int = 1) -> Optional[Union[List[Dict], str]]:
    """
    Runs the open-deep-research tool as a subprocess.
    Returns parsed JSON results (list of dicts) or raw text output on failure/non-JSON.
    """
    logger.info(f"[ProspectResearcher] Running ODR query (depth {depth}): '{query[:50]}...'")
    output_filename = f"odr_output_{random.randint(1000, 9999)}.json"
    output_path = os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, output_filename)
    node_script_path = os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT)

    if not os.path.exists(settings.NODE_EXECUTABLE_PATH):
        logger.error(f"Node executable not found at: {settings.NODE_EXECUTABLE_PATH}")
        return f"Error: Node executable not found at {settings.NODE_EXECUTABLE_PATH}"
    if not os.path.exists(node_script_path):
        logger.error(f"ODR entry point not found at: {node_script_path}")
        return f"Error: ODR entry point not found at {node_script_path}"

    cmd = [
        settings.NODE_EXECUTABLE_PATH,
        node_script_path,
        "--query", query,
        "--output", output_path,
        "--json", # Assume JSON output flag
        "--depth", str(depth), # Control research depth
        # Add model selection if ODR supports it and we want to control it here
        # "--model", settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest"
    ]

    raw_output = ""
    try:
        logger.info(f"[ProspectResearcher] Executing ODR: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.OPEN_DEEP_RESEARCH_REPO_PATH
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=OPEN_DEEP_RESEARCH_TIMEOUT)
        stdout_decoded = stdout.decode(errors='ignore').strip() if stdout else ""
        stderr_decoded = stderr.decode(errors='ignore').strip() if stderr else ""
        raw_output = f"STDOUT:\n{stdout_decoded}\n\nSTDERR:\n{stderr_decoded}"
        logger.info(f"[ProspectResearcher] ODR exited with code: {process.returncode}")
        if stdout_decoded: logger.debug(f"[ProspectResearcher] ODR STDOUT: {stdout_decoded[:500]}...")
        if stderr_decoded: logger.warning(f"[ProspectResearcher] ODR STDERR: {stderr_decoded[:500]}...")

        if process.returncode == 0 and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"[ProspectResearcher] Successfully parsed JSON results from {output_path}")
                # Basic validation - adapt based on expected ODR JSON structure
                if isinstance(results, list): return results
                if isinstance(results, dict) and 'results' in results and isinstance(results['results'], list): return results['results']
                if isinstance(results, dict) and 'companies' in results and isinstance(results['companies'], list): return results['companies']
                logger.warning(f"ODR JSON output is not a list or expected dict structure: {type(results)}")
                return raw_output # Return raw output if JSON structure is unexpected
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {output_path}: {e}")
                return raw_output # Return raw output if JSON parsing fails
            except Exception as e:
                logger.error(f"Error reading results file {output_path}: {e}")
                return raw_output
        else:
            logger.error(f"ODR failed (Code: {process.returncode}) or output file missing/empty ({output_path}).")
            return raw_output # Return raw output on failure

    except asyncio.TimeoutError:
         logger.error(f"ODR timed out after {OPEN_DEEP_RESEARCH_TIMEOUT}s for query: '{query[:50]}...'")
         return f"Error: ODR subprocess timed out after {OPEN_DEEP_RESEARCH_TIMEOUT}s."
    except FileNotFoundError as e:
         logger.error(f"FileNotFoundError running ODR: {e}")
         return f"Error: FileNotFoundError running ODR - {e}"
    except Exception as e:
        logger.error(f"Unexpected error running ODR: {e}", exc_info=True)
        return f"Error: Unexpected error running ODR - {e}"
    finally:
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError as e: logger.warning(f"Error removing ODR output file {output_path}: {e}")


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single cycle of prospecting using ODR and LLM enrichment."""
    logger.info("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client = await get_httpx_client()

    # Define research queries - MCOL should ideally suggest/refine these
    # Example: Fetch queries from DB or config?
    research_queries = getattr(settings, 'PROSPECTING_QUERIES', [
        "List B2B SaaS companies in marketing technology that received Series A funding recently",
        "Identify companies launching new AI-powered analytics products",
        "Find e-commerce platforms expanding into international markets",
        "Companies announcing large layoffs in tech sector", # Example of a different signal
        "Pharmaceutical companies with recent phase 3 trial failures" # Example signal
    ])
    random.shuffle(research_queries) # Process in random order

    try:
        for query in research_queries:
            if shutdown_event.is_set(): break
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

            odr_results = await run_opendeepresearch_query(query, depth=1) # Initial shallow search

            if isinstance(odr_results, list): # Check if we got structured data
                logger.info(f"[ProspectResearcher] Processing {len(odr_results)} results from query: '{query[:50]}...'")
                for result in odr_results:
                    if shutdown_event.is_set(): break
                    if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

                    # --- Adapt parsing based on actual ODR output ---
                    company_name = result.get("name") or result.get("company_name")
                    website = result.get("website") or result.get("url")
                    description = result.get("description") or result.get("summary") or result.get("signal")
                    # --- End Adapt ---

                    if not company_name: continue
                    company_name = company_name.strip()
                    logger.info(f"[ProspectResearcher] Processing lead: {company_name}")

                    odr_context_for_llm = None
                    if ODR_FOR_PROSPECT_DETAILS and website:
                        # Run deeper ODR search specifically for this company
                        logger.info(f"Running deeper ODR search for {company_name}...")
                        detail_query = f"Detailed analysis of {company_name} ({website or ''}) focusing on recent activities, challenges, and strategic direction."
                        detail_results = await run_opendeepresearch_query(detail_query, depth=2) # Deeper search
                        if isinstance(detail_results, str): # If ODR returned raw text
                            odr_context_for_llm = detail_results[:2000] # Use truncated raw text
                        elif isinstance(detail_results, list) and detail_results:
                            # Try to synthesize list results into context
                            odr_context_for_llm = json.dumps(detail_results[:5], indent=2)[:2000] # Use first few results
                        else:
                            logger.info(f"No detailed ODR context found for {company_name}.")


                    # Infer pain points using LLM, potentially with ODR context
                    inferred_pain = await infer_pain_points_llm(client, company_name, description, odr_context_for_llm)

                    # Attempt to find contact email using LLM guessing
                    contact_email = await find_contact_llm_guess(client, company_name, website)

                    # Validate email if enabled and found
                    is_valid_email = True
                    if contact_email and EMAIL_VALIDATION_ENABLED:
                        is_valid_email = await validate_email_api(client, contact_email)

                    prospect_status = "NEW" if is_valid_email else "INVALID_EMAIL"
                    if not contact_email: prospect_status = "NEW" # Keep NEW if no email guessed

                    # Save prospect
                    try:
                        created = await crud.create_prospect(
                            db=db,
                            company_name=company_name,
                            email=contact_email if is_valid_email else None, # Store None if invalid
                            website=website,
                            pain_point=inferred_pain,
                            source=f"odr_{query[:30].replace(' ', '_')}", # Source indicates ODR query
                            status=prospect_status # Set status based on validation
                        )
                        if created:
                            await db.commit() # Commit each prospect individually
                            prospects_added_this_cycle += 1
                            logger.info(f"Prospect '{created.company_name}' (ID: {created.prospect_id}) saved with status {prospect_status}. Total added: {prospects_added_this_cycle}")
                        else:
                            await db.rollback() # Duplicate or other issue during save
                    except Exception as e:
                        logger.error(f"Error saving prospect from ODR ({company_name}) to DB: {e}", exc_info=True)
                        await db.rollback()

                    await asyncio.sleep(random.uniform(1.0, 2.5)) # Slightly longer delay

            elif isinstance(odr_results, str): # ODR failed or returned raw text
                logger.error(f"ODR query failed or returned non-JSON for query '{query[:50]}...': {odr_results}")
            else:
                 logger.info(f"No results returned from ODR query: '{query[:50]}...'")


    except Exception as e:
        logger.error(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}", exc_info=True)
    finally:
        if client: await client.aclose()

    logger.info(f"[ProspectResearcher] Prospecting cycle finished. Added {prospects_added_this_cycle} new prospects.")
