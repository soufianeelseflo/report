
# autonomous_agency/app/agents/prospect_researcher.py
import asyncio
import random
import json
import subprocess
import os
import logging
import traceback
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urljoin, urlparse
import datetime
import time # For circuit breaker timestamp

from sqlalchemy.ext.asyncio import AsyncSession
import httpx
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
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Configuration ---
MAX_PROSPECTS_PER_CYCLE = settings.MAX_PROSPECTS_PER_CYCLE or 20
OPEN_DEEP_RESEARCH_TIMEOUT = settings.OPEN_DEEP_RESEARCH_TIMEOUT or 300
EMAIL_VALIDATION_ENABLED = getattr(settings, 'EMAIL_VALIDATION_ENABLED', False)
EMAIL_VALIDATION_API_URL = getattr(settings, 'EMAIL_VALIDATION_API_URL', None)
EMAIL_VALIDATION_API_KEY = getattr(settings, 'EMAIL_VALIDATION_API_KEY', None)
ODR_FOR_PROSPECT_DETAILS = getattr(settings, 'ODR_FOR_PROSPECT_DETAILS', False)
ODR_DETAIL_DEPTH = getattr(settings, 'ODR_DETAIL_DEPTH', 2) # Configurable depth for detail search
VALIDATION_API_FAILURE_THRESHOLD = 3 # Number of consecutive failures before tripping circuit breaker
VALIDATION_API_COOLDOWN_SECONDS = 300 # Cooldown period for validation API

# --- State for Email Validation Circuit Breaker ---
_validation_api_consecutive_failures = 0
_validation_api_disabled_until: Optional[float] = None

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
        guessed_email = llm_response["raw_inference"].strip().lower() # Normalize to lowercase
        # Basic validation: ensure it looks like an email and isn't just "none"
        if "@" in guessed_email and "." in guessed_email.split("@")[-1] and guessed_email != "none":
            logger.info(f"[ProspectResearcher] LLM guessed email: {guessed_email} for {company_name}")
            return guessed_email
        else:
             logger.info(f"[ProspectResearcher] LLM did not provide a valid email guess for {company_name}. Response: {guessed_email}")
    else:
         logger.warning(f"[ProspectResearcher] LLM email guessing failed for {company_name}.")
    return None

async def validate_email_api(client: httpx.AsyncClient, email: str) -> bool:
    """
    Validates an email using a configured external API with circuit breaker logic.
    Returns True if valid or if validation is disabled/fails/in cooldown, False only if explicitly invalid.
    """
    global _validation_api_consecutive_failures, _validation_api_disabled_until

    if not EMAIL_VALIDATION_ENABLED:
        return True
    if not EMAIL_VALIDATION_API_URL:
        logger.warning("Email validation enabled but API URL not configured.")
        return True # Assume valid if misconfigured
    if not email:
        return False # Cannot validate empty email

    # Check circuit breaker
    if _validation_api_disabled_until and time.monotonic() < _validation_api_disabled_until:
        logger.warning(f"Email validation API is in cooldown. Skipping validation for {email}.")
        return True # Assume valid during cooldown

    logger.info(f"[ProspectResearcher] Validating email via API: {email}")
    headers = {"Authorization": f"Bearer {EMAIL_VALIDATION_API_KEY}"} if EMAIL_VALIDATION_API_KEY else {}
    params = {"email": email} # Common parameter name

    try:
        # Use the main httpx client from agent_utils
        response = await client.get(EMAIL_VALIDATION_API_URL, params=params, headers=headers, timeout=15.0)
        response.raise_for_status()
        result = response.json()

        # Reset failure count on success
        _validation_api_consecutive_failures = 0
        _validation_api_disabled_until = None

        # --- Adapt parsing based on the specific API's response structure ---
        status = str(result.get("status", result.get("result", result.get("deliverability", "")))).lower()
        is_valid_flag = result.get("is_valid", None)
        is_valid = False
        if is_valid_flag is not None: is_valid = bool(is_valid_flag)
        elif status in ["valid", "deliverable", "ok", "risky"]: is_valid = True
        # --- End Adapt ---

        logger.info(f"Validation result for {email}: {'Valid/Deliverable' if is_valid else 'Invalid/Undeliverable'}. API Response: {result}")
        return is_valid

    except Exception as e:
        logger.error(f"Email validation API call failed for {email}: {e}")
        _validation_api_consecutive_failures += 1
        if _validation_api_consecutive_failures >= VALIDATION_API_FAILURE_THRESHOLD:
            _validation_api_disabled_until = time.monotonic() + VALIDATION_API_COOLDOWN_SECONDS
            logger.critical(f"Email validation API failed {_validation_api_consecutive_failures} consecutive times. Disabling for {VALIDATION_API_COOLDOWN_SECONDS} seconds.")
            # TODO: Log a suggestion for MCOL/Operator to investigate the validation API
            # await log_critical_event(f"Email validation API circuit breaker tripped. Disabled until {datetime.datetime.now() + datetime.timedelta(seconds=VALIDATION_API_COOLDOWN_SECONDS)}")
        return True # Default to true (assume valid) on API failure

async def run_opendeepresearch_query(query: str, depth: int = 1) -> Optional[Union[List[Dict], str]]:
    """
    Runs the open-deep-research tool as a subprocess.
    Returns parsed JSON results (list of dicts) or raw text output on failure/non-JSON.
    """
    logger.info(f"[ProspectResearcher] Running ODR query (depth {depth}): '{query[:50]}...'")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_filename = f"odr_output_{timestamp}_{random.randint(1000, 9999)}.json"
    # Ensure the output directory exists relative to the ODR repo path
    odr_repo_path = settings.OPEN_DEEP_RESEARCH_REPO_PATH
    output_path = os.path.join(odr_repo_path, output_filename)
    node_script_path = os.path.join(odr_repo_path, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT)

    # Validate paths
    if not os.path.exists(settings.NODE_EXECUTABLE_PATH):
        logger.error(f"Node executable not found at: {settings.NODE_EXECUTABLE_PATH}")
        return f"Error: Node executable not found at {settings.NODE_EXECUTABLE_PATH}"
    if not os.path.exists(odr_repo_path):
        logger.error(f"ODR repository path not found: {odr_repo_path}")
        return f"Error: ODR repository path not found: {odr_repo_path}"
    if not os.path.exists(node_script_path):
        logger.error(f"ODR entry point not found at: {node_script_path}")
        # Attempt common alternatives
        alt_paths = ["index.js", "cli.js", "dist/index.js", "dist/cli.js"] # Added common build outputs
        found_alt = False
        for alt in alt_paths:
            alt_full_path = os.path.join(odr_repo_path, alt)
            if os.path.exists(alt_full_path):
                logger.warning(f"Default ODR entry point not found, using alternative: {alt}")
                node_script_path = alt_full_path
                found_alt = True
                break
        if not found_alt:
            return f"Error: ODR entry point not found at {settings.OPEN_DEEP_RESEARCH_ENTRY_POINT} or common alternatives in {odr_repo_path}."

    # Prepare command
    cmd = [
        settings.NODE_EXECUTABLE_PATH,
        node_script_path,
        "--query", query,
        "--output", output_path, # ODR needs to write to this path
        "--json", # Ensure JSON output is requested
        "--depth", str(depth),
        # Potentially pass model if ODR supports it via args
        # "--model", settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest"
    ]

    # Environment setup (e.g., for API keys ODR might use)
    env = os.environ.copy()
    # Example: If ODR reads OPENROUTER_API_KEY
    # active_key_info = get_next_api_key() # Needs adaptation if called from here
    # if active_key_info: env['OPENROUTER_API_KEY'] = active_key_info['key']

    raw_output = ""
    process = None
    try:
        logger.info(f"[ProspectResearcher] Executing ODR in '{odr_repo_path}': {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=odr_repo_path, # Crucial: Run from the tool's directory
            env=env
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=OPEN_DEEP_RESEARCH_TIMEOUT)
        stdout_decoded = stdout.decode(errors='ignore').strip() if stdout else ""
        stderr_decoded = stderr.decode(errors='ignore').strip() if stderr else ""
        raw_output = f"ODR Return Code: {process.returncode}\n\nSTDOUT:\n{stdout_decoded}\n\nSTDERR:\n{stderr_decoded}"
        logger.info(f"[ProspectResearcher] ODR exited with code: {process.returncode}")
        if stdout_decoded: logger.debug(f"[ProspectResearcher] ODR STDOUT: {stdout_decoded[:500]}...")
        if stderr_decoded: logger.warning(f"[ProspectResearcher] ODR STDERR: {stderr_decoded[:500]}...")

        # Check results
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 2:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"[ProspectResearcher] Successfully parsed JSON results from {output_path}")
                # Adapt parsing based on expected ODR JSON structure
                if isinstance(results, list): return results
                if isinstance(results, dict):
                    if isinstance(results.get('results'), list): return results['results']
                    if isinstance(results.get('companies'), list): return results['companies']
                    # Add other potential top-level keys if ODR structure varies
                logger.warning(f"ODR JSON output is not a list or expected dict structure: {type(results)}. Content: {str(results)[:200]}...")
                return raw_output # Return raw output if JSON structure is unexpected
            except json.JSONDecodeError as e:
                file_content = ""
                try:
                    with open(output_path, 'r', encoding='utf-8') as f_err: file_content = f_err.read(500)
                except Exception: pass
                logger.error(f"Error decoding JSON from {output_path}: {e}. Raw content start: {file_content}...")
                return raw_output # Return raw output if JSON parsing fails
            except Exception as e:
                logger.error(f"Error reading results file {output_path}: {e}", exc_info=True)
                return raw_output
        else:
            logger.error(f"ODR failed (Code: {process.returncode}) or output file missing/empty ({output_path}). Raw output:\n{raw_output}")
            return raw_output # Return raw output on failure

    except asyncio.TimeoutError:
         logger.error(f"ODR timed out after {OPEN_DEEP_RESEARCH_TIMEOUT}s for query: '{query[:50]}...'")
         if process and process.returncode is None: # Terminate runaway process
             try: process.terminate()
             except ProcessLookupError: pass
         return f"Error: ODR subprocess timed out after {OPEN_DEEP_RESEARCH_TIMEOUT}s."
    except FileNotFoundError as e:
         logger.error(f"FileNotFoundError running ODR: {e}")
         return f"Error: FileNotFoundError running ODR - {e}"
    except Exception as e:
        logger.error(f"Unexpected error running ODR: {e}", exc_info=True)
        return f"Error: Unexpected error running ODR - {e}"
    finally:
        # Ensure output file is removed
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError as e: logger.warning(f"Error removing ODR output file {output_path}: {e}")


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single cycle of prospecting using ODR and LLM enrichment."""
    logger.info("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client: Optional[httpx.AsyncClient] = None

    # Fetch queries from settings
    research_queries = getattr(settings, 'PROSPECTING_QUERIES', [])
    if not research_queries:
        logger.warning("No prospecting queries defined in settings.PROSPECTING_QUERIES. Skipping cycle.")
        return

    random.shuffle(research_queries) # Process in random order

    try:
        client = await get_httpx_client()
        for query in research_queries:
            if shutdown_event.is_set():
                logger.info("Shutdown signal received during prospecting.")
                break
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE:
                logger.info(f"Reached max prospects per cycle ({MAX_PROSPECTS_PER_CYCLE}).")
                break

            odr_results = await run_opendeepresearch_query(query, depth=1) # Initial shallow search

            if isinstance(odr_results, list): # Check if we got structured data
                logger.info(f"[ProspectResearcher] Processing {len(odr_results)} results from query: '{query[:50]}...'")
                for result in odr_results:
                    if shutdown_event.is_set(): break
                    if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

                    # --- Adapt parsing based on actual ODR output ---
                    company_name = result.get("name") or result.get("company_name") or result.get("title")
                    website = result.get("website") or result.get("url") or result.get("link")
                    description = result.get("description") or result.get("summary") or result.get("snippet") or result.get("text")
                    # --- End Adapt ---

                    if not company_name:
                        logger.debug("Skipping result with no company name.")
                        continue
                    company_name = company_name.strip()
                    logger.info(f"[ProspectResearcher] Processing lead: {company_name}")

                    odr_context_for_llm = None
                    if ODR_FOR_PROSPECT_DETAILS and website:
                        # Run deeper ODR search specifically for this company
                        logger.info(f"Running deeper ODR search (depth {ODR_DETAIL_DEPTH}) for {company_name}...")
                        detail_query = f"Detailed analysis of {company_name} ({website or ''}) focusing on recent activities, challenges, strategic direction, key personnel, and potential needs related to market intelligence."
                        detail_results = await run_opendeepresearch_query(detail_query, depth=ODR_DETAIL_DEPTH)
                        if isinstance(detail_results, str): # If ODR returned raw text
                            odr_context_for_llm = detail_results[:2000] # Use truncated raw text
                        elif isinstance(detail_results, list) and detail_results:
                            # Synthesize list results into context (simple approach)
                            odr_context_for_llm = "\n".join([f"- {item.get('name', item.get('title', ''))}: {item.get('description', item.get('snippet', ''))[:200]}" for item in detail_results[:5]])
                        else:
                            logger.info(f"No detailed ODR context found for {company_name}.")

                    # Infer pain points using LLM, potentially with ODR context
                    inferred_pain = await infer_pain_points_llm(client, company_name, description, odr_context_for_llm)

                    # Attempt to find contact email using LLM guessing
                    contact_email = await find_contact_llm_guess(client, company_name, website)

                    # Validate email if enabled and found
                    is_valid_email = True # Assume valid by default
                    if contact_email and EMAIL_VALIDATION_ENABLED:
                        is_valid_email = await validate_email_api(client, contact_email)

                    prospect_status = "NEW"
                    email_to_save = contact_email
                    if not contact_email:
                        prospect_status = "NEW" # No email guessed
                    elif not is_valid_email:
                        prospect_status = "INVALID_EMAIL"
                        email_to_save = None # Don't save invalid email
                        logger.info(f"Marking prospect {company_name} as INVALID_EMAIL.")

                    # Save prospect using isolated session
                    try:
                        async with get_worker_session() as prospect_session:
                            created = await crud.create_prospect(
                                db=prospect_session,
                                company_name=company_name,
                                email=email_to_save,
                                website=website,
                                pain_point=inferred_pain,
                                source=f"odr_{query[:30].replace(' ', '_')}",
                                status=prospect_status
                            )
                            if created:
                                await prospect_session.commit()
                                prospects_added_this_cycle += 1
                                logger.info(f"Prospect '{created.company_name}' (ID: {created.prospect_id}) saved with status {prospect_status}. Total added: {prospects_added_this_cycle}")
                            else:
                                logger.warning(f"Failed to create or update prospect {company_name} (returned None).")
                                # No explicit rollback needed, session scope handles it on exit if commit fails

                    except Exception as e:
                        logger.error(f"Error saving prospect {company_name} to DB: {e}", exc_info=True)
                        # No explicit rollback needed

                    await asyncio.sleep(random.uniform(1.5, 3.0)) # Delay between processing leads

            elif isinstance(odr_results, str): # ODR failed or returned raw text
                logger.error(f"ODR query failed or returned non-JSON for query '{query[:50]}...'. Error/Output: {odr_results[:500]}...")
            else:
                 logger.info(f"No results returned from ODR query: '{query[:50]}...'")

            # Delay between different ODR queries
            await asyncio.sleep(random.uniform(5.0, 10.0))

    except Exception as e:
        logger.error(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}", exc_info=True)
    finally:
        if client: await client.aclose()

    logger.info(f"[ProspectResearcher] Prospecting cycle finished. Added {prospects_added_this_cycle} new prospects.")