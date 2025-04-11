import asyncio
import random
import json
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
import datetime

from sqlalchemy.ext.asyncio import AsyncSession
import httpx # Use the client from agent_utils
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Corrected relative imports for package structure
from Acumenis.app.core.config import settings
from Acumenis.app.db import crud, models
from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api # Import shared LLM caller

# --- Configuration ---
# Removed SIGNAL_SOURCES as NewsAPI is deprecated

MAX_PROSPECTS_PER_CYCLE = 20 # Limit prospects generated per run
# SIGNAL_ANALYSIS_TIMEOUT = 60 # Seconds to wait for LLM inference (Now handled by call_llm_api timeout)
OPEN_DEEP_RESEARCH_TIMEOUT = 300 # Seconds timeout for the external tool

# Use tenacity for retrying API calls
RETRY_WAIT = wait_fixed(3)
RETRY_ATTEMPTS = 3

# Removed local call_llm_api - use shared one from agent_utils


async def infer_pain_points_llm(client: httpx.AsyncClient, company_name: str, company_description: Optional[str] = None) -> Optional[str]:
    """Uses LLM to infer potential pain points based on company name and optional description."""
    context = f"Company: {company_name}"
    if company_description:
        context += f"\nDescription: {company_description}"

    prompt = f"""
    Analyze the following company information:
    {context}

    Infer 1-2 specific, high-probability business pain points this company might face related to data analysis, market intelligence, competitive research, or strategic reporting, suitable for pitching Acumenis AI reports ($499/$999).
    Focus on needs addressable by rapid, AI-driven research. Be concise and action-oriented. Frame them as potential needs. Avoid generic statements. If no clear pain point can be inferred, respond with "None".

    Example format:
    - Need for faster competitor benchmarking in a dynamic market.
    - Requirement for data-driven validation of new feature ideas.
    """
    # Use shared call_llm_api from agent_utils
    inference_result = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest") # Use a standard model
    if inference_result and isinstance(inference_result.get("raw_inference"), str):
        inferred_text = inference_result["raw_inference"].strip()
        if inferred_text.lower() != "none" and len(inferred_text) > 5:
            print(f"[ProspectResearcher] Inferred pain points for {company_name}: {inferred_text}")
            return inferred_text
    print(f"[ProspectResearcher] Could not infer pain points for {company_name}.")
    return None

async def find_contact_llm_guess(client: httpx.AsyncClient, company_name: str, website: Optional[str]) -> Optional[str]:
    """
    Attempts to find a contact email using LLM pattern guessing.
    Strategy C: LLM Guessing - Low Reliability.
    """
    if not website:
        print(f"[ProspectResearcher] Cannot guess email for {company_name} without website.")
        return None

    parsed_url = urlparse(website)
    domain = parsed_url.netloc or parsed_url.path # Handle cases with/without scheme
    domain = domain.replace("www.", "") # Clean domain
    if not domain:
         print(f"[ProspectResearcher] Could not extract domain from website: {website}")
         return None

    prompt = f"""
    Given the company name "{company_name}" and domain "{domain}", suggest the single most likely professional contact email address suitable for B2B outreach regarding market research services.
    Consider common patterns like:
    - info@{domain}
    - contact@{domain}
    - marketing@{domain}
    - sales@{domain}
    - hello@{domain}
    - press@{domain}
    - partnerships@{domain}
    - [firstname].[lastname]@{domain} (if name is guessable, but avoid making up names)
    - [f].[lastname]@{domain}

    Prioritize generic functional addresses (marketing, sales, info) unless a specific contact strategy is implied.
    Respond ONLY with the single most likely email address, or "None" if no reasonable guess can be made.
    """
    print(f"[ProspectResearcher] Attempting LLM email guess for {company_name} ({domain})...")
    # Use shared call_llm_api from agent_utils
    llm_response = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest") # Use a standard model

    if llm_response and isinstance(llm_response.get("raw_inference"), str):
        guessed_email = llm_response["raw_inference"].strip()
        # Basic validation
        if "@" in guessed_email and "." in guessed_email.split("@")[-1] and guessed_email.lower() != "none":
            print(f"[ProspectResearcher] LLM guessed email: {guessed_email} for {company_name}")
            return guessed_email
        else:
             print(f"[ProspectResearcher] LLM did not provide a valid email guess for {company_name}. Response: {guessed_email}")
    else:
         print(f"[ProspectResearcher] LLM email guessing failed for {company_name}.")

    # Fallback or default if LLM fails
    # return f"info@{domain}" # Optionally return a default guess
    return None

async def run_opendeepresearch_query(query: str) -> Optional[List[Dict]]:
    """
    Runs the open-deep-research tool as a subprocess with a given query.
    Parses the JSON output file.
    """
    print(f"[ProspectResearcher] Running open-deep-research for query: '{query[:50]}...'")
    results = None
    output_filename = f"research_output_{random.randint(1000, 9999)}.json"
    # Assume output is saved relative to the tool's repo path for simplicity
    output_path = os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, output_filename)

    # Construct command
    node_script_path = os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT)
    if not os.path.exists(settings.NODE_EXECUTABLE_PATH):
        print(f"Error: Node executable not found at: {settings.NODE_EXECUTABLE_PATH}")
        return None
    if not os.path.exists(node_script_path):
        print(f"Error: open-deep-research entry point not found at: {node_script_path}")
        return None

    cmd = [
        settings.NODE_EXECUTABLE_PATH,
        node_script_path,
        "--query", query,
        "--output", output_path, # Assuming the tool saves JSON to this path
        "--json", # Assuming a flag for JSON output
        # Add other relevant flags like --depth if needed
        # "--depth", "3",
    ]

    try:
        print(f"[ProspectResearcher] Executing: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.OPEN_DEEP_RESEARCH_REPO_PATH
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=OPEN_DEEP_RESEARCH_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"[ProspectResearcher] open-deep-research timed out after {OPEN_DEEP_RESEARCH_TIMEOUT}s. Terminating.")
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except Exception: process.kill() # Force kill if terminate fails
            raise TimeoutError("open-deep-research subprocess timed out.")

        stdout_decoded = stdout.decode(errors='ignore').strip() if stdout else ""
        stderr_decoded = stderr.decode(errors='ignore').strip() if stderr else ""
        print(f"[ProspectResearcher] open-deep-research exited with code: {process.returncode}")
        if stdout_decoded: print(f"[ProspectResearcher] ODR STDOUT: {stdout_decoded[:500]}...")
        if stderr_decoded: print(f"[ProspectResearcher] ODR STDERR: {stderr_decoded[:500]}...")

        if process.returncode == 0 and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"[ProspectResearcher] Successfully parsed results from {output_path}")
                # Assuming results is a list of dictionaries, one per company/lead
                if not isinstance(results, list):
                     print(f"[ProspectResearcher] Warning: Expected list from JSON output, got {type(results)}. Trying to adapt.")
                     # Attempt to adapt if it's a dict with a known key, otherwise fail
                     if isinstance(results, dict) and 'results' in results and isinstance(results['results'], list):
                          results = results['results']
                     elif isinstance(results, dict) and 'companies' in results and isinstance(results['companies'], list):
                          results = results['companies']
                     else:
                          results = None # Cannot adapt
            except json.JSONDecodeError as e:
                print(f"[ProspectResearcher] Error decoding JSON from {output_path}: {e}")
            except Exception as e:
                print(f"[ProspectResearcher] Error reading results file {output_path}: {e}")
        else:
            print(f"[ProspectResearcher] open-deep-research failed (Code: {process.returncode}) or output file missing ({output_path}).")

    except TimeoutError as e:
         print(f"[ProspectResearcher] TimeoutError: {e}")
    except FileNotFoundError as e:
         print(f"[ProspectResearcher] FileNotFoundError running open-deep-research: {e}")
    except Exception as e:
        print(f"[ProspectResearcher] Unexpected error running open-deep-research: {e}")
    finally:
        # Clean up the output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                print(f"[ProspectResearcher] Error removing output file {output_path}: {e}")

    return results
# Removed duplicate return statement
    return added_count


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """
    Runs a single cycle of prospecting using open-deep-research.
    """
    print("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client = await get_httpx_client() # For LLM enrichment

    # Define research queries/strategies for open-deep-research
    # TODO: Make these dynamic or configurable
    research_queries = [
        "List B2B SaaS companies in marketing technology that received Series A funding recently",
        "Identify companies launching new AI-powered analytics products",
        "Find e-commerce platforms expanding into international markets",
    ]

    try:
        for query in research_queries:
            if shutdown_event.is_set(): break
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

            research_results = await run_opendeepresearch_query(query)

            if research_results:
                print(f"[ProspectResearcher] Processing {len(research_results)} results from query: '{query[:50]}...'")
                for result in research_results:
                    if shutdown_event.is_set(): break
                    if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

                    # --- Adapt parsing based on actual open-deep-research output ---
                    # Assuming output provides at least 'name' and potentially 'website', 'description'
                    company_name = result.get("name") or result.get("company_name")
                    website = result.get("website") or result.get("url")
                    description = result.get("description") or result.get("summary") or result.get("signal")
                    # --- End Adapt ---

                    if not company_name: continue # Skip if no company name found

                    company_name = company_name.strip()
                    print(f"[ProspectResearcher] Processing lead: {company_name}")

                    # Infer pain points
                    inferred_pain = await infer_pain_points_llm(client, company_name, description)

                    # Attempt to find contact email using LLM guessing
                    contact_email = await find_contact_llm_guess(client, company_name, website)

                    # Save prospect
                    try:
                        created = await crud.create_prospect(
                            db=db,
                            company_name=company_name,
                            email=contact_email,
                            website=website,
                            pain_point=inferred_pain,
                            source=f"odr_{query[:30]}" # Source indicates ODR query
                        )
                        if created:
                            await db.commit()
                            prospects_added_this_cycle += 1
                            print(f"[ProspectResearcher] Prospect '{created.company_name}' added from ODR (ID: {created.prospect_id}). Total added: {prospects_added_this_cycle}")
                        else:
                            await db.rollback() # Duplicate or other issue
                    except Exception as e:
                        print(f"[ProspectResearcher] Error saving prospect from ODR ({company_name}) to DB: {e}")
                        await db.rollback()

                    await asyncio.sleep(random.uniform(0.5, 1.5)) # Small delay between processing results

    except Exception as e:
        print(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}")
        # Log error
    finally:
        await client.aclose()

    print(f"[ProspectResearcher] Prospecting cycle finished. Added {prospects_added_this_cycle} new prospects.")