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
    # MODIFIED: Ensure get_httpx_client is used for proxied requests
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api, get_worker_session
    # Optional: Import Playwright if adding visual extraction
    # from playwright.async_api import async_playwright
except ImportError:
    print("[ProspectResearcher] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api, get_worker_session
    # from playwright.async_api import async_playwright


# Setup logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Configuration ---
MAX_PROSPECTS_PER_CYCLE = settings.MAX_PROSPECTS_PER_CYCLE # Use configured value (e.g., 50)
OPEN_DEEP_RESEARCH_TIMEOUT = settings.REPORT_GENERATOR_TIMEOUT_SECONDS # Reuse report timeout
EMAIL_VALIDATION_ENABLED = getattr(settings, 'EMAIL_VALIDATION_ENABLED', False) # Keep disabled by default
EMAIL_VALIDATION_API_URL = getattr(settings, 'EMAIL_VALIDATION_API_URL', None)
EMAIL_VALIDATION_API_KEY = getattr(settings, 'EMAIL_VALIDATION_API_KEY', None)
ODR_FOR_PROSPECT_DETAILS = getattr(settings, 'ODR_FOR_PROSPECT_DETAILS', True) # Keep enabled
ODR_DETAIL_DEPTH = getattr(settings, 'ODR_DETAIL_DEPTH', 1) # Use faster depth
VALIDATION_API_FAILURE_THRESHOLD = 3
VALIDATION_API_COOLDOWN_SECONDS = 300

# --- State for Email Validation Circuit Breaker ---
_validation_api_consecutive_failures = 0
_validation_api_disabled_until: Optional[float] = None

# --- Helper Functions ---

async def infer_pain_points_llm(client: httpx.AsyncClient, company_name: str, company_description: Optional[str] = None, odr_context: Optional[str] = None) -> Optional[str]:
    """Uses LLM to infer potential pain points based on company info and optional ODR context."""
    context = f"Company: {company_name}"
    if company_description: context += f"\nDescription: {company_description}"
    if odr_context:
        context += f"\n\nDeep Research Context (Prioritize this):\n{odr_context[:2000]}" # Keep ODR context priority
    elif company_description:
         context += f"\nDescription: {company_description}"

    prompt = f"""
    Analyze the following company information, paying close attention to the Deep Research Context if provided:
    {context}

    Based *primarily* on the Deep Research Context (or the description if context is unavailable), infer 1-2 specific, high-probability business pain points this company might face related to data analysis, market intelligence, competitive research, or strategic reporting. These pain points should be directly addressable by Acumenis AI reports ($499/$999).
    Focus on needs suggested by recent activities, challenges, or strategic directions mentioned in the context. Be concise and action-oriented. Frame them as potential needs or challenges. Avoid generic statements. If no clear pain point can be inferred from the provided information, respond with "None".

    Example format (if context mentioned competitor X launching feature Y):
    - Challenge adapting to competitor X's recent launch of feature Y.
    - Need for rapid analysis of market response to feature Y.
    """
    # Use the single API key via call_llm_api
    inference_result = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL)
    if inference_result and isinstance(inference_result.get("raw_inference"), str):
        inferred_text = inference_result["raw_inference"].strip()
        if inferred_text.lower() != "none" and len(inferred_text) > 5:
            logger.info(f"[ProspectResearcher] Inferred pain points for {company_name}: {inferred_text}")
            return inferred_text
    logger.info(f"[ProspectResearcher] Could not infer specific pain points for {company_name} from context.")
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
    # Use the single API key via call_llm_api
    llm_response = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL)

    if llm_response and isinstance(llm_response.get("raw_inference"), str):
        guessed_email = llm_response["raw_inference"].strip().lower() # Normalize to lowercase
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
        # Use the main httpx client (which might be proxied)
        response = await client.get(str(EMAIL_VALIDATION_API_URL), params=params, headers=headers, timeout=15.0) # Cast URL
        response.raise_for_status()
        result = response.json()

        _validation_api_consecutive_failures = 0
        _validation_api_disabled_until = None

        status = str(result.get("status", result.get("result", result.get("deliverability", "")))).lower()
        is_valid_flag = result.get("is_valid", None)

        is_valid = False
        if is_valid_flag is not None:
            is_valid = bool(is_valid_flag)
        elif status in ["valid", "deliverable", "ok", "risky", "catch_all"]:
            is_valid = True

        logger.info(f"Validation result for {email}: {'Valid/Deliverable' if is_valid else 'Invalid/Undeliverable'}. API Response Status: {status}")
        return is_valid

    except Exception as e:
        logger.error(f"Email validation API call failed for {email}: {e}")
        _validation_api_consecutive_failures += 1
        if _validation_api_consecutive_failures >= VALIDATION_API_FAILURE_THRESHOLD:
            _validation_api_disabled_until = time.monotonic() + VALIDATION_API_COOLDOWN_SECONDS
            logger.critical(f"Email validation API failed {_validation_api_consecutive_failures} consecutive times. Disabling for {VALIDATION_API_COOLDOWN_SECONDS} seconds.")
        return True # Default to true (assume valid) on API failure

async def run_opendeepresearch_query(query: str, depth: int = 1) -> Optional[Union[List[Dict], str]]:
    """
    Runs the open-deep-research tool via the internal service URL.
    Returns parsed JSON results (list of dicts) or raw text output on failure/non-JSON.
    """
    logger.info(f"[ProspectResearcher] Calling ODR service (depth {depth}) for query: '{query[:50]}...'")

    odr_service_url = getattr(settings, 'OPEN_DEEP_RESEARCH_SERVICE_URL', None)
    if not odr_service_url:
        error_message = "Configuration Error: OPEN_DEEP_RESEARCH_SERVICE_URL is not set."
        logger.critical(f"[ProspectResearcher] {error_message}")
        return f"Error: {error_message}"

    # Ensure URL has http/https scheme
    if not str(odr_service_url).startswith(('http://', 'https://')):
        odr_service_url_str = f"http://{odr_service_url}" # Default to http for internal service
    else:
        odr_service_url_str = str(odr_service_url)

    odr_endpoint = f"{odr_service_url_str.rstrip('/')}/api/search" # Assuming ODR service has a /api/search endpoint

    # Prepare payload for ODR service search endpoint
    # This needs to match what the ODR service expects
    odr_payload = {
        "query": query,
        "depth": depth, # Pass depth if ODR service supports it
        # Add other parameters ODR service might need (e.g., model preference)
        "platformModel": settings.STANDARD_REPORT_MODEL # Pass the agency's standard model
    }

    try:
        # Use the main httpx client (which might be proxied)
        async with get_httpx_client() as client:
            logger.info(f"[ProspectResearcher] Sending request to ODR service: POST {odr_endpoint}")
            response = await client.post(
                odr_endpoint,
                json=odr_payload,
                timeout=OPEN_DEEP_RESEARCH_TIMEOUT # Use configured timeout
            )
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses
            odr_result = response.json()

        # Process ODR service result - Adapt based on ODR service's /api/search response structure
        # Assuming it returns a structure similar to the original ODR tool's output file
        if isinstance(odr_result, list):
            logger.info(f"[ProspectResearcher] Received {len(odr_result)} results from ODR service.")
            return odr_result
        elif isinstance(odr_result, dict) and isinstance(odr_result.get('results'), list):
             logger.info(f"[ProspectResearcher] Received {len(odr_result['results'])} results from ODR service.")
             return odr_result['results']
        elif isinstance(odr_result, dict) and isinstance(odr_result.get('webPages', {}).get('value'), list):
             logger.info(f"[ProspectResearcher] Received {len(odr_result['webPages']['value'])} results from ODR service (Bing format?).")
             # Transform Bing-like format if needed
             return [
                 {
                     "id": item.get('id', f"odr_{i}"),
                     "url": item.get('url'),
                     "name": item.get('name'),
                     "snippet": item.get('snippet'),
                     "description": item.get('snippet') # Use snippet as description fallback
                 } for i, item in enumerate(odr_result['webPages']['value'])
             ]
        else:
            logger.warning(f"ODR service returned unexpected format. Response: {str(odr_result)[:500]}")
            return str(odr_result) # Return raw string representation

    except httpx.HTTPStatusError as e:
        error_body = ""
        try: error_body = e.response.text[:500]
        except Exception: pass
        error_message = f"ODR service call failed: Status {e.response.status_code}. Response: {error_body}"
        logger.error(f"[ProspectResearcher] {error_message}")
        return f"Error: {error_message}"
    except httpx.RequestError as e:
        error_message = f"Network error calling ODR service: {e}"
        logger.error(f"[ProspectResearcher] {error_message}")
        return f"Error: {error_message}"
    except asyncio.TimeoutError:
        error_message = f"ODR service call timed out after {OPEN_DEEP_RESEARCH_TIMEOUT} seconds."
        logger.error(f"[ProspectResearcher] {error_message}")
        return f"Error: {error_message}"
    except Exception as e:
        error_message = f"Unexpected error calling ODR service: {str(e)[:500]}"
        logger.error(f"[ProspectResearcher] Error: {error_message}", exc_info=True)
        return f"Error: {error_message}"


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single cycle of prospecting using ODR service and LLM enrichment."""
    logger.info("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client: Optional[httpx.AsyncClient] = None

    research_queries = getattr(settings, 'PROSPECTING_QUERIES', [])
    if not research_queries:
        logger.warning("No prospecting queries defined in settings.PROSPECTING_QUERIES. Skipping cycle.")
        return

    random.shuffle(research_queries) # Process in random order

    try:
        # Get client once for the cycle (will use current proxy setting)
        client = await get_httpx_client()

        for query in research_queries:
            if shutdown_event.is_set():
                logger.info("Shutdown signal received during prospecting.")
                break
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE:
                logger.info(f"Reached max prospects per cycle ({MAX_PROSPECTS_PER_CYCLE}).")
                break

            # --- Run ODR for initial lead generation via internal service ---
            logger.info(f"[ProspectResearcher] Running initial ODR query for leads: '{query[:50]}...'")
            odr_lead_results = await run_opendeepresearch_query(query, depth=1) # Initial shallow search

            if isinstance(odr_lead_results, list): # Check if we got structured data
                logger.info(f"[ProspectResearcher] Processing {len(odr_lead_results)} potential leads from query: '{query[:50]}...'")
                for lead in odr_lead_results:
                    if shutdown_event.is_set(): break
                    if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE: break

                    # Adapt parsing based on ODR service output structure
                    company_name = lead.get("name") or lead.get("company_name") or lead.get("title")
                    website = lead.get("website") or lead.get("url") or lead.get("link")
                    description = lead.get("description") or lead.get("summary") or lead.get("snippet") or lead.get("text")

                    if not company_name:
                        logger.debug("Skipping lead with no company name.")
                        continue
                    company_name = company_name.strip()
                    logger.info(f"[ProspectResearcher] Processing lead: {company_name}")

                    odr_context_for_llm = None
                    if ODR_FOR_PROSPECT_DETAILS:
                        logger.info(f"Running deeper ODR search (depth {ODR_DETAIL_DEPTH}) for {company_name}...")
                        detail_query = f"Analyze {company_name} ({website or ''}). Focus on recent business challenges, strategic shifts, product launches, funding rounds, executive changes, or stated goals relevant to market intelligence needs."
                        detail_results = await run_opendeepresearch_query(detail_query, depth=ODR_DETAIL_DEPTH)

                        if isinstance(detail_results, str):
                            if detail_results.startswith("Error:"):
                                logger.warning(f"Detailed ODR search failed for {company_name}: {detail_results}")
                                odr_context_for_llm = description # Fallback
                            else:
                                odr_context_for_llm = detail_results[:2000] # Use raw text
                        elif isinstance(detail_results, list) and detail_results:
                            odr_context_for_llm = "\n".join([
                                f"- {item.get('name', item.get('title', ''))}: {item.get('description', item.get('snippet', ''))[:200]}"
                                for item in detail_results[:5]
                            ])
                        else:
                            odr_context_for_llm = description # Fallback

                    else:
                        odr_context_for_llm = description

                    # Infer pain points using LLM (uses the same client, hence same proxy)
                    inferred_pain = await infer_pain_points_llm(client, company_name, description, odr_context_for_llm)

                    # Attempt to find contact email using LLM guessing (uses same client/proxy)
                    contact_email = await find_contact_llm_guess(client, company_name, website)

                    # Validate email if enabled and found (uses same client/proxy)
                    is_valid_email = True
                    if contact_email and EMAIL_VALIDATION_ENABLED:
                        is_valid_email = await validate_email_api(client, contact_email)

                    prospect_status = "NEW"
                    email_to_save = contact_email
                    if not contact_email:
                        prospect_status = "NEW"
                    elif not is_valid_email:
                        prospect_status = "INVALID_EMAIL"
                        email_to_save = None

                    # Save prospect using isolated session
                    try:
                        async with get_worker_session() as prospect_session:
                            created = await crud.create_or_update_prospect(
                                db=prospect_session,
                                company_name=company_name,
                                email=email_to_save,
                                website=website,
                                pain_point=inferred_pain,
                                source=f"odr_{query[:30].replace(' ', '_')}",
                                status_if_new=prospect_status
                            )
                            if created:
                                await prospect_session.commit()
                                prospects_added_this_cycle += 1
                                logger.info(f"Prospect '{created.company_name}' (ID: {created.prospect_id}) saved/updated. Status: {created.status}. Total added this cycle: {prospects_added_this_cycle}")
                            else:
                                logger.info(f"Prospect {company_name} not created or updated.")
                    except Exception as e:
                        logger.error(f"Error saving prospect {company_name} to DB: {e}", exc_info=True)

                    await asyncio.sleep(random.uniform(0.5, 1.5)) # Faster delay between leads

            elif isinstance(odr_lead_results, str): # ODR service failed or returned raw text/error
                logger.error(f"ODR lead generation query failed or returned non-JSON for query '{query[:50]}...'. Error/Output: {odr_lead_results[:500]}...")
            else:
                 logger.info(f"No initial leads returned from ODR query: '{query[:50]}...'")

            # Faster delay between different ODR queries
            await asyncio.sleep(random.uniform(2.0, 5.0))

    except Exception as e:
        logger.error(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}", exc_info=True)
    finally:
        if client: await client.aclose()

    logger.info(f"[ProspectResearcher] Prospecting cycle finished. Added/Updated {prospects_added_this_cycle} prospects this cycle.")