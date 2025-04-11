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
SIGNAL_ANALYSIS_TIMEOUT = 60 # Seconds to wait for LLM inference

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


async def process_llm_brainstorm_signals(client: httpx.AsyncClient, db: AsyncSession) -> int:
    """Generates prospect ideas using LLM brainstorming."""
    added_count = 0
    print("[ProspectResearcher] Starting LLM brainstorming for prospects...")

    # Define criteria for LLM brainstorming
    # TODO: Make these criteria dynamic or configurable
    criteria = "B2B SaaS companies in the marketing technology (MarTech) or sales technology space that have announced Series A or Series B funding rounds in the last 6-12 months."
    prompt = f"""
    List 15 distinct company names that fit the following criteria: {criteria}.
    Focus on companies likely to benefit from competitive analysis or market research reports.
    Format the output as a simple JSON list of strings, like: ["Company A Inc.", "Company B Solutions", "Company C Tech"].
    Output ONLY the JSON list.
    """

    # Use shared call_llm_api from agent_utils
    llm_response = await call_llm_api(client, prompt, model=settings.STANDARD_REPORT_MODEL or "google/gemini-1.5-flash-latest")

    company_names = []
    if llm_response:
        if isinstance(llm_response, list):
            company_names = [name for name in llm_response if isinstance(name, str)]
        elif isinstance(llm_response.get("raw_inference"), str):
             # Try parsing raw inference if LLM didn't return clean JSON list
             try:
                 raw_list = json.loads(llm_response["raw_inference"])
                 if isinstance(raw_list, list):
                     company_names = [name for name in raw_list if isinstance(name, str)]
             except json.JSONDecodeError:
                 # Fallback: Try regex to extract names if JSON fails (less reliable)
                 print("[ProspectResearcher] LLM brainstorm response not valid JSON list, attempting regex extraction.")
                 # Example regex (adjust as needed): find quoted strings or lines starting with '-'
                 potential_names = re.findall(r'["\']([^"\']+)["\']|^- (.*)', llm_response["raw_inference"], re.MULTILINE)
                 company_names = [match[0] or match[1] for match in potential_names if match[0] or match[1]]

    if not company_names:
        print("[ProspectResearcher] LLM brainstorming did not yield company names.")
        return 0

    print(f"[ProspectResearcher] LLM brainstormed {len(company_names)} potential companies.")

    for company_name in company_names:
        if added_count >= MAX_PROSPECTS_PER_CYCLE: break
        if shutdown_event.is_set(): break # Check for shutdown signal

        company_name = company_name.strip()
        if not company_name: continue

        print(f"[ProspectResearcher] Processing brainstormed company: {company_name}")

        # Attempt to find website (simple heuristic or another LLM call)
        # For now, we'll skip website finding to focus on email guessing if needed
        website = None
        # TODO: Implement website finding if needed for email guessing strategy

        # Infer pain points
        inferred_pain = await infer_pain_points_llm(client, company_name)

        # Attempt to find contact email using LLM guessing
        contact_email = await find_contact_llm_guess(client, company_name, website)

        # Save prospect
        try:
            created = await crud.create_prospect(
                db=db,
                company_name=company_name,
                email=contact_email, # Might be None
                website=website, # Might be None
                pain_point=inferred_pain,
                source="llm_brainstorm"
            )
            if created:
                # Commit immediately after adding one prospect? Or batch commit?
                # Committing here ensures data is saved even if subsequent steps fail.
                await db.commit()
                added_count += 1
                print(f"[ProspectResearcher] Prospect '{created.company_name}' added from LLM brainstorm (ID: {created.prospect_id}). Total added: {added_count}")
            else:
                await db.rollback() # Duplicate or other issue during creation
        except Exception as e:
            print(f"[ProspectResearcher] Error saving prospect from LLM brainstorm ({company_name}) to DB: {e}")
            await db.rollback()

        await asyncio.sleep(random.uniform(0.5, 1.5)) # Small delay

    return added_count


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """
    Runs a single cycle of prospecting using LLM brainstorming.
    """
    print("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client = await get_httpx_client() # Get shared client

    try:
        # Use LLM Brainstorming as the primary source
        if not shutdown_event.is_set():
            added = await process_llm_brainstorm_signals(client, db)
            prospects_added_this_cycle += added
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE:
                print("[ProspectResearcher] Reached max prospects for this cycle via LLM Brainstorming.")

        # TODO: Add calls to other prospecting methods here if implemented
        # (e.g., process_scraping_signals, process_api_signals)

    except Exception as e:
        print(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}")
        # Log error
    finally:
        await client.aclose()

    print(f"[ProspectResearcher] Prospecting cycle finished. Added {prospects_added_this_cycle} new prospects.")