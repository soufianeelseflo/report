import asyncio
import random
import json
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
import datetime

from sqlalchemy.ext.asyncio import AsyncSession
import httpx # Use the client from agent_utils
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from app.core.config import settings
from app.db import crud, models
from app.agents.agent_utils import get_httpx_client

# --- Configuration for Signal Detection ---
# Focus on high-value signals, not generic keywords
# These might require specific API integrations (free tiers where possible)
SIGNAL_SOURCES = {
    "news_api": { # Example: NewsAPI.org free tier (limited requests)
        "enabled": False, # Set to True if API key is available
        "api_key": os.environ.get("NEWS_API_KEY"),
        "keywords": ["series A funding", "series B funding", "product launch", "executive change", "data breach", "market expansion", "acquisition"],
        "endpoint": "https://newsapi.org/v2/everything" # Or /top-headlines
    },
    "alpha_vantage": { # Example: Alpha Vantage free tier (limited requests)
        "enabled": False, # Set to True if API key is available
        "api_key": os.environ.get("ALPHA_VANTAGE_KEY"),
        "functions": ["NEWS_SENTIMENT"], # Function to get news/sentiment for tickers
        "target_tickers": ["IBM", "AAPL", "MSFT"] # Needs dynamic target identification
    },
    # Add more sources: Reddit keyword monitoring (Pushshift API?), specific industry forums (requires custom scraping logic)
}

# LLM Configuration (Conceptual - replace with actual API call logic)
LLM_INFERENCE_ENDPOINT = "YOUR_LLM_API_ENDPOINT" # e.g., OpenAI, Claude, Gemini free tier endpoint
LLM_API_KEY = os.environ.get("YOUR_LLM_API_KEY")
LLM_HEADERS = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

MAX_PROSPECTS_PER_CYCLE = 20 # Limit prospects generated per run
SIGNAL_ANALYSIS_TIMEOUT = 60 # Seconds to wait for LLM inference

# Use tenacity for retrying API calls
RETRY_WAIT = wait_fixed(3)
RETRY_ATTEMPTS = 3

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception_type(httpx.RequestError))
async def call_llm_api(client: httpx.AsyncClient, prompt: str) -> Optional[Dict[str, Any]]:
    """Generic function to call a configured LLM API for inference."""
    if not LLM_INFERENCE_ENDPOINT or not LLM_API_KEY:
        print("[ProspectResearcher] LLM API endpoint or key not configured. Skipping inference.")
        return None

    # --- Adapt payload structure based on your chosen LLM API ---
    payload = {
        "model": "gpt-4o-mini", # Or equivalent powerful 2025 model available on free tier
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 250,
        "temperature": 0.7,
    }
    # --- End API specific payload ---

    try:
        response = await client.post(LLM_INFERENCE_ENDPOINT, headers=LLM_HEADERS, json=payload, timeout=SIGNAL_ANALYSIS_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        # --- Adapt response parsing based on your chosen LLM API ---
        # Example for OpenAI-like response:
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            # Attempt to parse if LLM returns structured JSON, otherwise return raw text
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw_inference": content}
        # --- End API specific parsing ---
        return None
    except httpx.HTTPStatusError as e:
        print(f"[ProspectResearcher] LLM API request failed: {e.response.status_code} - {e.response.text}")
        # Handle specific errors like rate limits (429) if needed
        if e.response.status_code == 429:
            print("[ProspectResearcher] LLM Rate limit hit. Consider slowing down requests.")
            # Optionally raise a specific exception to stop retrying immediately for rate limits
        return None
    except Exception as e:
        print(f"[ProspectResearcher] Error calling LLM API: {e}")
        return None


async def infer_pain_points_from_signal(client: httpx.AsyncClient, company_name: str, signal_description: str) -> Optional[str]:
    """Uses LLM to infer potential pain points based on a detected signal."""
    prompt = f"""
    Analyze the following signal regarding the company '{company_name}':
    Signal: "{signal_description}"

    Based *only* on this signal, infer 1-2 specific, high-probability business pain points or challenges this company might be facing right now related to data analysis, market intelligence, competitive research, or strategic reporting.
    Be concise and action-oriented. Frame them as potential needs. Avoid generic statements. If no clear pain point can be inferred, respond with "None".

    Example format:
    - Need for deeper competitive analysis following recent funding.
    - Requirement for data-driven market entry strategy validation.
    """
    inference_result = await call_llm_api(client, prompt)
    if inference_result and isinstance(inference_result.get("raw_inference"), str):
        inferred_text = inference_result["raw_inference"].strip()
        if inferred_text.lower() != "none" and len(inferred_text) > 5:
             print(f"[ProspectResearcher] Inferred pain points for {company_name}: {inferred_text}")
             return inferred_text
    return None

async def find_contact_heuristic(client: httpx.AsyncClient, company_name: str, website: Optional[str]) -> Optional[str]:
    """
    Placeholder/Heuristic: Tries to find a contact email.
    This remains challenging. Could involve:
    1. Scraping 'Contact Us' / 'About Us' pages (if website provided).
    2. Using LLM to guess common email patterns (e.g., first.last@domain.com) - unreliable.
    3. Integrating with *paid* email finder APIs (violates free trial constraint).
    For now, returns a placeholder or None.
    """
    if website:
        # Basic: Check if email is on homepage (already done in previous version's _extract...)
        # Advanced: Navigate to 'Contact' page and scrape (requires more complex scraping logic)
        pass
    # Placeholder - in a real scenario, this needs a robust (likely paid) solution or manual step
    print(f"[ProspectResearcher] Placeholder: Could not automatically find contact for {company_name}.")
    return None # Or return a placeholder like "info@domain.com" if domain is known


async def process_news_api_signals(client: httpx.AsyncClient, db: AsyncSession) -> int:
    """Fetches signals from NewsAPI and processes them."""
    config = SIGNAL_SOURCES["news_api"]
    if not config["enabled"] or not config["api_key"]:
        return 0

    added_count = 0
    headers = {"X-Api-Key": config["api_key"]}
    # Combine keywords for a single query if possible, or make separate requests
    query = " OR ".join([f'"{kw}"' for kw in config["keywords"]])
    # Look for recent news (e.g., last 3 days)
    from_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50, # Max 100 for free/dev tier? Check limits
        "from": from_date,
    }

    try:
        response = await client.get(config["endpoint"], headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        print(f"[ProspectResearcher] NewsAPI found {len(articles)} articles for keywords.")

        for article in articles:
            if added_count >= MAX_PROSPECTS_PER_CYCLE: break

            title = article.get("title")
            description = article.get("description")
            url = article.get("url")
            source_name = article.get("source", {}).get("name")

            # Use LLM to extract company name from title/description (can be noisy)
            # This is a crucial step where AI helps structure unstructured signal data
            company_extract_prompt = f"Extract the primary company name mentioned in this news headline and description. Respond with only the company name or 'None'.\nHeadline: {title}\nDescription: {description}"
            company_info = await call_llm_api(client, company_extract_prompt)
            company_name = company_info.get("raw_inference", "None").strip() if company_info else "None"

            if company_name != "None" and len(company_name) > 1:
                signal_desc = f"News Signal ({source_name}): {title}"
                print(f"[ProspectResearcher] Processing signal for company: {company_name} -> {title}")

                # Infer pain points using LLM
                inferred_pain = await infer_pain_points_from_signal(client, company_name, signal_desc)

                # Attempt to find website/contact (placeholder)
                # Could use another LLM call: "Find the official website URL for company '{company_name}'."
                website = None # Placeholder
                contact_email = await find_contact_heuristic(client, company_name, website)

                # Save prospect if company name found (even without email initially)
                try:
                    created = await crud.create_prospect(
                        db=db,
                        company_name=company_name,
                        email=contact_email, # Might be None
                        website=website, # Might be None
                        pain_point=inferred_pain,
                        source=f"news_api_{url}"
                    )
                    if created:
                        await db.commit()
                        added_count += 1
                        print(f"[ProspectResearcher] Prospect '{created.company_name}' added from NewsAPI signal (ID: {created.prospect_id}). Total added: {added_count}")
                    else:
                        await db.rollback() # Duplicate detected by CRUD
                except Exception as e:
                    print(f"[ProspectResearcher] Error saving prospect from NewsAPI signal ({company_name}) to DB: {e}")
                    await db.rollback()
            else:
                 print(f"[ProspectResearcher] Could not extract company name from signal: {title}")

            # Add small delay between processing articles to avoid hitting LLM limits too fast
            await asyncio.sleep(random.uniform(1, 3))

    except Exception as e:
        print(f"[ProspectResearcher] Error processing NewsAPI signals: {e}")

    return added_count


async def run_prospecting_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """
    Runs a single cycle of prospecting using configured signal sources.
    """
    print("[ProspectResearcher] Starting prospecting cycle...")
    prospects_added_this_cycle = 0
    client = await get_httpx_client()

    try:
        # Process NewsAPI signals
        if not shutdown_event.is_set() and SIGNAL_SOURCES["news_api"]["enabled"]:
            added = await process_news_api_signals(client, db)
            prospects_added_this_cycle += added
            if prospects_added_this_cycle >= MAX_PROSPECTS_PER_CYCLE:
                 print("[ProspectResearcher] Reached max prospects for this cycle via NewsAPI.")

        # Process Alpha Vantage signals (if enabled and not maxed out)
        # if not shutdown_event.is_set() and prospects_added_this_cycle < MAX_PROSPECTS_PER_CYCLE and SIGNAL_SOURCES["alpha_vantage"]["enabled"]:
            # Implement similar logic for Alpha Vantage API calls
            # Requires identifying target stock tickers dynamically or having a predefined list
            # added = await process_alpha_vantage_signals(client, db)
            # prospects_added_this_cycle += added
            # ...

        # Process other signal sources...

    except Exception as e:
        print(f"[ProspectResearcher] Unexpected error during prospecting cycle: {e}")
        # Log error
    finally:
        await client.aclose()

    print(f"[ProspectResearcher] Prospecting cycle finished. Added {prospects_added_this_cycle} new prospects.")