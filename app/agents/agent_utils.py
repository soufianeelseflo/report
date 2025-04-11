import httpx
from typing import Optional, Dict, Any, List
import json
import os
import re # Import regex
import random # To potentially select keys later
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception, RetryError # Use retry_if_exception
import traceback # Import traceback at the top

import asyncio # For async operations like key loading
from sqlalchemy.ext.asyncio import AsyncSession

# Corrected relative import for package structure
# Assuming Acumenis is the root package name based on later files
from Acumenis.app.core.config import settings # Adjusted path
from Acumenis.app.db.base import get_worker_session # Function to get DB session for background tasks
from Acumenis.app.db import crud # Import CRUD functions

# Use tenacity for retrying API calls
RETRY_WAIT = wait_fixed(2) # Wait 2 seconds between retries
RETRY_ATTEMPTS = 3 # Retry up to 3 times
LLM_TIMEOUT = 120.0 # Increased timeout for potentially long generations (e.g., report analysis)

# --- Global variables for API keys ---
AVAILABLE_API_KEYS: List[str] = [] # Initialize empty, will be loaded from DB
CURRENT_KEY_INDEX: int = 0
_keys_loaded: bool = False # Flag to track if keys have been loaded from DB
_key_load_lock = asyncio.Lock() # Lock to prevent concurrent loading attempts
from Acumenis.app.core.security import decrypt_data # Ensure decrypt is imported

# --- Key Refresh Task ---
_key_refresh_task: Optional[asyncio.Task] = None
_key_refresh_interval: int = 300 # Refresh every 5 minutes (300 seconds)

def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    proxy_url = settings.PROXY_URL
    if proxy_url:
        if not re.match(r"^(http|https|socks\d?)://", proxy_url):
             print(f"[Proxy] WARNING: Configured PROXY_URL '{proxy_url}' doesn't look like a valid URL.")
             return None
        return {"http://": proxy_url, "https://": proxy_url}
    # Handle PROXY_LIST if PROXY_URL is not set
    elif settings.PROXY_LIST:
        # Simple approach: use the first valid proxy from the list for the client
        # KeyAcquirer uses random selection per attempt. This provides a default.
        valid_proxies = [p for p in settings.PROXY_LIST if re.match(r"^(http|https|socks\d?)://", p)]
        if valid_proxies:
            selected_proxy = valid_proxies[0]
            print(f"[Proxy] Using first valid proxy from PROXY_LIST for client: {selected_proxy}")
            return {"http://": selected_proxy, "https://": selected_proxy}
        else:
            print("[Proxy] PROXY_LIST configured but contains no valid URLs.")
            return None
    return None

async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies."""
    proxies = get_proxy_map()
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=40)
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=15.0)

    headers = {
        "User-Agent": f"AcumenisAgent/1.0 ({settings.AGENCY_BASE_URL or 'http://localhost'})",
    }

    client = httpx.AsyncClient(
        proxies=proxies,
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        http2=True
    )
    return client

async def load_and_update_api_keys():
    """
    Loads active API keys from the database and updates the global list.
    Handles decryption and potential errors.
    """
    global AVAILABLE_API_KEYS, CURRENT_KEY_INDEX, _keys_loaded
    async with _key_load_lock: # Ensure only one load happens at a time
        print("[API Keys] Attempting to refresh API keys from database...")
        session: AsyncSession = None
        new_keys_loaded = []
        try:
            session = await get_worker_session()
            provider = settings.LLM_PROVIDER or "openrouter"
            # crud.get_active_api_keys now returns decrypted keys directly
            loaded_keys = await crud.get_active_api_keys(session, provider=provider)

            if loaded_keys:
                new_keys_loaded = loaded_keys
                print(f"[API Keys] Successfully loaded {len(new_keys_loaded)} active keys for provider '{provider}'.")
            else:
                print(f"[API Keys] No active keys found in DB for provider '{provider}'.")
                # Fallback: Use the key from .env ONLY if DB is empty AND it's OpenRouter
                if settings.OPENROUTER_API_KEY and provider == "openrouter":
                    print("[API Keys] Using fallback API key from settings.")
                    new_keys_loaded = [settings.OPENROUTER_API_KEY]
                else:
                    new_keys_loaded = [] # Ensure it's empty if DB and fallback fail

            # Atomically update the global list and reset index
            AVAILABLE_API_KEYS = new_keys_loaded
            CURRENT_KEY_INDEX = 0 if AVAILABLE_API_KEYS else -1 # Reset index or set to -1 if empty
            _keys_loaded = True # Mark that a load attempt was made

        except Exception as e:
            print(f"[API Keys] Error loading API keys from database: {e}")
            traceback.print_exc()
            # Keep existing keys if load fails? Or clear? Keep for resilience.
            print(f"[API Keys] Keeping previously loaded keys ({len(AVAILABLE_API_KEYS)}) due to load error.")
            _keys_loaded = True # Still mark as loaded attempt
        finally:
            if session:
                await session.close()

async def _key_refresh_loop():
    """Background loop to periodically refresh API keys."""
    print("[API Keys] Starting background key refresh loop...")
    while True:
        try:
            await load_and_update_api_keys()
        except Exception as e:
            print(f"[API Keys] Error in refresh loop: {e}")
        await asyncio.sleep(_key_refresh_interval)

def start_key_refresh_task():
    """Starts the background key refresh task if not already running."""
    global _key_refresh_task
    if _key_refresh_task is None or _key_refresh_task.done():
        _key_refresh_task = asyncio.create_task(_key_refresh_loop())
        print("[API Keys] Background key refresh task started.")
    else:
        print("[API Keys] Background key refresh task already running.")

# Call this on application startup (e.g., in main.py's startup event)
# start_key_refresh_task() # Moved to main.py startup

def get_next_api_key() -> Optional[str]:
    """
    Selects the next API key for use from the globally loaded list.
    Simple rotation. Returns None if no keys are available.
    Relies on the background task to keep AVAILABLE_API_KEYS updated.
    """
    global CURRENT_KEY_INDEX

    if not _keys_loaded:
        # This should ideally not happen if keys are loaded on startup
        print("[LLM Call] CRITICAL: API keys have not been loaded yet. Call load_and_update_api_keys() or ensure startup load.")
        # Attempt a synchronous load as a last resort? Risky.
        # asyncio.run(load_and_update_api_keys()) # Avoid running async code synchronously here
        return None

    if not AVAILABLE_API_KEYS:
        # print("[LLM Call] Warning: No API keys available in the loaded list.") # Reduce noise
        return None

    # Rotate through available keys
    # Ensure index is valid even if list shrank since last call
    if CURRENT_KEY_INDEX >= len(AVAILABLE_API_KEYS) or CURRENT_KEY_INDEX < 0:
        CURRENT_KEY_INDEX = 0

    if not AVAILABLE_API_KEYS: # Double check after index reset
        return None

    key = AVAILABLE_API_KEYS[CURRENT_KEY_INDEX]
    # print(f"[LLM Call] Using API Key Index: {CURRENT_KEY_INDEX} of {len(AVAILABLE_API_KEYS)}") # Debugging
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(AVAILABLE_API_KEYS)

    # Mark key as used asynchronously after the call succeeds
    # asyncio.create_task(crud.mark_api_key_used(db_session_if_available, key_value=key, provider=settings.LLM_PROVIDER))

    return key

# Custom retry condition: Retry on 429 (Rate Limit) and 5xx (Server Errors)
def should_retry(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on rate limits and server errors
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    # Retry on general request errors (network issues, timeouts)
    return isinstance(exception, httpx.RequestError)

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception(should_retry))
async def call_llm_api(client: httpx.AsyncClient, prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (specifically handles OpenRouter).
    Uses the specified model, falling back to a default if needed.
    Handles API key selection, rotation, and retries.
    Attempts to parse JSON, falls back to raw inference string.
    """
    api_key = get_next_api_key()
    if not api_key:
        print("[LLM Call] CRITICAL: No API Key available for LLM call.")
        return None # No key, cannot proceed

    endpoint = None
    headers = {}
    # Determine model: Use provided, fallback to standard, then premium as last resort
    selected_model = model or settings.STANDARD_REPORT_MODEL or settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-flash-latest" # Ensure a final fallback

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.HTTP_REFERER or settings.AGENCY_BASE_URL or "https://acumenis.ai",
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME or "Acumenis",
        }
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            # Add optional parameters if needed
            # "temperature": 0.7,
        }
    # Add elif blocks here for other providers if needed
    else:
        print(f"[LLM Call] Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}. Only 'openrouter' is explicitly handled.")
        return None

    # --- Make API Call ---
    # print(f"[LLM Call] Sending request to {endpoint} using model {selected_model} with key ending ...{api_key[-4:]}") # Reduce noise
    last_exception = None
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
        result = response.json()

        # Mark key as used *after* successful call
        # Need a DB session here - best handled by the calling agent function
        # For now, fire-and-forget task (requires crud function to get its own session)
        # asyncio.create_task(crud.mark_api_key_used_by_value(api_key, settings.LLM_PROVIDER)) # Needs new crud func

        # --- Parse Response ---
        if "error" in result:
             error_details = result["error"]
             print(f"[LLM Call] API returned error in body: {error_details}")
             # Potentially mark key as inactive based on error type here
             # Example: if 'invalid api key' in str(error_details).lower():
             #    asyncio.create_task(crud.set_api_key_inactive(api_key, settings.LLM_PROVIDER, f"Invalid Key Error: {error_details}"))
             return None

        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Clean potential markdown ```json ... ``` blocks
                cleaned_content = content.strip()
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = cleaned_content

                # Attempt to parse as JSON
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError:
                # If not JSON, return the raw string content wrapped in a dict
                return {"raw_inference": content}
        else:
             print(f"[LLM Call] No content found in LLM response choices: {result}")
             return None

    except httpx.HTTPStatusError as e:
        last_exception = e
        print(f"[LLM Call] API request failed: Status {e.response.status_code} for model {selected_model} with key ...{api_key[-4:]}")
        try: error_body = e.response.json(); print(f"[LLM Call] Error Body: {error_body}")
        except: print(f"[LLM Call] Error Body (non-JSON): {e.response.text[:200]}...")

        # Mark potentially bad keys based on status code
        if e.response.status_code in [401, 403]: # Unauthorized or Forbidden
             print(f"[LLM Call] Deactivating key ...{api_key[-4:]} due to {e.response.status_code} error.")
             # Fire-and-forget task to update DB status
             asyncio.create_task(crud.set_api_key_inactive(api_key, settings.LLM_PROVIDER, f"API Error {e.response.status_code}"))
        elif e.response.status_code == 429: # Rate limited
             print(f"[LLM Call] Rate limit hit for key ...{api_key[-4:]}.")
             # Optionally mark as 'rate_limited' status?
             # asyncio.create_task(crud.set_api_key_status_by_value(api_key, settings.LLM_PROVIDER, 'rate_limited', 'API Error 429'))

        raise e # Re-raise to trigger tenacity retry if applicable
    except httpx.RequestError as e:
        last_exception = e
        print(f"[LLM Call] Network or Request Error: {e}")
        raise e
    except RetryError as e:
        # This is caught after all retry attempts fail
        print(f"[LLM Call] API call failed after {RETRY_ATTEMPTS} attempts for model {selected_model}. Last exception: {e.last_attempt.exception()}")
        # Deactivate the key that failed the last attempt if it was a 401/403
        if isinstance(e.last_attempt.exception(), httpx.HTTPStatusError):
             status_code = e.last_attempt.exception().response.status_code
             if status_code in [401, 403]:
                  print(f"[LLM Call] Deactivating key ...{api_key[-4:]} after exhausting retries due to {status_code}.")
                  asyncio.create_task(crud.set_api_key_inactive(api_key, settings.LLM_PROVIDER, f"API Error {status_code} after retries"))
        return None
    except Exception as e:
        last_exception = e
        print(f"[LLM Call] Unexpected error calling API: {e}")
        traceback.print_exc()
        return None