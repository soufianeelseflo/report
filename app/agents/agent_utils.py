# autonomous_agency/app/agents/agent_utils.py
import httpx
from typing import Optional, Dict, Any, List
import json
import os
import re # Import regex
import random # To potentially select keys later
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception, RetryError # Use retry_if_exception
import traceback # Import traceback at the top
import datetime # For rate limit cooldown

import asyncio # For async operations like key loading
from sqlalchemy.ext.asyncio import AsyncSession

# Corrected relative import for package structure
# Assuming Acumenis is the root package name based on later files
try:
    from Acumenis.app.core.config import settings # Adjusted path
    from Acumenis.app.db.base import get_worker_session # Function to get DB session for background tasks
    from Acumenis.app.db import crud # Import CRUD functions
    from Acumenis.app.core.security import decrypt_data # Ensure decrypt is imported
except ImportError:
    print("[AgentUtils] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db.base import get_worker_session
    from app.db import crud
    from app.core.security import decrypt_data

# Use tenacity for retrying API calls
RETRY_WAIT = wait_fixed(3) # Wait 3 seconds between retries
RETRY_ATTEMPTS = 3 # Retry up to 3 times
LLM_TIMEOUT = 120.0 # Increased timeout for potentially long generations (e.g., report analysis)
RATE_LIMIT_COOLDOWN_SECONDS = 60 # Cooldown period for rate-limited keys

# --- Global variables for API keys ---
# Store more key info for better management
AVAILABLE_API_KEYS: List[Dict[str, Any]] = [] # Stores {'key': decrypted_key, 'id': db_id, 'rate_limited_until': datetime | None}
CURRENT_KEY_INDEX: int = 0
_keys_loaded: bool = False # Flag to track if keys have been loaded from DB
_key_load_lock = asyncio.Lock() # Lock to prevent concurrent loading attempts

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
        valid_proxies = [p for p in settings.PROXY_LIST if re.match(r"^(http|https|socks\d?)://", p)]
        if valid_proxies:
            # Use a random proxy from the list for the client for better distribution
            selected_proxy = random.choice(valid_proxies)
            proxy_display = selected_proxy.split('@')[-1] if '@' in selected_proxy else selected_proxy
            print(f"[Proxy] Using random proxy from PROXY_LIST for client: {proxy_display}")
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
        # Add other common headers if needed
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
    Handles decryption and potential errors, stores key ID and rate limit info.
    """
    global AVAILABLE_API_KEYS, CURRENT_KEY_INDEX, _keys_loaded
    async with _key_load_lock: # Ensure only one load happens at a time
        print("[API Keys] Attempting to refresh API keys from database...")
        session: AsyncSession = None
        new_keys_loaded = []
        try:
            session = await get_worker_session()
            provider = settings.LLM_PROVIDER or "openrouter"
            # crud.get_active_api_keys_with_details should return list of dicts {id, key_encrypted, rate_limited_until}
            loaded_key_details = await crud.get_active_api_keys_with_details(session, provider=provider)

            if loaded_key_details:
                for key_detail in loaded_key_details:
                    try:
                        decrypted_key = decrypt_data(key_detail['api_key_encrypted'])
                        if decrypted_key:
                            new_keys_loaded.append({
                                'id': key_detail['id'],
                                'key': decrypted_key,
                                'rate_limited_until': key_detail.get('rate_limited_until') # Get timestamp if exists
                            })
                        else:
                            print(f"[API Keys] Decryption failed for key ID {key_detail['id']}. Skipping.")
                            # Optionally mark as error in DB here?
                            # await crud.set_api_key_status_by_id(session, key_detail['id'], 'error', 'Decryption failed during load')
                    except Exception as decrypt_err:
                        print(f"[API Keys] Error decrypting key ID {key_detail['id']}: {decrypt_err}. Skipping.")
                        # await crud.set_api_key_status_by_id(session, key_detail['id'], 'error', f'Decryption exception: {decrypt_err}')

                print(f"[API Keys] Successfully loaded and decrypted {len(new_keys_loaded)} active keys for provider '{provider}'.")
                # Commit any status changes made during loading (like marking decryption errors)
                # await session.commit() # Commit error status updates if any
            else:
                print(f"[API Keys] No active keys found in DB for provider '{provider}'.")
                # Fallback: Use the key from .env ONLY if DB is empty AND it's OpenRouter
                if settings.OPENROUTER_API_KEY and provider == "openrouter":
                    print("[API Keys] Using fallback API key from settings.")
                    # Add with a placeholder ID or handle differently? For now, just the key.
                    new_keys_loaded = [{'id': None, 'key': settings.OPENROUTER_API_KEY, 'rate_limited_until': None}]
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
            if session: await session.rollback() # Rollback on general load error
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
            traceback.print_exc() # Log full trace for loop errors
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

def get_next_api_key() -> Optional[Dict[str, Any]]:
    """
    Selects the next valid API key dict {'id', 'key', 'rate_limited_until'} for use.
    Rotates through keys, skipping those that are currently rate-limited.
    Returns None if no valid keys are available.
    """
    global CURRENT_KEY_INDEX

    if not _keys_loaded:
        print("[LLM Call] CRITICAL: API keys have not been loaded yet. Call load_and_update_api_keys() or ensure startup load.")
        return None

    if not AVAILABLE_API_KEYS:
        # print("[LLM Call] Warning: No API keys available in the loaded list.") # Reduce noise
        return None

    num_keys = len(AVAILABLE_API_KEYS)
    start_index = CURRENT_KEY_INDEX

    for i in range(num_keys):
        current_check_index = (start_index + i) % num_keys
        key_info = AVAILABLE_API_KEYS[current_check_index]

        # Check if key is rate-limited
        rate_limited_until = key_info.get('rate_limited_until')
        is_rate_limited = rate_limited_until and datetime.datetime.now(datetime.timezone.utc) < rate_limited_until

        if not is_rate_limited:
            # Found a valid key
            CURRENT_KEY_INDEX = (current_check_index + 1) % num_keys # Point to the next one for the next call
            # print(f"[LLM Call] Using API Key ID: {key_info.get('id', 'N/A')} (Index: {current_check_index})") # Debugging
            return key_info # Return the whole dict

        # else: print(f"[LLM Call] Skipping rate-limited key ID: {key_info.get('id', 'N/A')}")

    # If we loop through all keys and find none are valid
    print("[LLM Call] Warning: All available API keys are currently rate-limited or none exist.")
    return None

# Custom retry condition: Retry on 429 (Rate Limit) and 5xx (Server Errors)
def should_retry(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on rate limits and server errors
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    # Retry on general request errors (network issues, timeouts)
    return isinstance(exception, httpx.RequestError)

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception(should_retry))
async def call_llm_api(
    client: httpx.AsyncClient,
    prompt: str,
    model: Optional[str] = None,
    image_data: Optional[str] = None # Optional base64 encoded image data
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (specifically handles OpenRouter).
    Uses the specified model, falling back to a default if needed.
    Handles API key selection, rotation, rate limiting, and retries.
    Attempts to parse JSON, falls back to raw inference string.
    Supports optional image data for multimodal calls.
    """
    key_info = get_next_api_key()
    if not key_info:
        print("[LLM Call] CRITICAL: No valid API Key available for LLM call.")
        return None # No key, cannot proceed

    api_key = key_info['key']
    key_id = key_info.get('id') # DB ID for status updates

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
            "HTTP-Referer": settings.HTTP_REFERER or settings.AGENCY_BASE_URL or "https://acumenis.ai", # Use setting or default
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME or "Acumenis", # Use setting or default
        }
        # Construct messages payload, potentially including image data
        messages = []
        content_parts = [{"type": "text", "text": prompt}]
        if image_data and selected_model and 'vision' in selected_model or 'gemini-1.5-pro' in selected_model or 'gemini-1.5-flash' in selected_model: # Basic check for vision models
            # Assuming base64 encoded image data, determine mime type (needs improvement)
            # This is a basic guess, might need more robust mime type detection
            mime_type = "image/jpeg" if image_data.startswith("/9j/") else "image/png" if image_data.startswith("iVBOR") else "image/jpeg"
            messages.append({
                "role": "user",
                "content": [
                     {"type": "text", "text": prompt},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:{mime_type};base64,{image_data}"
                         }
                     }
                ]
            })
            print(f"[LLM Call] Including image data in request for model {selected_model}")
        else:
             messages.append({"role": "user", "content": prompt})
             if image_data:
                  print(f"[LLM Call] WARNING: Image data provided but model '{selected_model}' might not support it. Sending text only.")


        payload = {
            "model": selected_model,
            "messages": messages,
            # Add optional parameters if needed
            # "temperature": 0.7,
            # "max_tokens": 4096, # Example
        }
    # Add elif blocks here for other providers if needed
    # elif settings.LLM_PROVIDER == "openai": ...
    else:
        print(f"[LLM Call] Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}. Only 'openrouter' is explicitly handled.")
        return None

    # --- Make API Call ---
    key_suffix = api_key[-4:] if api_key else 'N/A'
    # print(f"[LLM Call] Sending request to {endpoint} using model {selected_model} with key ID {key_id or 'N/A'} (...{key_suffix})") # Reduce noise
    last_exception = None
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
        result = response.json()

        # Mark key as used *after* successful call (fire-and-forget)
        if key_id: # Only if it's a key from the DB
            asyncio.create_task(crud.mark_api_key_used_by_id(key_id))

        # --- Parse Response ---
        if "error" in result:
             error_details = result["error"]
             print(f"[LLM Call] API returned error in body: {error_details}")
             # Potentially mark key as inactive based on error type here
             error_str = str(error_details).lower()
             if key_id and ('invalid api key' in error_str or 'authentication error' in error_str):
                 print(f"[LLM Call] Deactivating key ID {key_id} due to API error: {error_details.get('message', 'Invalid Key')}")
                 asyncio.create_task(crud.set_api_key_status_by_id(key_id, 'inactive', f"Invalid Key Error: {error_details.get('message', '')}"))
             return None

        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Clean potential markdown ```json ... ``` blocks
                cleaned_content = content.strip()
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE | re.DOTALL)
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
        status_code = e.response.status_code
        print(f"[LLM Call] API request failed: Status {status_code} for model {selected_model} with key ID {key_id or 'N/A'} (...{key_suffix})")
        try: error_body = e.response.json(); print(f"[LLM Call] Error Body: {error_body}")
        except: print(f"[LLM Call] Error Body (non-JSON): {e.response.text[:200]}...")

        # Handle specific status codes
        if key_id: # Only update status for keys managed in DB
            if status_code in [401, 403]: # Unauthorized or Forbidden
                 print(f"[LLM Call] Deactivating key ID {key_id} due to {status_code} error.")
                 asyncio.create_task(crud.set_api_key_status_by_id(key_id, 'inactive', f"API Error {status_code}"))
            elif status_code == 429: # Rate limited
                 print(f"[LLM Call] Rate limit hit for key ID {key_id}. Marking as rate-limited.")
                 cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=RATE_LIMIT_COOLDOWN_SECONDS)
                 asyncio.create_task(crud.set_api_key_rate_limited(key_id, cooldown_until, f"API Error {status_code}"))
                 # Update local cache immediately to prevent reuse in this cycle
                 for k in AVAILABLE_API_KEYS:
                     if k.get('id') == key_id:
                         k['rate_limited_until'] = cooldown_until
                         break

        raise e # Re-raise to trigger tenacity retry if applicable
    except httpx.RequestError as e:
        last_exception = e
        print(f"[LLM Call] Network or Request Error: {e}")
        raise e
    except RetryError as e:
        # This is caught after all retry attempts fail
        print(f"[LLM Call] API call failed after {RETRY_ATTEMPTS} attempts for model {selected_model}. Last exception: {e.last_attempt.exception()}")
        # Deactivate the key that failed the last attempt if it was a 401/403
        if key_id and isinstance(e.last_attempt.exception(), httpx.HTTPStatusError):
             status_code = e.last_attempt.exception().response.status_code
             if status_code in [401, 403]:
                  print(f"[LLM Call] Deactivating key ID {key_id} after exhausting retries due to {status_code}.")
                  asyncio.create_task(crud.set_api_key_status_by_id(key_id, 'inactive', f"API Error {status_code} after retries"))
        return None
    except Exception as e:
        last_exception = e
        print(f"[LLM Call] Unexpected error calling API: {e}")
        traceback.print_exc()
        return None