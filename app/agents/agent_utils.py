# autonomous_agency/app/agents/agent_utils.py
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

def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    proxy_url = settings.PROXY_URL
    if proxy_url:
        # Basic validation - check if it looks like a URL structure
        if not re.match(r"^(http|https|socks\d?)://", proxy_url):
             print(f"[Proxy] WARNING: Configured PROXY_URL '{proxy_url}' doesn't look like a valid URL. Check format (e.g., http://user:pass@host:port).")
             # Optionally return None or raise an error if format is critical
             # return None
        return {"http://": proxy_url, "https://": proxy_url}
    return None

async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies."""
    proxies = get_proxy_map()
    # Increased limits for potentially higher concurrency
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=40)
    # Connect timeout remains reasonable, main timeout is longer
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=15.0) # Use LLM_TIMEOUT directly

    headers = {
        "User-Agent": f"AcumenisAgent/1.0 ({settings.AGENCY_BASE_URL or 'http://localhost'})", # Identify bot
    }
    # Add proxy auth headers if needed (some proxies use headers instead of URL format)
    # if settings.PROXY_USER and settings.PROXY_PASSWORD and not settings.PROXY_URL:
    #     # Example for basic auth header proxy
    #     auth = base64.b64encode(f"{settings.PROXY_USER}:{settings.PROXY_PASSWORD}".encode()).decode()
    #     headers['Proxy-Authorization'] = f'Basic {auth}'

    client = httpx.AsyncClient(
        proxies=proxies,
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        http2=True # Enable HTTP/2 if supported by endpoints
    )
    return client
async def load_and_update_api_keys():
    """
    Loads active API keys from the database and updates the global list.
    This should be called periodically or on startup by an external process/worker.
    """
    global AVAILABLE_API_KEYS, CURRENT_KEY_INDEX, _keys_loaded
    async with _key_load_lock: # Ensure only one load happens at a time
        print("[API Keys] Attempting to refresh API keys from database...")
        session: AsyncSession = None
        try:
            session = await get_worker_session()
            # Load keys specifically for the configured provider (e.g., "openrouter")
            provider = settings.LLM_PROVIDER or "openrouter" # Default to openrouter if not set
            # This now returns a list of decrypted key strings
            loaded_keys = await crud.get_active_api_keys(session, provider=provider)

            if loaded_keys:
                AVAILABLE_API_KEYS = loaded_keys
                CURRENT_KEY_INDEX = 0 # Reset index when keys are refreshed
                _keys_loaded = True # Mark that a load attempt (successful or not) was made
                print(f"[API Keys] Successfully loaded {len(AVAILABLE_API_KEYS)} active keys for provider '{provider}'.")
            else:
                print(f"[API Keys] No active keys found in DB for provider '{provider}'.")
                # Fallback: Use the key from .env if available and DB is empty
                if settings.OPENROUTER_API_KEY and provider == "openrouter":
                    print("[API Keys] Using fallback API key from settings.")
                    AVAILABLE_API_KEYS = [settings.OPENROUTER_API_KEY]
                    CURRENT_KEY_INDEX = 0
                else:
                    AVAILABLE_API_KEYS = [] # Ensure it's empty if DB and fallback fail
                _keys_loaded = True # Mark that a load attempt was made

        except Exception as e:
            print(f"[API Keys] Error loading API keys from database: {e}")
            # Keep existing keys if load fails? Or clear them? Clearing might be safer.
            # AVAILABLE_API_KEYS = [] # Optional: Clear keys on DB error
            # _keys_loaded = False # Allow retry on next call? Risky if DB is down.
            # For now, just log the error and keep whatever keys were there (or empty list).
            # Ensure _keys_loaded is True so get_next_api_key doesn't try to load itself.
            _keys_loaded = True
        finally:
            if session:
                await session.close()

def get_next_api_key() -> Optional[str]:
    """
    Selects the next API key for use from the globally loaded list.
    Simple rotation. Returns None if no keys are available in the list.
    IMPORTANT: Assumes `load_and_update_api_keys` has been called previously
    (e.g., on startup or periodically) to populate AVAILABLE_API_KEYS.
    """
    global CURRENT_KEY_INDEX

    if not AVAILABLE_API_KEYS:
        # If the list is empty, it means either loading failed, returned no keys,
        # or hasn't been called yet. Log a warning if loading was attempted.
        if _keys_loaded:
            print("[LLM Call] Warning: No API keys available in the loaded list.")
        else:
            # This case should ideally be avoided by ensuring keys are loaded on startup.
            print("[LLM Call] CRITICAL: API keys have not been loaded yet. Call load_and_update_api_keys().")
        return None

    # Rotate through available keys
    # Ensure index is valid even if list shrank since last call
    if CURRENT_KEY_INDEX >= len(AVAILABLE_API_KEYS):
        CURRENT_KEY_INDEX = 0

    if not AVAILABLE_API_KEYS: # Double check after index reset
        return None

    key = AVAILABLE_API_KEYS[CURRENT_KEY_INDEX]
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(AVAILABLE_API_KEYS)
    # print(f"[LLM Call] Using API Key Index: {CURRENT_KEY_INDEX} of {len(AVAILABLE_API_KEYS)}") # Debugging

    # Note: Marking key as used is removed from here.
    # It should be done by the calling function AFTER a successful API call,
    # potentially using the async crud.mark_api_key_used function.

    return key

# Custom retry condition: Retry on 429 (Rate Limit) and 5xx (Server Errors)
def should_retry(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on rate limits and server errors
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    # Retry on general request errors (network issues, timeouts)
    return isinstance(exception, httpx.RequestError)

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception(should_retry)) # Correct decorator usage
async def call_llm_api(client: httpx.AsyncClient, prompt: str, model: str = "google/gemini-1.5-pro-latest") -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (specifically handles OpenRouter).
    Uses the specified model, falling back to a default if needed.
    Handles API key selection and retries.
    Attempts to parse JSON, falls back to raw inference string.
    """
    api_key = get_next_api_key() # Use the synchronous key selection function
    endpoint = None
    headers = {}
    selected_model = model # Use the passed model by default

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        if not api_key:
            print("[LLM Call] CRITICAL: No OpenRouter API Key available.")
            return None # No key, cannot proceed
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Recommended headers for OpenRouter tracking/moderation
            "HTTP-Referer": settings.HTTP_REFERER or settings.AGENCY_BASE_URL or "https://acumenis.ai", # Provide a default
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME or "Acumenis", # Provide a default
        }
        # Use the specific model requested, e.g., "google/gemini-1.5-pro-latest"
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            # Add optional parameters if needed, e.g., max_tokens, temperature
            # "max_tokens": 4096, # Example: Set max tokens
            # "temperature": 0.7, # Example: Set temperature
        }
    # Add elif blocks here for other providers like 'openai' if needed
    # elif settings.LLM_PROVIDER == "openai":
    #     api_key = settings.LLM_API_KEY ... etc ...
    else:
        print(f"[LLM Call] Unsupported LLM_PROVIDER configured: {settings.LLM_PROVIDER}. Only 'openrouter' is explicitly handled.")
        return None

    if not endpoint: # Should be set if provider is supported
        print("[LLM Call] API endpoint not determined.")
        return None

    # --- Make API Call ---
    print(f"[LLM Call] Sending request to {endpoint} using model {selected_model}")
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
        result = response.json()

        # --- Parse Response (OpenAI/OpenRouter format assumed) ---
        # Check for errors in the response body first
        if "error" in result:
             error_details = result["error"]
             print(f"[LLM Call] API returned an error: {error_details}")
             # Specific check for model not found or invalid request
             if isinstance(error_details, dict):
                 msg = error_details.get("message", "")
                 code = error_details.get("code")
                 if "model_not_found" in msg or code == "model_not_found":
                     print(f"[LLM Call] Model '{selected_model}' not found or invalid on OpenRouter.")
                 elif response.status_code == 400:
                     print(f"[LLM Call] Bad Request (400): Check payload or model parameters. Message: {msg}")
             return None # Return None on API error

        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Clean potential markdown ```json ... ``` blocks
                cleaned_content = content.strip()
                # Regex to find JSON block, handles potential variations in spacing/newlines
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE)
                if json_match:
                    json_str = json_match.group(1).strip()
                    # print(f"[LLM Call] Extracted JSON string from markdown: {json_str[:100]}...") # Debugging
                else:
                    # If no markdown block, assume it might be direct JSON or just text
                    json_str = cleaned_content
                    # print(f"[LLM Call] No JSON markdown block found, trying direct parse: {json_str[:100]}...") # Debugging

                # Attempt to parse as JSON
                parsed_json = json.loads(json_str)
                # print("[LLM Call] Successfully parsed JSON response.") # Debugging
                return parsed_json
            except json.JSONDecodeError:
                # If not JSON, return the raw string content wrapped in a dict
                # print(f"[LLM Call] Response content is not JSON, returning raw: {content[:100]}...") # Debugging
                return {"raw_inference": content}
        else:
             print(f"[LLM Call] No content found in LLM response choices: {result}")
             return None

    except httpx.HTTPStatusError as e:
        # This block catches errors raised by response.raise_for_status()
        print(f"[LLM Call] API request failed: Status {e.response.status_code}")
        try:
            # Attempt to get more details from the response body
            error_body = e.response.json()
            print(f"[LLM Call] Error Body: {error_body}")
            if e.response.status_code == 429:
                print("[LLM Call] Rate limit hit (429).")
            elif e.response.status_code == 401:
                 print("[LLM Call] Authentication error (401): Check API Key.")
            elif e.response.status_code == 400:
                 print(f"[LLM Call] Bad Request (400): Possible issue with model '{selected_model}' or request format.")
            elif e.response.status_code == 404:
                 print(f"[LLM Call] Not Found (404): Possible issue with model '{selected_model}' or endpoint.")
            elif e.response.status_code >= 500:
                 print(f"[LLM Call] Server error ({e.response.status_code}): Temporary issue with the API provider.")
        except json.JSONDecodeError:
            # If response body is not JSON
            print(f"[LLM Call] Error Body (non-JSON): {e.response.text[:500]}") # Log first 500 chars
        # Re-raise the exception to trigger tenacity retry if applicable
        raise e
    except httpx.RequestError as e:
        # Handles network errors, timeouts, etc.
        print(f"[LLM Call] Network or Request Error: {e}")
        # Re-raise the exception to trigger tenacity retry if applicable
        raise e
    except RetryError as e:
        # This is caught after all retry attempts fail
        print(f"[LLM Call] API call failed after {RETRY_ATTEMPTS} attempts. Last exception: {e.last_attempt.exception()}")
        return None # Return None after exhausting retries
    except Exception as e:
        # Catch any other unexpected errors
        print(f"[LLM Call] Unexpected error calling API: {e}")
        traceback.print_exc() # Use traceback imported at the top
        return None