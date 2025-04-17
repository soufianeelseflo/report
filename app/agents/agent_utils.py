# autonomous_agency/app/agents/agent_utils.py
import httpx
from typing import Optional, Dict, Any, List, Tuple
import json
import os
import re
import random
import traceback
import datetime
import logging
import asyncio
import ssl # For checking SSL errors specifically
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception, RetryError, wait_exponential

# Corrected relative import for package structure
try:
    # Use absolute path based on assumed project structure if possible
    from Acumenis.app.core.config import settings
    from Acumenis.app.db.base import get_worker_session
    from Acumenis.app.db import crud
    from Acumenis.app.core.security import decrypt_data
except ImportError:
    print("[AgentUtils] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    # Assume running from within app directory for fallback
    from app.core.config import settings
    from app.db.base import get_worker_session
    from app.db import crud
    from app.core.security import decrypt_data

# Setup logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Custom Exceptions ---
class NoValidApiKeyError(Exception):
    """Raised when no usable API keys are found in the pool."""
    pass

class APIKeyInvalidError(Exception):
    """Raised when an API key is definitively identified as invalid (e.g., 401/403)."""
    pass

# --- Tenacity Configuration ---
RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=15) # Increased max wait slightly
RETRY_ATTEMPTS = 4 # Increased attempts slightly
LLM_TIMEOUT = 240.0 # Increased timeout significantly for complex/multimodal tasks
RATE_LIMIT_COOLDOWN_SECONDS = getattr(settings, 'RATE_LIMIT_COOLDOWN_SECONDS', 180) # Longer default cooldown

# --- Global variables for API keys ---
AVAILABLE_API_KEYS: List[Dict[str, Any]] = [] # Structure: {'id': int | None, 'key': str, 'rate_limited_until': datetime | None}
CURRENT_KEY_INDEX: int = 0
_keys_loaded: bool = False
_key_load_lock = asyncio.Lock()

# --- Key Refresh Task ---
_key_refresh_task: Optional[asyncio.Task] = None
_key_refresh_interval: int = 300 # Refresh every 5 minutes
shutdown_event = asyncio.Event() # Shared shutdown event

# --- HTTP Client & Proxy Setup ---
def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    proxy_url = settings.PROXY_URL
    if proxy_url:
        if not re.match(r"^(http|https|socks\d?)://", proxy_url):
             logger.warning(f"[Proxy] Configured PROXY_URL '{proxy_url}' doesn't look like a valid URL.")
             return None
        return {"http://": proxy_url, "https://": proxy_url}
    elif settings.PROXY_LIST:
        valid_proxies = [p for p in settings.PROXY_LIST if re.match(r"^(http|https|socks\d?)://", p)]
        if valid_proxies:
            selected_proxy = random.choice(valid_proxies)
            proxy_display = selected_proxy.split('@')[-1] if '@' in selected_proxy else selected_proxy
            logger.debug(f"[Proxy] Using random proxy from PROXY_LIST for client: {proxy_display}")
            return {"http://": selected_proxy, "https://": selected_proxy}
        else:
            logger.warning("[Proxy] PROXY_LIST configured but contains no valid URLs.")
            return None
    return None

async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies and enhanced headers."""
    proxies = get_proxy_map()
    limits = httpx.Limits(max_connections=250, max_keepalive_connections=60) # Increased limits
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=45.0) # Increased connect timeout

    # Enhanced Headers
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"
    ]
    platforms = ['"Windows"', '"macOS"', '"Linux"']
    accept_languages = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "en;q=0.7"]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": random.choice(accept_languages),
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": random.choice(platforms),
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document", # More common for initial requests
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none", # Or "cross-site" depending on context
        "Sec-Fetch-User": "?1",
        "Connection": "keep-alive",
        "DNT": "1", # Do Not Track
    }
    # Add Referer dynamically if needed before specific requests

    client = httpx.AsyncClient(
        proxies=proxies,
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        http2=True # Enable HTTP/2
    )
    return client

# --- API Key Management Logic ---
async def load_and_update_api_keys():
    """
    Loads active API keys from the database and updates the global list.
    Handles decryption, stores key ID and rate limit info. Marks decryption errors.
    """
    global AVAILABLE_API_KEYS, CURRENT_KEY_INDEX, _keys_loaded
    async with _key_load_lock: # Ensure only one refresh runs at a time
        logger.info("[API Keys] Refreshing API keys from database...")
        session: AsyncSession = None
        new_keys_loaded = []
        keys_marked_for_error = []
        try:
            session = await get_worker_session()
            provider = settings.LLM_PROVIDER or "openrouter"
            loaded_key_details = await crud.get_active_api_keys_with_details(session, provider=provider)

            if loaded_key_details:
                for key_detail in loaded_key_details:
                    try:
                        decrypted_key = decrypt_data(key_detail['api_key_encrypted'])
                        if decrypted_key:
                            new_keys_loaded.append({
                                'id': key_detail['id'],
                                'key': decrypted_key,
                                'rate_limited_until': key_detail.get('rate_limited_until')
                            })
                        else:
                            logger.warning(f"[API Keys] Decryption failed for key ID {key_detail['id']}. Skipping and marking as error.")
                            keys_marked_for_error.append((key_detail['id'], 'Decryption failed during load'))
                    except Exception as decrypt_err:
                        logger.error(f"[API Keys] Error decrypting key ID {key_detail['id']}: {decrypt_err}. Skipping and marking as error.", exc_info=True)
                        keys_marked_for_error.append((key_detail['id'], f'Decryption exception: {str(decrypt_err)[:200]}'))

                # Mark keys with decryption errors in the DB within the same transaction
                if keys_marked_for_error:
                    for key_id, reason in keys_marked_for_error:
                        await crud.set_api_key_status_by_id(session, key_id, 'error', reason)
                    await session.commit() # Commit error status updates immediately
                    logger.warning(f"[API Keys] Marked {len(keys_marked_for_error)} keys as 'error' due to decryption issues.")

                logger.info(f"[API Keys] Successfully loaded and decrypted {len(new_keys_loaded)} active keys for provider '{provider}'.")
            else:
                logger.info(f"[API Keys] No active keys found in DB for provider '{provider}'.")
                # Fallback: Use the key from .env ONLY if DB is empty AND it's OpenRouter
                if settings.OPENROUTER_API_KEY and provider == "openrouter":
                    logger.info("[API Keys] Using fallback API key from settings (ID: 0).")
                    new_keys_loaded = [{'id': 0, 'key': settings.OPENROUTER_API_KEY, 'rate_limited_until': None}] # Use ID 0 for fallback
                else:
                    new_keys_loaded = []

            # Atomically update the global list and reset index
            AVAILABLE_API_KEYS = new_keys_loaded
            CURRENT_KEY_INDEX = 0 if AVAILABLE_API_KEYS else -1
            _keys_loaded = True
            logger.info(f"[API Keys] Key pool updated. Active keys: {len(AVAILABLE_API_KEYS)}. Index reset.")

        except Exception as e:
            logger.error(f"[API Keys] CRITICAL Error loading API keys from database: {e}", exc_info=True)
            logger.warning(f"[API Keys] Keeping previously loaded keys ({len(AVAILABLE_API_KEYS)}) due to load error.")
            _keys_loaded = True # Mark as loaded attempt failed, but keep flag true
            if session: await session.rollback()
        finally:
            if session:
                await session.close()

async def _key_refresh_loop():
    """Background loop to periodically refresh API keys."""
    logger.info("[API Keys] Starting background key refresh loop...")
    while not shutdown_event.is_set():
        try:
            await load_and_update_api_keys()
        except Exception as e:
            logger.error(f"[API Keys] Unhandled error in refresh loop: {e}", exc_info=True)
        # Use asyncio.sleep, but check shutdown_event frequently
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=_key_refresh_interval)
            logger.info("[API Keys] Shutdown signal received during wait, exiting refresh loop.")
            break
        except asyncio.TimeoutError:
            continue # Normal timeout, continue loop
        except asyncio.CancelledError:
             logger.info("[API Keys] Refresh task cancelled.")
             break
        except Exception as e:
             logger.error(f"[API Keys] Error during refresh loop wait: {e}")
             break # Exit loop on unexpected error
    logger.info("[API Keys] Background key refresh loop stopped.")


def start_key_refresh_task():
    """Starts the background key refresh task if not already running."""
    global _key_refresh_task
    if _key_refresh_task is None or _key_refresh_task.done():
        logger.info("Attempting to start background key refresh task...")
        _key_refresh_task = asyncio.create_task(_key_refresh_loop())
        logger.info("[API Keys] Background key refresh task started.")
    else:
        logger.info("[API Keys] Background key refresh task already running.")

def get_next_api_key() -> Optional[Dict[str, Any]]:
    """
    Selects the next valid API key dict {'id', 'key', 'rate_limited_until'} for use.
    Rotates through keys, skipping those that are currently rate-limited.
    Returns None if no valid keys are available.
    """
    global CURRENT_KEY_INDEX

    if not _keys_loaded:
        logger.critical("[LLM Call] CRITICAL: API keys have not been loaded yet. Cannot select key.")
        # Cannot trigger load here safely from potentially sync context or different loop.
        return None

    if not AVAILABLE_API_KEYS:
        logger.warning("[LLM Call] No API keys available in the pool.")
        return None

    num_keys = len(AVAILABLE_API_KEYS)
    start_index = CURRENT_KEY_INDEX % num_keys
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    for i in range(num_keys):
        current_check_index = (start_index + i) % num_keys
        key_info = AVAILABLE_API_KEYS[current_check_index]
        key_id_display = key_info.get('id', 'Fallback')

        # Check if key is rate-limited
        rate_limited_until = key_info.get('rate_limited_until')
        is_rate_limited = False
        if rate_limited_until:
            if isinstance(rate_limited_until, datetime.datetime):
                # Ensure comparison is timezone-aware
                if rate_limited_until.tzinfo is None:
                    logger.warning(f"[API Keys] Key ID {key_id_display} has naive datetime for rate_limited_until. Assuming UTC.")
                    rate_limited_until = rate_limited_until.replace(tzinfo=datetime.timezone.utc)

                if now_utc < rate_limited_until:
                    is_rate_limited = True
                    logger.debug(f"[LLM Call] Skipping rate-limited key ID: {key_id_display} (until {rate_limited_until.isoformat()})")
                else:
                    # Rate limit expired, clear it locally for immediate use (DB refresh will catch up)
                    logger.debug(f"[API Keys] Rate limit expired for key ID {key_id_display}. Clearing local cooldown.")
                    key_info['rate_limited_until'] = None
                    # Optionally trigger an async task to clear it in DB? Low priority.
            else:
                logger.warning(f"[API Keys] Invalid rate_limited_until format for key ID {key_id_display}: {type(rate_limited_until)}. Assuming not rate-limited.")

        if not is_rate_limited:
            # Found a valid key
            CURRENT_KEY_INDEX = (current_check_index + 1) % num_keys # Point to the next one for the next call
            logger.debug(f"[LLM Call] Using API Key ID: {key_id_display} (Index: {current_check_index})")
            return key_info

    logger.warning("[LLM Call] All available API keys are currently rate-limited or none exist.")
    return None

# --- Retry Logic ---
def should_retry(exception: BaseException) -> bool:
    """Determines if an HTTP request should be retried based on exception type and status code."""
    if isinstance(exception, APIKeyInvalidError):
        logger.warning(f"API Key Invalid Error encountered ({exception}). Retrying with next key...")
        return True # Retry immediately, get_next_api_key will skip the invalid one
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        # Retry on 429 (Rate Limit), 500, 502, 503, 504 (Server Errors)
        should = status_code == 429 or status_code in [500, 502, 503, 504]
        if should:
            logger.warning(f"Retrying due to HTTP Status {status_code}...")
        else:
             logger.warning(f"Not retrying HTTP Status {status_code}.")
        return should
    if isinstance(exception, (httpx.RequestError, httpx.TimeoutException)):
        # Check for specific non-retriable network errors like SSL
        original_exc = getattr(exception, '__cause__', None)
        if isinstance(original_exc, ssl.SSLError):
             logger.error(f"SSL Error encountered, not retrying: {exception}")
             return False
        logger.warning(f"Network/Timeout error encountered, retrying: {type(exception).__name__} - {exception}")
        return True
    # Do not retry NoValidApiKeyError
    if isinstance(exception, NoValidApiKeyError):
        logger.critical("No valid API keys available, cannot retry.")
        return False

    logger.warning(f"Not retrying unhandled exception type {type(exception).__name__}: {exception}")
    return False

# --- Async Task Helpers for DB Updates ---
async def _run_db_task(coro, task_description: str):
    """Runs a DB task in the background, handling session and errors."""
    session: AsyncSession = None
    try:
        session = await get_worker_session()
        await coro(session)
        await session.commit()
        logger.debug(f"Successfully completed DB task: {task_description}")
    except Exception as e:
        logger.error(f"Error in background DB task '{task_description}': {e}", exc_info=True)
        if session: await session.rollback()
    finally:
        if session: await session.close()

async def _mark_key_used_task(key_id: Optional[int]):
    """Marks a key as used in the DB."""
    if key_id and key_id != 0:
        await _run_db_task(lambda s: crud.mark_api_key_used_by_id(s, key_id), f"Mark Key Used (ID: {key_id})")

async def _update_key_status_task(key_id: Optional[int], status: str, reason: str):
    """Updates key status in the DB."""
    if key_id and key_id != 0:
        await _run_db_task(lambda s: crud.set_api_key_status_by_id(s, key_id, status, reason), f"Update Key Status (ID: {key_id}, Status: {status})")

async def _set_rate_limit_cooldown_task(key_id: Optional[int], cooldown_until: datetime.datetime, reason: str):
    """Sets rate limit status and cooldown time in the DB."""
    if key_id and key_id != 0:
        await _run_db_task(lambda s: crud.set_api_key_rate_limited(s, key_id, cooldown_until, reason), f"Set Rate Limit (ID: {key_id})")

# --- Main LLM API Call Function ---
@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception(should_retry))
async def call_llm_api(
    client: httpx.AsyncClient,
    prompt: str,
    model: Optional[str] = None,
    image_data: Optional[str] = None # Optional base64 encoded image data
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (handles OpenRouter). Handles key rotation, rate limiting, retries, multimodal input.
    Returns parsed JSON or {'raw_inference': content}.
    Raises APIKeyInvalidError or NoValidApiKeyError on critical key issues.
    """
    key_info = get_next_api_key()
    if not key_info:
        # No keys available at all, raise immediately, don't enter retry loop.
        raise NoValidApiKeyError("No valid API keys available to make the LLM call.")

    api_key = key_info['key']
    key_id = key_info.get('id') # DB ID (0 or None for fallback)
    key_id_for_logs = key_id if key_id else "Fallback"
    key_suffix = api_key[-4:] if api_key else 'N/A'

    endpoint = None
    headers = {}
    selected_model = model or settings.STANDARD_REPORT_MODEL or settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-flash-latest"

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.HTTP_REFERER or str(settings.AGENCY_BASE_URL), # Use validated URL
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME,
        }
        messages = []
        is_vision_model = any(vm in selected_model for vm in ['vision', 'gemini-1.5', 'gpt-4o', 'claude-3'])

        if image_data and is_vision_model:
            mime_type = "image/jpeg" # Default
            if image_data.startswith("/9j/"): mime_type = "image/jpeg"
            elif image_data.startswith("iVBOR"): mime_type = "image/png"
            elif image_data.startswith("UklGR"): mime_type = "image/webp"
            elif image_data.startswith("R0lG"): mime_type = "image/gif"
            messages.append({
                "role": "user",
                "content": [
                     {"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                ]
            })
            logger.info(f"[LLM Call] Including image data ({mime_type}) for model {selected_model}")
        else:
             messages.append({"role": "user", "content": prompt})
             if image_data:
                  logger.warning(f"[LLM Call] Image data provided but model '{selected_model}' not vision-capable. Sending text only.")

        payload = {"model": selected_model, "messages": messages}
        # Add other parameters like temperature, max_tokens from settings if needed
        # payload["temperature"] = getattr(settings, 'LLM_TEMPERATURE', 0.7)
        # payload["max_tokens"] = getattr(settings, 'LLM_MAX_TOKENS', 4096)

    else:
        logger.error(f"[LLM Call] Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}.")
        # Raise an error instead of returning None to prevent silent failures in callers
        raise ValueError(f"Unsupported LLM_PROVIDER configured: {settings.LLM_PROVIDER}")

    # --- Make API Call ---
    logger.debug(f"[LLM Call] Sending request to {endpoint} using model {selected_model} with key ID {key_id_for_logs} (...{key_suffix})")
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)

        # Check for specific errors BEFORE raise_for_status to handle key invalidation correctly
        if response.status_code in [401, 403]:
            logger.warning(f"[LLM Call] Received {response.status_code} for key ID {key_id_for_logs}. Deactivating key.")
            if key_id and key_id != 0:
                asyncio.create_task(_update_key_status_task(key_id, 'inactive', f"API Error {response.status_code}"))
                # Invalidate key locally immediately
                for k in AVAILABLE_API_KEYS:
                    if k.get('id') == key_id: k['rate_limited_until'] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365); break
            raise APIKeyInvalidError(f"Key ID {key_id_for_logs} received {response.status_code} error.") # Raise to trigger retry with next key

        if response.status_code == 429:
            logger.warning(f"[LLM Call] Rate limit (429) hit for key ID {key_id_for_logs}. Marking as rate-limited.")
            if key_id and key_id != 0:
                 cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=RATE_LIMIT_COOLDOWN_SECONDS)
                 asyncio.create_task(_set_rate_limit_cooldown_task(key_id, cooldown_until, f"API Error {response.status_code}"))
                 # Update local cache immediately
                 for k in AVAILABLE_API_KEYS:
                     if k.get('id') == key_id: k['rate_limited_until'] = cooldown_until; break
            # Raise the original exception to trigger Tenacity retry
            response.raise_for_status()

        # Raise for other non-2xx codes if not handled above
        response.raise_for_status()
        result = response.json()

        # Mark key as used asynchronously (only if it's a DB key)
        if key_id and key_id != 0: asyncio.create_task(_mark_key_used_task(key_id))

        # --- Parse Response ---
        if "error" in result:
             error_details = result["error"]
             error_message = error_details.get('message', str(error_details))
             logger.error(f"[LLM Call] API returned error in response body: {error_message}")
             # Check if this body error indicates an invalid key
             error_str_lower = error_message.lower()
             if key_id and key_id != 0 and ('invalid api key' in error_str_lower or 'authentication error' in error_str_lower or 'incorrect api key' in error_str_lower or 'api key is invalid' in error_str_lower):
                 logger.warning(f"[LLM Call] Deactivating key ID {key_id} due to API error in body: {error_message[:200]}")
                 asyncio.create_task(_update_key_status_task(key_id, 'inactive', f"Invalid Key Error in Body: {error_message[:200]}"))
                 for k in AVAILABLE_API_KEYS: # Invalidate locally
                     if k.get('id') == key_id: k['rate_limited_until'] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365); break
                 raise APIKeyInvalidError(f"Key ID {key_id} reported as invalid by API in response body.")
             return None # Return None for other non-key-related errors in body

        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Improved JSON extraction
                cleaned_content = content.strip()
                try:
                    # Try direct parse first
                    parsed_json = json.loads(cleaned_content)
                    return parsed_json
                except json.JSONDecodeError:
                    # Try parsing from markdown code block
                    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", cleaned_content, re.IGNORECASE | re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        try:
                            parsed_json = json.loads(json_str)
                            return parsed_json
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON from code block: {e}. Content snippet: {json_str[:100]}... Returning raw inference.")
                            return {"raw_inference": content}
                    else:
                        # If not direct JSON and not in code block, return raw
                        logger.debug("LLM response was not JSON or in a JSON code block. Returning raw inference.")
                        return {"raw_inference": content}
            except Exception as parse_e:
                logger.error(f"Error processing LLM response content: {parse_e}", exc_info=True)
                return {"raw_inference": content} # Fallback to raw
        else:
             logger.warning(f"[LLM Call] No content found in LLM response choices: {result}")
             return None

    # --- Exception Handling within @retry block ---
    except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException, APIKeyInvalidError) as e:
        # These errors are handled by the `should_retry` logic or specific checks above.
        # Re-raise them so Tenacity can decide whether to retry based on `should_retry`.
        logger.debug(f"Raising exception for Tenacity retry check: {type(e).__name__}")
        raise e
    except NoValidApiKeyError:
        # This is caught by the initial check, re-raise to stop retries.
        raise
    except Exception as e:
        # Catch any other unexpected errors during the API call or processing
        logger.error(f"[LLM Call] Unexpected error during API call/processing for key ID {key_id_for_logs}: {e}", exc_info=True)
        # Do not retry unexpected errors by default
        # Optionally mark key as 'error'?
        # if key_id and key_id != 0:
        #     asyncio.create_task(_update_key_status_task(key_id, 'error', f"Unexpected API Call Error: {str(e)[:200]}"))
        return None # Indicate failure for unexpected errors

# Note: The @retry decorator handles the final RetryError if all attempts fail.
# The logic inside the `except RetryError` block in the previous version is now implicitly handled by Tenacity.
# The final action (like deactivating a key after all retries fail due to auth) needs to be inferred from the last exception raised within the try block.