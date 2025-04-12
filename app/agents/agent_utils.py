
# autonomous_agency/app/agents/agent_utils.py
import httpx
from typing import Optional, Dict, Any, List
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
    from Acumenis.app.core.config import settings
    from Acumenis.app.db.base import get_worker_session
    from Acumenis.app.db import crud
    from Acumenis.app.core.security import decrypt_data
except ImportError:
    print("[AgentUtils] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db.base import get_worker_session
    from app.db import crud
    from app.core.security import decrypt_data

# Setup logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Tenacity Configuration ---
# Exponential backoff for retries, max 10 seconds
RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=10)
RETRY_ATTEMPTS = 3
LLM_TIMEOUT = 180.0 # Increased timeout further for complex/multimodal tasks
RATE_LIMIT_COOLDOWN_SECONDS = getattr(settings, 'RATE_LIMIT_COOLDOWN_SECONDS', 120) # Longer default cooldown

# --- Global variables for API keys ---
# Structure: {'id': int | None, 'key': str, 'rate_limited_until': datetime | None}
AVAILABLE_API_KEYS: List[Dict[str, Any]] = []
CURRENT_KEY_INDEX: int = 0
_keys_loaded: bool = False
_key_load_lock = asyncio.Lock()

# --- Key Refresh Task ---
_key_refresh_task: Optional[asyncio.Task] = None
_key_refresh_interval: int = 300 # Refresh every 5 minutes
shutdown_event = asyncio.Event() # Shared shutdown event

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
    """Creates an async httpx client, potentially configured with proxies."""
    proxies = get_proxy_map()
    # Increased limits for potentially higher concurrency needs across agents
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=30.0) # Increased connect timeout

    headers = {
        "User-Agent": f"AcumenisAgent/3.0 ({settings.AGENCY_BASE_URL or 'http://localhost'})", # Version bump
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        # Standard security headers
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site", # More realistic for external API calls
    }

    # Add common browser headers to reduce fingerprinting
    headers.update({
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"' # Can be randomized or set based on environment
    })


    client = httpx.AsyncClient(
        proxies=proxies,
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        http2=True # Enable HTTP/2 if supported by proxies/endpoints
    )
    return client

async def load_and_update_api_keys():
    """
    Loads active API keys from the database and updates the global list.
    Handles decryption, stores key ID and rate limit info. Marks decryption errors.
    """
    global AVAILABLE_API_KEYS, CURRENT_KEY_INDEX, _keys_loaded
    async with _key_load_lock:
        logger.info("[API Keys] Refreshing API keys from database...")
        session: AsyncSession = None
        new_keys_loaded = []
        keys_marked_for_error = []
        try:
            session = await get_worker_session()
            provider = settings.LLM_PROVIDER or "openrouter"
            # Expects crud function returning: [{'id': int, 'api_key_encrypted': str, 'rate_limited_until': Optional[datetime]}]
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
                        logger.warning(f"[API Keys] Error decrypting key ID {key_detail['id']}: {decrypt_err}. Skipping and marking as error.")
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
                    logger.info("[API Keys] Using fallback API key from settings.")
                    new_keys_loaded = [{'id': 0, 'key': settings.OPENROUTER_API_KEY, 'rate_limited_until': None}] # Use ID 0 for fallback
                else:
                    new_keys_loaded = []

            # Atomically update the global list and reset index
            AVAILABLE_API_KEYS = new_keys_loaded
            CURRENT_KEY_INDEX = 0 if AVAILABLE_API_KEYS else -1
            _keys_loaded = True

        except Exception as e:
            logger.error(f"[API Keys] Error loading API keys from database: {e}", exc_info=True)
            logger.warning(f"[API Keys] Keeping previously loaded keys ({len(AVAILABLE_API_KEYS)}) due to load error.")
            _keys_loaded = True # Still mark as loaded attempt
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
            logger.error(f"[API Keys] Error in refresh loop: {e}", exc_info=True)
        # Use asyncio.sleep, but check shutdown_event frequently
        try:
            # Wait for the interval, but break early if shutdown is signaled
            await asyncio.wait_for(shutdown_event.wait(), timeout=_key_refresh_interval)
            logger.info("[API Keys] Shutdown signal received during wait, exiting refresh loop.")
            break
        except asyncio.TimeoutError:
            continue # Timeout is expected, continue loop
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
        logger.critical("[LLM Call] CRITICAL: API keys have not been loaded yet.")
        return None

    if not AVAILABLE_API_KEYS:
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
                    rate_limited_until = rate_limited_until.replace(tzinfo=datetime.timezone.utc)
                if now_utc < rate_limited_until:
                    is_rate_limited = True
                    logger.debug(f"[LLM Call] Skipping rate-limited key ID: {key_id_display} (until {rate_limited_until.strftime('%H:%M:%S')})")
            else:
                logger.warning(f"[API Keys] Invalid rate_limited_until format for key ID {key_id_display}: {type(rate_limited_until)}. Assuming not rate-limited.")

        if not is_rate_limited:
            # Found a valid key
            CURRENT_KEY_INDEX = (current_check_index + 1) % num_keys # Point to the next one for the next call
            logger.debug(f"[LLM Call] Using API Key ID: {key_id_display} (Index: {current_check_index})")
            return key_info

    logger.warning("[LLM Call] All available API keys are currently rate-limited or none exist.")
    return None

# Custom retry condition: Retry on 429 (Rate Limit) and 5xx (Server Errors)
def should_retry(exception: BaseException) -> bool:
    """Determines if an HTTP request should be retried."""
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        should = status_code == 429 or status_code >= 500
        if should:
            logger.warning(f"Retrying due to HTTP Status {status_code}...")
        return should
    if isinstance(exception, httpx.RequestError):
        original_exc = getattr(exception, 'original_exception', getattr(exception, '__cause__', None))
        if isinstance(original_exc, ssl.SSLError):
             logger.error(f"SSL Error encountered, not retrying: {exception}")
             return False
        logger.warning(f"Network error encountered, retrying: {exception}")
        return True
    logger.warning(f"Not retrying exception of type {type(exception).__name__}: {exception}")
    return False

# --- Async Task Helpers for DB Updates ---
async def _run_db_task(coro, task_description: str):
    """Runs a DB task in the background, handling session and errors."""
    session = None
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
    if key_id and key_id != 0: # Don't try to mark fallback key (ID 0)
        await _run_db_task(lambda s: crud.mark_api_key_used_by_id(s, key_id), f"Mark Key Used (ID: {key_id})")

async def _update_key_status_task(key_id: Optional[int], status: str, reason: str):
    if key_id and key_id != 0:
        await _run_db_task(lambda s: crud.set_api_key_status_by_id(s, key_id, status, reason), f"Update Key Status (ID: {key_id}, Status: {status})")

async def _set_rate_limit_cooldown_task(key_id: Optional[int], cooldown_until: datetime.datetime, reason: str):
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
    """
    key_info = get_next_api_key()
    if not key_info:
        logger.critical("[LLM Call] No valid API Key available for LLM call.")
        return None

    api_key = key_info['key']
    key_id = key_info.get('id') # DB ID (0 or None for fallback)
    key_id_for_logs = key_id if key_id else "Fallback"
    key_suffix = api_key[-4:] if api_key else 'N/A'

    endpoint = None
    headers = {}
    # Determine model: Use provided, fallback to standard, then premium
    selected_model = model or settings.STANDARD_REPORT_MODEL or settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-flash-latest"

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.HTTP_REFERER or settings.AGENCY_BASE_URL or "https://acumenis.ai",
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME or "Acumenis",
        }
        # Construct messages payload with multimodal support
        messages = []
        # More robust check for vision models
        is_vision_model = any(vm in selected_model for vm in ['vision', 'gemini-1.5', 'gpt-4o', 'claude-3'])

        if image_data and is_vision_model:
            # Basic mime type detection
            mime_type = "image/jpeg" # Default assumption
            if image_data.startswith("/9j/"): mime_type = "image/jpeg"
            elif image_data.startswith("iVBOR"): mime_type = "image/png"
            elif image_data.startswith("UklGR"): mime_type = "image/webp"
            elif image_data.startswith("R0lG"): mime_type = "image/gif"

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
            logger.info(f"[LLM Call] Including image data ({mime_type}) in request for model {selected_model}")
        else:
             messages.append({"role": "user", "content": prompt})
             if image_data:
                  logger.warning(f"[LLM Call] Image data provided but model '{selected_model}' is not recognized as vision-capable. Sending text only.")

        payload = {
            "model": selected_model,
            "messages": messages,
            # Example: Add temperature from settings if defined
            # "temperature": getattr(settings, 'LLM_TEMPERATURE', 0.7),
        }
    else:
        logger.error(f"[LLM Call] Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}.")
        return None

    # --- Make API Call ---
    logger.debug(f"[LLM Call] Sending request to {endpoint} using model {selected_model} with key ID {key_id_for_logs} (...{key_suffix})")
    last_exception = None
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Raises HTTPStatusError for non-retried 4xx/5xx
        result = response.json()

        # Mark key as used asynchronously (only if it's a DB key)
        if key_id: asyncio.create_task(_mark_key_used_task(key_id))

        # --- Parse Response ---
        if "error" in result:
             error_details = result["error"]
             error_message = error_details.get('message', str(error_details))
             logger.error(f"[LLM Call] API returned error in body: {error_message}")
             error_str_lower = error_message.lower()
             # Deactivate key only on definitive key errors
             if key_id and key_id != 0 and ('invalid api key' in error_str_lower or 'authentication error' in error_str_lower or 'incorrect api key' in error_str_lower or 'api key is invalid' in error_str_lower):
                 logger.warning(f"[LLM Call] Deactivating key ID {key_id} due to API error: {error_message[:200]}")
                 asyncio.create_task(_update_key_status_task(key_id, 'inactive', f"Invalid Key Error: {error_message[:200]}"))
             return None

        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Improved JSON extraction: handles direct JSON or JSON within markdown code blocks
                cleaned_content = content.strip()
                try:
                    parsed_json = json.loads(cleaned_content)
                    return parsed_json
                except json.JSONDecodeError:
                    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", cleaned_content, re.IGNORECASE | re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        try:
                            parsed_json = json.loads(json_str)
                            return parsed_json
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON from code block: {e}. Returning raw inference.")
                            return {"raw_inference": content}
                    else:
                        logger.debug("LLM response was not JSON or in a JSON code block. Returning raw inference.")
                        return {"raw_inference": content}
            except Exception as parse_e:
                logger.error(f"Error processing LLM response content: {parse_e}", exc_info=True)
                return {"raw_inference": content} # Fallback to raw
        else:
             logger.warning(f"[LLM Call] No content found in LLM response choices: {result}")
             return None

    except httpx.HTTPStatusError as e:
        last_exception = e
        status_code = e.response.status_code
        error_text = e.response.text[:500]
        logger.warning(f"[LLM Call] API request failed: Status {status_code} for model {selected_model} with key ID {key_id_for_logs} (...{key_suffix}). Response: {error_text}...")

        if key_id and key_id != 0: # Only update status for keys managed in DB
            if status_code in [401, 403]:
                 logger.warning(f"[LLM Call] Deactivating key ID {key_id} due to {status_code} error.")
                 asyncio.create_task(_update_key_status_task(key_id, 'inactive', f"API Error {status_code}"))
            elif status_code == 429:
                 logger.warning(f"[LLM Call] Rate limit hit for key ID {key_id}. Marking as rate-limited for {RATE_LIMIT_COOLDOWN_SECONDS}s.")
                 cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=RATE_LIMIT_COOLDOWN_SECONDS)
                 asyncio.create_task(_set_rate_limit_cooldown_task(key_id, cooldown_until, f"API Error {status_code}"))
                 # Update local cache immediately
                 for k in AVAILABLE_API_KEYS:
                     if k.get('id') == key_id:
                         k['rate_limited_until'] = cooldown_until; break

        # Re-raise ONLY if the error should be retried by Tenacity
        if should_retry(e):
            raise e
        else:
            logger.error(f"Non-retriable HTTP error {status_code} encountered. Returning None.")
            return None # Return None for non-retried HTTP errors

    except httpx.RequestError as e:
        last_exception = e
        logger.error(f"[LLM Call] Network or Request Error: {e}")
        raise e # Trigger retry
    except RetryError as e:
        logger.error(f"[LLM Call] API call failed after {RETRY_ATTEMPTS} attempts for model {selected_model}. Last exception: {e.last_attempt.exception()}")
        last_exc = e.last_attempt.exception()
        if key_id and key_id != 0 and isinstance(last_exc, httpx.HTTPStatusError):
             status_code = last_exc.response.status_code
             if status_code in [401, 403]:
                  logger.warning(f"[LLM Call] Deactivating key ID {key_id} after exhausting retries due to {status_code}.")
                  asyncio.create_task(_update_key_status_task(key_id, 'inactive', f"API Error {status_code} after retries"))
        return None
    except Exception as e:
        last_exception = e
        logger.error(f"[LLM Call] Unexpected error calling API: {e}", exc_info=True)
        # Optionally deactivate key on unexpected errors? Risky.
        # if key_id and key_id != 0:
        #     asyncio.create_task(_update_key_status_task(key_id, 'error', f"Unexpected API Call Error: {str(e)[:200]}"))
        return None