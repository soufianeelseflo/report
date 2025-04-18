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

# CORRECTED IMPORTS: Replaced "Nexus Plan.app" with "app"
try:
    from app.core.config import settings
    from app.db.base import get_worker_session
    from app.db import crud
    from app.core.security import decrypt_data
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
    """Raised when the single API key is missing or invalid."""
    pass

class APIKeyFailedError(Exception):
    """Raised when the single API key fails (401/403/429)."""
    pass

# --- Tenacity Configuration ---
RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=10) # Shorter max wait for faster failure detection
RETRY_ATTEMPTS = 2 # Fewer attempts, fail faster
LLM_TIMEOUT = 180.0 # Reduced timeout slightly
RATE_LIMIT_COOLDOWN_SECONDS = getattr(settings, 'RATE_LIMIT_COOLDOWN_SECONDS', 60) # Shorter cooldown, assume manual fix

# --- Global variables ---
# We only use the single key from settings now
shutdown_event = asyncio.Event() # Shared shutdown event

# --- Proxy Rotation State ---
# Simple global state for proxy rotation trigger. MCOL agent can set this.
# In a more robust system, this might be in Redis or DB.
_trigger_proxy_rotation = False
_current_proxy_index = 0 # Index for PROXY_LIST if used

def signal_proxy_rotation():
    """Sets the flag to rotate proxy on the next client creation."""
    global _trigger_proxy_rotation
    logger.warning("SIGNAL RECEIVED: Rotating proxy on next client request due to API key failure.")
    _trigger_proxy_rotation = True

# --- HTTP Client & Proxy Setup ---
def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    global _trigger_proxy_rotation, _current_proxy_index

    if not settings.PROXY_ENABLED:
        logger.debug("[Proxy] Proxy usage disabled in settings.")
        return None

    proxy_url = None
    proxy_list = settings.PROXY_LIST or [] # Use PROXY_LIST if available

    if proxy_list:
        if _trigger_proxy_rotation:
            _current_proxy_index = (_current_proxy_index + 1) % len(proxy_list)
            logger.info(f"[Proxy] Rotating proxy index to {_current_proxy_index} due to trigger.")
            _trigger_proxy_rotation = False # Reset flag after rotation

        if not proxy_list: # Should not happen if PROXY_LIST was set, but safety check
             logger.error("[Proxy] PROXY_LIST is configured but empty.")
             return None

        selected_proxy = proxy_list[_current_proxy_index]
        if not re.match(r"^(http|https|socks\d?)://.*:.*@.*:\d+", selected_proxy):
             logger.warning(f"[Proxy] Proxy '{selected_proxy}' from PROXY_LIST doesn't look valid (http://user:pass@host:port). Skipping.")
             # Try next proxy in list? For now, just return None.
             return None
        proxy_url = selected_proxy
        proxy_display = selected_proxy.split('@')[-1]
        logger.debug(f"[Proxy] Using proxy from PROXY_LIST (Index: {_current_proxy_index}): {proxy_display}")

    elif settings.PROXY_HOST and settings.PROXY_PORT and settings.PROXY_USER and settings.PROXY_PASSWORD:
        # Fallback to single proxy config if PROXY_LIST is not set
        # Rotation doesn't apply to single proxy config
        if _trigger_proxy_rotation:
             logger.warning("[Proxy] Proxy rotation triggered, but only single proxy configured. Using the same proxy.")
             _trigger_proxy_rotation = False # Reset flag

        proxy_url = f"http://{settings.PROXY_USER}:{settings.PROXY_PASSWORD}@{settings.PROXY_HOST}:{settings.PROXY_PORT}"
        proxy_display = f"{settings.PROXY_HOST}:{settings.PROXY_PORT}"
        logger.debug(f"[Proxy] Using single proxy from settings: {proxy_display}")
    else:
        logger.warning("[Proxy] Proxy enabled, but no valid PROXY_LIST or single proxy credentials configured.")
        return None

    return {"http://": proxy_url, "https://": proxy_url}


async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies and enhanced headers."""
    proxies = get_proxy_map() # Get current proxy setting
    limits = httpx.Limits(max_connections=250, max_keepalive_connections=60) # Keep high limits
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=30.0) # Slightly shorter connect timeout

    # Enhanced Headers (Keep randomization)
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
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Connection": "keep-alive",
        "DNT": "1",
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

# --- API Key Management Logic ---
# REMOVED: load_and_update_api_keys, _key_refresh_loop, start_key_refresh_task
# REMOVED: get_next_api_key (replaced by direct setting usage)
# REMOVED: DB update tasks (_run_db_task, _mark_key_used_task, etc.) - No DB keys to manage

# --- Retry Logic ---
def should_retry(exception: BaseException) -> bool:
    """Determines if an HTTP request should be retried based on exception type and status code."""
    # DO NOT RETRY on APIKeyFailedError - the key is bad, retrying won't help.
    if isinstance(exception, APIKeyFailedError):
        logger.critical(f"Single API Key failed ({exception}). Cannot retry LLM call.")
        return False
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        # Retry ONLY on transient server errors (500, 502, 503, 504)
        # DO NOT retry 429 (Rate Limit) or 401/403 (Auth) as these indicate the key failed.
        should = status_code in [500, 502, 503, 504]
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
        logger.critical("API key missing/invalid, cannot retry.")
        return False

    logger.warning(f"Not retrying unhandled exception type {type(exception).__name__}: {exception}")
    return False

# --- Main LLM API Call Function ---
@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception(should_retry))
async def call_llm_api(
    client: httpx.AsyncClient,
    prompt: str,
    model: Optional[str] = None,
    image_data: Optional[str] = None # Optional base64 encoded image data
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (handles OpenRouter) using the SINGLE API key from settings.
    Handles retries for transient errors. Triggers proxy rotation on key failure.
    Returns parsed JSON or {'raw_inference': content}.
    Raises APIKeyFailedError or NoValidApiKeyError on critical key issues.
    """
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise NoValidApiKeyError("OPENROUTER_API_KEY is not configured in settings.")

    key_suffix = api_key[-4:]
    key_id_for_logs = "SettingsKey" # Identifier for logs

    endpoint = None
    headers = {}
    # Use provided model, fallback to standard, then premium, then hardcoded default
    selected_model = model or settings.STANDARD_REPORT_MODEL or settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-flash-latest"

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Use AGENCY_BASE_URL directly now
            "HTTP-Referer": str(settings.AGENCY_BASE_URL),
            "X-Title": settings.PROJECT_NAME,
        }
        messages = []
        # Assume the single key is for a visual model if image_data is provided
        is_vision_model = bool(image_data)

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
                  logger.warning(f"[LLM Call] Image data provided but model '{selected_model}' might not be visual or key doesn't support it. Sending text only.")

        payload = {"model": selected_model, "messages": messages}
        # Add other parameters like temperature, max_tokens from settings if needed
        # payload["temperature"] = getattr(settings, 'LLM_TEMPERATURE', 0.7)
        # payload["max_tokens"] = getattr(settings, 'LLM_MAX_TOKENS', 4096)

    else:
        logger.error(f"[LLM Call] Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}.")
        raise ValueError(f"Unsupported LLM_PROVIDER configured: {settings.LLM_PROVIDER}")

    # --- Make API Call ---
    logger.debug(f"[LLM Call] Sending request to {endpoint} using model {selected_model} with key {key_id_for_logs} (...{key_suffix})")
    try:
        response = await client.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT)

        # Check for specific errors BEFORE raise_for_status to handle key failure correctly
        if response.status_code in [401, 403, 429]:
            error_reason = f"Key {key_id_for_logs} failed with status {response.status_code}."
            logger.critical(f"[LLM Call] CRITICAL: {error_reason} Triggering proxy rotation.")
            signal_proxy_rotation() # Signal rotation for *next* client use
            # Raise specific error to stop retries for THIS call
            raise APIKeyFailedError(error_reason)

        # Raise for other non-2xx codes if not handled above
        response.raise_for_status() # This will trigger retry for 5xx errors via should_retry
        result = response.json()

        # --- Parse Response ---
        if "error" in result:
             error_details = result["error"]
             error_message = error_details.get('message', str(error_details))
             logger.error(f"[LLM Call] API returned error in response body: {error_message}")
             # Check if this body error indicates an invalid key
             error_str_lower = error_message.lower()
             if ('invalid api key' in error_str_lower or 'authentication error' in error_str_lower or 'incorrect api key' in error_str_lower or 'api key is invalid' in error_str_lower):
                 error_reason = f"Key {key_id_for_logs} reported as invalid by API in response body."
                 logger.critical(f"[LLM Call] CRITICAL: {error_reason} Triggering proxy rotation.")
                 signal_proxy_rotation()
                 raise APIKeyFailedError(error_reason) # Raise to stop retries
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
    except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
        # These errors are handled by the `should_retry` logic or specific checks above.
        # Re-raise them so Tenacity can decide whether to retry based on `should_retry`.
        logger.debug(f"Raising exception for Tenacity retry check: {type(e).__name__}")
        raise e
    except (APIKeyFailedError, NoValidApiKeyError):
        # These should not be retried. Re-raise to stop.
        raise
    except Exception as e:
        # Catch any other unexpected errors during the API call or processing
        logger.error(f"[LLM Call] Unexpected error during API call/processing for key {key_id_for_logs}: {e}", exc_info=True)
        # Do not retry unexpected errors by default
        return None # Indicate failure for unexpected errors