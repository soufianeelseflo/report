import httpx
from typing import Optional, Dict, Any
import json
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Corrected relative import for package structure
from autonomous_agency.app.core.config import settings

# Use tenacity for retrying API calls
RETRY_WAIT = wait_fixed(3)
RETRY_ATTEMPTS = 3
LLM_TIMEOUT = 90.0 # Increased timeout for potentially long generations

def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    proxy_url = settings.PROXY_URL
    if proxy_url:
        return {"http://": proxy_url, "https://": proxy_url}
    return None

async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies."""
    proxies = get_proxy_map()
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    timeout = httpx.Timeout(LLM_TIMEOUT + 10.0, connect=10.0) # Ensure connect timeout is reasonable

    headers = {
        "User-Agent": f"AutonomousAgencyBot/1.0 ({settings.AGENCY_BASE_URL})", # Identify bot
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

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=RETRY_WAIT, retry=retry_if_exception_type(httpx.RequestError))
async def call_llm_api(client: httpx.AsyncClient, prompt: str, model: str = "google/gemini-1.5-pro-latest") -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM API (specifically handles OpenRouter).
    Uses the specified model, falling back to a default if needed.
    """
    api_key = None
    endpoint = None
    headers = {}
    selected_model = model # Use the passed model by default

    # --- Configure based on LLM Provider ---
    if settings.LLM_PROVIDER == "openrouter":
        api_key = settings.OPENROUTER_API_KEY
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        if not api_key:
            print("[LLM Call] CRITICAL: OpenRouter API Key (OPENROUTER_API_KEY) not configured.")
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.HTTP_REFERER or settings.AGENCY_BASE_URL, # Recommended by OpenRouter
            "X-Title": settings.X_TITLE or settings.PROJECT_NAME, # Recommended by OpenRouter
        }
        # Use the specific model requested, e.g., "google/gemini-1.5-pro-latest"
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            # "max_tokens": 2048, # Optional: Adjust as needed
            # "temperature": 0.7, # Optional: Adjust as needed
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
        response.raise_for_status()
        result = response.json()

        # --- Parse Response (OpenAI/OpenRouter format assumed) ---
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            try:
                # Clean potential markdown ```json ... ``` blocks
                cleaned_content = content.strip()
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_content)
                if json_match:
                    cleaned_content = json_match.group(1).strip()
                else:
                    # If no markdown block, assume it might be direct JSON
                    pass

                # Attempt to parse as JSON
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                # If not JSON, return the raw string content
                # print(f"[LLM Call] Response content is not JSON: {content[:100]}...") # Debugging
                return {"raw_inference": content}
        else:
             print(f"[LLM Call] No content found in LLM response choices: {result}")
             return None

    except httpx.HTTPStatusError as e:
        print(f"[LLM Call] API request failed: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429: print("[LLM Call] Rate limit hit.")
        # Add specific handling for model not found (e.g., 404 or specific error code from OpenRouter)
        if e.response.status_code == 400 or e.response.status_code == 404:
             print(f"[LLM Call] Possible issue with model '{selected_model}' or request format.")
        return None
    except Exception as e:
        print(f"[LLM Call] Error calling API: {e}")
        import traceback
        traceback.print_exc()
        return None