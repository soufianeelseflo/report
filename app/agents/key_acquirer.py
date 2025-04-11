import asyncio
import httpx
import re
import json
import random
import traceback
from typing import Optional, Dict, Any, List, Tuple
from bs4 import BeautifulSoup
import base64 # For potential future use with captcha solvers

from sqlalchemy.ext.asyncio import AsyncSession

# Corrected relative imports
from Acumenis.app.core.config import settings
from Acumenis.app.db.base import get_worker_session
from Acumenis.app.db import crud
from Acumenis.app.agents.agent_utils import get_httpx_client # Reuse client creation logic initially

# --- Constants ---
# WARNING: These URLs and selectors WILL change and break the script.
# Make these configurable in settings? For now, hardcoded examples.
TEMP_EMAIL_PROVIDER_URL = "https://inboxes.com/" # Example provider
OPENROUTER_SIGNUP_URL = "https://openrouter.ai/auth?callbackUrl=%2Fkeys" # Check current URL
OPENROUTER_KEYS_URL = "https://openrouter.ai/keys" # Check current URL

# Example Selectors (MUST be updated by inspecting the live websites)
TEMP_EMAIL_SELECTOR = "#email" # Example CSS selector for the email address
SIGNUP_EMAIL_FIELD_NAME = "email" # Example name attribute for email input
SIGNUP_PASSWORD_FIELD_NAME = "password" # Example name attribute for password input
SIGNUP_SUBMIT_SELECTOR = "button[type='submit']" # Example selector for submit button
API_KEY_GENERATE_SELECTOR = "button:contains('Generate Key')" # Example selector
API_KEY_DISPLAY_SELECTOR = "input[readonly]" # Example selector for the key field

DEFAULT_TIMEOUT = 45.0 # Increased timeout
ACQUISITION_RETRY_ATTEMPTS = 2 # Retries for transient network errors per step

# --- Helper Functions ---

def get_random_proxy() -> Optional[str]:
    """Selects a random proxy from the configured list or single URL."""
    proxies = settings.PROXY_LIST or []
    if not proxies and settings.PROXY_URL:
        proxies = [settings.PROXY_URL]

    if not proxies:
        # print("[KeyAcquirer] No proxies configured.") # Reduce log noise
        return None

    valid_proxies = [p for p in proxies if re.match(r"^(http|https|socks\d?)://", p)]
    if not valid_proxies:
         print("[KeyAcquirer] No VALID proxies configured (check format http://user:pass@host:port).")
         return None

    return random.choice(valid_proxies)

async def create_proxied_client() -> Optional[httpx.AsyncClient]:
    """Creates an httpx client configured with a randomly selected proxy."""
    proxy_url = get_random_proxy()
    if not proxy_url:
        # print("[KeyAcquirer] No valid proxy available for this attempt.") # Reduce log noise
        return None # Proceed without proxy? Or fail? Fail for now.

    proxies = {"http://": proxy_url, "https://": proxy_url}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=2)
    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0)
    # Use a more common user agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"', # Or Windows/macOS
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        client = httpx.AsyncClient(
            proxies=proxies,
            limits=limits,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            http2=True
        )
        # print(f"[KeyAcquirer] Client created with proxy: {proxy_url}") # Reduce log noise
        return client
    except Exception as e:
        print(f"[KeyAcquirer] Failed to create client with proxy {proxy_url}: {e}")
        return None

# --- Web Interaction Steps (Highly Unreliable - Expect Breakage & Blocks) ---

async def get_temporary_email(client: httpx.AsyncClient) -> Optional[str]:
    """Attempts to get a temporary email. Logs URL if selector fails."""
    print(f"[KeyAcquirer] Attempting temporary email from {TEMP_EMAIL_PROVIDER_URL}...")
    try:
        response = await client.get(TEMP_EMAIL_PROVIDER_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        email_element = soup.select_one(TEMP_EMAIL_SELECTOR)

        if email_element:
            temp_email = email_element.get('value') or email_element.get('data-original-title') or email_element.text # Try common attributes
            temp_email = temp_email.strip()
            if temp_email and '@' in temp_email:
                print(f"[KeyAcquirer] Obtained temporary email: {temp_email}")
                return temp_email
            else:
                 print(f"[KeyAcquirer] Failed to extract valid email from element: {email_element}")
        else:
            print(f"[KeyAcquirer] Could not find temp email element '{TEMP_EMAIL_SELECTOR}'. Website structure changed? Operator check needed at {TEMP_EMAIL_PROVIDER_URL}")

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error getting temp email: {e}")
    except Exception as e:
        print(f"[KeyAcquirer] Error parsing temp email page: {e}")
    return None

async def signup_openrouter(client: httpx.AsyncClient, temp_email: str) -> Tuple[bool, str]:
    """Attempts signup. Returns (success, status_message). Logs CAPTCHA info."""
    print(f"[KeyAcquirer] Attempting OpenRouter signup for {temp_email}...")
    generated_password = f"Acum3n!s_{random.randint(1000000, 9999999)}" # More complex password

    try:
        # GET signup page first to potentially capture CSRF, cookies, or initial state
        get_response = await client.get(OPENROUTER_SIGNUP_URL, timeout=DEFAULT_TIMEOUT)
        get_response.raise_for_status()
        # Potentially parse get_response.text for tokens if needed

        # Prepare POST data
        payload = {
            SIGNUP_EMAIL_FIELD_NAME: temp_email,
            SIGNUP_PASSWORD_FIELD_NAME: generated_password,
            # Add any other hidden fields discovered from the form
        }
        print("[KeyAcquirer] Preparing signup POST...")

        # *** CAPTCHA Handling ***
        # Check response HTML for CAPTCHA elements (hCaptcha, reCAPTCHA)
        soup_get = BeautifulSoup(get_response.text, 'html.parser')
        h_captcha_sitekey = soup_get.find('div', {'class': 'h-captcha'}) or soup_get.find('iframe', {'title': 'hCaptcha challenge'})
        g_recaptcha_sitekey = soup_get.find('div', {'class': 'g-recaptcha'})

        if h_captcha_sitekey or g_recaptcha_sitekey:
            captcha_type = "hCaptcha" if h_captcha_sitekey else "reCAPTCHA"
            site_key = h_captcha_sitekey.get('data-sitekey') if h_captcha_sitekey else g_recaptcha_sitekey.get('data-sitekey') if g_recaptcha_sitekey else 'Unknown'
            captcha_info = f"CAPTCHA Detected ({captcha_type}, SiteKey: {site_key}). Manual intervention required. URL: {OPENROUTER_SIGNUP_URL}"
            print(f"[KeyAcquirer] {captcha_info}")
            # TODO: Integrate CAPTCHA solving service API call here if budget allows
            # captcha_solution = await solve_captcha(captcha_type, site_key, OPENROUTER_SIGNUP_URL)
            # if captcha_solution: payload['captcha-response-field'] = captcha_solution
            # else: return False, captcha_info # Fail if no solution
            return False, captcha_info # Currently fails here

        # Make the POST request
        response_post = await client.post(OPENROUTER_SIGNUP_URL, data=payload, timeout=DEFAULT_TIMEOUT)

        # Analyze response
        if response_post.status_code in [200, 201, 302]: # Check for success/redirect
             # Check for success text or redirect location
             if "Verification email sent" in response_post.text or "check your email" in response_post.text.lower():
                 print(f"[KeyAcquirer] Signup POST successful (pending verification) for {temp_email}.")
                 return True, "Signup successful, verification needed."
             elif response_post.is_redirect:
                  print(f"[KeyAcquirer] Signup resulted in redirect to {response_post.headers.get('location')}. Assuming success (pending verification).")
                  return True, "Signup successful (redirect), verification needed."
             else:
                  # Status 200 but no clear success message? Might be error page.
                  print(f"[KeyAcquirer] Signup POST status {response_post.status_code}, but success message unclear. Response: {response_post.text[:200]}...")
                  return False, f"Signup status {response_post.status_code}, outcome unclear."
        elif response_post.status_code == 429:
             print(f"[KeyAcquirer] Signup failed for {temp_email} - Rate Limited (429).")
             return False, "Rate Limited during signup."
        elif response_post.status_code == 403:
             print(f"[KeyAcquirer] Signup failed for {temp_email} - Forbidden (403). Likely IP block or detection.")
             return False, "Forbidden (403) during signup."
        else:
             print(f"[KeyAcquirer] Signup POST failed for {temp_email}. Status: {response_post.status_code}. Response: {response_post.text[:200]}...")
             return False, f"Signup failed with status {response_post.status_code}."

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error during signup: {e}")
        return False, f"Network error: {e}"
    except Exception as e:
        print(f"[KeyAcquirer] Error processing signup: {e}")
        traceback.print_exc()
        return False, f"Unexpected signup error: {e}"

async def check_and_verify_email(client: httpx.AsyncClient, temp_email: str) -> Tuple[bool, str]:
    """Checks temp inbox, attempts to click verification link. Logs URL if fails."""
    print(f"[KeyAcquirer] Checking inbox for {temp_email} verification...")
    verification_link = None
    max_check_attempts = 8 # Reduced attempts
    check_interval = 10 # seconds

    for attempt in range(max_check_attempts):
        if attempt > 0: await asyncio.sleep(check_interval) # Wait before next check
        print(f"[KeyAcquirer] Inbox check attempt {attempt + 1}/{max_check_attempts}...")
        try:
            # Re-fetch inbox page (simple approach, might need stateful interaction)
            inbox_response = await client.get(TEMP_EMAIL_PROVIDER_URL, timeout=DEFAULT_TIMEOUT)
            inbox_response.raise_for_status()
            inbox_soup = BeautifulSoup(inbox_response.text, 'html.parser')

            # Find link containing 'openrouter.ai/verify' or similar pattern
            link_element = inbox_soup.find('a', href=re.compile(r'openrouter\.ai/(auth/verify|verify)')) # More robust regex

            if link_element and link_element.get('href'):
                verification_link = link_element['href']
                # Ensure link is absolute
                if not verification_link.startswith('http'):
                    base_url = urlparse(OPENROUTER_SIGNUP_URL).scheme + "://" + urlparse(OPENROUTER_SIGNUP_URL).netloc
                    verification_link = urljoin(base_url, verification_link)
                print(f"[KeyAcquirer] Found verification link: {verification_link}")
                break # Exit loop
            else:
                 print("[KeyAcquirer] Verification email not found yet.")

        except httpx.RequestError as e:
            print(f"[KeyAcquirer] Network error checking inbox: {e}")
        except Exception as e:
            print(f"[KeyAcquirer] Error parsing inbox: {e}")

    if verification_link:
        print(f"[KeyAcquirer] Attempting to visit verification link: {verification_link}")
        try:
            verify_response = await client.get(verification_link, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
            verify_response.raise_for_status()
            # Check for success indicator (highly dependent on OpenRouter's response)
            # Look for keywords, successful status code, or redirection to a specific page like /keys
            if ("Email verified" in verify_response.text or "verification successful" in verify_response.text.lower() or
                (verify_response.status_code == 200 and "/keys" in str(verify_response.url))):
                print(f"[KeyAcquirer] Email verification successful for {temp_email}.")
                return True, "Verification successful."
            else:
                print(f"[KeyAcquirer] Verification link visited, but success unclear. Status: {verify_response.status_code}, URL: {verify_response.url}")
                return False, f"Verification unclear (Status: {verify_response.status_code})."
        except httpx.RequestError as e:
            print(f"[KeyAcquirer] Network error visiting verification link: {e}")
            return False, f"Network error during verification: {e}"
        except Exception as e:
            print(f"[KeyAcquirer] Error processing verification response: {e}")
            return False, f"Error processing verification: {e}"
    else:
        msg = f"Verification link not found for {temp_email} after {max_check_attempts} attempts. Operator check needed at {TEMP_EMAIL_PROVIDER_URL}"
        print(f"[KeyAcquirer] {msg}")
        return False, msg

async def extract_api_key(client: httpx.AsyncClient) -> Tuple[Optional[str], str]:
    """Attempts to navigate to keys page and extract key. Logs URL if fails."""
    print("[KeyAcquirer] Attempting to extract API key from keys page...")
    try:
        # Assume client is logged in from verification step. Might need explicit login if session lost.
        keys_page_response = await client.get(OPENROUTER_KEYS_URL, timeout=DEFAULT_TIMEOUT)
        keys_page_response.raise_for_status()
        keys_soup = BeautifulSoup(keys_page_response.text, 'html.parser')

        # Attempt to find the key directly (common pattern: readonly input)
        key_element = keys_soup.find('input', {'readonly': True, 'value': re.compile(r'^sk-or-')})
        if not key_element: # Fallback: try any input with the key pattern
             key_element = keys_soup.find('input', {'value': re.compile(r'^sk-or-')})
        if not key_element: # Fallback: try code blocks
             code_elements = keys_soup.find_all('code')
             for code_el in code_elements:
                  if re.match(r'^sk-or-', code_el.text.strip()):
                       key_element = code_el
                       break

        if key_element:
            api_key = key_element.get('value') or key_element.text # Handle input value or code text
            api_key = api_key.strip()
            if api_key and api_key.startswith("sk-or-"):
                print(f"[KeyAcquirer] Successfully extracted API key: {api_key[:10]}...")
                return api_key, "Key extracted successfully."
            else:
                msg = f"Found potential key element but content invalid: {str(key_element)[:100]}..."
                print(f"[KeyAcquirer] {msg}")
                return None, msg
        else:
            # If key not found, maybe needs generation? Try finding a "Generate" button.
            # This part is highly speculative and likely needs JS execution (Playwright)
            generate_button = keys_soup.find('button', string=re.compile(r'generate|create', re.IGNORECASE))
            if generate_button:
                 msg = f"API key not found directly. 'Generate Key' button detected. Manual click likely required at {OPENROUTER_KEYS_URL}"
                 print(f"[KeyAcquirer] {msg}")
                 return None, msg
            else:
                 msg = f"Could not find API key element or Generate button. Page structure changed or login failed? Operator check needed at {OPENROUTER_KEYS_URL}"
                 print(f"[KeyAcquirer] {msg}")
                 # print(f"Keys page content sample: {keys_page_response.text[:500]}") # Debugging
                 return None, msg

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error accessing keys page: {e}")
        return None, f"Network error accessing keys page: {e}"
    except Exception as e:
        print(f"[KeyAcquirer] Error parsing keys page: {e}")
        traceback.print_exc()
        return None, f"Error parsing keys page: {e}"

# --- Main Acquisition Orchestration ---

async def acquire_one_key(session: AsyncSession) -> Optional[str]:
    """Attempts the full sequence for one key. Logs detailed status."""
    client = None
    proxy_used = get_random_proxy() # Get proxy info for logging
    temp_email_used = None
    api_key = None
    status_log = []
    start_time = asyncio.get_event_loop().time()

    print(f"\n--- Starting Key Acquisition Attempt (Proxy: {proxy_used or 'None'}) ---")

    try:
        client = await create_proxied_client()
        if not client:
            raise Exception("Failed to create proxied client.")
        status_log.append("Client created.")

        # 1. Get Temporary Email
        temp_email_used = await get_temporary_email(client)
        if not temp_email_used:
            raise Exception("Failed to obtain temporary email.")
        status_log.append(f"Temp email obtained: {temp_email_used}")

        # 2. Sign Up
        signup_success, signup_msg = await signup_openrouter(client, temp_email_used)
        status_log.append(f"Signup attempt: {signup_msg}")
        if not signup_success:
            raise Exception(f"Signup step failed: {signup_msg}")

        # 3. Check & Verify Email
        verification_success, verify_msg = await check_and_verify_email(client, temp_email_used)
        status_log.append(f"Verification attempt: {verify_msg}")
        if not verification_success:
            raise Exception(f"Verification step failed: {verify_msg}")

        # 4. Extract API Key
        api_key, extract_msg = await extract_api_key(client)
        status_log.append(f"Key extraction attempt: {extract_msg}")
        if not api_key:
            raise Exception(f"Key extraction step failed: {extract_msg}")

        # 5. Store Key if successful
        print(f"[KeyAcquirer] SUCCESS! Acquired key: {api_key[:10]}... for email {temp_email_used}")
        await crud.add_api_key(
            db=session,
            key=api_key,
            provider="openrouter",
            email=temp_email_used,
            proxy=proxy_used,
            notes=f"Acquired autonomously. Status log: {' -> '.join(status_log)}"
        )
        # No commit here, handled by the calling worker loop per task
        return api_key # Return the acquired key

    except Exception as e:
        error_msg = f"Attempt FAILED: {e}. Log: {' -> '.join(status_log)}"
        print(f"[KeyAcquirer] {error_msg}")
        # Optionally store failure info in DB?
        # await crud.log_acquisition_attempt(session, status='failed', email=temp_email_used, proxy=proxy_used, error=str(e))
        return None # Indicate failure
    finally:
        if client:
            await client.aclose()
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"--- Finished Key Acquisition Attempt (Success: {bool(api_key)}, Duration: {elapsed:.2f}s) ---")


async def run_acquisition_process(target_keys: int, shutdown_event: asyncio.Event):
    """Runs the key acquisition process until target is reached or stops."""
    print(f"[KeyAcquirer] Starting acquisition process. Target: {target_keys} keys.")
    acquired_count = 0
    attempt_count = 0
    session: AsyncSession = None

    # Initial count of existing keys
    try:
        session = await get_worker_session()
        initial_keys = await crud.get_active_api_keys(session, provider="openrouter")
        acquired_count = len(initial_keys)
        print(f"[KeyAcquirer] Initial active OpenRouter keys in DB: {acquired_count}")
    except Exception as e:
        print(f"[KeyAcquirer] Error checking initial keys: {e}")
    finally:
        if session: await session.close()

    max_concurrent_attempts = settings.KEY_ACQUIRER_CONCURRENCY or 5 # Increase concurrency
    active_tasks = set()

    while acquired_count < target_keys and not shutdown_event.is_set():
        # Clean up completed tasks and update count
        if len(active_tasks) >= max_concurrent_attempts:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result_key = await task # Await task result to catch exceptions
                    if result_key:
                        acquired_count += 1
                        print(f"[KeyAcquirer] Progress: {acquired_count}/{target_keys} keys acquired.")
                except Exception as task_e:
                     print(f"[KeyAcquirer] Acquisition task failed with error: {task_e}")
                     # Error already logged within acquire_one_key
        else:
            # Start a new acquisition attempt
            attempt_count += 1
            print(f"[KeyAcquirer] Starting attempt #{attempt_count} ({len(active_tasks)+1}/{max_concurrent_attempts}). Current count: {acquired_count}/{target_keys}")
            session = await get_worker_session() # Get session for this attempt
            task = asyncio.create_task(acquire_one_key(session))
            active_tasks.add(task)
            # Ensure session is closed after task finishes
            task.add_done_callback(lambda t: asyncio.create_task(session.close()) if session else None)

        # Check if target reached or shutdown signaled
        if acquired_count >= target_keys:
            print(f"[KeyAcquirer] Target of {target_keys} keys reached.")
            break
        if shutdown_event.is_set():
             print("[KeyAcquirer] Shutdown signal received, stopping new attempts.")
             break

        # Optional: Small delay between starting new tasks if needed
        await asyncio.sleep(0.5) # Short delay

    # Wait for any remaining tasks to finish if shutdown wasn't immediate
    if active_tasks:
        print(f"[KeyAcquirer] Waiting for {len(active_tasks)} remaining acquisition attempts to finish...")
        results = await asyncio.gather(*active_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, str) and result.startswith("sk-or-"): # Check if it's a successfully acquired key
                 # Avoid double counting if target was met exactly before gather
                 # This check is tricky, might slightly undercount if gather finishes after loop break
                 pass # Count was likely updated already
             elif isinstance(result, Exception):
                  print(f"[KeyAcquirer] A remaining task failed: {result}")
             # Else: Task returned None (failure already logged)

    # Final count check
    try:
        session = await get_worker_session()
        final_keys = await crud.get_active_api_keys(session, provider="openrouter")
        final_acquired_count = len(final_keys)
        print(f"[KeyAcquirer] Acquisition process finished. Total attempts: {attempt_count}. Final active keys in DB: {final_acquired_count}")
    except Exception as e:
        print(f"[KeyAcquirer] Error checking final keys: {e}")
        print(f"[KeyAcquirer] Acquisition process finished. Total attempts: {attempt_count}. Last known count: {acquired_count}")
    finally:
        if session: await session.close()