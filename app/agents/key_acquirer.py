import asyncio
import httpx
import re
import json
import random
import traceback
from typing import Optional, Dict, Any, List, Tuple
from bs4 import BeautifulSoup
import base64 # For potential future use with captcha solvers
from urllib.parse import urlparse, urljoin # Added for URL handling

from sqlalchemy.ext.asyncio import AsyncSession

# Corrected relative imports
# Ensure Acumenis is the root package name if running modules directly
# If running via the main app, relative imports might need adjustment based on execution context
try:
    # Attempt package-relative imports first
    from Acumenis.app.core.config import settings
    from Acumenis.app.db.base import get_worker_session
    from Acumenis.app.db import crud
    from Acumenis.app.agents.agent_utils import get_httpx_client # Reuse client creation logic initially
except ImportError:
    # Fallback for direct execution or different structure
    print("[KeyAcquirer] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db.base import get_worker_session
    from app.db import crud
    from app.agents.agent_utils import get_httpx_client


# --- Constants ---
# Use settings for URLs and selectors, with defaults
TEMP_EMAIL_PROVIDER_URL = getattr(settings, 'TEMP_EMAIL_PROVIDER_URL', "https://inboxes.com/")
OPENROUTER_SIGNUP_URL = getattr(settings, 'OPENROUTER_SIGNUP_URL', "https://openrouter.ai/auth?callbackUrl=%2Fkeys")
OPENROUTER_KEYS_URL = getattr(settings, 'OPENROUTER_KEYS_URL', "https://openrouter.ai/keys")
TEMP_EMAIL_SELECTOR = getattr(settings, 'TEMP_EMAIL_SELECTOR', "#email")
SIGNUP_EMAIL_FIELD_NAME = getattr(settings, 'SIGNUP_EMAIL_FIELD_NAME', "email")
SIGNUP_PASSWORD_FIELD_NAME = getattr(settings, 'SIGNUP_PASSWORD_FIELD_NAME', "password")
SIGNUP_SUBMIT_SELECTOR = getattr(settings, 'SIGNUP_SUBMIT_SELECTOR', "button[type='submit']")
API_KEY_DISPLAY_SELECTOR = getattr(settings, 'API_KEY_DISPLAY_SELECTOR', "input[readonly][value^='sk-or-']")

DEFAULT_TIMEOUT = 60.0 # Increased timeout for potentially slow pages/proxies
ACQUISITION_RETRY_ATTEMPTS = 1 # Reduce retries per step to fail faster on persistent issues

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
         print("[KeyAcquirer] WARNING: No VALID proxies configured (check format http://user:pass@host:port).")
         return None

    return random.choice(valid_proxies)

async def create_proxied_client() -> Optional[httpx.AsyncClient]:
    """Creates an httpx client configured with a randomly selected proxy."""
    proxy_url = get_random_proxy()
    proxies = None
    proxy_display = "None"
    if proxy_url:
        proxies = {"http://": proxy_url, "https://": proxy_url}
        proxy_display = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
        print(f"[KeyAcquirer] Attempting client creation with proxy: {proxy_display}")
    else:
        # If circumvention is key, failing without a proxy might be desired
        print("[KeyAcquirer] ERROR: No valid proxy available for this attempt. Cannot proceed without proxy.")
        return None # Fail if no proxy

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=2)
    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0)
    # Use a more common user agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36", # Slightly updated UA
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
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
            follow_redirects=True, # Keep redirects on for simplicity unless causing issues
            http2=True
        )
        # print(f"[KeyAcquirer] Client created successfully.")
        return client
    except Exception as e:
        print(f"[KeyAcquirer] Failed to create client with proxy {proxy_display}: {e}")
        return None

# --- Web Interaction Steps (Highly Unreliable - Expect Breakage & Blocks) ---

async def get_temporary_email(client: httpx.AsyncClient) -> Tuple[Optional[str], str]:
    """Attempts to get a temporary email. Returns (email, status_message)."""
    step_name = "Get Temporary Email"
    print(f"[KeyAcquirer] [{step_name}] Attempting from {TEMP_EMAIL_PROVIDER_URL}...")
    try:
        response = await client.get(TEMP_EMAIL_PROVIDER_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        email_element = soup.select_one(TEMP_EMAIL_SELECTOR)

        if email_element:
            # Try common attributes/methods to extract email
            temp_email = email_element.get('value') or \
                         email_element.get('data-original-title') or \
                         email_element.get('data-clipboard-text') or \
                         email_element.text
            temp_email = temp_email.strip() if temp_email else None

            if temp_email and '@' in temp_email:
                msg = f"Obtained temporary email: {temp_email}"
                print(f"[KeyAcquirer] [{step_name}] SUCCESS: {msg}")
                return temp_email, f"SUCCESS: {msg}"
            else:
                 msg = f"FAILED: Failed to extract valid email content from element: {str(email_element)[:100]}..."
                 print(f"[KeyAcquirer] [{step_name}] {msg}")
                 return None, msg
        else:
            msg = f"FAILED: Could not find temp email element '{TEMP_EMAIL_SELECTOR}'. Website structure changed? Operator check needed at {TEMP_EMAIL_PROVIDER_URL}"
            print(f"[KeyAcquirer] [{step_name}] {msg}")
            return None, msg

    except httpx.HTTPStatusError as e:
        msg = f"FAILED: HTTP error {e.response.status_code} getting temp email: {e.response.text[:100]}..."
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return None, msg
    except httpx.RequestError as e:
        msg = f"FAILED: Network error getting temp email: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return None, msg
    except Exception as e:
        msg = f"FAILED: Error parsing temp email page: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        traceback.print_exc()
        return None, msg

async def signup_openrouter(client: httpx.AsyncClient, temp_email: str) -> Tuple[bool, str]:
    """Attempts signup. Returns (success, status_message). Logs CAPTCHA info."""
    step_name = "Signup OpenRouter"
    print(f"[KeyAcquirer] [{step_name}] Attempting for {temp_email}...")
    generated_password = f"Acum3n!s_{random.randint(1000000, 9999999)}" # More complex password

    try:
        # GET signup page first
        get_response = await client.get(OPENROUTER_SIGNUP_URL, timeout=DEFAULT_TIMEOUT)
        get_response.raise_for_status()
        soup_get = BeautifulSoup(get_response.text, 'html.parser')

        # *** CAPTCHA Detection ***
        is_captcha_present = soup_get.find('div', {'class': re.compile(r'captcha|challenge', re.I)}) or \
                             soup_get.find('iframe', {'title': re.compile(r'captcha|challenge', re.I)}) or \
                             'captcha' in get_response.text.lower() or \
                             'turnstile' in get_response.text.lower() or \
                             'challenge' in get_response.text.lower() # Broader check

        if is_captcha_present:
            captcha_info = f"FAILED: CAPTCHA Detected during signup attempt. Cannot proceed automatically. URL: {OPENROUTER_SIGNUP_URL}"
            print(f"[KeyAcquirer] [{step_name}] {captcha_info}")
            return False, captcha_info # Fail immediately

        # Prepare POST data (Extract CSRF if needed)
        csrf_token = None
        csrf_input = soup_get.find('input', {'name': re.compile(r'csrf', re.I)})
        if csrf_input:
            csrf_token = csrf_input.get('value')

        payload = {
            SIGNUP_EMAIL_FIELD_NAME: temp_email,
            SIGNUP_PASSWORD_FIELD_NAME: generated_password,
        }
        if csrf_token:
            payload[csrf_input['name']] = csrf_token # Add CSRF if found

        print(f"[KeyAcquirer] [{step_name}] Preparing signup POST...")
        response_post = await client.post(OPENROUTER_SIGNUP_URL, data=payload, timeout=DEFAULT_TIMEOUT)
        status_code = response_post.status_code
        response_text_lower = response_post.text.lower()

        # Analyze response
        if status_code in [200, 201, 302] or response_post.is_redirect:
             if "verification email sent" in response_text_lower or \
                "check your email" in response_text_lower or \
                "confirm your email" in response_text_lower or \
                response_post.is_redirect:
                 msg = f"SUCCESS: Signup POST successful (Status: {status_code}, Pending Verification) for {temp_email}."
                 print(f"[KeyAcquirer] [{step_name}] {msg}")
                 return True, msg
             else:
                  msg = f"FAILED: Signup POST status {status_code}, but success message unclear. Response: {response_post.text[:200]}..."
                  print(f"[KeyAcquirer] [{step_name}] {msg}")
                  return False, msg
        elif status_code == 429:
             msg = f"FAILED: Signup failed for {temp_email} - Rate Limited (429)."
             print(f"[KeyAcquirer] [{step_name}] {msg}")
             return False, msg
        elif status_code == 403:
             msg = f"FAILED: Signup failed for {temp_email} - Forbidden (403). Likely IP block or detection."
             print(f"[KeyAcquirer] [{step_name}] {msg}")
             return False, msg
        else:
             msg = f"FAILED: Signup POST failed for {temp_email}. Status: {status_code}. Response: {response_post.text[:200]}..."
             print(f"[KeyAcquirer] [{step_name}] {msg}")
             return False, msg

    except httpx.HTTPStatusError as e:
        msg = f"FAILED: HTTP error {e.response.status_code} during signup GET/POST: {e.response.text[:100]}..."
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return False, msg
    except httpx.RequestError as e:
        msg = f"FAILED: Network error during signup: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return False, msg
    except Exception as e:
        msg = f"FAILED: Unexpected error processing signup: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        traceback.print_exc()
        return False, msg

async def check_and_verify_email(client: httpx.AsyncClient, temp_email: str) -> Tuple[bool, str]:
    """Checks temp inbox, attempts to click verification link. Returns (success, status_message)."""
    step_name = "Verify Email"
    print(f"[KeyAcquirer] [{step_name}] Checking inbox for {temp_email} verification...")
    verification_link = None
    max_check_attempts = 8
    check_interval = 15 # Increased interval

    for attempt in range(max_check_attempts):
        if attempt > 0: await asyncio.sleep(check_interval)
        print(f"[KeyAcquirer] [{step_name}] Inbox check attempt {attempt + 1}/{max_check_attempts}...")
        try:
            # Fetch inbox page
            inbox_response = await client.get(TEMP_EMAIL_PROVIDER_URL, timeout=DEFAULT_TIMEOUT)
            inbox_response.raise_for_status()
            inbox_soup = BeautifulSoup(inbox_response.text, 'html.parser')

            # Find link - adjust selector based on temp mail provider structure
            link_element = inbox_soup.find('a', href=re.compile(r'openrouter\.ai/(auth/verify|verify|confirm)'))
            # Example refinement: link_element = inbox_soup.select_one('div.email-preview a[href*="openrouter.ai/verify"]')

            if link_element and link_element.get('href'):
                found_link = link_element['href']
                # Ensure link is absolute
                if not found_link.startswith('http'):
                    base_url = urlparse(TEMP_EMAIL_PROVIDER_URL).scheme + "://" + urlparse(TEMP_EMAIL_PROVIDER_URL).netloc
                    try:
                         verification_link = urljoin(base_url, found_link)
                    except ValueError:
                         print(f"[KeyAcquirer] [{step_name}] Failed to resolve relative link: {found_link}")
                         continue
                else:
                     verification_link = found_link

                print(f"[KeyAcquirer] [{step_name}] Found potential verification link: {verification_link}")
                break
            else:
                 print(f"[KeyAcquirer] [{step_name}] Verification email not found yet.")

        except httpx.HTTPStatusError as e:
            print(f"[KeyAcquirer] [{step_name}] HTTP error {e.response.status_code} checking inbox.")
            # Don't immediately fail, maybe transient issue or provider change
        except httpx.RequestError as e:
            print(f"[KeyAcquirer] [{step_name}] Network error checking inbox: {e}")
            # Don't immediately fail
        except Exception as e:
            print(f"[KeyAcquirer] [{step_name}] Error parsing inbox: {e}")
            traceback.print_exc() # Log full trace for parsing errors

    if verification_link:
        print(f"[KeyAcquirer] [{step_name}] Attempting to visit verification link: {verification_link}")
        try:
            verify_response = await client.get(verification_link, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
            verify_response.raise_for_status()
            response_text_lower = verify_response.text.lower()

            if ("email verified" in response_text_lower or \
                "verification successful" in response_text_lower or \
                "account confirmed" in response_text_lower or \
                (verify_response.status_code == 200 and "/keys" in str(verify_response.url))):
                msg = f"SUCCESS: Email verification successful for {temp_email}."
                print(f"[KeyAcquirer] [{step_name}] {msg}")
                return True, msg
            else:
                msg = f"FAILED: Verification link visited, but success unclear. Status: {verify_response.status_code}, URL: {verify_response.url}, Response: {verify_response.text[:100]}..."
                print(f"[KeyAcquirer] [{step_name}] {msg}")
                return False, msg
        except httpx.HTTPStatusError as e:
             msg = f"FAILED: HTTP error {e.response.status_code} visiting verification link: {e.response.text[:100]}..."
             print(f"[KeyAcquirer] [{step_name}] {msg}")
             return False, msg
        except httpx.RequestError as e:
            msg = f"FAILED: Network error visiting verification link: {e}"
            print(f"[KeyAcquirer] [{step_name}] {msg}")
            return False, msg
        except Exception as e:
            msg = f"FAILED: Error processing verification response: {e}"
            print(f"[KeyAcquirer] [{step_name}] {msg}")
            traceback.print_exc()
            return False, msg
    else:
        msg = f"FAILED: Verification link not found for {temp_email} after {max_check_attempts} attempts. Operator check needed at {TEMP_EMAIL_PROVIDER_URL}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return False, msg

async def extract_api_key(client: httpx.AsyncClient) -> Tuple[Optional[str], str]:
    """Attempts to navigate to keys page and extract key. Returns (key, status_message)."""
    step_name = "Extract API Key"
    print(f"[KeyAcquirer] [{step_name}] Attempting from {OPENROUTER_KEYS_URL}...")
    try:
        keys_page_response = await client.get(OPENROUTER_KEYS_URL, timeout=DEFAULT_TIMEOUT)
        keys_page_response.raise_for_status()
        keys_soup = BeautifulSoup(keys_page_response.text, 'html.parser')

        # Attempt to find the key directly using the more specific selector
        key_element = keys_soup.select_one(API_KEY_DISPLAY_SELECTOR)

        # Fallback selectors
        if not key_element: key_element = keys_soup.find('input', {'value': re.compile(r'^sk-or-')})
        if not key_element:
             code_elements = keys_soup.find_all('code')
             for code_el in code_elements:
                  if re.match(r'^sk-or-', code_el.text.strip()):
                       key_element = code_el; break

        if key_element:
            api_key = key_element.get('value') or key_element.text
            api_key = api_key.strip()
            if api_key and api_key.startswith("sk-or-"):
                msg = f"SUCCESS: Successfully extracted API key: {api_key[:10]}..."
                print(f"[KeyAcquirer] [{step_name}] {msg}")
                return api_key, msg
            else:
                msg = f"FAILED: Found potential key element but content invalid: {str(key_element)[:100]}..."
                print(f"[KeyAcquirer] [{step_name}] {msg}")
                return None, msg
        else:
            # Check for common failure reasons
            if "you haven't created any keys yet" in keys_page_response.text.lower():
                 msg = f"FAILED: API key not found. Page indicates no keys generated yet. Manual generation likely required at {OPENROUTER_KEYS_URL}"
            elif "login" in str(keys_page_response.url).lower() or keys_page_response.status_code != 200:
                 msg = f"FAILED: Failed to access keys page (Status: {keys_page_response.status_code}). Likely not logged in or session expired. URL: {keys_page_response.url}"
            else:
                 msg = f"FAILED: Could not find API key element. Page structure changed? Operator check needed at {OPENROUTER_KEYS_URL}"
                 # print(f"Keys page content sample: {keys_page_response.text[:500]}") # Debugging
            print(f"[KeyAcquirer] [{step_name}] {msg}")
            return None, msg

    except httpx.HTTPStatusError as e:
        msg = f"FAILED: HTTP error {e.response.status_code} accessing keys page: {e.response.text[:100]}..."
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return None, msg
    except httpx.RequestError as e:
        msg = f"FAILED: Network error accessing keys page: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        return None, msg
    except Exception as e:
        msg = f"FAILED: Error parsing keys page: {e}"
        print(f"[KeyAcquirer] [{step_name}] {msg}")
        traceback.print_exc()
        return None, msg

# --- Main Acquisition Orchestration ---

async def acquire_one_key(session: AsyncSession) -> Tuple[Optional[str], List[str]]:
    """
    Attempts the full sequence for one key. Commits on success.
    Returns (api_key, status_log)
    """
    client = None
    proxy_used = get_random_proxy()
    temp_email_used = None
    api_key = None
    status_log = ["Start"]
    start_time = asyncio.get_event_loop().time()
    proxy_display = proxy_used.split('@')[-1] if proxy_used and '@' in proxy_used else proxy_used or 'None'
    print(f"\n--- Starting Key Acquisition Attempt (Proxy: {proxy_display}) ---")

    try:
        client = await create_proxied_client()
        if not client:
            raise Exception("FAILED: Client Creation (No valid proxy?)")
        status_log.append("Client created")

        # 1. Get Temporary Email
        temp_email_used, msg = await get_temporary_email(client)
        status_log.append(f"TempEmail: {msg}")
        if not temp_email_used: raise Exception("Temp Email Failed")

        # 2. Sign Up
        signup_success, msg = await signup_openrouter(client, temp_email_used)
        status_log.append(f"Signup: {msg}")
        if not signup_success: raise Exception("Signup Failed")

        # 3. Check & Verify Email
        verification_success, msg = await check_and_verify_email(client, temp_email_used)
        status_log.append(f"VerifyEmail: {msg}")
        if not verification_success: raise Exception("Verification Failed")

        # 4. Extract API Key
        api_key, msg = await extract_api_key(client)
        status_log.append(f"ExtractKey: {msg}")
        if not api_key: raise Exception("Key Extraction Failed")

        # 5. Store Key if successful
        print(f"[KeyAcquirer] SUCCESS! Acquired key: {api_key[:10]}... for email {temp_email_used}")
        status_log.append(f"SUCCESS: Key {api_key[:10]}... acquired.")
        db_key = await crud.add_api_key(
            db=session,
            key=api_key,
            provider="openrouter",
            email=temp_email_used,
            proxy=proxy_used,
            notes=f"Acquired autonomously. Log: {' -> '.join(status_log)}"
        )
        if db_key:
            await session.commit() # Commit the successful key addition
            print(f"[KeyAcquirer] Key ID {db_key.id} committed to DB.")
        else:
            # This case means add_api_key failed (e.g., duplicate or encryption error)
            status_log.append("FAILED: DB Add operation failed (duplicate/error).")
            await session.rollback()
            api_key = None # Mark as failure for the overall attempt
            print(f"[KeyAcquirer] Failed to add key to DB (duplicate or error).")


        return api_key, status_log # Return key and log

    except Exception as e:
        # Capture the specific failure reason from the exception message if possible
        failure_reason = str(e)
        if not failure_reason.startswith("FAILED:"): # Add prefix if not already there
            failure_reason = f"FAILED: {failure_reason}"
        status_log.append(failure_reason)
        print(f"[KeyAcquirer] Attempt FAILED: {failure_reason}")
        await session.rollback() # Rollback any potential partial changes
        return None, status_log # Indicate failure, return log
    finally:
        if client:
            await client.aclose()
        elapsed = asyncio.get_event_loop().time() - start_time
        result_status = "Success" if api_key else "Failed"
        print(f"--- Finished Key Acquisition Attempt ({result_status}, Duration: {elapsed:.2f}s) ---")
        print(f"--- Status Log: {' -> '.join(status_log)}")
        # Session is closed by the calling worker loop's callback

# Note: run_acquisition_process remains the same as the 'new' version provided previously,
# as its logic for handling consecutive failures and task management is sound.
# It will now correctly receive the detailed status_log from the revised acquire_one_key.

async def run_acquisition_process(target_keys: int, shutdown_event: asyncio.Event):
    """Runs the key acquisition process until target is reached or stops."""
    print(f"[KeyAcquirer] Starting acquisition process. Target: {target_keys} keys.")
    acquired_count = 0
    attempt_count = 0
    session: AsyncSession = None
    consecutive_failures = 0
    max_consecutive_failures = settings.KEY_ACQUIRER_MAX_FAILURES or 5 # Use setting or default

    # Initial count of existing keys
    try:
        session = await get_worker_session()
        initial_keys = await crud.get_active_api_keys(session, provider="openrouter")
        acquired_count = len(initial_keys)
        print(f"[KeyAcquirer] Initial active OpenRouter keys in DB: {acquired_count}")
    except Exception as e:
        print(f"[KeyAcquirer] Error checking initial keys: {e}")
        acquired_count = 0 # Assume zero if check fails
    finally:
        if session: await session.close()

    max_concurrent_attempts = settings.KEY_ACQUIRER_CONCURRENCY or 3 # Reduce default concurrency slightly
    active_tasks = set()

    while acquired_count < target_keys and not shutdown_event.is_set() and consecutive_failures < max_consecutive_failures:
        # Launch new tasks up to the concurrency limit
        while len(active_tasks) < max_concurrent_attempts and acquired_count + len(active_tasks) < target_keys:
            attempt_count += 1
            print(f"[KeyAcquirer] Starting attempt #{attempt_count} ({len(active_tasks)+1}/{max_concurrent_attempts}). Current count: {acquired_count}/{target_keys}")
            session = await get_worker_session() # Get session for this attempt
            # Pass session to acquire_one_key
            task = asyncio.create_task(acquire_one_key(session))
            active_tasks.add(task)
            # Ensure session is closed after task finishes, regardless of outcome
            # Pass the session explicitly to the lambda
            task.add_done_callback(lambda t, s=session: asyncio.create_task(s.close()) if s else None)


        # Wait for at least one task to complete
        if not active_tasks: break # Should not happen if loop condition is correct
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

        # Process completed tasks
        for task in done:
            try:
                result_key, status_log = await task # Await task result to catch exceptions and get log
                if result_key:
                    acquired_count += 1
                    consecutive_failures = 0 # Reset failure count on success
                    print(f"[KeyAcquirer] Progress: {acquired_count}/{target_keys} keys acquired.")
                    # Commit is now handled within acquire_one_key on success
                else:
                    # Task failed
                    consecutive_failures += 1
                    failure_summary = status_log[-1] if status_log else "Unknown failure"
                    print(f"[KeyAcquirer] Acquisition attempt failed. Reason: {failure_summary}")
                    # Check for specific blocking failures
                    if "CAPTCHA Detected" in failure_summary:
                         print("[KeyAcquirer] CAPTCHA detected, stopping further attempts as it's likely systemic.")
                         consecutive_failures = max_consecutive_failures # Force stop
                    elif "Forbidden (403)" in failure_summary:
                         print("[KeyAcquirer] Forbidden (403) error detected, likely IP block. Stopping further attempts.")
                         consecutive_failures = max_consecutive_failures # Force stop

                    if consecutive_failures >= max_consecutive_failures:
                         print(f"[KeyAcquirer] Reached max consecutive failures ({max_consecutive_failures}). Stopping.")

            except Exception as task_e:
                 print(f"[KeyAcquirer] Acquisition task raised an unexpected exception: {task_e}")
                 traceback.print_exc()
                 consecutive_failures += 1
                 if consecutive_failures >= max_consecutive_failures:
                      print(f"[KeyAcquirer] Reached max consecutive failures ({max_consecutive_failures}) after task exception. Stopping.")

        # Check exit conditions
        if acquired_count >= target_keys:
            print(f"[KeyAcquirer] Target of {target_keys} keys reached.")
            break
        if shutdown_event.is_set():
             print("[KeyAcquirer] Shutdown signal received, stopping new attempts.")
             break
        if consecutive_failures >= max_consecutive_failures:
             print("[KeyAcquirer] Stopping due to excessive consecutive failures or critical block.")
             # Log this event for MCOL/operator
             log_session = None
             try:
                 log_session = await get_worker_session()
                 # Use a generic logging function if available, otherwise print clearly
                 # await crud.log_system_event(log_session, agent="KeyAcquirer", event_type="Error", details=f"Stopped due to max consecutive failures ({consecutive_failures}) or critical block.")
                 # await log_session.commit()
                 print(f"[KeyAcquirer][ALERT] Process stopped. Max failures ({max_consecutive_failures}) reached or critical block encountered.")
             except Exception as log_e:
                 print(f"[KeyAcquirer] Failed to log stopping event: {log_e}")
                 if log_session: await log_session.rollback()
             finally:
                 if log_session: await log_session.close()
             break

        await asyncio.sleep(1.0) # Short delay between checks

    # --- Cleanup ---
    if active_tasks:
        print(f"[KeyAcquirer] Waiting for {len(active_tasks)} remaining acquisition attempts to finish...")
        if shutdown_event.is_set() or consecutive_failures >= max_consecutive_failures:
             print("[KeyAcquirer] Cancelling remaining tasks...")
             for task in active_tasks: task.cancel()

        results = await asyncio.gather(*active_tasks, return_exceptions=True)
        # Process remaining results for logging purposes
        for result in results:
             if isinstance(result, tuple) and result[0] and isinstance(result[0], str) and result[0].startswith("sk-or-"):
                 print(f"[KeyAcquirer] A remaining task succeeded. Log: {' -> '.join(result[1])}")
             elif isinstance(result, tuple) and result[0] is None:
                  print(f"[KeyAcquirer] A remaining task failed. Log: {' -> '.join(result[1])}")
             elif isinstance(result, asyncio.CancelledError):
                  print("[KeyAcquirer] A remaining task was cancelled.")
             elif isinstance(result, Exception):
                  print(f"[KeyAcquirer] A remaining task failed with exception: {result}")
                  # traceback.print_exc() # Optional: full trace for remaining task exceptions
             # Else: Task returned None (failure already logged)

    # Final count check
    final_session = None
    try:
        final_session = await get_worker_session()
        final_keys = await crud.get_active_api_keys(final_session, provider="openrouter")
        final_acquired_count = len(final_keys)
        print(f"[KeyAcquirer] Acquisition process finished. Total attempts: {attempt_count}. Final active keys in DB: {final_acquired_count}")
    except Exception as e:
        print(f"[KeyAcquirer] Error checking final keys: {e}")
        print(f"[KeyAcquirer] Acquisition process finished. Total attempts: {attempt_count}. Last known count: {acquired_count}")
    finally:
        if final_session: await final_session.close()