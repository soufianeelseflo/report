# autonomous_agency/app/agents/key_acquirer.py
import asyncio
import httpx
import re
import json
import random
import traceback
from typing import Optional, Dict, Any, List, Tuple
from bs4 import BeautifulSoup # For HTML parsing (might need Playwright for JS)

from sqlalchemy.ext.asyncio import AsyncSession

# Corrected relative imports
from Acumenis.app.core.config import settings
from Acumenis.app.db.base import get_worker_session
from Acumenis.app.db import crud
from Acumenis.app.agents.agent_utils import get_httpx_client # Reuse client creation logic initially

# --- Constants ---
# WARNING: These URLs and selectors WILL change and break the script.
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

DEFAULT_TIMEOUT = 30.0
ACQUISITION_RETRY_ATTEMPTS = 2 # Retries for transient network errors per step

# --- Helper Functions ---

def get_random_proxy() -> Optional[str]:
    """Selects a random proxy from the configured list."""
    if settings.PROXY_LIST:
        return random.choice(settings.PROXY_LIST)
    elif settings.PROXY_URL: # Fallback to single proxy if list not defined
        return settings.PROXY_URL
    return None

async def create_proxied_client() -> Optional[httpx.AsyncClient]:
    """Creates an httpx client configured with a randomly selected proxy."""
    proxy_url = get_random_proxy()
    if not proxy_url:
        print("[KeyAcquirer] No proxies configured or available.")
        return None

    # Basic validation
    if not re.match(r"^(http|https|socks\d?)://", proxy_url):
        print(f"[KeyAcquirer] WARNING: Proxy '{proxy_url}' format invalid. Skipping.")
        return None

    proxies = {"http://": proxy_url, "https://": proxy_url}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=2) # Lower limits for single use
    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=15.0)
    headers = { # Use less specific user agent?
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
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
        # Test proxy connection (optional, adds overhead)
        # await client.get("https://httpbin.org/ip", timeout=10.0)
        # print(f"[KeyAcquirer] Client created with proxy: {proxy_url}")
        return client
    except Exception as e:
        print(f"[KeyAcquirer] Failed to create client or test proxy {proxy_url}: {e}")
        return None

# --- Web Interaction Steps (Highly Unreliable - Expect Breakage) ---

async def get_temporary_email(client: httpx.AsyncClient) -> Optional[str]:
    """
    Attempts to get a temporary email from the provider.
    NOTE: Likely requires Playwright if JS is involved in email generation.
    """
    print(f"[KeyAcquirer] Attempting to get temporary email from {TEMP_EMAIL_PROVIDER_URL}...")
    try:
        response = await client.get(TEMP_EMAIL_PROVIDER_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # *** This selector is a GUESS - MUST be updated by inspecting inboxes.com ***
        email_element = soup.select_one(TEMP_EMAIL_SELECTOR)

        if email_element:
            # Extract email based on how it's stored (e.g., value attribute, text content)
            temp_email = email_element.get('value') or email_element.text
            temp_email = temp_email.strip()
            if temp_email and '@' in temp_email:
                print(f"[KeyAcquirer] Obtained temporary email: {temp_email}")
                return temp_email
            else:
                 print(f"[KeyAcquirer] Failed to extract valid email from element: {email_element}")
        else:
            print(f"[KeyAcquirer] Could not find temporary email element using selector '{TEMP_EMAIL_SELECTOR}'. Website structure may have changed.")
            # print(f"Page content: {response.text[:500]}") # Debugging

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error getting temp email: {e}")
    except Exception as e:
        print(f"[KeyAcquirer] Error parsing temp email page: {e}")
        # traceback.print_exc() # Uncomment for detailed debug
    return None

async def signup_openrouter(client: httpx.AsyncClient, temp_email: str) -> bool:
    """
    Attempts to sign up on OpenRouter using the temporary email.
    NOTE: Will likely FAIL due to CAPTCHA. Requires Playwright for complex forms.
    """
    print(f"[KeyAcquirer] Attempting OpenRouter signup for {temp_email}...")
    generated_password = f"AcumenisPass_{random.randint(100000, 999999)}!" # Simple generated password

    try:
        # 1. GET signup page to potentially get CSRF tokens, cookies etc. (if needed)
        # response_get = await client.get(OPENROUTER_SIGNUP_URL)
        # response_get.raise_for_status()
        # soup_get = BeautifulSoup(response_get.text, 'html.parser')
        # Extract CSRF token if present: csrf_token = soup_get.find('input', {'name': 'csrf_token'})['value']

        # 2. POST signup data
        payload = {
            SIGNUP_EMAIL_FIELD_NAME: temp_email,
            SIGNUP_PASSWORD_FIELD_NAME: generated_password,
            # 'csrf_token': csrf_token, # Include if needed
            # *** CAPTCHA Handling Placeholder ***
            # 'h-captcha-response': 'CAPTCHA_SOLUTION_FROM_SOLVER_SERVICE', # Example
            # 'g-recaptcha-response': 'CAPTCHA_SOLUTION_FROM_SOLVER_SERVICE', # Example
        }
        print("[KeyAcquirer] Preparing signup POST...")
        # print(f"Payload (excluding captcha): {payload}") # Debug

        # *** CAPTCHA BLOCKER ***
        print("[KeyAcquirer] WARNING: CAPTCHA challenge expected here. Without a CAPTCHA solving service integration, signup WILL fail.")
        # If you integrate a solver, get the solution token and add it to the payload here.
        # await asyncio.sleep(2) # Simulate waiting for CAPTCHA

        # Make the POST request
        response_post = await client.post(OPENROUTER_SIGNUP_URL, data=payload, timeout=DEFAULT_TIMEOUT)
        # print(f"Signup POST status: {response_post.status_code}") # Debug
        # print(f"Signup POST response URL: {response_post.url}") # Debug
        # print(f"Signup POST response text: {response_post.text[:500]}") # Debug

        # Check for success indicators (highly dependent on OpenRouter's response)
        # - Status code 200 might not mean success (could be error page)
        # - Redirect to a specific page?
        # - Specific text on the response page?
        if response_post.status_code == 200 and "Verification email sent" in response_post.text: # GUESS
            print(f"[KeyAcquirer] Signup POST successful (pending email verification) for {temp_email}.")
            return True
        elif "CAPTCHA" in response_post.text or response_post.status_code == 403:
             print(f"[KeyAcquirer] Signup failed for {temp_email} - Likely CAPTCHA challenge or block.")
        else:
             print(f"[KeyAcquirer] Signup POST failed for {temp_email}. Status: {response_post.status_code}. Check response.")

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error during signup: {e}")
    except Exception as e:
        print(f"[KeyAcquirer] Error processing signup: {e}")
        # traceback.print_exc()
    return False


async def check_and_verify_email(client: httpx.AsyncClient, temp_email: str) -> bool:
    """
    Checks the temporary inbox and clicks the verification link.
    NOTE: Extremely likely to require Playwright due to JS/dynamic inbox updates.
    """
    print(f"[KeyAcquirer] Checking inbox for {temp_email} for verification email...")
    verification_link = None
    max_check_attempts = 10
    check_interval = 15 # seconds

    for attempt in range(max_check_attempts):
        try:
            print(f"[KeyAcquirer] Inbox check attempt {attempt + 1}/{max_check_attempts}...")
            # *** This part is highly unreliable with httpx ***
            # Need to simulate refreshing/interacting with the inbox page.
            # Playwright would navigate, wait for email to appear, then extract link.
            inbox_response = await client.get(TEMP_EMAIL_PROVIDER_URL) # Re-fetch or use persistent session
            inbox_response.raise_for_status()
            inbox_soup = BeautifulSoup(inbox_response.text, 'html.parser')

            # *** Find email from OpenRouter and extract link (GUESSING selectors/structure) ***
            # Example: Find link containing 'openrouter.ai/verify'
            link_element = inbox_soup.find('a', href=re.compile(r'openrouter\.ai/verify'))

            if link_element:
                verification_link = link_element['href']
                if not verification_link.startswith('http'): # Handle relative links
                    # Need base URL of the email provider or OpenRouter? Assume absolute.
                     print(f"[KeyAcquirer] Found potential verification link: {verification_link}")
                     break # Exit loop once link is found
            else:
                 print("[KeyAcquirer] Verification email not found yet.")

            await asyncio.sleep(check_interval) # Wait before next check

        except httpx.RequestError as e:
            print(f"[KeyAcquirer] Network error checking inbox: {e}")
            await asyncio.sleep(check_interval) # Wait before retrying check
        except Exception as e:
            print(f"[KeyAcquirer] Error parsing inbox: {e}")
            # traceback.print_exc()
            await asyncio.sleep(check_interval) # Wait before retrying check

    if verification_link:
        print(f"[KeyAcquirer] Attempting to visit verification link: {verification_link}")
        try:
            verify_response = await client.get(verification_link, timeout=DEFAULT_TIMEOUT)
            verify_response.raise_for_status()
            # Check for success indicator on verification page
            if "Email verified" in verify_response.text or verify_response.status_code == 200: # GUESS
                print(f"[KeyAcquirer] Email verification successful for {temp_email}.")
                return True
            else:
                print(f"[KeyAcquirer] Verification link visited, but success indicator not found. Status: {verify_response.status_code}")
        except httpx.RequestError as e:
            print(f"[KeyAcquirer] Network error visiting verification link: {e}")
        except Exception as e:
            print(f"[KeyAcquirer] Error processing verification response: {e}")
    else:
        print(f"[KeyAcquirer] Verification link not found in inbox for {temp_email} after {max_check_attempts} attempts.")

    return False

async def extract_api_key(client: httpx.AsyncClient) -> Optional[str]:
    """
    Attempts to log in (if needed) and extract an API key from the keys page.
    NOTE: Requires successful signup/verification first. Likely needs Playwright.
    """
    print("[KeyAcquirer] Attempting to navigate to API keys page and extract key...")
    try:
        # This likely requires the client to be logged in from the verification step.
        # If the session was lost, a login step would be needed here.
        keys_page_response = await client.get(OPENROUTER_KEYS_URL, timeout=DEFAULT_TIMEOUT)
        keys_page_response.raise_for_status()
        keys_soup = BeautifulSoup(keys_page_response.text, 'html.parser')

        # 1. Find and "click" the generate key button (if necessary)
        #    This might be a POST request triggered by JS (needs Playwright)
        #    Example: Find button -> Get its target URL/form data -> Make POST request
        # generate_button = keys_soup.select_one(API_KEY_GENERATE_SELECTOR) # GUESS
        # if generate_button:
        #     print("[KeyAcquirer] Found generate key button (simulation - actual click likely needs JS/Playwright)")
        #     # Simulate POST if possible, or assume key exists/is generated on page load

        # 2. Find the API key display element
        # *** This selector is a GUESS - MUST be updated by inspecting OpenRouter ***
        key_element = keys_soup.select_one(API_KEY_DISPLAY_SELECTOR)

        if key_element:
            # Extract key based on how it's stored (e.g., value attribute)
            api_key = key_element.get('value') or key_element.text # GUESS
            api_key = api_key.strip()
            if api_key and api_key.startswith("sk-or-"): # Basic validation
                print(f"[KeyAcquirer] Successfully extracted API key: {api_key[:10]}...")
                return api_key
            else:
                print(f"[KeyAcquirer] Found key element but content invalid: {key_element}")
        else:
            print(f"[KeyAcquirer] Could not find API key element using selector '{API_KEY_DISPLAY_SELECTOR}'. Page structure changed or login failed?")
            # print(f"Keys page content: {keys_page_response.text[:500]}") # Debugging

    except httpx.RequestError as e:
        print(f"[KeyAcquirer] Network error accessing keys page: {e}")
    except Exception as e:
        print(f"[KeyAcquirer] Error parsing keys page: {e}")
        # traceback.print_exc()
    return None

# --- Main Acquisition Orchestration ---

async def acquire_one_key(session: AsyncSession) -> Optional[str]:
    """
    Attempts the full sequence to acquire a single OpenRouter API key.
    Uses a dedicated proxied client for the attempt.
    """
    client = None
    proxy_used = get_random_proxy() # Get proxy info for logging
    temp_email_used = None
    api_key = None
    success = False

    print(f"\n--- Starting new key acquisition attempt (Proxy: {proxy_used or 'None'}) ---")

    try:
        client = await create_proxied_client()
        if not client:
            raise Exception("Failed to create proxied client.")

        # 1. Get Temporary Email
        temp_email_used = await get_temporary_email(client)
        if not temp_email_used:
            raise Exception("Failed to obtain temporary email.")

        # 2. Sign Up (Handles CAPTCHA placeholder)
        signup_success = await signup_openrouter(client, temp_email_used)
        if not signup_success:
            # Specific handling if CAPTCHA was the known failure point
            if "CAPTCHA" in traceback.format_exc(): # Simple check
                 print("[KeyAcquirer] Attempt failed at CAPTCHA step.")
            raise Exception("Signup step failed (likely CAPTCHA or form issue).")

        # 3. Check & Verify Email
        verification_success = await check_and_verify_email(client, temp_email_used)
        if not verification_success:
            raise Exception("Email verification step failed.")

        # 4. Extract API Key
        api_key = await extract_api_key(client)
        if not api_key:
            raise Exception("API key extraction step failed.")

        # 5. Store Key if successful
        print(f"[KeyAcquirer] SUCCESS! Acquired key: {api_key[:10]}... for email {temp_email_used}")
        await crud.add_api_key(
            db=session,
            key=api_key,
            provider="openrouter",
            email=temp_email_used,
            proxy=proxy_used,
            notes="Acquired autonomously"
        )
        success = True
        return api_key # Return the acquired key

    except Exception as e:
        print(f"[KeyAcquirer] Attempt FAILED: {e}")
        # Optionally store failure info in DB?
        # crud.log_acquisition_attempt(session, status='failed', email=temp_email_used, proxy=proxy_used, error=str(e))
        return None # Indicate failure
    finally:
        if client:
            await client.aclose()
        print(f"--- Finished key acquisition attempt (Success: {success}) ---")


async def run_acquisition_process(target_keys: int = 10):
    """Runs the key acquisition process until target is reached or stops."""
    print(f"[KeyAcquirer] Starting acquisition process. Target: {target_keys} keys.")
    acquired_count = 0
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

    # Limit concurrency to avoid overwhelming resources or getting banned quickly
    max_concurrent_attempts = settings.KEY_ACQUIRER_CONCURRENCY or 3
    active_tasks = set()

    while acquired_count < target_keys:
        if len(active_tasks) >= max_concurrent_attempts:
            # Wait for one task to complete before starting another
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result_key = task.result()
                if result_key:
                    acquired_count += 1
                    print(f"[KeyAcquirer] Progress: {acquired_count}/{target_keys} keys acquired.")
        else:
            # Start a new acquisition attempt
            print(f"[KeyAcquirer] Starting new attempt ({len(active_tasks)+1}/{max_concurrent_attempts})...")
            session = await get_worker_session() # Get session for this attempt
            task = asyncio.create_task(acquire_one_key(session))
            active_tasks.add(task)
            # Ensure session is closed after task finishes (or handle within acquire_one_key)
            task.add_done_callback(lambda t: asyncio.create_task(session.close()) if session else None)

        # Check if target reached after processing completed tasks
        if acquired_count >= target_keys:
            print(f"[KeyAcquirer] Target of {target_keys} keys reached.")
            break

        # Optional: Add delay between starting batches of attempts
        await asyncio.sleep(5) # Wait 5 seconds before checking/starting next attempt

    # Wait for any remaining tasks to finish
    if active_tasks:
        print("[KeyAcquirer] Waiting for remaining acquisition attempts to finish...")
        done, _ = await asyncio.wait(active_tasks)
        for task in done:
            result_key = task.result()
            if result_key and acquired_count < target_keys: # Avoid double counting if target was met exactly
                acquired_count += 1
                print(f"[KeyAcquirer] Progress: {acquired_count}/{target_keys} keys acquired.")

    print(f"[KeyAcquirer] Acquisition process finished. Total active keys: {acquired_count}")

# Example of how to run (e.g., from a worker)
# if __name__ == "__main__":
#     asyncio.run(run_acquisition_process(target_keys=5))