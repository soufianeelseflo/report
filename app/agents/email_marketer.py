import asyncio
import random
import json
import datetime
import smtplib # For exception types
import re
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Optional, Dict, Any, List # Add List

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import httpx

# Corrected relative imports for package structure
from Acumenis.app.core.config import settings
from Acumenis.app.db import crud, models
from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api # Use shared call_llm_api
from Acumenis.app.core.security import decrypt_data

# --- Configuration ---
EMAIL_GENERATION_TIMEOUT = 90 # Seconds for LLM to generate email
SMTP_TIMEOUT = 30 # Seconds for SMTP operations
SEND_DELAY_MIN = settings.EMAIL_SEND_DELAY_MIN or 5.0 # Minimum delay between sends in seconds
SEND_DELAY_MAX = settings.EMAIL_SEND_DELAY_MAX or 15.0 # Maximum delay between sends in seconds
WARMUP_THRESHOLD = settings.EMAIL_WARMUP_THRESHOLD or 10 # Accounts with sends below this count get extra delay
WARMUP_DELAY_MULTIPLIER = settings.EMAIL_WARMUP_DELAY_MULTIPLIER or 2.0 # Extra delay factor for warming up accounts

# --- Email Content Generation ---

def extract_signal_context(source_string: Optional[str]) -> str:
    """Extracts a concise description of the signal source for the LLM."""
    if not source_string: return "an unspecified business signal"
    if source_string.startswith("news_api_"): return "a recent news event relevant to their industry or company"
    if source_string.startswith("scraping_web_"):
        domain = source_string.replace("scraping_web_", "")
        return f"insights gathered from their website ({domain})"
    if source_string.startswith("signal_linkedin"): return "activity or changes observed related to their LinkedIn presence"
    return "an identified business signal"

async def humanize_email_content(client: httpx.AsyncClient, subject: str, body: str) -> Optional[Dict[str, str]]:
    """Uses LLM to rewrite email content to sound more human and bypass AI detectors."""
    humanize_prompt = f"""
    Rewrite the following cold email subject and body to sound completely human-written, conversational, and less like AI output. Focus on natural language, avoiding overly formal structures or buzzwords. Ensure it bypasses AI detection. Keep the core message and call-to-action intact.

    Original Subject: "{subject}"
    Original Body:
    "{body}"

    Output ONLY a JSON object with the rewritten "subject" and "body". Example: {{"subject": "Re: {company_name} data point", "body": "Hey [Name],\n\nSaw [brief observation] - reminded me of something...\n\nCompanies like yours often hit [challenge]. We help with quick, AI-powered reports for [benefit]...\n\nWorth a quick chat next week if you're curious?\n\nCheers,\n[Your Name]"}}
    """
    print("[EmailMarketer] Humanizing email content with LLM...")
    llm_response = await call_llm_api(client, humanize_prompt, model="google/gemini-1.5-pro-latest") # Use powerful model

    if llm_response and isinstance(llm_response, dict) and "subject" in llm_response and "body" in llm_response:
        h_subject = llm_response["subject"].strip()
        h_body = llm_response["body"].strip()
        if h_subject and h_body:
            print(f"[EmailMarketer] Humanized Subject: {h_subject}")
            return {"subject": h_subject, "body": h_body}
    print("[EmailMarketer] Failed to humanize email content.")
    return None


async def generate_personalized_email(client: httpx.AsyncClient, prospect: models.Prospect) -> Optional[Dict[str, str]]:
    """Uses LLM to generate a personalized subject and body, then humanizes it."""
    company_name = prospect.company_name
    pain_point = prospect.potential_pain_point or "potential needs for enhanced data insights or reporting"
    signal_context = extract_signal_context(prospect.source)

    # Step 1: Generate Initial Draft
    initial_prompt = f"""
    Objective: Generate a hyper-personalized, concise cold email subject and body targeting '{company_name}'.
    Your Role: Act as a skilled B2B sales professional specializing in data analysis and research reporting.
    Key Information:
    - Company: {company_name}
    - Context/Signal: We detected {signal_context}.
    - Inferred Need: Based on this, they might be facing challenges related to: "{pain_point}".
    - Our Solution: We provide expert-level, AI-generated research reports (market analysis, competitive intelligence, etc.) rapidly.

    Instructions:
    1.  **Subject Line:** Create a short, intriguing, personalized subject line (under 8 words). Reference '{company_name}' or a relevant aspect of the inferred need. Avoid generic phrases.
    2.  **Email Body (Max 100 words):** Opener referencing need/signal subtly -> Connect challenge to reporting value -> Introduce our rapid AI reports & benefit -> Soft CTA (brief call or relevant example).
    3.  **Tone:** Professional, concise, helpful, slightly informal but respectful.
    4.  **Output Format:** Respond ONLY with a JSON object containing "subject" and "body".

    Strict Command: Generate the JSON output directly. No preamble.
    """

    print(f"[EmailMarketer] Generating initial email draft for {prospect.company_name} (ID: {prospect.prospect_id})")
    initial_llm_response = await call_llm_api(client, initial_prompt)

    initial_subject = None
    initial_body = None
    if initial_llm_response and isinstance(initial_llm_response, dict) and "subject" in initial_llm_response and "body" in initial_llm_response:
        initial_subject = initial_llm_response["subject"].strip()
        initial_body = initial_llm_response["body"].strip()
    elif initial_llm_response and isinstance(initial_llm_response.get("raw_inference"), str):
         # Basic fallback parsing
         raw = initial_llm_response["raw_inference"]
         try:
             parsed = json.loads(raw)
             initial_subject = parsed.get("subject","").strip()
             initial_body = parsed.get("body","").strip()
         except:
             # Try regex if JSON fails
             subject_match = re.search(r"Subject:\s*(.*)", raw, re.IGNORECASE)
             body_match = re.search(r"Body:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
             if subject_match: initial_subject = subject_match.group(1).strip()
             if body_match: initial_body = body_match.group(1).strip()

    if not initial_subject or not initial_body:
        print(f"[EmailMarketer] Failed to generate initial email draft for {prospect.company_name}")
        return None

    # Step 2: Humanize the Draft
    humanized_content = await humanize_email_content(client, initial_subject, initial_body)

    if humanized_content:
        return humanized_content
    else:
        # Fallback: Use the initial draft if humanization fails
        print("[EmailMarketer] Humanization failed, using initial draft.")
        return {"subject": initial_subject, "body": initial_body}


# --- SMTP Sending Logic ---
async def send_email_via_smtp(email_content: Dict[str, str], prospect_email: str, account: models.EmailAccount):
    """
    Connects to SMTP, authenticates, and sends the generated email.
    Raises specific exceptions on failure for the caller to handle.
    Does NOT handle retries, account deactivation, or prospect status updates.
    """
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        # Raise a specific error if password cannot be decrypted
        raise ValueError(f"Password decryption failed for account {account.email_address}")

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = account.email_address
    msg['To'] = prospect_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=account.email_address.split('@')[-1])
    msg.set_content(email_content['body'])

    print(f"[EmailMarketer] Sending email to {prospect_email} via {account.email_address}...")

    # Exceptions like SMTPAuthenticationError, SMTPRecipientsRefused, SMTPSenderRefused,
    # SMTPConnectError, SMTPDataError, SMTPException will propagate up to the caller.
    async with aiosmtplib.SMTP(hostname=account.smtp_host, port=account.smtp_port, use_tls=True, timeout=SMTP_TIMEOUT) as smtp:
        await smtp.login(account.smtp_user, decrypted_password)
        await smtp.send_message(msg)
        print(f"[EmailMarketer] Email sent successfully to {prospect_email} via {account.email_address}")
    # If no exception is raised, sending is considered successful by this function.


# --- Main Agent Logic ---
async def process_email_batch(db: AsyncSession, shutdown_event: asyncio.Event):
    """Fetches prospects, generates emails, and sends using a rotating pool of accounts."""
    if shutdown_event.is_set(): return

    prospects_processed_count = 0
    prospects_emailed_count = 0
    client: Optional[httpx.AsyncClient] = None
    active_accounts: List[models.EmailAccount] = []
    account_index = -1 # Start at -1 so first increment makes it 0

    try:
        client = await get_httpx_client()

        # Fetch a batch of active accounts for this run
        # Ensure EMAIL_ACCOUNTS_PER_BATCH is set in config, default to 5
        batch_account_limit = settings.EMAIL_ACCOUNTS_PER_BATCH or 5
        active_accounts = await crud.get_batch_of_active_accounts(db, limit=batch_account_limit)
        if not active_accounts:
            print("[EmailMarketer] No active email accounts found in DB. Cannot send emails.")
            return
        print(f"[EmailMarketer] Fetched {len(active_accounts)} active accounts for this batch.")

        # Fetch prospects to process
        batch_prospect_limit = settings.EMAIL_BATCH_SIZE or 50
        prospects = await crud.get_new_prospects_for_emailing(db, limit=batch_prospect_limit)
        if not prospects:
            print("[EmailMarketer] No new prospects found for emailing.")
            return

        print(f"[EmailMarketer] Fetched {len(prospects)} prospects for emailing.")

        # --- Main Prospect Loop ---
        for prospect in prospects:
            if shutdown_event.is_set():
                print("[EmailMarketer] Shutdown signal received during batch processing.")
                break
            prospects_processed_count += 1
            email_account: Optional[models.EmailAccount] = None
            prospect_status_update = None # Tuple: (status, last_contacted_at)
            account_to_deactivate = None # Tuple: (account_id, reason)
            increment_send_count_for_account_id = None
            current_account_list_index = -1 # Track index in the current active_accounts list

            try:
                # --- Select Account with Rotation ---
                if not active_accounts:
                    print("[EmailMarketer] Ran out of active accounts for this batch.")
                    break # Stop processing this batch if all accounts failed
                # Cycle through the fetched accounts for this batch
                account_index = (account_index + 1) % len(active_accounts)
                current_account_list_index = account_index # Store index in case we need to remove it
                email_account = active_accounts[current_account_list_index]

                # --- Simple Warm-up Delay ---
                warmup_delay = 0
                # Check emails_sent_today which was reset by get_batch_of_active_accounts if needed
                if email_account.emails_sent_today < WARMUP_THRESHOLD:
                    warmup_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX) * (WARMUP_DELAY_MULTIPLIER - 1)
                    print(f"[EmailMarketer] Applying warm-up delay ({warmup_delay:.2f}s) for account {email_account.email_address} (Sent today: {email_account.emails_sent_today})")

                # --- Standard Delay ---
                standard_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX)
                await asyncio.sleep(standard_delay + warmup_delay)

                # --- Generate Email ---
                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    print(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} due to email generation failure.")
                    prospect_status_update = ("FAILED_GENERATION", None)
                    # Use continue to skip to the finally block for this prospect
                    continue

                # --- Send Email ---
                # send_email_via_smtp will raise specific exceptions on failure
                await send_email_via_smtp(email_content, prospect.contact_email, email_account)

                # --- Handle Success ---
                print(f"[EmailMarketer] Successfully sent email to {prospect.contact_email} via {email_account.email_address}")
                prospect_status_update = ("CONTACTED", datetime.datetime.now(datetime.timezone.utc))
                increment_send_count_for_account_id = email_account.account_id
                prospects_emailed_count += 1
                # Increment count in memory for warm-up check in the same batch run
                email_account.emails_sent_today += 1

            except (aiosmtplib.SMTPAuthenticationError, ValueError) as auth_err: # ValueError for decryption failure
                print(f"[EmailMarketer] Auth/Setup error for account {email_account.email_address if email_account else 'N/A'}: {auth_err}. Deactivating.")
                if email_account:
                    account_to_deactivate = (email_account.account_id, f"Auth/Setup Error: {str(auth_err)[:100]}")
                    # Remove from current batch list to prevent reuse
                    if current_account_list_index != -1:
                         try:
                             active_accounts.pop(current_account_list_index)
                             # Adjust index for next iteration if needed
                             account_index = current_account_list_index -1 # Point to previous index before the removed one
                         except IndexError:
                              print(f"[EmailMarketer] Error removing account from active list index {current_account_list_index}")
                              account_index = -1 # Reset index
                    # Prospect status remains NEW, will be retried later

            except aiosmtplib.SMTPRecipientsRefused as bounce_err:
                print(f"[EmailMarketer] Hard bounce for {prospect.contact_email}: {bounce_err}. Marking as BOUNCED.")
                prospect_status_update = ("BOUNCED", None)
                if email_account: # Still count as sent attempt
                    increment_send_count_for_account_id = email_account.account_id
                    email_account.emails_sent_today += 1 # Increment memory count

            except (aiosmtplib.SMTPSenderRefused, aiosmtplib.SMTPDataError, smtplib.SMTPException) as smtp_err:
                # Treat sender refused, data errors, and other SMTP issues as potential account problems or blocks
                # Check for specific Gmail error codes/messages if known
                error_message = str(smtp_err)
                deactivate_reason = f"SMTP Send Error: {error_message[:100]}"
                print(f"[EmailMarketer] SMTP Error for account {email_account.email_address if email_account else 'N/A'} sending to {prospect.contact_email}: {smtp_err}. Deactivating account.")
                # Example: Check for common Gmail suspension codes/messages
                # if "Suspended" in error_message or "5.7.0" in error_message:
                #     deactivate_reason = f"Account Likely Suspended: {error_message[:100]}"

                if email_account:
                    account_to_deactivate = (email_account.account_id, deactivate_reason)
                    if current_account_list_index != -1:
                         try:
                             active_accounts.pop(current_account_list_index)
                             account_index = current_account_list_index - 1
                         except IndexError:
                              print(f"[EmailMarketer] Error removing account from active list index {current_account_list_index}")
                              account_index = -1 # Reset index
                # Prospect status remains NEW

            except Exception as e:
                print(f"[EmailMarketer] Unexpected error processing prospect {prospect.prospect_id}: {e}")
                # Mark prospect as failed send to avoid retrying immediately
                prospect_status_update = ("FAILED_SEND", None)
                if email_account: # Count as attempt if account was selected
                     increment_send_count_for_account_id = email_account.account_id
                     email_account.emails_sent_today += 1 # Increment memory count

            finally:
                # --- Database Updates for this Prospect/Account ---
                # This block executes for each prospect, committing changes individually.
                try:
                    if account_to_deactivate:
                        await crud.set_email_account_inactive(db, account_to_deactivate[0], account_to_deactivate[1])
                    if increment_send_count_for_account_id:
                        await crud.increment_email_sent_count(db, increment_send_count_for_account_id)
                    if prospect_status_update:
                        await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status_update[0], last_contacted_at=prospect_status_update[1])
                    await db.commit() # Commit changes for this prospect/account interaction
                except Exception as db_e:
                    print(f"[EmailMarketer] CRITICAL: Failed to commit DB updates for prospect {prospect.prospect_id}: {db_e}")
                    await db.rollback() # Rollback if updates fail

    except Exception as e:
        print(f"[EmailMarketer] Error in email processing batch: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client: await client.aclose()

    print(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}.")