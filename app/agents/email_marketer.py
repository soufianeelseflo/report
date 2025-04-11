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
EMAIL_GENERATION_TIMEOUT = 120 # Increased timeout for complex prompts
SMTP_TIMEOUT = 30 # Seconds for SMTP operations
SEND_DELAY_MIN = settings.EMAIL_SEND_DELAY_MIN or 1.0 # Aggressive: Minimum delay 1 second
SEND_DELAY_MAX = settings.EMAIL_SEND_DELAY_MAX or 5.0 # Aggressive: Maximum delay 5 seconds
WARMUP_THRESHOLD = settings.EMAIL_WARMUP_THRESHOLD or 5 # Very low threshold for "warmup"
WARMUP_DELAY_MULTIPLIER = settings.EMAIL_WARMUP_DELAY_MULTIPLIER or 1.5 # Minimal extra delay

# --- Email Content Generation ---

def extract_signal_context(source_string: Optional[str]) -> str:
    """Extracts a concise description of the signal source for the LLM."""
    if not source_string: return "an unspecified critical business signal we detected"
    if source_string.startswith("odr_"):
        query_part = source_string.replace("odr_", "").replace("_", " ")[:50]
        return f"intelligence derived from our analysis related to '{query_part}'"
    # Add more specific signal parsers if ProspectResearcher source format changes
    return "a high-priority business signal identified by our AI"

async def humanize_email_content(client: httpx.AsyncClient, subject: str, body: str) -> Optional[Dict[str, str]]:
    """Uses LLM to rewrite email content to sound human and bypass AI detectors, while maintaining persuasive edge."""
    humanize_prompt = f"""
    Rewrite the following hyper-personalized cold email subject and body.
    Goal: Make it sound like an urgent, direct message from a busy but insightful human expert. Bypass AI detection filters aggressively.
    Style: Extremely concise, high-impact, slightly informal, confident, bordering on provocative. Use short sentences, maybe subtle typos if natural. Avoid corporate jargon. Focus on immediate value/pain relief.
    Constraint: Retain the core value proposition (rapid AI reports for strategic advantage) and the direct call-to-action.

    Original Subject: "{subject}"
    Original Body:
    "{body}"

    Output ONLY a JSON object with the rewritten "subject" and "body". Example: {{"subject": "Re: {company_name} - quick thought", "body": "Hey [Name],\n\nSaw [signal context]. Data blind spots kill growth.\n\nWe generate deep AI intel reports (competitors, market shifts) in hours, not weeks. Solves [pain point].\n\nWorth $499 to act fast? [Link to Order Page]\n\n[Your Name]\nAcumenis AI"}}
    """
    print("[EmailMarketer] Humanizing/Sharpening email content with LLM...")
    # Use a powerful model capable of nuanced tone and evasion
    llm_response = await call_llm_api(client, humanize_prompt, model=settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-pro-latest")

    if llm_response and isinstance(llm_response, dict) and "subject" in llm_response and "body" in llm_response:
        h_subject = llm_response["subject"].strip()
        h_body = llm_response["body"].strip()
        if h_subject and h_body:
            print(f"[EmailMarketer] Sharpened Subject: {h_subject}")
            return {"subject": h_subject, "body": h_body}
    print("[EmailMarketer] Failed to humanize/sharpen email content.")
    return None


async def generate_personalized_email(client: httpx.AsyncClient, prospect: models.Prospect) -> Optional[Dict[str, str]]:
    """Uses LLM to generate a hyper-personalized, aggressive subject and body, then humanizes/sharpens it."""
    company_name = prospect.company_name
    pain_point = prospect.potential_pain_point or "critical gaps in strategic intelligence"
    signal_context = extract_signal_context(prospect.source)
    contact_name = prospect.contact_name or company_name # Use company name if contact name unknown

    # Step 1: Generate Initial Aggressive Draft
    initial_prompt = f"""
    Objective: Generate an extremely concise, high-impact, psychologically persuasive cold email subject and body for '{contact_name}' at '{company_name}'. Goal is IMMEDIATE click-through and purchase ($499/$999).
    Your Role: Act as a 'rogue' AI strategist providing exclusive, time-sensitive intelligence.
    Key Information:
    - Prospect: {contact_name} @ {company_name}
    - Context/Signal: Detected {signal_context}. This implies vulnerability or missed opportunity related to: "{pain_point}".
    - Our Solution: Acumenis AI - We deliver deep competitive/market intelligence reports synthesized by advanced AI within HOURS. This provides [Specific Benefit, e.g., 'first-mover advantage', 'risk mitigation', 'competitor blindspot exploitation'].
    - Offer: Standard ($499) / Premium ($999) report activation.

    Instructions:
    1.  **Subject Line:** Max 6 words. Create extreme urgency/curiosity. Use prospect name/company. Examples: "{contact_name} - Urgent Intel?", "{company_name} Blindspot?", "Re: Your Competitor Risk".
    2.  **Email Body (Max 70 words):**
        -   Opener: Direct reference to signal/pain point (make it sound exclusive/discovered).
        -   Problem Agitation: Briefly state consequence of inaction (e.g., "Losing ground?", "Decision paralysis?").
        -   Solution Intro: Introduce Acumenis AI as the rapid antidote. Mention HOURS turnaround.
        -   Value Prop: State the core benefit bluntly.
        -   Call to Action (CTA): Direct link to the order page (`{settings.AGENCY_BASE_URL}/order`). Frame as immediate action. Example: "Activate $499 stream now: [Link]", "Get Nexus intel ($999): [Link]".
    3.  **Tone:** Confident, direct, slightly mysterious, implies insider knowledge, extremely urgent.
    4.  **Output Format:** Respond ONLY with a JSON object containing "subject" and "body".

    Strict Command: Generate the JSON output directly. No preamble. Be aggressive.
    """

    print(f"[EmailMarketer] Generating aggressive draft for {prospect.company_name} (ID: {prospect.prospect_id})")
    initial_llm_response = await call_llm_api(client, initial_prompt, model=settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-pro-latest") # Use powerful model

    initial_subject = None
    initial_body = None
    # Try parsing logic (handle raw inference or direct JSON)
    if initial_llm_response and isinstance(initial_llm_response, dict):
        if "subject" in initial_llm_response and "body" in initial_llm_response:
            initial_subject = initial_llm_response["subject"].strip()
            initial_body = initial_llm_response["body"].strip()
        elif isinstance(initial_llm_response.get("raw_inference"), str):
             raw = initial_llm_response["raw_inference"]
             try:
                 # Clean potential markdown ```json ... ``` blocks first
                 json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
                 json_str = json_match.group(1).strip() if json_match else raw.strip()
                 parsed = json.loads(json_str)
                 initial_subject = parsed.get("subject","").strip()
                 initial_body = parsed.get("body","").strip()
             except (json.JSONDecodeError, AttributeError):
                 # Fallback: Try simple regex if JSON parsing fails
                 subject_match = re.search(r'"?subject"?\s*:\s*"(.*?)"', raw, re.IGNORECASE)
                 body_match = re.search(r'"?body"?\s*:\s*"([\s\S]*?)"', raw, re.IGNORECASE) # Use [\s\S]*? for multiline body
                 if subject_match: initial_subject = subject_match.group(1).strip()
                 if body_match: initial_body = body_match.group(1).strip().replace('\\n', '\n') # Handle escaped newlines

    if not initial_subject or not initial_body:
        print(f"[EmailMarketer] Failed to generate initial aggressive draft for {prospect.company_name}")
        return None

    # Step 2: Humanize/Sharpen the Draft
    humanized_content = await humanize_email_content(client, initial_subject, initial_body)

    if humanized_content:
        # Inject the order link into the body (ensure placeholder exists or append)
        order_link = f"{settings.AGENCY_BASE_URL}/order" # Ensure AGENCY_BASE_URL is correct
        # Simple replacement, might need refinement based on LLM output structure
        humanized_content["body"] = humanized_content["body"].replace("[Link to Order Page]", order_link).replace("[Link]", order_link)
        # Append if placeholder missing (crude fallback)
        if order_link not in humanized_content["body"]:
             humanized_content["body"] += f"\n\nActivate here: {order_link}"

        # Replace [Name] placeholder
        humanized_content["body"] = humanized_content["body"].replace("[Name]", contact_name.split()[0]) # Use first name
        humanized_content["subject"] = humanized_content["subject"].replace("{company_name}", company_name).replace("{Name}", contact_name.split()[0])

        return humanized_content
    else:
        # Fallback: Use the initial draft if humanization fails (less ideal)
        print("[EmailMarketer] Humanization/Sharpening failed, using initial draft.")
        # Inject link and name into initial draft as fallback
        order_link = f"{settings.AGENCY_BASE_URL}/order"
        initial_body = initial_body.replace("[Link to Order Page]", order_link).replace("[Link]", order_link)
        if order_link not in initial_body: initial_body += f"\n\nActivate here: {order_link}"
        initial_body = initial_body.replace("[Name]", contact_name.split()[0])
        initial_subject = initial_subject.replace("{company_name}", company_name).replace("{Name}", contact_name.split()[0])
        return {"subject": initial_subject, "body": initial_body}


# --- SMTP Sending Logic ---
async def send_email_via_smtp(email_content: Dict[str, str], prospect_email: str, account: models.EmailAccount):
    """
    Connects to SMTP, authenticates, and sends the generated email using the account's ALIAS as the FROM address.
    Raises specific exceptions on failure for the caller to handle.
    """
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        raise ValueError(f"Password decryption failed for account {account.email_address}")

    # *** USE ALIAS for FROM address ***
    from_address = account.alias_email or account.email_address # CRITICAL: Use alias!
    if not account.alias_email:
         print(f"[EmailMarketer] WARNING: Sending from {account.email_address} because alias_email is not set in DB!")

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = from_address # Use the alias here
    msg['To'] = prospect_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=from_address.split('@')[-1]) # Use alias domain for Message-ID
    # Optional: Set Reply-To header to the actual Gmail account if needed for replies?
    # msg['Reply-To'] = account.email_address
    msg.set_content(email_content['body'])

    print(f"[EmailMarketer] Sending email to {prospect_email} via {account.smtp_user} (From: {from_address})...")

    # Exceptions like SMTPAuthenticationError, SMTPRecipientsRefused, SMTPSenderRefused,
    # SMTPConnectError, SMTPDataError, SMTPException will propagate up to the caller.
    async with aiosmtplib.SMTP(hostname=account.smtp_host, port=account.smtp_port, use_tls=True, timeout=SMTP_TIMEOUT) as smtp:
        # Login using the *actual* Gmail account credentials
        await smtp.login(account.smtp_user, decrypted_password)
        # Send message specifying the alias as the FROM address
        await smtp.send_message(msg, sender=from_address) # Pass sender explicitly if needed by server
        print(f"[EmailMarketer] Email sent successfully to {prospect_email} (From: {from_address})")
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
        batch_account_limit = settings.EMAIL_ACCOUNTS_PER_BATCH or 10 # Increase batch size
        active_accounts = await crud.get_batch_of_active_accounts(db, limit=batch_account_limit)
        if not active_accounts:
            print("[EmailMarketer] No active email accounts found in DB. Cannot send emails.")
            return
        print(f"[EmailMarketer] Fetched {len(active_accounts)} active accounts for this batch.")

        # Fetch prospects to process
        batch_prospect_limit = settings.EMAIL_BATCH_SIZE or 100 # Increase batch size
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
                account_index = (account_index + 1) % len(active_accounts)
                current_account_list_index = account_index
                email_account = active_accounts[current_account_list_index]

                # Check if alias is set, skip account if not (critical for strategy)
                if not email_account.alias_email:
                     print(f"[EmailMarketer] Skipping account {email_account.email_address} - alias_email not set.")
                     # Remove from current batch list to prevent reuse without alias
                     try:
                         active_accounts.pop(current_account_list_index)
                         account_index = current_account_list_index - 1
                     except IndexError:
                          account_index = -1
                     continue # Try next account

                # --- Minimal Warm-up Delay ---
                warmup_delay = 0
                if email_account.emails_sent_today < WARMUP_THRESHOLD:
                    warmup_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX) * (WARMUP_DELAY_MULTIPLIER - 1)
                    # print(f"[EmailMarketer] Applying minimal warm-up delay ({warmup_delay:.2f}s) for account {email_account.email_address}")

                # --- Aggressive Standard Delay ---
                standard_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX)
                await asyncio.sleep(standard_delay + warmup_delay)

                # --- Generate Email ---
                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    print(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} due to email generation failure.")
                    prospect_status_update = ("FAILED_GENERATION", None)
                    continue

                # --- Send Email ---
                await send_email_via_smtp(email_content, prospect.contact_email, email_account)

                # --- Handle Success ---
                print(f"[EmailMarketer] Successfully sent email to {prospect.contact_email} via {email_account.smtp_user} (From: {email_account.alias_email})")
                prospect_status_update = ("CONTACTED", datetime.datetime.now(datetime.timezone.utc))
                increment_send_count_for_account_id = email_account.account_id
                prospects_emailed_count += 1
                email_account.emails_sent_today += 1 # Increment memory count

            except (aiosmtplib.SMTPAuthenticationError, ValueError) as auth_err: # ValueError for decryption failure
                print(f"[EmailMarketer] Auth/Setup error for account {email_account.email_address if email_account else 'N/A'}: {auth_err}. Deactivating.")
                if email_account:
                    account_to_deactivate = (email_account.account_id, f"Auth/Setup Error: {str(auth_err)[:100]}")
                    if current_account_list_index != -1:
                         try:
                             active_accounts.pop(current_account_list_index)
                             account_index = current_account_list_index -1
                         except IndexError:
                              account_index = -1
                # Prospect status remains NEW

            except aiosmtplib.SMTPRecipientsRefused as bounce_err:
                print(f"[EmailMarketer] Hard bounce for {prospect.contact_email}: {bounce_err}. Marking as BOUNCED.")
                prospect_status_update = ("BOUNCED", None)
                if email_account: # Still count as sent attempt
                    increment_send_count_for_account_id = email_account.account_id
                    email_account.emails_sent_today += 1

            except (aiosmtplib.SMTPSenderRefused, aiosmtplib.SMTPDataError, smtplib.SMTPException, aiosmtplib.SMTPResponseException) as smtp_err:
                # Treat most SMTP errors as potential account issues
                error_message = str(smtp_err)
                deactivate_reason = f"SMTP Send Error: {error_message[:100]}"
                print(f"[EmailMarketer] SMTP Error for account {email_account.email_address if email_account else 'N/A'} sending to {prospect.contact_email}: {smtp_err}. Deactivating account.")
                # Check for common Gmail suspension/limit codes/messages
                if "Suspended" in error_message or "5.7.0" in error_message or "421" in error_message or "rate limit" in error_message.lower():
                    deactivate_reason = f"Account Likely Suspended/Limited: {error_message[:100]}"

                if email_account:
                    account_to_deactivate = (email_account.account_id, deactivate_reason)
                    if current_account_list_index != -1:
                         try:
                             active_accounts.pop(current_account_list_index)
                             account_index = current_account_list_index - 1
                         except IndexError:
                              account_index = -1
                # Prospect status remains NEW

            except Exception as e:
                print(f"[EmailMarketer] Unexpected error processing prospect {prospect.prospect_id}: {e}")
                import traceback
                traceback.print_exc()
                prospect_status_update = ("FAILED_SEND", None)
                if email_account: # Count as attempt if account was selected
                     increment_send_count_for_account_id = email_account.account_id
                     email_account.emails_sent_today += 1

            finally:
                # --- Database Updates for this Prospect/Account ---
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
                    await db.rollback()

    except Exception as e:
        print(f"[EmailMarketer] Error in email processing batch: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client: await client.aclose()

    print(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}.")