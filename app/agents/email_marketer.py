import asyncio
import random
import json
import datetime
import smtplib # For exception types
import re
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Optional, Dict, Any

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import httpx

from autonomous_agency.app.core.config import settings
from autonomous_agency.app.db import crud, models
from autonomous_agency.app.agents.agent_utils import get_httpx_client, call_llm_api # Use updated call_llm_api
from autonomous_agency.app.core.security import decrypt_data

# --- Configuration ---
EMAIL_GENERATION_TIMEOUT = 90 # Seconds for LLM to generate email
SMTP_TIMEOUT = 30 # Seconds for SMTP operations
SEND_DELAY_MIN = 2.0 # Minimum delay between sends in seconds
SEND_DELAY_MAX = 5.0 # Maximum delay between sends in seconds

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
# @retry remains the same as previous version
@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), retry=retry_if_exception_type(aiosmtplib.SMTPConnectError))
async def send_email_via_smtp(email_content: Dict[str, str], prospect_email: str, account: models.EmailAccount):
    """Connects to SMTP, authenticates, and sends the generated email."""
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        print(f"[EmailMarketer] CRITICAL: Failed to decrypt password for account {account.email_address}. Deactivating.")
        raise ValueError("Password decryption failed")

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = account.email_address
    msg['To'] = prospect_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=account.email_address.split('@')[-1])
    msg.set_content(email_content['body'])

    print(f"[EmailMarketer] Attempting to send email to {prospect_email} via {account.email_address} ({account.smtp_host}:{account.smtp_port})")

    try:
        async with aiosmtplib.SMTP(hostname=account.smtp_host, port=account.smtp_port, use_tls=True, timeout=SMTP_TIMEOUT) as smtp:
            await smtp.login(account.smtp_user, decrypted_password)
            await smtp.send_message(msg)
            print(f"[EmailMarketer] Email sent successfully to {prospect_email}")
            return True
    except aiosmtplib.SMTPAuthenticationError as e:
        print(f"[EmailMarketer] SMTP Auth Error for {account.email_address}: {e.code} - {e.message}. Deactivating.")
        raise
    except aiosmtplib.SMTPRecipientsRefused as e:
        print(f"[EmailMarketer] SMTP Recipient Refused for {prospect_email}: {e.recipients}. Marking as BOUNCED.")
        raise
    except aiosmtplib.SMTPSenderRefused as e:
        print(f"[EmailMarketer] SMTP Sender Refused for {account.email_address}: {e.sender}. Deactivating.")
        raise
    except aiosmtplib.SMTPDataError as e:
        print(f"[EmailMarketer] SMTP Data Error sending to {prospect_email}: {e.code} - {e.message}.")
        return False
    except aiosmtplib.SMTPConnectError as e:
        print(f"[EmailMarketer] SMTP Connect Error to {account.smtp_host}:{account.smtp_port}: {e}")
        raise
    except smtplib.SMTPException as e:
        print(f"[EmailMarketer] Generic SMTP Error sending to {prospect_email} via {account.email_address}: {e}")
        return False
    except Exception as e:
        print(f"[EmailMarketer] Unexpected SMTP error for {prospect_email}: {e}")
        return False


# --- Main Agent Logic ---
async def process_email_batch(db: AsyncSession, shutdown_event: asyncio.Event):
    """Fetches a batch of prospects and attempts to generate and send emails."""
    if shutdown_event.is_set(): return

    prospects_processed_count = 0
    prospects_emailed_count = 0
    client: Optional[httpx.AsyncClient] = None

    try:
        client = await get_httpx_client()
        prospects = await crud.get_new_prospects_for_emailing(db, limit=settings.EMAIL_BATCH_SIZE)
        if not prospects: return

        print(f"[EmailMarketer] Fetched {len(prospects)} prospects for emailing.")

        for prospect in prospects:
            if shutdown_event.is_set(): break
            prospects_processed_count += 1
            email_account: Optional[models.EmailAccount] = None
            prospect_status = "FAILED_SEND" # Default if send fails non-critically
            last_contact_time = None

            try:
                await asyncio.sleep(random.uniform(SEND_DELAY_MIN / 2, SEND_DELAY_MAX / 2))

                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    print(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} due to email generation failure.")
                    await crud.update_prospect_status(db, prospect.prospect_id, status="FAILED_GENERATION")
                    await db.commit()
                    continue

                email_account = await crud.get_active_email_account_for_sending(db)
                if not email_account:
                    print("[EmailMarketer] No available email sending accounts found. Pausing email sending for this batch.")
                    await db.rollback()
                    break # Stop processing this batch

                # Re-check prospect status before sending
                await db.refresh(prospect, with_for_update={'skip_locked': True})
                if prospect.status != 'NEW':
                     print(f"[EmailMarketer] Prospect {prospect.prospect_id} status changed, skipping.")
                     await db.rollback() # Release lock
                     continue

                send_success = await send_email_via_smtp(email_content, prospect.contact_email, email_account)

                if send_success:
                    prospect_status = "CONTACTED"
                    last_contact_time = datetime.datetime.now(datetime.timezone.utc)
                    prospects_emailed_count += 1
                    await crud.increment_email_sent_count(db, email_account.account_id)
                # else: prospect_status remains FAILED_SEND

                await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status, last_contacted_at=last_contact_time)
                await db.commit()

            except (aiosmtplib.SMTPAuthenticationError, aiosmtplib.SMTPSenderRefused, ValueError) as auth_err:
                print(f"[EmailMarketer] Auth/Setup error for account {email_account.email_address if email_account else 'N/A'}. Deactivating.")
                if email_account:
                    await crud.set_email_account_inactive(db, email_account.account_id, reason=str(auth_err))
                await db.commit() # Commit account deactivation
                # Prospect status remains NEW, will be retried later with another account
            except aiosmtplib.SMTPRecipientsRefused as bounce_err:
                print(f"[EmailMarketer] Hard bounce for {prospect.contact_email}. Marking as BOUNCED.")
                await crud.update_prospect_status(db, prospect.prospect_id, status="BOUNCED")
                if email_account: # Still count as sent attempt
                    await crud.increment_email_sent_count(db, email_account.account_id)
                await db.commit()
            except Exception as e:
                print(f"[EmailMarketer] Unexpected error processing prospect {prospect.prospect_id}: {e}")
                await db.rollback() # Rollback any changes for this prospect

            await asyncio.sleep(random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX))

    except Exception as e:
        print(f"[EmailMarketer] Error in email processing batch: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client: await client.aclose()

    print(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}.")