import asyncio
import random
import json
import datetime
import smtplib # For exception types
import re
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from app.core.config import settings
from app.db import crud, models
from app.agents.agent_utils import get_httpx_client # For LLM calls
from app.core.security import decrypt_data # To decrypt SMTP passwords
from app.agents.prospect_researcher import call_llm_api # Reuse the LLM utility

# --- Configuration ---
EMAIL_GENERATION_TIMEOUT = 90 # Seconds for LLM to generate email
SMTP_TIMEOUT = 30 # Seconds for SMTP operations
SEND_DELAY_MIN = 2.0 # Minimum delay between sends in seconds
SEND_DELAY_MAX = 5.0 # Maximum delay between sends in seconds

# --- Email Content Generation ---

def extract_signal_context(source_string: Optional[str]) -> str:
    """Extracts a concise description of the signal source for the LLM."""
    if not source_string:
        return "an unspecified business signal"
    if source_string.startswith("news_api_"):
        # Could potentially extract URL or source name if stored differently
        return "a recent news event relevant to their industry or company"
    if source_string.startswith("scraping_web_"):
        domain = source_string.replace("scraping_web_", "")
        return f"insights gathered from their website ({domain})"
    # Add more parsers for other sources
    return "an identified business signal"


async def generate_personalized_email(client: httpx.AsyncClient, prospect: models.Prospect) -> Optional[Dict[str, str]]:
    """Uses LLM to generate a personalized subject and body for a cold email."""
    company_name = prospect.company_name
    pain_point = prospect.potential_pain_point or "potential needs for enhanced data insights or reporting"
    signal_context = extract_signal_context(prospect.source)

    # Advanced Prompt Engineering for Human-like, High-Deliverability Email
    prompt = f"""
    Objective: Generate a hyper-personalized, concise cold email subject and body targeting '{company_name}'.
    Your Role: Act as a highly skilled B2B sales professional specializing in data analysis and research reporting.
    Key Information:
    - Company: {company_name}
    - Context/Signal: We detected {signal_context}.
    - Inferred Need: Based on this, they might be facing challenges related to: "{pain_point}".
    - Our Solution: We provide expert-level, AI-generated research reports (market analysis, competitive intelligence, etc.) rapidly.

    Instructions:
    1.  **Subject Line:** Create a short, intriguing, personalized subject line (under 8 words). Reference '{company_name}' or a highly relevant aspect of the inferred need. Avoid generic phrases like "Quick Question" or "Checking In". Make it sound like it's specifically for them.
    2.  **Email Body (Max 100 words):**
        -   **Opener:** Start with a brief, personalized observation referencing the inferred need or signal context *subtly*. Show you've done *some* homework, but don't sound creepy or overly familiar.
        -   **Connect Need to Solution:** Briefly connect their potential challenge ({pain_point}) to the value of timely, accurate research or reporting.
        -   **Introduce Solution:** Concisely mention our service: rapid, AI-powered, expert-level reports. Highlight the benefit (e.g., faster decisions, competitive edge, saved time).
        -   **Call to Action (Soft):** Suggest a brief call to discuss *if* this resonates, or offer a specific example report relevant to their potential need. Avoid hard sells.
        -   **Tone:** Professional, concise, helpful, slightly informal but respectful. Crucially, write like a human, not an AI. Use natural language, varied sentence structure, and avoid buzzwords or overly formal language. Aim to bypass AI detection filters.
    3.  **Output Format:** Respond ONLY with a JSON object containing two keys: "subject" and "body". Example: {{"subject": "Idea for {company_name} re: market data", "body": "Hi [Name - use placeholder if unknown],\n\nNoticed [brief observation related to pain point]...\n\nMany companies in your space find [challenge]. We help by providing [brief solution description]...\n\nOpen to a quick chat next week if this is relevant?\n\nBest,\n[Your Name/Agency Name]"}}

    Strict Command: Generate the JSON output directly. Do not include any preamble or explanation.
    """

    print(f"[EmailMarketer] Generating email for {prospect.company_name} (ID: {prospect.prospect_id}) based on pain: '{pain_point[:50]}...'")
    llm_response = await call_llm_api(client, prompt)

    if llm_response and isinstance(llm_response, dict) and "subject" in llm_response and "body" in llm_response:
        subject = llm_response["subject"].strip()
        body = llm_response["body"].strip()
        # Basic validation
        if subject and body and len(subject) < 80 and len(body) > 20:
             print(f"[EmailMarketer] LLM generated email - Subject: {subject}")
             return {"subject": subject, "body": body}
        else:
             print(f"[EmailMarketer] LLM response invalid or too short. Subject: {subject}, Body: {body[:50]}...")
    elif llm_response and isinstance(llm_response.get("raw_inference"), str):
         # Fallback if LLM didn't return JSON - try to parse raw text (less reliable)
         print("[EmailMarketer] LLM did not return valid JSON, attempting to parse raw text.")
         raw_text = llm_response["raw_inference"]
         subject_match = re.search(r"Subject:\s*(.*)", raw_text, re.IGNORECASE)
         body_match = re.search(r"Body:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
         if subject_match and body_match:
             subject = subject_match.group(1).strip()
             body = body_match.group(1).strip()
             if subject and body:
                 print(f"[EmailMarketer] Parsed from raw LLM response - Subject: {subject}")
                 return {"subject": subject, "body": body}
         print("[EmailMarketer] Could not parse subject/body from raw LLM response.")

    print(f"[EmailMarketer] Failed to generate valid email content for {prospect.company_name}")
    return None


# --- SMTP Sending Logic ---

@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), retry=retry_if_exception_type(aiosmtplib.SMTPConnectError))
async def send_email_via_smtp(email_content: Dict[str, str], prospect: models.Prospect, account: models.EmailAccount):
    """Connects to SMTP, authenticates, and sends the generated email."""
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        print(f"[EmailMarketer] CRITICAL: Failed to decrypt password for account {account.email_address}. Deactivating.")
        raise ValueError("Password decryption failed") # Raise specific error to trigger deactivation

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = account.email_address # Use the sending account's email
    msg['To'] = prospect.contact_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=account.email_address.split('@')[-1]) # Use sending domain

    # Set body (plain text is crucial for deliverability)
    plain_body = email_content['body']
    msg.set_content(plain_body)

    # Optionally add HTML alternative if LLM provides it or you format it
    # html_body = f"<html><body><p>{plain_body.replace('\n', '<br>')}</p></body></html>"
    # msg.add_alternative(html_body, subtype='html')

    # Add List-Unsubscribe header (requires setup of an endpoint/mailto) - Optional for cold email
    # msg['List-Unsubscribe'] = f'<mailto:unsubscribe@yourdomain.com?subject=unsubscribe:{prospect.contact_email}>, <https://yourdomain.com/unsubscribe?email={prospect.contact_email}>'

    print(f"[EmailMarketer] Attempting to send email to {prospect.contact_email} via {account.email_address} ({account.smtp_host}:{account.smtp_port})")

    try:
        async with aiosmtplib.SMTP(
            hostname=account.smtp_host,
            port=account.smtp_port,
            use_tls=True, # Assume STARTTLS, adjust if SSL needed directly
            timeout=SMTP_TIMEOUT,
        ) as smtp:
            # Login
            await smtp.login(account.smtp_user, decrypted_password)
            # Send
            await smtp.send_message(msg)
            print(f"[EmailMarketer] Email sent successfully to {prospect.contact_email}")
            return True # Indicate success

    except aiosmtplib.SMTPAuthenticationError as e:
        print(f"[EmailMarketer] SMTP Authentication Error for {account.email_address}: {e.code} - {e.message}. Deactivating account.")
        raise # Re-raise to be caught by the caller for deactivation
    except aiosmtplib.SMTPRecipientsRefused as e:
        # This often indicates the recipient address is invalid (hard bounce)
        print(f"[EmailMarketer] SMTP Recipient Refused for {prospect.contact_email}: {e.recipients}. Marking as BOUNCED.")
        # Specific handling for bounced emails
        raise # Re-raise to be caught by the caller for status update
    except aiosmtplib.SMTPSenderRefused as e:
        print(f"[EmailMarketer] SMTP Sender Refused for {account.email_address}: {e.sender}. Deactivating account.")
        raise # Re-raise for deactivation
    except aiosmtplib.SMTPDataError as e:
        print(f"[EmailMarketer] SMTP Data Error sending to {prospect.contact_email}: {e.code} - {e.message}. Might be spam filter related.")
        # Treat as failure, but might not require account deactivation immediately
        return False # Indicate failure
    except aiosmtplib.SMTPConnectError as e:
        print(f"[EmailMarketer] SMTP Connection Error to {account.smtp_host}:{account.smtp_port}: {e}")
        raise # Re-raise for retry logic
    except smtplib.SMTPException as e: # Catch broader SMTP errors
        print(f"[EmailMarketer] Generic SMTP Error sending to {prospect.contact_email} via {account.email_address}: {e}")
        return False # Indicate failure
    except Exception as e:
        print(f"[EmailMarketer] Unexpected error during SMTP operation for {prospect.contact_email}: {e}")
        return False # Indicate failure


# --- Main Agent Logic ---

async def process_email_batch(db: AsyncSession, shutdown_event: asyncio.Event):
    """Fetches a batch of prospects and attempts to generate and send emails."""
    if shutdown_event.is_set(): return

    prospects_processed_count = 0
    prospects_emailed_count = 0
    client = await get_httpx_client() # For LLM calls

    try:
        # 1. Fetch prospects needing emails
        prospects = await crud.get_new_prospects_for_emailing(db, limit=settings.EMAIL_BATCH_SIZE)
        if not prospects:
            # print("[EmailMarketer] No new prospects to email in this cycle.")
            return # Nothing to do

        print(f"[EmailMarketer] Fetched {len(prospects)} prospects for emailing.")

        for prospect in prospects:
            if shutdown_event.is_set(): break
            prospects_processed_count += 1
            email_account: Optional[models.EmailAccount] = None # Ensure defined in outer scope
            prospect_status = "FAILED" # Default status if things go wrong before sending
            last_contact_time = None

            try:
                # Add random delay before processing next prospect
                await asyncio.sleep(random.uniform(SEND_DELAY_MIN / 2, SEND_DELAY_MAX / 2))

                # 2. Generate personalized email content
                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    print(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} due to email generation failure.")
                    prospect_status = "FAILED_GENERATION" # Custom status?
                    await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status)
                    await db.commit() # Commit status update
                    continue # Move to next prospect

                # 3. Get an available sending account
                email_account = await crud.get_active_email_account_for_sending(db)
                if not email_account:
                    print("[EmailMarketer] No available email sending accounts found. Pausing email sending.")
                    # Should ideally notify admin or pause worker for longer
                    await db.rollback() # Rollback potential lock on prospect
                    break # Stop processing this batch

                # Lock the prospect row while we attempt sending
                # (already locked by get_new_prospects_for_emailing, but good practice)
                await db.refresh(prospect, with_for_update={'skip_locked': True})
                if prospect.status != 'NEW': # Check if status changed meanwhile
                     print(f"[EmailMarketer] Prospect {prospect.prospect_id} status changed, skipping.")
                     await db.rollback()
                     continue

                # 4. Attempt to send the email
                send_success = await send_email_via_smtp(email_content, prospect, email_account)

                if send_success:
                    prospect_status = "CONTACTED"
                    last_contact_time = datetime.datetime.now(datetime.timezone.utc)
                    prospects_emailed_count += 1
                    # 6. Increment sending count for the account
                    await crud.increment_email_sent_count(db, email_account.account_id)
                else:
                    # Send failed, but not necessarily a bounce or auth error (e.g., connection timeout after retries)
                    prospect_status = "FAILED_SEND" # Keep as NEW or use specific failed status?

                # 7. Update prospect status
                await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status, last_contacted_at=last_contact_time)
                await db.commit() # Commit prospect status and email count increment

            except (aiosmtplib.SMTPAuthenticationError, aiosmtplib.SMTPSenderRefused, ValueError) as auth_err:
                # Handle auth errors or decryption failure: Deactivate account, leave prospect as NEW
                print(f"[EmailMarketer] Authentication/Setup error for account {email_account.email_address if email_account else 'N/A'}. Deactivating.")
                if email_account:
                    await crud.set_email_account_inactive(db, email_account.account_id, reason=str(auth_err))
                # Rollback prospect status change attempt
                await db.rollback()
                # Do not commit prospect status change, it remains NEW
                # Break the inner loop for this batch if account issue is severe? Maybe not, try next prospect.
            except aiosmtplib.SMTPRecipientsRefused as bounce_err:
                # Handle hard bounce: Mark prospect as BOUNCED
                print(f"[EmailMarketer] Hard bounce detected for {prospect.contact_email}. Marking as BOUNCED.")
                prospect_status = "BOUNCED"
                await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status)
                # Still increment email count as an attempt was made & account was used
                if email_account:
                    await crud.increment_email_sent_count(db, email_account.account_id)
                await db.commit()
            except Exception as e:
                print(f"[EmailMarketer] Unexpected error processing prospect {prospect.prospect_id}: {e}")
                await db.rollback() # Rollback any changes for this prospect
                # Leave prospect status as NEW or set to FAILED? Let's leave as NEW for retry.

            # Add random delay after processing each prospect
            await asyncio.sleep(random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX))

    except Exception as e:
        print(f"[EmailMarketer] Error in email processing batch: {e}")
        # Log error
    finally:
        await client.aclose() # Close httpx client

    print(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}.")