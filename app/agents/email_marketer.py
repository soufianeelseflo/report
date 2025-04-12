# autonomous_agency/app/agents/email_marketer.py
import asyncio
import random
import json
import datetime
import smtplib # For exception types
import re
import logging
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Optional, Dict, Any, List

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import httpx

# Corrected relative imports for package structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api
    from Acumenis.app.core.security import decrypt_data
except ImportError:
    print("[EmailMarketer] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api
    from app.core.security import decrypt_data

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
EMAIL_GENERATION_TIMEOUT = 120 # Increased timeout for complex prompts
SMTP_TIMEOUT = 30 # Seconds for SMTP operations
SEND_DELAY_MIN = settings.EMAIL_SEND_DELAY_MIN or 1.0
SEND_DELAY_MAX = settings.EMAIL_SEND_DELAY_MAX or 5.0
WARMUP_THRESHOLD = settings.EMAIL_WARMUP_THRESHOLD or 5
WARMUP_DELAY_MULTIPLIER = settings.EMAIL_WARMUP_DELAY_MULTIPLIER or 1.5

# --- Email Content Generation ---

def extract_signal_context(source_string: Optional[str]) -> str:
    """Extracts a concise description of the signal source for the LLM."""
    if not source_string: return "an unspecified critical business signal we detected"
    if source_string.startswith("odr_"):
        query_part = source_string.replace("odr_", "").replace("_", " ")[:50]
        return f"intelligence derived from our analysis related to '{query_part}'"
    return "a high-priority business signal identified by our AI"

async def humanize_email_content(client: httpx.AsyncClient, subject: str, body: str) -> Optional[Dict[str, str]]:
    """Uses LLM to rewrite email content to sound human and bypass AI detectors, while maintaining persuasive edge."""
    humanize_prompt = f"""
    Rewrite the following hyper-personalized cold email subject and body.
    Goal: Make it sound like an urgent, direct message from a busy but insightful human expert. Bypass AI detection filters aggressively.
    Style: Extremely concise, high-impact, slightly informal, confident, bordering on provocative. Use short sentences, maybe subtle typos if natural. Avoid corporate jargon. Focus on immediate value/pain relief.
    Constraint: Retain the core value proposition (rapid AI reports for strategic advantage) and the direct call-to-action link placeholder '[Link to Order Page]'.

    Original Subject: "{subject}"
    Original Body:
    "{body}"

    Output ONLY a JSON object with the rewritten "subject" and "body". Example: {{"subject": "Re: {{company_name}} - quick thought", "body": "Hey [Name],\n\nSaw [signal context]. Data blind spots kill growth.\n\nWe generate deep AI intel reports (competitors, market shifts) in hours, not weeks. Solves [pain point].\n\nWorth $499 to act fast? [Link to Order Page]\n\n[Your Name]\nAcumenis AI"}}
    """
    logger.info("[EmailMarketer] Humanizing/Sharpening email content with LLM...")
    llm_response = await call_llm_api(client, humanize_prompt, model=settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-pro-latest")

    if llm_response and isinstance(llm_response, dict) and "subject" in llm_response and "body" in llm_response:
        h_subject = llm_response["subject"].strip()
        h_body = llm_response["body"].strip()
        if h_subject and h_body:
            logger.info(f"[EmailMarketer] Sharpened Subject: {h_subject}")
            return {"subject": h_subject, "body": h_body}
    logger.warning("[EmailMarketer] Failed to humanize/sharpen email content.")
    return None


async def generate_personalized_email(client: httpx.AsyncClient, prospect: models.Prospect) -> Optional[Dict[str, str]]:
    """Uses LLM to generate a hyper-personalized, aggressive subject and body, then humanizes/sharpens it."""
    company_name = prospect.company_name
    pain_point = prospect.potential_pain_point or "critical gaps in strategic intelligence"
    signal_context = extract_signal_context(prospect.source)
    # Use first name if available, otherwise fallback to a generic greeting part
    contact_first_name = (prospect.contact_name or "").split()[0] if prospect.contact_name else "there"
    greeting_name = contact_first_name if contact_first_name != "there" else company_name # Use company name if no contact name

    # Step 1: Generate Initial Aggressive Draft
    initial_prompt = f"""
    Objective: Generate an extremely concise, high-impact, psychologically persuasive cold email subject and body for '{greeting_name}' at '{company_name}'. Goal is IMMEDIATE click-through and purchase ($499/$999).
    Your Role: Act as a 'rogue' AI strategist providing exclusive, time-sensitive intelligence.
    Key Information:
    - Prospect: {greeting_name} @ {company_name}
    - Context/Signal: Detected {signal_context}. This implies vulnerability or missed opportunity related to: "{pain_point}".
    - Our Solution: Acumenis AI - We deliver deep competitive/market intelligence reports synthesized by advanced AI within HOURS. This provides [Specific Benefit, e.g., 'first-mover advantage', 'risk mitigation', 'competitor blindspot exploitation'].
    - Offer: Standard ($499) / Premium ($999) report activation.

    Instructions:
    1.  **Subject Line:** Max 6 words. Create extreme urgency/curiosity. Use prospect name/company if appropriate. Examples: "{greeting_name} - Urgent Intel?", "{company_name} Blindspot?", "Re: Your Competitor Risk".
    2.  **Email Body (Max 70 words):**
        -   Opener: Direct reference to signal/pain point (make it sound exclusive/discovered).
        -   Problem Agitation: Briefly state consequence of inaction (e.g., "Losing ground?", "Decision paralysis?").
        -   Solution Intro: Introduce Acumenis AI as the rapid antidote. Mention HOURS turnaround.
        -   Value Prop: State the core benefit bluntly.
        -   Call to Action (CTA): Use the placeholder "[Link to Order Page]". Frame as immediate action. Example: "Activate $499 stream now: [Link to Order Page]", "Get Nexus intel ($999): [Link to Order Page]".
    3.  **Tone:** Confident, direct, slightly mysterious, implies insider knowledge, extremely urgent.
    4.  **Output Format:** Respond ONLY with a JSON object containing "subject" and "body".

    Strict Command: Generate the JSON output directly. No preamble. Be aggressive.
    """

    logger.info(f"[EmailMarketer] Generating aggressive draft for {prospect.company_name} (ID: {prospect.prospect_id})")
    initial_llm_response = await call_llm_api(client, initial_prompt, model=settings.PREMIUM_REPORT_MODEL or "google/gemini-1.5-pro-latest")

    initial_subject = None
    initial_body = None
    if initial_llm_response and isinstance(initial_llm_response, dict):
        if "subject" in initial_llm_response and "body" in initial_llm_response:
            initial_subject = initial_llm_response["subject"].strip()
            initial_body = initial_llm_response["body"].strip()
        elif isinstance(initial_llm_response.get("raw_inference"), str):
             raw = initial_llm_response["raw_inference"]
             try:
                 json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE | re.DOTALL)
                 json_str = json_match.group(1).strip() if json_match else raw.strip()
                 # Handle potential leading/trailing text before/after JSON object
                 json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
                 parsed = json.loads(json_str)
                 initial_subject = parsed.get("subject","").strip()
                 initial_body = parsed.get("body","").strip()
             except (json.JSONDecodeError, AttributeError, IndexError) as e:
                 logger.warning(f"[EmailMarketer] Failed to parse LLM raw inference as JSON for {prospect.company_name}. Error: {e}")
                 # Attempt simple regex as last resort
                 subject_match = re.search(r'"?subject"?\s*:\s*"(.*?)"', raw, re.IGNORECASE)
                 body_match = re.search(r'"?body"?\s*:\s*"([\s\S]*?)"', raw, re.IGNORECASE | re.DOTALL)
                 if subject_match: initial_subject = subject_match.group(1).strip()
                 if body_match: initial_body = body_match.group(1).strip().replace('\\n', '\n')

    if not initial_subject or not initial_body:
        logger.error(f"[EmailMarketer] Failed to generate or parse initial aggressive draft for {prospect.company_name}. LLM Response: {initial_llm_response}")
        return None

    # Step 2: Humanize/Sharpen the Draft
    humanized_content = await humanize_email_content(client, initial_subject, initial_body)

    final_content = humanized_content or {"subject": initial_subject, "body": initial_body}
    if not humanized_content:
        logger.warning("[EmailMarketer] Humanization/Sharpening failed, using initial draft.")

    # Inject the order link and personalize placeholders
    order_link = f"{settings.AGENCY_BASE_URL}/order" # Ensure AGENCY_BASE_URL is correct
    final_content["body"] = final_content["body"].replace("[Link to Order Page]", order_link).replace("[Link]", order_link)
    # Append link if placeholder somehow missing
    if order_link not in final_content["body"]:
         final_content["body"] += f"\n\nActivate here: {order_link}"

    # Replace placeholders like [Name], [company_name], [signal context], [pain point], [Your Name]
    your_name = getattr(settings, 'EMAIL_SENDER_NAME', 'Acumenis AI Strategist') # Configurable sender name
    final_content["body"] = final_content["body"].replace("[Name]", contact_first_name)
    final_content["body"] = final_content["body"].replace("[company_name]", company_name)
    final_content["body"] = final_content["body"].replace("[signal context]", signal_context)
    final_content["body"] = final_content["body"].replace("[pain point]", pain_point)
    final_content["body"] = final_content["body"].replace("[Your Name]", your_name)
    final_content["subject"] = final_content["subject"].replace("{company_name}", company_name)
    final_content["subject"] = final_content["subject"].replace("{Name}", contact_first_name)

    return final_content


# --- SMTP Sending Logic ---
async def send_email_via_smtp(email_content: Dict[str, str], prospect_email: str, account: models.EmailAccount):
    """Connects to SMTP, authenticates, and sends the generated email using the account's ALIAS."""
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        raise ValueError(f"Password decryption failed for account {account.email_address}")

    from_address = account.alias_email # Already checked for non-null in process_email_batch

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = from_address
    msg['To'] = prospect_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=from_address.split('@')[-1])
    # Consider adding List-Unsubscribe headers for compliance
    # msg['List-Unsubscribe'] = f'<{settings.AGENCY_BASE_URL}/unsubscribe?email={prospect_email}>' # Example
    msg.set_content(email_content['body'])

    logger.info(f"[EmailMarketer] Sending email to {prospect_email} via {account.smtp_user} (From: {from_address})...")

    async with aiosmtplib.SMTP(hostname=account.smtp_host, port=account.smtp_port, use_tls=True, timeout=SMTP_TIMEOUT) as smtp:
        await smtp.login(account.smtp_user, decrypted_password)
        await smtp.send_message(msg, sender=from_address)
        logger.info(f"[EmailMarketer] Email sent successfully to {prospect_email} (From: {from_address})")


# --- SMTP Error Analysis ---
def analyze_smtp_error(error: Exception) -> Tuple[str, str]:
    """Analyzes SMTP errors to provide better reasons for account deactivation."""
    error_str = str(error).lower()
    status_code = getattr(error, 'code', None) # aiosmtplib exceptions often have a code

    # Authentication / Setup Errors
    if isinstance(error, (aiosmtplib.SMTPAuthenticationError, ValueError)):
        return "Auth/Setup Error", f"Auth/Config Error: {str(error)[:100]}"

    # Hard Bounces / Recipient Issues
    if isinstance(error, aiosmtplib.SMTPRecipientsRefused):
        # This indicates the recipient address is invalid, not necessarily an account issue.
        # We handle this by marking the prospect as BOUNCED, not deactivating the account.
        return "Recipient Issue", f"Recipient Refused: {str(error)[:100]}"

    # Sender / Account Issues (Suspension, Rate Limits, etc.)
    if isinstance(error, (aiosmtplib.SMTPSenderRefused, aiosmtplib.SMTPDataError, aiosmtplib.SMTPResponseException, smtplib.SMTPException)):
        reason = f"SMTP Send Error ({status_code or 'N/A'}): {error_str[:100]}"
        # Check for common suspension/limit indicators
        if "suspended" in error_str or \
           "rate limit" in error_str or \
           "too many messages" in error_str or \
           status_code in [421, 451, 550, 554] or \
           (status_code == 535 and "authentication failed" not in error_str): # 535 can be auth, but also other issues
            reason = f"Account Likely Suspended/Limited ({status_code or 'N/A'}): {error_str[:100]}"
        return "Account Issue", reason

    # Default / Unexpected
    return "Unknown SMTP Error", f"Unexpected SMTP Error: {error_str[:100]}"


# --- Main Agent Logic ---
async def process_email_batch(db: AsyncSession, shutdown_event: asyncio.Event):
    """Fetches prospects, generates emails, and sends using a rotating pool of accounts."""
    if shutdown_event.is_set(): return

    prospects_processed_count = 0
    prospects_emailed_count = 0
    client: Optional[httpx.AsyncClient] = None
    active_accounts: List[models.EmailAccount] = []
    account_index = -1

    try:
        client = await get_httpx_client()
        batch_account_limit = settings.EMAIL_ACCOUNTS_PER_BATCH or 10
        active_accounts = await crud.get_batch_of_active_accounts(db, limit=batch_account_limit)

        if not active_accounts:
            logger.warning("[EmailMarketer] No active email accounts with aliases found in DB. Cannot send emails.")
            return
        logger.info(f"[EmailMarketer] Fetched {len(active_accounts)} active accounts for this batch.")

        batch_prospect_limit = settings.EMAIL_BATCH_SIZE or 100
        prospects = await crud.get_new_prospects_for_emailing(db, limit=batch_prospect_limit)
        if not prospects:
            logger.info("[EmailMarketer] No new prospects found for emailing.")
            return
        logger.info(f"[EmailMarketer] Fetched {len(prospects)} prospects for emailing.")

        # --- Main Prospect Loop ---
        for prospect in prospects:
            if shutdown_event.is_set():
                logger.info("[EmailMarketer] Shutdown signal received during batch processing.")
                break
            prospects_processed_count += 1
            email_account: Optional[models.EmailAccount] = None
            prospect_status_update = None
            account_to_deactivate = None
            increment_send_count_for_account_id = None
            current_account_list_index = -1

            try:
                # --- Select Account with Rotation ---
                if not active_accounts:
                    logger.warning("[EmailMarketer] Ran out of active accounts for this batch.")
                    break
                account_index = (account_index + 1) % len(active_accounts)
                current_account_list_index = account_index
                email_account = active_accounts[current_account_list_index]

                # Alias check (should be guaranteed by get_batch_of_active_accounts, but double-check)
                if not email_account.alias_email:
                     logger.error(f"[EmailMarketer] CRITICAL: Account {email_account.email_address} selected but has no alias! Skipping.")
                     active_accounts.pop(current_account_list_index)
                     account_index -= 1
                     continue

                # --- Delays ---
                warmup_delay = 0
                if email_account.emails_sent_today < WARMUP_THRESHOLD:
                    warmup_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX) * (WARMUP_DELAY_MULTIPLIER - 1)
                standard_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX)
                await asyncio.sleep(standard_delay + warmup_delay)

                # --- Generate Email ---
                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    logger.warning(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} due to email generation failure.")
                    prospect_status_update = ("FAILED_GENERATION", None)
                    continue

                # --- Send Email ---
                await send_email_via_smtp(email_content, prospect.contact_email, email_account)

                # --- Handle Success ---
                logger.info(f"[EmailMarketer] Successfully sent email to {prospect.contact_email} via {email_account.smtp_user} (From: {email_account.alias_email})")
                prospect_status_update = ("CONTACTED", datetime.datetime.now(datetime.timezone.utc))
                increment_send_count_for_account_id = email_account.account_id
                prospects_emailed_count += 1
                # Increment in-memory count for immediate check in next loop iteration
                email_account.emails_sent_today += 1

            except Exception as e:
                # --- Enhanced Error Handling ---
                error_type, reason = analyze_smtp_error(e)
                logger.error(f"[EmailMarketer] Error processing prospect {prospect.prospect_id} with account {email_account.email_address if email_account else 'N/A'}. Type: {error_type}, Reason: {reason}", exc_info=False) # Don't need full trace for common errors

                if error_type == "Recipient Issue":
                    prospect_status_update = ("BOUNCED", None)
                    if email_account: # Still count as a send attempt
                        increment_send_count_for_account_id = email_account.account_id
                        email_account.emails_sent_today += 1
                elif error_type == "Account Issue" or error_type == "Auth/Setup Error":
                    if email_account:
                        account_to_deactivate = (email_account.account_id, reason)
                        # Remove account from current batch immediately
                        if current_account_list_index != -1:
                            try:
                                active_accounts.pop(current_account_list_index)
                                account_index -= 1 # Adjust index
                            except IndexError:
                                account_index = -1 # Reset if list becomes empty
                    # Prospect status remains NEW, will be retried later with a different account
                else: # Unknown or other errors
                    prospect_status_update = ("FAILED_SEND", None) # Mark prospect as failed send
                    if email_account: # Still count as attempt
                        increment_send_count_for_account_id = email_account.account_id
                        email_account.emails_sent_today += 1
                    # Log unexpected errors with full trace
                    logger.error(f"[EmailMarketer] Unexpected error details:", exc_info=True)


            finally:
                # --- Database Updates ---
                try:
                    if account_to_deactivate:
                        await crud.set_email_account_inactive(db, account_to_deactivate[0], account_to_deactivate[1])
                    if increment_send_count_for_account_id:
                        await crud.increment_email_sent_count(db, increment_send_count_for_account_id)
                    if prospect_status_update:
                        await crud.update_prospect_status(db, prospect.prospect_id, status=prospect_status_update[0], last_contacted_at=prospect_status_update[1])
                    await db.commit() # Commit changes for this prospect/account interaction
                except Exception as db_e:
                    logger.critical(f"[EmailMarketer] Failed to commit DB updates for prospect {prospect.prospect_id}: {db_e}", exc_info=True)
                    await db.rollback()

    except Exception as e:
        logger.error(f"[EmailMarketer] Error in email processing batch: {e}", exc_info=True)
    finally:
        if client: await client.aclose()

    logger.info(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}.")