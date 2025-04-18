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
from typing import Optional, Dict, Any, List, Tuple # Added Tuple

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select # Import select
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_random_exponential # Added wait_random_exponential
import httpx

# Corrected relative imports for package structure
try:
    from Nexus Plan.app.core.config import settings
    from Nexus Plan.app.db import crud, models
    from Nexus Plan.app.agents.agent_utils import get_httpx_client, call_llm_api
    from Nexus Plan.app.core.security import decrypt_data
    from Nexus Plan.app.db.base import get_worker_session # Import session getter
except ImportError:
    print("[EmailMarketer] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api
    from app.core.security import decrypt_data
    from app.db.base import get_worker_session

# Setup logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
     logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Configuration ---
EMAIL_GENERATION_TIMEOUT = 120 # Faster generation timeout
SMTP_TIMEOUT = 30 # Seconds for SMTP operations (faster)
SEND_DELAY_MIN = settings.EMAIL_SEND_DELAY_MIN # Use aggressive settings
SEND_DELAY_MAX = settings.EMAIL_SEND_DELAY_MAX # Use aggressive settings
# WARMUP_THRESHOLD = settings.EMAIL_WARMUP_THRESHOLD # REMOVED
# WARMUP_DELAY_MULTIPLIER = settings.EMAIL_WARMUP_DELAY_MULTIPLIER # REMOVED
EMAIL_MARKETER_SENDER_NAME = settings.EMAIL_SENDER_NAME # Use configured name

# --- Email Content Generation ---

def extract_signal_context(source_string: Optional[str]) -> str:
    """Extracts a concise description of the signal source for the LLM."""
    if not source_string: return "an unspecified critical business signal we detected"
    # Make matching more robust
    if source_string.lower().startswith("odr_"):
        query_part = source_string.split('_', 1)[-1].replace("_", " ")[:60] # Get part after odr_
        return f"intelligence derived from our deep analysis related to '{query_part}'"
    # Add more potential source parsers here if needed
    # Example: if source_string.startswith("manual_"): return "a specific opportunity identified for your company"
    return "a high-priority business signal identified by our AI" # Default fallback

async def mutate_and_humanize_email_content(client: httpx.AsyncClient, subject: str, body: str) -> Optional[Dict[str, str]]:
    """Uses LLM to rewrite email content to sound human AND aggressively evade filters."""
    # NEW Prompt: Focus on mutation and evasion
    humanize_prompt = f"""
    Rewrite and MUTATE the following hyper-personalized cold email subject and body.
    Goal: Make it sound like an urgent, direct message from a busy human expert. AGGRESSIVELY bypass AI detection and spam filters. Emphasize exclusivity and speed. Use unconventional phrasing, synonyms, maybe subtle natural-sounding typos or contractions. Vary sentence structure significantly.
    Constraint: Retain the core value proposition (rapid AI reports for strategic advantage) and the direct call-to-action link placeholder '[Link to Order Page]'. Ensure the sender name placeholder '[Your Name]' is present at the end. Keep body under 80 words.

    Original Subject: "{subject}"
    Original Body:
    "{body}"

    Output ONLY a JSON object with the rewritten "subject" and "body".
    Example JSON (focus on variation):
    {{
      "subject": "Quick thought re: {{company_name}} strategy",
      "body": "Hey [Name] - saw the [signal context] stuff. Looks like a gap.\n\nWe spin up deep AI intel reports (competitors, market shifts etc) fast - hours, not weeks. Fixes the [pain point] issue.\n\nWorth $499 to move quicker? [Link to Order Page]\n\nCheers,\n[Your Name]"
    }}
    """
    logger.info("[EmailMarketer] Mutating/Humanizing email content with LLM...")
    llm_response = await call_llm_api(
        client,
        humanize_prompt,
        model=settings.PREMIUM_REPORT_MODEL # Use premium for better mutation/nuance
    )

    if llm_response and isinstance(llm_response, dict) and "subject" in llm_response and "body" in llm_response:
        h_subject = llm_response["subject"].strip()
        h_body = llm_response["body"].strip()
        # Basic validation
        if h_subject and h_body and len(h_body) > 10 and '[Link to Order Page]' in h_body and '[Your Name]' in h_body:
            logger.info(f"[EmailMarketer] Mutated Subject: {h_subject}")
            return {"subject": h_subject, "body": h_body}
        else:
             logger.warning(f"[EmailMarketer] Mutated content failed validation (missing placeholders or too short). Subject: {h_subject}, Body: {h_body[:100]}...")
    else:
        logger.warning(f"[EmailMarketer] Failed to get valid JSON structure from mutate LLM. Response: {llm_response}")

    logger.warning("[EmailMarketer] Mutation/Humanization failed, using initial draft.")
    return None


async def generate_personalized_email(client: httpx.AsyncClient, prospect: models.Prospect) -> Optional[Dict[str, str]]:
    """Uses LLM to generate a hyper-personalized, aggressive subject and body, then mutates/humanizes it."""
    company_name = prospect.company_name
    pain_point = prospect.potential_pain_point or "closing critical gaps in strategic intelligence" # Make pain point more active
    signal_context = extract_signal_context(prospect.source)
    # Use first name if available, otherwise fallback to a generic greeting part
    contact_first_name = (prospect.contact_name or "").split()[0] if prospect.contact_name else "there"
    greeting_name = contact_first_name if contact_first_name != "there" else company_name # Use company name if no contact name

    # Step 1: Generate Initial Aggressive Draft (Keep this prompt aggressive)
    initial_prompt = f"""
    Objective: Generate an extremely concise, psychologically persuasive cold email subject and body for '{greeting_name}' at '{company_name}'. Goal is IMMEDIATE click-through and purchase ($499/$999).
    Your Role: Act as a 'rogue' AI strategist providing exclusive, time-sensitive intelligence based on a detected signal.
    Key Information:
    - Prospect: {greeting_name} @ {company_name}
    - Context/Signal: Detected signal related to: {signal_context}. This implies a potential vulnerability or missed opportunity regarding: "{pain_point}".
    - Our Solution: Nexus Plan AI - We deliver deep competitive/market intelligence reports synthesized by advanced AI within HOURS. This provides [Specific Benefit, e.g., 'first-mover advantage', 'risk mitigation', 'competitor blindspot exploitation'].
    - Offer: Standard ($499) / Premium ($999) report activation via link.

    Instructions:
    1.  **Subject Line:** Max 7 words. Create extreme urgency/curiosity. Use prospect name/company. Examples: "{greeting_name} - Urgent Intel on [Signal Context]?", "{company_name} Competitor Risk Analysis?", "Re: Your Strategy & [Pain Point]". Make it look like a reply or internal forward if possible (Re:, Fwd:).
    2.  **Email Body (Strict Max 75 words):**
        -   Opener: Direct reference to the signal/pain point (make it sound exclusive/discovered). E.g., "Saw [signal context] - looks like you need intel on [pain point] fast."
        -   Problem Agitation: Briefly state consequence of inaction (e.g., "Falling behind?", "Decision paralysis?", "Competitors moving?").
        -   Solution Intro: Introduce Nexus Plan AI as the rapid antidote. Mention HOURS turnaround.
        -   Value Prop: State the core benefit bluntly. E.g., "Get actionable intel before rivals do."
        -   Call to Action (CTA): Use the placeholder "[Link to Order Page]". Frame as immediate action. Example: "Activate $499 stream now: [Link to Order Page]", "Get Nexus intel ($999): [Link to Order Page]".
    3.  **Tone:** Confident, direct, slightly mysterious, implies insider knowledge, extremely urgent, minimalist.
    4.  **Output Format:** Respond ONLY with a JSON object containing "subject" and "body". Ensure '[Link to Order Page]' and '[Your Name]' placeholders are included in the body.

    Strict Command: Generate the JSON output directly. No preamble. Be aggressive and concise.
    """

    logger.info(f"[EmailMarketer] Generating aggressive draft for {prospect.company_name} (ID: {prospect.prospect_id})")
    initial_llm_response = await call_llm_api(
        client,
        initial_prompt,
        model=settings.PREMIUM_REPORT_MODEL # Use premium for generation
    )

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
                 start_index = json_str.find('{')
                 end_index = json_str.rfind('}')
                 if start_index != -1 and end_index != -1:
                     json_str = json_str[start_index : end_index + 1]
                     parsed = json.loads(json_str)
                     initial_subject = parsed.get("subject","").strip()
                     initial_body = parsed.get("body","").strip()
                 else:
                      logger.warning(f"[EmailMarketer] Could not extract JSON object from raw inference for {prospect.company_name}.")
             except (json.JSONDecodeError, AttributeError, IndexError) as e:
                 logger.warning(f"[EmailMarketer] Failed to parse LLM raw inference as JSON for {prospect.company_name}. Error: {e}")

    if not initial_subject or not initial_body or '[Link to Order Page]' not in initial_body: # Ensure placeholder exists
        logger.error(f"[EmailMarketer] Failed to generate valid initial draft for {prospect.company_name}. Missing subject, body, or placeholder. LLM Response: {initial_llm_response}")
        return None

    # Step 2: Mutate/Humanize the Draft
    mutated_content = await mutate_and_humanize_email_content(client, initial_subject, initial_body)

    final_content = mutated_content or {"subject": initial_subject, "body": initial_body}

    # Inject the order link and personalize placeholders
    order_link = f"{str(settings.AGENCY_BASE_URL).rstrip('/')}/order" # Ensure AGENCY_BASE_URL is correct
    final_content["body"] = final_content["body"].replace("[Link to Order Page]", order_link)
    # Append link if placeholder somehow missing after mutation (safety net)
    if order_link not in final_content["body"]:
         final_content["body"] = final_content["body"].rstrip() + f"\n\nActivate here: {order_link}"

    # Replace placeholders like [Name], [company_name], [signal context], [pain point], [Your Name]
    final_content["body"] = final_content["body"].replace("[Name]", contact_first_name)
    final_content["body"] = final_content["body"].replace("[company_name]", company_name)
    final_content["body"] = final_content["body"].replace("{{company_name}}", company_name) # Handle different placeholder styles
    final_content["body"] = final_content["body"].replace("[signal context]", signal_context)
    final_content["body"] = final_content["body"].replace("[pain point]", pain_point)
    final_content["body"] = final_content["body"].replace("[Your Name]", EMAIL_MARKETER_SENDER_NAME)
    final_content["subject"] = final_content["subject"].replace("{company_name}", company_name)
    final_content["subject"] = final_content["subject"].replace("[company_name]", company_name)
    final_content["subject"] = final_content["subject"].replace("{Name}", contact_first_name)
    final_content["subject"] = final_content["subject"].replace("[Name]", contact_first_name)

    # Final length check
    if len(final_content["body"]) > 500: # Arbitrary limit to catch runaway generation
        logger.warning(f"Generated email body for {prospect.company_name} is unusually long ({len(final_content['body'])} chars). Truncating.")
        final_content["body"] = final_content["body"][:500] + "..."

    return final_content


# --- SMTP Sending Logic ---
# Add retry logic to SMTP sending for transient network issues
@retry(
    stop=stop_after_attempt(2), # Fewer retries for faster failure
    wait=wait_random_exponential(multiplier=0.5, max=5), # Faster backoff
    retry=retry_if_exception_type((aiosmtplib.SMTPServerDisconnected, aiosmtplib.SMTPConnectError, TimeoutError))
)
async def send_email_via_smtp(email_content: Dict[str, str], prospect_email: str, account: models.EmailAccount):
    """Connects to SMTP, authenticates, and sends the generated email using the account's ALIAS."""
    decrypted_password = decrypt_data(account.smtp_password_encrypted)
    if not decrypted_password:
        logger.error(f"CRITICAL: Password decryption failed for account {account.email_address} (ID: {account.account_id}).")
        raise ValueError(f"Password decryption failed for account {account.email_address}")

    from_address = account.alias_email
    if not from_address:
        logger.error(f"CRITICAL: Account {account.email_address} (ID: {account.account_id}) has no alias_email. Cannot send.")
        raise ValueError(f"Missing alias_email for account {account.email_address}")

    msg = EmailMessage()
    msg['Subject'] = email_content['subject']
    msg['From'] = f"{EMAIL_MARKETER_SENDER_NAME} <{from_address}>" # Format From header nicely
    msg['To'] = prospect_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=from_address.split('@')[-1])
    # Add List-Unsubscribe headers for compliance and deliverability
    unsubscribe_url = f"{str(settings.AGENCY_BASE_URL).rstrip('/')}/unsubscribe?email={prospect_email}" # Basic unsubscribe link
    msg['List-Unsubscribe'] = f'<{unsubscribe_url}>'
    msg['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
    msg.set_content(email_content['body'])

    logger.info(f"[EmailMarketer] Attempting to send email to {prospect_email} via {account.smtp_user} (From: {from_address})...")

    use_tls_wrapper = account.smtp_port != 465
    use_starttls = account.smtp_port == 587

    smtp_client: Optional[aiosmtplib.SMTP] = None
    try:
        smtp_client = aiosmtplib.SMTP(
            hostname=account.smtp_host,
            port=account.smtp_port,
            use_tls=use_tls_wrapper,
            timeout=SMTP_TIMEOUT
        )
        await smtp_client.connect()
        if use_starttls:
            await smtp_client.starttls()

        await smtp_client.login(account.smtp_user, decrypted_password)
        await smtp_client.send_message(msg, sender=from_address)
        logger.info(f"[EmailMarketer] Email sent successfully to {prospect_email} (From: {from_address})")

    except aiosmtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Auth Error for {account.email_address} (User: {account.smtp_user}). Error: {e}")
        raise # Re-raise specific error type for analysis
    except aiosmtplib.SMTPRecipientsRefused as e:
        logger.warning(f"SMTP Recipient Refused for {prospect_email}: {e}")
        raise # Re-raise specific error type for analysis
    except aiosmtplib.SMTPSenderRefused as e:
         logger.error(f"SMTP Sender Refused for {from_address} (Account: {account.email_address}). Error: {e}")
         raise # Re-raise specific error type for analysis
    except (aiosmtplib.SMTPServerDisconnected, aiosmtplib.SMTPConnectError, TimeoutError) as e:
         logger.warning(f"SMTP Connection/Timeout Error for {account.smtp_host}:{account.smtp_port}. Error: {e}. Retrying...")
         raise # Re-raise to trigger Tenacity retry
    except Exception as e:
         logger.error(f"Unexpected SMTP Error sending report for prospect {prospect_email} via {from_address}: {e}", exc_info=True)
         raise # Re-raise for general error handling
    finally:
        if smtp_client and smtp_client.is_connected:
            await smtp_client.quit()


# --- SMTP Error Analysis ---
def analyze_smtp_error(error: Exception) -> Tuple[str, str]:
    """Analyzes SMTP errors to provide better reasons for account deactivation."""
    error_str = str(error).lower()
    status_code = getattr(error, 'code', None) # aiosmtplib exceptions often have a code

    # Authentication / Setup Errors (High Severity - Deactivate Account)
    if isinstance(error, (aiosmtplib.SMTPAuthenticationError, ValueError)):
        reason = f"Auth/Config Error: {str(error)[:150]}"
        logger.error(f"Analyzed error as Auth/Setup: {reason}")
        return "ACCOUNT_ISSUE", reason

    # Hard Bounces / Recipient Issues (Medium Severity - Mark Prospect)
    if isinstance(error, aiosmtplib.SMTPRecipientsRefused):
        reason = f"Recipient Refused: {str(error)[:150]}"
        logger.warning(f"Analyzed error as Recipient Issue: {reason}")
        return "RECIPIENT_ISSUE", reason

    # Sender / Account Issues (Suspension, Rate Limits, etc.) (High Severity - Deactivate Account)
    if isinstance(error, (aiosmtplib.SMTPSenderRefused, aiosmtplib.SMTPDataError, aiosmtplib.SMTPResponseException)):
        reason = f"SMTP Send Error ({status_code or 'N/A'}): {error_str[:150]}"
        # Check for common suspension/limit indicators
        if "suspended" in error_str or \
           "block" in error_str or \
           "spam" in error_str or \
           "rate limit" in error_str or \
           "too many messages" in error_str or \
           "policy violation" in error_str or \
           status_code in [421, 451, 550, 554] or \
           (status_code == 535 and "authentication failed" not in error_str): # 535 can be auth, but also other issues
            reason = f"Account Likely Suspended/Limited ({status_code or 'N/A'}): {error_str[:150]}"
        logger.error(f"Analyzed error as Account Issue: {reason}")
        return "ACCOUNT_ISSUE", reason

    # Transient Connection/Timeout Issues (Low Severity - Retryable, don't mark anything yet)
    if isinstance(error, (aiosmtplib.SMTPServerDisconnected, aiosmtplib.SMTPConnectError, TimeoutError)):
        reason = f"Transient Connection/Timeout: {str(error)[:150]}"
        logger.warning(f"Analyzed error as Transient: {reason}")
        return "TRANSIENT_ERROR", reason

    # Default / Unexpected (Medium Severity - Mark Prospect as Failed Send)
    reason = f"Unexpected SMTP Error: {str(error)[:150]}"
    logger.error(f"Analyzed error as Unknown: {reason}", exc_info=True) # Log full trace for unknowns
    return "UNKNOWN_ERROR", reason


# --- Main Agent Logic ---
async def process_email_batch(db: AsyncSession, shutdown_event: asyncio.Event):
    """Fetches prospects, generates emails, and sends using a rapidly rotating pool of accounts."""
    if shutdown_event.is_set(): return

    prospects_processed_count = 0
    prospects_emailed_count = 0
    prospects_failed_count = 0
    accounts_deactivated_count = 0
    client: Optional[httpx.AsyncClient] = None
    active_accounts: List[models.EmailAccount] = []
    account_index = -1 # Start before the first account
    # Track usage *within this batch* to enforce rapid rotation (e.g., 1-3 emails per account)
    batch_send_counts: Dict[int, int] = {}
    MAX_SENDS_PER_ACCOUNT_PER_BATCH = 3 # Configurable: How many emails max per account in one go

    try:
        client = await get_httpx_client()
        # Fetch ALL active accounts for rotation potential
        batch_account_limit = 100 # Fetch a large pool
        stmt = select(models.EmailAccount).where(models.EmailAccount.is_active == True).where(models.EmailAccount.alias_email.isnot(None)).limit(batch_account_limit)
        result = await db.execute(stmt)
        active_accounts = list(result.scalars().all()) # Convert to list

        if not active_accounts:
            logger.warning("[EmailMarketer] No active email accounts with valid aliases found in DB. Cannot send emails this cycle.")
            return
        logger.info(f"[EmailMarketer] Fetched {len(active_accounts)} active accounts for this batch.")
        random.shuffle(active_accounts) # Shuffle for randomness

        batch_prospect_limit = settings.EMAIL_BATCH_SIZE # Use configured batch size (e.g., 50)
        prospects = await crud.get_new_prospects_for_emailing(db, limit=batch_prospect_limit)
        await db.commit() # Commit the transaction locking prospects

        if not prospects:
            logger.info("[EmailMarketer] No new prospects found for emailing this cycle.")
            return
        logger.info(f"[EmailMarketer] Fetched {len(prospects)} prospects for emailing.")

        # --- Main Prospect Loop ---
        for prospect in prospects:
            if shutdown_event.is_set():
                logger.info("[EmailMarketer] Shutdown signal received during batch processing.")
                break
            prospects_processed_count += 1
            email_account: Optional[models.EmailAccount] = None
            prospect_status_update: Optional[Tuple[str, Optional[datetime.datetime]]] = None
            account_to_deactivate: Optional[Tuple[int, str]] = None
            increment_send_count_for_account_id: Optional[int] = None
            current_account_list_index = -1 # Track index in the current batch list

            try:
                # --- Select Account with RAPID Rotation & Check Limits ---
                if not active_accounts:
                    logger.warning("[EmailMarketer] Ran out of usable accounts for this batch.")
                    break # Stop processing this batch of prospects

                # Find the next usable account in the shuffled list
                account_found = False
                start_search_index = (account_index + 1) % len(active_accounts)
                for i in range(len(active_accounts)):
                    check_index = (start_search_index + i) % len(active_accounts)
                    potential_account = active_accounts[check_index]
                    account_id = potential_account.account_id
                    # Check batch usage limit first
                    if batch_send_counts.get(account_id, 0) < MAX_SENDS_PER_ACCOUNT_PER_BATCH:
                        # Then check daily limit (though less relevant with rapid rotation)
                        if potential_account.emails_sent_today < potential_account.daily_limit:
                            email_account = potential_account
                            current_account_list_index = check_index
                            account_index = check_index # Update main index
                            account_found = True
                            break
                        else:
                             logger.debug(f"Account {account_id} skipped (daily limit reached).")
                    else:
                         logger.debug(f"Account {account_id} skipped (batch limit {MAX_SENDS_PER_ACCOUNT_PER_BATCH} reached).")


                if not account_found:
                    logger.warning("[EmailMarketer] All available accounts reached their batch/daily limit.")
                    break # Stop processing prospects for this batch

                # --- Delays (Minimal) ---
                # REMOVED WARMUP LOGIC
                total_delay = random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX) # Use aggressive base delay only
                logger.debug(f"Sleeping for {total_delay:.2f}s before sending (Account: {email_account.account_id})")
                await asyncio.sleep(total_delay)

                # --- Generate Email (with Mutation) ---
                email_content = await generate_personalized_email(client, prospect)
                if not email_content:
                    logger.warning(f"[EmailMarketer] Skipping prospect {prospect.prospect_id} ({prospect.company_name}) due to email generation failure.")
                    prospect_status_update = ("FAILED_GENERATION", None)
                    prospects_failed_count += 1
                    continue # Move to the next prospect

                # --- Send Email ---
                await send_email_via_smtp(email_content, prospect.contact_email, email_account)

                # --- Handle Success ---
                logger.info(f"[EmailMarketer] Successfully sent email to {prospect.contact_email} via {email_account.smtp_user} (From: {email_account.alias_email})")
                prospect_status_update = ("CONTACTED", datetime.datetime.now(datetime.timezone.utc))
                increment_send_count_for_account_id = email_account.account_id
                prospects_emailed_count += 1
                # Increment in-memory counts for immediate check in next loop iteration
                email_account.emails_sent_today += 1
                batch_send_counts[email_account.account_id] = batch_send_counts.get(email_account.account_id, 0) + 1


            except Exception as e:
                # --- Enhanced Error Handling (Keep this logic) ---
                error_type, reason = analyze_smtp_error(e)
                logger.error(f"[EmailMarketer] Error processing prospect {prospect.prospect_id} ({prospect.company_name}) with account {email_account.email_address if email_account else 'N/A'}. Type: {error_type}, Reason: {reason}", exc_info=(error_type == "UNKNOWN_ERROR"))

                prospects_failed_count += 1 # Count any failure as a prospect failure for this attempt

                if error_type == "RECIPIENT_ISSUE":
                    prospect_status_update = ("BOUNCED", datetime.datetime.now(datetime.timezone.utc)) # Mark prospect as bounced
                    if email_account: # Still count as a send attempt for the account
                        increment_send_count_for_account_id = email_account.account_id
                        email_account.emails_sent_today += 1
                        batch_send_counts[email_account.account_id] = batch_send_counts.get(email_account.account_id, 0) + 1
                elif error_type == "ACCOUNT_ISSUE":
                    if email_account:
                        account_to_deactivate = (email_account.account_id, reason)
                        accounts_deactivated_count += 1
                        # Remove account from current batch immediately to prevent reuse
                        if current_account_list_index != -1:
                            try:
                                logger.warning(f"Removing account {email_account.account_id} from active batch due to ACCOUNT_ISSUE.")
                                active_accounts.pop(current_account_list_index)
                                # Adjust index carefully after removal
                                account_index = account_index -1 if account_index >= current_account_list_index else account_index
                                if account_index < -1: account_index = -1 # Reset if needed
                            except IndexError:
                                logger.error("IndexError removing account from batch, list might be inconsistent.")
                                active_accounts = [acc for acc in active_accounts if acc.account_id != email_account.account_id] # Rebuild list safely
                                account_index = -1 # Reset index
                    # Prospect status remains NEW, will be retried later with a different account
                    prospect_status_update = None # Ensure no status update for the prospect
                elif error_type == "TRANSIENT_ERROR":
                    # Don't update prospect status, don't deactivate account. Let retry handle it or fail later.
                    prospect_status_update = None
                    # Don't count transient errors as sends or batch usage
                else: # UNKNOWN_ERROR or other unexpected issues
                    prospect_status_update = ("FAILED_SEND", None) # Mark prospect as failed send
                    if email_account: # Still count as attempt
                        increment_send_count_for_account_id = email_account.account_id
                        email_account.emails_sent_today += 1
                        batch_send_counts[email_account.account_id] = batch_send_counts.get(email_account.account_id, 0) + 1

            finally:
                # --- Database Updates (within prospect loop - Keep this logic) ---
                prospect_session: AsyncSession = None
                try:
                    # Use get_worker_session for isolated transaction per prospect
                    prospect_session = await get_worker_session()
                    if account_to_deactivate:
                        await crud.set_email_account_inactive(prospect_session, account_to_deactivate[0], account_to_deactivate[1])
                    if increment_send_count_for_account_id:
                        await crud.increment_email_sent_count(prospect_session, increment_send_count_for_account_id)
                    if prospect_status_update:
                        await crud.update_prospect_status(prospect_session, prospect.prospect_id, status=prospect_status_update[0], last_contacted_at=prospect_status_update[1])
                    await prospect_session.commit() # Commit changes for this prospect/account interaction
                except Exception as db_e:
                    logger.critical(f"[EmailMarketer] CRITICAL DB update failed for prospect {prospect.prospect_id}: {db_e}", exc_info=True)
                    if prospect_session: await prospect_session.rollback()
                finally:
                    if prospect_session: await prospect_session.close()

    except Exception as e:
        logger.error(f"[EmailMarketer] Unhandled error in email processing batch: {e}", exc_info=True)
        # Rollback potentially pending transaction from account/prospect fetching if error happened before loop
        if db: await db.rollback() # Rollback the main session if it was used for fetching
    finally:
        if client: await client.aclose()

    logger.info(f"[EmailMarketer] Finished processing batch. Processed: {prospects_processed_count}, Emailed: {prospects_emailed_count}, Failed: {prospects_failed_count}, Accounts Deactivated: {accounts_deactivated_count}.")