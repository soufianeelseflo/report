import datetime
import json
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, cast, Float, Integer as SQLInteger, text, update # Import update

# Corrected relative imports for package structure
from . import models
from Acumenis.app.api import schemas # Assuming schemas is here
from Acumenis.app.core.config import settings # Import settings for variant IDs
from Acumenis.app.core.security import encrypt_data, decrypt_data # Import encryption functions
from .models import KpiSnapshot, McolDecisionLog, ReportRequest, Prospect, EmailAccount, ApiKey, AgentTask # Import AgentTask model

# --- Report Request CRUD ---
# (create_report_request, create_initial_report_request, update_report_request_status remain largely the same, ensure statuses match model)
async def create_report_request(db: AsyncSession, request: schemas.ReportRequestCreate) -> models.ReportRequest:
    """DEPRECATED? Creates a report request directly via API, bypassing payment initially."""
    print("[DEPRECATED PATH?] Creating report request directly via API.")
    db_request = models.ReportRequest(
        client_name=request.client_name,
        client_email=request.client_email,
        company_name=request.company_name,
        report_type=request.report_type,
        request_details=request.request_details,
        status="AWAITING_PAYMENT", # Requires payment step
        payment_status="unpaid"
    )
    db.add(db_request)
    await db.flush()
    await db.refresh(db_request)
    return db_request

async def create_initial_report_request(
    db: AsyncSession,
    order_data: dict # Parsed data from Lemon Squeezy webhook
) -> Optional[models.ReportRequest]:
    """
    Creates a report request after successful payment via webhook with status AWAITING_GENERATION.
    This is the first step before an AgentTask is created.
    """
    ls_order_id = order_data.get('id')
    if not ls_order_id:
        print("[Webhook] Error: Missing order ID in webhook data.")
        return None

    # Check if order already processed
    existing = await db.scalar(select(models.ReportRequest).where(models.ReportRequest.lemonsqueezy_order_id == str(ls_order_id)))
    if existing:
        print(f"[Webhook] Order ID {ls_order_id} already processed for ReportRequest ID {existing.request_id}. Skipping.")
        return None # Indicate duplicate or already processed

    attributes = order_data.get('attributes', {})
    customer_email = attributes.get('user_email')
    customer_name = attributes.get('user_name')
    variant_id = attributes.get('first_order_item', {}).get('variant_id')
    order_total_cents = attributes.get('total', 0)

    custom_data = attributes.get('custom_data', {})
    request_details = custom_data.get('research_topic', 'Not Provided - Check Order Notes')
    company_name_custom = custom_data.get('company_name')

    if not customer_email:
        print(f"[Webhook] Error: Missing customer email for order {ls_order_id}.")
        return None

    # Determine report type based on variant ID
    report_type_code = 'unknown_paid'
    if str(variant_id) == settings.LEMONSQUEEZY_VARIANT_STANDARD:
        report_type_code = 'standard_499'
    elif str(variant_id) == settings.LEMONSQUEEZY_VARIANT_PREMIUM:
        report_type_code = 'premium_999'
    else:
        print(f"[Webhook] Warning: Order {ls_order_id} variant ID {variant_id} doesn't match configured standard/premium IDs.")

    db_request = models.ReportRequest(
        client_name=customer_name,
        client_email=customer_email,
        company_name=company_name_custom,
        report_type=report_type_code,
        request_details=request_details,
        status="AWAITING_GENERATION", # Initial status after payment
        payment_status="paid",
        lemonsqueezy_order_id=str(ls_order_id),
        order_total_cents=order_total_cents
    )
    db.add(db_request)
    await db.flush() # Flush to get the ID assigned
    await db.refresh(db_request)
    print(f"[Webhook] Created initial ReportRequest ID {db_request.request_id} from LS Order {ls_order_id} with status AWAITING_GENERATION.")
    return db_request

async def get_report_request(db: AsyncSession, request_id: int) -> Optional[models.ReportRequest]:
     """Retrieves a specific report request by ID."""
     result = await db.execute(select(models.ReportRequest).where(models.ReportRequest.request_id == request_id))
     return result.scalars().first()

async def update_report_request_status(db: AsyncSession, request_id: int, status: str, output_path: Optional[str] = None, error_message: Optional[str] = None, payment_status: Optional[str] = None, checkout_id: Optional[str] = None) -> Optional[models.ReportRequest]:
    """Updates the status and potentially other fields of a report request."""
    db_request = await db.get(models.ReportRequest, request_id) # Use get for primary key lookup
    if db_request:
        if status: db_request.status = status
        if output_path is not None: db_request.report_output_path = output_path
        if error_message is not None: db_request.error_message = error_message
        if payment_status: db_request.payment_status = payment_status
        if checkout_id: db_request.lemonsqueezy_checkout_id = checkout_id
        # updated_at is handled automatically by onupdate
        await db.flush()
        await db.refresh(db_request)
    return db_request

# --- Prospect CRUD ---
async def create_prospect(db: AsyncSession, company_name: str, email: Optional[str] = None, website: Optional[str] = None, pain_point: Optional[str] = None, source: Optional[str] = None, linkedin_url: Optional[str] = None, executives: Optional[Dict] = None) -> Optional[models.Prospect]:
    """Creates or updates a prospect. Tries email first, then company name for uniqueness check."""
    existing = None
    if email:
        # Prioritize checking by email if provided
        existing = await db.scalar(select(models.Prospect).where(models.Prospect.contact_email == email))

    if not existing and company_name:
         # If no email match or no email provided, check by company name (less reliable for updates)
         existing = await db.scalar(select(models.Prospect).where(models.Prospect.company_name == company_name))

    if existing:
        # Update existing prospect - prioritize non-null new data
        print(f"Updating existing prospect: {existing.contact_email or existing.company_name}")
        existing.company_name = company_name or existing.company_name # Update company name if changed
        existing.website = website or existing.website
        existing.contact_email = email or existing.contact_email # Update email if provided
        existing.potential_pain_point = pain_point or existing.potential_pain_point
        existing.source = source or existing.source # Overwrite source? Or append? Overwrite for now.
        existing.linkedin_profile_url = linkedin_url or existing.linkedin_profile_url
        existing.key_executives = executives if executives else existing.key_executives # Replace for now
        # Don't reset status if already contacted/bounced etc. Only update if NEW?
        # if existing.status == 'NEW': existing.status = 'NEW' # No change needed
        await db.flush()
        await db.refresh(existing)
        return existing
    else:
        # Create new prospect
        db_prospect = models.Prospect(
            company_name=company_name, website=website, contact_email=email,
            potential_pain_point=pain_point, source=source, status="NEW",
            linkedin_profile_url=linkedin_url, key_executives=executives
        )
        db.add(db_prospect)
        await db.flush()
        await db.refresh(db_prospect)
        return db_prospect

async def get_new_prospects_for_emailing(db: AsyncSession, limit: int) -> list[models.Prospect]:
    """Gets NEW prospects ready for emailing, prioritizing those with emails."""
    result = await db.execute(
        select(models.Prospect)
        .where(models.Prospect.status == 'NEW')
        .where(models.Prospect.contact_email != None) # Ensure email exists (even if guessed)
        .order_by(models.Prospect.created_at.asc()) # Process oldest first
        .limit(limit)
        .with_for_update(skip_locked=True) # Lock rows for processing
    )
    return result.scalars().all()

async def update_prospect_status(db: AsyncSession, prospect_id: int, status: str, last_contacted_at: Optional[datetime.datetime] = None) -> Optional[models.Prospect]:
    """Updates the status of a prospect."""
    db_prospect = await db.get(models.Prospect, prospect_id)
    if db_prospect:
        db_prospect.status = status
        if last_contacted_at:
            db_prospect.last_contacted_at = last_contacted_at
        # updated_at handled automatically
        await db.flush()
        await db.refresh(db_prospect)
    return db_prospect

# --- Email Account CRUD ---
async def get_active_email_account_for_sending(db: AsyncSession) -> Optional[models.EmailAccount]:
    """Finds an active email account under its daily limit, prioritizing those with aliases."""
    today = datetime.date.today()
    result = await db.execute(
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.alias_email != None) # *** MUST have an alias ***
        .where(
            (models.EmailAccount.last_reset_date < today) |
            (models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        )
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst()) # Use least recently used
        .limit(1)
        .with_for_update(skip_locked=True) # Lock the selected account
    )
    account = result.scalars().first()

    if account and account.last_reset_date < today:
        # Reset count if it's a new day
        account.emails_sent_today = 0
        account.last_reset_date = today
        await db.flush()
        await db.refresh(account)

    # Final check after potential reset
    if account and account.emails_sent_today < account.daily_limit:
        return account
    else:
        # If no account found or limit reached, return None
        return None

async def increment_email_sent_count(db: AsyncSession, account_id: int) -> None:
    """Increments the sent count and updates last used time for an email account."""
    # Use SQLAlchemy update for potential race condition safety
    stmt = (
        update(models.EmailAccount)
        .where(models.EmailAccount.account_id == account_id)
        .values(
            emails_sent_today=models.EmailAccount.emails_sent_today + 1,
            last_used_at=datetime.datetime.now(datetime.timezone.utc)
        )
        .execution_options(synchronize_session="fetch") # Ensure session is aware of changes
    )
    await db.execute(stmt)
    # No commit here, handled by caller

async def set_email_account_inactive(db: AsyncSession, account_id: int, reason: str) -> None:
    """Marks an email account as inactive."""
    stmt = (
        update(models.EmailAccount)
        .where(models.EmailAccount.account_id == account_id)
        .values(
            is_active=False,
            notes=f"Deactivated on {datetime.datetime.now(datetime.timezone.utc)}: {reason}",
            updated_at=datetime.datetime.now(datetime.timezone.utc) # Explicitly set updated_at
        )
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(stmt)
    # No commit here

async def get_batch_of_active_accounts(db: AsyncSession, limit: int) -> List[models.EmailAccount]:
    """
    Finds a batch of active email accounts under their daily limit,
    ordered by least recently used. Handles daily reset logic. Prioritizes accounts with aliases.
    """
    today = datetime.date.today()
    # Step 1: Reset counts for accounts needing it
    reset_stmt = (
        update(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.last_reset_date < today)
        .values(emails_sent_today=0, last_reset_date=today)
        .execution_options(synchronize_session=False) # Don't need results back
    )
    await db.execute(reset_stmt)
    # print(f"[EmailAccountCRUD] Reset daily counts check complete.") # Reduce log noise

    # Step 2: Fetch usable accounts (active, alias set, under limit after potential reset)
    result = await db.execute(
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.alias_email != None) # *** Ensure alias is set ***
        .where(models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst())
        .limit(limit)
        .with_for_update(skip_locked=True) # Lock selected rows for update
    )
    accounts = result.scalars().all()
    return accounts

# --- MCOL CRUD ---
async def create_kpi_snapshot(db: AsyncSession) -> KpiSnapshot:
    """Calculates current KPIs and saves a snapshot."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(days=1)

    # Report Metrics
    report_counts = await db.execute(
        select(
            func.count().filter(models.ReportRequest.status == 'AWAITING_PAYMENT').label('awaiting_payment'),
            func.count().filter(models.ReportRequest.status == 'AWAITING_GENERATION').label('awaiting_generation'), # Added
            func.count().filter(models.ReportRequest.status == 'PENDING').label('pending'),
            func.count().filter(models.ReportRequest.status == 'PROCESSING').label('processing'),
            func.count().filter(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago).label('completed_24h'),
            func.count().filter(models.ReportRequest.status == 'FAILED', models.ReportRequest.updated_at >= one_day_ago).label('failed_24h'),
            func.count().filter(models.ReportRequest.status == 'DELIVERY_FAILED', models.ReportRequest.updated_at >= one_day_ago).label('delivery_failed_24h'), # Added
            func.count().filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago).label('orders_created_24h'),
            func.sum(
                 models.ReportRequest.order_total_cents / 100.0 # Convert cents to dollars
            ).filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago).label('revenue_24h')
        )
    )
    report_kpis = report_counts.first()

    avg_time_result = await db.execute(select(func.avg(func.extract('epoch', models.ReportRequest.updated_at - models.ReportRequest.created_at))).where(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago))
    avg_report_time_seconds = avg_time_result.scalar_one_or_none()

    # Prospecting Metrics
    new_prospects_24h = await db.scalar(select(func.count(models.Prospect.prospect_id)).where(models.Prospect.created_at >= one_day_ago))

    # Email Metrics
    email_counts = await db.execute(
        select(
            func.count().filter(models.Prospect.status == 'CONTACTED', models.Prospect.last_contacted_at >= one_day_ago).label('sent_24h'),
            func.count().filter(models.Prospect.status == 'BOUNCED', models.Prospect.last_contacted_at >= one_day_ago).label('bounced_24h'),
            func.count().filter(models.Prospect.last_contacted_at >= one_day_ago, models.Prospect.status.in_(['CONTACTED', 'BOUNCED'])).label('contacted_total_24h')
        )
    )
    email_kpis = email_counts.first()

    account_counts = await db.execute(
        select(
            func.count().filter(models.EmailAccount.is_active == True).label('active'),
            func.count().filter(models.EmailAccount.is_active == False, models.EmailAccount.updated_at >= one_day_ago).label('deactivated_24h')
        )
    )
    account_kpis = account_counts.first()

    bounce_rate_24h = (float(email_kpis.bounced_24h) / float(email_kpis.contacted_total_24h) * 100.0) if email_kpis.contacted_total_24h and email_kpis.contacted_total_24h > 0 else 0.0

    # API Key Metrics
    api_key_counts = await db.execute(
        select(
            func.count().filter(models.ApiKey.status == 'active').label('active_keys'),
            func.count().filter(models.ApiKey.status != 'active', models.ApiKey.updated_at >= one_day_ago).label('deactivated_keys_24h') # Count any non-active status updated recently
        )
    )
    api_key_kpis = api_key_counts.first()

    snapshot = KpiSnapshot(
        awaiting_payment_reports=report_kpis.awaiting_payment or 0,
        awaiting_generation_reports=report_kpis.awaiting_generation or 0, # Added
        pending_reports=report_kpis.pending or 0,
        processing_reports=report_kpis.processing or 0,
        completed_reports_24h=report_kpis.completed_24h or 0,
        failed_reports_24h=report_kpis.failed_24h or 0,
        delivery_failed_reports_24h=report_kpis.delivery_failed_24h or 0, # Added
        avg_report_time_seconds=avg_report_time_seconds,
        new_prospects_24h=new_prospects_24h or 0,
        emails_sent_24h=email_kpis.sent_24h or 0,
        active_email_accounts=account_kpis.active or 0,
        deactivated_accounts_24h=account_kpis.deactivated_24h or 0,
        bounce_rate_24h=bounce_rate_24h,
        revenue_24h=report_kpis.revenue_24h or 0.0,
        orders_created_24h=report_kpis.orders_created_24h or 0,
        active_api_keys=api_key_kpis.active_keys or 0, # Added
        deactivated_api_keys_24h=api_key_kpis.deactivated_keys_24h or 0 # Added
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    return snapshot

# (log_mcol_decision, update_mcol_decision_log, get_latest_kpi_snapshot remain the same)
async def log_mcol_decision(db: AsyncSession, kpi_snapshot_id: Optional[int] = None, **kwargs) -> McolDecisionLog:
    """Logs a decision made by the MCOL."""
    if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], str):
        kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])
    if 'action_parameters' in kwargs and not isinstance(kwargs['action_parameters'], (str, type(None))):
         kwargs['action_parameters'] = json.dumps(kwargs['action_parameters']) # Ensure params are JSON serializable

    decision = McolDecisionLog(kpi_snapshot_id=kpi_snapshot_id, **kwargs)
    db.add(decision)
    await db.flush()
    await db.refresh(decision)
    return decision

async def update_mcol_decision_log(db: AsyncSession, log_id: int, **kwargs) -> Optional[McolDecisionLog]:
    """Updates an existing MCOL decision log entry."""
    log_entry = await db.get(McolDecisionLog, log_id)
    if log_entry:
        if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], str):
            kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])
        if 'action_parameters' in kwargs and not isinstance(kwargs['action_parameters'], (str, type(None))):
             kwargs['action_parameters'] = json.dumps(kwargs['action_parameters'])

        for key, value in kwargs.items():
            setattr(log_entry, key, value)
        # updated_at handled automatically
        await db.flush()
        await db.refresh(log_entry)
    return log_entry

async def get_latest_kpi_snapshot(db: AsyncSession) -> Optional[KpiSnapshot]:
    """Retrieves the most recent KPI snapshot."""
    result = await db.execute(select(KpiSnapshot).order_by(KpiSnapshot.timestamp.desc()).limit(1))
    return result.scalars().first()

# --- API Key CRUD ---

async def add_api_key(db: AsyncSession, key: str, provider: str, email: Optional[str] = None, proxy: Optional[str] = None, notes: Optional[str] = None) -> Optional[models.ApiKey]:
    """Adds a new API key, storing it encrypted. Returns None if encryption fails or key exists."""
    try:
        encrypted_key = encrypt_data(key)
        if not encrypted_key:
            print(f"Error: Failed to encrypt API key for provider {provider}")
            return None

        # Check if the encrypted key already exists
        existing = await db.scalar(select(models.ApiKey).where(models.ApiKey.api_key_encrypted == encrypted_key))
        if existing:
            print(f"API Key already exists (ID: {existing.id}, Status: {existing.status}). Skipping add.")
            # Optionally reactivate if inactive? For now, just skip.
            # if existing.status != 'active':
            #     existing.status = 'active'
            #     existing.notes = f"Re-added/Activated on {datetime.datetime.now(datetime.timezone.utc)}. {notes or ''}".strip()
            #     await db.flush()
            #     await db.refresh(existing)
            #     return existing
            return None # Don't add duplicate

        db_key = models.ApiKey(
            api_key_encrypted=encrypted_key,
            provider=provider,
            email_used=email,
            proxy_used=proxy,
            status="active",
            notes=notes
        )
        db.add(db_key)
        await db.flush()
        await db.refresh(db_key)
        print(f"Added new API Key ID {db_key.id} for provider {provider}")
        return db_key
    except Exception as e:
         print(f"Error adding API key: {e}")
         await db.rollback() # Rollback on error
         return None


async def get_active_api_keys(db: AsyncSession, provider: str) -> List[str]:
    """Retrieves all active, decrypted API keys for a given provider."""
    result = await db.execute(
        select(models.ApiKey.api_key_encrypted, models.ApiKey.id) # Fetch ID for logging
        .where(models.ApiKey.provider == provider)
        .where(models.ApiKey.status == 'active')
    )
    encrypted_keys_with_ids = result.all()

    decrypted_keys = []
    for enc_key, key_id in encrypted_keys_with_ids:
        try:
            dec_key = decrypt_data(enc_key)
            if dec_key: # Check if decryption was successful
                decrypted_keys.append(dec_key)
            else:
                print(f"Warning: Failed to decrypt API key ID {key_id} for provider {provider}.")
                # Consider marking key as 'error' status here?
                # await set_api_key_status_by_id(db, key_id, 'error', 'Decryption failed')
        except Exception as decrypt_err:
             print(f"Warning: Exception during decryption for API key ID {key_id}: {decrypt_err}")
             # await set_api_key_status_by_id(db, key_id, 'error', f'Decryption exception: {decrypt_err}')

    return decrypted_keys

async def set_api_key_status_by_id(db: AsyncSession, key_id: int, status: str, reason: Optional[str] = None) -> bool:
    """Updates the status of an API key identified by its database ID."""
    db_key = await db.get(models.ApiKey, key_id)
    if db_key:
        db_key.status = status
        if reason:
            db_key.notes = f"[{status.upper()} at {datetime.datetime.now(datetime.timezone.utc)}] {reason}\n{db_key.notes or ''}".strip()
        # updated_at handled automatically
        await db.flush()
        print(f"Updated status for API Key ID {key_id} to '{status}'")
        return True
    else:
        print(f"API Key ID {key_id} not found for status update.")
        return False

async def set_api_key_status_by_value(db: AsyncSession, key_value: str, provider: str, status: str, reason: Optional[str] = None) -> bool:
    """
    Finds an API key by its DECRYPTED value and provider, then updates its status using its ID.
    Less efficient than updating by ID directly.
    """
    # This requires iterating through keys, decrypting, and then updating by ID.
    # It's better if the calling function can get the ID or the encrypted value.
    # For now, implement the search logic:
    result = await db.execute(
        select(models.ApiKey)
        .where(models.ApiKey.provider == provider)
    )
    potential_keys = result.scalars().all()

    target_key_id = None
    for db_key in potential_keys:
        try:
            decrypted_key = decrypt_data(db_key.api_key_encrypted)
            if decrypted_key == key_value:
                target_key_id = db_key.id
                break # Found the key
        except Exception:
             continue # Skip keys that fail decryption

    if target_key_id:
        return await set_api_key_status_by_id(db, target_key_id, status, reason)
    else:
        print(f"API Key matching '{key_value[:5]}...' for provider '{provider}' not found for status update.")
        return False

async def set_api_key_inactive(db: AsyncSession, key_to_deactivate: str, provider: str, reason: str) -> bool:
     """Convenience function to mark a key as inactive by its value."""
     return await set_api_key_status_by_value(db, key_value=key_to_deactivate, provider=provider, status='inactive', reason=reason)

async def mark_api_key_used(db: AsyncSession, key_value: str, provider: str) -> bool:
    """Updates the last_used_at timestamp for a given API key value."""
    # Similar inefficient search as set_api_key_status_by_value
    result = await db.execute(
        select(models.ApiKey)
        .where(models.ApiKey.provider == provider)
        .where(models.ApiKey.status == 'active') # Only mark active keys as used
    )
    active_keys = result.scalars().all()

    target_key_id = None
    for db_key in active_keys:
        try:
            decrypted_key = decrypt_data(db_key.api_key_encrypted)
            if decrypted_key == key_value:
                target_key_id = db_key.id
                break
        except Exception:
            continue

    if target_key_id:
        db_key_to_update = await db.get(models.ApiKey, target_key_id)
        if db_key_to_update:
            db_key_to_update.last_used_at = datetime.datetime.now(datetime.timezone.utc)
            await db.flush()
            return True
        else:
            print(f"Error: Found key ID {target_key_id} but failed to retrieve for last_used_at update.")
            return False
    else:
        # print(f"Warning: Active API Key matching '{key_value[:5]}...' not found for last_used_at update.") # Reduce noise
        return False

# --- Agent Task CRUD ---
# (create_agent_task, get_pending_tasks_for_agent, update_task_status, get_task remain the same)
async def create_agent_task(
    db: AsyncSession,
    agent_name: str,
    goal: str,
    parameters: Optional[Dict] = None,
    priority: int = 0,
    depends_on_task_id: Optional[int] = None
) -> models.AgentTask:
    """Creates a new agent task."""
    db_task = models.AgentTask(
        agent_name=agent_name,
        goal=goal,
        parameters=parameters,
        priority=priority,
        status='PENDING',
        depends_on_task_id=depends_on_task_id
    )
    db.add(db_task)
    await db.flush()
    await db.refresh(db_task)
    print(f"Created AgentTask ID {db_task.task_id} for agent {agent_name} with goal: {goal}")
    return db_task

async def get_pending_tasks_for_agent(
    db: AsyncSession,
    agent_name: str,
    limit: int = 1 # Process one task at a time per agent cycle usually
) -> List[models.AgentTask]:
    """
    Fetches pending tasks for a specific agent, ordered by priority (desc)
    and then creation time (asc). Uses FOR UPDATE SKIP LOCKED.
    """
    # Basic implementation without dependency check for now
    result = await db.execute(
        select(models.AgentTask)
        .where(models.AgentTask.agent_name == agent_name)
        .where(models.AgentTask.status == 'PENDING')
        .order_by(models.AgentTask.priority.desc(), models.AgentTask.created_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    tasks = result.scalars().all()
    return tasks

async def update_task_status(
    db: AsyncSession,
    task_id: int,
    status: str,
    result: Optional[str] = None
) -> Optional[models.AgentTask]:
    """Updates a task's status and optional result message."""
    db_task = await db.get(models.AgentTask, task_id) # Use db.get for primary key lookup
    if db_task:
        db_task.status = status
        if result is not None: # Allow clearing the result
            db_task.result = result
        # updated_at is handled automatically by onupdate
        await db.flush()
        await db.refresh(db_task)
        print(f"Updated AgentTask ID {task_id} to status {status}")
    else:
        print(f"Warning: AgentTask ID {task_id} not found for status update.")
    return db_task

async def get_task(db: AsyncSession, task_id: int) -> Optional[models.AgentTask]:
     """Retrieves a specific task by ID."""
     result = await db.execute(select(models.AgentTask).where(models.AgentTask.task_id == task_id))
     return result.scalars().first()