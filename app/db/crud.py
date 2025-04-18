# autonomous_agency/app/db/crud.py

import json
import logging # Use standard logging
from typing import Optional, List, Dict, Any, Union
import datetime # Ensure datetime is imported

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, cast, Float, Integer as SQLInteger, text, update, delete, Text as SQLText

# Corrected relative imports for package structure
try:
    from . import models
    # Ensure schemas are imported correctly if needed, adjust path if necessary
    # from Nexus Plan.app.api import schemas
    from Nexus Plan.app.core.config import settings # Import settings for variant IDs, etc.
    from Nexus Plan.app.core.security import encrypt_data, decrypt_data # Import encryption functions
    from .models import ( # Import all relevant models
        KpiSnapshot, McolDecisionLog, ReportRequest, Prospect, EmailAccount, ApiKey, AgentTask
    )
except ImportError:
    print("[CRUD] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.db import models
    from app.core.config import settings
    from app.core.security import encrypt_data, decrypt_data
    from app.db.models import (
        KpiSnapshot, McolDecisionLog, ReportRequest, Prospect, EmailAccount, ApiKey, AgentTask
    )


# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Adjust level as needed

# --- Report Request CRUD ---

async def create_initial_report_request(
    db: AsyncSession,
    order_data: dict # Parsed data from Lemon Squeezy webhook
) -> Optional[models.ReportRequest]:
    """
    Creates a report request after successful payment via webhook with status AWAITING_GENERATION.
    Checks for duplicates based on the Lemon Squeezy Order ID.
    """
    ls_order_id = order_data.get('id')
    if not ls_order_id:
        logger.error("[Webhook] Missing order ID in webhook data.")
        return None

    # Check if order already processed
    existing = await db.scalar(select(models.ReportRequest).where(models.ReportRequest.lemonsqueezy_order_id == str(ls_order_id)))
    if existing:
        logger.info(f"[Webhook] Order ID {ls_order_id} already processed for ReportRequest ID {existing.request_id}. Skipping.")
        return None # Indicate duplicate

    attributes = order_data.get('attributes', {})
    customer_email = attributes.get('user_email')
    customer_name = attributes.get('user_name')
    variant_id = attributes.get('first_order_item', {}).get('variant_id')
    order_total_cents = attributes.get('total', 0)

    custom_data = attributes.get('custom_data', {})
    request_details = custom_data.get('research_topic', 'Not Provided - Check Order Notes')
    company_name_custom = custom_data.get('company_name')

    if not customer_email:
        logger.error(f"[Webhook] Missing customer email for order {ls_order_id}.")
        return None

    # Determine report type based on variant ID
    report_type_code = 'unknown_paid'
    if str(variant_id) == settings.LEMONSQUEEZY_VARIANT_STANDARD:
        report_type_code = 'standard_499'
    elif str(variant_id) == settings.LEMONSQUEEZY_VARIANT_PREMIUM:
        report_type_code = 'premium_999'
    else:
        logger.warning(f"[Webhook] Order {ls_order_id} variant ID {variant_id} doesn't match configured standard/premium IDs.")

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
    logger.info(f"[Webhook] Created initial ReportRequest ID {db_request.request_id} from LS Order {ls_order_id} with status AWAITING_GENERATION.")
    return db_request

async def get_report_request(db: AsyncSession, request_id: int) -> Optional[models.ReportRequest]:
     """Retrieves a specific report request by ID."""
     result = await db.execute(select(models.ReportRequest).where(models.ReportRequest.request_id == request_id))
     return result.scalars().first()

# --- MODIFICATION: Add get_pending_report_request ---
async def get_pending_report_request(db: AsyncSession) -> Optional[models.ReportRequest]:
    """
    Gets the next report request with status 'AWAITING_GENERATION'
    and locks it for processing.
    """
    stmt = (
        select(models.ReportRequest)
        .where(models.ReportRequest.status == 'AWAITING_GENERATION')
        .order_by(models.ReportRequest.created_at.asc()) # Process oldest first
        .limit(1)
        .with_for_update(skip_locked=True) # Lock the row
    )
    result = await db.execute(stmt)
    request = result.scalars().first()
    if request:
        logger.info(f"Fetched pending ReportRequest ID {request.request_id} for processing.")
    return request
# --- END MODIFICATION ---


async def update_report_request_status(
    db: AsyncSession,
    request_id: int,
    status: str,
    output_path: Optional[str] = None,
    error_message: Optional[str] = None
) -> Optional[models.ReportRequest]:
    """Updates the status and potentially other fields of a report request."""
    stmt = (
        update(models.ReportRequest)
        .where(models.ReportRequest.request_id == request_id)
        .values(
            status=status,
            report_output_path=output_path if output_path is not None else models.ReportRequest.report_output_path, # Keep old if None
            error_message=error_message if error_message is not None else models.ReportRequest.error_message, # Keep old if None
            updated_at=func.now() # Explicitly set updated_at
        )
        .execution_options(synchronize_session="fetch")
        .returning(models.ReportRequest)
    )
    result = await db.execute(stmt)
    updated_request = result.scalar_one_or_none()
    if updated_request:
        logger.info(f"Updated ReportRequest ID {request_id} status to {status}.")
    else:
        logger.warning(f"ReportRequest ID {request_id} not found for status update.")
    return updated_request


# --- Prospect CRUD ---
# --- MODIFICATION: Replace create_prospect with create_or_update_prospect ---
async def create_or_update_prospect(
    db: AsyncSession,
    company_name: str,
    email: Optional[str] = None,
    website: Optional[str] = None,
    pain_point: Optional[str] = None,
    source: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    executives: Optional[Dict] = None,
    status_if_new: str = "NEW" # Status to set only if creating a new prospect
) -> Optional[models.Prospect]:
    """
    Creates a new prospect or updates an existing one based on company name.
    Prioritizes updating existing records over creating duplicates.
    Only sets status if creating a new record.
    """
    # Try to find existing prospect by company name first
    existing = await db.scalar(select(models.Prospect).where(models.Prospect.company_name == company_name))

    if existing:
        # Update existing prospect if new information is provided
        update_data = {}
        if website and website != existing.website: update_data['website'] = website
        if email and email != existing.contact_email: update_data['contact_email'] = email
        if pain_point and pain_point != existing.potential_pain_point: update_data['potential_pain_point'] = pain_point
        if source and source != existing.source: update_data['source'] = source
        if linkedin_url and linkedin_url != existing.linkedin_profile_url: update_data['linkedin_profile_url'] = linkedin_url
        if executives and executives != existing.key_executives: update_data['key_executives'] = executives

        # Only update if there's new data
        if update_data:
            update_data['updated_at'] = func.now()
            stmt = (
                update(models.Prospect)
                .where(models.Prospect.prospect_id == existing.prospect_id)
                .values(**update_data)
                .execution_options(synchronize_session="fetch")
                .returning(models.Prospect)
            )
            result = await db.execute(stmt)
            updated_prospect = result.scalar_one_or_none()
            logger.info(f"Updated existing prospect: {updated_prospect.company_name} (ID: {updated_prospect.prospect_id})")
            return updated_prospect
        else:
            logger.info(f"No new information to update for existing prospect: {company_name} (ID: {existing.prospect_id})")
            return existing # Return existing if no updates needed
    else:
        # Create new prospect
        db_prospect = models.Prospect(
            company_name=company_name,
            website=website,
            contact_email=email,
            potential_pain_point=pain_point,
            source=source,
            status=status_if_new, # Use the provided status only when creating
            linkedin_profile_url=linkedin_url,
            key_executives=executives
        )
        db.add(db_prospect)
        await db.flush()
        await db.refresh(db_prospect)
        logger.info(f"Created new prospect: {db_prospect.company_name} (ID: {db_prospect.prospect_id}) with status {status_if_new}")
        return db_prospect
# --- END MODIFICATION ---


async def get_new_prospects_for_emailing(db: AsyncSession, limit: int) -> list[models.Prospect]:
    """Gets NEW prospects ready for emailing, prioritizing those with emails and locking them."""
    stmt = (
        select(models.Prospect)
        .where(models.Prospect.status == 'NEW')
        .where(models.Prospect.contact_email.isnot(None)) # Ensure email exists
        .order_by(models.Prospect.created_at.asc()) # Process oldest first
        .limit(limit)
        .with_for_update(skip_locked=True) # Lock rows for processing
    )
    result = await db.execute(stmt)
    prospects = result.scalars().all()
    if prospects:
        logger.info(f"Fetched {len(prospects)} new prospects for emailing.")
    return prospects

async def update_prospect_status(
    db: AsyncSession,
    prospect_id: int,
    status: str,
    last_contacted_at: Optional[datetime.datetime] = None
) -> Optional[models.Prospect]:
    """Updates the status of a prospect."""
    update_values = {'status': status, 'updated_at': func.now()}
    if last_contacted_at:
        update_values['last_contacted_at'] = last_contacted_at

    stmt = (
        update(models.Prospect)
        .where(models.Prospect.prospect_id == prospect_id)
        .values(**update_values)
        .execution_options(synchronize_session="fetch")
        .returning(models.Prospect)
    )
    result = await db.execute(stmt)
    updated_prospect = result.scalar_one_or_none()
    if updated_prospect:
        logger.info(f"Updated Prospect ID {prospect_id} status to {status}.")
    else:
        logger.warning(f"Prospect ID {prospect_id} not found for status update.")
    return updated_prospect

# --- Email Account CRUD ---
async def get_active_email_account_for_sending(db: AsyncSession) -> Optional[models.EmailAccount]:
    """
    Finds an active email account under its daily limit, prioritizing those with aliases,
    and handles daily reset logic. Uses FOR UPDATE SKIP LOCKED.
    """
    today = datetime.date.today()
    # Use CTE or separate queries for reset for better locking scope
    reset_candidates = await db.scalars(
        select(models.EmailAccount.account_id)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.last_reset_date < today)
        .with_for_update(skip_locked=True) # Lock only accounts needing reset
    )
    reset_ids = reset_candidates.all()

    if reset_ids:
        reset_stmt = (
            update(models.EmailAccount)
            .where(models.EmailAccount.account_id.in_(reset_ids))
            .values(emails_sent_today=0, last_reset_date=today)
            .execution_options(synchronize_session=False)
        )
        await db.execute(reset_stmt)
        logger.info(f"Reset daily counts for {len(reset_ids)} email accounts.")
        # Commit the reset separately? Or rely on the main transaction? Rely on main for now.

    # Now fetch a usable account
    stmt = (
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.alias_email.isnot(None)) # *** MUST have an alias ***
        .where(models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst()) # Use least recently used
        .limit(1)
        .with_for_update(skip_locked=True) # Lock the selected account
    )
    result = await db.execute(stmt)
    account = result.scalars().first()

    if account:
        logger.info(f"Selected email account {account.email_address} (ID: {account.account_id}) for sending.")
    else:
        logger.info("No suitable active email account found for sending.")

    return account

async def increment_email_sent_count(db: AsyncSession, account_id: int) -> None:
    """Increments the sent count and updates last used time for an email account."""
    stmt = (
        update(models.EmailAccount)
        .where(models.EmailAccount.account_id == account_id)
        .values(
            emails_sent_today=models.EmailAccount.emails_sent_today + 1,
            last_used_at=func.now()
        )
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(stmt)
    # No commit here, handled by caller

async def set_email_account_inactive(db: AsyncSession, account_id: int, reason: str) -> None:
    """Marks an email account as inactive."""
    now = datetime.datetime.now(datetime.timezone.utc)
    stmt = (
        update(models.EmailAccount)
        .where(models.EmailAccount.account_id == account_id)
        .values(
            is_active=False,
            notes=f"Deactivated on {now.strftime('%Y-%m-%d %H:%M:%S UTC')}: {reason[:250]}", # Truncate reason
            updated_at=now # Explicitly set updated_at
        )
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(stmt)
    logger.warning(f"Deactivated Email Account ID {account_id}. Reason: {reason}")
    # No commit here

async def get_batch_of_active_accounts(db: AsyncSession, limit: int) -> List[models.EmailAccount]:
    """
    Finds a batch of active email accounts under their daily limit,
    ordered by least recently used. Handles daily reset logic. Prioritizes accounts with aliases.
    Uses FOR UPDATE SKIP LOCKED.
    """
    today = datetime.date.today()
    # Step 1: Reset counts for accounts needing it (lock only these)
    reset_candidates = await db.scalars(
        select(models.EmailAccount.account_id)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.last_reset_date < today)
        .with_for_update(skip_locked=True)
    )
    reset_ids = reset_candidates.all()

    if reset_ids:
        reset_stmt = (
            update(models.EmailAccount)
            .where(models.EmailAccount.account_id.in_(reset_ids))
            .values(emails_sent_today=0, last_reset_date=today)
            .execution_options(synchronize_session=False)
        )
        await db.execute(reset_stmt)
        # logger.info(f"[EmailAccountCRUD] Reset daily counts for {len(reset_ids)} accounts.")

    # Step 2: Fetch usable accounts (active, alias set, under limit after potential reset)
    stmt = (
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(models.EmailAccount.alias_email.isnot(None)) # *** Ensure alias is set ***
        .where(models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst())
        .limit(limit)
        .with_for_update(skip_locked=True) # Lock selected rows for update
    )
    result = await db.execute(stmt)
    accounts = result.scalars().all()
    return accounts

# --- MCOL CRUD ---
async def create_kpi_snapshot(db: AsyncSession) -> KpiSnapshot:
    """Calculates current KPIs and saves a snapshot."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(days=1)

    # Use explicit labels and coalesce for safety
    report_counts_stmt = select(
        func.count().filter(models.ReportRequest.status == 'AWAITING_GENERATION').label('awaiting_generation'),
        func.count().filter(models.ReportRequest.status == 'PENDING_TASK').label('pending_task'), # Use new status
        func.count().filter(models.ReportRequest.status == 'PROCESSING').label('processing'),
        func.count().filter(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago).label('completed_24h'),
        func.count().filter(models.ReportRequest.status == 'FAILED', models.ReportRequest.updated_at >= one_day_ago).label('failed_24h'),
        func.count().filter(models.ReportRequest.status == 'DELIVERY_FAILED', models.ReportRequest.updated_at >= one_day_ago).label('delivery_failed_24h'),
        func.count().filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago).label('orders_created_24h'),
        func.coalesce(func.sum(
             models.ReportRequest.order_total_cents / 100.0 # Convert cents to dollars
        ).filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago), 0.0).label('revenue_24h')
    )
    report_kpis = (await db.execute(report_counts_stmt)).first()

    avg_time_stmt = select(func.avg(func.extract('epoch', models.ReportRequest.updated_at - models.ReportRequest.created_at)))\
        .where(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago)
    avg_report_time_seconds = (await db.execute(avg_time_stmt)).scalar_one_or_none()

    new_prospects_24h = await db.scalar(select(func.count(models.Prospect.prospect_id)).where(models.Prospect.created_at >= one_day_ago)) or 0

    email_counts_stmt = select(
        func.count().filter(models.Prospect.status == 'CONTACTED', models.Prospect.last_contacted_at >= one_day_ago).label('sent_24h'),
        func.count().filter(models.Prospect.status == 'BOUNCED', models.Prospect.last_contacted_at >= one_day_ago).label('bounced_24h'),
        func.count().filter(models.Prospect.last_contacted_at >= one_day_ago, models.Prospect.status.in_(['CONTACTED', 'BOUNCED'])).label('contacted_total_24h')
    )
    email_kpis = (await db.execute(email_counts_stmt)).first()

    account_counts_stmt = select(
        func.count().filter(models.EmailAccount.is_active == True).label('active'),
        func.count().filter(models.EmailAccount.is_active == False, models.EmailAccount.updated_at >= one_day_ago).label('deactivated_24h')
    )
    account_kpis = (await db.execute(account_counts_stmt)).first()

    contacted_total = email_kpis.contacted_total_24h or 0
    bounce_rate_24h = (float(email_kpis.bounced_24h or 0) / float(contacted_total) * 100.0) if contacted_total > 0 else 0.0

    api_key_counts_stmt = select(
        func.count().filter(models.ApiKey.status == 'active').label('active_keys'),
        func.count().filter(models.ApiKey.status != 'active', models.ApiKey.updated_at >= one_day_ago).label('deactivated_keys_24h')
    )
    api_key_kpis = (await db.execute(api_key_counts_stmt)).first()

    snapshot = KpiSnapshot(
        awaiting_generation_reports=report_kpis.awaiting_generation or 0,
        pending_reports=report_kpis.pending_task or 0, # Use pending_task count
        processing_reports=report_kpis.processing or 0,
        completed_reports_24h=report_kpis.completed_24h or 0,
        failed_reports_24h=report_kpis.failed_24h or 0,
        delivery_failed_reports_24h=report_kpis.delivery_failed_24h or 0,
        avg_report_time_seconds=avg_report_time_seconds,
        new_prospects_24h=new_prospects_24h,
        emails_sent_24h=email_kpis.sent_24h or 0,
        active_email_accounts=account_kpis.active or 0,
        deactivated_accounts_24h=account_kpis.deactivated_24h or 0,
        bounce_rate_24h=bounce_rate_24h,
        revenue_24h=report_kpis.revenue_24h or 0.0,
        orders_created_24h=report_kpis.orders_created_24h or 0,
        active_api_keys=api_key_kpis.active_keys or 0,
        deactivated_api_keys_24h=api_key_kpis.deactivated_keys_24h or 0,
        # Keep awaiting_payment if needed
        awaiting_payment_reports=getattr(report_kpis, 'awaiting_payment', 0) or 0
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    logger.info(f"Created KPI Snapshot ID: {snapshot.snapshot_id}")
    return snapshot

async def log_mcol_decision(db: AsyncSession, kpi_snapshot_id: Optional[int] = None, **kwargs) -> McolDecisionLog:
    """Logs a decision made by the MCOL."""
    # Ensure complex types are JSON serialized
    if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], (str, type(None))):
        kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])
    if 'action_parameters' in kwargs and not isinstance(kwargs['action_parameters'], (str, type(None))):
         kwargs['action_parameters'] = json.dumps(kwargs['action_parameters'])

    decision = McolDecisionLog(kpi_snapshot_id=kpi_snapshot_id, **kwargs)
    db.add(decision)
    await db.flush()
    await db.refresh(decision)
    logger.info(f"Logged MCOL Decision ID: {decision.log_id}, Status: {decision.action_status}")
    return decision

async def update_mcol_decision_log(db: AsyncSession, log_id: int, **kwargs) -> Optional[McolDecisionLog]:
    """Updates an existing MCOL decision log entry."""
    log_entry = await db.get(McolDecisionLog, log_id)
    if log_entry:
        # Ensure complex types are JSON serialized before update
        if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], (str, type(None))):
            kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])
        if 'action_parameters' in kwargs and not isinstance(kwargs['action_parameters'], (str, type(None))):
             kwargs['action_parameters'] = json.dumps(kwargs['action_parameters'])

        for key, value in kwargs.items():
            setattr(log_entry, key, value)
        log_entry.updated_at = func.now() # Explicitly update timestamp
        await db.flush()
        await db.refresh(log_entry)
        logger.info(f"Updated MCOL Decision Log ID: {log_id}, Status: {log_entry.action_status}")
    else:
        logger.warning(f"MCOL Decision Log ID {log_id} not found for update.")
    return log_entry

async def get_latest_kpi_snapshot(db: AsyncSession) -> Optional[KpiSnapshot]:
    """Retrieves the most recent KPI snapshot."""
    result = await db.execute(select(KpiSnapshot).order_by(KpiSnapshot.timestamp.desc()).limit(1))
    return result.scalars().first()

# --- API Key CRUD ---

async def add_api_key(
    db: AsyncSession,
    key: str,
    provider: str,
    email: Optional[str] = None,
    proxy: Optional[str] = None,
    notes: Optional[str] = None
) -> Optional[models.ApiKey]:
    """Adds a new API key, storing it encrypted. Returns None if encryption fails or key exists."""
    try:
        encrypted_key = encrypt_data(key)
        if not encrypted_key:
            logger.error(f"Failed to encrypt API key for provider {provider}")
            return None

        # Check if the encrypted key already exists
        existing = await db.scalar(select(models.ApiKey).where(models.ApiKey.api_key_encrypted == encrypted_key))
        if existing:
            logger.info(f"API Key already exists (ID: {existing.id}, Status: {existing.status}). Skipping add.")
            # Optionally reactivate if inactive?
            if existing.status != 'active':
                 logger.info(f"Reactivating existing API Key ID {existing.id}.")
                 existing.status = 'active'
                 existing.last_failure_reason = None # Clear failure reason
                 existing.failure_count = 0
                 existing.notes = f"Re-added/Activated on {datetime.datetime.now(datetime.timezone.utc)}. {notes or ''}".strip()
                 await db.flush()
                 await db.refresh(existing)
                 return existing # Return the reactivated key
            return None # Don't add duplicate if already active

        db_key = models.ApiKey(
            api_key_encrypted=encrypted_key,
            provider=provider,
            email_used=email,
            proxy_used=proxy,
            status="active",
            notes=notes,
            failure_count=0 # Initialize failure count
        )
        db.add(db_key)
        await db.flush()
        await db.refresh(db_key)
        logger.info(f"Added new API Key ID {db_key.id} for provider {provider}")
        return db_key
    except Exception as e:
         logger.error(f"Error adding API key: {e}", exc_info=True)
         await db.rollback() # Rollback on error
         return None

async def get_active_api_keys(db: AsyncSession, provider: str) -> List[str]:
    """Retrieves all active, decrypted API keys for a given provider."""
    stmt = select(models.ApiKey.api_key_encrypted, models.ApiKey.id) \
        .where(models.ApiKey.provider == provider) \
        .where(models.ApiKey.status == 'active')
    result = await db.execute(stmt)
    encrypted_keys_with_ids = result.all()

    decrypted_keys = []
    keys_to_deactivate = []
    for enc_key, key_id in encrypted_keys_with_ids:
        try:
            dec_key = decrypt_data(enc_key)
            if dec_key: # Check if decryption was successful
                decrypted_keys.append(dec_key)
            else:
                logger.warning(f"Failed to decrypt API key ID {key_id} for provider {provider}. Marking as error.")
                keys_to_deactivate.append((key_id, 'Decryption failed'))
        except Exception as decrypt_err:
             logger.warning(f"Exception during decryption for API key ID {key_id}: {decrypt_err}")
             keys_to_deactivate.append((key_id, f'Decryption exception: {decrypt_err}'))

    # Deactivate keys that failed decryption in a separate step
    if keys_to_deactivate:
        logger.warning(f"Deactivating {len(keys_to_deactivate)} keys due to decryption issues.")
        for key_id, reason in keys_to_deactivate:
            await set_api_key_status_by_id(db, key_id, 'error', reason)
        # Commit these deactivations? Depends on caller context. Assume caller commits.

    return decrypted_keys

# --- MODIFICATION: Add get_active_api_keys_with_details ---
async def get_active_api_keys_with_details(db: AsyncSession, provider: str) -> List[Dict[str, Any]]:
    """Retrieves details (id, encrypted key, rate limit) of active API keys."""
    stmt = select(
        models.ApiKey.id,
        models.ApiKey.api_key_encrypted,
        models.ApiKey.rate_limited_until # Assuming this column exists
    ).where(models.ApiKey.provider == provider).where(models.ApiKey.status == 'active')
    result = await db.execute(stmt)
    # Convert rows to dictionaries
    keys_details = [
        {"id": row.id, "api_key_encrypted": row.api_key_encrypted, "rate_limited_until": getattr(row, 'rate_limited_until', None)}
        for row in result.all()
    ]
    return keys_details
# --- END MODIFICATION ---

# --- MODIFICATION: Add count_active_api_keys ---
async def count_active_api_keys(db: AsyncSession, provider: str) -> int:
    """Counts the number of active API keys for a given provider."""
    stmt = select(func.count(models.ApiKey.id))\
        .where(models.ApiKey.provider == provider)\
        .where(models.ApiKey.status == 'active')
    result = await db.execute(stmt)
    count = result.scalar_one_or_none()
    return count or 0
# --- END MODIFICATION ---


async def set_api_key_status_by_id(db: AsyncSession, key_id: int, status: str, reason: Optional[str] = None) -> bool:
    """Updates the status and failure reason/count of an API key identified by its database ID."""
    db_key = await db.get(models.ApiKey, key_id)
    if db_key:
        now = datetime.datetime.now(datetime.timezone.utc)
        update_values = {
            'status': status,
            'last_failure_reason': reason[:250] if reason else None, # Truncate reason
            'updated_at': now
        }
        if status != 'active':
            # Increment failure count if moving to a non-active state
            update_values['failure_count'] = models.ApiKey.failure_count + 1
        else:
            # Reset failure count if reactivating
            update_values['failure_count'] = 0
            update_values['last_failure_reason'] = None # Clear reason on reactivation

        stmt = (
            update(models.ApiKey)
            .where(models.ApiKey.id == key_id)
            .values(**update_values)
            .execution_options(synchronize_session="fetch")
        )
        await db.execute(stmt)
        logger.info(f"Updated status for API Key ID {key_id} to '{status}'. Reason: {reason}")
        return True
    else:
        logger.warning(f"API Key ID {key_id} not found for status update.")
        return False

async def find_api_key_id_by_value(db: AsyncSession, key_value: str, provider: str) -> Optional[int]:
    """Finds the database ID of an API key by its decrypted value and provider."""
    # This is inefficient but necessary if only the value is known.
    # Consider optimizing if this becomes a bottleneck (e.g., caching decrypted keys temporarily).
    stmt = select(models.ApiKey.id, models.ApiKey.api_key_encrypted).where(models.ApiKey.provider == provider)
    result = await db.execute(stmt)
    for key_id, encrypted_key in result.all():
        try:
            decrypted_key = decrypt_data(encrypted_key)
            if decrypted_key == key_value:
                return key_id
        except Exception:
            continue # Skip keys that fail decryption
    return None

async def set_api_key_status_by_value(db: AsyncSession, key_value: str, provider: str, status: str, reason: Optional[str] = None) -> bool:
    """Finds an API key by its DECRYPTED value and provider, then updates its status using its ID."""
    target_key_id = await find_api_key_id_by_value(db, key_value, provider)
    if target_key_id:
        return await set_api_key_status_by_id(db, target_key_id, status, reason)
    else:
        logger.warning(f"API Key matching value for provider '{provider}' not found for status update.")
        return False

async def set_api_key_inactive(db: AsyncSession, key_to_deactivate: str, provider: str, reason: str) -> bool:
     """Convenience function to mark a key as inactive by its value."""
     return await set_api_key_status_by_value(db, key_value=key_to_deactivate, provider=provider, status='inactive', reason=reason)

# --- MODIFICATION: Add mark_api_key_used_by_id ---
async def mark_api_key_used_by_id(db: AsyncSession, key_id: int) -> bool:
    """Updates the last_used_at timestamp for a given API key ID."""
    stmt = (
        update(models.ApiKey)
        .where(models.ApiKey.id == key_id)
        .values(last_used_at=func.now())
        .execution_options(synchronize_session="fetch") # Keep session consistent
    )
    result = await db.execute(stmt)
    if result.rowcount > 0:
        # logger.debug(f"Marked API Key ID {key_id} as used.") # Debug level
        return True
    else:
        logger.warning(f"API Key ID {key_id} not found for 'mark_api_key_used_by_id'.")
        return False
# --- END MODIFICATION ---

# --- MODIFICATION: Add set_api_key_rate_limited ---
async def set_api_key_rate_limited(db: AsyncSession, key_id: int, cooldown_until: datetime.datetime, reason: str) -> bool:
    """Sets the rate_limited_until timestamp and updates status."""
    stmt = (
        update(models.ApiKey)
        .where(models.ApiKey.id == key_id)
        .values(
            status='rate_limited', # Set status explicitly
            rate_limited_until=cooldown_until,
            last_failure_reason=f"Rate Limited: {reason[:200]}", # Store reason
            updated_at=func.now()
        )
        .execution_options(synchronize_session="fetch")
    )
    result = await db.execute(stmt)
    if result.rowcount > 0:
        logger.info(f"Marked API Key ID {key_id} as rate-limited until {cooldown_until}. Reason: {reason}")
        return True
    else:
        logger.warning(f"API Key ID {key_id} not found for setting rate limit.")
        return False
# --- END MODIFICATION ---


# --- Agent Task CRUD ---
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
    logger.info(f"Created AgentTask ID {db_task.task_id} for agent {agent_name} with goal: {goal}")
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
    # TODO: Add dependency check (task status should be WAITING if depends_on task is not COMPLETED)
    stmt = (
        select(models.AgentTask)
        .where(models.AgentTask.agent_name == agent_name)
        .where(models.AgentTask.status == 'PENDING')
        # .where(or_(
        #     models.AgentTask.depends_on_task_id.is_(None),
        #     # Add subquery/join to check dependent task status if needed
        # ))
        .order_by(models.AgentTask.priority.desc(), models.AgentTask.created_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    if tasks:
        logger.info(f"Fetched {len(tasks)} pending tasks for agent {agent_name}.")
    return tasks

async def update_task_status(
    db: AsyncSession,
    task_id: int,
    status: str,
    result: Optional[str] = None
) -> Optional[models.AgentTask]:
    """Updates a task's status and optional result message."""
    update_values = {
        'status': status,
        'result': result if result is not None else models.AgentTask.result, # Keep old result if None
        'updated_at': func.now()
    }
    stmt = (
        update(models.AgentTask)
        .where(models.AgentTask.task_id == task_id)
        .values(**update_values)
        .execution_options(synchronize_session="fetch")
        .returning(models.AgentTask)
    )
    result_proxy = await db.execute(stmt)
    updated_task = result_proxy.scalar_one_or_none()

    if updated_task:
        logger.info(f"Updated AgentTask ID {task_id} to status {status}.")
    else:
        logger.warning(f"AgentTask ID {task_id} not found for status update.")
    return updated_task

async def get_task(db: AsyncSession, task_id: int) -> Optional[models.AgentTask]:
     """Retrieves a specific task by ID."""
     result = await db.execute(select(models.AgentTask).where(models.AgentTask.task_id == task_id))
     return result.scalars().first()

# --- System Event Logging (Example) ---
# Add this if you want MCOL or other agents to log critical events
# async def log_system_event(db: AsyncSession, agent: str, event_type: str, details: str):
#     """Logs a system-level event."""
#     # Create a SystemEvent model if needed, or log to a generic table/file
#     logger.info(f"[SystemEvent] Agent: {agent}, Type: {event_type}, Details: {details}")
#     # Example DB logging:
#     # event = models.SystemEvent(agent_name=agent, event_type=event_type, details=details)
#     # db.add(event)
#     # await db.flush()

# --- MODIFICATION: Add function to update dynamic config (Example) ---
# This assumes you create a simple key-value table for dynamic settings
# async def update_dynamic_config(db: AsyncSession, key: str, value: Any) -> bool:
#     """Updates a dynamic configuration value in the database."""
#     # Example: Assumes a table 'dynamic_config' with 'config_key' and 'config_value' (JSON?)
#     # Implement proper upsert logic based on your DB model for dynamic config
#     logger.info(f"MCOL attempting to update config: {key} = {value}")
#     # Placeholder: Replace with actual DB update logic
#     # try:
#     #     stmt = upsert(...).values(config_key=key, config_value=json.dumps(value))
#     #     await db.execute(stmt)
#     #     await db.commit() # Commit immediately? Or let MCOL cycle commit?
#     #     logger.info(f"Successfully updated dynamic config '{key}'.")
#     #     # Need mechanism to signal relevant agents/app to reload config
#     #     return True
#     # except Exception as e:
#     #     logger.error(f"Failed to update dynamic config '{key}': {e}")
#     #     await db.rollback()
#     #     return False
#     return True # Placeholder success
# --- END MODIFICATION ---
