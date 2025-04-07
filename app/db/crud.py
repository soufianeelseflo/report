import datetime
import json
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, cast, Float, Integer as SQLInteger, text

from . import models
from app.api import schemas # Import schemas
from .models import KpiSnapshot, McolDecisionLog, ReportRequest, Prospect, EmailAccount

async def create_report_request(db: AsyncSession, request: schemas.ReportRequestCreate) -> models.ReportRequest:
    """Creates a new report request in the database, initially awaiting payment."""
    db_request = models.ReportRequest(
        client_name=request.client_name,
        client_email=request.client_email,
        company_name=request.company_name,
        report_type=request.report_type,
        request_details=request.request_details,
        status="AWAITING_PAYMENT", # Start as awaiting payment
        payment_status="unpaid"
    )
    db.add(db_request)
    await db.flush() # Flush to get the ID without full commit yet
    await db.refresh(db_request)
    return db_request

async def create_report_request_from_webhook(
    db: AsyncSession,
    order_data: dict # Parsed data from Lemon Squeezy webhook
) -> Optional[models.ReportRequest]:
    """Creates a report request after successful payment via webhook."""

    ls_order_id = order_data.get('id')
    if not ls_order_id: return None # Should not happen

    # Check if order already processed
    existing = await db.scalar(select(models.ReportRequest).where(models.ReportRequest.lemonsqueezy_order_id == str(ls_order_id)))
    if existing:
        print(f"[Webhook] Order ID {ls_order_id} already processed for ReportRequest ID {existing.request_id}. Skipping.")
        return None

    # Extract necessary details from webhook payload
    attributes = order_data.get('attributes', {})
    customer_email = attributes.get('user_email')
    customer_name = attributes.get('user_name')
    product_name = attributes.get('first_order_item', {}).get('product_name', 'Unknown Product')
    variant_name = attributes.get('first_order_item', {}).get('variant_name', 'Unknown Variant')
    variant_id = attributes.get('first_order_item', {}).get('variant_id')
    order_total = attributes.get('total', 0) # Total in cents

    # --- CRUCIAL: Get the custom data (research details) ---
    # This relies on you setting up custom fields in Lemon Squeezy checkout
    custom_data = attributes.get('custom_data', {})
    request_details = custom_data.get('research_topic', 'Not Provided - Check Order Notes') # Adjust key 'research_topic'
    company_name_custom = custom_data.get('company_name') # Optional custom field

    if not customer_email:
        print(f"[Webhook] Error: Missing customer email for order {ls_order_id}.")
        return None # Cannot proceed without email

    # Determine report type based on variant ID (more reliable than name)
    report_type_code = 'unknown_paid' # Fallback
    if str(variant_id) == settings.LEMONSQUEEZY_VARIANT_STANDARD:
        report_type_code = 'standard_499'
    elif str(variant_id) == settings.LEMONSQUEEZY_VARIANT_PREMIUM:
        report_type_code = 'premium_999'
    else:
        print(f"[Webhook] Warning: Order {ls_order_id} variant ID {variant_id} doesn't match configured standard/premium IDs.")

    db_request = models.ReportRequest(
        client_name=customer_name,
        client_email=customer_email,
        company_name=company_name_custom, # Use custom field if provided
        report_type=report_type_code,
        request_details=request_details,
        status="PENDING", # Set to PENDING, ready for ReportGenerator
        payment_status="paid",
        lemonsqueezy_order_id=str(ls_order_id)
        # lemonsqueezy_checkout_id = ? # Can get this from checkout creation if stored
    )
    db.add(db_request)
    await db.flush()
    await db.refresh(db_request)
    print(f"[Webhook] Created ReportRequest ID {db_request.request_id} from LS Order {ls_order_id}")
    return db_request

async def get_pending_report_request(db: AsyncSession) -> Optional[models.ReportRequest]:
    """Gets the oldest PENDING (paid, ready to process) report request."""
    result = await db.execute(
        select(models.ReportRequest)
        .where(models.ReportRequest.status == 'PENDING')
        .order_by(models.ReportRequest.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True) # Attempt to lock the row
    )
    return result.scalars().first()

async def update_report_request_status(db: AsyncSession, request_id: int, status: str, output_path: Optional[str] = None, error_message: Optional[str] = None, payment_status: Optional[str] = None, checkout_id: Optional[str] = None) -> Optional[models.ReportRequest]:
    """Updates the status and potentially other fields of a report request."""
    result = await db.execute(
        select(models.ReportRequest).where(models.ReportRequest.request_id == request_id)
    )
    db_request = result.scalars().first()
    if db_request:
        if status: db_request.status = status
        if output_path: db_request.report_output_path = output_path
        if error_message: db_request.error_message = error_message
        if payment_status: db_request.payment_status = payment_status
        if checkout_id: db_request.lemonsqueezy_checkout_id = checkout_id
        # updated_at is handled by DB onupdate trigger if set up, otherwise:
        # db_request.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await db.flush()
        await db.refresh(db_request)
    return db_request

# --- Prospect CRUD ---
async def create_prospect(db: AsyncSession, company_name: str, email: Optional[str] = None, website: Optional[str] = None, pain_point: Optional[str] = None, source: Optional[str] = None, linkedin_url: Optional[str] = None, executives: Optional[Dict] = None) -> Optional[models.Prospect]:
    """Creates or updates a prospect."""
    # Check for existing prospect by email or company name? Decide on uniqueness criteria.
    existing = None
    if email:
        existing = await db.scalar(select(models.Prospect).where(models.Prospect.contact_email == email))

    if existing:
        # Update existing prospect
        print(f"Updating existing prospect: {email or company_name}")
        update_data = {
            "website": website or existing.website,
            "potential_pain_point": pain_point or existing.potential_pain_point,
            "source": source or existing.source,
            "linkedin_profile_url": linkedin_url or existing.linkedin_profile_url,
            "key_executives": executives or existing.key_executives,
            # Don't reset status if already contacted etc.
        }
        for key, value in update_data.items():
            setattr(existing, key, value)
        await db.flush()
        await db.refresh(existing)
        return existing
    else:
        # Create new prospect
        db_prospect = models.Prospect(
            company_name=company_name,
            website=website,
            contact_email=email,
            potential_pain_point=pain_point,
            source=source,
            status="NEW",
            linkedin_profile_url=linkedin_url,
            key_executives=executives
        )
        db.add(db_prospect)
        await db.flush()
        await db.refresh(db_prospect)
        return db_prospect

async def get_new_prospects_for_emailing(db: AsyncSession, limit: int) -> list[models.Prospect]:
    """Gets NEW prospects ready for emailing."""
    result = await db.execute(
        select(models.Prospect)
        .where(models.Prospect.status == 'NEW')
        .where(models.Prospect.contact_email != None) # Only those with emails
        .order_by(models.Prospect.created_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True) # Lock rows for processing
    )
    return result.scalars().all()

async def update_prospect_status(db: AsyncSession, prospect_id: int, status: str, last_contacted_at: Optional[datetime.datetime] = None) -> Optional[models.Prospect]:
    """Updates the status of a prospect."""
    result = await db.execute(
        select(models.Prospect).where(models.Prospect.prospect_id == prospect_id)
    )
    db_prospect = result.scalars().first()
    if db_prospect:
        db_prospect.status = status
        if last_contacted_at:
            db_prospect.last_contacted_at = last_contacted_at
        await db.flush()
        await db.refresh(db_prospect)
    return db_prospect

# --- Email Account CRUD ---
async def get_active_email_account_for_sending(db: AsyncSession) -> Optional[models.EmailAccount]:
    """Finds an active email account under its daily limit."""
    today = datetime.date.today()
    result = await db.execute(
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        .where(
            (models.EmailAccount.last_reset_date < today) |
            (models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        )
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst()) # Prioritize least recently used
        .limit(1)
        .with_for_update(skip_locked=True) # Lock the account
    )
    account = result.scalars().first()

    if account and account.last_reset_date < today:
        account.emails_sent_today = 0
        account.last_reset_date = today
        await db.flush()
        await db.refresh(account)

    if account and account.emails_sent_today < account.daily_limit:
        return account
    else:
        return None

async def increment_email_sent_count(db: AsyncSession, account_id: int) -> None:
    """Increments the sent count and updates last used time for an email account."""
    # Use DB atomic increment if possible, otherwise select-update
    await db.execute(
        text(
            "UPDATE email_accounts "
            "SET emails_sent_today = emails_sent_today + 1, last_used_at = :now "
            "WHERE account_id = :account_id"
        ),
        {"now": datetime.datetime.now(datetime.timezone.utc), "account_id": account_id}
    )
    # No need to flush/refresh if just updating

async def set_email_account_inactive(db: AsyncSession, account_id: int, reason: str) -> None:
    """Marks an email account as inactive."""
    await db.execute(
        text(
            "UPDATE email_accounts "
            "SET is_active = false, notes = :notes, updated_at = :now "
            "WHERE account_id = :account_id"
        ),
        {
            "notes": f"Deactivated on {datetime.datetime.now(datetime.timezone.utc)}: {reason}",
            "now": datetime.datetime.now(datetime.timezone.utc),
            "account_id": account_id
        }
    )
    # No need to flush/refresh

# --- MCOL CRUD ---
async def create_kpi_snapshot(db: AsyncSession) -> KpiSnapshot:
    """Calculates current KPIs and saves a snapshot."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(days=1)

    # --- Calculate KPIs (Refined Queries) ---
    report_counts = await db.execute(
        select(
            func.count().filter(models.ReportRequest.status == 'AWAITING_PAYMENT').label('awaiting_payment'),
            func.count().filter(models.ReportRequest.status == 'PENDING').label('pending'),
            func.count().filter(models.ReportRequest.status == 'PROCESSING').label('processing'),
            func.count().filter(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago).label('completed_24h'),
            func.count().filter(models.ReportRequest.status == 'FAILED', models.ReportRequest.updated_at >= one_day_ago).label('failed_24h'),
            func.count().filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago).label('orders_created_24h'), # Count paid orders
            func.sum(
                cast(text("substring(report_type from '(\\d+)$')"), SQLInteger) / 100.0 # Extract price and convert cents to dollars
            ).filter(models.ReportRequest.payment_status == 'paid', models.ReportRequest.created_at >= one_day_ago).label('revenue_24h') # Sum price from paid orders
        )
    )
    report_kpis = report_counts.first()

    avg_time_result = await db.execute(select(func.avg(func.extract('epoch', models.ReportRequest.updated_at - models.ReportRequest.created_at))).where(models.ReportRequest.status == 'COMPLETED', models.ReportRequest.updated_at >= one_day_ago))
    avg_report_time_seconds = avg_time_result.scalar_one_or_none()

    new_prospects_24h = await db.scalar(select(func.count(models.Prospect.prospect_id)).where(models.Prospect.created_at >= one_day_ago))

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

    snapshot = KpiSnapshot(
        awaiting_payment_reports=report_kpis.awaiting_payment,
        pending_reports=report_kpis.pending,
        processing_reports=report_kpis.processing,
        completed_reports_24h=report_kpis.completed_24h,
        failed_reports_24h=report_kpis.failed_24h,
        avg_report_time_seconds=avg_report_time_seconds,
        new_prospects_24h=new_prospects_24h,
        emails_sent_24h=email_kpis.sent_24h,
        active_email_accounts=account_kpis.active,
        deactivated_accounts_24h=account_kpis.deactivated_24h,
        bounce_rate_24h=bounce_rate_24h,
        revenue_24h=report_kpis.revenue_24h or 0.0,
        orders_created_24h=report_kpis.orders_created_24h or 0
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    return snapshot

async def log_mcol_decision(db: AsyncSession, kpi_snapshot_id: Optional[int] = None, **kwargs) -> McolDecisionLog:
    """Logs a decision made by the MCOL."""
    # Ensure generated_strategy is stored as JSON string if it's a list/dict
    if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], str):
        kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])

    decision = McolDecisionLog(kpi_snapshot_id=kpi_snapshot_id, **kwargs)
    db.add(decision)
    await db.flush()
    await db.refresh(decision)
    return decision

async def update_mcol_decision_log(db: AsyncSession, log_id: int, **kwargs) -> Optional[McolDecisionLog]:
    """Updates an existing MCOL decision log entry."""
    result = await db.execute(select(McolDecisionLog).where(McolDecisionLog.log_id == log_id))
    log_entry = result.scalars().first()
    if log_entry:
        # Ensure generated_strategy is stored as JSON string if updated
        if 'generated_strategy' in kwargs and not isinstance(kwargs['generated_strategy'], str):
            kwargs['generated_strategy'] = json.dumps(kwargs['generated_strategy'])
        for key, value in kwargs.items():
            setattr(log_entry, key, value)
        await db.flush()
        await db.refresh(log_entry)
    return log_entry

async def get_latest_kpi_snapshot(db: AsyncSession) -> Optional[KpiSnapshot]:
    """Retrieves the most recent KPI snapshot."""
    result = await db.execute(select(KpiSnapshot).order_by(KpiSnapshot.timestamp.desc()).limit(1))
    return result.scalars().first()