from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from . import models
from app.api import schemas # Import schemas
from sqlalchemy import func, cast, Float, Integer as SQLInteger
from .models import KpiSnapshot, McolDecisionLog, ReportRequest, Prospect, EmailAccount
import datetime

async def create_report_request(db: AsyncSession, request: schemas.ReportRequestCreate) -> models.ReportRequest:
    """Creates a new report request in the database."""
    db_request = models.ReportRequest(
        client_name=request.client_name,
        client_email=request.client_email,
        company_name=request.company_name,
        report_type=request.report_type,
        request_details=request.request_details,
        status="PENDING" # Initial status
    )
    db.add(db_request)
    await db.flush() # Flush to get the ID without full commit yet
    await db.refresh(db_request)
    return db_request

# --- Add other CRUD functions for agents later ---

async def get_pending_report_request(db: AsyncSession) -> Optional[models.ReportRequest]:
    """Gets the oldest PENDING report request."""
    result = await db.execute(
        select(models.ReportRequest)
        .where(models.ReportRequest.status == 'PENDING')
        .order_by(models.ReportRequest.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True) # Attempt to lock the row
    )
    return result.scalars().first()

async def update_report_request_status(db: AsyncSession, request_id: int, status: str, output_path: Optional[str] = None, error_message: Optional[str] = None) -> Optional[models.ReportRequest]:
    """Updates the status and potentially output path or error of a report request."""
    result = await db.execute(
        select(models.ReportRequest).where(models.ReportRequest.request_id == request_id)
    )
    db_request = result.scalars().first()
    if db_request:
        db_request.status = status
        if output_path:
            db_request.report_output_path = output_path
        if error_message:
            db_request.error_message = error_message
        await db.flush()
        await db.refresh(db_request)
    return db_request

# --- Add CRUD for Prospects ---
async def create_prospect(db: AsyncSession, company_name: str, email: Optional[str] = None, website: Optional[str] = None, pain_point: Optional[str] = None, source: Optional[str] = None) -> models.Prospect:
    # Basic check to avoid duplicate emails if provided
    if email:
        existing = await db.execute(select(models.Prospect).where(models.Prospect.contact_email == email))
        if existing.scalars().first():
            # Handle duplicate: update existing, skip, or raise error
            print(f"Skipping duplicate email: {email}")
            return None # Or update logic

    db_prospect = models.Prospect(
        company_name=company_name,
        website=website,
        contact_email=email,
        potential_pain_point=pain_point,
        source=source,
        status="NEW"
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


# --- Add CRUD for Email Accounts ---
async def get_active_email_account_for_sending(db: AsyncSession) -> Optional[models.EmailAccount]:
    """Finds an active email account under its daily limit."""
    today = datetime.date.today()
    result = await db.execute(
        select(models.EmailAccount)
        .where(models.EmailAccount.is_active == True)
        # Reset daily count if the last reset date is before today
        .where(
            (models.EmailAccount.last_reset_date < today) |
            (models.EmailAccount.emails_sent_today < models.EmailAccount.daily_limit)
        )
        .order_by(models.EmailAccount.last_used_at.asc().nullsfirst()) # Prioritize least recently used
        .limit(1)
        .with_for_update(skip_locked=True) # Lock the account
    )
    account = result.scalars().first()

    # Reset counter if needed before returning
    if account and account.last_reset_date < today:
        account.emails_sent_today = 0
        account.last_reset_date = today
        await db.flush()
        await db.refresh(account)

    # Check limit again after potential reset
    if account and account.emails_sent_today < account.daily_limit:
        return account
    else:
        # If the selected account was reset but is *still* over limit (limit=0?), or no account found
        return None


async def increment_email_sent_count(db: AsyncSession, account_id: int) -> None:
    """Increments the sent count and updates last used time for an email account."""
    result = await db.execute(
        select(models.EmailAccount).where(models.EmailAccount.account_id == account_id)
    )
    account = result.scalars().first()
    if account:
        account.emails_sent_today += 1
        account.last_used_at = datetime.datetime.now(datetime.timezone.utc)
        await db.flush()

async def set_email_account_inactive(db: AsyncSession, account_id: int, reason: str) -> None:
    """Marks an email account as inactive."""
    result = await db.execute(
        select(models.EmailAccount).where(models.EmailAccount.account_id == account_id)
    )
    account = result.scalars().first()
    if account:
        account.is_active = False
        account.notes = f"Deactivated on {datetime.datetime.now(datetime.timezone.utc)}: {reason}"
        await db.flush()

async def create_kpi_snapshot(db: AsyncSession) -> KpiSnapshot:
    """Calculates current KPIs and saves a snapshot."""
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(days=1)

    # --- Calculate KPIs (Examples - refine queries for performance) ---

    # Report Metrics
    pending_reports = await db.scalar(select(func.count(ReportRequest.request_id)).where(ReportRequest.status == 'PENDING'))
    processing_reports = await db.scalar(select(func.count(ReportRequest.request_id)).where(ReportRequest.status == 'PROCESSING'))
    completed_reports_24h = await db.scalar(select(func.count(ReportRequest.request_id)).where(ReportRequest.status == 'COMPLETED', ReportRequest.updated_at >= one_day_ago))
    failed_reports_24h = await db.scalar(select(func.count(ReportRequest.request_id)).where(ReportRequest.status == 'FAILED', ReportRequest.updated_at >= one_day_ago))
    # Avg report time (simple example - could be more complex)
    avg_time_result = await db.execute(select(func.avg(func.extract('epoch', ReportRequest.updated_at - ReportRequest.created_at))).where(ReportRequest.status == 'COMPLETED', ReportRequest.updated_at >= one_day_ago))
    avg_report_time_seconds = avg_time_result.scalar_one_or_none()

    # Prospecting Metrics
    new_prospects_24h = await db.scalar(select(func.count(Prospect.prospect_id)).where(Prospect.created_at >= one_day_ago))

    # Email Metrics
    emails_sent_24h = await db.scalar(select(func.count(Prospect.prospect_id)).where(Prospect.status == 'CONTACTED', Prospect.last_contacted_at >= one_day_ago))
    active_email_accounts = await db.scalar(select(func.count(EmailAccount.account_id)).where(EmailAccount.is_active == True))
    deactivated_accounts_24h = await db.scalar(select(func.count(EmailAccount.account_id)).where(EmailAccount.is_active == False, EmailAccount.updated_at >= one_day_ago))
    # Bounce Rate (CONTACTED + BOUNCED in last 24h)
    contacted_total_24h = await db.scalar(select(func.count(Prospect.prospect_id)).where(Prospect.last_contacted_at >= one_day_ago, Prospect.status.in_(['CONTACTED', 'BOUNCED'])))
    bounced_24h = await db.scalar(select(func.count(Prospect.prospect_id)).where(Prospect.last_contacted_at >= one_day_ago, Prospect.status == 'BOUNCED'))
    bounce_rate_24h = (float(bounced_24h) / float(contacted_total_24h) * 100.0) if contacted_total_24h and contacted_total_24h > 0 else 0.0

    # Financial Metrics (Placeholder)
    revenue_24h = 0.0 # Replace with actual calculation if payments table exists

    # --- Create Snapshot ---
    snapshot = KpiSnapshot(
        pending_reports=pending_reports,
        processing_reports=processing_reports,
        completed_reports_24h=completed_reports_24h,
        failed_reports_24h=failed_reports_24h,
        avg_report_time_seconds=avg_report_time_seconds,
        new_prospects_24h=new_prospects_24h,
        emails_sent_24h=emails_sent_24h,
        active_email_accounts=active_email_accounts,
        deactivated_accounts_24h=deactivated_accounts_24h,
        bounce_rate_24h=bounce_rate_24h,
        revenue_24h=revenue_24h,
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    return snapshot

async def log_mcol_decision(db: AsyncSession, kpi_snapshot_id: Optional[int] = None, **kwargs) -> McolDecisionLog:
    """Logs a decision made by the MCOL."""
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
        for key, value in kwargs.items():
            setattr(log_entry, key, value)
        await db.flush()
        await db.refresh(log_entry)
    return log_entry

async def get_latest_kpi_snapshot(db: AsyncSession) -> Optional[KpiSnapshot]:
    """Retrieves the most recent KPI snapshot."""
    result = await db.execute(select(KpiSnapshot).order_by(KpiSnapshot.timestamp.desc()).limit(1))
    return result.scalars().first()