import datetime
import json
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Date, Float, JSON, Index, text
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from sqlalchemy import MetaData

# Define naming convention for constraints for Alembic autogenerate
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)

class ReportRequest(Base):
    __tablename__ = 'report_requests'

    request_id = Column(Integer, primary_key=True, index=True)
    client_name = Column(String(255), nullable=True)
    client_email = Column(String(255), nullable=False, index=True)
    company_name = Column(String(255), nullable=True)
    report_type = Column(String(50), nullable=False) # e.g., 'standard_499', 'premium_999', 'unknown_paid'
    request_details = Column(Text, nullable=False)
    # Consolidated Statuses: AWAITING_GENERATION, PENDING_TASK, PROCESSING, GENERATED, DELIVERY_FAILED, COMPLETED, FAILED
    status = Column(String(50), nullable=False, default='AWAITING_GENERATION', index=True)
    report_output_path = Column(String(1024), nullable=True) # Path to generated report file/data
    error_message = Column(Text, nullable=True) # Store errors if status is FAILED or DELIVERY_FAILED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # --- Payment Related Fields ---
    payment_status = Column(String(50), default='paid', index=True) # unpaid, paid, refunded (defaulting to paid as creation follows webhook)
    lemonsqueezy_order_id = Column(String(255), nullable=True, unique=True, index=True)
    lemonsqueezy_checkout_id = Column(String(255), nullable=True, index=True) # Optional: store checkout ID
    order_total_cents = Column(Integer, nullable=True) # Store amount paid (in cents)

class Prospect(Base):
    __tablename__ = 'prospects'

    prospect_id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False, index=True) # Added index
    website = Column(String(255), nullable=True)
    contact_name = Column(String(255), nullable=True)
    contact_email = Column(String(255), nullable=True, index=True) # Keep index, but uniqueness might be violated by guesses
    potential_pain_point = Column(Text, nullable=True)
    source = Column(String(100), nullable=True, index=True) # 'odr_<query>', 'manual' etc. Added index
    # Consolidated Statuses: NEW, INVALID_EMAIL, CONTACTED, BOUNCED, FAILED_GENERATION, FAILED_SEND, CONVERTED (optional)
    status = Column(String(50), nullable=False, default='NEW', index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    linkedin_profile_url = Column(String(512), nullable=True)
    key_executives = Column(JSON, nullable=True) # Store as {"title": "name", ...}

    # Add unique constraint on email if it's considered reliable, otherwise index is fine
    # __table_args__ = (UniqueConstraint('contact_email', name='uq_prospects_contact_email'),)


class EmailAccount(Base):
    __tablename__ = 'email_accounts'

    account_id = Column(Integer, primary_key=True, index=True)
    email_address = Column(String(255), nullable=False, unique=True) # The actual Gmail address
    alias_email = Column(String(255), nullable=True, index=True) # The alias to send FROM (e.g., research@yourdomain.com) - CRITICAL
    smtp_host = Column(String(255), nullable=False)
    smtp_port = Column(Integer, nullable=False)
    smtp_user = Column(String(255), nullable=False) # Usually same as email_address for Gmail
    smtp_password_encrypted = Column(Text, nullable=False) # Store encrypted App Password!
    provider = Column(String(100), nullable=True, default='gmail') # 'gmail', 'sendgrid_free', etc.
    daily_limit = Column(Integer, nullable=False, default=30) # Start low for warm-up
    emails_sent_today = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for sorting
    last_reset_date = Column(Date, nullable=False, default=func.current_date()) # Use func for DB default
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    notes = Column(Text, nullable=True) # e.g., domain associated, warmup status, deactivation reason
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class KpiSnapshot(Base):
    """Stores periodic snapshots of key performance indicators."""
    __tablename__ = 'kpi_snapshots'

    snapshot_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    # Report Metrics
    awaiting_generation_reports = Column(Integer, default=0) # Renamed from awaiting_payment
    pending_reports = Column(Integer, default=0) # Now means pending agent task pickup
    processing_reports = Column(Integer, default=0)
    completed_reports_24h = Column(Integer, default=0)
    failed_reports_24h = Column(Integer, default=0) # Generation failed
    delivery_failed_reports_24h = Column(Integer, default=0) # Delivery failed
    avg_report_time_seconds = Column(Float, nullable=True)
    # Prospecting Metrics
    new_prospects_24h = Column(Integer, default=0)
    # Email Metrics
    emails_sent_24h = Column(Integer, default=0)
    active_email_accounts = Column(Integer, default=0)
    deactivated_accounts_24h = Column(Integer, default=0)
    bounce_rate_24h = Column(Float, nullable=True) # Percentage
    # Financial Metrics
    revenue_24h = Column(Float, default=0.0) # Calculated from successful orders
    orders_created_24h = Column(Integer, default=0)
    # Resource Metrics
    active_api_keys = Column(Integer, default=0)
    deactivated_api_keys_24h = Column(Integer, default=0)
    # Add fields from initial migration if they were missed
    awaiting_payment_reports = Column(Integer, default=0) # Keep if direct API creation path exists

class McolDecisionLog(Base):
    """Logs decisions and actions taken by the MCOL."""
    __tablename__ = 'mcol_decision_log'

    log_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id', name=op.f('fk_mcol_decision_log_kpi_snapshot_id_kpi_snapshots')), nullable=True) # Link to snapshot that triggered decision
    priority_problem = Column(Text, nullable=True) # Problem identified (e.g., "High email bounce rate")
    analysis_summary = Column(Text, nullable=True) # LLM analysis output
    generated_strategy = Column(Text, nullable=True) # Store strategies as JSON string
    chosen_action = Column(Text, nullable=True) # Specific action MCOL decided to take
    action_parameters = Column(JSON, nullable=True) # Parameters for the action (e.g., code snippet, API endpoint)
    action_status = Column(String(50), default='PENDING', index=True) # PENDING, IMPLEMENTING, COMPLETED, FAILED, SUGGESTED
    action_result = Column(Text, nullable=True) # Outcome of the action (e.g., "Code modification suggested", "API call failed")
    follow_up_kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id', name=op.f('fk_mcol_decision_log_follow_up_kpi_snapshot_id_kpi_snapshots')), nullable=True) # Link to snapshot after action

    # Relationship (optional but good practice) - Define relationships outside the class if preferred
    # triggering_snapshot = relationship("KpiSnapshot", foreign_keys=[kpi_snapshot_id], backref="mcol_decisions")
    # followup_snapshot = relationship("KpiSnapshot", foreign_keys=[follow_up_kpi_snapshot_id], backref="mcol_followups")

class ApiKey(Base):
    """Stores API keys acquired or managed by the system."""
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    # Store the API key encrypted using functions from core.security
    api_key_encrypted = Column(Text, nullable=False, unique=True) # Encrypted key should be unique
    provider = Column(String(100), nullable=False, index=True, default='openrouter') # Default provider
    email_used = Column(String(255), nullable=True) # Temp email used for acquisition
    proxy_used = Column(String(255), nullable=True) # Proxy used for acquisition
    # Consolidated Statuses: active, inactive, error, rate_limited (banned implies inactive)
    status = Column(String(50), nullable=False, default='active', index=True)
    last_failure_reason = Column(String(255), nullable=True) # Store brief reason for failure/deactivation
    failure_count = Column(Integer, default=0, nullable=False) # Track consecutive failures
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for rotation
    notes = Column(Text, nullable=True) # General notes
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class AgentTask(Base):
    """Stores tasks for agents to perform."""
    __tablename__ = 'agent_tasks'

    task_id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True) # e.g., ReportGenerator, ProspectResearcher
    # Consolidated Statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED, WAITING (for dependency)
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    priority = Column(Integer, default=0, nullable=False, index=True) # Index for ordering
    goal = Column(Text, nullable=False) # e.g., "Generate Report", "Research Query"
    parameters = Column(JSON, nullable=True) # e.g., {"report_request_id": 123}, {"query": "..."}
    result = Column(Text, nullable=True) # Store outcome or error message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    depends_on_task_id = Column(Integer, ForeignKey('agent_tasks.task_id', name=op.f('fk_agent_tasks_depends_on_task_id_agent_tasks')), nullable=True)

    # Optional: Relationship for dependencies
    # dependent_task = relationship("AgentTask", remote_side=[task_id], backref="prerequisites")

# Add indexes explicitly if not covered by primary/foreign keys or specific queries
Index('ix_agent_tasks_status_priority_created', AgentTask.status, AgentTask.priority.desc(), AgentTask.created_at.asc())
Index('ix_email_accounts_active_alias_usage', EmailAccount.is_active, EmailAccount.alias_email, EmailAccount.last_used_at.asc())
Index('ix_api_keys_provider_status_usage', ApiKey.provider, ApiKey.status, ApiKey.last_used_at.asc())

# Define relationships after classes if preferred
McolDecisionLog.triggering_snapshot = relationship("KpiSnapshot", foreign_keys=[McolDecisionLog.kpi_snapshot_id], back_populates="mcol_decisions")
McolDecisionLog.followup_snapshot = relationship("KpiSnapshot", foreign_keys=[McolDecisionLog.follow_up_kpi_snapshot_id], back_populates="mcol_followups")
KpiSnapshot.mcol_decisions = relationship("McolDecisionLog", foreign_keys=[McolDecisionLog.kpi_snapshot_id], back_populates="triggering_snapshot")
KpiSnapshot.mcol_followups = relationship("McolDecisionLog", foreign_keys=[McolDecisionLog.follow_up_kpi_snapshot_id], back_populates="followup_snapshot")
# AgentTask.prerequisites = relationship("AgentTask", remote_side=[AgentTask.task_id], backref="dependent_task") # Example dependency relationship