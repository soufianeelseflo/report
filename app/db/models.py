# autonomous_agency/app/db/models.py
import datetime
import json
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Date, Float, JSON, Index, text, MetaData # Added MetaData
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
# from alembic import op # Only needed if using op directly in models, usually not

# Define naming convention for constraints for Alembic autogenerate
# Ensures consistent naming across the database and for migrations
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
    report_type = Column(String(50), nullable=False, index=True) # e.g., 'standard_499', 'premium_999', 'unknown_paid'
    request_details = Column(Text, nullable=False)
    # Statuses: AWAITING_GENERATION, PENDING_TASK, PROCESSING, GENERATED, DELIVERY_FAILED, COMPLETED, FAILED, REFUNDED
    status = Column(String(50), nullable=False, default='AWAITING_GENERATION', index=True)
    report_output_path = Column(String(1024), nullable=True) # Path to generated report file/data
    error_message = Column(Text, nullable=True) # Store errors if status is FAILED/DELIVERY_FAILED/REFUNDED
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # --- Payment Related Fields ---
    payment_status = Column(String(50), default='paid', index=True, nullable=False) # unpaid, paid, refunded
    lemonsqueezy_order_id = Column(String(255), nullable=True, unique=True, index=True) # Must be unique
    lemonsqueezy_checkout_id = Column(String(255), nullable=True, index=True)
    order_total_cents = Column(Integer, nullable=True) # Store amount paid (in cents)

class Prospect(Base):
    __tablename__ = 'prospects'

    prospect_id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False, index=True) # Crucial for lookups
    website = Column(String(512), nullable=True) # Increased length
    contact_name = Column(String(255), nullable=True)
    contact_email = Column(String(255), nullable=True, index=True) # Indexed, but not unique due to guesses
    potential_pain_point = Column(Text, nullable=True)
    source = Column(String(100), nullable=True, index=True) # 'odr_<query>', 'manual' etc.
    # Statuses: NEW, INVALID_EMAIL, CONTACTED, BOUNCED, FAILED_GENERATION, FAILED_SEND, CONVERTED
    status = Column(String(50), nullable=False, default='NEW', index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    linkedin_profile_url = Column(String(512), nullable=True)
    key_executives = Column(JSON, nullable=True) # Store as {"title": "name", ...}

class EmailAccount(Base):
    __tablename__ = 'email_accounts'

    account_id = Column(Integer, primary_key=True, index=True)
    email_address = Column(String(255), nullable=False, unique=True) # The actual login email address
    alias_email = Column(String(255), nullable=True, index=True) # The alias to send FROM - CRITICAL
    smtp_host = Column(String(255), nullable=False)
    smtp_port = Column(Integer, nullable=False)
    smtp_user = Column(String(255), nullable=False) # Usually same as email_address
    smtp_password_encrypted = Column(Text, nullable=False) # Store encrypted App Password securely
    provider = Column(String(100), nullable=True, default='smtp') # Generic 'smtp' or specific like 'gmail'
    daily_limit = Column(Integer, nullable=False, default=30) # Start low for warm-up
    emails_sent_today = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for sorting/rotation
    last_reset_date = Column(Date, nullable=False, server_default=func.current_date()) # Use server_default for DB consistency
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    notes = Column(Text, nullable=True) # e.g., domain associated, warmup status, deactivation reason
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('ix_email_accounts_active_alias_usage_limit', is_active, alias_email, daily_limit, emails_sent_today, last_used_at.asc().nullsfirst()),
    )

class KpiSnapshot(Base):
    __tablename__ = 'kpi_snapshots'

    snapshot_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    # Report Metrics
    awaiting_generation_reports = Column(Integer, default=0, nullable=False)
    pending_reports = Column(Integer, default=0, nullable=False) # Reports awaiting agent task pickup
    processing_reports = Column(Integer, default=0, nullable=False)
    completed_reports_24h = Column(Integer, default=0, nullable=False)
    failed_reports_24h = Column(Integer, default=0, nullable=False) # Generation failed
    delivery_failed_reports_24h = Column(Integer, default=0, nullable=False) # Delivery failed
    avg_report_time_seconds = Column(Float, nullable=True)
    # Prospecting Metrics
    new_prospects_24h = Column(Integer, default=0, nullable=False)
    # Email Metrics
    emails_sent_24h = Column(Integer, default=0, nullable=False) # Successful sends in last 24h
    active_email_accounts = Column(Integer, default=0, nullable=False)
    deactivated_accounts_24h = Column(Integer, default=0, nullable=False)
    bounce_rate_24h = Column(Float, nullable=True) # Percentage
    # Financial Metrics
    revenue_24h = Column(Float, default=0.0, nullable=False) # Calculated from successful orders
    orders_created_24h = Column(Integer, default=0, nullable=False)
    # Resource Metrics
    active_api_keys = Column(Integer, default=0, nullable=False)
    deactivated_api_keys_24h = Column(Integer, default=0, nullable=False)

    # Relationships for MCOL
    mcol_decisions = relationship("McolDecisionLog", foreign_keys="McolDecisionLog.kpi_snapshot_id", back_populates="triggering_snapshot", lazy="selectin", cascade="all, delete-orphan")
    mcol_followups = relationship("McolDecisionLog", foreign_keys="McolDecisionLog.follow_up_kpi_snapshot_id", back_populates="followup_snapshot", lazy="selectin", cascade="all, delete-orphan")


class McolDecisionLog(Base):
    __tablename__ = 'mcol_decision_log'

    log_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id', ondelete='SET NULL'), nullable=True, index=True) # Link to snapshot, set null on delete
    priority_problem = Column(Text, nullable=True)
    analysis_summary = Column(Text, nullable=True)
    generated_strategy = Column(Text, nullable=True) # Store strategies as JSON string
    chosen_action = Column(Text, nullable=True)
    action_parameters = Column(JSON, nullable=True)
    action_status = Column(String(50), default='PENDING', index=True, nullable=False) # PENDING, IMPLEMENTING, COMPLETED, FAILED, SUGGESTED
    action_result = Column(Text, nullable=True)
    follow_up_kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id', ondelete='SET NULL'), nullable=True) # Link to snapshot after action
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships defined via back_populates in KpiSnapshot
    triggering_snapshot = relationship("KpiSnapshot", foreign_keys=[kpi_snapshot_id], back_populates="mcol_decisions")
    followup_snapshot = relationship("KpiSnapshot", foreign_keys=[follow_up_kpi_snapshot_id], back_populates="mcol_followups")


class ApiKey(Base):
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    api_key_encrypted = Column(Text, nullable=False, unique=True) # Encrypted key MUST be unique
    provider = Column(String(100), nullable=False, index=True, default='openrouter')
    email_used = Column(String(255), nullable=True) # Temp email used for acquisition
    proxy_used = Column(String(255), nullable=True) # Proxy used for acquisition
    # Statuses: active, inactive, error, rate_limited
    status = Column(String(50), nullable=False, default='active', index=True)
    last_failure_reason = Column(String(255), nullable=True)
    failure_count = Column(Integer, default=0, nullable=False)
    rate_limited_until = Column(DateTime(timezone=True), nullable=True, index=True) # **ADDED** Track when rate limit expires
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for rotation
    notes = Column(Text, nullable=True) # General notes
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('ix_api_keys_provider_status_rate_limit_usage', provider, status, rate_limited_until, last_used_at.asc().nullsfirst()),
    )

class AgentTask(Base):
    __tablename__ = 'agent_tasks'

    task_id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True) # e.g., ReportGenerator, ProspectResearcher
    # Statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    priority = Column(Integer, default=0, nullable=False, index=True) # Index for ordering
    goal = Column(Text, nullable=False)
    parameters = Column(JSON, nullable=True) # e.g., {"report_request_id": 123}
    result = Column(Text, nullable=True) # Store outcome or error message
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    depends_on_task_id = Column(Integer, ForeignKey('agent_tasks.task_id', ondelete='SET NULL'), nullable=True, index=True) # Set NULL on delete

    # Optional: Relationship for dependencies
    # dependent_task = relationship("AgentTask", remote_side=[task_id], backref="prerequisites")

    __table_args__ = (
        Index('ix_agent_tasks_agent_status_priority_created', agent_name, status, priority.desc(), created_at.asc()),
    )