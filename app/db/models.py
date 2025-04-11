import datetime
import json
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Date, Float, JSON
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
    status = Column(String(50), nullable=False, default='AWAITING_GENERATION', index=True) # AWAITING_PAYMENT (deprecate?), AWAITING_GENERATION, PENDING, PROCESSING, COMPLETED, FAILED, DELIVERY_FAILED
    report_output_path = Column(String(1024), nullable=True) # Path to generated report file/data
    error_message = Column(Text, nullable=True) # Store errors if status is FAILED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # --- Payment Related Fields ---
    payment_status = Column(String(50), default='unpaid', index=True) # unpaid, paid, refunded
    lemonsqueezy_order_id = Column(String(255), nullable=True, unique=True, index=True)
    lemonsqueezy_checkout_id = Column(String(255), nullable=True, index=True) # Optional: store checkout ID
    order_total_cents = Column(Integer, nullable=True) # Store amount paid (in cents)

class Prospect(Base):
    __tablename__ = 'prospects'

    prospect_id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    website = Column(String(255), nullable=True)
    contact_name = Column(String(255), nullable=True)
    contact_email = Column(String(255), nullable=True, unique=True, index=True) # Email guessing means not always unique in practice if guesses overlap, but keep constraint for now
    potential_pain_point = Column(Text, nullable=True)
    source = Column(String(100), nullable=True) # 'odr_<query>', 'manual' etc.
    status = Column(String(50), nullable=False, default='NEW', index=True) # NEW, CONTACTED, BOUNCED, FAILED_GENERATION, FAILED_SEND, INVALID_EMAIL
    last_contacted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    linkedin_profile_url = Column(String(512), nullable=True)
    key_executives = Column(JSON, nullable=True) # Store as {"title": "name", ...}

class EmailAccount(Base):
    __tablename__ = 'email_accounts'

    account_id = Column(Integer, primary_key=True, index=True)
    email_address = Column(String(255), nullable=False, unique=True) # The actual Gmail address
    alias_email = Column(String(255), nullable=True, index=True) # The alias to send FROM (e.g., research@yourdomain.com)
    smtp_host = Column(String(255), nullable=False)
    smtp_port = Column(Integer, nullable=False)
    smtp_user = Column(String(255), nullable=False) # Usually same as email_address for Gmail
    smtp_password_encrypted = Column(Text, nullable=False) # Store encrypted App Password!
    provider = Column(String(100), nullable=True, default='gmail') # 'gmail', 'sendgrid_free', etc.
    daily_limit = Column(Integer, nullable=False, default=30) # Start low for warm-up
    emails_sent_today = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_reset_date = Column(Date, nullable=False, default=datetime.date.today)
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
    awaiting_payment_reports = Column(Integer, default=0)
    awaiting_generation_reports = Column(Integer, default=0) # Added
    pending_reports = Column(Integer, default=0) # Now means pending agent task pickup
    processing_reports = Column(Integer, default=0)
    completed_reports_24h = Column(Integer, default=0)
    failed_reports_24h = Column(Integer, default=0)
    delivery_failed_reports_24h = Column(Integer, default=0) # Added
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
    active_api_keys = Column(Integer, default=0) # Added
    deactivated_api_keys_24h = Column(Integer, default=0) # Added
    # MCOL can add more KPIs as it learns

class McolDecisionLog(Base):
    """Logs decisions and actions taken by the MCOL."""
    __tablename__ = 'mcol_decision_log'

    log_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id'), nullable=True) # Link to snapshot that triggered decision
    priority_problem = Column(Text, nullable=True) # Problem identified (e.g., "High email bounce rate")
    analysis_summary = Column(Text, nullable=True) # LLM analysis output
    generated_strategy = Column(Text, nullable=True) # Store strategies as JSON string
    chosen_action = Column(Text, nullable=True) # Specific action MCOL decided to take
    action_parameters = Column(JSON, nullable=True) # Parameters for the action (e.g., code snippet, API endpoint)
    action_status = Column(String(50), default='PENDING') # PENDING, IMPLEMENTING, COMPLETED, FAILED, SUGGESTED
    action_result = Column(Text, nullable=True) # Outcome of the action (e.g., "Code modification suggested", "API call failed")
    follow_up_kpi_snapshot_id = Column(Integer, ForeignKey('kpi_snapshots.snapshot_id'), nullable=True) # Link to snapshot after action

# Relationship (optional but good practice)
KpiSnapshot.mcol_decisions = relationship("McolDecisionLog", backref="triggering_snapshot", foreign_keys=[McolDecisionLog.kpi_snapshot_id])
KpiSnapshot.mcol_followups = relationship("McolDecisionLog", backref="followup_snapshot", foreign_keys=[McolDecisionLog.follow_up_kpi_snapshot_id])

class ApiKey(Base):
    """Stores API keys acquired or managed by the system."""
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    # Store the API key encrypted using functions from core.security
    api_key_encrypted = Column(Text, nullable=False, unique=True) # Encrypted key should be unique
    provider = Column(String(100), nullable=False, index=True, default='openrouter') # Default provider
    email_used = Column(String(255), nullable=True) # Temp email used for acquisition
    proxy_used = Column(String(255), nullable=True) # Proxy used for acquisition
    status = Column(String(50), nullable=False, default='active', index=True) # active, inactive, banned, error, rate_limited
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    # Optional: Add notes field for errors during acquisition or usage issues
    notes = Column(Text, nullable=True)
    # Add updated_at for tracking changes like status updates
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class AgentTask(Base):
    """Stores tasks for agents to perform."""
    __tablename__ = 'agent_tasks'

    task_id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True) # e.g., ReportGenerator, ProspectResearcher
    status = Column(String(50), nullable=False, default='PENDING', index=True) # PENDING, IN_PROGRESS, COMPLETED, FAILED, WAITING
    priority = Column(Integer, default=0, nullable=False) # Higher number = higher priority
    goal = Column(Text, nullable=False) # e.g., "Generate Report", "Research Query"
    parameters = Column(JSON, nullable=True) # e.g., {"report_request_id": 123}, {"query": "..."}
    result = Column(Text, nullable=True) # Store outcome or error message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    depends_on_task_id = Column(Integer, ForeignKey('agent_tasks.task_id'), nullable=True)

    # Optional: Relationship for dependencies
    # dependent_task = relationship("AgentTask", remote_side=[task_id], backref="prerequisites")