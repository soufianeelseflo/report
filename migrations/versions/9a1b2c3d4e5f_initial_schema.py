"""Initial schema setup (ReportRequest, Prospect, EmailAccount, PaymentFields)

Revision ID: 9a1b2c3d4e5f
Revises: 
Create Date: 2025-04-08 10:00:00.000000 

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9a1b2c3d4e5f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('email_accounts',
    sa.Column('account_id', sa.Integer(), nullable=False),
    sa.Column('email_address', sa.String(length=255), nullable=False),
    sa.Column('smtp_host', sa.String(length=255), nullable=False),
    sa.Column('smtp_port', sa.Integer(), nullable=False),
    sa.Column('smtp_user', sa.String(length=255), nullable=False),
    sa.Column('smtp_password_encrypted', sa.Text(), nullable=False),
    sa.Column('provider', sa.String(length=100), nullable=True),
    sa.Column('daily_limit', sa.Integer(), nullable=False, server_default='100'),
    sa.Column('emails_sent_today', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_reset_date', sa.Date(), nullable=False, server_default=sa.text('CURRENT_DATE')),
    sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('account_id', name=op.f('pk_email_accounts')),
    sa.UniqueConstraint('email_address', name=op.f('uq_email_accounts_email_address'))
    )
    op.create_table('prospects',
    sa.Column('prospect_id', sa.Integer(), nullable=False),
    sa.Column('company_name', sa.String(length=255), nullable=False),
    sa.Column('website', sa.String(length=255), nullable=True),
    sa.Column('contact_name', sa.String(length=255), nullable=True),
    sa.Column('contact_email', sa.String(length=255), nullable=True),
    sa.Column('potential_pain_point', sa.Text(), nullable=True),
    sa.Column('source', sa.String(length=100), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=False, server_default='NEW'),
    sa.Column('last_contacted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('linkedin_profile_url', sa.String(length=512), nullable=True),
    sa.Column('key_executives', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.PrimaryKeyConstraint('prospect_id', name=op.f('pk_prospects'))
    )
    op.create_index(op.f('ix_prospects_contact_email'), 'prospects', ['contact_email'], unique=True)
    op.create_index(op.f('ix_prospects_prospect_id'), 'prospects', ['prospect_id'], unique=False)
    op.create_index(op.f('ix_prospects_status'), 'prospects', ['status'], unique=False)
    op.create_table('report_requests',
    sa.Column('request_id', sa.Integer(), nullable=False),
    sa.Column('client_name', sa.String(length=255), nullable=True),
    sa.Column('client_email', sa.String(length=255), nullable=False),
    sa.Column('company_name', sa.String(length=255), nullable=True),
    sa.Column('report_type', sa.String(length=50), nullable=False),
    sa.Column('request_details', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=50), nullable=False, server_default='AWAITING_PAYMENT'),
    sa.Column('report_output_path', sa.String(length=1024), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('payment_status', sa.String(length=50), server_default='unpaid', nullable=True),
    sa.Column('lemonsqueezy_order_id', sa.String(length=255), nullable=True),
    sa.Column('lemonsqueezy_checkout_id', sa.String(length=255), nullable=True),
    sa.Column('order_total_cents', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('request_id', name=op.f('pk_report_requests'))
    )
    op.create_index(op.f('ix_report_requests_client_email'), 'report_requests', ['client_email'], unique=False)
    op.create_index(op.f('ix_report_requests_lemonsqueezy_checkout_id'), 'report_requests', ['lemonsqueezy_checkout_id'], unique=False)
    op.create_index(op.f('ix_report_requests_lemonsqueezy_order_id'), 'report_requests', ['lemonsqueezy_order_id'], unique=True)
    op.create_index(op.f('ix_report_requests_payment_status'), 'report_requests', ['payment_status'], unique=False)
    op.create_index(op.f('ix_report_requests_request_id'), 'report_requests', ['request_id'], unique=False)
    op.create_index(op.f('ix_report_requests_status'), 'report_requests', ['status'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_report_requests_status'), table_name='report_requests')
    op.drop_index(op.f('ix_report_requests_request_id'), table_name='report_requests')
    op.drop_index(op.f('ix_report_requests_payment_status'), table_name='report_requests')
    op.drop_index(op.f('ix_report_requests_lemonsqueezy_order_id'), table_name='report_requests')
    op.drop_index(op.f('ix_report_requests_lemonsqueezy_checkout_id'), table_name='report_requests')
    op.drop_index(op.f('ix_report_requests_client_email'), table_name='report_requests')
    op.drop_table('report_requests')
    op.drop_index(op.f('ix_prospects_status'), table_name='prospects')
    op.drop_index(op.f('ix_prospects_prospect_id'), table_name='prospects')
    op.drop_index(op.f('ix_prospects_contact_email'), table_name='prospects')
    op.drop_table('prospects')
    op.drop_table('email_accounts')
    # ### end Alembic commands ###