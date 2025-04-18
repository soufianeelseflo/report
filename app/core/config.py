# autonomous_agency/app/core/config.py
import os
import base64
import re # Import re for proxy list validation
from pydantic_settings import BaseSettings
from typing import Optional, List, Union, Any, Dict # Added Any, Dict
from pydantic import validator, PostgresDsn, AnyHttpUrl, Field, ValidationError # Added ValidationError
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Nexus Plan AI Agency"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "3.1-Velocity" # Version tracking

    # --- Security ---
    SECRET_KEY: str = Field(default="change_this_in_production_a_very_secret_key_32_chars_min", min_length=32) # Used for signing, etc. MUST be strong and long
    ENCRYPTION_KEY: str # MUST be set via env (32 url-safe base64-encoded bytes)

    @validator('ENCRYPTION_KEY')
    def validate_encryption_key(cls, v):
        if not v:
            raise ValueError("ENCRYPTION_KEY must be set in environment.")
        try:
            # Ensure correct padding for URL-safe base64
            # Python's base64 decoder handles padding automatically if needed,
            # but requires the input length to be valid. Fernet expects exactly 32 bytes decoded.
            key_bytes = base64.urlsafe_b64decode(v + '==') # Add padding just in case, decoder ignores extra if not needed
            if len(key_bytes) != 32:
                raise ValueError(f"ENCRYPTION_KEY must decode to 32 bytes, but got {len(key_bytes)} bytes.")
        except (TypeError, ValueError, base64.binascii.Error) as e: # Catch specific base64 errors
            # Raise a ValueError that Pydantic understands, but avoid echoing the key
            raise ValueError(f"ENCRYPTION_KEY is invalid or incorrectly formatted: {e}")
        return v # Return original value if valid

    # --- Database Configuration ---
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    ASYNC_DATABASE_URL: Optional[PostgresDsn] = None
    DATABASE_URL: Optional[PostgresDsn] = None # For Alembic sync usage

    @validator('ASYNC_DATABASE_URL', pre=True, always=True)
    def assemble_async_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        server = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        if not all([user, password, server, port, db]):
             raise ValueError("Missing PostgreSQL connection details.")
        # Ensure port is integer before building DSN
        try:
            port_int = int(port)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid POSTGRES_PORT: '{port}'. Must be an integer.")
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=user,
            password=password,
            host=server,
            port=port_int,
            path=f"/{db}",
        )

    @validator('DATABASE_URL', pre=True, always=True)
    def assemble_sync_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        server = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        if not all([user, password, server, port, db]):
             raise ValueError("Missing PostgreSQL connection details.")
        # Ensure port is integer before building DSN
        try:
            port_int = int(port)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid POSTGRES_PORT: '{port}'. Must be an integer.")
        return PostgresDsn.build(
            scheme="postgresql", # Use standard scheme for psycopg2
            username=user,
            password=password,
            host=server,
            port=port_int,
            path=f"/{db}",
        )

    DB_ECHO: bool = False # Set to True for debugging SQL queries

    # --- Agency & Web Presence ---
    AGENCY_BASE_URL: AnyHttpUrl = Field(..., description="Public base URL (MUST include http/https)") # !! MUST BE SET IN .ENV !!

    # --- Payment (Lemon Squeezy) ---
    LEMONSQUEEZY_API_KEY: str # Changed to required
    LEMONSQUEEZY_STORE_ID: str # Changed to required
    LEMONSQUEEZY_VARIANT_STANDARD: str # Changed to required
    LEMONSQUEEZY_VARIANT_PREMIUM: str # Changed to required
    LEMONSQUEEZY_WEBHOOK_SECRET: str # Changed to required

    # --- LLM Provider Configuration ---
    LLM_PROVIDER: str = "openrouter" # Keep as only supported option for now
    OPENROUTER_API_KEY: str # Changed to required - The single key
    STANDARD_REPORT_MODEL: str = "google/gemini-1.5-flash-latest" # Default model for standard reports/tasks
    PREMIUM_REPORT_MODEL: str = "google/gemini-1.5-pro-latest" # Default model for premium reports/tasks
    RATE_LIMIT_COOLDOWN_SECONDS: int = 60 # Shorter cooldown, assume manual fix
    API_KEY_LOW_THRESHOLD: int = 1 # MCOL warns if the single key fails (effectively count < 1)

    # --- External Tools & Internal Services ---
    OPEN_DEEP_RESEARCH_SERVICE_URL: AnyHttpUrl = Field(default="http://odr-service:3000", description="Internal ODR service URL")

    # --- Proxies ---
    PROXY_ENABLED: bool = False # Default to False if not set
    PROXY_HOST: Optional[str] = None
    PROXY_PORT: Optional[int] = None
    PROXY_USER: Optional[str] = None
    PROXY_PASSWORD: Optional[str] = None
    PROXY_LIST: Optional[List[str]] = Field(None, description="List of Proxy URLs (overrides single proxy if set)") # Keep list option

    @validator('PROXY_LIST', pre=True)
    def process_proxy_list(cls, v):
        if isinstance(v, str):
            # Ensure format user:pass@host:port
            valid_proxies = []
            for p in v.split(','):
                p_strip = p.strip()
                # Regex updated to handle optional scheme
                if re.match(r"^(?:(http|https|socks\d?)://)?.*:.*@.*:\d+", p_strip):
                    # Prepend http:// if scheme is missing (common for user:pass@host:port format)
                    if not re.match(r"^(http|https|socks\d?)://", p_strip):
                        p_strip = f"http://{p_strip}"
                        logger.debug(f"Prepended 'http://' to proxy: {p_strip}")
                    valid_proxies.append(p_strip)
                else:
                    logger.warning(f"Invalid proxy format in PROXY_LIST skipped: {p_strip}")
            return valid_proxies
        elif isinstance(v, list): # Handle case where it might already be a list
            return v
        return None # Return None if input is neither string nor list

    # --- Prospecting & Marketing Configuration ---
    MAX_PROSPECTS_PER_CYCLE: int = 50 # Increased prospecting volume
    ODR_FOR_PROSPECT_DETAILS: bool = True # Keep detailed ODR search
    ODR_DETAIL_DEPTH: int = 1 # Reduce depth slightly for speed/cost
    PROSPECTING_QUERIES: List[str] = [ # Keep defaults, user can override in .env
        "List B2B SaaS companies in marketing technology that received Series A funding recently",
        "Identify companies launching new AI-powered analytics products",
        "Find e-commerce platforms expanding into international markets",
        "Companies announcing large layoffs in tech sector",
        "Pharmaceutical companies with recent phase 3 trial failures",
        "Fintech startups developing blockchain-based payment solutions",
        "Renewable energy companies securing new large-scale projects",
    ]
    EMAIL_VALIDATION_ENABLED: bool = False # Keep disabled unless user explicitly configures API
    EMAIL_VALIDATION_API_URL: Optional[AnyHttpUrl] = None
    EMAIL_VALIDATION_API_KEY: Optional[str] = None
    EMAIL_SEND_DELAY_MIN: float = 0.5 # Drastically reduced min delay
    EMAIL_SEND_DELAY_MAX: float = 2.0 # Drastically reduced max delay
    EMAIL_WARMUP_THRESHOLD: int = 0 # No warmup
    EMAIL_WARMUP_DELAY_MULTIPLIER: float = 1.0 # No warmup multiplier

    # --- Agent Worker Configuration ---
    REPORT_GENERATOR_INTERVAL_SECONDS: int = 2 # Check very frequently
    REPORT_GENERATOR_TIMEOUT_SECONDS: int = 600 # Keep timeout reasonable
    PROSPECT_RESEARCHER_INTERVAL_SECONDS: int = 1800 # Run every 30 mins
    EMAIL_MARKETER_INTERVAL_SECONDS: int = 5 # Check very frequently
    EMAIL_BATCH_SIZE: int = 50 # Fetch larger batches for mass email
    EMAIL_ACCOUNTS_PER_BATCH: int = 20 # Load more accounts if available
    MCOL_ANALYSIS_INTERVAL_SECONDS: int = 120 # Analyze every 2 mins
    MCOL_IMPLEMENTATION_MODE: str = "EXECUTE_SAFE_CONFIG" # Default to execution
    MCOL_PROMPT_TUNING_ENABLED: bool = False # Keep disabled
    MCOL_MAX_STRATEGIES: int = 1 # Focus on executing one thing fast

    # --- Email Account Credentials (Reference - Add to DB manually) ---
    EMAIL_SENDER_NAME: str = "Nexus Plan AI Strategist" # Default sender name

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore' # Allow extra fields not defined in the model

# Instantiate settings
try:
    settings = Settings()
    # Post-validation checks (optional)
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
         logger.warning("LEMONSQUEEZY_WEBHOOK_SECRET is not set. Webhook verification will fail.")
    if not settings.LEMONSQUEEZY_API_KEY:
         logger.warning("LEMONSQUEEZY_API_KEY is not set. Payment creation will fail.")
    if not settings.OPENROUTER_API_KEY:
         logger.critical("FATAL: OPENROUTER_API_KEY (your single key) is not set.")
         raise ValueError("OPENROUTER_API_KEY is mandatory.")
    if settings.PROXY_ENABLED and not (settings.PROXY_LIST or (settings.PROXY_HOST and settings.PROXY_PORT and settings.PROXY_USER and settings.PROXY_PASSWORD)):
        logger.warning("PROXY_ENABLED is True, but no valid PROXY_LIST or single proxy credentials found.")

except ValidationError as e:
    # --- MODIFIED EXCEPTION HANDLING ---
    error_summary = []
    is_encryption_key_error = False
    for error in e.errors():
        loc = ".".join(map(str, error['loc']))
        msg = error['msg']
        if loc == 'ENCRYPTION_KEY':
            is_encryption_key_error = True
            # Log generic error for encryption key, avoid logging the value
            error_summary.append(f"FATAL: ENCRYPTION_KEY validation failed: {msg}. Ensure it is a 32-byte URL-safe base64 encoded string.")
        else:
            # Log other validation errors normally
            error_summary.append(f"Config Error [{loc}]: {msg}")

    log_message = "\n".join(error_summary)
    logger.critical(f"FATAL: Failed to initialize settings:\n{log_message}")
    if is_encryption_key_error:
        logger.critical("Please generate a new valid ENCRYPTION_KEY and set it in your environment.")
    raise SystemExit(f"Configuration Error(s):\n{log_message}")
    # --- END MODIFIED EXCEPTION HANDLING ---

except Exception as e: # Catch other potential errors during init
    logger.critical(f"FATAL: Unexpected error initializing settings: {e}", exc_info=True)
    raise SystemExit(f"Unexpected Configuration Error: {e}")