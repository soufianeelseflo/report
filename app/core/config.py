# autonomous_agency/app/core/config.py
import os
import base64
from pydantic_settings import BaseSettings
from typing import Optional, List, Union, Any, Dict # Added Any, Dict
from pydantic import validator, PostgresDsn, AnyHttpUrl, Field # Added validators, Field
import logging # Added logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Acumenis AI Agency"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "3.0.1-Prime" # Version tracking

    # --- Security ---
    SECRET_KEY: str = Field(default="a_very_secret_key_change_this_in_production_32_chars_min", min_length=32) # Used for signing, etc. MUST be strong and long
    ENCRYPTION_KEY: str # MUST be set via env (32 url-safe base64-encoded bytes)

    @validator('ENCRYPTION_KEY')
    def validate_encryption_key(cls, v):
        if not v:
            raise ValueError("ENCRYPTION_KEY must be set in environment.")
        try:
            key_bytes = v.encode()
            # Check if it's valid base64 and 32 bytes long when decoded
            if len(base64.urlsafe_b64decode(key_bytes)) != 32:
                raise ValueError("ENCRYPTION_KEY must be 32 url-safe base64-encoded bytes.")
        except Exception as e:
            raise ValueError(f"ENCRYPTION_KEY is invalid: {e}")
        return v

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
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=user,
            password=password,
            host=server,
            port=int(port), # Ensure port is integer
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
        return PostgresDsn.build(
            scheme="postgresql", # Use standard scheme for psycopg2
            username=user,
            password=password,
            host=server,
            port=int(port),
            path=f"/{db}",
        )

    DB_ECHO: bool = False # Set to True for debugging SQL queries

    # --- Agency & Web Presence ---
    AGENCY_BASE_URL: AnyHttpUrl = Field(default="http://localhost:8000", description="Public base URL (MUST include http/https)") # !! MUST BE SET IN .ENV/COOLIFY FOR PRODUCTION !!
    EMAIL_SENDER_NAME: str = "Acumenis AI Strategist" # Name used in email signatures

    # --- Payment (Lemon Squeezy) ---
    LEMONSQUEEZY_API_KEY: Optional[str] = None
    LEMONSQUEEZY_STORE_ID: Optional[str] = None
    LEMONSQUEEZY_VARIANT_STANDARD: Optional[str] = None # e.g., 12345
    LEMONSQUEEZY_VARIANT_PREMIUM: Optional[str] = None # e.g., 67890
    LEMONSQUEEZY_WEBHOOK_SECRET: Optional[str] = None # CRITICAL for security

    # --- LLM Provider Configuration ---
    LLM_PROVIDER: str = "openrouter" # Currently only 'openrouter' fully supported in agent_utils
    OPENROUTER_API_KEY: Optional[str] = None # Fallback if DB is empty
    HTTP_REFERER: Optional[str] = None # Set automatically from AGENCY_BASE_URL below if None
    X_TITLE: Optional[str] = None # Set automatically from PROJECT_NAME below if None
    STANDARD_REPORT_MODEL: str = "google/gemini-1.5-flash-latest" # Default model for standard reports/tasks
    PREMIUM_REPORT_MODEL: str = "google/gemini-1.5-pro-latest" # Default model for premium reports/tasks
    RATE_LIMIT_COOLDOWN_SECONDS: int = 120 # How long to sideline a key after a 429 error (increased default)
    API_KEY_LOW_THRESHOLD: int = 5 # MCOL warns below this number of active keys

    @validator('HTTP_REFERER', pre=True, always=True)
    def set_http_referer(cls, v, values):
        return v or str(values.get('AGENCY_BASE_URL'))

    @validator('X_TITLE', pre=True, always=True)
    def set_x_title(cls, v, values):
        return v or values.get('PROJECT_NAME')

    # --- External Tools & Internal Services ---
    OPEN_DEEP_RESEARCH_SERVICE_URL: AnyHttpUrl = Field(default="http://odr-service:3000", description="Internal ODR service URL")
    INTERNAL_ODR_API_KEY: Optional[str] = Field(None, description="Optional API key for securing internal ODR calls")

    # --- Proxies (Essential for KeyAcquirer if enabled) ---
    PROXY_URL: Optional[str] = Field(None, description="Single Proxy URL (e.g., http://user:pass@host:port)")
    PROXY_LIST: Optional[List[str]] = Field(None, description="List of Proxy URLs (overrides PROXY_URL)")

    @validator('PROXY_LIST', pre=True)
    def process_proxy_list(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(',') if p.strip()]
        return v

    # --- Key Acquirer Configuration (High Risk - Default OFF) ---
    KEY_ACQUIRER_RUN_ON_STARTUP: bool = False # Default to False for safety
    KEY_ACQUIRER_TARGET_COUNT: int = 10 # How many active keys to aim for
    KEY_ACQUIRER_CONCURRENCY: int = 2 # Reduced default concurrency
    KEY_ACQUIRER_MAX_FAILURES: int = 5 # Stop worker after this many consecutive failures
    TEMP_EMAIL_PROVIDER_URL: str = "https://inboxes.com/"
    OPENROUTER_SIGNUP_URL: str = "https://openrouter.ai/auth?callbackUrl=%2Fkeys"
    OPENROUTER_KEYS_URL: str = "https://openrouter.ai/keys"
    TEMP_EMAIL_SELECTOR: str = "#email"
    SIGNUP_EMAIL_FIELD_NAME: str = "email"
    SIGNUP_PASSWORD_FIELD_NAME: str = "password"
    SIGNUP_SUBMIT_SELECTOR: str = "button[type='submit']"
    API_KEY_DISPLAY_SELECTOR: str = "input[readonly][value^='sk-or-']"
    CAPTCHA_IMAGE_SELECTOR: str = 'img[alt="CAPTCHA"], iframe[title*="CAPTCHA"], div.captcha-image img' # Example
    CAPTCHA_INPUT_SELECTOR: str = 'input[name*="captcha"], input[id*="captcha"]' # Example

    # --- Prospecting & Marketing Configuration ---
    MAX_PROSPECTS_PER_CYCLE: int = 25 # Increased slightly
    ODR_FOR_PROSPECT_DETAILS: bool = True # Enable deeper ODR search per prospect by default
    ODR_DETAIL_DEPTH: int = 2 # Depth for detailed ODR search
    PROSPECTING_QUERIES: List[str] = [ # Default queries if not overridden
        "List B2B SaaS companies in marketing technology that received Series A funding recently",
        "Identify companies launching new AI-powered analytics products",
        "Find e-commerce platforms expanding into international markets",
        "Companies announcing large layoffs in tech sector",
        "Pharmaceutical companies with recent phase 3 trial failures",
        "Fintech startups developing blockchain-based payment solutions",
        "Renewable energy companies securing new large-scale projects",
    ]
    EMAIL_VALIDATION_ENABLED: bool = False # Set to True to enable external validation (Requires API URL/Key)
    EMAIL_VALIDATION_API_URL: Optional[AnyHttpUrl] = Field(None, description="URL of the external email validation service")
    EMAIL_VALIDATION_API_KEY: Optional[str] = None # API key if required by the service
    EMAIL_SEND_DELAY_MIN: float = 2.0 # Increased min delay
    EMAIL_SEND_DELAY_MAX: float = 7.0 # Increased max delay
    EMAIL_WARMUP_THRESHOLD: int = 7 # Slightly increased warmup threshold
    EMAIL_WARMUP_DELAY_MULTIPLIER: float = 1.8 # Increased warmup multiplier

    # --- Agent Worker Configuration ---
    REPORT_GENERATOR_INTERVAL_SECONDS: int = 5 # Check frequently for new tasks
    REPORT_GENERATOR_TIMEOUT_SECONDS: int = 720 # Timeout for the ODR service call
    PROSPECT_RESEARCHER_INTERVAL_SECONDS: int = 3600 # Run hourly
    EMAIL_MARKETER_INTERVAL_SECONDS: int = 15 # Check frequently for prospects
    EMAIL_BATCH_SIZE: int = 15 # Number of prospects to fetch per cycle
    EMAIL_ACCOUNTS_PER_BATCH: int = 8 # Number of email accounts to load per cycle
    MCOL_ANALYSIS_INTERVAL_SECONDS: int = 300 # Analyze every 5 mins
    MCOL_IMPLEMENTATION_MODE: str = "SUGGEST" # SUGGEST | EXECUTE_SAFE_CONFIG | EXECUTE_PROMPT_TUNING | EXECUTE_CODE (Risky!) - Default SUGGEST
    MCOL_PROMPT_TUNING_ENABLED: bool = False # Disable LLM prompt tuning by MCOL by default
    MCOL_MAX_STRATEGIES: int = 3 # Max strategies MCOL should generate per cycle

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore' # Allow extra fields not defined in the model

# Instantiate settings
try:
    settings = Settings()
    # Post-validation checks
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
         logger.warning("LEMONSQUEEZY_WEBHOOK_SECRET is not set. Webhook verification will fail.")
    if not settings.LEMONSQUEEZY_API_KEY:
         logger.warning("LEMONSQUEEZY_API_KEY is not set. Payment creation will fail.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize settings: {e}", exc_info=True)
    # Exit or raise to prevent app start with invalid config
    raise SystemExit(f"Configuration Error: {e}")