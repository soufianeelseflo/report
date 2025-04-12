# autonomous_agency/app/core/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional, List, Union

class Settings(BaseSettings):
    PROJECT_NAME: str = "Acumenis AI Agency"
    API_V1_STR: str = "/api/v1"

    # --- Database Configuration ---
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    DB_ECHO: bool = False

    # --- Core Agent Configuration ---
    REPORT_GENERATOR_INTERVAL_SECONDS: int = 10 # Check frequently for new tasks
    REPORT_GENERATOR_TIMEOUT_SECONDS: int = 600 # Timeout for the ODR subprocess
    PROSPECT_RESEARCHER_INTERVAL_SECONDS: int = 3600 # Run hourly
    EMAIL_MARKETER_INTERVAL_SECONDS: int = 20 # Check frequently for prospects
    EMAIL_BATCH_SIZE: int = 10 # Number of prospects to fetch per cycle
    EMAIL_ACCOUNTS_PER_BATCH: int = 5 # Number of email accounts to load per cycle
    MCOL_ANALYSIS_INTERVAL_SECONDS: int = 300 # Analyze every 5 mins
    MCOL_IMPLEMENTATION_MODE: str = "SUGGEST" # SUGGEST | EXECUTE_SAFE_CONFIG | EXECUTE_PROMPT_TUNING | EXECUTE_CODE (Risky!)

    # --- Agency & Web Presence ---
    AGENCY_BASE_URL: str = "http://localhost:8000" # !! MUST BE SET IN .ENV/COOLIFY FOR PRODUCTION !!
    EMAIL_SENDER_NAME: str = "Acumenis AI Strategist" # Name used in email signatures

    # --- Payment (Lemon Squeezy) ---
    LEMONSQUEEZY_API_KEY: Optional[str] = None
    LEMONSQUEEZY_STORE_ID: Optional[str] = None
    LEMONSQUEEZY_VARIANT_STANDARD: Optional[str] = None # e.g., 12345
    LEMONSQUEEZY_VARIANT_PREMIUM: Optional[str] = None # e.g., 67890
    LEMONSQUEEZY_WEBHOOK_SECRET: Optional[str] = None

    # --- LLM Provider Configuration ---
    LLM_PROVIDER: str = "openrouter" # Currently only 'openrouter' fully supported in agent_utils
    # OpenRouter Specific (Primary method if LLM_PROVIDER is 'openrouter')
    OPENROUTER_API_KEY: Optional[str] = None # Fallback if DB is empty
    HTTP_REFERER: Optional[str] = AGENCY_BASE_URL # Your site URL for OpenRouter header
    X_TITLE: Optional[str] = PROJECT_NAME # App name for OpenRouter header
    STANDARD_REPORT_MODEL: str = "google/gemini-1.5-flash-latest" # Default model for standard reports/tasks
    PREMIUM_REPORT_MODEL: str = "google/gemini-1.5-pro-latest" # Default model for premium reports/tasks
    RATE_LIMIT_COOLDOWN_SECONDS: int = 60 # How long to sideline a key after a 429 error
    API_KEY_LOW_THRESHOLD: int = 5 # MCOL warns below this number of active keys

    # --- External Tools ---
    # open-deep-research
    OPEN_DEEP_RESEARCH_REPO_PATH: str = "/app/open-deep-research-main" # Adjusted path based on file tree
    OPEN_DEEP_RESEARCH_ENTRY_POINT: str = "app/page.tsx" # Verify this - likely needs a build step or different entry
    NODE_EXECUTABLE_PATH: str = "/usr/local/bin/node" # Example path in container, adjust based on Dockerfile
    OPEN_DEEP_RESEARCH_TIMEOUT: int = 300 # Timeout for ODR subprocess in ProspectResearcher

    # --- Proxies (Essential for KeyAcquirer) ---
    # Option 1: Single Proxy URL
    PROXY_URL: Optional[str] = None # e.g., "http://user:pass@host:port"
    # Option 2: List of Proxy URLs (Takes precedence over PROXY_URL if set)
    PROXY_LIST: Optional[List[str]] = None # e.g., ["http://u1:p1@h1:p1", "http://u2:p2@h2:p2"]

    # --- Key Acquirer Configuration (High Risk - Use with Caution) ---
    KEY_ACQUIRER_RUN_ON_STARTUP: bool = False # Default to False for safety
    KEY_ACQUIRER_TARGET_COUNT: int = 10 # How many active keys to aim for
    KEY_ACQUIRER_CONCURRENCY: int = 3 # How many acquisition attempts to run in parallel
    KEY_ACQUIRER_MAX_FAILURES: int = 5 # Stop worker after this many consecutive failures
    # URLs and Selectors (Highly likely to change - keep updated!)
    TEMP_EMAIL_PROVIDER_URL: str = "https://inboxes.com/"
    OPENROUTER_SIGNUP_URL: str = "https://openrouter.ai/auth?callbackUrl=%2Fkeys"
    OPENROUTER_KEYS_URL: str = "https://openrouter.ai/keys"
    TEMP_EMAIL_SELECTOR: str = "#email"
    SIGNUP_EMAIL_FIELD_NAME: str = "email"
    SIGNUP_PASSWORD_FIELD_NAME: str = "password"
    SIGNUP_SUBMIT_SELECTOR: str = "button[type='submit']"
    API_KEY_DISPLAY_SELECTOR: str = "input[readonly][value^='sk-or-']"

    # --- Prospecting & Marketing Configuration ---
    MAX_PROSPECTS_PER_CYCLE: int = 20
    ODR_FOR_PROSPECT_DETAILS: bool = False # Whether to run a deep ODR search per prospect
    PROSPECTING_QUERIES: List[str] = [ # Default queries if not overridden
        "List B2B SaaS companies in marketing technology that received Series A funding recently",
        "Identify companies launching new AI-powered analytics products",
        "Find e-commerce platforms expanding into international markets",
        "Companies announcing large layoffs in tech sector",
        "Pharmaceutical companies with recent phase 3 trial failures"
    ]
    EMAIL_VALIDATION_ENABLED: bool = False # Set to True to enable external validation
    EMAIL_VALIDATION_API_URL: Optional[str] = None # URL of the validation service
    EMAIL_VALIDATION_API_KEY: Optional[str] = None # API key if required by the service
    EMAIL_SEND_DELAY_MIN: float = 1.0
    EMAIL_SEND_DELAY_MAX: float = 5.0
    EMAIL_WARMUP_THRESHOLD: int = 5
    EMAIL_WARMUP_DELAY_MULTIPLIER: float = 1.5

    # --- Security ---
    SECRET_KEY: str = "a_very_secret_key_change_this_in_production" # Used for signing, etc.
    ENCRYPTION_KEY: str # MUST be set via env (32 url-safe base64-encoded bytes)

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore' # Allow extra fields not defined in the model

# Instantiate settings
settings = Settings()

# Post-process PROXY_LIST from potential comma-separated string in env var
if isinstance(settings.PROXY_LIST, str):
    settings.PROXY_LIST = [p.strip() for p in settings.PROXY_LIST.split(',') if p.strip()]
elif settings.PROXY_LIST is None:
    settings.PROXY_LIST = []

# Validate ENCRYPTION_KEY format (basic check)
if not settings.ENCRYPTION_KEY or len(settings.ENCRYPTION_KEY) < 32:
     print("CRITICAL WARNING: ENCRYPTION_KEY is missing or too short in environment variables. Data encryption will fail.")
     # Consider raising an exception here to prevent startup without a valid key
     # raise ValueError("ENCRYPTION_KEY is missing or invalid.")