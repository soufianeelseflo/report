import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous AI Reporting Agency"
    API_V1_STR: str = "/api/v1"

    # Database Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    # Construct synchronous and asynchronous database URLs

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    DB_ECHO: bool = False # Set to True for SQLAlchemy logging

    # Agent Configuration
    REPORT_GENERATOR_INTERVAL_SECONDS: int = 15 # How often to check for new reports
    PROSPECT_RESEARCHER_INTERVAL_SECONDS: int = 3600 # How often to run research (e.g., 1 hour)
    EMAIL_MARKETER_INTERVAL_SECONDS: int = 30 # How often to check for prospects to email
    EMAIL_BATCH_SIZE: int = 5 # How many emails to send per email worker cycle

    MCOL_ANALYSIS_INTERVAL_SECONDS: int = 300 # e.g., 5 minutes
    MCOL_IMPLEMENTATION_MODE: str = "SUGGEST" # "SUGGEST" or "ATTEMPT_EXECUTE"

    LEMONSQUEEZY_API_KEY: Optional[str] = None
    LEMONSQUEEZY_STORE_ID: Optional[str] = None
    LEMONSQUEEZY_VARIANT_STANDARD: Optional[str] = None # Variant ID for $499 product
    LEMONSQUEEZY_VARIANT_PREMIUM: Optional[str] = None # Variant ID for $999 product
    LEMONSQUEEZY_WEBHOOK_SECRET: Optional[str] = None # For verifying webhooks
    # Base URL of your deployed agency (for redirects and webhooks)
    AGENCY_BASE_URL: str = "http://localhost:8000" # !! MUST BE SET IN .ENV FOR PRODUCTION !!

    # LLM Configuration (Centralize if not already done)
    LLM_INFERENCE_ENDPOINT: Optional[str] = None
    LLM_API_KEY: Optional[str] = None

    # News API Key (Example signal source)
    NEWS_API_KEY: Optional[str] = None

    # open-deep-research Configuration
    OPEN_DEEP_RESEARCH_REPO_PATH: str = "/app/open-deep-research" # Path inside the container
    OPEN_DEEP_RESEARCH_ENTRY_POINT: str = "main.js" # Adjust if needed
    NODE_EXECUTABLE_PATH: str = "node" # Assumes node is in PATH

    # Proxy Configuration (Smartproxy)
    PROXY_ENABLED: bool = True
    PROXY_HOST: str
    PROXY_PORT: int
    PROXY_USER: str
    PROXY_PASSWORD: str
    @property
    def PROXY_URL(self) -> Optional[str]:
        if self.PROXY_ENABLED and self.PROXY_HOST and self.PROXY_PORT and self.PROXY_USER and self.PROXY_PASSWORD:
            return f"http://{self.PROXY_USER}:{self.PROXY_PASSWORD}@{self.PROXY_HOST}:{self.PROXY_PORT}"
        return None

    # Placeholder for Client Contact
    CLIENT_CONTACT_PHONE: str = "+1-800-555-0199" # Placeholder

    # Security (Example: Secret key for potential JWT later)
    SECRET_KEY: str = "a_very_secret_key_change_this_in_production"

    # Email Account Encryption Key (MUST be set securely, e.g., via env var)
    # Generate a strong key: python -c 'import os; print(os.urandom(32).hex())'
    ENCRYPTION_KEY: str = "default_encryption_key_replace_me_32_bytes_long"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()