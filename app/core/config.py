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
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    DB_ECHO: bool = False

    # Agent Configuration
    REPORT_GENERATOR_INTERVAL_SECONDS: int = 15
    PROSPECT_RESEARCHER_INTERVAL_SECONDS: int = 3600
    EMAIL_MARKETER_INTERVAL_SECONDS: int = 30
    EMAIL_BATCH_SIZE: int = 5
    MCOL_ANALYSIS_INTERVAL_SECONDS: int = 300
    MCOL_IMPLEMENTATION_MODE: str = "SUGGEST" # Start safe

    # --- Agency URL (CRITICAL) ---
    AGENCY_BASE_URL: str = "http://localhost:8000" # !! MUST BE SET IN .ENV/COOLIFY FOR PRODUCTION !!

    # --- Payment (Lemon Squeezy) ---
    LEMONSQUEEZY_API_KEY: Optional[str] = None
    LEMONSQUEEZY_STORE_ID: Optional[str] = None
    LEMONSQUEEZY_VARIANT_STANDARD: Optional[str] = None
    LEMONSQUEEZY_VARIANT_PREMIUM: Optional[str] = None
    LEMONSQUEEZY_WEBHOOK_SECRET: Optional[str] = None

    # --- LLM Provider (OpenRouter Recommended) ---
    LLM_PROVIDER: str = "openrouter" # 'openrouter' or 'openai' or other
    # OpenRouter Specific
    OPENROUTER_API_KEY: Optional[str] = None
    HTTP_REFERER: Optional[str] = None # Optional: Your site URL for OpenRouter header
    X_TITLE: Optional[str] = PROJECT_NAME # Optional: App name for OpenRouter header
    # Generic / OpenAI Specific
    LLM_API_KEY: Optional[str] = None # Use if provider is not OpenRouter
    LLM_INFERENCE_ENDPOINT: Optional[str] = None # Use if provider is not OpenRouter

    # --- Signal Sources ---
    NEWS_API_KEY: Optional[str] = None

    # --- open-deep-research ---
    OPEN_DEEP_RESEARCH_REPO_PATH: str = "/app/open-deep-research"
    OPEN_DEEP_RESEARCH_ENTRY_POINT: str = "main.js" # Verify this!
    NODE_EXECUTABLE_PATH: str = "node"

    # --- Proxies ---
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

    # --- Security ---
    SECRET_KEY: str = "a_very_secret_key_change_this_in_production" # For potential future use
    ENCRYPTION_KEY: str # MUST be set via env

    class Config:
        # Load .env file from the root directory where docker-compose is run
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        # Allow extra fields if needed, though BaseSettings usually handles this
        # extra = 'ignore'

settings = Settings()