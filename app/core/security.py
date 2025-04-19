# autonomous_agency/app/core/security.py
import base64
import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# CORRECTED IMPORTS: Replaced "Nexus Plan.app" with "app"
try:
    from app.core.config import settings
except ImportError:
    print("[Security] WARNING: Using fallback imports.")
    from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Key Derivation (Use only if ENCRYPTION_KEY is not pre-generated correctly) ---
# WARNING: Using a fixed salt is not ideal for production.
# Prefer generating a strong 32-byte URL-safe base64 key and setting it directly.
FIXED_SALT = b'NexusPlan_salt_16' # MUST be 16 bytes - Replace in production!

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derives a Fernet key from a password string using PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600000, # Increased iterations (OWASP recommendation 2023+)
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    logger.info("Derived Fernet key from ENCRYPTION_KEY setting.")
    return key

# --- Initialize Fernet ---
fernet_instance: Optional[Fernet] = None
try:
    # Assume settings.ENCRYPTION_KEY is validated by config loader
    key_bytes = settings.ENCRYPTION_KEY.encode()
    # Check if it's already a valid key (32 URL-safe base64 bytes)
    # The validator in config.py should handle this check now.
    fernet_key = key_bytes
    logger.info("Using provided ENCRYPTION_KEY for Fernet.")
    fernet_instance = Fernet(fernet_key)
except ValueError as e:
    # This should ideally not happen if config validation works, but handle defensively
    logger.critical(f"CRITICAL: ENCRYPTION_KEY validation failed during Fernet initialization: {e}. Security features will fail.")
    # Optionally, attempt derivation as a last resort? Risky.
    # logger.warning("Attempting to derive key from potentially invalid ENCRYPTION_KEY setting...")
    # try:
    #     fernet_key = _derive_key(settings.ENCRYPTION_KEY, FIXED_SALT)
    #     fernet_instance = Fernet(fernet_key)
    # except Exception as deriv_e:
    #     logger.critical(f"FATAL: Failed to derive key after validation error: {deriv_e}. Cannot proceed.")
    #     fernet_instance = None # Ensure it's None if init fails
except Exception as e:
    logger.critical(f"FATAL: Unexpected error initializing Fernet with ENCRYPTION_KEY: {e}", exc_info=True)
    fernet_instance = None


def encrypt_data(data: str) -> Optional[str]:
    """Encrypts a string and returns it as a string, or None on failure."""
    if not fernet_instance:
        logger.error("Encryption failed: Fernet not initialized (check ENCRYPTION_KEY).")
        return None
    if not data:
        return "" # Encrypting empty string is empty string
    try:
        return fernet_instance.encrypt(data.encode()).decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}", exc_info=True)
        return None

def decrypt_data(encrypted_data: str) -> Optional[str]:
    """Decrypts a string and returns it, or None on failure."""
    if not fernet_instance:
        logger.error("Decryption failed: Fernet not initialized (check ENCRYPTION_KEY).")
        return None
    if not encrypted_data:
        return "" # Decrypting empty string is empty string
    try:
        return fernet_instance.decrypt(encrypted_data.encode()).decode()
    except InvalidToken:
        logger.error("Decryption failed: Invalid token (likely wrong key or corrupted data).")
        return None
    except Exception as e:
        logger.error(f"Decryption error: {e}", exc_info=True)
        return None