import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.core.config import settings
import os

# Ensure the key is the correct length for Fernet (32 url-safe base64-encoded bytes)
# We derive a key from the setting using PBKDF2HMAC for better practice if the key isn't pre-generated correctly.
def _derive_key(password: str, salt: bytes) -> bytes:
    """Derives a Fernet key from a password string."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000, # OWASP recommendation as of 2023
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

# Use a fixed salt stored alongside the key or derive from a setting.
# For simplicity here, using a fixed salt. In production, manage this securely.
# DO NOT use a hardcoded salt like this in real production without careful consideration.
# It's better if the salt is unique per installation or stored securely.
FIXED_SALT = b'a_fixed_salt_replace_me_16_bytes' # MUST be 16 bytes

try:
    # Attempt to use the key directly if it's already valid base64
    key_bytes = settings.ENCRYPTION_KEY.encode()
    if len(base64.urlsafe_b64decode(key_bytes)) == 32:
         fernet_key = key_bytes
         print("Using provided ENCRYPTION_KEY directly.")
    else:
        print("Provided ENCRYPTION_KEY is not valid base64 or wrong length. Deriving key...")
        fernet_key = _derive_key(settings.ENCRYPTION_KEY, FIXED_SALT)
except Exception:
    print("Error processing ENCRYPTION_KEY. Deriving key...")
    fernet_key = _derive_key(settings.ENCRYPTION_KEY, FIXED_SALT)

f = Fernet(fernet_key)

def encrypt_data(data: str) -> str:
    """Encrypts a string and returns it as a string."""
    if not data:
        return ""
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    """Decrypts a string and returns it."""
    if not encrypted_data:
        return ""
    try:
        return f.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        print(f"Error decrypting data: {e}") # Log properly
        # Handle error appropriately - return empty string, raise exception?
        return "" # Or raise DecryptionError("Failed to decrypt")

# Example usage (can be tested independently):
# encrypted = encrypt_data("my secret password")
# print(encrypted)
# decrypted = decrypt_data(encrypted)
# print(decrypted)