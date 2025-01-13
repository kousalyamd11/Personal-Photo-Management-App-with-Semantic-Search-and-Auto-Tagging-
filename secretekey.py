import os
import secrets

# Generate a secure random key
secret_key = secrets.token_hex(32)  # 32 bytes = 64 characters
print(secret_key)