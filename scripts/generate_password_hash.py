#!/usr/bin/env python3
"""
Generate an Argon2 password hash for RainRAG authentication.

This script helps administrators create secure password hashes for the
RAINRAG_PASSWORD_HASH environment variable.

Usage:
    python scripts/generate_password_hash.py

Security:
    - Uses Argon2 with secure parameters (memory-hard, resistant to attacks)
    - Passwords are not logged or stored
    - Hash can be safely stored in environment variables
"""

import getpass
import sys


try:
    from argon2 import PasswordHasher
except ImportError:
    print("ERROR: argon2-cffi library not installed")
    print("Please install it with: pip install argon2-cffi")
    sys.exit(1)


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength against security best practices.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if len(password) < 12:
        issues.append("Password should be at least 12 characters long")

    if not any(c.isupper() for c in password):
        issues.append("Password should contain at least one uppercase letter")

    if not any(c.islower() for c in password):
        issues.append("Password should contain at least one lowercase letter")

    if not any(c.isdigit() for c in password):
        issues.append("Password should contain at least one number")

    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("Password should contain at least one special character")

    return len(issues) == 0, issues


def generate_hash(password: str) -> str:
    """
    Generate an Argon2 hash for the given password.

    Args:
        password: Plain text password

    Returns:
        Argon2 hash as a string
    """
    # Initialize Argon2 hasher with secure parameters
    hasher = PasswordHasher(
        time_cost=2,  # Number of iterations (moderate for web app)
        memory_cost=102400,  # Memory usage in KiB (100MB)
        parallelism=8,  # Number of parallel threads
        hash_len=32,  # Hash length in bytes
        salt_len=16,  # Salt length in bytes
    )

    return hasher.hash(password)


def main():
    """Main function to generate password hash."""
    print("=" * 70)
    print("RainRAG Password Hash Generator")
    print("=" * 70)
    print()
    print("This tool generates a secure Argon2 hash for your password.")
    print("The hash will be used in the RAINRAG_PASSWORD_HASH environment variable.")
    print()
    print("Password Requirements:")
    print("  - At least 12 characters long")
    print("  - Contains uppercase and lowercase letters")
    print("  - Contains at least one number")
    print("  - Contains at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)")
    print()

    # Get password from user
    while True:
        password = getpass.getpass("Enter password: ")
        password_confirm = getpass.getpass("Confirm password: ")

        if password != password_confirm:
            print("[ERROR] Passwords do not match. Please try again.\n")
            continue

        if not password:
            print("[ERROR] Password cannot be empty. Please try again.\n")
            continue

        # Validate password strength
        is_valid, issues = validate_password_strength(password)
        if not is_valid:
            print("\n[WARNING]  Password does not meet security requirements:")
            for issue in issues:
                print(f"   - {issue}")
            print()

            response = input("Use this password anyway? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                print("Please try again.\n")
                continue

        break

    # Generate hash
    print("\nGenerating secure hash...")
    password_hash = generate_hash(password)

    # Display results
    print("\n" + "=" * 70)
    print("[OK] Password hash generated successfully!")
    print("=" * 70)
    print("\nAdd this to your environment variables:")
    print()
    print(f"export RAINRAG_PASSWORD_HASH='{password_hash}'")
    print()
    print("For Docker:")
    print(f"RAINRAG_PASSWORD_HASH={password_hash}")
    print()
    print("For .env file:")
    print(f"RAINRAG_PASSWORD_HASH={password_hash}")
    print()
    print("=" * 70)
    print("[WARNING]  SECURITY REMINDERS:")
    print("  - Store this hash securely (e.g., in a password manager)")
    print("  - Never commit the hash to version control")
    print("  - Use different passwords for different environments")
    print("  - Rotate passwords regularly")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[ERROR] Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
