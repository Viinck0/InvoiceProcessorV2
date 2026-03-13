"""
Security Module for Invoice Processing System

Centralized security features:
- Data encryption/decryption (Fernet)
- LLM output sanitization (anti-injection)
- Input file validation (MIME + size)
- Configuration integrity verification (SHA-256)
- Audit logging
- Data retention / cleanup

Usage:
    from core.security import (
        DataEncryptor, sanitize_llm_output, validate_input_file,
        verify_config_integrity, AuditLogger, cleanup_old_files
    )
"""

import os
import re
import json
import hashlib
import logging
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Encryption: Fernet symmetric encryption
# ─────────────────────────────────────────────

# Try to import cryptography; graceful fallback if not installed
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning(
        "⚠️ cryptography not installed. Encryption disabled. "
        "Install with: pip install cryptography"
    )

# Default key file location (next to this module's parent directory)
_DEFAULT_KEY_PATH = Path(__file__).parent.parent / ".security_key"


class DataEncryptor:
    """
    Encrypts and decrypts sensitive data using Fernet (AES-128-CBC + HMAC).

    Key management:
      - Auto-generates a key on first use and saves to .security_key
      - In production, replace with KMS or environment variable

    Usage:
        enc = DataEncryptor()
        encrypted = enc.encrypt_text("sensitive data")
        decrypted = enc.decrypt_text(encrypted)
    """

    def __init__(self, key_path: Optional[Path] = None):
        self.key_path = key_path or _DEFAULT_KEY_PATH
        self._fernet = None
        if HAS_CRYPTO:
            self._init_fernet()

    def _init_fernet(self):
        """Load or generate encryption key."""
        key = self._load_or_create_key()
        if key:
            self._fernet = Fernet(key)

    def _load_or_create_key(self) -> Optional[bytes]:
        """Load key from file or generate a new one."""
        try:
            if self.key_path.exists():
                key = self.key_path.read_bytes().strip()
                # Validate key format
                Fernet(key)
                return key
            else:
                key = Fernet.generate_key()
                self.key_path.write_bytes(key)
                # Restrict file permissions (best effort on Windows)
                try:
                    os.chmod(str(self.key_path), 0o600)
                except OSError:
                    pass
                logger.info(f"🔑 New encryption key generated: {self.key_path}")
                return key
        except Exception as e:
            logger.error(f"❌ Encryption key error: {e}")
            return None

    @property
    def available(self) -> bool:
        """Check if encryption is available."""
        return self._fernet is not None

    def encrypt_text(self, plaintext: str) -> Optional[bytes]:
        """Encrypt a text string. Returns encrypted bytes or None if unavailable."""
        if not self._fernet:
            logger.warning("Encryption not available — storing plaintext")
            return None
        try:
            return self._fernet.encrypt(plaintext.encode("utf-8"))
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None

    def decrypt_text(self, encrypted_data: bytes) -> Optional[str]:
        """Decrypt encrypted bytes back to text string."""
        if not self._fernet:
            logger.warning("Decryption not available")
            return None
        try:
            return self._fernet.decrypt(encrypted_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def encrypt_file(self, source_path: Path, dest_path: Optional[Path] = None) -> Optional[Path]:
        """Encrypt a file and save with .enc extension."""
        if not self._fernet:
            return None
        try:
            plaintext = source_path.read_text(encoding="utf-8")
            encrypted = self.encrypt_text(plaintext)
            if encrypted is None:
                return None

            out_path = dest_path or source_path.with_suffix(source_path.suffix + ".enc")
            out_path.write_bytes(encrypted)
            logger.debug(f"🔒 Encrypted: {source_path.name} → {out_path.name}")
            return out_path
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return None

    def decrypt_file(self, encrypted_path: Path) -> Optional[str]:
        """Decrypt an encrypted file and return the plaintext content."""
        if not self._fernet:
            return None
        try:
            encrypted_data = encrypted_path.read_bytes()
            return self.decrypt_text(encrypted_data)
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return None


# ─────────────────────────────────────────────
# LLM Output Sanitization
# ─────────────────────────────────────────────

# Patterns that could indicate injection attempts in LLM output
_INJECTION_PATTERNS = [
    # JSON / code injection
    (re.compile(r'\{[^}]{50,}\}', re.DOTALL), "[FILTERED_JSON]"),
    # Code block injection
    (re.compile(r'```[\s\S]*?```', re.DOTALL), "[FILTERED_CODE]"),
    # System prompt leakage patterns
    (re.compile(r'(?i)(system\s*prompt|instructions?:\s*you\s+are)', re.DOTALL), "[FILTERED]"),
    # HTML/script injection
    (re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE), "[FILTERED_SCRIPT]"),
    (re.compile(r'<iframe[^>]*>.*?</iframe>', re.DOTALL | re.IGNORECASE), "[FILTERED_IFRAME]"),
]

# Maximum reasoning length to prevent data exfiltration through very long outputs
_MAX_REASONING_LENGTH = 2000


def sanitize_llm_output(
    text: str,
    max_length: int = _MAX_REASONING_LENGTH,
    is_reasoning: bool = True
) -> str:
    """
    Sanitize LLM output to prevent prompt injection and data leakage.

    When is_reasoning=True (default, for human-readable 'reasoning' fields):
    - Removes embedded JSON structures (potential injection payloads)
    - Removes code blocks (could contain executable instructions)
    - Removes system prompt leakage patterns
    - Removes HTML/script tags
    - Truncates excessive length

    When is_reasoning=False (for structured/machine data like dicts, JSON):
    - Only removes dangerous HTML/script injection
    - Only removes control characters
    - Does NOT truncate or strip JSON/code blocks

    Args:
        text: Raw text from LLM
        max_length: Maximum allowed length (default: 2000 chars)
        is_reasoning: If True, apply full sanitization incl. JSON stripping
                      and truncation. If False, only strip HTML/scripts.

    Returns:
        Sanitized text safe for storage and display
    """
    if not text:
        return ""

    sanitized = text

    if is_reasoning:
        # Full sanitization: strip JSON, code blocks, prompt leakage, HTML
        for pattern, replacement in _INJECTION_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        # Truncate if too long (only for human-readable reasoning)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "… [TRUNCATED]"
    else:
        # Lightweight sanitization: only strip dangerous HTML/script tags
        # Preserve JSON and code blocks needed for machine processing
        for pattern, replacement in _INJECTION_PATTERNS:
            # Only apply HTML/script patterns (last two in the list)
            if replacement in ("[FILTERED_SCRIPT]", "[FILTERED_IFRAME]"):
                sanitized = pattern.sub(replacement, sanitized)

    # Always remove null bytes and control characters (except newline/tab)
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)

    return sanitized.strip()


# ─────────────────────────────────────────────
# Input File Validation
# ─────────────────────────────────────────────

# Allowed MIME types for invoice documents
_ALLOWED_MIME_TYPES = {
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/tiff',
    'image/bmp',
    'image/gif',
}

# Extension-to-MIME mapping (fallback if python-magic not available)
_EXTENSION_MIME_MAP = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
    '.gif': 'image/gif',
}

# Default maximum file size: 50 MB
_DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024

# Try to import python-magic for MIME detection
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


def validate_input_file(
    file_path: Path,
    max_size_bytes: int = _DEFAULT_MAX_FILE_SIZE,
    allowed_mimes: set = None
) -> Tuple[bool, str]:
    """
    Validate an input file before processing.

    Checks:
    1. File exists and is a regular file
    2. File size within limits
    3. File extension is allowed
    4. MIME type matches (if python-magic is installed)

    Args:
        file_path: Path to the file to validate
        max_size_bytes: Maximum allowed file size in bytes
        allowed_mimes: Set of allowed MIME types (uses defaults if None)

    Returns:
        Tuple (is_valid: bool, reason: str)
    """
    allowed = allowed_mimes or _ALLOWED_MIME_TYPES

    # 1. Existence check
    if not file_path.exists():
        return False, f"Soubor neexistuje: {file_path}"

    if not file_path.is_file():
        return False, f"Není běžný soubor: {file_path}"

    # 2. Size check
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "Soubor je prázdný (0 bajtů)"

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        return False, f"Soubor příliš velký: {size_mb:.1f} MB (limit: {limit_mb:.0f} MB)"

    # 3. Extension check
    ext = file_path.suffix.lower()
    if ext not in _EXTENSION_MIME_MAP:
        return False, f"Nepodporovaná přípona: {ext}"

    # 4. MIME type check (if python-magic installed)
    if HAS_MAGIC:
        try:
            detected_mime = magic.from_file(str(file_path), mime=True)
            if detected_mime not in allowed:
                return False, (
                    f"Neplatný typ souboru: {detected_mime} "
                    f"(přípona: {ext}, očekáváno: {_EXTENSION_MIME_MAP.get(ext, '?')})"
                )
        except Exception as e:
            logger.warning(f"MIME detection failed, falling back to extension check: {e}")
    else:
        # Fallback: Magic header check for common file types
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)

            if ext == '.pdf' and not header.startswith(b'%PDF'):
                return False, "Soubor nemá platnou PDF hlavičku"
            elif ext in ('.jpg', '.jpeg') and not header.startswith(b'\xff\xd8\xff'):
                return False, "Soubor nemá platnou JPEG hlavičku"
            elif ext == '.png' and not header.startswith(b'\x89PNG'):
                return False, "Soubor nemá platnou PNG hlavičku"
        except IOError as e:
            return False, f"Nelze přečíst soubor: {e}"

    return True, "OK"


# ─────────────────────────────────────────────
# Configuration Integrity Verification
# ─────────────────────────────────────────────

_HASH_FILE_SUFFIX = ".sha256"


def compute_config_hash(config_path: Path) -> str:
    """Compute SHA-256 hash of a configuration file."""
    with open(config_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_config_hash(config_path: Path) -> str:
    """
    Compute and save SHA-256 hash of config file.

    Saves hash to a .sha256 file alongside the config.
    Returns the computed hash.
    """
    config_hash = compute_config_hash(config_path)
    hash_path = config_path.with_suffix(config_path.suffix + _HASH_FILE_SUFFIX)
    hash_path.write_text(config_hash, encoding="utf-8")
    logger.info(f"🔏 Config hash saved: {hash_path.name}")
    return config_hash


def verify_config_integrity(config_path: Path) -> Tuple[bool, str]:
    """
    Verify configuration file integrity against stored SHA-256 hash.

    If no hash file exists, creates one (first-run initialization).

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple (is_valid: bool, message: str)
    """
    if not config_path.exists():
        return False, f"Config file not found: {config_path}"

    hash_path = config_path.with_suffix(config_path.suffix + _HASH_FILE_SUFFIX)

    current_hash = compute_config_hash(config_path)

    if not hash_path.exists():
        # First run — save initial hash
        save_config_hash(config_path)
        return True, "Config hash initialized (first run)"

    try:
        stored_hash = hash_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, f"Cannot read hash file: {e}"

    if current_hash == stored_hash:
        return True, "Config integrity verified ✓"
    else:
        return False, (
            f"⚠️ CONFIG INTEGRITY FAILURE! "
            f"Expected: {stored_hash[:16]}… "
            f"Got: {current_hash[:16]}… "
            f"File may have been tampered with."
        )


# ─────────────────────────────────────────────
# Audit Logging
# ─────────────────────────────────────────────

_DEFAULT_AUDIT_LOG = Path(__file__).parent.parent / "audit.log"


class AuditLogger:
    """
    Structured audit logging for data access and classification decisions.

    Logs events in JSON Lines format for easy parsing.

    Usage:
        audit = AuditLogger()
        audit.log_access("raw_text_logs/invoice_001.md", "read")
        audit.log_classification("invoice_001.pdf", "auto_accept", 0.95)
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or _DEFAULT_AUDIT_LOG
        self._logger = logging.getLogger("audit")

        # Ensure handler is set up (file handler, no propagation)
        if not self._logger.handlers:
            handler = logging.FileHandler(
                str(self.log_path), encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def _log_event(self, event_type: str, **kwargs):
        """Write a structured audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **kwargs
        }
        self._logger.info(json.dumps(event, ensure_ascii=False))

    def log_access(self, resource: str, action: str, user: str = "system"):
        """Log a data access event."""
        self._log_event(
            "data_access",
            user=user,
            resource=resource,
            action=action
        )

    def log_classification(
        self,
        filename: str,
        decision: str,
        confidence: float,
        reasoning: str = ""
    ):
        """Log a classification decision."""
        self._log_event(
            "classification",
            filename=filename,
            decision=decision,
            confidence=round(confidence, 4),
            reasoning=reasoning[:200]  # Truncate for audit log
        )

    def log_security_event(self, event: str, details: str = ""):
        """Log a security-related event (failed validation, integrity check, etc.)."""
        self._log_event(
            "security",
            event=event,
            details=details
        )


# ─────────────────────────────────────────────
# Data Retention / Cleanup
# ─────────────────────────────────────────────

def cleanup_old_files(
    directory: Path,
    max_age_days: int = 30,
    extensions: Optional[set] = None,
    dry_run: bool = False
) -> list:
    """
    Remove files older than max_age_days from a directory.

    Args:
        directory: Directory to clean
        max_age_days: Maximum age of files in days
        extensions: Set of file extensions to target (e.g., {'.md', '.log'}).
                    If None, targets all files.
        dry_run: If True, only list files that would be deleted

    Returns:
        List of deleted (or would-be-deleted) file paths
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Cleanup directory not found: {directory}")
        return []

    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = []

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        # Filter by extension if specified
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                if dry_run:
                    logger.info(f"🗑️ Would delete: {file_path.name} (age: {(datetime.now() - mtime).days}d)")
                else:
                    file_path.unlink()
                    logger.info(f"🗑️ Deleted: {file_path.name} (age: {(datetime.now() - mtime).days}d)")
                deleted.append(str(file_path))
        except FileNotFoundError:
            # File was already deleted by another process — goal achieved
            logger.debug(f"File already removed (concurrent deletion): {file_path.name}")
            deleted.append(str(file_path))
        except Exception as e:
            logger.warning(f"Cleanup error for {file_path.name}: {e}")

    if deleted:
        logger.info(f"Cleanup: {'would delete' if dry_run else 'deleted'} {len(deleted)} file(s) from {directory}")

    return deleted


# ─────────────────────────────────────────────
# Module-level singleton instances
# ─────────────────────────────────────────────

# Lazy singletons — created on first access
_encryptor_instance = None
_audit_instance = None


def get_encryptor() -> DataEncryptor:
    """Get the module-level DataEncryptor singleton."""
    global _encryptor_instance
    if _encryptor_instance is None:
        _encryptor_instance = DataEncryptor()
    return _encryptor_instance


def get_audit_logger() -> AuditLogger:
    """Get the module-level AuditLogger singleton."""
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = AuditLogger()
    return _audit_instance
