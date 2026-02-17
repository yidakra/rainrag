"""Streamlit frontend for RainRAG - Multilingual RAG system for video transcripts."""

import contextlib
import hmac
import html
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import redis
import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from loguru import logger


# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("RAINRAG_API_URL", "http://localhost:8001").rstrip("/")
# API base for server-side calls (health/query). If API_BASE_URL ends with /api, keep it.
API_BASE = API_BASE_URL
# Asset base for browser-facing URLs (video/vtt/docs). If API_BASE_URL ends with /api, strip it.
ASSET_BASE_URL = API_BASE[:-4] if API_BASE.endswith("/api") else API_BASE
# Allow disabling SSL verification for self-signed/internal certs
API_VERIFY_SSL = os.getenv("RAINRAG_API_VERIFY", "true").lower() not in ("0", "false", "no", "off")

# Security Configuration
# REQUIRED: Argon2-hashed password (use scripts/generate_password_hash.py to create)
AUTH_PASSWORD_HASH = os.getenv("RAINRAG_PASSWORD_HASH", "")
# Optional: Legacy plain-text token support (DEPRECATED - use hashed password instead)
AUTH_TOKEN = os.getenv("STREAMLIT_AUTH_TOKEN", "")
# Session timeout in minutes (default: 8 hours)
try:
    _timeout = int(os.getenv("RAINRAG_SESSION_TIMEOUT", "480"))
    if _timeout <= 0:
        _timeout = 480
        logger.warning("RAINRAG_SESSION_TIMEOUT must be positive, using default 480")
    session_timeout_minutes = _timeout
except (ValueError, TypeError):
    session_timeout_minutes = 480
    logger.warning("Invalid RAINRAG_SESSION_TIMEOUT, using default 480")

# Maximum failed login attempts before temporary lockout
_max_attempts = 5
try:
    temp = int(os.getenv("RAINRAG_MAX_LOGIN_ATTEMPTS", "5"))
    if temp > 0:
        _max_attempts = temp
    else:
        logger.warning("RAINRAG_MAX_LOGIN_ATTEMPTS must be positive, using default 5")
except (ValueError, TypeError):
    logger.warning("Invalid RAINRAG_MAX_LOGIN_ATTEMPTS, using default 5")
MAX_LOGIN_ATTEMPTS = _max_attempts

# Lockout duration in seconds after max attempts
try:
    _lockout_duration = int(os.getenv("RAINRAG_LOCKOUT_DURATION", "300"))
    if _lockout_duration <= 0:
        _lockout_duration = 300
        logger.warning("RAINRAG_LOCKOUT_DURATION must be positive, using default 300")
except (ValueError, TypeError):
    _lockout_duration = 300
    logger.warning("Invalid RAINRAG_LOCKOUT_DURATION, using default 300")
LOCKOUT_DURATION_SECONDS = _lockout_duration
# Enable audit logging for authentication events
AUDIT_LOG_ENABLED = os.getenv("RAINRAG_AUDIT_LOG", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Redis configuration for server-side session storage
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

DEFAULT_LANGUAGE = "ru"
DEFAULT_TOP_K = 3
REQUEST_TIMEOUT = 60.0  # 60 seconds timeout for API requests
DOCS_PATH = os.getenv("RAINRAG_DOCS_PATH", "./data/docs.jsonl")

# Initialize Argon2 password hasher for secure password hashing
# Using recommended parameters: time_cost=2, memory_cost=102400 (100MB), parallelism=8
password_hasher = PasswordHasher(
    time_cost=2,  # Number of iterations
    memory_cost=102400,  # Memory usage in KiB (100MB)
    parallelism=8,  # Number of parallel threads
    hash_len=32,  # Hash length in bytes
    salt_len=16,  # Salt length in bytes
)

# Initialize Redis client for server-side storage
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )
    # Test connection
    redis_client.ping()  # type: ignore
    logger.info("Connected to Redis for session storage")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None


# Translations
TRANSLATIONS = {
    "ru": {
        "title": "RainRAG - Поиск по видео-транскриптам",
        "subtitle": "Задайте вопрос о содержимом видео на русском или английском языке",
        "language_label": "Язык / Language",
        "num_chunks_label": "Количество контекстных фрагментов",
        "system_info_label": "Информация о системе",
        "model_label": "Модель LLM",
        "embedding_label": "Модель эмбеддингов",
        "collection_label": "Коллекция Qdrant",
        "status_label": "Статус",
        "connected": "Подключено",
        "disconnected": "Отключено",
        "input_placeholder": "Введите ваш вопрос здесь...",
        "send_button": "Отправить",
        "clear_button": "Очистить историю",
        "context_header": "Найденные фрагменты контекста",
        "error_auth": "Ошибка аутентификации. Проверьте токен доступа.",
        "error_connection": "Ошибка подключения к серверу. Убедитесь, что API запущен.",
        "error_timeout": "Превышено время ожидания. Попробуйте снова.",
        "error_general": "Произошла ошибка",
        "thinking": "Думаю...",
        "source_label": "Источник",
        "score_label": "Релевантность",
        "language_field": "Язык",
        "loading_system": "Загрузка информации о системе...",
        "auth_title": "Вход в систему",
        "auth_prompt": "Введите токен доступа:",
        "auth_help": "Введите системный пароль, предоставленный администратором",
        "auth_button": "Войти",
        "auth_invalid": "Неверный токен доступа",
        "health_check_failed": "Не удалось подключиться к API",
        "video_label": "Видео",
        "vtt_label": "Субтитры",
        "download_vtt": "Скачать VTT",
        "view_vtt": "Просмотр VTT",
        "no_video": "Видео не найдено",
        "date_label": "Дата",
        "duration_label": "Длит.",
        "timecode_label": "Тайм-код",
        "date_filter_label": "Фильтр по дате",
        "date_from_label": "С",
        "date_to_label": "По",
        "clear_dates": "Сбросить даты",
        "web_title_label": "Заголовок",
        "web_date_label": "Дата веб-страницы",
        "url_label": "URL",
        "description_label": "Описание",
        "copy_answer": "Копировать ответ",
        "copied": "Скопировано!",
        "search_history": "История поиска",
        "no_history": "История пуста",
        "export_chat": "Экспорт беседы",
        "export_markdown": "Скачать Markdown",
        "export_text": "Скачать текст",
        "example_queries": "Примеры запросов",
        "try_example": "Попробовать",
        "sidebar_search_settings": "⚙️ Настройки поиска",
        "sidebar_conversation": "💬 Беседа",
        "sidebar_session": "👤 Сессия",
        "search_features_label": "Функции поиска:",
        "search_features_standard": "Стандартный векторный поиск",
        "searching": "Поиск видео транскриптов...",
        "found_clips": "Найдено {count} релевантных фрагментов!",
        "session_expires": "Сессия истекает через:",
        "session_expired_warning": "Сессия истекла из-за неактивности. Пожалуйста, войдите снова.",
        "logout": "Выход",
        "vtt_language": "Язык",
        "vtt_error": "Не удалось загрузить VTT",
        "no_vtt": "VTT недоступен",
        "find_related": "Найти похожее",
        "related_content": "Похожий контент:",
        "no_related": "Похожих фрагментов не найдено",
        "finding_related": "Поиск похожих фрагментов...",
        "error_finding_related": "Ошибка поиска похожих фрагментов:",
        "auth_not_configured": "ОШИБКА БЕЗОПАСНОСТИ: Аутентификация не настроена!",
        "auth_set_password": "Пожалуйста, установите переменную окружения RAINRAG_PASSWORD_HASH.",
        "auth_use_script": "Используйте scripts/generate_password_hash.py для генерации безопасного хеша.",
        "account_locked_message": "Аккаунт заблокирован из-за большого количества неудачных попыток.",
        "try_again_in": "Пожалуйста, попробуйте снова через {time}",
        "attempts_remaining": "{count} попыток осталось до блокировки аккаунта",
        "auth_success": "Аутентификация успешна!",
        "attempts_left": "{count} попыток осталось",
        "account_locked_final": "Аккаунт заблокирован! Слишком много неудачных попыток.",
    },
    "en": {
        "title": "RainRAG - Video Transcript Search",
        "subtitle": "Ask questions about video content in Russian or English",
        "language_label": "Language / Язык",
        "num_chunks_label": "Number of context chunks",
        "system_info_label": "System Information",
        "model_label": "LLM Model",
        "embedding_label": "Embedding Model",
        "collection_label": "Qdrant Collection",
        "status_label": "Status",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "input_placeholder": "Type your question here...",
        "send_button": "Send",
        "clear_button": "Clear History",
        "context_header": "Retrieved Context Chunks",
        "error_auth": "Authentication error. Please check your access token.",
        "error_connection": "Connection error. Make sure the API is running.",
        "error_timeout": "Request timeout. Please try again.",
        "error_general": "An error occurred",
        "thinking": "Thinking...",
        "source_label": "Source",
        "score_label": "Relevance",
        "language_field": "Language",
        "loading_system": "Loading system information...",
        "auth_title": "System Login",
        "auth_prompt": "Enter access token:",
        "auth_help": "Enter the system password provided by your administrator",
        "auth_button": "Login",
        "auth_invalid": "Invalid access token",
        "health_check_failed": "Failed to connect to API",
        "video_label": "Video",
        "vtt_label": "Subtitles",
        "download_vtt": "Download VTT",
        "view_vtt": "View VTT",
        "no_video": "Video not found",
        "date_label": "Date",
        "duration_label": "Dur.",
        "timecode_label": "TC",
        "date_filter_label": "Date Filter",
        "date_from_label": "From",
        "date_to_label": "To",
        "clear_dates": "Clear Dates",
        "web_title_label": "Title",
        "web_date_label": "Web Date",
        "url_label": "URL",
        "description_label": "Description",
        "copy_answer": "Copy Answer",
        "copied": "Copied!",
        "search_history": "Search History",
        "no_history": "No history",
        "export_chat": "Export Conversation",
        "export_markdown": "Download Markdown",
        "export_text": "Download Text",
        "example_queries": "Example Queries",
        "try_example": "Try it",
        "sidebar_search_settings": "⚙️ Search Settings",
        "sidebar_conversation": "💬 Conversation",
        "sidebar_session": "👤 Session",
        "search_features_label": "Search Features:",
        "search_features_standard": "Standard vector search",
        "searching": "Searching video transcripts...",
        "found_clips": "Found {count} relevant clips!",
        "session_expires": "Session expires in:",
        "session_expired_warning": "Session expired due to inactivity. Please login again.",
        "logout": "Logout",
        "vtt_language": "Language",
        "vtt_error": "Could not load VTT",
        "no_vtt": "No VTT available",
        "find_related": "Find Related",
        "related_content": "Related Content:",
        "no_related": "No related chunks found",
        "finding_related": "Finding related chunks...",
        "error_finding_related": "Error finding related chunks:",
        "auth_not_configured": "SECURITY ERROR: No authentication configured!",
        "auth_set_password": "Please set RAINRAG_PASSWORD_HASH environment variable.",
        "auth_use_script": "Use scripts/generate_password_hash.py to generate a secure hash.",
        "account_locked_message": "Account locked due to too many failed attempts.",
        "try_again_in": "Please try again in {time}",
        "attempts_remaining": "{count} attempt(s) remaining before account lockout",
        "auth_success": "Authentication successful!",
        "attempts_left": "{count} attempt(s) remaining",
        "account_locked_final": "Account locked! Too many failed attempts.",
    },
}


def get_text(key: str, lang: str = "ru") -> str:
    """Get translated text for given key and language."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["ru"]).get(key, key)


def audit_log(event: str, details: str = "", success: bool = True):
    """Log authentication events for security auditing."""
    if not AUDIT_LOG_ENABLED:
        return

    timestamp = datetime.now().isoformat()
    status = "SUCCESS" if success else "FAILURE"
    client_ip = st.session_state.get("client_ip", "unknown")

    log_message = f"[AUDIT] {timestamp} | {status} | {event} | IP: {client_ip} | {details}"

    if success:
        logger.info(log_message)
    else:
        logger.warning(log_message)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against an Argon2 hash.

    Args:
        password: Plain text password to verify
        password_hash: Argon2 hash to verify against

    Returns:
        True if password matches hash, False otherwise
    """
    try:
        password_hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def is_session_expired() -> bool:
    """Check if the current session has expired."""
    if "last_activity" not in st.session_state:
        return True

    last_activity = st.session_state.get("last_activity")
    if not last_activity:
        return True

    timeout_delta = timedelta(minutes=session_timeout_minutes)
    return datetime.now() - last_activity > timeout_delta


def update_session_activity():
    """Update the last activity timestamp for session timeout tracking."""
    st.session_state["last_activity"] = datetime.now()


# Redis-based server-side lockout helpers
def _get_lockout_key(identifier: str) -> str:
    """Generate a unique Redis key for lockout tracking."""
    return f"lockout:{identifier}"


def redis_get_failed_attempts(identifier: str) -> dict[str, Any]:
    """
    Get failed login attempts from Redis.

    Args:
        identifier: Unique identifier (e.g., IP address or user ID)

    Returns:
        Dictionary with 'count' and 'lockout_until' keys
    """
    if redis_client is None:
        # Fallback to in-memory dict if Redis unavailable
        return {"count": 0, "lockout_until": None}

    key = _get_lockout_key(identifier)
    data = cast(dict[str, Any], redis_client.hgetall(key))  # type: ignore

    if not data:
        return {"count": 0, "lockout_until": None}

    count = int(data.get("count", "0"))
    lockout_until_str = data.get("lockout_until")
    lockout_until = None
    if lockout_until_str:
        try:
            lockout_until = datetime.fromisoformat(lockout_until_str)
        except ValueError:
            lockout_until = None

    return {"count": count, "lockout_until": lockout_until}


def redis_set_failed_attempts(identifier: str, count: int, lockout_until: datetime | None = None):
    """
    Set failed login attempts in Redis.

    Args:
        identifier: Unique identifier
        count: Number of failed attempts
        lockout_until: Lockout expiration timestamp
    """
    if redis_client is None:
        return

    key = _get_lockout_key(identifier)
    data = {"count": str(count)}

    if lockout_until:
        data["lockout_until"] = lockout_until.isoformat()
        # Set TTL to lockout duration
        ttl = int((lockout_until - datetime.now()).total_seconds())
        if ttl > 0:
            redis_client.expire(key, ttl)
    else:
        # Remove TTL if no lockout
        redis_client.persist(key)

    redis_client.hset(key, mapping=data)  # type: ignore


def redis_incr_failed_attempts(identifier: str) -> int:
    """
    Increment failed attempts counter in Redis.

    Args:
        identifier: Unique identifier

    Returns:
        New count value
    """
    if redis_client is None:
        return 0

    key = _get_lockout_key(identifier)
    count = cast(int, redis_client.hincrby(key, "count", 1))  # type: ignore
    return count


def redis_reset_failed_attempts(identifier: str):
    """
    Reset failed login attempts in Redis.

    Args:
        identifier: Unique identifier
    """
    if redis_client is None:
        return

    key = _get_lockout_key(identifier)
    redis_client.delete(key)


def get_failed_attempts() -> dict[str, Any]:
    """
    Get failed login attempts tracking data.

    Returns:
        Dictionary with 'count' and 'lockout_until' keys
    """
    # Use client IP as identifier
    identifier = st.session_state.get("client_ip", "unknown")
    return redis_get_failed_attempts(identifier)


def is_account_locked() -> tuple[bool, int]:
    """
    Check if account is temporarily locked due to failed attempts.

    Returns:
        Tuple of (is_locked, seconds_remaining)
    """
    attempts = get_failed_attempts()
    lockout_until = attempts.get("lockout_until")

    if lockout_until is None:
        return False, 0

    remaining = (lockout_until - datetime.now()).total_seconds()
    if remaining > 0:
        return True, int(remaining)
    else:
        # Lockout period expired (shouldn't happen with Redis TTL, but handle anyway)
        return False, 0


def record_failed_attempt():
    """Record a failed login attempt and apply lockout if threshold reached."""
    identifier = st.session_state.get("client_ip", "unknown")
    count = redis_incr_failed_attempts(identifier)

    if count >= MAX_LOGIN_ATTEMPTS:
        lockout_until = datetime.now() + timedelta(seconds=LOCKOUT_DURATION_SECONDS)
        redis_set_failed_attempts(identifier, count, lockout_until)
        audit_log(
            "ACCOUNT_LOCKED",
            f"Account locked after {MAX_LOGIN_ATTEMPTS} failed attempts",
            success=False,
        )


def reset_failed_attempts():
    """Reset failed login attempts counter after successful login."""
    identifier = st.session_state.get("client_ip", "unknown")
    redis_reset_failed_attempts(identifier)


def check_authentication() -> bool:
    """
    Check authentication and handle login with security best practices.

    Features:
    - Mandatory authentication (no bypass)
    - Argon2 password hashing
    - Rate limiting with account lockout
    - Session timeout
    - Audit logging

    Returns:
        True if authenticated, False otherwise
    """
    # SECURITY: Authentication is now MANDATORY
    lang = st.session_state.get("language", "ru")
    if not AUTH_PASSWORD_HASH and not AUTH_TOKEN:
        st.error(get_text("auth_not_configured", lang))
        st.error(get_text("auth_set_password", lang))
        st.error(get_text("auth_use_script", lang))
        st.stop()
        return False

    # Check if user is already authenticated and session is valid
    if st.session_state.get("authenticated", False):
        # Check for session timeout
        if is_session_expired():
            st.session_state["authenticated"] = False
            audit_log("SESSION_EXPIRED", "Session timed out due to inactivity", success=False)
            st.warning(get_text("session_expired_warning", st.session_state.get("language", "ru")))
            time.sleep(1)
            st.rerun()
            return False

        # Update activity timestamp
        update_session_activity()
        return True

    # Check if account is locked
    lang = st.session_state.get("language", "ru")
    is_locked, seconds_remaining = is_account_locked()
    if is_locked:
        minutes = int(seconds_remaining / 60)
        seconds = int(seconds_remaining % 60)
        st.error(get_text("account_locked_message", lang))
        st.error(get_text("try_again_in", lang).format(time=f"{minutes}m {seconds}s"))
        audit_log(
            "LOGIN_BLOCKED",
            f"Login attempt while locked ({seconds_remaining}s remaining)",
            success=False,
        )
        time.sleep(1)
        st.stop()
        return False

    # Show login form
    st.title(get_text("auth_title", lang))

    # Display security info
    attempts = get_failed_attempts()
    remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts["count"]
    if attempts["count"] > 0:
        st.warning(get_text("attempts_remaining", lang).format(count=remaining_attempts))

    with st.form("login_form"):
        password_input = st.text_input(
            get_text("auth_prompt", lang),
            type="password",
            help=get_text("auth_help", lang),
        )
        submit = st.form_submit_button(get_text("auth_button", lang))

        if submit:
            if not password_input:
                st.error(get_text("auth_invalid", lang))
                audit_log("LOGIN_FAILED", "Empty password submitted", success=False)
                return False

            # Verify password (use Argon2 hash, fallback to legacy token)
            password_valid = False

            if AUTH_PASSWORD_HASH:
                # Use Argon2 hash (secure method)
                password_valid = verify_password(password_input, AUTH_PASSWORD_HASH)
            elif AUTH_TOKEN:
                # Legacy plain-text comparison (DEPRECATED)
                password_valid = hmac.compare_digest(password_input or "", AUTH_TOKEN or "")
                if password_valid:
                    logger.warning(
                        "SECURITY WARNING: Using deprecated plain-text AUTH_TOKEN. "
                        "Please migrate to RAINRAG_PASSWORD_HASH using argon2."
                    )

            if password_valid:
                # Successful authentication
                st.session_state["authenticated"] = True
                st.session_state["last_activity"] = datetime.now()
                reset_failed_attempts()
                audit_log("LOGIN_SUCCESS", "User authenticated successfully", success=True)
                st.success(get_text("auth_success", lang))
                time.sleep(0.5)
                st.rerun()
                return True
            else:
                # Failed authentication
                record_failed_attempt()
                attempts_left = MAX_LOGIN_ATTEMPTS - get_failed_attempts()["count"]
                audit_log(
                    "LOGIN_FAILED",
                    f"Invalid password attempt ({attempts_left} attempts remaining)",
                    success=False,
                )
                st.error(get_text("auth_invalid", lang))

                if attempts_left > 0:
                    st.warning(get_text("attempts_left", lang).format(count=attempts_left))
                else:
                    st.error(get_text("account_locked_final", lang))

                return False

    return False


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False  # Always require authentication
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = None
    if "date_from" not in st.session_state:
        st.session_state.date_from = None
    if "date_to" not in st.session_state:
        st.session_state.date_to = None
    if "date_input_reset_counter" not in st.session_state:
        st.session_state.date_input_reset_counter = 0
    # Search history tracking
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    # Query suggestions (based on actual topics in web metadata)
    if "example_queries" not in st.session_state:
        st.session_state.example_queries = {
            "ru": [
                "30 тысяч погибли на протестах в Иране",
                "«Совет мира» с Путиным и Лукашенко",
                "ДТП в Грозном: что с Адамом Кадыровым?",
                "«Госуслуги» зовут в университет спецназа",
            ],
            "en": [
                "30 thousand killed in Iran protests",
                "Peace Council with Putin and Lukashenko",
                "Car accident in Grozny: what happened to Adam Kadyrov?",
                "Gosuslugi invites to special forces university",
            ],
        }
    # Pending query flag for example buttons
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "client_ip" not in st.session_state:
        # Try to get client IP from Streamlit context (for audit logging)
        try:
            # Use new st.context API (Streamlit >= 1.31)
            if hasattr(st, "context") and hasattr(st.context, "headers"):
                headers = st.context.headers
                x_forwarded_for = headers.get("X-Forwarded-For", "").strip()
                if x_forwarded_for:
                    # X-Forwarded-For may contain multiple IPs, use the first one
                    client_ip = x_forwarded_for.split(",")[0].strip()
                else:
                    client_ip = headers.get("Remote-Addr", "unknown")
                st.session_state.client_ip = client_ip
            else:
                st.session_state.client_ip = "unknown"
        except Exception:
            st.session_state.client_ip = "unknown"


@st.cache_data(show_spinner=False)
def get_archive_date_range() -> tuple[date | None, date | None]:
    """
    Compute min/max available dates from the docs JSONL.

    Returns:
        (min_date, max_date) or (None, None) if unavailable.
    """
    docs_path = Path(DOCS_PATH)
    if not docs_path.exists():
        return None, None

    min_d: date | None = None
    max_d: date | None = None

    try:
        with docs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                date_str = record.get("date") or record.get("date_iso")
                if date_str:
                    try:
                        d = date.fromisoformat(date_str.split("T")[0])
                    except ValueError:
                        d = None
                else:
                    d = None

                if d is None:
                    date_ts = record.get("date_ts")
                    if isinstance(date_ts, int | float):
                        try:
                            d = date.fromtimestamp(date_ts)
                        except (OverflowError, OSError, ValueError):
                            d = None

                if d:
                    if min_d is None or d < min_d:
                        min_d = d
                    if max_d is None or d > max_d:
                        max_d = d
    except FileNotFoundError:
        return None, None

    return min_d, max_d


def get_api_headers() -> dict[str, str]:
    """Get API request headers including auth token if configured."""
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


async def check_api_health() -> dict[str, Any] | None:
    """Check API health status."""
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=API_VERIFY_SSL) as client:
            response = await client.get(f"{API_BASE}/health", headers=get_api_headers())
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return None


async def query_rag(
    question: str,
    language: str,
    top_k: int,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """
    Query the RAG system via API.

    Args:
        question: User's question
        language: Response language (ru or en)
        top_k: Number of context chunks to retrieve

    Returns:
        API response dictionary

    Raises:
        httpx.HTTPStatusError: If API returns error status
        httpx.TimeoutException: If request times out
    """
    payload: dict[str, Any] = {"question": question, "language": language, "top_k": top_k}
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.post(
            f"{API_BASE}/query",
            json=payload,
            headers=get_api_headers(),
        )
        response.raise_for_status()
        return response.json()


async def get_related_chunks(
    chunk_id: str, top_k: int = 3, same_video_only: bool = False
) -> list[dict[str, Any]]:
    """
    Get related chunks for a given chunk.

    Args:
        chunk_id: The ID of the source chunk
        top_k: Number of related chunks to retrieve
        same_video_only: If True, only return chunks from the same video

    Returns:
        List of related chunks

    Raises:
        httpx.HTTPStatusError: If API returns error status
    """
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "top_k": top_k,
        "same_video_only": same_video_only,
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.post(
            f"{API_BASE}/related-chunks",
            json=payload,
            headers=get_api_headers(),
        )
        response.raise_for_status()
        result = response.json()
        return result.get("related_chunks", [])


def fetch_vtt_content(vtt_url: str) -> str | None:
    """
    Fetch VTT file content from the API.

    Args:
        vtt_url: VTT file URL (relative to API base URL)

    Returns:
        VTT file content as string, or None if failed
    """
    try:
        import requests

        vtt_full_url = f"{ASSET_BASE_URL}{vtt_url}"
        headers = get_api_headers()
        response = requests.get(vtt_full_url, headers=headers, timeout=10, verify=API_VERIFY_SSL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch VTT content: {e}")
        return None


def group_chunks_by_video(chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Group context chunks by their group_id (same video, different languages).

    Args:
        chunks: List of context chunks

    Returns:
        List of groups, where each group contains chunks for the same video
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        group_id = chunk.get("group_id") or chunk.get("filename", "unknown")
        groups.setdefault(group_id, []).append(chunk)

    # Sort groups by the best score in each group
    sorted_groups = sorted(
        groups.values(),
        key=lambda g: max(c.get("score", 0.0) for c in g),
        reverse=True,
    )

    return sorted_groups


def sanitize_web_url(url: str) -> str | None:
    """Sanitize web URL for safe Markdown embedding.

    - Allow only http/https schemes
    - Escape Markdown-breaking characters
    - Return None if invalid
    """
    if not url:
        return None

    # Parse URL to check scheme
    from urllib.parse import urlparse

    parsed = urlparse(url)

    # Allow only http and https schemes
    if parsed.scheme not in ("http", "https"):
        return None

    # Escape Markdown-breaking characters (like parentheses)
    safe_url = url.replace(")", "\\)").replace("(", "\\(")

    return safe_url


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text and clean up whitespace.

    Args:
        text: Text potentially containing HTML tags

    Returns:
        Plain text with HTML tags removed
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Clean up excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def format_context_chunk(chunk: dict[str, Any], lang: str) -> str:
    """Format a context chunk metadata for display (without text content)."""
    filename = chunk.get("filename", "Unknown")
    score = chunk.get("score", 0.0)
    chunk_lang = chunk.get("language", "unknown")
    chunk_date = chunk.get("date")
    duration_seconds = chunk.get("duration_seconds")
    start_time = chunk.get("start_time")
    end_time = chunk.get("end_time")
    is_chunk = chunk.get("is_chunk", False)
    chunk_index = chunk.get("chunk_index")
    total_chunks = chunk.get("total_chunks")

    # New metadata fields
    rerank_score = chunk.get("rerank_score")
    original_score = chunk.get("original_score")
    time_boost = chunk.get("time_boost")
    fusion_method = chunk.get("fusion_method")

    # Web metadata fields
    web_title = chunk.get("web_title")
    web_description = chunk.get("web_description")
    web_url = chunk.get("web_url")

    duration_str = ""
    if duration_seconds:
        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        duration_str = f"{mins}:{secs:02d}"

    timecode_str = ""
    if start_time and end_time:
        timecode_str = f"{start_time}-{end_time}"

    # Build chunk info string with time range
    chunk_info = ""
    if is_chunk and chunk_index is not None and total_chunks is not None:
        if timecode_str:
            chunk_info = f" **[Chunk {chunk_index + 1}/{total_chunks}: {timecode_str}]**"
        else:
            chunk_info = f" **[Chunk {chunk_index + 1}/{total_chunks}]**"

    # Base metadata
    meta_parts = [f"**Score:** {score:.3f}"]

    # Show reranking info if available
    if rerank_score is not None and original_score is not None:
        meta_parts[0] = f"**Score:** {score:.3f} (reranked from {original_score:.3f})"

    # Show time boost if available
    if time_boost is not None:
        meta_parts.append(f"**Time Boost:** {time_boost:.2f}x")

    # Show fusion method if available
    if fusion_method:
        meta_parts.append(f"**Fusion:** {fusion_method.upper()}")

    if chunk_date:
        meta_parts.append(f"**{get_text('date_label', lang)}:** {chunk_date}")
    if duration_str:
        meta_parts.append(f"**{get_text('duration_label', lang)}:** {duration_str}")
    # Don't duplicate timecode in metadata if already shown in chunk info
    if timecode_str and not is_chunk:
        meta_parts.append(f"**{get_text('timecode_label', lang)}:** {timecode_str}")
    meta_parts.append(f"**{get_text('language_field', lang)}:** {chunk_lang}")

    # Build metadata display - prioritize user-facing information
    lines: list[str] = []

    # 1. Title (most important - show prominently if available)
    if web_title:
        lines.append(f"### {web_title}")

    # 2. Chunk info (if this is a chunk)
    if chunk_info:
        lines.append(chunk_info)

    # 3. Video metadata (date, duration, timecode)
    video_meta: list[str] = []
    if chunk_date:
        video_meta.append(f"📅 {chunk_date}")
    if duration_str:
        video_meta.append(f"⏱ {duration_str}")
    if timecode_str:
        video_meta.append(f"⏰ {timecode_str}")

    if video_meta:
        lines.append(" • ".join(video_meta))

    # 4. Additional metadata (score, boost, fusion, language)
    if meta_parts:
        lines.append("\n".join(meta_parts))

    # 5. Description (if available)
    if web_description:
        clean_description = strip_html_tags(web_description)
        lines.append(f"\n{clean_description}")

    # 6. URL (if available)
    if web_url:
        sanitized_url = sanitize_web_url(web_url)
        if sanitized_url:
            lines.append(f"\n🔗 <{sanitized_url}>")

    # 7. Technical details (collapsed/minimized)
    tech_details: list[str] = []
    tech_details.append(f"Score: {score:.3f}")
    if rerank_score is not None and original_score is not None:
        tech_details[-1] = f"Score: {score:.3f} (reranked from {original_score:.3f})"
    if time_boost is not None:
        tech_details.append(f"Boost: {time_boost:.2f}x")
    if fusion_method:
        tech_details.append(f"Fusion: {fusion_method.upper()}")
    tech_details.append(f"Lang: {chunk_lang}")
    # Sanitize filename to prevent breaking Markdown inline code formatting
    sanitized_filename = filename.replace("`", "'").replace("|", "│")
    tech_details.append(f"File: {sanitized_filename}")

    lines.append(f"\n`{' | '.join(tech_details)}`")

    return "\n".join(lines)


def get_text_preview(text: str, max_lines: int = 3, max_chars: int = 200) -> tuple[str, bool]:
    """
    Get a preview of text content.

    Args:
        text: Full text content
        max_lines: Maximum number of lines to show in preview
        max_chars: Maximum number of characters to show in preview

    Returns:
        Tuple of (preview_text, is_truncated)
    """
    lines = text.split("\n")
    preview_lines = lines[:max_lines]
    preview_text = "\n".join(preview_lines)

    # Check if we need to truncate by character count
    if len(preview_text) > max_chars:
        preview_text = preview_text[:max_chars] + "..."
        is_truncated = True
    elif len(lines) > max_lines:
        preview_text += "..."
        is_truncated = True
    else:
        is_truncated = len(text) > len(preview_text)

    return preview_text, is_truncated


def render_message_bubble(message: dict[str, Any], lang: str):
    """Render a message bubble with appropriate styling."""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

    # Show context if available
    if role == "assistant" and "context" in message:
        # Search insights are available in chunk metadata if needed for debugging

        with st.expander(get_text("context_header", lang), expanded=False):
            # Group chunks by video (to show en/ru versions together)
            grouped_chunks = group_chunks_by_video(message["context"])

            for group_idx, group in enumerate(grouped_chunks, 1):
                # Create 2/3 - 1/3 layout for video and VTT
                video_col, vtt_col = st.columns([2, 1])

                with video_col:
                    # Display video (2/3 width on left)
                    video_url = group[0].get("video_url")
                    if video_url:
                        st.markdown(f"**{get_text('video_label', lang)}:**")
                        # Strip any existing fragment
                        base_video_url = video_url.split("#", 1)[0]
                        video_full_url = f"{ASSET_BASE_URL}{base_video_url}"

                        # Add timestamp fragment for chunks to seek to start time
                        start_time_seconds = group[0].get("start_time_seconds")
                        if start_time_seconds is not None:
                            with contextlib.suppress(ValueError, TypeError):
                                # HTML5 video supports #t=seconds for seeking
                                video_full_url += f"#t={int(float(start_time_seconds))}"

                        try:
                            # HTML5 video player with native controls
                            st.markdown(
                                f"""
                                <video controls controlsList="nodownload"
                                       style="max-width: 100%; height: auto; border-radius: 8px;"
                                       preload="metadata">
                                    <source src="{html.escape(video_full_url)}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                                """,
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            logger.warning(f"Could not load video: {e}")
                            st.warning(get_text("no_video", lang))
                    else:
                        st.info(get_text("no_video", lang))

                with vtt_col:
                    # Display VTT selector and viewer (1/3 width on right)
                    st.markdown(f"**{get_text('vtt_label', lang)}:**")

                    # Create language selector if multiple VTT files exist for this group
                    vtt_languages: dict[str, dict[str, Any]] = {}
                    for chunk in group:
                        vtt_url = chunk.get("vtt_url")
                        if vtt_url:
                            chunk_lang = chunk.get("language", "unknown")
                            vtt_languages[chunk_lang] = {
                                "url": vtt_url,
                                "filename": chunk.get("filename", "subtitle.vtt"),
                                "chunk": chunk,
                            }

                    if vtt_languages:
                        # Language selector for VTT
                        if len(vtt_languages) > 1:
                            lang_display = {"ru": "🇷🇺 Русский", "en": "🇬🇧 English"}
                            selected_vtt_lang = st.radio(
                                get_text("vtt_language", lang),
                                options=list(vtt_languages.keys()),
                                format_func=lambda x, ld=lang_display: ld.get(x, x),
                                horizontal=True,
                                key=f"vtt_lang_{group_idx}",
                                label_visibility="collapsed",
                            )
                        else:
                            selected_vtt_lang = list(vtt_languages.keys())[0]

                        # Get selected VTT info
                        vtt_info = vtt_languages[selected_vtt_lang]
                        vtt_url = vtt_info["url"]
                        vtt_full_url = f"{ASSET_BASE_URL}{vtt_url}"
                        vtt_filename = vtt_info["filename"].split("/")[-1]

                        # Download button
                        st.markdown(
                            f'<a href="{vtt_full_url}" download="{vtt_filename}" style="display: inline-block; padding: 0.4rem 0.8rem; background-color: #0084ff; color: white; text-decoration: none; border-radius: 0.25rem; text-align: center; width: 100%; box-sizing: border-box; margin-bottom: 0.5rem;">{get_text("download_vtt", lang)}</a>',
                            unsafe_allow_html=True,
                        )

                        # VTT content viewer (scrollable)
                        vtt_content = fetch_vtt_content(vtt_url)
                        if vtt_content:
                            # Display VTT in a scrollable container
                            # Use dark background that works in both light and dark modes
                            st.markdown(
                                f'<div style="height: 400px; overflow-y: auto; border: 1px solid #4a4a4a; border-radius: 0.25rem; padding: 0.5rem; background-color: #1e1e1e; color: #e0e0e0; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap;">{vtt_content}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.error(get_text("vtt_error", lang))
                    else:
                        st.info(get_text("no_vtt", lang))

                # Display text context and metadata below the video/VTT layout
                st.markdown("---")
                for chunk_idx, chunk in enumerate(group):
                    # Display context chunk metadata
                    st.markdown(format_context_chunk(chunk, lang))

                    # Add "Find Related" button for each chunk
                    doc_id = chunk.get("doc_id")
                    if doc_id:
                        col1 = st.columns([1, 4])[0]
                        with col1:
                            if st.button(
                                get_text("find_related", lang),
                                key=f"related_{doc_id}_{chunk_idx}_{group_idx}",
                                help=get_text("find_related", lang),
                            ):
                                # Store the doc_id to fetch related chunks
                                st.session_state[f"show_related_{doc_id}"] = True

                        # Show related chunks if button was clicked
                        if st.session_state.get(f"show_related_{doc_id}", False):
                            # Cache related chunks in session state
                            cache_key = f"related_chunks_{doc_id}"
                            if cache_key not in st.session_state:
                                with st.spinner(get_text("finding_related", lang)):
                                    import asyncio

                                    try:
                                        st.session_state[cache_key] = asyncio.run(
                                            get_related_chunks(
                                                doc_id, top_k=3, same_video_only=False
                                            )
                                        )
                                    except Exception as e:
                                        st.error(f"{get_text('error_finding_related', lang)} {e}")
                                        st.session_state[cache_key] = []

                            related_chunks = st.session_state[cache_key]
                            if related_chunks:
                                st.markdown(f"**{get_text('related_content', lang)}**")
                                for rel_idx, rel_chunk in enumerate(related_chunks, 1):
                                    rel_filename = html.escape(rel_chunk.get("filename", "Unknown"))
                                    rel_score = rel_chunk.get("score", 0.0)
                                    rel_text = html.escape(rel_chunk.get("text", "")[:150])
                                    st.markdown(
                                        f"{rel_idx}. `{rel_filename}` (Score: {rel_score:.3f})<br>_{rel_text}..._",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info(get_text("no_related", lang))

                    # Do not display transcript text here (VTT viewer already provides full context)

                    # Add a small separator between language versions within a group
                    if chunk_idx < len(group) - 1:
                        st.markdown("---")

                # Add a larger divider between groups
                if group_idx < len(grouped_chunks):
                    st.divider()


def render_sidebar(lang: str):
    """Render the sidebar with controls and system information."""
    with st.sidebar:
        # ===== SEARCH SETTINGS =====
        st.markdown(f"### {get_text('sidebar_search_settings', lang)}")

        # Language selection
        language_options = {"ru": "🇷🇺 Русский", "en": "🇬🇧 English"}
        selected_lang = st.radio(
            get_text("language_label", lang),
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0 if st.session_state.language == "ru" else 1,
            key="lang_select",
            horizontal=True,
        )

        # Update language if changed
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()

        # Number of context chunks
        st.session_state.top_k = st.slider(
            get_text("num_chunks_label", lang),
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            step=1,
        )

        # Date range filter
        with st.expander(get_text("date_filter_label", lang), expanded=False):
            min_date, max_date = get_archive_date_range()
            # Clamp stored dates to available range (if known)
            if min_date and st.session_state.date_from and st.session_state.date_from < min_date:
                st.session_state.date_from = min_date
            if max_date and st.session_state.date_from and st.session_state.date_from > max_date:
                st.session_state.date_from = max_date
            if min_date and st.session_state.date_to and st.session_state.date_to < min_date:
                st.session_state.date_to = min_date
            if max_date and st.session_state.date_to and st.session_state.date_to > max_date:
                st.session_state.date_to = max_date

            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input(
                    get_text("date_from_label", lang),
                    value=st.session_state.date_from,
                    min_value=min_date or date(1900, 1, 1),
                    max_value=max_date or date.today(),
                    key=f"date_from_input_{st.session_state.date_input_reset_counter}",
                )
                st.session_state.date_from = date_from if date_from else None
            with col2:
                date_to = st.date_input(
                    get_text("date_to_label", lang),
                    value=st.session_state.date_to,
                    min_value=min_date or date(1900, 1, 1),
                    max_value=max_date or date.today(),
                    key=f"date_to_input_{st.session_state.date_input_reset_counter}",
                )
                st.session_state.date_to = date_to if date_to else None

            if st.button(get_text("clear_dates", lang), use_container_width=True):
                st.session_state.date_from = None
                st.session_state.date_to = None
                st.session_state.date_input_reset_counter += 1
                st.rerun()

        st.divider()

        # ===== CONVERSATION ACTIONS =====
        st.markdown(f"### {get_text('sidebar_conversation', lang)}")

        # Search history
        if st.session_state.search_history:
            with st.expander(get_text("search_history", lang), expanded=False):
                for idx, query in enumerate(st.session_state.search_history[:5]):
                    if st.button(
                        f"{query[:40]}..." if len(query) > 40 else query,
                        key=f"history_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.messages.append({"role": "user", "content": query})
                        st.rerun()

        # Export conversation
        if st.session_state.messages:
            with st.expander(get_text("export_chat", lang), expanded=False):
                # Generate markdown export
                markdown_content = f"# {get_text('title', lang)}\n\n"
                markdown_content += f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}**\n\n"
                for msg in st.session_state.messages:
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                    markdown_content += f"## {role_label}\n\n{msg['content']}\n\n"

                # Generate text export
                text_content = f"{get_text('title', lang)}\n"
                text_content += f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                text_content += "=" * 50 + "\n\n"
                for msg in st.session_state.messages:
                    role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
                    text_content += f"{role_label}:\n{msg['content']}\n\n{'-' * 50}\n\n"

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=get_text("export_markdown", lang),
                        data=markdown_content,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col2:
                    st.download_button(
                        label=get_text("export_text", lang),
                        data=text_content,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

        # Clear history button
        if st.button(get_text("clear_button", lang), use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.search_history = []
            st.rerun()

        st.divider()

        # ===== SESSION & SYSTEM =====
        st.markdown(f"### {get_text('sidebar_session', lang)}")

        # Session info and logout
        if st.session_state.get("authenticated", False):
            last_activity = st.session_state.get("last_activity")
            if last_activity:
                time_since_activity = datetime.now() - last_activity
                minutes_remaining = session_timeout_minutes - int(
                    time_since_activity.total_seconds() / 60
                )
                if minutes_remaining > 60:
                    hours = minutes_remaining // 60
                    mins = minutes_remaining % 60
                    timeout_str = f"{hours}h {mins}m"
                else:
                    timeout_str = f"{minutes_remaining}m"
                st.caption(f"{get_text('session_expires', lang)} {timeout_str}")

            if st.button(get_text("logout", lang), use_container_width=True, type="secondary"):
                audit_log("LOGOUT", "User logged out", success=True)
                st.session_state.authenticated = False
                st.session_state.last_activity = None
                st.session_state.messages = []
                st.rerun()

        # System information (collapsible + compact)
        with st.expander(get_text("system_info_label", lang), expanded=False):
            with st.spinner(get_text("loading_system", lang)):
                import asyncio

                try:
                    health_info = asyncio.run(check_api_health())
                    if health_info:
                        # Get configuration
                        llm_provider = health_info.get("llm_provider", "Unknown")
                        llm_model = health_info.get("llm_model", "Unknown")
                        embedding_provider = health_info.get("embedding_provider", "Unknown")
                        embedding_model = health_info.get("embedding_model", "Unknown")
                        collection = health_info.get("qdrant_collection", "Unknown")

                        # Search features
                        features: list[str] = []
                        if health_info.get("hybrid_search_enabled", False):
                            features.append(
                                f"Hybrid ({health_info.get('fusion_method', 'rrf').upper()})"
                            )
                        if health_info.get("reranker_enabled", False):
                            features.append("Reranker")
                        if health_info.get("temporal_enabled", False):
                            features.append("Temporal")

                        features_display = " | ".join(features) if features else "Vector"

                        # Simple list format
                        st.text(f"LLM: {llm_provider} ({llm_model})")
                        st.text(f"Embeddings: {embedding_provider} ({embedding_model})")
                        st.text(f"Collection: {collection}")
                        st.text(f"Search: {features_display}")
                    else:
                        st.error(get_text("health_check_failed", lang))
                except Exception as e:
                    logger.error(f"Failed to get health info: {e}")
                    st.error(get_text("health_check_failed", lang))


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="RainRAG - Video Transcript Search",
        page_icon="📹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply security headers (Streamlit doesn't provide direct header control,
    # but these should be set at the reverse proxy level in production)
    # Recommended nginx/caddy headers:
    # - X-Frame-Options: DENY
    # - X-Content-Type-Options: nosniff
    # - X-XSS-Protection: 1; mode=block
    # - Strict-Transport-Security: max-age=31536000; includeSubDomains
    # - Content-Security-Policy: default-src 'self'

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
        }

        /* Download button styling */
        .stDownloadButton > button {
            border-radius: 8px;
        }

        /* Video controls */
        video {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        /* Video player control buttons */
        button[onclick] {
            transition: background-color 0.2s ease;
        }
        button[onclick]:hover {
            background-color: #e0e0e0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    initialize_session_state()

    # Check authentication
    if not check_authentication():
        return

    lang = st.session_state.language

    # Render sidebar
    render_sidebar(lang)

    # Main content
    st.title(get_text("title", lang))
    st.caption(get_text("subtitle", lang))

    st.divider()

    # Display chat messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            render_message_bubble(message, lang)
    else:
        # Show example queries when no messages
        st.markdown(f"### {get_text('example_queries', lang)}")
        st.markdown("")

        examples = st.session_state.example_queries.get(
            lang, st.session_state.example_queries["ru"]
        )
        cols = st.columns(2)

        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                if st.button(example, key=f"example_{idx}", use_container_width=True):
                    # Set the pending query flag
                    st.session_state.pending_query = example
                    st.rerun()

        st.markdown("")
        st.markdown("---")

    # Chat input
    user_input = st.chat_input(get_text("input_placeholder", lang))

    # Check for pending query from example buttons
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Add to search history (keep last 10)
        if user_input not in st.session_state.search_history:
            st.session_state.search_history.insert(0, user_input)
            st.session_state.search_history = st.session_state.search_history[:10]

        # Display user message immediately
        render_message_bubble({"role": "user", "content": user_input}, lang)

        # Show enhanced loading indicator
        status_container = st.empty()
        with st.spinner(get_text("thinking", lang)):
            try:
                import asyncio

                # Show loading steps
                status_container.info(get_text("searching", lang))

                # Query the RAG system
                date_from = (
                    st.session_state.date_from.isoformat() if st.session_state.date_from else None
                )
                date_to = st.session_state.date_to.isoformat() if st.session_state.date_to else None
                response = asyncio.run(
                    query_rag(
                        user_input,
                        st.session_state.language,
                        st.session_state.top_k,
                        date_from=date_from,
                        date_to=date_to,
                    )
                )

                # Add assistant response to chat
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response["answer"],
                    "context": response.get("context", []),
                }
                st.session_state.messages.append(assistant_message)

                # Show completion and clear status
                clip_count = len(response.get("context", []))
                status_container.success(get_text("found_clips", lang).format(count=clip_count))
                time.sleep(0.5)
                status_container.empty()

                # Rerun to display the response
                st.rerun()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    st.error(get_text("error_auth", lang))
                else:
                    st.error(f"{get_text('error_general', lang)}: {e}")
                logger.error(f"HTTP error: {e}")

            except httpx.TimeoutException:
                st.error(get_text("error_timeout", lang))
                logger.error("Request timeout")

            except httpx.ConnectError:
                st.error(get_text("error_connection", lang))
                logger.error(f"Connection error to {API_BASE}")

            except Exception as e:
                st.error(f"{get_text('error_general', lang)}: {str(e)}")
                logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run the main function when the module is executed directly
    main()
