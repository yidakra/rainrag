"""Streamlit frontend for RainRAG - Multilingual RAG system for video transcripts."""

import asyncio
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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import redis
import streamlit as st
import streamlit.components.v1 as components
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from loguru import logger


# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("RAINRAG_API_URL", "http://localhost:8001").rstrip("/")
# API base for server-side calls (health/query)
API_BASE = API_BASE_URL
# Asset base for browser-facing URLs (video/vtt/docs).
# Defaults to API base. Override with RAINRAG_ASSET_URL when assets are served elsewhere.
ASSET_BASE_URL = os.getenv("RAINRAG_ASSET_URL", API_BASE_URL).rstrip("/")
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
REQUEST_TIMEOUT = float(os.getenv("RAINRAG_REQUEST_TIMEOUT_SECONDS", "240"))
DOCS_PATH = os.getenv("RAINRAG_DOCS_PATH", "./data/docs.jsonl")
ENABLE_DATE_FILTER = os.getenv("RAINRAG_ENABLE_DATE_FILTER", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENABLE_RELATED_CHUNKS = os.getenv("RAINRAG_ENABLE_RELATED_CHUNKS", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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
    redis_client.ping()
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
        "metadata_fallback_hits_label": "Восстановлено метаданных (fallback)",
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
        "name_search_toggle": "Поиск по названию видео",
        "name_search_placeholder": "Введите название видео...",
        "name_search_button": "Найти",
        "name_search_no_results": "Видео с таким названием не найдено в локальном архиве.",
        "name_search_no_local_files": "Файлы не найдены в локальном архиве.",
        "name_search_show": "программа",
        "name_search_searching": "Поиск видео по названию...",
        # Mode selector
        "mode_selector_label": "Режим поиска",
        "mode_content": "По содержанию",
        "mode_name": "По названию",
        "mode_video": "Своё видео",
        "mode_content_caption": "Поиск по расшифровкам архива Дождя",
        "mode_name_caption": "Поиск видео в архиве по названию",
        "mode_video_caption": "Загрузите своё видео — мы его расшифруем и ответим на вопросы по нему",
        # Single-video upload mode
        "video_upload_prompt": "Перетащите видео сюда, чтобы расшифровать его и задавать вопросы по нему",
        "video_uploading": "Загрузка видео…",
        "video_queued": "В очереди на обработку…",
        "video_waiting_for_gpu": "Ожидание GPU…",
        "video_waiting_for_transcriber": "Ожидание очереди распознавания…",
        "video_transcribing": "Распознавание речи…",
        "video_indexing": "Индексация транскрипта…",
        "video_ready": "Готово! Задайте вопрос по видео.",
        "video_error": "Ошибка обработки",
        "video_new_video": "Загрузить другое видео",
        "video_cancel": "Отменить",
        "video_chat_placeholder": "Спросите что-нибудь об этом видео…",
        "video_processing_title": "Обрабатываем ваше видео",
        "video_queue_position": "Перед вами в очереди: {n}",
        "video_detected_language": "Язык видео",
        "video_language_auto": "определён автоматически",
        "video_jump_to": "Перейти к моменту:",
        "video_jump_to_fragment": "▶ Смотреть этот фрагмент",
        "video_download_transcript": "⬇ Скачать расшифровку (VTT)",
        "video_url_label": "или введите ссылку на видео",
        "video_url_placeholder": "https://t.me/...",
        "video_url_button": "Загрузить по ссылке",
        "video_url_downloading": "Скачиваем видео…",
        "video_url_empty": "Введите ссылку на видео",
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
        "metadata_fallback_hits_label": "Metadata fallback enrichments",
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
        "name_search_toggle": "Search by video name",
        "name_search_placeholder": "Enter video title...",
        "name_search_button": "Search",
        "name_search_no_results": "No videos with that title found in the local archive.",
        "name_search_no_local_files": "Files not found in local archive.",
        "name_search_show": "show",
        "name_search_searching": "Searching videos by name...",
        # Mode selector
        "mode_selector_label": "Search mode",
        "mode_content": "By content",
        "mode_name": "By title",
        "mode_video": "Your video",
        "mode_content_caption": "Search across the TV Rain archive transcripts",
        "mode_name_caption": "Find archive videos by title",
        "mode_video_caption": "Upload your own video — we'll transcribe it and answer questions about it",
        # Single-video upload mode
        "video_upload_prompt": "Drop a video here to transcribe it and ask questions about it",
        "video_uploading": "Uploading video…",
        "video_queued": "Queued for processing…",
        "video_waiting_for_gpu": "Waiting for GPU…",
        "video_waiting_for_transcriber": "Waiting for a transcription slot…",
        "video_transcribing": "Transcribing…",
        "video_indexing": "Indexing transcript…",
        "video_ready": "Ready! Ask a question about the video.",
        "video_error": "Processing error",
        "video_new_video": "Upload another video",
        "video_cancel": "Cancel",
        "video_chat_placeholder": "Ask anything about this video…",
        "video_processing_title": "Processing your video",
        "video_queue_position": "Uploads ahead of you: {n}",
        "video_detected_language": "Video language",
        "video_language_auto": "detected automatically",
        "video_jump_to": "Jump to:",
        "video_jump_to_fragment": "▶ Play this fragment",
        "video_download_transcript": "⬇ Download transcript (VTT)",
        "video_url_label": "or paste a video link",
        "video_url_placeholder": "https://t.me/...",
        "video_url_button": "Download from link",
        "video_url_downloading": "Downloading video…",
        "video_url_empty": "Please enter a video URL",
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
    data = cast(dict[str, str], redis_client.hgetall(key))

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

    redis_client.hset(key, mapping=data)


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
    count = cast(int, redis_client.hincrby(key, "count", 1))
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

    # Check if user is already authenticated and session is valid
    if st.session_state.get("authenticated", False):
        # Check for session timeout
        if is_session_expired():
            st.session_state["authenticated"] = False
            audit_log("SESSION_EXPIRED", "Session timed out due to inactivity", success=False)
            st.warning(get_text("session_expired_warning", st.session_state.get("language", "ru")))
            time.sleep(1)
            st.rerun()

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
                        "SECURITY WARNING: Using deprecated plain-text AUTH_TOKEN. Please migrate to RAINRAG_PASSWORD_HASH using argon2."
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
    # Search mode: "content" (default RAG), "name" (title search), or "video" (upload)
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "content"
    if "name_search_results" not in st.session_state:
        st.session_state.name_search_results = None
    # Single-video upload mode state
    if "video_session_id" not in st.session_state:
        st.session_state.video_session_id = None
    if "video_messages" not in st.session_state:
        st.session_state.video_messages = []
    if "video_upload_error" not in st.session_state:
        st.session_state.video_upload_error = None
    if "video_seek_seconds" not in st.session_state:
        st.session_state.video_seek_seconds = 0.0
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


def build_asset_url(path_or_url: str) -> str:
    """Build an absolute URL for browser-facing media assets.

    Accepts either relative API paths (e.g. /video/foo.mp4) or already-absolute URLs.
    """
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    if path_or_url.startswith("/"):
        return f"{ASSET_BASE_URL}{path_or_url}"
    return f"{ASSET_BASE_URL}/{path_or_url}"


def build_video_source_urls(
    video_url: str,
    start_time_seconds: float | None = None,
    preferred_quality: str | None = None,
) -> list[str]:
    """Build ordered video source URLs with quality fallback.

    If the URL points to a quality-suffixed MP4 (e.g. *_1080p.mp4), add
    fallback variants so the browser can try lower qualities automatically.
    """
    base_video_url = video_url.split("#", 1)[0]
    start_fragment = ""
    if start_time_seconds is not None:
        with contextlib.suppress(ValueError, TypeError):
            start_fragment = f"#t={int(float(start_time_seconds))}"

    quality_order = ["1080p", "720p", "480p", "360p", "180p"]
    if preferred_quality in quality_order:
        quality_order = [preferred_quality] + [q for q in quality_order if q != preferred_quality]
    candidates: list[str] = []

    match = re.search(r"_(1080p|720p|480p|360p|180p)\.mp4$", base_video_url)
    if match:
        for quality in quality_order:
            candidate = re.sub(
                r"_(1080p|720p|480p|360p|180p)\.mp4$",
                f"_{quality}.mp4",
                base_video_url,
            )
            if candidate not in candidates:
                candidates.append(candidate)
    else:
        candidates.append(base_video_url)

    return [
        append_auth_query(build_asset_url(candidate)) + start_fragment for candidate in candidates
    ]


def build_hls_master_url(video_url: str, start_time_seconds: float | None = None) -> str | None:
    """Build HLS master playlist URL from a /video/... URL."""
    base_video_url = video_url.split("#", 1)[0]
    if not base_video_url.startswith("/video/"):
        return None
    hls_path = "/hls/master/" + base_video_url[len("/video/") :]
    hls_url = append_auth_query(build_asset_url(hls_path))
    # Avoid stale HLS manifests from intermediary caches.
    hls_url = append_auth_query(
        hls_url + ("&" if "?" in hls_url else "?") + f"cb={int(time.time())}"
    )
    if start_time_seconds is not None:
        with contextlib.suppress(ValueError, TypeError):
            hls_url += f"#t={int(float(start_time_seconds))}"
    return hls_url


def render_adaptive_hls_player(
    hls_url: str,
    mp4_fallback_url: str,
    element_id: str,
    height: int = 420,
) -> None:
    """Render an HLS.js player with MP4/native fallback."""
    html_block = f"""
<video id="{html.escape(element_id)}" controls playsinline
       style="width: 100%; height: auto; border-radius: 8px;" preload="metadata"></video>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1"></script>
<script>
(() => {{
  const video = document.getElementById("{html.escape(element_id)}");
  const hlsUrl = {json.dumps(hls_url)};
  const mp4Fallback = {json.dumps(mp4_fallback_url)};
  const resumeKey = `rainrag_resume_{html.escape(element_id)}`;
  const getSavedTime = () => {{
    try {{
      const raw = sessionStorage.getItem(resumeKey);
      if (!raw) return null;
      const parsed = parseFloat(raw);
      return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
    }} catch (e) {{
      return null;
    }}
  }};
  const saveTime = () => {{
    try {{
      if (Number.isFinite(video.currentTime) && video.currentTime > 0) {{
        sessionStorage.setItem(resumeKey, String(video.currentTime));
      }}
    }} catch (e) {{}}
  }};
  const startFromFragment = (() => {{
    try {{
      const frag = new URL(hlsUrl).hash || "";
      const m = frag.match(/t=(\\d+)/);
      return m ? parseInt(m[1], 10) : null;
    }} catch (e) {{
      return null;
    }}
  }})();

  const applyStartTime = () => {{
    const saved = getSavedTime();
    if (saved !== null) {{
      video.currentTime = saved;
      return;
    }}
    if (startFromFragment !== null && !Number.isNaN(startFromFragment)) {{
      video.currentTime = startFromFragment;
    }}
  }};
  let fallbackTriggered = false;
  const fallbackToMp4 = () => {{
    if (fallbackTriggered) return;
    fallbackTriggered = true;
    video.src = mp4Fallback;
    video.load();
    video.addEventListener('loadedmetadata', applyStartTime, {{ once: true }});
  }};
  const clearWatchdogs = () => {{
    clearTimeout(fallbackTimer);
    clearTimeout(playbackStartTimer);
  }};
  const fallbackTimer = setTimeout(() => {{
    // If HLS is stalled (no fatal event), force MP4 fallback.
    if (video.readyState < 1) {{
      fallbackToMp4();
    }}
  }}, 8000);
  let playbackStartTimer = null;
  const armPlaybackWatchdog = () => {{
    if (playbackStartTimer) return;
    playbackStartTimer = setTimeout(() => {{
      // Manifest may parse, but playback can still stall before first frame.
      const noPlaybackProgress = video.currentTime < 0.1 && video.paused;
      if (noPlaybackProgress) {{
        fallbackToMp4();
      }}
    }}, 5000);
  }};
  video.addEventListener('play', armPlaybackWatchdog, {{ once: true }});
  video.addEventListener('playing', clearWatchdogs, {{ once: true }});
  video.addEventListener('timeupdate', saveTime);
  video.addEventListener('pause', saveTime);
  video.addEventListener('timeupdate', () => {{
    if (video.currentTime > 0.1) {{
      clearWatchdogs();
    }}
  }});

  if (window.Hls && window.Hls.isSupported()) {{
    const hls = new window.Hls({{
      startLevel: 0,
      capLevelToPlayerSize: true,
      testBandwidth: false,
    }});
    hls.loadSource(hlsUrl.split('#')[0]);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.MANIFEST_PARSED, () => {{
      applyStartTime();
    }});
    hls.on(window.Hls.Events.ERROR, (_event, data) => {{
      if (data && data.fatal) {{
        clearWatchdogs();
        fallbackToMp4();
      }}
    }});
  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
    video.src = hlsUrl;
    video.addEventListener('loadedmetadata', () => clearWatchdogs(), {{ once: true }});
    video.addEventListener('loadedmetadata', applyStartTime, {{ once: true }});
  }} else {{
    clearWatchdogs();
    fallbackToMp4();
  }}
}})();
</script>
"""
    components.html(html_block, height=height)


def render_html5_video_player(
    source_tags: str,
    track_tags: str,
    element_id: str,
) -> None:
    """Render MP4 player and preserve playback position across Streamlit reruns."""
    html_block = f"""
<video id="{html.escape(element_id)}" controls playsinline
       style="max-width: 100%; height: auto; border-radius: 8px;"
       preload="metadata">
    {source_tags}
    {track_tags}
    Your browser does not support the video tag.
</video>
<script>
(() => {{
  const video = document.getElementById("{html.escape(element_id)}");
  if (!video) return;
  const resumeKey = `rainrag_resume_{html.escape(element_id)}`;
  const loadSavedTime = () => {{
    try {{
      const raw = sessionStorage.getItem(resumeKey);
      if (!raw) return;
      const parsed = parseFloat(raw);
      if (Number.isFinite(parsed) && parsed >= 0) {{
        video.currentTime = parsed;
      }}
    }} catch (e) {{}}
  }};
  const saveTime = () => {{
    try {{
      if (Number.isFinite(video.currentTime) && video.currentTime > 0) {{
        sessionStorage.setItem(resumeKey, String(video.currentTime));
      }}
    }} catch (e) {{}}
  }};
  video.addEventListener("loadedmetadata", loadSavedTime, {{ once: true }});
  video.addEventListener("timeupdate", saveTime);
  video.addEventListener("pause", saveTime);
}})();
</script>
"""
    st.markdown(html_block, unsafe_allow_html=True)


def append_auth_query(url: str) -> str:
    """Append auth token query param for browser media requests when configured."""
    if not AUTH_TOKEN:
        return url

    parts = urlsplit(url)
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_params.setdefault("auth", AUTH_TOKEN)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment)
    )


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


async def search_by_name_api(query: str) -> dict[str, Any]:
    """Call the /search-by-name endpoint and return the parsed response."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.get(
            f"{API_BASE}/search-by-name",
            params={"q": query},
            headers=get_api_headers(),
        )
        response.raise_for_status()
        return response.json()


def get_auth_headers() -> dict[str, str]:
    """Auth-only headers (no Content-Type) for multipart/file requests."""
    headers: dict[str, str] = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


async def upload_video_api(data: bytes, filename: str, content_type: str) -> dict[str, Any]:
    """Upload a video to start a transcription+indexing session."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(1200.0), verify=API_VERIFY_SSL) as client:
        response = await client.post(
            f"{API_BASE}/video-sessions",
            files={"file": (filename, data, content_type or "application/octet-stream")},
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


async def upload_video_url_api(url: str) -> dict[str, Any]:
    """Ask the API to download a video from a URL and start a session."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0), verify=API_VERIFY_SSL) as client:
        response = await client.post(
            f"{API_BASE}/video-sessions/from-url",
            json={"url": url},
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


async def get_video_session_api(session_id: str) -> dict[str, Any]:
    """Fetch the status/progress of a video-upload session."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.get(
            f"{API_BASE}/video-sessions/{session_id}",
            headers=get_api_headers(),
        )
        response.raise_for_status()
        return response.json()


async def query_video_session_api(
    session_id: str, question: str, language: str, top_k: int
) -> dict[str, Any]:
    """Query a single uploaded video's transcript (scoped Q&A)."""
    payload = {"question": question, "language": language, "top_k": top_k}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.post(
            f"{API_BASE}/video-sessions/{session_id}/query",
            json=payload,
            headers=get_api_headers(),
        )
        response.raise_for_status()
        return response.json()


async def delete_video_session_api(session_id: str) -> bool:
    """Delete a video-upload session and its ephemeral collection."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=API_VERIFY_SSL) as client:
        response = await client.delete(
            f"{API_BASE}/video-sessions/{session_id}",
            headers=get_api_headers(),
        )
        return response.status_code in (200, 404)


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

        vtt_full_url = build_asset_url(vtt_url)
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


# Chunks run ~300s of speech, so the full text of five of them buries the panel.
_FRAGMENT_PREVIEW_CHARS = 160


def render_video_session_context(message_key: str, chunks: list[dict[str, Any]], lang: str) -> None:
    """Render retrieved fragments for an uploaded video.

    The archive renderer builds ``/video``, ``/hls`` and ``/vtt`` URLs from the
    transcript path, which only resolve under the archive root; a session
    transcript lives in the session directory, so those routes reject it. The
    session already shows one player above the chat, so a fragment needs only
    its text and a button that seeks that player.
    """
    for idx, chunk in enumerate(chunks):
        start = str(chunk.get("start_time") or "")
        end = str(chunk.get("end_time") or "")
        span = " – ".join(part for part in (start, end) if part)
        score = chunk.get("score")
        heading = f"**{span}**" if span else f"**{idx + 1}**"
        if isinstance(score, (int, float)):
            heading += f" · {score:.3f}"
        st.markdown(heading)

        text = str(chunk.get("text") or "")
        if text:
            # A native <details> rather than st.expander: this already runs
            # inside the context expander, and Streamlit forbids nesting those.
            preview = text[:_FRAGMENT_PREVIEW_CHARS].rstrip()
            if len(text) > _FRAGMENT_PREVIEW_CHARS:
                preview += "…"
            st.markdown(
                f"<details><summary style='cursor: pointer; color: #9aa0a6;'>"
                f"{html.escape(preview)}</summary>"
                f"<div style='white-space: pre-wrap; margin-top: 0.5rem;'>"
                f"{html.escape(text)}</div></details>",
                unsafe_allow_html=True,
            )

        timecodes = extract_timecodes(start, limit=1)
        if timecodes and st.button(
            get_text("video_jump_to_fragment", lang),
            key=f"ctx_seek_{message_key}_{idx}",
        ):
            st.session_state.video_seek_seconds = float(timecodes[0][1])
            st.rerun()

        if idx < len(chunks) - 1:
            st.markdown("---")


def render_message_bubble(message: dict[str, Any], lang: str, video_session_key: str | None = None):
    """Render a message bubble with appropriate styling.

    ``video_session_key`` marks the message as belonging to an uploaded-video
    session, which renders context fragments without archive media widgets.
    """
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

    # Show context if available
    if role == "assistant" and "context" in message:
        # Search insights are available in chunk metadata if needed for debugging

        with st.expander(get_text("context_header", lang), expanded=False):
            if video_session_key is not None:
                render_video_session_context(video_session_key, message["context"], lang)
                return

            fallback_hits = message.get("metadata_fallback_hits")
            if isinstance(fallback_hits, int):
                st.caption(f"{get_text('metadata_fallback_hits_label', lang)}: {fallback_hits}")

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
                        quality_choice = st.selectbox(
                            "Quality",
                            options=["Auto", "1080p", "720p", "480p", "360p", "180p"],
                            index=0,
                            key=f"context_video_quality_{group_idx}",
                            label_visibility="collapsed",
                        )
                        start_time_seconds = group[0].get("start_time_seconds")
                        preferred_quality = None if quality_choice == "Auto" else quality_choice
                        source_urls = build_video_source_urls(
                            video_url,
                            start_time_seconds,
                            preferred_quality=preferred_quality,
                        )
                        source_tags = "\n".join(
                            f'<source src="{html.escape(url)}" type="video/mp4">'
                            for url in source_urls
                        )

                        try:
                            rendered_hls = False
                            if quality_choice == "Auto":
                                hls_url = build_hls_master_url(video_url, start_time_seconds)
                                if hls_url:
                                    render_adaptive_hls_player(
                                        hls_url=hls_url,
                                        mp4_fallback_url=source_urls[0],
                                        element_id=f"ctx_hls_{group_idx}",
                                    )
                                    rendered_hls = True
                            if not rendered_hls:
                                # HTML5 video player so the browser streams /video directly
                                st.markdown(
                                    f"""
                                    <video controls playsinline
                                           style="max-width: 100%; height: auto; border-radius: 8px;"
                                           preload="metadata">
                                        {source_tags}
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

                            def format_vtt_lang(code: str, display=lang_display) -> str:
                                return display.get(code, code)

                            selected_vtt_lang = st.radio(
                                get_text("vtt_language", lang),
                                options=list(vtt_languages.keys()),
                                format_func=format_vtt_lang,
                                horizontal=True,
                                key=f"vtt_lang_{group_idx}",
                                label_visibility="collapsed",
                            )
                        else:
                            selected_vtt_lang = list(vtt_languages.keys())[0]

                        # Get selected VTT info
                        vtt_info = vtt_languages[selected_vtt_lang]
                        vtt_url = vtt_info["url"]
                        vtt_full_url = append_auth_query(build_asset_url(vtt_url))
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
                    if ENABLE_RELATED_CHUNKS and doc_id:
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


def render_name_search_result(result: dict[str, Any], idx: int, lang: str):
    """Render a single video name search result card."""
    name = result.get("name", "")
    date = result.get("date", "")
    web_url = sanitize_web_url(result.get("web_url", "") or "")
    teleshow_name = result.get("teleshow_name", "")
    languages: dict[str, Any] = result.get("languages", {})

    # Title row
    title_parts = [f"**{html.escape(name)}**"]
    if teleshow_name:
        title_parts.append(f"_{get_text('name_search_show', lang)}: {html.escape(teleshow_name)}_")
    if date:
        title_parts.append(date[:10])
    st.markdown(" · ".join(title_parts))

    if web_url:
        st.markdown(f"[{get_text('url_label', lang)}]({web_url})")

    if not languages:
        st.warning(get_text("name_search_no_local_files", lang))
        st.divider()
        return

    video_col, vtt_col = st.columns([2, 1])

    with video_col:
        st.markdown(f"**{get_text('video_label', lang)}:**")
        # Use the first available language for the video player (both share the same file)
        video_url = None
        for lang_key in ("ru", "en"):
            v = languages.get(lang_key, {}).get("video_url")
            if v:
                video_url = v
                break
        if video_url:
            quality_choice = st.selectbox(
                "Quality",
                options=["Auto", "1080p", "720p", "480p", "360p", "180p"],
                index=0,
                key=f"name_video_quality_{idx}",
                label_visibility="collapsed",
            )
            preferred_quality = None if quality_choice == "Auto" else quality_choice
            source_urls = build_video_source_urls(video_url, preferred_quality=preferred_quality)
            source_tags = "\n".join(
                f'<source src="{html.escape(url)}" type="video/mp4">' for url in source_urls
            )
            # Build <track> elements for every available VTT language
            track_labels = {"ru": "Russian", "en": "English"}
            track_tags = ""
            for t_idx, t_lang in enumerate(("ru", "en")):
                t_vtt = languages.get(t_lang, {}).get("vtt_url")
                if t_vtt:
                    t_full = html.escape(append_auth_query(build_asset_url(t_vtt)))
                    default_attr = " default" if t_idx == 0 else ""
                    track_tags += (
                        f'<track kind="subtitles" src="{t_full}"'
                        f' srclang="{t_lang}" label="{track_labels[t_lang]}"{default_attr}>\n'
                    )
            if quality_choice == "Auto":
                hls_url = build_hls_master_url(video_url)
                if hls_url:
                    render_adaptive_hls_player(
                        hls_url=hls_url,
                        mp4_fallback_url=source_urls[0],
                        element_id=f"name_hls_{idx}",
                    )
                else:
                    render_html5_video_player(
                        source_tags=source_tags,
                        track_tags=track_tags,
                        element_id=f"name_mp4_{idx}",
                    )
            else:
                render_html5_video_player(
                    source_tags=source_tags,
                    track_tags=track_tags,
                    element_id=f"name_mp4_{idx}",
                )
        else:
            st.info(get_text("no_video", lang))

    with vtt_col:
        st.markdown(f"**{get_text('vtt_label', lang)}:**")
        lang_display = {"ru": "🇷🇺 Русский", "en": "🇬🇧 English"}
        available_langs = [lk for lk in ("ru", "en") if lk in languages]
        if available_langs:
            if len(available_langs) > 1:
                selected_vtt_lang = st.radio(
                    get_text("vtt_language", lang),
                    options=available_langs,
                    format_func=lambda c: lang_display.get(c, c),
                    horizontal=True,
                    key=f"ns_vtt_lang_{idx}",
                    label_visibility="collapsed",
                )
            else:
                selected_vtt_lang = available_langs[0]

            vtt_url = languages[selected_vtt_lang].get("vtt_url")
            if vtt_url:
                vtt_full_url = append_auth_query(build_asset_url(vtt_url))
                vtt_filename = vtt_url.split("/")[-1]
                st.markdown(
                    f'<a href="{vtt_full_url}" download="{vtt_filename}" style="display: inline-block; padding: 0.4rem 0.8rem; background-color: #0084ff; color: white; text-decoration: none; border-radius: 0.25rem; text-align: center; width: 100%; box-sizing: border-box; margin-bottom: 0.5rem;">{get_text("download_vtt", lang)}</a>',
                    unsafe_allow_html=True,
                )
                # Cache VTT content by URL to avoid re-fetching on every Streamlit rerun
                cache_key = f"ns_vtt_{vtt_url}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = fetch_vtt_content(vtt_url)

                vtt_content = st.session_state[cache_key]
                if vtt_content:
                    # Escape content to prevent XSS via malformed/malicious VTT markup
                    st.markdown(
                        f'<div style="height: 400px; overflow-y: auto; border: 1px solid #4a4a4a; border-radius: 0.25rem; padding: 0.5rem; background-color: #1e1e1e; color: #e0e0e0; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap;">{html.escape(vtt_content)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(get_text("vtt_error", lang))
        else:
            st.info(get_text("no_vtt", lang))

    st.divider()


# HH:MM:SS or MM:SS, as cited by the single-video prompt.
_TIMECODE_PATTERN = re.compile(r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b")

# Human-readable names for the language codes Whisper reports most often here.
_LANGUAGE_NAMES = {
    "ru": {"ru": "русский", "en": "Russian"},
    "en": {"ru": "английский", "en": "English"},
    "uk": {"ru": "украинский", "en": "Ukrainian"},
    "be": {"ru": "белорусский", "en": "Belarusian"},
    "de": {"ru": "немецкий", "en": "German"},
    "fr": {"ru": "французский", "en": "French"},
    "es": {"ru": "испанский", "en": "Spanish"},
    "ka": {"ru": "грузинский", "en": "Georgian"},
    "hy": {"ru": "армянский", "en": "Armenian"},
    "kk": {"ru": "казахский", "en": "Kazakh"},
}


def language_display_name(code: str, ui_lang: str) -> str:
    """Render a detected language code for humans, falling back to the code."""
    return _LANGUAGE_NAMES.get(code, {}).get(ui_lang, code.upper())


def extract_timecodes(text: str, limit: int = 12) -> list[tuple[str, int]]:
    """Extract cited timecodes from an answer as (label, seconds), deduplicated.

    Used to turn the timecodes the model cites into seek buttons.
    """
    found: list[tuple[str, int]] = []
    seen: set[int] = set()
    for match in _TIMECODE_PATTERN.finditer(text):
        seconds = int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
        if seconds in seen:
            continue
        seen.add(seconds)
        found.append((match.group(0), seconds))
        if len(found) >= limit:
            break
    return found


# Bounds for the player frame: tall uploads (4:3, phone video) must not push the
# chat off screen, and a very wide one must stay big enough to use.
_PLAYER_MIN_HEIGHT = 260
_PLAYER_MAX_HEIGHT = 720
# Typical content width of the main column. Only used to turn an aspect ratio
# into a frame height; the video is centred and scaled to fit, so a narrower
# window costs some empty frame rather than cropping anything.
_PLAYER_ASSUMED_WIDTH = 1100


def player_frame_height(width: int | None, height: int | None) -> int:
    """Frame height that shows an upload of this shape whole.

    The component iframe's height is fixed when Streamlit renders it, and
    growing it from inside does not reflow the elements below -- they end up
    drawn over the video. So the height is decided here, from the real
    dimensions when ffprobe supplied them and 16:9 when it did not.
    """
    ratio = 9 / 16
    if width and height and width > 0 and height > 0:
        ratio = height / width
    wanted = round(_PLAYER_ASSUMED_WIDTH * ratio)
    return int(min(max(wanted, _PLAYER_MIN_HEIGHT), _PLAYER_MAX_HEIGHT))


def render_video_session_player(
    media_url: str,
    element_id: str,
    start_seconds: float = 0.0,
    height: int = 420,
) -> None:
    """Play an uploaded session's video, optionally seeking to a cited timecode.

    Rendered through components.html (an iframe) rather than st.markdown so the
    seek/resume script actually executes. An explicit start time wins over the
    remembered position, since it means the user just clicked a timecode.

    ``height`` comes from :func:`player_frame_height`. Inside the frame the
    video is scaled to fit rather than stretched, so whatever the caller got
    wrong shows up as empty space, never as a cropped picture or hidden
    controls.
    """
    html_block = f"""
<style>
  html, body {{
    margin: 0; padding: 0; height: 100%; background: transparent;
    display: flex; align-items: center; justify-content: center;
  }}
  #{html.escape(element_id)} {{
    max-width: 100%; max-height: 100%; width: auto; height: auto;
    display: block; border-radius: 8px; background: #000;
  }}
</style>
<video id="{html.escape(element_id)}" controls playsinline preload="metadata">
    <source src="{html.escape(media_url)}">
</video>
<script>
(() => {{
  const video = document.getElementById("{html.escape(element_id)}");
  if (!video) return;
  const startAt = {json.dumps(float(start_seconds))};
  const resumeKey = `rainrag_session_resume_{html.escape(element_id)}`;
  const applyStart = () => {{
    if (startAt > 0) {{
      video.currentTime = startAt;
      video.play().catch(() => {{}});
      return;
    }}
    try {{
      const raw = sessionStorage.getItem(resumeKey);
      const parsed = raw === null ? NaN : parseFloat(raw);
      if (Number.isFinite(parsed) && parsed > 0) {{
        video.currentTime = parsed;
      }}
    }} catch (e) {{}}
  }};
  const saveTime = () => {{
    try {{
      if (Number.isFinite(video.currentTime) && video.currentTime > 0) {{
        sessionStorage.setItem(resumeKey, String(video.currentTime));
      }}
    }} catch (e) {{}}
  }};
  video.addEventListener("loadedmetadata", applyStart, {{ once: true }});
  video.addEventListener("timeupdate", saveTime);
  video.addEventListener("pause", saveTime);
}})();
</script>
"""
    components.html(html_block, height=height)


def render_timecode_buttons(message_key: str, content: str, lang: str) -> None:
    """Render the timecodes cited in an answer as buttons that seek the player."""
    timecodes = extract_timecodes(content)
    if not timecodes:
        return
    st.caption(get_text("video_jump_to", lang))
    columns = st.columns(min(len(timecodes), 6))
    for idx, (label, seconds) in enumerate(timecodes):
        with columns[idx % len(columns)]:
            if st.button(label, key=f"seek_{message_key}_{idx}", use_container_width=True):
                st.session_state.video_seek_seconds = float(seconds)
                st.rerun()


def _reset_video_session(delete_remote: bool = True) -> None:
    """Clear local video-mode state and optionally delete the remote session."""
    session_id = st.session_state.get("video_session_id")
    if delete_remote and session_id:
        try:
            asyncio.run(delete_video_session_api(session_id))
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning(f"Failed to delete video session {session_id}: {exc}")
    st.session_state.video_session_id = None
    st.session_state.video_messages = []
    st.session_state.video_upload_error = None
    st.session_state.video_seek_seconds = 0.0
    # st.file_uploader keeps its value across reruns, so without a fresh widget
    # key the next rerun lands on the uploader branch with the previous file
    # still set and immediately re-uploads it.
    st.session_state.video_uploader_seq = int(st.session_state.get("video_uploader_seq", 0)) + 1


def render_video_mode(lang: str):
    """Upload a single video, transcribe it, and chat scoped to that video."""
    session_id = st.session_state.get("video_session_id")

    # --- No active session: show the uploader ---
    if not session_id:
        if st.session_state.get("video_upload_error"):
            st.error(st.session_state.video_upload_error)
            st.session_state.video_upload_error = None
        st.markdown(f"### {get_text('video_upload_prompt', lang)}")
        uploaded = st.file_uploader(
            get_text("video_upload_prompt", lang),
            type=["mp4", "mkv", "webm", "avi", "mov", "m4v"],
            label_visibility="collapsed",
            key=f"video_uploader_{st.session_state.get('video_uploader_seq', 0)}",
        )
        if uploaded is not None:
            with st.spinner(get_text("video_uploading", lang)):
                try:
                    result = asyncio.run(
                        upload_video_api(
                            uploaded.getvalue(),
                            uploaded.name,
                            uploaded.type or "application/octet-stream",
                        )
                    )
                    st.session_state.video_session_id = result.get("id")
                    st.session_state.video_messages = []
                    st.rerun()
                except httpx.HTTPStatusError as e:
                    detail = e.response.text
                    st.session_state.video_upload_error = (
                        f"{get_text('error_general', lang)}: {detail}"
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.session_state.video_upload_error = f"{get_text('error_general', lang)}: {e}"
                    st.rerun()

        st.divider()
        st.caption(get_text("video_url_label", lang))
        url_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
        with url_col:
            video_url_input = st.text_input(
                get_text("video_url_label", lang),
                placeholder=get_text("video_url_placeholder", lang),
                label_visibility="collapsed",
                key="video_url_input",
            )
        with btn_col:
            url_submit = st.button(get_text("video_url_button", lang), use_container_width=True)
        if url_submit:
            url_val = (video_url_input or "").strip()
            if not url_val:
                st.session_state.video_upload_error = get_text("video_url_empty", lang)
                st.rerun()
            else:
                with st.spinner(get_text("video_url_downloading", lang)):
                    try:
                        result = asyncio.run(upload_video_url_api(url_val))
                        st.session_state.video_session_id = result.get("id")
                        st.session_state.video_messages = []
                        st.rerun()
                    except httpx.HTTPStatusError as e:
                        detail = e.response.text
                        st.session_state.video_upload_error = (
                            f"{get_text('error_general', lang)}: {detail}"
                        )
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.session_state.video_upload_error = (
                            f"{get_text('error_general', lang)}: {e}"
                        )
                        st.rerun()
        return

    # --- Active session: fetch status ---
    try:
        status = asyncio.run(get_video_session_api(session_id))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Session expired / gone on the server — reset locally.
            _reset_video_session(delete_remote=False)
            st.rerun()
        st.error(f"{get_text('error_general', lang)}: {e}")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"{get_text('error_general', lang)}: {e}")
        return

    state = status.get("status", "queued")
    filename = status.get("filename", "")

    processing = state in ("queued", "transcribing", "indexing")

    header_col, btn_col = st.columns([4, 1])
    with header_col:
        if filename:
            st.markdown(f"**{html.escape(filename)}**")
    with btn_col:
        # While work is in flight the same action means "stop this", and the
        # backend kills the transcriber so the GPU goes to the next upload.
        button_label = get_text("video_cancel" if processing else "video_new_video", lang)
        if st.button(button_label, use_container_width=True):
            _reset_video_session()
            st.rerun()

    detected_language = status.get("language")

    # --- Still processing: show progress and auto-refresh ---
    if processing:
        stage = status.get("stage", state)
        percent = float(status.get("percent") or 0.0)
        st.markdown(f"#### {get_text('video_processing_title', lang)}")
        # Prefer the fine-grained stage label (e.g. waiting_for_gpu) when the
        # backend reports one we have a translation for; else fall back to status.
        known_stages = (
            "queued",
            "waiting_for_gpu",
            "waiting_for_transcriber",
            "transcribing",
            "indexing",
        )
        stage_key = stage if stage in known_stages else state
        st.caption(get_text(f"video_{stage_key}", lang))
        queue_position = int(status.get("queue_position") or 0)
        if stage in ("waiting_for_gpu", "waiting_for_transcriber") and queue_position > 0:
            st.caption(get_text("video_queue_position", lang).format(n=queue_position))
        if detected_language:
            st.caption(
                f"{get_text('video_detected_language', lang)}: "
                f"{language_display_name(detected_language, lang)}"
            )
        st.progress(min(max(percent / 100.0, 0.0), 1.0))
        time.sleep(2.0)
        st.rerun()
        return

    # --- Error ---
    if state == "error":
        st.error(f"{get_text('video_error', lang)}: {status.get('error', '')}")
        return

    # --- Ready: player + chat scoped to this video ---
    media_url = append_auth_query(build_asset_url(f"/video-sessions/{session_id}/media"))
    render_video_session_player(
        media_url,
        f"video_session_{session_id}",
        start_seconds=float(st.session_state.get("video_seek_seconds") or 0.0),
        height=player_frame_height(status.get("width"), status.get("height")),
    )
    if detected_language:
        st.caption(
            f"{get_text('video_detected_language', lang)}: "
            f"{language_display_name(detected_language, lang)} "
            f"({get_text('video_language_auto', lang)})"
        )
    st.success(get_text("video_ready", lang))

    # Sessions are ephemeral, so this is the only way to keep the transcript.
    transcript_url = append_auth_query(build_asset_url(f"/video-sessions/{session_id}/transcript"))
    download_name = f"{Path(status.get('filename') or 'transcript').stem}.vtt"
    st.markdown(
        f'<a href="{html.escape(transcript_url)}" download="{html.escape(download_name)}" '
        f'style="display: inline-block; padding: 0.4rem 0.8rem; background-color: #0084ff; '
        f'color: white; text-decoration: none; border-radius: 0.25rem; margin-bottom: 0.5rem;">'
        f"{get_text('video_download_transcript', lang)}</a>",
        unsafe_allow_html=True,
    )

    for idx, message in enumerate(st.session_state.video_messages):
        render_message_bubble(message, lang, video_session_key=f"{session_id}_{idx}")
        if message.get("role") == "assistant":
            render_timecode_buttons(f"{session_id}_{idx}", message.get("content", ""), lang)

    user_input = st.chat_input(get_text("video_chat_placeholder", lang))
    if user_input:
        st.session_state.video_messages.append({"role": "user", "content": user_input})
        render_message_bubble({"role": "user", "content": user_input}, lang)
        with st.spinner(get_text("thinking", lang)):
            try:
                response = asyncio.run(
                    query_video_session_api(
                        session_id,
                        user_input,
                        st.session_state.language,
                        st.session_state.top_k,
                    )
                )
                st.session_state.video_messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "context": response.get("context", []),
                    }
                )
                st.rerun()
            except httpx.HTTPStatusError as e:
                st.error(f"{get_text('error_general', lang)}: {e}")
            except httpx.TimeoutException:
                st.error(get_text("error_timeout", lang))
            except Exception as e:  # noqa: BLE001
                st.error(f"{get_text('error_general', lang)}: {e}")


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

        # Optional date range filter (disabled by default for deployed UI)
        if ENABLE_DATE_FILTER:
            with st.expander(get_text("date_filter_label", lang), expanded=False):
                min_date, max_date = get_archive_date_range()
                # Clamp stored dates to available range (if known)
                if (
                    min_date
                    and st.session_state.date_from
                    and st.session_state.date_from < min_date
                ):
                    st.session_state.date_from = min_date
                if (
                    max_date
                    and st.session_state.date_from
                    and st.session_state.date_from > max_date
                ):
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
        else:
            st.session_state.date_from = None
            st.session_state.date_to = None

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

    # Search-mode selector: content RAG / name search / single-video upload
    modes = ["content", "name", "video"]
    current_mode = (
        st.session_state.search_mode if st.session_state.search_mode in modes else "content"
    )
    selected_mode = st.segmented_control(
        get_text("mode_selector_label", lang),
        modes,
        default=current_mode,
        format_func=lambda m: get_text(f"mode_{m}", lang),
        label_visibility="collapsed",
        key="search_mode_selector_widget",
    )
    # segmented_control returns None when the active option is clicked again;
    # treat that as "keep the current mode" so the UI never ends up mode-less.
    if selected_mode is None:
        selected_mode = current_mode
    st.session_state.search_mode = selected_mode

    # Tell the user what the active mode actually does.
    st.caption(get_text(f"mode_{selected_mode}_caption", lang))

    st.divider()

    # ---- SINGLE-VIDEO UPLOAD MODE ----
    if st.session_state.search_mode == "video":
        render_video_mode(lang)
        return

    # ---- NAME SEARCH MODE ----
    if st.session_state.search_mode == "name":
        search_col, btn_col = st.columns([5, 1])
        with search_col:
            name_query = st.text_input(
                get_text("name_search_toggle", lang),
                placeholder=get_text("name_search_placeholder", lang),
                label_visibility="collapsed",
                key="name_search_input",
            )
        with btn_col:
            name_search_clicked = st.button(
                get_text("name_search_button", lang),
                use_container_width=True,
            )

        if name_search_clicked and name_query.strip():
            with st.spinner(get_text("name_search_searching", lang)):
                try:
                    import asyncio

                    response = asyncio.run(search_by_name_api(name_query.strip()))
                    st.session_state.name_search_results = response.get("results", [])
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        st.error(get_text("error_auth", lang))
                    else:
                        st.error(f"{get_text('error_general', lang)}: {e}")
                    st.session_state.name_search_results = []
                except httpx.ConnectError:
                    st.error(get_text("error_connection", lang))
                    st.session_state.name_search_results = []
                except Exception as e:
                    st.error(f"{get_text('error_general', lang)}: {e}")
                    st.session_state.name_search_results = []

        results = st.session_state.get("name_search_results")
        if results is not None:
            if not results:
                st.info(get_text("name_search_no_results", lang))
            else:
                for idx, result in enumerate(results):
                    render_name_search_result(result, idx, lang)
        return

    # ---- CONTENT SEARCH MODE (existing RAG chat) ----
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
                date_from = None
                date_to = None
                if ENABLE_DATE_FILTER:
                    date_from = (
                        st.session_state.date_from.isoformat()
                        if st.session_state.date_from
                        else None
                    )
                    date_to = (
                        st.session_state.date_to.isoformat() if st.session_state.date_to else None
                    )
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
                    "metadata_fallback_hits": response.get("metadata_fallback_hits"),
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
