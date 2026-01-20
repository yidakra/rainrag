"""Streamlit frontend for RainRAG - Multilingual RAG system for video transcripts."""

import html
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from loguru import logger


# Configuration
API_BASE_URL = os.getenv("RAINRAG_API_URL", "http://localhost:8001").rstrip("/")
# API base for server-side calls (health/query). If API_BASE_URL ends with /api, keep it.
API_BASE = API_BASE_URL
# Asset base for browser-facing URLs (video/vtt/docs). If API_BASE_URL ends with /api, strip it.
ASSET_BASE_URL = API_BASE[:-4] if API_BASE.endswith("/api") else API_BASE
# Allow disabling SSL verification for self-signed/internal certs
API_VERIFY_SSL = os.getenv("RAINRAG_API_VERIFY", "true").lower() not in ("0", "false", "no", "off")
AUTH_TOKEN = os.getenv("STREAMLIT_AUTH_TOKEN", "")
DEFAULT_LANGUAGE = "ru"
DEFAULT_TOP_K = 3
REQUEST_TIMEOUT = 60.0  # 60 seconds timeout for API requests
DOCS_PATH = os.getenv("RAINRAG_DOCS_PATH", "./data/docs.jsonl")


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
    },
}


def get_text(key: str, lang: str = "ru") -> str:
    """Get translated text for given key and language."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["ru"]).get(key, key)


def check_authentication() -> bool:
    """Check if authentication is required and handle login."""
    # If no auth token is configured, skip authentication
    if not AUTH_TOKEN:
        return True

    # Check if user is already authenticated
    if st.session_state.get("authenticated", False):
        return True

    # Show login form
    st.title(get_text("auth_title", st.session_state.get("language", "ru")))

    with st.form("login_form"):
        token_input = st.text_input(
            get_text("auth_prompt", st.session_state.get("language", "ru")),
            type="password",
        )
        submit = st.form_submit_button(
            get_text("auth_button", st.session_state.get("language", "ru"))
        )

        if submit:
            if token_input == AUTH_TOKEN:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error(get_text("auth_invalid", st.session_state.get("language", "ru")))
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
        st.session_state.authenticated = not bool(AUTH_TOKEN)
    if "date_from" not in st.session_state:
        st.session_state.date_from = None
    if "date_to" not in st.session_state:
        st.session_state.date_to = None
    if "date_input_reset_counter" not in st.session_state:
        st.session_state.date_input_reset_counter = 0


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
    payload = {
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


def format_context_chunk(chunk: dict[str, Any], index: int, lang: str) -> str:
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

    return f"""
**{get_text("source_label", lang)}:** `{filename}`{chunk_info}

{" | ".join(meta_parts)}
"""


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

    if role == "user":
        # User message (right-aligned, blue)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                <div style="background-color: #0084ff; color: white; padding: 10px 15px;
                           border-radius: 18px; max-width: 70%; text-align: left;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant message (left-aligned, gray)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="background-color: #f0f0f0; color: black; padding: 10px 15px;
                           border-radius: 18px; max-width: 70%; text-align: left;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show context if available
    if role == "assistant" and "context" in message:
        # Show search insights if we have context
        if message["context"]:
            chunks = message["context"]
            # Detect active features from chunk metadata
            has_reranking = any(c.get("rerank_score") is not None for c in chunks)
            has_time_boost = any(c.get("time_boost") is not None for c in chunks)
            has_hybrid = any(c.get("fusion_method") is not None for c in chunks)

            insights = []
            if has_hybrid:
                fusion_method = next(
                    (chunk.get("fusion_method") for chunk in chunks if chunk.get("fusion_method")),
                    "rrf",
                )
                fusion = fusion_method.upper()
                insights.append(f"Hybrid Search ({fusion})")
            if has_reranking:
                insights.append("Reranked")
            if has_time_boost:
                insights.append("Time-Boosted")

            if insights:
                st.info(f"**Search Features Active:** {' | '.join(insights)}")

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
                            # HTML5 video supports #t=seconds for seeking
                            video_full_url += f"#t={int(float(start_time_seconds))}"

                        try:
                            # Use HTML video player for better compatibility
                            st.markdown(
                                f"""
                                <video controls style="max-width: 100%; height: auto;">
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
                    vtt_languages = {}
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
                            lang_display = {"ru": "Русский 🇷🇺", "en": "English 🇬🇧"}
                            selected_vtt_lang = st.radio(
                                "Language",
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
                            f'<a href="{vtt_full_url}" download="{vtt_filename}" '
                            f'style="display: inline-block; padding: 0.4rem 0.8rem; '
                            f'background-color: #0084ff; color: white; text-decoration: none; '
                            f'border-radius: 0.25rem; text-align: center; width: 100%; '
                            f'box-sizing: border-box; margin-bottom: 0.5rem;">'
                            f'{get_text("download_vtt", lang)}</a>',
                            unsafe_allow_html=True,
                        )

                        # VTT content viewer (scrollable)
                        vtt_content = fetch_vtt_content(vtt_url)
                        if vtt_content:
                            # Display VTT in a scrollable container
                            # Use dark background that works in both light and dark modes
                            st.markdown(
                                f'<div style="height: 400px; overflow-y: auto; '
                                f"border: 1px solid #4a4a4a; border-radius: 0.25rem; "
                                f"padding: 0.5rem; background-color: #1e1e1e; "
                                f"color: #e0e0e0; "
                                f"font-family: monospace; font-size: 0.8rem; "
                                f'white-space: pre-wrap;">{vtt_content}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.error("Could not load VTT")
                    else:
                        st.info("No VTT available")

                # Display text context and metadata below the video/VTT layout
                st.markdown("---")
                for chunk_idx, chunk in enumerate(group):
                    # Display context chunk metadata
                    st.markdown(format_context_chunk(chunk, chunk_idx + 1, lang))

                    # Add "Find Related" button for each chunk
                    doc_id = chunk.get("doc_id")
                    if doc_id:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button(
                                "Find Related",
                                key=f"related_{doc_id}_{chunk_idx}_{group_idx}",
                                help="Find similar content",
                            ):
                                # Store the doc_id to fetch related chunks
                                st.session_state[f"show_related_{doc_id}"] = True

                        # Show related chunks if button was clicked
                        if st.session_state.get(f"show_related_{doc_id}", False):
                            # Cache related chunks in session state
                            cache_key = f"related_chunks_{doc_id}"
                            if cache_key not in st.session_state:
                                with st.spinner("Finding related chunks..."):
                                    import asyncio

                                    try:
                                        st.session_state[cache_key] = asyncio.run(
                                            get_related_chunks(
                                                doc_id, top_k=3, same_video_only=False
                                            )
                                        )
                                    except Exception as e:
                                        st.error(f"Error finding related chunks: {e}")
                                        st.session_state[cache_key] = []

                            related_chunks = st.session_state[cache_key]
                            if related_chunks:
                                st.markdown("**Related Content:**")
                                for rel_idx, rel_chunk in enumerate(related_chunks, 1):
                                    rel_filename = html.escape(rel_chunk.get("filename", "Unknown"))
                                    rel_score = rel_chunk.get("score", 0.0)
                                    rel_text = html.escape(rel_chunk.get("text", "")[:150])
                                    st.markdown(
                                        f"{rel_idx}. `{rel_filename}` (Score: {rel_score:.3f})<br>"
                                        f"_{rel_text}..._",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info("No related chunks found")

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
        # Language selection
        language_options = {"ru": "Русский 🇷🇺", "en": "English 🇬🇧"}
        selected_lang = st.selectbox(
            get_text("language_label", lang),
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0 if st.session_state.language == "ru" else 1,
            key="lang_select",
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

        st.divider()

        # Date range filter
        st.markdown(f"**{get_text('date_filter_label', lang)}**")

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

        # System information (collapsible + compact)
        with st.expander(get_text("system_info_label", lang), expanded=False):
            with st.spinner(get_text("loading_system", lang)):
                import asyncio

                try:
                    health_info = asyncio.run(check_api_health())
                    if health_info:
                        status_color = "🟢" if health_info.get("status") == "healthy" else "🟡"
                        qdrant_status = "🟢" if health_info.get("qdrant_connected") else "🔴"
                        model_status = "🟢" if health_info.get("model_loaded") else "🔴"

                        st.markdown(
                            f"**{get_text('status_label', lang)}:** {status_color} | **Qdrant:** {qdrant_status} | **Embeddings:** {model_status}"
                        )

                        llm_provider = health_info.get("llm_provider", "Unknown")
                        llm_model = health_info.get("llm_model", "Unknown")
                        embedding_provider = health_info.get("embedding_provider", "Unknown")
                        embedding_model = health_info.get("embedding_model", "Unknown")
                        collection_name = health_info.get("qdrant_collection", "Unknown")

                        st.markdown(
                            f"**{get_text('model_label', lang)}:** {llm_provider} ({llm_model})"
                        )
                        st.markdown(
                            f"**{get_text('embedding_label', lang)}:** {embedding_provider} ({embedding_model})"
                        )
                        st.markdown(f"**{get_text('collection_label', lang)}:** {collection_name}")
                    else:
                        st.error(get_text("health_check_failed", lang))
                except Exception as e:
                    logger.error(f"Failed to get health info: {e}")
                    st.error(get_text("health_check_failed", lang))

        st.divider()

        # Clear history button
        if st.button(get_text("clear_button", lang), use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="RainRAG - Video Transcript Search",
        page_icon="📹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        .stButton > button {
            border-radius: 20px;
            width: 100%;
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
    for message in st.session_state.messages:
        render_message_bubble(message, lang)

    # Chat input
    user_input = st.chat_input(get_text("input_placeholder", lang))

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message immediately
        render_message_bubble({"role": "user", "content": user_input}, lang)

        # Show thinking indicator
        with st.spinner(get_text("thinking", lang)):
            try:
                import asyncio

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
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "context": response.get("context", []),
                }
                st.session_state.messages.append(assistant_message)

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
    main()
