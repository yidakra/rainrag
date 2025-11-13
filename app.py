"""Streamlit frontend for RainRAG - Multilingual RAG system for video transcripts."""

import os
import time
from typing import Dict, List, Any, Optional

import streamlit as st
import httpx
from loguru import logger


# Configuration
API_BASE_URL = os.getenv("RAINRAG_API_URL", "http://localhost:8001")
AUTH_TOKEN = os.getenv("STREAMLIT_AUTH_TOKEN", "")
DEFAULT_LANGUAGE = "ru"
DEFAULT_TOP_K = 3
REQUEST_TIMEOUT = 60.0  # 60 seconds timeout for API requests


# Translations
TRANSLATIONS = {
    "ru": {
        "title": "🎬 RainRAG - Поиск по видео-транскриптам",
        "subtitle": "Задайте вопрос о содержимом видео на русском или английском языке",
        "language_label": "Язык / Language",
        "num_chunks_label": "Количество контекстных фрагментов",
        "system_info_label": "Информация о системе",
        "model_label": "Модель LLM",
        "collection_label": "Коллекция Qdrant",
        "status_label": "Статус",
        "connected": "Подключено",
        "disconnected": "Отключено",
        "input_placeholder": "Введите ваш вопрос здесь...",
        "send_button": "Отправить",
        "clear_button": "Очистить историю",
        "context_header": "📄 Найденные фрагменты контекста",
        "error_auth": "❌ Ошибка аутентификации. Проверьте токен доступа.",
        "error_connection": "❌ Ошибка подключения к серверу. Убедитесь, что API запущен.",
        "error_timeout": "⏱️ Превышено время ожидания. Попробуйте снова.",
        "error_general": "❌ Произошла ошибка",
        "thinking": "Думаю...",
        "source_label": "Источник",
        "score_label": "Релевантность",
        "language_field": "Язык",
        "loading_system": "Загрузка информации о системе...",
        "auth_title": "🔐 Вход в систему",
        "auth_prompt": "Введите токен доступа:",
        "auth_button": "Войти",
        "auth_invalid": "Неверный токен доступа",
        "health_check_failed": "Не удалось подключиться к API",
        "video_label": "Видео",
        "vtt_label": "Субтитры",
        "download_vtt": "Скачать VTT",
        "view_vtt": "Просмотр VTT",
        "no_video": "Видео не найдено",
    },
    "en": {
        "title": "🎬 RainRAG - Video Transcript Search",
        "subtitle": "Ask questions about video content in Russian or English",
        "language_label": "Language / Язык",
        "num_chunks_label": "Number of context chunks",
        "system_info_label": "System Information",
        "model_label": "LLM Model",
        "collection_label": "Qdrant Collection",
        "status_label": "Status",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "input_placeholder": "Type your question here...",
        "send_button": "Send",
        "clear_button": "Clear History",
        "context_header": "📄 Retrieved Context Chunks",
        "error_auth": "❌ Authentication error. Please check your access token.",
        "error_connection": "❌ Connection error. Make sure the API is running.",
        "error_timeout": "⏱️ Request timeout. Please try again.",
        "error_general": "❌ An error occurred",
        "thinking": "Thinking...",
        "source_label": "Source",
        "score_label": "Relevance",
        "language_field": "Language",
        "loading_system": "Loading system information...",
        "auth_title": "🔐 System Login",
        "auth_prompt": "Enter access token:",
        "auth_button": "Login",
        "auth_invalid": "Invalid access token",
        "health_check_failed": "Failed to connect to API",
        "video_label": "Video",
        "vtt_label": "Subtitles",
        "download_vtt": "Download VTT",
        "view_vtt": "View VTT",
        "no_video": "Video not found",
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


def get_api_headers() -> Dict[str, str]:
    """Get API request headers including auth token if configured."""
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


async def check_api_health() -> Optional[Dict[str, Any]]:
    """Check API health status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/health", headers=get_api_headers())
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return None


async def query_rag(question: str, language: str, top_k: int) -> Dict[str, Any]:
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
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "language": language, "top_k": top_k},
            headers=get_api_headers(),
        )
        response.raise_for_status()
        return response.json()


def fetch_vtt_content(vtt_url: str) -> Optional[str]:
    """
    Fetch VTT file content from the API.

    Args:
        vtt_url: VTT file URL (relative to API base URL)

    Returns:
        VTT file content as string, or None if failed
    """
    try:
        import requests

        vtt_full_url = f"{API_BASE_URL}{vtt_url}"
        headers = get_api_headers()
        response = requests.get(vtt_full_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch VTT content: {e}")
        return None


def format_context_chunk(chunk: Dict[str, Any], index: int, lang: str) -> str:
    """Format a context chunk for display."""
    filename = chunk.get("filename", "Unknown")
    score = chunk.get("score", 0.0)
    text = chunk.get("text", "")
    chunk_lang = chunk.get("language", "unknown")

    return f"""
**{get_text("source_label", lang)}:** `{filename}`
**{get_text("score_label", lang)}:** {score:.3f} | **{get_text("language_field", lang)}:** {chunk_lang}

{text}
"""


def render_message_bubble(message: Dict[str, Any], lang: str):
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
        with st.expander(get_text("context_header", lang), expanded=False):
            for idx, chunk in enumerate(message["context"], 1):
                # Display context chunk info
                st.markdown(format_context_chunk(chunk, idx, lang))

                # Display video if available
                video_url = chunk.get("video_url")
                if video_url:
                    st.markdown(f"**🎥 {get_text('video_label', lang)}:**")
                    video_full_url = f"{API_BASE_URL}{video_url}"
                    try:
                        st.video(video_full_url)
                    except Exception as e:
                        logger.warning(f"Could not load video: {e}")
                        st.warning(get_text("no_video", lang))
                else:
                    st.info(get_text("no_video", lang))

                # Display VTT download link and viewer if available
                vtt_url = chunk.get("vtt_url")
                if vtt_url:
                    vtt_full_url = f"{API_BASE_URL}{vtt_url}"
                    filename = chunk.get("filename", "subtitle.vtt")
                    vtt_filename = filename.split("/")[-1]  # Get just the filename

                    st.markdown(f"**📄 {get_text('vtt_label', lang)}:**")

                    # Create columns for VTT actions
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(vtt_filename, language="text")
                    with col2:
                        st.markdown(
                            f'<a href="{vtt_full_url}" download="{vtt_filename}" '
                            f'style="display: inline-block; padding: 0.25rem 0.75rem; '
                            f'background-color: #0084ff; color: white; text-decoration: none; '
                            f'border-radius: 0.25rem; text-align: center;">'
                            f'{get_text("download_vtt", lang)}</a>',
                            unsafe_allow_html=True,
                        )

                    # Add expandable VTT content viewer
                    with st.expander(get_text("view_vtt", lang)):
                        vtt_content = fetch_vtt_content(vtt_url)
                        if vtt_content:
                            # Display VTT content with syntax highlighting
                            st.code(vtt_content, language="vtt", line_numbers=True)
                        else:
                            st.error("Could not load VTT content")

                if idx < len(message["context"]):
                    st.divider()


def render_sidebar(lang: str):
    """Render the sidebar with controls and system information."""
    with st.sidebar:
        st.title("⚙️ " + get_text("system_info_label", lang))

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

        # System information
        with st.spinner(get_text("loading_system", lang)):
            import asyncio

            try:
                health_info = asyncio.run(check_api_health())
                if health_info:
                    status_color = "🟢" if health_info.get("status") == "healthy" else "🟡"
                    st.markdown(f"**{get_text('status_label', lang)}:** {status_color} {health_info.get('status', 'unknown').title()}")
                    st.markdown(f"**{get_text('model_label', lang)}:**")
                    st.code(health_info.get("vllm_model", "Unknown"), language="text")
                    st.markdown(f"**{get_text('collection_label', lang)}:**")
                    st.code(health_info.get("qdrant_collection", "Unknown"), language="text")

                    # Connection statuses
                    qdrant_status = "🟢" if health_info.get("qdrant_connected") else "🔴"
                    model_status = "🟢" if health_info.get("model_loaded") else "🔴"
                    st.markdown(f"**Qdrant:** {qdrant_status}")
                    st.markdown(f"**Embedding Model:** {model_status}")
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
        page_icon="🎬",
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
                response = asyncio.run(
                    query_rag(user_input, st.session_state.language, st.session_state.top_k)
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
                logger.error(f"Connection error to {API_BASE_URL}")

            except Exception as e:
                st.error(f"{get_text('error_general', lang)}: {str(e)}")
                logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
