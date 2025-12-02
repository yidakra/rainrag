"""Query interface for RainRAG using Mistral/OpenAI/Claude/Gemini API and Qdrant."""

from typing import Any

import google.generativeai as genai
from anthropic import Anthropic
from loguru import logger
from mistralai import Mistral
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rainrag.config import Config


class RAGQueryEngine:
    """
    RAG Query Engine that retrieves relevant documents and generates answers.

    This class handles the complete query pipeline:
    1. Embed the query using the same model as documents
    2. Search Qdrant for relevant chunks
    3. Build a prompt with retrieved context
    4. Send to Mistral API for answer generation
    """

    def __init__(self, config: Config):
        """
        Initialize the query engine.

        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.embedding_model: SentenceTransformer | None = None
        self.qdrant_client: QdrantClient | None = None

        # Initialize clients based on what's needed for LLM and embeddings
        needs_mistral = config.llm.provider == "mistral" or config.embedding.provider == "mistral"
        needs_openai = config.llm.provider == "openai" or config.embedding.provider == "openai"
        needs_claude = config.llm.provider == "claude"  # Claude only supports LLM, not embeddings
        needs_gemini = config.llm.provider == "gemini" or config.embedding.provider == "gemini"

        # Initialize Mistral client if needed
        if needs_mistral:
            self.mistral_client = Mistral(api_key=config.mistral.api_key)
            logger.info("Initialized Mistral client")
        else:
            self.mistral_client = None

        # Initialize OpenAI client if needed
        if needs_openai:
            self.openai_client = OpenAI(api_key=config.openai.api_key)
            logger.info("Initialized OpenAI client")
        else:
            self.openai_client = None

        # Initialize Claude client if needed
        if needs_claude:
            self.claude_client = Anthropic(api_key=config.claude.api_key)
            logger.info("Initialized Claude client")
        else:
            self.claude_client = None

        # Initialize Gemini client if needed
        if needs_gemini:
            genai.configure(api_key=config.gemini.api_key)
            logger.info("Initialized Gemini client")

        # Log which provider is being used for LLM
        if config.llm.provider == "mistral":
            logger.info(f"Using Mistral for LLM: {config.mistral.model_name}")
        elif config.llm.provider == "openai":
            logger.info(f"Using OpenAI for LLM: {config.openai.model_name}")
        elif config.llm.provider == "claude":
            logger.info(f"Using Claude for LLM: {config.claude.model_name}")
        elif config.llm.provider == "gemini":
            logger.info(f"Using Gemini for LLM: {config.gemini.model_name}")
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm.provider}")

    def initialize(self) -> None:
        """Initialize the embedding model and Qdrant client."""
        logger.info("Initializing query engine...")

        # Load embedding model only if using local provider
        if self.config.embedding.provider == "local":
            logger.info(f"Loading local embedding model: {self.config.embedding.model_name}")
            try:
                try:
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding.model_name,
                        device=self.config.embedding.device,
                        model_kwargs={"dtype": "auto"},  # Prefer new dtype kwarg when supported
                    )
                except TypeError:
                    # Older sentence-transformers versions don't accept model_kwargs
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding.model_name,
                        device=self.config.embedding.device,
                    )
            except OSError as e:
                # Handle offline mode / model not cached
                error_msg = (
                    f"Failed to load embedding model '{self.config.embedding.model_name}'. "
                    f"The model is not cached locally and cannot be downloaded. "
                    f"\n\nTo fix this:"
                    f"\n1. Connect to the internet"
                    f"\n2. Run: python scripts/download_models.py"
                    f"\n3. Or run: poetry run python scripts/download_models.py"
                    f"\n\nOriginal error: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        elif self.config.embedding.provider == "mistral":
            logger.info("Using Mistral API for embeddings (mistral-embed)")
        elif self.config.embedding.provider == "openai":
            logger.info(f"Using OpenAI API for embeddings ({self.config.openai.embedding_model})")
        elif self.config.embedding.provider == "gemini":
            logger.info(f"Using Gemini API for embeddings ({self.config.gemini.embedding_model})")
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding.provider}")

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {self.config.qdrant.host}:{self.config.qdrant.port}")
        # Disable version check to avoid warnings when client/server versions differ slightly
        # The HTTP API is stable and compatible across minor versions
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            prefer_grpc=False,
            api_key=None,  # No authentication for local Qdrant
            timeout=60,
        )

        # Test connection
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(
                f"Connected to Qdrant. Available collections: {len(collections.collections)}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        logger.info("Query engine initialized successfully")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed the query text using configured provider.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding
        """
        if self.config.embedding.provider == "mistral":
            # Use Mistral API embeddings
            logger.debug(f"Embedding query using Mistral API: {query[:100]}...")
            try:
                response = self.mistral_client.embeddings.create(
                    model="mistral-embed", inputs=[query]
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Failed to generate embeddings with Mistral API: {e}")
                raise RuntimeError(f"Mistral embeddings API error: {e}") from e

        elif self.config.embedding.provider == "local":
            # Use local SentenceTransformer model
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized. Call initialize() first.")

            # Add "query: " prefix for E5 model (improves retrieval performance)
            prefixed_query = f"query: {query}"

            logger.debug(f"Embedding query using local model: {query[:100]}...")
            embedding = self.embedding_model.encode(
                prefixed_query,
                normalize_embeddings=self.config.embedding.normalize_embeddings,
            )

            return embedding.tolist()

        elif self.config.embedding.provider == "openai":
            # Use OpenAI API embeddings
            logger.debug(f"Embedding query using OpenAI API: {query[:100]}...")
            try:
                response = self.openai_client.embeddings.create(
                    model=self.config.openai.embedding_model, input=query
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Failed to generate embeddings with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI embeddings API error: {e}") from e

        elif self.config.embedding.provider == "gemini":
            # Use Gemini API embeddings
            logger.debug(f"Embedding query using Gemini API: {query[:100]}...")
            try:
                result = genai.embed_content(
                    model=self.config.gemini.embedding_model,
                    content=query,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except Exception as e:
                logger.error(f"Failed to generate embeddings with Gemini API: {e}")
                raise RuntimeError(f"Gemini embeddings API error: {e}") from e

        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding.provider}")

    def retrieve_documents(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents from Qdrant.

        Args:
            query_vector: The query embedding
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with metadata
        """
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized. Call initialize() first.")

        logger.info(f"Searching for top {top_k} documents...")

        try:
            results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query=query_vector,
                limit=top_k,
            ).points

            documents = []
            for idx, hit in enumerate(results):
                doc = {
                    "rank": idx + 1,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "path": hit.payload.get("path", ""),
                    "language": hit.payload.get("language", ""),
                    "doc_id": hit.payload.get("doc_id", ""),
                    "date": hit.payload.get("date"),
                    "duration_seconds": hit.payload.get("duration_seconds"),
                }
                documents.append(doc)
                logger.debug(f"Rank {idx + 1}: Score={hit.score:.4f}, Path={doc['path']}")

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def build_prompt(
        self, query: str, documents: list[dict[str, Any]], language: str = "en"
    ) -> list[dict[str, str]]:
        """
        Build the messages for the chat LLM with retrieved context.

        Args:
            query: The user's question
            documents: List of retrieved documents
            language: Language code (e.g., "en", "ru") for response

        Returns:
            List of message dictionaries for Mistral API
        """
        # Build context from retrieved documents
        context_parts = []
        max_chars_per_doc = 1200
        for doc in documents:
            text = doc["text"]
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc].rstrip() + "..."
            context_parts.append(f"[Document {doc['rank']}]")
            context_parts.append(f"Source: {doc['path']}")
            # Include date if available
            if doc.get("date"):
                context_parts.append(f"Date: {doc['date']}")
            # Include duration if available (format as mm:ss)
            if doc.get("duration_seconds"):
                mins = int(doc["duration_seconds"] // 60)
                secs = int(doc["duration_seconds"] % 60)
                context_parts.append(f"Duration: {mins}:{secs:02d}")
            context_parts.append(f"Text: {text}")
            context_parts.append("")  # Empty line between documents

        context = "\n".join(context_parts)

        # Language-specific system messages
        system_messages = {
            "ru": """Вы — ассистент для журналистов и редакторов, помогающий находить видеоматериалы в новостном архиве. КРИТИЧЕСКИ ВАЖНО: Вы ДОЛЖНЫ отвечать ТОЛЬКО на русском языке.

ВСЕ ВИДЕО — АРХИВНЫЕ ЗАПИСИ, не текущие новости. Правила ответа:
- Всегда указывайте дату записи из метаданных (поле "Date")
- Упоминайте длительность видео (поле "Duration") — важно для редакторов
- Используйте прошедшее время: "В архивном видео от 2021-05-11 показано..."
- Цитируйте путь к файлу (Source), чтобы редактор мог найти видео
- Если несколько видео релевантны, перечислите каждое с датой и описанием
- Объясните, почему материал может быть полезен для текущего сюжета

Если дата отсутствует, укажите это. Если материал не найден — скажите прямо, не выдумывайте.""",
            "en": """You are an assistant for journalists and editors, helping them find video footage from a news archive. CRITICAL: You MUST answer ONLY in English.

ALL VIDEOS ARE ARCHIVAL RECORDINGS, not current news. Response rules:
- Always cite the recording date from metadata (the "Date" field)
- Mention video duration (the "Duration" field) — important for editors
- Use past tense: "Archive footage from 2021-05-11 shows..."
- Cite the file path (Source) so editors can locate the video
- If multiple videos are relevant, list each with date and description
- Explain why the footage might be useful for the current story

If date is missing, note this. If no relevant footage is found, say so clearly — do not fabricate.""",
        }

        # Get system message (default to English if not found)
        system_message = system_messages.get(language, system_messages["en"])

        # Build user message with context and question
        user_message = f"""Context from video transcripts:
{context}

Question: {query}"""

        # Return messages in Mistral API format
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def generate_answer(self, messages: list[dict[str, str]]) -> str:
        """
        Generate an answer using configured LLM provider.

        Args:
            messages: List of message dictionaries for the chat

        Returns:
            The generated answer text

        Raises:
            RuntimeError: If API call fails
        """
        if self.config.llm.provider == "mistral":
            logger.info("Generating answer using Mistral API...")
            try:
                response = self.mistral_client.chat.complete(
                    model=self.config.mistral.model_name,
                    messages=messages,
                    max_tokens=self.config.mistral.max_tokens,
                    temperature=self.config.mistral.temperature,
                )
                answer = response.choices[0].message.content.strip()
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Mistral API: {e}")
                raise RuntimeError(f"Mistral API error: {e}") from e

        elif self.config.llm.provider == "openai":
            logger.info("Generating answer using OpenAI API...")
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.openai.model_name,
                    messages=messages,
                    max_tokens=self.config.openai.max_tokens,
                    temperature=self.config.openai.temperature,
                )
                answer = response.choices[0].message.content.strip()
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI API error: {e}") from e

        elif self.config.llm.provider == "claude":
            logger.info("Generating answer using Claude API...")
            try:
                # Extract system message and user messages for Claude API
                system_message = ""
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        claude_messages.append(msg)

                response = self.claude_client.messages.create(
                    model=self.config.claude.model_name,
                    max_tokens=self.config.claude.max_tokens,
                    temperature=self.config.claude.temperature,
                    system=system_message,
                    messages=claude_messages,
                )
                answer = response.content[0].text.strip()
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Claude API: {e}")
                raise RuntimeError(f"Claude API error: {e}") from e

        elif self.config.llm.provider == "gemini":
            logger.info("Generating answer using Gemini API...")
            try:
                # Convert messages to Gemini format
                model = genai.GenerativeModel(self.config.gemini.model_name)

                # Extract system message and build conversation
                system_instruction = ""
                conversation_parts = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        conversation_parts.append(msg["content"])
                    elif msg["role"] == "assistant":
                        # Gemini doesn't use explicit assistant messages in the same way
                        # For now, we'll skip assistant messages or handle them differently
                        pass

                # Combine system instruction with user message
                if system_instruction:
                    prompt = f"{system_instruction}\n\n{conversation_parts[-1]}"
                else:
                    prompt = conversation_parts[-1]

                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.config.gemini.max_tokens,
                        temperature=self.config.gemini.temperature,
                    ),
                )
                answer = response.text.strip()
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Gemini API: {e}")
                raise RuntimeError(f"Gemini API error: {e}") from e

        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm.provider}")

    def query(
        self, question: str, top_k: int | None = None, language: str = "en"
    ) -> dict[str, Any]:
        """
        Execute the complete query pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve (defaults to config value)
            language: Language code for response (e.g., "en", "ru")

        Returns:
            Dictionary containing the answer and metadata
        """
        if top_k is None:
            # Get top_k from the appropriate LLM config
            if self.config.llm.provider == "mistral":
                top_k = self.config.mistral.top_k
            elif self.config.llm.provider == "openai":
                top_k = self.config.openai.top_k
            elif self.config.llm.provider == "claude":
                top_k = self.config.claude.top_k
            elif self.config.llm.provider == "gemini":
                top_k = self.config.gemini.top_k
            else:
                top_k = 5  # Default fallback

        logger.info(f"Processing query: {question[:100]}... (language: {language})")

        # Step 1: Embed the query
        query_vector = self.embed_query(question)

        # Step 2: Retrieve relevant documents
        documents = self.retrieve_documents(query_vector, top_k)

        # Step 3: Build the messages with language specification
        messages = self.build_prompt(question, documents, language=language)

        # Step 4: Generate the answer
        answer = self.generate_answer(messages)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": documents,
            "num_documents": len(documents),
        }


def run_query(config_path: str, question: str, top_k: int | None = None) -> dict[str, Any]:
    """
    Run a query against the RAG system.

    This is a convenience function for the CLI.

    Args:
        config_path: Path to configuration file
        question: The user's question
        top_k: Number of documents to retrieve

    Returns:
        Dictionary containing the answer and metadata
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    engine = RAGQueryEngine(config)
    engine.initialize()

    return engine.query(question, top_k)
