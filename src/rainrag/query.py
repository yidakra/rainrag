"""Query interface for RainRAG using Mistral/OpenAI/Claude/Gemini API and Qdrant."""

from typing import Any

import cohere
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

        # Initialize Cohere client if reranker is enabled
        if config.reranker.enabled and config.reranker.provider == "cohere":
            try:
                # Newer SDK
                self.cohere_client = cohere.ClientV2(api_key=config.cohere.api_key)
            except AttributeError:
                # Older SDK fallback
                self.cohere_client = cohere.Client(api_key=config.cohere.api_key)
            logger.info(f"Initialized Cohere reranker: {config.cohere.model_name}")
        else:
            self.cohere_client = None

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
                response = self.openai_client.embeddings.create(  # type: ignore[assignment]
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

    def retrieve_documents(
        self,
        query_vector: list[float],
        top_k: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents from Qdrant.

        Args:
            query_vector: The query embedding
            top_k: Number of documents to retrieve
            date_from: Filter results from this date (YYYY-MM-DD)
            date_to: Filter results up to this date (YYYY-MM-DD)

        Returns:
            List of retrieved documents with metadata
        """
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized. Call initialize() first.")

        logger.info(f"Searching for top {top_k} documents...")

        # Build date filter if specified - we'll do client-side filtering
        # to avoid Qdrant issues with None date values
        query_filter = None
        if date_from or date_to:
            logger.info(f"Will apply client-side date filter: {date_from} to {date_to}")

        try:
            # If date filtering is requested, retrieve more documents and filter client-side
            # to handle cases where some documents don't have dates
            effective_limit = top_k * 3 if (date_from or date_to) else top_k

            logger.info(f"Querying Qdrant with filter: {query_filter}, limit: {effective_limit}")
            results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query=query_vector,
                limit=effective_limit,
                query_filter=query_filter,
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
                    "start_time": hit.payload.get("start_time"),
                    "end_time": hit.payload.get("end_time"),
                }
                documents.append(doc)
                logger.debug(f"Rank {idx + 1}: Score={hit.score:.4f}, Path={doc['path']}")

            # Apply date filtering client-side if requested
            if date_from or date_to:
                from datetime import date as _date
                from datetime import datetime as _dt

                def _parse_date(value: Any) -> _date | None:
                    """Best-effort parse for ISO date strings or datetime/date objects."""
                    if value is None:
                        return None
                    if isinstance(value, _dt):
                        return value.date()
                    if isinstance(value, _date):
                        return value
                    try:
                        return _dt.fromisoformat(str(value)).date()
                    except Exception:
                        return None

                parsed_from = _parse_date(date_from)
                parsed_to = _parse_date(date_to)

                filtered_documents = []
                for doc in documents:
                    doc_date_raw = doc.get("date")
                    doc_date = _parse_date(doc_date_raw)
                    if doc_date is None:
                        continue  # Skip documents without dates

                    include_doc = True
                    if parsed_from and doc_date < parsed_from:
                        include_doc = False
                    if parsed_to and doc_date > parsed_to:
                        include_doc = False

                    if include_doc:
                        filtered_documents.append(doc)

                # Re-rank and limit to top_k
                documents = filtered_documents[:top_k]

                # Update ranks after filtering
                for idx, doc in enumerate(documents):
                    doc["rank"] = idx + 1

            logger.info(f"Retrieved {len(documents)} documents after filtering")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def rerank_documents(
        self, query: str, documents: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using Cohere Rerank API.

        Args:
            query: The user's query
            documents: List of retrieved documents
            top_n: Number of documents to return after reranking

        Returns:
            Reranked list of documents
        """
        if not self.cohere_client:
            logger.warning("Cohere client not initialized, skipping reranking")
            return documents[:top_n]

        if not documents:
            return documents

        logger.info(f"Reranking {len(documents)} documents with Cohere...")

        try:
            # Prepare documents for reranking
            texts = [doc["text"] for doc in documents]

            # Call Cohere Rerank API
            response = self.cohere_client.rerank(
                model=self.config.cohere.model_name,
                query=query,
                documents=texts,
                top_n=min(top_n, len(documents)),
            )

            # Reorder documents based on rerank results
            reranked = []
            for idx, result in enumerate(response.results):
                doc = documents[result.index].copy()
                doc["rank"] = idx + 1
                doc["rerank_score"] = result.relevance_score
                doc["original_score"] = doc["score"]
                doc["score"] = result.relevance_score  # Use rerank score as primary
                reranked.append(doc)

            logger.info(f"Reranking complete. Returning top {len(reranked)} documents")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original documents.")
            return documents[:top_n]

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
            List of message dictionaries for the chat API
        """
        # Build context from retrieved documents
        context_parts = []
        # Allow more context per document
        max_chars_per_doc = 2000
        for doc in documents:
            text = doc["text"]
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc].rstrip() + "..."

            # Document header
            doc_header = f"[Document {doc['rank']}]"
            if doc.get("is_chunk"):
                doc_header += f" [Chunk {doc.get('chunk_index', 0) + 1}/{doc.get('total_chunks', 1)}]"
            context_parts.append(doc_header)

            # Include date if available
            if doc.get("date"):
                context_parts.append(f"Date: {doc['date']}")

            # Include duration if available (format as mm:ss)
            if doc.get("duration_seconds"):
                mins = int(doc["duration_seconds"] // 60)
                secs = int(doc["duration_seconds"] % 60)
                context_parts.append(f"Duration: {mins}:{secs:02d}")

            # Include timecodes if available
            if doc.get("start_time") and doc.get("end_time"):
                context_parts.append(f"Timecodes: {doc['start_time']} - {doc['end_time']}")

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
- Если несколько видео релевантны, перечислите каждое с датой и описанием
- Делайте хорошее, развернутое "Описание" для каждого релевантного видео
- Объясните, почему материал может быть полезен для текущего сюжета

Если дата отсутствует, укажите это. Если материал не найден — скажите прямо, не выдумывайте.""",
            "en": """You are an assistant for journalists and editors, helping them find video footage from a news archive. CRITICAL: You MUST answer ONLY in English.

ALL VIDEOS ARE ARCHIVAL RECORDINGS, not current news. Response rules:
- Always cite the recording date from metadata (the "Date" field)
- Mention video duration (the "Duration" field) — important for editors
- Use past tense: "Archive footage from 2021-05-11 shows..."
- If multiple videos are relevant, list each with date and description
- Provide rich, detailed descriptions explaining the content of each video
- Explain why the footage might be useful for the current story

If date is missing, note this. If no relevant footage is found, say so clearly — do not fabricate.""",
        }

        # Get system message (default to English if not found)
        system_message = system_messages.get(language, system_messages["en"])

        # Build user message with context and question
        user_message = f"""Context from video transcripts:
{context}

Question: {query}"""

        # Return messages in provider-agnostic chat format
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
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.config.mistral.max_tokens,
                    temperature=self.config.mistral.temperature,
                )
                answer = response.choices[0].message.content.strip()  # type: ignore[union-attr]
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Mistral API: {e}")
                raise RuntimeError(f"Mistral API error: {e}") from e

        elif self.config.llm.provider == "openai":
            logger.info("Generating answer using OpenAI API...")
            try:
                response = self.openai_client.chat.completions.create(  # type: ignore[assignment]
                    model=self.config.openai.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.config.openai.max_tokens,
                    temperature=self.config.openai.temperature,
                )
                answer = response.choices[0].message.content.strip()  # type: ignore[union-attr]
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

                response = self.claude_client.messages.create(  # type: ignore[assignment]
                    model=self.config.claude.model_name,
                    max_tokens=self.config.claude.max_tokens,
                    temperature=self.config.claude.temperature,
                    system=system_message,
                    messages=claude_messages,  # type: ignore[arg-type]
                )
                answer = response.content[0].text.strip()  # type: ignore[attr-defined]
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

                response = model.generate_content(  # type: ignore[assignment]
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.config.gemini.max_tokens,
                        temperature=self.config.gemini.temperature,
                    ),
                )
                answer = response.text.strip()  # type: ignore[attr-defined]
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Gemini API: {e}")
                raise RuntimeError(f"Gemini API error: {e}") from e

        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm.provider}")

    def query(
        self,
        question: str,
        top_k: int | None = None,
        language: str = "en",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete query pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve (defaults to config value)
            language: Language code for response (e.g., "en", "ru")
            date_from: Filter results from this date (YYYY-MM-DD)
            date_to: Filter results up to this date (YYYY-MM-DD)

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

        # Step 2: Retrieve relevant documents with optional date filter
        # If reranking is enabled, retrieve more candidates
        retrieval_k = self.config.reranker.initial_k if self.config.reranker.enabled else top_k

        documents = self.retrieve_documents(
            query_vector, retrieval_k, date_from=date_from, date_to=date_to
        )

        # Step 3: Rerank if enabled
        if self.config.reranker.enabled and documents:
            documents = self.rerank_documents(question, documents, top_n=top_k)

        # Step 4: Build the messages with language specification
        messages = self.build_prompt(question, documents, language=language)

        # Step 5: Generate the answer
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
