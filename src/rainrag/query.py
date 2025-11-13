"""Query interface for RainRAG using vLLM and Qdrant."""

from typing import List, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from loguru import logger

from rainrag.config import Config


class RAGQueryEngine:
    """
    RAG Query Engine that retrieves relevant documents and generates answers.

    This class handles the complete query pipeline:
    1. Embed the query using the same model as documents
    2. Search Qdrant for relevant chunks
    3. Build a prompt with retrieved context
    4. Send to vLLM for answer generation
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
        # Select API endpoint based on configuration
        endpoint = "chat/completions" if config.vllm.use_chat_completions else "completions"
        self.vllm_url = f"http://{config.vllm.host}:{config.vllm.port}/v1/{endpoint}"
        self.use_chat_api = config.vllm.use_chat_completions

        # Create a session with retry strategy for resilient HTTP requests
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
            allowed_methods=["POST"],  # Retry POST requests (for vLLM API calls)
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def initialize(self) -> None:
        """Initialize the embedding model and Qdrant client."""
        logger.info("Initializing query engine...")

        # Load embedding model
        logger.info(f"Loading embedding model: {self.config.embedding.model_name}")
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

        # Connect to Qdrant
        logger.info(
            f"Connecting to Qdrant at {self.config.qdrant.host}:{self.config.qdrant.port}"
        )
        # Disable version check to avoid warnings when client/server versions differ slightly
        # The HTTP API is stable and compatible across minor versions
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            prefer_grpc=False,
        )

        # Test connection
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant. Available collections: {len(collections.collections)}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        logger.info("Query engine initialized successfully")

    def embed_query(self, query: str) -> List[float]:
        """
        Embed the query text using the same model as documents.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized. Call initialize() first.")

        # Add "query: " prefix for E5 model (improves retrieval performance)
        prefixed_query = f"query: {query}"

        logger.debug(f"Embedding query: {query[:100]}...")
        embedding = self.embedding_model.encode(
            prefixed_query,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
        )

        return embedding.tolist()

    def retrieve_documents(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
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
            results = self.qdrant_client.search(
                collection_name=self.config.qdrant.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

            documents = []
            for idx, hit in enumerate(results):
                doc = {
                    "rank": idx + 1,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "path": hit.payload.get("path", ""),
                    "language": hit.payload.get("language", ""),
                    "doc_id": hit.payload.get("doc_id", ""),
                }
                documents.append(doc)
                logger.debug(f"Rank {idx + 1}: Score={hit.score:.4f}, Path={doc['path']}")

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def build_prompt(self, query: str, documents: List[Dict[str, Any]], language: str = "en") -> tuple[str, str]:
        """
        Build the system and user prompts for the chat LLM with retrieved context.

        Args:
            query: The user's question
            documents: List of retrieved documents
            language: Language code (e.g., "en", "ru") for response

        Returns:
            Tuple of (system_message, user_message)
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
            context_parts.append(f"Text: {text}")
            context_parts.append("")  # Empty line between documents

        context = "\n".join(context_parts)

        # Language-specific system messages
        system_messages = {
            "ru": """Вы помощник, который помогает пользователям понимать видео-транскрипты. КРИТИЧЕСКИ ВАЖНО: Вы ДОЛЖНЫ отвечать ТОЛЬКО на русском языке. Каждое слово вашего ответа должно быть на русском языке.

Вам предоставлены релевантные фрагменты видео-транскриптов. Отвечайте на вопрос пользователя на основе предоставленного контекста. Если контекста недостаточно для полного ответа, укажите это в ответе.

ПОВТОРЯЮ: Отвечайте ТОЛЬКО на русском языке.""",
            "en": """You are an assistant that helps users understand video transcripts. CRITICAL: You MUST answer ONLY in English. Every word of your response must be in English.

You have been provided with relevant excerpts from video transcripts. Answer the user's question based on the provided context. If the context doesn't contain enough information to fully answer the question, acknowledge this in your response.

REPEATING: Answer ONLY in English.""",
        }

        # Get system message (default to English if not found)
        system_message = system_messages.get(language, system_messages["en"])

        # Build user message with context and question
        user_message = f"""Context from video transcripts:
{context}

Question: {query}"""

        return system_message, user_message

    def generate_answer(self, system_message: str, user_message: str) -> str:
        """
        Generate an answer using the vLLM server.

        Uses a requests session with automatic retry logic for transient failures.
        Retries up to 3 times with exponential backoff on HTTP 429, 500, 502, 503, 504.

        Args:
            system_message: The system instruction message
            user_message: The user's question with context

        Returns:
            The generated answer text

        Raises:
            RuntimeError: If connection fails, request times out, or server returns error
        """
        # Try chat completions API first if enabled
        if self.use_chat_api:
            try:
                return self._generate_with_chat_api(system_message, user_message)
            except requests.exceptions.HTTPError as e:
                # If chat API fails with 404 or 422, fall back to completions API
                if e.response.status_code in [404, 422]:
                    logger.warning(
                        f"Chat completions API failed ({e.response.status_code}), "
                        "falling back to completions API"
                    )
                    self.use_chat_api = False
                    # Update URL for future requests
                    self.vllm_url = f"http://{self.config.vllm.host}:{self.config.vllm.port}/v1/completions"
                else:
                    raise

        # Use completions API (either configured or as fallback)
        return self._generate_with_completions_api(system_message, user_message)

    def _generate_with_chat_api(self, system_message: str, user_message: str) -> str:
        """Generate answer using chat completions API."""
        logger.info("Generating answer using vLLM chat completions API...")

        payload = {
            "model": self.config.vllm.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": self.config.vllm.max_tokens,
            "temperature": self.config.vllm.temperature,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.vllm_url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()

            logger.info("Answer generated successfully (chat API)")
            return answer

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to vLLM server at {self.vllm_url}: {e}")
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.vllm_url}. "
                "Make sure the vLLM server is running."
            ) from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Request to vLLM server timed out after 30 seconds: {e}")
            raise RuntimeError(
                "Request to vLLM server timed out after 30 seconds. "
                "The model may be overloaded or the prompt may be too complex."
            ) from e
        except requests.exceptions.HTTPError as e:
            error_details = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"vLLM chat API error {e.response.status_code}: {error_details}")
            raise  # Re-raise to allow fallback logic in generate_answer
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from vLLM server: {e}")
            raise RuntimeError(
                "vLLM server returned an unexpected response format. "
                "The response may be missing expected fields."
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise RuntimeError(f"Unexpected error during answer generation: {e}") from e

    def _generate_with_completions_api(self, system_message: str, user_message: str) -> str:
        """Generate answer using completions API (fallback for older vLLM versions)."""
        logger.info("Generating answer using vLLM completions API...")

        # Combine system and user messages into a single prompt
        combined_prompt = f"{system_message}\n\n{user_message}\n\nAnswer:"

        payload = {
            "model": self.config.vllm.model_name,
            "prompt": combined_prompt,
            "max_tokens": self.config.vllm.max_tokens,
            "temperature": self.config.vllm.temperature,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.vllm_url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["text"].strip()

            logger.info("Answer generated successfully (completions API)")
            return answer

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to vLLM server at {self.vllm_url}: {e}")
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.vllm_url}. "
                "Make sure the vLLM server is running."
            ) from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Request to vLLM server timed out after 30 seconds: {e}")
            raise RuntimeError(
                "Request to vLLM server timed out after 30 seconds. "
                "The model may be overloaded or the prompt may be too complex."
            ) from e
        except requests.exceptions.HTTPError as e:
            error_details = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"vLLM completions API error {e.response.status_code}: {error_details}")
            raise RuntimeError(
                f"vLLM server returned HTTP error: {e.response.status_code} - {error_details}"
            ) from e
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from vLLM server: {e}")
            raise RuntimeError(
                "vLLM server returned an unexpected response format. "
                "The response may be missing expected fields."
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise RuntimeError(f"Unexpected error during answer generation: {e}") from e

    def query(self, question: str, top_k: int | None = None, language: str = "en") -> Dict[str, Any]:
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
            top_k = self.config.vllm.top_k

        logger.info(f"Processing query: {question[:100]}... (language: {language})")

        # Step 1: Embed the query
        query_vector = self.embed_query(question)

        # Step 2: Retrieve relevant documents
        documents = self.retrieve_documents(query_vector, top_k)

        # Step 3: Build the prompt with language specification
        system_message, user_message = self.build_prompt(question, documents, language=language)

        # Step 4: Generate the answer
        answer = self.generate_answer(system_message, user_message)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": documents,
            "num_documents": len(documents),
        }


def run_query(config_path: str, question: str, top_k: int | None = None) -> Dict[str, Any]:
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
