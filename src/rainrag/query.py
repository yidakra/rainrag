"""Query interface for RainRAG using vLLM and Qdrant."""

from typing import List, Dict, Any
import requests
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
        self.vllm_url = f"http://{config.vllm.host}:{config.vllm.port}/v1/completions"

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

    def build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for the LLM with retrieved context.

        Args:
            query: The user's question
            documents: List of retrieved documents

        Returns:
            The complete prompt string
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

        # Build the complete prompt
        prompt = f"""You are an assistant that helps users understand video transcripts. You have been provided with relevant excerpts from video transcripts. Answer the user's question based on the provided context.

Context from video transcripts:
{context}

User Question: {query}

Provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge this in your response. Answer in the same language as the question."""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the vLLM server.

        Args:
            prompt: The complete prompt with context

        Returns:
            The generated answer text
        """
        logger.info("Generating answer using vLLM...")

        payload = {
            "model": self.config.vllm.model_name,
            "prompt": prompt,
            "max_tokens": self.config.vllm.max_tokens,
            "temperature": self.config.vllm.temperature,
            "stream": False,
        }

        try:
            response = requests.post(
                self.vllm_url,
                json=payload,
                timeout=120,  # 2 minute timeout
            )
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["text"].strip()

            logger.info("Answer generated successfully")
            return answer

        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to vLLM server at {self.vllm_url}")
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.vllm_url}. "
                "Make sure the vLLM server is running."
            )
        except requests.exceptions.Timeout:
            logger.error("Request to vLLM server timed out")
            raise RuntimeError("Request to vLLM server timed out after 120 seconds")
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise

    def query(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        """
        Execute the complete query pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve (defaults to config value)

        Returns:
            Dictionary containing the answer and metadata
        """
        if top_k is None:
            top_k = self.config.vllm.top_k

        logger.info(f"Processing query: {question[:100]}...")

        # Step 1: Embed the query
        query_vector = self.embed_query(question)

        # Step 2: Retrieve relevant documents
        documents = self.retrieve_documents(query_vector, top_k)

        # Step 3: Build the prompt
        prompt = self.build_prompt(question, documents)

        # Step 4: Generate the answer
        answer = self.generate_answer(prompt)

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
