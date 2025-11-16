#!/usr/bin/env python3
"""
Download and cache required models for RainRAG.

This script downloads the embedding model and caches it locally.
Run this once when you have internet access.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sentence_transformers import SentenceTransformer

from rainrag.config import load_config


def download_embedding_model(model_name: str, device: str = "cpu") -> bool:
    """
    Download and cache the embedding model.

    Args:
        model_name: HuggingFace model name
        device: Device to use (cpu/cuda)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading embedding model: {model_name}")
        logger.info("This may take several minutes...")

        model = SentenceTransformer(
            model_name,
            device=device,
        )

        logger.success(f"✓ Successfully downloaded and cached: {model_name}")
        logger.info(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
        logger.info(f"  Cached location: ~/.cache/huggingface/hub/")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False


def main():
    """Download all required models."""
    logger.info("=== RainRAG Model Download Script ===")
    logger.info("")

    # Load config to get model name
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = load_config(str(config_path))
        model_name = config.embedding.model_name
        device = config.embedding.device

        logger.info(f"Config file: {config_path}")
        logger.info(f"Embedding model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Using default model: intfloat/multilingual-e5-large")
        model_name = "intfloat/multilingual-e5-large"
        device = "cpu"

    # Download embedding model
    success = download_embedding_model(model_name, device)

    if success:
        logger.info("")
        logger.success("=== All models downloaded successfully! ===")
        logger.info("You can now run RainRAG offline.")
        return 0
    else:
        logger.error("")
        logger.error("=== Model download failed! ===")
        logger.info("Please check your internet connection and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
