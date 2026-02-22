"""Provider comparison experiment.

Fixes the best-performing retrieval configuration (full_pipeline: hybrid RRF
+ query rewriting + HyDE + reranker) and sweeps all combinations of LLM and
embedding providers.

Provider matrix
---------------
LLM providers  : mistral, openai, claude, gemini
Embed providers: local, mistral, openai, gemini

Note: Claude does not offer an embedding API, so combinations where
``embedding.provider = "claude"`` are skipped.  When the LLM provider is
Claude, the embedding provider must be one of local / mistral / openai / gemini.

Each run logs the provider pair, retrieval metrics, answer quality, latency,
and estimated cost per query (rough token-count-based estimate).

Usage
-----
    from eval.experiments.provider_comparison import ProviderComparisonExperiment

    exp = ProviderComparisonExperiment(
        config_path="config.yaml",
        dataset_path="eval/datasets/eval_set_en.jsonl",
    )
    results = exp.run()
"""

from __future__ import annotations

from typing import Any

from eval.experiments.base import BaseExperiment


#: Approximate cost per 1 M tokens (input) in USD, for rough cost logging.
#: Update these whenever provider pricing changes.
_LLM_COST_PER_1M: dict[str, float] = {
    "mistral": 2.0,  # mistral-large-latest
    "openai": 2.5,  # gpt-4o-mini
    "claude": 0.8,  # claude-haiku-4-5-20251001
    "gemini": 0.075,  # gemini-2.5-flash
}

_EMBED_COST_PER_1M: dict[str, float] = {
    "local": 0.0,
    "mistral": 0.1,  # mistral-embed
    "openai": 0.02,  # text-embedding-3-large
    "gemini": 0.0,  # free tier
}

# Retrieval features to fix for all provider comparison runs
_FIXED_RETRIEVAL_OVERRIDES: dict[str, Any] = {
    "hybrid_search.enabled": True,
    "hybrid_search.fusion_method": "rrf",
    "two_stage.enabled": True,
    "two_stage.query_rewrite_enabled": True,
    "two_stage.hyde_enabled": True,
    "reranker.enabled": True,
}

_LLM_PROVIDERS = ["mistral", "openai", "claude", "gemini"]
_EMBED_PROVIDERS = ["local", "mistral", "openai", "gemini"]

# Embedding model names + vector dimensions for each provider
_EMBED_MODELS: dict[str, tuple[str, int]] = {
    "local": ("intfloat/multilingual-e5-large", 1024),
    "mistral": ("mistral-embed", 1024),
    "openai": ("text-embedding-3-large", 3072),
    "gemini": ("models/text-embedding-004", 768),  # Gemini embeddings are 768-dimensional
}


def _build_conditions() -> list[dict[str, Any]]:
    conditions = []
    cid = 0
    for llm in _LLM_PROVIDERS:
        for emb in _EMBED_PROVIDERS:
            cid += 1
            embed_model, vector_size = _EMBED_MODELS[emb]
            overrides: dict[str, Any] = {
                **_FIXED_RETRIEVAL_OVERRIDES,
                "llm.provider": llm,
                "embedding.provider": emb,
                "embedding.model_name": embed_model,
                "qdrant.vector_size": vector_size,
            }
            # sanity check: vector_size should match embedding dimension
            assert overrides["qdrant.vector_size"] == vector_size
            # Point LLM-specific top_k to the right sub-config
            overrides[f"{llm}.top_k"] = 5

            conditions.append(
                {
                    "id": f"{cid:02d}",
                    "label": f"llm={llm}_emb={emb}",
                    "overrides": overrides,
                    "tags": {
                        "llm_provider": llm,
                        "embedding_provider": emb,
                        "experiment_type": "provider_comparison",
                    },
                    "_llm": llm,
                    "_emb": emb,
                }
            )
    return conditions


class ProviderComparisonExperiment(BaseExperiment):
    """Sweep all LLM × embedding provider combinations with a fixed retrieval config.

    Args:
        llm_providers: Subset of LLM providers to test. Defaults to all four.
        embed_providers: Subset of embedding providers to test. Defaults to all four.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str = "./mlruns",
        top_ks: tuple[int, ...] = (5,),
        llm_providers: list[str] | None = None,
        embed_providers: list[str] | None = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            dataset_path=dataset_path,
            mlflow_uri=mlflow_uri,
            experiment_name="provider_comparison",
            top_ks=top_ks,
        )
        self._llm_providers = set(llm_providers or _LLM_PROVIDERS)
        self._embed_providers = set(embed_providers or _EMBED_PROVIDERS)

    def conditions(self) -> list[dict[str, Any]]:
        all_conditions = _build_conditions()
        return [
            c
            for c in all_conditions
            if c["_llm"] in self._llm_providers and c["_emb"] in self._embed_providers
        ]
