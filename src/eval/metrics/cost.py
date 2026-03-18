"""API cost estimation for the provider comparison experiment.

Estimates token counts from character lengths (English: ~4 chars/token)
and applies published per-1M-token prices.  All figures are labelled
*estimated* — use your provider dashboards for billing accuracy.

NOTE: This estimator only accounts for the *main answer generation* LLM call
(and the query embedding call). It does **not** include additional LLM calls
(e.g., query rewrite, HyDE) or third-party reranker calls. For those, the
evaluation suite logs usage metrics (`cost.llm_calls_count`, etc.) to enable
post-hoc cost modeling.

Pricing tables (update when providers change rates).
Costs are modelled separately for input tokens and output tokens using
``LLM_INPUT_COST_PER_1M`` and ``LLM_OUTPUT_COST_PER_1M`` dictionaries.

LLM input cost (USD / 1 M tokens)
  mistral  $2.00   mistral-large-latest
  openai   $0.15   gpt-4o-mini (input price)
  claude   $0.80   claude-haiku-4-5-20251001 (input)
  gemini   $0.075  gemini-2.5-flash (input)

LLM output cost (USD / 1 M tokens)
  mistral  $6.00   mistral-large-latest (output)
  openai   $0.60   gpt-4o-mini (output)
  claude   $4.00   claude-haiku-4-5-20251001 (output)
  gemini   $0.30  gemini-2.5-flash (output)

Embedding (USD / 1 M tokens)
  local    $0.00   SentenceTransformer (electricity not counted)
  mistral  $0.10   mistral-embed
  openai   $0.02   text-embedding-3-large
  gemini   $0.00   free tier
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Pricing tables (USD / 1 M tokens)
# ---------------------------------------------------------------------------

LLM_INPUT_COST_PER_1M: dict[str, float] = {
    "mistral": 2.00,
    "openai": 0.15,  # gpt-4o-mini input price
    "claude": 0.80,  # claude-haiku-4-5-20251001 input
    "gemini": 0.075,  # gemini-2.5-flash input
}

LLM_OUTPUT_COST_PER_1M: dict[str, float] = {
    "mistral": 6.00,  # mistral-large-latest output
    "openai": 0.60,  # gpt-4o-mini output
    "claude": 4.00,  # claude-haiku-4-5-20251001 output
    "gemini": 0.30,  # gemini-2.5-flash output
}

EMBED_COST_PER_1M: dict[str, float] = {
    "local": 0.0,
    "mistral": 0.10,
    "openai": 0.02,
    "gemini": 0.0,
}

# Character-to-token ratio (rough estimate for English / mixed text)
_CHARS_PER_TOKEN: float = 4.0


# ---------------------------------------------------------------------------
# Core estimation functions
# ---------------------------------------------------------------------------


def chars_to_tokens(text: str | None) -> float:
    """Estimate token count from character length.

    Returns 0.0 for ``None`` to support optional text fields in pipelines.
    """
    if text is None:
        return 0.0
    return len(text) / _CHARS_PER_TOKEN


def estimate_query_cost(
    query: str,
    contexts: list[str | None],
    answer: str,
    llm_provider: str,
    embed_provider: str,
    llm_query_rewrite_calls: int = 0,
    llm_hyde_calls: int = 0,
    reranker_calls: int = 0,
) -> dict[str, float]:
    """Estimate API cost for a single query-answer cycle.

    Args:
        query: The user query text.
        contexts: Retrieved document texts passed to the LLM as context (may include None).
        answer: The LLM-generated answer.
        llm_provider: Name of the LLM provider ("mistral", "openai", "claude", "gemini").
        embed_provider: Name of the embedding provider.
        llm_query_rewrite_calls: Number of query rewrite LLM calls.
        llm_hyde_calls: Number of HyDE LLM calls.
        reranker_calls: Number of reranker API calls.

    Returns:
        Dict with keys:
          ``cost.input_tokens_est``         – estimated prompt token count
          ``cost.output_tokens_est``        – estimated completion token count
          ``cost.embed_tokens_est``         – estimated embedding token count (query only)
          ``cost.llm_base_usd_est``         – estimated USD cost for the main answer generation call
          ``cost.llm_rewrite_usd_est``      – estimated USD cost for query rewrite calls
          ``cost.llm_hyde_usd_est``         – estimated USD cost for HyDE calls
          ``cost.reranker_usd_est``         – estimated USD cost for reranker calls (zero by default)
          ``cost.llm_usd_est``              – legacy field (same as ``cost.llm_base_usd_est``)
          ``cost.embed_usd_est``            – estimated embedding cost in USD
          ``cost.total_usd_est``            – sum of all cost components
    """
    # Input = system prompt + context + query (system prompt approximated as 200 tokens)
    # Normalize contexts so we never pass None / non-str values into chars_to_tokens.
    # (e.g., some retrieval pipelines may return None for missing documents.)
    contexts = contexts or []
    cleaned_contexts = [str(c) for c in contexts if c is not None]
    context_text = " ".join(cleaned_contexts)
    input_tokens = 200 + chars_to_tokens(context_text) + chars_to_tokens(query)
    output_tokens = chars_to_tokens(answer)
    embed_tokens = chars_to_tokens(query)  # one embedding call per query

    # Unknown providers degrade gracefully to zero cost.
    llm_input_rate = LLM_INPUT_COST_PER_1M.get(llm_provider, 0.0)
    llm_output_rate = LLM_OUTPUT_COST_PER_1M.get(llm_provider, 0.0)
    embed_rate = EMBED_COST_PER_1M.get(embed_provider, 0.0)

    input_cost = input_tokens / 1_000_000 * llm_input_rate
    output_cost = output_tokens / 1_000_000 * llm_output_rate
    embed_cost = embed_tokens / 1_000_000 * embed_rate

    # Base LLM cost is the cost for the final answer generation call.
    llm_base_cost = input_cost + output_cost

    # Additional LLM calls (rewrite, HyDE) are estimated using the same per-call cost.
    # This is a simple proxy; the real cost depends on prompt lengths and output sizes.
    llm_rewrite_cost = llm_base_cost * llm_query_rewrite_calls
    llm_hyde_cost = llm_base_cost * llm_hyde_calls

    # Reranker cost is not known (provider-dependent). Default to zero and allow later extension.
    reranker_cost = 0.0

    total_cost = embed_cost + llm_base_cost + llm_rewrite_cost + llm_hyde_cost + reranker_cost

    return {
        "cost.input_tokens_est": round(input_tokens),
        "cost.output_tokens_est": round(output_tokens),
        "cost.embed_tokens_est": round(embed_tokens),
        "cost.llm_base_usd_est": llm_base_cost,
        "cost.llm_rewrite_usd_est": llm_rewrite_cost,
        "cost.llm_hyde_usd_est": llm_hyde_cost,
        "cost.reranker_usd_est": reranker_cost,
        # Legacy key for backwards compatibility
        "cost.llm_usd_est": llm_base_cost,
        "cost.embed_usd_est": embed_cost,
        "cost.total_usd_est": total_cost,
    }


def aggregate_costs(per_query_costs: list[dict[str, float]]) -> dict[str, float]:
    """Sum token counts and costs across all queries, add per-query averages.

    Returns keys like ``cost.total_usd_est`` (total) and
    ``cost.total_usd_est_per_query`` (mean).
    """
    if not per_query_costs:
        return {}

    keys = per_query_costs[0].keys()
    totals = {k: sum(d.get(k, 0.0) for d in per_query_costs) for k in keys}
    n = len(per_query_costs)
    averages = {f"{k}_per_query": v / n for k, v in totals.items()}

    # Clarify that the per-query cost is a mean across the evaluated queries.
    # Provide an explicit (and more descriptive) key for the main total cost metric.
    if "cost.total_usd_est_per_query" in averages:
        averages["cost.total_mean_usd_est_per_query"] = averages["cost.total_usd_est_per_query"]

    return {**totals, **averages}
