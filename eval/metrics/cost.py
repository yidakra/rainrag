"""API cost estimation for the provider comparison experiment.

Estimates token counts from character lengths (English: ~4 chars/token)
and applies published per-1M-token prices.  All figures are labelled
*estimated* — use your provider dashboards for billing accuracy.

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


def chars_to_tokens(text: str) -> float:
    """Estimate token count from character length."""
    return len(text) / _CHARS_PER_TOKEN


def estimate_query_cost(
    query: str,
    contexts: list[str],
    answer: str,
    llm_provider: str,
    embed_provider: str,
) -> dict[str, float]:
    """Estimate API cost for a single query-answer cycle.

    Args:
        query: The user query text.
        contexts: Retrieved document texts passed to the LLM as context.
        answer: The LLM-generated answer.
        llm_provider: Name of the LLM provider ("mistral", "openai", "claude", "gemini").
        embed_provider: Name of the embedding provider.

    Returns:
        Dict with keys:
          ``cost.input_tokens_est``   – estimated prompt token count
          ``cost.output_tokens_est``  – estimated completion token count
          ``cost.embed_tokens_est``   – estimated embedding token count (query only)
          ``cost.llm_usd_est``        – estimated LLM cost in USD
          ``cost.embed_usd_est``      – estimated embedding cost in USD
          ``cost.total_usd_est``      – sum of llm + embed cost
    """
    # Input = system prompt + context + query (system prompt approximated as 200 tokens)
    context_text = " ".join(contexts)
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

    llm_cost = input_cost + output_cost
    total_cost = llm_cost + embed_cost

    return {
        "cost.input_tokens_est": round(input_tokens),
        "cost.output_tokens_est": round(output_tokens),
        "cost.embed_tokens_est": round(embed_tokens),
        "cost.llm_usd_est": llm_cost,
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

    return {**totals, **averages}
