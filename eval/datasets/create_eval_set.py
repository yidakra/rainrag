"""Generate a synthetic evaluation dataset from the live Qdrant collection.

Usage
-----
    python -m eval.datasets.create_eval_set \\
        --config config.yaml \\
        --lang en \\
        --n 50 \\
        --output eval/datasets/eval_set_en.jsonl

For each sampled chunk the configured LLM generates:
  * A natural query that can be answered from the chunk text.
  * A reference answer grounded in the chunk.
  * A category label (factual | temporal | entity | multilingual).
  * Whether the query uses recency/temporal language.

The chunk's own ``doc_id`` is written as the single ``relevant_doc_ids``
entry.  You can extend the JSONL manually to add additional relevant docs
or merge duplicates before running experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path


# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rainrag.config import load_config
from rainrag.query import RAGQueryEngine


_GENERATION_PROMPT = textwrap.dedent("""\
    You are creating an evaluation dataset for an information-retrieval system.

    Below is a transcript chunk from a broadcast archive.  Your task is to
    invent one realistic question that a journalist or researcher might ask
    whose answer is contained *entirely* within this chunk.

    Rules:
    - The question must be answerable from the chunk alone.
    - The reference answer must be a concise factual sentence (≤ 3 sentences).
    - category must be exactly one of: factual, temporal, entity, multilingual
      * factual   – a general fact question
      * temporal  – asks about a date, time, or recency ("latest", "recent", …)
      * entity    – asks about a specific person, organisation, or place
      * multilingual – the question mixes languages or asks about cross-language content
    - temporal_query: true only when the question explicitly uses temporal language.

    Respond with ONLY a JSON object (no markdown fences), for example:
    {{"question": "…", "reference_answer": "…", "category": "factual", "temporal_query": false}}

    Chunk:
    {chunk_text}
""")

_FILTER_PROMPT = textwrap.dedent("""\
    You are a quality-control reviewer for an evaluation dataset.

    Decide whether the following (question, answer, chunk) triple is
    high-quality for evaluating a retrieval-augmented generation system.

    High quality means:
    - The question is clear and unambiguous.
    - The answer is fully supported by the chunk.
    - The question cannot be answered without the chunk.

    Reply with ONLY "yes" or "no".

    Question: {question}
    Reference answer: {reference_answer}
    Chunk: {chunk_text}
""")


def _scroll_chunks(engine: RAGQueryEngine, lang: str, limit: int) -> list[dict]:
    """Scroll through the Qdrant collection and return chunks matching *lang*."""
    client = engine.qdrant_client
    collection = engine.config.qdrant.collection_name
    all_chunks: list[dict] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload or {}
            if (
                payload.get("language", "en") == lang
                and payload.get("text", "")
                and not payload.get("is_speech_free", False)  # skip metadata-only docs
            ):
                all_chunks.append({"doc_id": payload.get("doc_id", str(point.id)), **payload})

        if next_offset is None or len(results) == 0:
            break
        offset = next_offset

    return all_chunks


def _stratified_sample(chunks: list[dict], n: int) -> list[dict]:
    """Roughly stratify by video_id then sample n chunks uniformly."""
    if len(chunks) <= n:
        return chunks
    # Group by video_id to avoid sampling too many chunks from one video
    by_video: dict[str, list[dict]] = {}
    for chunk in chunks:
        vid = chunk.get("video_id") or chunk.get("path", "unknown")
        by_video.setdefault(vid, []).append(chunk)

    sampled: list[dict] = []
    videos = list(by_video.keys())
    random.shuffle(videos)
    per_video = max(1, n // len(videos))

    for vid in videos:
        sampled.extend(random.sample(by_video[vid], min(per_video, len(by_video[vid]))))
        if len(sampled) >= n:
            break

    # Top-up if we're short
    remaining = [c for c in chunks if c not in sampled]
    if len(sampled) < n and remaining:
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled[:n]


def _call_llm(engine: RAGQueryEngine, prompt: str) -> str:
    """Call the configured LLM with a single user message and return the response text."""
    messages = [{"role": "user", "content": prompt}]
    return engine.generate_answer(messages, temperature=0.3)


def _generate_pair(engine: RAGQueryEngine, chunk: dict, lang: str) -> dict | None:
    """Generate a (question, reference_answer, category) triple for one chunk."""
    chunk_text = chunk.get("text", "")
    if len(chunk_text.strip()) < 50:
        return None

    prompt = _GENERATION_PROMPT.format(chunk_text=chunk_text[:1500])
    try:
        raw = _call_llm(engine, prompt)
        # Strip potential markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
    except Exception as exc:
        print(f"  [warn] Generation failed for doc_id={chunk.get('doc_id')}: {exc}")
        return None

    question = data.get("question", "").strip()
    reference_answer = data.get("reference_answer", "").strip()
    category = data.get("category", "factual")
    temporal = bool(data.get("temporal_query", False))

    if not question or not reference_answer:
        return None

    return {
        "query_id": None,  # filled later
        "query": question,
        "language": lang,
        "relevant_doc_ids": [chunk["doc_id"]],
        "reference_answer": reference_answer,
        "category": category
        if category in {"factual", "temporal", "entity", "multilingual"}
        else "factual",
        "temporal": temporal,
        "source_doc_id": chunk["doc_id"],
        "source_path": chunk.get("path", ""),
        "source_date": chunk.get("web_date") or chunk.get("date", ""),
    }


def _filter_pair(engine: RAGQueryEngine, pair: dict, chunk: dict) -> bool:
    """Ask the LLM to quality-gate the generated pair. Returns True if accepted."""
    prompt = _FILTER_PROMPT.format(
        question=pair["query"],
        reference_answer=pair["reference_answer"],
        chunk_text=chunk.get("text", "")[:1000],
    )
    try:
        verdict = _call_llm(engine, prompt).strip().lower()
        return verdict.startswith("yes")
    except Exception:
        return True  # Accept on error to avoid losing too many pairs


def create_eval_set(
    config_path: str,
    lang: str,
    n: int,
    output: str,
    seed: int = 42,
    skip_filter: bool = False,
) -> None:
    """Main entry point for dataset creation.

    Args:
        config_path: Path to config.yaml.
        lang: Language code ("en" or "ru").
        n: Target number of eval pairs.
        output: Output JSONL file path.
        seed: Random seed for reproducibility.
        skip_filter: Skip the LLM quality-filter pass (faster but noisier).
    """
    random.seed(seed)
    print(f"Loading config from {config_path} ...")
    config = load_config(config_path)
    engine = RAGQueryEngine(config)
    engine.initialize()

    print(f"Scrolling Qdrant collection '{config.qdrant.collection_name}' for lang='{lang}' ...")
    all_chunks = _scroll_chunks(engine, lang, limit=10_000)
    print(f"  Found {len(all_chunks)} chunks.")

    if not all_chunks:
        print("ERROR: no chunks found. Check Qdrant connection and language filter.")
        sys.exit(1)

    # Sample 2× target to allow for filter failures
    candidates = _stratified_sample(all_chunks, n * 2)
    print(f"  Sampled {len(candidates)} candidate chunks (2× target for filter headroom).")

    pairs: list[dict] = []
    for i, chunk in enumerate(candidates):
        if len(pairs) >= n:
            break
        print(
            f"  [{i+1}/{len(candidates)}] Generating pair for doc_id={chunk.get('doc_id', '?')} ...",
            end=" ",
        )
        pair = _generate_pair(engine, chunk, lang)
        if pair is None:
            print("skipped (generation failed)")
            continue
        if not skip_filter and not _filter_pair(engine, pair, chunk):
            print("skipped (quality filter)")
            continue
        pair["query_id"] = f"{lang}_{len(pairs)+1:03d}"
        pairs.append(pair)
        print(f"ok [{pair['category']}]")

    print(f"\nGenerated {len(pairs)} pairs (target: {n}).")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written to {output_path}")

    # Category breakdown
    from collections import Counter

    cats = Counter(p["category"] for p in pairs)
    print("Category distribution:", dict(cats))


def load_eval_set(path: str) -> list[dict]:
    """Load an eval JSONL file into a list of record dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic eval dataset from Qdrant.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--lang", choices=["en", "ru"], default="en")
    parser.add_argument("--n", type=int, default=50, help="Number of eval pairs to generate")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-filter", action="store_true", help="Skip LLM quality filter")
    args = parser.parse_args()

    create_eval_set(
        config_path=args.config,
        lang=args.lang,
        n=args.n,
        output=args.output,
        seed=args.seed,
        skip_filter=args.skip_filter,
    )
