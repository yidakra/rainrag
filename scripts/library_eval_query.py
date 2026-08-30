#!/usr/bin/env python3
"""Score the similarity ranking against the answers Varya wrote by hand.

Her first query has a stated correct answer — six content_ids, each with a
reason — which makes it a real test rather than a demo:

    Вот пример видео [RohuZGgpC_k], найди похожие по темам и спикерам
    видео длиной от 30 минут в жанре лекции или интервью
    → 454556, 454501, 77645, 449964, 484740, 431298

What this reports, and why:

    recall@k    of the six she named, how many are in the top k. This is the
                number that matters: an editor scanning a shortlist will not
                scroll past the first screen.
    MRR         how high the first correct answer lands.
    found       how many of the six are even in the tagged pool -- a miss
                because an episode was never tagged is a different failure
                from a miss because it ranked badly, and conflating them
                would flatter the ranker.

    scripts/library_eval_query.py
    scripts/library_eval_query.py --k 20 --show 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_TAGS = REPO_ROOT / "data" / "library_tags.jsonl"
DEFAULT_GOLD = REPO_ROOT / "data" / "library_gold.json"


def load_episodes(path: Path) -> list:
    from rainrag.library_similar import Episode, dedupe_latest

    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        # A failed tagging row has no subjects; keeping it would inflate the
        # candidate pool with episodes that can never match.
        if record.get("error"):
            continue
        episodes.append(Episode.from_record(record))
    # The tag file is appended to, so a re-tagged episode has more than one
    # row. Counting those separately would overstate the pool and rank the
    # same episode twice.
    return dedupe_latest(episodes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tags", default=str(DEFAULT_TAGS))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--k", type=int, default=10, help="cut-off for recall@k")
    parser.add_argument("--show", type=int, default=10, help="how many results to print")
    args = parser.parse_args(argv)

    from rainrag.library_similar import find_similar

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    query = gold["queries"][0]
    episodes = load_episodes(Path(args.tags))
    by_content = {e.content_id: e for e in episodes if e.content_id}

    seed = by_content.get(query["seed_content_id"])
    if seed is None:
        print(
            f"Seed {query['seed_content_id']} is not in the tagged pool "
            f"({len(episodes)} episodes tagged) — tag it before evaluating."
        )
        return 1

    expected = query["expected_content_ids"]
    present = [cid for cid in expected if cid in by_content]
    missing = [cid for cid in expected if cid not in by_content]

    filters = query.get("filters", {})
    results = find_similar(
        seed,
        episodes,
        min_duration_minutes=filters.get("min_duration_minutes"),
        genres=filters.get("genre"),
        # The full filtered pool: truncating here would report MRR 0 whenever
        # the first correct answer falls below the display cut-off.
        limit=len(episodes),
    )
    ranked = [r.episode.content_id for r in results]

    hits = [cid for cid in expected if cid in ranked[: args.k]]
    first_rank = next((i + 1 for i, cid in enumerate(ranked) if cid in expected), None)

    print(f"Query: {query['query']}\n")
    print(f"Pool: {len(episodes)} tagged episodes")
    print(f"Seed: {seed.content_id} — {(seed.title or '')[:60]}")
    print(f"      speakers={seed.speakers} genre={seed.genre} subjects={len(seed.subject)}")
    print()
    print(f"Expected {len(expected)}; {len(present)} are in the tagged pool", end="")
    if missing:
        print(f", {len(missing)} not tagged yet: {', '.join(missing)}")
    else:
        print()
    print()
    print(
        f"  recall@{args.k}: {len(hits)}/{len(expected)} overall"
        f"   {len(hits)}/{len(present) or 1} of those available"
    )
    print(f"  MRR: {1 / first_rank:.2f}" if first_rank else "  MRR: 0 (no expected result ranked)")
    print()

    for i, r in enumerate(results[: args.show], 1):
        mark = "✓" if r.episode.content_id in expected else " "
        title = (r.episode.title or "")[:52]
        print(f"  {mark} {i:>2}. [{r.score:5.2f}] {r.episode.content_id or '—':>7}  {title}")
        print(f"           {r.explain()[:104]}")

    if missing:
        print(f"\n  Not yet tagged, so unrankable: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
