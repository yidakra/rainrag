"""Evaluate and fit selectors from judged candidate snapshots, without provider calls.

Run with ``python -m rainrag_eval.selection --help``. Input is JSONL containing
query_id, group_id, relevant_doc_ids and the opt-in retrieval_selection trace.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from rainrag.selection import (
    RetrievalAlternative,
    feature_matrix,
    fit_ridge,
    select_alternative,
)


app = typer.Typer(help=__doc__, add_completion=False)


@dataclass(frozen=True)
class JudgedCase:
    """One information need with explicit leakage group and pooled positive labels."""

    id: str
    group: str
    relevant: frozenset[str]
    alternatives: list[RetrievalAlternative]
    signature: str


def load_cases(path: Path) -> list[JudgedCase]:
    """Validate trace schema and labels before fitting; do not guess event groups."""
    cases = []
    revisions: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                trace = record["retrieval_selection"]
                if (
                    trace["stage"] != "first_stage_before_merge"
                    or not trace["corpus_revision"].strip()
                ):
                    raise ValueError(
                        "An explicit corpus revision and first-stage trace are required"
                    )
                revisions.add(trace["corpus_revision"])
                if (
                    not isinstance(trace["configuration_signature"], str)
                    or not trace["configuration_signature"]
                ):
                    raise ValueError("A configuration signature is required")
                for key in ("query_id", "group_id"):
                    if not isinstance(record[key], str) or not record[key].strip():
                        raise ValueError(f"{key} must be a non-empty string")
                labels = record["relevant_doc_ids"]
                if (
                    not isinstance(labels, list)
                    or not labels
                    or not all(isinstance(x, str) and x for x in labels)
                ):
                    raise ValueError("Pooled relevant_doc_ids are required")
                alternatives = [
                    RetrievalAlternative(a["id"], a["query"], a["documents"])
                    for a in trace["alternatives"]
                ]
                if any(
                    not isinstance(a.id, str) or not a.id or not isinstance(a.query, str)
                    for a in alternatives
                ):
                    raise ValueError("Option IDs and query texts must be strings")
                if not any(a.id == "original" for a in alternatives):
                    raise ValueError("Every query must retain its original alternative")
                feature_matrix(alternatives)
                cases.append(
                    JudgedCase(
                        record["query_id"],
                        record["group_id"],
                        frozenset(labels),
                        alternatives,
                        trace["configuration_signature"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid judged snapshot at line {line_number}: {exc}") from exc
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Dataset must contain unique query IDs")
    if len({c.signature for c in cases}) != 1 or len(revisions) != 1:
        raise ValueError("Do not mix retrieval configurations or corpus revisions")
    label_groups: dict[tuple[str, str], str] = {}
    for case in cases:
        keys = [("doc", doc_id) for doc_id in case.relevant]
        keys.extend(
            ("video", str(d["video_id"]))
            for alternative in case.alternatives
            for d in alternative.documents
            if d["doc_id"] in case.relevant and d.get("video_id")
        )
        for label_key in keys:
            if label_key in label_groups and label_groups[label_key] != case.group:
                raise ValueError("Shared relevant documents/videos must remain in one group")
            label_groups[label_key] = case.group
    return cases


def utility(alternative: RetrievalAlternative, relevant: frozenset[str], k: int = 10) -> float:
    """Binary nDCG on saved retrieval order, never prompt order."""
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    actual = sum(
        1 / math.log2(i + 2)
        for i, doc in enumerate(alternative.documents[:k])
        if doc["doc_id"] in relevant
    )
    return actual / ideal if ideal else 0.0


class CorpusStatistics:
    """Unsupervised lexical statistics from a fixed corpus snapshot, not retrieved hits."""

    def __init__(self, texts: Iterable[str]):
        self.count = 0
        self.frequency: Counter[str] = Counter()
        for text in texts:
            self.count += 1
            self.frequency.update(set(re.findall(r"\w+", text.lower())))
        if not self.count:
            raise ValueError("Corpus snapshot is empty")

    def matrix(self, alternatives: list[RetrievalAlternative]) -> np.ndarray:
        """A bounded corpus-statistic control, not a reproduction of all eleven QPP predictors."""
        rows = []
        for alternative in alternatives:
            tokens = re.findall(r"\w+", alternative.query.lower())
            idf = [math.log((self.count + 1) / (self.frequency[t] + 1)) for t in tokens]
            rows.append(
                [
                    len(tokens),
                    len(alternative.query),
                    sum(idf),
                    float(np.mean(idf)) if idf else 0.0,
                    float(np.std(idf)) if idf else 0.0,
                    min(idf, default=0.0),
                    max(idf, default=0.0),
                ]
            )
        return np.asarray(rows, dtype=float)


def grouped_folds(
    cases: list[JudgedCase], folds: int, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Keep explicit event/video/duplicate groups whole on both sides of each fold."""
    groups = sorted({c.group for c in cases})
    if folds < 2 or len(groups) < folds:
        raise ValueError("At least one independent group per fold, with folds >= 2, is required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    splits = []
    for held_out in np.array_split(groups, folds):
        test_groups = set(held_out)
        train = [i for i, case in enumerate(cases) if case.group not in test_groups]
        test = [i for i, case in enumerate(cases) if case.group in test_groups]
        splits.append((train, test))
    return splits


def signflip_p(differences: np.ndarray, cases: list[JudgedCase], draws: int = 2000) -> float:
    """One-sided, query-weighted group sign-flip test, with a finite-sample correction."""
    totals: dict[str, float] = {}
    for difference, case in zip(differences, cases, strict=True):
        totals[case.group] = totals.get(case.group, 0.0) + float(difference)
    observed = float(differences.sum())
    group_totals = np.array(list(totals.values()))
    rng = np.random.default_rng(0)
    null = rng.choice([-1, 1], size=(draws, len(group_totals))) @ group_totals
    return float((1 + np.sum(null >= observed)) / (draws + 1))


def evaluate(
    cases: list[JudgedCase],
    folds: int = 5,
    alpha: float = 1.0,
    corpus: CorpusStatistics | None = None,
    min_gain: float = 0.0,
) -> dict[str, Any]:
    """Compare out-of-fold selectors with original and training-chosen fixed policies.

    Outcomes are first-stage ranking utility only. A separate end-to-end pilot is
    required to establish gains after time adjustment, deduplication and reranking.
    """
    post = [feature_matrix(c.alternatives) for c in cases]
    targets = [np.array([utility(a, c.relevant) for a in c.alternatives]) for c in cases]
    original = np.array(
        [utility(next(a for a in c.alternatives if a.id == "original"), c.relevant) for c in cases]
    )
    pre = [corpus.matrix(c.alternatives) for c in cases] if corpus is not None else None
    selected = np.zeros(len(cases))
    pre_selected = np.zeros(len(cases))
    fixed = np.zeros(len(cases))
    missing_fixed_options = 0
    chosen_ids = [""] * len(cases)
    fold_details = []
    for train, test in grouped_folds(cases, folds):
        model = fit_ridge(
            [post[i] for i in train], [targets[i] for i in train], cases[0].signature, alpha
        )
        pre_model = (
            fit_ridge([pre[i] for i in train], [targets[i] for i in train], "pre", alpha)
            if pre is not None
            else None
        )
        common = set.intersection(*[{a.id for a in cases[i].alternatives} for i in train])
        best_fixed = max(
            sorted(common),
            key=lambda option_id: np.mean(
                [
                    utility(
                        next(a for a in cases[i].alternatives if a.id == option_id),
                        cases[i].relevant,
                    )
                    for i in train
                ]
            ),
        )
        for i in test:
            case = cases[i]
            choice, _ = select_alternative(case.alternatives, model, min_gain)
            selected[i] = utility(choice, case.relevant)
            chosen_ids[i] = choice.id
            fixed_option = next(
                (a for a in case.alternatives if a.id == best_fixed),
                next(a for a in case.alternatives if a.id == "original"),
            )
            missing_fixed_options += int(not any(a.id == best_fixed for a in case.alternatives))
            fixed[i] = utility(fixed_option, case.relevant)
            if pre_model is not None and pre is not None:
                predictions = pre_model.predict(pre[i])
                available = [j for j, a in enumerate(case.alternatives) if a.documents]
                baseline = next(j for j, a in enumerate(case.alternatives) if a.id == "original")
                pick = max(available, key=lambda j: predictions[j]) if available else baseline
                if (
                    case.alternatives[baseline].documents
                    and predictions[pick] - predictions[baseline] <= min_gain
                ):
                    pick = baseline
                pre_selected[i] = targets[i][pick]
        fold_details.append(
            {
                "train_groups": sorted({cases[i].group for i in train}),
                "test_groups": sorted({cases[i].group for i in test}),
                "fixed_policy": best_fixed,
            }
        )
    result: dict[str, Any] = {
        "metric": "binary_ndcg@10",
        "stage": "first_stage_before_merge",
        "queries": len(cases),
        "groups": len({c.group for c in cases}),
        "fixed_alpha": alpha,
        "min_predicted_gain": min_gain,
        "original": float(original.mean()),
        "training_chosen_fixed": float(fixed.mean()),
        "fixed_missing_options": missing_fixed_options,
        "post_retrieval": float(selected.mean()),
        "post_gain_points_vs_fixed": float(100 * (selected - fixed).mean()),
        "post_p_vs_fixed": signflip_p(selected - fixed, cases),
        "tied_options_queries": sum(float(t.max() - t.min()) <= 1e-9 for t in targets),
        "single_positive_queries": sum(len(c.relevant) == 1 for c in cases),
        "folds": fold_details,
        "selected_ids": dict(zip([c.id for c in cases], chosen_ids, strict=True)),
        "caveat": "First-stage utility only; no claim of end-to-end answer gains or cost savings.",
    }
    if pre is not None:
        result.update(
            {
                "pre_retrieval": float(pre_selected.mean()),
                "pre_p_vs_fixed": signflip_p(pre_selected - fixed, cases),
                "pre_features": "query length and corpus IDF moments (seven features)",
            }
        )
        p_values = [result["post_p_vs_fixed"], result["pre_p_vs_fixed"]]
        ordered = sorted(range(2), key=lambda i: p_values[i])
        adjusted = np.maximum.accumulate(
            [min(1.0, (2 - rank) * p_values[i]) for rank, i in enumerate(ordered)]
        )
        for i, value in zip(ordered, adjusted, strict=True):
            result[("post", "pre")[i] + "_holm_p_vs_fixed"] = float(value)
    return result


@app.command()
def train(
    dataset: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Judged candidate JSONL")
    ],
    report: Annotated[Path, typer.Option(help="New evaluation JSON output")],
    model: Annotated[Path | None, typer.Option(help="Optional new archive model artifact")] = None,
    corpus: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Optional fixed docs.jsonl for the matched corpus-statistic control",
        ),
    ] = None,
    folds: Annotated[int, typer.Option(min=2)] = 5,
    alpha: Annotated[float, typer.Option(min=0.000001)] = 1.0,
    min_gain: Annotated[float, typer.Option(min=0.0)] = 0.0,
) -> None:
    """Evaluate cached first-stage candidates and optionally fit an opt-in model."""
    for output in (report, model):
        if output is not None and output.exists():
            raise typer.BadParameter(f"Refusing to overwrite existing output: {output}")
    if model is not None and model.resolve() == report.resolve():
        raise typer.BadParameter("Report and model paths must differ")
    try:
        cases = load_cases(dataset)
        statistics = None
        if corpus is not None:
            with corpus.open(encoding="utf-8") as stream:
                statistics = CorpusStatistics(
                    json.loads(line)["text"] for line in stream if line.strip()
                )
        result = evaluate(cases, folds, alpha, statistics, min_gain)
        if result["single_positive_queries"]:
            typer.echo(
                "Warning: single-positive labels can hide other relevant segments; pool judgments.",
                err=True,
            )
        with report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if model is not None:
            fitted = fit_ridge(
                [feature_matrix(c.alternatives) for c in cases],
                [np.array([utility(a, c.relevant) for a in c.alternatives]) for c in cases],
                cases[0].signature,
                alpha,
            )
            fitted.save(model)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote first-stage evaluation to {report}; production selection remains opt-in.")


if __name__ == "__main__":
    app()
