"""Unit tests for eval/experiments/two_stage_sweep.py.

Tests cover:
- Default condition counts for each axis and all-axes combined
- Condition structure: required keys, override keys, tag keys
- Per-axis override values (correct parameter wired to correct override key)
- Custom axis values (constructor overrides respected)
- axes= subset filtering
- Unknown axis raises ValueError
- No duplicate condition IDs across a full sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.experiments.two_stage_sweep import (
    _DEFAULT_HYDE_ALPHA,
    _DEFAULT_MERGE_RRF_K,
    _DEFAULT_MERGE_STRATEGY,
    _DEFAULT_POOL_SIZE,
    _DEFAULT_REWRITE_VARIANTS,
    DOC_ORDERS,
    HYDE_ALPHAS,
    MERGE_RRF_KS,
    MERGE_STRATEGIES,
    POOL_SIZES,
    REWRITE_VARIANTS,
    TwoStageSweepExperiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_CONDITION_KEYS = {"id", "label", "overrides", "tags"}
_BASE_OVERRIDE_KEYS = {
    "hybrid_search.enabled",
    "hybrid_search.fusion_method",
    "two_stage.enabled",
    "two_stage.query_rewrite_enabled",
    "two_stage.hyde_enabled",
    "reranker.enabled",
}


def _make_exp(**kwargs) -> TwoStageSweepExperiment:
    """Construct experiment bypassing BaseExperiment.__init__ file loading."""
    exp = TwoStageSweepExperiment.__new__(TwoStageSweepExperiment)
    # Replicate just the attribute assignments made by TwoStageSweepExperiment.__init__
    # without calling super().__init__ (which requires a real config file).

    axes = kwargs.get("axes")
    exp._axes = set(axes) if axes is not None else TwoStageSweepExperiment._ALL_AXES
    exp._hyde_alphas = kwargs.get("hyde_alphas", HYDE_ALPHAS)
    exp._rewrite_variants = kwargs.get("rewrite_variants", REWRITE_VARIANTS)
    exp._pool_sizes = kwargs.get("pool_sizes", POOL_SIZES)
    exp._merge_strategies = kwargs.get("merge_strategies", MERGE_STRATEGIES)
    exp._merge_rrf_ks = kwargs.get("merge_rrf_ks", MERGE_RRF_KS)
    exp._doc_orders = kwargs.get("doc_orders", DOC_ORDERS)
    return exp


# ---------------------------------------------------------------------------
# Condition counts
# ---------------------------------------------------------------------------


class TestConditionCounts:
    def test_full_sweep_count(self):
        """Total conditions = sum of all six axis lengths."""
        exp = _make_exp()
        expected = (
            len(HYDE_ALPHAS)
            + len(REWRITE_VARIANTS)
            + len(POOL_SIZES)
            + len(MERGE_STRATEGIES)
            + len(MERGE_RRF_KS)
            + len(DOC_ORDERS)
        )
        assert len(exp.conditions()) == expected

    def test_hyde_alpha_axis_count(self):
        exp = _make_exp(axes=["hyde_alpha"])
        assert len(exp.conditions()) == len(HYDE_ALPHAS)

    def test_rewrite_variants_axis_count(self):
        exp = _make_exp(axes=["rewrite_variants"])
        assert len(exp.conditions()) == len(REWRITE_VARIANTS)

    def test_pool_size_axis_count(self):
        exp = _make_exp(axes=["pool_size"])
        assert len(exp.conditions()) == len(POOL_SIZES)

    def test_merge_strategy_axis_count(self):
        exp = _make_exp(axes=["merge_strategy"])
        assert len(exp.conditions()) == len(MERGE_STRATEGIES)

    def test_merge_rrf_k_axis_count(self):
        exp = _make_exp(axes=["merge_rrf_k"])
        assert len(exp.conditions()) == len(MERGE_RRF_KS)

    def test_doc_order_axis_count(self):
        exp = _make_exp(axes=["doc_order"])
        assert len(exp.conditions()) == len(DOC_ORDERS)

    def test_two_axis_subset_count(self):
        exp = _make_exp(axes=["hyde_alpha", "merge_strategy"])
        assert len(exp.conditions()) == len(HYDE_ALPHAS) + len(MERGE_STRATEGIES)

    def test_empty_custom_list_produces_zero_conditions_for_that_axis(self):
        exp = _make_exp(axes=["hyde_alpha"], hyde_alphas=[])
        assert len(exp.conditions()) == 0

    def test_custom_values_respected(self):
        exp = _make_exp(axes=["merge_rrf_k"], merge_rrf_ks=[10, 30])
        assert len(exp.conditions()) == 2


# ---------------------------------------------------------------------------
# Condition structure
# ---------------------------------------------------------------------------


class TestConditionStructure:
    @pytest.fixture(
        params=[
            "hyde_alpha",
            "rewrite_variants",
            "pool_size",
            "merge_strategy",
            "merge_rrf_k",
            "doc_order",
        ]
    )
    def single_axis_conditions(self, request):
        exp = _make_exp(axes=[request.param])
        return exp.conditions()

    def test_required_keys_present(self, single_axis_conditions):
        for cond in single_axis_conditions:
            assert (
                set(cond.keys()) >= _REQUIRED_CONDITION_KEYS
            ), f"Missing keys in condition {cond.get('id')}"

    def test_id_is_non_empty_string(self, single_axis_conditions):
        for cond in single_axis_conditions:
            assert isinstance(cond["id"], str) and cond["id"]

    def test_label_is_non_empty_string(self, single_axis_conditions):
        for cond in single_axis_conditions:
            assert isinstance(cond["label"], str) and cond["label"]

    def test_base_overrides_present_in_every_condition(self, single_axis_conditions):
        for cond in single_axis_conditions:
            assert (
                set(cond["overrides"].keys()) >= _BASE_OVERRIDE_KEYS
            ), f"Missing base override keys in {cond['id']}"

    def test_sweep_axis_tag_present(self, single_axis_conditions):
        for cond in single_axis_conditions:
            assert "sweep_axis" in cond["tags"], f"Missing sweep_axis tag in {cond['id']}"


# ---------------------------------------------------------------------------
# Axis A – HyDE alpha overrides
# ---------------------------------------------------------------------------


class TestHydeAlphaAxis:
    def _conditions(self, alphas=None):
        kwargs = {"axes": ["hyde_alpha"]}
        if alphas is not None:
            kwargs["hyde_alphas"] = alphas
        return _make_exp(**kwargs).conditions()

    def test_override_key_is_hyde_alpha(self):
        for cond in self._conditions():
            assert "two_stage.hyde_alpha" in cond["overrides"]

    def test_override_values_match_defaults(self):
        for cond, expected in zip(self._conditions(), HYDE_ALPHAS, strict=True):
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(expected)

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert (
                cond["overrides"]["two_stage.query_rewrite_variants"] == _DEFAULT_REWRITE_VARIANTS
            )
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == _DEFAULT_POOL_SIZE

    def test_tag_records_alpha(self):
        for cond, expected in zip(self._conditions(), HYDE_ALPHAS, strict=True):
            assert cond["tags"]["hyde_alpha"] == str(expected)

    def test_custom_alphas(self):
        conds = self._conditions(alphas=[0.2, 0.8])
        vals = [c["overrides"]["two_stage.hyde_alpha"] for c in conds]
        assert vals == pytest.approx([0.2, 0.8])


# ---------------------------------------------------------------------------
# Axis B – Rewrite variants overrides
# ---------------------------------------------------------------------------


class TestRewriteVariantsAxis:
    def _conditions(self, variants=None):
        kwargs = {"axes": ["rewrite_variants"]}
        if variants is not None:
            kwargs["rewrite_variants"] = variants
        return _make_exp(**kwargs).conditions()

    def test_override_key_is_query_rewrite_variants(self):
        for cond in self._conditions():
            assert "two_stage.query_rewrite_variants" in cond["overrides"]

    def test_override_values_match_defaults(self):
        for cond, expected in zip(self._conditions(), REWRITE_VARIANTS, strict=True):
            assert cond["overrides"]["two_stage.query_rewrite_variants"] == expected

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(_DEFAULT_HYDE_ALPHA)
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == _DEFAULT_POOL_SIZE

    def test_tag_records_variant_count(self):
        for cond, expected in zip(self._conditions(), REWRITE_VARIANTS, strict=True):
            assert cond["tags"]["rewrite_variants"] == str(expected)


# ---------------------------------------------------------------------------
# Axis C – Pool size overrides
# ---------------------------------------------------------------------------


class TestPoolSizeAxis:
    def _conditions(self, pool_sizes=None):
        kwargs = {"axes": ["pool_size"]}
        if pool_sizes is not None:
            kwargs["pool_sizes"] = pool_sizes
        return _make_exp(**kwargs).conditions()

    def test_override_key_is_top_k_multiplier(self):
        for cond in self._conditions():
            assert "hybrid_search.top_k_multiplier" in cond["overrides"]

    def test_override_values_match_defaults(self):
        for cond, expected in zip(self._conditions(), POOL_SIZES, strict=True):
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == expected

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(_DEFAULT_HYDE_ALPHA)
            assert (
                cond["overrides"]["two_stage.query_rewrite_variants"] == _DEFAULT_REWRITE_VARIANTS
            )

    def test_tag_records_multiplier(self):
        for cond, expected in zip(self._conditions(), POOL_SIZES, strict=True):
            assert cond["tags"]["top_k_multiplier"] == str(expected)


# ---------------------------------------------------------------------------
# Axis D – Merge strategy overrides
# ---------------------------------------------------------------------------


class TestMergeStrategyAxis:
    def _conditions(self, strategies=None):
        kwargs = {"axes": ["merge_strategy"]}
        if strategies is not None:
            kwargs["merge_strategies"] = strategies
        return _make_exp(**kwargs).conditions()

    def test_override_key_is_merge_strategy(self):
        for cond in self._conditions():
            assert "two_stage.merge_strategy" in cond["overrides"]

    def test_override_values_match_defaults(self):
        for cond, expected in zip(self._conditions(), MERGE_STRATEGIES, strict=True):
            assert cond["overrides"]["two_stage.merge_strategy"] == expected

    def test_rrf_k_held_at_default(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.merge_rrf_k"] == _DEFAULT_MERGE_RRF_K

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(_DEFAULT_HYDE_ALPHA)
            assert (
                cond["overrides"]["two_stage.query_rewrite_variants"] == _DEFAULT_REWRITE_VARIANTS
            )
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == _DEFAULT_POOL_SIZE

    def test_tag_records_strategy(self):
        for cond, expected in zip(self._conditions(), MERGE_STRATEGIES, strict=True):
            assert cond["tags"]["merge_strategy"] == expected

    def test_tag_sweep_axis_value(self):
        for cond in self._conditions():
            assert cond["tags"]["sweep_axis"] == "merge_strategy"

    def test_custom_strategies(self):
        conds = self._conditions(strategies=["coverage"])
        assert len(conds) == 1
        assert conds[0]["overrides"]["two_stage.merge_strategy"] == "coverage"


# ---------------------------------------------------------------------------
# Axis E – Merge RRF k overrides
# ---------------------------------------------------------------------------


class TestMergeRrfKAxis:
    def _conditions(self, rrf_ks=None):
        kwargs = {"axes": ["merge_rrf_k"]}
        if rrf_ks is not None:
            kwargs["merge_rrf_ks"] = rrf_ks
        return _make_exp(**kwargs).conditions()

    def test_override_key_is_merge_rrf_k(self):
        for cond in self._conditions():
            assert "two_stage.merge_rrf_k" in cond["overrides"]

    def test_override_values_match_defaults(self):
        for cond, expected in zip(self._conditions(), MERGE_RRF_KS, strict=True):
            assert cond["overrides"]["two_stage.merge_rrf_k"] == expected

    def test_merge_strategy_fixed_to_diverse_rrf(self):
        """RRF k axis must lock merge_strategy=diverse_rrf so the k effect is isolated."""
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.merge_strategy"] == "diverse_rrf"

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(_DEFAULT_HYDE_ALPHA)
            assert (
                cond["overrides"]["two_stage.query_rewrite_variants"] == _DEFAULT_REWRITE_VARIANTS
            )
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == _DEFAULT_POOL_SIZE

    def test_tag_records_rrf_k(self):
        for cond, expected in zip(self._conditions(), MERGE_RRF_KS, strict=True):
            assert cond["tags"]["merge_rrf_k"] == str(expected)

    def test_tag_sweep_axis_value(self):
        for cond in self._conditions():
            assert cond["tags"]["sweep_axis"] == "merge_rrf_k"

    def test_custom_rrf_ks(self):
        conds = self._conditions(rrf_ks=[10, 50])
        vals = [c["overrides"]["two_stage.merge_rrf_k"] for c in conds]
        assert vals == [10, 50]


# ---------------------------------------------------------------------------
# Axis F: doc_order
# ---------------------------------------------------------------------------


class TestDocOrderAxis:
    def _conditions(self, **kwargs):
        exp = _make_exp(axes=["doc_order"], **kwargs)
        return exp.conditions()

    def test_default_doc_orders_count(self):
        assert len(self._conditions()) == len(DOC_ORDERS)

    def test_override_key_is_prompt_doc_order(self):
        for cond in self._conditions():
            assert "two_stage.prompt_doc_order" in cond["overrides"]

    def test_override_values_match_doc_orders(self):
        for cond, expected in zip(self._conditions(), DOC_ORDERS, strict=True):
            assert cond["overrides"]["two_stage.prompt_doc_order"] == expected

    def test_other_axes_held_at_defaults(self):
        for cond in self._conditions():
            assert cond["overrides"]["two_stage.hyde_alpha"] == pytest.approx(_DEFAULT_HYDE_ALPHA)
            assert (
                cond["overrides"]["two_stage.query_rewrite_variants"] == _DEFAULT_REWRITE_VARIANTS
            )
            assert cond["overrides"]["hybrid_search.top_k_multiplier"] == _DEFAULT_POOL_SIZE
            assert cond["overrides"]["two_stage.merge_strategy"] == _DEFAULT_MERGE_STRATEGY
            assert cond["overrides"]["two_stage.merge_rrf_k"] == _DEFAULT_MERGE_RRF_K

    def test_tag_records_doc_order(self):
        for cond, expected in zip(self._conditions(), DOC_ORDERS, strict=True):
            assert cond["tags"]["prompt_doc_order"] == expected

    def test_tag_sweep_axis_value(self):
        for cond in self._conditions():
            assert cond["tags"]["sweep_axis"] == "doc_order"

    def test_custom_doc_orders(self):
        conds = self._conditions(doc_orders=["rank", "book_end"])
        vals = [c["overrides"]["two_stage.prompt_doc_order"] for c in conds]
        assert vals == ["rank", "book_end"]


# ---------------------------------------------------------------------------
# Unknown axis / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_axis_raises(self):
        # ensure constructor validation path is exercised with minimal setup
        with pytest.raises(ValueError, match="Unknown sweep axes"):
            TwoStageSweepExperiment(axes=["not_a_real_axis"])

    def test_no_duplicate_condition_ids_full_sweep(self):
        exp = _make_exp()
        ids = [c["id"] for c in exp.conditions()]
        assert len(ids) == len(
            set(ids)
        ), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"

    def test_no_duplicate_condition_ids_per_axis(self):
        for axis in (
            "hyde_alpha",
            "rewrite_variants",
            "pool_size",
            "merge_strategy",
            "merge_rrf_k",
            "doc_order",
        ):
            exp = _make_exp(axes=[axis])
            ids = [c["id"] for c in exp.conditions()]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in axis={axis}: {ids}"
