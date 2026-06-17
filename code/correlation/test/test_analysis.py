# -*- coding: utf-8 -*-
"""
Tests for analysis.py

Covers: calc_delta, calc_baseline, apply_fdr, spearman_ci,
        normalize_ctx_index, normalize_sub_index, format_stat_text
"""

import numpy as np
import pandas as pd
import pytest

# ── Minimal stubs so analysis.py imports without real config files ────────────
import sys
from types import ModuleType

_config_stub = ModuleType("config")
_config_stub.timepoint_to_months = {
    "bl": 0, "m06": 6, "m12": 12, "m24": 24,
    "m36": 36, "m48": 48, "m60": 60,
}
_config_stub.subcortical_cols = [
    "left-hippocampus", "left-amygdala", "right-hippocampus", "right-amygdala",
]
_config_stub.color_cortical    = "#253a6b"
_config_stub.color_subcortical = "#de5f2d"
_config_stub.color_trendline   = "green"
_config_stub.receptors_csv     = ""          # not used in unit tests
sys.modules.setdefault("config", _config_stub)

from analysis import (  # noqa: E402
    calc_delta,
    calc_baseline,
    apply_fdr,
    spearman_ci,
    normalize_ctx_index,
    normalize_sub_index,
    format_stat_text,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def simple_df():
    """Two patients with two timepoints each."""
    return pd.DataFrame({
        "patient_id": [1, 1, 2, 2],
        "timepoint":  ["bl", "m24", "bl", "m24"],
        "lh_entorhinal": [1.0, 3.0, 2.0, 4.0],
        "rh_entorhinal": [0.5, 1.5, 1.0, 3.0],
    })


@pytest.fixture()
def cols():
    return ["lh_entorhinal", "rh_entorhinal"]


# =============================================================================
# calc_delta
# =============================================================================

class TestCalcDelta:
    def test_shape(self, simple_df, cols):
        result = calc_delta(simple_df, cols)
        assert result.shape == (2, 2), "Should return one row per patient"

    def test_values_per_year(self, simple_df, cols):
        result = calc_delta(simple_df, cols)
        # Patient 1: (3-1)/2 years = 1.0 per year for lh_entorhinal
        assert pytest.approx(result.loc[1, "lh_entorhinal"], abs=1e-6) == 1.0

    def test_single_timepoint_returns_nan(self, cols):
        df = pd.DataFrame({
            "patient_id": [1],
            "timepoint":  ["bl"],
            "lh_entorhinal": [1.0],
            "rh_entorhinal": [0.5],
        })
        result = calc_delta(df, cols)
        assert result["lh_entorhinal"].isna().all()

    def test_does_not_mutate_input(self, simple_df, cols):
        original_cols = simple_df.columns.tolist()
        calc_delta(simple_df, cols)
        assert simple_df.columns.tolist() == original_cols


# =============================================================================
# calc_baseline
# =============================================================================

class TestCalcBaseline:
    def test_returns_first_timepoint(self, simple_df, cols):
        result = calc_baseline(simple_df, cols)
        assert result.loc[1, "lh_entorhinal"] == pytest.approx(1.0)
        assert result.loc[2, "lh_entorhinal"] == pytest.approx(2.0)

    def test_shape(self, simple_df, cols):
        result = calc_baseline(simple_df, cols)
        assert result.shape == (2, 2)


# =============================================================================
# apply_fdr
# =============================================================================

class TestApplyFdr:
    def _make_records(self, pvalues):
        return [
            {"receptor": f"R{i}", "region_type": "cortical",
             "rho": 0.5, "pvalue": p}
            for i, p in enumerate(pvalues)
        ]

    def test_adds_fdr_fields(self):
        records = self._make_records([0.001, 0.5, 0.8])
        result = apply_fdr(records)
        assert all("pvalue_fdr" in r for r in result)
        assert all("sig_fdr" in r for r in result)

    def test_empty_input(self):
        assert apply_fdr([]) == []

    def test_significant_small_pvalue(self):
        records = self._make_records([1e-10, 0.9, 0.95])
        result = apply_fdr(records)
        assert result[0]["sig_fdr"] is True

    def test_nonsignificant_large_pvalue(self):
        records = self._make_records([0.7, 0.8, 0.9])
        result = apply_fdr(records)
        assert all(not r["sig_fdr"] for r in result)

    def test_fdr_pvalue_between_0_and_1(self):
        records = self._make_records([0.01, 0.05, 0.1])
        result = apply_fdr(records)
        for r in result:
            assert 0.0 <= r["pvalue_fdr"] <= 1.0


# =============================================================================
# spearman_ci
# =============================================================================

class TestSpearmanCI:
    def test_small_n_returns_nan(self):
        lo, hi = spearman_ci(0.5, n=3)
        assert np.isnan(lo) and np.isnan(hi)

    def test_ci_contains_rho(self):
        rho = 0.6
        lo, hi = spearman_ci(rho, n=30)
        assert lo < rho < hi

    def test_ci_ordering(self):
        lo, hi = spearman_ci(0.3, n=50)
        assert lo < hi

    def test_negative_rho(self):
        lo, hi = spearman_ci(-0.7, n=40)
        assert lo < -0.7 < hi

    def test_extreme_rho_clipped(self):
        lo, hi = spearman_ci(1.0, n=20)
        assert np.isfinite(lo) and np.isfinite(hi)


# =============================================================================
# normalize_ctx_index
# =============================================================================

class TestNormalizeCtxIndex:
    def test_standard_mode(self):
        s = pd.Series([1.0, 2.0], index=["lh_entorhinal", "rh_parahippocampal"])
        cfg = {
            "ctx_index_transform": lambda idx: "ctx-" + idx.str.replace("_", "-"),
        }
        result = normalize_ctx_index(s, cfg)
        assert "ctx-lh-entorhinal" in result.index
        assert "ctx-rh-parahippocampal" in result.index

    def test_does_not_mutate_input(self):
        s = pd.Series([1.0], index=["lh_entorhinal"])
        original_index = s.index.tolist()
        cfg = {"ctx_index_transform": lambda idx: "ctx-" + idx.str.replace("_", "-")}
        normalize_ctx_index(s, cfg)
        assert s.index.tolist() == original_index


# =============================================================================
# normalize_sub_index
# =============================================================================

class TestNormalizeSubIndex:
    def test_accumbens_area_casing(self):
        s = pd.Series([1.0], index=["left-accumbens-area"])
        result = normalize_sub_index(s)
        assert "Left-Accumbens-area" in result.index

    def test_ventraldc_casing(self):
        s = pd.Series([1.0], index=["left-ventraldc"])
        result = normalize_sub_index(s)
        assert "Left-VentralDC" in result.index

    def test_generic_region(self):
        s = pd.Series([1.0], index=["left-hippocampus"])
        result = normalize_sub_index(s)
        assert "Left-Hippocampus" in result.index


# =============================================================================
# format_stat_text
# =============================================================================

class TestFormatStatText:
    def test_significant_marker(self):
        text = format_stat_text(0.7, 0.001, 0.004, sig_fdr=True)
        assert "✱" in text

    def test_nonsignificant_marker(self):
        text = format_stat_text(0.2, 0.3, 0.6, sig_fdr=False)
        assert "ns" in text

    def test_contains_rho(self):
        text = format_stat_text(0.55, 0.01, 0.03, sig_fdr=True)
        assert "ρ=0.550" in text

    def test_scientific_notation_small_pvalue(self):
        text = format_stat_text(0.9, 1e-8, 1e-7, sig_fdr=True)
        assert "e" in text.lower()
