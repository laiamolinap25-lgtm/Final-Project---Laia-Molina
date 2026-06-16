import numpy as np
import pandas as pd
import pytest

# Try to import the functions under test. If analysis.py (or one of its
# dependencies) cannot be imported, skip the whole module cleanly.
analysis = pytest.importorskip(
    "analysis",
    reason="analysis.py could not be imported (check dependencies like brainspace).",
)

calc_delta = analysis.calc_delta
calc_baseline = analysis.calc_baseline
calc_delta_declining = analysis.calc_delta_declining
get_hemisphere = analysis.get_hemisphere
normalize_points_to_sphere = analysis.normalize_points_to_sphere


def test_delta_one_year():
    """From 10 to 16 over 12 months (1 year) -> 6.0 per year."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "timepoint": ["bl", "m12"],
        "region_x": [10.0, 16.0],
    })
    result = calc_delta(df, ["region_x"])
    assert abs(result.loc["P1", "region_x"] - 6.0) < 1e-9


def test_delta_two_years():
    """Same change (10 to 16) but over 24 months (2 years) -> 3.0 per year."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "timepoint": ["bl", "m24"],
        "region_x": [10.0, 16.0],
    })
    result = calc_delta(df, ["region_x"])
    assert abs(result.loc["P1", "region_x"] - 3.0) < 1e-9


def test_delta_unsorted_visits():
    """Visits given out of order must still use first and last by time."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "timepoint": ["m12", "bl", "m06"],
        "region_x": [16.0, 10.0, 13.0],
    })
    result = calc_delta(df, ["region_x"])
    assert abs(result.loc["P1", "region_x"] - 6.0) < 1e-9

def test_baseline_takes_first_visit():
    """Baseline must take the earliest visit, regardless of row order."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "timepoint": ["m12", "bl", "m06"],
        "region_x": [16.0, 10.0, 13.0],
    })
    result = calc_baseline(df, ["region_x"])
    assert abs(result.loc["P1", "region_x"] - 10.0) < 1e-9


def test_baseline_two_patients():
    """Baseline computed independently for each patient."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2"],
        "timepoint": ["bl", "m12", "bl", "m12"],
        "region_x": [10.0, 16.0, 20.0, 30.0],
    })
    result = calc_baseline(df, ["region_x"])
    assert abs(result.loc["P1", "region_x"] - 10.0) < 1e-9
    assert abs(result.loc["P2", "region_x"] - 20.0) < 1e-9

def test_declining_uses_only_selected_patients():
    """Only patients listed in df_decline should appear in the result."""
    df_all = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2"],
        "timepoint": ["bl", "m12", "bl", "m12"],
        "region_x": [10.0, 16.0, 5.0, 5.0],
    })
    df_decline = pd.DataFrame({"patient_id": ["P1"]})
    result = calc_delta_declining(df_decline, df_all, ["region_x"])
    assert list(result.index) == ["P1"]
    assert abs(result.loc["P1", "region_x"] - 6.0) < 1e-9

def test_hemisphere_left():
    assert get_hemisphere("ctx-lh-superiorfrontal") == "lh"


def test_hemisphere_right():
    assert get_hemisphere("ctx-rh-precuneus") == "rh"


def test_hemisphere_none_for_subcortical():
    """A subcortical region name has no lh/rh prefix -> None."""
    assert get_hemisphere("Left-Hippocampus") is None


def test_points_projected_to_unit_sphere():
    """After normalization, every point must lie at distance 1 from origin."""
    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [-1.0, -1.0, -1.0],
    ])
    sphere = normalize_points_to_sphere(pts)
    norms = np.linalg.norm(sphere, axis=1)
    assert np.allclose(norms, 1.0)