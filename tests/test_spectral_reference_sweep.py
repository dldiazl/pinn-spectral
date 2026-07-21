"""Tests for full-window spectral validation against the reference solution."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.spectral import (  # noqa: E402
    annotate_metric_minimum_paths,
    annotate_reference_window_sweep,
    build_metric_path_comparison,
    compute_temporal_stability_metrics,
    compute_window_reference_metrics,
    select_minimum_reference_rmse_at_tmax,
)


def test_window_rmse_is_independent_of_sample_replication() -> None:
    """Repeating identical errors must not change normalized MSE or RMSE."""
    reference = np.zeros((2, 3), dtype=np.float64)
    raw = np.full((2, 3), 2.0, dtype=np.float64)
    filtered = np.full((2, 3), 0.5, dtype=np.float64)

    original = compute_window_reference_metrics(filtered, raw, reference)
    repeated = compute_window_reference_metrics(
        np.tile(filtered, (3, 4)),
        np.tile(raw, (3, 4)),
        np.tile(reference, (3, 4)),
    )

    assert np.isclose(original["filtered_window_mse"], 0.25)
    assert np.isclose(original["filtered_window_rmse"], 0.5)
    assert np.isclose(original["raw_window_rmse"], 2.0)
    assert np.isclose(original["rmse_ratio"], 0.25)
    assert np.isclose(original["rmse_reduction"], 0.75)
    for key in original:
        assert np.isclose(original[key], repeated[key])


def test_window_metrics_reject_mismatched_grids() -> None:
    """Reference comparisons must never interpolate or truncate silently."""
    filtered = np.zeros((3, 4), dtype=np.float64)
    raw = np.zeros((3, 4), dtype=np.float64)
    reference = np.zeros((3, 5), dtype=np.float64)

    try:
        compute_window_reference_metrics(filtered, raw, reference)
    except ValueError as error:
        assert "identical shapes" in str(error)
    else:
        raise AssertionError("Expected a shape-mismatch error.")


def test_reference_sweep_keeps_both_minimum_paths() -> None:
    """Sensitivity and reference-RMSE paths must be annotated independently."""
    sensitivity = pd.DataFrame(
        {
            "tmax": [0.4, 0.4, 0.5, 0.5],
            "tcut": [0.1, 0.2, 0.1, 0.2],
            "is_minimum_for_tmax": [True, False, False, True],
            "is_reduced_sensitivity_onset": [False, True, True, False],
            "is_selected": [False, False, False, True],
        }
    )
    reference = pd.DataFrame(
        {
            "tmax": [0.4, 0.4, 0.5, 0.5],
            "tcut": [0.1, 0.2, 0.1, 0.2],
            "filtered_window_rmse": [0.4, 0.2, 0.1, 0.3],
        }
    )

    annotated = annotate_reference_window_sweep(reference, sensitivity)

    sensitivity_path = annotated[annotated["is_minimum_sensitivity_for_tmax"]]
    reduced_sensitivity_path = annotated[annotated["is_reduced_sensitivity_onset_for_tmax"]]
    rmse_path = annotated[annotated["is_minimum_reference_rmse_for_tmax"]]
    selected = annotated[annotated["is_selected_by_sensitivity"]]

    assert list(sensitivity_path["tcut"]) == [0.1, 0.2]
    assert list(reduced_sensitivity_path["tcut"]) == [0.2, 0.1]
    assert list(rmse_path["tcut"]) == [0.2, 0.1]
    assert len(selected) == 1
    assert float(selected.iloc[0]["tmax"]) == 0.5
    assert float(selected.iloc[0]["tcut"]) == 0.2
    assert int(annotated["is_global_reference_rmse"].sum()) == 1


def test_select_minimum_reference_rmse_at_requested_tmax() -> None:
    """The Figure 12b cutoff must come from the RMSE path at the same tmax."""
    sweep = pd.DataFrame(
        {
            "tmax": [4.9, 4.9, 5.0, 5.0, 5.0],
            "tcut": [4.7, 4.8, 4.7, 4.8, 4.9],
            "filtered_window_rmse": [0.2, 0.1, 0.08, 0.03, 0.05],
        }
    )

    selected = select_minimum_reference_rmse_at_tmax(sweep, 5.0)

    assert np.isclose(float(selected["tmax"]), 5.0)
    assert np.isclose(float(selected["tcut"]), 4.8)
    assert np.isclose(float(selected["filtered_window_rmse"]), 0.03)


def test_select_minimum_reference_rmse_resolves_ties_deterministically() -> None:
    """Equal RMSE values must select the earliest cutoff."""
    sweep = pd.DataFrame(
        {
            "tmax": [5.0, 5.0],
            "tcut": [4.9, 4.8],
            "filtered_window_rmse": [0.03, 0.03],
        }
    )

    selected = select_minimum_reference_rmse_at_tmax(sweep, 5.0)

    assert np.isclose(float(selected["tcut"]), 4.8)



def test_temporal_stability_metrics_include_boundary_nodes() -> None:
    """Boundary variation must contribute to the all-node stability metrics."""
    filtered = np.array(
        [
            [0.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    metrics = compute_temporal_stability_metrics(
        filtered_segment=filtered,
        dt=0.5,
        window_duration=1.0,
    )

    assert metrics["temporal_std_rms"] > 0.0
    assert metrics["boundary_temporal_std_rms"] > 0.0
    assert np.isclose(metrics["interior_temporal_std_rms"], 0.0)
    assert metrics["temporal_derivative_rms"] > 0.0
    assert metrics["boundary_temporal_derivative_rms"] > 0.0
    assert np.isclose(metrics["interior_temporal_derivative_rms"], 0.0)


def test_duration_normalized_std_prefers_longer_equal_deviation_window() -> None:
    """Equal temporal deviation over a longer duration must score lower."""
    filtered = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)

    short = compute_temporal_stability_metrics(
        filtered_segment=filtered,
        dt=0.5,
        window_duration=1.0,
    )
    long = compute_temporal_stability_metrics(
        filtered_segment=filtered,
        dt=1.0,
        window_duration=2.0,
    )

    assert np.isclose(short["temporal_std_rms"], long["temporal_std_rms"])
    assert np.isclose(
        short["temporal_std_per_duration"],
        2.0 * long["temporal_std_per_duration"],
    )


def test_temporal_derivative_rms_matches_linear_slope() -> None:
    """A linear signal must recover its absolute forward-difference slope."""
    t = np.linspace(0.0, 1.0, 6)
    filtered = np.vstack([2.0 * t + 1.0, -2.0 * t + 4.0])

    metrics = compute_temporal_stability_metrics(
        filtered_segment=filtered,
        dt=float(t[1] - t[0]),
        window_duration=1.0,
    )

    assert np.isclose(metrics["temporal_derivative_rms"], 2.0)


def test_metric_minimum_paths_and_reference_penalty() -> None:
    """Reference-free metric paths must be compared without changing them."""
    sweep = pd.DataFrame(
        {
            "tmax": [1.0, 1.0, 2.0, 2.0],
            "tcut": [0.2, 0.4, 0.2, 0.4],
            "filtered_window_rmse": [0.3, 0.1, 0.2, 0.1],
            "is_minimum_reference_rmse_for_tmax": [False, True, False, True],
            "is_minimum_sensitivity_for_tmax": [True, False, True, False],
            "temporal_std_rms": [0.2, 0.3, 0.4, 0.1],
            "temporal_std_per_duration": [0.4, 0.6, 0.8, 0.2],
            "temporal_derivative_rms": [0.5, 0.2, 0.6, 0.3],
        }
    )
    annotated = annotate_metric_minimum_paths(
        sweep,
        [
            "temporal_std_rms",
            "temporal_std_per_duration",
            "temporal_derivative_rms",
        ],
    )
    comparison = build_metric_path_comparison(
        annotated,
        {
            "sensitivity": "is_minimum_sensitivity_for_tmax",
            "temporal_std": "is_minimum_temporal_std_rms_for_tmax",
            "temporal_std_per_duration": (
                "is_minimum_temporal_std_per_duration_for_tmax"
            ),
            "temporal_derivative": (
                "is_minimum_temporal_derivative_rms_for_tmax"
            ),
        },
    )

    first = comparison.iloc[0]
    second = comparison.iloc[1]
    assert np.isclose(first["sensitivity_rmse_ratio_to_optimum"], 3.0)
    assert np.isclose(first["temporal_derivative_tcut"], 0.4)
    assert np.isclose(first["temporal_derivative_rmse_ratio_to_optimum"], 1.0)
    assert np.isclose(second["temporal_std_tcut"], 0.4)
    assert np.isclose(second["temporal_std_rmse_ratio_to_optimum"], 1.0)


def test_metric_path_comparison_contains_all_path_columns() -> None:
    """Path comparison must expose every cutoff path needed by Figure 11f."""
    sweep = pd.DataFrame(
        {
            "tmax": [1.0, 1.0, 2.0, 2.0],
            "tcut": [0.2, 0.4, 0.2, 0.4],
            "filtered_window_rmse": [0.3, 0.1, 0.2, 0.1],
            "is_minimum_reference_rmse_for_tmax": [False, True, False, True],
            "is_minimum_sensitivity_for_tmax": [True, False, True, False],
            "temporal_std_rms": [0.2, 0.3, 0.4, 0.1],
            "temporal_std_per_duration": [0.4, 0.6, 0.8, 0.2],
            "temporal_derivative_rms": [0.5, 0.2, 0.6, 0.3],
        }
    )
    annotated = annotate_metric_minimum_paths(
        sweep,
        [
            "temporal_std_rms",
            "temporal_std_per_duration",
            "temporal_derivative_rms",
        ],
    )
    comparison = build_metric_path_comparison(
        annotated,
        {
            "sensitivity": "is_minimum_sensitivity_for_tmax",
            "temporal_std": "is_minimum_temporal_std_rms_for_tmax",
            "temporal_std_per_duration": (
                "is_minimum_temporal_std_per_duration_for_tmax"
            ),
            "temporal_derivative": (
                "is_minimum_temporal_derivative_rms_for_tmax"
            ),
        },
    )

    expected = {
        "reference_rmse_optimal_tcut",
        "sensitivity_tcut",
        "temporal_std_tcut",
        "temporal_std_per_duration_tcut",
        "temporal_derivative_tcut",
    }
    assert expected.issubset(set(comparison.columns))
