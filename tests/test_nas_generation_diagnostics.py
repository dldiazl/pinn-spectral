"""Unit tests for generation-level NAS convergence diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.tools import (  # noqa: E402
    _dominated_hypervolume_2d,
    build_nas_generation_diagnostics,
)


def test_hypervolume_of_single_corner_point_is_full_square() -> None:
    """A front at the origin must dominate the whole unit square."""
    assert _dominated_hypervolume_2d(np.array([[0.0, 0.0]])) == 1.0
    assert _dominated_hypervolume_2d(np.array([[1.0, 1.0]])) == 0.0


def test_hypervolume_of_staircase_front() -> None:
    """A two-point staircase front must sum its dominated rectangles."""
    front = np.array([[0.0, 0.5], [0.5, 0.0]])
    assert np.isclose(_dominated_hypervolume_2d(front), 0.75)


def test_generation_diagnostics_track_cumulative_best_and_hypervolume() -> None:
    """Cumulative best loss must be monotone and hypervolume nondecreasing."""
    trials = pd.DataFrame(
        {
            "generation": [0, 0, 1, 1, 2, 2],
            "final_loss": [1e-6, 1e-7, 5e-7, 1e-8, 1e-8, 2e-6],
            "total_time_optimizer": [100.0, 400.0, 200.0, 300.0, 300.0, 50.0],
            "architecture_key": ["a", "b", "c", "d", "d", "e"],
        }
    )

    table, summary = build_nas_generation_diagnostics(
        trials, selected_architecture_key="d"
    )

    assert list(table["generation"]) == [1, 2, 3]
    assert np.allclose(table["cumulative_best_loss"], [1e-7, 1e-8, 1e-8])
    assert (table["hypervolume_increment"].to_numpy() >= 0.0).all()
    assert table["cumulative_hypervolume"].is_monotonic_increasing
    assert summary["trial_count"] == 6
    assert summary["distinct_architecture_count"] == 5
    assert summary["duplicate_evaluation_count"] == 1
    assert summary["selected_first_generation"] == 2
    assert summary["selected_evaluation_count"] == 2


def test_generation_diagnostics_reject_missing_selected_architecture() -> None:
    """An unknown selected architecture must raise a clear error."""
    trials = pd.DataFrame(
        {
            "generation": [0, 0],
            "final_loss": [1e-6, 1e-7],
            "total_time_optimizer": [100.0, 200.0],
            "architecture_key": ["a", "b"],
        }
    )
    try:
        build_nas_generation_diagnostics(trials, selected_architecture_key="zz")
    except ValueError as error:
        assert "does not" in str(error)
    else:
        raise AssertionError("Expected a missing-architecture error.")
