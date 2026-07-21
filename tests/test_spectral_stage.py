"""Unit tests for late-time temporal spectral analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.spectral import (  # noqa: E402
    annotate_sweep_selection,
    build_cutoff_schedule,
    compute_adjacent_training_discrepancy,
    compute_window_sensitivity_rows,
    discover_completed_pinn_windows,
    fit_reduced_sensitivity_regime,
    filter_field_after_cutoff,
    summarize_adjacent_training_discrepancy,
    symmetric_energy_lowpass,
)


def test_energy_filter_uses_conjugate_symmetric_mask() -> None:
    """The retained FFT mask must contain complete conjugate pairs."""
    n_time = 128
    dt = 0.01
    t = np.arange(n_time, dtype=np.float64) * dt
    signal = 10.0 + np.sin(2.0 * np.pi * 25.0 * t)
    u = np.vstack([signal, 2.0 * signal])

    result = symmetric_energy_lowpass(u, dt=dt, retained_energy_fraction=0.95)

    assert result.cutoff_shell == 0
    assert result.retained_bin_count == 1
    assert result.retained_energy_fraction >= 0.95
    for index in range(n_time):
        conjugate = (-index) % n_time
        assert result.retained_mask[index] == result.retained_mask[conjugate]
    assert np.allclose(result.filtered_segment, u.mean(axis=1, keepdims=True), atol=1e-12)


def test_filter_preserves_all_samples_before_cutoff() -> None:
    """The pre-cutoff field must remain bitwise unchanged."""
    dt = 0.01
    t = np.arange(101, dtype=np.float64) * dt
    u = np.vstack(
        [
            0.5 + 0.2 * np.sin(2.0 * np.pi * 3.0 * t),
            1.0 + 0.1 * np.cos(2.0 * np.pi * 7.0 * t),
        ]
    )

    filtered, _, cutoff_index = filter_field_after_cutoff(
        u,
        t,
        t_cut=0.4,
        dt=dt,
        retained_energy_fraction=0.95,
    )

    assert cutoff_index == 40
    assert np.array_equal(filtered[:, :cutoff_index], u[:, :cutoff_index])
    assert filtered.shape == u.shape


def test_cutoff_schedule_excludes_tmax() -> None:
    """Tested cutoffs must stop one increment before the window endpoint."""
    values = build_cutoff_schedule(t_max=0.5, first_cutoff_time=0.1, cutoff_increment=0.1)
    assert np.array_equal(values, np.array([0.1, 0.2, 0.3, 0.4]))


def test_inter_filter_sensitivity_compares_consecutive_cutoffs() -> None:
    """Sensitivity must compare final fields from adjacent tested cutoffs."""
    dt = 0.01
    t = np.arange(51, dtype=np.float64) * dt
    u = np.vstack(
        [
            0.5 + 0.1 * np.sin(2.0 * np.pi * 4.0 * t),
            0.8 + 0.2 * np.cos(2.0 * np.pi * 5.0 * t),
        ]
    )
    cutoffs = np.array([0.1, 0.2, 0.3])
    rows = compute_window_sensitivity_rows(
        u=u,
        t=t,
        t_max=0.5,
        dt=dt,
        cutoff_values=cutoffs,
        retained_energy_fraction=0.95,
    )

    filtered_01, _, _ = filter_field_after_cutoff(u, t, 0.1, dt, 0.95)
    filtered_02, _, _ = filter_field_after_cutoff(u, t, 0.2, dt, 0.95)
    expected = np.linalg.norm(filtered_02[:, -1] - filtered_01[:, -1])

    assert np.isnan(rows[0]["inter_filter_sensitivity"])
    assert rows[1]["previous_tcut"] == 0.1
    assert np.isclose(rows[1]["inter_filter_sensitivity"], expected)


def test_adjacent_discrepancy_is_unchanged_before_common_cutoff() -> None:
    """Raw and filtered adjacent curves must coincide before the cutoff."""
    dt = 0.1
    t = np.arange(4, dtype=np.float64) * dt
    earlier = np.vstack([t, 2.0 * t])
    later = earlier.copy()
    later[:, -1] += np.array([0.2, -0.1])

    metrics = compute_adjacent_training_discrepancy(
        earlier_field=earlier,
        earlier_time=t,
        later_field_on_common_grid=later,
        later_time_on_common_grid=t,
        t_cut=0.4,
        dt=dt,
        retained_energy_fraction=0.95,
    )

    assert not metrics["filtering_applied"]
    assert np.isclose(
        metrics["raw_adjacent_final_rmse"],
        metrics["filtered_adjacent_final_rmse"],
    )
    assert np.isclose(metrics["filtered_reduction"], 0.0)


def test_adjacent_filter_removes_high_frequency_training_difference() -> None:
    """A late high-frequency discrepancy should be reduced by filtering."""
    dt = 0.01
    t = np.arange(101, dtype=np.float64) * dt
    base = np.vstack(
        [
            np.ones_like(t),
            2.0 * np.ones_like(t),
        ]
    )
    perturbation = 0.2 * np.cos(2.0 * np.pi * 20.0 * t)
    later = base + np.vstack([perturbation, perturbation])

    metrics = compute_adjacent_training_discrepancy(
        earlier_field=base,
        earlier_time=t,
        later_field_on_common_grid=later,
        later_time_on_common_grid=t,
        t_cut=0.4,
        dt=dt,
        retained_energy_fraction=0.95,
    )

    assert metrics["filtering_applied"]
    assert metrics["raw_adjacent_final_rmse"] > 0.0
    assert (
        metrics["filtered_adjacent_final_rmse"]
        < metrics["raw_adjacent_final_rmse"]
    )
    assert metrics["filtered_reduction"] > 0.0


def test_adjacent_discrepancy_summary_reports_post_onset_reduction() -> None:
    """The summary must quantify only pairs for which filtering is active."""
    discrepancy = pd.DataFrame(
        {
            "filtering_applied": [False, True, True, True],
            "raw_adjacent_final_rmse": [1.0, 2.0, 4.0, 2.0],
            "filtered_adjacent_final_rmse": [1.0, 1.0, 2.0, 3.0],
            "filtered_to_raw_ratio": [1.0, 0.5, 0.5, 1.5],
            "filtered_reduction": [0.0, 0.5, 0.5, -0.5],
        }
    )

    summary = summarize_adjacent_training_discrepancy(discrepancy)

    assert summary["post_onset_pair_count"] == 3
    assert summary["reduced_pair_count"] == 2
    assert np.isclose(summary["reduced_pair_fraction"], 2.0 / 3.0)
    assert np.isclose(summary["median_filtered_to_raw_ratio"], 0.5)
    assert np.isclose(summary["median_filtered_reduction"], 0.5)


def test_sweep_annotation_separates_global_and_selected_pairs() -> None:
    """Global and final-time selection flags must be explicit and deterministic."""
    sweep = pd.DataFrame(
        {
            "tmax": [0.4, 0.4, 0.5, 0.5],
            "tcut": [0.2, 0.3, 0.2, 0.3],
            "inter_filter_sensitivity": [0.5, 0.2, 0.1, 0.3],
        }
    )

    annotated, global_row, selected_row = annotate_sweep_selection(
        sweep,
        selection_mode="minimum_at_final_time",
        final_time=0.5,
    )

    assert float(global_row["tmax"]) == 0.5
    assert float(global_row["tcut"]) == 0.2
    assert float(selected_row["tmax"]) == 0.5
    assert int(annotated["is_minimum_for_tmax"].sum()) == 2
    assert int(annotated["is_global_minimum"].sum()) == 1
    assert int(annotated["is_selected"].sum()) == 1


def test_reduced_sensitivity_fit_recovers_reduced_slope_breakpoint() -> None:
    """The BIC model must recover a clear reduced-slope regime."""
    tcut = np.arange(0.1, 1.1, 0.1, dtype=np.float64)
    tau = 0.5
    log_sensitivity = np.where(
        tcut < tau,
        -0.5 - 3.0 * tcut,
        (-0.5 - 3.0 * tau) - 0.2 * (tcut - tau),
    )
    log_sensitivity += np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.03, -0.02, 0.025, -0.015, 0.01]
    )
    sweep = pd.DataFrame(
        {
            "tmax": np.full(tcut.size, 5.0),
            "tcut": tcut,
            "inter_filter_sensitivity": 10.0**log_sensitivity,
        }
    )

    fit = fit_reduced_sensitivity_regime(
        sweep,
        tmax=5.0,
        minimum_points_per_segment=3,
    )

    assert fit.preferred_model == "M1"
    assert np.isclose(fit.tau, tau)
    assert fit.m1_bic < fit.m0_bic
    assert fit.m1_slope_before < 0.0
    assert abs(fit.m1_slope_after) < abs(fit.m1_slope_before)


def test_reduced_sensitivity_selection_marks_breakpoint_not_sensitivity_minimum() -> None:
    """The selected cutoff must be tau even when the minimum occurs later."""
    tcut = np.arange(0.1, 1.1, 0.1, dtype=np.float64)
    tau = 0.5
    log_sensitivity = np.where(
        tcut < tau,
        -0.5 - 3.0 * tcut,
        (-0.5 - 3.0 * tau) - 0.2 * (tcut - tau),
    )
    log_sensitivity += np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.03, -0.02, 0.025, -0.015, 0.01]
    )
    sweep = pd.DataFrame(
        {
            "tmax": np.full(tcut.size, 5.0),
            "tcut": tcut,
            "previous_tcut": np.r_[np.nan, tcut[:-1]],
            "inter_filter_sensitivity": 10.0**log_sensitivity,
        }
    )

    annotated, _, selected = annotate_sweep_selection(
        sweep,
        selection_mode="reduced_sensitivity_at_final_time",
        final_time=5.0,
        reduced_sensitivity_minimum_points_per_segment=3,
    )

    minimum_row = annotated[annotated["is_minimum_for_tmax"]].iloc[0]
    assert np.isclose(float(selected["tcut"]), tau)
    assert np.isclose(float(minimum_row["tcut"]), 0.9)
    assert bool(selected["is_reduced_sensitivity_onset"])
    assert bool(selected["is_selected"])
    assert str(selected["reduced_sensitivity_preferred_model"]) == "M1"


def test_reduced_sensitivity_selection_rejects_single_linear_trend() -> None:
    """A single trend must not be forced into a reduced-sensitivity selection."""
    tcut = np.arange(0.1, 1.1, 0.1, dtype=np.float64)
    sweep = pd.DataFrame(
        {
            "tmax": np.full(tcut.size, 5.0),
            "tcut": tcut,
            "previous_tcut": np.r_[np.nan, tcut[:-1]],
            "inter_filter_sensitivity": 10.0 ** (-0.5 - 0.8 * tcut),
        }
    )

    try:
        annotate_sweep_selection(
            sweep,
            selection_mode="reduced_sensitivity_at_final_time",
            final_time=5.0,
            reduced_sensitivity_minimum_points_per_segment=3,
        )
    except RuntimeError as error:
        assert "M0 was preferred" in str(error)
    else:
        raise AssertionError("Expected a single-trend profile to reject M1.")



def test_reduced_sensitivity_fit_accepts_small_positive_post_break_slope() -> None:
    """A reduced-sensitivity regime may drift upward after the main decay."""
    tcut = np.arange(0.1, 1.5, 0.1, dtype=np.float64)
    tau = 0.6
    log_sensitivity = np.where(
        tcut < tau,
        -0.4 - 3.2 * tcut,
        (-0.4 - 3.2 * tau) + 0.15 * (tcut - tau),
    )
    log_sensitivity += np.array(
        [
            0.0,
            -0.01,
            0.01,
            0.0,
            -0.01,
            0.0,
            0.02,
            -0.015,
            0.01,
            -0.01,
            0.015,
            -0.01,
            0.0,
            0.01,
        ]
    )
    sweep = pd.DataFrame(
        {
            "tmax": np.full(tcut.size, 5.0),
            "tcut": tcut,
            "inter_filter_sensitivity": 10.0**log_sensitivity,
        }
    )

    fit = fit_reduced_sensitivity_regime(
        sweep,
        tmax=5.0,
        minimum_points_per_segment=3,
    )

    assert fit.preferred_model == "M1"
    assert np.isclose(fit.tau, tau)
    assert fit.m1_slope_before < 0.0
    assert fit.m1_slope_after > 0.0
    assert abs(fit.m1_slope_after) < abs(fit.m1_slope_before)


def test_reduced_sensitivity_path_is_annotated_for_multiple_tmax_values() -> None:
    """The robustness path must repeat the same fit at every eligible horizon."""
    tcut = np.arange(0.1, 1.1, 0.1, dtype=np.float64)
    tau = 0.5
    rows: list[pd.DataFrame] = []
    for tmax, offset in [(4.0, 0.0), (5.0, 0.03)]:
        log_sensitivity = np.where(
            tcut < tau,
            -0.5 - 3.0 * tcut + offset,
            (-0.5 - 3.0 * tau + offset) + 0.12 * (tcut - tau),
        )
        log_sensitivity += np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.015, 0.01, -0.01, 0.005]
        )
        rows.append(
            pd.DataFrame(
                {
                    "tmax": np.full(tcut.size, tmax),
                    "tcut": tcut,
                    "previous_tcut": np.r_[np.nan, tcut[:-1]],
                    "inter_filter_sensitivity": 10.0**log_sensitivity,
                }
            )
        )
    sweep = pd.concat(rows, ignore_index=True)

    annotated, _, selected = annotate_sweep_selection(
        sweep,
        selection_mode="reduced_sensitivity_at_final_time",
        final_time=5.0,
        reduced_sensitivity_minimum_points_per_segment=3,
    )

    reduced_sensitivity_path = annotated.loc[
        annotated["is_reduced_sensitivity_onset"]
    ].sort_values("tmax")
    assert list(reduced_sensitivity_path["tmax"]) == [4.0, 5.0]
    assert np.allclose(reduced_sensitivity_path["tcut"], tau)
    assert set(reduced_sensitivity_path["reduced_sensitivity_fit_status"]) == {
        "M1_preferred"
    }
    assert np.isclose(float(selected["tmax"]), 5.0)
    assert np.isclose(float(selected["tcut"]), tau)

def _partial_pinn_config() -> dict:
    """Return a small progressive PINN configuration for artifact tests."""
    return {
        "case": {
            "length": 1.0,
            "u_left": 0.5,
            "u_right": 1.0,
            "velocity": 1.0,
            "diffusivity": 0.025,
        },
        "grid": {"n_space": 5, "dt": 0.1, "final_time": 0.5},
        "training": {
            "first_window_final_time": 0.1,
            "window_increment": 0.1,
        },
        "outputs": {
            "data_dir": "data/pinn",
            "histories_dir": "data/pinn/histories",
            "models_dir": "results/models/pinn",
            "metrics_dir": "results/metrics",
        },
    }


def _write_complete_window(root: Path, config: dict, window_index: int, tmax: float) -> None:
    """Write the minimum atomic artifact bundle used by discovery tests."""
    from pinn_spectral.benchmark import BenchmarkConfig
    from pinn_spectral.pinn import build_pinn_output_paths, build_window_artifacts

    benchmark = BenchmarkConfig.from_mapping(config["case"])
    paths = build_pinn_output_paths(config, root)
    artifacts = build_window_artifacts(
        paths=paths,
        benchmark=benchmark,
        n_space=int(config["grid"]["n_space"]),
        dt=float(config["grid"]["dt"]),
        window_final_time=tmax,
    )
    for path in [artifacts.solution, artifacts.history, artifacts.checkpoint]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    artifacts.completion.parent.mkdir(parents=True, exist_ok=True)
    artifacts.completion.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "window_index": window_index,
                "window_final_time": tmax,
            }
        ),
        encoding="utf-8",
    )


def test_completed_window_discovery_ignores_in_progress_window(tmp_path: Path) -> None:
    """Only windows with an atomic completion marker may enter a partial sweep."""
    config = _partial_pinn_config()
    _write_complete_window(tmp_path, config, 1, 0.1)
    _write_complete_window(tmp_path, config, 2, 0.2)

    # Simulate an in-progress t=0.3 solution without its final completion marker.
    from pinn_spectral.benchmark import BenchmarkConfig
    from pinn_spectral.pinn import build_pinn_output_paths, build_window_artifacts

    benchmark = BenchmarkConfig.from_mapping(config["case"])
    paths = build_pinn_output_paths(config, tmp_path)
    in_progress = build_window_artifacts(paths, benchmark, 5, 0.1, 0.3)
    in_progress.solution.parent.mkdir(parents=True, exist_ok=True)
    in_progress.solution.write_bytes(b"not-yet-complete")

    state = discover_completed_pinn_windows(config, tmp_path)

    assert np.array_equal(state.completed_schedule, np.array([0.1, 0.2]))
    assert state.processed_final_time == 0.2
    assert not state.training_complete


def test_completed_window_discovery_rejects_marker_after_gap(tmp_path: Path) -> None:
    """A later completed marker after a missing window must be rejected."""
    config = _partial_pinn_config()
    _write_complete_window(tmp_path, config, 1, 0.1)
    _write_complete_window(tmp_path, config, 3, 0.3)

    try:
        discover_completed_pinn_windows(config, tmp_path)
    except RuntimeError as error:
        assert "after an incomplete window" in str(error)
    else:
        raise AssertionError("Expected a progressive-window gap error.")
