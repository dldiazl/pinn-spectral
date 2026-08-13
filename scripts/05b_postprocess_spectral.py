"""Postprocess spectral reconstructions against the analytical reference.

Interval selection remains reference-free in stage 05a. This stage validates
the BIC-selected reduced-sensitivity cutoff and computes the article error
curves for that pair,
builds a full-window RMSE map, and evaluates temporal-stability metrics for
every tested ``(tcut, tmax)`` pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.benchmark import BenchmarkConfig  # noqa: E402
from pinn_spectral.spectral import (  # noqa: E402
    annotate_metric_minimum_paths,
    annotate_reference_window_sweep,
    build_metric_path_comparison,
    build_spectral_output_paths,
    compute_adjacent_training_discrepancy,
    compute_temporal_stability_metrics,
    compute_window_reference_metrics,
    filter_field_after_cutoff,
    read_generation_metadata,
    select_minimum_reference_rmse_at_tmax,
    summarize_adjacent_training_discrepancy,
)
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    assert_same_grid,
    compute_error_space,
    compute_error_time,
    output_name,
    print_skip_message,
    read_solution_matrix,
    read_yaml,
    reject_partial_outputs,
    require_file,
    save_csv,
    save_json,
    should_skip,
)


METHOD_LABELS = {
    "CDS_EF": "CDS-EF",
    "CDS_CN": "CDS-CN",
    "CompactSchemes_CN": "Compact-CN",
}

TEMPORAL_STABILITY_METRICS = [
    "temporal_std_rms",
    "temporal_std_per_duration",
    "temporal_derivative_rms",
]

TEMPORAL_STABILITY_PATH_FLAGS = {
    "sensitivity": "is_minimum_sensitivity_for_tmax",
    "temporal_std": "is_minimum_temporal_std_rms_for_tmax",
    "temporal_std_per_duration": (
        "is_minimum_temporal_std_per_duration_for_tmax"
    ),
    "temporal_derivative": "is_minimum_temporal_derivative_rms_for_tmax",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Postprocess spectral-filter diagnostics.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/spectral.yaml",
        help="Path to the spectral-analysis configuration YAML file.",
    )
    return parser.parse_args()


def load_config(path_value: str | Path) -> dict[str, Any]:
    """Load one project-relative YAML configuration."""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return read_yaml(path)


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def selected_pinn_solution_path(
    pinn_config: dict[str, Any],
    tmax: float,
) -> Path:
    """Return the progressive PINN solution for one trained window."""
    benchmark = BenchmarkConfig.from_mapping(pinn_config["case"])
    grid = pinn_config["grid"]
    data_dir = resolve_path(pinn_config.get("outputs", {}).get("data_dir", "data/pinn"))
    stem = output_name(
        "PINNs",
        int(grid["n_space"]),
        float(grid["dt"]),
        float(tmax),
        benchmark.peclet,
    )
    return data_dir / f"{stem}.parquet"


def reference_solution_path(
    reference_config: dict[str, Any],
    tmax: float,
) -> Path:
    """Return the full reference path; temporal slicing is performed after reading."""
    benchmark = BenchmarkConfig.from_mapping(reference_config["case"])
    grid = reference_config["grid"]
    stem = output_name(
        "Analytical",
        int(grid["n_space"]),
        float(grid["dt"]),
        float(grid["final_time"]),
        benchmark.peclet,
    )
    data_dir = resolve_path(reference_config.get("outputs", {}).get("data_dir", "data/reference"))
    return data_dir / f"{stem}.parquet"


def numerical_solution_path(
    numerical_config: dict[str, Any],
    method: str,
) -> Path:
    """Return the full-domain finite-difference solution path for one method."""
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown finite-difference method: {method}")
    benchmark = BenchmarkConfig.from_mapping(numerical_config["case"])
    grid = numerical_config["grid"]
    data_dir = resolve_path(
        numerical_config.get("outputs", {}).get("data_dir", "data/numerical")
    )
    stem = output_name(
        method,
        int(grid["n_space"]),
        float(grid["dt"]),
        float(grid["final_time"]),
        benchmark.peclet,
    )
    return data_dir / f"{stem}.parquet"


def slice_solution_to_tmax(
    x_full: np.ndarray,
    t_full: np.ndarray,
    u_full: np.ndarray,
    tmax: float,
    dt: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a full-domain solution at an exact progressive-window endpoint."""
    index = int(np.argmin(np.abs(t_full - float(tmax))))
    if abs(float(t_full[index]) - float(tmax)) > max(1e-12, dt * 1e-7):
        raise ValueError(
            f"Selected tmax={tmax:.3f} is not present in the {label} time grid."
        )
    return x_full, t_full[: index + 1], u_full[:, : index + 1]


def build_reference_window_sweep(
    sweep: pd.DataFrame,
    pinn_config: dict[str, Any],
    x_ref_full: np.ndarray,
    t_ref_full: np.ndarray,
    u_ref_full: np.ndarray,
    retained_energy_fraction: float,
) -> pd.DataFrame:
    """Compute normalized reference errors on each complete filter window.

    For every tested pair, the filtered, raw PINN, and analytical fields are
    compared only on ``[tcut, tmax]``. MSE and RMSE therefore use exactly the
    samples belonging to that filter window and are normalized by ``nx * nt``.
    """
    required = {"tmax", "tcut", "inter_filter_sensitivity"}
    missing = required.difference(sweep.columns)
    if missing:
        raise ValueError(f"Spectral sweep is missing columns: {sorted(missing)}")

    dt = float(pinn_config["grid"]["dt"])
    rows: list[dict[str, Any]] = []
    for tmax, group in sweep.groupby("tmax", sort=True):
        tmax_value = float(tmax)
        pinn_path = selected_pinn_solution_path(pinn_config, tmax_value)
        require_file(pinn_path, "Run: python scripts\\04a_run_pinn.py")
        x_pinn, t_pinn, u_pinn = read_solution_matrix(pinn_path)
        x_ref, t_ref, u_ref = slice_solution_to_tmax(
            x_ref_full,
            t_ref_full,
            u_ref_full,
            tmax_value,
            dt,
            "reference",
        )
        assert_same_grid(x_ref, t_ref, x_pinn, t_pinn, f"PINN window tmax={tmax_value:.3f}")

        for sweep_row in group.sort_values("tcut").itertuples(index=False):
            tcut_value = float(sweep_row.tcut)
            filtered, result, cutoff_index = filter_field_after_cutoff(
                u=u_pinn,
                t=t_pinn,
                t_cut=tcut_value,
                dt=dt,
                retained_energy_fraction=retained_energy_fraction,
            )
            filtered_segment = filtered[:, cutoff_index:]
            raw_segment = u_pinn[:, cutoff_index:]
            reference_segment = u_ref[:, cutoff_index:]
            window_duration = float(t_pinn[-1] - t_pinn[cutoff_index])
            metrics = compute_window_reference_metrics(
                filtered_segment=filtered_segment,
                raw_segment=raw_segment,
                reference_segment=reference_segment,
            )
            temporal_stability_metrics = compute_temporal_stability_metrics(
                filtered_segment=filtered_segment,
                dt=dt,
                window_duration=window_duration,
            )
            rows.append(
                {
                    "tmax": tmax_value,
                    "tcut": tcut_value,
                    "nx": int(filtered_segment.shape[0]),
                    "nt": int(filtered_segment.shape[1]),
                    "sample_count": int(filtered_segment.size),
                    "window_duration": window_duration,
                    "cutoff_index": int(cutoff_index),
                    "cutoff_frequency": float(result.cutoff_frequency),
                    "cutoff_shell": int(result.cutoff_shell),
                    "inter_filter_sensitivity": float(sweep_row.inter_filter_sensitivity),
                    **temporal_stability_metrics,
                    **metrics,
                }
            )
        print(
            "Computed reference-window sweep: "
            f"tmax={tmax_value:.3f}, tested_cutoffs={len(group)}"
        )

    reference_sweep = pd.DataFrame(rows)
    annotated = annotate_reference_window_sweep(reference_sweep, sweep)
    return annotate_metric_minimum_paths(
        annotated,
        TEMPORAL_STABILITY_METRICS,
    )


def build_adjacent_training_discrepancy(
    pinn_config: dict[str, Any],
    generation_metadata: dict[str, Any],
    tcut_common: float,
    retained_energy_fraction: float,
) -> pd.DataFrame:
    """Compare raw and filtered fields from adjacent trained horizons.

    For each consecutive pair ``(T_k, T_{k+1})``, the longer-horizon PINN is
    first restricted to ``[0, T_k]``. Both fields are then compared at
    ``t=T_k``. Before the common cutoff the filtered curve is exactly equal to
    the raw curve; after it, both common-grid fields are filtered independently
    on ``[tcut_common, T_k]`` using the same retained-energy rule.
    """
    schedule_values = generation_metadata.get("sweep", {}).get(
        "processed_window_schedule", []
    )
    schedule = np.asarray(schedule_values, dtype=np.float64)
    if schedule.ndim != 1 or schedule.size < 2:
        raise ValueError(
            "Adjacent-training discrepancy requires at least two completed "
            "progressive PINN windows."
        )
    if not np.all(np.isfinite(schedule)) or np.any(np.diff(schedule) <= 0.0):
        raise ValueError(
            "Processed progressive-window endpoints must be finite and strictly increasing."
        )

    dt = float(pinn_config["grid"]["dt"])
    rows: list[dict[str, Any]] = []
    for earlier_tmax, later_tmax in zip(schedule[:-1], schedule[1:]):
        earlier_path = selected_pinn_solution_path(pinn_config, float(earlier_tmax))
        later_path = selected_pinn_solution_path(pinn_config, float(later_tmax))
        require_file(earlier_path, "Run: python scripts\\04a_run_pinn.py")
        require_file(later_path, "Run: python scripts\\04a_run_pinn.py")

        x_earlier, t_earlier, u_earlier = read_solution_matrix(earlier_path)
        x_later_full, t_later_full, u_later_full = read_solution_matrix(later_path)
        x_later, t_later, u_later = slice_solution_to_tmax(
            x_later_full,
            t_later_full,
            u_later_full,
            float(earlier_tmax),
            dt,
            f"later PINN window tmax={float(later_tmax):.3f}",
        )
        assert_same_grid(
            x_earlier,
            t_earlier,
            x_later,
            t_later,
            (
                "adjacent progressive PINNs "
                f"({float(earlier_tmax):.3f}, {float(later_tmax):.3f})"
            ),
        )

        metrics = compute_adjacent_training_discrepancy(
            earlier_field=u_earlier,
            earlier_time=t_earlier,
            later_field_on_common_grid=u_later,
            later_time_on_common_grid=t_later,
            t_cut=float(tcut_common),
            dt=dt,
            retained_energy_fraction=float(retained_energy_fraction),
        )
        rows.append(
            {
                "tmax_earlier": float(earlier_tmax),
                "tmax_later": float(later_tmax),
                "training_increment": float(later_tmax - earlier_tmax),
                "source_earlier": str(earlier_path.relative_to(ROOT)),
                "source_later": str(later_path.relative_to(ROOT)),
                **metrics,
            }
        )
        print(
            "Computed adjacent-training discrepancy: "
            f"T_k={float(earlier_tmax):.3f}, "
            f"T_(k+1)={float(later_tmax):.3f}, "
            f"raw={metrics['raw_adjacent_final_rmse']:.6e}, "
            f"filtered={metrics['filtered_adjacent_final_rmse']:.6e}"
        )

    return pd.DataFrame(rows)


def build_preprocessing_sensitivity(
    u_pinn: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    tcut: float,
    dt: float,
    q_values: tuple[float, ...] = (0.90, 0.95, 0.99),
    curve_q: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare raw and mean-removed spectrum preprocessing at the selected interval.

    Every combination of ``spectrum_source`` and retained-energy fraction in
    ``q_values`` is applied at the same selected cutoff, and each
    reconstruction is scored against the reference through the window RMSE on
    ``[tcut, tmax]`` and the final-time spatial error norm. The returned
    curves table contains the full error-time histories of the unfiltered
    field and of the two reconstructions at ``curve_q``, and the diagnostics
    dictionary summarizes the spectral composition that explains the outcome:
    the zero-frequency energy fraction of the raw segment, the maximum
    difference between the raw reconstruction and the per-node temporal mean
    of the segment, and the fluctuation-energy fraction of the first nonzero
    frequency pair.
    """
    window_mask = t >= float(tcut) - max(1e-12, float(dt) * 1e-7)
    window_sample_count = int(np.count_nonzero(window_mask))
    frequency_resolution = float(1.0 / (window_sample_count * float(dt)))

    def window_rmse(u_field: np.ndarray) -> float:
        difference = u_field[:, window_mask] - u_ref[:, window_mask]
        return float(np.sqrt(np.mean(difference**2)))

    error_unfiltered = compute_error_time(u_pinn, u_ref)
    final_unfiltered = float(error_unfiltered[-1])
    rows: list[dict[str, Any]] = [
        {
            "spectrum_source": "unfiltered",
            "retained_energy_fraction_target": np.nan,
            "achieved_energy_fraction": np.nan,
            "cutoff_frequency": np.nan,
            "cutoff_shell": np.nan,
            "retained_bin_count": np.nan,
            "window_rmse": window_rmse(u_pinn),
            "final_error_time": final_unfiltered,
            "final_error_reduction_factor": 1.0,
        }
    ]
    curves: list[pd.DataFrame] = [
        pd.DataFrame(
            {
                "method": "PINN",
                "label": "PINN",
                "t": t,
                "error_l2": error_unfiltered,
            }
        )
    ]
    curve_labels = {
        "raw_pinn_field": (
            "PINN_Filtered",
            "PINN (filtered, reduced-sensitivity cutoff)",
        ),
        "mean_removed_field": (
            "PINN_Filtered_Mean_Removed",
            "PINN (filtered, mean-removed spectrum)",
        ),
    }
    diagnostics: dict[str, Any] = {
        "tcut": float(tcut),
        "window_sample_count": window_sample_count,
        "frequency_resolution": frequency_resolution,
        "q_values": [float(value) for value in q_values],
        "curve_q": float(curve_q),
    }
    for source in ("raw_pinn_field", "mean_removed_field"):
        for q_value in q_values:
            filtered, result, cutoff_index = filter_field_after_cutoff(
                u=u_pinn,
                t=t,
                t_cut=float(tcut),
                dt=float(dt),
                retained_energy_fraction=float(q_value),
                spectrum_source=source,
            )
            error_filtered = compute_error_time(filtered, u_ref)
            final_filtered = float(error_filtered[-1])
            rows.append(
                {
                    "spectrum_source": source,
                    "retained_energy_fraction_target": float(q_value),
                    "achieved_energy_fraction": float(result.retained_energy_fraction),
                    "cutoff_frequency": float(result.cutoff_frequency),
                    "cutoff_shell": int(result.cutoff_shell),
                    "retained_bin_count": int(result.retained_bin_count),
                    "window_rmse": window_rmse(filtered),
                    "final_error_time": final_filtered,
                    "final_error_reduction_factor": (
                        float(final_unfiltered / final_filtered)
                        if final_filtered > 0.0
                        else float("inf")
                    ),
                }
            )
            if np.isclose(float(q_value), float(curve_q), rtol=0.0, atol=1e-12):
                method, label = curve_labels[source]
                curves.append(
                    pd.DataFrame(
                        {
                            "method": method,
                            "label": label,
                            "t": t,
                            "error_l2": error_filtered,
                        }
                    )
                )
                segment = u_pinn[:, cutoff_index:]
                if source == "raw_pinn_field":
                    total_energy = float(np.sum(result.spectral_energy))
                    diagnostics["zero_frequency_energy_fraction"] = float(
                        result.spectral_energy[0] / total_energy
                    )
                    diagnostics["mean_projection_max_abs_difference"] = float(
                        np.max(
                            np.abs(
                                result.filtered_segment
                                - segment.mean(axis=1, keepdims=True)
                            )
                        )
                    )
                else:
                    fluctuation_energy = result.spectral_energy
                    total_fluctuation = float(np.sum(fluctuation_energy))
                    first_pair = float(
                        fluctuation_energy[1] + fluctuation_energy[-1]
                    )
                    diagnostics["first_shell_fluctuation_energy_fraction"] = float(
                        first_pair / total_fluctuation
                    )

    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True), diagnostics


def main() -> None:
    """Generate spectral postprocess tables and metadata."""
    args = parse_args()
    config = load_config(args.config)
    inputs = config.get("inputs", {})
    pinn_config = load_config(inputs.get("pinn_config", "configs/pinn.yaml"))
    reference_config = load_config(inputs.get("reference_config", "configs/reference.yaml"))
    numerical_config = load_config(inputs.get("numerical_config", "configs/numerical.yaml"))
    paths = build_spectral_output_paths(config, ROOT)

    validation = config.get("reference_validation", {})
    validation_enabled = bool(validation.get("enabled", True))
    validation_metric = str(validation.get("metric", "rmse_window")).strip().lower()
    if validation_enabled and validation_metric != "rmse_window":
        raise ValueError("reference_validation.metric must be 'rmse_window'.")

    temporal_stability_config = config.get("temporal_stability", {})
    temporal_stability_enabled = bool(
        temporal_stability_config.get("enabled", True)
    )
    if not bool(
        temporal_stability_config.get("include_all_spatial_nodes", True)
    ):
        raise ValueError(
            "temporal_stability.include_all_spatial_nodes must remain true for "
            "the current all-node metric definition."
        )

    sensitivity_csv = paths.postprocess_dir / "spectral_sensitivity.csv"
    time_csv = paths.postprocess_dir / "spectral_error_time.csv"
    rmse_path_time_csv = paths.postprocess_dir / "spectral_error_time_rmse_path.csv"
    space_csv = paths.postprocess_dir / "spectral_error_space.csv"
    spectrum_csv = paths.postprocess_dir / "selected_spectrum.csv"
    adjacent_training_csv = (
        paths.postprocess_dir / "spectral_adjacent_training_discrepancy.csv"
    )
    preprocessing_table_csv = (
        paths.postprocess_dir / "spectral_preprocessing_sensitivity.csv"
    )
    preprocessing_time_csv = (
        paths.postprocess_dir / "spectral_error_time_preprocessing.csv"
    )
    legacy_esens_samples_csv = (
        paths.postprocess_dir / "spectral_error_time_esens_path_samples.csv"
    )
    if args.overwrite:
        legacy_esens_samples_csv.unlink(missing_ok=True)
    expected = [
        sensitivity_csv,
        time_csv,
        space_csv,
        spectrum_csv,
        adjacent_training_csv,
        preprocessing_table_csv,
        preprocessing_time_csv,
        paths.postprocess_metadata,
    ]
    if validation_enabled:
        expected.extend([paths.reference_window_sweep, rmse_path_time_csv])
    if temporal_stability_enabled:
        expected.extend(
            [paths.temporal_stability_sweep, paths.metric_path_comparison]
        )
    reject_partial_outputs(expected, overwrite=args.overwrite)
    if should_skip(expected, overwrite=args.overwrite):
        print_skip_message(expected, ROOT)
        return

    require_file(paths.sweep, "Run: python scripts\\05a_generate_spectral_data.py --overwrite")
    require_file(paths.filtered_solution, "Run: python scripts\\05a_generate_spectral_data.py --overwrite")
    require_file(paths.spectrum, "Run: python scripts\\05a_generate_spectral_data.py --overwrite")
    require_file(paths.generation_metadata, "Run: python scripts\\05a_generate_spectral_data.py --overwrite")
    generation_metadata = read_generation_metadata(paths.generation_metadata)
    selected = generation_metadata["selected_pair"]
    selected_tmax = float(selected["tmax"])
    selected_tcut = float(selected["tcut"])
    status = str(generation_metadata.get("status", "complete"))
    training_complete = bool(generation_metadata.get("training_complete", status == "complete"))
    configured_final_time = float(generation_metadata["grid"]["configured_final_time"])
    processed_final_time = float(generation_metadata["grid"]["processed_final_time"])

    pinn_path = selected_pinn_solution_path(pinn_config, selected_tmax)
    reference_path = reference_solution_path(reference_config, selected_tmax)
    require_file(pinn_path, "Run: python scripts\\04a_run_pinn.py")
    require_file(reference_path, "Run: python scripts\\01a_generate_reference_data.py")

    x_pinn, t_pinn, u_pinn = read_solution_matrix(pinn_path)
    x_filtered, t_filtered, u_filtered = read_solution_matrix(paths.filtered_solution)
    assert_same_grid(x_pinn, t_pinn, x_filtered, t_filtered, "filtered PINN")

    x_ref_full, t_ref_full, u_ref_full = read_solution_matrix(reference_path)
    dt = float(pinn_config["grid"]["dt"])
    x_ref, t_ref, u_ref = slice_solution_to_tmax(
        x_ref_full,
        t_ref_full,
        u_ref_full,
        selected_tmax,
        dt,
        "reference",
    )
    assert_same_grid(x_ref, t_ref, x_pinn, t_pinn, "selected PINN window")

    error_unfiltered_time = compute_error_time(u_pinn, u_ref)
    error_filtered_time = compute_error_time(u_filtered, u_ref)
    error_unfiltered_space = compute_error_space(u_pinn, u_ref)
    error_filtered_space = compute_error_space(u_filtered, u_ref)

    numerical_postprocess_dir = resolve_path(
        numerical_config.get("outputs", {}).get("postprocess_dir", "results/postprocess/numerical")
    )
    fdm_time_path = numerical_postprocess_dir / "fdm_error_time.csv"
    require_file(fdm_time_path, "Run: python scripts\\02b_postprocess_numerical.py --overwrite")
    fdm_time = pd.read_csv(fdm_time_path)
    fdm_time = fdm_time[fdm_time["t"] <= selected_tmax + max(1e-12, dt * 1e-7)].copy()

    time_df = pd.concat(
        [
            fdm_time,
            pd.DataFrame(
                {
                    "method": "PINN",
                    "label": "PINN",
                    "t": t_ref,
                    "error_l2": error_unfiltered_time,
                }
            ),
            pd.DataFrame(
                {
                    "method": "PINN_Filtered",
                    "label": "PINN (filtered, reduced-sensitivity cutoff)",
                    "t": t_ref,
                    "error_l2": error_filtered_time,
                }
            ),
        ],
        ignore_index=True,
    )

    space_parts: list[pd.DataFrame] = []
    numerical_sources: dict[str, str] = {}
    numerical_methods = list(numerical_config.get("methods", METHOD_LABELS.keys()))
    for method in numerical_methods:
        method_path = numerical_solution_path(numerical_config, str(method))
        require_file(
            method_path,
            "Run: python scripts\\02a_generate_numerical_data.py --overwrite",
        )
        x_method_full, t_method_full, u_method_full = read_solution_matrix(method_path)
        x_method, t_method, u_method = slice_solution_to_tmax(
            x_method_full,
            t_method_full,
            u_method_full,
            selected_tmax,
            dt,
            str(method),
        )
        assert_same_grid(x_ref, t_ref, x_method, t_method, str(method))
        space_parts.append(
            pd.DataFrame(
                {
                    "method": str(method),
                    "label": METHOD_LABELS[str(method)],
                    "x": x_ref,
                    "error_l2": compute_error_space(u_method, u_ref),
                }
            )
        )
        numerical_sources[str(method)] = str(method_path.relative_to(ROOT))

    space_parts.extend(
        [
            pd.DataFrame(
                {
                    "method": "PINN",
                    "label": "PINN",
                    "x": x_ref,
                    "error_l2": error_unfiltered_space,
                }
            ),
            pd.DataFrame(
                {
                    "method": "PINN_Filtered",
                    "label": "PINN (filtered, reduced-sensitivity cutoff)",
                    "x": x_ref,
                    "error_l2": error_filtered_space,
                }
            ),
        ]
    )
    space_df = pd.concat(space_parts, ignore_index=True)

    sweep = pd.read_parquet(paths.sweep)
    spectrum = pd.read_parquet(paths.spectrum)
    reference_window_sweep: pd.DataFrame | None = None
    temporal_stability_sweep: pd.DataFrame | None = None
    metric_path_comparison: pd.DataFrame | None = None
    rmse_path_time_df: pd.DataFrame | None = None
    rmse_path_details: dict[str, Any] | None = None
    q = float(config.get("analysis", {}).get("retained_energy_fraction", 0.95))
    adjacent_training_df = build_adjacent_training_discrepancy(
        pinn_config=pinn_config,
        generation_metadata=generation_metadata,
        tcut_common=selected_tcut,
        retained_energy_fraction=q,
    )
    adjacent_training_summary = summarize_adjacent_training_discrepancy(
        adjacent_training_df
    )
    if validation_enabled or temporal_stability_enabled:
        reference_window_sweep = build_reference_window_sweep(
            sweep=sweep,
            pinn_config=pinn_config,
            x_ref_full=x_ref_full,
            t_ref_full=t_ref_full,
            u_ref_full=u_ref_full,
            retained_energy_fraction=q,
        )

        if temporal_stability_enabled:
            stability_columns = [
                "tmax",
                "tcut",
                "nx",
                "nt",
                "sample_count",
                "window_duration",
                "cutoff_index",
                "cutoff_frequency",
                "cutoff_shell",
                "inter_filter_sensitivity",
                "temporal_std_rms",
                "temporal_std_per_duration",
                "temporal_derivative_rms",
                "boundary_temporal_std_rms",
                "interior_temporal_std_rms",
                "boundary_temporal_derivative_rms",
                "interior_temporal_derivative_rms",
                "is_minimum_sensitivity_for_tmax",
                "is_minimum_reference_rmse_for_tmax",
                "is_minimum_temporal_std_rms_for_tmax",
                "is_global_temporal_std_rms",
                "is_minimum_temporal_std_per_duration_for_tmax",
                "is_global_temporal_std_per_duration",
                "is_minimum_temporal_derivative_rms_for_tmax",
                "is_global_temporal_derivative_rms",
            ]
            temporal_stability_sweep = reference_window_sweep[stability_columns].copy()
            metric_path_comparison = build_metric_path_comparison(
                reference_window_sweep,
                TEMPORAL_STABILITY_PATH_FLAGS,
            )

    if validation_enabled:
        if reference_window_sweep is None:
            raise RuntimeError("Reference-window sweep was not generated.")

        # Use the reference-RMSE path at the same tmax as Figure 12 so the
        # additional curve changes only the cutoff-selection criterion.
        rmse_path_row = select_minimum_reference_rmse_at_tmax(
            reference_window_sweep,
            selected_tmax,
        )
        rmse_path_tcut = float(rmse_path_row["tcut"])
        u_rmse_path_filtered, rmse_filter_result, rmse_cutoff_index = (
            filter_field_after_cutoff(
                u=u_pinn,
                t=t_pinn,
                t_cut=rmse_path_tcut,
                dt=dt,
                retained_energy_fraction=q,
            )
        )
        error_rmse_path_time = compute_error_time(u_rmse_path_filtered, u_ref)
        rmse_path_time_df = pd.concat(
            [
                fdm_time,
                pd.DataFrame(
                    {
                        "method": "PINN",
                        "label": "PINN",
                        "t": t_ref,
                        "error_l2": error_unfiltered_time,
                    }
                ),
                pd.DataFrame(
                    {
                        "method": "PINN_Filtered_RMSE_Path",
                        "label": "PINN (filtered, minimum RMSE path)",
                        "t": t_ref,
                        "error_l2": error_rmse_path_time,
                    }
                ),
            ],
            ignore_index=True,
        )
        rmse_path_details = {
            "tmax": selected_tmax,
            "tcut": rmse_path_tcut,
            "filtered_window_mse": float(rmse_path_row["filtered_window_mse"]),
            "filtered_window_rmse": float(rmse_path_row["filtered_window_rmse"]),
            "raw_window_mse": float(rmse_path_row["raw_window_mse"]),
            "raw_window_rmse": float(rmse_path_row["raw_window_rmse"]),
            "rmse_ratio": float(rmse_path_row["rmse_ratio"]),
            "rmse_reduction": float(rmse_path_row["rmse_reduction"]),
            "cutoff_index": int(rmse_cutoff_index),
            "cutoff_frequency": float(rmse_filter_result.cutoff_frequency),
            "cutoff_shell": int(rmse_filter_result.cutoff_shell),
            "final_error_time": float(error_rmse_path_time[-1]),
        }

    preprocessing_table, preprocessing_time_df, preprocessing_diagnostics = (
        build_preprocessing_sensitivity(
            u_pinn=u_pinn,
            t=t_ref,
            u_ref=u_ref,
            tcut=selected_tcut,
            dt=dt,
        )
    )

    paths.postprocess_dir.mkdir(parents=True, exist_ok=True)
    save_csv(sweep, sensitivity_csv)
    save_csv(time_df, time_csv)
    save_csv(space_df, space_csv)
    save_csv(spectrum, spectrum_csv)
    save_csv(adjacent_training_df, adjacent_training_csv)
    save_csv(preprocessing_table, preprocessing_table_csv)
    save_csv(preprocessing_time_df, preprocessing_time_csv)
    if reference_window_sweep is not None:
        save_csv(reference_window_sweep, paths.reference_window_sweep)
    if rmse_path_time_df is not None:
        save_csv(rmse_path_time_df, rmse_path_time_csv)
    if temporal_stability_sweep is not None:
        save_csv(temporal_stability_sweep, paths.temporal_stability_sweep)
    if metric_path_comparison is not None:
        save_csv(metric_path_comparison, paths.metric_path_comparison)

    final_unfiltered = float(error_unfiltered_time[-1])
    final_filtered = float(error_filtered_time[-1])
    reduction_factor = float(final_unfiltered / final_filtered) if final_filtered > 0.0 else float("inf")
    metadata = {
        "stage": "spectral_postprocess",
        "status": status,
        "training_complete": training_complete,
        "selection": selected,
        "reduced_sensitivity_analysis": generation_metadata.get("reduced_sensitivity_analysis"),
        "reduced_sensitivity_horizon_validation": generation_metadata.get(
            "reduced_sensitivity_horizon_validation"
        ),
        "reference": str(reference_path.relative_to(ROOT)),
        "unfiltered_pinn": str(pinn_path.relative_to(ROOT)),
        "filtered_pinn": str(paths.filtered_solution.relative_to(ROOT)),
        "numerical_solutions": numerical_sources,
        "space_error_methods": [
            *[str(method) for method in numerical_methods],
            "PINN",
            "PINN_Filtered",
        ],
        "error_definition": {
            "E_t": "unnormalized spatial L2 norm at each time",
            "E_x": "unnormalized time-aggregated L2 norm at each spatial node",
            "E_sens": "unnormalized L2 distance between consecutive-cutoff final spatial fields",
            "E_adjacent_raw": (
                "spatial RMSE at t=T_k between PINNs trained through adjacent "
                "horizons T_k and T_(k+1), with the later solution restricted "
                "to the common grid [0, T_k]"
            ),
            "E_adjacent_filtered": (
                "the same adjacent-horizon spatial RMSE after filtering both "
                "common-grid fields on [tcut_RS, T_k]; it is set exactly "
                "equal to E_adjacent_raw when T_k <= tcut_RS"
            ),
            "RMSE_window": (
                "sqrt(mean((u_filtered-u_reference)^2)) over the complete "
                "filter window [tcut, tmax]"
            ),
            "temporal_std_rms": (
                "RMS across all spatial nodes of the temporal population "
                "standard deviation within [tcut, tmax]"
            ),
            "temporal_std_per_duration": (
                "temporal_std_rms divided by (tmax-tcut)"
            ),
            "temporal_derivative_rms": (
                "RMS forward-difference du_filtered/dt over all spatial nodes "
                "and times within [tcut, tmax]"
            ),
        },
        "temporal_stability": {
            "enabled": temporal_stability_enabled,
            "reference_used_by_metrics": False,
            "spatial_nodes": "all nodes, including both boundaries",
            "standard_deviation_ddof": 0,
            "duration_normalization": "divide temporal_std_rms by tmax-tcut",
            "temporal_derivative": "forward difference divided by dt",
            "output": (
                str(paths.temporal_stability_sweep.relative_to(ROOT))
                if temporal_stability_enabled
                else None
            ),
            "path_comparison_output": (
                str(paths.metric_path_comparison.relative_to(ROOT))
                if temporal_stability_enabled
                else None
            ),
        },
        "adjacent_training_consistency": {
            "reference_used": False,
            "comparison": "consecutive progressive PINN horizons",
            "common_time": "t=T_k, the earlier endpoint of each adjacent pair",
            "later_solution_policy": "restricted to [0, T_k] before comparison",
            "common_cutoff": selected_tcut,
            "pre_cutoff_policy": (
                "filtered discrepancy equals raw discrepancy exactly when "
                "T_k <= tcut_RS"
            ),
            "post_cutoff_policy": (
                "both common-grid fields are filtered independently on the "
                "same interval [tcut_RS, T_k]"
            ),
            "pair_count": int(len(adjacent_training_df)),
            "filtered_pair_count": int(
                adjacent_training_df["filtering_applied"].astype(bool).sum()
            ),
            "post_onset_summary": adjacent_training_summary,
            "maximum_pre_cutoff_curve_difference": float(
                np.max(
                    np.abs(
                        adjacent_training_df.loc[
                            ~adjacent_training_df["filtering_applied"].astype(bool),
                            "raw_adjacent_final_rmse",
                        ].to_numpy(dtype=float)
                        - adjacent_training_df.loc[
                            ~adjacent_training_df["filtering_applied"].astype(bool),
                            "filtered_adjacent_final_rmse",
                        ].to_numpy(dtype=float)
                    )
                )
            )
            if (
                ~adjacent_training_df["filtering_applied"].astype(bool)
            ).any()
            else 0.0,
            "output": str(adjacent_training_csv.relative_to(ROOT)),
        },
        "reference_validation": {
            "enabled": validation_enabled,
            "metric": validation_metric,
            "reference_used_for_selection": False,
            "comparison_domain": "[tcut, tmax] for every tested pair",
            "normalization": "mean over nx * nt window samples before square root",
            "output": (
                str(paths.reference_window_sweep.relative_to(ROOT))
                if validation_enabled
                else None
            ),
        },
        "preprocessing_sensitivity": {
            **preprocessing_diagnostics,
            "tmax": selected_tmax,
            "reference_used_for_selection": False,
            "output": str(preprocessing_table_csv.relative_to(ROOT)),
            "time_error_output": str(preprocessing_time_csv.relative_to(ROOT)),
        },
        "pre_cutoff_identity": {
            "tcut": selected_tcut,
            "maximum_absolute_difference": float(
                np.max(np.abs(u_filtered[:, t_ref < selected_tcut] - u_pinn[:, t_ref < selected_tcut]))
            )
            if np.any(t_ref < selected_tcut)
            else 0.0,
        },
        "summary": {
            "configured_final_time": configured_final_time,
            "processed_final_time": processed_final_time,
            "selected_tmax": selected_tmax,
            "selected_tcut": selected_tcut,
            "unfiltered_final_error_time": final_unfiltered,
            "filtered_final_error_time": final_filtered,
            "final_error_reduction_factor": reduction_factor,
            "unfiltered_global_l2_error": float(np.linalg.norm(u_pinn - u_ref)),
            "filtered_global_l2_error": float(np.linalg.norm(u_filtered - u_ref)),
            "maximum_unfiltered_error_time": float(np.max(error_unfiltered_time)),
            "maximum_filtered_error_time": float(np.max(error_filtered_time)),
        },
        "outputs": {
            "sensitivity_csv": str(sensitivity_csv.relative_to(ROOT)),
            "time_error_csv": str(time_csv.relative_to(ROOT)),
            "space_error_csv": str(space_csv.relative_to(ROOT)),
            "spectrum_csv": str(spectrum_csv.relative_to(ROOT)),
            "adjacent_training_discrepancy_csv": str(
                adjacent_training_csv.relative_to(ROOT)
            ),
            "preprocessing_sensitivity_csv": str(
                preprocessing_table_csv.relative_to(ROOT)
            ),
            "preprocessing_time_error_csv": str(
                preprocessing_time_csv.relative_to(ROOT)
            ),
            "reference_window_sweep_csv": (
                str(paths.reference_window_sweep.relative_to(ROOT))
                if validation_enabled
                else None
            ),
            "rmse_path_time_error_csv": (
                str(rmse_path_time_csv.relative_to(ROOT))
                if validation_enabled
                else None
            ),
            "temporal_stability_sweep_csv": (
                str(paths.temporal_stability_sweep.relative_to(ROOT))
                if temporal_stability_enabled
                else None
            ),
            "metric_path_comparison_csv": (
                str(paths.metric_path_comparison.relative_to(ROOT))
                if temporal_stability_enabled
                else None
            ),
        },
    }
    if rmse_path_details is not None:
        metadata["reference_rmse_path_at_selected_tmax"] = rmse_path_details
        metadata["summary"].update(
            {
                "rmse_path_tmax": float(rmse_path_details["tmax"]),
                "rmse_path_tcut": float(rmse_path_details["tcut"]),
                "rmse_path_filtered_window_rmse": float(
                    rmse_path_details["filtered_window_rmse"]
                ),
                "rmse_path_raw_window_rmse": float(
                    rmse_path_details["raw_window_rmse"]
                ),
                "rmse_path_rmse_ratio": float(rmse_path_details["rmse_ratio"]),
                "rmse_path_rmse_reduction": float(
                    rmse_path_details["rmse_reduction"]
                ),
                "rmse_path_final_error_time": float(
                    rmse_path_details["final_error_time"]
                ),
            }
        )

    if reference_window_sweep is not None:
        selected_validation = reference_window_sweep[
            reference_window_sweep["is_selected_by_sensitivity"]
        ]
        if len(selected_validation) != 1:
            raise RuntimeError(
                "Expected exactly one reference-validation row selected by the "
                "reference-free reduced-slope sensitivity criterion."
            )
        selected_validation_row = selected_validation.iloc[0]
        metadata["summary"].update(
            {
                "selected_window_filtered_rmse": float(
                    selected_validation_row["filtered_window_rmse"]
                ),
                "selected_window_raw_rmse": float(selected_validation_row["raw_window_rmse"]),
                "selected_window_rmse_ratio": float(selected_validation_row["rmse_ratio"]),
                "selected_window_rmse_reduction": float(
                    selected_validation_row["rmse_reduction"]
                ),
            }
        )
    if metric_path_comparison is not None:
        selected_comparison = metric_path_comparison[
            np.isclose(
                metric_path_comparison["tmax"].to_numpy(dtype=float),
                selected_tmax,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if len(selected_comparison) == 1:
            comparison_row = selected_comparison.iloc[0]
            metadata["temporal_stability_at_selected_tmax"] = {
                key: (
                    float(value)
                    if pd.notna(value)
                    else None
                )
                for key, value in comparison_row.to_dict().items()
            }
    save_json(metadata, paths.postprocess_metadata)

    print(
        "Spectral postprocess source: "
        f"PINN sweep through t={processed_final_time:.3f} of configured "
        f"t={configured_final_time:.3f} ({status}); "
        f"selected tmax={selected_tmax:.3f}."
    )
    print("Saved spectral postprocess outputs:")
    for path in expected:
        print(f"  {path.relative_to(ROOT)}")
    print(
        "Reduced-sensitivity cutoff reference comparison: "
        f"tmax={selected_tmax:.3f}, tcut={selected_tcut:.3f}, "
        f"final E_unfiltered={final_unfiltered:.6e}, "
        f"final E_filtered={final_filtered:.6e}, "
        f"reduction={reduction_factor:.3f}x"
    )
    post_onset_pairs = int(adjacent_training_summary["post_onset_pair_count"])
    if post_onset_pairs > 0:
        reduced_pairs = int(adjacent_training_summary["reduced_pair_count"])
        reduced_fraction = float(
            adjacent_training_summary["reduced_pair_fraction"]
        )
        median_ratio = float(
            adjacent_training_summary["median_filtered_to_raw_ratio"]
        )
        print(
            "Adjacent-training post-onset consistency: "
            f"filtered discrepancy lower in {reduced_pairs}/{post_onset_pairs} "
            f"pairs ({100.0 * reduced_fraction:.1f}%); "
            f"median filtered/raw ratio={median_ratio:.6f}"
        )
    if rmse_path_details is not None:
        print(
            "Minimum-RMSE-path comparison at the same tmax: "
            f"tmax={rmse_path_details['tmax']:.3f}, "
            f"tcut={rmse_path_details['tcut']:.3f}, "
            f"window RMSE={rmse_path_details['filtered_window_rmse']:.6e}, "
            f"final E={rmse_path_details['final_error_time']:.6e}"
        )
    print(
        "Preprocessing sensitivity at the selected interval: "
        f"M={preprocessing_diagnostics['window_sample_count']}, "
        f"zero-frequency fraction="
        f"{preprocessing_diagnostics['zero_frequency_energy_fraction']:.7f}, "
        "mean-projection max deviation="
        f"{preprocessing_diagnostics['mean_projection_max_abs_difference']:.3e}, "
        "first-pair fluctuation fraction="
        f"{preprocessing_diagnostics['first_shell_fluctuation_energy_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
