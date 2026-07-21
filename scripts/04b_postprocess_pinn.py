"""Postprocess the latest completed progressive PINN window.

This stage does not require the full ``0 <= t <= 5`` training run to be
finished. It reads the atomic progressive-window summaries written by
``04a_run_pinn.py``, selects the latest continuous completed window, truncates
the analytical and finite-difference solutions to the same time interval, and
computes the unnormalized discrete error norms used in the article.
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Postprocess the latest completed progressive PINN window."
    )
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/pinn.yaml",
        help="Path to the PINN configuration YAML file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load one project-relative YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return read_yaml(path)


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_paths(config: dict[str, Any], numerical_config: dict[str, Any]) -> dict[str, Path]:
    """Build the stable inputs and outputs used by partial PINN postprocessing."""
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    outputs = config["outputs"]
    numerical_outputs = numerical_config["outputs"]

    n_space = int(grid["n_space"])
    dt = float(grid["dt"])
    configured_final_time = float(grid["final_time"])
    pe = benchmark.peclet

    reference_stem = output_name(
        "Analytical", n_space, dt, configured_final_time, pe
    )
    postprocess_dir = resolve_path(
        outputs.get("postprocess_dir", "results/postprocess/pinn")
    )
    metrics_dir = resolve_path(outputs.get("metrics_dir", "results/metrics"))

    return {
        "reference": ROOT / "data" / "reference" / f"{reference_stem}.parquet",
        "numerical_data_dir": resolve_path(
            numerical_outputs.get("data_dir", "data/numerical")
        ),
        "windows_csv": metrics_dir / "pinn_windows.csv",
        "training_metadata": metrics_dir / "pinn_training_metadata.json",
        "time_csv": postprocess_dir / "pinn_error_time.csv",
        "space_csv": postprocess_dir / "pinn_error_space.csv",
        "profiles_csv": postprocess_dir / "pinn_profiles.csv",
        "history_csv": postprocess_dir / "pinn_training_history.csv",
        "metadata_json": metrics_dir / "pinn_diagnostics_metadata.json",
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _resolve_recorded_path(value: Any) -> Path:
    """Resolve a path recorded in a training summary."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("A required artifact path is missing from pinn_windows.csv.")
    return resolve_path(str(value))


def latest_completed_window(paths: dict[str, Path]) -> dict[str, Any]:
    """Return the latest continuous completed PINN window.

    ``04a_run_pinn.py`` writes ``pinn_windows.csv`` only after a complete
    solution bundle and its atomic completion marker have been stored. This
    function validates the continuous window prefix and selects its last row.
    """
    require_file(paths["windows_csv"], "Run: python scripts\\04a_run_pinn.py")
    require_file(paths["training_metadata"], "Run: python scripts\\04a_run_pinn.py")

    windows = pd.read_csv(paths["windows_csv"])
    required_columns = {
        "window_index",
        "window_final_time",
        "solution_path",
        "history_path",
        "completion_path",
    }
    missing = required_columns.difference(windows.columns)
    if missing:
        raise ValueError(
            f"{paths['windows_csv']} is missing columns: {sorted(missing)}"
        )
    if windows.empty:
        raise RuntimeError(
            "No completed PINN window is available yet. Wait until 04a finishes "
            "at least the first temporal window."
        )

    windows = windows.copy()
    windows["window_index"] = pd.to_numeric(
        windows["window_index"], errors="raise"
    ).astype(np.int64)
    windows["window_final_time"] = pd.to_numeric(
        windows["window_final_time"], errors="raise"
    ).astype(np.float64)
    windows = windows.sort_values("window_index").reset_index(drop=True)

    expected_indices = np.arange(1, len(windows) + 1, dtype=np.int64)
    actual_indices = windows["window_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise RuntimeError(
            "pinn_windows.csv does not contain a continuous completed-window prefix. "
            f"Expected indices {expected_indices.tolist()}, got {actual_indices.tolist()}."
        )

    window_times = windows["window_final_time"].to_numpy(dtype=np.float64)
    if np.any(np.diff(window_times) <= 0.0):
        raise RuntimeError(
            "Completed PINN window final times are not strictly increasing."
        )

    for row in windows.itertuples(index=False):
        solution_path = _resolve_recorded_path(row.solution_path)
        history_path = _resolve_recorded_path(row.history_path)
        completion_path = _resolve_recorded_path(row.completion_path)
        require_file(
            solution_path,
            "The window is listed as complete, but its solution file is missing.",
        )
        require_file(
            history_path,
            "The window is listed as complete, but its history file is missing.",
        )
        require_file(
            completion_path,
            "The window is listed as complete, but its completion marker is missing.",
        )
        if solution_path.stat().st_size == 0:
            raise RuntimeError(f"Completed PINN solution is empty: {solution_path}")
        if history_path.stat().st_size == 0:
            raise RuntimeError(f"Completed PINN history is empty: {history_path}")

        marker = read_json(completion_path)
        if marker.get("status") != "COMPLETE":
            raise RuntimeError(
                f"Invalid completion status in {completion_path}: "
                f"{marker.get('status')!r}"
            )
        marker_index = int(marker.get("window_index", -1))
        marker_time = float(marker.get("window_final_time", np.nan))
        if marker_index != int(row.window_index) or not np.isclose(
            marker_time,
            float(row.window_final_time),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Completion marker does not match pinn_windows.csv: {completion_path}"
            )

    latest = windows.iloc[-1]
    return {
        "window_index": int(latest["window_index"]),
        "window_final_time": float(latest["window_final_time"]),
        "solution_path": _resolve_recorded_path(latest["solution_path"]),
        "history_path": _resolve_recorded_path(latest["history_path"]),
        "completion_path": _resolve_recorded_path(latest["completion_path"]),
        "completed_windows": int(len(windows)),
        "windows": windows,
    }



_HISTORY_SERIES = [
    ("Loss", "TotalLoss", "Total loss (scaled)", True),
    ("LossIC", "LossIC", "IC loss", False),
    ("LossBC", "LossBC", "BC loss", False),
    ("LossPhysics", "LossPhysics", "Physics loss", False),
]


def build_training_history_dataframe(windows: pd.DataFrame) -> pd.DataFrame:
    """Combine completed-window histories into a long plotting table.

    The horizontal coordinate is the continuous global outer epoch written by
    ``04a_run_pinn.py``. Component losses remain unscaled, while ``Loss`` is
    preserved exactly as the recorded weighted training objective.
    """
    parts: list[pd.DataFrame] = []
    previous_global_epoch = 0

    for row in windows.sort_values("window_index").itertuples(index=False):
        history_path = _resolve_recorded_path(row.history_path)
        history = pd.read_parquet(history_path)
        if history.empty:
            raise RuntimeError(f"Completed PINN history is empty: {history_path}")

        required = {"Epoch", "Loss", "LossIC", "LossBC", "LossPhysics"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(
                f"{history_path} is missing training-history columns: {sorted(missing)}"
            )

        global_epoch = pd.to_numeric(history["Epoch"], errors="raise").astype(np.int64)
        expected = np.arange(
            previous_global_epoch + 1,
            previous_global_epoch + len(history) + 1,
            dtype=np.int64,
        )
        if not np.array_equal(global_epoch.to_numpy(dtype=np.int64), expected):
            raise RuntimeError(
                "Completed PINN histories do not contain a continuous global epoch "
                f"sequence at window {int(row.window_index)}."
            )
        previous_global_epoch = int(expected[-1])

        if "EpochInWindow" in history.columns:
            epoch_in_window = pd.to_numeric(
                history["EpochInWindow"], errors="raise"
            ).astype(np.int64)
        else:
            epoch_in_window = pd.Series(
                np.arange(1, len(history) + 1, dtype=np.int64),
                index=history.index,
            )

        for source_column, loss_name, label, is_scaled in _HISTORY_SERIES:
            values = pd.to_numeric(history[source_column], errors="raise").astype(
                np.float64
            )
            parts.append(
                pd.DataFrame(
                    {
                        "window_index": int(row.window_index),
                        "window_final_time": float(row.window_final_time),
                        "epoch_in_window": epoch_in_window.to_numpy(dtype=np.int64),
                        "global_epoch": global_epoch.to_numpy(dtype=np.int64),
                        "loss_name": loss_name,
                        "label": label,
                        "loss_value": values.to_numpy(dtype=np.float64),
                        "is_scaled": bool(is_scaled),
                    }
                )
            )

    if not parts:
        raise RuntimeError("No completed PINN training histories are available.")
    return pd.concat(parts, ignore_index=True)

def _truncate_to_target_grid(
    x_full: np.ndarray,
    t_full: np.ndarray,
    u_full: np.ndarray,
    x_target: np.ndarray,
    t_target: np.ndarray,
    label: str,
) -> np.ndarray:
    """Return a full-domain solution restricted to a target prefix grid."""
    if t_target.size > t_full.size:
        raise ValueError(
            f"{label} has only {t_full.size} time levels, but the PINN requires "
            f"{t_target.size}."
        )
    t_prefix = t_full[: t_target.size]
    u_prefix = u_full[:, : t_target.size]
    assert_same_grid(x_target, t_target, x_full, t_prefix, label)
    return u_prefix


def _effective_profile_times(
    requested_times: list[float],
    available_times: np.ndarray,
) -> list[float]:
    """Map configured profile times to the available partial PINN interval.

    Requested times after the latest completed window collapse to that latest
    time. Duplicate mapped times are removed while preserving their order.
    """
    if available_times.size == 0:
        raise ValueError("The PINN solution contains no time levels.")

    first_time = float(available_times[0])
    last_time = float(available_times[-1])
    effective: list[float] = []
    for requested in requested_times:
        bounded = min(max(float(requested), first_time), last_time)
        index = int(np.argmin(np.abs(available_times - bounded)))
        actual = float(available_times[index])
        if not any(np.isclose(actual, value, rtol=0.0, atol=1e-12) for value in effective):
            effective.append(actual)
    return effective


def profile_rows(
    solution_name: str,
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    requested_times: list[float],
) -> list[pd.DataFrame]:
    """Extract one solution profile at each requested time."""
    rows: list[pd.DataFrame] = []
    for requested_time in requested_times:
        time_index = int(np.argmin(np.abs(t - float(requested_time))))
        rows.append(
            pd.DataFrame(
                {
                    "solution": solution_name,
                    "t": float(t[time_index]),
                    "x": x,
                    "u": u[:, time_index],
                }
            )
        )
    return rows


def compute_pinn_postprocess(
    config: dict[str, Any],
    numerical_config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute diagnostics and training history for completed PINN windows."""
    require_file(
        paths["reference"],
        "Run: python scripts\\01a_generate_reference_data.py --overwrite",
    )
    latest = latest_completed_window(paths)
    training_history_df = build_training_history_dataframe(latest["windows"])

    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    n_space = int(grid["n_space"])
    dt = float(grid["dt"])
    configured_final_time = float(grid["final_time"])
    pe = benchmark.peclet

    x_pinn, t_pinn, u_pinn = read_solution_matrix(latest["solution_path"])
    if not np.isclose(
        float(t_pinn[-1]),
        latest["window_final_time"],
        rtol=0.0,
        atol=max(1e-12, abs(dt) * 1e-9),
    ):
        raise RuntimeError(
            "The latest PINN solution final time does not match its completed-window marker: "
            f"solution={t_pinn[-1]:.12g}, marker={latest['window_final_time']:.12g}."
        )

    x_ref_full, t_ref_full, u_ref_full = read_solution_matrix(paths["reference"])
    u_ref = _truncate_to_target_grid(
        x_ref_full,
        t_ref_full,
        u_ref_full,
        x_pinn,
        t_pinn,
        "Analytical",
    )
    x_ref = x_pinn
    t_ref = t_pinn

    time_parts: list[pd.DataFrame] = []
    space_parts: list[pd.DataFrame] = []
    numerical_solutions: list[tuple[str, np.ndarray]] = []
    numerical_data_dir = paths["numerical_data_dir"]
    methods = list(numerical_config.get("methods", METHOD_LABELS.keys()))

    for method in methods:
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown finite-difference method: {method}")
        stem = output_name(method, n_space, dt, configured_final_time, pe)
        method_path = numerical_data_dir / f"{stem}.parquet"
        require_file(
            method_path,
            "Run: python scripts\\02a_generate_numerical_data.py --overwrite",
        )
        x_method, t_method, u_method_full = read_solution_matrix(method_path)
        u_method = _truncate_to_target_grid(
            x_method,
            t_method,
            u_method_full,
            x_pinn,
            t_pinn,
            method,
        )
        label = METHOD_LABELS[method]
        numerical_solutions.append((label, u_method))
        time_parts.append(
            pd.DataFrame(
                {
                    "method": method,
                    "label": label,
                    "t": t_ref,
                    "error_l2": compute_error_time(u_method, u_ref),
                }
            )
        )
        space_parts.append(
            pd.DataFrame(
                {
                    "method": method,
                    "label": label,
                    "x": x_ref,
                    "error_l2": compute_error_space(u_method, u_ref),
                }
            )
        )

    pinn_error_time = compute_error_time(u_pinn, u_ref)
    pinn_error_space = compute_error_space(u_pinn, u_ref)
    time_parts.append(
        pd.DataFrame(
            {
                "method": "PINN",
                "label": "PINN",
                "t": t_ref,
                "error_l2": pinn_error_time,
            }
        )
    )
    space_parts.append(
        pd.DataFrame(
            {
                "method": "PINN",
                "label": "PINN",
                "x": x_ref,
                "error_l2": pinn_error_space,
            }
        )
    )
    time_df = pd.concat(time_parts, ignore_index=True)
    space_df = pd.concat(space_parts, ignore_index=True)

    requested_times = [
        float(value)
        for value in config.get("postprocess", {}).get("profile_times", [])
    ]
    if not requested_times:
        raise ValueError("postprocess.profile_times must contain at least one time.")
    effective_times = _effective_profile_times(requested_times, t_pinn)

    profile_parts = profile_rows(
        "Analytical", x_ref, t_ref, u_ref, effective_times
    )
    for label, u_method in numerical_solutions:
        profile_parts.extend(
            profile_rows(label, x_ref, t_ref, u_method, effective_times)
        )
    profile_parts.extend(
        profile_rows("PINN", x_pinn, t_pinn, u_pinn, effective_times)
    )
    profiles_df = pd.concat(profile_parts, ignore_index=True)

    training_metadata = read_json(paths["training_metadata"])
    training_complete = np.isclose(
        latest["window_final_time"],
        configured_final_time,
        rtol=0.0,
        atol=max(1e-12, abs(dt) * 1e-9),
    )
    metadata = {
        "stage": "pinn_postprocess",
        "status": "complete" if training_complete else "partial",
        "architecture": training_metadata.get("architecture"),
        "reference": str(paths["reference"].relative_to(ROOT)),
        "pinn_solution": str(latest["solution_path"].relative_to(ROOT)),
        "source_window_index": latest["window_index"],
        "completed_windows": latest["completed_windows"],
        "expected_windows": training_metadata.get("expected_windows"),
        "training_complete": bool(training_complete),
        "configured_final_time": configured_final_time,
        "processed_final_time": float(t_pinn[-1]),
        "n_space": int(x_ref.size),
        "n_time": int(t_ref.size),
        "dt": dt,
        "requested_profile_times": requested_times,
        "profile_times": effective_times,
        "training_history_csv": str(paths["history_csv"].relative_to(ROOT)),
        "training_history": {
            "global_epochs": int(training_history_df["global_epoch"].max()),
            "rows": int(len(training_history_df)),
            "series": [
                {
                    "loss_name": loss_name,
                    "label": label,
                    "is_scaled": bool(is_scaled),
                }
                for _, loss_name, label, is_scaled in _HISTORY_SERIES
            ],
            "total_loss_definition": "1000*LossBC + 1000*LossIC + LossPhysics",
            "component_losses": "unscaled mean-squared losses",
        },
        "error_definition": {
            "E_t": "unnormalized spatial L2 norm at each time",
            "E_x": "unnormalized time-aggregated L2 norm over the processed interval at each spatial node",
        },
        "summary": {
            "final_error_time": float(pinn_error_time[-1]),
            "minimum_error_time": float(np.min(pinn_error_time)),
            "maximum_error_time": float(np.max(pinn_error_time)),
            "maximum_error_space": float(np.max(pinn_error_space)),
            "global_l2_error": float(np.linalg.norm(u_pinn - u_ref)),
            "relative_global_l2_error": float(
                np.linalg.norm(u_pinn - u_ref) / np.linalg.norm(u_ref)
            ),
        },
    }
    return time_df, space_df, profiles_df, training_history_df, metadata


def main() -> None:
    """Run PINN postprocessing for the latest completed window."""
    args = parse_args()
    config = load_config(args.config)
    numerical_config_path = config.get("inputs", {}).get(
        "numerical_config", "configs/numerical.yaml"
    )
    numerical_config = load_config(numerical_config_path)
    paths = build_paths(config, numerical_config)

    expected_outputs = [
        paths["time_csv"],
        paths["space_csv"],
        paths["profiles_csv"],
        paths["history_csv"],
        paths["metadata_json"],
    ]
    reject_partial_outputs(expected_outputs, overwrite=args.overwrite)
    if should_skip(expected_outputs, overwrite=args.overwrite):
        print_skip_message(expected_outputs, ROOT)
        return

    time_df, space_df, profiles_df, training_history_df, metadata = compute_pinn_postprocess(
        config, numerical_config, paths
    )
    save_csv(time_df, paths["time_csv"])
    save_csv(space_df, paths["space_csv"])
    save_csv(profiles_df, paths["profiles_csv"])
    save_csv(training_history_df, paths["history_csv"])
    save_json(metadata, paths["metadata_json"])

    status = "complete" if metadata["training_complete"] else "partial"
    print(
        "PINN postprocess source: "
        f"window {metadata['source_window_index']} at "
        f"t={metadata['processed_final_time']:.3f} ({status})."
    )
    print("Saved PINN postprocess outputs:")
    for key in ["time_csv", "space_csv", "profiles_csv", "history_csv", "metadata_json"]:
        print(f"  {paths[key].relative_to(ROOT)}")
    summary = metadata["summary"]
    print(
        "PINN reference error over the processed interval: "
        f"final E(t)={summary['final_error_time']:.6e}, "
        f"max E(t)={summary['maximum_error_time']:.6e}, "
        f"relative global L2={summary['relative_global_l2_error']:.6e}"
    )


if __name__ == "__main__":
    main()
