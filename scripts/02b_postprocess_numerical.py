"""Postprocess finite-difference baselines into error diagnostics.

This script reads the analytical reference and finite-difference parquet files,
computes L2 error curves in time and in space, and writes reusable CSV files. It
does not generate figures.
"""

from __future__ import annotations

import argparse
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
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Postprocess finite-difference error diagnostics.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/numerical.yaml",
        help="Path to the numerical configuration YAML file.",
    )
    return parser.parse_args()


def load_numerical_config(config_path: str | Path) -> dict[str, Any]:
    """Load the finite-difference YAML configuration.

    Parameters
    ----------
    config_path:
        Path relative to the project root or an absolute path.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return read_yaml(path)


def compute_numerical_diagnostics(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute finite-difference error curves against the analytical reference.

    Parameters
    ----------
    config:
        Numerical configuration mapping.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]
        Time-error table, space-error table, and serializable metadata.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    outputs = config["outputs"]
    methods = list(config.get("methods", METHOD_LABELS.keys()))

    n_space = int(grid["n_space"])
    final_time = float(grid["final_time"])
    dt = float(grid["dt"])
    pe = benchmark.peclet

    reference_stem = output_name("Analytical", n_space, dt, final_time, pe)
    reference_path = ROOT / "data" / "reference" / f"{reference_stem}.parquet"
    numerical_dir = ROOT / outputs.get("data_dir", "data/numerical")

    require_file(reference_path, "Run: python scripts\\01a_generate_reference_data.py --overwrite")
    x_ref, t_ref, u_ref = read_solution_matrix(reference_path)

    rows_time: list[pd.DataFrame] = []
    rows_space: list[pd.DataFrame] = []
    summary: dict[str, dict[str, float]] = {}

    print("Generating numerical postprocess diagnostics:")
    for method in methods:
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown finite-difference method: {method}")

        stem = output_name(method, n_space, dt, final_time, pe)
        method_path = numerical_dir / f"{stem}.parquet"
        require_file(method_path, "Run: python scripts\\02a_generate_numerical_data.py --overwrite")

        x_num, t_num, u_num = read_solution_matrix(method_path)
        assert_same_grid(x_ref, t_ref, x_num, t_num, method)

        error_time = compute_error_time(u_num, u_ref)
        error_space = compute_error_space(u_num, u_ref)
        label = METHOD_LABELS[method]

        rows_time.append(pd.DataFrame({"method": method, "label": label, "t": t_ref, "error_l2": error_time}))
        rows_space.append(pd.DataFrame({"method": method, "label": label, "x": x_ref, "error_l2": error_space}))

        summary[method] = {
            "max_error_time": float(np.max(error_time)),
            "final_error_time": float(error_time[-1]),
            "max_error_space": float(np.max(error_space)),
        }
        print(
            f"  {label}: final E(t)={error_time[-1]:.6e}, "
            f"max E(t)={np.max(error_time):.6e}, max E(x)={np.max(error_space):.6e}"
        )

    metadata = {
        "reference": str(reference_path.relative_to(ROOT)),
        "methods": methods,
        "length": benchmark.length,
        "u_left": benchmark.u_left,
        "u_right": benchmark.u_right,
        "velocity": benchmark.velocity,
        "diffusivity": benchmark.diffusivity,
        "peclet": pe,
        "n_space": n_space,
        "n_time": int(t_ref.size),
        "dt": dt,
        "final_time": final_time,
        "summary": summary,
    }
    return pd.concat(rows_time, ignore_index=True), pd.concat(rows_space, ignore_index=True), metadata


def save_numerical_postprocess_outputs(
    time_df: pd.DataFrame,
    space_df: pd.DataFrame,
    metadata: dict[str, Any],
    postprocess_dir: Path,
    metrics_dir: Path,
) -> None:
    """Write finite-difference postprocess CSV and metadata files.

    Parameters
    ----------
    time_df, space_df:
        Error diagnostic tables.
    metadata:
        Serializable metadata mapping.
    postprocess_dir:
        Directory for CSV outputs.
    metrics_dir:
        Directory for JSON metadata.
    """
    save_csv(time_df, postprocess_dir / "fdm_error_time.csv")
    save_csv(space_df, postprocess_dir / "fdm_error_space.csv")
    save_json(metadata, metrics_dir / "fdm_diagnostics_metadata.json")


def main() -> None:
    """Run finite-difference postprocessing."""
    args = parse_args()
    config = load_numerical_config(args.config)
    outputs = config["outputs"]
    postprocess_dir = ROOT / outputs.get("postprocess_dir", "results/postprocess/numerical")
    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    expected_outputs = [
        postprocess_dir / "fdm_error_time.csv",
        postprocess_dir / "fdm_error_space.csv",
        metrics_dir / "fdm_diagnostics_metadata.json",
    ]
    if should_skip(expected_outputs, overwrite=args.overwrite):
        print_skip_message(expected_outputs, ROOT)
        return

    time_df, space_df, metadata = compute_numerical_diagnostics(config)
    save_numerical_postprocess_outputs(time_df, space_df, metadata, postprocess_dir, metrics_dir)
    print(f"Saved numerical postprocess data to {postprocess_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
