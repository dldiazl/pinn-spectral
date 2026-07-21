"""Generate raw finite-difference baseline solutions.

This script reads ``configs/numerical.yaml``, runs the selected finite-difference
schemes, and writes one long-format parquet file plus one metadata JSON per
method. It does not compute error diagnostics or figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.analytical import solution_to_dataframe  # noqa: E402
from pinn_spectral.benchmark import BenchmarkConfig, make_space_time_grid  # noqa: E402
from pinn_spectral.numerical import (  # noqa: E402
    solve_cds_crank_nicolson,
    solve_cds_explicit,
    solve_compact_crank_nicolson,
)
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    output_name,
    print_skip_message,
    read_yaml,
    save_json,
    save_parquet,
    should_skip,
)

SOLVERS = {
    "CDS_EF": solve_cds_explicit,
    "CDS_CN": solve_cds_crank_nicolson,
    "CompactSchemes_CN": solve_compact_crank_nicolson,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Generate finite-difference baseline data.")
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


def build_numerical_paths(config: dict[str, Any], method: str) -> dict[str, Path]:
    """Build data and metadata output paths for a numerical method.

    Parameters
    ----------
    config:
        Numerical configuration mapping.
    method:
        Method key, for example ``CDS_EF``.

    Returns
    -------
    dict[str, Path]
        Mapping with ``data`` and ``metadata`` paths.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    outputs = config["outputs"]
    stem = output_name(method, int(grid["n_space"]), float(grid["dt"]), float(grid["final_time"]), benchmark.peclet)
    return {
        "data": ROOT / outputs.get("data_dir", "data/numerical") / f"{stem}.parquet",
        "metadata": ROOT / outputs.get("metrics_dir", "results/metrics") / f"{stem}_metadata.json",
    }


def run_numerical_solver(
    method: str,
    benchmark: BenchmarkConfig,
    n_space: int,
    final_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a selected finite-difference solver.

    Parameters
    ----------
    method:
        Method key present in ``SOLVERS``.
    benchmark:
        Physical benchmark configuration.
    n_space:
        Number of spatial grid points.
    final_time:
        Final simulation time.
    dt:
        Time step.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Spatial grid, temporal grid, and solution matrix with shape ``(nx, nt)``.
    """
    if method not in SOLVERS:
        raise ValueError(f"Unknown finite-difference method: {method}")
    return SOLVERS[method](benchmark, n_space=int(n_space), final_time=float(final_time), dt=float(dt))


def save_numerical_solution(
    method: str,
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Write one numerical solution and its metadata.

    Parameters
    ----------
    method:
        Method key used for console output.
    x, t, u:
        Spatial grid, temporal grid, and solution matrix.
    paths:
        Mapping with ``data`` and ``metadata`` output paths.
    metadata:
        Serializable metadata mapping.
    """
    df = solution_to_dataframe(x, t, u)
    save_parquet(df, paths["data"])
    save_json(metadata, paths["metadata"])
    print(f"Saved {method} to {paths['data'].relative_to(ROOT)}")
    print(f"Saved metadata to {paths['metadata'].relative_to(ROOT)}")


def main() -> None:
    """Run the finite-difference data-generation workflow."""
    args = parse_args()
    config = load_numerical_config(args.config)
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    methods = list(config.get("methods", SOLVERS.keys()))

    n_space = int(grid["n_space"])
    final_time = float(grid["final_time"])
    dt = float(grid["dt"])
    _, t_template = make_space_time_grid(benchmark.length, n_space, final_time, dt)

    for method in methods:
        paths = build_numerical_paths(config, method)
        if should_skip([paths["data"], paths["metadata"]], overwrite=args.overwrite):
            print_skip_message([paths["data"], paths["metadata"]], ROOT)
            continue

        print(f"Generating {method}: nx={n_space}, nt={t_template.size}, dt={dt}, T={final_time}, Pe={benchmark.peclet:.3f}")
        x, t, u = run_numerical_solver(method, benchmark, n_space, final_time, dt)
        metadata = {
            "method": method,
            "length": benchmark.length,
            "u_left": benchmark.u_left,
            "u_right": benchmark.u_right,
            "velocity": benchmark.velocity,
            "diffusivity": benchmark.diffusivity,
            "peclet": benchmark.peclet,
            "n_space": n_space,
            "n_time": int(t.size),
            "dt": dt,
            "final_time": final_time,
        }
        save_numerical_solution(method, x, t, u, paths, metadata)


if __name__ == "__main__":
    main()
