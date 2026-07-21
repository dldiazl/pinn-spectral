"""Postprocess analytical-reference data into reusable diagnostics.

This script creates CSV and JSON files used by the reference figures. It does
not write any figures. Endpoint diagnostics intentionally use the literal
partial-sum construction to preserve the finite-precision behavior documented in
the manuscript diagnostics.
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

from pinn_spectral.analytical import analytical_solution, stationary_solution  # noqa: E402
from pinn_spectral.benchmark import BenchmarkConfig, initial_condition, make_space_time_grid  # noqa: E402
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    print_skip_message,
    read_yaml,
    save_csv,
    save_json,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Postprocess analytical-reference diagnostics.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/reference.yaml",
        help="Path to the reference configuration YAML file.",
    )
    return parser.parse_args()


def load_reference_config(config_path: str | Path) -> dict[str, Any]:
    """Load the analytical-reference YAML configuration.

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


def legacy_coefficient_bn(
    length: float,
    velocity: float,
    diffusivity: float,
    n: np.ndarray,
    u_left: float,
    u_right: float,
) -> np.ndarray:
    """Return the legacy closed-form modal coefficient ``B_n``.

    Parameters
    ----------
    length, velocity, diffusivity:
        Physical parameters of the advection-diffusion problem.
    n:
        One-dimensional array of modal indices.
    u_left, u_right:
        Dirichlet boundary values.

    Returns
    -------
    np.ndarray
        Modal coefficient values with the same shape as ``n``.

    Notes
    -----
    The expression is retained only for endpoint convergence diagnostics because
    it reproduces the numerical behavior of the original convergence scripts.
    """
    L = np.float64(length)
    v = np.float64(velocity)
    D = np.float64(diffusivity)
    nn = np.asarray(n, dtype=np.float64)
    u0 = np.float64(u_left)
    uL = np.float64(u_right)

    int_f = (
        8
        * np.pi
        * D**2
        * nn
        * (
            2
            * (-1.0) ** nn
            * np.pi
            * D
            * L
            * u0
            * v
            * np.exp(L * v / (2.0 * D))
            + (-1.0) ** (nn + 1.0)
            * uL
            * (4.0 * np.pi**2 * D**2 * nn**2 - np.pi**2 * D**2 + L**2 * v**2)
            * np.exp(L * v / (2.0 * D))
            + 2.0 * np.pi * D * L * uL * v * np.exp(L * v / D)
            + u0
            * (4.0 * np.pi**2 * D**2 * nn**2 - np.pi**2 * D**2 + L**2 * v**2)
            * np.exp(L * v / D)
        )
        * np.exp(-L * v / D)
        / (
            16.0 * np.pi**4 * D**4 * nn**4
            - 8.0 * np.pi**4 * D**4 * nn**2
            + np.pi**4 * D**4
            + 8.0 * np.pi**2 * D**2 * L**2 * nn**2 * v**2
            + 2.0 * np.pi**2 * D**2 * L**2 * v**2
            + L**4 * v**4
        )
    )

    int_us = (
        8.0
        * np.pi
        * D**2
        * nn
        * (
            (-1.0) ** nn * uL * np.exp(L * v / D)
            + (-1.0) ** (nn + 1.0) * uL
            - u0 * np.exp(3.0 * L * v / (2.0 * D))
            + u0 * np.exp(L * v / (2.0 * D))
        )
        * np.exp(-L * v / (2.0 * D))
        / (
            4.0 * np.pi**2 * D**2 * nn**2 * np.exp(L * v / D)
            - 4.0 * np.pi**2 * D**2 * nn**2
            + L**2 * v**2 * np.exp(L * v / D)
            - L**2 * v**2
        )
    )

    return np.asarray(int_f + int_us, dtype=np.float64)


def last_k_average_columns(partial_sums: np.ndarray, averaging_width: int) -> np.ndarray:
    """Average the last K columns at every truncation level.

    Parameters
    ----------
    partial_sums:
        Matrix whose column ``j`` corresponds to truncation ``N=j+1``.
    averaging_width:
        Backward averaging width ``K``.

    Returns
    -------
    np.ndarray
        Matrix with the same shape as ``partial_sums`` where each column is the
        average over columns ``max(0, j-K+1), ..., j``.
    """
    k = int(averaging_width)
    if k <= 1:
        return partial_sums

    n_terms = partial_sums.shape[1]
    k = min(k, n_terms)
    cumulative = np.cumsum(partial_sums, axis=1, dtype=np.float64)
    shifted = np.zeros_like(cumulative)
    shifted[:, k:] = cumulative[:, :-k]
    window_sum = cumulative - shifted
    counts = np.minimum(np.arange(1, n_terms + 1), k).astype(np.float64)
    return window_sum / counts[None, :]


def literal_partial_solutions_at_time(
    x: np.ndarray,
    time_value: float,
    v: float,
    D: float,
    L: float,
    u0: float,
    uL: float,
    n_terms: int,
    averaging_width: int,
) -> np.ndarray:
    """Return literal last-K averaged partial solutions for all truncations.

    Parameters
    ----------
    x:
        Spatial grid.
    time_value:
        Time at which all partial solutions are evaluated.
    v, D, L, u0, uL:
        Physical parameters and boundary values.
    n_terms:
        Maximum truncation index.
    averaging_width:
        Backward averaging width ``K``.

    Returns
    -------
    np.ndarray
        Matrix with shape ``(nx, n_terms)``. Column ``j`` corresponds to the
        averaged solution at truncation ``N=j+1``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_terms = int(n_terms)
    n = np.arange(1, n_terms + 1, dtype=np.float64)

    bn = legacy_coefficient_bn(L, v, D, n, u0, uL)
    sine = np.sin((n[:, None] * np.pi / np.float64(L)) * x[None, :])
    decay = np.exp(-np.float64(D) * (n * np.pi / np.float64(L)) ** 2 * np.float64(time_value))
    modal_terms = (bn[:, None] * sine * decay[:, None]).T
    partial_transient = np.cumsum(modal_terms, axis=1, dtype=np.float64)

    factor = np.exp(
        -np.float64(v) ** 2 * np.float64(time_value) / (4.0 * np.float64(D))
        + np.float64(v) * x / (2.0 * np.float64(D))
    )
    stationary = stationary_solution(x, v, D, L, u0, uL)

    partial_solutions = stationary[:, None] + factor[:, None] * partial_transient
    return last_k_average_columns(partial_solutions, averaging_width)


def endpoint_consistency_dataframe(
    x: np.ndarray,
    time_value: float,
    target: np.ndarray,
    v: float,
    D: float,
    L: float,
    u0: float,
    uL: float,
    n_terms: int,
    k_values: list[int],
) -> pd.DataFrame:
    """Compute endpoint consistency errors for all ``N`` and ``K`` values.

    Parameters
    ----------
    x:
        Spatial grid.
    time_value:
        Endpoint time to evaluate.
    target:
        Target profile, either the initial condition or stationary solution.
    v, D, L, u0, uL:
        Physical parameters and boundary values.
    n_terms:
        Maximum truncation index.
    k_values:
        Averaging widths to evaluate.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``N``, ``K``, and ``error_l2``.
    """
    rows: list[pd.DataFrame] = []
    n_values = np.arange(1, int(n_terms) + 1, dtype=np.int64)

    for k in k_values:
        print(f"    K={k}...")
        partial_solutions = literal_partial_solutions_at_time(
            x=x,
            time_value=time_value,
            v=v,
            D=D,
            L=L,
            u0=u0,
            uL=uL,
            n_terms=n_terms,
            averaging_width=k,
        )
        errors = np.linalg.norm(partial_solutions - target[:, None], axis=0)
        rows.append(pd.DataFrame({"N": n_values, "K": int(k), "error_l2": errors}))

    return pd.concat(rows, ignore_index=True)


def compute_temporal_convergence(config: dict[str, Any]) -> pd.DataFrame:
    """Compute convergence of the unsmoothed series to the stationary profile.

    Parameters
    ----------
    config:
        Reference configuration mapping.

    Returns
    -------
    pd.DataFrame
        Table with columns ``t`` and ``error_l2``.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    diagnostics_cfg = config.get("diagnostics", {})
    x, t = make_space_time_grid(
        benchmark.length,
        int(grid["n_space"]),
        float(grid["final_time"]),
        float(grid["dt"]),
    )
    stationary = stationary_solution(
        x,
        benchmark.velocity,
        benchmark.diffusivity,
        benchmark.length,
        benchmark.u_left,
        benchmark.u_right,
    )
    u_temporal = analytical_solution(
        x,
        t,
        benchmark.velocity,
        benchmark.diffusivity,
        benchmark.length,
        benchmark.u_left,
        benchmark.u_right,
        n_terms=int(diagnostics_cfg.get("temporal_n_terms", ref["n_terms"])),
        averaging_width=int(diagnostics_cfg.get("temporal_averaging_width", 1)),
        n_modes_per_block=int(ref.get("n_modes_per_block", 2000)),
        n_times_per_block=int(ref.get("n_times_per_block", 256)),
    )
    error = np.linalg.norm(u_temporal - stationary[:, None], axis=0)
    return pd.DataFrame({"t": t, "error_l2": error})


def compute_initial_consistency(config: dict[str, Any]) -> pd.DataFrame:
    """Compute initial-condition consistency over all retained truncations.

    Parameters
    ----------
    config:
        Reference configuration mapping.

    Returns
    -------
    pd.DataFrame
        Table with columns ``N``, ``K``, and ``error_l2``.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    diagnostics_cfg = config.get("diagnostics", {})
    x, _ = make_space_time_grid(benchmark.length, int(grid["n_space"]), float(grid["final_time"]), float(grid["dt"]))
    target = initial_condition(x, benchmark)
    k_values = [int(k) for k in diagnostics_cfg.get("k_values", [1, 10, 100, 1000, 10000])]
    k_values = [k for k in k_values if 1 <= k <= int(ref["n_terms"])]
    return endpoint_consistency_dataframe(
        x=x,
        time_value=0.0,
        target=target,
        v=benchmark.velocity,
        D=benchmark.diffusivity,
        L=benchmark.length,
        u0=benchmark.u_left,
        uL=benchmark.u_right,
        n_terms=int(ref["n_terms"]),
        k_values=k_values,
    )


def compute_stationary_consistency(config: dict[str, Any]) -> pd.DataFrame:
    """Compute stationary-profile consistency over all retained truncations.

    Parameters
    ----------
    config:
        Reference configuration mapping.

    Returns
    -------
    pd.DataFrame
        Table with columns ``N``, ``K``, and ``error_l2``.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    diagnostics_cfg = config.get("diagnostics", {})
    x, _ = make_space_time_grid(benchmark.length, int(grid["n_space"]), float(grid["final_time"]), float(grid["dt"]))
    target = stationary_solution(
        x,
        benchmark.velocity,
        benchmark.diffusivity,
        benchmark.length,
        benchmark.u_left,
        benchmark.u_right,
    )
    k_values = [int(k) for k in diagnostics_cfg.get("k_values", [1, 10, 100, 1000, 10000])]
    k_values = [k for k in k_values if 1 <= k <= int(ref["n_terms"])]
    return endpoint_consistency_dataframe(
        x=x,
        time_value=float(grid["final_time"]),
        target=target,
        v=benchmark.velocity,
        D=benchmark.diffusivity,
        L=benchmark.length,
        u0=benchmark.u_left,
        uL=benchmark.u_right,
        n_terms=int(ref["n_terms"]),
        k_values=k_values,
    )


def save_reference_postprocess_outputs(
    temporal_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    stationary_df: pd.DataFrame,
    metadata: dict[str, Any],
    postprocess_dir: Path,
    metrics_dir: Path,
) -> None:
    """Write all reference postprocess outputs.

    Parameters
    ----------
    temporal_df, initial_df, stationary_df:
        Diagnostic tables for the three reference figures.
    metadata:
        Serializable metadata mapping.
    postprocess_dir:
        Directory for CSV outputs.
    metrics_dir:
        Directory for JSON metadata.
    """
    save_csv(temporal_df, postprocess_dir / "temporal_convergence.csv")
    save_csv(initial_df, postprocess_dir / "initial_consistency.csv")
    save_csv(stationary_df, postprocess_dir / "stationary_consistency.csv")
    save_json(metadata, metrics_dir / "reference_diagnostics_metadata.json")


def main() -> None:
    """Run reference postprocessing."""
    args = parse_args()
    config = load_reference_config(args.config)
    outputs = config["outputs"]
    postprocess_dir = ROOT / outputs.get("postprocess_dir", "results/postprocess/reference")
    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    paths = [
        postprocess_dir / "temporal_convergence.csv",
        postprocess_dir / "initial_consistency.csv",
        postprocess_dir / "stationary_consistency.csv",
        metrics_dir / "reference_diagnostics_metadata.json",
    ]
    if should_skip(paths, overwrite=args.overwrite):
        print_skip_message(paths, ROOT)
        return

    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    diagnostics_cfg = config.get("diagnostics", {})
    k_values = [int(k) for k in diagnostics_cfg.get("k_values", [1, 10, 100, 1000, 10000])]
    k_values = [k for k in k_values if 1 <= k <= int(ref["n_terms"])]

    print(
        "Generating reference postprocess data: "
        f"nx={int(grid['n_space'])}, dt={float(grid['dt'])}, T={float(grid['final_time'])}, "
        f"N={int(ref['n_terms'])}, K values={k_values}"
    )
    print("  temporal convergence...")
    temporal_df = compute_temporal_convergence(config)
    print("  initial consistency...")
    initial_df = compute_initial_consistency(config)
    print("  stationary consistency...")
    stationary_df = compute_stationary_consistency(config)

    metadata = {
        "length": benchmark.length,
        "u_left": benchmark.u_left,
        "u_right": benchmark.u_right,
        "velocity": benchmark.velocity,
        "diffusivity": benchmark.diffusivity,
        "peclet": benchmark.peclet,
        "n_space": int(grid["n_space"]),
        "dt": float(grid["dt"]),
        "final_time": float(grid["final_time"]),
        "n_terms": int(ref["n_terms"]),
        "k_values": k_values,
        "n_values_count": int(ref["n_terms"]),
        "n_values": "all",
        "temporal_n_terms": int(diagnostics_cfg.get("temporal_n_terms", ref["n_terms"])),
        "temporal_averaging_width": int(diagnostics_cfg.get("temporal_averaging_width", 1)),
    }
    save_reference_postprocess_outputs(temporal_df, initial_df, stationary_df, metadata, postprocess_dir, metrics_dir)
    print(f"Saved reference postprocess data to {postprocess_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
