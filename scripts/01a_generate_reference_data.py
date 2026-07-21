"""Generate raw analytical reference data for the pre-NAS workflow.

This script writes the main optimized reference by default and can also generate
an intentionally slow literal implementation for validation. The literal option
is not part of the normal validation run because it is computationally expensive.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.analytical import (  # noqa: E402
    analytical_solution,
    analytical_solution_legacy,
    consistency_errors,
    solution_to_dataframe,
)
from pinn_spectral.benchmark import BenchmarkConfig, make_space_time_grid  # noqa: E402
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    output_name,
    print_skip_message,
    read_yaml,
    save_json,
    save_parquet,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the configuration path, implementation selector,
        progress reporting interval, and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Generate analytical reference data.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/reference.yaml",
        help="Path to the reference configuration YAML file.",
    )
    parser.add_argument(
        "--implementation",
        choices=["optimized", "literal"],
        default="optimized",
        help="Reference implementation to generate. The literal implementation is slow.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help=(
            "For --implementation literal, print progress every N completed time "
            "levels. Use 0 to disable progress messages inside the time loop."
        ),
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


def build_reference_paths(config: dict[str, Any], implementation: str) -> tuple[Path, Path, str]:
    """Build output paths for a reference implementation.

    Parameters
    ----------
    config:
        Reference configuration mapping.
    implementation:
        Either ``optimized`` or ``literal``.

    Returns
    -------
    tuple[Path, Path, str]
        Data path, metadata path, and output file stem.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    outputs = config["outputs"]

    prefix = "Analytical" if implementation == "optimized" else "Analytical_literal"
    stem = output_name(
        prefix=prefix,
        n_space=int(grid["n_space"]),
        dt=float(grid["dt"]),
        final_time=float(grid["final_time"]),
        pe=benchmark.peclet,
    )
    data_path = ROOT / outputs.get("data_dir", "data/reference") / f"{stem}.parquet"
    metadata_path = ROOT / outputs.get("metrics_dir", "results/metrics") / f"{stem}_metadata.json"
    return data_path, metadata_path, stem


def generate_optimized_reference(config: dict[str, Any]) -> tuple[object, object, object]:
    """Generate the optimized analytical reference field.

    Parameters
    ----------
    config:
        Reference configuration mapping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Spatial grid, temporal grid, and solution matrix with shape ``(nx, nt)``.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    x, t = make_space_time_grid(
        benchmark.length,
        int(grid["n_space"]),
        float(grid["final_time"]),
        float(grid["dt"]),
    )
    u = analytical_solution(
        x,
        t,
        benchmark.velocity,
        benchmark.diffusivity,
        benchmark.length,
        benchmark.u_left,
        benchmark.u_right,
        n_terms=int(ref["n_terms"]),
        averaging_width=int(ref["averaging_width"]),
        n_modes_per_block=int(ref.get("n_modes_per_block", 2000)),
        n_times_per_block=int(ref.get("n_times_per_block", 256)),
    )
    return x, t, u


def make_literal_progress_callback(progress_every: int) -> Callable[[int, int, float], None] | None:
    """Build a console progress callback for the literal implementation.

    Parameters
    ----------
    progress_every:
        Number of completed time levels between progress messages. Values less
        than one return ``None`` and disable inner-loop progress messages.

    Returns
    -------
    callable | None
        Callback compatible with ``analytical_solution_legacy`` or ``None``.
    """
    progress_every = int(progress_every)
    if progress_every < 1:
        return None

    def report(completed: int, total: int, time_value: float) -> None:
        """Print one progress line if the current time level should be reported."""
        if completed == 1 or completed == total or completed % progress_every == 0:
            percent = 100.0 * completed / max(total, 1)
            print(
                "  Literal progress: "
                f"{completed}/{total} time levels ({percent:.1f}%), t={time_value:.6g}",
                flush=True,
            )

    return report


def generate_literal_reference(config: dict[str, Any], progress_every: int = 100) -> tuple[object, object, object]:
    """Generate the slow literal analytical reference field.

    Parameters
    ----------
    config:
        Reference configuration mapping.
    progress_every:
        Number of completed time levels between progress messages. Values less
        than one disable inner-loop progress messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Spatial grid, temporal grid, and solution matrix with shape ``(nx, nt)``.

    Notes
    -----
    This implementation constructs all literal partial sums before applying the
    last-K average. It is retained as a reference implementation and is expected
    to be much slower than the optimized workflow.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]

    print("Literal reference stage 1/3: building the space-time grid.")
    x, t = make_space_time_grid(
        benchmark.length,
        int(grid["n_space"]),
        float(grid["final_time"]),
        float(grid["dt"]),
    )

    progress_callback = make_literal_progress_callback(progress_every)
    print(
        "Literal reference stage 2/3: evaluating literal partial sums "
        f"for {t.size} time levels."
    )
    u = analytical_solution_legacy(
        x,
        t,
        benchmark.velocity,
        benchmark.diffusivity,
        benchmark.length,
        benchmark.u_left,
        benchmark.u_right,
        n_terms=int(ref["n_terms"]),
        averaging_width=int(ref["averaging_width"]),
        progress_callback=progress_callback,
    )
    print("Literal reference stage 3/3: solution matrix completed.")
    return x, t, u


def build_reference_metadata(
    config: dict[str, Any],
    implementation: str,
    x: object,
    t: object,
    u: object,
) -> dict[str, Any]:
    """Build metadata for an analytical reference output.

    Parameters
    ----------
    config:
        Reference configuration mapping.
    implementation:
        Implementation name stored in the metadata.
    x, t, u:
        Spatial grid, temporal grid, and solution matrix.

    Returns
    -------
    dict[str, Any]
        Serializable metadata including physical parameters, grid parameters,
        truncation settings, and endpoint consistency errors.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    return {
        "implementation": implementation,
        "length": benchmark.length,
        "u_left": benchmark.u_left,
        "u_right": benchmark.u_right,
        "velocity": benchmark.velocity,
        "diffusivity": benchmark.diffusivity,
        "peclet": benchmark.peclet,
        "n_space": int(grid["n_space"]),
        "n_time": int(len(t)),
        "dt": float(grid["dt"]),
        "final_time": float(grid["final_time"]),
        "n_terms": int(ref["n_terms"]),
        "averaging_width": int(ref["averaging_width"]),
        **consistency_errors(
            x,
            t,
            u,
            benchmark.velocity,
            benchmark.diffusivity,
            benchmark.length,
            benchmark.u_left,
            benchmark.u_right,
        ),
    }


def save_reference_outputs(df: object, metadata: dict[str, Any], data_path: Path, metadata_path: Path) -> None:
    """Write reference data and metadata files.

    Parameters
    ----------
    df:
        Long-form solution DataFrame with columns ``t``, ``x``, and ``u``.
    metadata:
        Serializable metadata mapping.
    data_path:
        Output parquet path.
    metadata_path:
        Output JSON path.
    """
    save_parquet(df, data_path)
    save_json(metadata, metadata_path)


def main() -> None:
    """Run the selected analytical reference generation workflow."""
    args = parse_args()
    config = load_reference_config(args.config)
    data_path, metadata_path, _ = build_reference_paths(config, args.implementation)

    if should_skip([data_path, metadata_path], overwrite=args.overwrite):
        print_skip_message([data_path, metadata_path], ROOT)
        return

    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    ref = config["reference"]
    print(
        "Generating analytical reference: "
        f"implementation={args.implementation}, nx={int(grid['n_space'])}, "
        f"dt={float(grid['dt'])}, T={float(grid['final_time'])}, "
        f"Pe={benchmark.peclet:.3f}, N={int(ref['n_terms'])}, K={int(ref['averaging_width'])}"
    )

    if args.implementation == "optimized":
        x, t, u = generate_optimized_reference(config)
    else:
        x, t, u = generate_literal_reference(config, progress_every=args.progress_every)

    print("Preparing long-form output table.")
    df = solution_to_dataframe(x, t, u)
    metadata = build_reference_metadata(config, args.implementation, x, t, u)
    save_reference_outputs(df, metadata, data_path, metadata_path)

    print(f"Saved reference data to {data_path.relative_to(ROOT)}")
    print(f"Saved reference metadata to {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
