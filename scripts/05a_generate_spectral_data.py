"""Generate late-time spectral-filter data from progressive PINN solutions.

The stage reads the saved solution for every progressive PINN window, performs
the cutoff sweep using only inter-filter sensitivity, and writes the selected
filtered field and spectrum. It does not use the analytical reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.spectral import (  # noqa: E402
    build_spectral_output_paths,
    run_spectral_data_generation,
)
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    print_skip_message,
    read_yaml,
    reject_partial_outputs,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate late-time spectral-filter data.")
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


def main() -> None:
    """Run the spectral data-generation stage."""
    args = parse_args()
    config = load_config(args.config)
    pinn_config_path = config.get("inputs", {}).get("pinn_config", "configs/pinn.yaml")
    pinn_config = load_config(pinn_config_path)
    paths = build_spectral_output_paths(config, ROOT)
    expected = [paths.sweep, paths.filtered_solution, paths.spectrum, paths.generation_metadata]
    reject_partial_outputs(expected, overwrite=args.overwrite)
    if should_skip(expected, overwrite=args.overwrite):
        print_skip_message(expected, ROOT)
        return

    metadata = run_spectral_data_generation(config, pinn_config, root=ROOT)
    selected = metadata["selected_pair"]
    global_minimum = metadata["global_minimum"]
    status = str(metadata.get("status", "complete"))
    processed_final_time = float(metadata["grid"]["processed_final_time"])
    configured_final_time = float(metadata["grid"]["configured_final_time"])
    completed_windows = int(metadata["sweep"]["completed_windows"])
    expected_windows = int(metadata["sweep"]["expected_windows"])
    print(
        "Spectral data source: "
        f"{completed_windows}/{expected_windows} completed PINN windows through "
        f"t={processed_final_time:.3f} of configured t={configured_final_time:.3f} "
        f"({status})."
    )
    print("Saved spectral data outputs:")
    for path in expected:
        print(f"  {path.relative_to(ROOT)}")
    print(
        "Global sensitivity minimum: "
        f"tmax={global_minimum['tmax']:.3f}, "
        f"tcut={global_minimum['tcut']:.3f}, "
        f"Esens={global_minimum['inter_filter_sensitivity']:.6e}"
    )
    reduced_sensitivity = metadata.get("reduced_sensitivity_analysis")
    if reduced_sensitivity is not None:
        print(
            "BIC-selected sensitivity change point: "
            f"tmax={reduced_sensitivity['tmax']:.3f}, "
            f"tau={reduced_sensitivity['tau']:.3f}, "
            f"BIC(M0)={reduced_sensitivity['M0']['bic']:.6f}, "
            f"BIC(M1)={reduced_sensitivity['M1']['bic']:.6f}, "
            f"delta_BIC={reduced_sensitivity['delta_bic_m1_minus_m0']:.6f}, "
            f"slopes=({reduced_sensitivity['M1']['slope_before']:.6e}, "
            f"{reduced_sensitivity['M1']['slope_after']:.6e})"
        )
    horizon_validation = metadata.get("reduced_sensitivity_horizon_validation")
    if horizon_validation is not None:
        path_rows = horizon_validation.get("reduced_sensitivity_path", [])
        if path_rows:
            path_summary = ", ".join(
                f"{float(row['tmax']):.1f}->{float(row['tcut']):.1f}"
                for row in path_rows
            )
        else:
            path_summary = "no M1-supported horizons"
        print(
            "Reduced-sensitivity onset validation across tmax: "
            f"{path_summary}; "
            f"status_counts={horizon_validation.get('status_counts', {})}"
        )
    print(
        "Selected reduced-sensitivity reconstruction: "
        f"tmax={selected['tmax']:.3f}, "
        f"tcut={selected['tcut']:.3f}, "
        f"fc={selected['cutoff_frequency']:.6e}, "
        f"retained={selected['achieved_energy_fraction']:.6f}"
    )


if __name__ == "__main__":
    main()
