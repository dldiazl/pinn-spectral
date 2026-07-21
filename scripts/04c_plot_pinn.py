"""Generate PINN comparison figures from postprocessed CSV files.

This script performs no training and does not recompute numerical errors. It
reads only the outputs of ``04b_postprocess_pinn.py`` and writes the figures
corresponding to the PINN error and profile comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    compute_pooled_log_ylim,
    error_space_source_paths,
    error_time_source_paths,
    plot_grouped_error_curves,
    plot_solution_profiles_with_zooms,
    plot_training_loss_history,
    print_skip_message,
    read_yaml,
    reject_partial_outputs,
    require_file,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot PINN comparison figures.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/pinn.yaml",
        help="Path to the PINN configuration YAML file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the PINN configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return read_yaml(path)


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON metadata object."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def main() -> None:
    """Generate PINN comparison figures and the training-loss diagnostic."""
    args = parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    postprocess = config.get("postprocess", {})
    postprocess_dir = resolve_path(outputs.get("postprocess_dir", "results/postprocess/pinn"))
    figure_dir = resolve_path(outputs.get("figure_dir", "results/figures/pinn"))
    metrics_dir = resolve_path(outputs.get("metrics_dir", "results/metrics"))

    time_csv = postprocess_dir / "pinn_error_time.csv"
    space_csv = postprocess_dir / "pinn_error_space.csv"
    profiles_csv = postprocess_dir / "pinn_profiles.csv"
    history_csv = postprocess_dir / "pinn_training_history.csv"
    metadata_json = metrics_dir / "pinn_diagnostics_metadata.json"
    for path in [time_csv, space_csv, profiles_csv, history_csv, metadata_json]:
        require_file(path, "Run: python scripts\\04b_postprocess_pinn.py --overwrite")

    figures = {
        "fig01": figure_dir / "fig01_pinn_error_time.png",
        "fig02": figure_dir / "fig02_pinn_error_space.png",
        "fig03": figure_dir / "fig03_pinn_profiles.png",
        "fig04": figure_dir / "fig04_pinn_training_loss_history.png",
    }
    reject_partial_outputs(list(figures.values()), overwrite=args.overwrite)
    if should_skip(list(figures.values()), overwrite=args.overwrite):
        print_skip_message(list(figures.values()), ROOT)
        return

    plot_grouped_error_curves(
        input_path=time_csv,
        output_path=figures["fig01"],
        x_column="t",
        xlabel=r"$t$",
        ylabel=r"$E(t)$",
        ylim=compute_pooled_log_ylim(error_time_source_paths(ROOT)),
    )
    plot_grouped_error_curves(
        input_path=space_csv,
        output_path=figures["fig02"],
        x_column="x",
        xlabel=r"$x$",
        ylabel=r"$E(x)$",
        ylim=compute_pooled_log_ylim(error_space_source_paths(ROOT)),
    )

    metadata = read_json(metadata_json)
    profiles_df = pd.read_csv(profiles_csv)
    metadata_profile_times = metadata.get("profile_times")
    if isinstance(metadata_profile_times, list) and metadata_profile_times:
        profile_times = [float(value) for value in metadata_profile_times]
    else:
        profile_times = sorted(
            pd.to_numeric(profiles_df["t"], errors="coerce").dropna().unique().tolist()
        )

    zoom_points = [tuple(float(value) for value in pair) for pair in postprocess.get("zoom_points", [])]
    plot_solution_profiles_with_zooms(
        df=profiles_df,
        output_path=figures["fig03"],
        solution_order=[str(value) for value in postprocess.get("solution_order", [])] or None,
        reference_solution="Analytical",
        profile_times=profile_times or None,
        zoom_points=zoom_points,
        delta=postprocess.get("zoom_delta", 0.05),
        text_locations=postprocess.get("text_locations"),
        colors=postprocess.get("colors"),
        linewidth=0.5,
        markersize=1.0,
    )

    plot_training_loss_history(
        input_path=history_csv,
        output_path=figures["fig04"],
        x_column="global_epoch",
        value_column="loss_value",
        series_column="loss_name",
        label_column="label",
        xlabel="Global outer epoch",
        ylabel="Loss",
    )

    status = str(metadata.get("status", "unknown"))
    processed_final_time = metadata.get("processed_final_time")
    if processed_final_time is not None:
        print(
            "PINN figure source: "
            f"postprocessed interval 0 <= t <= {float(processed_final_time):.3f} "
            f"({status})."
        )
    print("Saved PINN figures:")
    for path in figures.values():
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
