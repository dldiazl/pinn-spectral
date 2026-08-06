"""Generate supervised NAS figures from saved postprocess files.

This script does not run Optuna or train neural networks. It reads files created
by ``03a_run_nas.py`` and ``03b_postprocess_nas.py`` and writes figures.

The Pareto and loss-history figures can be generated while the NAS optimization
is still running. If the final ``nas_trials.csv`` file does not exist yet, the
script builds a provisional trials table from the history parquet files that
have already been written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    compute_pooled_log_ylim,
    error_time_source_paths,
    load_or_build_nas_trials_table,
    plot_grouped_error_curves_from_dataframe,
    plot_pareto_front_from_dataframe,
    plot_solution_profiles_with_zooms,
    print_skip_message,
    read_yaml,
    require_file,
    save_figure,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with NAS configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Generate supervised NAS figures from saved data.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/nas.yaml",
        help="Path to the NAS configuration YAML file.",
    )
    parser.add_argument(
        "--skip-full-domain",
        action="store_true",
        help="Only plot NAS Pareto and training histories; skip figures that require 03b outputs.",
    )
    return parser.parse_args()


def load_nas_config(config_path: str | Path) -> dict[str, Any]:
    """Load the NAS YAML configuration.

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


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Build figure input and output paths.

    Parameters
    ----------
    config:
        NAS configuration mapping.

    Returns
    -------
    dict[str, Path]
        Named paths for this stage.
    """
    outputs = config["outputs"]
    training = config["training"]
    study_name = str(training["study_name"])

    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    postprocess_dir = ROOT / outputs.get("postprocess_dir", "results/postprocess/nas")
    figure_dir = ROOT / outputs.get("figure_dir", "results/figures/nas")
    histories_base = ROOT / outputs.get("histories_dir", "data/nas/histories")
    models_base = ROOT / outputs.get("models_dir", "results/models/nas")
    histories_dir = histories_base if histories_base.name == study_name else histories_base / study_name
    models_dir = models_base if models_base.name == study_name else models_base / study_name
    fdm_dir = ROOT / "results" / "postprocess" / "numerical"

    return {
        "trials_csv": metrics_dir / "nas_trials.csv",
        "partial_trials_csv": metrics_dir / "nas_trials_partial.csv",
        "histories_dir": histories_dir,
        "models_dir": models_dir,
        "profiles_csv": postprocess_dir / "nas_profiles.csv",
        "nas_error_time_csv": postprocess_dir / "nas_error_time.csv",
        "fdm_error_time_csv": fdm_dir / "fdm_error_time.csv",
        "fig01a": figure_dir / "fig01a_nas_pareto.png",
        "fig01b": figure_dir / "fig01b_nas_loss_histories.png",
        "fig02a": figure_dir / "fig02a_nas_profiles.png",
        "fig02b": figure_dir / "fig02b_nas_error_time.png",
    }


def load_trials_for_plotting(paths: dict[str, Path]) -> pd.DataFrame:
    """Load final or provisional NAS trials for plotting.

    Parameters
    ----------
    paths:
        Named paths returned by ``build_paths``.

    Returns
    -------
    pd.DataFrame
        Completed NAS trials table.
    """
    trials, source = load_or_build_nas_trials_table(
        trials_csv=paths["trials_csv"],
        histories_dir=paths["histories_dir"],
        models_dir=paths["models_dir"],
        partial_trials_csv=paths["partial_trials_csv"],
        root=ROOT,
        include_manual=False,
    )
    complete = trials[trials["state"] == "COMPLETE"].copy() if "state" in trials.columns else trials.copy()
    if complete.empty:
        raise ValueError("No completed NAS trials were found for plotting.")

    print(f"Using NAS trials source: {source}")
    return complete.reset_index(drop=True)


def plot_nas_pareto(trials: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the supervised NAS cloud, Pareto front, and selected trial.

    Parameters
    ----------
    trials:
        Completed NAS trials table, either final or provisional.
    output_path:
        Output figure path.
    """
    plot_data = trials.copy()
    if "final_mse" not in plot_data.columns:
        if "final_loss" in plot_data.columns:
            plot_data["final_mse"] = plot_data["final_loss"]
        else:
            raise ValueError("NAS trials table must contain final_mse for Pareto plotting.")

    plot_pareto_front_from_dataframe(
        plot_data,
        output_path=output_path,
        time_column="total_time_seconds",
        loss_column="final_mse",
        activation_column="activation",
        architecture_column="architecture_name",
        trial_column="trial_number",
        selected_label="Selected",
        xlabel="Training time [s]",
        ylabel="MSE",
    )


def read_history(path: str | Path) -> pd.DataFrame:
    """Read one training-history parquet file and normalize column names.

    Parameters
    ----------
    path:
        History parquet path.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Epoch`` and a plottable loss column.
    """
    history = pd.read_parquet(path)
    if "Epoch" not in history.columns:
        raise ValueError(f"{path} is missing column: Epoch")
    if "Loss" not in history.columns and "LossBC" not in history.columns:
        raise ValueError(f"{path} is missing both Loss and LossBC columns.")
    return history


def plot_nas_loss_histories(trials: pd.DataFrame, output_path: str | Path) -> None:
    """Plot supervised-loss histories for all completed NAS trials.

    Parameters
    ----------
    trials:
        Completed NAS trials table, either final or provisional.
    output_path:
        Output figure path.
    """
    complete = trials.copy()
    if "final_mse" not in complete.columns:
        if "final_loss" in complete.columns:
            complete["final_mse"] = complete["final_loss"]
        else:
            raise ValueError("NAS trials table must contain final_mse and history_path columns.")
    if "history_path" not in complete.columns:
        raise ValueError("NAS trials table must contain final_mse and history_path columns.")

    if "trial_number" not in complete.columns:
        complete["trial_number"] = np.arange(len(complete), dtype=int)
    if "architecture_name" not in complete.columns:
        complete["architecture_name"] = complete.index.astype(str)

    complete = complete.sort_values(["final_mse", "trial_number"], ascending=True, na_position="last").reset_index(drop=True)
    selected_history_path = ROOT / str(complete.loc[0, "history_path"])
    selected_name = str(complete.loc[0, "architecture_name"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted_all = False
    for _, row in complete.iterrows():
        history_path = ROOT / str(row["history_path"])
        if not history_path.exists():
            continue
        try:
            history = read_history(history_path)
        except Exception as exc:  # noqa: BLE001 - monitoring should tolerate partial files.
            print(f"Skipping history during plotting: {history_path} ({exc})")
            continue
        loss_column = "LossBC" if "LossBC" in history.columns else "Loss"
        label = "All NAS trials" if not plotted_all else None
        ax.semilogy(history["Epoch"], history[loss_column], color="0.55", linewidth=0.7, alpha=1.0, label=label)
        plotted_all = True

    if selected_history_path.exists():
        history = read_history(selected_history_path)
        loss_column = "LossBC" if "LossBC" in history.columns else "Loss"
        ax.semilogy(history["Epoch"], history[loss_column], color="black", linewidth=1.8, label=f"Selected: {selected_name}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    if plotted_all or selected_history_path.exists():
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(output_path)
    plt.close(fig)


def plot_nas_profiles(input_path: str | Path, output_path: str | Path) -> None:
    """Plot analytical and NAS profiles using the reusable profile-zoom tool.

    Parameters
    ----------
    input_path:
        Profile CSV written by ``03b_postprocess_nas.py``.
    output_path:
        Output figure path.
    """
    input_path = Path(input_path)
    require_file(input_path, "Run: python scripts\\03b_postprocess_nas.py --overwrite")
    df = pd.read_csv(input_path)
    plot_solution_profiles_with_zooms(
        df=df,
        output_path=output_path,
        solution_column="solution",
        time_column="t",
        x_column="x",
        value_column="u",
        solution_order=["Analytical", "NN"],
        reference_solution="Analytical",
        zoom_points=[(0.0, 0.5055), (1.0, 0.998)],
        delta=[0.007, 0.005],
        text_locations=[None, None, 0.65, 0.8, 0.9],
        colors={"Analytical": "royalblue", "NN": "crimson"},
        linewidth=1.2,
        markersize=2.5,
        figsize=(12.0, 5.0),
        show_legend=True,
    )


def plot_nas_error_time(nas_path: str | Path, fdm_path: str | Path, output_path: str | Path) -> None:
    """Plot NAS temporal error, optionally with finite-difference baselines.

    Parameters
    ----------
    nas_path:
        NAS temporal-error CSV written by ``03b_postprocess_nas.py``.
    fdm_path:
        Finite-difference temporal-error CSV from stage 02b. If it does not
        exist, only the NN curve is plotted.
    output_path:
        Output figure path.
    """
    nas_path = Path(nas_path)
    require_file(nas_path, "Run: python scripts\\03b_postprocess_nas.py --overwrite")
    frames = [pd.read_csv(nas_path)]
    fdm_path = Path(fdm_path)
    if fdm_path.exists():
        frames.append(pd.read_csv(fdm_path))
    else:
        print(f"Finite-difference error file not found; plotting NN only: {fdm_path}")
    df = pd.concat(frames, ignore_index=True)
    plot_grouped_error_curves_from_dataframe(
        df,
        output_path,
        x_column="t",
        xlabel=r"$t$",
        ylabel=r"$E(t)$",
        ylim=compute_pooled_log_ylim(error_time_source_paths(ROOT)),
    )


def main() -> None:
    """Run the NAS plotting script."""
    args = parse_args()
    config = load_nas_config(args.config)
    paths = build_paths(config)

    figure_paths = [paths["fig01a"], paths["fig01b"]]
    if not args.skip_full_domain:
        figure_paths.extend([paths["fig02a"], paths["fig02b"]])

    if should_skip(figure_paths, overwrite=args.overwrite):
        print_skip_message(figure_paths, ROOT)
        return

    trials = load_trials_for_plotting(paths)
    plot_nas_pareto(trials, paths["fig01a"])
    plot_nas_loss_histories(trials, paths["fig01b"])

    if args.skip_full_domain:
        print("Skipping full-domain NAS figures because --skip-full-domain was used.")
    else:
        plot_nas_profiles(paths["profiles_csv"], paths["fig02a"])
        plot_nas_error_time(paths["nas_error_time_csv"], paths["fdm_error_time_csv"], paths["fig02b"])

    print("Saved NAS figures:")
    for path in figure_paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
