"""Generate spectral figures focused on E_sens and reference-RMSE diagnostics."""

from __future__ import annotations

import argparse
import json
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

from pinn_spectral.spectral import build_spectral_output_paths  # noqa: E402
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    compute_pooled_log_ylim,
    error_space_source_paths,
    error_time_source_paths,
    plot_grouped_error_curves_from_dataframe,
    plot_log_heatmap_from_dataframe,
    print_skip_message,
    read_yaml,
    reject_partial_outputs,
    require_file,
    should_skip,
)


DEFAULT_TIME_CURVE_ORDER = [
    "CDS_EF",
    "CDS_CN",
    "CompactSchemes_CN",
    "PINN",
    "PINN_Filtered",
]
DEFAULT_SPACE_CURVE_ORDER = DEFAULT_TIME_CURVE_ORDER.copy()
DEFAULT_RMSE_TIME_CURVE_ORDER = [
    "CDS_EF",
    "CDS_CN",
    "CompactSchemes_CN",
    "PINN",
    "PINN_Filtered_RMSE_Path",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot late-time spectral diagnostics.")
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


def ordered_methods(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Return grouped curve rows in a configured stable method order."""
    if not order:
        return df
    rank = {name: index for index, name in enumerate(order)}
    result = df.copy()
    result["_order"] = result["method"].map(rank).fillna(len(rank)).astype(int)
    return result.sort_values(["_order"]).drop(columns="_order")


def complete_method_order(
    configured_order: list[str],
    required_methods: list[str],
) -> list[str]:
    """Return a stable order that always contains every required curve."""
    result: list[str] = []
    for method in [*configured_order, *required_methods]:
        if method not in result:
            result.append(method)
    return result


def require_methods(
    df: pd.DataFrame,
    required_methods: list[str],
    context: str,
) -> None:
    """Validate that a plot-ready table contains all required methods."""
    available = set(df["method"].astype(str).unique())
    missing = [method for method in required_methods if method not in available]
    if missing:
        raise ValueError(
            f"{context} is missing required methods: {missing}. "
            "Run: python scripts\\05b_postprocess_spectral.py --overwrite"
        )



def plot_minimum_path_value_comparison(
    sensitivity_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_path: Path,
    xlim: tuple[float, float],
) -> None:
    """Compare minimum sensitivity and reference-RMSE path values versus ``tmax``.

    This plot preserves the original diagnostic that compares the shapes of
    the two minimum-value paths. It is descriptive only: the reference-RMSE
    path does not participate in the reference-free cutoff selection.
    """
    sensitivity_required = {
        "tmax",
        "inter_filter_sensitivity",
        "is_minimum_for_tmax",
    }
    reference_required = {
        "tmax",
        "filtered_window_rmse",
        "is_minimum_reference_rmse_for_tmax",
    }
    missing_sensitivity = sensitivity_required.difference(sensitivity_df.columns)
    missing_reference = reference_required.difference(reference_df.columns)
    if missing_sensitivity:
        raise ValueError(
            "spectral_sensitivity.csv is missing minimum-path columns: "
            f"{sorted(missing_sensitivity)}."
        )
    if missing_reference:
        raise ValueError(
            "spectral_reference_window_sweep.csv is missing minimum-path columns: "
            f"{sorted(missing_reference)}."
        )

    sensitivity_path = sensitivity_df.loc[
        sensitivity_df["is_minimum_for_tmax"].astype(bool),
        ["tmax", "inter_filter_sensitivity"],
    ].copy()
    reference_path = reference_df.loc[
        reference_df["is_minimum_reference_rmse_for_tmax"].astype(bool),
        ["tmax", "filtered_window_rmse"],
    ].copy()

    for column in ["tmax", "inter_filter_sensitivity"]:
        sensitivity_path[column] = pd.to_numeric(
            sensitivity_path[column], errors="coerce"
        )
    for column in ["tmax", "filtered_window_rmse"]:
        reference_path[column] = pd.to_numeric(
            reference_path[column], errors="coerce"
        )

    sensitivity_path = sensitivity_path.dropna().sort_values("tmax")
    reference_path = reference_path.dropna().sort_values("tmax")
    sensitivity_path = sensitivity_path[
        sensitivity_path["inter_filter_sensitivity"] > 0.0
    ]
    reference_path = reference_path[reference_path["filtered_window_rmse"] > 0.0]
    if sensitivity_path.empty or reference_path.empty:
        raise ValueError(
            "Minimum-path comparison requires positive finite sensitivity and RMSE values."
        )
    if sensitivity_path.duplicated("tmax").any():
        raise ValueError("The minimum-sensitivity path contains duplicate tmax values.")
    if reference_path.duplicated("tmax").any():
        raise ValueError("The minimum-reference-RMSE path contains duplicate tmax values.")

    fig, left_axis = plt.subplots(figsize=(7.5, 4.8))
    right_axis = left_axis.twinx()

    sensitivity_line = left_axis.plot(
        sensitivity_path["tmax"],
        sensitivity_path["inter_filter_sensitivity"],
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label=r"Minimum $E_{\mathrm{sens}}$ path value",
    )[0]
    rmse_line = right_axis.plot(
        reference_path["tmax"],
        reference_path["filtered_window_rmse"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="tab:orange",
        label="Minimum reference RMSE path value",
    )[0]

    left_axis.set_xlabel(r"$t_{\max}$")
    left_axis.set_ylabel(r"Minimum $E_{\mathrm{sens}}$", color="tab:blue")
    right_axis.set_ylabel("Minimum reference RMSE", color="tab:orange")
    left_axis.set_yscale("log")
    right_axis.set_yscale("log")
    left_axis.tick_params(axis="y", colors="tab:blue")
    right_axis.tick_params(axis="y", colors="tab:orange")
    left_axis.spines["left"].set_color("tab:blue")
    right_axis.spines["right"].set_color("tab:orange")
    left_axis.set_xlim(*xlim)
    left_axis.grid(True, which="major", linewidth=0.5, alpha=0.4)
    left_axis.legend(
        [sensitivity_line, rmse_line],
        [sensitivity_line.get_label(), rmse_line.get_label()],
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_final_time_reduced_sensitivity_profile(
    sensitivity_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_path: Path,
    selected_tmax: float,
    selected_tcut: float,
    rmse_tcut: float,
    xlim: tuple[float, float],
) -> None:
    """Plot the final-time sensitivity change point and reference-RMSE profile."""
    sensitivity_required = {
        "tmax",
        "tcut",
        "inter_filter_sensitivity",
        "reduced_sensitivity_m0_fitted_log10_sensitivity",
        "reduced_sensitivity_m1_fitted_log10_sensitivity",
        "reduced_sensitivity_m0_bic",
        "reduced_sensitivity_m1_bic",
    }
    reference_required = {"tmax", "tcut", "filtered_window_rmse"}
    missing_sensitivity = sensitivity_required.difference(sensitivity_df.columns)
    missing_reference = reference_required.difference(reference_df.columns)
    if missing_sensitivity:
        raise ValueError(
            "spectral_sensitivity.csv is missing reduced-sensitivity-analysis columns: "
            f"{sorted(missing_sensitivity)}. "
            "Run: python scripts\05a_generate_spectral_data.py --overwrite"
        )
    if missing_reference:
        raise ValueError(
            "spectral_reference_window_sweep.csv is missing required columns: "
            f"{sorted(missing_reference)}."
        )

    sensitivity = sensitivity_df.copy()
    reference = reference_df.copy()
    for column in [
        "tmax",
        "tcut",
        "inter_filter_sensitivity",
        "reduced_sensitivity_m0_fitted_log10_sensitivity",
        "reduced_sensitivity_m1_fitted_log10_sensitivity",
        "reduced_sensitivity_m0_bic",
        "reduced_sensitivity_m1_bic",
    ]:
        sensitivity[column] = pd.to_numeric(sensitivity[column], errors="coerce")
    for column in ["tmax", "tcut", "filtered_window_rmse"]:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")

    sensitivity = sensitivity.loc[
        np.isclose(
            sensitivity["tmax"].to_numpy(dtype=float),
            float(selected_tmax),
            rtol=0.0,
            atol=1e-12,
        )
    ].dropna(
        subset=[
            "tcut",
            "inter_filter_sensitivity",
            "reduced_sensitivity_m0_fitted_log10_sensitivity",
            "reduced_sensitivity_m1_fitted_log10_sensitivity",
        ]
    )
    sensitivity = sensitivity[
        sensitivity["inter_filter_sensitivity"] > 0.0
    ].sort_values("tcut")
    if sensitivity.empty:
        raise ValueError(
            f"No fitted reduced-sensitivity profile is available at tmax={selected_tmax:.3f}."
        )

    reference = reference.loc[
        np.isclose(
            reference["tmax"].to_numpy(dtype=float),
            float(selected_tmax),
            rtol=0.0,
            atol=1e-12,
        )
    ].dropna(subset=["tcut", "filtered_window_rmse"])
    reference = reference[reference["filtered_window_rmse"] > 0.0].sort_values(
        "tcut"
    )
    if reference.empty:
        raise ValueError(
            f"No positive reference-RMSE profile is available at tmax={selected_tmax:.3f}."
        )

    m0_bic = float(sensitivity["reduced_sensitivity_m0_bic"].dropna().iloc[0])
    m1_bic = float(sensitivity["reduced_sensitivity_m1_bic"].dropna().iloc[0])
    delta_bic = m1_bic - m0_bic

    fig, left_axis = plt.subplots(figsize=(7.5, 4.8))
    right_axis = left_axis.twinx()

    sensitivity_points = left_axis.plot(
        sensitivity["tcut"],
        sensitivity["inter_filter_sensitivity"],
        marker="o",
        linewidth=0.0,
        color="tab:blue",
        label=rf"$E_{{\mathrm{{sens}}}}$ at $t_{{\max}}={selected_tmax:g}$",
    )[0]
    m0_line = left_axis.plot(
        sensitivity["tcut"],
        10.0
        ** sensitivity["reduced_sensitivity_m0_fitted_log10_sensitivity"].to_numpy(
            dtype=float
        ),
        linestyle=":",
        linewidth=1.6,
        color="0.4",
        label="M0: single log-linear trend",
    )[0]
    m1_line = left_axis.plot(
        sensitivity["tcut"],
        10.0
        ** sensitivity["reduced_sensitivity_m1_fitted_log10_sensitivity"].to_numpy(
            dtype=float
        ),
        linewidth=2.0,
        color="tab:blue",
        label="M1: two-regime log-linear trend",
    )[0]
    rmse_line = right_axis.plot(
        reference["tcut"],
        reference["filtered_window_rmse"],
        marker="s",
        linestyle="--",
        linewidth=1.6,
        color="tab:orange",
        label="Reference RMSE",
    )[0]
    reduced_sensitivity_marker = left_axis.axvline(
        float(selected_tcut),
        linestyle="--",
        linewidth=1.2,
        color="tab:blue",
        label=r"BIC-selected reduced-sensitivity onset $t_{\mathrm{cut}}^{\mathrm{RS}}$",
    )
    rmse_marker = left_axis.axvline(
        float(rmse_tcut),
        linestyle="-.",
        linewidth=1.2,
        color="tab:orange",
        label=r"$t_{\mathrm{cut}}^{\mathrm{RMSE}}$",
    )

    left_axis.set_xlabel(r"$t_{\mathrm{cut}}$")
    left_axis.set_ylabel(r"$E_{\mathrm{sens}}(t_{\mathrm{cut}}, t_{\max})$")
    right_axis.set_ylabel(r"Reference $\mathrm{RMSE}_{[t_{\mathrm{cut}},t_{\max}]}$")
    left_axis.tick_params(axis="y", colors="tab:blue")
    right_axis.tick_params(axis="y", colors="tab:orange")
    left_axis.spines["left"].set_color("tab:blue")
    right_axis.spines["right"].set_color("tab:orange")
    left_axis.set_yscale("log")
    right_axis.set_yscale("log")
    left_axis.set_xlim(*xlim)
    left_axis.grid(True, linewidth=0.5, alpha=0.4)
    left_axis.text(
        0.02,
        0.03,
        rf"$\Delta\mathrm{{BIC}}_{{M1-M0}}={delta_bic:.2f}$",
        transform=left_axis.transAxes,
        fontsize=9,
    )
    handles = [
        sensitivity_points,
        m0_line,
        m1_line,
        rmse_line,
        reduced_sensitivity_marker,
        rmse_marker,
    ]
    left_axis.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="best",
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate spectral sensitivity, reference validation, and error figures."""
    args = parse_args()
    config = load_config(args.config)
    paths = build_spectral_output_paths(config, ROOT)
    postprocess = config.get("postprocess", {})
    validation_enabled = bool(config.get("reference_validation", {}).get("enabled", True))

    sensitivity_csv = paths.postprocess_dir / "spectral_sensitivity.csv"
    time_csv = paths.postprocess_dir / "spectral_error_time.csv"
    rmse_path_time_csv = paths.postprocess_dir / "spectral_error_time_rmse_path.csv"
    space_csv = paths.postprocess_dir / "spectral_error_space.csv"
    required_inputs = [
        sensitivity_csv,
        time_csv,
        space_csv,
        paths.postprocess_metadata,
    ]
    if validation_enabled:
        required_inputs.extend([paths.reference_window_sweep, rmse_path_time_csv])
    for path in required_inputs:
        require_file(path, "Run: python scripts\\05b_postprocess_spectral.py --overwrite")

    figures = {
        "fig01": paths.figure_dir / "fig01_spectral_sensitivity.png",
    }
    if validation_enabled:
        figures["fig02"] = paths.figure_dir / "fig02_filtered_reference_window_rmse.png"
        figures["fig03"] = paths.figure_dir / "fig03_esens_reference_rmse_path_values.png"
        figures["fig04"] = paths.figure_dir / "fig04_reduced_sensitivity_analysis.png"
    figures["fig05"] = paths.figure_dir / "fig05_filtered_error_time.png"
    if validation_enabled:
        figures["fig06"] = paths.figure_dir / "fig06_filtered_error_time_rmse_path.png"
    figures["fig07"] = paths.figure_dir / "fig07_filtered_error_space.png"
    legacy_figures = [
        paths.figure_dir / "fig02_adjacent_training_discrepancy.png",
        paths.figure_dir / "fig08_filtered_error_space.png",
        paths.figure_dir / "fig12c_esens_path_filtered_samples.png",
    ]
    if args.overwrite:
        for legacy_path in legacy_figures:
            legacy_path.unlink(missing_ok=True)
    reject_partial_outputs(list(figures.values()), overwrite=args.overwrite)
    if should_skip(list(figures.values()), overwrite=args.overwrite):
        print_skip_message(list(figures.values()), ROOT)
        return

    with paths.postprocess_metadata.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    summary = metadata["summary"]
    selected_tmax = float(summary["selected_tmax"])
    selected_tcut = float(summary["selected_tcut"])
    rmse_path_tcut = float(summary["rmse_path_tcut"]) if validation_enabled else None
    processed_final_time = float(summary.get("processed_final_time", summary["selected_tmax"]))
    configured_final_time = float(summary.get("configured_final_time", processed_final_time))
    status = str(metadata.get("status", "complete"))

    configured_xlim = tuple(
        float(value)
        for value in postprocess.get("sensitivity_xlim", [0.0, configured_final_time])
    )
    configured_ylim = tuple(
        float(value)
        for value in postprocess.get("sensitivity_ylim", [0.0, configured_final_time])
    )
    if bool(postprocess.get("adapt_axes_to_processed_time", True)):
        sensitivity_xlim = (configured_xlim[0], min(configured_xlim[1], processed_final_time))
        sensitivity_ylim = (configured_ylim[0], min(configured_ylim[1], processed_final_time))
    else:
        sensitivity_xlim = configured_xlim
        sensitivity_ylim = configured_ylim

    sensitivity = pd.read_csv(sensitivity_csv)
    plot_log_heatmap_from_dataframe(
        df=sensitivity,
        output_path=figures["fig01"],
        x_column="tmax",
        y_column="tcut",
        value_column="inter_filter_sensitivity",
        xlabel=r"$t_{\max}$",
        ylabel=r"$t_{\mathrm{cut}}$",
        colorbar_label=r"$E_{\mathrm{sens}}(t_{\mathrm{cut}},t_{\max})$",
        path_flag_column="is_reduced_sensitivity_onset",
        path_label=r"Reduced-sensitivity onset path",
        secondary_path_flag_column="is_minimum_for_tmax",
        secondary_path_label=r"Minimum $E_{\mathrm{sens}}$ path",
        marker_flag_column="is_selected",
        marker_label=r"Final selected cutoff",
        cmap=str(postprocess.get("sensitivity_cmap", "coolwarm")),
        xlim=sensitivity_xlim,
        ylim=sensitivity_ylim,
    )

    if validation_enabled:
        reference_sweep = pd.read_csv(paths.reference_window_sweep)
        plot_log_heatmap_from_dataframe(
            df=reference_sweep,
            output_path=figures["fig02"],
            x_column="tmax",
            y_column="tcut",
            value_column="filtered_window_rmse",
            xlabel=r"$t_{\max}$",
            ylabel=r"$t_{\mathrm{cut}}$",
            colorbar_label=r"$\mathrm{RMSE}_{[t_{\mathrm{cut}},t_{\max}]}$",
            path_flag_column="is_reduced_sensitivity_onset_for_tmax",
            path_label=r"Reduced-sensitivity onset path",
            secondary_path_flag_column="is_minimum_reference_rmse_for_tmax",
            secondary_path_label="Minimum reference RMSE path",
            marker_flag_column="is_selected_by_sensitivity",
            marker_label="Final selected cutoff",
            cmap=str(postprocess.get("reference_rmse_cmap", "viridis")),
            xlim=sensitivity_xlim,
            ylim=sensitivity_ylim,
        )
        plot_minimum_path_value_comparison(
            sensitivity_df=sensitivity,
            reference_df=reference_sweep,
            output_path=figures["fig03"],
            xlim=sensitivity_xlim,
        )
        plot_final_time_reduced_sensitivity_profile(
            sensitivity_df=sensitivity,
            reference_df=reference_sweep,
            output_path=figures["fig04"],
            selected_tmax=selected_tmax,
            selected_tcut=selected_tcut,
            rmse_tcut=float(rmse_path_tcut),
            xlim=sensitivity_ylim,
        )

    time_ylim = compute_pooled_log_ylim(error_time_source_paths(ROOT))
    space_ylim = compute_pooled_log_ylim(error_space_source_paths(ROOT))

    time_raw = pd.read_csv(time_csv)
    require_methods(time_raw, DEFAULT_TIME_CURVE_ORDER, "spectral_error_time.csv")
    time_order = complete_method_order(
        [str(value) for value in postprocess.get("time_curve_order", [])],
        DEFAULT_TIME_CURVE_ORDER,
    )
    time_df = ordered_methods(time_raw, time_order)
    plot_grouped_error_curves_from_dataframe(
        df=time_df,
        output_path=figures["fig05"],
        x_column="t",
        xlabel=r"$t$",
        ylabel=r"$E(t)$",
        ylim=time_ylim,
        vertical_markers=[
            {
                "x": selected_tcut,
                "label": r"$t_{\mathrm{cut}}^{\mathrm{RS}}$",
                "linestyle": "--",
                "linewidth": 1.0,
            }
        ],
    )

    if validation_enabled:
        rmse_time_raw = pd.read_csv(rmse_path_time_csv)
        require_methods(
            rmse_time_raw,
            DEFAULT_RMSE_TIME_CURVE_ORDER,
            "spectral_error_time_rmse_path.csv",
        )
        rmse_time_order = complete_method_order(
            [str(value) for value in postprocess.get("rmse_time_curve_order", [])],
            DEFAULT_RMSE_TIME_CURVE_ORDER,
        )
        rmse_time_df = ordered_methods(rmse_time_raw, rmse_time_order)
        plot_grouped_error_curves_from_dataframe(
            df=rmse_time_df,
            output_path=figures["fig06"],
            x_column="t",
            xlabel=r"$t$",
            ylabel=r"$E(t)$",
            ylim=time_ylim,
            vertical_markers=[
                {
                    "x": float(rmse_path_tcut),
                    "label": r"$t_{\mathrm{cut}}^{\mathrm{RMSE}}$",
                    "linestyle": "--",
                    "linewidth": 1.0,
                }
            ],
        )

    space_raw = pd.read_csv(space_csv)
    require_methods(space_raw, DEFAULT_SPACE_CURVE_ORDER, "spectral_error_space.csv")
    space_order = complete_method_order(
        [str(value) for value in postprocess.get("space_curve_order", [])],
        DEFAULT_SPACE_CURVE_ORDER,
    )
    space_df = ordered_methods(space_raw, space_order)
    plot_grouped_error_curves_from_dataframe(
        df=space_df,
        output_path=figures["fig07"],
        x_column="x",
        xlabel=r"$x$",
        ylabel=r"$E(x)$",
        ylim=space_ylim,
    )

    print(
        "Spectral figure source: "
        f"PINN sweep through t={processed_final_time:.3f} of configured "
        f"t={configured_final_time:.3f} ({status})."
    )
    print("Saved spectral figures:")
    for path in figures.values():
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
