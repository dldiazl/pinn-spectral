"""Postprocess the selected supervised NAS network over the full time domain.

The supervised NAS is trained only on ``0 <= t <= 1``. This script deliberately
evaluates the selected network on the complete reference grid ``0 <= t <= 5`` so
that extrapolation beyond the supervised interval can be quantified.

The script can also be used while the NAS optimization is still running. If the
final ``nas_trials.csv`` file does not exist yet, it builds a provisional trials
table from the history parquet files that have already been written.
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
    load_or_build_nas_trials_table,
    output_name,
    predict_with_model,
    print_skip_message,
    read_solution_matrix,
    read_yaml,
    require_file,
    save_csv,
    save_json,
    save_parquet,
    should_skip,
    solution_dataframe_from_matrix,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with NAS configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Postprocess the selected NAS network on the full grid.")
    add_overwrite_argument(parser)
    parser.add_argument(
        "--config",
        default="configs/nas.yaml",
        help="Path to the NAS configuration YAML file.",
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
    """Build input and output paths for NAS postprocessing.

    Parameters
    ----------
    config:
        NAS configuration mapping.

    Returns
    -------
    dict[str, Path]
        Named paths used by this stage.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    outputs = config["outputs"]
    training = config["training"]

    n_space = int(grid["n_space"])
    final_time = float(grid["final_time"])
    dt = float(grid["dt"])
    pe = benchmark.peclet
    study_name = str(training["study_name"])

    reference_stem = output_name("Analytical", n_space, dt, final_time, pe)
    nn_stem = output_name("NN", n_space, dt, final_time, pe)

    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    postprocess_dir = ROOT / outputs.get("postprocess_dir", "results/postprocess/nas")
    data_dir = ROOT / outputs.get("data_dir", "data/nas")
    histories_base = ROOT / outputs.get("histories_dir", "data/nas/histories")
    models_base = ROOT / outputs.get("models_dir", "results/models/nas")
    histories_dir = histories_base if histories_base.name == study_name else histories_base / study_name
    models_dir = models_base if models_base.name == study_name else models_base / study_name

    return {
        "reference_path": ROOT / "data" / "reference" / f"{reference_stem}.parquet",
        "trials_csv": metrics_dir / "nas_trials.csv",
        "partial_trials_csv": metrics_dir / "nas_trials_partial.csv",
        "histories_dir": histories_dir,
        "models_dir": models_dir,
        "nn_data_path": data_dir / f"{nn_stem}.parquet",
        "time_csv": postprocess_dir / "nas_error_time.csv",
        "space_csv": postprocess_dir / "nas_error_space.csv",
        "profiles_csv": postprocess_dir / "nas_profiles.csv",
        "metadata_json": metrics_dir / "nas_selected_model_metadata.json",
    }


def select_trial(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Select the best currently available NAS trial.

    The final ``nas_trials.csv`` is used when it exists. Otherwise, the function
    builds a provisional trials table from completed history parquet files.

    Parameters
    ----------
    config:
        NAS configuration mapping.
    paths:
        Input and output paths for this stage.

    Returns
    -------
    dict[str, Any]
        Selected trial row.
    """
    del config  # The parameter is kept for a stable call signature.

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
        raise ValueError("No completed NAS trials were found.")

    if "final_mse" not in complete.columns:
        if "final_loss" in complete.columns:
            complete["final_mse"] = complete["final_loss"]
        else:
            raise ValueError("The NAS trials table must contain a 'final_mse' column.")

    # Full-domain postprocessing requires a saved model. If a history exists but
    # the corresponding model is missing, that trial is valid for monitoring
    # curves but cannot be selected for prediction on the reference grid.
    if "model_path" not in complete.columns:
        complete["model_path"] = ""
    complete["model_path"] = complete["model_path"].fillna("").astype(str)
    complete["model_exists"] = complete["model_path"].apply(lambda value: bool(value) and (ROOT / value).exists())
    selectable = complete[complete["model_exists"]].copy()
    if selectable.empty:
        raise FileNotFoundError(
            "Completed NAS histories were found, but none of them has a matching saved model.\n"
            f"Histories directory: {paths['histories_dir']}\n"
            f"Models directory: {paths['models_dir']}"
        )

    sort_columns = ["final_mse"]
    if "trial_number" in selectable.columns:
        sort_columns.append("trial_number")
    selectable = selectable.sort_values(sort_columns, ascending=True, na_position="last")

    selected = selectable.iloc[0].to_dict()
    selected["selection_source"] = source

    print("Selected NAS trial:")
    for key in [
        "selection_source",
        "trial_number",
        "architecture_name",
        "activation",
        "num_layers",
        "neurons_per_layer",
        "epochs",
        "final_mse",
        "final_scaled_loss",
        "total_time_seconds",
        "history_path",
        "model_path",
    ]:
        if key in selected:
            print(f"  {key}: {selected[key]}")

    return selected



def compute_profiles(
    x: np.ndarray,
    t: np.ndarray,
    u_reference: np.ndarray,
    u_nn: np.ndarray,
    profile_times: list[float],
) -> pd.DataFrame:
    """Build profile rows for selected times.

    Parameters
    ----------
    x, t:
        Spatial and temporal grids.
    u_reference, u_nn:
        Reference and neural-network solution matrices with shape ``(nx, nt)``.
    profile_times:
        Times requested in the NAS configuration.

    Returns
    -------
    pd.DataFrame
        Long-form profile table with columns ``solution``, ``t``, ``x``, and ``u``.
    """
    rows: list[pd.DataFrame] = []
    for requested_time in profile_times:
        idx = int(np.argmin(np.abs(t - float(requested_time))))
        actual_time = float(t[idx])
        rows.append(pd.DataFrame({"solution": "Analytical", "t": actual_time, "x": x, "u": u_reference[:, idx]}))
        rows.append(pd.DataFrame({"solution": "NN", "t": actual_time, "x": x, "u": u_nn[:, idx]}))
    return pd.concat(rows, ignore_index=True)


def selected_architecture_metadata(selected: dict[str, Any]) -> dict[str, Any]:
    """Build the architecture contract consumed by the PINN stage.

    Parameters
    ----------
    selected:
        Selected NAS trial row.

    Returns
    -------
    dict[str, Any]
        Serializable architecture metadata containing the complete layer list
        and the activation function.
    """
    required = ["activation", "num_layers", "neurons_per_layer"]
    missing = [key for key in required if key not in selected or pd.isna(selected[key])]
    if missing:
        raise ValueError(f"Selected NAS trial is missing architecture fields: {missing}")

    activation = str(selected["activation"]).lower()
    num_hidden_layers = int(selected["num_layers"])
    neurons_per_hidden_layer = int(selected["neurons_per_layer"])
    layers = [2] + [neurons_per_hidden_layer] * num_hidden_layers + [1]
    return {
        "source": "nas_postprocess",
        "architecture_name": str(selected.get("architecture_name", f"{activation}_{num_hidden_layers}x{neurons_per_hidden_layer}")),
        "activation": activation,
        "layers": layers,
        "input_features": int(layers[0]),
        "output_features": int(layers[-1]),
        "num_hidden_layers": num_hidden_layers,
        "neurons_per_hidden_layer": neurons_per_hidden_layer,
        "source_trial_number": int(selected["trial_number"]) if "trial_number" in selected and not pd.isna(selected["trial_number"]) else None,
    }


def compute_nas_postprocess(config: dict[str, Any], paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the selected NAS model and compute full-domain diagnostics.

    Parameters
    ----------
    config:
        NAS configuration mapping.
    paths:
        Input and output paths for this stage.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]
        Neural-network solution table, time-error table, space-error table,
        profile table, and metadata.
    """
    require_file(paths["reference_path"], "Run: python scripts\\01a_generate_reference_data.py --overwrite")
    selected = select_trial(config, paths)
    model_path = ROOT / str(selected["model_path"])
    require_file(model_path, "Run: python scripts\\03a_run_nas.py --overwrite")

    x_ref, t_ref, u_ref = read_solution_matrix(paths["reference_path"])
    X_full = np.column_stack((np.repeat(t_ref, x_ref.size), np.tile(x_ref, t_ref.size)))

    print("Evaluating selected NAS model on the full reference grid:")
    print(f"  model: {model_path.relative_to(ROOT)}")
    print(f"  grid: nx={x_ref.size}, nt={t_ref.size}, points={X_full.shape[0]}")
    y_pred = predict_with_model(X_full, model_path=model_path)
    u_nn = y_pred.reshape(t_ref.size, x_ref.size).T

    nn_df = solution_dataframe_from_matrix(x_ref, t_ref, u_nn)
    x_nn, t_nn, u_nn_check = x_ref, t_ref, u_nn
    assert_same_grid(x_ref, t_ref, x_nn, t_nn, "NN")

    error_time = compute_error_time(u_nn_check, u_ref)
    error_space = compute_error_space(u_nn_check, u_ref)
    time_df = pd.DataFrame({"method": "NN", "label": "NN", "t": t_ref, "error_l2": error_time})
    space_df = pd.DataFrame({"method": "NN", "label": "NN", "x": x_ref, "error_l2": error_space})

    profile_times = [float(value) for value in config.get("postprocess", {}).get("profile_times", [0.0, 0.25, 0.5, 0.75, 1.0])]
    profiles_df = compute_profiles(x_ref, t_ref, u_ref, u_nn_check, profile_times)

    architecture = selected_architecture_metadata(selected)
    metadata = {
        "stage": "nas_postprocess",
        "selected_trial": selected,
        "selected_architecture": architecture,
        "reference": str(paths["reference_path"].relative_to(ROOT)),
        "prediction": str(paths["nn_data_path"].relative_to(ROOT)),
        "n_space": int(x_ref.size),
        "n_time": int(t_ref.size),
        "full_prediction_final_time": float(t_ref[-1]),
        "training_final_time": float(config["training"]["training_final_time"]),
        "profile_times": profile_times,
        "summary": {
            "max_error_time": float(np.max(error_time)),
            "final_error_time": float(error_time[-1]),
            "max_error_space": float(np.max(error_space)),
            "global_l2_error": float(np.linalg.norm(u_nn_check - u_ref)),
            "relative_global_l2_error": float(np.linalg.norm(u_nn_check - u_ref) / np.linalg.norm(u_ref)),
        },
    }
    return nn_df, time_df, space_df, profiles_df, metadata


def save_nas_postprocess_outputs(
    paths: dict[str, Path],
    nn_df: pd.DataFrame,
    time_df: pd.DataFrame,
    space_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Write NAS postprocess outputs.

    Parameters
    ----------
    paths:
        Output paths.
    nn_df, time_df, space_df, profiles_df:
        DataFrames written by this stage.
    metadata:
        Serializable metadata mapping.
    """
    save_parquet(nn_df, paths["nn_data_path"])
    save_csv(time_df, paths["time_csv"])
    save_csv(space_df, paths["space_csv"])
    save_csv(profiles_df, paths["profiles_csv"])
    save_json(metadata, paths["metadata_json"])


def main() -> None:
    """Run the NAS postprocess script."""
    args = parse_args()
    config = load_nas_config(args.config)
    paths = build_paths(config)
    expected_outputs = [paths["nn_data_path"], paths["time_csv"], paths["space_csv"], paths["profiles_csv"], paths["metadata_json"]]
    if should_skip(expected_outputs, overwrite=args.overwrite):
        print_skip_message(expected_outputs, ROOT)
        return

    nn_df, time_df, space_df, profiles_df, metadata = compute_nas_postprocess(config, paths)
    save_nas_postprocess_outputs(paths, nn_df, time_df, space_df, profiles_df, metadata)

    summary = metadata["summary"]
    print("Saved NAS postprocess outputs:")
    for key in ["nn_data_path", "time_csv", "space_csv", "profiles_csv", "metadata_json"]:
        print(f"  {paths[key].relative_to(ROOT)}")
    print(
        "NAS full-domain error: "
        f"final E(t)={summary['final_error_time']:.6e}, "
        f"max E(t)={summary['max_error_time']:.6e}, "
        f"relative global L2={summary['relative_global_l2_error']:.6e}"
    )


if __name__ == "__main__":
    main()
