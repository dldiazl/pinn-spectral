"""Run the supervised neural architecture search with Optuna NSGA-II.

The NAS stage trains fully connected neural networks against the analytical
reference over the supervised interval ``0 <= t <= 1``. The internal neural
network training loss remains the legacy scaled loss, i.e.
``Loss = 1000*LossBC`` for the supervised NAS case. The NAS objective reported
to Optuna is the unscaled known-data mean squared error, ``LossBC``.

This script is restartable. It stores the Optuna study in SQLite, writes a
partial CSV after each completed trial, and avoids retraining architectures that
already exist either in the current study folders or in the flat cache folder
``data/nas/cache``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import optuna
import pandas as pd
import torch
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.benchmark import BenchmarkConfig  # noqa: E402
from pinn_spectral.neural import FullyConnectedNN  # noqa: E402
from pinn_spectral.tools import (  # noqa: E402
    add_overwrite_argument,
    architecture_key,
    copy_architecture_pair_to_trial,
    dataframe_to_xy,
    find_architecture_pair_in_flat_cache,
    find_architecture_pair_in_study_outputs,
    get_trial_generation_safe,
    output_name,
    print_skip_message,
    read_nas_objectives_from_history,
    read_yaml,
    require_file,
    save_csv,
    save_json,
    should_skip,
    study_to_nas_trials_dataframe,
    validate_nas_generation_config,
    xy_to_tensor,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with NAS configuration path and overwrite flag.
    """
    parser = argparse.ArgumentParser(description="Run supervised NAS with Optuna NSGA-II.")
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


def build_nas_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Build the main output paths for the NAS stage.

    Parameters
    ----------
    config:
        NAS configuration mapping.

    Returns
    -------
    dict[str, Path]
        Named output paths and directories.
    """
    outputs = config["outputs"]
    training = config["training"]
    study_name = str(training["study_name"])
    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    histories_dir = ROOT / outputs.get("histories_dir", "data/nas/histories") / study_name
    models_dir = ROOT / outputs.get("models_dir", "results/models/nas") / study_name
    cache_dir = ROOT / "data" / "nas" / "cache"
    storage_path = metrics_dir / f"{study_name}.db"
    return {
        "metrics_dir": metrics_dir,
        "histories_dir": histories_dir,
        "models_dir": models_dir,
        "cache_dir": cache_dir,
        "storage_path": storage_path,
        "trials_csv": metrics_dir / "nas_trials.csv",
        "partial_trials_csv": metrics_dir / "nas_trials_partial.csv",
        "metadata_json": metrics_dir / "nas_metadata.json",
    }


def reference_data_path(config: dict[str, Any]) -> Path:
    """Return the expected analytical-reference parquet path.

    Parameters
    ----------
    config:
        NAS configuration mapping.

    Returns
    -------
    Path
        Reference parquet path used as supervised NAS data.
    """
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    stem = output_name(
        "Analytical",
        int(grid["n_space"]),
        float(grid["dt"]),
        float(grid["final_time"]),
        benchmark.peclet,
    )
    return ROOT / "data" / "reference" / f"{stem}.parquet"


def load_training_tensors(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Load analytical data and return supervised NAS tensors.

    Parameters
    ----------
    config:
        NAS configuration mapping.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, Path]
        Input tensor, target tensor, and reference parquet path.
    """
    reference_path = reference_data_path(config)
    require_file(reference_path, "Run: python scripts\\01a_generate_reference_data.py --overwrite")

    df = pd.read_parquet(reference_path)
    training_final_time = float(config["training"]["training_final_time"])
    X_np, y_np = dataframe_to_xy(df, time_max=training_final_time)
    X, y = xy_to_tensor(X_np, y_np)
    return X, y, reference_path


def parse_sampler_seed(value: Any) -> int | None:
    """Parse an optional sampler seed from configuration."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def write_trials_snapshot(study: optuna.Study, paths: dict[str, Path]) -> pd.DataFrame:
    """Write the current Optuna trial table to the partial CSV file.

    Parameters
    ----------
    study:
        Optuna study being optimized.
    paths:
        NAS output paths.

    Returns
    -------
    pd.DataFrame
        Snapshot table.
    """
    trials_df = study_to_nas_trials_dataframe(study, root=ROOT)
    save_csv(trials_df, paths["partial_trials_csv"])
    return trials_df


def make_trial_callback(paths: dict[str, Path]):
    """Return an Optuna callback that writes a partial CSV snapshot."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Write a partial trials snapshot after each completed Optuna trial."""
        del trial
        write_trials_snapshot(study, paths)

    return callback


def register_trial_attrs(
    trial: optuna.Trial,
    *,
    generation: int,
    name: str,
    history_path: Path,
    model_path: Path,
    objectives: dict[str, Any],
    cache_hit: bool,
    cache_source: str,
) -> None:
    """Store NAS metadata in Optuna user attributes."""
    trial.set_user_attr("generation", int(generation))
    trial.set_user_attr("architecture_name", name)
    trial.set_user_attr("architecture_key", name)
    trial.set_user_attr("history_path", str(history_path.relative_to(ROOT)))
    trial.set_user_attr("model_path", str(model_path.relative_to(ROOT)))
    trial.set_user_attr("cache_hit", bool(cache_hit))
    trial.set_user_attr("cache_source", cache_source)
    for key in [
        "final_mse",
        "min_mse",
        "final_scaled_loss",
        "min_scaled_loss",
        "total_time_seconds",
        "total_time_cpu",
        "flops",
        "final_epoch",
        "epochs",
    ]:
        if key in objectives:
            trial.set_user_attr(key, objectives[key])

    # Backward-compatible aliases. In this repo, final_loss now means the NAS
    # objective, i.e. the unscaled known-data MSE.
    trial.set_user_attr("final_loss", objectives.get("final_mse"))
    trial.set_user_attr("known_data_mse", objectives.get("final_mse"))


def run_nas(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    """Run Optuna NSGA-II and write model/history files for each trial.

    Parameters
    ----------
    config:
        NAS configuration mapping.
    paths:
        Output paths produced by :func:`build_nas_paths`.

    Returns
    -------
    pd.DataFrame
        Trial summary table.
    """
    validate_nas_generation_config(config["training"])
    X, y, reference_path = load_training_tensors(config)

    training = config["training"]
    search_space = config["search_space"]
    max_epochs = int(training["max_epochs"])
    n_trials = int(training["n_trials"])
    population_size = int(training["population_size"])
    generations = int(training["generations"])
    seed = parse_sampler_seed(training.get("sampler_seed"))
    study_name = str(training["study_name"])

    paths["histories_dir"].mkdir(parents=True, exist_ok=True)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=population_size)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=study_name,
        sampler=sampler,
        storage=f"sqlite:///{paths['storage_path'].as_posix()}",
        load_if_exists=True,
    )

    complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    remaining_trials = max(0, n_trials - len(complete_trials))

    print(
        "Running supervised NAS: "
        f"study={study_name}, trials={n_trials}, population_size={population_size}, "
        f"generations={generations}, remaining_trials={remaining_trials}, "
        f"max_epochs={max_epochs}, training_t_max={float(training['training_final_time'])}, "
        f"sampler_seed={seed}"
    )
    print(f"Training data: {reference_path.relative_to(ROOT)}")
    print(f"Optuna storage: {paths['storage_path'].relative_to(ROOT)}")
    print(f"Flat cache: {paths['cache_dir'].relative_to(ROOT)}")

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        """Sample one architecture and return its (final MSE, optimizer time) objectives.

        A cache hit in the current study output or the flat cache reuses the
        existing history and model files instead of retraining.
        """
        num_layers = trial.suggest_int(
            "num_layers",
            int(search_space["num_layers"]["low"]),
            int(search_space["num_layers"]["high"]),
        )
        neurons_per_layer = trial.suggest_int(
            "neurons_per_layer",
            int(search_space["neurons_per_layer"]["low"]),
            int(search_space["neurons_per_layer"]["high"]),
            step=int(search_space["neurons_per_layer"].get("step", 1)),
        )
        activation = trial.suggest_categorical("activation", list(search_space["activation"]))

        layers = [2] + [neurons_per_layer] * num_layers + [1]
        name = architecture_key(activation, num_layers, neurons_per_layer)
        generation = get_trial_generation_safe(sampler, study, trial, population_size=population_size)
        file_prefix = f"trial{trial.number:03d}_{name}"
        history_path = paths["histories_dir"] / f"{file_prefix}.parquet"
        model_path = paths["models_dir"] / f"{file_prefix}.pth"

        print(
            f"Trial {trial.number}: generation={generation}, architecture={name}, "
            f"layers={layers}, activation={activation}"
        )

        # Priority 1: already completed in the current study output folders.
        current_pair = find_architecture_pair_in_study_outputs(
            architecture=name,
            histories_dir=paths["histories_dir"],
            models_dir=paths["models_dir"],
            exclude_history_path=history_path,
            exclude_model_path=model_path,
        )
        if current_pair is not None:
            source_history, source_model = current_pair
            copy_architecture_pair_to_trial(source_history, source_model, history_path, model_path)
            objectives = read_nas_objectives_from_history(history_path)
            register_trial_attrs(
                trial,
                generation=generation,
                name=name,
                history_path=history_path,
                model_path=model_path,
                objectives=objectives,
                cache_hit=True,
                cache_source="current_study",
            )
            print(
                f"[CACHE HIT/current_study] Trial {trial.number}: {name}, "
                f"MSE={objectives['final_mse']:.6e}, "
                f"optimizer time={objectives['total_time_seconds']:.3f}s"
            )
            return float(objectives["final_mse"]), float(objectives["total_time_seconds"])

        # Priority 2: flat cache under data/nas/cache.
        flat_pair = find_architecture_pair_in_flat_cache(name, paths["cache_dir"])
        if flat_pair is not None:
            source_history, source_model = flat_pair
            copy_architecture_pair_to_trial(source_history, source_model, history_path, model_path)
            objectives = read_nas_objectives_from_history(history_path)
            register_trial_attrs(
                trial,
                generation=generation,
                name=name,
                history_path=history_path,
                model_path=model_path,
                objectives=objectives,
                cache_hit=True,
                cache_source="flat_cache",
            )
            print(
                f"[CACHE HIT/flat_cache] Trial {trial.number}: {name}, "
                f"MSE={objectives['final_mse']:.6e}, "
                f"optimizer time={objectives['total_time_seconds']:.3f}s"
            )
            return float(objectives["final_mse"]), float(objectives["total_time_seconds"])

        # Priority 3: train a new architecture.
        print(f"[CACHE MISS] Trial {trial.number}: training {name}")
        model = FullyConnectedNN(layers, activation, max_epochs=max_epochs).double()
        model.fit(X, y)

        model.history.to_parquet(history_path, index=False)
        torch.save(model, model_path)
        objectives = read_nas_objectives_from_history(history_path)
        register_trial_attrs(
            trial,
            generation=generation,
            name=name,
            history_path=history_path,
            model_path=model_path,
            objectives=objectives,
            cache_hit=False,
            cache_source="trained",
        )

        print(
            f"Trial {trial.number} complete: MSE={objectives['final_mse']:.6e}, "
            f"scaled loss={objectives['final_scaled_loss']:.6e}, "
            f"optimizer time={objectives['total_time_seconds']:.3f}s, "
            f"CPU time={objectives['total_time_cpu']:.3f}s, FLOPS={objectives['flops']}"
        )
        return float(objectives["final_mse"]), float(objectives["total_time_seconds"])

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, callbacks=[make_trial_callback(paths)])
    else:
        print("Requested number of completed trials is already available in the Optuna study.")

    trials_df = write_trials_snapshot(study, paths)
    return trials_df


def selected_trial_summary(trials_df: pd.DataFrame) -> dict[str, Any]:
    """Return the minimum-MSE completed trial summary.

    Parameters
    ----------
    trials_df:
        NAS trial table.

    Returns
    -------
    dict[str, Any]
        Selected trial row as a serializable dictionary.
    """
    complete = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if complete.empty:
        raise ValueError("No completed NAS trials were found.")
    if "final_mse" not in complete.columns:
        complete["final_mse"] = complete["final_loss"]
    complete = complete.sort_values(["final_mse", "trial_number"], ascending=True)
    return complete.iloc[0].to_dict()


def save_nas_outputs(config: dict[str, Any], paths: dict[str, Path], trials_df: pd.DataFrame) -> None:
    """Write NAS trial and metadata files.

    Parameters
    ----------
    config:
        NAS configuration mapping.
    paths:
        Output paths produced by :func:`build_nas_paths`.
    trials_df:
        Trial summary table.
    """
    save_csv(trials_df, paths["trials_csv"])
    selected = selected_trial_summary(trials_df)
    training = config["training"]
    metadata = {
        "stage": "supervised_nas",
        "study_name": str(training["study_name"]),
        "training_final_time": float(training["training_final_time"]),
        "max_epochs": int(training["max_epochs"]),
        "n_trials": int(training["n_trials"]),
        "population_size": int(training["population_size"]),
        "generations": int(training["generations"]),
        "sampler": "NSGAIISampler",
        "sampler_seed": parse_sampler_seed(training.get("sampler_seed")),
        "storage": str(paths["storage_path"].relative_to(ROOT)),
        "flat_cache_dir": str(paths["cache_dir"].relative_to(ROOT)),
        "selection_criterion": str(config.get("selection", {}).get("criterion", "minimum_final_mse")),
        "selected_trial": selected,
        "loss_interpretation": {
            "training_Loss": "legacy scaled training loss, equal to 1000*LossBC + 1000*LossIC + LossPhysics",
            "training_Loss_in_supervised_NAS": "1000*known-data MSE because LossIC=0 and LossPhysics=0",
            "NAS_objective_1": "final_mse, equal to the final LossBC value",
            "NAS_objective_2": "total_time_seconds, equal to the cumulative optimizer wall-clock time",
            "LossBC_in_supervised_NAS": "known-data mean squared error",
        },
    }
    save_json(metadata, paths["metadata_json"])


def main() -> None:
    """Run the NAS script."""
    args = parse_args()
    config = load_nas_config(args.config)
    paths = build_nas_paths(config)
    expected_outputs = [paths["trials_csv"], paths["metadata_json"]]
    if should_skip(expected_outputs, overwrite=args.overwrite):
        print_skip_message(expected_outputs, ROOT)
        return

    trials_df = run_nas(config, paths)
    save_nas_outputs(config, paths, trials_df)
    selected = selected_trial_summary(trials_df)
    print("Selected architecture by minimum final MSE:")
    print(
        f"  trial={int(selected['trial_number'])}, architecture={selected['architecture_name']}, "
        f"final_mse={float(selected['final_mse']):.6e}, "
        f"scaled_loss={float(selected['final_scaled_loss']):.6e}"
    )
    print(f"Saved NAS trials to {paths['trials_csv'].relative_to(ROOT)}")
    print(f"Saved NAS metadata to {paths['metadata_json'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
