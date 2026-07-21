"""Physics-informed training utilities for the advection-diffusion benchmark.

This module keeps equation-specific PINN logic separate from the generic neural
network implementation. The network architecture is resolved from a metadata
file produced by the NAS postprocessing stage unless an explicit architecture
is supplied for testing or another controlled workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from pinn_spectral.benchmark import BenchmarkConfig, initial_condition, make_space_time_grid
from pinn_spectral.neural import FullyConnectedNN
from pinn_spectral.tools import (
    output_name,
    predict_with_model,
    solution_dataframe_from_matrix,
    xy_to_tensor,
)

_HISTORY_COLUMNS = [
    "Epoch",
    "Loss",
    "LossIC",
    "LossBC",
    "LossPhysics",
    "Time_optimizer",
    "Time_cpu",
    "Flops",
    "LR",
]


@dataclass(frozen=True)
class NetworkArchitecture:
    """Architecture contract used to construct a PINN model."""

    name: str
    layers: tuple[int, ...]
    activation: str
    source: str
    metadata_path: str | None = None
    source_trial_number: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the architecture."""
        return {
            "name": self.name,
            "layers": list(self.layers),
            "activation": self.activation,
            "source": self.source,
            "metadata_path": self.metadata_path,
            "source_trial_number": self.source_trial_number,
        }


@dataclass(frozen=True)
class PinnOutputPaths:
    """Output directories and summary files for progressive PINN training."""

    data_dir: Path
    histories_dir: Path
    models_dir: Path
    metrics_dir: Path
    windows_csv: Path
    training_metadata_json: Path


@dataclass(frozen=True)
class WindowArtifacts:
    """Files that define one completed progressive training window."""

    solution: Path
    history: Path
    checkpoint: Path
    completion: Path


@dataclass(frozen=True)
class PinnTrainingSets:
    """Initial, boundary, and residual-point arrays for one time window."""

    X_ic: np.ndarray
    y_ic: np.ndarray
    X_bc: np.ndarray
    y_bc: np.ndarray
    X_physics: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _resolve_root_path(root: Path, value: str | Path) -> Path:
    """Resolve a project-relative path without introducing machine paths."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _architecture_from_metadata(metadata: dict[str, Any], metadata_path: Path) -> NetworkArchitecture:
    """Extract a complete architecture from NAS postprocess metadata."""
    architecture = metadata.get("selected_architecture")
    if isinstance(architecture, dict):
        layers = architecture.get("layers")
        activation = architecture.get("activation")
        name = architecture.get("architecture_name") or architecture.get("name")
        trial_number = architecture.get("source_trial_number")
    else:
        selected = metadata.get("selected_trial")
        if not isinstance(selected, dict):
            raise ValueError(
                f"{metadata_path} contains neither 'selected_architecture' nor 'selected_trial'. "
                "Rerun scripts/03b_postprocess_nas.py."
            )
        required = ["activation", "num_layers", "neurons_per_layer"]
        missing = [key for key in required if key not in selected]
        if missing:
            raise ValueError(f"Selected NAS trial in {metadata_path} is missing fields: {missing}")
        activation = selected["activation"]
        hidden_layers = int(selected["num_layers"])
        hidden_width = int(selected["neurons_per_layer"])
        layers = [2] + [hidden_width] * hidden_layers + [1]
        name = selected.get("architecture_name") or f"{activation}_{hidden_layers}x{hidden_width}"
        trial_number = selected.get("trial_number")

    if layers is None or activation is None:
        raise ValueError(f"Incomplete selected architecture in {metadata_path}.")

    parsed_layers = tuple(int(value) for value in layers)
    parsed_trial = None if trial_number is None else int(trial_number)
    return NetworkArchitecture(
        name=str(name or "nas_selected_architecture"),
        layers=parsed_layers,
        activation=str(activation).lower(),
        source="nas_postprocess",
        metadata_path=str(metadata_path),
        source_trial_number=parsed_trial,
    )


def validate_pinn_architecture(
    architecture: NetworkArchitecture,
    require_twice_differentiable_activation: bool = True,
) -> None:
    """Validate a network architecture for the strong-form PINN residual."""
    if len(architecture.layers) < 3:
        raise ValueError("A PINN architecture must contain input, hidden, and output layers.")
    if any(width <= 0 for width in architecture.layers):
        raise ValueError(f"All layer widths must be positive: {architecture.layers}")
    if architecture.layers[0] != 2:
        raise ValueError(f"The PINN input layer must have width 2 for (t, x), got {architecture.layers[0]}.")
    if architecture.layers[-1] != 1:
        raise ValueError(f"The PINN output layer must have width 1 for u, got {architecture.layers[-1]}.")

    supported = {"tanh", "relu", "sigmoid"}
    if architecture.activation not in supported:
        raise ValueError(
            f"Unsupported activation '{architecture.activation}'. Supported activations are {sorted(supported)}."
        )
    if require_twice_differentiable_activation and architecture.activation == "relu":
        raise ValueError(
            "The NAS-selected activation is ReLU, which is not suitable for the strong-form PINN residual "
            "containing u_xx. Select a twice-differentiable NAS architecture or change the PINN formulation."
        )


def resolve_pinn_architecture(config: dict[str, Any], root: str | Path) -> NetworkArchitecture:
    """Resolve the PINN architecture from the configured architecture source.

    The production configuration uses ``source: nas_postprocess`` and reads the
    architecture contract written by ``03b_postprocess_nas.py``. ``explicit``
    remains available for lightweight tests and controlled alternative studies.
    """
    root = Path(root)
    architecture_config = config.get("architecture", {})
    source = str(architecture_config.get("source", "nas_postprocess")).lower()

    if source == "nas_postprocess":
        metadata_value = architecture_config.get("metadata_path")
        if not metadata_value:
            raise KeyError("architecture.metadata_path is required for source='nas_postprocess'.")
        metadata_path = _resolve_root_path(root, str(metadata_value))
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"NAS postprocess metadata not found: {metadata_path}\n"
                "Run: python scripts/03b_postprocess_nas.py --overwrite"
            )
        resolved = _architecture_from_metadata(_read_json(metadata_path), metadata_path)
        architecture = NetworkArchitecture(
            name=resolved.name,
            layers=resolved.layers,
            activation=resolved.activation,
            source=resolved.source,
            metadata_path=_relative_path(metadata_path, root),
            source_trial_number=resolved.source_trial_number,
        )
    elif source == "explicit":
        layers = architecture_config.get("layers")
        activation = architecture_config.get("activation")
        if layers is None or activation is None:
            raise KeyError("Explicit architectures require architecture.layers and architecture.activation.")
        architecture = NetworkArchitecture(
            name=str(architecture_config.get("name", "explicit_architecture")),
            layers=tuple(int(value) for value in layers),
            activation=str(activation).lower(),
            source="explicit",
        )
    else:
        raise ValueError(f"Unknown architecture source: {source}")

    validate_pinn_architecture(
        architecture,
        require_twice_differentiable_activation=bool(
            architecture_config.get("require_twice_differentiable_activation", True)
        ),
    )
    return architecture


def build_progressive_window_schedule(
    first_window_final_time: float,
    window_increment: float,
    final_time: float,
) -> np.ndarray:
    """Return the ordered terminal times of the progressive PINN windows."""
    first = float(first_window_final_time)
    increment = float(window_increment)
    final = float(final_time)
    if first <= 0.0 or increment <= 0.0 or final <= 0.0:
        raise ValueError("Progressive window times must be positive.")
    if first > final:
        raise ValueError("The first window final time cannot exceed the full final time.")

    count_float = (final - first) / increment
    count = int(round(count_float))
    if not np.isclose(count_float, count, rtol=0.0, atol=1e-12):
        raise ValueError(
            "The final time must be reachable from first_window_final_time using whole window increments."
        )
    schedule = first + increment * np.arange(count + 1, dtype=np.float64)
    schedule[-1] = final
    return np.round(schedule, decimals=12)


def build_pinn_training_sets(
    x: np.ndarray,
    t: np.ndarray,
    window_final_time: float,
    benchmark: BenchmarkConfig,
) -> PinnTrainingSets:
    """Build full-batch IC, BC, and residual points for one time window."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.ndim != 1 or t.ndim != 1:
        raise ValueError("x and t must be one-dimensional arrays.")
    if x.size < 3:
        raise ValueError("At least three spatial nodes are required for interior residual points.")

    final_index = int(np.argmin(np.abs(t - float(window_final_time))))
    if not np.isclose(t[final_index], float(window_final_time), rtol=0.0, atol=1e-12):
        raise ValueError(f"Window final time {window_final_time} is not present on the configured time grid.")
    t_window = t[: final_index + 1]
    x_interior = x[1:-1]

    X_ic = np.column_stack((np.zeros(x_interior.size, dtype=np.float64), x_interior))
    y_ic = initial_condition(x_interior, benchmark)

    boundary_x = np.array([x[0], x[-1]], dtype=np.float64)
    X_bc = np.column_stack((np.repeat(t_window, 2), np.tile(boundary_x, t_window.size)))
    y_bc = np.tile(np.array([benchmark.u_left, benchmark.u_right], dtype=np.float64), t_window.size)

    residual_times = t_window[1:]
    X_physics = np.column_stack(
        (
            np.repeat(residual_times, x_interior.size),
            np.tile(x_interior, residual_times.size),
        )
    )
    return PinnTrainingSets(X_ic=X_ic, y_ic=y_ic, X_bc=X_bc, y_bc=y_bc, X_physics=X_physics)


def advection_diffusion_physics_loss(
    velocity: float,
    diffusivity: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the strong-form residual loss ``mean((u_t + v u_x - D u_xx)^2)``."""
    velocity_value = float(velocity)
    diffusivity_value = float(diffusivity)

    def loss_function(
        X: torch.Tensor,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean-squared advection-diffusion residual at the given points."""
        del X, u
        residual = u_t + velocity_value * u_x - diffusivity_value * u_xx
        return torch.mean(residual**2)

    return loss_function


def build_pinn_output_paths(config: dict[str, Any], root: str | Path) -> PinnOutputPaths:
    """Build output directories and summary paths from ``configs/pinn.yaml``."""
    root = Path(root)
    outputs = config["outputs"]
    metrics_dir = _resolve_root_path(root, outputs.get("metrics_dir", "results/metrics"))
    return PinnOutputPaths(
        data_dir=_resolve_root_path(root, outputs.get("data_dir", "data/pinn")),
        histories_dir=_resolve_root_path(root, outputs.get("histories_dir", "data/pinn/histories")),
        models_dir=_resolve_root_path(root, outputs.get("models_dir", "results/models/pinn")),
        metrics_dir=metrics_dir,
        windows_csv=metrics_dir / "pinn_windows.csv",
        training_metadata_json=metrics_dir / "pinn_training_metadata.json",
    )


def build_window_artifacts(
    paths: PinnOutputPaths,
    benchmark: BenchmarkConfig,
    n_space: int,
    dt: float,
    window_final_time: float,
) -> WindowArtifacts:
    """Return the four files associated with one progressive time window."""
    stem = output_name("PINNs", n_space, dt, window_final_time, benchmark.peclet)
    return WindowArtifacts(
        solution=paths.data_dir / f"{stem}.parquet",
        history=paths.histories_dir / f"{stem}_history.parquet",
        checkpoint=paths.models_dir / f"{stem}_weights.pth",
        completion=paths.models_dir / f"{stem}_complete.json",
    )


def _temporary_path(path: Path) -> Path:
    """Return the deterministic temporary path used for atomic replacement."""
    return path.with_name(f"{path.name}.tmp")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file and atomically replace the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    df.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV file and atomically replace the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON and atomically replace the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_torch_save(data: Any, path: Path) -> None:
    """Write a PyTorch checkpoint and atomically replace the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    torch.save(data, temporary)
    os.replace(temporary, path)


def _remove_window_artifacts(artifacts: WindowArtifacts) -> None:
    """Remove final and temporary files for one incomplete window."""
    for path in [artifacts.solution, artifacts.history, artifacts.checkpoint, artifacts.completion]:
        path.unlink(missing_ok=True)
        _temporary_path(path).unlink(missing_ok=True)


def _validate_complete_window(artifacts: WindowArtifacts) -> dict[str, Any]:
    """Validate a completion marker and all artifacts in its bundle."""
    marker = _read_json(artifacts.completion)
    if marker.get("status") != "COMPLETE":
        raise ValueError(f"Invalid completion status in {artifacts.completion}.")
    missing = [
        path
        for path in [artifacts.solution, artifacts.history, artifacts.checkpoint]
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "A PINN window is marked complete but its artifact bundle is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    return marker


def _completed_window_prefix(
    schedule: np.ndarray,
    artifacts_by_time: dict[float, WindowArtifacts],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Return the continuous completed prefix and reject markers after a gap."""
    completed_times: list[float] = []
    markers: list[dict[str, Any]] = []
    gap_found = False
    for value in schedule:
        time_value = float(value)
        artifacts = artifacts_by_time[time_value]
        if artifacts.completion.exists():
            if gap_found:
                raise RuntimeError(
                    "A later PINN window is marked complete after an incomplete window. "
                    f"Unexpected marker: {artifacts.completion}"
                )
            markers.append(_validate_complete_window(artifacts))
            completed_times.append(time_value)
        else:
            gap_found = True
    return completed_times, markers


def _load_completed_history(
    completed_times: list[float],
    artifacts_by_time: dict[float, WindowArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load local histories and reconstruct legacy and annotated histories."""
    annotated_parts: list[pd.DataFrame] = []
    for time_value in completed_times:
        history = pd.read_parquet(artifacts_by_time[time_value].history)
        missing = [column for column in _HISTORY_COLUMNS if column not in history.columns]
        if missing:
            raise ValueError(f"PINN history for t={time_value:.3f} is missing columns: {missing}")
        annotated_parts.append(history)

    if not annotated_parts:
        empty_legacy = pd.DataFrame(columns=_HISTORY_COLUMNS)
        empty_annotated = pd.DataFrame(columns=["WindowIndex", "WindowFinalTime", "EpochInWindow"] + _HISTORY_COLUMNS)
        return empty_legacy, empty_annotated

    annotated = pd.concat(annotated_parts, ignore_index=True)
    legacy = annotated[_HISTORY_COLUMNS].copy()
    epochs = legacy["Epoch"].to_numpy(dtype=np.int64)
    expected = np.arange(1, len(legacy) + 1, dtype=np.int64)
    if not np.array_equal(epochs, expected):
        raise ValueError("Completed PINN histories do not contain a continuous global Epoch sequence.")
    return legacy, annotated


def _evaluate_window_solution(
    model: FullyConnectedNN,
    x: np.ndarray,
    t: np.ndarray,
    window_final_time: float,
) -> pd.DataFrame:
    """Evaluate a trained model on the configured tensor grid up to one window."""
    final_index = int(np.argmin(np.abs(t - float(window_final_time))))
    t_window = t[: final_index + 1]
    X_eval = np.column_stack((np.repeat(t_window, x.size), np.tile(x, t_window.size)))
    prediction = predict_with_model(X_eval, model=model)
    u = prediction.reshape(t_window.size, x.size).T
    return solution_dataframe_from_matrix(x, t_window, u)


def _relative_path(path: Path, root: Path) -> str:
    """Serialize paths relative to the repository root when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _window_summary_from_history(
    window_index: int,
    window_final_time: float,
    history: pd.DataFrame,
    artifacts: WindowArtifacts,
    root: Path,
) -> dict[str, Any]:
    """Build one serializable summary row from a local window history."""
    if history.empty:
        raise ValueError(f"Training produced no history rows for window t={window_final_time:.3f}.")
    last = history.iloc[-1]
    return {
        "window_index": int(window_index),
        "window_final_time": float(window_final_time),
        "epochs_in_window": int(len(history)),
        "global_epoch_final": int(last["Epoch"]),
        "final_scaled_loss": float(last["Loss"]),
        "final_loss_ic": float(last["LossIC"]),
        "final_loss_bc": float(last["LossBC"]),
        "final_loss_physics": float(last["LossPhysics"]),
        "optimizer_time_seconds": float(history["Time_optimizer"].sum()),
        "cpu_time_seconds": float(history["Time_cpu"].sum()),
        "flops": float(history["Flops"].sum()),
        "solution_path": _relative_path(artifacts.solution, root),
        "history_path": _relative_path(artifacts.history, root),
        "checkpoint_path": _relative_path(artifacts.checkpoint, root),
        "completion_path": _relative_path(artifacts.completion, root),
    }


def _save_training_summaries(
    markers: list[dict[str, Any]],
    config: dict[str, Any],
    architecture: NetworkArchitecture,
    paths: PinnOutputPaths,
    benchmark: BenchmarkConfig,
    schedule: np.ndarray,
    root: Path,
) -> None:
    """Write the progressive-window table and training metadata."""
    summary_rows = [dict(marker["summary"]) for marker in markers]
    windows_df = pd.DataFrame(summary_rows)
    _atomic_write_csv(windows_df, paths.windows_csv)

    grid = config["grid"]
    training = config["training"]
    metadata = {
        "stage": "pinn_progressive_training",
        "architecture": architecture.as_dict(),
        "architecture_transfer": "architecture only; NAS weights are not loaded",
        "benchmark": {
            "length": benchmark.length,
            "u_left": benchmark.u_left,
            "u_right": benchmark.u_right,
            "velocity": benchmark.velocity,
            "diffusivity": benchmark.diffusivity,
            "peclet": benchmark.peclet,
        },
        "grid": {
            "n_space": int(grid["n_space"]),
            "dt": float(grid["dt"]),
            "final_time": float(grid["final_time"]),
        },
        "training": {
            "first_window_final_time": float(training["first_window_final_time"]),
            "window_increment": float(training["window_increment"]),
            "max_epochs_per_window": int(training["max_epochs_per_window"]),
            "patience": int(training["patience"]),
            "update_learning_rate": bool(training.get("update_learning_rate", False)),
            "transfer_nas_weights": False,
            "optimizer": {
                "name": "LBFGS",
                "lr": 1.0,
                "max_iter": 20,
                "tolerance_grad": 1e-9,
                "tolerance_change": float(np.finfo(float).eps),
                "line_search_fn": "strong_wolfe",
                "history_size": 100,
            },
            "loss": "1000*LossBC + 1000*LossIC + LossPhysics",
        },
        "window_schedule": [float(value) for value in schedule],
        "completed_windows": len(markers),
        "expected_windows": int(schedule.size),
        "windows_csv": _relative_path(paths.windows_csv, root),
    }
    _atomic_write_json(metadata, paths.training_metadata_json)


def clear_pinn_training_outputs(paths: PinnOutputPaths) -> None:
    """Remove PINN training outputs while leaving other repository stages intact."""
    for directory in [paths.data_dir, paths.models_dir]:
        if directory.exists():
            shutil.rmtree(directory)
    if paths.histories_dir.exists() and not paths.histories_dir.is_relative_to(paths.data_dir):
        shutil.rmtree(paths.histories_dir)
    paths.windows_csv.unlink(missing_ok=True)
    paths.training_metadata_json.unlink(missing_ok=True)


def run_progressive_pinn_training(
    config: dict[str, Any],
    root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train or resume the complete progressive-window PINN workflow.

    A window is considered complete only after its solution, local history, and
    checkpoint have been written and a completion marker has been atomically
    installed. Restarting resumes from the first missing window and never skips
    over a gap.
    """
    root = Path(root)
    benchmark = BenchmarkConfig.from_mapping(config["case"])
    grid = config["grid"]
    training = config["training"]
    n_space = int(grid["n_space"])
    dt = float(grid["dt"])
    final_time = float(grid["final_time"])
    max_epochs = int(training["max_epochs_per_window"])
    patience = int(training["patience"])
    update_learning_rate = bool(training.get("update_learning_rate", False))
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs_per_window and patience must be positive integers.")
    if bool(training.get("transfer_nas_weights", False)):
        raise ValueError(
            "The article transfers the NAS-selected architecture, not the supervised NAS weights. "
            "Set training.transfer_nas_weights to false."
        )

    architecture = resolve_pinn_architecture(config, root)
    schedule = build_progressive_window_schedule(
        first_window_final_time=float(training["first_window_final_time"]),
        window_increment=float(training["window_increment"]),
        final_time=final_time,
    )
    x, t = make_space_time_grid(benchmark.length, n_space, final_time, dt)
    paths = build_pinn_output_paths(config, root)
    if overwrite:
        clear_pinn_training_outputs(paths)

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.histories_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    artifacts_by_time = {
        float(value): build_window_artifacts(paths, benchmark, n_space, dt, float(value))
        for value in schedule
    }
    completed_times, markers = _completed_window_prefix(schedule, artifacts_by_time)
    legacy_history, _ = _load_completed_history(completed_times, artifacts_by_time)

    previous_state_dict: dict[str, torch.Tensor] | None = None
    if completed_times:
        last_checkpoint = artifacts_by_time[completed_times[-1]].checkpoint
        payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        previous_state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        print(
            f"Resuming PINN training after {len(completed_times)} completed windows; "
            f"last completed time={completed_times[-1]:.3f}."
        )
    else:
        print("Starting PINN progressive training from the first window.")

    physics_loss = advection_diffusion_physics_loss(benchmark.velocity, benchmark.diffusivity)
    for zero_based_index, schedule_value in enumerate(schedule):
        window_index = zero_based_index + 1
        window_time = float(schedule_value)
        if window_time in completed_times:
            continue

        artifacts = artifacts_by_time[window_time]
        _remove_window_artifacts(artifacts)
        training_sets = build_pinn_training_sets(x, t, window_time, benchmark)
        X_bc, y_bc = xy_to_tensor(training_sets.X_bc, training_sets.y_bc)
        X_ic, y_ic = xy_to_tensor(training_sets.X_ic, training_sets.y_ic)
        X_physics = torch.as_tensor(training_sets.X_physics, dtype=torch.float64)

        model = FullyConnectedNN(
            layers=list(architecture.layers),
            activation=architecture.activation,
            max_epochs=max_epochs,
        ).double()
        if previous_state_dict is not None:
            model.load_state_dict(previous_state_dict)
        model.history = legacy_history.copy()
        model.current_epoch = int(legacy_history["Epoch"].iloc[-1]) + 1 if not legacy_history.empty else 1
        history_start = len(model.history)

        print(
            f"Training PINN window {window_index}/{schedule.size}: "
            f"0 <= t <= {window_time:.3f}, architecture={architecture.name}, "
            f"points(IC={len(X_ic)}, BC={len(X_bc)}, Physics={len(X_physics)})"
        )
        model.fit(
            X_bc,
            y_bc,
            XIC=X_ic,
            yIC=y_ic,
            XPhysics=X_physics,
            loss_physics_function=physics_loss,
            patience=patience,
            updateLR=update_learning_rate,
        )

        local_legacy_history = model.history.iloc[history_start:].copy().reset_index(drop=True)
        if local_legacy_history.empty:
            raise RuntimeError(f"No optimizer epochs were recorded for window t={window_time:.3f}.")
        local_history = local_legacy_history.copy()
        local_history.insert(0, "EpochInWindow", np.arange(1, len(local_history) + 1, dtype=np.int64))
        local_history.insert(0, "WindowFinalTime", window_time)
        local_history.insert(0, "WindowIndex", window_index)

        solution_df = _evaluate_window_solution(model, x, t, window_time)
        summary = _window_summary_from_history(
            window_index=window_index,
            window_final_time=window_time,
            history=local_history,
            artifacts=artifacts,
            root=root,
        )
        checkpoint_payload = {
            "state_dict": model.state_dict(),
            "architecture": architecture.as_dict(),
            "window_index": window_index,
            "window_final_time": window_time,
            "global_epoch_final": int(local_history["Epoch"].iloc[-1]),
        }
        marker = {
            "status": "COMPLETE",
            "window_index": window_index,
            "window_final_time": window_time,
            "architecture": architecture.as_dict(),
            "summary": summary,
        }

        _atomic_write_parquet(solution_df, artifacts.solution)
        _atomic_write_parquet(local_history, artifacts.history)
        _atomic_torch_save(checkpoint_payload, artifacts.checkpoint)
        _atomic_write_json(marker, artifacts.completion)

        legacy_history = pd.concat([legacy_history, local_legacy_history], ignore_index=True)
        previous_state_dict = model.state_dict()
        markers.append(marker)
        completed_times.append(window_time)
        _save_training_summaries(markers, config, architecture, paths, benchmark, schedule, root)
        print(
            f"Completed window t={window_time:.3f}: "
            f"Loss={summary['final_scaled_loss']:.6e}, "
            f"LossIC={summary['final_loss_ic']:.6e}, "
            f"LossBC={summary['final_loss_bc']:.6e}, "
            f"LossPhysics={summary['final_loss_physics']:.6e}"
        )

    _save_training_summaries(markers, config, architecture, paths, benchmark, schedule, root)
    final_artifacts = artifacts_by_time[float(schedule[-1])]
    return {
        "architecture": architecture.as_dict(),
        "completed_windows": len(markers),
        "expected_windows": int(schedule.size),
        "final_solution": _relative_path(final_artifacts.solution, root),
        "final_checkpoint": _relative_path(final_artifacts.checkpoint, root),
        "windows_csv": _relative_path(paths.windows_csv, root),
        "training_metadata": _relative_path(paths.training_metadata_json, root),
    }
