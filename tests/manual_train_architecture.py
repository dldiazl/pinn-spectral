"""Manual training test for one NAS architecture.

This script is intentionally outside the normal ``unittest discover`` workflow.
It trains one architecture, for example ``tanh_8x90``, using the same training
class and printed metrics used by the NAS stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.benchmark import BenchmarkConfig  # noqa: E402
from pinn_spectral.neural import FullyConnectedNN  # noqa: E402
from pinn_spectral.tools import output_name, read_yaml, require_file, save_json  # noqa: E402


def df2Xy(df, time_max=None):
    """Convert a solution DataFrame to ``X`` and ``y`` exactly as in the old tools.py."""
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    
    if time_max != None:
        y = y[X[:,0]<=time_max]
        X = X[X[:,0]<=time_max,:]
    return X, y


def Xy2tensor(X,y):
    """Convert ``X`` and ``y`` to float64 PyTorch tensors exactly as in the old tools.py."""
    X_tensor = torch.as_tensor(X, dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.float64).view(-1, 1)
    
    return X_tensor, y_tensor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Manually train one architecture such as tanh_8x90.")
    parser.add_argument(
        "--config",
        default="configs/nas.yaml",
        help="Path to the NAS configuration YAML file.",
    )
    parser.add_argument(
        "--architecture",
        default="tanh_8x90",
        help="Architecture in activation_layersxneurons format, for example tanh_8x90.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional override for the maximum number of outer L-BFGS epochs.",
    )
    parser.add_argument(
        "--input-parquet",
        default=None,
        help=(
            "Optional input parquet. Use this to compare against the old workflow, "
            "for example solutions\\Analytical_nx57_dt0.001.parquet."
        ),
    )
    parser.add_argument(
        "--no-time-filter",
        action="store_true",
        help="Do not filter the input DataFrame by time. This matches the old NAS_Optuna.py call.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration relative to the project root if needed."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return read_yaml(path)


def parse_architecture(text: str) -> tuple[str, int, int]:
    """Parse an architecture string such as ``tanh_8x90``."""
    try:
        activation, shape = text.split("_", maxsplit=1)
        layers_text, neurons_text = shape.lower().split("x", maxsplit=1)
        num_layers = int(layers_text)
        neurons_per_layer = int(neurons_text)
    except ValueError as exc:
        raise ValueError("Architecture must use activation_layersxneurons format, for example tanh_8x90.") from exc
    return activation, num_layers, neurons_per_layer


def default_reference_path(config: dict) -> Path:
    """Return the default analytical-reference parquet path for the refactored workflow."""
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


def resolve_input_path(config: dict, input_parquet: str | None) -> Path:
    """Resolve the manual-training input parquet."""
    if input_parquet is None:
        return default_reference_path(config)
    path = Path(input_parquet)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    """Run the manual architecture training test."""
    args = parse_args()
    config = load_config(args.config)
    activation, num_layers, neurons_per_layer = parse_architecture(args.architecture)
    max_epochs = int(args.max_epochs if args.max_epochs is not None else config["training"]["max_epochs"])

    input_path = resolve_input_path(config, args.input_parquet)
    require_file(input_path, "Generate the analytical reference first or pass --input-parquet with an existing file.")

    time_max = None if args.no_time_filter else float(config["training"]["training_final_time"])
    df = pd.read_parquet(input_path)
    X_np, y_np = df2Xy(df, time_max=time_max)
    X, y = Xy2tensor(X_np, y_np)

    outputs = config["outputs"]
    study_name = str(config["training"].get("study_name", "nas_manual"))
    manual_name = f"manual_{args.architecture}_e{max_epochs}"
    histories_dir = ROOT / outputs.get("histories_dir", "data/nas/histories") / study_name / "manual"
    models_dir = ROOT / outputs.get("models_dir", "results/models/nas") / study_name / "manual"
    metrics_dir = ROOT / outputs.get("metrics_dir", "results/metrics")
    history_path = histories_dir / f"{manual_name}.parquet"
    model_path = models_dir / f"{manual_name}.pth"
    metadata_path = metrics_dir / f"{manual_name}_metadata.json"

    if (not args.overwrite) and history_path.exists() and model_path.exists() and metadata_path.exists():
        print("Outputs already exist. Use --overwrite to regenerate them.")
        print(f"  {history_path.relative_to(ROOT)}")
        print(f"  {model_path.relative_to(ROOT)}")
        print(f"  {metadata_path.relative_to(ROOT)}")
        return

    layers = [2] + [neurons_per_layer] * num_layers + [1]

    print("Manual architecture training test:")
    print(f"  architecture: {args.architecture}")
    print(f"  layers: {layers}")
    print(f"  activation: {activation}")
    print(f"  max_epochs: {max_epochs}")
    print(f"  input parquet: {input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path}")
    print(f"  time filter: {'disabled' if time_max is None else f't <= {time_max}'}")
    print(f"  training points: {X_np.shape[0]}")
    print("Training metrics will be printed once per outer L-BFGS epoch.")

    histories_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = FullyConnectedNN(layers, activation, max_epochs=max_epochs).double()
    model.fit(X, y)

    model.history.to_parquet(history_path, index=False)
    torch.save(model, model_path)

    metadata = {
        "stage": "manual_architecture_training_test",
        "architecture": args.architecture,
        "layers": layers,
        "activation": activation,
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
        "max_epochs": max_epochs,
        "input_parquet": str(input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path),
        "time_filter": None if time_max is None else f"t <= {time_max}",
        "training_points": int(X_np.shape[0]),
        "history_path": str(history_path.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
        "final_epoch": int(model.history['Epoch'].iloc[-1]),
        "final_loss": float(model.history['Loss'].iloc[-1]),
        "known_data_mse": float(model.history['LossBC'].iloc[-1]),
    }
    save_json(metadata, metadata_path)

    print("Manual architecture training completed:")
    print(f"  final epoch: {metadata['final_epoch']}")
    print(f"  final Loss: {metadata['final_loss']:.6e}")
    print(f"  LossBC / known-data MSE: {metadata['known_data_mse']:.6e}")
    print(f"  history: {history_path.relative_to(ROOT)}")
    print(f"  model: {model_path.relative_to(ROOT)}")
    print(f"  metadata: {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
