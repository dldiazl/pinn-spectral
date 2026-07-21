"""Lightweight tests for the progressive PINN stage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.benchmark import BenchmarkConfig, make_space_time_grid  # noqa: E402
from pinn_spectral.pinn import (  # noqa: E402
    WindowArtifacts,
    _completed_window_prefix,
    advection_diffusion_physics_loss,
    build_pinn_training_sets,
    build_progressive_window_schedule,
    resolve_pinn_architecture,
)


def test_progressive_schedule_contains_fifty_ordered_windows() -> None:
    """The article schedule must be 0.1, 0.2, ..., 5.0."""
    schedule = build_progressive_window_schedule(0.1, 0.1, 5.0)
    assert schedule.shape == (50,)
    assert schedule[0] == pytest.approx(0.1)
    assert schedule[-1] == pytest.approx(5.0)
    assert np.allclose(np.diff(schedule), 0.1, rtol=0.0, atol=1e-12)


def test_training_sets_match_full_grid_rules() -> None:
    """IC, BC, and residual points must match the historical full-grid PINN."""
    benchmark = BenchmarkConfig()
    x, t = make_space_time_grid(benchmark.length, 57, 5.0, 0.001)
    sets = build_pinn_training_sets(x, t, 0.1, benchmark)

    assert sets.X_ic.shape == (55, 2)
    assert sets.y_ic.shape == (55,)
    assert sets.X_bc.shape == (202, 2)
    assert sets.y_bc.shape == (202,)
    assert sets.X_physics.shape == (5500, 2)
    assert np.all(sets.X_ic[:, 0] == 0.0)
    assert np.all((sets.X_ic[:, 1] > 0.0) & (sets.X_ic[:, 1] < 1.0))
    assert set(np.unique(sets.X_bc[:, 1])) == {0.0, 1.0}
    assert np.all(sets.X_physics[:, 0] > 0.0)
    assert np.all((sets.X_physics[:, 1] > 0.0) & (sets.X_physics[:, 1] < 1.0))


def test_architecture_is_loaded_from_nas_postprocess_metadata(tmp_path: Path) -> None:
    """The PINN must construct its layers from NAS postprocess metadata."""
    metadata_path = tmp_path / "results" / "metrics" / "nas_selected_model_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "selected_architecture": {
                    "architecture_name": "tanh_custom",
                    "activation": "tanh",
                    "layers": [2, 16, 24, 16, 1],
                    "source_trial_number": 7,
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "architecture": {
            "source": "nas_postprocess",
            "metadata_path": "results/metrics/nas_selected_model_metadata.json",
            "require_twice_differentiable_activation": True,
        }
    }

    architecture = resolve_pinn_architecture(config, tmp_path)
    assert architecture.name == "tanh_custom"
    assert architecture.layers == (2, 16, 24, 16, 1)
    assert architecture.activation == "tanh"
    assert architecture.source_trial_number == 7


def test_relu_architecture_is_rejected_for_second_derivative_residual(tmp_path: Path) -> None:
    """ReLU cannot be silently accepted for a strong-form u_xx residual."""
    config = {
        "architecture": {
            "source": "explicit",
            "name": "relu_test",
            "layers": [2, 8, 1],
            "activation": "relu",
            "require_twice_differentiable_activation": True,
        }
    }
    with pytest.raises(ValueError, match="ReLU"):
        resolve_pinn_architecture(config, tmp_path)


def test_physics_loss_uses_advection_diffusion_residual() -> None:
    """The residual must be u_t + v*u_x - D*u_xx."""
    loss_function = advection_diffusion_physics_loss(velocity=1.0, diffusivity=0.025)
    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    loss = loss_function(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.float64),
        values,
        2.0 * values,
        4.0 * values,
    )
    residual = values + 2.0 * values - 0.025 * 4.0 * values
    assert loss.item() == pytest.approx(torch.mean(residual**2).item())



def _write_complete_bundle(base: Path, name: str) -> WindowArtifacts:
    """Create a minimal nonempty artifact bundle for resume tests."""
    artifacts = WindowArtifacts(
        solution=base / f"{name}.parquet",
        history=base / f"{name}_history.parquet",
        checkpoint=base / f"{name}_weights.pth",
        completion=base / f"{name}_complete.json",
    )
    base.mkdir(parents=True, exist_ok=True)
    artifacts.solution.write_bytes(b"solution")
    artifacts.history.write_bytes(b"history")
    artifacts.checkpoint.write_bytes(b"checkpoint")
    artifacts.completion.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    return artifacts


def test_resume_rejects_a_completed_window_after_a_gap(tmp_path: Path) -> None:
    """A restart must never skip an unfinished progressive window."""
    schedule = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    artifacts = {
        0.1: _write_complete_bundle(tmp_path, "w01"),
        0.2: WindowArtifacts(
            solution=tmp_path / "w02.parquet",
            history=tmp_path / "w02_history.parquet",
            checkpoint=tmp_path / "w02_weights.pth",
            completion=tmp_path / "w02_complete.json",
        ),
        0.3: _write_complete_bundle(tmp_path, "w03"),
    }
    with pytest.raises(RuntimeError, match="after an incomplete window"):
        _completed_window_prefix(schedule, artifacts)
