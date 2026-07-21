"""Benchmark definition for the 1D advection-diffusion test case.

This module contains only data and functions that define the benchmark case used
in the paper. Equation-specific objects such as the stationary solution and modal
coefficients are implemented in :mod:`pinn_spectral.analytical`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    """Physical parameters of the benchmark problem."""

    length: float = 1.0
    u_left: float = 0.5
    u_right: float = 1.0
    velocity: float = 1.0
    diffusivity: float = 0.025

    @classmethod
    def from_mapping(cls, data: dict) -> "BenchmarkConfig":
        """Build a benchmark configuration from a YAML mapping."""
        return cls(
            length=float(data["length"]),
            u_left=float(data["u_left"]),
            u_right=float(data["u_right"]),
            velocity=float(data["velocity"]),
            diffusivity=float(data["diffusivity"]),
        )

    @property
    def peclet(self) -> float:
        """Return the Peclet number ``Pe = v L / D``."""
        return self.velocity * self.length / self.diffusivity


def initial_condition(x: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    """Return the benchmark initial condition ``f(x)``.

    The initial profile is

        f(x) = u_left cos(pi x / 2L) + u_right sin(pi x / 2L).
    """
    x = np.asarray(x, dtype=np.float64)
    return config.u_left * np.cos(np.pi * x / (2.0 * config.length)) + config.u_right * np.sin(
        np.pi * x / (2.0 * config.length)
    )


def make_space_time_grid(
    length: float,
    n_space: int,
    final_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the spatial and temporal grids used by the benchmark."""
    x = np.linspace(0.0, float(length), int(n_space), dtype=np.float64)
    n_time = int(round(float(final_time) / float(dt))) + 1
    t = np.linspace(0.0, float(final_time), n_time, dtype=np.float64)
    return x, t
