"""Finite-difference solvers for the 1D advection-diffusion benchmark.

The implementations in this module preserve the discretizations used in the
original manuscript scripts while returning NumPy arrays instead of writing files
or appending pandas DataFrames inside the time loop.

The equation is

    u_t = D u_xx - v u_x,

with Dirichlet boundary conditions ``u(0,t)=u_left`` and ``u(L,t)=u_right``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from pinn_spectral.benchmark import BenchmarkConfig, initial_condition, make_space_time_grid


def _initial_profile(x: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    """Return the benchmark initial profile as float64."""
    return np.asarray(initial_condition(x, config), dtype=np.float64)


def solve_cds_crank_nicolson(
    config: BenchmarkConfig,
    n_space: int,
    final_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve using central differences in space and Crank-Nicolson in time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x, t, u)`` where ``u`` has shape ``(n_space, n_time)``.
    """
    L = np.float64(config.length)
    u0 = np.float64(config.u_left)
    uL = np.float64(config.u_right)
    v = np.float64(config.velocity)
    D = np.float64(config.diffusivity)

    x, t = make_space_time_grid(L, n_space, final_time, dt)
    n_time = t.size
    h = np.float64(L / np.float64(n_space - 1))
    dt64 = np.float64(dt)

    alpha = np.float64(D * dt64 / (2.0 * h**2))
    beta = np.float64(v * dt64 / (4.0 * h))

    n_inner = int(n_space) - 2
    A = np.zeros((n_inner, n_inner), dtype=np.float64)
    B = np.zeros((n_inner, n_inner), dtype=np.float64)
    bcs = np.zeros(n_inner, dtype=np.float64)

    a1 = np.float64(-alpha - beta)
    b1 = np.float64(1.0 + 2.0 * alpha)
    c1 = np.float64(beta - alpha)

    a2 = np.float64(alpha + beta)
    b2 = np.float64(1.0 - 2.0 * alpha)
    c2 = np.float64(alpha - beta)

    bcs[0] = np.float64(a2 * u0 - a1 * u0)
    bcs[-1] = np.float64(c2 * uL - c1 * uL)

    A[0, 0] = b1
    A[0, 1] = c1
    B[0, 0] = b2
    B[0, 1] = c2

    for i in range(1, n_space - 3):
        A[i, i - 1] = a1
        A[i, i] = b1
        A[i, i + 1] = c1
        B[i, i - 1] = a2
        B[i, i] = b2
        B[i, i + 1] = c2

    A[-1, -2] = a1
    A[-1, -1] = b1
    B[-1, -2] = a2
    B[-1, -1] = b2

    lu, piv = lu_factor(A)

    u = np.empty((int(n_space), n_time), dtype=np.float64)
    u[:, 0] = _initial_profile(x, config)
    current = u[1:-1, 0].copy()

    for j in range(1, n_time):
        next_inner = lu_solve((lu, piv), B @ current + bcs)
        u[:, j] = np.concatenate(([u0], next_inner, [uL]))
        current = next_inner

    return x, t, u


def solve_cds_explicit(
    config: BenchmarkConfig,
    n_space: int,
    final_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve using central differences in space and explicit Euler in time."""
    L = np.float64(config.length)
    u0 = np.float64(config.u_left)
    uL = np.float64(config.u_right)
    v = np.float64(config.velocity)
    D = np.float64(config.diffusivity)

    x, t = make_space_time_grid(L, n_space, final_time, dt)
    n_time = t.size
    h = np.float64(L / np.float64(n_space - 1))
    dt64 = np.float64(dt)

    alpha = np.float64(D * dt64 / h**2)
    beta = np.float64(v * dt64 / (2.0 * h))

    n_inner = int(n_space) - 2
    A = np.zeros((n_inner, n_inner), dtype=np.float64)
    bcs = np.zeros(n_inner, dtype=np.float64)

    a1 = np.float64(alpha + beta)
    b1 = np.float64(1.0 - 2.0 * alpha)
    c1 = np.float64(alpha - beta)

    bcs[0] = np.float64(a1 * u0)
    bcs[-1] = np.float64(c1 * uL)

    A[0, 0] = b1
    A[0, 1] = c1
    for i in range(1, n_space - 3):
        A[i, i - 1] = a1
        A[i, i] = b1
        A[i, i + 1] = c1
    A[-1, -2] = a1
    A[-1, -1] = b1

    u = np.empty((int(n_space), n_time), dtype=np.float64)
    u[:, 0] = _initial_profile(x, config)
    current = u[1:-1, 0].copy()

    for j in range(1, n_time):
        next_inner = A @ current + bcs
        u[:, j] = np.concatenate(([u0], next_inner, [uL]))
        current = next_inner

    return x, t, u


def build_compact_diff_matrix_34643(n_nodes: int, h: float) -> np.ndarray:
    """Build the compact first-derivative matrix labeled 3-4-6-4-3."""
    n_nodes = int(n_nodes)
    h = np.float64(h)
    alpha = np.float64(1.0 / 3.0)
    fac01 = np.float64(1.0 / 36.0)
    fac02 = np.float64(14.0 / 18.0)

    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    B = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    A[0, 0] = np.float64(1.0)
    A[0, 1] = np.float64(2.0)
    A[1, 0] = np.float64(0.25)
    A[1, 1] = np.float64(1.0)
    A[1, 2] = np.float64(0.25)

    B[0, 0] = np.float64(-15.0 / 6.0)
    B[0, 1] = np.float64(2.0)
    B[0, 2] = np.float64(0.5)
    B[1, 0] = np.float64(-3.0 / 4.0)
    B[1, 2] = np.float64(+3.0 / 4.0)

    for i in range(2, n_nodes - 2):
        A[i, i - 1] = alpha
        A[i, i] = np.float64(1.0)
        A[i, i + 1] = alpha
        B[i, i - 2] = -fac01
        B[i, i - 1] = -fac02
        B[i, i + 1] = fac02
        B[i, i + 2] = fac01

    A[-2, -3] = np.float64(0.25)
    A[-2, -2] = np.float64(1.0)
    A[-2, -1] = np.float64(0.25)
    A[-1, -2] = np.float64(2.0)
    A[-1, -1] = np.float64(1.0)

    B[-2, -3] = np.float64(-3.0 / 4.0)
    B[-2, -1] = np.float64(+3.0 / 4.0)
    B[-1, -3] = np.float64(-0.5)
    B[-1, -2] = np.float64(-2.0)
    B[-1, -1] = np.float64(15.0 / 6.0)

    return np.float64(1.0 / h) * (np.linalg.inv(A) @ B)


def solve_compact_crank_nicolson(
    config: BenchmarkConfig,
    n_space: int,
    final_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve using compact finite differences and Crank-Nicolson in time.

    The compact differentiation matrix is the 3-4-6-4-3 operator used in the
    original script. Boundary values are reset after every time step, matching
    the legacy implementation.
    """
    L = np.float64(config.length)
    u0 = np.float64(config.u_left)
    uL = np.float64(config.u_right)
    v = np.float64(config.velocity)
    D = np.float64(config.diffusivity)

    x, t = make_space_time_grid(L, n_space, final_time, dt)
    n_time = t.size
    h = np.float64(L / np.float64(n_space - 1))
    dt64 = np.float64(dt)

    diff = build_compact_diff_matrix_34643(n_space, h)
    identity = np.eye(int(n_space), dtype=np.float64)
    diff2 = diff @ diff

    oper_a = identity - np.float64(0.5 * dt64 * D) * diff2 + np.float64(0.5 * dt64 * v) * diff
    oper_b = identity + np.float64(0.5 * dt64 * D) * diff2 - np.float64(0.5 * dt64 * v) * diff
    step_matrix = np.linalg.inv(oper_a) @ oper_b

    u = np.empty((int(n_space), n_time), dtype=np.float64)
    u[:, 0] = _initial_profile(x, config)

    current = u[:, 0].copy()
    for j in range(1, n_time):
        current = step_matrix @ current
        current[0] = u0
        current[-1] = uL
        u[:, j] = current

    return x, t, u
