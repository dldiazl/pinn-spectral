"""Analytical reference solution for the 1D advection-diffusion equation.

This module implements the closed-form eigenfunction expansion used in the
manuscript for

    u_t = D u_xx - v u_x,
    u(0,t)=u0, u(L,t)=uL,
    u(x,0)=u0 cos(pi x / 2L) + uL sin(pi x / 2L).

The computable reference is a truncated expansion with an optional backward
average of the last K partial sums. The backward average is evaluated through
mode weights, so the full array of partial sums is never stored in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceDiagnostics:
    """Endpoint diagnostics for the analytical reference."""

    initial_error: float
    stationary_error: float
    max_endpoint_error: float


def initial_condition(x: np.ndarray, L: float, u0: float, uL: float) -> np.ndarray:
    """Return the prescribed initial condition ``f(x)``."""
    x = np.asarray(x, dtype=np.float64)
    return u0 * np.cos(np.pi * x / (2.0 * L)) + uL * np.sin(np.pi * x / (2.0 * L))


def stationary_solution(x: np.ndarray, v: float, D: float, L: float, u0: float, uL: float) -> np.ndarray:
    """Return the stationary solution ``S(x)``.

    The zero-advection limit is handled explicitly to avoid a removable
    singularity in the exponential formula.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.isclose(v, 0.0):
        return u0 + (uL - u0) * x / L

    pe = v * L / D
    return u0 + (uL - u0) * (np.exp(pe * x / L) - 1.0) / (np.exp(pe) - 1.0)


def initial_integral(v: float, D: float, L: float, u0: float, uL: float, n: np.ndarray) -> np.ndarray:
    """Return the coefficient contribution ``I_n^(f)``.

    This expression evaluates

        (2/L) int_0^L f(x) exp[-Pe x/(2L)] sin(n pi x/L) dx

    for the initial condition used in the manuscript. It follows the appendix
    derivation based on sine-product and sine-cosine identities.
    """
    n = np.asarray(n, dtype=np.float64)
    a = v / (2.0 * D)
    b = n * np.pi / L
    alpha = np.pi / (2.0 * L)

    def J(k: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form antiderivative used for the sine terms (Appendix A.1)."""
        return ((-a * np.sin(k * L) - k * np.cos(k * L)) * np.exp(-a * L) + k) / (a**2 + k**2)

    def K(k: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form antiderivative used for the cosine terms (Appendix A.2)."""
        return ((-a * np.cos(k * L) + k * np.sin(k * L)) * np.exp(-a * L) + a) / (a**2 + k**2)

    i1 = 0.5 * (J(b + alpha) + J(b - alpha))
    i2 = 0.5 * (K(alpha - b) - K(alpha + b))
    return (2.0 / L) * (u0 * i1 + uL * i2)


def stationary_integral(v: float, D: float, L: float, u0: float, uL: float, n: np.ndarray) -> np.ndarray:
    """Return the coefficient contribution ``I_n^(S)``."""
    n = np.asarray(n, dtype=np.float64)
    pe = v * L / D
    return (2.0 * n * np.pi / ((n * np.pi) ** 2 + (pe / 2.0) ** 2)) * (
        uL * (-1.0) ** n * np.exp(-pe / 2.0) - u0
    )


def coefficients(v: float, D: float, L: float, u0: float, uL: float, n_terms: int) -> np.ndarray:
    """Return the modal coefficients ``B_n`` for ``n=1,...,n_terms``."""
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    return initial_integral(v, D, L, u0, uL, n) + stationary_integral(v, D, L, u0, uL, n)


def backward_average_weights(n_terms: int, averaging_width: int) -> np.ndarray:
    """Return modal weights equivalent to averaging the last K partial sums.

    The averaged field is

        mean_{M=N-K+1}^N sum_{n=1}^M a_n.

    A mode ``n`` therefore contributes with weight 1 if it is present in all
    averaged partial sums, and with weight ``(N-n+1)/K`` near the truncation end.
    """
    if averaging_width <= 1:
        return np.ones(n_terms, dtype=np.float64)

    width = min(int(averaging_width), int(n_terms))
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    included_count = np.minimum(width, n_terms - n + 1.0)
    return included_count / width


def analytical_solution(
    x: np.ndarray,
    t: np.ndarray,
    v: float,
    D: float,
    L: float,
    u0: float,
    uL: float,
    n_terms: int,
    averaging_width: int = 1,
    n_modes_per_block: int = 2000,
    n_times_per_block: int = 256,
    mode_chunk_size: int | None = None,
    time_chunk_size: int | None = None,
) -> np.ndarray:
    """Return the truncated/averaged analytical solution on a tensor grid.

    Parameters
    ----------
    x, t:
        One-dimensional spatial and temporal grids.
    v, D, L, u0, uL:
        Problem parameters.
    n_terms:
        Number of retained modes ``N``.
    averaging_width:
        Backward averaging width ``K``. ``K=1`` gives the unsmoothed partial sum.
    mode_chunk_size, time_chunk_size:
        Chunk sizes used to control memory consumption.

    Returns
    -------
    np.ndarray
        Array of shape ``(nx, nt)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()

    if n_terms < 1:
        raise ValueError("n_terms must be at least 1.")
    if averaging_width < 1:
        raise ValueError("averaging_width must be at least 1.")

    if mode_chunk_size is not None:
        n_modes_per_block = int(mode_chunk_size)
    if time_chunk_size is not None:
        n_times_per_block = int(time_chunk_size)
    n_modes_per_block = int(n_modes_per_block)
    n_times_per_block = int(n_times_per_block)
    if n_modes_per_block < 1:
        raise ValueError("n_modes_per_block must be at least 1.")
    if n_times_per_block < 1:
        raise ValueError("n_times_per_block must be at least 1.")

    pe = v * L / D
    s_x = stationary_solution(x, v, D, L, u0, uL)
    prefactor = np.exp(0.5 * pe * x / L)

    b_n = coefficients(v, D, L, u0, uL, n_terms)
    weights = backward_average_weights(n_terms, averaging_width)

    u = np.empty((x.size, t.size), dtype=np.float64)
    n_all = np.arange(1, n_terms + 1, dtype=np.float64)
    weighted_coefficients = b_n * weights

    # The t=0 column is the only one that requires all retained modes.
    # For every strictly positive time, high modes decay exponentially as
    # exp(-lambda_n t). We therefore skip modes whose exponent is already
    # below machine-level relevance for the first positive time in a block.
    # This does not alter the mathematical definition of the reference; it only
    # avoids adding terms whose floating-point contribution is effectively zero.
    max_decay_exponent = 745.0  # exp(-745) is close to the smallest useful float64 contribution.

    if t.size and np.isclose(t[0], 0.0):
        transient0 = np.zeros(x.size, dtype=np.float64)
        for n_start in range(0, n_terms, n_modes_per_block):
            n_stop = min(n_start + n_modes_per_block, n_terms)
            n = n_all[n_start:n_stop]
            coeff = weighted_coefficients[n_start:n_stop]
            sine = np.sin((n[:, None] * np.pi / L) * x[None, :])
            transient0 += coeff @ sine
        u[:, 0] = s_x + prefactor * transient0
        first_time_index = 1
    else:
        first_time_index = 0

    for t_start in range(first_time_index, t.size, n_times_per_block):
        t_stop = min(t_start + n_times_per_block, t.size)
        t_block = t[t_start:t_stop]
        min_positive_t = float(np.min(t_block[t_block > 0.0]))

        argument = max_decay_exponent * L**2 / (D * min_positive_t) - pe**2 / 4.0
        if argument <= 0.0:
            effective_terms = 1
        else:
            effective_terms = min(n_terms, max(1, int(np.ceil(np.sqrt(argument) / np.pi))))

        transient = np.zeros((x.size, t_block.size), dtype=np.float64)

        for n_start in range(0, effective_terms, n_modes_per_block):
            n_stop = min(n_start + n_modes_per_block, effective_terms)
            n = n_all[n_start:n_stop]
            coeff = weighted_coefficients[n_start:n_stop]

            sine = np.sin((n[:, None] * np.pi / L) * x[None, :])
            lambdas = (D / L**2) * ((n * np.pi) ** 2 + pe**2 / 4.0)
            decay = np.exp(-lambdas[:, None] * t_block[None, :])

            transient += (coeff[:, None] * sine).T @ decay

        u[:, t_start:t_stop] = s_x[:, None] + prefactor[:, None] * transient

    return u


def endpoint_diagnostics(x: np.ndarray, t: np.ndarray, u: np.ndarray, v: float, D: float, L: float, u0: float, uL: float) -> ReferenceDiagnostics:
    """Compute initial- and final-time consistency errors for a reference field."""
    f = initial_condition(x, L, u0, uL)
    s = stationary_solution(x, v, D, L, u0, uL)
    initial_error = float(np.linalg.norm(u[:, 0] - f))
    stationary_error = float(np.linalg.norm(u[:, -1] - s))
    return ReferenceDiagnostics(
        initial_error=initial_error,
        stationary_error=stationary_error,
        max_endpoint_error=max(initial_error, stationary_error),
    )


def solution_to_dataframe(x: np.ndarray, t: np.ndarray, u: np.ndarray) -> pd.DataFrame:
    """Convert a tensor-grid solution with shape ``(nx, nt)`` to long format.

    The output order matches the legacy files: for each time level, all spatial
    nodes are stored consecutively.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if u.shape != (x.size, t.size):
        raise ValueError(f"Expected u with shape {(x.size, t.size)}, got {u.shape}.")

    return pd.DataFrame(
        {
            "t": np.repeat(t, x.size),
            "x": np.tile(x, t.size),
            "u": u.T.ravel(order="C"),
        }
    )

# Backward-compatible aliases used by scripts and tests.
from pinn_spectral.benchmark import make_space_time_grid as make_space_time_grid  # noqa: E402,F401


def _last_k_column_average(partial_solutions: np.ndarray, averaging_width: int) -> np.ndarray:
    """Average the last K partial solution columns."""
    k = int(averaging_width)
    if k <= 1:
        return partial_solutions[:, -1]
    k = min(k, partial_solutions.shape[1])
    return np.mean(partial_solutions[:, -k:], axis=1, dtype=np.float64)


def analytical_solution_legacy(
    x: np.ndarray,
    t: np.ndarray,
    v: float,
    D: float,
    L: float,
    u0: float,
    uL: float,
    n_terms: int,
    averaging_width: int = 1,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> np.ndarray:
    """Literal last-K partial-sum analytical solution.

    This implementation is intentionally slower than :func:`analytical_solution`.
    It constructs all partial sums for each requested time and then averages the
    last K partial solutions. It is used only for validation and small tests.

    Parameters
    ----------
    x, t:
        One-dimensional spatial and temporal grids.
    v, D, L, u0, uL:
        Problem parameters.
    n_terms:
        Number of retained modes.
    averaging_width:
        Width of the last-K average applied to literal partial sums.
    progress_callback:
        Optional callable invoked after each completed time level as
        ``progress_callback(completed, total, time_value)``. It is intended for
        command-line progress reporting and does not affect the numerical result.

    Returns
    -------
    np.ndarray
        Solution matrix with shape ``(nx, nt)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    n_terms = int(n_terms)
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1.")
    if averaging_width < 1:
        raise ValueError("averaging_width must be at least 1.")

    pe = v * L / D
    s_x = stationary_solution(x, v, D, L, u0, uL)
    prefactor = np.exp(0.5 * pe * x / L)
    b_n = coefficients(v, D, L, u0, uL, n_terms)
    n_all = np.arange(1, n_terms + 1, dtype=np.float64)

    u = np.empty((x.size, t.size), dtype=np.float64)
    sine = np.sin((n_all[:, None] * np.pi / L) * x[None, :])

    for j, time_value in enumerate(t):
        lambdas = (D / L**2) * ((n_all * np.pi) ** 2 + pe**2 / 4.0)
        decay = np.exp(-lambdas * np.float64(time_value))
        modal_terms = (b_n * decay)[:, None] * sine
        partial_transient = np.cumsum(modal_terms, axis=0, dtype=np.float64).T
        partial_solutions = s_x[:, None] + prefactor[:, None] * partial_transient
        u[:, j] = _last_k_column_average(partial_solutions, averaging_width)

        if progress_callback is not None:
            progress_callback(j + 1, t.size, float(time_value))

    return u


def consistency_errors(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    v: float,
    D: float,
    L: float,
    u0: float,
    uL: float,
) -> dict[str, float]:
    """Return endpoint consistency errors as a serializable dictionary."""
    diag = endpoint_diagnostics(x, t, u, v, D, L, u0, uL)
    return {
        "initial_l2_error": diag.initial_error,
        "stationary_l2_error": diag.stationary_error,
        "max_endpoint_l2_error": diag.max_endpoint_error,
    }
