"""Shared utilities for the advection-diffusion workflow.

The functions in this module are limited to operations reused by multiple
reference, finite-difference, NAS, PINN, or spectral-analysis stages. Solution
files use the long format with columns ``t``, ``x``, and ``u``. When reshaped to
a matrix, the convention is ``u.shape == (nx, nt)``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1)), (0, (5, 2)), (0, (1, 2))]
MARKERS = ["o", "s", "D", "^", "v", "x", "*", "P", "<", ">"]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def add_overwrite_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the common ``--overwrite`` flag to a command-line parser.

    Parameters
    ----------
    parser:
        Parser that will receive the flag.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, returned for convenience.
    """
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping generation.",
    )
    return parser


def output_name(prefix: str, n_space: int, dt: float, final_time: float, pe: float) -> str:
    """Return the file stem used for solution and metadata outputs.

    Parameters
    ----------
    prefix:
        Method or solution name, for example ``Analytical`` or ``CDS_EF``.
    n_space:
        Number of spatial grid points.
    dt:
        Time step.
    final_time:
        Final simulation time.
    pe:
        Domain Peclet number.

    Returns
    -------
    str
        File stem compatible with the existing manuscript naming convention.
    """
    return f"{prefix}_nx{int(n_space)}_dt{float(dt):.3f}_t{float(final_time):.3f}_Pe{float(pe):.3f}"


def should_skip(outputs: list[Path], overwrite: bool) -> bool:
    """Return whether a script should skip generation for existing outputs.

    Parameters
    ----------
    outputs:
        Expected output paths for the current stage.
    overwrite:
        Whether the user requested regeneration.

    Returns
    -------
    bool
        ``True`` if all outputs exist and regeneration was not requested.
    """
    return (not overwrite) and all(Path(path).exists() for path in outputs)


def reject_partial_outputs(outputs: list[Path], overwrite: bool) -> None:
    """Reject a partially existing stage bundle unless overwrite is explicit.

    Parameters
    ----------
    outputs:
        Complete output bundle expected from one stage.
    overwrite:
        Whether replacement of existing artifacts was explicitly requested.

    Raises
    ------
    FileExistsError
        If only part of the output bundle exists and ``overwrite`` is false.
    """
    if overwrite:
        return
    existing = [Path(path) for path in outputs if Path(path).exists()]
    if existing and len(existing) != len(outputs):
        listed = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "A partial output bundle already exists. Refusing to overwrite existing files "
            "without --overwrite:\n" + listed
        )


def print_skip_message(outputs: list[Path], root: Path) -> None:
    """Print a standard skip message for existing outputs.

    Parameters
    ----------
    outputs:
        Paths that caused the stage to be skipped.
    root:
        Project root used to display relative paths when possible.
    """
    print("Outputs already exist. Skipping generation:")
    for path in outputs:
        path = Path(path)
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        print(f"  {shown}")
    print("Use --overwrite to regenerate them.")


def require_file(path: str | Path, hint: str) -> None:
    """Raise a helpful error when a required input file is missing.

    Parameters
    ----------
    path:
        Required file path.
    hint:
        Command or explanation that helps the user create the missing file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}\n{hint}")


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory of a file path if it does not exist.

    Parameters
    ----------
    path:
        File path whose parent directory must exist before writing.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_solution_matrix(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a long-form solution file and return ``x``, ``t``, and ``u``.

    Parameters
    ----------
    path:
        Parquet file with columns ``t``, ``x``, and ``u``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Spatial grid, temporal grid, and solution matrix with shape ``(nx, nt)``.

    Raises
    ------
    ValueError
        If required columns are missing or the rows do not form a tensor grid.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    required = {"t", "x", "u"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.sort_values(["t", "x"]).reset_index(drop=True)
    x = np.sort(df["x"].unique()).astype(np.float64)
    t = np.sort(df["t"].unique()).astype(np.float64)
    expected_rows = int(x.size * t.size)
    if len(df) != expected_rows:
        raise ValueError(
            f"{path} has {len(df)} rows, but {expected_rows} were expected "
            f"from nx={x.size}, nt={t.size}."
        )

    u = df["u"].to_numpy(dtype=np.float64).reshape(t.size, x.size).T
    return x, t, u


def solution_dataframe_from_matrix(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
) -> pd.DataFrame:
    """Convert a tensor-grid solution matrix to standard long format.

    Parameters
    ----------
    x, t:
        One-dimensional spatial and temporal grids.
    u:
        Solution matrix with shape ``(nx, nt)``.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``t``, ``x``, and ``u`` ordered by time
        first and space second.

    Raises
    ------
    ValueError
        If the grid arrays are not one-dimensional or ``u`` has an
        incompatible shape.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if x.ndim != 1 or t.ndim != 1:
        raise ValueError("x and t must be one-dimensional arrays.")
    expected_shape = (x.size, t.size)
    if u.shape != expected_shape:
        raise ValueError(f"Expected u with shape {expected_shape}, got {u.shape}.")
    return pd.DataFrame(
        {
            "t": np.repeat(t, x.size),
            "x": np.tile(x, t.size),
            "u": u.T.ravel(order="C"),
        }
    )


def assert_same_grid(
    reference_x: np.ndarray,
    reference_t: np.ndarray,
    candidate_x: np.ndarray,
    candidate_t: np.ndarray,
    label: str,
) -> None:
    """Check that two tensor-grid solutions share the same coordinates.

    Parameters
    ----------
    reference_x, reference_t:
        Spatial and temporal grids of the reference solution.
    candidate_x, candidate_t:
        Spatial and temporal grids of the candidate solution.
    label:
        Human-readable label used in error messages.

    Raises
    ------
    ValueError
        If either grid has a different shape or coordinate values.
    """
    if reference_x.shape != candidate_x.shape or not np.allclose(reference_x, candidate_x, rtol=0.0, atol=1e-15):
        raise ValueError(f"Spatial grid mismatch for {label}.")
    if reference_t.shape != candidate_t.shape or not np.allclose(reference_t, candidate_t, rtol=0.0, atol=1e-15):
        raise ValueError(f"Temporal grid mismatch for {label}.")


def compute_error_time(u_candidate: np.ndarray, u_reference: np.ndarray) -> np.ndarray:
    """Compute the spatial L2 error norm at every time level.

    Parameters
    ----------
    u_candidate, u_reference:
        Solution matrices with shape ``(nx, nt)``.

    Returns
    -------
    np.ndarray
        Array with shape ``(nt,)`` containing ``||u_candidate(:,t)-u_reference(:,t)||_2``.
    """
    if u_candidate.shape != u_reference.shape:
        raise ValueError(f"Shape mismatch: candidate={u_candidate.shape}, reference={u_reference.shape}.")
    return np.linalg.norm(np.asarray(u_candidate) - np.asarray(u_reference), axis=0)


def compute_error_space(u_candidate: np.ndarray, u_reference: np.ndarray) -> np.ndarray:
    """Compute the temporal L2 error norm at every spatial node.

    Parameters
    ----------
    u_candidate, u_reference:
        Solution matrices with shape ``(nx, nt)``.

    Returns
    -------
    np.ndarray
        Array with shape ``(nx,)`` containing ``||u_candidate(x,:)-u_reference(x,:)||_2``.
    """
    if u_candidate.shape != u_reference.shape:
        raise ValueError(f"Shape mismatch: candidate={u_candidate.shape}, reference={u_reference.shape}.")
    return np.linalg.norm(np.asarray(u_candidate) - np.asarray(u_reference), axis=1)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV without an index.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Output CSV path.
    """
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable indentation and sorted keys.

    Parameters
    ----------
    data:
        Serializable mapping.
    path:
        Output JSON path.
    """
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to parquet without an index.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Output parquet path.
    """
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def save_figure(path: str | Path, dpi: int = 600) -> None:
    """Save the current Matplotlib figure with consistent settings.

    Parameters
    ----------
    path:
        Output image path.
    dpi:
        Figure resolution in dots per inch.
    """
    ensure_parent_dir(path)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def compute_pareto_front_indices(objectives: np.ndarray) -> np.ndarray:
    """Return indices of the non-dominated front for minimization objectives.

    Parameters
    ----------
    objectives:
        Two-dimensional array with shape ``(n_points, n_objectives)``. Every
        objective is assumed to be minimized.

    Returns
    -------
    np.ndarray
        Integer indices of the non-dominated points, ordered by the first
        objective and then by the second objective.
    """
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D objective array, got shape {values.shape}.")

    finite = np.all(np.isfinite(values), axis=1)
    valid_indices = np.flatnonzero(finite)
    valid_values = values[finite]
    if valid_values.size == 0:
        return np.array([], dtype=int)

    n_points = valid_values.shape[0]
    is_dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            no_worse = np.all(valid_values[j] <= valid_values[i])
            strictly_better = np.any(valid_values[j] < valid_values[i])
            if no_worse and strictly_better:
                is_dominated[i] = True
                break

    front_local = np.flatnonzero(~is_dominated)
    front_global = valid_indices[front_local]
    order = np.lexsort((values[front_global, 1], values[front_global, 0]))
    return front_global[order]


def _dominated_hypervolume_2d(front: np.ndarray) -> float:
    """Return the area dominated by a 2-D minimization front in [0, 1]^2.

    The front coordinates must already be normalized so that the reference
    point is ``(1, 1)``; points outside the unit square are rejected.
    """
    values = np.asarray(front, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected a (n, 2) front array, got shape {values.shape}.")
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Front coordinates must lie within the unit square.")
    order = np.lexsort((values[:, 1], values[:, 0]))
    u = values[order, 0]
    v = values[order, 1]
    lowest_v = np.minimum.accumulate(v)
    next_u = np.append(u[1:], 1.0)
    return float(np.sum((next_u - u) * (1.0 - lowest_v)))


def build_nas_generation_diagnostics(
    trials: pd.DataFrame,
    selected_architecture_key: str | None = None,
    loss_column: str = "final_loss",
    time_column: str = "total_time_optimizer",
    generation_column: str = "generation",
    architecture_column: str = "architecture_key",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize generation-level convergence of the NSGA-II search.

    For every generation, the returned table reports the number of trials,
    the best supervised loss inside the generation, the cumulative best loss
    over all trials evaluated so far, the size of the cumulative
    non-dominated front in the (loss, optimizer-time) plane, and the
    dominated hypervolume of that front. The hypervolume is evaluated in a
    normalized objective space: ``log10`` of the loss and the raw optimizer
    time are min-max scaled over all evaluated trials, and the reference
    point is ``(1, 1)``. The summary records the distinct and duplicated
    architecture counts and, when ``selected_architecture_key`` is given,
    the generation in which that architecture was first evaluated.
    Generations are reported one-based.
    """
    required = {loss_column, time_column, generation_column, architecture_column}
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"NAS trials table is missing columns: {sorted(missing)}")

    table = trials[
        [generation_column, loss_column, time_column, architecture_column]
    ].copy()
    for column in (generation_column, loss_column, time_column):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna()
    if table.empty:
        raise ValueError("The NAS trials table contains no complete rows.")
    if (table[loss_column] <= 0.0).any():
        raise ValueError("Hypervolume normalization requires strictly positive losses.")

    log_loss = np.log10(table[loss_column].to_numpy(dtype=np.float64))
    times = table[time_column].to_numpy(dtype=np.float64)

    def _normalize(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        if span == 0.0:
            return np.zeros_like(values)
        return (values - float(values.min())) / span

    normalized = np.column_stack([_normalize(log_loss), _normalize(times)])
    generations = np.sort(table[generation_column].unique().astype(int))

    rows: list[dict[str, Any]] = []
    cumulative_best = float("inf")
    previous_hypervolume = 0.0
    generation_values = table[generation_column].to_numpy(dtype=np.float64)
    for generation in generations:
        in_generation = generation_values == float(generation)
        cumulative_mask = generation_values <= float(generation)
        generation_best = float(table.loc[in_generation, loss_column].min())
        cumulative_best = min(cumulative_best, generation_best)
        cumulative_points = normalized[cumulative_mask]
        front_indices = compute_pareto_front_indices(cumulative_points)
        hypervolume = _dominated_hypervolume_2d(cumulative_points[front_indices])
        rows.append(
            {
                "generation": int(generation) + 1,
                "trials_in_generation": int(np.count_nonzero(in_generation)),
                "best_loss_in_generation": generation_best,
                "cumulative_best_loss": cumulative_best,
                "cumulative_front_size": int(front_indices.size),
                "cumulative_hypervolume": hypervolume,
                "hypervolume_increment": hypervolume - previous_hypervolume,
            }
        )
        previous_hypervolume = hypervolume

    architecture_keys = table[architecture_column].astype(str)
    distinct_count = int(architecture_keys.nunique())
    summary: dict[str, Any] = {
        "trial_count": int(len(table)),
        "generation_count": int(generations.size),
        "distinct_architecture_count": distinct_count,
        "duplicate_evaluation_count": int(len(table) - distinct_count),
        "hypervolume_normalization": (
            "log10 loss and optimizer wall-clock time min-max scaled over all "
            "evaluated trials; reference point (1, 1)"
        ),
    }
    if selected_architecture_key is not None:
        matches = table.loc[
            architecture_keys == str(selected_architecture_key), generation_column
        ]
        if matches.empty:
            raise ValueError(
                f"Selected architecture '{selected_architecture_key}' does not "
                "appear in the trials table."
            )
        summary["selected_architecture_key"] = str(selected_architecture_key)
        summary["selected_first_generation"] = int(matches.min()) + 1
        summary["selected_evaluation_count"] = int(matches.size)

    return pd.DataFrame(rows), summary


def plot_pareto_front_from_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    time_column: str = "total_time_seconds",
    loss_column: str = "final_loss",
    activation_column: str = "activation",
    architecture_column: str = "architecture_name",
    trial_column: str = "trial_number",
    selected_label: str = "Selected",
    xlabel: str = "Training time [s]",
    ylabel: str = "Final supervised loss",
) -> None:
    """Plot a two-objective NAS Pareto front from a trials table.

    The plotted objectives are training time and final supervised loss. Both are
    minimized. The selected point is the completed trial with the smallest final
    loss, using the trial number as a deterministic tie-breaker when available.

    Parameters
    ----------
    df:
        Trials table containing time and loss columns.
    output_path:
        Figure path.
    time_column, loss_column:
        Columns used as the two minimization objectives.
    activation_column:
        Optional categorical column used to style the scatter markers.
    architecture_column:
        Optional label used to annotate the selected architecture.
    trial_column:
        Optional deterministic tie-breaker column.
    selected_label:
        Legend label for the selected point.
    xlabel, ylabel:
        Axis labels.
    """
    required = {time_column, loss_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pareto data are missing columns: {sorted(missing)}")

    complete = df.copy()
    complete[time_column] = pd.to_numeric(complete[time_column], errors="coerce")
    complete[loss_column] = pd.to_numeric(complete[loss_column], errors="coerce")
    complete = complete[np.isfinite(complete[time_column]) & np.isfinite(complete[loss_column])]
    complete = complete[(complete[time_column] > 0.0) & (complete[loss_column] > 0.0)].copy()
    if complete.empty:
        raise ValueError("No finite positive NAS objectives were found for the Pareto plot.")

    if activation_column not in complete.columns:
        complete[activation_column] = "unknown"
    if architecture_column not in complete.columns:
        complete[architecture_column] = complete.index.astype(str)
    if trial_column not in complete.columns:
        complete[trial_column] = np.arange(len(complete), dtype=int)

    complete = complete.sort_values(trial_column, na_position="last").reset_index(drop=True)
    objectives = complete[[time_column, loss_column]].to_numpy(dtype=np.float64)
    pareto_indices = compute_pareto_front_indices(objectives)
    pareto = complete.iloc[pareto_indices].sort_values(time_column)

    selected_idx = int(
        np.lexsort(
            (
                complete[trial_column].fillna(10**9).to_numpy(dtype=np.int64),
                complete[loss_column].to_numpy(dtype=np.float64),
            )
        )[0]
    )
    selected = complete.iloc[selected_idx]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    activations = list(dict.fromkeys(str(value) for value in complete[activation_column].fillna("unknown")))
    for idx, activation in enumerate(activations):
        group = complete[complete[activation_column].fillna("unknown") == activation]
        ax.loglog(
            group[time_column],
            group[loss_column],
            marker=MARKERS[idx % len(MARKERS)],
            linestyle="None",
            markersize=4.0,
            alpha=0.85,
            label=activation,
        )

    if not pareto.empty:
        ax.loglog(
            pareto[time_column],
            pareto[loss_column],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Pareto front",
            zorder=8,
        )

    ax.loglog(
        [float(selected[time_column])],
        [float(selected[loss_column])],
        marker="*",
        markersize=12,
        linestyle="None",
        color="black",
        label=selected_label,
        zorder=10,
    )
    ax.annotate(
        str(selected[architecture_column]),
        xy=(float(selected[time_column]), float(selected[loss_column])),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _profile_zoom_delta(delta: float | list[Any] | tuple[Any, ...], index: int) -> tuple[float, float]:
    """Normalize one zoom-window delta specification."""
    if isinstance(delta, (int, float)):
        value = float(delta)
        return value, value

    item = delta[index] if index < len(delta) else delta[0]
    if isinstance(item, (int, float)):
        value = float(item)
        return value, value
    if len(item) != 2:
        raise ValueError("Each zoom delta tuple must contain exactly two values: (dx, dy).")
    return float(item[0]), float(item[1])


def _profile_method_colors(solution_names: list[str], colors: dict[str, str] | list[str] | tuple[str, ...] | None) -> dict[str, str | None]:
    """Build a method-color mapping for profile plots."""
    if colors is None:
        return {name: None for name in solution_names}
    if isinstance(colors, dict):
        return {name: colors.get(name) for name in solution_names}
    if len(colors) < len(solution_names):
        raise ValueError(f"Expected at least {len(solution_names)} colors, got {len(colors)}.")
    return {name: str(colors[idx]) for idx, name in enumerate(solution_names)}


def plot_solution_profiles_with_zooms(
    df: pd.DataFrame,
    output_path: str | Path,
    solution_column: str = "solution",
    time_column: str = "t",
    x_column: str = "x",
    value_column: str = "u",
    solution_order: list[str] | None = None,
    reference_solution: str | None = "Analytical",
    profile_times: list[float] | None = None,
    zoom_points: list[tuple[float, float]] | None = None,
    delta: float | list[Any] | tuple[Any, ...] = 0.05,
    text_locations: list[float | None] | None = None,
    colors: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    linewidth: float = 0.8,
    markersize: float = 1.0,
    figsize: tuple[float, float] = (12.0, 5.0),
    show_legend: bool = True,
) -> None:
    """Plot solution profiles at selected times with optional endpoint zooms.

    This is a reusable version of the historical ``plot_stride`` routine. It is
    method-agnostic: any number of methods can be compared as long as the input
    table contains one row per ``(solution, t, x)`` value.

    Parameters
    ----------
    df:
        Long-form profile table.
    output_path:
        Figure path.
    solution_column, time_column, x_column, value_column:
        Column names in ``df``.
    solution_order:
        Optional order for the plotted methods. Methods not listed are appended
        in their original order of appearance.
    reference_solution:
        Solution used to place time labels on the main panel. If it is missing,
        the first plotted solution is used.
    profile_times:
        Optional time values to plot. The nearest available times are used.
    zoom_points:
        Optional list of ``(x0, y0)`` centers for the zoom panels.
    delta:
        Zoom half-width. It can be one scalar, one scalar per zoom, or one
        ``(dx, dy)`` tuple per zoom.
    text_locations:
        Optional x-locations for time labels. Use ``None`` entries to place a
        label near the half arc-length of the reference curve.
    colors:
        Optional method colors as a mapping or list.
    linewidth, markersize, figsize:
        Matplotlib styling parameters.
    show_legend:
        Whether to draw the method legend below the main panel.
    """
    required = {solution_column, time_column, x_column, value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Profile data are missing columns: {sorted(missing)}")

    data = df.copy()
    data[time_column] = pd.to_numeric(data[time_column], errors="coerce")
    data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=[solution_column, time_column, x_column, value_column])
    if data.empty:
        raise ValueError("Profile data are empty after removing invalid rows.")

    available_solutions = list(dict.fromkeys(str(value) for value in data[solution_column]))
    if solution_order is None:
        solution_names = available_solutions
    else:
        solution_names = [name for name in solution_order if name in available_solutions]
        solution_names.extend(name for name in available_solutions if name not in solution_names)
    if not solution_names:
        raise ValueError("No valid solution names were found for profile plotting.")

    available_times = np.array(sorted(data[time_column].unique()), dtype=np.float64)
    if profile_times is None:
        times_to_plot = available_times
    else:
        times_to_plot = np.array([available_times[int(np.argmin(np.abs(available_times - float(t))))] for t in profile_times], dtype=np.float64)
        times_to_plot = np.array(list(dict.fromkeys(times_to_plot)), dtype=np.float64)

    method_colors = _profile_method_colors(solution_names, colors)
    zoom_points = zoom_points or []

    fig = plt.figure(figsize=figsize)
    if zoom_points:
        gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.1)
        ax = fig.add_subplot(gs[0])
        zoom_gs = GridSpecFromSubplotSpec(len(zoom_points), 1, subplot_spec=gs[1], hspace=0.4)
        zoom_axes = [fig.add_subplot(zoom_gs[i]) for i in range(len(zoom_points))]
    else:
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        zoom_axes = []

    method_lines: list[Line2D] = []
    reference_for_labels = reference_solution if reference_solution in solution_names else solution_names[0]

    for method_idx, solution_name in enumerate(solution_names):
        solution_data = data[data[solution_column].astype(str) == solution_name]
        marker = MARKERS[method_idx % len(MARKERS)]
        color = method_colors.get(solution_name)
        if color is None:
            probe = ax.plot([], [])[0]
            color = probe.get_color()
            probe.remove()

        method_lines.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                marker=marker,
                markersize=max(markersize * 2.0, markersize),
                linewidth=linewidth,
                label=solution_name,
            )
        )

        for time_idx, time_value in enumerate(times_to_plot):
            profile = solution_data[np.isclose(solution_data[time_column], time_value)].sort_values(x_column)
            if profile.empty:
                continue
            x = profile[x_column].to_numpy(dtype=np.float64)
            y = profile[value_column].to_numpy(dtype=np.float64)
            linestyle = LINESTYLES[time_idx % len(LINESTYLES)]
            for target_ax in [ax] + zoom_axes:
                is_zoom = target_ax in zoom_axes
                target_ax.plot(
                    x,
                    y,
                    linestyle=linestyle,
                    color=color,
                    marker=marker,
                    markersize=markersize * 5.0 if is_zoom else markersize,
                    linewidth=linewidth,
                    label=None,
                )

    label_data = data[data[solution_column].astype(str) == reference_for_labels]
    for time_idx, time_value in enumerate(times_to_plot):
        profile = label_data[np.isclose(label_data[time_column], time_value)].sort_values(x_column)
        if profile.empty:
            continue
        x = profile[x_column].to_numpy(dtype=np.float64)
        y = profile[value_column].to_numpy(dtype=np.float64)
        if text_locations is not None and time_idx < len(text_locations) and text_locations[time_idx] is not None:
            label_index = int(np.argmin(np.abs(x - float(text_locations[time_idx]))))
        else:
            dx_values = np.diff(x)
            dy_values = np.diff(y)
            arc_length = np.insert(np.cumsum(np.sqrt(dx_values**2 + dy_values**2)), 0, 0.0)
            label_index = int(np.searchsorted(arc_length, arc_length[-1] / 2.0))
            label_index = min(max(label_index, 0), len(x) - 1)

        if 1 <= label_index < len(x) - 1:
            slope = (y[label_index + 1] - y[label_index - 1]) / (x[label_index + 1] - x[label_index - 1])
        else:
            slope = 0.0
        angle = float(np.degrees(np.arctan(slope)))
        ax.text(
            x[label_index],
            y[label_index],
            rf"$t={time_value:.2f}$",
            fontsize=9,
            rotation=angle,
            rotation_mode="anchor",
            color="black",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    time_lines = [
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=2.0,
            label=rf"$t={time_value:.2f}$",
        )
        for idx, time_value in enumerate(times_to_plot)
    ]

    method_legend = None
    if show_legend:
        method_legend = Legend(
            ax,
            method_lines,
            [handle.get_label() for handle in method_lines],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, max(1, len(method_lines))),
            frameon=False,
        )
        ax.add_artist(method_legend)

    time_legend = Legend(
        ax,
        time_lines,
        [handle.get_label() for handle in time_lines],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=8,
        handlelength=4.0,
        handletextpad=1.0,
        title="Times",
        title_fontsize=9,
    )
    ax.add_artist(time_legend)

    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$u(x,t)$", fontsize=16)
    ax.grid(True)

    for idx, (zoom_ax, (x0, y0)) in enumerate(zip(zoom_axes, zoom_points)):
        dx_zoom, dy_zoom = _profile_zoom_delta(delta, idx)
        zoom_ax.set_xlim(float(x0) - dx_zoom, float(x0) + dx_zoom)
        zoom_ax.set_ylim(float(y0) - dy_zoom, float(y0) + dy_zoom)
        zoom_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        zoom_ax.set_title(f"Zoom @ ({float(x0):.1f}, {float(y0):.1f})", fontsize=10)
        zoom_ax.grid(True)

    if zoom_points:
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.18, wspace=0.12)
    else:
        fig.tight_layout()
    ensure_parent_dir(output_path)
    extra_artists = [method_legend] if method_legend is not None else []
    fig.savefig(output_path, dpi=600, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def _percentile_lower_limit(
    positive_values: np.ndarray,
    percentile: float,
    margin_factor: float,
) -> float:
    """Return a percentile-based lower limit with a visual margin.

    ``positive_values`` must already be filtered to finite, strictly
    positive values. Returns a small positive fallback when empty.
    """
    if positive_values.size == 0:
        return 1.0e-16
    return float(np.percentile(positive_values, percentile) / margin_factor)


def compute_pooled_log_ylim(
    csv_paths: list[str | Path],
    lower_percentile: float = 3.0,
    lower_margin_factor: float = 10.0,
    upper_margin_factor: float = 1.2,
) -> tuple[float, float] | None:
    """Compute one shared logarithmic y-limit pooled from several error CSVs.

    Every existing CSV in ``csv_paths`` contributes its finite, strictly
    positive ``error_l2`` values to a single pooled sample before the limit
    is computed. Passing the same result to several grouped-error-curve
    figures makes them share identical axis limits and therefore directly
    comparable, even though each figure is produced by a different pipeline
    stage. Paths that do not exist yet are skipped silently, so one stage can
    still be replotted before every other stage has run; once every listed
    CSV exists, rerunning the affected scripts yields the same fully pooled
    bounds everywhere.

    Parameters
    ----------
    csv_paths:
        Paths to grouped error CSVs, each containing an ``error_l2`` column.
    lower_percentile:
        Percentile of the pooled positive values used for the lower limit.
        See :func:`plot_grouped_error_curves_from_dataframe` for why a low
        percentile is used instead of the raw minimum.
    lower_margin_factor:
        Divisor applied to the percentile value for a visual margin below
        the plotted data.
    upper_margin_factor:
        Multiplier applied to the pooled maximum for a visual margin above
        the plotted data.

    Returns
    -------
    tuple[float, float] | None
        ``(lower, upper)`` limits, or ``None`` if none of the listed CSVs
        exist yet or none contain a finite positive ``error_l2`` value. The
        caller should fall back to per-figure automatic limits in that case.
    """
    pooled: list[np.ndarray] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "error_l2" not in df.columns:
            continue
        values = pd.to_numeric(df["error_l2"], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            pooled.append(values)

    if not pooled:
        return None

    all_values = np.concatenate(pooled)
    lower = _percentile_lower_limit(all_values, lower_percentile, lower_margin_factor)
    upper = float(np.max(all_values) * upper_margin_factor)
    return lower, upper


def error_time_source_paths(root: str | Path) -> list[Path]:
    """Return the standard cross-stage ``E(t)`` postprocessed CSV paths.

    Used to pool a shared y-axis limit across every temporal grouped-error
    figure (finite-difference, NAS, PINN, and spectral stages) so they are
    directly comparable. Assumes the standard, unconfigured output
    directories used throughout this repository.
    """
    root = Path(root)
    return [
        root / "results" / "postprocess" / "numerical" / "fdm_error_time.csv",
        root / "results" / "postprocess" / "nas" / "nas_error_time.csv",
        root / "results" / "postprocess" / "pinn" / "pinn_error_time.csv",
        root / "results" / "postprocess" / "spectral" / "spectral_error_time.csv",
        root / "results" / "postprocess" / "spectral" / "spectral_error_time_rmse_path.csv",
    ]


def error_space_source_paths(root: str | Path) -> list[Path]:
    """Return the standard cross-stage ``E(x)`` postprocessed CSV paths.

    Used to pool a shared y-axis limit across every spatial grouped-error
    figure (finite-difference, PINN, and spectral stages) so they are
    directly comparable. Assumes the standard, unconfigured output
    directories used throughout this repository.
    """
    root = Path(root)
    return [
        root / "results" / "postprocess" / "numerical" / "fdm_error_space.csv",
        root / "results" / "postprocess" / "pinn" / "pinn_error_space.csv",
        root / "results" / "postprocess" / "spectral" / "spectral_error_space.csv",
    ]


def plot_grouped_error_curves_from_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    x_column: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    vertical_markers: list[dict[str, Any]] | None = None,
    auto_lower_limit_percentile: float = 3.0,
) -> None:
    """Plot all error curves contained in a grouped DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing at least ``method``, ``label``, ``error_l2``, and
        the coordinate column specified by ``x_column``.
    output_path:
        Figure path written by Matplotlib.
    x_column:
        Coordinate column to draw on the horizontal axis, typically ``t`` or ``x``.
    xlabel, ylabel:
        Axis labels.
    ylim:
        Optional positive limits for the logarithmic vertical axis. Finite values
        at or below the lower limit are displayed at that limit with downward
        triangle markers, while their exact values remain unchanged in the input
        data. A matching triangle entry is added to the method legend. When
        omitted, the lower limit is chosen automatically from
        ``auto_lower_limit_percentile``.
    vertical_markers:
        Optional dictionaries defining vertical reference lines. Each mapping
        must contain ``x`` and may contain ``label``, ``linestyle``,
        ``linewidth``, and ``color``.
    auto_lower_limit_percentile:
        Percentile of the pooled positive ``error_l2`` values used to set the
        automatic lower limit when ``ylim`` is not given, divided by 10 for a
        visual margin. A handful of samples can sit many orders of magnitude
        below the rest of a curve, for example right where a scheme starts
        from the same value as the reference or at a boundary node held
        exactly at its Dirichlet value; using the raw minimum would stretch
        the axis across that isolated dip and compress the informative part
        of the curve into a thin band at the top. The low percentile still
        adapts to the data but ignores that handful of outliers; values below
        it are still shown, clipped to the limit with a triangle marker.
        Ignored when ``ylim`` is given explicitly.
    """
    required = {"method", "label", x_column, "error_l2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Grouped error data are missing columns: {sorted(missing)}")

    if ylim is not None:
        lower_limit = float(ylim[0])
        upper_limit = float(ylim[1])
        if not np.isfinite(lower_limit) or not np.isfinite(upper_limit):
            raise ValueError("Logarithmic axis limits must be finite.")
        if lower_limit <= 0.0 or upper_limit <= lower_limit:
            raise ValueError("Logarithmic axis limits must satisfy 0 < lower < upper.")
    else:
        all_errors = pd.to_numeric(df["error_l2"], errors="coerce").to_numpy(dtype=np.float64)
        positive_errors = all_errors[np.isfinite(all_errors) & (all_errors > 0.0)]
        lower_limit = _percentile_lower_limit(positive_errors, auto_lower_limit_percentile, 10.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted_any = False
    clipped_any = False
    for idx, (_, group) in enumerate(df.groupby("method", sort=False)):
        group = group.sort_values(x_column)
        x = group[x_column].to_numpy(dtype=np.float64)
        y = group["error_l2"].to_numpy(dtype=np.float64)
        finite = np.isfinite(y)
        negative = finite & (y < 0.0)
        if np.any(negative):
            method = str(group["method"].iloc[0])
            raise ValueError(f"Error curve '{method}' contains negative values.")

        clipped = finite & (y <= lower_limit)
        y_plot = y.copy()
        y_plot[clipped] = lower_limit
        valid = np.isfinite(y_plot)
        if not np.any(valid):
            continue

        label = str(group["label"].iloc[0])
        line = ax.semilogy(
            x[valid],
            y_plot[valid],
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=1.3,
            label=label,
        )[0]
        if np.any(clipped):
            ax.scatter(
                x[clipped],
                np.full(np.count_nonzero(clipped), lower_limit, dtype=np.float64),
                marker="v",
                s=24.0,
                facecolor=line.get_color(),
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
                clip_on=False,
            )
            clipped_any = True
        plotted_any = True

    for marker in vertical_markers or []:
        if "x" not in marker:
            raise ValueError("Each vertical marker must contain an 'x' value.")
        ax.axvline(
            x=float(marker["x"]),
            label=marker.get("label"),
            linestyle=str(marker.get("linestyle", "--")),
            linewidth=float(marker.get("linewidth", 1.0)),
            color=marker.get("color", "black"),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    if plotted_any:
        handles, labels = ax.get_legend_handles_labels()
        if clipped_any:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="v",
                    markersize=6.5,
                    markerfacecolor="0.35",
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    label=f"value ≤ {lower_limit:.0e}",
                )
            )
            labels.append(f"value ≤ {lower_limit:.0e}")
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    save_figure(output_path)
    plt.close(fig)


def plot_grouped_error_curves(
    input_path: str | Path,
    output_path: str | Path,
    x_column: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    vertical_markers: list[dict[str, Any]] | None = None,
) -> None:
    """Plot all error curves contained in a grouped CSV file.

    Parameters
    ----------
    input_path:
        CSV file containing at least ``method``, ``label``, ``error_l2``, and
        the coordinate column specified by ``x_column``.
    output_path:
        Figure path written by Matplotlib.
    x_column:
        Coordinate column to draw on the horizontal axis, typically ``t`` or ``x``.
    xlabel, ylabel:
        Axis labels.
    ylim:
        Optional positive limits for the logarithmic vertical axis.

    Notes
    -----
    The function is method-agnostic. It can be reused for finite-difference,
    NAS, PINN, or spectral postprocessing curves as long as the input CSV follows
    the grouped long-format convention.
    """
    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    plot_grouped_error_curves_from_dataframe(
        df,
        output_path,
        x_column,
        xlabel,
        ylabel,
        ylim=ylim,
        vertical_markers=vertical_markers,
    )


def plot_log_heatmap_from_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    x_column: str,
    y_column: str,
    value_column: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    path_flag_column: str | None = None,
    path_label: str = "Minimum-sensitivity path",
    secondary_path_flag_column: str | None = None,
    secondary_path_label: str = "Secondary path",
    marker_flag_column: str | None = None,
    marker_label: str = "Selected",
    cmap: str = "viridis",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a sparse positive-valued heatmap on a logarithmic color scale.

    Parameters
    ----------
    df:
        Long-form table containing one row per tested coordinate pair.
    output_path:
        Figure path written by Matplotlib.
    x_column, y_column, value_column:
        Column names defining the heatmap coordinates and positive values.
    xlabel, ylabel, colorbar_label:
        Figure labels.
    path_flag_column:
        Optional Boolean column identifying the primary path to overlay.
    path_label:
        Legend label for the primary path.
    secondary_path_flag_column:
        Optional Boolean column identifying a second path to overlay.
    secondary_path_label:
        Legend label for the secondary path.
    marker_flag_column:
        Optional Boolean column identifying one highlighted point.
    marker_label:
        Legend label for the highlighted point.
    cmap:
        Matplotlib colormap name.
    xlim, ylim:
        Optional explicit axis limits.
    """
    required = {x_column, y_column, value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Heatmap data are missing columns: {sorted(missing)}")

    data = df.copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data.loc[~np.isfinite(data[value_column]) | (data[value_column] <= 0.0), value_column] = np.nan
    valid_values = data[value_column].dropna().to_numpy(dtype=np.float64)
    if valid_values.size == 0:
        raise ValueError("Heatmap data contain no finite positive values.")

    x_values = np.sort(data[x_column].dropna().unique().astype(np.float64))
    y_values = np.sort(data[y_column].dropna().unique().astype(np.float64))
    matrix = (
        data.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="first")
        .reindex(index=y_values, columns=x_values)
        .to_numpy(dtype=np.float64)
    )

    vmin = float(np.nanmin(valid_values))
    vmax = float(np.nanmax(valid_values))
    if np.isclose(vmin, vmax, rtol=0.0, atol=0.0):
        vmax = vmin * (1.0 + 1e-12)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    mesh = ax.pcolormesh(
        x_values,
        y_values,
        matrix,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="nearest",
        cmap=cmap,
    )
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    def flag_mask(column: str) -> pd.Series:
        """Return a boolean mask for a column that may be stored as bool or text."""
        values = data[column]
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    if path_flag_column is not None:
        if path_flag_column not in data.columns:
            raise ValueError(f"Heatmap path flag column not found: {path_flag_column}")
        path_rows = data[flag_mask(path_flag_column)].sort_values(x_column)
        if not path_rows.empty:
            ax.plot(
                path_rows[x_column],
                path_rows[y_column],
                linestyle="--",
                linewidth=1.5,
                color="black",
                label=path_label,
            )

    if secondary_path_flag_column is not None:
        if secondary_path_flag_column not in data.columns:
            raise ValueError(
                f"Heatmap secondary path flag column not found: {secondary_path_flag_column}"
            )
        secondary_rows = data[flag_mask(secondary_path_flag_column)].sort_values(x_column)
        if not secondary_rows.empty:
            ax.plot(
                secondary_rows[x_column],
                secondary_rows[y_column],
                linestyle="-",
                linewidth=1.3,
                marker="s",
                markersize=3.5,
                color="white",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=secondary_path_label,
            )

    if marker_flag_column is not None:
        if marker_flag_column not in data.columns:
            raise ValueError(f"Heatmap marker flag column not found: {marker_flag_column}")
        marker_rows = data[flag_mask(marker_flag_column)]
        if len(marker_rows) > 1:
            raise ValueError(f"Expected at most one highlighted heatmap point, found {len(marker_rows)}.")
        if not marker_rows.empty:
            row = marker_rows.iloc[0]
            ax.plot(
                float(row[x_column]),
                float(row[y_column]),
                marker="o",
                markersize=5.0,
                color="red",
                linestyle="none",
                label=marker_label,
            )
            ax.annotate(
                f"{float(row[value_column]):.1e}",
                xy=(float(row[x_column]), float(row[y_column])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if (
        path_flag_column is not None
        or secondary_path_flag_column is not None
        or marker_flag_column is not None
    ):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(output_path)
    plt.close(fig)



def plot_training_loss_history_from_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    x_column: str = "global_epoch",
    value_column: str = "loss_value",
    series_column: str = "loss_name",
    label_column: str = "label",
    xlabel: str = "Global outer epoch",
    ylabel: str = "Loss",
) -> None:
    """Plot multiple training-loss series on a logarithmic vertical axis.

    Parameters
    ----------
    df:
        Long-format training history. Each row represents one loss value at one
        outer epoch.
    output_path:
        Figure path written by Matplotlib.
    x_column, value_column:
        Columns containing the horizontal coordinate and loss magnitude.
    series_column, label_column:
        Columns identifying each curve and its display label.
    xlabel, ylabel:
        Axis labels.

    Notes
    -----
    The function does not rescale or recompute any loss. This is important for
    PINN histories, where the total objective is weighted while IC, BC, and
    physics components are stored as unscaled mean-squared losses.
    """
    required = {x_column, value_column, series_column, label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Training-history data are missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plotted_any = False
    for idx, (_, group) in enumerate(df.groupby(series_column, sort=False)):
        group = group.sort_values(x_column)
        x = pd.to_numeric(group[x_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        y = pd.to_numeric(group[value_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if not np.any(valid):
            continue
        label = str(group[label_column].iloc[0])
        ax.semilogy(
            x[valid],
            y[valid],
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=1.2,
            label=label,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        raise ValueError("Training history contains no finite positive loss values.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(output_path)
    plt.close(fig)


def plot_training_loss_history(
    input_path: str | Path,
    output_path: str | Path,
    x_column: str = "global_epoch",
    value_column: str = "loss_value",
    series_column: str = "loss_name",
    label_column: str = "label",
    xlabel: str = "Global outer epoch",
    ylabel: str = "Loss",
) -> None:
    """Read a long-format training history and plot all recorded loss curves."""
    df = pd.read_csv(Path(input_path))
    plot_training_loss_history_from_dataframe(
        df=df,
        output_path=output_path,
        x_column=x_column,
        value_column=value_column,
        series_column=series_column,
        label_column=label_column,
        xlabel=xlabel,
        ylabel=ylabel,
    )

def dataframe_to_xy(df: pd.DataFrame, time_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convert a long-form solution DataFrame into input and target arrays.

    Parameters
    ----------
    df:
        DataFrame whose first two columns are interpreted as ``(t, x)`` and
        whose third column is interpreted as the scalar target ``u``. This
        preserves the convention used by the original NAS scripts.
    time_max:
        Optional maximum time used to filter the rows. In the supervised NAS
        stage this is the end of the known-data interval, usually ``t=1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` with shape ``(n_samples, 2)`` and ``y`` with shape
        ``(n_samples,)``.
    """
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    if time_max is not None:
        mask = X[:, 0] <= float(time_max)
        X = X[mask, :]
        y = y[mask]
    return X, y


def xy_to_tensor(X: np.ndarray, y: np.ndarray):
    """Convert input and target arrays to float64 PyTorch tensors.

    Parameters
    ----------
    X:
        Input array with shape ``(n_samples, 2)``.
    y:
        Target array with shape ``(n_samples,)`` or ``(n_samples, 1)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Input tensor and column target tensor in double precision.
    """
    import torch

    X_tensor = torch.as_tensor(X, dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.float64).view(-1, 1)
    return X_tensor, y_tensor



def relative_path_string(path: str | Path, root: str | Path) -> str:
    """Return a stable path string relative to ``root`` when possible.

    Parameters
    ----------
    path:
        Path to display or serialize.
    root:
        Project root used as the relative-path anchor.

    Returns
    -------
    str
        Relative path string if ``path`` is inside ``root``; otherwise the
        original absolute path string.
    """
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)




def architecture_key(activation: str, num_layers: int, neurons_per_layer: int) -> str:
    """Return the canonical NAS architecture key.

    Parameters
    ----------
    activation:
        Activation function name.
    num_layers:
        Number of hidden layers.
    neurons_per_layer:
        Number of neurons in each hidden layer.

    Returns
    -------
    str
        Canonical key such as ``tanh_8x90``.
    """
    return f"{str(activation).lower()}_{int(num_layers)}x{int(neurons_per_layer)}"


def validate_nas_generation_config(training: dict[str, Any]) -> None:
    """Validate that the NAS configuration has complete generations.

    Parameters
    ----------
    training:
        Training block from ``configs/nas.yaml``.

    Raises
    ------
    KeyError
        If ``n_trials``, ``population_size``, or ``generations`` is missing.
    ValueError
        If ``population_size * generations != n_trials``.
    """
    required = ["n_trials", "population_size", "generations"]
    missing = [key for key in required if key not in training]
    if missing:
        raise KeyError(
            "NAS training configuration is missing required keys: "
            f"{missing}. Add n_trials, population_size, and generations."
        )

    n_trials = int(training["n_trials"])
    population_size = int(training["population_size"])
    generations = int(training["generations"])
    if population_size <= 0 or generations <= 0 or n_trials <= 0:
        raise ValueError("n_trials, population_size, and generations must be positive integers.")
    expected = population_size * generations
    if expected != n_trials:
        raise ValueError(
            "Invalid NAS generation configuration: "
            f"population_size * generations = {population_size} * {generations} = {expected}, "
            f"but n_trials = {n_trials}. Use complete generations only."
        )


def _normalise_pair_path(path: str | Path | None) -> Path | None:
    """Return a resolved path or ``None`` for missing path-like values."""
    if path is None:
        return None
    value = str(path)
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return Path(value)


def find_architecture_pair_in_flat_cache(architecture: str, cache_dir: str | Path) -> tuple[Path, Path] | None:
    """Find a valid flat-cache pair for one NAS architecture.

    A cache hit is valid only when both ``<architecture>.parquet`` and
    ``<architecture>.pth`` exist in the same flat cache directory.
    """
    cache_dir = Path(cache_dir)
    history_path = cache_dir / f"{architecture}.parquet"
    model_path = cache_dir / f"{architecture}.pth"
    if history_path.exists() and model_path.exists():
        return history_path, model_path
    return None


def find_architecture_pair_in_study_outputs(
    architecture: str,
    histories_dir: str | Path,
    models_dir: str | Path,
    exclude_history_path: str | Path | None = None,
    exclude_model_path: str | Path | None = None,
) -> tuple[Path, Path] | None:
    """Find an already completed architecture inside current study outputs.

    Parameters
    ----------
    architecture:
        Canonical architecture key, for example ``tanh_8x90``.
    histories_dir, models_dir:
        Current-study output directories.
    exclude_history_path, exclude_model_path:
        Optional trial paths that should not be used as a source for copying.

    Returns
    -------
    tuple[Path, Path] | None
        Matching ``(history_path, model_path)`` pair if both files exist.
    """
    histories_dir = Path(histories_dir)
    models_dir = Path(models_dir)
    exclude_history = _normalise_pair_path(exclude_history_path)
    exclude_model = _normalise_pair_path(exclude_model_path)
    exclude_history = exclude_history.resolve() if exclude_history is not None and exclude_history.exists() else exclude_history
    exclude_model = exclude_model.resolve() if exclude_model is not None and exclude_model.exists() else exclude_model

    if not histories_dir.exists() or not models_dir.exists():
        return None

    history_candidates: list[Path] = []
    for history_path in sorted(histories_dir.rglob("*.parquet")):
        if exclude_history is not None and history_path.resolve() == exclude_history:
            continue
        metadata = parse_nas_history_metadata(history_path)
        if str(metadata.get("architecture_name")) == architecture:
            history_candidates.append(history_path)

    for history_path in history_candidates:
        model_path = find_matching_nas_model_path(history_path, models_dir)
        if model_path is None:
            continue
        if exclude_model is not None and model_path.exists() and model_path.resolve() == exclude_model:
            continue
        if model_path.exists():
            return history_path, model_path
    return None


def copy_architecture_pair_to_trial(
    source_history: str | Path,
    source_model: str | Path,
    target_history: str | Path,
    target_model: str | Path,
) -> None:
    """Copy an existing architecture pair into the current trial filenames.

    Parameters
    ----------
    source_history, source_model:
        Existing valid pair.
    target_history, target_model:
        Trial-specific output paths.
    """
    source_history = Path(source_history)
    source_model = Path(source_model)
    target_history = Path(target_history)
    target_model = Path(target_model)
    if not source_history.exists() or not source_model.exists():
        raise FileNotFoundError(
            "Cannot copy cached architecture because the source pair is incomplete: "
            f"history={source_history}, model={source_model}"
        )
    ensure_parent_dir(target_history)
    ensure_parent_dir(target_model)
    if source_history.resolve() != target_history.resolve():
        shutil.copy2(source_history, target_history)
    if source_model.resolve() != target_model.resolve():
        shutil.copy2(source_model, target_model)


def read_nas_objectives_from_history(history_path: str | Path) -> dict[str, Any]:
    """Read NAS objectives and diagnostics from a history parquet file.

    The NAS objective is the unscaled known-data MSE. For current histories this
    is stored in ``LossBC``. The scaled legacy training loss is preserved as
    ``final_scaled_loss`` from the ``Loss`` column.
    """
    history_path = Path(history_path)
    history = pd.read_parquet(history_path)
    if history.empty:
        raise ValueError(f"NAS history is empty: {history_path}")

    mse_column = "LossBC" if "LossBC" in history.columns else "Loss" if "Loss" in history.columns else None
    if mse_column is None:
        raise ValueError(f"NAS history does not contain LossBC or Loss: {history_path}")

    final_mse = float(history[mse_column].iloc[-1])
    min_mse = float(history[mse_column].min())
    final_scaled_loss = float(history["Loss"].iloc[-1]) if "Loss" in history.columns else float("nan")
    min_scaled_loss = float(history["Loss"].min()) if "Loss" in history.columns else float("nan")

    if "Time_optimizer" in history.columns:
        total_time_seconds = float(history["Time_optimizer"].sum())
    elif "Time" in history.columns:
        total_time_seconds = float(history["Time"].sum())
    else:
        total_time_seconds = float("nan")

    total_time_cpu = float(history["Time_cpu"].sum()) if "Time_cpu" in history.columns else float("nan")
    flops = int(history["Flops"].sum()) if "Flops" in history.columns else 0
    final_epoch = int(history["Epoch"].iloc[-1]) if "Epoch" in history.columns else int(len(history))

    return {
        "final_mse": final_mse,
        "min_mse": min_mse,
        "final_loss": final_mse,
        "min_loss": min_mse,
        "final_scaled_loss": final_scaled_loss,
        "min_scaled_loss": min_scaled_loss,
        "total_time_seconds": total_time_seconds,
        "total_time_optimizer": total_time_seconds,
        "total_time_cpu": total_time_cpu,
        "flops": flops,
        "final_epoch": final_epoch,
        "epochs": int(len(history)),
    }


def get_trial_generation_safe(sampler: Any, study: Any, trial: Any, population_size: int) -> int:
    """Return the Optuna NSGA-II generation with a deterministic fallback."""
    try:
        return int(sampler.get_trial_generation(study, trial))
    except Exception:  # noqa: BLE001 - compatibility with older Optuna versions.
        return int(trial.number) // int(population_size)


def study_to_nas_trials_dataframe(study: Any, root: str | Path) -> pd.DataFrame:
    """Convert an Optuna study to the NAS trials CSV schema.

    Parameters
    ----------
    study:
        Optuna study object.
    root:
        Project root used for relative path normalization when user attributes
        already contain absolute paths.
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        values = trial.values or [float("nan"), float("nan")]
        final_mse = trial.user_attrs.get("final_mse", values[0])
        total_time_seconds = trial.user_attrs.get("total_time_seconds", values[1])
        history_path = trial.user_attrs.get("history_path", "")
        model_path = trial.user_attrs.get("model_path", "")
        rows.append(
            {
                "trial_number": trial.number,
                "generation": trial.user_attrs.get("generation"),
                "state": trial.state.name,
                "num_layers": trial.params.get("num_layers"),
                "neurons_per_layer": trial.params.get("neurons_per_layer"),
                "activation": trial.params.get("activation"),
                "architecture_name": trial.user_attrs.get("architecture_name"),
                "architecture_key": trial.user_attrs.get("architecture_key", trial.user_attrs.get("architecture_name")),
                "final_mse": final_mse,
                "min_mse": trial.user_attrs.get("min_mse"),
                "final_loss": final_mse,
                "min_loss": trial.user_attrs.get("min_mse"),
                "final_scaled_loss": trial.user_attrs.get("final_scaled_loss"),
                "min_scaled_loss": trial.user_attrs.get("min_scaled_loss"),
                "total_time_seconds": total_time_seconds,
                "total_time_optimizer": total_time_seconds,
                "total_time_cpu": trial.user_attrs.get("total_time_cpu"),
                "flops": trial.user_attrs.get("flops"),
                "final_epoch": trial.user_attrs.get("final_epoch"),
                "epochs": trial.user_attrs.get("epochs", trial.user_attrs.get("final_epoch")),
                "cache_hit": trial.user_attrs.get("cache_hit"),
                "cache_source": trial.user_attrs.get("cache_source"),
                "history_path": history_path,
                "model_path": model_path,
            }
        )

    trials = pd.DataFrame(rows)
    if not trials.empty:
        trials = trials.sort_values("trial_number").reset_index(drop=True)
    return trials

def parse_nas_history_metadata(history_path: str | Path) -> dict[str, Any]:
    """Parse NAS architecture metadata from a history parquet file name.

    The parser intentionally accepts several naming styles used by historical
    scripts, for example ``trial000_tanh_8x90``, ``trial_000_tanh_8x90``,
    ``tanh_8x90``, and ``8x90_tanh``.

    Parameters
    ----------
    history_path:
        Path to a saved NAS history parquet file.

    Returns
    -------
    dict[str, Any]
        Parsed trial number, architecture name, activation, number of hidden
        layers, and neurons per hidden layer. Unknown fields are returned as
        ``None`` while the architecture name falls back to the file stem.
    """
    stem = Path(history_path).stem
    trial_match = re.search(r"trial[_-]?(\d+)", stem, flags=re.IGNORECASE)
    trial_number = int(trial_match.group(1)) if trial_match else None

    activation = None
    num_layers = None
    neurons_per_layer = None

    match = re.search(r"(relu|tanh|sigmoid)[_-](\d+)x(\d+)", stem, flags=re.IGNORECASE)
    if match:
        activation = match.group(1).lower()
        num_layers = int(match.group(2))
        neurons_per_layer = int(match.group(3))
    else:
        match = re.search(r"(\d+)x(\d+)[_-](relu|tanh|sigmoid)", stem, flags=re.IGNORECASE)
        if match:
            num_layers = int(match.group(1))
            neurons_per_layer = int(match.group(2))
            activation = match.group(3).lower()

    if activation is None or num_layers is None or neurons_per_layer is None:
        architecture_name = stem
    else:
        architecture_name = f"{activation}_{num_layers}x{neurons_per_layer}"

    return {
        "trial_number": trial_number,
        "architecture_name": architecture_name,
        "architecture": architecture_name,
        "activation": activation,
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
    }


def find_matching_nas_model_path(history_path: str | Path, models_dir: str | Path | None) -> Path | None:
    """Find the saved PyTorch model that matches a NAS history file.

    Parameters
    ----------
    history_path:
        History parquet file.
    models_dir:
        Directory that stores NAS ``.pth`` files. If it is missing, ``None`` is
        returned.

    Returns
    -------
    Path | None
        Matching model path, or ``None`` when no match is found.
    """
    history_path = Path(history_path)
    if models_dir is None:
        return None

    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None

    exact = list(models_dir.rglob(f"{history_path.stem}.pth"))
    if exact:
        return exact[0]

    metadata = parse_nas_history_metadata(history_path)
    architecture = str(metadata["architecture_name"])
    trial_number = metadata["trial_number"]
    trial_tokens = []
    if trial_number is not None:
        trial_tokens = [
            f"trial{trial_number}",
            f"trial{trial_number:03d}",
            f"trial_{trial_number}",
            f"trial_{trial_number:03d}",
            f"trial-{trial_number}",
            f"trial-{trial_number:03d}",
        ]

    candidates = sorted(models_dir.rglob("*.pth"))
    for candidate in candidates:
        stem = candidate.stem.lower()
        if history_path.stem.lower() in stem or stem in history_path.stem.lower():
            return candidate

    for candidate in candidates:
        stem = candidate.stem.lower()
        has_architecture = architecture.lower() in stem
        has_trial = not trial_tokens or any(token.lower() in stem for token in trial_tokens)
        if has_architecture and has_trial:
            return candidate

    return None


def build_partial_nas_trials_from_histories(
    histories_dir: str | Path,
    models_dir: str | Path | None,
    root: str | Path,
    include_manual: bool = False,
) -> pd.DataFrame:
    """Build a provisional NAS trial table from finished history parquet files.

    This function is intended for monitoring a NAS run before Optuna has written
    its final ``nas_trials.csv``. It only sees trials whose history parquet files
    already exist. Files that are still being written or cannot be read are
    skipped with a warning.

    Parameters
    ----------
    histories_dir:
        Directory containing NAS history parquet files.
    models_dir:
        Directory containing saved PyTorch models. Matching models are recorded
        when available.
    root:
        Project root used to store relative paths in the generated table.
    include_manual:
        Whether histories inside folders named ``manual`` should be included.

    Returns
    -------
    pd.DataFrame
        Provisional trials table with a schema compatible with the final NAS
        CSV used by plotting and postprocessing scripts.
    """
    histories_dir = Path(histories_dir)
    root = Path(root)

    if not histories_dir.exists():
        raise FileNotFoundError(f"NAS histories directory not found: {histories_dir}")

    parquet_files = sorted(histories_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No NAS history parquet files were found under: {histories_dir}")

    rows: list[dict[str, Any]] = []
    for history_path in parquet_files:
        parts_lower = {part.lower() for part in history_path.parts}
        if not include_manual and "manual" in parts_lower:
            continue

        try:
            objectives = read_nas_objectives_from_history(history_path)
        except Exception as exc:  # noqa: BLE001 - deliberate monitoring fallback.
            print(f"Skipping unreadable or invalid NAS history file: {history_path} ({exc})")
            continue

        metadata = parse_nas_history_metadata(history_path)
        model_path = find_matching_nas_model_path(history_path, models_dir)

        row = {
            **metadata,
            "state": "COMPLETE",
            "generation": np.nan,
            **objectives,
            "history_path": relative_path_string(history_path, root),
            "model_path": relative_path_string(model_path, root) if model_path is not None else "",
            "cache_hit": np.nan,
            "cache_source": "partial_history_folder",
            "source": "partial_history_folder",
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No valid completed NAS history parquet files were found under: {histories_dir}")

    trials = pd.DataFrame(rows)
    sort_columns = [column for column in ["trial_number", "final_mse", "total_time_seconds"] if column in trials.columns]
    if sort_columns:
        trials = trials.sort_values(sort_columns, na_position="last").reset_index(drop=True)
    return trials


def load_or_build_nas_trials_table(
    trials_csv: str | Path,
    histories_dir: str | Path,
    models_dir: str | Path | None,
    partial_trials_csv: str | Path,
    root: str | Path,
    include_manual: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load final NAS trials or build a provisional table from histories.

    Parameters
    ----------
    trials_csv:
        Final trials CSV written by the NAS stage.
    histories_dir:
        NAS histories directory used as fallback while the NAS is still running.
    models_dir:
        NAS models directory used to associate histories with saved models.
    partial_trials_csv:
        Path where the provisional table is saved.
    root:
        Project root used to serialize relative paths.
    include_manual:
        Whether manual diagnostic histories should be included in the fallback.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Trials table and source label: ``final_trials_csv`` or
        ``partial_history_folder``.
    """
    trials_csv = Path(trials_csv)
    if trials_csv.exists():
        trials = pd.read_csv(trials_csv)
        return trials, "final_trials_csv"

    print(
        "Final NAS trials CSV was not found. "
        "Building a provisional table from finished history parquet files."
    )
    trials = build_partial_nas_trials_from_histories(
        histories_dir=histories_dir,
        models_dir=models_dir,
        root=root,
        include_manual=include_manual,
    )
    partial_trials_csv = Path(partial_trials_csv)
    ensure_parent_dir(partial_trials_csv)
    trials.to_csv(partial_trials_csv, index=False)
    print(f"Partial NAS trials table written to: {partial_trials_csv}")
    return trials, "partial_history_folder"

def predict_with_model(X: np.ndarray, model_path: str | Path | None = None, model: Any = None, weights_path: str | Path | None = None) -> np.ndarray:
    """Evaluate a saved or in-memory neural-network model.

    Parameters
    ----------
    X:
        Input array with shape ``(n_samples, 2)``.
    model_path:
        Path to a PyTorch model saved with ``torch.save``. Required when
        ``model`` is not provided.
    model:
        Optional in-memory model. If provided, ``model_path`` is ignored.
    weights_path:
        Optional state-dict path loaded into the model before prediction.

    Returns
    -------
    np.ndarray
        One-dimensional predicted values.
    """
    import torch

    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided.")
        model = torch.load(Path(model_path), weights_only=False)
    model = model.to(torch.float64)
    model.eval()

    if weights_path is not None:
        model.load_state_dict(torch.load(Path(weights_path)))

    X_tensor = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        y_tensor = model.forward(X_tensor)
    return y_tensor.detach().cpu().numpy().ravel()
