"""Temporal spectral-analysis utilities for late-time PINN fields.

The implementation follows the article definitions directly. A temporal DFT is
computed independently at every spatial node, the two-sided energy is averaged
across space, and the smallest conjugate-symmetric low-pass band retaining the
configured energy fraction is selected. The field before ``t_cut`` is copied
without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pinn_spectral.benchmark import BenchmarkConfig
from pinn_spectral.pinn import (
    build_pinn_output_paths,
    build_progressive_window_schedule,
    build_window_artifacts,
)
from pinn_spectral.tools import (
    output_name,
    read_solution_matrix,
    solution_dataframe_from_matrix,
)


@dataclass(frozen=True)
class CompletedPinnWindows:
    """Continuous prefix of atomically completed progressive PINN windows."""

    configured_schedule: np.ndarray
    completed_schedule: np.ndarray
    markers: tuple[dict[str, Any], ...]

    @property
    def training_complete(self) -> bool:
        """Return whether every configured PINN window is complete."""
        return self.completed_schedule.size == self.configured_schedule.size

    @property
    def processed_final_time(self) -> float:
        """Return the latest completed progressive-window endpoint."""
        if self.completed_schedule.size == 0:
            raise RuntimeError("No completed PINN window is available.")
        return float(self.completed_schedule[-1])


@dataclass(frozen=True)
class SpectralFilterResult:
    """Result of one conjugate-symmetric temporal low-pass reconstruction."""

    filtered_segment: np.ndarray
    frequencies: np.ndarray
    spectral_energy: np.ndarray
    retained_mask: np.ndarray
    cutoff_frequency: float
    cutoff_shell: int
    retained_energy_fraction: float
    retained_bin_count: int
    total_bin_count: int
    maximum_imaginary_residual: float


@dataclass(frozen=True)
class SpectralOutputPaths:
    """Paths written by the spectral data-generation stage."""

    data_dir: Path
    postprocess_dir: Path
    figure_dir: Path
    metrics_dir: Path
    sweep: Path
    filtered_solution: Path
    spectrum: Path
    generation_metadata: Path
    postprocess_metadata: Path
    reference_window_sweep: Path
    temporal_stability_sweep: Path
    metric_path_comparison: Path


@dataclass(frozen=True)
class ReducedSensitivityRegimeFit:
    """BIC comparison used to identify a reduced-sensitivity onset.

    The fit is performed on ``log10(E_sens)`` at one fixed ``tmax``. ``M0``
    uses one log-linear trend. ``M1`` uses a continuous two-regime trend with
    one discrete change point. The post-change slope may remain non-zero, but
    its absolute magnitude must be smaller than the initial descending slope.
    The selected ``tau`` is therefore interpreted as the onset of a
    reduced-sensitivity regime, not as the start of a horizontal segment.
    """

    tmax: float
    n_points: int
    minimum_points_per_segment: int
    preferred_model: str
    tau: float
    m0_intercept: float
    m0_slope: float
    m0_sse: float
    m0_bic: float
    m1_intercept: float
    m1_slope_before: float
    m1_slope_after: float
    m1_sse: float
    m1_bic: float
    delta_bic_m1_minus_m0: float
    tcut: np.ndarray
    log10_sensitivity: np.ndarray
    m0_fitted: np.ndarray
    m1_fitted: np.ndarray


def _resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative_path(path: Path, root: Path) -> str:
    """Serialize a path relative to the repository root when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _temporary_path(path: Path) -> Path:
    """Return the temporary sibling used for one atomic file replacement."""
    return path.with_suffix(path.suffix + ".tmp")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file and atomically install the completed artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    df.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    """Write one JSON object and atomically install the completed artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


def build_spectral_output_paths(
    config: dict[str, Any],
    root: str | Path,
) -> SpectralOutputPaths:
    """Build all paths used by spectral generation and postprocessing."""
    root = Path(root)
    outputs = config.get("outputs", {})
    data_dir = _resolve_path(root, outputs.get("data_dir", "data/spectral"))
    postprocess_dir = _resolve_path(root, outputs.get("postprocess_dir", "results/postprocess/spectral"))
    figure_dir = _resolve_path(root, outputs.get("figure_dir", "results/figures/spectral"))
    metrics_dir = _resolve_path(root, outputs.get("metrics_dir", "results/metrics"))
    return SpectralOutputPaths(
        data_dir=data_dir,
        postprocess_dir=postprocess_dir,
        figure_dir=figure_dir,
        metrics_dir=metrics_dir,
        sweep=data_dir / str(outputs.get("sweep_file", "spectral_sweep.parquet")),
        filtered_solution=data_dir
        / str(outputs.get("filtered_solution_file", "selected_filtered_solution.parquet")),
        spectrum=data_dir / str(outputs.get("spectrum_file", "selected_spectrum.parquet")),
        generation_metadata=metrics_dir
        / str(outputs.get("generation_metadata_file", "spectral_generation_metadata.json")),
        postprocess_metadata=metrics_dir
        / str(outputs.get("postprocess_metadata_file", "spectral_diagnostics_metadata.json")),
        reference_window_sweep=postprocess_dir
        / str(outputs.get("reference_window_sweep_file", "spectral_reference_window_sweep.csv")),
        temporal_stability_sweep=postprocess_dir
        / str(
            outputs.get(
                "temporal_stability_sweep_file",
                "spectral_temporal_stability_sweep.csv",
            )
        ),
        metric_path_comparison=postprocess_dir
        / str(
            outputs.get(
                "metric_path_comparison_file",
                "spectral_metric_path_comparison.csv",
            )
        ),
    )


def _validate_segment(u_segment: np.ndarray, dt: float, retained_energy_fraction: float) -> np.ndarray:
    """Validate and normalize one temporal signal matrix."""
    u_segment = np.asarray(u_segment, dtype=np.float64)
    if u_segment.ndim != 2:
        raise ValueError("u_segment must have shape (nx, nt_segment).")
    if u_segment.shape[1] < 2:
        raise ValueError("At least two temporal samples are required for a DFT reconstruction.")
    if not np.all(np.isfinite(u_segment)):
        raise ValueError("u_segment contains non-finite values.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value.")
    if not 0.0 < retained_energy_fraction <= 1.0:
        raise ValueError("retained_energy_fraction must lie in (0, 1].")
    return u_segment


def symmetric_energy_lowpass(
    u_segment: np.ndarray,
    dt: float,
    retained_energy_fraction: float = 0.95,
) -> SpectralFilterResult:
    """Filter a temporal segment using the article's two-sided energy rule.

    Parameters
    ----------
    u_segment:
        Matrix with shape ``(nx, nt_segment)``. The transform is applied along
        the temporal axis.
    dt:
        Uniform temporal sampling interval.
    retained_energy_fraction:
        Target fraction ``q`` in the article. The smallest symmetric frequency
        shell whose cumulative two-sided energy reaches ``q`` is retained.

    Returns
    -------
    SpectralFilterResult
        Filtered segment and complete spectral diagnostics.
    """
    u_segment = _validate_segment(u_segment, dt, retained_energy_fraction)
    n_time = int(u_segment.shape[1])
    spectrum = np.fft.fft(u_segment, axis=1)
    spectral_energy = np.mean(np.abs(spectrum) ** 2, axis=0)
    total_energy = float(np.sum(spectral_energy))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("The temporal segment has zero or invalid discrete spectral energy.")

    frequency_indices = np.arange(n_time, dtype=np.int64)
    shell_indices = np.minimum(frequency_indices, n_time - frequency_indices)
    shell_count = n_time // 2 + 1
    shell_energy = np.bincount(
        shell_indices,
        weights=spectral_energy,
        minlength=shell_count,
    )[:shell_count]
    cumulative_fraction = np.cumsum(shell_energy) / total_energy
    cutoff_shell = int(np.searchsorted(cumulative_fraction, retained_energy_fraction, side="left"))
    cutoff_shell = min(cutoff_shell, shell_count - 1)
    retained_mask = shell_indices <= cutoff_shell

    filtered_complex = np.fft.ifft(spectrum * retained_mask[np.newaxis, :], axis=1)
    maximum_imaginary_residual = float(np.max(np.abs(filtered_complex.imag)))
    filtered_segment = filtered_complex.real.astype(np.float64, copy=False)
    frequencies = np.fft.fftfreq(n_time, d=float(dt)).astype(np.float64)
    cutoff_frequency = float(cutoff_shell / (n_time * float(dt)))
    achieved_fraction = float(np.sum(spectral_energy[retained_mask]) / total_energy)

    return SpectralFilterResult(
        filtered_segment=filtered_segment,
        frequencies=frequencies,
        spectral_energy=spectral_energy.astype(np.float64),
        retained_mask=retained_mask,
        cutoff_frequency=cutoff_frequency,
        cutoff_shell=cutoff_shell,
        retained_energy_fraction=achieved_fraction,
        retained_bin_count=int(np.count_nonzero(retained_mask)),
        total_bin_count=n_time,
        maximum_imaginary_residual=maximum_imaginary_residual,
    )


def find_time_index(t: np.ndarray, requested_time: float, dt: float) -> int:
    """Return the grid index matching a requested time within a strict tolerance."""
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a non-empty one-dimensional array.")
    index = int(np.argmin(np.abs(t - float(requested_time))))
    tolerance = max(1e-12, abs(float(dt)) * 1e-7)
    if abs(float(t[index]) - float(requested_time)) > tolerance:
        raise ValueError(
            f"Requested time {requested_time:.15g} is not present on the temporal grid; "
            f"nearest value is {t[index]:.15g}."
        )
    return index


def filter_field_after_cutoff(
    u: np.ndarray,
    t: np.ndarray,
    t_cut: float,
    dt: float,
    retained_energy_fraction: float = 0.95,
) -> tuple[np.ndarray, SpectralFilterResult, int]:
    """Filter ``u`` from ``t_cut`` onward and preserve all earlier samples.

    The returned field satisfies Eq. (41) of the article: indices strictly
    before the cutoff are copied exactly, while the cutoff sample and all later
    samples are replaced by the IDFT reconstruction.
    """
    u = np.asarray(u, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != t.size:
        raise ValueError("u must have shape (nx, len(t)).")
    cutoff_index = find_time_index(t, t_cut, dt)
    result = symmetric_energy_lowpass(
        u[:, cutoff_index:],
        dt=dt,
        retained_energy_fraction=retained_energy_fraction,
    )
    filtered = u.copy()
    filtered[:, cutoff_index:] = result.filtered_segment
    return filtered, result, cutoff_index


def compute_adjacent_training_discrepancy(
    earlier_field: np.ndarray,
    earlier_time: np.ndarray,
    later_field_on_common_grid: np.ndarray,
    later_time_on_common_grid: np.ndarray,
    t_cut: float,
    dt: float,
    retained_energy_fraction: float = 0.95,
) -> dict[str, Any]:
    """Compare adjacent progressive PINNs at their last common time.

    ``earlier_field`` is the solution trained through ``T_k``. The later
    solution must first be restricted to the same temporal grid ``[0, T_k]``.
    The raw metric is the spatial RMSE between both final fields. For
    ``T_k <= t_cut`` the filtered metric is defined to be exactly the raw
    metric because the spectral filter has not yet acted on the comparison
    time. For ``T_k > t_cut``, both common-grid fields are filtered
    independently on the identical window ``[t_cut, T_k]`` before their final
    fields are compared. No analytical reference is used.
    """
    earlier = np.asarray(earlier_field, dtype=np.float64)
    later = np.asarray(later_field_on_common_grid, dtype=np.float64)
    earlier_time = np.asarray(earlier_time, dtype=np.float64)
    later_time = np.asarray(later_time_on_common_grid, dtype=np.float64)

    if earlier.ndim != 2 or later.ndim != 2:
        raise ValueError(
            "Adjacent-training fields must have shape (nx, nt_common)."
        )
    if earlier.shape != later.shape:
        raise ValueError(
            "Adjacent-training fields must have identical shapes on the "
            f"common grid; received {earlier.shape} and {later.shape}."
        )
    if earlier.shape[1] != earlier_time.size or later.shape[1] != later_time.size:
        raise ValueError(
            "Adjacent-training time arrays must match their field dimensions."
        )
    if earlier_time.shape != later_time.shape or not np.allclose(
        earlier_time,
        later_time,
        rtol=0.0,
        atol=max(1e-12, abs(float(dt)) * 1e-7),
    ):
        raise ValueError(
            "Adjacent-training solutions must use the same common time grid."
        )
    if not np.all(np.isfinite(earlier)) or not np.all(np.isfinite(later)):
        raise ValueError(
            "Adjacent-training discrepancy received non-finite field values."
        )
    if earlier_time.size == 0:
        raise ValueError(
            "Adjacent-training discrepancy requires a non-empty common grid."
        )
    if not np.isfinite(t_cut):
        raise ValueError("t_cut must be finite.")

    common_tmax = float(earlier_time[-1])
    raw_difference = earlier[:, -1] - later[:, -1]
    raw_rmse = float(np.sqrt(np.mean(raw_difference**2)))
    tolerance = max(1e-12, abs(float(dt)) * 1e-7)
    filtering_applied = common_tmax > float(t_cut) + tolerance

    if not filtering_applied:
        filtered_rmse = raw_rmse
        cutoff_index = None
        earlier_cutoff_shell = None
        later_cutoff_shell = None
        post_cutoff_points = 0
    else:
        earlier_filtered, earlier_result, cutoff_index = filter_field_after_cutoff(
            u=earlier,
            t=earlier_time,
            t_cut=float(t_cut),
            dt=float(dt),
            retained_energy_fraction=float(retained_energy_fraction),
        )
        later_filtered, later_result, later_cutoff_index = (
            filter_field_after_cutoff(
                u=later,
                t=later_time,
                t_cut=float(t_cut),
                dt=float(dt),
                retained_energy_fraction=float(retained_energy_fraction),
            )
        )
        if int(cutoff_index) != int(later_cutoff_index):
            raise RuntimeError(
                "Common-grid adjacent filters produced different cutoff indices."
            )
        filtered_difference = (
            earlier_filtered[:, -1] - later_filtered[:, -1]
        )
        filtered_rmse = float(np.sqrt(np.mean(filtered_difference**2)))
        earlier_cutoff_shell = int(earlier_result.cutoff_shell)
        later_cutoff_shell = int(later_result.cutoff_shell)
        post_cutoff_points = int(earlier_time.size - int(cutoff_index))

    if raw_rmse > 0.0:
        filtered_to_raw_ratio = float(filtered_rmse / raw_rmse)
        filtered_reduction = float(1.0 - filtered_to_raw_ratio)
    elif filtered_rmse == 0.0:
        filtered_to_raw_ratio = 1.0
        filtered_reduction = 0.0
    else:
        filtered_to_raw_ratio = float("inf")
        filtered_reduction = float("-inf")

    return {
        "common_tmax": common_tmax,
        "tcut_common": float(t_cut),
        "filtering_applied": bool(filtering_applied),
        "raw_adjacent_final_rmse": raw_rmse,
        "filtered_adjacent_final_rmse": filtered_rmse,
        "filtered_to_raw_ratio": filtered_to_raw_ratio,
        "filtered_reduction": filtered_reduction,
        "cutoff_index": cutoff_index,
        "post_cutoff_points": post_cutoff_points,
        "earlier_cutoff_shell": earlier_cutoff_shell,
        "later_cutoff_shell": later_cutoff_shell,
    }


def summarize_adjacent_training_discrepancy(
    discrepancy: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize the post-onset effect of filtering across adjacent horizons.

    The summary uses only rows for which filtering was active. It reports how
    often the filtered adjacent-horizon discrepancy is lower than the raw
    discrepancy and the median filtered-to-raw ratio. No analytical reference
    is used.
    """
    required = {
        "filtering_applied",
        "raw_adjacent_final_rmse",
        "filtered_adjacent_final_rmse",
        "filtered_to_raw_ratio",
        "filtered_reduction",
    }
    missing = required.difference(discrepancy.columns)
    if missing:
        raise ValueError(
            "Adjacent-training discrepancy is missing columns: "
            f"{sorted(missing)}"
        )

    data = discrepancy.copy()
    for column in [
        "raw_adjacent_final_rmse",
        "filtered_adjacent_final_rmse",
        "filtered_to_raw_ratio",
        "filtered_reduction",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    post_onset = data.loc[data["filtering_applied"].astype(bool)].dropna(
        subset=[
            "raw_adjacent_final_rmse",
            "filtered_adjacent_final_rmse",
            "filtered_to_raw_ratio",
            "filtered_reduction",
        ]
    )
    finite_mask = np.isfinite(
        post_onset[
            [
                "raw_adjacent_final_rmse",
                "filtered_adjacent_final_rmse",
                "filtered_to_raw_ratio",
                "filtered_reduction",
            ]
        ].to_numpy(dtype=np.float64)
    ).all(axis=1)
    post_onset = post_onset.loc[finite_mask]

    pair_count = int(len(post_onset))
    if pair_count == 0:
        return {
            "post_onset_pair_count": 0,
            "reduced_pair_count": 0,
            "reduced_pair_fraction": float("nan"),
            "median_filtered_to_raw_ratio": float("nan"),
            "median_filtered_reduction": float("nan"),
        }

    reduced_mask = (
        post_onset["filtered_adjacent_final_rmse"].to_numpy(dtype=np.float64)
        < post_onset["raw_adjacent_final_rmse"].to_numpy(dtype=np.float64)
    )
    reduced_count = int(np.count_nonzero(reduced_mask))
    return {
        "post_onset_pair_count": pair_count,
        "reduced_pair_count": reduced_count,
        "reduced_pair_fraction": float(reduced_count / pair_count),
        "median_filtered_to_raw_ratio": float(
            post_onset["filtered_to_raw_ratio"].median()
        ),
        "median_filtered_reduction": float(
            post_onset["filtered_reduction"].median()
        ),
    }


def build_cutoff_schedule(
    t_max: float,
    first_cutoff_time: float,
    cutoff_increment: float,
) -> np.ndarray:
    """Build tested cutoff times strictly smaller than ``t_max``."""
    t_max = float(t_max)
    first_cutoff_time = float(first_cutoff_time)
    cutoff_increment = float(cutoff_increment)
    if cutoff_increment <= 0.0:
        raise ValueError("cutoff_increment must be positive.")
    if first_cutoff_time < 0.0:
        raise ValueError("first_cutoff_time must be non-negative.")
    if first_cutoff_time >= t_max:
        return np.array([], dtype=np.float64)
    count = int(np.floor((t_max - first_cutoff_time) / cutoff_increment + 1e-10)) + 1
    values = first_cutoff_time + cutoff_increment * np.arange(count, dtype=np.float64)
    values = values[values < t_max - 1e-12]
    return np.round(values, 12)


def compute_window_sensitivity_rows(
    u: np.ndarray,
    t: np.ndarray,
    t_max: float,
    dt: float,
    cutoff_values: np.ndarray,
    retained_energy_fraction: float,
) -> list[dict[str, Any]]:
    """Compute consecutive-cutoff sensitivity rows for one trained window."""
    rows: list[dict[str, Any]] = []
    previous_final_field: np.ndarray | None = None
    previous_cutoff: float | None = None

    for cutoff_value in np.asarray(cutoff_values, dtype=np.float64):
        filtered, result, cutoff_index = filter_field_after_cutoff(
            u=u,
            t=t,
            t_cut=float(cutoff_value),
            dt=dt,
            retained_energy_fraction=retained_energy_fraction,
        )
        final_field = filtered[:, -1]
        sensitivity = (
            float(np.linalg.norm(final_field - previous_final_field))
            if previous_final_field is not None
            else np.nan
        )
        rows.append(
            {
                "tmax": float(t_max),
                "tcut": float(cutoff_value),
                "previous_tcut": previous_cutoff,
                "inter_filter_sensitivity": sensitivity,
                "cutoff_index": int(cutoff_index),
                "segment_points": int(t.size - cutoff_index),
                "cutoff_frequency": result.cutoff_frequency,
                "cutoff_shell": result.cutoff_shell,
                "target_energy_fraction": float(retained_energy_fraction),
                "achieved_energy_fraction": result.retained_energy_fraction,
                "retained_bin_count": result.retained_bin_count,
                "total_bin_count": result.total_bin_count,
                "maximum_imaginary_residual": result.maximum_imaginary_residual,
            }
        )
        previous_final_field = final_field.copy()
        previous_cutoff = float(cutoff_value)
    return rows


def _bic_from_sse(sse: float, n_points: int, parameter_count: int) -> float:
    """Return the Gaussian least-squares BIC for one fitted mean model."""
    if n_points <= parameter_count:
        raise ValueError(
            "BIC requires more observations than fitted mean-model parameters."
        )
    if not np.isfinite(sse) or sse < 0.0:
        raise ValueError("SSE must be finite and non-negative.")
    if sse == 0.0:
        return float("-inf")
    return float(
        n_points * np.log(sse / float(n_points))
        + parameter_count * np.log(float(n_points))
    )


def fit_reduced_sensitivity_regime(
    sweep: pd.DataFrame,
    tmax: float,
    minimum_points_per_segment: int = 3,
) -> ReducedSensitivityRegimeFit:
    """Fit one-line and reduced-slope models at a fixed ``tmax``.

    The response is ``log10(inter_filter_sensitivity)``. Breakpoints are
    searched only on available cutoff values. For a candidate ``tau``, the
    segmented model is

    ``a + b_before * x + delta * max(0, x - tau)``,

    with ``b_after = b_before + delta``. A candidate is structurally valid
    only when the initial slope is negative and the post-break trend is flatter
    in absolute magnitude. BIC counts ``tau`` as an estimated parameter.
    """
    required = {"tmax", "tcut", "inter_filter_sensitivity"}
    missing = required.difference(sweep.columns)
    if missing:
        raise ValueError(f"Spectral sweep is missing columns: {sorted(missing)}")
    if minimum_points_per_segment < 2:
        raise ValueError("minimum_points_per_segment must be at least 2.")

    tmax_values = pd.to_numeric(sweep["tmax"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    profile = sweep.loc[
        np.isclose(tmax_values, float(tmax), rtol=0.0, atol=1e-12),
        ["tcut", "inter_filter_sensitivity"],
    ].copy()
    profile["tcut"] = pd.to_numeric(profile["tcut"], errors="coerce")
    profile["inter_filter_sensitivity"] = pd.to_numeric(
        profile["inter_filter_sensitivity"], errors="coerce"
    )
    profile = profile.dropna().sort_values("tcut")
    if profile.duplicated("tcut").any():
        raise ValueError(
            f"Duplicate cutoff values exist in the sensitivity profile at tmax={tmax:.3f}."
        )
    if profile.empty:
        raise ValueError(
            f"No finite sensitivity profile is available at tmax={float(tmax):.3f}."
        )
    if (profile["inter_filter_sensitivity"] <= 0.0).any():
        raise ValueError(
            "Reduced-sensitivity analysis requires strictly positive inter-filter sensitivity "
            f"values at tmax={float(tmax):.3f}."
        )

    x = profile["tcut"].to_numpy(dtype=np.float64)
    y = np.log10(
        profile["inter_filter_sensitivity"].to_numpy(dtype=np.float64)
    )
    n_points = int(x.size)
    minimum_required = 2 * int(minimum_points_per_segment)
    if n_points < minimum_required:
        raise ValueError(
            "Reduced-sensitivity analysis requires at least "
            f"{minimum_required} positive sensitivity values at one tmax; "
            f"received {n_points} at tmax={float(tmax):.3f}."
        )

    m0_design = np.column_stack([np.ones_like(x), x])
    m0_coefficients, _, _, _ = np.linalg.lstsq(m0_design, y, rcond=None)
    m0_fitted = m0_design @ m0_coefficients
    m0_residual = y - m0_fitted
    m0_sse = float(np.dot(m0_residual, m0_residual))
    m0_bic = _bic_from_sse(m0_sse, n_points, parameter_count=2)

    candidates: list[dict[str, Any]] = []
    for tau in x:
        left_count = int(np.count_nonzero(x < tau))
        right_count = int(np.count_nonzero(x >= tau))
        if (
            left_count < minimum_points_per_segment
            or right_count < minimum_points_per_segment
        ):
            continue

        hinge = np.maximum(0.0, x - float(tau))
        design = np.column_stack([np.ones_like(x), x, hinge])
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(coefficients[0])
        slope_before = float(coefficients[1])
        slope_after = float(coefficients[1] + coefficients[2])
        slope_scale = max(1.0, abs(slope_before), abs(slope_after))
        numerical_slope_tolerance = (
            100.0 * np.finfo(np.float64).eps * slope_scale
        )
        if not (
            slope_before < 0.0
            and abs(slope_before) - abs(slope_after)
            > numerical_slope_tolerance
        ):
            continue

        fitted = design @ coefficients
        residual = y - fitted
        sse = float(np.dot(residual, residual))
        bic = _bic_from_sse(sse, n_points, parameter_count=4)
        candidates.append(
            {
                "tau": float(tau),
                "intercept": intercept,
                "slope_before": slope_before,
                "slope_after": slope_after,
                "sse": sse,
                "bic": bic,
                "fitted": fitted,
            }
        )

    if candidates:
        best = sorted(candidates, key=lambda item: (item["bic"], item["tau"]))[0]
        m1_bic = float(best["bic"])
        preferred_model = "M1" if m1_bic < m0_bic else "M0"
        return ReducedSensitivityRegimeFit(
            tmax=float(tmax),
            n_points=n_points,
            minimum_points_per_segment=int(minimum_points_per_segment),
            preferred_model=preferred_model,
            tau=float(best["tau"]),
            m0_intercept=float(m0_coefficients[0]),
            m0_slope=float(m0_coefficients[1]),
            m0_sse=m0_sse,
            m0_bic=m0_bic,
            m1_intercept=float(best["intercept"]),
            m1_slope_before=float(best["slope_before"]),
            m1_slope_after=float(best["slope_after"]),
            m1_sse=float(best["sse"]),
            m1_bic=m1_bic,
            delta_bic_m1_minus_m0=float(m1_bic - m0_bic),
            tcut=x.copy(),
            log10_sensitivity=y.copy(),
            m0_fitted=m0_fitted.copy(),
            m1_fitted=np.asarray(best["fitted"], dtype=np.float64).copy(),
        )

    return ReducedSensitivityRegimeFit(
        tmax=float(tmax),
        n_points=n_points,
        minimum_points_per_segment=int(minimum_points_per_segment),
        preferred_model="M0",
        tau=float("nan"),
        m0_intercept=float(m0_coefficients[0]),
        m0_slope=float(m0_coefficients[1]),
        m0_sse=m0_sse,
        m0_bic=m0_bic,
        m1_intercept=float("nan"),
        m1_slope_before=float("nan"),
        m1_slope_after=float("nan"),
        m1_sse=float("nan"),
        m1_bic=float("inf"),
        delta_bic_m1_minus_m0=float("inf"),
        tcut=x.copy(),
        log10_sensitivity=y.copy(),
        m0_fitted=m0_fitted.copy(),
        m1_fitted=np.full_like(y, np.nan),
    )


def _store_reduced_sensitivity_fit_on_sweep(
    annotated: pd.DataFrame,
    fit: ReducedSensitivityRegimeFit,
) -> Any | None:
    """Store one fixed-``tmax`` reduced-sensitivity fit on its sweep rows.

    The returned index is the discrete breakpoint row when ``M1`` is preferred.
    ``None`` is returned when the single-trend model is preferred.
    """
    tmax_values = pd.to_numeric(annotated["tmax"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    profile_mask = np.isclose(
        tmax_values,
        float(fit.tmax),
        rtol=0.0,
        atol=1e-12,
    )
    positive_profile = annotated.loc[profile_mask].copy()
    positive_profile["tcut"] = pd.to_numeric(
        positive_profile["tcut"], errors="coerce"
    )
    positive_profile["inter_filter_sensitivity"] = pd.to_numeric(
        positive_profile["inter_filter_sensitivity"], errors="coerce"
    )
    positive_profile = positive_profile[
        np.isfinite(positive_profile["tcut"])
        & np.isfinite(positive_profile["inter_filter_sensitivity"])
        & (positive_profile["inter_filter_sensitivity"] > 0.0)
    ].sort_values("tcut")
    profile_index = positive_profile.index
    if len(profile_index) != fit.n_points:
        raise RuntimeError(
            "Could not map a fitted reduced-sensitivity profile back to the spectral sweep."
        )

    annotated.loc[
        profile_index, "log10_inter_filter_sensitivity"
    ] = fit.log10_sensitivity
    annotated.loc[
        profile_index, "reduced_sensitivity_m0_fitted_log10_sensitivity"
    ] = fit.m0_fitted
    annotated.loc[
        profile_index, "reduced_sensitivity_m1_fitted_log10_sensitivity"
    ] = fit.m1_fitted
    scalar_values = {
        "reduced_sensitivity_n_points": fit.n_points,
        "reduced_sensitivity_tau": fit.tau,
        "reduced_sensitivity_m0_intercept": fit.m0_intercept,
        "reduced_sensitivity_m0_slope": fit.m0_slope,
        "reduced_sensitivity_m0_sse": fit.m0_sse,
        "reduced_sensitivity_m0_bic": fit.m0_bic,
        "reduced_sensitivity_m1_intercept": fit.m1_intercept,
        "reduced_sensitivity_m1_slope_before": fit.m1_slope_before,
        "reduced_sensitivity_m1_slope_after": fit.m1_slope_after,
        "reduced_sensitivity_m1_sse": fit.m1_sse,
        "reduced_sensitivity_m1_bic": fit.m1_bic,
        "reduced_sensitivity_delta_bic_m1_minus_m0": (
            fit.delta_bic_m1_minus_m0
        ),
    }
    for column, value in scalar_values.items():
        annotated.loc[profile_index, column] = float(value)
    annotated.loc[profile_mask, "reduced_sensitivity_preferred_model"] = fit.preferred_model
    annotated.loc[profile_mask, "reduced_sensitivity_fit_status"] = (
        f"{fit.preferred_model}_preferred"
    )

    if fit.preferred_model != "M1":
        return None

    breakpoint_candidates = annotated.loc[
        profile_mask
        & np.isclose(
            pd.to_numeric(annotated["tcut"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            float(fit.tau),
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(breakpoint_candidates) != 1:
        raise RuntimeError(
            "A reduced-sensitivity breakpoint does not map to exactly one cutoff row."
        )
    breakpoint_index = breakpoint_candidates.index[0]
    annotated.loc[breakpoint_index, "is_reduced_sensitivity_onset"] = True
    return breakpoint_index


def annotate_sweep_selection(
    sweep: pd.DataFrame,
    selection_mode: str,
    final_time: float,
    reduced_sensitivity_minimum_points_per_segment: int = 3,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Annotate diagnostic minima and the configured reference-free selection.

    When ``reduced_sensitivity_at_final_time`` is active, the same BIC comparison is also
    evaluated independently at every structurally identifiable ``tmax``. Those
    additional fits do not change the final-time selection; they provide a
    reference-free robustness path showing whether the inferred cutoff remains
    stable as the trained horizon increases.
    """
    required = {"tmax", "tcut", "inter_filter_sensitivity"}
    missing = required.difference(sweep.columns)
    if missing:
        raise ValueError(f"Spectral sweep is missing columns: {sorted(missing)}")

    annotated = sweep.copy()
    annotated["is_minimum_for_tmax"] = False
    annotated["is_global_minimum"] = False
    annotated["is_reduced_sensitivity_onset"] = False
    annotated["is_selected"] = False
    reduced_sensitivity_numeric_columns = [
        "log10_inter_filter_sensitivity",
        "reduced_sensitivity_m0_fitted_log10_sensitivity",
        "reduced_sensitivity_m1_fitted_log10_sensitivity",
        "reduced_sensitivity_n_points",
        "reduced_sensitivity_tau",
        "reduced_sensitivity_m0_intercept",
        "reduced_sensitivity_m0_slope",
        "reduced_sensitivity_m0_sse",
        "reduced_sensitivity_m0_bic",
        "reduced_sensitivity_m1_intercept",
        "reduced_sensitivity_m1_slope_before",
        "reduced_sensitivity_m1_slope_after",
        "reduced_sensitivity_m1_sse",
        "reduced_sensitivity_m1_bic",
        "reduced_sensitivity_delta_bic_m1_minus_m0",
    ]
    for column in reduced_sensitivity_numeric_columns:
        annotated[column] = np.nan
    annotated["reduced_sensitivity_preferred_model"] = None
    annotated["reduced_sensitivity_fit_status"] = "not_evaluated"

    sensitivity_values = pd.to_numeric(
        annotated["inter_filter_sensitivity"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid = annotated.loc[np.isfinite(sensitivity_values)].copy()
    if valid.empty:
        raise ValueError(
            "No inter-filter sensitivity values were produced. At least two tested cutoffs "
            "are required within one temporal window."
        )

    for _, group in valid.groupby("tmax", sort=True):
        chosen_index = group.sort_values(
            ["inter_filter_sensitivity", "tcut"],
            ascending=[True, True],
        ).index[0]
        annotated.loc[chosen_index, "is_minimum_for_tmax"] = True

    global_index = valid.sort_values(
        ["inter_filter_sensitivity", "tmax", "tcut"],
        ascending=[True, False, True],
    ).index[0]
    annotated.loc[global_index, "is_global_minimum"] = True
    global_row = annotated.loc[global_index].copy()

    normalized_mode = str(selection_mode).strip().lower()
    if normalized_mode == "global_minimum":
        selected_index = global_index
    elif normalized_mode == "minimum_at_final_time":
        at_final = valid[
            np.isclose(
                pd.to_numeric(valid["tmax"], errors="coerce").to_numpy(
                    dtype=np.float64
                ),
                float(final_time),
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if at_final.empty:
            raise ValueError(
                f"No finite sensitivity values exist at final_time={final_time:.3f}."
            )
        selected_index = at_final.sort_values(
            ["inter_filter_sensitivity", "tcut"],
            ascending=[True, True],
        ).index[0]
    elif normalized_mode == "reduced_sensitivity_at_final_time":
        minimum_required = 2 * int(reduced_sensitivity_minimum_points_per_segment)
        tmax_values = np.sort(
            pd.to_numeric(valid["tmax"], errors="coerce")
            .dropna()
            .unique()
            .astype(np.float64)
        )
        final_fit: ReducedSensitivityRegimeFit | None = None
        for tmax_value in tmax_values:
            profile_mask = np.isclose(
                pd.to_numeric(annotated["tmax"], errors="coerce").to_numpy(
                    dtype=np.float64
                ),
                float(tmax_value),
                rtol=0.0,
                atol=1e-12,
            )
            profile_sensitivity = pd.to_numeric(
                annotated.loc[profile_mask, "inter_filter_sensitivity"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            finite_values = profile_sensitivity[np.isfinite(profile_sensitivity)]
            if finite_values.size and np.any(finite_values <= 0.0):
                annotated.loc[
                    profile_mask, "reduced_sensitivity_fit_status"
                ] = "nonpositive_sensitivity"
                if np.isclose(
                    float(tmax_value), float(final_time), rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        "The final-time reduced-sensitivity analysis requires strictly positive "
                        "inter-filter sensitivity values."
                    )
                continue

            positive_count = int(np.count_nonzero(finite_values > 0.0))
            if positive_count < minimum_required:
                annotated.loc[
                    profile_mask, "reduced_sensitivity_fit_status"
                ] = "insufficient_points"
                continue

            fit = fit_reduced_sensitivity_regime(
                annotated,
                tmax=float(tmax_value),
                minimum_points_per_segment=int(
                    reduced_sensitivity_minimum_points_per_segment
                ),
            )
            _store_reduced_sensitivity_fit_on_sweep(annotated, fit)
            if np.isclose(
                float(tmax_value), float(final_time), rtol=0.0, atol=1e-12
            ):
                final_fit = fit

        if final_fit is None:
            raise RuntimeError(
                "The final-time sensitivity profile does not contain enough positive "
                "values for the configured segmented analysis."
            )
        if final_fit.preferred_model != "M1":
            raise RuntimeError(
                "The final-time sensitivity profile does not support a reduced-sensitivity "
                "regime under BIC. M0 was preferred at "
                f"tmax={float(final_time):.3f}: "
                f"BIC(M0)={final_fit.m0_bic:.6f}, "
                f"BIC(M1)={final_fit.m1_bic:.6f}."
            )

        final_mask = np.isclose(
            pd.to_numeric(annotated["tmax"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            float(final_time),
            rtol=0.0,
            atol=1e-12,
        )
        selected_candidates = annotated.loc[
            final_mask & annotated["is_reduced_sensitivity_onset"].astype(bool)
        ]
        if len(selected_candidates) != 1:
            raise RuntimeError(
                "The selected final-time reduced-sensitivity breakpoint does not map to exactly "
                "one cutoff row."
            )
        selected_index = selected_candidates.index[0]
    else:
        raise ValueError(
            "analysis.selection_mode must be 'global_minimum', "
            "'minimum_at_final_time', or 'reduced_sensitivity_at_final_time'."
        )

    annotated.loc[selected_index, "is_selected"] = True
    selected_row = annotated.loc[selected_index].copy()
    return annotated, global_row, selected_row


def compute_window_reference_metrics(
    filtered_segment: np.ndarray,
    raw_segment: np.ndarray,
    reference_segment: np.ndarray,
) -> dict[str, float]:
    """Compute normalized full-window errors against one reference segment.

    All arrays must describe the same space-time grid and have shape
    ``(nx, nt_window)``. The mean-square normalization makes the metrics
    independent of the number of sampled points, although different cutoff
    pairs still represent different physical time intervals.
    """
    filtered = np.asarray(filtered_segment, dtype=np.float64)
    raw = np.asarray(raw_segment, dtype=np.float64)
    reference = np.asarray(reference_segment, dtype=np.float64)
    if filtered.ndim != 2:
        raise ValueError("filtered_segment must have shape (nx, nt_window).")
    if filtered.shape != raw.shape or filtered.shape != reference.shape:
        raise ValueError(
            "Filtered, raw, and reference segments must have identical shapes; "
            f"received {filtered.shape}, {raw.shape}, and {reference.shape}."
        )
    if filtered.size == 0:
        raise ValueError("Window error metrics require at least one sampled point.")
    if not (
        np.all(np.isfinite(filtered))
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(reference))
    ):
        raise ValueError("Window error metrics received non-finite values.")

    filtered_mse = float(np.mean((filtered - reference) ** 2))
    raw_mse = float(np.mean((raw - reference) ** 2))
    filtered_rmse = float(np.sqrt(filtered_mse))
    raw_rmse = float(np.sqrt(raw_mse))
    if raw_rmse > 0.0:
        ratio = float(filtered_rmse / raw_rmse)
        reduction = float(1.0 - ratio)
    elif filtered_rmse == 0.0:
        ratio = 1.0
        reduction = 0.0
    else:
        ratio = float("inf")
        reduction = float("-inf")

    return {
        "filtered_window_mse": filtered_mse,
        "filtered_window_rmse": filtered_rmse,
        "raw_window_mse": raw_mse,
        "raw_window_rmse": raw_rmse,
        "rmse_ratio": ratio,
        "rmse_reduction": reduction,
    }


def compute_temporal_stability_metrics(
    filtered_segment: np.ndarray,
    dt: float,
    window_duration: float,
) -> dict[str, float]:
    """Measure temporal variation within one filtered space-time window.

    The main metrics include every spatial node, including the two boundary
    nodes. Temporal variability is first measured independently at each
    spatial position and then aggregated with an RMS across space, so spatial
    profile variation is not mixed with temporal variation.

    ``temporal_std_per_duration`` divides the temporal standard-deviation RMS
    by the physical window duration. It therefore prefers a longer window when
    two windows contain the same temporal deviation. The derivative metric is
    the RMS forward-difference approximation of ``du/dt`` over the window.
    """
    filtered = np.asarray(filtered_segment, dtype=np.float64)
    if filtered.ndim != 2:
        raise ValueError("filtered_segment must have shape (nx, nt_window).")
    if filtered.shape[0] < 1 or filtered.shape[1] < 2:
        raise ValueError(
            "Temporal-stability metrics require at least one spatial node and "
            "two temporal samples."
        )
    if not np.all(np.isfinite(filtered)):
        raise ValueError("Temporal-stability metrics received non-finite values.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value.")
    if not np.isfinite(window_duration) or window_duration <= 0.0:
        raise ValueError("window_duration must be a positive finite value.")

    temporal_mean = np.mean(filtered, axis=1, keepdims=True)
    temporal_deviation = filtered - temporal_mean
    temporal_std_by_x = np.sqrt(np.mean(temporal_deviation**2, axis=1))

    temporal_derivative = np.diff(filtered, axis=1) / float(dt)
    temporal_derivative_rms_by_x = np.sqrt(
        np.mean(temporal_derivative**2, axis=1)
    )

    def subset_rms(values: np.ndarray, indices: np.ndarray) -> float:
        """Return an RMS over one non-empty spatial-index subset."""
        selected = values[indices]
        return float(np.sqrt(np.mean(selected**2)))

    nx = int(filtered.shape[0])
    all_indices = np.arange(nx, dtype=np.int64)
    boundary_indices = np.unique(np.array([0, nx - 1], dtype=np.int64))
    interior_indices = np.arange(1, nx - 1, dtype=np.int64)

    temporal_std_rms = subset_rms(temporal_std_by_x, all_indices)
    temporal_derivative_rms = subset_rms(
        temporal_derivative_rms_by_x,
        all_indices,
    )
    boundary_temporal_std_rms = subset_rms(
        temporal_std_by_x,
        boundary_indices,
    )
    boundary_temporal_derivative_rms = subset_rms(
        temporal_derivative_rms_by_x,
        boundary_indices,
    )
    if interior_indices.size:
        interior_temporal_std_rms = subset_rms(
            temporal_std_by_x,
            interior_indices,
        )
        interior_temporal_derivative_rms = subset_rms(
            temporal_derivative_rms_by_x,
            interior_indices,
        )
    else:
        interior_temporal_std_rms = float("nan")
        interior_temporal_derivative_rms = float("nan")

    return {
        "temporal_std_rms": temporal_std_rms,
        "temporal_std_per_duration": float(
            temporal_std_rms / float(window_duration)
        ),
        "temporal_derivative_rms": temporal_derivative_rms,
        "boundary_temporal_std_rms": boundary_temporal_std_rms,
        "interior_temporal_std_rms": interior_temporal_std_rms,
        "boundary_temporal_derivative_rms": boundary_temporal_derivative_rms,
        "interior_temporal_derivative_rms": interior_temporal_derivative_rms,
    }


def annotate_metric_minimum_paths(
    sweep: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Annotate per-``tmax`` and global minima for arbitrary sweep metrics.

    Ties are resolved toward the earliest cutoff. Global ties are resolved
    toward the latest ``tmax`` and then the earliest cutoff, matching the
    existing sensitivity and reference-RMSE conventions.
    """
    required = {"tmax", "tcut", *metric_columns}
    missing = required.difference(sweep.columns)
    if missing:
        raise ValueError(f"Metric sweep is missing columns: {sorted(missing)}")

    annotated = sweep.copy()
    for metric in metric_columns:
        path_flag = f"is_minimum_{metric}_for_tmax"
        global_flag = f"is_global_{metric}"
        annotated[path_flag] = False
        annotated[global_flag] = False

        values = pd.to_numeric(annotated[metric], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = annotated.loc[np.isfinite(values)].copy()
        if finite.empty:
            raise ValueError(f"Metric column {metric!r} contains no finite values.")

        for _, group in finite.groupby("tmax", sort=True):
            chosen_index = group.sort_values(
                [metric, "tcut"],
                ascending=[True, True],
            ).index[0]
            annotated.loc[chosen_index, path_flag] = True

        global_index = finite.sort_values(
            [metric, "tmax", "tcut"],
            ascending=[True, False, True],
        ).index[0]
        annotated.loc[global_index, global_flag] = True

    return annotated


def build_metric_path_comparison(
    annotated_sweep: pd.DataFrame,
    criterion_flags: dict[str, str],
) -> pd.DataFrame:
    """Compare reference-free paths with the reference-RMSE optimum.

    The criteria themselves remain reference-free. Reference RMSE is used only
    afterward to report how much precision is lost relative to the best
    available cutoff at each ``tmax``.
    """
    required = {
        "tmax",
        "tcut",
        "filtered_window_rmse",
        "is_minimum_reference_rmse_for_tmax",
        *criterion_flags.values(),
    }
    missing = required.difference(annotated_sweep.columns)
    if missing:
        raise ValueError(
            f"Annotated sweep is missing path-comparison columns: {sorted(missing)}"
        )

    rows: list[dict[str, float]] = []
    for tmax, group in annotated_sweep.groupby("tmax", sort=True):
        optimum = group[group["is_minimum_reference_rmse_for_tmax"].astype(bool)]
        if len(optimum) != 1:
            raise ValueError(
                "Expected exactly one minimum reference-RMSE row at "
                f"tmax={float(tmax):.3f}."
            )
        optimum_row = optimum.iloc[0]
        optimum_rmse = float(optimum_row["filtered_window_rmse"])
        optimum_tcut = float(optimum_row["tcut"])
        row: dict[str, float] = {
            "tmax": float(tmax),
            "reference_rmse_optimal_tcut": optimum_tcut,
            "reference_rmse_optimal": optimum_rmse,
        }

        for criterion, flag_column in criterion_flags.items():
            selected = group[group[flag_column].astype(bool)]
            if selected.empty:
                selected_tcut = float("nan")
                selected_rmse = float("nan")
            elif len(selected) == 1:
                selected_row = selected.iloc[0]
                selected_tcut = float(selected_row["tcut"])
                selected_rmse = float(selected_row["filtered_window_rmse"])
            else:
                raise ValueError(
                    f"Expected at most one {criterion!r} path row at "
                    f"tmax={float(tmax):.3f}."
                )

            row[f"{criterion}_tcut"] = selected_tcut
            row[f"{criterion}_reference_rmse"] = selected_rmse
            row[f"{criterion}_absolute_tcut_distance"] = (
                abs(selected_tcut - optimum_tcut)
                if np.isfinite(selected_tcut)
                else float("nan")
            )
            if np.isfinite(selected_rmse) and optimum_rmse > 0.0:
                row[f"{criterion}_rmse_ratio_to_optimum"] = float(
                    selected_rmse / optimum_rmse
                )
            elif np.isfinite(selected_rmse) and selected_rmse == optimum_rmse == 0.0:
                row[f"{criterion}_rmse_ratio_to_optimum"] = 1.0
            else:
                row[f"{criterion}_rmse_ratio_to_optimum"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def annotate_reference_window_sweep(
    reference_sweep: pd.DataFrame,
    sensitivity_sweep: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate reference-RMSE and reference-free sensitivity paths.

    The reference minimum is computed independently for every ``tmax``. The
    minimum-sensitivity path, reduced-slope onset path, and selected pair are
    copied from stage 05a so reference validation cannot alter any
    reference-free decision.
    """
    reference_required = {"tmax", "tcut", "filtered_window_rmse"}
    sensitivity_required = {
        "tmax",
        "tcut",
        "is_minimum_for_tmax",
        "is_selected",
    }
    missing_reference = reference_required.difference(reference_sweep.columns)
    missing_sensitivity = sensitivity_required.difference(sensitivity_sweep.columns)
    if missing_reference:
        raise ValueError(
            f"Reference window sweep is missing columns: {sorted(missing_reference)}"
        )
    if missing_sensitivity:
        raise ValueError(
            f"Sensitivity sweep is missing columns: {sorted(missing_sensitivity)}"
        )

    annotated = reference_sweep.copy()
    annotated["is_minimum_reference_rmse_for_tmax"] = False
    annotated["is_global_reference_rmse"] = False

    finite = annotated[np.isfinite(annotated["filtered_window_rmse"])].copy()
    if finite.empty:
        raise ValueError("Reference window sweep contains no finite RMSE values.")
    for _, group in finite.groupby("tmax", sort=True):
        chosen_index = group.sort_values(
            ["filtered_window_rmse", "tcut"], ascending=[True, True]
        ).index[0]
        annotated.loc[chosen_index, "is_minimum_reference_rmse_for_tmax"] = True

    global_index = finite.sort_values(
        ["filtered_window_rmse", "tmax", "tcut"],
        ascending=[True, False, True],
    ).index[0]
    annotated.loc[global_index, "is_global_reference_rmse"] = True

    flag_columns = ["tmax", "tcut", "is_minimum_for_tmax", "is_selected"]
    if "is_reduced_sensitivity_onset" in sensitivity_sweep.columns:
        flag_columns.append("is_reduced_sensitivity_onset")
    flags = sensitivity_sweep[flag_columns].copy()
    rename_columns = {
        "is_minimum_for_tmax": "is_minimum_sensitivity_for_tmax",
        "is_selected": "is_selected_by_sensitivity",
    }
    if "is_reduced_sensitivity_onset" in flags.columns:
        rename_columns["is_reduced_sensitivity_onset"] = "is_reduced_sensitivity_onset_for_tmax"
    flags = flags.rename(columns=rename_columns)
    if flags.duplicated(["tmax", "tcut"]).any():
        raise ValueError("Sensitivity sweep contains duplicate (tmax, tcut) pairs.")
    annotated = annotated.merge(
        flags,
        on=["tmax", "tcut"],
        how="left",
        validate="one_to_one",
    )
    boolean_columns = [
        "is_minimum_sensitivity_for_tmax",
        "is_selected_by_sensitivity",
    ]
    if "is_reduced_sensitivity_onset_for_tmax" in annotated.columns:
        boolean_columns.append("is_reduced_sensitivity_onset_for_tmax")
    else:
        annotated["is_reduced_sensitivity_onset_for_tmax"] = False
    for column in boolean_columns:
        annotated[column] = annotated[column].fillna(False).astype(bool)
    return annotated


def select_minimum_reference_rmse_at_tmax(
    reference_sweep: pd.DataFrame,
    tmax: float,
) -> pd.Series:
    """Return the minimum-reference-RMSE row at one exact ``tmax``.

    The row is selected independently from the reference-free sensitivity
    criterion. Ties are resolved toward the earliest cutoff so repeated runs
    remain deterministic.
    """
    required = {"tmax", "tcut", "filtered_window_rmse"}
    missing = required.difference(reference_sweep.columns)
    if missing:
        raise ValueError(
            f"Reference window sweep is missing columns: {sorted(missing)}"
        )

    tmax_values = pd.to_numeric(reference_sweep["tmax"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    rmse_values = pd.to_numeric(
        reference_sweep["filtered_window_rmse"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    mask = np.isclose(tmax_values, float(tmax), rtol=0.0, atol=1e-12) & np.isfinite(
        rmse_values
    )
    candidates = reference_sweep.loc[mask].copy()
    if candidates.empty:
        raise ValueError(
            f"No finite reference-window RMSE values exist at tmax={float(tmax):.3f}."
        )

    return candidates.sort_values(
        ["filtered_window_rmse", "tcut"],
        ascending=[True, True],
    ).iloc[0].copy()


def spectrum_dataframe(result: SpectralFilterResult) -> pd.DataFrame:
    """Convert one averaged two-sided spectrum to a plot-ready table."""
    energy_total = float(np.sum(result.spectral_energy))
    frequency_indices = np.arange(result.total_bin_count, dtype=np.int64)
    shell_indices = np.minimum(frequency_indices, result.total_bin_count - frequency_indices)
    return pd.DataFrame(
        {
            "frequency": result.frequencies,
            "absolute_frequency": np.abs(result.frequencies),
            "frequency_shell": shell_indices,
            "spectral_energy": result.spectral_energy,
            "energy_fraction": result.spectral_energy / energy_total,
            "retained": result.retained_mask.astype(bool),
        }
    ).sort_values("frequency").reset_index(drop=True)


def discover_completed_pinn_windows(
    pinn_config: dict[str, Any],
    root: str | Path,
) -> CompletedPinnWindows:
    """Return the continuous prefix of completed PINN windows.

    The completion marker written by ``04a_run_pinn.py`` is installed only
    after the solution, history, and checkpoint have been written atomically.
    A window currently being trained is therefore ignored safely. Later
    completion markers after a missing window are rejected to preserve the
    deterministic progressive sequence.
    """
    root = Path(root)
    benchmark = BenchmarkConfig.from_mapping(pinn_config["case"])
    grid = pinn_config["grid"]
    training = pinn_config["training"]
    configured_schedule = build_progressive_window_schedule(
        first_window_final_time=float(training["first_window_final_time"]),
        window_increment=float(training["window_increment"]),
        final_time=float(grid["final_time"]),
    )
    paths = build_pinn_output_paths(pinn_config, root)
    artifacts_by_time = {
        float(value): build_window_artifacts(
            paths=paths,
            benchmark=benchmark,
            n_space=int(grid["n_space"]),
            dt=float(grid["dt"]),
            window_final_time=float(value),
        )
        for value in configured_schedule
    }

    completed: list[float] = []
    markers: list[dict[str, Any]] = []
    gap_found = False
    for expected_index, value in enumerate(configured_schedule, start=1):
        time_value = float(value)
        artifacts = artifacts_by_time[time_value]
        if not artifacts.completion.exists():
            gap_found = True
            continue
        if gap_found:
            raise RuntimeError(
                "A later PINN window is marked complete after an incomplete window. "
                f"Unexpected marker: {artifacts.completion}"
            )

        marker = _read_json(artifacts.completion)
        if marker.get("status") != "COMPLETE":
            raise ValueError(f"Invalid completion status in {artifacts.completion}.")
        marker_index = int(marker.get("window_index", -1))
        marker_time = float(marker.get("window_final_time", np.nan))
        if marker_index != expected_index or not np.isclose(
            marker_time,
            time_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "PINN completion marker does not match the configured progressive schedule: "
                f"{artifacts.completion}"
            )

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
        completed.append(time_value)
        markers.append(marker)

    if not completed:
        raise RuntimeError(
            "No completed PINN window is available yet. Wait until 04a finishes "
            "at least one temporal window."
        )

    return CompletedPinnWindows(
        configured_schedule=configured_schedule,
        completed_schedule=np.asarray(completed, dtype=np.float64),
        markers=tuple(markers),
    )


def _window_solution_path(
    pinn_config: dict[str, Any],
    root: Path,
    benchmark: BenchmarkConfig,
    window_final_time: float,
) -> Path:
    """Return the saved progressive PINN solution path for one window."""
    grid = pinn_config["grid"]
    data_dir = _resolve_path(root, pinn_config.get("outputs", {}).get("data_dir", "data/pinn"))
    stem = output_name(
        "PINNs",
        int(grid["n_space"]),
        float(grid["dt"]),
        float(window_final_time),
        benchmark.peclet,
    )
    return data_dir / f"{stem}.parquet"


def run_spectral_data_generation(
    config: dict[str, Any],
    pinn_config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Generate the cutoff sweep and selected filtered PINN field.

    This stage does not read the analytical reference. The interval selection is
    therefore based only on inter-filter sensitivity, preserving the article's
    distinction between selection stability and reference-based accuracy.
    """
    root = Path(root)
    paths = build_spectral_output_paths(config, root)
    benchmark = BenchmarkConfig.from_mapping(pinn_config["case"])
    grid = pinn_config["grid"]
    dt = float(grid["dt"])
    final_time = float(grid["final_time"])
    n_space = int(grid["n_space"])
    analysis = config.get("analysis", {})
    q = float(analysis.get("retained_energy_fraction", 0.95))
    first_cutoff = float(analysis.get("first_cutoff_time", 0.1))
    cutoff_increment = float(analysis.get("cutoff_increment", 0.1))
    selection_mode = str(analysis.get("selection_mode", "global_minimum"))
    reduced_sensitivity_minimum_points_per_segment = int(
        analysis.get("reduced_sensitivity_minimum_points_per_segment", 3)
    )
    if str(analysis.get("spectrum_source", "raw_pinn_field")) != "raw_pinn_field":
        raise ValueError("Only analysis.spectrum_source='raw_pinn_field' is implemented for the article workflow.")

    window_state = discover_completed_pinn_windows(pinn_config, root)
    schedule = window_state.completed_schedule
    processed_final_time = window_state.processed_final_time
    all_rows: list[dict[str, Any]] = []
    first_x: np.ndarray | None = None

    for window_index, t_max in enumerate(schedule, start=1):
        solution_path = _window_solution_path(pinn_config, root, benchmark, float(t_max))
        if not solution_path.exists():
            raise FileNotFoundError(
                f"Required progressive PINN solution not found: {solution_path}\n"
                "Run: python scripts\\04a_run_pinn.py"
            )
        x, t, u = read_solution_matrix(solution_path)
        if x.size != n_space:
            raise ValueError(f"Unexpected spatial size in {solution_path}: {x.size} != {n_space}.")
        if not np.isclose(t[-1], float(t_max), rtol=0.0, atol=max(1e-12, dt * 1e-7)):
            raise ValueError(f"PINN window {solution_path} ends at t={t[-1]}, expected {t_max}.")
        if first_x is None:
            first_x = x
        elif first_x.shape != x.shape or not np.allclose(first_x, x, rtol=0.0, atol=1e-15):
            raise ValueError(f"Spatial grid mismatch in progressive PINN solution {solution_path}.")

        cutoff_values = build_cutoff_schedule(float(t_max), first_cutoff, cutoff_increment)
        rows = compute_window_sensitivity_rows(
            u=u,
            t=t,
            t_max=float(t_max),
            dt=dt,
            cutoff_values=cutoff_values,
            retained_energy_fraction=q,
        )
        for row in rows:
            row["window_index"] = int(window_index)
            row["source_solution"] = _relative_path(solution_path, root)
        all_rows.extend(rows)
        print(
            f"Processed spectral window {window_index}/{schedule.size}: "
            f"tmax={float(t_max):.3f}, tested_cutoffs={len(rows)}"
        )

    sweep = pd.DataFrame(all_rows)
    if sweep.empty or (
        "inter_filter_sensitivity" not in sweep.columns
        or not np.isfinite(sweep["inter_filter_sensitivity"]).any()
    ):
        minimum_endpoint = first_cutoff + 2.0 * cutoff_increment
        raise RuntimeError(
            "The completed PINN prefix is still too short for inter-filter sensitivity. "
            "At least one window with two tested cutoff times is required. "
            f"With the current spectral configuration, wait until 04a completes "
            f"approximately t={minimum_endpoint:.3f}; latest completed t={processed_final_time:.3f}."
        )
    sweep, global_row, selected_row = annotate_sweep_selection(
        sweep=sweep,
        selection_mode=selection_mode,
        final_time=processed_final_time,
        reduced_sensitivity_minimum_points_per_segment=(
            reduced_sensitivity_minimum_points_per_segment
        ),
    )

    selected_tmax = float(selected_row["tmax"])
    selected_tcut = float(selected_row["tcut"])
    selected_solution_path = _window_solution_path(pinn_config, root, benchmark, selected_tmax)
    x_selected, t_selected, u_selected = read_solution_matrix(selected_solution_path)
    u_filtered, selected_result, cutoff_index = filter_field_after_cutoff(
        u=u_selected,
        t=t_selected,
        t_cut=selected_tcut,
        dt=dt,
        retained_energy_fraction=q,
    )

    _atomic_write_parquet(sweep, paths.sweep)
    _atomic_write_parquet(
        solution_dataframe_from_matrix(x_selected, t_selected, u_filtered),
        paths.filtered_solution,
    )
    _atomic_write_parquet(spectrum_dataframe(selected_result), paths.spectrum)

    metadata = {
        "stage": "spectral_data_generation",
        "status": "complete" if window_state.training_complete else "partial",
        "training_complete": window_state.training_complete,
        "method": {
            "spectrum_source": "raw_pinn_field",
            "temporal_transform": "DFT",
            "spectral_energy": "spatial mean of the two-sided squared Fourier magnitude",
            "filter": "hard conjugate-symmetric low-pass mask",
            "target_retained_energy_fraction": q,
            "pre_cutoff_policy": "samples with t < tcut are copied exactly",
            "selection_metric": "inter-filter final-field L2 sensitivity",
            "selection_mode": selection_mode,
            "reduced_sensitivity_response": "log10(inter_filter_sensitivity)",
            "reduced_sensitivity_models": {
                "M0": "a + b*tcut",
                "M1": (
                    "continuous two-segment line with a discrete breakpoint "
                    "and a lower-magnitude post-break slope of either sign"
                ),
            },
            "reduced_sensitivity_minimum_points_per_segment": (
                reduced_sensitivity_minimum_points_per_segment
            ),
            "reference_used_for_selection": False,
        },
        "grid": {
            "n_space": n_space,
            "dt": dt,
            "configured_final_time": final_time,
            "processed_final_time": processed_final_time,
        },
        "sweep": {
            "first_cutoff_time": first_cutoff,
            "cutoff_increment": cutoff_increment,
            "configured_window_schedule": [
                float(value) for value in window_state.configured_schedule
            ],
            "processed_window_schedule": [float(value) for value in schedule],
            "completed_windows": int(schedule.size),
            "expected_windows": int(window_state.configured_schedule.size),
            "tested_pairs": int(len(sweep)),
            "finite_sensitivity_pairs": int(np.isfinite(sweep["inter_filter_sensitivity"]).sum()),
            "path": _relative_path(paths.sweep, root),
        },
        "global_minimum": {
            "tmax": float(global_row["tmax"]),
            "tcut": float(global_row["tcut"]),
            "previous_tcut": float(global_row["previous_tcut"]),
            "inter_filter_sensitivity": float(global_row["inter_filter_sensitivity"]),
        },
        "selected_pair": {
            "tmax": selected_tmax,
            "tcut": selected_tcut,
            "selection_mode": selection_mode,
            "previous_tcut": float(selected_row["previous_tcut"]),
            "inter_filter_sensitivity": float(selected_row["inter_filter_sensitivity"]),
            "source_pinn_solution": _relative_path(selected_solution_path, root),
            "cutoff_index": int(cutoff_index),
            "segment_points": int(t_selected.size - cutoff_index),
            "cutoff_frequency": selected_result.cutoff_frequency,
            "cutoff_shell": selected_result.cutoff_shell,
            "achieved_energy_fraction": selected_result.retained_energy_fraction,
            "retained_bin_count": selected_result.retained_bin_count,
            "total_bin_count": selected_result.total_bin_count,
            "maximum_imaginary_residual": selected_result.maximum_imaginary_residual,
        },
        "outputs": {
            "filtered_solution": _relative_path(paths.filtered_solution, root),
            "selected_spectrum": _relative_path(paths.spectrum, root),
            "generation_metadata": _relative_path(paths.generation_metadata, root),
        },
    }
    if selection_mode.strip().lower() == "reduced_sensitivity_at_final_time":
        metadata["reduced_sensitivity_analysis"] = {
            "tmax": selected_tmax,
            "response": "log10(inter_filter_sensitivity)",
            "statistical_name": "BIC-selected sensitivity change point",
            "operational_name": "onset of the reduced-sensitivity regime",
            "symbol": "tcut_RS",
            "interpretation": (
                "the change point separates the initial rapid decrease of "
                "log10(inter_filter_sensitivity) from a second regime with a "
                "lower-magnitude slope; the post-change slope is not required "
                "to be zero and may have either sign"
            ),
            "preferred_model": str(selected_row["reduced_sensitivity_preferred_model"]),
            "minimum_points_per_segment": reduced_sensitivity_minimum_points_per_segment,
            "tau": float(selected_row["reduced_sensitivity_tau"]),
            "M0": {
                "intercept": float(selected_row["reduced_sensitivity_m0_intercept"]),
                "slope": float(selected_row["reduced_sensitivity_m0_slope"]),
                "sse": float(selected_row["reduced_sensitivity_m0_sse"]),
                "bic": float(selected_row["reduced_sensitivity_m0_bic"]),
            },
            "M1": {
                "intercept": float(selected_row["reduced_sensitivity_m1_intercept"]),
                "slope_before": float(
                    selected_row["reduced_sensitivity_m1_slope_before"]
                ),
                "slope_after": float(
                    selected_row["reduced_sensitivity_m1_slope_after"]
                ),
                "sse": float(selected_row["reduced_sensitivity_m1_sse"]),
                "bic": float(selected_row["reduced_sensitivity_m1_bic"]),
            },
            "delta_bic_m1_minus_m0": float(
                selected_row["reduced_sensitivity_delta_bic_m1_minus_m0"]
            ),
        }

        profile_status = (
            sweep[["tmax", "reduced_sensitivity_fit_status"]]
            .drop_duplicates()
            .sort_values("tmax")
        )
        reduced_sensitivity_path = sweep.loc[
            sweep["is_reduced_sensitivity_onset"].astype(bool)
        ].sort_values("tmax")
        metadata["reduced_sensitivity_horizon_validation"] = {
            "reference_used": False,
            "purpose": (
                "evaluate whether the BIC-selected reduced-sensitivity onset "
                "remains stable when the same analysis is repeated "
                "independently for different trained horizons"
            ),
            "evaluated_tmax_count": int(len(profile_status)),
            "status_counts": {
                str(key): int(value)
                for key, value in profile_status["reduced_sensitivity_fit_status"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
            "reduced_sensitivity_path": [
                {
                    "tmax": float(row.tmax),
                    "tcut": float(row.tcut),
                    "n_points": int(row.reduced_sensitivity_n_points),
                    "slope_before": float(row.reduced_sensitivity_m1_slope_before),
                    "slope_after": float(row.reduced_sensitivity_m1_slope_after),
                    "bic_m0": float(row.reduced_sensitivity_m0_bic),
                    "bic_m1": float(row.reduced_sensitivity_m1_bic),
                    "delta_bic_m1_minus_m0": float(
                        row.reduced_sensitivity_delta_bic_m1_minus_m0
                    ),
                }
                for row in reduced_sensitivity_path.itertuples(index=False)
            ],
        }
    _atomic_write_json(metadata, paths.generation_metadata)

    return metadata


def read_generation_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate spectral-generation metadata."""
    path = Path(path)
    metadata = _read_json(path)
    if metadata.get("stage") != "spectral_data_generation":
        raise ValueError(f"Unexpected spectral metadata stage in {path}.")
    if not isinstance(metadata.get("selected_pair"), dict):
        raise ValueError(f"Missing selected_pair in {path}.")
    return metadata
