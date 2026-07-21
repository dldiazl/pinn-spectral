"""Integration test for full optimized and literal reference files.

This test intentionally does not generate data. It must be run only after both
production reference files have been created with ``scripts/01a_generate_reference_data.py``.
The purpose is to verify that the full-grid optimized reference and the full-grid
literal reference exist, use the same tensor grid, and agree numerically.

The comparison reports several error measures:

* global_l2_error: unnormalized discrete L2 norm over the full space-time grid,
  equivalent to the Frobenius norm of the difference matrix.
* relative_global_l2_error: global_l2_error divided by the Frobenius norm of the
  literal reference matrix.
* max_absolute_error: maximum pointwise absolute difference over the full grid.
* max_time_l2_error: maximum spatial L2 norm over all time levels.
* max_space_l2_error: maximum temporal L2 norm over all spatial nodes.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_spectral.benchmark import BenchmarkConfig  # noqa: E402
from pinn_spectral.tools import (  # noqa: E402
    assert_same_grid,
    compute_error_space,
    compute_error_time,
    output_name,
    read_solution_matrix,
    read_yaml,
    save_json,
)


MAX_ABSOLUTE_ERROR_TOL = 1.0e-7
RELATIVE_GLOBAL_L2_ERROR_TOL = 1.0e-8


class FullReferenceFileEquivalenceTest(unittest.TestCase):
    """Compare the complete optimized and literal reference parquet files."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the reference configuration and resolve expected file paths."""
        cls.config_path = ROOT / "configs" / "reference.yaml"
        cls.config = read_yaml(cls.config_path)
        cls.optimized_data_path, cls.optimized_metadata_path = cls._build_paths("Analytical")
        cls.literal_data_path, cls.literal_metadata_path = cls._build_paths("Analytical_literal")
        cls.comparison_report_path = (
            ROOT / "results" / "metrics" / "reference_optimized_vs_literal_comparison.json"
        )

    @classmethod
    def _build_paths(cls, prefix: str) -> tuple[Path, Path]:
        """Return the expected parquet and metadata paths for one reference."""
        benchmark = BenchmarkConfig.from_mapping(cls.config["case"])
        grid = cls.config["grid"]
        outputs = cls.config["outputs"]
        stem = output_name(
            prefix=prefix,
            n_space=int(grid["n_space"]),
            dt=float(grid["dt"]),
            final_time=float(grid["final_time"]),
            pe=benchmark.peclet,
        )
        data_path = ROOT / outputs.get("data_dir", "data/reference") / f"{stem}.parquet"
        metadata_path = ROOT / outputs.get("metrics_dir", "results/metrics") / f"{stem}_metadata.json"
        return data_path, metadata_path

    @staticmethod
    def _relative(path: Path) -> str:
        """Format a path relative to the repository root when possible."""
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def _assert_required_files_exist(self) -> None:
        """Fail with generation commands if any full reference file is missing."""
        required_paths = [
            self.optimized_data_path,
            self.optimized_metadata_path,
            self.literal_data_path,
            self.literal_metadata_path,
        ]
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            missing_text = "\n".join(f"  - {self._relative(path)}" for path in missing)
            self.fail(
                "Missing required full reference files:\n"
                f"{missing_text}\n\n"
                "Generate them from the repository root before running this integration test:\n"
                "  set PYTHONPATH=src\n"
                "  python scripts\\01a_generate_reference_data.py --overwrite\n"
                "  python scripts\\01a_generate_reference_data.py --implementation literal --overwrite\n"
            )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read one metadata JSON file."""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_error_report(
        self,
        x: np.ndarray,
        t: np.ndarray,
        u_optimized: np.ndarray,
        u_literal: np.ndarray,
    ) -> dict[str, Any]:
        """Compute comparison metrics for the optimized and literal fields."""
        difference = np.asarray(u_optimized, dtype=np.float64) - np.asarray(u_literal, dtype=np.float64)
        absolute_difference = np.abs(difference)
        time_l2_error = compute_error_time(u_optimized, u_literal)
        space_l2_error = compute_error_space(u_optimized, u_literal)

        global_l2_error = float(np.linalg.norm(difference))
        literal_global_l2_norm = float(np.linalg.norm(u_literal))
        relative_global_l2_error = (
            global_l2_error / literal_global_l2_norm if literal_global_l2_norm > 0.0 else float("nan")
        )

        max_abs_flat_index = int(np.argmax(absolute_difference))
        max_abs_indices = np.unravel_index(max_abs_flat_index, difference.shape)
        max_abs_x_index = int(max_abs_indices[0])
        max_abs_t_index = int(max_abs_indices[1])

        max_time_index = int(np.argmax(time_l2_error))
        max_space_index = int(np.argmax(space_l2_error))

        return {
            "comparison": "optimized analytical reference vs literal analytical reference",
            "norm_convention": {
                "matrix_shape": "u.shape = (nx, nt)",
                "global_l2_error": "||u_optimized - u_literal||_2 after flattening the full (nx, nt) matrix; equivalent to the Frobenius norm.",
                "relative_global_l2_error": "global_l2_error / ||u_literal||_2 using the same flattened full-grid convention.",
                "max_absolute_error": "max_{x,t} |u_optimized(x,t) - u_literal(x,t)|.",
                "time_l2_error": "E(t_n) = ||u_optimized(:, n) - u_literal(:, n)||_2.",
                "space_l2_error": "E(x_j) = ||u_optimized(j, :) - u_literal(j, :)||_2.",
                "normalization": "All L2 norms are unnormalized discrete Euclidean norms unless explicitly marked as relative.",
            },
            "files": {
                "optimized_data": self._relative(self.optimized_data_path),
                "literal_data": self._relative(self.literal_data_path),
            },
            "grid": {
                "nx": int(u_optimized.shape[0]),
                "nt": int(u_optimized.shape[1]),
                "x_min": float(x[0]),
                "x_max": float(x[-1]),
                "t_min": float(t[0]),
                "t_max": float(t[-1]),
            },
            "errors": {
                "global_l2_error": global_l2_error,
                "literal_global_l2_norm": literal_global_l2_norm,
                "relative_global_l2_error": float(relative_global_l2_error),
                "max_absolute_error": float(absolute_difference[max_abs_x_index, max_abs_t_index]),
                "max_absolute_error_x": float(x[max_abs_x_index]),
                "max_absolute_error_t": float(t[max_abs_t_index]),
                "max_absolute_error_x_index": max_abs_x_index,
                "max_absolute_error_t_index": max_abs_t_index,
                "max_time_l2_error": float(time_l2_error[max_time_index]),
                "max_time_l2_error_t": float(t[max_time_index]),
                "max_time_l2_error_t_index": max_time_index,
                "final_time_l2_error": float(time_l2_error[-1]),
                "max_space_l2_error": float(space_l2_error[max_space_index]),
                "max_space_l2_error_x": float(x[max_space_index]),
                "max_space_l2_error_x_index": max_space_index,
            },
            "tolerances": {
                "max_absolute_error": MAX_ABSOLUTE_ERROR_TOL,
                "relative_global_l2_error": RELATIVE_GLOBAL_L2_ERROR_TOL,
            },
        }

    def test_full_reference_files_exist(self) -> None:
        """The optimized and literal parquet/metadata files must exist."""
        self._assert_required_files_exist()

    def test_full_reference_metadata_is_consistent(self) -> None:
        """The optimized and literal metadata must describe the same case."""
        self._assert_required_files_exist()
        optimized = self._read_json(self.optimized_metadata_path)
        literal = self._read_json(self.literal_metadata_path)

        self.assertEqual(optimized["implementation"], "optimized")
        self.assertEqual(literal["implementation"], "literal")

        exact_keys = ["n_space", "n_time", "n_terms", "averaging_width"]
        for key in exact_keys:
            self.assertEqual(optimized[key], literal[key], msg=f"Metadata mismatch for {key}.")

        float_keys = [
            "length",
            "u_left",
            "u_right",
            "velocity",
            "diffusivity",
            "peclet",
            "dt",
            "final_time",
        ]
        for key in float_keys:
            self.assertAlmostEqual(
                float(optimized[key]),
                float(literal[key]),
                places=15,
                msg=f"Metadata mismatch for {key}.",
            )

    def test_full_reference_grids_and_values_match(self) -> None:
        """The optimized and literal solution files must agree on the full grid."""
        self._assert_required_files_exist()

        try:
            x_optimized, t_optimized, u_optimized = read_solution_matrix(self.optimized_data_path)
            x_literal, t_literal, u_literal = read_solution_matrix(self.literal_data_path)
        except ImportError as exc:
            self.fail(
                "Reading parquet files requires a parquet engine such as pyarrow or fastparquet. "
                f"Original error: {exc}"
            )

        grid = self.config["grid"]
        expected_nx = int(grid["n_space"])
        expected_nt = int(round(float(grid["final_time"]) / float(grid["dt"]))) + 1

        self.assertEqual(u_optimized.shape, (expected_nx, expected_nt))
        self.assertEqual(u_literal.shape, (expected_nx, expected_nt))
        assert_same_grid(
            x_optimized,
            t_optimized,
            x_literal,
            t_literal,
            label="optimized vs literal analytical reference",
        )

        report = self._build_error_report(
            x=x_optimized,
            t=t_optimized,
            u_optimized=u_optimized,
            u_literal=u_literal,
        )
        save_json(report, self.comparison_report_path)

        errors = report["errors"]
        print("\nFull reference optimized-vs-literal comparison:")
        print(f"  global_l2_error          = {errors['global_l2_error']:.16e}")
        print(f"  relative_global_l2_error = {errors['relative_global_l2_error']:.16e}")
        print(f"  max_absolute_error       = {errors['max_absolute_error']:.16e}")
        print(f"  max_time_l2_error        = {errors['max_time_l2_error']:.16e}")
        print(f"  final_time_l2_error      = {errors['final_time_l2_error']:.16e}")
        print(f"  max_space_l2_error       = {errors['max_space_l2_error']:.16e}")
        print(f"  report                   = {self._relative(self.comparison_report_path)}")

        self.assertLessEqual(
            errors["max_absolute_error"],
            MAX_ABSOLUTE_ERROR_TOL,
            msg=(
                "Full-grid optimized/literal mismatch in maximum absolute norm: "
                f"max_absolute_error={errors['max_absolute_error']:.6e}, "
                f"tolerance={MAX_ABSOLUTE_ERROR_TOL:.6e}. "
                f"See {self._relative(self.comparison_report_path)}."
            ),
        )
        self.assertLessEqual(
            errors["relative_global_l2_error"],
            RELATIVE_GLOBAL_L2_ERROR_TOL,
            msg=(
                "Full-grid optimized/literal mismatch in relative global L2 norm: "
                f"relative_global_l2_error={errors['relative_global_l2_error']:.6e}, "
                f"tolerance={RELATIVE_GLOBAL_L2_ERROR_TOL:.6e}. "
                f"See {self._relative(self.comparison_report_path)}."
            ),
        )


if __name__ == "__main__":
    unittest.main()
