"""Tests for analytical reference implementation equivalence.

These tests compare the optimized analytical implementation against the slow
literal partial-sum implementation on a small grid. The full production
reference remains too expensive for a routine test, so this file uses a reduced
case that preserves the same mathematical path: modal coefficients, last-K
partial-sum averaging, and tensor-grid reconstruction.
"""

from __future__ import annotations

import unittest

import numpy as np

from pinn_spectral.analytical import analytical_solution, analytical_solution_legacy


class ReferenceImplementationEquivalenceTest(unittest.TestCase):
    """Compare optimized and literal analytical references on small cases."""

    def test_optimized_matches_literal_last_k_average(self) -> None:
        """The optimized last-K weighted sum must match literal averaging."""
        x = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        t = np.linspace(0.0, 0.2, 5, dtype=np.float64)

        common_kwargs = {
            "v": 1.0,
            "D": 0.025,
            "L": 1.0,
            "u0": 0.5,
            "uL": 1.0,
            "n_terms": 64,
            "averaging_width": 16,
        }

        optimized = analytical_solution(
            x,
            t,
            **common_kwargs,
            n_modes_per_block=7,
            n_times_per_block=2,
        )
        literal = analytical_solution_legacy(x, t, **common_kwargs)

        self.assertEqual(optimized.shape, (x.size, t.size))
        self.assertEqual(literal.shape, (x.size, t.size))
        np.testing.assert_allclose(optimized, literal, rtol=1.0e-10, atol=1.0e-10)

    def test_optimized_matches_literal_without_averaging(self) -> None:
        """The optimized implementation must also match the K=1 truncation."""
        x = np.linspace(0.0, 1.0, 7, dtype=np.float64)
        t = np.array([0.0, 0.01, 0.05, 0.1], dtype=np.float64)

        common_kwargs = {
            "v": 1.0,
            "D": 0.025,
            "L": 1.0,
            "u0": 0.5,
            "uL": 1.0,
            "n_terms": 40,
            "averaging_width": 1,
        }

        optimized = analytical_solution(
            x,
            t,
            **common_kwargs,
            n_modes_per_block=5,
            n_times_per_block=2,
        )
        literal = analytical_solution_legacy(x, t, **common_kwargs)

        np.testing.assert_allclose(optimized, literal, rtol=1.0e-10, atol=1.0e-10)

    def test_literal_progress_callback_reports_each_time_level(self) -> None:
        """The literal progress hook must be called once per time level."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        t = np.array([0.0, 0.05, 0.1], dtype=np.float64)
        calls: list[tuple[int, int, float]] = []

        analytical_solution_legacy(
            x,
            t,
            v=1.0,
            D=0.025,
            L=1.0,
            u0=0.5,
            uL=1.0,
            n_terms=12,
            averaging_width=4,
            progress_callback=lambda completed, total, time_value: calls.append(
                (completed, total, time_value)
            ),
        )

        self.assertEqual(len(calls), t.size)
        self.assertEqual(calls[0], (1, t.size, float(t[0])))
        self.assertEqual(calls[-1], (t.size, t.size, float(t[-1])))


if __name__ == "__main__":
    unittest.main()
