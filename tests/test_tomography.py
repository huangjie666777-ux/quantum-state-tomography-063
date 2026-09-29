import copy
import unittest

import numpy as np

from quantum_feedback import (
    PostselectionError,
    TomographyError,
    pauli_measurement_plan,
    reconstruct_density_matrix,
    simulate_tomography_counts,
    bind_parameters,
    validate_template,
)
from quantum_feedback.energy import evaluate_bound, validate_hamiltonian


class MeasurementPlanTests(unittest.TestCase):
    def test_complete_plan_sizes_and_bit_order(self):
        self.assertEqual(pauli_measurement_plan(1), ("X", "Y", "Z"))
        self.assertEqual(len(pauli_measurement_plan(2)), 9)
        three_qubit = pauli_measurement_plan(3)
        self.assertEqual(len(three_qubit), 27)
        self.assertEqual(three_qubit[0], "XXX")
        self.assertIn("ZYX", three_qubit)
        with self.assertRaises(TomographyError):
            pauli_measurement_plan(0)


class TomographyExperimentTests(unittest.TestCase):
    def test_postselected_counts_are_reproducible_and_isolated(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "H", "qubit": 1, "condition": [0, 0]},
        ]
        snapshot = copy.deepcopy(operations)
        first = simulate_tomography_counts(
            2, 1, operations, postselection={0: 0}, shots=64, seed=17
        )
        second = simulate_tomography_counts(
            2, 1, operations, postselection={0: 0}, shots=64, seed=17
        )
        self.assertEqual(first.counts, second.counts)
        self.assertAlmostEqual(first.success_probability, 0.5)
        self.assertEqual(operations, snapshot)
        for setting, table in first.counts.items():
            self.assertEqual(sum(table.values()), 64)
            self.assertTrue(all(len(outcome) == 2 for outcome in table))

        before = np.random.get_state()[1][:5].copy()
        simulate_tomography_counts(1, 1, [{"op": "H", "qubit": 0}], shots=8, seed=3)
        after = np.random.get_state()[1][:5]
        np.testing.assert_array_equal(before, after)

    def test_per_setting_shots_and_parameter_binding(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
        ]
        plan = pauli_measurement_plan(1)
        shots = {setting: 10 + index for index, setting in enumerate(plan)}
        experiment = simulate_tomography_counts(
            1, 1, operations, {"theta": 0.2}, shots=shots, seed=5
        )
        self.assertEqual(experiment.shots, shots)
        for setting in plan:
            self.assertEqual(sum(experiment.counts[setting].values()), shots[setting])

    def test_vanishing_postselection_fails(self):
        operations = [
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        with self.assertRaises(PostselectionError):
            simulate_tomography_counts(1, 1, operations, postselection={0: 0}, seed=1)

    def test_invalid_shots(self):
        with self.assertRaises(TomographyError):
            simulate_tomography_counts(1, 1, [], shots=0, seed=1)
        with self.assertRaises(TomographyError):
            simulate_tomography_counts(1, 1, [], shots=1.5, seed=1)
        with self.assertRaises(TomographyError):
            simulate_tomography_counts(1, 1, [], shots={"X": 2}, seed=1)


class ReconstructionTests(unittest.TestCase):
    def test_bell_tomography_recovers_state_from_simulated_counts(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
        ]
        experiment = simulate_tomography_counts(2, 1, operations, shots=4000, seed=42)
        result = reconstruct_density_matrix(experiment.plan, experiment.counts)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = expected[3, 3] = 0.5
        expected[0, 3] = expected[3, 0] = 0.5
        np.testing.assert_allclose(result.physical_density_matrix, expected, atol=0.06)
        self.assertAlmostEqual(np.trace(result.density_matrix), 1.0)
        eigenvalues = np.linalg.eigvalsh(result.physical_density_matrix)
        self.assertGreaterEqual(eigenvalues.min(), -1e-10)
        self.assertAlmostEqual(eigenvalues.sum(), 1.0)

    def test_three_qubit_ghz_reconstruction_and_bit_order(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
            {"op": "CX", "control": 0, "target": 2},
        ]
        experiment = simulate_tomography_counts(3, 1, operations, shots=3000, seed=77)
        self.assertEqual(len(experiment.plan), 27)
        self.assertTrue(all(len(outcome) == 3 for table in experiment.counts.values() for outcome in table))
        result = reconstruct_density_matrix(experiment.plan, experiment.counts)
        expected = np.zeros((8, 8), dtype=complex)
        expected[0, 0] = expected[7, 7] = 0.5
        expected[0, 7] = expected[7, 0] = 0.5
        np.testing.assert_allclose(result.physical_density_matrix, expected, atol=0.08)
        self.assertAlmostEqual(result.expectations["XXX"], 1.0, delta=0.05)
        self.assertAlmostEqual(result.expectations["ZZI"], 1.0, delta=0.05)

    def test_y_basis_identifies_imaginary_coherence(self):
        operations = [{"op": "H", "qubit": 0}]
        experiment = simulate_tomography_counts(1, 1, operations, shots=4000, seed=11)
        result = reconstruct_density_matrix(experiment.plan, experiment.counts)
        self.assertAlmostEqual(result.expectations["X"], 1.0, delta=0.05)
        self.assertAlmostEqual(result.expectations["Y"], 0.0, delta=0.05)
        self.assertAlmostEqual(result.expectations["Z"], 0.0, delta=0.05)

    def test_weighted_marginals_use_all_compatible_settings(self):
        counts = {
            "XX": {"00": 80},
            "XY": {"00": 20},
            "XZ": {"00": 100},
            "YX": {"00": 40},
            "YY": {"00": 40},
            "YZ": {"00": 40},
            "ZX": {"00": 10},
            "ZY": {"00": 10},
            "ZZ": {"00": 20},
        }
        plan = tuple(counts)
        snapshot = copy.deepcopy(counts)
        result = reconstruct_density_matrix(plan, counts)
        # Weighted <I X>: (80 + 20 + 100 + 40 + 40 + 40) / 320 = 1.0
        self.assertAlmostEqual(result.expectations["IX"], 1.0)
        # Weighted <Z I>: (10 + 10 + 20) / 40 = 1.0
        self.assertAlmostEqual(result.expectations["ZI"], 1.0)
        self.assertEqual(counts, snapshot)

    def test_invalid_plan_and_counts(self):
        good_counts = {setting: {"0": 5} for setting in "XYZ"}
        with self.assertRaises(TomographyError):
            reconstruct_density_matrix(["X", "Y"], good_counts)
        with self.assertRaises(TomographyError):
            reconstruct_density_matrix([42], good_counts)
        with self.assertRaises(TomographyError):
            reconstruct_density_matrix(["X", "Y", "Y"], good_counts)
        bad_tables = [
            {"X": {"00": 5}, "Y": {"0": 5}, "Z": {"0": 5}},
            {"X": {"2": 5}, "Y": {"0": 5}, "Z": {"0": 5}},
            {"X": {"0": -1}, "Y": {"0": 5}, "Z": {"0": 5}},
            {"X": {"0": 2.5}, "Y": {"0": 5}, "Z": {"0": 5}},
            {"X": {"0": True}, "Y": {"0": 5}, "Z": {"0": 5}},
            {"X": {}, "Y": {"0": 5}, "Z": {"0": 5}},
        ]
        for table in bad_tables:
            with self.assertRaises(TomographyError):
                reconstruct_density_matrix(["X", "Y", "Z"], table)

    def test_missing_outcomes_count_as_zero(self):
        result = reconstruct_density_matrix(
            ["X", "Y", "Z"],
            {"X": {"0": 4, "1": 6}, "Y": {"0": 10}, "Z": {"1": 3}},
        )
        self.assertAlmostEqual(result.expectations["X"], -0.2)
        self.assertAlmostEqual(result.expectations["Y"], 1.0)
        self.assertAlmostEqual(result.expectations["Z"], -1.0)

    def test_physical_projection_repairs_inconsistent_counts(self):
        result = reconstruct_density_matrix(
            ["X", "Y", "Z"],
            {"X": {"0": 10}, "Y": {"0": 10}, "Z": {"0": 10}},
        )
        self.assertLess(result.min_eigenvalue, 0.0)
        self.assertGreater(result.correction_distance, 0.0)
        eigenvalues = np.linalg.eigvalsh(result.physical_density_matrix)
        self.assertGreaterEqual(eigenvalues.min(), -1e-12)
        self.assertAlmostEqual(eigenvalues.sum(), 1.0)

    def test_high_unused_postselection_bit_does_not_overflow(self):
        template = validate_template(1, 1, [{"op": "H", "qubit": 0}])
        bound = bind_parameters(template, {})
        terms = validate_hamiltonian([(1.0, "Z")], 1)
        result = evaluate_bound(1, bound, terms, {1: 0})
        self.assertAlmostEqual(result.success_probability, 1.0)
        with self.assertRaises(PostselectionError):
            evaluate_bound(1, bound, terms, {1: 1})


if __name__ == "__main__":
    unittest.main()
