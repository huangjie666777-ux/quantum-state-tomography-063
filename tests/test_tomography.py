import copy
import unittest

import numpy as np

from quantum_feedback import (
    PostselectionError,
    TomographyError,
    local_pauli_plan,
    reconstruct_state,
    tomography_experiment,
)


class MeasurementPlanTests(unittest.TestCase):
    def test_plan_size_and_bit_ordering(self):
        plan1 = local_pauli_plan(1)
        plan2 = local_pauli_plan(2)
        plan3 = local_pauli_plan(3)
        self.assertEqual(plan1.settings, ("X", "Y", "Z"))
        self.assertEqual(len(plan2.settings), 9)
        self.assertEqual(len(plan3.settings), 27)
        self.assertEqual(plan2.settings[0], "XX")
        self.assertEqual(plan2.settings[-1], "ZZ")

    def test_rejects_unsupported_size(self):
        with self.assertRaises(TomographyError):
            local_pauli_plan(0)
        with self.assertRaises(TomographyError):
            local_pauli_plan(4)
        with self.assertRaises(TomographyError):
            local_pauli_plan(2.0)


class ReconstructionValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = local_pauli_plan(1)

    def counts(self, z0=10, z1=2):
        return {
            "X": {"0": 12},
            "Y": {"0": 6, "1": 6},
            "Z": {"0": z0, "1": z1},
        }

    def test_requires_each_setting_exactly_once(self):
        counts = self.counts()
        del counts["X"]
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, counts)
        counts["X"] = {"0": 1}
        counts["W"] = {"0": 1}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, counts)

    def test_rejects_bad_outcomes_and_counts(self):
        bad = self.counts()
        bad["X"] = {"00": 12}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, bad)
        bad["X"] = {"2": 12}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, bad)
        bad["X"] = {"0": -1}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, bad)
        bad["X"] = {"0": 1.5}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, bad)
        bad["X"] = {}
        with self.assertRaises(TomographyError):
            reconstruct_state(self.plan, bad)

    def test_missing_outcomes_count_as_zero_and_input_unchanged(self):
        counts = {"X": {"0": 12}, "Y": {"1": 12}, "Z": {"1": 12}}
        snapshot = copy.deepcopy(counts)
        result = reconstruct_state(self.plan, counts)
        self.assertEqual(counts, snapshot)
        self.assertAlmostEqual(result.expectations["X"], 1.0)
        self.assertAlmostEqual(result.expectations["Y"], -1.0)
        self.assertAlmostEqual(result.expectations["Z"], -1.0)


class ReconstructionPhysicsTests(unittest.TestCase):
    def test_pure_states_reconstructed_from_synthetic_counts(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": 0.37},
            {"op": "H", "qubit": 1},
            {"op": "CX", "control": 0, "target": 1},
        ]
        collected = tomography_experiment(2, 1, operations, None, shots=4000, seed=17)
        result = reconstruct_state(local_pauli_plan(2), collected.counts)
        self.assertAlmostEqual(np.trace(result.density_matrix).real, 1.0, places=10)
        eigenvalues = np.linalg.eigvalsh(result.density_matrix)
        self.assertGreaterEqual(eigenvalues.min(), -1e-12)
        self.assertGreater(result.min_eigenvalue, -0.25)
        self.assertLess(result.correction_distance, 0.3)

        from quantum_feedback import run

        exact = run(2, 1, operations).density_matrix
        difference = result.density_matrix - exact
        distance = np.sqrt(np.real(np.trace(difference @ difference.conj().T)))
        self.assertLess(distance, 0.25)

    def test_noiseless_counts_give_physical_state_without_correction(self):
        counts = {}
        for setting in local_pauli_plan(1).settings:
            counts[setting] = {"0": 100}
        result = reconstruct_state(local_pauli_plan(1), counts)
        self.assertAlmostEqual(result.expectations["X"], 1.0)
        self.assertAlmostEqual(result.expectations["Y"], 1.0)
        self.assertAlmostEqual(result.expectations["Z"], 1.0)
        expected = 0.5 * (
            np.eye(2) + np.array([[0.0, 1.0], [1.0, 0.0]]) + np.array([[0.0, -1j], [1j, 0.0]])
            + np.array([[1.0, 0.0], [0.0, -1.0]])
        ).astype(complex)
        np.testing.assert_allclose(result.raw_density_matrix, expected, atol=1e-12)
        # (X+Y+Z)/sqrt(3) is a physical eigenstate, but the unnormalized
        # state with all Bloch components equal to 1 is not.
        self.assertLess(result.min_eigenvalue, 0.0)
        self.assertAlmostEqual(result.expectations["I"], 1.0)
        projected_eigenvalues = np.linalg.eigvalsh(result.density_matrix)
        self.assertGreaterEqual(projected_eigenvalues.min(), -1e-12)
        self.assertAlmostEqual(np.trace(result.density_matrix).real, 1.0)

    def test_impossible_counts_are_reported_and_projected(self):
        counts = {
            "X": {"0": 100},
            "Y": {"0": 100},
            "Z": {"0": 100},
        }
        result = reconstruct_state(local_pauli_plan(1), counts)
        self.assertLess(result.min_eigenvalue, 0.0)
        self.assertGreater(result.correction_distance, 0.0)
        eigenvalues = np.linalg.eigvalsh(result.density_matrix)
        np.testing.assert_allclose(eigenvalues, [0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(np.trace(result.density_matrix).real, 1.0)


class TomographyExperimentTests(unittest.TestCase):
    def test_seeded_reproducibility_and_no_global_state_pollution(self):
        operations = [{"op": "H", "qubit": 0}]
        first = tomography_experiment(1, 1, operations, None, 20, seed=5).counts
        before = np.random.get_state()[1][:5].copy()
        second = tomography_experiment(1, 1, operations, None, 20, seed=5).counts
        after = np.random.get_state()[1][:5]
        self.assertEqual(first, second)
        np.testing.assert_array_equal(before, after)
        for setting in ("X", "Y", "Z"):
            self.assertEqual(sum(first[setting].values()), 20)

    def test_independent_shot_counts(self):
        collected = tomography_experiment(
            1, 1, [{"op": "H", "qubit": 0}], None,
            shots={"X": 5, "Y": 7, "Z": 11}, seed=3,
        )
        self.assertEqual(sum(collected.counts["X"].values()), 5)
        self.assertEqual(sum(collected.counts["Y"].values()), 7)
        self.assertEqual(sum(collected.counts["Z"].values()), 11)

    def test_postselection_success_probability_and_classical_register_preserved(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "X", "qubit": 1, "condition": [0, 1]},
        ]
        selected = tomography_experiment(
            2, 2, operations, None, 30, seed=4, postselection={0: 1}
        )
        self.assertAlmostEqual(selected.success_probability, 0.5)
        self.assertEqual(set(selected.counts["ZZ"]), {"11"})
        with self.assertRaises(PostselectionError):
            tomography_experiment(
                1, 2,
                [{"op": "X", "qubit": 0}, {"op": "MEASURE", "qubit": 0, "clbit": 0}],
                None, 30, seed=4, postselection={0: 0},
            )

    def test_unused_high_postselection_bit_no_longer_raises_index_error(self):
        from quantum_feedback import energy

        result = energy(1, 3, [{"op": "H", "qubit": 0}], [(1.0, "Z")], postselection={2: 0})
        self.assertAlmostEqual(result.success_probability, 1.0)
        self.assertAlmostEqual(result.energy, 0.0)

    def test_bound_parameters_are_used(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
        ]
        plus = tomography_experiment(1, 1, operations, {"theta": 0.0}, 1000, seed=8)
        minus = tomography_experiment(1, 1, operations, {"theta": np.pi}, 1000, seed=8)
        plus_z = plus.counts["Z"].get("0", 0) - plus.counts["Z"].get("1", 0)
        minus_z = minus.counts["Z"].get("0", 0) - minus.counts["Z"].get("1", 0)
        self.assertGreater(plus_z - minus_z, 500)


if __name__ == "__main__":
    unittest.main()
