import copy
import math
import unittest

import numpy as np

from quantum_feedback import (
    CircuitValidationError,
    HamiltonianValidationError,
    PostselectionError,
    TrainingError,
    bind_parameters,
    energy,
    gradient,
    minimize_energy,
    run,
    template_parameters,
    validate_hamiltonian,
    validate_parameters,
    validate_template,
)


def central_difference(fn, point, parameter, epsilon=1e-6):
    plus = dict(point)
    minus = dict(point)
    plus[parameter] += epsilon
    minus[parameter] -= epsilon
    return (fn(plus) - fn(minus)) / (2 * epsilon)


class ParameterBindingTests(unittest.TestCase):
    def test_angle_expression_scale_offset_and_shared_parameter(self):
        template = validate_template(1, 1, [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": 2.0, "offset": 0.1}},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": -0.5, "offset": 0.0}},
        ])
        self.assertEqual(template_parameters(template), ("theta",))
        bound = bind_parameters(template, {"theta": 0.3})
        self.assertAlmostEqual(bound[1].angle, 0.7)
        self.assertAlmostEqual(bound[2].angle, -0.15)

    def test_string_angle_is_named_parameter_with_unit_scale(self):
        bound = bind_parameters(
            validate_template(1, 1, [{"op": "RZ", "qubit": 0, "angle": "phi"}]),
            {"phi": 0.25},
        )
        self.assertAlmostEqual(bound[0].angle, 0.25)

    def test_binding_validates_completely_and_does_not_mutate_inputs(self):
        operations = [
            {"op": "RZ", "qubit": 0, "angle": {"param": "a", "scale": -1.0}},
            {"op": "RZ", "qubit": 0, "angle": {"param": "b"}},
        ]
        snapshot = copy.deepcopy(operations)
        values = {"a": 0.4}
        with self.assertRaises(CircuitValidationError):
            bind_parameters(validate_template(1, 1, operations), values)
        self.assertEqual(operations, snapshot)
        self.assertEqual(values, {"a": 0.4})

        with self.assertRaises(CircuitValidationError):
            validate_parameters(validate_template(1, 1, operations), {"a": 1.0, "b": 2.0, "c": 3.0})
        with self.assertRaises(CircuitValidationError):
            validate_parameters(validate_template(1, 1, operations), {"a": float("nan"), "b": 1.0})
        with self.assertRaises(CircuitValidationError):
            validate_template(1, 1, [{"op": "RZ", "qubit": 0, "angle": {"param": "a", "scale": 0.0}}])

    def test_legacy_constant_circuit_rejects_parameterized_angles(self):
        with self.assertRaises(CircuitValidationError):
            run(1, 1, [{"op": "RZ", "qubit": 0, "angle": {"param": "a"}}])


class HamiltonianTests(unittest.TestCase):
    def test_hamiltonian_validation_length_and_characters(self):
        with self.assertRaises(HamiltonianValidationError):
            validate_hamiltonian([], 2)
        with self.assertRaises(HamiltonianValidationError):
            energy(2, 1, [], [(1.0, "Z")])
        with self.assertRaises(HamiltonianValidationError):
            energy(2, 1, [], [(1.0, "ZA")])
        with self.assertRaises(HamiltonianValidationError):
            energy(2, 1, [], [(float("inf"), "ZZ")])

    def test_pauli_bit_ordering_matches_qubit_zero(self):
        result = energy(
            2, 1,
            [{"op": "H", "qubit": 0}],
            [(1.0, "IZ"), (-0.5, "IX")],
        )
        self.assertAlmostEqual(result.energy, -0.5)
        self.assertAlmostEqual(result.success_probability, 1.0)

    def test_postselection_success_probability_and_conditional_energy(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "X", "qubit": 1, "condition": [0, 1]},
        ]
        hamiltonian = [(1.0, "ZI"), (1.0, "IZ")]
        total = energy(2, 1, operations, hamiltonian)
        self.assertAlmostEqual(total.energy, 0.0)
        selected_zero = energy(2, 1, operations, hamiltonian, postselection={0: 0})
        self.assertAlmostEqual(selected_zero.success_probability, 0.5)
        self.assertAlmostEqual(selected_zero.conditional_energy, 2.0)
        selected_one = energy(2, 1, operations, hamiltonian, postselection={0: 1})
        self.assertAlmostEqual(selected_one.conditional_energy, -2.0)

    def test_vanishing_postselection_fails_explicitly(self):
        operations = [
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        with self.assertRaises(PostselectionError):
            energy(1, 1, operations, [(1.0, "Z")], postselection={0: 0})
        with self.assertRaises(PostselectionError):
            gradient(1, 1, operations, [(1.0, "Z")], {}, postselection={0: 0})


class GradientTests(unittest.TestCase):
    TEMPLATE = [
        {"op": "H", "qubit": 0},
        {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": 2.0, "offset": 0.1}},
        {"op": "H", "qubit": 0},
        {"op": "MEASURE", "qubit": 0, "clbit": 0},
        {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": -1.0},
         "condition": [0, 0]},
    ]
    HAMILTONIAN = [(1.0, "Z"), (0.3, "X")]

    def test_exact_gradient_matches_reference_with_shared_and_conditional_gates(self):
        point = {"theta": 0.7}
        exact = gradient(1, 1, self.TEMPLATE, self.HAMILTONIAN, point)
        reference = central_difference(
            lambda values: energy(1, 1, self.TEMPLATE, self.HAMILTONIAN, values).energy,
            point, "theta",
        )
        self.assertAlmostEqual(exact["theta"], reference, places=8)

    def test_conditional_energy_gradient_includes_probability_derivative(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "H", "qubit": 1},
            {"op": "RZ", "qubit": 1, "angle": {"param": "phi"}, "condition": [0, 0]},
            {"op": "H", "qubit": 1},
        ]
        hamiltonian = [(1.0, "ZI"), (0.5, "XI")]
        point = {"theta": 0.5, "phi": 0.7}
        exact = gradient(2, 1, operations, hamiltonian, point, postselection={0: 0})
        probability_reference = central_difference(
            lambda values: energy(2, 1, operations, hamiltonian, values, {0: 0}).success_probability,
            point, "theta",
        )
        energy_reference = central_difference(
            lambda values: energy(2, 1, operations, hamiltonian, values, {0: 0}).energy,
            point, "theta",
        )
        self.assertGreater(abs(probability_reference), 1e-3)
        self.assertAlmostEqual(energy_reference, 0.0, places=8)
        self.assertAlmostEqual(exact["theta"], energy_reference, places=8)
        phi_reference = central_difference(
            lambda values: energy(2, 1, operations, hamiltonian, values, {0: 0}).energy,
            point, "phi",
        )
        self.assertAlmostEqual(exact["phi"], phi_reference, places=8)
        self.assertGreater(abs(exact["phi"]), 1e-3)

    def test_zero_shift_branch_probability_does_not_invalidate_gradient(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        point = {"theta": 0.5}
        values = gradient(1, 1, operations, [(1.0, "X"), (-1.0, "Z")], point, {0: 0})
        reference = central_difference(
            lambda p: energy(1, 1, operations, [(1.0, "X"), (-1.0, "Z")], p, {0: 0}).energy,
            point, "theta",
        )
        self.assertAlmostEqual(values["theta"], reference, places=7)

    def test_negative_scale_gradient_sign(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "phi", "scale": -2.0}},
            {"op": "H", "qubit": 0},
        ]
        point = {"phi": 0.3}
        exact = gradient(1, 1, operations, [(1.0, "Z")], point)
        reference = central_difference(
            lambda values: energy(1, 1, operations, [(1.0, "Z")], values).energy,
            point, "phi",
        )
        self.assertAlmostEqual(exact["phi"], reference, places=8)


class TrainingTests(unittest.TestCase):
    def test_local_minimizer_converges_and_reports_history(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": 2.0}},
            {"op": "H", "qubit": 0},
        ]
        result = minimize_energy(
            1, 1, operations, [(1.0, "Z")], {"theta": 0.1},
            max_iterations=200, gradient_tolerance=1e-8, initial_step_size=0.3,
        )
        self.assertEqual(result.stop_reason, "gradient_tolerance")
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energy, -1.0, places=6)
        self.assertLess(abs(math.sin(2.0 * result.parameters["theta"])), 1e-6)
        self.assertTrue(all(record.accepted for record in result.history))

    def test_budget_exhaustion_is_not_reported_as_convergence(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
        ]
        result = minimize_energy(
            1, 1, operations, [(1.0, "Z")], {"theta": 0.2},
            max_iterations=1, gradient_tolerance=1e-10, initial_step_size=0.1,
        )
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_failed_trials_shrink_step_and_last_valid_point_is_kept(self):
        with self.assertRaises(TrainingError):
            minimize_energy(
                1, 1, [{"op": "RZ", "qubit": 0, "angle": {"param": "theta"}}],
                [(1.0, "Z")], {"theta": 0.0}, max_iterations=2,
                gradient_tolerance=1e-8, initial_step_size=0.0,
            )

    def test_training_with_postselection_and_feedback(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "H", "qubit": 1, "condition": [0, 0]},
            {"op": "RZ", "qubit": 1, "angle": {"param": "theta"}, "condition": [0, 0]},
        ]
        hamiltonian = [(0.5, "XX"), (-0.3, "ZI")]
        result = minimize_energy(
            2, 1, operations, hamiltonian, {"theta": 0.2},
            max_iterations=100, gradient_tolerance=1e-7,
            postselection={0: 0}, initial_step_size=0.3,
        )
        self.assertIn(result.stop_reason, {"gradient_tolerance", "max_iterations"})
        self.assertGreater(result.success_probability, 0.2)
        current = energy(
            2, 1, operations, hamiltonian, result.parameters, postselection={0: 0}
        )
        self.assertAlmostEqual(current.energy, result.energy, places=10)


if __name__ == "__main__":
    unittest.main()
