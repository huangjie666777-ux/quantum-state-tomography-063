"""Train a one-qubit state, collect local-Pauli counts and reconstruct it."""

import numpy as np

from quantum_feedback import (
    bind_parameters,
    execute,
    local_pauli_plan,
    minimize_energy,
    reconstruct_state,
    tomography_experiment,
    validate_template,
)


def main() -> None:
    operations = [
        {"op": "H", "qubit": 0},
        {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
        {"op": "H", "qubit": 0},
    ]
    hamiltonian = [(1.0, "Z")]
    trained = minimize_energy(
        1,
        1,
        operations,
        hamiltonian,
        {"theta": 0.2},
        max_iterations=100,
        gradient_tolerance=1e-8,
        initial_step_size=0.3,
    )
    print(f"trained energy: {trained.energy:.6f}")
    print(f"parameters: {trained.parameters}")

    collected = tomography_experiment(
        1,
        1,
        operations,
        trained.parameters,
        shots={"X": 1000, "Y": 1000, "Z": 1000},
        seed=2026,
    )
    print(f"postselection success probability: {collected.success_probability:.6f}")
    print(f"counts: {collected.counts}")

    result = reconstruct_state(local_pauli_plan(1), collected.counts)
    rounded_expectations = {name: round(value, 4) for name, value in result.expectations.items()}
    print(f"Pauli expectations: {rounded_expectations}")
    print(f"raw minimum eigenvalue: {result.min_eigenvalue:.6f}")
    print(f"Frobenius projection distance: {result.correction_distance:.6f}")
    print("projected density matrix:")
    print(np.round(result.density_matrix, 6))

    constant = bind_parameters(validate_template(1, 1, operations), trained.parameters)
    exact = execute(constant, 1, 1).density_matrix
    error = np.linalg.norm(result.density_matrix - exact)
    print(f"Frobenius error against exact trained state: {error:.4f}")


if __name__ == "__main__":
    main()
