"""Train a two-qubit variational state, collect finite counts and reconstruct it."""

import numpy as np

from quantum_feedback import (
    minimize_energy,
    pauli_measurement_plan,
    reconstruct_density_matrix,
    simulate_tomography_counts,
)

NUM_QUBITS = 2
NUM_CLBITS = 1
SHOTS_PER_SETTING = 2000

OPERATIONS = [
    {"op": "H", "qubit": 0},
    {"op": "RZ", "qubit": 0, "angle": {"param": "theta"}},
    {"op": "CX", "control": 0, "target": 1},
]

HAMILTONIAN = [
    (1.0, "ZZ"),
    (0.7, "XX"),
    (-0.4, "ZI"),
]


def main() -> None:
    trained = minimize_energy(
        NUM_QUBITS,
        NUM_CLBITS,
        OPERATIONS,
        HAMILTONIAN,
        {"theta": 0.2},
        max_iterations=200,
        gradient_tolerance=1e-8,
        initial_step_size=0.3,
    )
    print("training stop reason:", trained.stop_reason)
    print("trained parameters:", trained.parameters)
    print("trained energy:", trained.energy)

    experiment = simulate_tomography_counts(
        NUM_QUBITS,
        NUM_CLBITS,
        OPERATIONS,
        trained.parameters,
        shots=SHOTS_PER_SETTING,
        seed=2026,
    )
    print("tomography settings:", len(experiment.plan))
    print("total shots:", sum(experiment.shots.values()))
    print("success probability:", experiment.success_probability)

    tomography = reconstruct_density_matrix(experiment.plan, experiment.counts)
    print("raw trace:", complex(np.trace(tomography.density_matrix)))
    print("raw minimum eigenvalue:", tomography.min_eigenvalue)
    print("physical correction Frobenius distance:", tomography.correction_distance)
    print("selected Pauli expectations:")
    for pauli in ("XX", "ZZ", "ZI", "IZ"):
        print(f"  <{pauli}> = {tomography.expectations[pauli]: .4f}")
    reconstructed_energy = sum(
        coefficient * tomography.expectations[pauli]
        for coefficient, pauli in HAMILTONIAN
    )
    print("energy from reconstructed expectations:", reconstructed_energy)


if __name__ == "__main__":
    main()
