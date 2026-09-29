"""Variational state preparation with a shared parameter, feedback and postselection.

The template trains one named parameter theta that appears in two RZ gates with
different scales (including a negative one). A measurement drives a conditional
RZ feedback gate, and the objective is the energy conditioned on classical bit
0 reading zero.
"""

from quantum_feedback import energy, gradient, minimize_energy


NUM_QUBITS = 2
NUM_CLBITS = 1

OPERATIONS = [
    {"op": "H", "qubit": 0},
    {"op": "RZ", "qubit": 0, "angle": {"param": "theta", "scale": 1.0, "offset": 0.05}},
    {"op": "H", "qubit": 0},
    {"op": "MEASURE", "qubit": 0, "clbit": 0},
    {"op": "H", "qubit": 1},
    {"op": "RZ", "qubit": 1, "angle": {"param": "theta", "scale": -1.0},
     "condition": [0, 0]},
    {"op": "H", "qubit": 1},
]

HAMILTONIAN = [
    (0.8, "IX"),
    (-0.5, "ZI"),
]


def main() -> None:
    initial = {"theta": 0.2}
    before = energy(
        NUM_QUBITS, NUM_CLBITS, OPERATIONS, HAMILTONIAN,
        initial, postselection={0: 0},
    )
    exact_gradient = gradient(
        NUM_QUBITS, NUM_CLBITS, OPERATIONS, HAMILTONIAN,
        initial, postselection={0: 0},
    )
    print("initial parameters:", initial)
    print("initial conditional energy:", before.conditional_energy)
    print("initial success probability:", before.success_probability)
    print("exact gradient:", exact_gradient)

    result = minimize_energy(
        NUM_QUBITS, NUM_CLBITS, OPERATIONS, HAMILTONIAN,
        initial_parameters=initial,
        max_iterations=200,
        gradient_tolerance=1e-8,
        postselection={0: 0},
        initial_step_size=0.3,
    )
    print("stop reason:", result.stop_reason)
    print("iterations used:", result.iterations)
    print("final parameters:", result.parameters)
    print("final conditional energy:", result.energy)
    print("final success probability:", result.success_probability)
    print("accepted history length:", len(result.history))


if __name__ == "__main__":
    main()
