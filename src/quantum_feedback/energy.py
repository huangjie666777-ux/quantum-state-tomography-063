"""Pauli energy objectives, deterministic expectations and postselection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from .circuit import (
    AngleExpression,
    ParameterizedOperation,
    bind_parameters,
    validate_template,
)
from .engine import Branch, simulate_branches
from .errors import HamiltonianValidationError, PostselectionError

PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}

POSTSELECTION_TOLERANCE = 1e-12
_SHIFT = np.pi / 2.0


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    pauli_string: str


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    success_probability: float
    conditional_energy: float
    postselection: dict[int, int]


def validate_hamiltonian(terms: Any, num_qubits: Any) -> list[PauliTerm]:
    """Validate real-coefficient Pauli sums of fixed length equal to num_qubits."""
    if not _is_plain_int(num_qubits) or not 1 <= num_qubits <= 6:
        raise HamiltonianValidationError("num_qubits must be an integer from 1 to 6")
    if not isinstance(terms, (list, tuple)) or not terms:
        raise HamiltonianValidationError("hamiltonian terms must be a non-empty list or tuple")
    validated: list[PauliTerm] = []
    for index, raw_term in enumerate(terms):
        where = f"term {index}"
        if isinstance(raw_term, Mapping):
            if set(raw_term) != {"coeff", "pauli"}:
                raise HamiltonianValidationError(f"{where}: term must contain 'coeff' and 'pauli'")
            coefficient = raw_term["coeff"]
            pauli_string = raw_term["pauli"]
        elif isinstance(raw_term, (list, tuple)) and len(raw_term) == 2:
            coefficient, pauli_string = raw_term
        else:
            raise HamiltonianValidationError(f"{where}: term must be (coeff, pauli_string)")
        if not isinstance(coefficient, (int, float, np.floating, np.integer)) or isinstance(
            coefficient, bool
        ):
            raise HamiltonianValidationError(f"{where}: coefficient must be numeric")
        coefficient = float(coefficient)
        if not isfinite(coefficient):
            raise HamiltonianValidationError(f"{where}: coefficient must be finite")
        if not isinstance(pauli_string, str) or len(pauli_string) != num_qubits:
            raise HamiltonianValidationError(
                f"{where}: Pauli string must be a string of length {num_qubits}"
            )
        if any(character not in PAULI_MATRICES for character in pauli_string):
            raise HamiltonianValidationError(f"{where}: Pauli string may contain only I, X, Y, Z")
        validated.append(PauliTerm(coefficient, pauli_string))
    return validated


def hamiltonian_matrix(terms: Sequence[PauliTerm], num_qubits: int) -> np.ndarray:
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for term in terms:
        factors = [
            PAULI_MATRICES[term.pauli_string[num_qubits - 1 - qubit]]
            for qubit in range(num_qubits - 1, -1, -1)
        ]
        pauli = factors[0]
        for factor in factors[1:]:
            pauli = np.kron(pauli, factor)
        matrix = matrix + term.coefficient * pauli
    return matrix


def validate_postselection(postselection: Any, num_clbits: int) -> dict[int, int]:
    if postselection is None:
        return {}
    if not isinstance(postselection, Mapping):
        raise PostselectionError("postselection must be a mapping of clbit -> value")
    selected: dict[int, int] = {}
    for clbit, value in postselection.items():
        if not _is_plain_int(clbit) or not 0 <= clbit < num_clbits:
            raise PostselectionError("postselection classical bit is out of range")
        if not _is_plain_int(value) or value not in (0, 1):
            raise PostselectionError("postselection value must be 0 or 1")
        selected[int(clbit)] = int(value)
    return selected


def energy(
    num_qubits: int,
    num_clbits: int,
    operations: Sequence,
    hamiltonian: Sequence,
    parameter_values: Mapping[str, float] | None = None,
    postselection: Mapping[int, int] | None = None,
) -> EnergyResult:
    """Compute the deterministic expectation, optionally conditioned on classical bits."""
    template = validate_template(num_qubits, num_clbits, operations)
    terms = validate_hamiltonian(hamiltonian, num_qubits)
    selected = validate_postselection(postselection, num_clbits)
    bound = bind_parameters(template, parameter_values or _required_mapping(template))
    return evaluate_bound(num_qubits, bound, terms, selected)


def _required_mapping(template: Sequence[ParameterizedOperation]) -> dict[str, float]:
    from .circuit import template_parameters

    required = template_parameters(template)
    if required:
        from .errors import CircuitValidationError

        raise CircuitValidationError(f"missing parameter value(s): {list(required)}")
    return {}


def evaluate_bound(
    num_qubits: int,
    bound_operations: Sequence,
    terms: Sequence[PauliTerm],
    postselection: Mapping[int, int],
    num_clbits: int | None = None,
) -> EnergyResult:
    if num_clbits is None:
        num_clbits = infer_num_clbits(bound_operations, postselection)
    branches = simulate_branches(list(bound_operations), num_qubits, num_clbits)
    return evaluate_branches(num_qubits, branches, terms, postselection)


def _infer_num_clbits(bound_operations: Sequence) -> int:
    clbits = [0]
    for operation in bound_operations:
        if operation.clbit is not None:
            clbits.append(operation.clbit)
        if operation.condition_clbit is not None:
            clbits.append(operation.condition_clbit)
    return max(clbits) + 1


def infer_num_clbits(
    bound_operations: Sequence, postselection: Mapping[int, int] | None = None
) -> int:
    """Infer a register wide enough for operations and any postselection."""
    inferred = _infer_num_clbits(bound_operations)
    if postselection:
        inferred = max(inferred, max(int(clbit) for clbit in postselection) + 1)
    return inferred


def evaluate_branches(
    num_qubits: int,
    branches: Sequence[Branch],
    terms: Sequence[PauliTerm],
    postselection: Mapping[int, int],
) -> EnergyResult:
    observable = hamiltonian_matrix(terms, num_qubits)
    success_probability = 0.0
    numerator = 0.0
    for branch in branches:
        if not _branch_selected(branch, postselection):
            continue
        weight = branch.probability
        success_probability += weight
        numerator += weight * float(np.real(np.trace(observable @ branch.rho)))
    if success_probability <= POSTSELECTION_TOLERANCE:
        raise PostselectionError(
            f"postselection success probability {success_probability:.3e} is not above "
            f"{POSTSELECTION_TOLERANCE:.0e}; no conditional energy is defined"
        )
    conditional_energy = numerator / success_probability
    return EnergyResult(
        energy=float(conditional_energy),
        success_probability=float(success_probability),
        conditional_energy=float(conditional_energy),
        postselection=dict(postselection),
    )


def unnormalized_quantities(
    num_qubits: int,
    num_clbits: int,
    bound_operations: Sequence,
    terms: Sequence[PauliTerm],
    postselection: Mapping[int, int],
) -> tuple[float, float]:
    """Return (numerator, success_probability) without division or tolerance failure."""
    branches = simulate_branches(list(bound_operations), num_qubits, num_clbits)
    observable = hamiltonian_matrix(terms, num_qubits)
    success_probability = 0.0
    numerator = 0.0
    for branch in branches:
        if not _branch_selected(branch, postselection):
            continue
        success_probability += branch.probability
        numerator += branch.probability * float(np.real(np.trace(observable @ branch.rho)))
    return numerator, success_probability


def _branch_selected(branch: Branch, postselection: Mapping[int, int]) -> bool:
    return all(branch.classical_bits[clbit] == value for clbit, value in postselection.items())


def gradient(
    num_qubits: int,
    num_clbits: int,
    operations: Sequence,
    hamiltonian: Sequence,
    parameter_values: Mapping[str, float],
    postselection: Mapping[int, int] | None = None,
) -> dict[str, float]:
    """Exact parameter gradients via gate-by-gate parameter shift.

    Shared parameters and negative scales are handled per gate. The conditional
    energy is a ratio N/P; both numerator and probability are shifted, so the
    normalization derivative is included.
    """
    template = validate_template(num_qubits, num_clbits, operations)
    terms = validate_hamiltonian(hamiltonian, num_qubits)
    selected = validate_postselection(postselection, num_clbits)
    return gradient_template(
        num_qubits, num_clbits, template, terms, parameter_values, selected
    )


def gradient_template(
    num_qubits: int,
    num_clbits: int,
    template: Sequence[ParameterizedOperation],
    terms: Sequence[PauliTerm],
    parameter_values: Mapping[str, float],
    selected: Mapping[int, int],
) -> dict[str, float]:
    from .circuit import template_parameters

    values = dict(parameter_values)
    base_bound = bind_parameters(template, values)
    base_numerator, base_probability = unnormalized_quantities(
        num_qubits, num_clbits, base_bound, terms, selected
    )
    if base_probability <= POSTSELECTION_TOLERANCE:
        raise PostselectionError(
            f"postselection success probability {base_probability:.3e} is not above "
            f"{POSTSELECTION_TOLERANCE:.0e}; gradient is undefined"
        )

    parameter_names = template_parameters(template)
    derivatives_numerator = {name: 0.0 for name in parameter_names}
    derivatives_probability = {name: 0.0 for name in parameter_names}
    for position, parameterized in enumerate(template):
        angle = parameterized.angle
        if not isinstance(angle, AngleExpression):
            continue
        plus_operations, minus_operations = _shift_operations(
            template, values, position, angle
        )
        plus_n, plus_p = unnormalized_quantities(
            num_qubits, num_clbits, plus_operations, terms, selected
        )
        minus_n, minus_p = unnormalized_quantities(
            num_qubits, num_clbits, minus_operations, terms, selected
        )
        coefficient = 0.5 * angle.scale
        derivatives_numerator[angle.parameter] += coefficient * (plus_n - minus_n)
        derivatives_probability[angle.parameter] += coefficient * (plus_p - minus_p)

    gradients: dict[str, float] = {}
    for name in parameter_names:
        gradients[name] = (
            derivatives_numerator[name]
            - (base_numerator / base_probability) * derivatives_probability[name]
        ) / base_probability
    return gradients


def _shift_operations(
    template: Sequence[ParameterizedOperation],
    values: Mapping[str, float],
    position: int,
    angle: AngleExpression,
) -> tuple[list, list]:
    return (
        _shift_bound(template, values, position, _SHIFT),
        _shift_bound(template, values, position, -_SHIFT),
    )


def _shift_bound(
    template: Sequence[ParameterizedOperation],
    values: Mapping[str, float],
    position: int,
    shift: float,
) -> list:
    bound = bind_parameters(template, values)
    operation = bound[position]
    shifted_angle = operation.angle + shift
    from .circuit import Operation

    bound[position] = Operation(
        operation.name,
        qubit=operation.qubit,
        control=operation.control,
        target=operation.target,
        clbit=operation.clbit,
        angle=shifted_angle,
        probability=operation.probability,
        condition_clbit=operation.condition_clbit,
        condition_value=operation.condition_value,
    )
    return bound


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
