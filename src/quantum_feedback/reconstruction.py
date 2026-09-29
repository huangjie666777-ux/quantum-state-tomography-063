"""Linear Pauli reconstruction of a density matrix from finite counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .energy import PAULI_MATRICES
from .errors import TomographyError
from .measurement import validate_plan
from .projection import project_density_matrix


@dataclass(frozen=True)
class TomographyResult:
    plan: tuple[str, ...]
    expectations: dict[str, float]
    density_matrix: np.ndarray
    physical_density_matrix: np.ndarray
    min_eigenvalue: float
    correction_distance: float


def reconstruct_density_matrix(
    plan: Sequence,
    counts: Mapping[str, Mapping[str, int]],
) -> TomographyResult:
    """Reconstruct a state using only a complete local-Pauli plan and counts.

    The simulator state is never consulted.  Each setting must occur exactly
    once; every legal n-bit result string may be omitted and is then zero.
    """
    validated_plan = validate_plan(plan)
    num_qubits = len(validated_plan[0])
    if not isinstance(counts, Mapping):
        raise TomographyError("counts must be a mapping of basis setting -> outcome table")
    if set(counts) != set(validated_plan):
        raise TomographyError("counts must contain exactly one table for each plan setting")

    expected_outcomes = [format(index, f"0{num_qubits}b") for index in range(1 << num_qubits)]
    validated_counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for setting in validated_plan:
        table_input = counts[setting]
        if not isinstance(table_input, Mapping):
            raise TomographyError(f"counts for setting {setting!r} must be a mapping")
        table = {outcome: 0 for outcome in expected_outcomes}
        for outcome, count in table_input.items():
            if not isinstance(outcome, str) or len(outcome) != num_qubits or any(
                bit not in "01" for bit in outcome
            ):
                raise TomographyError(
                    f"setting {setting!r}: outcome strings must be length-{num_qubits} strings of 0 and 1"
                )
            if not isinstance(count, (int, np.integer)) or isinstance(count, bool):
                raise TomographyError(f"setting {setting!r}, outcome {outcome!r}: count must be an integer")
            count = int(count)
            if count < 0:
                raise TomographyError(f"setting {setting!r}, outcome {outcome!r}: counts cannot be negative")
            table[outcome] = count
        total = sum(table.values())
        if total == 0:
            raise TomographyError(f"setting {setting!r}: total count must be greater than zero")
        validated_counts[setting] = table
        totals[setting] = total

    expectations = _estimate_expectations(validated_plan, validated_counts, totals)
    dimension = 1 << num_qubits
    raw_rho = np.zeros((dimension, dimension), dtype=complex)
    for pauli_string, expectation in expectations.items():
        raw_rho += expectation * _pauli_matrix(pauli_string, num_qubits)
    raw_rho /= dimension
    raw_rho = 0.5 * (raw_rho + raw_rho.conj().T)

    physical_rho = project_density_matrix(raw_rho)
    eigenvalues = np.linalg.eigvalsh(raw_rho)
    correction_distance = float(np.linalg.norm(physical_rho - raw_rho, ord="fro"))
    result = TomographyResult(
        plan=validated_plan,
        expectations=expectations,
        density_matrix=raw_rho,
        physical_density_matrix=physical_rho,
        min_eigenvalue=float(np.real(eigenvalues[0])),
        correction_distance=correction_distance,
    )
    return result


def _estimate_expectations(
    plan: tuple[str, ...],
    counts: Mapping[str, Mapping[str, int]],
    totals: Mapping[str, int],
) -> dict[str, float]:
    """Combine parity statistics from compatible settings, weighted by shots."""
    num_qubits = len(plan[0])
    weighted: dict[str, float] = {}
    weights: dict[str, int] = {}
    for setting in plan:
        total = totals[setting]
        for mask in range(1, 1 << num_qubits):
            characters = [
                setting[position] if (mask >> (num_qubits - 1 - position)) & 1 else "I"
                for position in range(num_qubits)
            ]
            pauli_string = "".join(characters)
            measured_positions = [
                position for position, character in enumerate(characters) if character != "I"
            ]
            parity_sum = 0
            for outcome, count in counts[setting].items():
                negative = sum(1 for position in measured_positions if outcome[position] == "1")
                parity_sum += count * (-1 if negative % 2 else 1)
            weighted[pauli_string] = weighted.get(pauli_string, 0.0) + parity_sum
            weights[pauli_string] = weights.get(pauli_string, 0) + total

    identity = "I" * num_qubits
    expectations = {pauli: weighted[pauli] / weights[pauli] for pauli in weighted if weights[pauli]}
    expectations[identity] = 1.0
    return expectations


def _pauli_matrix(pauli_string: str, num_qubits: int) -> np.ndarray:
    factors = [
        PAULI_MATRICES[pauli_string[num_qubits - 1 - qubit]]
        for qubit in range(num_qubits - 1, -1, -1)
    ]
    matrix = factors[0]
    for factor in factors[1:]:
        matrix = np.kron(matrix, factor)
    return matrix
