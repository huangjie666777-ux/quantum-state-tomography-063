"""Complete local-Pauli measurement plans for 1-3 qubit tomography."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from .errors import TomographyError
from .operators import H_MATRIX, apply_single_gate

PAULI_BASES = ("X", "Y", "Z")

# Measurement eigenstate rotation: rotating the state by U maps the chosen
# Pauli eigenstates onto the computational (Z) eigenstates.
_ROTATIONS = {
    "X": H_MATRIX,
    "Y": H_MATRIX @ np.diag([1.0, -1.0j]).astype(complex),
    "Z": None,
}


def pauli_measurement_plan(num_qubits: Any) -> tuple[str, ...]:
    """Return all 3**n local-Pauli settings; the rightmost character is qubit 0."""
    if not isinstance(num_qubits, (int, np.integer)) or isinstance(num_qubits, bool):
        raise TomographyError("num_qubits must be an integer from 1 to 3")
    num_qubits = int(num_qubits)
    if not 1 <= num_qubits <= 3:
        raise TomographyError("num_qubits must be an integer from 1 to 3")
    return tuple("".join(settings) for settings in product(PAULI_BASES, repeat=num_qubits))


def validate_plan(plan: Any) -> tuple[str, ...]:
    """Validate that a plan contains exactly one copy of every complete setting."""
    if not isinstance(plan, (list, tuple)):
        raise TomographyError("measurement plan must be a list or tuple of basis strings")
    if not plan:
        raise TomographyError("measurement plan must not be empty")
    if not isinstance(plan[0], str):
        raise TomographyError("every plan setting must be a basis string")
    num_qubits = len(plan[0])
    expected = pauli_measurement_plan(num_qubits)
    if len(plan) != len(expected):
        raise TomographyError(
            f"measurement plan must contain exactly {len(expected)} settings for {num_qubits} qubits"
        )
    seen: set[str] = set()
    for setting in plan:
        if not isinstance(setting, str) or len(setting) != num_qubits:
            raise TomographyError("every plan setting must be a basis string of equal length")
        if any(character not in PAULI_BASES for character in setting):
            raise TomographyError("basis strings may contain only X, Y, Z")
        if setting in seen:
            raise TomographyError(f"measurement plan contains repeated setting {setting!r}")
        seen.add(setting)
    if seen != set(expected):
        raise TomographyError("measurement plan is incomplete")
    return tuple(plan)


def rotate_to_computational_basis(rho: np.ndarray, basis_string: str) -> np.ndarray:
    """Rotate a prepared state so a local-Pauli measurement becomes a Z measurement."""
    rotated = rho
    for qubit, basis in enumerate(basis_string):
        rotation = _ROTATIONS[basis]
        if rotation is not None:
            rotated = apply_single_gate(rotated, rotation, qubit)
    return rotated
