"""Complete local-Pauli measurement plans for one to three qubits."""

from __future__ import annotations

from itertools import product
from typing import Any

from .errors import TomographyError

PAULI_BASES = ("X", "Y", "Z")


class MeasurementPlan:
    """Ordered collection of local Pauli basis strings.

    Basis strings are indexed with the rightmost character corresponding to
    qubit 0.  Reading 0 (1) in a result string denotes the positive
    (negative) eigenvalue of that qubit's basis observable.
    """

    def __init__(self, num_qubits: int, settings: Any) -> None:
        validated = validate_plan(num_qubits, settings)
        self.num_qubits = validated[0]
        self.settings = validated[1]

    def __len__(self) -> int:
        return len(self.settings)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasurementPlan):
            return NotImplemented
        return self.num_qubits == other.num_qubits and self.settings == other.settings

    def __repr__(self) -> str:
        return f"MeasurementPlan(num_qubits={self.num_qubits}, settings={self.settings!r})"


def local_pauli_plan(num_qubits: Any) -> MeasurementPlan:
    """Return the full X/Y/Z-per-qubit plan with 3**n settings."""
    if not _is_plain_int(num_qubits) or not 1 <= int(num_qubits) <= 3:
        raise TomographyError("tomography supports an integer number of qubits from 1 to 3")
    num_qubits = int(num_qubits)
    letters = ("X", "Y", "Z")
    settings = tuple(
        "".join(choice)
        for choice in product(letters, repeat=num_qubits)
    )
    return MeasurementPlan(num_qubits, settings)


def validate_plan(num_qubits: Any, settings: Any) -> tuple[int, tuple[str, ...]]:
    """Validate that every complete local-Pauli setting appears exactly once."""
    if not _is_plain_int(num_qubits) or not 1 <= int(num_qubits) <= 3:
        raise TomographyError("tomography supports an integer number of qubits from 1 to 3")
    num_qubits = int(num_qubits)
    if not isinstance(settings, (list, tuple)):
        raise TomographyError("measurement settings must be a list or tuple")
    expected = 3**num_qubits
    if len(settings) != expected:
        raise TomographyError(
            f"expected exactly {expected} settings for {num_qubits} qubits, got {len(settings)}"
        )
    seen: set[str] = set()
    normalized: list[str] = []
    for index, setting in enumerate(settings):
        if not isinstance(setting, str) or len(setting) != num_qubits:
            raise TomographyError(f"setting {index}: must be a string of length {num_qubits}")
        if any(character not in PAULI_BASES for character in setting):
            raise TomographyError(f"setting {index}: basis characters must be X, Y or Z")
        if setting in seen:
            raise TomographyError(f"setting {index}: duplicated setting {setting!r}")
        seen.add(setting)
        normalized.append(setting)
    complete = {"".join(choice) for choice in product(PAULI_BASES, repeat=num_qubits)}
    if seen != complete:
        raise TomographyError("measurement plan must contain every local Pauli setting once")
    return num_qubits, tuple(normalized)


def all_pauli_strings(num_qubits: int) -> tuple[str, ...]:
    """Return all 4**n Pauli strings, including identities, in fixed order."""
    return tuple("".join(choice) for choice in product(("I", "X", "Y", "Z"), repeat=num_qubits))


def compatible_settings(pauli_string: str, plan: MeasurementPlan) -> tuple[str, ...]:
    """Return plan settings that measure every non-identity letter of a Pauli string."""
    compatible = []
    for setting in plan.settings:
        if all(
            letter == "I" or setting[len(pauli_string) - 1 - position] == letter
            for position, letter in enumerate(pauli_string)
        ):
            compatible.append(setting)
    return tuple(compatible)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
