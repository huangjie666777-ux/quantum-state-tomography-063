"""Linear Pauli reconstruction from local-basis count records."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from .energy import PAULI_MATRICES
from .errors import TomographyError
from .projection import frobenius_distance, project_density_matrix
from .tomography_plan import (
    MeasurementPlan,
    all_pauli_strings,
    compatible_settings,
    validate_plan,
)


@dataclass(frozen=True)
class TomographyResult:
    expectations: dict[str, float]
    raw_density_matrix: np.ndarray
    density_matrix: np.ndarray
    min_eigenvalue: float
    correction_distance: float

    @property
    def trace(self) -> float:
        return float(np.real(np.trace(self.raw_density_matrix)))


def reconstruct_state(plan: Any, counts: Any) -> TomographyResult:
    """Reconstruct a state from count data without consulting a simulator."""
    normalized_plan = _coerce_plan(plan)
    normalized_counts = _validate_counts(normalized_plan, counts)
    expectations = _estimate_expectations(normalized_plan, normalized_counts)
    raw = _linear_inversion(normalized_plan.num_qubits, expectations)
    raw_hermitian = 0.5 * (raw + raw.conj().T)
    eigenvalues = np.linalg.eigvalsh(raw_hermitian)
    projected = project_density_matrix(raw_hermitian)
    distance = frobenius_distance(raw_hermitian, projected)
    return TomographyResult(
        expectations=expectations,
        raw_density_matrix=raw_hermitian,
        density_matrix=projected,
        min_eigenvalue=float(np.min(eigenvalues).real),
        correction_distance=distance,
    )


def _coerce_plan(plan: Any) -> MeasurementPlan:
    if isinstance(plan, MeasurementPlan):
        return plan
    if isinstance(plan, int) and not isinstance(plan, bool):
        from .tomography_plan import local_pauli_plan

        return local_pauli_plan(plan)
    raise TomographyError("plan must be a MeasurementPlan")


def _validate_counts(
    plan: MeasurementPlan, counts: Any
) -> dict[str, dict[str, int]]:
    if not isinstance(counts, Mapping):
        raise TomographyError("counts must be a mapping of basis setting -> outcome counts")
    settings = set(plan.settings)
    if set(counts) != settings:
        missing = sorted(settings - set(counts))
        extra = sorted(set(counts) - settings)
        raise TomographyError(
            f"each setting must appear exactly once; missing={missing}, extra={extra}"
        )
    normalized: dict[str, dict[str, int]] = {}
    for setting in plan.settings:
        outcome_counts = counts[setting]
        if not isinstance(outcome_counts, Mapping):
            raise TomographyError(f"counts for setting {setting!r} must be a mapping")
        per_setting: dict[str, int] = {}
        total = 0
        for outcome, count in outcome_counts.items():
            if not isinstance(outcome, str) or len(outcome) != plan.num_qubits:
                raise TomographyError(
                    f"setting {setting!r}: outcome must be a string of length {plan.num_qubits}"
                )
            if any(character not in ("0", "1") for character in outcome):
                raise TomographyError(f"setting {setting!r}: outcome {outcome!r} must contain only 0/1")
            if not _is_plain_integer(count) or isinstance(count, bool):
                raise TomographyError(f"setting {setting!r}, outcome {outcome!r}: count must be an integer")
            count = int(count)
            if count < 0:
                raise TomographyError(f"setting {setting!r}, outcome {outcome!r}: count must be non-negative")
            if outcome in per_setting:
                raise TomographyError(f"setting {setting!r}: duplicated outcome {outcome!r}")
            per_setting[outcome] = count
            total += count
        if total <= 0:
            raise TomographyError(f"setting {setting!r}: total count must be positive")
        normalized[setting] = per_setting
    return normalized


def _estimate_expectations(
    plan: MeasurementPlan, counts: Mapping[str, Mapping[str, int]]
) -> dict[str, float]:
    expectations: dict[str, float] = {}
    for pauli_string in all_pauli_strings(plan.num_qubits):
        signed_total = 0.0
        weighted_total = 0.0
        for setting in compatible_settings(pauli_string, plan):
            outcome_counts = counts[setting]
            total = sum(outcome_counts.values())
            signed = sum(
                count * parity(pauli_string, outcome) for outcome, count in outcome_counts.items()
            )
            signed_total += signed
            weighted_total += total
        expectations[pauli_string] = signed_total / weighted_total
    return expectations


def parity(pauli_string: str, outcome: str) -> int:
    """Return the eigenvalue parity of an outcome for a measured Pauli string."""
    sign = 1
    for letter, bit in zip(reversed(pauli_string), reversed(outcome)):
        if letter != "I" and bit == "1":
            sign = -sign
    return sign


def _linear_inversion(num_qubits: int, expectations: Mapping[str, float]) -> np.ndarray:
    dimension = 1 << num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    for pauli_string, expectation in expectations.items():
        matrix = pauli_matrix(pauli_string)
        rho += expectation * matrix
    return rho / dimension


def pauli_matrix(pauli_string: str) -> np.ndarray:
    factors = [PAULI_MATRICES[letter] for letter in reversed(pauli_string)]
    matrix = factors[0]
    for factor in factors[1:]:
        matrix = np.kron(matrix, factor)
    return matrix


def _is_plain_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, np.integer):
        return bool(isfinite(int(value)))
    return False
