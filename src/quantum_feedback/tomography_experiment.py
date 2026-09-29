"""Simulated data collection for complete local-Pauli tomography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .circuit import bind_parameters, validate_template
from .engine import simulate_branches
from .energy import POSTSELECTION_TOLERANCE, _branch_selected, validate_postselection
from .errors import PostselectionError, SamplingError, TomographyError
from .tomography_plan import MeasurementPlan, local_pauli_plan


@dataclass(frozen=True)
class TomographyCounts:
    counts: dict[str, dict[str, int]]
    success_probability: float
    num_qubits: int


def tomography_experiment(
    num_qubits: int,
    num_clbits: int,
    operations: Any,
    parameter_values: Mapping[str, float] | None,
    shots: Any,
    seed: Any,
    postselection: Mapping[int, int] | None = None,
) -> TomographyCounts:
    """Bind parameters, evolve branches and sample one local-Pauli data set.

    Postselection is applied to existing classical branches only; tomography
    measurements never write into that classical register.
    """
    template = validate_template(num_qubits, num_clbits, operations)
    selected = validate_postselection(postselection, num_clbits)
    values = parameter_values if parameter_values is not None else {}
    bound = bind_parameters(template, values)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise SamplingError("seed must be an integer")
    rng = np.random.default_rng(int(seed))

    plan = local_pauli_plan(num_qubits)
    shots_per_setting = _validate_shots(plan, shots)

    branches = simulate_branches(bound, num_qubits, num_clbits)
    success_probability = 0.0
    conditional_rho = np.zeros_like(branches[0].rho)
    for branch in branches:
        if not _branch_selected(branch, selected):
            continue
        success_probability += branch.probability
        conditional_rho += branch.probability * branch.rho
    if success_probability <= POSTSELECTION_TOLERANCE:
        raise PostselectionError(
            f"postselection success probability {success_probability:.3e} is not above "
            f"{POSTSELECTION_TOLERANCE:.0e}; tomography sampling failed"
        )
    conditional_rho /= success_probability

    counts: dict[str, dict[str, int]] = {}
    dimension = 1 << num_qubits
    for setting in plan.settings:
        rotated_rho = _rotate_to_computational_basis(conditional_rho, setting)
        probabilities = np.clip(np.real(np.diag(rotated_rho)), 0.0, 1.0)
        probabilities /= probabilities.sum()
        sampled = rng.multinomial(shots_per_setting[setting], probabilities)
        counts[setting] = {
            _format_outcome(index, num_qubits): int(count)
            for index, count in enumerate(sampled)
            if count > 0
        }
    return TomographyCounts(
        counts=counts,
        success_probability=float(success_probability),
        num_qubits=num_qubits,
    )


def _validate_shots(plan: MeasurementPlan, shots: Any) -> dict[str, int]:
    if isinstance(shots, Mapping):
        if set(shots) != set(plan.settings):
            raise TomographyError("shots mapping must provide one positive integer per setting")
        normalized = {}
        for setting in plan.settings:
            value = shots[setting]
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or int(value) <= 0
            ):
                raise TomographyError(f"shots for {setting!r} must be a positive integer")
            normalized[setting] = int(value)
        return normalized
    if isinstance(shots, (int, np.integer)) and not isinstance(shots, bool) and int(shots) > 0:
        return {setting: int(shots) for setting in plan.settings}
    raise TomographyError("shots must be a positive integer or a setting -> positive-integer mapping")


def _rotate_to_computational_basis(rho: np.ndarray, setting: str) -> np.ndarray:
    from .operators import H_MATRIX, apply_single_gate

    rotated = rho
    num_qubits = len(setting)
    for position, letter in enumerate(setting):
        qubit = num_qubits - 1 - position
        if letter == "X":
            rotated = apply_single_gate(rotated, H_MATRIX, qubit)
        elif letter == "Y":
            s_dagger = np.diag([1.0, -1.0j]).astype(complex)
            rotated = apply_single_gate(rotated, s_dagger, qubit)
            rotated = apply_single_gate(rotated, H_MATRIX, qubit)
    return rotated


def _format_outcome(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")[::-1]
