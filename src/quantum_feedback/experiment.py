"""Simulate finite-count tomography experiments on parameterized circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .circuit import bind_parameters, validate_template
from .energy import POSTSELECTION_TOLERANCE, validate_postselection
from .engine import simulate_branches
from .errors import PostselectionError, TomographyError
from .measurement import pauli_measurement_plan, rotate_to_computational_basis


@dataclass(frozen=True)
class TomographyExperiment:
    plan: tuple[str, ...]
    counts: dict[str, dict[str, int]]
    shots: dict[str, int]
    success_probability: float


def simulate_tomography_counts(
    num_qubits: int,
    num_clbits: int,
    operations: Sequence,
    parameter_values: Mapping[str, float] | None = None,
    postselection: Mapping[int, int] | None = None,
    shots: Any = 1000,
    seed: Any = None,
) -> TomographyExperiment:
    """Run every local-Pauli setting once against the postselected final state.

    The template is bound and the existing branching/feedback evolution is
    reused.  Classical records are read for postselection but never modified;
    tomography readouts are returned in separate count tables.
    """
    template = validate_template(num_qubits, num_clbits, operations)
    selected = validate_postselection(postselection, num_clbits)
    bound = bind_parameters(template, dict(parameter_values or {}))
    plan = pauli_measurement_plan(num_qubits)
    shot_table = _validate_shots(shots, plan)
    if seed is not None and (
        not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
    ):
        raise TomographyError("seed must be an integer or None")

    branches = simulate_branches(bound, num_qubits, num_clbits)
    success_probability = 0.0
    dimension = 1 << num_qubits
    conditional_rho = np.zeros((dimension, dimension), dtype=complex)
    for branch in branches:
        if all(branch.classical_bits[clbit] == value for clbit, value in selected.items()):
            success_probability += branch.probability
            conditional_rho += branch.probability * branch.rho
    if success_probability <= POSTSELECTION_TOLERANCE:
        raise PostselectionError(
            f"postselection success probability {success_probability:.3e} is not above "
            f"{POSTSELECTION_TOLERANCE:.0e}; tomography is undefined"
        )
    conditional_rho /= success_probability

    rng = np.random.default_rng(seed)
    counts: dict[str, dict[str, int]] = {}
    for setting in plan:
        rotated = rotate_to_computational_basis(conditional_rho, setting)
        probabilities = np.clip(np.real(np.diag(rotated)), 0.0, 1.0)
        probabilities = probabilities / probabilities.sum()
        setting_shots = shot_table[setting]
        draws = rng.multinomial(setting_shots, probabilities)
        table = {
            _format_outcome(index, num_qubits): int(count)
            for index, count in enumerate(draws)
            if count > 0
        }
        counts[setting] = table
    return TomographyExperiment(
        plan=plan,
        counts=counts,
        shots=dict(shot_table),
        success_probability=float(success_probability),
    )


def _validate_shots(shots: Any, plan: tuple[str, ...]) -> dict[str, int]:
    if isinstance(shots, Mapping):
        if set(shots) != set(plan):
            raise TomographyError("shots mapping must provide a count for every plan setting")
        table = {}
        for setting in plan:
            value = shots[setting]
            table[setting] = _positive_shots(value, setting)
        return table
    return {setting: _positive_shots(shots, setting) for setting in plan}


def _positive_shots(value: Any, setting: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TomographyError(f"shots for setting {setting!r} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise TomographyError(f"shots for setting {setting!r} must be a positive integer")
    return value


def _format_outcome(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")[::-1]
