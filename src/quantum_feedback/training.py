"""Deterministic local variational minimization with backtracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from .circuit import (
    ParameterizedOperation,
    bind_parameters,
    template_parameters,
    validate_parameters,
    validate_template,
)
from .energy import (
    POSTSELECTION_TOLERANCE,
    PauliTerm,
    validate_hamiltonian,
    validate_postselection,
)
from .errors import PostselectionError, TrainingError

STEP_SHRINK_FACTOR = 0.5
MIN_STEP_SIZE = 1e-12
ENERGY_REL_SLACK = 1e-12


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    parameters: dict[str, float]
    energy: float
    success_probability: float
    gradient_norm: float
    step_size_used: float | None
    accepted: bool


@dataclass(frozen=True)
class TrainingResult:
    parameters: dict[str, float]
    energy: float
    success_probability: float
    gradient: dict[str, float]
    history: tuple[IterationRecord, ...] = field(default_factory=tuple)
    iterations: int = 0
    stop_reason: str = ""

    @property
    def converged(self) -> bool:
        return self.stop_reason == "gradient_tolerance"


def minimize_energy(
    num_qubits: int,
    num_clbits: int,
    operations: Sequence,
    hamiltonian: Sequence,
    initial_parameters: Mapping[str, float],
    max_iterations: Any,
    gradient_tolerance: Any,
    postselection: Mapping[int, int] | None = None,
    initial_step_size: float = 0.5,
) -> TrainingResult:
    """Locally minimize a conditional energy with deterministic gradient descent.

    Stop reasons:

    - gradient_tolerance: gradient norm reached the requested tolerance.
    - max_iterations: the iteration budget was exhausted (not convergence).
    - step_size_underflow: no accepted trial remained; the last valid point
      is returned (not convergence and no claim of global optimality).
    """
    template: list[ParameterizedOperation] = validate_template(num_qubits, num_clbits, operations)
    terms: list[PauliTerm] = validate_hamiltonian(hamiltonian, num_qubits)
    selected = validate_postselection(postselection, num_clbits)
    parameter_names = template_parameters(template)
    current = validate_parameters(template, initial_parameters)
    if not _is_plain_int(max_iterations) or max_iterations <= 0:
        raise TrainingError("max_iterations must be a positive integer")
    if (
        not isinstance(gradient_tolerance, (int, float, np.floating, np.integer))
        or isinstance(gradient_tolerance, bool)
        or not isfinite(float(gradient_tolerance))
        or float(gradient_tolerance) <= 0.0
    ):
        raise TrainingError("gradient_tolerance must be a finite positive number")
    if (
        not isinstance(initial_step_size, (int, float, np.floating, np.integer))
        or isinstance(initial_step_size, bool)
        or not isfinite(float(initial_step_size))
        or float(initial_step_size) <= 0.0
    ):
        raise TrainingError("initial_step_size must be a finite positive number")

    gradient_tolerance = float(gradient_tolerance)
    step_size = float(initial_step_size)
    history: list[IterationRecord] = []

    energy_value, success_probability, current_gradient = _objective_and_gradient(
        num_qubits, num_clbits, template, terms, selected, current
    )

    for iteration in range(max_iterations):
        grad_norm = _norm(current_gradient, parameter_names)
        if grad_norm <= gradient_tolerance:
            return TrainingResult(
                parameters=dict(current),
                energy=energy_value,
                success_probability=success_probability,
                gradient=dict(current_gradient),
                history=tuple(history),
                iterations=iteration,
                stop_reason="gradient_tolerance",
            )

        accepted = False
        trial_step = step_size
        candidate = None
        candidate_energy = None
        candidate_probability = None
        candidate_gradient = None
        while trial_step >= MIN_STEP_SIZE:
            trial_parameters = {
                name: current[name] - trial_step * current_gradient[name]
                for name in parameter_names
            }
            try:
                trial_energy, trial_probability, trial_gradient = _objective_and_gradient(
                    num_qubits, num_clbits, template, terms, selected, trial_parameters
                )
            except PostselectionError:
                trial_step *= STEP_SHRINK_FACTOR
                continue
            slack = ENERGY_REL_SLACK * max(1.0, abs(energy_value))
            if trial_energy <= energy_value + slack:
                accepted = True
                candidate = trial_parameters
                candidate_energy = trial_energy
                candidate_probability = trial_probability
                candidate_gradient = trial_gradient
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        parameters=dict(trial_parameters),
                        energy=trial_energy,
                        success_probability=trial_probability,
                        gradient_norm=grad_norm,
                        step_size_used=trial_step,
                        accepted=True,
                    )
                )
                break
            trial_step *= STEP_SHRINK_FACTOR

        if not accepted:
            return TrainingResult(
                parameters=dict(current),
                energy=energy_value,
                success_probability=success_probability,
                gradient=dict(current_gradient),
                history=tuple(history),
                iterations=iteration,
                stop_reason="step_size_underflow",
            )

        if trial_step >= step_size:
            step_size = min(step_size / STEP_SHRINK_FACTOR, initial_step_size)
        else:
            step_size = trial_step
        current = candidate
        energy_value = candidate_energy
        success_probability = candidate_probability
        current_gradient = candidate_gradient

    return TrainingResult(
        parameters=dict(current),
        energy=energy_value,
        success_probability=success_probability,
        gradient=dict(current_gradient),
        history=tuple(history),
        iterations=max_iterations,
        stop_reason="max_iterations",
    )


def _objective_and_gradient(
    num_qubits: int,
    num_clbits: int,
    template: Sequence[ParameterizedOperation],
    terms: Sequence[PauliTerm],
    selected: Mapping[int, int],
    values: Mapping[str, float],
) -> tuple[float, float, dict[str, float]]:
    from .energy import evaluate_bound, gradient_template

    bound = bind_parameters(template, values)
    result = evaluate_bound(num_qubits, bound, terms, selected, num_clbits)
    gradients = gradient_template(
        num_qubits,
        num_clbits,
        template,
        terms,
        values,
        selected,
    )
    if result.success_probability <= POSTSELECTION_TOLERANCE:
        raise PostselectionError("postselection success probability is too small")
    return result.energy, result.success_probability, gradients


def _norm(gradients: Mapping[str, float], parameter_names: Sequence[str]) -> float:
    return sqrt(sum(gradients[name] * gradients[name] for name in parameter_names))


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
