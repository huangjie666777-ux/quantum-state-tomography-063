"""Small embeddable density-matrix quantum-circuit SDK with variational training."""

from .circuit import (
    AngleExpression,
    Operation,
    ParameterizedOperation,
    bind_parameters,
    template_parameters,
    validate_circuit,
    validate_parameters,
    validate_template,
)
from .energy import (
    EnergyResult,
    PauliTerm,
    energy,
    evaluate_branches,
    gradient,
    hamiltonian_matrix,
    validate_hamiltonian,
    validate_postselection,
)
from .engine import execute, run, simulate_branches
from .errors import (
    CircuitValidationError,
    HamiltonianValidationError,
    PostselectionError,
    SamplingError,
    TomographyError,
    TrainingError,
)
from .experiment import TomographyExperiment, simulate_tomography_counts
from .measurement import pauli_measurement_plan, validate_plan
from .reconstruction import TomographyResult, reconstruct_density_matrix
from .results import ExecutionResult
from .training import IterationRecord, TrainingResult, minimize_energy

__all__ = [
    "AngleExpression",
    "CircuitValidationError",
    "EnergyResult",
    "ExecutionResult",
    "HamiltonianValidationError",
    "IterationRecord",
    "Operation",
    "PauliTerm",
    "ParameterizedOperation",
    "PostselectionError",
    "SamplingError",
    "TrainingError",
    "TrainingResult",
    "TomographyError",
    "TomographyExperiment",
    "TomographyResult",
    "bind_parameters",
    "energy",
    "evaluate_branches",
    "execute",
    "gradient",
    "run",
    "simulate_branches",
    "template_parameters",
    "validate_circuit",
    "validate_hamiltonian",
    "validate_parameters",
    "validate_postselection",
    "validate_template",
    "minimize_energy",
    "pauli_measurement_plan",
    "reconstruct_density_matrix",
    "simulate_tomography_counts",
    "validate_plan",
]
