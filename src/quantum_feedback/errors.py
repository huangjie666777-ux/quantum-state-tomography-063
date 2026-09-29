"""Exception types raised by the quantum feedback SDK."""


class CircuitValidationError(ValueError):
    """Raised when a circuit or an operation is invalid."""


class SamplingError(ValueError):
    """Raised when sampling arguments are invalid."""


class PostselectionError(ValueError):
    """Raised when postselection is invalid or has vanishing success probability."""


class HamiltonianValidationError(ValueError):
    """Raised when an energy objective is invalid."""


class TrainingError(ValueError):
    """Raised when variational training cannot proceed."""


class TomographyError(ValueError):
    """Raised when a tomography plan or count table is invalid."""
