"""Frobenius-nearest projection onto trace-one positive semidefinite matrices."""

from __future__ import annotations

import numpy as np


def project_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix onto {rho >= 0, Tr(rho) = 1} in Frobenius norm.

    Unitary invariance reduces the problem to projecting the eigenvalue vector
    onto the probability simplex.
    """
    hermitian = 0.5 * (rho + rho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    projected_eigenvalues = _project_simplex(np.real(eigenvalues))
    projected = (
        eigenvectors * projected_eigenvalues
    ) @ eigenvectors.conj().T
    return 0.5 * (projected + projected.conj().T)


def _project_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    running_sum = np.cumsum(sorted_values)
    candidates = (running_sum - 1.0) / np.arange(1, len(values) + 1)
    feasible = sorted_values > candidates
    feasible_indices = np.flatnonzero(feasible)
    cutoff = candidates[feasible_indices[-1]]
    return np.maximum(values - cutoff, 0.0)
