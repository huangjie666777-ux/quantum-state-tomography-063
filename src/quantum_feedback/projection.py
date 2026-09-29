"""Frobenius-nearest projection onto trace-one positive density matrices."""

from __future__ import annotations

import numpy as np


def project_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix onto PSD matrices with unit trace.

    The Frobenius norm is unitarily invariant, so the eigenvalues are
    projected onto the probability simplex while eigenvectors are retained.
    """
    matrix = np.asarray(rho, dtype=complex)
    dimension = matrix.shape[0]
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    projected_eigenvalues = _project_simplex(eigenvalues)
    projected = (
        eigenvectors * projected_eigenvalues
    ) @ eigenvectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    trace_correction = (1.0 - np.trace(projected).real) / dimension
    if trace_correction:
        projected = projected + trace_correction * np.eye(dimension, dtype=complex)
    return projected


def frobenius_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.real(np.trace(difference @ difference.conj().T))))


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of a real vector onto {x >= 0, sum(x) = 1}."""
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    cumulative = np.cumsum(sorted_values)
    rho = (cumulative - 1.0) / (np.arange(len(values)) + 1)
    candidates = sorted_values - rho
    positive = np.flatnonzero(candidates > 0.0)
    cutoff = positive[-1]
    theta = rho[cutoff]
    return np.maximum(values - theta, 0.0)
