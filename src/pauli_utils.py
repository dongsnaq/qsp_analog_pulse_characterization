from __future__ import annotations
from typing import Callable
import numpy as np

# Pauli matrices
paulix = np.array([[0, 1], [1, 0]], dtype=complex)
pauliy = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauliz = np.array([[1, 0], [0, -1]], dtype=complex)
paulii = np.eye(2, dtype=complex)


def sigma_phi(phi: float) -> np.ndarray:
    """Return sigma(phi) = cos(phi) * sigma_x + sin(phi) * sigma_y."""
    return np.cos(phi) * paulix + np.sin(phi) * pauliy


def rk4_integrate(
    f: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    t1: float,
    n: int,
) -> np.ndarray:
    """Classic RK4 for dU/dt = f(U, t) with U(t0) = I."""
    h = (t1 - t0) / n
    u = paulii.copy()
    for i in range(n):
        t = t0 + i * h
        k1 = h * f(u, t)
        k2 = h * f(u + k1 / 2, t + h / 2)
        k3 = h * f(u + k2 / 2, t + h / 2)
        k4 = h * f(u + k3, t + h)
        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u


def build_first_quadrant_midpoint_grid(
    N: int,
    left: float = 0.0,
    right: float = np.pi / 2,
) -> np.ndarray:
    """Return N midpoints on [left, right] in the first quadrant."""
    h = (right - left) / N
    return left + (np.arange(N) + 0.5) * h

