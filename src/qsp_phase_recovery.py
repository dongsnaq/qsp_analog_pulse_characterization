from __future__ import annotations
from typing import Dict
import numpy as np


def extend_QSP_data(data: np.ndarray, deg: int) -> np.ndarray:
    """
    Extend first-quadrant data (N, 2, 2) to a 4-quadrant full-circle grid (4N, 2, 2).
    """
    N = data.shape[0]
    ret = np.zeros((2 * N, 2, 2), dtype=data.dtype)
    ret[:N, :, :] = data

    # Reflect across quadrant boundary with sign flips.
    rev_data = data[::-1, :, :].copy()
    rev_data[:, 0, 1] = -rev_data[:, 0, 1]
    rev_data[:, 1, 0] = -rev_data[:, 1, 0]
    rev_data = ((-1) ** deg) * rev_data
    ret[N:, :, :] = rev_data

    full_ret = np.zeros((4 * N, 2, 2), dtype=data.dtype)
    full_ret[: 2 * N, :, :] = ret

    # Reverse again to fill remaining 2N points.
    ret_rev = ret[::-1, :, :].copy()
    ret_rev[:, 0, 1] = -ret_rev[:, 0, 1]
    ret_rev[:, 1, 0] = -ret_rev[:, 1, 0]
    full_ret[2 * N :, :, :] = ret_rev
    return full_ret


def get_matrix_fourier_coeffs_midpoint(
    samples: np.ndarray,
    d: int,
) -> Dict[int, np.ndarray]:
    """
    Compute matrix-valued Fourier coefficients on a midpoint grid.

    Args:
        samples: Array of shape (M, 2, 2) sampled on [0, 2*pi), M even.
        d: Maximum harmonic to keep. Requires d < M / 4.
    """
    samples = np.asarray(samples)
    assert samples.ndim == 3 and samples.shape[1:] == (2, 2)
    M = samples.shape[0]
    assert M % 2 == 0, "M must be even (midpoint full-circle grid)"
    assert d < M // 4, f"Need d < M/4 to avoid aliasing, got d={d}, M/4={M/4}"

    C_fft = (1.0 / M) * np.fft.fft(samples, axis=0)
    coeffs: Dict[int, np.ndarray] = {}

    # Positive harmonics including 0.
    for p in range(d + 1):
        coeffs[p] = C_fft[p, :, :] * np.exp(-1j * p * np.pi / M)

    # Negative harmonics from the high-frequency tail.
    for p in range(M - d, M):
        k = p - M
        coeffs[k] = C_fft[p, :, :] * np.exp(-1j * k * np.pi / M)
    return coeffs


def convert_C_to_projection(C: np.ndarray) -> np.ndarray:
    """Return rank-1 projector proportional to C^dagger C."""
    CdagC = C.transpose().conjugate() @ C
    tr_val = np.trace(CdagC)
    return CdagC / tr_val


def reduction_and_compute_phase_factors(coeffs: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Single-sided recursive reduction to recover QSP phases from Fourier data.

    coeffs maps k in [-d, ..., d] to 2x2 matrices.
    """
    d = (len(coeffs) - 1) // 2
    phases = np.zeros(d, dtype=float)
    coeffs_curr = coeffs.copy()

    for j in range(d, 0, -1):
        P = convert_C_to_projection(coeffs_curr[j])
        sin_phi = np.imag(P[0, 1] - P[1, 0])
        phi = np.arcsin(sin_phi)
        phases[j - 1] = phi

        Q = convert_C_to_projection(coeffs_curr[-j])
        prev = coeffs_curr.copy()
        coeffs_curr = {}
        for k in range(-j + 1, j):
            coeffs_curr[k] = prev[k - 1] @ Q + prev[k + 1] @ P

    return phases


def double_sided_reduction_and_compute_phase_factors(
    coeffs: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Combine right- and left-sided reductions to stabilize the phase sequence.
    """
    phi_right = reduction_and_compute_phase_factors(coeffs)
    phi_left = reduction_and_compute_phase_factors(
        {k: C.transpose() for k, C in coeffs.items()}
    )
    d = len(phi_right)
    phi_out = np.zeros_like(phi_right)
    mid = d // 2
    phi_out[mid:] = phi_right[mid:]
    phi_out[:mid] = -phi_left[: -mid - 1 : -1]
    return phi_out


class QSPPhaseRecoveryFromQuadrant:
    """
    End-to-end recovery of QSP phases from first-quadrant midpoint samples.
    """

    def __init__(self, samples_quadrant: np.ndarray, deg: int) -> None:
        self.samples_quadrant = np.asarray(samples_quadrant)
        self.deg = int(deg)
        assert self.samples_quadrant.ndim == 3 and self.samples_quadrant.shape[1:] == (
            2,
            2,
        )
        N = self.samples_quadrant.shape[0]
        assert self.deg < N, f"Require deg < N (here N={N}, deg={self.deg})"

    def run(self) -> np.ndarray:
        """Run extension, Fourier transform, and double-sided reduction."""
        qsp_extend = extend_QSP_data(self.samples_quadrant, self.deg)
        coeffs = get_matrix_fourier_coeffs_midpoint(qsp_extend, self.deg)
        phases = double_sided_reduction_and_compute_phase_factors(coeffs)
        return phases

