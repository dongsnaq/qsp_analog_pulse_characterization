from __future__ import annotations
from typing import Callable, Sequence
import numpy as np
import scipy.integrate

from .pauli_utils import sigma_phi, rk4_integrate, build_first_quadrant_midpoint_grid
from .tomography import recover_so3_from_U, recover_U_from_so3, NoisyTomographySimulator


class Pulse_Signal:
    """Represent phi(t) on [0, T] with H(t) = omega * sigma(phi(t))."""

    def __init__(self, signal: Callable[[float], float], T: float):
        self.signal = signal
        self.T = float(T)
        self._num_segments: int | None = None

    def set_num_segments(self, num_segments: int) -> None:
        """Set number of segments used for cell averages."""
        if num_segments <= 0:
            raise ValueError("num_segments must be positive.")
        self._num_segments = int(num_segments)

    def reset_num_segments(self) -> None:
        """Reset segment count so that num_segments must be set again."""
        self._num_segments = None

    @property
    def num_segments(self) -> int:
        """Return current number of segments."""
        if self._num_segments is None:
            raise ValueError(
                "num_segments is not set. Call set_num_segments(...) first."
            )
        return self._num_segments

    def _segment_grid(self) -> tuple[np.ndarray, float]:
        """Return segment endpoints and step size."""
        n = self.num_segments
        times = np.linspace(0.0, self.T, n + 1)
        h = self.T / n
        return times, h

    def average_over_segments(self) -> np.ndarray:
        """Return cell averages of phi over each of L segments."""
        n = self.num_segments
        T = self.T
        tau = T / n
        times = np.linspace(0.0, T, n + 1)
        avgs = np.zeros(n, dtype=float)
        for i in range(n):
            a, b = times[i], times[i + 1]
            integral, _ = scipy.integrate.quad(self.signal, a, b)
            avgs[i] = integral / tau
        return avgs

    def H(self, t: float, omega: float) -> np.ndarray:
        """Return single-qubit Hamiltonian H(t) = omega * sigma(phi(t))."""
        return omega * sigma_phi(self.signal(t))

    def evolve_rk4_full(
        self,
        omega: float,
        n_steps: int = 400,
    ) -> np.ndarray:
        """
        Evolve from t = 0 to t = T using RK4:
            dU/dt = -i H(t, omega) U(t), U(0) = I.
        """

        n_steps = int(max(1, n_steps))

        def _f(U: np.ndarray, t: float) -> np.ndarray:
            return -1j * self.H(t, omega) @ U

        return rk4_integrate(_f, 0.0, self.T, n_steps)


def generate_quadrant_samples_with_rk4(
    pulse: Pulse_Signal,
    omegas: Sequence[float],
    *,
    n_segments: int,
    n_rk4_per_segment: int = 400,
) -> np.ndarray:
    """Generate U(T) for each omega using RK4 and a fixed segmentation."""
    pulse.set_num_segments(n_segments)
    N = len(omegas)
    samples = np.zeros((N, 2, 2), dtype=complex)
    for i, w in enumerate(omegas):
        U = pulse.evolve_rk4_full(w, n_steps=n_rk4_per_segment)
        samples[i] = U
    return samples


def generate_true_unitaries(
    pulse: Pulse_Signal,
    N: int,
    L: int,
    n_rk4_per_segment: int = 400,
) -> np.ndarray:
    """
    Generate "true" unitaries for given pulse and discretization L.

    The frequency grid is chosen as midpoint points on [0, pi/2],
    rescaled by the segment width h = T / L.
    """
    omega_vals = build_first_quadrant_midpoint_grid(N, left=0.0, right=np.pi / 2)
    h = pulse.T / L
    omega_vals = omega_vals / h

    samples = generate_quadrant_samples_with_rk4(
        pulse,
        omega_vals,
        n_segments=L,
        n_rk4_per_segment=n_rk4_per_segment,
    )
    return samples


def generate_noisy_unitaries(
    true_unitaries: np.ndarray,
    sim: NoisyTomographySimulator,
) -> np.ndarray:
    """
    Apply SO(3)-level tomography noise model to true_unitaries.

    This maps each U in SU(2) to an estimated U_hat through:
    1) PTM representation,
    2) noisy SO(3) tomography,
    3) polar projection back to SU(2).
    """
    ptms = np.array([recover_so3_from_U(U) for U in true_unitaries])
    rvecs = np.array([A[1:, 1:] for A in ptms])
    Rhat_1 = sim.estimate_all(rvecs)
    recovered_Us = np.array([recover_U_from_so3(rmat) for rmat in Rhat_1])

    for i in range(len(recovered_Us)):
        err_pos = np.linalg.norm(true_unitaries[i] - recovered_Us[i])
        # Fix the global sign if needed.
        if err_pos > 1:
            recovered_Us[i] *= -1
    return recovered_Us

