from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy.linalg import fractional_matrix_power

from .pauli_utils import paulii, paulix, pauliy, pauliz


def hat(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix hat(v) for v in R^3."""
    x, y, z = v
    return np.array(
        [[0, -z, y], [z, 0, -x], [-y, x, 0]],
        float,
    )


def random_SO3(rng: np.random.Generator) -> np.ndarray:
    """Draw a random rotation matrix in SO(3) using axis-angle sampling."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(0, 2 * np.pi)
    K = hat(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def polar_orthogonal(B: np.ndarray) -> np.ndarray:
    """Project B to the closest orthogonal matrix via polar decomposition."""
    H = B.T @ B
    H_inv_sqrt = fractional_matrix_power(H, -0.5)
    return B @ H_inv_sqrt


def rand_gen(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Generate a random traceless 3x3 Gaussian matrix."""
    G = rng.normal(size=(3, 3))
    G -= np.trace(G) / 3.0 * np.eye(3)
    return scale * G


def random_pair(
    rng: np.random.Generator,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair of independent traceless Gaussian matrices."""
    return rand_gen(rng, scale), rand_gen(rng, scale)


def symmetric_diff_pair(
    rng: np.random.Generator,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A + H, A - H where H is symmetric traceless."""
    A = rand_gen(rng, scale)
    Q = rng.normal(size=(3, 3))
    H = 0.5 * (Q + Q.T)
    H -= np.trace(H) / 3.0 * np.eye(3)
    return A + H, A - H


def ideal_r_vectors(A: np.ndarray) -> dict[tuple[str, str | None], np.ndarray]:
    """Return ideal Bloch vectors for x, y, z measurement settings."""
    return {
        ("z", "+"): A[:, 2],
        ("z", "-"): -A[:, 2],
        ("x", None): A[:, 0],
        ("y", None): A[:, 1],
    }


def sample_expectations(
    r_dict: dict[tuple[str, str | None], np.ndarray],
    shots: int | None,
    rng: np.random.Generator,
) -> dict[tuple[str, str | None], np.ndarray]:
    """
    Sample expectation vectors with binomial noise per component.

    If shots is None, return the means deterministically.
    """
    r_hat: dict[tuple[str, str | None], np.ndarray] = {}
    for key, vec in r_dict.items():
        mu = np.clip(vec, -1.0, 1.0)
        if shots is None:
            r_hat[key] = mu.copy()
        else:
            p = (1.0 + mu) * 0.5
            k = rng.binomial(shots, p)
            m = (k - (shots - k)) / shots
            r_hat[key] = m
    return r_hat


def assemble_A_from_r(
    r_hat: dict[tuple[str, str | None], np.ndarray],
) -> np.ndarray:
    """Assemble a 3x3 matrix from estimated Bloch vectors."""
    A = np.zeros((3, 3))
    A[:, 0] = r_hat[("x", None)]
    A[:, 1] = r_hat[("y", None)]
    A[:, 2] = 0.5 * (r_hat[("z", "+")] - r_hat[("z", "-")])
    return A


class NoisyTomographySimulator:
    """
    Tomography simulator with fixed SPAM maps (M, S) and per-call sampling noise.
    """

    def __init__(
        self,
        delta: float,
        alpha: float,
        mode: str = "random",
        shots: int | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            delta: Scale of SPAM perturbations.
            alpha: Overall visibility factor.
            mode: 'random' or 'symdiff' for generating SPAM structure.
            shots: Number of shots per measurement, or None for noiseless.
            seed: Random seed for reproducibility.
        """
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

        if mode == "random":
            gM, gS = random_pair(self.rng)
        elif mode == "symdiff":
            gM, gS = symmetric_diff_pair(self.rng)
        else:
            raise ValueError("mode must be 'random' or 'symdiff'")

        self.M = np.eye(3) + self.delta * gM
        self.S = np.eye(3) + self.delta * gS

    def _estimate_single(self, R_true: np.ndarray) -> np.ndarray:
        """
        Estimate an SO(3) rotation R_true under SPAM and sampling noise.
        """
        K_true = self.alpha * (self.M @ self.S)
        Atilde_true = self.alpha * (self.M @ R_true @ self.S)

        r_ref = ideal_r_vectors(K_true)
        r_ref_hat = sample_expectations(r_ref, self.shots, self.rng)
        K_hat = assemble_A_from_r(r_ref_hat)

        r_meas = ideal_r_vectors(Atilde_true)
        r_meas_hat = sample_expectations(r_meas, self.shots, self.rng)
        Atilde_hat = assemble_A_from_r(r_meas_hat)

        K_inv_sqrt = fractional_matrix_power(K_hat, -0.5)
        B = K_inv_sqrt @ Atilde_hat @ K_inv_sqrt
        R_hat = polar_orthogonal(B)
        return R_hat

    def estimate_all(self, R_list: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Estimate a list of SO(3) rotations."""
        out: list[np.ndarray] = []
        for R_true in R_list:
            R_hat = self._estimate_single(np.asarray(R_true, float))
            out.append(R_hat)
        return out


def recover_so3_from_U(U: np.ndarray) -> np.ndarray:
    """Recover the SO(3) representation of a single-qubit unitary U."""
    ixyz = [paulii, paulix, pauliy, pauliz]
    ptm = np.zeros((4, 4))
    for j in range(4):
        out_rho = U @ ixyz[j] @ U.conj().T
        for i in range(4):
            ptm[i, j] = np.real(0.5 * np.trace(ixyz[i] @ out_rho))
    return ptm


def recover_U_from_so3(R: np.ndarray) -> np.ndarray:
    """
    Recover a single-qubit unitary U from its SO(3) rotation matrix R.

    This uses the standard mapping from SO(3) to SU(2).
    """
    w = 0.5 * np.sqrt(1 + np.trace(R))
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    U = np.array(
        [[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]],
        dtype=complex,
    )
    return U / np.linalg.norm(U, ord=2)

