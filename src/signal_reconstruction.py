from __future__ import annotations
from typing import Callable, Literal
import numpy as np
from scipy.interpolate import CubicSpline


class SignalReconstructor:
    """Reconstruct phi(t) on [0, T] from cell averages psi_i, i = 0..L-1."""

    def __init__(self, n_segments: int, T: float, phase_sequence):
        if n_segments <= 0:
            raise ValueError("n_segments must be positive.")
        self.L = int(n_segments)
        self.T = float(T)
        self.psi = np.asarray(phase_sequence, dtype=float).reshape(-1)
        if self.psi.shape[0] != self.L:
            raise ValueError(
                f"phase_sequence length {self.psi.shape[0]} != n_segments {self.L}"
            )

    def _segment_grid(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Return segment endpoints, step size, and midpoints."""
        times = np.linspace(0.0, self.T, self.L + 1)
        h = self.T / self.L
        mids = 0.5 * (times[:-1] + times[1:])
        return times, h, mids

    def _prefix_integral_nodes(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Build F(t) whose discrete derivative recovers the cell averages psi_i.
        """
        times, h, _ = self._segment_grid()
        F_vals = np.zeros(self.L + 1, dtype=float)
        for k in range(1, self.L + 1):
            F_vals[k] = F_vals[k - 1] + h * self.psi[k - 1]
        return times, F_vals, h

    def differentiate_spline(
        self,
        bc: Literal["clamped", "not-a-knot", "natural"] = "clamped",
        ghost: bool = False,
    ) -> Callable[[np.ndarray | float], np.ndarray]:
        """
        Reconstruct phi(t) via cubic spline on the primitive F(t),
        then differentiate the spline.

        Args:
            bc: Boundary condition for the spline ('clamped'|'not-a-knot'|'natural').
            ghost: Whether to add ghost nodes to stabilize boundaries.
        """
        times, F_vals, h = self._prefix_integral_nodes()
        t_nodes, F_nodes = times, F_vals

        # Optional ghost nodes to stabilize boundaries.
        if ghost:
            t_left = times[0] - 0.5 * h
            t_right = times[-1] + 0.5 * h
            F_left = F_vals[0] - 0.5 * h * self.psi[0]
            F_right = F_vals[-1] + 0.5 * h * self.psi[-1]
            t_nodes = np.concatenate(([t_left], times, [t_right]))
            F_nodes = np.concatenate(([F_left], F_vals, [F_right]))

        if bc == "clamped":
            if self.L >= 2:
                phi0_est = 2.0 * self.psi[0] - self.psi[1]
                phiT_est = 2.0 * self.psi[-1] - self.psi[-2]
            else:
                phi0_est = phiT_est = self.psi[0]
            bc_type = ((1, phi0_est), (1, phiT_est))
        elif bc == "not-a-knot":
            bc_type = "not-a-knot"
        elif bc == "natural":
            bc_type = "natural"
        else:
            raise ValueError("bc must be 'clamped'|'not-a-knot'|'natural'")

        spline_F = CubicSpline(t_nodes, F_nodes, bc_type=bc_type)
        spline_phi = spline_F.derivative()
        return lambda t: np.asarray(spline_phi(t))

    def midpoint_direct_spline(
        self,
        bc: Literal["clamped", "not-a-knot", "natural"] = "not-a-knot",
        *,
        avg_to_point: bool = False,
        return_samples: bool = False,
    ):
        """
        Directly interpolate phi at segment midpoints with an optional
        average-to-point correction and cubic spline.

        Args:
            bc: Boundary condition for the spline ('clamped'|'not-a-knot'|'natural').
            avg_to_point: Whether to convert cell averages to point values.
            return_samples: If True, also return (mids, samples).
        """
        times, h, mids = self._segment_grid()
        L = self.L
        samples = self.psi.copy()

        if avg_to_point and L >= 3:
            phi_pt = samples.copy()
            # Interior correction.
            phi_pt[1:-1] = (
                samples[1:-1]
                - (samples[0:-2] - 2.0 * samples[1:-1] + samples[2:]) / 24.0
            )
            # Boundary correction.
            if L >= 5:
                left_num = (
                    35 * samples[0]
                    - 104 * samples[1]
                    + 114 * samples[2]
                    - 56 * samples[3]
                    + 11 * samples[4]
                )
                right_num = (
                    35 * samples[-1]
                    - 104 * samples[-2]
                    + 114 * samples[-3]
                    - 56 * samples[-4]
                    + 11 * samples[-5]
                )
                phi_pt[0] = samples[0] - left_num / 288.0
                phi_pt[-1] = samples[-1] - right_num / 288.0
            else:
                phi_pt[0] = (
                    samples[0]
                    - (samples[0] - 2.0 * samples[1] + samples[2]) / 24.0
                )
                phi_pt[-1] = (
                    samples[-1]
                    - (samples[-1] - 2.0 * samples[-2] + samples[-3]) / 24.0
                )
            samples = phi_pt

        midpoint_spline = CubicSpline(mids, samples, bc_type=bc)
        if return_samples:
            return (lambda t: np.asarray(midpoint_spline(t))), mids, samples
        return lambda t: np.asarray(midpoint_spline(t))

    def build_evaluator(
        self,
        route: Literal[
            "differentiate-spline",
            "midpoint-direct-spline",
        ] = "differentiate-spline",
        **kwargs,
    ) -> Callable[[np.ndarray | float], np.ndarray]:
        """
        Build a callable phi_hat(t) according to the chosen reconstruction route.

        Routes:
            'differentiate-spline'  : spline on F(t), then differentiation.
            'midpoint-direct-spline': spline on midpoint samples of phi.
        """
        route = route.lower()

        if route == "differentiate-spline":
            # Only pass args used by differentiate_spline
            allowed = {}
            for k in ("bc", "ghost"):
                if k in kwargs:
                    allowed[k] = kwargs[k]
            return self.differentiate_spline(**allowed)

        if route == "midpoint-direct-spline":
            # Only pass args used by midpoint_direct_spline
            allowed = {}
            for k in ("bc", "avg_to_point", "return_samples"):
                if k in kwargs:
                    allowed[k] = kwargs[k]
            out = self.midpoint_direct_spline(**allowed)
            if isinstance(out, tuple):
                return out[0]
            return out

        raise ValueError("Unknown route. Use 'differentiate-spline' or 'midpoint-direct-spline'.")

