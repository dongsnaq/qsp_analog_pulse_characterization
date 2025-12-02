from __future__ import annotations
from typing import Callable
import numpy as np
import scipy as sp
from functools import partial


def piecewise_linear_pulse(t, values, breakpoints=None):
    """
    Simple piecewise linear pulse on [0, 1] with given values at segments.
    """
    values = np.asarray(values)
    if breakpoints is None:
        breakpoints = np.linspace(0, 1, len(values) + 1)[1:-1]
    conditions = (
        [t < breakpoints[0]]
        + [
            (t >= breakpoints[i]) & (t < breakpoints[i + 1])
            for i in range(len(breakpoints) - 1)
        ]
        + [t >= breakpoints[-1]]
    )
    return np.piecewise(t, conditions, values)


def _smoothed_piecewise_linear_pulse_midpts(
    t,
    breakpoints,
    values,
    width: float = 0.1,
):
    """Internal helper for Gaussian smoothed pulse."""
    kernel = np.exp(-(t - breakpoints) ** 2 / (2 * width**2))
    kernel = kernel / np.sum(kernel)
    return np.sum(kernel * values)


def smoothed_piecewise_linear_pulse(
    t,
    values,
    width: float = 0.1,
):
    """
    Evaluate a smoothed version of a piecewise linear pulse.

    The values are taken at midpoints of equal segments on [0, 1].
    """
    values = np.asarray(values)
    new_breakpoints = np.array(
        [(j + 0.5) / len(values) for j in range(len(values))]
    )
    smoothing_fn = partial(
        _smoothed_piecewise_linear_pulse_midpts,
        breakpoints=new_breakpoints,
        values=values,
        width=width,
    )
    return np.vectorize(smoothing_fn)(t)


def convert_to_smoothed_piecewise_linear(
    pulse_function: Callable[[float], float],
    num_points: int,
    width: float = 0.01,
) -> Callable[[float], float]:
    """
    Approximate a general pulse_function by a smoothed piecewise linear one.
    """
    out = np.zeros(num_points)
    for i in range(num_points):
        a = i / num_points
        b = (i + 1) / num_points
        out[i] = sp.integrate.quad(pulse_function, a, b)[0] / (b - a)
    return partial(smoothed_piecewise_linear_pulse, values=out, width=width)


def perturb_and_window_pulse_fn(
    pulse_fn: Callable[[float], float],
    perturbations: np.ndarray,
    num_points: int = 50,
    width: float = 0.02,
) -> Callable[[float], float]:
    """
    Apply a fixed piecewise linear perturbation to pulse_fn and smooth it.
    """
    perturbations = np.asarray(perturbations)

    def perturbed_pulse_fn(t):
        return pulse_fn(t) + piecewise_linear_pulse(t, perturbations)

    return convert_to_smoothed_piecewise_linear(
        perturbed_pulse_fn,
        num_points=num_points,
        width=width,
    )

