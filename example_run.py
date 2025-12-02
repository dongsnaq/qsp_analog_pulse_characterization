from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src import (
    NoisyTomographySimulator,
    run_experiments_fixed_perturb,
)
from src.plotting import plot_pulse_reconstruction_single


def phi_cos_on_0_1(T: float = 1.0):
    """Return a pulse function phi(t) = cos(t) on [0, T]."""

    def phi(t):
        return np.cos(np.pi * t)

    return phi


def main():
    T = 1.0
    pulse_name = "cos"
    phi_cos = phi_cos_on_0_1(T=T)
    pulse_fns = [(phi_cos, pulse_name)]

    # Choose one discretization level for visualization.
    L = 50
    L_list = [L,]

    # Noise configuration.
    sim = NoisyTomographySimulator(
        delta=0.01,
        alpha=0.95,
        mode="symdiff",
        shots=10_000,
        seed=123,
    )

    rng = np.random.default_rng(42)

    results = run_experiments_fixed_perturb(
        L_list=L_list,
        pulse_fns=pulse_fns,
        sim=sim,
        T=T,
        rng=rng,
    )

    fig, axes = plot_pulse_reconstruction_single(
        results_for_all_L=results,
        pulse_name=pulse_name,
        L=L,
        T=T,
        recon_route="midpoint-direct-spline",
        recon_kwargs={"bc": "not-a-knot", "avg_to_point": True},
        title=r"Pulse learning example: $\phi(t) = \cos(\pi t)$",
    )
    plt.show()


if __name__ == "__main__":
    main()

