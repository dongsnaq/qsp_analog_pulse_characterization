from __future__ import annotations
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_pulse_reconstruction_single(
    results_for_all_L: Dict[int, dict],
    pulse_name: str,
    L: int,
    *,
    T: float = 1.0,
    num_time_points: int = 2000,
    recon_route: str = "differentiate-spline",
    recon_kwargs: dict | None = None,
    title: str | None = None,
):
    """
    Plot a single pulse reconstruction example for a fixed L.

    The input `results_for_all_L` is the dict returned by
    `run_experiments_fixed_perturb` for one noise configuration.

    For the chosen L and `pulse_name`, this shows:
      - Top: ideal phi(t), actual phi(t), reconstructed phi_hat(t).
      - Bottom: pointwise error phi_hat(t) - phi_actual(t).

    Args:
        results_for_all_L: Output of run_experiments_fixed_perturb (one config).
        pulse_name: Name of the pulse used in that experiment.
        L: Discretization level to visualize.
        T: Total time interval [0, T].
        num_time_points: Number of time samples for plotting.
        recon_route: Reconstruction route for SignalReconstructor.
        recon_kwargs: Extra kwargs passed to `build_evaluator(...)`.
        title: Optional figure title.
    """
    if recon_kwargs is None:
        recon_kwargs = {"bc": "clamped", "ghost": False}

    if L not in results_for_all_L:
        raise KeyError(f"L={L} not found in results.")

    recovered = results_for_all_L[L]["recovered"]
    if pulse_name not in recovered:
        raise KeyError(f"pulse_name={pulse_name} not found in recovered dict.")

    rec_entry = recovered[pulse_name]
    ideal_fn = rec_entry["ideal_pulse_fn"]
    actual_fn = rec_entry["actual_pulse_fn"]
    recon = rec_entry["recon"]

    t = np.linspace(0.0, T, num_time_points)
    phi_ideal = ideal_fn(t)
    phi_actual = actual_fn(t)

    phi_eval = recon.build_evaluator(
        route=recon_route,
        **recon_kwargs,
    )
    phi_hat = phi_eval(t)
    error = phi_hat - phi_actual

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax_top, ax_bottom = axes

    # Top: ideal vs actual vs reconstructed.
    ax_top.plot(t, phi_ideal, label="ideal", linestyle="--", linewidth=1.5, color="0.7")
    ax_top.plot(t, phi_actual, label="actual", linewidth=2.0, color="k")
    ax_top.plot(t, phi_hat, label=r"reconstructed $\hat\phi$", linewidth=1.5)

    ax_top.set_ylabel(r"$\phi(t)$")
    ax_top.legend(loc="best")

    # Bottom: error.
    ax_bottom.plot(t, error, linewidth=1.5)
    ax_bottom.axhline(0.0, color="0.7", linestyle="--", linewidth=1.0)
    ax_bottom.set_xlabel("t")
    ax_bottom.set_ylabel(r"$\hat\phi(t) - \phi(t)$")

    if title is None:
        title = f"Pulse reconstruction (L = {L}, pulse = '{pulse_name}')"

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

