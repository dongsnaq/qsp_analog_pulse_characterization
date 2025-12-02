from __future__ import annotations
from typing import Sequence
import numpy as np

from .pulse_signal import Pulse_Signal, generate_true_unitaries, generate_noisy_unitaries
from .perturbation import perturb_and_window_pulse_fn
from .qsp_phase_recovery import QSPPhaseRecoveryFromQuadrant
from .signal_reconstruction import SignalReconstructor
from .tomography import NoisyTomographySimulator


def run_experiments_fixed_perturb(
    L_list: Sequence[int],
    pulse_fns,
    sim: NoisyTomographySimulator,
    T: float = 1.0,
    L_perturb: int = 100,
    perturb_scale: float = 0.5,
    perturb_num_points: int = 50,
    perturb_width: float = 0.02,
    n_rk4_per_segment: int = 400,
    base_perturbations: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Run tomography + QSP phase recovery + reconstruction with a fixed perturbation.

    Steps:
      1. Generate or reuse a perturbation vector of length L_perturb.
      2. Use this perturbation to define actual pulses for each ideal pulse.
      3. For each L in L_list, perform tomography, phase recovery, and reconstruction.

    Returns:
      A dict keyed by L with detailed results per pulse name.
    """
    if rng is None:
        rng = np.random.default_rng()

    if base_perturbations is None:
        perturbations = (rng.random(L_perturb) * 2.0 - 1.0) * perturb_scale
    else:
        perturbations = np.asarray(base_perturbations, dtype=float)
        if perturbations.shape[0] != L_perturb:
            raise ValueError(
                f"base_perturbations length {perturbations.shape[0]} != L_perturb {L_perturb}"
            )

    ideal_pulse_map = {name: fn for (fn, name) in pulse_fns}

    perturbed_pulse_fns = []
    for pulse_fn, name in pulse_fns:
        perturbed_fn = perturb_and_window_pulse_fn(
            pulse_fn,
            perturbations=perturbations,
            num_points=perturb_num_points,
            width=perturb_width,
        )
        perturbed_pulse_fns.append((perturbed_fn, name))

    actual_pulse_map = {name: fn for (fn, name) in perturbed_pulse_fns}

    results: dict[int, dict] = {}

    for L in L_list:
        N_omega = L + 1
        true_unitaries_dict = {}
        pulses_dict = {}
        recovered_dict = {}

        for (actual_fn, name) in perturbed_pulse_fns:
            pulse = Pulse_Signal(signal=actual_fn, T=T)
            true_unitaries = generate_true_unitaries(
                pulse,
                N_omega,
                L,
                n_rk4_per_segment=n_rk4_per_segment,
            )
            noisy_unitaries = generate_noisy_unitaries(true_unitaries, sim)
            runner = QSPPhaseRecoveryFromQuadrant(noisy_unitaries, deg=L)
            phases = runner.run()
            recon = SignalReconstructor(n_segments=L, T=T, phase_sequence=phases[::-1])

            true_unitaries_dict[name] = true_unitaries
            pulses_dict[name] = pulse
            recovered_dict[name] = {
                "ideal_pulse_fn": ideal_pulse_map[name],
                "actual_pulse_fn": actual_pulse_map[name],
                "pulse_signal": pulse,
                "true_unitaries": true_unitaries,
                "noisy_unitaries": noisy_unitaries,
                "phases": phases,
                "recon": recon,
            }

        results[L] = {
            "unitaries": true_unitaries_dict,
            "pulses": pulses_dict,
            "recovered": recovered_dict,
            "perturbations": perturbations,
            "L_perturb": L_perturb,
        }

    return results

