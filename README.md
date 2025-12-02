# In Situ Quantum Analog Pulse Characterization via Structured Signal Processing

This is the support code for the Paper: In Situ Quantum Analog Pulse 
Characterization via Structured Signal Processing. By combining Quantum 
Signal Processing (QSP) with a logical-level analog–digital mapping paradigm, 
our method reconstructs a smooth pulse directly from queries of the time-
ordered propagator, without requiring mid-circuit measurements or additional 
evolution. The code is organized under the `src/` directory and includes 
an example script `example_run.py` that demonstrates a full workflow.

## Installation

This repository is structured as a standard Python project.

Dependencies:

    numpy
    scipy
    matplotlib

## Project Layout

project/
    src/
        __init__.py
        pauli_utils.py            # Pauli matrices, Hamiltonian, RK4 integrator
        pulse_signal.py           # Pulse_Signal class and unitary generation
        qsp_phase_recovery.py     # Fourier-based QSP phase recovery
        signal_reconstruction.py  # Reconstruction of phi(t) from phase averages
        tomography.py             # Robust tomography method
        perturbation.py           # pulse perturbation utilities
        experiments.py            # High-level experiment runner
        plotting.py               # Minimal plotting utilities
    example_run.py                # Example
    README.txt

## Spline-Based Reconstruction Methods

There are two main methods provided in (src/signal_reconstruction.py).

Constructor:

    SignalReconstructor(n_segments, T, phase_sequence)

Two reconstruction routes:

    differentiate-spline:
        Build a cubic spline for the primitive F(t),
        then differentiate to get phi(t).

    midpoint-direct-spline:
        Interpolate midpoint values directly
        Optionally applying high-order stencils.

Public API Example:

    phi_hat = recon.build_evaluator(
        route="midpoint-direct-spline",
        bc="not-a-knot",
        avg_to_point=True
    )

## Example: Cosine Pulse

The file `example_run.py` demonstrates an end-to-end reconstruction of

    phi(t) = cos(pi * t),   t in [0, 1].

How to run:

    python example_run.py

Workflow in example_run.py:

    1. Define ideal pulse phi(t) = cos(t).
    2. Choose L as the reconstruction resolution.
    3. Initialize a noisy simulator subjected to SPAM and depolarizing noise.
    4. Run.
    5. Recover QSP phases.
    6. Build a SignalReconstructor.
    7. Visualize:
         - ideal phi(t)
         - actual perturbed pulse
         - reconstructed phi_hat(t)
         - pointwise error

You will see a two-row matplotlib figure.
