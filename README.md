# Velocity of a Quantum Particle in a Classically Forbidden Region — simulation code

Code written by Dustin Lazarovici (Technion) with Claude AI. 

Simulation and analysis code accompanying:

> C. Beck, S. Goldstein, D. Lazarovici, R. Tumulka, and N. Zanghì,
> *Velocity of a Quantum Particle in a Classically Forbidden Region* (2026).

A 2D split-step (GPU/CuPy) solver for a Gaussian wavepacket scattering off a
potential step in a coupled double-well waveguide, with Bohmian-trajectory
integration and dwell-time statistics.

## Requirements

- CUDA-capable GPU and `cupy`
- `numpy >= 2.0` (the code uses `np.trapezoid`)
- `scipy >= 1.7`, `matplotlib`, `tqdm`
- optional: `numba` (CPU trajectory interpolation), `imageio` (video output)

```bash
conda create -n dwtun python=3.11
conda activate dwtun
conda install -c conda-forge cupy "numpy>=2" scipy matplotlib tqdm
```

## Files

| file | role |
|---|---|
| `DWtunneling.py` | main simulation; one run per invocation |
| `batch_sigx_sweep.py` | drives a multi-GPU sweep over `sigx` |
| `plot_dwell_time_from_batch.py` | plots the resulting `VEL_*.json` files |

## Verifying a correct build

Two quantities are printed at startup and identify a correct build:

- `hbar*J0 = 26.2 ueV`
- `d0 = -Delta - hbar*J0` **exactly** — e.g. `d0 = +0.017510` at `DETUNING_MEV = -0.07`

`d0` follows from the Eq. (3) offset `delta_y`, which is **added** to the
right-side potential (see the "Right-side theory" block). Subtracting it instead
doubles the transverse zero-point mismatch and is wrong; at `DETUNING_MEV = -0.07`
it opens the symmetric channel and the run is no longer evanescent.

## Reproducing the figures

> **TODO (author):** confirm each row against the run logs before release —
> in particular `x_max` for the propagative runs, which must exceed the
> population-transfer length `X_tun = pi/(2*dk)` (~52 code units at
> `DETUNING_MEV = +0.15`). The default `x_max = +100` places the right
> absorber at `x = 25`, which is too small for the propagative regime.

| figure | regime | key parameters |
|---|---|---|
| Fig. 4 (trajectories) | evanescent | `DETUNING_MEV = -0.07`, `sigx = 100` (sigma_k = 2.5e-3 um^-1) |
| Fig. 5 top (migration histogram) | propagative | `DETUNING_MEV = +0.15`, `sigx = 100`, larger `x_max` |
| Fig. 5 bottom | evanescent | `DETUNING_MEV = -0.07`, `sigx = 100` |
| Fig. 6 (density + trajectories) | propagative | `DETUNING_MEV = +0.15`, `x_max` >= ~300 |
| Fig. 7 (dwell time vs sigma_k) | evanescent | `DETUNING_MEV = -0.09`, sweep `sigx` via `batch_sigx_sweep.py` |

Batch invocation (one run -> one `VEL_<run_tag>.json`):

```bash
python DWtunneling.py --DETUNING_MEV -0.09 --sigx 100 \
    --outdir batch_out --run_tag sigx=100 --traj --no_anim --no_video
```

## Unit system

Code units set `hbar = m = 1`. The length unit is fixed by the transverse well
position, `L0 = y0_exp / y0_code = 2e-6 m`; then `E0 = hbar^2/(m*L0^2)` and
`T0 = hbar/E0`. So 1 code length = 2 um, 1 code energy = 2.4993 meV,
1 code time = 0.2635 ps.

Note that the module-level `L0` is in **metres**; the plotting blocks use a
separate `CODE_TO_UM = 2.0` for micrometres.

## Known limitations

See the "Numerical caveats" section of the module docstring: absorber geometry
versus the ROI, the velocity-field regularisation thresholds, the truncated
initial ensemble, the first-order-in-time RK2 step, and the ROI storage memory
cost. None affects the published runs; all constrain reuse at other parameters.

## License

GPL-3.0-or-later — see `LICENSE`. Derivative works that are distributed must
also be released under the GPL.
