"""
Quantum-particle speed in a classically forbidden region: split-step (GPU/CuPy)
simulation of a Gaussian wavepacket scattering off a reflective potential step in
a coupled double-well waveguide, with Bohmian-trajectory and dwell-time analysis.

This code accompanies the paper "Velocity of a Quantum Particle in a Classically
Forbidden Region" (Beck, Goldstein, Lazarovici, Tumulka, Zanghi), a response to
Sharoglazova et al., Nature 643, 67 (2025). It produces the plots and 
additional diagnostics.

Unit system
-----------
Code units set hbar = m = 1. The length unit is fixed by the transverse well
position, L0 = y0_exp / y0_code; the energy unit then follows as
E0 = hbar_SI^2 / (m_SI * L0^2), and the time unit as T0 = hbar_SI / E0. These
are defined once in the UNIT SYSTEM block below and
reported in the PHYSICAL UNIT CONVERSIONS section.

Detuning convention
-------------------
The detuning is Delta = E_kinetic - V0 (incident longitudinal kinetic energy
relative to the step). This coincides numerically with Sharoglazova et al.: they
write Delta = E - V0 + hbar*J0 relative to a different energy reference (their
implicit potential shift puts the band-average at -hbar*J0), so both definitions
yield the same value -- DETUNING_MEV is the detuning as reported in either paper.
Eq. (3), (1/2) hbar*omega = Ebar, is enforced numerically by adding the transverse
zero-point mismatch delta_y to the right-side potential (see the Right-side theory
block); with Eq. (3) holding, the symmetric channel closes (evanescent regime) at
Delta < -hbar*J0, exactly as in the paper.

Running
-------
Interactive (e.g. Spyder): set the parameters in the PARAMETERS section and run.
Batch worker (one run -> one VEL_<run_tag>.json):
    python DWtunneling.py --DETUNING_MEV -0.126 --sigx 100 \
        --outdir batch_out --run_tag sigx=100 --traj --no_anim --no_video
Use batch_sigx_sweep.py to drive a multi-GPU parameter sweep, and
plot_dwell_time_from_batch.py to plot the resulting JSON files.

Provenance
----------
This is the script that produced the figures in the paper. Two checks identify a
correct build, both printed at startup: hbar*J0 = 26.2 ueV, and the evanescent
decay constants satisfying d0 = -Delta - hbar*J0 exactly (e.g. d0 = +0.017510 at
DETUNING_MEV = -0.07). Variants circulating under other names may carry the
Eq. (3) offset with the wrong sign; see the Right-side theory block.

Numerical caveats
-----------------
These do not affect the published runs but constrain reuse at other parameters:

* Absorber vs. ROI. absorber() uses width_frac=0.08 capped at max_width=75, so
  with x_max = +100 the right mask departs from 1 at x = 25 and is effectively
  opaque past x ~ 32.5 -- inside ROI_X2 = 40. Harmless in the evanescent regime
  (decay length 1/(2*kappa_bar) ~ 2 code units), but in the PROPAGATIVE regime the
  population-transfer length is X_tun = pi/(2*dk) ~ 52 code units, so raise x_max
  before trusting anything measured beyond x ~ 25.
* Velocity-field regularisation. VELOCITY_GLOBAL_CUT zeroes the velocity where
  rho < cut * rho_max, and velocities above 5 x the 95th percentile are rescaled.
  Both modify the Bohmian field; check they do not bind in the region analysed.
* Initial ensemble. CLAMP_SIGMA_X / CLAMP_SIGMA_Y truncate the sampled ensemble,
  so it is |psi|^2 conditioned on a band rather than exactly Born-distributed.
* Trajectory integrator. RK2 evaluates the velocity field at a single time, so the
  scheme is midpoint-accurate in space but first order in time.
* Memory. ROI histories are Python lists appended every ROI_HIST_STRIDE steps for
  every trajectory that has ever entered the ROI, until the run ends -- several GB
  at n_traj = 1e6 with a percent-level penetration probability.

Requires: cupy (CUDA), numpy >= 2.0 (np.trapezoid), scipy >= 1.7, matplotlib,
tqdm. Optional: numba, imageio.
"""

import cupy as cp
import numpy as np

import matplotlib
matplotlib.use('Agg')   # force non-interactive, no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit
import gc
import os
import sys
import atexit
from datetime import datetime

# Try to import numba for CPU trajectory interpolation speedup
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[WARNING] numba not found; CPU trajectory interpolation will be slower. Install: pip install numba")

# Try to import imageio for video saving
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("[INFO] imageio not found; video saving will be unavailable. Install: pip install imageio[ffmpeg]")
    

# Alias for CPU-only math (used in CN path)
onp = np


#%% =========== BATCH CLI  =================
# DWfinal can be driven both interactively (Spyder: args_cli is empty, all the
# in-file defaults below apply) and as a batch worker:
#   python DWfinal.py --kx0 ... --sigx ... --outdir batch_out --run_tag tagX
# When run as a batch worker it writes a flat VEL_<run_tag>.json (paper
# convention, corrected sigma_k, dwell fields) that plot_dwell_time_from_batch.py
# consumes -- so DWbatch.py is no longer needed.
import argparse, json
_parser = argparse.ArgumentParser()
_parser.add_argument("--kx0", type=float, default=None)
_parser.add_argument("--sigx", type=float, default=None, help="override sigma_x")
_parser.add_argument("--V_STEP", type=float, default=None)
_parser.add_argument("--DETUNING_MEV", type=float, default=None,
                     help="paper-convention detuning Delta = E_k - V0 (meV)")
_parser.add_argument("--x_max", type=float, default=None)
_parser.add_argument("--x_min", type=float, default=None)
_parser.add_argument("--n_steps", type=int, default=None)
_parser.add_argument("--n_traj", type=int, default=None)
_parser.add_argument("--outdir", type=str, default=None)
_parser.add_argument("--run_tag", type=str, default=None)
_parser.add_argument("--no_video", action="store_true")
_parser.add_argument("--no_anim", action="store_true")
_parser.add_argument("--traj", action="store_true")
_parser.add_argument("--no_save_npz", action="store_true",
                     help="skip the heavy per-run NPZ (JSON summary is enough for batch)")
_parser.add_argument("--device", type=int, default=None)
# Parse only when run as a script; ignore unknown args (robust to Spyder/Jupyter).
if __name__ == "__main__" and sys.argv and sys.argv[0]:
    args_cli, _unknown = _parser.parse_known_args()
else:
    args_cli = argparse.Namespace()

# BATCH_MODE is True whenever a run_tag or outdir is supplied on the CLI.
BATCH_MODE = (getattr(args_cli, "run_tag", None) is not None) or \
             (getattr(args_cli, "outdir", None) is not None)

if getattr(args_cli, "device", None) is not None:
    cp.cuda.Device(args_cli.device).use()
    print(f"[BATCH] Using CUDA device {args_cli.device}")


#%% =========== CLEANUP ======================================================

plt.close('all')

# Clear any figure manager references (Spyder-specific)
try:
    from matplotlib._pylab_helpers import Gcf
    Gcf.destroy_all()
except (ImportError, AttributeError):
    pass

# CLEAR OLD VARIABLES FROM PREVIOUS RUNS (CPU RAM cleanup)
# List of heavy variables to explicitly delete
vars_to_clear = [
    # Trajectory data (biggest RAM users)
    'traj_hist_X_roi', 'traj_hist_Y_roi', 'traj_times_roi',
    'traj_hist_X_bulk', 'traj_hist_Y_bulk', 'traj_times_bulk',
    'traj_positions', 'traj_velocities',
    
    # Large numpy arrays
    'psi_cpu', 'psi_snapshots', 'density_history',
    'y_cpu', 'x_cpu', 'y_np', 'x_np',
    
    # Analysis results
    'residence_lower', 'residence_upper', 'residence_barrier',
    'rho_a_raw', 'rho_a_corrected',
    
    # GPU arrays (belt and suspenders)
    'psi', 'psi_next', 'phi_y_L', 'phi_y_R',
    'y', 'x', 'Vx', 'Vy', 'kx', 'ky', 'kx2', 'ky2',
    'prop_x', 'prop_y'
]

cleared_count = 0
for var_name in vars_to_clear:
    if var_name in globals():
        del globals()[var_name]
        cleared_count += 1
        
if cleared_count > 0:
    print(f"[CLEANUP] Deleted {cleared_count} variables from previous run")

# Force garbage collection (CRITICAL for RAM)
gc.collect()

# Clear CuPy memory pools
try:
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    total_before = mempool.total_bytes()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    total_after = mempool.total_bytes()
    
    if total_before > 0:
        print(f"[CLEANUP] GPU memory freed: {(total_before - total_after)/1e9:.2f} GB")
        
except (NameError, AttributeError, RuntimeError):
    pass

print("[CLEANUP] Cleanup complete")



#%% =========== TOGGLES & SETTINGS =====================

# ====================================================================
# MASTER TRAJECTORY TOGGLE  
# ====================================================================
COMPUTE_TRAJECTORIES = True  # Set to False to skip ALL trajectory computations

USE_ABSORBER        = True   # absorbers on all x-edges and y top/bottom


# Trajectory plots
FULL_TRAJ_PLOT      = False and COMPUTE_TRAJECTORIES  # Requires trajectories   
FULL_TRAJ_MAX       = 2000        # Moderate sample for clean visualization

DO_PLOT_ROI_TURNS   = True and COMPUTE_TRAJECTORIES  # Enable turnaround trajectory plotting

# Detection mode:
#   'any_turn_in_ROI': Detect peaks anywhere within [ROI_X1, ROI_X2]
#   'any_turn_xpos':   Detect peaks only for x > 0 (evanescent region past barrier)
TURN_MODE           = 'any_turn_in_ROI'

# Smoothing settings (crucial for noisy trajectories in low-density regions):
SMOOTH_X_FOR_TURNS  = True                       # Apply Savitzky-Golay smoothing before peak detection
SMOOTH_SEC_TURNS    = 0.02                       # Smoothing time window (seconds)
                                                 #  Smaller: more sensitive to fluctuations, may get false positives
                                                 #  Larger: smoother curves, may miss sharp/fast turnarounds
SMOOTH_MIN_WIN_PTS  = 9                          # Minimum smoothing window (must be odd)
                                            


# Performance warning for disabled trajectories
if not COMPUTE_TRAJECTORIES:
    print("\n" + "="*70)
    print("  TRAJECTORY COMPUTATION DISABLED")
    print("="*70)
    print("  Performance improvements:")
    print("  • Velocity field calculations: SKIPPED")
    print("  • Trajectory integration: SKIPPED")
    print("  • Trajectory storage: SKIPPED")
    print("  • Expected speedup: 30-60%")
    print("  ")
    print("  Note: The following features are automatically disabled:")
    print("  • DO_MEAN_TEST (requires trajectories)")
    print("  • ANALYZE_TURNAROUND_TIMING (requires trajectories)")
    print("  • FULL_TRAJ_PLOT (requires trajectories)")
    print("="*70 + "\n")

# ====================================================================
# ANALYTICAL FEATURES
# ====================================================================
DO_MEAN_TEST        = True and COMPUTE_TRAJECTORIES  # Requires trajectories
DO_RESOLUTION_CHECK      = True   # Analyze grid/time resolution

DO_TIME_AVG_DENSITY      = True   # Time-averaged |ψ|² in ROI
TIME_AVG_WINDOW          = None   # None = auto-detect (uses DO_TRANSMISSION_ANALYTICS if available)
BARRIER_HIT_THRESHOLD    = 0.12   # Start averaging when 12% of peak density reaches barrier

DO_RIGHT_IMBALANCE  = False   # Compute D_ROI(t) from wavefunction

DO_OSCILLATION_ANALYSIS  = False and DO_TIME_AVG_DENSITY  # Requires time-averaged density (propagative: spatial beating; evanescent: equilibration)

ANALYZE_TUNNELING_SPEED = False and COMPUTE_TRAJECTORIES  # Requires trajectories
VELOCITY_ANALYSIS = False and COMPUTE_TRAJECTORIES  # Requires trajectories

# Dwell-time analysis (NEW):
#   (a) wavefunction integral tau = ∫ P(x>0, t) dt, accumulated in the time loop
#   (b) per-trajectory dwell statistics after the run
# Meaningful only for the normalized Gaussian initial condition (not INJECT_PLANE).
DO_DWELL_TIME = True
DWELL_SAMPLE_INTERVAL = 5   # steps between P(x>0) samples for the integral

# Migration-time histogram (transverse upper->lower well tunneling clock).
# Plots p(tau) on the normalized Rabi axis tau = t/(pi/2 J0); the title adapts
# to the detected regime. Requires trajectories.
DO_MIGRATION_HISTOGRAM = True and COMPUTE_TRAJECTORIES
MIGRATION_THRESHOLD_SIGMA_WELL = 0.0   # 0 -> y<0; 1 -> y<-sigma_well; etc.
MIGRATION_MODE = 'redefine'            # 'redefine' (time to y<threshold) or 'filter'
MIGRATION_N_BINS = 30

# Waveguide population ratio analysis (ρ_a)
ANALYZE_RHO_A = False and COMPUTE_TRAJECTORIES  # Requires trajectories (evanescent only)
RHO_A_SPATIAL_BIN = 0.1

# Transmission/Reflection analytics
DO_TRANSMISSION_ANALYTICS = False   # Track T/R coefficients (also enables auto window detection)
FLUX_SAMPLE_INTERVAL = 50       # Steps between flux measurements 

# ====================================================================
# INTEGRATION METHODS
# ====================================================================
# --- Propagator: 'FFT' (GPU split-step) or 'CN_ADI' (CPU Crank–Nicolson ADI) ---
PROPAGATOR          = 'FFT'   # 'FFT' or 'CN_ADI'

# --- VELOCITY FIELD METHOD ---
# 'ADAPTIVE', 'PHASE',  or 'CURRENT_SPECTRAL' (recommended for evanescent)
VELOCITY_METHOD     = 'CURRENT_SPECTRAL'
 
# Velocity method parameters
# NOTE: the next three thresholds regularise the Bohmian velocity field and are
# therefore physics-affecting. VELOCITY_GLOBAL_CUT zeroes v where rho falls below
# this fraction of rho_max; in the evanescent regime rho ~ exp(-2*kappa_bar*x), so
# check that the cut is not reached inside the ROI. CURRENT_SPECTRAL additionally
# rescales any |v| above 5 x the 95th percentile of the masked field.
VELOCITY_GLOBAL_CUT = 1e-10   # Low for deep evanescent
VELOCITY_NOISE_FLOOR = 1e-18  # Extremely low floor
VELOCITY_GRAD_FACTOR = 0.2    # Permissive (for ADAPTIVE)
VELOCITY_PHASE_SMOOTH = False  # Apply smoothing to phase (PHASE method)


# Adaptive/Selective Storage
# ══════════════════════════════════════════════════════════════════
# BULK STORAGE (coarse, all trajectories):
#   - Stride: BULK_HIST_STRIDE (typically 50)
#   - Used for: full trajectory plots, mean position, penetration stats
#
# ROI STORAGE (fine, selective):
#   - Stride: ROI_HIST_STRIDE (typically 5-10)
#   - Used for: turnaround analysis, ROI trajectory plot, tunneling speed, trajectory poluation
# ══════════════════════════════════════════════════════════════════
# MEMORY: once a trajectory enters the ROI it is appended to five host-side lists
# every ROI_HIST_STRIDE steps for the REST of the run, whether or not it is still
# inside. Cost ~ n_entered * (n_steps/ROI_HIST_STRIDE) * 5 Python floats.
# Also sets the resolution of the per-trajectory dwell time (ROI_HIST_STRIDE*dt).
BULK_HIST_STRIDE = 50           
ROI_HIST_STRIDE = 10            

# Enhanced Adaptive CFL + Curvature-aware RK2 substeps
ADAPTIVE_SUBSTEPS      = False
SUBSTEP_CFL_FRAC       = 0.3   
SUBSTEP_CURVATURE_FRAC = 0.3    
SUBSTEP_NEAR_STEP_X    = 4.0    
SUBSTEP_NEAR_STEP_MULT = 2.0    
SUBSTEP_MIN            = 1      
SUBSTEP_MAX            = 8     


# ====================================================================
# LIVE ANIMATION AND VIDEO SAVING
# ====================================================================

LIVE_ANIM           = False     #live animation ON/OFF (turn off for faster execution)
LIVE_ANIM_FIG_SIZE  = (14, 5)   # inches - larger for better visibility
ANIM_EVERY          = 100
NOLOG               = False    #Disable directory creation and print output file

SAVE_VIDEO          = False      # Save animation as video file (disabled for faster run)
VIDEO_FILENAME      = 'simulation.mp4'
VIDEO_FIG_SIZE      = (12, 5)    # inches for video
VIDEO_DPI           = 128        # Recomm: 112, 128, 144, 160 
VIDEO_FPS           = 30         # Frames per second
VIDEO_STRIDE        = 100         # Save every N steps

SAVE_DATA           = True       # Save NPZ with data for key plots

#%%=========== PARAMETERS ============================
# Spatial domain & numerics
hbar, m = 1.0, 1.0
Nx, Ny  = 4096, 512
x_min, x_max = -1100.0, +100.0   
y_min, y_max = -30.0, +30.0
dt      = 0.02                 
n_steps = 75000               
n_traj  = 1000000 

 
# ROI on the right side (size adapts to expected regime)
# NOTE: keep ROI_X2 inside the absorber-free region. absorber() reserves
# min(0.08*(x_max-x_min), 75) at each edge, so the right mask starts at
# x_max - 75 and is opaque ~7.5 units further in. See "Numerical caveats".
ROI_X1, ROI_X2 = 0.0, 40.0      # Extended for propagative

# Double-well parameters and well center
mu, y0  =  4.36e-4, 5.0 #Experiment: mu = 4.36e-4

# Initial Gaussian (used only if INJECT_PLANE=False)
x0, sigx = -450.0, 100  
kx0 = 0.6  # Will be overridden if SET_KX0_FROM_DETUNING=True 

SET_KX0_FROM_DETUNING = True  # True: use detuning, False: use kx0 directly
DETUNING_MEV = -0.07  # Detuning Δ in meV (can be negative)


# Potential Step

SHARP_STEP  = True   
STEP_CENTER = 0.0
STEP_WIDTH_MULT  = 2.0  #Multiple of dx for smooth step
V_STEP      = 0.216     #Experiment: V_STEP  = 0.216

# ---- Apply CLI overrides (batch mode) ----
if getattr(args_cli, "kx0", None) is not None:
    kx0 = float(args_cli.kx0); SET_KX0_FROM_DETUNING = False
if getattr(args_cli, "DETUNING_MEV", None) is not None:
    DETUNING_MEV = float(args_cli.DETUNING_MEV); SET_KX0_FROM_DETUNING = True
if getattr(args_cli, "sigx", None) is not None:   sigx = float(args_cli.sigx)
if getattr(args_cli, "V_STEP", None) is not None: V_STEP = float(args_cli.V_STEP)
if getattr(args_cli, "x_max", None) is not None:  x_max = float(args_cli.x_max)
if getattr(args_cli, "x_min", None) is not None:  x_min = float(args_cli.x_min)
if getattr(args_cli, "n_steps", None) is not None: n_steps = int(args_cli.n_steps)
if getattr(args_cli, "n_traj", None) is not None:  n_traj = int(args_cli.n_traj)
if getattr(args_cli, "no_anim", False):  LIVE_ANIM = False
if getattr(args_cli, "no_video", False): SAVE_VIDEO = False
if getattr(args_cli, "traj", False):     COMPUTE_TRAJECTORIES = True

if BATCH_MODE:
    # Headless, quiet, deterministic batch worker. NOLOG keeps a flat output dir
    # (no per-run timestamped results/ folder); the JSON summary is written at the end.
    LIVE_ANIM = False
    SAVE_VIDEO = False
    NOLOG = True
    COMPUTE_TRAJECTORIES = True   # dwell statistics require trajectories
    DO_DWELL_TIME = True
    if getattr(args_cli, "no_save_npz", False):
        SAVE_DATA = False
    _BATCH_OUTDIR = getattr(args_cli, "outdir", None) or "batch_out"
    os.makedirs(_BATCH_OUTDIR, exist_ok=True)
    RUN_TAG = (getattr(args_cli, "run_tag", None)
               or f"kx0={kx0:.4f}_V={float(V_STEP):.3f}_sigx={float(sigx):.3f}")
    print(f"[BATCH] run_tag={RUN_TAG}, outdir={_BATCH_OUTDIR}")

# Trajectory Sampling

# The sampled ensemble is |psi|^2 TRUNCATED to these bands, not exactly Born-
# distributed (2.5 sigma in y discards ~1.2% of the transverse weight). If the
# bands are tightened enough that the surviving pool falls below n_traj, the
# shortfall is refilled from a UNIFORM distribution -- which is not |psi|^2. At
# the values below the refill never triggers.
CLAMP_OUTLIERS = True
CLAMP_SIGMA_X  = 3.5  
CLAMP_SIGMA_Y  = 2.5  

SAMPLE_Y_LOWER_HALF = False     

# Forward-band x-seeding
X_SEED_FORWARD_BAND = False      
X_SEED_A_SIG        = 1.0       
X_SEED_B_SIG        = 3.7       


# --- Use numeric left ground state instead of analytic HO? ---
LEFT_GROUND_NUMERIC     = True     # True: compute ground state numerically

# Injection (additive source with spectral control)
# ══════════════════════════════════════════════════════════════════════════
# Alternative to Gaussian initial condition: inject a monochromatic plane wave
# from the left boundary with controlled spatial and spectral profiles.
# ══════════════════════════════════════════════════════════════════════════
INJECT_PLANE        = False   # Enable injection (overrides Gaussian initial condition)
INJECT_WIDTH        = 75.0    # Width of injection region (spatial extent)
INJECT_X0           = -250    # Left edge of injection region (x-coordinate)
INJECT_RAMP_STEPS   = 800     # Temporal ramp duration (gradual turn-on to avoid transients)
GAMMA_ADD           = 0.25    # Injection strength (controls amplitude of added wavefunction)

# Spatial window function (controls injection profile in x)
INJECT_WINDOW       = 'TUKEY'     # Options: 'TUKEY' (flexible taper), 'HANN' (smooth cosine)
INJECT_TUKEY_ALPHA  = 0.10        # Tukey window parameter (0=rect, 1=Hann); controls taper width

# Temporal ramp function (gradual turn-on)
TEMPORAL_RAMP_KIND  = 'HANN'      # Options: 'HANN' (smooth sin² ramp), 'LINEAR' (linear ramp)

# Transverse (y) profile of injected beam
Y_PROFILE_KIND      = 'phi'       # Options: 'phi' (left ground state), 'gauss' (Gaussian beam)
Y_FOCUS_SIGMA       = 1.5         # Width parameter for 'gauss' (in units of σ_y)

if INJECT_PLANE:
    x_min = INJECT_X0 - INJECT_WIDTH - 75

    
#%% ============ SETUP =========================  

# ============== Grids =========================
x = cp.linspace(x_min, x_max, Nx)
y = cp.linspace(y_min, y_max, Ny)
dx, dy = (x_max-x_min)/(Nx-1), (y_max-y_min)/(Ny-1)
X, Y = cp.meshgrid(x, y, indexing="xy")

print(f"[VELOCITY] Using method: {VELOCITY_METHOD}")
print(f"[RESOLUTION] Nx={Nx}, Ny={Ny}, dx={dx:.4f}, dy={dy:.4f}")
print(f"[DOMAIN] x ∈ [{float(x_min):.0f}, {float(x_max):.0f}], range = {float(x_max-x_min):.0f}")
print(f"[TIME] dt={dt}, n_steps={n_steps}, total_time={n_steps*dt:.2f}")
if COMPUTE_TRAJECTORIES:
    print(f"[TRAJECTORIES] n_traj={n_traj}, visualization_sample={FULL_TRAJ_MAX}")
else:
    print(f"[TRAJECTORIES] DISABLED (n_traj=0)")


#%% =========== UNIT SYSTEM  ===========
# NOTE: this block is placed here (rather than after the Absorbers section, where it
# used to sit) so that E0/E0_ueV exist before the [EQ3] diagnostic below prints the
# transverse zero-point mismatch in ueV. It depends only on y0 from PARAMETERS.
# Code units set hbar = m = 1. Everything physical is derived from these four
# SI constants plus the transverse well position y0 (code) and y0_exp (SI).
# Defined once here; the PHYSICAL UNIT CONVERSIONS section below only reports them.
hbar_SI  = 1.055e-34     # J s
h_planck = 6.626e-34     # J s
m_SI     = 6.95e-36      # kg  (photon effective mass)
y0_exp   = 10e-6         # m   (experimental transverse well position)
meV_to_J = 1.602e-22     # J per meV

L0     = y0_exp / y0                 # length unit (m): 1 code length = L0
E0     = hbar_SI**2 / (m_SI * L0**2) # energy unit (J), forced by hbar=m=1
T0     = hbar_SI / E0                # time unit (s)
E0_meV = E0 / meV_to_J
E0_ueV = E0 / 1.602e-25
E0_GHz = E0 / h_planck * 1e-9
ueV_to_J = 1.602e-25     # J per ueV


# ===================== Left HO (x<0) transverse ground & energy =====================
omega0 = float(2.0 * y0 * np.sqrt(mu/m))  

def left_ground_true_tridiag(y_cpu, V_y_cpu, hbar=1.0, m=1.0):
    """Ground state solver using symmetric tridiagonal."""
    y_cpu = onp.asarray(y_cpu, float)
    V_y_cpu = onp.asarray(V_y_cpu, float)
    Ny_loc = y_cpu.size
    if Ny_loc < 3:
        raise ValueError("Need at least 3 points on y-grid.")
    dy_loc = y_cpu[1] - y_cpu[0]
    t_main = (hbar**2) / (m * dy_loc**2)
    t_off  = -(hbar**2) / (2 * m * dy_loc**2)
    main_diag = t_main * onp.ones(Ny_loc) + V_y_cpu
    off_diag  = t_off  * onp.ones(Ny_loc - 1)
    Evals, Evecs = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, 0))
    E0 = float(Evals[0])
    phi0 = Evecs[:, 0].copy()
    norm = onp.sqrt(onp.trapezoid(phi0**2, y_cpu) + 1e-300)
    if norm > 0:
        phi0 /= norm
    return E0, phi0

# Build left transverse mode
y_cpu = np.linspace(float(y_min), float(y_max), int(Ny))
dy_cpu = y_cpu[1] - y_cpu[0]

if LEFT_GROUND_NUMERIC:
    V_yL_cpu = 0.5 * m * (omega0**2) * (y_cpu - y0)**2
    E_y_inj, phi_y0_cpu = left_ground_true_tridiag(y_cpu, V_yL_cpu, hbar=hbar, m=m)
    phi_y_L = cp.asarray(phi_y0_cpu, dtype=cp.float64)
else:
    sigy_L = cp.sqrt(hbar/(m*omega0))
    phi_y_L = (1/(cp.pi*sigy_L**2))**0.25 * cp.exp(-0.5*((y - y0)/sigy_L)**2)
    phi_y_L /= cp.sqrt(cp.sum(cp.abs(phi_y_L)**2) * float(dy))
    def Ey_from_phi_HO(phi_y_gpu, omega0_, y0_):
        dphi = cp.gradient(phi_y_gpu, dy)
        Ty = 0.5 * cp.sum(cp.abs(dphi)**2) * float(dy)
        Vy = cp.sum(0.5*m*(omega0_**2)*(y - y0_)**2 * (cp.abs(phi_y_gpu)**2)) * float(dy)
        return float(Ty + Vy)
    E_y_inj = Ey_from_phi_HO(phi_y_L, omega0, y0)

# ===================== Right-side theory =====================
# Right double-well transverse eigenproblem. Solved with eigh_tridiagonal
# (deterministic ascending eigenvalue AND eigenvector ordering). NOTE: the
# previous eigsh + np.sort(vals_R) sorted the eigenVALUES without reordering
# the eigenVECTORS, so vecs_R[:, 0] was not guaranteed to be the ground state.
_main_R = (hbar**2) / (m * dy_cpu**2)
_off_R  = -(hbar**2) / (2 * m * dy_cpu**2)
V_yR = 0.5*mu*(y_cpu**2 - y0**2)**2
vals_R, vecs_R = eigh_tridiagonal(_main_R * np.ones(int(Ny)) + V_yR,
                                  _off_R * np.ones(int(Ny) - 1),
                                  select='i', select_range=(0, 3))
E0_R, E1_R = float(vals_R[0]), float(vals_R[1])
dE_right   = E1_R - E0_R
J0_energy  = 0.5 * dE_right
f_split_R  = dE_right/(2*np.pi*hbar)
T_split_R  = 1.0/ max(f_split_R, 1e-12)

# ===================== Enforce paper Eq. (3): (1/2) hbar*omega = Ebar =====================
# Transverse zero-point mismatch between the left harmonic guide and the right
# double-well doublet (nonzero here because mu_bar = mu; the paper tunes mu_bar
# so this vanishes). E_y_inj sits ABOVE Ebar_R by delta_y, so to enforce Eq.(3)
# (doublet average = left transverse ground) we RAISE the right doublet, i.e. ADD
# delta_y to the right-side potential. Effective average then = Ebar_R + delta_y
# = E_y_inj, so the ground-channel evanescence threshold lands at the paper value
# Delta < -hbar*J0.  DO NOT flip this sign: subtracting delta_y would lower the
# right-side barrier and DOUBLE the mismatch (error 2*delta_y ~ 60 ueV > hbar*J0),
# which shifts kappa_+ by a factor of ~4 at Delta = -0.09 meV and opens the
# symmetric channel outright at Delta = -0.07 meV.  Sanity check on any run:
# d0 printed below must equal -Delta - hbar*J0.
Ebar_R  = 0.5 * (E0_R + E1_R)
delta_y = float(E_y_inj - Ebar_R)
print(f"[EQ3] E_y_inj = {E_y_inj:.6f} (code), Ebar_R = {Ebar_R:.6f} (code)")
print(f"[EQ3] delta_y = E_y_inj - Ebar_R = {delta_y:.6f} (code) "
      f"= {delta_y * E0 / ueV_to_J:.2f} ueV  -> ADDED to right-side potential")
# Keep stored eigenvalues consistent with the SHIFTED potential so that all
# downstream theory (threshR, d0, d1, kappa0, kappa1, L_decay, ...) is paper-
# consistent. J0_energy and dE_right are invariant under this constant shift.
E0_R += delta_y
E1_R += delta_y


# ===================== Potentials =====================
if SHARP_STEP:
    s_x = (X >= STEP_CENTER).astype(cp.float64)
else:
    STEP_WIDTH = STEP_WIDTH_MULT * dx
    s_x = 0.5*(1.0 + cp.tanh((X - STEP_CENTER)/STEP_WIDTH))

V_single = 0.5 * m * (omega0**2) * (Y - y0)**2
V_double = 0.5 * mu * (Y**2 - y0**2)**2
# Right side shifted by +delta_y (paper Eq. (3)); see Right-side theory block.
V = (1.0 - s_x) * V_single + s_x * (V_STEP + V_double + delta_y)
Vhalf = cp.exp(-0.5j*dt*V/hbar)

# Spectral kinetic propagator (2D FFT)
kx = 2*cp.pi*cp.fft.fftfreq(Nx, d=dx)
ky = 2*cp.pi*cp.fft.fftfreq(Ny, d=dy)
KX, KY = cp.meshgrid(kx, ky, indexing="xy")
Kprop = cp.exp(-1j * (hbar/(2*m)) * (KX**2 + KY**2) * dt)

# ===================== Absorbers =====================
def absorber(coord, cmin, cmax, width_frac=0.08, max_width=None, power=6, both=True):
    L = cmax - cmin
    Ledge = width_frac * L
    
    # Apply maximum width constraint
    if max_width is not None:
        Ledge = min(Ledge, max_width)
    
    f = cp.ones_like(coord, dtype=cp.float64)
    
    # Left boundary
    left = (coord - cmin) < Ledge
    if cp.any(left):
        s = (coord[left] - cmin) / Ledge
        f[left] = cp.exp(-((1 - s) * 10)**power)
    
    # Right boundary
    if both:
        right = (cmax - coord) < Ledge
        if cp.any(right):
            s = (cmax - coord[right]) / Ledge
            f[right] = cp.exp(-((1 - s) * 10)**power)
    
    return f

if USE_ABSORBER:
    mask_x = absorber(x, x_min, x_max, width_frac=0.08, max_width=75.0, power=8, both=True)
    mask_y = absorber(y, y_min, y_max, width_frac=0.05, power=6, both=True)
else:
    mask_x = cp.ones(Nx, dtype=cp.float64)
    mask_y = cp.ones(Ny, dtype=cp.float64)

mask = mask_y[:, None] * mask_x[None, :]

#%% =========== COMPUTE kx0 FROM DETUNING (if enabled) =====================

if SET_KX0_FROM_DETUNING:
    print("\n" + "="*70)
    print("COMPUTING kx0 FROM DETUNING (paper convention)")
    print("="*70)
    
    # J0 from the right-well splitting was already computed in SETUP and is
    # invariant under the Eq. (3) potential shift. (The previous version re-solved
    # the eigenproblem here, which also clobbered vecs_R.)
    J0_energy_temp = J0_energy
    
    # Convert detuning from meV to code units (E0, meV_to_J from UNIT SYSTEM block)
    Delta_code = DETUNING_MEV * meV_to_J / E0
    
    # Calculate kx0 from detuning -- PAPER convention:
    #   Delta = E_kinetic - V_STEP            (Delta = E_k - V_0)
    #   E_kinetic = kx0**2/2 (since hbar=m=1)
    #   => kx0 = sqrt(2(Delta + V_STEP))
    # (This Delta coincides with Sharoglazova et al.'s -- same physical detuning,
    #  different energy reference -- so no separate conversion is needed.)
    E_kinetic_target = Delta_code + V_STEP
    
    if E_kinetic_target < 0:
        print(f"[ERROR] Negative kinetic energy!")
        print(f"  Δ = {Delta_code:.6f} (code) = {DETUNING_MEV:.3f} meV")
        print(f"  V_STEP = {V_STEP:.6f}")
        print(f"  Required: Δ + V_STEP > 0")
        print(f"  Try: Δ > {-V_STEP * E0 / meV_to_J:.3f} meV")
        raise ValueError("Cannot have negative kinetic energy")
    
    kx0 = np.sqrt(2 * E_kinetic_target)
    
    print(f"\n[PARAMETERS]")
    print(f"  V_STEP = {V_STEP:.6f} (code) = {V_STEP * E0 / meV_to_J:.3f} meV")
    print(f"  J₀ = {J0_energy_temp:.6f} (code) = {J0_energy_temp * E0 / meV_to_J:.3f} meV")
    print(f"  Detuning Δ = E_k - V₀ = {Delta_code:.6f} (code) = {DETUNING_MEV:.3f} meV")
    
    print(f"\n[CALCULATED]")
    print(f"  E_kinetic = {E_kinetic_target:.6f} (code) = {E_kinetic_target * E0 / meV_to_J:.3f} meV")
    print(f"  kx0 = {kx0:.6f}")
    
    # Regime indication (paper convention): channel +/- closes iff Delta < -/+ hbar*J0.
    # The authoritative channel check (with the actual transverse energies) happens
    # in REGIME DETERMINATION below.
    if Delta_code < -J0_energy_temp:
        print(f"\n  → EVANESCENT regime (Δ < -ℏJ₀: both channels closed)")
    elif Delta_code <= J0_energy_temp:
        print(f"\n  → MIXED regime (|Δ| ≤ ℏJ₀: one channel open); see exact check below")
    else:
        print(f"\n  → PROPAGATIVE regime (Δ > ℏJ₀: both channels open)")
    
    print("="*70 + "\n")

# ===================== Initial ψ =====================
def gauss(grid, x0_, k0_, sigma_):
    return (1/(2*cp.pi*sigma_**2))**0.25 * cp.exp(-(grid-x0_)**2/(4*sigma_**2)) * cp.exp(1j*k0_*(grid-x0_))

if INJECT_PLANE:
    psi = cp.zeros((Ny, Nx), dtype=cp.complex128)
else:
    phi_x = gauss(x, x0, kx0, sigx)
    psi = cp.outer(phi_y_L, phi_x)
    # Normalize with dx*dy for consistency
    psi /= cp.sqrt(cp.sum(cp.abs(psi)**2) * dx * dy + 1e-300)

# ===================== Injection precompute =====================
def _temporal_ramp(step, n_ramp):
    if (not n_ramp) or (step >= n_ramp):
        return 1.0
    x_ = step / float(n_ramp)
    return np.sin(0.5*np.pi*x_)**2 if TEMPORAL_RAMP_KIND.upper() == 'HANN' else x_

def _make_x_window(x_cpu, x0w, x1w):
    """Create spatial window for injection (simplified to TUKEY and HANN only)."""
    L = int(max(8, np.round((x1w - x0w)/float(dx))))
    xs = np.linspace(x0w, x1w, L, endpoint=True)
    u  = np.linspace(0.0, 1.0, L, endpoint=True)
    wname = INJECT_WINDOW.upper()
    
    if wname == 'HANN':
        w = 0.5*(1 - np.cos(2*np.pi*u))
    elif wname == 'TUKEY':
        a = float(INJECT_TUKEY_ALPHA)
        w = np.ones_like(u)
        if a > 0:
            m1 = u <  a/2
            m2 = u > 1 - a/2
            w[m1] = 0.5*(1 - np.cos(2*np.pi*u[m1]/a))
            w[m2] = 0.5*(1 - np.cos(2*np.pi*(1 - u[m2])/a))
    else:
        raise ValueError(f"Unknown INJECT_WINDOW = {INJECT_WINDOW}. Options: 'TUKEY', 'HANN'")

    s = np.zeros(Nx, dtype=np.float64)
    idx = (x_cpu >= x0w) & (x_cpu <= x1w)
    if np.any(idx):
        s[idx] = np.interp(x_cpu[idx], xs, w)
    return s

if INJECT_PLANE:
    OMEGA_INJ = 0.5*float(kx0**2) + E_y_inj
    print(f"[INJECT] Ey_left≈{E_y_inj:.6e}, omega_inj≈{OMEGA_INJ:.6e}, k0={kx0:.3f}")
    Lx = float(x_max - x_min)
    left_abs_L = (0.06 * Lx) if USE_ABSORBER else 0.0
    inj_x0 = float(x_min + left_abs_L + 2*float(dx)) if (INJECT_X0 is None) else float(INJECT_X0)
    inj_x1 = inj_x0 + float(INJECT_WIDTH)
    x_cpu_full = np.linspace(float(x_min), float(x_max), Nx)
    s = _make_x_window(x_cpu_full, inj_x0, inj_x1)
    s2d = cp.asarray(s, dtype=cp.float64)[None, :]

    if Y_PROFILE_KIND.lower() == 'phi':
        y_prof = phi_y_L
    elif Y_PROFILE_KIND.lower() == 'gauss':
        sigy_L_est = cp.sqrt(hbar/(m*omega0))
        sigma = float(Y_FOCUS_SIGMA) * float(cp.asnumpy(sigy_L_est))
        y_prof = cp.exp(-0.5*((y - y0)/sigma)**2)
        y_prof /= cp.sqrt(cp.sum(y_prof**2) * float(dy))
    else:
        raise ValueError(f"Unknown Y_PROFILE_KIND = {Y_PROFILE_KIND}. Options: 'phi', 'gauss'")

    inj_carrier_x = cp.exp(1j * float(kx0) * (x - x_min))
    if USE_ABSORBER:
        inj_mask = (x >= inj_x0) & (x <= inj_x1)
        mask_x[inj_mask] = 1.0
        mask = mask_y[:, None] * mask_x[None, :]

    s2d_c = s2d.astype(cp.complex128, copy=False)
    S0 = (y_prof[:, None]) * (inj_carrier_x[None, :]) * s2d_c

# Regime detection and informative output
E_inj_total = 0.5*float(kx0**2) + E_y_inj
threshR     = float(V_STEP) + E0_R

print("\n" + "="*70)
print("REGIME DETERMINATION")
print("="*70)
print(f"Configuration: V_STEP={V_STEP}, kx0={kx0}")
print(f"Total incident energy:    E_inj = {E_inj_total:.6f}")
print(f"Right ground threshold:   V_STEP + E0_R = {threshR:.6f}")
print(f"Right excited threshold:  V_STEP + E1_R = {float(V_STEP + E1_R):.6f}")

if E_inj_total >= float(V_STEP + E1_R):
    print(f"\n✓ PROPAGATIVE REGIME: Both channels open (above-barrier transmission)")

elif E_inj_total >= threshR:
    print(f"\n⚠ MIXED REGIME: Ground state propagating, excited state closed")
    print(f"  Partial transmission expected")
elif E_inj_total < threshR:
    print(f"\n✓ EVANESCENT REGIME: Both channels closed (tunneling)")


# Calculate theoretical T/R based on regime
d0 = (float(V_STEP) + float(E0_R) - float(E_inj_total))
d1 = (float(V_STEP) + float(E1_R) - float(E_inj_total))
# Define regime flag for later use
is_evanescent = (d0 > 0.0) and (d1 > 0.0)
is_propagative = (d0 <= 0.0) and (d1 <= 0.0)
regime_name = "Propagative" if is_propagative else ("Evanescent" if is_evanescent else "Mixed")

if (d0 > 0.0) and (d1 > 0.0):
    # EVANESCENT REGIME (tunneling)
    kappa0 = np.sqrt(2.0 * d0)
    kappa1 = np.sqrt(2.0 * d1)
    L_decay = 1.0 / (kappa0 + kappa1)
    L_equil = 1.0 / (kappa1 - kappa0)
    print(f"[theory] L_decay ≈ {L_decay:.6f}, L_equil ≈ {L_equil:.6f}")
    
    # WKB approximation for transmission coefficient
    kappa_avg = (kappa0 + kappa1) / 2
    L_eff = 2.5 * L_decay
    T_theory = np.exp(-2 * kappa_avg * L_eff)
    R_theory = 1.0 - T_theory
    print(f"[theory] T_WKB ≈ {T_theory:.6e}, R_WKB ≈ {R_theory:.6f}")
    
elif d0 <= 0.0 and d1 <= 0.0:
    # PROPAGATING REGIME - both channels open (above-barrier transmission)
    print(f"         d0 = {d0:.6f}, d1 = {d1:.6f}")
    print(f"         E_inj = {E_inj_total:.6f} > V_STEP + E1_R = {float(V_STEP + E1_R):.6f}")
    
    # Step potential transmission coefficient from momentum matching
    k_left = float(kx0)  # Incident momentum
    
    # Right-side momentum (accounting for potential step and transverse energy)
    E_kinetic_right_0 = E_inj_total - (V_STEP + E0_R)
    E_kinetic_right_1 = E_inj_total - (V_STEP + E1_R)
    
    if E_kinetic_right_0 > 0:
        k_right_0 = np.sqrt(2.0 * E_kinetic_right_0)
        k_right_1 = np.sqrt(2.0 * E_kinetic_right_1) if E_kinetic_right_1 > 0 else 0.0
        
        # Transmission coefficient for step potential (momentum matching)
        TR0 = 4 * k_left * k_right_0 / (k_left + k_right_0)**2
        TR1 = 4 * k_left * k_right_1 / (k_left + k_right_1)**2 if k_right_1 > 0 else 0.0
        
        # Weight by ground state coupling (crude estimate)
        T_theory = 0.7 * TR0 + 0.3 * TR1  # Favor ground state
        R_theory = 1.0 - T_theory
        
        print(f"[theory] Momentum matching:")
        print(f"         k_left = {k_left:.4f}")
        print(f"         k_right_0 = {k_right_0:.4f}, k_right_1 = {k_right_1:.4f}")
        print(f"         TR0 = {TR0:.4f}, TR1 = {TR1:.4f}")
        print(f"[theory] T_step ≈ {T_theory:.6f}, R_step ≈ {R_theory:.6f}")
    else:
        T_theory = None
        R_theory = None
        print(f"[theory] Warning: Cannot compute transmission (k_right would be imaginary)")
    
else:
    # MIXED REGIME (one evanescent, one propagating)
    print(f"[theory] MIXED regime: ground state {'evanescent' if d0>0 else 'propagating'}, "
          f"excited state {'evanescent' if d1>0 else 'propagating'}")
    print(f"         d0 = {d0:.6f}, d1 = {d1:.6f}")
    
    # Rough estimate: if ground state is evanescent, transmission will be very low
    if d0 > 0.0:
        kappa0 = np.sqrt(2.0 * d0)
        L_decaymix = 1.0 / kappa0
        T_theory = np.exp(-2 * kappa0 * 3 * L_decaymix)  # Rough barrier width
        R_theory = 1.0 - T_theory
        print(f"[theory] T_approx ≈ {T_theory:.6e}, R_approx ≈ {R_theory:.6f}")
    else:
        # Ground propagating, so transmission should be substantial
        T_theory = 0.5  # Very crude estimate
        R_theory = 0.5
        print(f"[theory] T_approx ≈ {T_theory:.6f} (crude mixed-regime estimate)")

# NOTE: the duplicated "Right-side theory" block that used to sit here has been
# REMOVED. It re-solved the UNSHIFTED eigenproblem and overwrote E0_R/E1_R,
# silently undoing the Eq. (3) shift for everything below. vals_R/vecs_R from the
# SETUP block remain in scope and are used by the well-localization code next.

# ===================== Compute well localization width from ground state =====================
# Extract ground state (even superposition, localized in both wells)
phi0_R = vecs_R[:, 0]
phi0_R = phi0_R / np.sqrt(np.trapezoid(phi0_R**2, y_cpu))  # Normalize

# Isolate upper well component (y > 0)
mask_upper = y_cpu > 0
rho_upper = phi0_R**2 * mask_upper
rho_upper = rho_upper / (np.trapezoid(rho_upper, y_cpu) + 1e-300)  # Renormalize

# Compute width of localized component in upper well
y_loc_upper = np.trapezoid(y_cpu * rho_upper, y_cpu)
y2_loc_upper = np.trapezoid(y_cpu**2 * rho_upper, y_cpu)
sigma_loc_upper = np.sqrt(y2_loc_upper - y_loc_upper**2)

# Use this as the physical well width
sigma_well = sigma_loc_upper
y_threshold_physical = y0 - 2*sigma_well

print(f"\n[DOUBLE-WELL GEOMETRY]")
print(f"  Well separation: 2y0 = {2*y0:.2f}")
print(f"  Upper well center (expected): y0 = {y0:.2f}")
print(f"  Upper well center (measured): <y>_upper = {y_loc_upper:.3f}")
print(f"  Well localization width: σ_well = {sigma_well:.3f}")
print(f"  Wells are well-separated: 2y0/σ_well = {2*y0/sigma_well:.2f}")

# ========================================================================
# PROPAGATIVE REGIME: SPATIAL BEATING AND PROPAGATION PREDICTIONS
# ========================================================================

if is_propagative:
    print("\n" + "="*70)
    print("PROPAGATIVE REGIME PREDICTIONS")
    print("="*70)
    
    
    # Spatial beating wavenumber
    k0 = np.sqrt(2.0 * abs(d0)) if d0 < 0 else 0.0
    k1 = np.sqrt(2.0 * abs(d1)) if d1 < 0 else 0.0
    
    if k0 > 0 and k1 > 0:
        Delta_k = abs(k1 - k0)
        lambda_beat = 2*np.pi / Delta_k
        
        print(f"\n[SPATIAL BEATING]")
        print(f"  Channel momenta after barrier:")
        print(f"    k₀ (lower) = √(2|d₀|) = {k0:.6f}")
        print(f"    k₁ (upper) = √(2|d₁|) = {k1:.6f}")
        print(f"  Spatial beating:")
        print(f"    Δk = |k₁ - k₀| = {Delta_k:.6f}")
        print(f"    λ_beat = 2π/Δk = {lambda_beat:.3f} (code)")
       # print(f"    λ_beat = {lambda_beat * L0 * 1e6:.3f} μm (physical)")
    else:
        print(f"\n[SPATIAL BEATING]")
        print(f"  WARNING: One or both channels have k=0 (threshold crossing)")
        Delta_k = 0.0
        lambda_beat = np.inf
    
    # Expected propagation distance
    print(f"\n[PROPAGATION DISTANCE]")
    print(f"  Initial conditions:")
    print(f"    x₀ = {x0:.2f} (starting position)")
    print(f"    kx₀ = {kx0:.6f} (incident momentum)")
    print(f"    σ_x = {sigx:.2f} (packet width)")
    print(f"    v_before = kx₀/m = {kx0:.6f} (velocity before barrier)")
    
    # Time to reach barrier
    if x0 < 0:
        t_to_barrier = abs(x0) / kx0
        print(f"    Distance to barrier: {abs(x0):.2f}")
        print(f"    Time to barrier: t_barrier = {t_to_barrier:.2f}")
    else:
        t_to_barrier = 0.0
        print(f"    Already past barrier")
    
    # After barrier: use average momentum
    if k0 > 0 and k1 > 0:
        k_avg_after = (k0 + k1) / 2.0
        v_after = k_avg_after  # Since m=1
        
        print(f"\n  After barrier (x > 0):")
        print(f"    Average momentum: k_avg = {k_avg_after:.6f}")
        print(f"    Average velocity: v_avg = {v_after:.6f}")
        
        # Remaining time
        T_total = n_steps * dt
        t_remaining = T_total - t_to_barrier
        
        if t_remaining > 0:
            x_center_after_barrier = v_after * t_remaining
            x_max_center = x_center_after_barrier  # Since barrier is at x=0
            x_max_leading = x_max_center + 3*sigx  # Leading edge (3σ)
            
            print(f"\n  Expected propagation:")
            print(f"    Total simulation time: T = {T_total:.2f}")
            print(f"    Time after barrier: {t_remaining:.2f}")
            print(f"    Distance after barrier: Δx = v_avg × t = {x_center_after_barrier:.2f}")
            print(f"\n    Packet center: x_max ≈ {x_max_center:.2f}")
            print(f"    Leading edge (3σ): x_max ≈ {x_max_leading:.2f}")
            print(f"    (neglecting dispersion)")
            
            # Check against domain
            if x_max_leading > x_max:
                print(f"\n  ⚠️  WARNING: Leading edge will reach domain edge!")
                print(f"      Domain: x_max = {x_max:.2f}")
                print(f"      Leading edge enters absorber at t ≈ {((x_max - 3*sigx) / v_after):.2f}")
            elif x_max_center > x_max:
                print(f"\n  ⚠️  WARNING: Packet center will reach domain edge!")
                print(f"      Domain: x_max = {x_max:.2f}")
                print(f"      (But leading edge is already beyond)")
            elif x_max_leading > 0.8 * x_max:
                print(f"\n  ⚠️  CAUTION: Leading edge approaches domain edge (>80%)")
                print(f"      Domain: x_max = {x_max:.2f}")
            else:
                print(f"\n  ✓ Packet will remain within domain (x_max = {x_max:.2f})")
            
            # Number of beating cycles
            if 0 < lambda_beat < np.inf:
                n_cycles_spatial = x_center_after_barrier / lambda_beat
                print(f"\n  Spatial beating cycles:")
                print(f"    Number of λ_beat traversed: {n_cycles_spatial:.2f} cycles")
                if n_cycles_spatial < 2:
                    print(f"    ⚠️  Less than 2 cycles - may be hard to resolve")
                elif n_cycles_spatial > 10:
                    print(f"    ✓ Many cycles - excellent for fitting")
                else:
                    print(f"    ✓ Good number of cycles for analysis")
        else:
            print(f"\n  ⚠️  WARNING: Simulation ends before barrier crossing!")
    else:
        print(f"\n  ⚠️  Cannot predict propagation (one or both channels closed)")
    
    print("="*70)


# ============================================================================
# AUTOMATIC RESULTS DIRECTORY CREATION
# ============================================================================
if not NOLOG:    
    # Determine regime name for folder
    if E_inj_total >= float(V_STEP + E1_R):
        regime_name = "propagative"
    elif E_inj_total >= threshR:
        regime_name = "mixed"
    else:
        regime_name = "evanescent"
    
    # Create unique results directory inside existing results/ folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if SET_KX0_FROM_DETUNING:
        results_dir = os.path.join("results", f"{timestamp}_V{V_STEP:.3f}_Δ{DETUNING_MEV:.3f}_{regime_name}")# True: use detuning, False: use kx0 directly
    else:
        results_dir = os.path.join("results", f"{timestamp}_V{V_STEP:.3f}_k{kx0:.2f}_{regime_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n[RESULTS] Saving all plots to: {results_dir}/")
    print("="*70 + "\n")
    
    # --- Console → .tex inside results_dir ---------------------------------------
    
    class Tee:
        def __init__(self, *streams, autoflush=True):
            self.streams = streams
            self.autoflush = autoflush
        def write(self, data):
            for s in self.streams:
                s.write(data)
                if self.autoflush and "\n" in data:
                    s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    
    # Save original stdout BEFORE redirecting
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    tex_path = os.path.join(results_dir, "console_output.tex")
    f = open(tex_path, "w", encoding="utf-8", buffering=1)
    print(r"\begin{verbatim}", file=f)
    
    def _close_log():
        """Close log file and restore original stdout."""
        if sys.stdout != _original_stdout:
            sys.stdout.flush()
            print(r"\end{verbatim}", file=f)
            f.close()
            sys.stdout = _original_stdout
            sys.stderr = _original_stderr
            print(f"[LOG] Console output saved to: {tex_path}")
    
    # Register cleanup for abnormal exit (Ctrl+C, errors, etc.)
    atexit.register(_close_log)
    
    # Mirror stdout to the file
    sys.stdout = Tee(_original_stdout, f)
    # ------------------------------------------------------------------------------

    
else:
    # NOLOG / batch mode: still define a valid results_dir so the many
    # plt.savefig(os.path.join(results_dir, ...)) calls don't crash. In batch
    # mode we send figures into a per-run subfolder of the batch outdir; the
    # JSON summary (written at the very end) is the actual batch artifact.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if BATCH_MODE:
        results_dir = os.path.join(_BATCH_OUTDIR, f"figs_{RUN_TAG}")
    else:
        results_dir = "."
    os.makedirs(results_dir, exist_ok=True)

# ============================================================================
# ADAPTIVE TRANSMISSION/REFLECTION BOUNDARIES
# ============================================================================
if DO_TRANSMISSION_ANALYTICS:
    
    # LEFT boundary: Fixed distance from barrier (regime-independent)
    TRANSMISSION_X_LEFT = -50.0
    
    # RIGHT boundary: Regime-adaptive
    if is_evanescent:
        if 'L_decay' in globals() and L_decay > 0:
            # Use 2.5 decay lengths (captures ~8% of transmitted amplitude)
            TRANSMISSION_X_RIGHT = max(2.0, min(2.5 * L_decay, 0.5 * ROI_X2))
            signal_pct = 100 * np.exp(-TRANSMISSION_X_RIGHT / L_decay)
            print(f"[T/R] Evanescent: L_decay={L_decay:.2f}, x_right={TRANSMISSION_X_RIGHT:.1f} (~{signal_pct:.1f}% signal)")
        else:
            TRANSMISSION_X_RIGHT = 5.0
            print(f"[T/R] Evanescent: x_right={TRANSMISSION_X_RIGHT:.1f} (fallback)")
    else:
        # Propagative: 60-70% through domain, minimum 50 from barrier
        TRANSMISSION_X_RIGHT = 150
        print(f"[T/R] Propagative: x_right={TRANSMISSION_X_RIGHT:.1f} (far-field)")
    
    # Sanity check
    if TRANSMISSION_X_RIGHT >= 0.85 * x_max:
        print(f"[T/R] WARNING: Right boundary near absorber edge")


# Precompute y-potentials (VY_RIGHT carries the same +delta_y shift as V)
VY_LEFT  = 0.5*m*((2*y0*cp.sqrt(mu/m))**2) * (y[:, None] - y0)**2
VY_RIGHT = 0.5*mu*(y[:, None]**2 - y0**2)**2 + delta_y

#%% ===== PHYSICAL UNIT CONVERSIONS =========

print("\n" + "="*70)
print("PHYSICAL UNIT CONVERSIONS AND KEY PARAMETERS")
print("="*70)

# ========================================================================
# EXPERIMENTAL REFERENCE VALUES
# ========================================================================

# Experimental reference values (SI constants & unit system come from UNIT SYSTEM block)
V_STEP_exp = 0.538e-3 * 1.602e-19      # meV → J (experimental barrier)
J0_exp_theory = 2 * np.pi * 6.34e9     # rad/s (ℏJ0 = 26.22 μeV)


# ========================================================================
# COMPUTE UNIT CONVERSIONS FROM CODE PARAMETERS
# ========================================================================

print(f"\n[CODE PARAMETERS (given)]")
print(f"  hbar = {hbar}")
print(f"  m = {m}")
print(f"  y0 = {y0}")
print(f"  V_STEP = {V_STEP}")
print(f"  mu = {mu:.4e}")
print(f"  x0 = {x0}")
print(f"  σ_x = {sigx}")

print(f"\n[NUMERICAL PARAMETERS (given)]")
print(f"  Nx = {Nx}")
print(f"  Ny = {Ny}")
print(f"  y0 = {y0}")
print(f"  x_min = {x_min}")
print(f"  x_max = {x_max}")
print(f"  dt = {dt}")
print(f"  n_steps = {n_steps}")
print(f"  Total time = {n_steps * dt}")
print(f"  CLAMP_SIGMA_X = {CLAMP_SIGMA_X}")
print(f"  CLAMP_SIGMA_Y = {CLAMP_SIGMA_Y}")
print(f"  ADAPTIVE_SUBSTEPS: {ADAPTIVE_SUBSTEPS}")


# Length unit (L0 defined in UNIT SYSTEM block)
print(f"\n[LENGTH UNIT]")
print(f"  L0 = y0_exp / y0_code = {L0:.4e} m")
print(f"  1 code unit = {L0*1e6:.4f} μm")

# Energy unit (determined by hbar, m, L0)
# With hbar_code = hbar_SI / hbar_unit and m_code = m_SI / m_unit:
# hbar_code * m_code = (hbar_SI / hbar_unit) * (m_SI / m_unit) = 1
# This gives: hbar_unit * m_unit = hbar_SI * m_SI
# Also: E_unit = hbar_unit^2 / (m_unit * L0^2)
# With hbar = m = 1 in code: hbar_unit = m_unit
# So: hbar_unit^2 = hbar_SI * m_SI
# And: E_unit = (hbar_SI * m_SI) / (m_unit * L0^2)

# Actually, simpler: with hbar_code = 1, m_code = 1:
# [hbar^2 / (m * L^2)] must have same value in code and SI
# hbar_code^2 / (m_code * L_code^2) = hbar_SI^2 / (m_SI * L_SI^2)
# 1^2 / (1 * 1^2) = hbar_SI^2 / (m_SI * L0^2)
# So: E0 = hbar_SI^2 / (m_SI * L0^2)   (E0, E0_meV, E0_ueV, E0_GHz from UNIT SYSTEM block)

print(f"\n[ENERGY UNIT (determined by hbar=m=1)]")
print(f"  E0 = {E0:.4e} J")
print(f"  E0 = {E0_meV:.4f} meV")
print(f"  E0 = {E0_GHz:.3f} GHz (as ℏω)")

# Time unit (T0 defined in UNIT SYSTEM block)
print(f"\n[TIME UNIT]")
print(f"  T0 = hbar_SI / E0 = {T0:.4e} s")
print(f"  1 code unit = {T0*1e12:.4f} ps")

# ========================================================================
# CHECK V_STEP AGAINST EXPERIMENT
# ========================================================================

V_STEP_physical = V_STEP * E0
V_STEP_physical_meV = V_STEP_physical / 1.602e-22

print(f"\n" + "="*70)
print("BARRIER HEIGHT CHECK")
print("="*70)

print(f"\n[X-DIRECTION BARRIER (V_STEP)]")
print(f"  Code: V_STEP = {V_STEP:.6f}")
print(f"  Physical: {V_STEP_physical:.4e} J = {V_STEP_physical_meV:.4f} meV")
print(f"  Experimental target: {V_STEP_exp/1.602e-22:.4f} meV")
print(f"  Ratio: {V_STEP_physical/V_STEP_exp:.4f}")

if abs(V_STEP_physical/V_STEP_exp - 1.0) < 0.1:
    print(f"  Status: ✓ Within 10% of experiment")
elif abs(V_STEP_physical/V_STEP_exp - 1.0) < 0.3:
    print(f"  Status: ~ Moderate match")
else:
    print(f"  Status: ⚠ Does not match experiment!")
    V_STEP_needed = V_STEP_exp / E0
    print(f"  → To match 0.538 meV, set V_STEP = {V_STEP_needed:.6f} (not {V_STEP})")



# ========================================================================
# TUNNELING SPLITTING: NUMERICAL vs THEORETICAL
# ========================================================================

print(f"\n" + "="*70)
print("TUNNELING SPLITTING")
print("="*70)

# Numerical J0 (already computed as J0_energy = 0.5 * dE_right)
hbar_J0_numerical_code = J0_energy  # Energy in code units
J0_angular_code = J0_energy  # Angular frequency in code units (same when ℏ=1)

# Convert to physical units
J0_angular_SI = J0_angular_code / T0  # rad/s
f0_numerical_GHz = J0_angular_SI / (2 * np.pi) * 1e-9  # GHz
hbar_J0_numerical_J = hbar_J0_numerical_code * E0  # Joules
hbar_J0_numerical_meV = hbar_J0_numerical_J / 1.602e-22  # meV
hbar_J0_numerical_ueV = hbar_J0_numerical_J / 1.602e-25  # μeV

# Theoretical
J0_angular_theory_SI = J0_exp_theory  # Already in rad/s
f0_theory_GHz = J0_angular_theory_SI / (2 * np.pi) * 1e-9  # GHz
hbar_J0_theory_J = hbar_SI * J0_angular_theory_SI  # Joules
hbar_J0_theory_meV = hbar_J0_theory_J / 1.602e-22  # meV
hbar_J0_theory_ueV = hbar_J0_theory_J / 1.602e-25  # μeV
J0_theory_code = J0_angular_theory_SI * T0  # Convert to code units

print(f"\n[NUMERICAL (from right-well eigenstates)]")
print(f"  E1_R - E0_R = {dE_right:.6e} (code)")
print(f"  ℏJ0 = ½(E1_R - E0_R) = {hbar_J0_numerical_code:.6e} (code)")
print(f"  ℏJ0 = {hbar_J0_numerical_ueV:.3f} μeV = {hbar_J0_numerical_meV:.6f} meV")
print(f"  J0/(2π) = {f0_numerical_GHz:.3f} GHz")

print(f"\n[THEORETICAL (experimental reference)]")
print(f"  J0/(2π) = {f0_theory_GHz:.2f} GHz")
print(f"  ℏJ0 = {hbar_J0_theory_ueV:.2f} μeV = {hbar_J0_theory_meV:.6f} meV")

# Comparison
ratio_J0 = f0_numerical_GHz / f0_theory_GHz
print(f"\n[COMPARISON]")
print(f"  J0_numerical / J0_theory = {ratio_J0:.4f}")
if abs(ratio_J0 - 1.0) < 0.1:
    print(f"  Status: ✓ Within 10% of theory")
elif abs(ratio_J0 - 1.0) < 0.3:
    print(f"  Status: ~ Moderate agreement")
else:
    print(f"  Status: ⚠ Significant deviation")
    print(f"  → Check: mu value, y-grid range/resolution")
    
# ========================================================================
# DOUBLE WELL POTENTIAL PARAMETERS
# ========================================================================

print(f"\n" + "="*70)
print("DOUBLE WELL POTENTIAL")
print("="*70)

# Well separation
y0_physical = y0 * L0
print(f"\n[WELL SEPARATION]")
print(f"  y0 = {y0:.4f} (code) = {y0_physical*1e6:.3f} μm")

# Coupling parameter mu
# Units: [Energy]/[Length]^4
mu_physical = mu * E0 / (L0**4)
mu_physical_SI = mu_physical  # J/m^4
print(f"\n[COUPLING PARAMETER]")
print(f"  mu = {mu:.4e} (code)")
print(f"  mu = {mu_physical:.4e} J/m⁴")

# More intuitive energy scale
mu_meV_per_um4 = mu_physical * 1e24 / 1.602e-22
print(f"  mu = {mu_meV_per_um4:.4e} meV/μm⁴")

# Y-direction barrier
V_barrier_y_code = 0.5 * mu * y0**4
V_barrier_y_physical = V_barrier_y_code * E0
V_barrier_y_meV = V_barrier_y_physical / 1.602e-22

print(f"  Barrier height: {V_barrier_y_meV:.4f} meV")

# Harmonic frequency ω0 = 2 y0 √(mu/m)
# Units: [1/Time]
omega0_physical = omega0 / T0  # rad/s
omega0_physical_GHz = omega0_physical / (2 * np.pi) * 1e-9  # GHz
print(f"\n[HARMONIC FREQUENCY]")
print(f"  ω0 = 2y0√(μ/m) = {omega0:.6f} (code)")
print(f"  ω0 = {omega0_physical:.4e} rad/s")
print(f"  ω0/(2π) = {omega0_physical_GHz:.3f} GHz")
hbar_omega0_meV = omega0 * E0_meV
print(f"  ℏω0 = {hbar_omega0_meV:.4f} meV")

# Well localization widths
sigma_well_physical = sigma_well * L0
print(f"\n[WELL LOCALIZATION WIDTH]")
print(f"  σ_well (numerical) = {sigma_well:.6f} (code) = {sigma_well_physical*1e6:.3f} μm")

# Analytic estimate from harmonic oscillator
sigma_well_analytic = np.sqrt(hbar / (m * omega0))
sigma_well_analytic_physical = sigma_well_analytic * L0
print(f"  σ_well (harmonic approximation)")
print(f"    = √(ℏ/(m×ω0)) = {sigma_well_analytic:.6f} (code) = {sigma_well_analytic_physical*1e6:.3f} μm")

ratio = sigma_well / sigma_well_analytic
print(f"\n  Ratio σ_numerical/σ_analytic = {ratio:.4f}")
print(f"  → Double well ground state is {(1-ratio)*100:.1f}% more localized than harmonic approximation")

# Separation parameter
X_param = y0 / sigma_well_analytic
print(f"\n[DEEP WELL PARAMETER]")
print(f"  X = y0/σ_well = {X_param:.4f} (dimensionless)")



print("="*70)

# ========================================================================
# INJECTION ENERGY AND DETUNING (CORRECTED)
# ========================================================================

print(f"\n" + "="*70)
print("INJECTION ENERGY AND DETUNING")
print("="*70)

# Total injection energy
if 'kx0' in locals() or 'kx0' in globals():
    E_kinetic_x = (hbar * kx0)**2 / (2 * m)
    E_total_inj = E_kinetic_x + E_y_inj
    
    print(f"\n[INJECTION ENERGY COMPONENTS]")
    print(f"  Transverse (E_y_inj) = {E_y_inj:.6f} (code) = {E_y_inj*E0_meV:.4f} meV")
    print(f"  Longitudinal (E_kinetic_x) = {E_kinetic_x:.6f} (code) = {E_kinetic_x*E0_meV:.4f} meV")
    print(f"  Total E_inj = {E_total_inj:.6f} (code) = {E_total_inj*E0_meV:.4f} meV")
else:
    E_total_inj = E_y_inj
    E_kinetic_x = 0.0
    print(f"\n[INJECTION ENERGY]")
    print(f"  E_inj (transverse only) = {E_y_inj:.6f} (code) = {E_y_inj*E0_meV:.4f} meV")
    print(f"  (Note: Longitudinal kx0 not yet defined)")

# Detuning Δ = E_k - V_STEP. (Same value as Sharoglazova et al.; their
# Δ = E - V0 + ℏJ0 uses a different energy reference but gives the same number.)
Delta_code = E_kinetic_x - V_STEP
Delta_J = Delta_code * E0
Delta_meV = Delta_code * E0_meV
Delta_GHz = Delta_code * E0_GHz

print(f"  V_STEP = {V_STEP:.6f} (code) = {V_STEP*E0_meV:.4f} meV")
print(f"  E_kinetic_x = {E_kinetic_x:.6f} (code) = {E_kinetic_x*E0_meV:.4f} meV")

print(f"\n[DETUNING (paper convention: Δ = E_k - V_STEP)]")
print(f"  Δ = E - V_STEP")
print(f"  Δ = {Delta_code:.6f} (code)")
print(f"  Δ = {Delta_J:.4e} J")
print(f"  Δ = {Delta_meV:.6f} meV")
print(f"  Δ = {Delta_GHz:.3f} GHz")

kx0_threshold = np.sqrt(2 * m * V_STEP)
sigk = 1 / (2 * sigx)
print(f"\n[MOMENTUM THRESHOLD]")
print(f"  kx0_threshold = √(2m×V_STEP) = {kx0_threshold:.6f}")
if 'kx0' in locals() or 'kx0' in globals():
    print(f"  kx0_actual = {kx0:.6f}")
    print(f"  Ratio: kx0/kx0_threshold = {kx0/kx0_threshold:.4f}")
    if kx0 < kx0_threshold * 0.95:
        print(f"  → Subcritical momentum (evanescent)")
    elif kx0 < kx0_threshold * 1.05:
        print(f"  → Near-critical momentum")
    else:
        print(f"  → Supercritical momentum (propagative)")
print(f"  σ_k = {sigk}")

# ========================================================================
# SUMMARY TABLE
# ========================================================================

print(f"\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

print(f"\n{'Parameter':<40} {'Code':<15} {'Physical':<30}")
print("="*85)

# Geometry
print(f"{'Well center (y0)':<40} {y0:<15.2f} {y0*L0*1e6:.3f} μm")
print(f"{'X-barrier (V_STEP)':<40} {V_STEP:<15.6f} {V_STEP_physical_meV:.4f} meV")
print(f"{'Y-barrier (well)':<40} {V_barrier_y_code:<15.6f} {V_barrier_y_meV:.4f} meV")

# Right-well states
print(f"{'Ground state (E0_R)':<40} {E0_R:<15.6f} {E0_R*E0_meV:.4f} meV")
print(f"{'First excited (E1_R)':<40} {E1_R:<15.6f} {E1_R*E0_meV:.4f} meV")
print(f"{'Splitting (E1_R - E0_R)':<40} {dE_right:<15.6e} {dE_right*E0_meV:.6f} meV")

# Tunneling
print(f"{'Tunneling splitting (ℏJ0)':<40} {hbar_J0_numerical_code:<15.6e} {hbar_J0_numerical_ueV:.3f} μeV")
f0_numerical_code = J0_angular_code / (2 * np.pi)
print(f"{'Tunneling frequency (J0/2π)':<40} {f0_numerical_code:<15.6e} {f0_numerical_GHz:.2f} GHz (exp: {f0_theory_GHz:.2f} GHz)")

# Injection
print(f"{'Injection energy (E_inj)':<40} {E_total_inj:<15.6f} {E_total_inj*E0_meV:.4f} meV")
print(f"{'Detuning (Δ = E_k - V_STEP)':<40} {Delta_code:<15.6f} {Delta_meV:.6f} meV")

# Thresholds
E_prop_threshold = V_STEP + E0_R
print(f"{'Propagative threshold (V_STEP + E0_R)':<40} {E_prop_threshold:<15.6f} {E_prop_threshold*E0_meV:.4f} meV")

print("="*85)






#%% =========== OPTIMIZED VELOCITY METHODS =======================

# NEW: SPECTRAL DERIVATIVES

def spectral_gradients(psi_field):
    """
    Compute gradients using spectral derivatives (FFT-based).
    Much smoother than finite differences in low-SNR regions.
    Uses global kx, ky wavenumber arrays.
    """
    Psi = cp.fft.fft2(psi_field)
    # Multiply by ik in Fourier space for derivatives
    # kx corresponds to x-direction (axis=1), ky to y-direction (axis=0)
    dpsi_dx = cp.fft.ifft2(1j * kx[None, :] * Psi)
    dpsi_dy = cp.fft.ifft2(1j * ky[:, None] * Psi)
    return dpsi_dx, dpsi_dy

def velocity_field_adaptive(psi_, dx_, dy_):
    """Multi-scale adaptive cutoff method."""
    dpsi_dx = cp.gradient(psi_, dx_, axis=1)
    dpsi_dy = cp.gradient(psi_, dy_, axis=0)
    rho = cp.abs(psi_)**2
    rho_max = float(rho.max())
    
    global_threshold = VELOCITY_GLOBAL_CUT * rho_max
    grad_mag = cp.sqrt(cp.abs(dpsi_dx)**2 + cp.abs(dpsi_dy)**2)
    local_gradient_ratio = grad_mag / (cp.abs(psi_) + VELOCITY_NOISE_FLOOR)
    
    mask = (rho > global_threshold) | ((rho > VELOCITY_NOISE_FLOOR) & 
                                        (local_gradient_ratio < 10 * float(local_gradient_ratio.mean())))
    
    jx = (hbar/m) * cp.imag(cp.conj(psi_) * dpsi_dx)
    jy = (hbar/m) * cp.imag(cp.conj(psi_) * dpsi_dy)
    
    vx = cp.zeros_like(rho)
    vy = cp.zeros_like(rho)
    
    if cp.any(mask):
        rho_safe = cp.maximum(rho[mask], VELOCITY_NOISE_FLOOR)
        vx[mask] = jx[mask] / rho_safe
        vy[mask] = jy[mask] / rho_safe
        
        v_mag = cp.sqrt(vx[mask]**2 + vy[mask]**2)
        if v_mag.size > 10:
            v_p95 = float(cp.percentile(v_mag, 95))
            v_max_allowed = 5.0 * v_p95
            outliers = v_mag > v_max_allowed
            if cp.any(outliers):
                scale = v_max_allowed / v_mag[outliers]
                vx_flat = vx[mask]
                vy_flat = vy[mask]
                vx_flat[outliers] *= scale
                vy_flat[outliers] *= scale
                vx[mask] = vx_flat
                vy[mask] = vy_flat
    
    return vx, vy

def velocity_field_phase(psi_, dx_, dy_):
    """Phase gradient method with row-by-row unwrapping."""
    rho = cp.abs(psi_)**2
    rho_max = float(rho.max())
    mask = (rho > VELOCITY_NOISE_FLOOR) & (rho > VELOCITY_GLOBAL_CUT * rho_max)
    
    vx = cp.zeros_like(rho)
    vy = cp.zeros_like(rho)
    
    if not cp.any(mask):
        return vx, vy
    
    phase = cp.angle(psi_)
    
    if VELOCITY_PHASE_SMOOTH:
        from scipy.ndimage import gaussian_filter
        phase_cpu = cp.asnumpy(phase)
        phase_cpu = gaussian_filter(phase_cpu, sigma=1.0)
        phase = cp.asarray(phase_cpu)
    
    # Unwrap row-by-row
    phase_np = cp.asnumpy(phase)
    for i in range(phase_np.shape[0]):
        if np.any(cp.asnumpy(mask[i, :])):
            phase_np[i, :] = np.unwrap(phase_np[i, :])
    phase = cp.asarray(phase_np)
    
    dphi_dy, dphi_dx = cp.gradient(phase, dy_, dx_, edge_order=2)
    
    # Detect and zero out 2π jumps
    jump_threshold_x = 0.8 * cp.pi / dx_
    jump_threshold_y = 0.8 * cp.pi / dy_
    jumps_x = cp.abs(dphi_dx) > jump_threshold_x
    jumps_y = cp.abs(dphi_dy) > jump_threshold_y
    dphi_dx[jumps_x] = 0.0
    dphi_dy[jumps_y] = 0.0
    
    vx[mask] = (hbar/m) * dphi_dx[mask]
    vy[mask] = (hbar/m) * dphi_dy[mask]
    
    return vx, vy


def velocity_field_current_spectral(psi_):
    """
    Current-based velocity with spectral derivatives.
    Optimal for evanescent regime - no unwrapping, stays on GPU, smooth gradients.
    """
    # Spectral derivatives (smooth!)
    dpsi_dx, dpsi_dy = spectral_gradients(psi_)
    
    rho = cp.abs(psi_)**2
    rho_safe = cp.maximum(rho, VELOCITY_NOISE_FLOOR)
    
    # Current density
    jx = (hbar/m) * cp.imag(cp.conj(psi_) * dpsi_dx)
    jy = (hbar/m) * cp.imag(cp.conj(psi_) * dpsi_dy)
    
    # Velocity
    vx = jx / rho_safe
    vy = jy / rho_safe
    
    # Simple masking based on global cutoff
    rho_max = float(rho.max())
    mask = (rho > VELOCITY_GLOBAL_CUT * rho_max)
    
    vx = cp.where(mask, vx, 0.0)
    vy = cp.where(mask, vy, 0.0)
    
    # Gentle clipping based on percentiles
    v_mag = cp.sqrt(vx*vx + vy*vy)
    v_mag_masked = v_mag[mask]
    if v_mag_masked.size > 10:
        v95 = float(cp.percentile(v_mag_masked, 95))
        vmax = 5.0 * max(v95, 1e-12)
        scale = cp.minimum(1.0, vmax / (v_mag + 1e-30))
        vx *= scale
        vy *= scale
    
    return vx, vy

def velocity_field(psi_):
    """Main velocity field dispatcher for GPU (FFT path)."""
    if VELOCITY_METHOD.upper() == 'ADAPTIVE':
        return velocity_field_adaptive(psi_, float(dx), float(dy))
    elif VELOCITY_METHOD.upper() == 'PHASE':
        return velocity_field_phase(psi_, float(dx), float(dy))
    elif VELOCITY_METHOD.upper() == 'CURRENT_SPECTRAL':
        return velocity_field_current_spectral(psi_)
    else:
        raise ValueError(f"Unknown VELOCITY_METHOD: {VELOCITY_METHOD}")

# ===================== CPU VERSIONS FOR CN PATH =====================

def velocity_field_adaptive_cpu(psi_cpu, dx_, dy_):
    """CPU version of adaptive method."""
    dpsi_dy, dpsi_dx = onp.gradient(psi_cpu, dy_, dx_, edge_order=2)
    rho = onp.abs(psi_cpu)**2
    rho_max = float(rho.max())
    
    global_threshold = VELOCITY_GLOBAL_CUT * rho_max
    grad_mag = onp.sqrt(onp.abs(dpsi_dx)**2 + onp.abs(dpsi_dy)**2)
    local_gradient_ratio = grad_mag / (onp.abs(psi_cpu) + VELOCITY_NOISE_FLOOR)
    
    mask = (rho > global_threshold) | ((rho > VELOCITY_NOISE_FLOOR) & 
                                        (local_gradient_ratio < 10 * float(local_gradient_ratio.mean())))
    
    jx = (hbar/m) * onp.imag(onp.conj(psi_cpu) * dpsi_dx)
    jy = (hbar/m) * onp.imag(onp.conj(psi_cpu) * dpsi_dy)
    
    vx = onp.zeros_like(rho)
    vy = onp.zeros_like(rho)
    
    if onp.any(mask):
        rho_safe = onp.maximum(rho[mask], VELOCITY_NOISE_FLOOR)
        vx[mask] = jx[mask] / rho_safe
        vy[mask] = jy[mask] / rho_safe
        
        v_mag = onp.sqrt(vx[mask]**2 + vy[mask]**2)
        if v_mag.size > 10:
            v_p95 = float(onp.percentile(v_mag, 95))
            v_max_allowed = 5.0 * v_p95
            outliers = v_mag > v_max_allowed
            if onp.any(outliers):
                scale = v_max_allowed / v_mag[outliers]
                vx_flat = vx[mask]
                vy_flat = vy[mask]
                vx_flat[outliers] *= scale
                vy_flat[outliers] *= scale
                vx[mask] = vx_flat
                vy[mask] = vy_flat
    
    return vx, vy

def velocity_field_cpu(psi_cpu, dx_, dy_, x_grid_cpu):
    """Main velocity field dispatcher for CPU (CN path) - simplified to ADAPTIVE only."""
    return velocity_field_adaptive_cpu(psi_cpu, dx_, dy_)

# ===================== Bohmian trajectory helpers (GPU) =====================

def interp2d(xp, yp, grid_x, grid_y, F):
    """Bilinear interpolation with proper floor operation."""
    ix = cp.clip(cp.floor((xp - grid_x[0]) / dx).astype(cp.int32), 0, len(grid_x)-2)
    iy = cp.clip(cp.floor((yp - grid_y[0]) / dy).astype(cp.int32), 0, len(grid_y)-2)
    x1, x2 = grid_x[ix], grid_x[ix+1]
    y1, y2 = grid_y[iy], grid_y[iy+1]
    tx = (xp - x1) / (x2 - x1 + 1e-300)
    ty = (yp - y1) / (y2 - y1 + 1e-300)
    f00 = F[iy, ix]; f10 = F[iy, ix+1]
    f01 = F[iy+1, ix]; f11 = F[iy+1, ix+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def rk2_step_bohm_once(xp, yp, vx, vy, dt_sub):
    """Single RK2 step.

    Both stages use the SAME velocity field (the one built from psi after the
    current split-step update), so the scheme is midpoint-accurate in space but
    first order in time. Adequate here because the field varies on the wavepacket
    timescale, which is long compared with dt.
    """
    v1x = interp2d(xp, yp, x, y, vx)
    v1y = interp2d(xp, yp, x, y, vy)
    xm, ym = xp + 0.5*dt_sub*v1x, yp + 0.5*dt_sub*v1y
    v2x = interp2d(xm, ym, x, y, vx)
    v2y = interp2d(xm, ym, x, y, vy)
    xn = cp.clip(xp + dt_sub*v2x, x[0], x[-1])
    yn = cp.clip(yp + dt_sub*v2y, y[0], y[-1])
    return xn, yn

def _compute_global_nsub_curvature(xp, yp, vx, vy, dt_):
    """
    NEW: Enhanced adaptive substepping with CFL + curvature detection.
    """
    vxp = interp2d(xp, yp, x, y, vx)
    vyp = interp2d(xp, yp, x, y, vy)
    speed = cp.sqrt(vxp*vxp + vyp*vyp) + 1e-30
    
    # CFL criterion
    ds_allow = float(SUBSTEP_CFL_FRAC) * float(min(dx, dy))
    n_cfl = cp.ceil((speed * dt_) / max(ds_allow, 1e-15))
    
    # Curvature criterion: check velocity change at midpoint
    xm = xp + 0.5*dt_*vxp
    ym = yp + 0.5*dt_*vyp
    vxm = interp2d(xm, ym, x, y, vx)
    vym = interp2d(xm, ym, x, y, vy)
    bend = cp.sqrt((vxm - vxp)**2 + (vym - vyp)**2) / speed
    n_curv = cp.ceil(1 + SUBSTEP_CURVATURE_FRAC * bend)
    
    # Take maximum of both criteria
    n = cp.clip(cp.maximum(n_cfl, n_curv), SUBSTEP_MIN, SUBSTEP_MAX)
    
    # Extra refinement near barrier
    if float(cp.any(cp.abs(xp - STEP_CENTER) < SUBSTEP_NEAR_STEP_X)):
        nsub = max(int(n.max()), int(np.ceil(int(n.max()) * SUBSTEP_NEAR_STEP_MULT)))
        nsub = min(nsub, SUBSTEP_MAX)
    else:
        nsub = int(n.max())
    
    return nsub

def rk2_step_bohm_adaptive(xp, yp, vx, vy, dt_):
    """RK2 with adaptive substepping (CFL + curvature aware)."""
    if not ADAPTIVE_SUBSTEPS:
        return rk2_step_bohm_once(xp, yp, vx, vy, dt_)
    
    nsub = _compute_global_nsub_curvature(xp, yp, vx, vy, dt_)
    dt_sub = dt_ / float(nsub)
    for _ in range(nsub):
        xp, yp = rk2_step_bohm_once(xp, yp, vx, vy, dt_sub)
    return xp, yp

# ===== CPU trajectory interpolation =====
if HAS_NUMBA:
    @njit
    def _interp2d_numba_kernel(xp, yp, grid_x0, grid_y0, dx_, dy_, F):
        n_traj = xp.size
        Nx = F.shape[1]
        Ny = F.shape[0]
        vx_out = onp.zeros(n_traj)
        for i in range(n_traj):
            ix_f = (xp[i] - grid_x0) / dx_
            iy_f = (yp[i] - grid_y0) / dy_
            ix_i = int(ix_f)
            iy_i = int(iy_f)
            ix_i = max(0, min(ix_i, Nx - 2))
            iy_i = max(0, min(iy_i, Ny - 2))
            
            x1 = grid_x0 + ix_i * dx_
            x2 = grid_x0 + (ix_i + 1) * dx_
            y1 = grid_y0 + iy_i * dy_
            y2 = grid_y0 + (iy_i + 1) * dy_
            
            tx = (xp[i] - x1) / (x2 - x1 + 1e-300)
            ty = (yp[i] - y1) / (y2 - y1 + 1e-300)
            
            f00 = F[iy_i, ix_i]
            f10 = F[iy_i, ix_i + 1]
            f01 = F[iy_i + 1, ix_i]
            f11 = F[iy_i + 1, ix_i + 1]
            
            vx_out[i] = (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11
        return vx_out

    def interp2d_cpu(xp, yp, grid_x, grid_y, F, dx_, dy_):
        grid_x0 = float(grid_x[0])
        grid_y0 = float(grid_y[0])
        return _interp2d_numba_kernel(xp, yp, grid_x0, grid_y0, dx_, dy_, F)

else:
    def interp2d_cpu(xp, yp, grid_x, grid_y, F, dx_, dy_):
        ix = onp.clip(onp.floor((xp - grid_x[0]) / dx_).astype(int), 0, len(grid_x)-2)
        iy = onp.clip(onp.floor((yp - grid_y[0]) / dy_).astype(int), 0, len(grid_y)-2)
        x1, x2 = grid_x[ix], grid_x[ix+1]
        y1, y2 = grid_y[iy], grid_y[iy+1]
        tx = (xp - x1) / (x2 - x1 + 1e-300)
        ty = (yp - y1) / (y2 - y1 + 1e-300)
        f00 = F[iy, ix]; f10 = F[iy, ix+1]
        f01 = F[iy+1, ix]; f11 = F[iy+1, ix+1]
        return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11
    
def compute_rho_a_heatmap(traj_hist_X_dict, traj_hist_Y_dict, traj_times_dict,
                          roi_x1, roi_x2, y_threshold, 
                          spatial_bin_width=0.1, time_stride_steps=50, dt_sim=0.01):
    """
    Compute ρ_a(t, x): fraction of trajectories in lower waveguide (y < -threshold)
    at each (time, position) bin.
    
    Parameters:
    -----------
    traj_hist_X_dict, traj_hist_Y_dict, traj_times_dict : dict
        Trajectory storage dictionaries from ROI
    roi_x1, roi_x2 : float
        ROI bounds in x
    y_threshold : float
        Lower waveguide defined as y < -threshold
    spatial_bin_width : float
        Width of spatial bins
    time_stride_steps : int
        Analysis time resolution in simulation steps
    dt_sim : float
        Simulation time step
        
    Returns:
    --------
    t_grid : np.ndarray
        Time points (1D array)
    x_grid : np.ndarray
        Spatial bin centers (1D array)
    rho_a : np.ndarray
        2D array (len(x_grid), len(t_grid)) with ρ_a values
    t_start : float
        Start time (first barrier crossing)
    """
    
    if len(traj_hist_X_dict) == 0:
        print("[RHO_A] No trajectories in ROI, cannot compute ρ_a")
        return None, None, None, None
    
    # Find t_start: earliest time when any trajectory crosses x=0
    t_start = np.inf
    for traj_idx in traj_hist_X_dict.keys():
        times = np.array(traj_times_dict[traj_idx])
        if len(times) > 0:
            t_start = min(t_start, times[0])
    
    if np.isinf(t_start):
        print("[RHO_A] No valid trajectory times found")
        return None, None, None, None
    
    print(f"[RHO_A] First barrier crossing at t = {t_start:.2f}")
    
    # Find t_end: latest time in any trajectory
    t_end = 0.0
    for traj_idx in traj_hist_X_dict.keys():
        times = np.array(traj_times_dict[traj_idx])
        if len(times) > 0:
            t_end = max(t_end, times[-1])
    
    # Create time grid (analysis resolution)
    dt_analysis = time_stride_steps * dt_sim
    n_time_bins = int(np.ceil((t_end - t_start) / dt_analysis))
    t_grid = t_start + np.arange(n_time_bins) * dt_analysis
    
    # Create spatial grid
    n_spatial_bins = int(np.ceil((roi_x2 - roi_x1) / spatial_bin_width))
    x_edges = np.linspace(roi_x1, roi_x2, n_spatial_bins + 1)
    x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])  # Bin centers
    
    print(f"[RHO_A] Time grid: {len(t_grid)} points from t={t_start:.2f} to t={t_end:.2f}")
    print(f"[RHO_A] Spatial grid: {len(x_grid)} bins from x={roi_x1:.1f} to x={roi_x2:.1f}")
    print(f"[RHO_A] Total grid size: {len(x_grid)} × {len(t_grid)} = {len(x_grid)*len(t_grid)} cells")
    
    # Initialize counters
    rho_a = np.zeros((len(x_grid), len(t_grid)))  # ρ_a values
    N_total = np.zeros((len(x_grid), len(t_grid)), dtype=int)  # Total count
    N_lower = np.zeros((len(x_grid), len(t_grid)), dtype=int)  # Lower waveguide count
    
    # Process each trajectory
    n_traj_processed = 0
    for traj_idx in traj_hist_X_dict.keys():
        x_traj = np.array(traj_hist_X_dict[traj_idx])
        y_traj = np.array(traj_hist_Y_dict[traj_idx])
        t_traj = np.array(traj_times_dict[traj_idx])
        
        if len(x_traj) < 2:
            continue  # Need at least 2 points for interpolation
        
        # Only consider times >= t_start
        valid_mask = t_traj >= t_start
        if not np.any(valid_mask):
            continue
        
        x_traj = x_traj[valid_mask]
        y_traj = y_traj[valid_mask]
        t_traj = t_traj[valid_mask]
        
        # For each analysis time point, interpolate trajectory position
        for j, t_query in enumerate(t_grid):
            # Check if trajectory exists at this time
            if t_query < t_traj[0] or t_query > t_traj[-1]:
                continue
            
            # Linear interpolation
            idx_after = np.searchsorted(t_traj, t_query)
            
            if idx_after == 0:
                # Exact match at first point
                x_interp = x_traj[0]
                y_interp = y_traj[0]
            elif idx_after >= len(t_traj):
                # Exact match at last point
                x_interp = x_traj[-1]
                y_interp = y_traj[-1]
            else:
                # Interpolate between idx_after-1 and idx_after
                t_before = t_traj[idx_after - 1]
                t_after = t_traj[idx_after]
                alpha = (t_query - t_before) / (t_after - t_before)
                
                x_interp = x_traj[idx_after - 1] + alpha * (x_traj[idx_after] - x_traj[idx_after - 1])
                y_interp = y_traj[idx_after - 1] + alpha * (y_traj[idx_after] - y_traj[idx_after - 1])
            
            # Check if particle is within ROI
            if x_interp < roi_x1 or x_interp > roi_x2:
                continue
            
            # Find spatial bin
            i_bin = np.searchsorted(x_edges, x_interp) - 1
            i_bin = np.clip(i_bin, 0, len(x_grid) - 1)
            
            # Update counts
            N_total[i_bin, j] += 1
            if y_interp < -y_threshold:
                N_lower[i_bin, j] += 1
        
        n_traj_processed += 1
    
    print(f"[RHO_A] Processed {n_traj_processed} trajectories")
    
    # Compute ρ_a
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_a = N_lower / N_total
        rho_a[N_total == 0] = 0.0  # Empty bins set to 0
    
    # Statistics
    populated_bins = N_total > 0
    if np.any(populated_bins):
        avg_occupancy = np.mean(N_total[populated_bins])
        max_occupancy = np.max(N_total)
        pct_populated = 100 * np.sum(populated_bins) / populated_bins.size
        print(f"[RHO_A] Bin occupancy: avg={avg_occupancy:.1f}, max={max_occupancy}, "
              f"populated={pct_populated:.1f}%")
    
    return t_grid, x_grid, rho_a, t_start

# %% =========== Trajectory sampling =====================

if COMPUTE_TRAJECTORIES:
    # Set random seed for reproducibility
    np.random.seed(42)
    print("[TRAJECTORY SAMPLING] Random seed set to 42 for reproducibility")

    
    def sample_from_pdf_1d(grid_cpu, pdf_cpu, n_samples):
        pdf = np.clip(pdf_cpu, 0, None)
        area = np.trapezoid(pdf, grid_cpu) + 1e-300
        pdf /= area
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        u = np.random.rand(n_samples)
        return np.interp(u, cdf, grid_cpu)
    
    def sample_from_pdf_band(grid_cpu, pdf_cpu, xlo, xhi, n_samples):
        m = (grid_cpu >= xlo) & (grid_cpu <= xhi)
        if not np.any(m):
            return np.random.uniform(xlo, xhi, size=n_samples)
        g = grid_cpu[m]; p = np.clip(pdf_cpu[m], 0, None)
        area = np.trapezoid(p, g)
        if area < 1e-12:
            return np.random.uniform(xlo, xhi, size=n_samples)
        p = p / area
        cdf = np.cumsum(p); cdf /= cdf[-1]
        u = np.random.rand(n_samples)
        return np.interp(u, cdf, g)
    
    if INJECT_PLANE:
        rng_x = np.random.uniform(float(inj_x0) + 2, float(inj_x1), size=n_traj)
        y_pdf = cp.asnumpy(cp.abs(phi_y_L)**2)
        y_pool = sample_from_pdf_1d(cp.asnumpy(y), y_pdf, n_traj*3)
        if not LEFT_GROUND_NUMERIC:
            sigy_L_est = float(cp.asnumpy(cp.sqrt(hbar/(m*omega0))))
        else:
            sigy_L_est = float(np.sqrt(hbar/(m*omega0)))
        y_band = CLAMP_SIGMA_Y * sigy_L_est
        keep   = np.abs(y_pool - y0) <= y_band
        y_pool = y_pool[keep]
        if SAMPLE_Y_LOWER_HALF:
            y_pool = y_pool[y_pool <= y0]
        if y_pool.size < n_traj:
            need   = n_traj - y_pool.size
            y_low  = max(float(y_min), y0 - y_band) if CLAMP_OUTLIERS else float(y_min)
            y_high = y0 if SAMPLE_Y_LOWER_HALF else (y0 + y_band if CLAMP_OUTLIERS else float(y_max))
            y_extra = np.random.uniform(y_low, y_high, size=need)
            y_pool  = np.concatenate([y_pool, y_extra])
        trajX = cp.asarray(rng_x[:n_traj])
        trajY = cp.asarray(y_pool[:n_traj])
    
    else:
        phi_x = gauss(x, x0, kx0, sigx)
        x_cpu = cp.asnumpy(x)
        px_cpu = cp.asnumpy(cp.abs(phi_x)**2)
    
        if X_SEED_FORWARD_BAND:
            x_lo = float(x0 + X_SEED_A_SIG*sigx)
            x_hi = float(x0 + X_SEED_B_SIG*sigx)
            x_pool = sample_from_pdf_band(x_cpu, px_cpu, x_lo, x_hi, n_traj*2)
        else:
            x_pool = sample_from_pdf_1d(x_cpu, px_cpu, n_traj*3)
    
        y_pdf = cp.asnumpy(cp.abs(phi_y_L)**2)
        y_pool = sample_from_pdf_1d(cp.asnumpy(y), y_pdf, n_traj*3)
    
        if CLAMP_OUTLIERS and not X_SEED_FORWARD_BAND:
            x_band = CLAMP_SIGMA_X * sigx
        else:
            x_band = None
        y_band = CLAMP_SIGMA_Y * (float(cp.asnumpy(cp.sqrt(hbar/(m*omega0)))))
    
        if CLAMP_OUTLIERS and not X_SEED_FORWARD_BAND:
            keep_x = np.abs(x_pool - x0) <= x_band
            x_pool = x_pool[keep_x]
        keep_y = np.abs(y_pool - y0) <= y_band
        y_pool = y_pool[keep_y]
    
        if SAMPLE_Y_LOWER_HALF:
            y_pool = y_pool[y_pool <= y0]
    
        need = max(0, n_traj - min(x_pool.size, y_pool.size))
        if need > 0:
            if X_SEED_FORWARD_BAND:
                x_extra = np.random.uniform(float(x0 + X_SEED_A_SIG*sigx),
                                            float(x0 + X_SEED_B_SIG*sigx),
                                            size=need)
            else:
                x_extra = sample_from_pdf_1d(x_cpu, px_cpu, need)
                if CLAMP_OUTLIERS:
                    x_extra = x_extra[np.abs(x_extra - x0) <= x_band]
                    if x_extra.size < need:
                        x_pad = np.random.uniform(x0 - x_band, x0 + x_band, size=need - x_extra.size)
                        x_extra = np.concatenate([x_extra, x_pad])
    
            y_low  = max(float(y_min), y0 - y_band) if CLAMP_OUTLIERS else float(y_min)
            y_high = y0 if SAMPLE_Y_LOWER_HALF else (y0 + y_band if CLAMP_OUTLIERS else float(y_max))
            y_extra = np.random.uniform(y_low, y_high, size=need)
    
            x_pool = np.concatenate([x_pool, x_extra])
            y_pool = np.concatenate([y_pool, y_extra])
    
        trajX = cp.asarray(x_pool[:n_traj])
        trajY = cp.asarray(y_pool[:n_traj])
    

else:
    # Skip trajectory sampling
    print("[TRAJECTORY SAMPLING] Skipped (COMPUTE_TRAJECTORIES=False)")
    n_traj_original = n_traj
    n_traj = 0
    if PROPAGATOR.upper() == 'FFT':
        trajX = cp.array([], dtype=cp.float64)
        trajY = cp.array([], dtype=cp.float64)
    else:
        trajX = onp.array([], dtype=onp.float64)
        trajY = onp.array([], dtype=onp.float64)

# ---- CN precompute containers (only if used) ----
if PROPAGATOR.upper() == 'CN_ADI':
    print("\n[CN_ADI] Initializing sparse solvers...")
    print("[CN_ADI] Note: CN_ADI is inherently slow (~1.5 it/s is expected for this grid size)")
    print("[CN_ADI] FFT method is 100-200x faster. Consider using PROPAGATOR='FFT' instead.")
    
    x_np = onp.linspace(float(x_min), float(x_max), int(Nx))
    y_np = onp.linspace(float(y_min), float(y_max), int(Ny))
    dx_np = (x_np[-1] - x_np[0]) / (len(x_np) - 1)
    dy_np = (y_np[-1] - y_np[0]) / (len(y_np) - 1)

    V_np = onp.asarray(cp.asnumpy(V), dtype=onp.complex128)
    Vhalf_np = onp.exp(-0.5j * dt * V_np / hbar)
    mask_np = onp.asarray(cp.asnumpy(mask), dtype=onp.float64)
    psi_np = onp.asarray(cp.asnumpy(psi), dtype=onp.complex128)

    if INJECT_PLANE:
        S0_np = onp.asarray(cp.asnumpy(S0), dtype=onp.complex128)
        omega_inj_np = 0.5 * float(kx0**2) + float(E_y_inj)
        phase_array = onp.exp(-1j * omega_inj_np * (onp.arange(n_steps + 1) * dt))
    else:
        phase_array = None

    Nx_, Ny_ = int(Nx), int(Ny)
    Lx_main = -2.0/dx_np**2 * onp.ones(Nx_)
    Lx_off  =  1.0/dx_np**2 * onp.ones(Nx_-1)
    Ly_main = -2.0/dy_np**2 * onp.ones(Ny_)
    Ly_off  =  1.0/dy_np**2 * onp.ones(Ny_-1)

    Ix = diags([onp.ones(Nx_)], [0], shape=(Nx_, Nx_)).tocsc()
    Iy = diags([onp.ones(Ny_)], [0], shape=(Ny_, Ny_)).tocsc()
    Lx = diags([Lx_off, Lx_main, Lx_off], [-1, 0, 1], shape=(Nx_, Nx_)).tocsc()
    Ly = diags([Ly_off, Ly_main, Ly_off], [-1, 0, 1], shape=(Ny_, Ny_)).tocsc()

    alpha = 1j * hbar * dt / (2.0 * m)
    Ax = (Ix - alpha * Lx).tocsc()
    Bx = (Ix + alpha * Lx).tocsc()
    Ay = (Iy - alpha * Ly).tocsc()
    By = (Iy + alpha * Ly).tocsc()

    print("[CN_ADI] Computing LU factorizations (this takes a moment)...")
    LUx = splu(Ax)
    LUy = splu(Ay)
    print("[CN_ADI] LU factorizations complete. Starting evolution...")

    # Convert trajectories to numpy for CN_ADI
    trajX = onp.asarray(cp.asnumpy(trajX))
    trajY = onp.asarray(cp.asnumpy(trajY))

#%% =========== Storage & Live Animation =====================
print("[STORAGE] Initializing adaptive storage system...")

# Bulk storage (coarse resolution for all trajectories)
# Note: This stores ALL trajectories at BULK_HIST_STRIDE for:
#   1. Mean trajectory position (DO_MEAN_TEST)
#   2. Full trajectory plotting (FULL_TRAJ_PLOT)
#   3. Deep penetration statistics
# Memory usage: ~2-3 MB for n_traj=1000, T_store=160 (negligible)
# To disable bulk storage if not needed, set STORE_BULK_TRAJECTORIES=False
if COMPUTE_TRAJECTORIES:
    STORE_BULK_TRAJECTORIES = DO_MEAN_TEST or FULL_TRAJ_PLOT
else:
    STORE_BULK_TRAJECTORIES = False

T_store_bulk = n_steps//BULK_HIST_STRIDE + 1
store_i_bulk = 1

if STORE_BULK_TRAJECTORIES:
    if PROPAGATOR.upper() == 'FFT':
        traj_hist_X_bulk = cp.full((T_store_bulk, n_traj), cp.nan, dtype=cp.float32)
        traj_hist_Y_bulk = cp.full((T_store_bulk, n_traj), cp.nan, dtype=cp.float32)
        traj_hist_X_bulk[0, :] = trajX.astype(cp.float32)
        traj_hist_Y_bulk[0, :] = trajY.astype(cp.float32)
        traj_entered_roi = cp.zeros(n_traj, dtype=bool)
    else:
        traj_hist_X_bulk = onp.full((T_store_bulk, n_traj), onp.nan, dtype=onp.float32)
        traj_hist_Y_bulk = onp.full((T_store_bulk, n_traj), onp.nan, dtype=onp.float32)
        traj_hist_X_bulk[0, :] = trajX.astype(onp.float32)
        traj_hist_Y_bulk[0, :] = trajY.astype(onp.float32)
        traj_entered_roi = onp.zeros(n_traj, dtype=bool)
    
    print(f"[STORAGE] Bulk trajectory storage: {T_store_bulk} timesteps × {n_traj} trajectories")
    print(f"[STORAGE] Memory per array: ~{T_store_bulk * n_traj * 4 / 1024**2:.2f} MB")
else:
    if PROPAGATOR.upper() == 'FFT':
        traj_entered_roi = cp.zeros(n_traj, dtype=bool)
    else:
        traj_entered_roi = onp.zeros(n_traj, dtype=bool)
    print("[STORAGE] Bulk trajectory storage DISABLED (saving memory)")

# High-res storage for ROI (only trajectories that enter ROI)
# This is efficient: typically only 10-30% of trajectories enter ROI
traj_hist_X_roi = {}
traj_hist_Y_roi = {}
traj_times_roi = {}
traj_hist_VX_roi = {} #initialize velocity storage
traj_hist_VY_roi = {}
print(f"[STORAGE] ROI trajectory storage: High-resolution (stride={ROI_HIST_STRIDE}), selective")

# --- Wavefunction-based dwell-time accumulator: tau = ∫ P(x>0, t) dt ---
# Exact Buettiker check; must equal the all-trajectory mean dwell time.
# Disabled for INJECT_PLANE (the additive source breaks the unit-norm assumption).
if DO_DWELL_TIME and INJECT_PLANE:
    print("[DWELL] INJECT_PLANE=True -> wavefunction dwell integral disabled "
          "(norm not unit/conserved)")
    DO_DWELL_TIME = False
if DO_DWELL_TIME:
    _x_grid_cpu_dwell = np.linspace(float(x_min), float(x_max), Nx)
    ix_x0_dwell = int(np.searchsorted(_x_grid_cpu_dwell, 0.0))
    P_barrier_int = 0.0
    print(f"[DWELL] Wavefunction dwell integral enabled "
          f"(sampled every {DWELL_SAMPLE_INTERVAL} steps, x>0 from index {ix_x0_dwell})")

# Time-averaged density with FIXED barrier detection
if DO_TIME_AVG_DENSITY:
    x_cpu_full = np.linspace(float(x_min), float(x_max), Nx)
    ix_roi1 = int(np.searchsorted(x_cpu_full, ROI_X1))
    ix_roi2 = int(np.searchsorted(x_cpu_full, ROI_X2))
    ix_roi1 = max(0, min(ix_roi1, Nx-1))
    ix_roi2 = max(ix_roi1+1, min(ix_roi2, Nx))
    
    if PROPAGATOR.upper() == 'FFT':
        density_sum_roi = cp.zeros((Ny, ix_roi2 - ix_roi1), dtype=cp.float64)
    else:
        density_sum_roi = onp.zeros((Ny, ix_roi2 - ix_roi1), dtype=onp.float64)
    
    density_count_roi = 0
    barrier_hit_step = None
    averaging_start_step = None
    averaging_end_step = None
    averaging_started = False
    averaging_window_committed = False
    historical_peak = 0.0  # Track historical peak properly
    barrier_check_startup_steps = 100  # Don't check for barrier hit in first N steps
    barrier_min_absolute_density = 1e-3  # Minimum absolute density to consider "significant"
    
    print(f"[TIME_AVG] Window={TIME_AVG_WINDOW} time units")
    print(f"[TIME_AVG] Will start when {BARRIER_HIT_THRESHOLD*100:.0f}% of peak density reaches x=0")


# Enhanced Transmission/Reflection analytics
if DO_TRANSMISSION_ANALYTICS:
    x_cpu_full = np.linspace(float(x_min), float(x_max), Nx)
    ix_left = int(np.searchsorted(x_cpu_full, TRANSMISSION_X_LEFT))
    ix_right = int(np.searchsorted(x_cpu_full, TRANSMISSION_X_RIGHT))
    ix_barrier = int(np.searchsorted(x_cpu_full, STEP_CENTER))
    
    # Flux-based calculation storage
    flux_incident_history = []
    flux_transmitted_history = []
    flux_reflected_history = []
    
    # Traditional probability-based for comparison
    T_history = []  # Transmission coefficient
    R_history = []  # Reflection coefficient
    t_history = []  # Time points
    
    # Trajectory-based T/R
    traj_transmitted = 0
    traj_reflected = 0
    traj_counted = set()
    
    print("[TRANSMISSION] Enhanced T/R tracking with flux method:")
    print(f"               Left (incident/reflected): x = {TRANSMISSION_X_LEFT}")
    print(f"               Right (transmitted): x = {TRANSMISSION_X_RIGHT}")
    print(f"               Barrier: x = {STEP_CENTER}")
    print("               Note: Trajectory T/R will be computed from final positions")

if DO_MEAN_TEST:
    mean_x_psi = np.zeros(T_store_bulk)
    mean_x_trj = np.zeros(T_store_bulk)
    if PROPAGATOR.upper() == 'FFT':
        rho0 = cp.abs(psi)**2
        rho_x0 = cp.sum(rho0, axis=0)
        mean_x_psi[0] = float(cp.dot(rho_x0, x)) / float(cp.sum(rho_x0)) if float(cp.sum(rho_x0)) > 0 else np.nan
    else:
        rho0 = onp.abs(psi_np)**2
        rho_x0 = onp.sum(rho0, axis=0)
        mean_x_psi[0] = float(onp.dot(rho_x0, x_np)) / float(onp.sum(rho_x0)) if float(onp.sum(rho_x0)) > 0 else np.nan
    mean_x_trj[0] = float(onp.mean(trajX)) if PROPAGATOR.upper() == 'CN_ADI' else float(cp.mean(trajX))

if DO_RIGHT_IMBALANCE:
    x_cpu_full = np.linspace(float(x_min), float(x_max), Nx)
    ix1 = int(np.searchsorted(x_cpu_full, ROI_X1))
    ix2 = int(np.searchsorted(x_cpu_full, ROI_X2))
    ix1 = max(0, min(ix1, Nx-1)); ix2 = max(ix1+1, min(ix2, Nx))
    PR = np.zeros(T_store_bulk); PL = np.zeros(T_store_bulk); D = np.zeros(T_store_bulk)
    if PROPAGATOR.upper() == 'FFT':
        rho0 = cp.abs(psi)**2
        slab = rho0[:, ix1:ix2]
        rho_y_roi = cp.sum(slab, axis=1) * float(dx)
        y_cpu_plot = np.linspace(float(y_min), float(y_max), Ny)
        split = int(np.searchsorted(y_cpu_plot, y_threshold_physical))
        PR[0] = float(cp.sum(rho_y_roi[split:]) * float(dy))
        PL[0] = float(cp.sum(rho_y_roi[:split]) * float(dy))
    else:
        rho0 = onp.abs(psi_np)**2
        slab = rho0[:, ix1:ix2]
        rho_y_roi = onp.sum(slab, axis=1) * float(dx_np)
        y_cpu_plot = onp.linspace(float(y_min), float(y_max), Ny)
        split = int(np.searchsorted(y_cpu_plot, y_threshold_physical))
        PR[0] = float(onp.sum(rho_y_roi[split:]) * float(dy_np))
        PL[0] = float(onp.sum(rho_y_roi[:split]) * float(dy_np))
    D[0]  = (PR[0]-PL[0])/(PR[0]+PL[0]) if (PR[0]+PL[0])>0 else 0.0

if LIVE_ANIM:
    plt.ion()
    fig, ax = plt.subplots(figsize=LIVE_ANIM_FIG_SIZE)
    if PROPAGATOR.upper() == 'FFT':
        cp.cuda.runtime.deviceSynchronize()
        dens_cpu = cp.asnumpy(cp.abs(psi)**2)
        trajX_np = cp.asnumpy(trajX); trajY_np = cp.asnumpy(trajY)
    else:
        dens_cpu = onp.abs(psi_np)**2
        trajX_np = trajX; trajY_np = trajY
    im = ax.imshow(dens_cpu, extent=[float(x_min), float(x_max), float(y_min), float(y_max)],
                   origin="lower", cmap="inferno", aspect="auto")
    ax.axvline(ROI_X1, color='c', ls='--', lw=1)
    ax.axvline(ROI_X2, color='c', ls='--', lw=1, label="ROI")
    traj_plot = ax.plot(trajX_np, trajY_np, 'wo', ms=1)[0]
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Live |ψ|² + Bohmian ({VELOCITY_METHOD}) - t=0.00")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    fig.show()  # Explicitly show the window
    fig.canvas.draw()  # Force drawing
    fig.canvas.flush_events()  # Process events
    print("[ANIMATION] Live animation window should be visible")
else:
    # Just disable interactivity during simulation - DON'T switch backend
    plt.ioff()
    print("[ANIMATION] Interactive plotting disabled during simulation")

# Initialize video writer if requested
video_writer = None
video_fig = None
video_ax = None
video_im = None
video_traj_plot = None
video_filename_final = None

if SAVE_VIDEO:
    if not HAS_IMAGEIO:
        print("[VIDEO] Cannot save video: imageio not installed")
        print("[VIDEO] Install with: pip install imageio[ffmpeg]")
        SAVE_VIDEO = False
    else:
        # Test if we can create a video writer (checks for ffmpeg backend)
        try:
            # Try to create a dummy writer to test
            test_writer = imageio.get_writer('test_dummy.mp4', fps=30, codec='libx264')
            test_writer.close()
            if os.path.exists('test_dummy.mp4'):
                os.remove('test_dummy.mp4')
            HAS_VIDEO_BACKEND = True
        except Exception as e:
            print("[VIDEO] Cannot save video: ffmpeg backend not available")
            print("[VIDEO] Install with: pip install imageio[ffmpeg]")
            print("[VIDEO] Or run: python -m imageio_download_bin ffmpeg")
            print(f"[VIDEO] Error details: {e}")
            SAVE_VIDEO = False
            HAS_VIDEO_BACKEND = False

if SAVE_VIDEO:
    # Save video in the results directory (no counter needed - unique dir!)
    video_filename_final = os.path.join(results_dir, VIDEO_FILENAME)
    
    print(f"[VIDEO] Initializing video: {video_filename_final}")
    print(f"[VIDEO] Resolution: {VIDEO_FIG_SIZE[0]}×{VIDEO_FIG_SIZE[1]} inches at {VIDEO_DPI} DPI")
    print(f"[VIDEO] Output size: {int(VIDEO_FIG_SIZE[0]*VIDEO_DPI)}×{int(VIDEO_FIG_SIZE[1]*VIDEO_DPI)} px")
    print(f"[VIDEO] Frame rate: {VIDEO_FPS} fps, saving every {VIDEO_STRIDE} steps")
    
    # Create separate figure for video using Agg backend (no window)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    video_fig = Figure(figsize=VIDEO_FIG_SIZE, dpi=VIDEO_DPI)
    video_canvas = FigureCanvasAgg(video_fig)
    video_ax = video_fig.add_subplot(111)
    
    if PROPAGATOR.upper() == 'FFT':
        cp.cuda.runtime.deviceSynchronize()
        dens_cpu = cp.asnumpy(cp.abs(psi)**2)
        trajX_np = cp.asnumpy(trajX); trajY_np = cp.asnumpy(trajY)
    else:
        dens_cpu = onp.abs(psi_np)**2
        trajX_np = trajX; trajY_np = trajY
    
    video_im = video_ax.imshow(dens_cpu, extent=[float(x_min), float(x_max), float(y_min), float(y_max)],
                                origin="lower", cmap="inferno", aspect="auto")
    video_ax.axvline(ROI_X1, color='c', ls='--', lw=1)
    video_ax.axvline(ROI_X2, color='c', ls='--', lw=1, label="ROI")
    video_traj_plot = video_ax.plot(trajX_np, trajY_np, 'wo', ms=0.8)[0]
    video_fig.colorbar(video_im, ax=video_ax)
    video_ax.set_title(f"|ψ|² + Bohmian trajectories ({VELOCITY_METHOD}) - t=0.00")
    video_ax.set_xlabel('x')
    video_ax.set_ylabel('y')
    video_ax.legend(loc='upper right', fontsize=8)
    video_fig.tight_layout()
    
    # Initialize video writer
    try:
        video_writer = imageio.get_writer(video_filename_final, fps=VIDEO_FPS, 
                                          codec='mpeg4',
                                          quality=8)
    except (ValueError, IOError, RuntimeError):
        print("[VIDEO] mpeg4 failed, trying fallback...")
        video_writer = imageio.get_writer(video_filename_final, fps=VIDEO_FPS)
    
    # Save initial frame
    video_canvas.draw()
    buf = video_canvas.buffer_rgba()
    image = np.asarray(buf)
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    video_writer.append_data(image)
    
    print("[VIDEO] Video writer initialized successfully")


resolution_check_done = False
resolution_check_step = 500

# CFL Safety Check (Phase 2 improvement)
print("\n" + "="*60)
print("CFL SAFETY CHECK")
print("="*60)
v_max_estimate = float(kx0)  # Maximum velocity estimate from initial momentum
cfl_criterion = 0.5 * min(float(dx), float(dy))
cfl_number = dt * v_max_estimate / cfl_criterion
print(f"[CFL] Estimated max velocity: v_max ≈ {v_max_estimate:.4f}")
print(f"[CFL] CFL criterion: dt * v_max < {cfl_criterion:.4f}")
print(f"[CFL] CFL number: {cfl_number:.4f}")
if cfl_number > 1.0:
    print("[CFL] ⚠️  WARNING: CFL number > 1.0! Consider reducing dt or enabling adaptive substeps.")
    print(f"[CFL]            Recommended: dt < {cfl_criterion / v_max_estimate:.6f}")
elif cfl_number > 0.8:
    print("[CFL] ⚠️  CAUTION: CFL number > 0.8. Simulation may be stable but consider smaller dt.")
else:
    print("[CFL] ✓ CFL condition satisfied (CFL < 0.8)")
print("="*60 + "\n")

# ============================================================================
# AUTOMATIC TIME-AVERAGING WINDOW DETECTION
# ============================================================================
# Add this section BEFORE the main time loop starts (around line 1900)
# Insert after the storage initialization section

print("\n" + "="*70)
print("TIME-AVERAGING WINDOW CONFIGURATION")
print("="*70)

# Manual override: if TIME_AVG_WINDOW is set to a number, use it
# If set to None, use automatic detection
USE_AUTO_WINDOW = (TIME_AVG_WINDOW is None) and is_evanescent

if USE_AUTO_WINDOW:
    print("[AUTO] Automatic window detection ENABLED")
    
    # Parameters for automatic detection
    AUTO_WINDOW_DT_THRESHOLD = 1e-3      # dT/dt threshold for convergence
    AUTO_WINDOW_STABLE_TIME = 50.0       # Must be stable for this long
    AUTO_WINDOW_CHECK_POINTS = 30
    AUTO_WINDOW_MIN_DURATION = 200.0     # Minimum averaging duration
    AUTO_WINDOW_FRACTION = 0.6           # Use 60% of remaining time
    AUTO_WINDOW_FALLBACK_START = 200.0   # Fallback if no T(t) data
    
    print(f"[AUTO] Convergence threshold: dT/dt < {AUTO_WINDOW_DT_THRESHOLD:.1e}")
    print(f"[AUTO] Stability window: {AUTO_WINDOW_STABLE_TIME:.1f} time units")
    print(f"[AUTO] Duration: {AUTO_WINDOW_FRACTION*100:.0f}% of remaining time (min {AUTO_WINDOW_MIN_DURATION:.0f})")
    
    # Storage for automatic detection
    T_for_convergence = []
    t_for_convergence = []
    convergence_detected = False
    convergence_time = None
    
else:
    # Manual window
    if TIME_AVG_WINDOW is None:
        TIME_AVG_WINDOW = 400.0  # Default value
    print("[MANUAL] Using fixed window")
    print("[MANUAL] Window starts at barrier hit + 2.0 time units")

print("="*70 + "\n")

#%% =========== Time loop =====================
for step in tqdm(range(1, n_steps+1), desc="Time evolution", file=sys.stderr):
    
    if PROPAGATOR.upper() == 'FFT':
        psi *= Vhalf
        psi_k = cp.fft.fftn(psi)
        psi_k *= Kprop
        psi = cp.fft.ifftn(psi_k)
        psi *= Vhalf
        if USE_ABSORBER:
            psi *= mask
        if INJECT_PLANE:
            phase_t = cp.exp(-1j * (0.5*float(kx0**2) + E_y_inj) * (step * dt))
            ramp = _temporal_ramp(step, INJECT_RAMP_STEPS)
            coef = (-1j) * (GAMMA_ADD * ramp * dt) * phase_t
            psi = psi + coef * S0
        
        # Trajectory computation (conditional)
        if COMPUTE_TRAJECTORIES:
            vx, vy = velocity_field(psi)
            
            # NEW: Store velocities at current trajectory positions
            trajVX = interp2d(trajX, trajY, x, y, vx)
            trajVY = interp2d(trajX, trajY, x, y, vy)
            
            # Update positions
            trajX, trajY = rk2_step_bohm_adaptive(trajX, trajY, vx, vy, dt)
            
    else:
        psi_np *= Vhalf_np
        rhs_x = Bx @ psi_np.T
        psi_np = LUx.solve(rhs_x).T
        rhs_y = By @ psi_np
        psi_np = LUy.solve(rhs_y)
        psi_np *= Vhalf_np
        if USE_ABSORBER:
            psi_np *= mask_np
        if INJECT_PLANE:
            phase_t = phase_array[step]
            ramp = _temporal_ramp(step, INJECT_RAMP_STEPS)
            coef = (-1j) * (GAMMA_ADD * ramp * dt) * phase_t
            psi_np = psi_np + coef * S0_np
        
        # Trajectory computation (conditional)
        if COMPUTE_TRAJECTORIES:
            vx_np, vy_np = velocity_field_cpu(psi_np, dx_np, dy_np, x_np)
            v1x = interp2d_cpu(trajX, trajY, x_np, y_np, vx_np, dx_np, dy_np)
            v1y = interp2d_cpu(trajX, trajY, x_np, y_np, vy_np, dx_np, dy_np)
            
            # NEW: Store velocities at current trajectory positions
            trajVX = v1x
            trajVY = v1y
            
            # RK2 update
            xm = onp.clip(trajX + 0.5*dt*v1x, x_np[0], x_np[-1])
            ym = onp.clip(trajY + 0.5*dt*v1y, y_np[0], y_np[-1])
            v2x = interp2d_cpu(xm, ym, x_np, y_np, vx_np, dx_np, dy_np)
            v2y = interp2d_cpu(xm, ym, x_np, y_np, vy_np, dx_np, dy_np)
            trajX = onp.clip(trajX + dt*v2x, x_np[0], x_np[-1])
            trajY = onp.clip(trajY + dt*v2y, y_np[0], y_np[-1])

    # --- Wavefunction-based dwell integral: accumulate P(x>0, t) dt ---
    if DO_DWELL_TIME and step % DWELL_SAMPLE_INTERVAL == 0:
        if PROPAGATOR.upper() == 'FFT':
            P_now = float(cp.sum(cp.abs(psi[:, ix_x0_dwell:])**2) * dx * dy)
        else:
            P_now = float(onp.sum(onp.abs(psi_np[:, ix_x0_dwell:])**2) * dx_np * dy_np)
        P_barrier_int += P_now * (DWELL_SAMPLE_INTERVAL * dt)

    # Resolution check
    if DO_RESOLUTION_CHECK and not resolution_check_done and step == resolution_check_step:
        print("\n" + "="*60)
        print("RESOLUTION ANALYSIS")
        print("="*60)
        
        if PROPAGATOR.upper() == 'FFT':
            psi_check = psi
        else:
            psi_check = cp.asarray(psi_np)
        
        # Use spectral derivatives for consistency with evolution
        dpsi_dx, dpsi_dy = spectral_gradients(psi_check)
        psi_mag = cp.abs(psi_check)
        
        local_lambda_x = 2*cp.pi * psi_mag / (cp.abs(dpsi_dx) + 1e-20)
        local_lambda_y = 2*cp.pi * psi_mag / (cp.abs(dpsi_dy) + 1e-20)
        
        mask_sig = psi_mag > 0.01 * float(psi_mag.max())
        
        if cp.any(mask_sig):
            lambda_x_vals = local_lambda_x[mask_sig]
            lambda_y_vals = local_lambda_y[mask_sig]
            
            # Different thresholds for x and y based on physics
            # Y: confined by harmonic potential → shorter wavelengths
            # X: can have very large wavelengths in evanescent regions
            LAMBDA_MAX_Y = max(100 * float(dy), 20.0)   # Tighter threshold for y
            LAMBDA_MAX_X = max(1000 * float(dx), 100.0)  # Much more lenient for x
            
            lambda_x_vals = lambda_x_vals[(lambda_x_vals > 0) & (lambda_x_vals < LAMBDA_MAX_X)]
            lambda_y_vals = lambda_y_vals[(lambda_y_vals > 0) & (lambda_y_vals < LAMBDA_MAX_Y)]
            
            if lambda_x_vals.size > 0:
                lambda_x_min = float(cp.percentile(lambda_x_vals, 5))
                ppw_x = lambda_x_min / float(dx)
                print(f"[SPATIAL X] Min wavelength: {lambda_x_min:.4f}, Points per wavelength: {ppw_x:.1f}")
                if ppw_x < 8:
                    print("            ⚠️  WARNING: dx too large!")
                else:
                    print("            ✓ Spatial resolution is adequate")
            else:
                print(f"[SPATIAL X] No valid wavelengths (all > {LAMBDA_MAX_X:.1f} or ≤ 0)")
                print(f"            Expected in deep evanescent regions")
            
            if lambda_y_vals.size > 0:
                lambda_y_min = float(cp.percentile(lambda_y_vals, 5))
                ppw_y = lambda_y_min / float(dy)
                print(f"[SPATIAL Y] Min wavelength: {lambda_y_min:.4f}, Points per wavelength: {ppw_y:.1f}")
                if ppw_y < 8:
                    print("            ⚠️  WARNING: dy too large!")
                else:
                    print("            ✓ Spatial resolution is adequate")
            else:
                print(f"[SPATIAL Y] No valid wavelengths (all > {LAMBDA_MAX_Y:.1f} or ≤ 0)")
        
        if PROPAGATOR.upper() == 'CN_ADI':
            dt_cfl_x = 0.5 * (float(dx)**2) / (hbar/m)
            dt_cfl_y = 0.5 * (float(dy)**2) / (hbar/m)
            dt_cfl = min(dt_cfl_x, dt_cfl_y)
            print(f"[TEMPORAL]  Current dt: {dt:.6f}")
            print(f"            CFL limit: {dt_cfl:.6f}")
            if dt > 0.8 * dt_cfl:
                print("            ⚠️  WARNING: dt too large!")
            else:
                print("            ✓ dt satisfies CFL condition")
        
        print("="*60 + "\n")
        resolution_check_done = True

   
# FIXED barrier hit detection with DEFERRED window selection
    if DO_TIME_AVG_DENSITY:
        # Step 1: Detect when packet reaches barrier
        if barrier_hit_step is None and step > barrier_check_startup_steps:
            if PROPAGATOR.upper() == 'FFT':
                rho = cp.abs(psi)**2
            else:
                rho = onp.abs(psi_np)**2
            
            # Get density at barrier (column-sum)
            ix_barrier = int(np.searchsorted(x_cpu_full, 0.0))
            if PROPAGATOR.upper() == 'FFT':
                col = rho[:, ix_barrier]
                density_at_barrier = float(cp.sum(col) * dy)
            else:
                col = rho[:, ix_barrier]
                density_at_barrier = float(onp.sum(col) * dy_np)
            
            # Update historical peak (also using column-sum)
            historical_peak = max(historical_peak, density_at_barrier)
            
            # Trigger when barrier density exceeds threshold AND is above absolute minimum
            threshold_met = density_at_barrier > BARRIER_HIT_THRESHOLD * historical_peak
            absolute_met = density_at_barrier > barrier_min_absolute_density
            
            if threshold_met and absolute_met:
                barrier_hit_step = step
                barrier_hit_time = step * dt
                
                print(f"\n[TIME_AVG] Significant packet reached barrier at step {step} (t={barrier_hit_time:.2f})")
                print(f"[TIME_AVG] Column-integrated density: {density_at_barrier:.3e}, Historical peak: {historical_peak:.3e}")
                
                if USE_AUTO_WINDOW:
                    print("[AUTO WINDOW] Waiting for T(t) convergence to optimize window...")
                else:
                    # Manual mode: commit immediately
                    averaging_start_step = step + int(2.0 / dt)
                    
                    # REGIME-SPECIFIC WINDOW
                    if is_evanescent:
                        # Evanescent: fixed window
                        averaging_end_step = averaging_start_step + int(TIME_AVG_WINDOW / dt)
                        print(f"[TIME_AVG] Evanescent: fixed window of {TIME_AVG_WINDOW:.1f} time units")
                    else:
                        # Propagative: adaptive window (90% of remaining time)
                        t_remaining = n_steps * dt - (averaging_start_step * dt)
                        duration = 0.9 * t_remaining
                        averaging_end_step = averaging_start_step + int(duration / dt)
                        print(f"[TIME_AVG] Propagative: adaptive window of {duration:.1f} time units (90% of remaining)")
        
                    averaging_window_committed = True
                    print(f"[TIME_AVG] Window: start at t={averaging_start_step*dt:.2f}, end at t={averaging_end_step*dt:.2f}")
        
        # Step 2: Decide on window (runs every step after barrier hit)
        if USE_AUTO_WINDOW and barrier_hit_step is not None and not averaging_window_committed:
            
            # Option A: Convergence detected - use optimal window
            if convergence_detected:
                t_convergence = convergence_time
                delay_after_convergence = 2.0
                averaging_start_step = int((t_convergence + delay_after_convergence) / dt)
                
                # Duration: fraction of remaining time, with minimum
                t_remaining = n_steps * dt - (averaging_start_step * dt)
                duration = max(AUTO_WINDOW_MIN_DURATION, AUTO_WINDOW_FRACTION * t_remaining)
                averaging_end_step = averaging_start_step + int(duration / dt)
                
                print(f"\n[AUTO WINDOW] ✓ Convergence detected at t = {t_convergence:.2f}")
                print(f"[AUTO WINDOW]   Start: t = {averaging_start_step*dt:.2f} (convergence + {delay_after_convergence:.1f})")
                print(f"[AUTO WINDOW]   Duration: {duration:.1f} time units ({AUTO_WINDOW_FRACTION*100:.0f}% of remaining)")
                print(f"[AUTO WINDOW]   End: t = {averaging_end_step*dt:.2f}")
                
                averaging_window_committed = True
            
            # Option B: Running out of time - use fallback
            elif step > int(0.7 * n_steps):
                barrier_hit_time = barrier_hit_step * dt
                averaging_start_step = int((barrier_hit_time + AUTO_WINDOW_FALLBACK_START + 2.0) / dt)
                
                t_remaining = n_steps * dt - (averaging_start_step * dt)
                duration = max(AUTO_WINDOW_MIN_DURATION, AUTO_WINDOW_FRACTION * t_remaining)
                averaging_end_step = averaging_start_step + int(duration / dt)
                
                print(f"\n[AUTO WINDOW] ⚠ Fallback triggered at t = {step*dt:.2f} (70% through simulation)")
                print(f"[AUTO WINDOW]   No convergence detected, using heuristic window")
                print(f"[AUTO WINDOW]   Start: t = {averaging_start_step*dt:.2f} (barrier + {AUTO_WINDOW_FALLBACK_START:.0f} + 2)")
                print(f"[AUTO WINDOW]   Duration: {duration:.1f} time units")
                print(f"[AUTO WINDOW]   End: t = {averaging_end_step*dt:.2f}")
                
                averaging_window_committed = True
        
        # Step 3: Start accumulation when ready
        if barrier_hit_step is not None and averaging_window_committed and not averaging_started:
            if step >= averaging_start_step:
                averaging_started = True
                print(f"[TIME_AVG] Starting density accumulation at step {step} (t={step*dt:.2f})")
        
        # Step 4: Accumulate density during window
        if averaging_started and step <= averaging_end_step:
            if PROPAGATOR.upper() == 'FFT':
                rho_roi = cp.abs(psi[:, ix_roi1:ix_roi2])**2
            else:
                rho_roi = onp.abs(psi_np[:, ix_roi1:ix_roi2])**2
            density_sum_roi += rho_roi
            density_count_roi += 1
            
            if step == averaging_end_step:
                print(f"[TIME_AVG] Completed density accumulation at step {step} (t={step*dt:.2f})")
                print(f"[TIME_AVG] Accumulated {density_count_roi} snapshots over {(averaging_end_step-averaging_start_step)*dt:.1f} time units")

    # Enhanced Transmission/Reflection analytics with flux method
    if DO_TRANSMISSION_ANALYTICS and step % FLUX_SAMPLE_INTERVAL == 0:
        if PROPAGATOR.upper() == 'FFT':
            # Use spectral derivatives for consistency (Phase 2 improvement)
            dpsi_dx_full, _ = spectral_gradients(psi)
            
            # Flux-based calculation at specific x-planes
            if ix_barrier < Nx:
                jx_barrier = (hbar/m) * cp.imag(cp.conj(psi[:, ix_barrier]) * dpsi_dx_full[:, ix_barrier])
                flux_barrier = float(cp.sum(jx_barrier) * dy)
            else:
                flux_barrier = 0.0
            
            if ix_left < Nx:
                jx_left = (hbar/m) * cp.imag(cp.conj(psi[:, ix_left]) * dpsi_dx_full[:, ix_left])
                flux_left = float(cp.sum(jx_left) * dy)
            else:
                flux_left = 0.0
            
            if ix_right < Nx:
                jx_right = (hbar/m) * cp.imag(cp.conj(psi[:, ix_right]) * dpsi_dx_full[:, ix_right])
                flux_right = float(cp.sum(jx_right) * dy)
            else:
                flux_right = 0.0
            
            # Traditional probability-based
            rho = cp.abs(psi)**2
            P_left = float(cp.sum(rho[:, :ix_left]) * dx * dy)
            P_right = float(cp.sum(rho[:, ix_right:]) * dx * dy)
            P_total = float(cp.sum(rho) * dx * dy)
        else:
            # CPU path - use finite differences (CN_ADI doesn't have spectral grads readily available)
            if 1 < ix_barrier < Nx-1:
                dpsi_dx_barrier = (psi_np[:, ix_barrier+1] - psi_np[:, ix_barrier-1]) / (2*dx_np)
                jx_barrier = (hbar/m) * onp.imag(onp.conj(psi_np[:, ix_barrier]) * dpsi_dx_barrier)
                flux_barrier = float(onp.sum(jx_barrier) * dy_np)
            else:
                flux_barrier = 0.0
            
            if 1 < ix_left < Nx-1:
                dpsi_dx_left = (psi_np[:, ix_left+1] - psi_np[:, ix_left-1]) / (2*dx_np)
                jx_left = (hbar/m) * onp.imag(onp.conj(psi_np[:, ix_left]) * dpsi_dx_left)
                flux_left = float(onp.sum(jx_left) * dy_np)
            else:
                flux_left = 0.0
            
            if 1 < ix_left < Nx-1:
                dpsi_dx_right = (psi_np[:, ix_right+1] - psi_np[:, ix_right-1]) / (2*dx_np)
                jx_right = (hbar/m) * onp.imag(onp.conj(psi_np[:, ix_right]) * dpsi_dx_right)
                flux_right = float(onp.sum(jx_right) * dy_np)
            else:
                flux_right = 0.0
            
            rho = onp.abs(psi_np)**2
            P_left = float(onp.sum(rho[:, :ix_left]) * dx_np * dy_np)
            P_right = float(onp.sum(rho[:, ix_right:]) * dx_np * dy_np)
            P_total = float(onp.sum(rho) * dx_np * dy_np)
        
        # Store flux data
        flux_incident_history.append(max(flux_barrier, 0))  # Forward flux at barrier
        flux_transmitted_history.append(max(flux_right, 0))
        flux_reflected_history.append(abs(min(flux_left, 0)))  # Backward flux
        
        # Probability-based T/R
        if P_total > 1e-10:
            T = P_right / P_total
            R = P_left / P_total
        else:
            T = R = 0.0
        
        T_history.append(T)
        R_history.append(R)
        t_history.append(step * dt)
    # ========================================================================
    # AUTOMATIC WINDOW DETECTION (if enabled)
    # ========================================================================
    if USE_AUTO_WINDOW and DO_TRANSMISSION_ANALYTICS and not convergence_detected:
        if len(T_history) > 0:
            T_for_convergence.append(T_history[-1])
            t_for_convergence.append(t_history[-1])
            
            # Check for convergence once we have enough data
            if len(T_for_convergence) > 10:
                # Compute derivative over last N points
                N_check = min(AUTO_WINDOW_CHECK_POINTS, len(T_for_convergence))
                t_recent = np.array(t_for_convergence[-N_check:])
                T_recent = np.array(T_for_convergence[-N_check:])
                
                # Linear fit to recent data
                if len(t_recent) > 1 and (t_recent[-1] - t_recent[0]) > 0:
                    slope = (T_recent[-1] - T_recent[0]) / (t_recent[-1] - t_recent[0])
                    
                    # Check if converged
                    if abs(slope) < AUTO_WINDOW_DT_THRESHOLD:
                        # Check if stable for required time
                        time_stable = t_recent[-1] - t_recent[0]
                        if time_stable >= AUTO_WINDOW_STABLE_TIME:
                            convergence_detected = True
                            convergence_time = t_recent[-1]
                            print(f"\n[AUTO WINDOW] T(t) converged at t = {convergence_time:.2f}")
                            print(f"[AUTO WINDOW] Slope: {abs(slope):.2e} < {AUTO_WINDOW_DT_THRESHOLD:.2e}")
    # Bulk storage for trajectories
    if step % BULK_HIST_STRIDE == 0 and STORE_BULK_TRAJECTORIES:
        if PROPAGATOR.upper() == 'FFT':
            traj_hist_X_bulk[store_i_bulk, :] = trajX.astype(cp.float32)
            traj_hist_Y_bulk[store_i_bulk, :] = trajY.astype(cp.float32)
            if DO_MEAN_TEST:
                rho = cp.abs(psi)**2
                rho_x = cp.sum(rho, axis=0)
                mean_x_psi[store_i_bulk] = float(cp.dot(rho_x, x)) / float(cp.sum(rho_x)) if float(cp.sum(rho_x)) > 0 else np.nan
        else:
            traj_hist_X_bulk[store_i_bulk, :] = trajX.astype(onp.float32)
            traj_hist_Y_bulk[store_i_bulk, :] = trajY.astype(onp.float32)
            if DO_MEAN_TEST:
                rho = onp.abs(psi_np)**2
                rho_x = onp.sum(rho, axis=0)
                mean_x_psi[store_i_bulk] = float(onp.dot(rho_x, x_np)) / float(onp.sum(rho_x)) if float(onp.sum(rho_x)) > 0 else np.nan

        if DO_MEAN_TEST:
            mean_x_trj[store_i_bulk] = float(onp.mean(trajX)) if PROPAGATOR.upper() == 'CN_ADI' else float(cp.mean(trajX))
    
    # D_ROI computation (INDEPENDENT of trajectories!)
    if step % BULK_HIST_STRIDE == 0 and DO_RIGHT_IMBALANCE:
        if PROPAGATOR.upper() == 'FFT':
            rho = cp.abs(psi)**2
            slab = rho[:, ix1:ix2]
            rho_y_roi = cp.sum(slab, axis=1) * float(dx)
            PR_t = float(cp.sum(rho_y_roi[split:]) * float(dy))
            PL_t = float(cp.sum(rho_y_roi[:split]) * float(dy))
        else:
            rho = onp.abs(psi_np)**2
            slab = rho[:, ix1:ix2]
            rho_y_roi = onp.sum(slab, axis=1) * float(dx_np)
            PR_t = float(onp.sum(rho_y_roi[split:]) * float(dy_np))
            PL_t = float(onp.sum(rho_y_roi[:split]) * float(dy_np))
        
        PR[store_i_bulk], PL[store_i_bulk] = PR_t, PL_t
        D[store_i_bulk] = (PR_t-PL_t)/(PR_t+PL_t) if (PR_t+PL_t)>0 else 0.0
    
    # Increment bulk storage index (whether storing trajectories or not)
    if step % BULK_HIST_STRIDE == 0 and (STORE_BULK_TRAJECTORIES or DO_RIGHT_IMBALANCE):
        store_i_bulk += 1

    # High-res ROI storage
    if COMPUTE_TRAJECTORIES and step % ROI_HIST_STRIDE == 0:
        if PROPAGATOR.upper() == 'FFT':
            in_roi = (trajX >= ROI_X1) & (trajX <= ROI_X2)
            newly_in_roi = in_roi & (~traj_entered_roi)
            if cp.any(newly_in_roi):
                traj_entered_roi |= in_roi
            
            if cp.any(traj_entered_roi):
                indices_roi = cp.where(traj_entered_roi)[0]
                current_time = step * dt
                
                indices_cpu = cp.asnumpy(indices_roi).astype(int)
                x_vals_cpu = cp.asnumpy(trajX[indices_roi])
                y_vals_cpu = cp.asnumpy(trajY[indices_roi])
                vx_vals_cpu = cp.asnumpy(trajVX[indices_roi])  # NEW
                vy_vals_cpu = cp.asnumpy(trajVY[indices_roi])  # NEW
                
                for i, idx_int in enumerate(indices_cpu):
                    if idx_int not in traj_hist_X_roi:
                        traj_hist_X_roi[idx_int] = []
                        traj_hist_Y_roi[idx_int] = []
                        traj_times_roi[idx_int] = []
                        traj_hist_VX_roi[idx_int] = []  # NEW
                        traj_hist_VY_roi[idx_int] = [] 
                    traj_hist_X_roi[idx_int].append(x_vals_cpu[i])
                    traj_hist_Y_roi[idx_int].append(y_vals_cpu[i])
                    traj_times_roi[idx_int].append(current_time)
                    traj_hist_VX_roi[idx_int].append(vx_vals_cpu[i])  # New
                    traj_hist_VY_roi[idx_int].append(vy_vals_cpu[i])
        else:
            in_roi = (trajX >= ROI_X1) & (trajX <= ROI_X2)
            newly_in_roi = in_roi & (~traj_entered_roi)
            if onp.any(newly_in_roi):
                traj_entered_roi = traj_entered_roi | in_roi
            
            if onp.any(traj_entered_roi):
                indices_roi = onp.where(traj_entered_roi)[0]
                current_time = step * dt
                
                indices_cpu = indices_roi.astype(int)
                x_vals_cpu = trajX[indices_roi]
                y_vals_cpu = trajY[indices_roi]
                vx_vals_cpu = trajVX[indices_roi]  # NEW
                vy_vals_cpu = trajVY[indices_roi]  # NEW
                
                for i, idx_int in enumerate(indices_cpu):
                    if idx_int not in traj_hist_X_roi:
                        traj_hist_X_roi[idx_int] = []
                        traj_hist_Y_roi[idx_int] = []
                        traj_times_roi[idx_int] = []
                        traj_hist_VX_roi[idx_int] = []  # NEW
                        traj_hist_VY_roi[idx_int] = []  # NEW
                    traj_hist_X_roi[idx_int].append(x_vals_cpu[i])
                    traj_hist_Y_roi[idx_int].append(y_vals_cpu[i])
                    traj_times_roi[idx_int].append(current_time)
                    traj_hist_VX_roi[idx_int].append(vx_vals_cpu[i])  # NEW
                    traj_hist_VY_roi[idx_int].append(vy_vals_cpu[i])  # NEW

    if LIVE_ANIM and step % ANIM_EVERY == 0:
        if PROPAGATOR.upper() == 'FFT':
            cp.cuda.runtime.deviceSynchronize()
            dens_cpu = cp.asnumpy(cp.abs(psi)**2)
            trajX_np = cp.asnumpy(trajX)
            trajY_np = cp.asnumpy(trajY)
        else:
            dens_cpu = onp.abs(psi_np)**2
            trajX_np = trajX
            trajY_np = trajY
    
        im.set_data(dens_cpu)
        im.set_clim(0, dens_cpu.max()*0.9 if dens_cpu.max() > 0 else 1.0)
        if COMPUTE_TRAJECTORIES:
            traj_plot.set_data(trajX_np, trajY_np)
            ax.set_title(f"Live |ψ|² + Bohmian ({VELOCITY_METHOD}) - t={step*dt:.2f}")
        else:
            ax.set_title(f"Live |ψ|² - t={step*dt:.2f}")
        
        plt.pause(0.05)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # Video frame saving (independent of live animation)
    if SAVE_VIDEO and step % VIDEO_STRIDE == 0:
        if PROPAGATOR.upper() == 'FFT':
            cp.cuda.runtime.deviceSynchronize()
            dens_cpu = cp.asnumpy(cp.abs(psi)**2)
            trajX_np = cp.asnumpy(trajX)
            trajY_np = cp.asnumpy(trajY)
        else:
            dens_cpu = onp.abs(psi_np)**2
            trajX_np = trajX
            trajY_np = trajY
        
        video_im.set_data(dens_cpu)
        video_im.set_clim(0, dens_cpu.max()*0.9 if dens_cpu.max() > 0 else 1.0)
        if COMPUTE_TRAJECTORIES:
            video_traj_plot.set_data(trajX_np, trajY_np)
            video_ax.set_title(f"|ψ|² + Bohmian trajectories ({VELOCITY_METHOD}) - t={step*dt:.2f}")
        else:
            video_ax.set_title(f"|ψ|² - t={step*dt:.2f}")
        
        # Render to image and save (modern matplotlib way)
        video_canvas.draw()
        buf = video_canvas.buffer_rgba()
        image = np.asarray(buf)
        # Convert RGBA to RGB and ensure contiguous
        image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
        video_writer.append_data(image)

if LIVE_ANIM:
    plt.ioff()
    plt.show()

# Close video writer if it was being used
if SAVE_VIDEO and video_writer is not None:
    video_writer.close()
    del video_fig, video_canvas  # Clean up video objects completely
    print(f"\n[VIDEO] Video saved successfully: {video_filename_final}")
    
    # Try to get file size
    try:
        import os
        file_size_mb = os.path.getsize(video_filename_final) / (1024 * 1024)
        print(f"[VIDEO] File size: {file_size_mb:.2f} MB")
        
        # Calculate total frames and duration
        total_frames = (n_steps // VIDEO_STRIDE) + 1
        duration_sec = total_frames / VIDEO_FPS
        print(f"[VIDEO] Total frames: {total_frames}, Duration: {duration_sec:.1f} seconds")
    except (OSError, IOError):
        pass

# Ensure interactive mode is ON for analysis plots
plt.ion()
print(f"[PLOTS] Interactive plotting enabled for analysis (backend: {matplotlib.get_backend()})")

# =========== Compute trajectory-based T/R from FINAL positions ===========
if DO_TRANSMISSION_ANALYTICS and COMPUTE_TRAJECTORIES and n_traj > 0:
    print("\n[TRAJECTORY T/R] Classifying trajectories based on final positions...")
    
    if PROPAGATOR.upper() == 'FFT':
        trajX_final = cp.asnumpy(trajX)
    else:
        trajX_final = trajX
    
    # Classify based on final position
    for i in range(n_traj):
        x_final = float(trajX_final[i])
        
        # Transmitted: ended up to the right of the transmission boundary
        if x_final > TRANSMISSION_X_RIGHT:
            traj_transmitted += 1
            traj_counted.add(i)
        # Reflected: ended up to the left of the reflection boundary
        elif x_final < TRANSMISSION_X_LEFT:
            traj_reflected += 1
            traj_counted.add(i)
        # In between: check if closer to left or right, or if crossed barrier
        else:
            # If between left boundary and barrier, count as reflected
            if x_final < STEP_CENTER:
                traj_reflected += 1
                traj_counted.add(i)
            # If between barrier and right boundary, count as transmitted
            else:
                traj_transmitted += 1
                traj_counted.add(i)
    
    print("[TRAJECTORY T/R] Final classification:")
    print(f"                 Transmitted (x > {TRANSMISSION_X_RIGHT}): {traj_transmitted}")
    print(f"                 Reflected (x < {TRANSMISSION_X_LEFT}): {traj_reflected}")
    print(f"                 Total classified: {len(traj_counted)} / {n_traj}")

# Report statistics
if COMPUTE_TRAJECTORIES and n_traj > 0:
    if PROPAGATOR.upper() == 'FFT':
        n_roi_entered = int(cp.sum(traj_entered_roi))
    else:
        n_roi_entered = int(onp.sum(traj_entered_roi))
    
    print(f"\n[ROI STATS] {n_roi_entered} / {n_traj} trajectories ({100*n_roi_entered/n_traj:.2f}%) entered ROI")
else:
    n_roi_entered = 0
    if not COMPUTE_TRAJECTORIES:
        print("\n[ROI STATS] Trajectory computation disabled - ROI statistics not available")


#%% =========== DWELL TIME ANALYSIS (trajectories + wavefunction) ===========
# Per-trajectory dwell = (number of ROI samples with x>0) * (ROI sampling dt).
# This counts total time spent at x>0 and is robust to re-entries (a trajectory
# that turns around, exits, and re-enters is handled correctly), unlike an
# entry-to-exit span. The all-trajectory mean (zeros for never-entered) must
# equal the wavefunction integral ∫P(x>0)dt by the Buettiker theorem, since the
# trajectories are Born-distributed.
dwell_time_entered_mean = None
dwell_time_entered_se   = None
dwell_time_all_mean     = None
dwell_time_all_se       = None
dwell_wf_integral       = (P_barrier_int if DO_DWELL_TIME else None)
tau_entered_theory      = None
tau_d2d_theory          = None
n_still_inside          = None
sigma_k_code            = 1.0 / (2.0 * sigx)

if COMPUTE_TRAJECTORIES and len(traj_hist_X_roi) > 0:
    print("\n" + "="*70)
    print("DWELL TIME ANALYSIS")
    print("="*70)

    dt_sample = ROI_HIST_STRIDE * dt
    _keys = list(traj_hist_X_roi.keys())
    print(f"  Reducing dwell over {len(_keys)} ROI trajectories "
          f"(stride={ROI_HIST_STRIDE}, dt_sample={dt_sample:.4g} code)...")

    # Per-trajectory dwell = (# samples with x>0) * dt_sample. The histories are
    # host-side float lists; np.fromiter + count_nonzero is much faster than
    # np.asarray + np.sum, and the tqdm bar makes the reduction visibly progress
    # (this loop touches every entered trajectory's full history, so for large
    #  n_traj it can take tens of seconds).
    _counts = np.empty(len(_keys), dtype=np.float64)
    n_still_inside = 0
    for _j, _idx in enumerate(tqdm(_keys, desc="Dwell reduce", file=sys.stderr)):
        _lst = traj_hist_X_roi[_idx]
        _x_t = np.fromiter(_lst, dtype=np.float64, count=len(_lst))
        _counts[_j] = np.count_nonzero(_x_t > 0.0)
        if _x_t.size and _x_t[-1] > 0.0:
            n_still_inside += 1                          # truncation diagnostic

    _counts *= dt_sample
    dwell_entered = _counts[_counts > 0.0]               # total time at x>0
    n_entered = dwell_entered.size

    if n_entered > 0:
        dwell_time_entered_mean = float(np.mean(dwell_entered))
        dwell_time_entered_se = (float(np.std(dwell_entered, ddof=1) / np.sqrt(n_entered))
                                 if n_entered > 1 else 0.0)

        dwell_all = np.concatenate([dwell_entered, np.zeros(max(n_traj - n_entered, 0))])
        dwell_time_all_mean = float(np.mean(dwell_all))
        dwell_time_all_se = (float(np.std(dwell_all, ddof=1) / np.sqrt(dwell_all.size))
                             if dwell_all.size > 1 else 0.0)

        time_to_ps = T0 * 1e12
        print(f"  Entered (n={n_entered}):  {dwell_time_entered_mean:.3f} "
              f"± {dwell_time_entered_se:.3f} code "
              f"= {dwell_time_entered_mean*time_to_ps:.3f} "
              f"± {dwell_time_entered_se*time_to_ps:.3f} ps")
        print(f"  All (n={n_traj}):         {dwell_time_all_mean:.4f} "
              f"± {dwell_time_all_se:.4f} code "
              f"= {dwell_time_all_mean*time_to_ps:.4f} "
              f"± {dwell_time_all_se*time_to_ps:.4f} ps")
        print(f"  Still inside at t_end: {n_still_inside} "
              f"({100*n_still_inside/max(n_entered,1):.2f}% of entered)  <- truncation check")

        # Wavefunction-based Buettiker integral (must equal the all-trajectory mean)
        if DO_DWELL_TIME:
            print(f"  Wavefunction integral ∫P(x>0)dt: {P_barrier_int:.4f} code "
                  f"= {P_barrier_int*time_to_ps:.4f} ps")
            if P_barrier_int > 0:
                print(f"  Ratio traj_all / wf_integral = "
                      f"{dwell_time_all_mean/P_barrier_int:.4f}  (expect 1.000)")

        # Theory references (paper-consistent after the Eq. (3) delta_y shift)
        tau_entered_theory = float(np.sqrt(np.pi/2) / (kx0 * sigma_k_code))   # Eq. (62)
        print(f"  Theory entered, Eq.(62): {tau_entered_theory:.3f} code "
              f"= {tau_entered_theory*time_to_ps:.3f} ps")

        if is_evanescent:
            # kappa0 = kappa_+ (symmetric), kappa1 = kappa_- (antisymmetric)
            _gp2 = 4*kx0**2 / (kx0**2 + kappa0**2)
            _gm2 = 4*kx0**2 / (kx0**2 + kappa1**2)
            tau_d2d_theory = float((0.5/kx0) * (_gp2/(2*kappa0) + _gm2/(2*kappa1)))
            print(f"  Theory all, tau_d^2D(k0): {tau_d2d_theory:.4f} code "
                  f"= {tau_d2d_theory*time_to_ps:.4f} ps")
    else:
        print("  No trajectories entered x>0; dwell statistics unavailable.")


#%% =========== MIGRATION-TIME HISTOGRAM (tunneling clock) ===================
# Histogram of transverse migration times (upper -> lower well) on the
# normalized Rabi axis tau = t / (pi/2 J0). The 1D double-well flux gives
# p(tau) = (pi/2) sin(pi tau) on [0, 1]; this clock is set by the coupling J0
# and is the same in either longitudinal regime, so only the title adapts.
if DO_MIGRATION_HISTOGRAM and COMPUTE_TRAJECTORIES and len(traj_hist_X_roi) > 0:
    print("\n" + "="*70)
    print("MIGRATION-TIME HISTOGRAM")
    print("="*70)

    _CODE_TO_UM   = L0 * 1e6
    _time_to_ps   = T0 * 1e12
    _y_thr_code   = -MIGRATION_THRESHOLD_SIGMA_WELL * sigma_well

    # Extract migration times from the in-memory ROI trajectories.
    _mig_times = []
    _n_upper = 0
    for _k in traj_hist_X_roi.keys():
        _xt = np.asarray(traj_hist_X_roi[_k])
        _yt = np.asarray(traj_hist_Y_roi[_k])
        _tt = np.asarray(traj_times_roi[_k])
        if _xt.size < 2:
            continue
        _ninit = min(3, _yt.size)
        if np.mean(_yt[:_ninit]) <= 0:          # must start in the UPPER waveguide
            continue
        _n_upper += 1
        _t_enter = _tt[0]
        if MIGRATION_MODE == 'redefine':
            _idx = np.where(_yt < _y_thr_code)[0]
            if _idx.size == 0:
                continue
            _t_cross = _tt[_idx[0]]
        elif MIGRATION_MODE == 'filter':
            if not np.any(_yt < _y_thr_code):
                continue
            _idx = np.where(_yt < 0)[0]
            if _idx.size == 0:
                continue
            _t_cross = _tt[_idx[0]]
        else:
            raise ValueError(f"Unknown MIGRATION_MODE: {MIGRATION_MODE}")
        _mt = _t_cross - _t_enter
        if _mt > 0:
            _mig_times.append(_mt)

    _mig_times = np.asarray(_mig_times)
    n_migrated = _mig_times.size
    print(f"  Upper-waveguide trajectories: {_n_upper}")
    print(f"  Migration events: {n_migrated}"
          + (f" ({100*n_migrated/_n_upper:.1f}%)" if _n_upper else ""))

    if n_migrated >= 10:
        J0_ps = J0_energy / _time_to_ps          # J0 in ps^-1 (hbar=1 -> E=omega)
        T_rabi_half_ps = np.pi / (2.0 * J0_ps)
        _tau = (_mig_times * _time_to_ps) / T_rabi_half_ps
        _tau = _tau[_tau > 0]

        # Regime-adaptive title (clock is regime-independent; label is not).
        if is_evanescent:
            _regime_title = "Evanescent Regime"
        elif is_propagative:
            _regime_title = "Propagative Regime"
        else:
            _regime_title = "Mixed Regime"

        print(f"  J0 = {J0_ps:.6f} ps^-1, pi/(2 J0) = {T_rabi_half_ps:.2f} ps (tau=1)")
        print(f"  tau: mean={np.mean(_tau):.3f}, median={np.median(_tau):.3f}, "
              f"max={np.max(_tau):.3f}  [{_regime_title}]")

        _fig_mig, _ax_mig = plt.subplots(figsize=(12, 6))
        # NOTE: density=True normalises over [0, 2.5] only; migration times with
        # tau > 2.5 are DISCARDED and the remainder rescaled to unit area. The
        # theory curve is normalised over [0, 1]. Compare the printed max(tau)
        # against 2.5 before reading the vertical scale quantitatively.
        _bins = np.linspace(0, 2.5, MIGRATION_N_BINS)
        _ax_mig.hist(_tau, bins=_bins, alpha=0.6, color='#4575B4',
                     label=f'2D simulation (n={_tau.size})', density=True,
                     edgecolor='black', linewidth=0.5)
        _tau_th = np.linspace(0, 1.0, 500)
        _ax_mig.plot(_tau_th, (np.pi/2)*np.sin(np.pi*_tau_th), color='#D55E00', lw=3,
                     label=r'1D double well: $p(\tau)=(\pi/2)\sin(\pi\tau)$', zorder=10)
        _ax_mig.axvline(0.5, color='#7B1FA2', ls='--', lw=1.5, alpha=0.7,
                        label=r'$\tau=1/2$')
        _ax_mig.axvline(1.0, color='#D55E00', ls=':', lw=1.5, alpha=0.6,
                        label=r'$\tau=1$ (Rabi half-period)')
        _ax_mig.set_xlabel(r"Normalized tunneling time  $\tau = t\,/\,(\pi/2J_0)$",
                           fontsize=12)
        _ax_mig.set_ylabel(r"Probability distribution  $p(\tau)$", fontsize=12)
        _ax_mig.set_title(f"Tunneling Time Distribution: {_regime_title}",
                          fontsize=13, fontweight='bold')
        _ax_mig.legend(fontsize=11, loc='best', framealpha=0.95)
        _ax_mig.grid(alpha=0.3)
        _ax_mig.set_xlim(0, 2.5)
        plt.tight_layout()
        _mig_png = os.path.join(results_dir, 'migration_time_histogram.png')
        plt.savefig(_mig_png, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'migration_time_histogram.pdf'),
                    bbox_inches='tight')
        print(f"  [SAVED] {_mig_png}")
    else:
        print("  Too few migration events (<10); histogram skipped.")


#%% =========== Transmission/Reflection Analysis =====================
t_array_bulk = np.arange(T_store_bulk) * (dt * BULK_HIST_STRIDE)

if STORE_BULK_TRAJECTORIES:
    if PROPAGATOR.upper() == 'FFT':
        Xhist_bulk = cp.asnumpy(traj_hist_X_bulk)
        Yhist_bulk = cp.asnumpy(traj_hist_Y_bulk)
    else:
        Xhist_bulk = traj_hist_X_bulk
        Yhist_bulk = traj_hist_Y_bulk
else:
    # No bulk trajectory data available
    Xhist_bulk = None
    Yhist_bulk = None

# Enhanced Transmission/Reflection plot with all three methods
if DO_TRANSMISSION_ANALYTICS and len(T_history) > 0:
    print("\n" + "="*60)
    print("ENHANCED TRANSMISSION/REFLECTION ANALYSIS")
    print("="*60)
    
    t_arr = np.array(t_history)
    T_arr = np.array(T_history)
    R_arr = np.array(R_history)
    
    # Trajectory-based T/R
    if len(traj_counted) > 0:
        T_traj = traj_transmitted / len(traj_counted)
        R_traj = traj_reflected / len(traj_counted)
    else:
        T_traj = R_traj = 0.0
    
    # Steady-state values (last 20% of simulation)
    steady_idx = int(0.8 * len(T_arr))
    if steady_idx < len(T_arr):
        T_steady = np.mean(T_arr[steady_idx:])
        R_steady = np.mean(R_arr[steady_idx:])
    else:
        T_steady = T_arr[-1] if len(T_arr) > 0 else 0.0
        R_steady = R_arr[-1] if len(R_arr) > 0 else 0.0
    
    print("\n[WAVEFUNCTION - Probability-based]")
    print(f"  Transmission coefficient: T = {T_steady:.6f}")
    print(f"  Reflection coefficient:   R = {R_steady:.6f}")
    print(f"  T + R = {T_steady + R_steady:.6f}")
    
    if COMPUTE_TRAJECTORIES and n_traj > 0:
        print("\n[TRAJECTORIES]")
        print("  Classification based on FINAL positions (end of simulation)")
        print(f"  Total classified: {len(traj_counted)} / {n_traj}")
        print(f"  Transmitted: {traj_transmitted} ({T_traj:.6f})")
        print(f"  Reflected:   {traj_reflected} ({R_traj:.6f})")
        print(f"  T + R = {T_traj + R_traj:.6f}")
        print(f"  Boundaries: x_left={TRANSMISSION_X_LEFT}, x_right={TRANSMISSION_X_RIGHT}, barrier={STEP_CENTER}")
    else:
        print("\n[TRAJECTORIES] Disabled - trajectory-based T/R not available")
    
    if COMPUTE_TRAJECTORIES and n_traj > 0:
        if T_theory is not None:
            methods = ['Wavefunction\n(steady)', 'Trajectories', 'Theory (WKB)']
            T_vals = [T_steady, T_traj, T_theory]
            R_vals = [R_steady, R_traj, R_theory]
        else:
            methods = ['Wavefunction\n(steady)', 'Trajectories']
            T_vals = [T_steady, T_traj]
            R_vals = [R_steady, R_traj]
    else:
        # No trajectories - only show wavefunction and theory
        if T_theory is not None:
            methods = ['Wavefunction\n(steady)', 'Theory (WKB)']
            T_vals = [T_steady, T_theory]
            R_vals = [R_steady, R_theory]
        else:
            methods = ['Wavefunction\n(steady)']
            T_vals = [T_steady]
            R_vals = [R_steady]

    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time evolution
    ax1.plot(t_arr, T_arr, label='T (wavefunction)', lw=2, color='blue')
    ax1.plot(t_arr, R_arr, label='R (wavefunction)', lw=2, color='orange')
    ax1.plot(t_arr, T_arr + R_arr, '--', label='T + R', lw=1, alpha=0.7, color='green')
    ax1.axhline(T_steady, color='blue', ls=':', alpha=0.5, label=f'T steady = {T_steady:.4f}')
    ax1.axhline(R_steady, color='orange', ls=':', alpha=0.5, label=f'R steady = {R_steady:.4f}')
    if T_theory is not None:
        ax1.axhline(T_theory, color='red', ls='--', alpha=0.7, lw=2, label=f'T theory = {T_theory:.2e}')
        ax1.axhline(R_theory, color='purple', ls='--', alpha=0.7, lw=2, label=f'R theory = {R_theory:.2e}')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Coefficient', fontsize=12)
    ax1.set_title('Transmission/Reflection vs Time', fontsize=13)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)
    
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax2.bar(x_pos - width/2, T_vals, width, label='Transmission', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, R_vals, width, label='Reflection', color='orange', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Coefficient', fontsize=12)
    ax2.set_title('T/R Comparison', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    # Add values on bars
    for i, (t, r) in enumerate(zip(T_vals, R_vals)):
        if t > 0.001:
            ax2.text(i - width/2, t + 0.02, f'{t:.4f}', ha='center', fontsize=8)
        else:
            ax2.text(i - width/2, max(t, 0.001) + 0.02, f'{t:.2e}', ha='center', fontsize=8)
        if r > 0.001:
            ax2.text(i + width/2, r + 0.02, f'{r:.4f}', ha='center', fontsize=8)
        else:
            ax2.text(i + width/2, max(r, 0.001) + 0.02, f'{r:.2e}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'transmission_reflection.png'), dpi=200)
    plt.draw()  # Force draw
    plt.pause(0.001)  # Brief pause to ensure display
    
    
#%% =========== Equilibration analysis ===========
if DO_RIGHT_IMBALANCE and DO_TIME_AVG_DENSITY and is_evanescent:
    print("\n" + "="*70)
    print("SPATIAL IMBALANCE EQUILIBRATION (within ROI)")
    print("="*70)
    
    # ========================================================================
    # Get time-averaged density in ROI using the actual counter
    # ========================================================================
    
    # density_count_roi is the actual number of samples accumulated
    n_samples = density_count_roi
    
    print(f"Using time-averaged density (accumulated over {n_samples} samples)")
    
    # Normalize density_sum_roi to get actual time-averaged density
    if PROPAGATOR.upper() == 'FFT':
        rho_roi_avg = cp.asnumpy(density_sum_roi) / float(n_samples)
    else:
        rho_roi_avg = density_sum_roi / float(n_samples)
    
    # Shape: (Ny, ix_roi2 - ix_roi1)
    print(f"Analysis window: ROI x ∈ [{ROI_X1:.1f}, {ROI_X2:.1f}]")
    print(f"  ROI spans {ix_roi2 - ix_roi1} grid points in x")
    print(f"  Shape of time-averaged density: {rho_roi_avg.shape}")
    
    # ========================================================================
    # Get grid arrays and compute 1σ threshold
    # ========================================================================
    
    if PROPAGATOR.upper() == 'FFT':
        x_cpu = cp.asnumpy(x)
        y_cpu = cp.asnumpy(y)
        dy_val = float(dy)
    else:
        x_cpu = x_np.copy()
        y_cpu = y_np.copy()
        dy_val = float(dy_np)
    
    # x-coordinates in ROI
    x_roi = x_cpu[ix_roi1:ix_roi2]
    
    # UPDATED: Use 1σ threshold instead of y=0
   
    #y_threshold_1sigma = -(y0 - 1.0 * sigma_well)
    
    # Find threshold index
    #iy_threshold = np.searchsorted(y_cpu, y_threshold_1sigma)
    iy_threshold = np.searchsorted(y_cpu, y_threshold_physical)
    
    print("\n[THRESHOLD] Using 2σ physical threshold:")
    print(f"  Well center: y₀ = {y0:.4f}")
    print(f"  Well width: σ_well = {sigma_well:.4f}")
    print(f"  Threshold: y = {y_threshold_physical:.4f} (y₀ - 2σ)")

    # ========================================================================
    # Compute spatial imbalance D(x) within ROI with 1σ threshold
    # ========================================================================
    
    n_x_roi = len(x_roi)
    D_spatial = np.zeros(n_x_roi)
    P_upper_arr = np.zeros(n_x_roi)
    P_lower_arr = np.zeros(n_x_roi)
    
    for i in range(n_x_roi):
        rho_col = rho_roi_avg[:, i]  # Column i in ROI
        
        #Split into upper (y>threshold) and lower (y<threshold) regions
        P_upper = np.sum(rho_col[iy_threshold:]) * dy_val
        P_lower = np.sum(rho_col[:iy_threshold]) * dy_val
        
        P_upper_arr[i] = P_upper
        P_lower_arr[i] = P_lower
        
        # Compute imbalance
        P_total = P_upper + P_lower
        if P_total > 1e-12:
            D_spatial[i] = (P_upper - P_lower) / P_total
        else:
            D_spatial[i] = 0.0
    
    # ========================================================================
    # Analyze equilibration within ROI
    # ========================================================================
    
    # Initial value at barrier (x=0)
    # Find index closest to barrier in x_roi
    ix_barrier_in_roi = np.searchsorted(x_roi, 0.0)
    
    if ix_barrier_in_roi >= len(x_roi):
        print("\n[SPATIAL IMBALANCE] WARNING: Barrier not in ROI, cannot analyze equilibration")
        print(f"  ROI: x ∈ [{x_roi[0]:.2f}, {x_roi[-1]:.2f}], Barrier at x=0")
    else:
        D_initial = D_spatial[ix_barrier_in_roi]
        
        print(f"\nImbalance at barrier (x=0): D = {D_initial:+.4f}")
        
        # ========================================================================
        # Try to fit equilibration using sech (hyperbolic secant) form
        # ========================================================================
        # Theory: D(x) = D_final + (D_initial - D_final) * sech(x/L_equil)
        # This allows for asymmetric equilibration (D_final ≠ 0)
        
        # Extract data from barrier onwards
        x_fit_data = x_roi[ix_barrier_in_roi:]
        D_fit_data = D_spatial[ix_barrier_in_roi:]
        P_upper_fit = P_upper_arr[ix_barrier_in_roi:]
        P_lower_fit = P_lower_arr[ix_barrier_in_roi:]
        
        # Shift x to start from 0 at barrier
        x_fit = x_fit_data - x_fit_data[0]
        
        # Only fit where we have significant population (no weighting)
        P_total_fit = P_upper_fit + P_lower_fit
        fit_mask = P_total_fit > 0.00001 * P_total_fit[0]
        
        if np.sum(fit_mask) > 5 and abs(D_initial) > 0.05:
            try:
                
                # Sech functional form (allow asymmetric equilibration)
                def equil_func(x, L_equil, D_final):
                    return D_final + (D_initial - D_final) / np.cosh(x / L_equil)
                
                # Initial guess for L_equil: use theoretical value if available
                if 'kappa0' in globals() and 'kappa1' in globals():
                    Delta_kappa = abs(kappa1 - kappa0)
                    if Delta_kappa > 1e-6:
                        L_guess = 1.0 / Delta_kappa
                    else:
                        L_guess = np.max(x_fit) / 3.0
                else:
                    L_guess = np.max(x_fit) / 3.0
                
                # Perform fit on masked data
                x_fit_masked = x_fit[fit_mask]
                D_fit_masked = D_fit_data[fit_mask]
            
                # Initial guess for D_final
                D_final_guess = 0.0
                
                popt, pcov = curve_fit(equil_func, x_fit_masked, D_fit_masked, 
                                      p0=[L_guess, D_final_guess],
                                      bounds=([0.1, -1.0], [1000.0, 1.0]))
                L_equil_fit = popt[0]
                D_final_fit = popt[1]
                
                # Uncertainty estimates
                L_equil_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0
                D_final_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
                
                # Quality of fit
                D_fit_curve = equil_func(x_fit_masked, L_equil_fit, D_final_fit)
                residuals = D_fit_masked - D_fit_curve
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((D_fit_masked - np.mean(D_fit_masked))**2)
                r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                print(f"\nEquilibration length (sech fit): L_equil = {L_equil_fit:.2f} ± {L_equil_err:.2f}")
                print(f"Equilibrium imbalance: D_final = {D_final_fit:+.4f} ± {D_final_err:.4f}")
                print(f"Fit quality: R² = {r_squared:.4f}")
                print(f"Fit window: x ∈ [{x_fit_masked[0]:.2f}, {x_fit_masked[-1]:.2f}] (barrier to end)")
                
                # Compare to theoretical expectations
                if 'kappa0' in globals() and 'kappa1' in globals():
                    kappa_avg = (kappa0 + kappa1) / 2.0
                    L_decay = 1.0 / (kappa0 + kappa1)
                    
                    print(f"\nComparison to theoretical scales:")
                    print(f"  L_decay = 1/(κ₀+κ₁) = {L_decay:.2f}")
                    print(f"  L_equil_fit / L_decay = {L_equil_fit / L_decay:.2f}")
                    
                    Delta_kappa = abs(kappa1 - kappa0)
                    if Delta_kappa > 1e-6:
                        L_equil_theory = 1.0 / Delta_kappa
                        print(f"  L_equil_theory = 1/|Δκ| = {L_equil_theory:.2f}")
                        print(f"  L_fit / L_theory = {L_equil_fit / L_equil_theory:.2f}")
                        
                        if abs(L_equil_fit - L_equil_theory) / L_equil_theory < 0.3:
                            print(f"  ✓ Good agreement with theory!")
                        else:
                            print(f"  ⚠ Significant deviation from 1/Δκ theory")
                    
                    if L_equil_fit < L_decay:
                        print(f"\n  → Equilibration is FASTER than decay")
                        print(f"     (Strong coupling between waveguides)")
                    elif L_equil_fit < 2 * L_decay:
                        print(f"\n  → Equilibration comparable to decay")
                    else:
                        print(f"\n  → Equilibration is SLOWER than decay")
                        print(f"     (Weak coupling between waveguides)")
                
                fit_success = True
                
            except Exception as e:
                print(f"\n⚠ Could not fit equilibration: {e}")
                print("  (Data may be too noisy or window too short)")
                fit_success = False
        else:
            if abs(D_initial) <= 0.05:
                print("\n→ Initial imbalance too small (<5%) - system already balanced")
            else:
                print(f"\n⚠ Not enough data points for fitting (need >5, have {np.sum(fit_mask)})")
            fit_success = False
        
        # ========================================================================
        # Plot results
        # ========================================================================
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top panel: Upper vs Lower populations
        ax1.semilogy(x_roi, P_upper_arr, 'r-', lw=2, 
                    label=f'P_upper (y > {y_threshold_physical:.2f})')
        ax1.semilogy(x_roi, P_lower_arr, 'b-', lw=2, 
                    label=f'P_lower (y < {y_threshold_physical:.2f})')
        ax1.axvline(STEP_CENTER, color='gray', ls='--', alpha=0.5, label='Barrier (x=0)')
        ax1.axvline(ROI_X1, color='cyan', ls=':', alpha=0.5)
        ax1.axvline(ROI_X2, color='cyan', ls=':', alpha=0.5, label='ROI bounds')
        
        # Add theoretical equilibration length marker
        if 'L_equil' in globals():
            ax1.axvline(L_equil, color='orange', ls='-.', lw=1.5, alpha=0.7, 
                       label=f'L_equil (theory)={L_equil:.2f}')
        
        ax1.set_xlabel('Position x', fontsize=12)
        ax1.set_ylabel('Population (time-averaged)', fontsize=12)
        ax1.set_title(f'Upper vs Lower Waveguide Populations (2σ threshold)', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Bottom panel: Imbalance D(x)
        ax2.plot(x_roi, D_spatial, 'ko-', markersize=4, lw=1.5, alpha=0.7,
                label='D(x) = (P_upper - P_lower) / (P_upper + P_lower)')
        
        if fit_success:
            # Plot fitted equilibration curve
            x_fit_dense = np.linspace(0, x_fit[-1], 200)
            D_fit_dense = equil_func(x_fit_dense, L_equil_fit, D_final_fit)
            ax2.plot(x_fit_dense, D_fit_dense, 'r--', lw=2,
                    label=f'Sech fit: L_equil={L_equil_fit:.2f}, R²={r_squared:.3f}')
            
            # Add theory line if available
            if 'L_equil' in globals():
                D_theory_dense = equil_func(x_fit_dense, L_equil, D_final_fit)
                ax2.plot(x_fit_dense, D_theory_dense, 'g:', lw=2,
                        label=f'Theory: L_equil={L_equil:.2f}')
        
        ax2.axhline(0, color='gray', ls=':', alpha=0.5, label='D=0 (balanced)')
        ax2.axvline(STEP_CENTER, color='gray', ls='--', alpha=0.5, label='Barrier (x=0)')
        ax2.axvline(ROI_X1, color='cyan', ls=':', alpha=0.5)
        ax2.axvline(ROI_X2, color='cyan', ls=':', alpha=0.5)
        
        # Add theoretical equilibration length marker
        if 'kappa0' in globals() and 'kappa1' in globals():
            Delta_kappa = abs(kappa1 - kappa0)
            if Delta_kappa > 1e-6:
                L_equil_theory = 1.0 / Delta_kappa
                ax2.axvline(L_equil_theory, color='orange', ls='-.', lw=1.5, alpha=0.7, 
                           label=f'L_equil (theory)={L_equil_theory:.2f}')
        
        ax2.set_xlabel('Position x', fontsize=12)
        ax2.set_ylabel('Imbalance D(x)', fontsize=12)
        ax2.set_title(f'Spatial Population Imbalance (2σ threshold: y={y_threshold_physical:.2f})', 
                     fontsize=13)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(alpha=0.3)
        ax2.set_ylim([-1.1, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'spatial_imbalance_analysis.png'), dpi=200)
        plt.draw()
        plt.pause(0.001)
        
        print("="*70 + "\n")

elif not DO_RIGHT_IMBALANCE:
    print("\n[SPATIAL IMBALANCE] Requires DO_RIGHT_IMBALANCE = True")
elif not DO_TIME_AVG_DENSITY:
    print("\n[SPATIAL IMBALANCE] Requires DO_TIME_AVG_DENSITY = True")
    print("  (Time-averaged density needed for robust statistics)")


# Full trajectory plot (with smart sampling)
if FULL_TRAJ_PLOT:
    if not STORE_BULK_TRAJECTORIES or Xhist_bulk is None:
        print("\n[FULL_TRAJ] WARNING: Bulk trajectory storage disabled, cannot plot full trajectories")
        print("[FULL_TRAJ] Enable by setting FULL_TRAJ_PLOT=True or DO_MEAN_TEST=True")
    else:
        print("\n" + "="*60)
        print("FULL TRAJECTORY PLOT")
        print("="*60)
        
        # Determine which trajectories to plot
        if n_traj > FULL_TRAJ_MAX:
            # Randomly sample trajectories
            np.random.seed(42)  # Reproducible sampling
            traj_indices = np.random.choice(n_traj, size=FULL_TRAJ_MAX, replace=False)
            traj_indices = np.sort(traj_indices)  # Sort for cleaner plotting
            print(f"[FULL_TRAJ] Sampling {FULL_TRAJ_MAX} trajectories from {n_traj} total")
        else:
            traj_indices = np.arange(n_traj)
            print(f"[FULL_TRAJ] Plotting all {n_traj} trajectories")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot sampled trajectories
        for idx in traj_indices:
            x_traj = Xhist_bulk[:, idx]
            y_traj = Yhist_bulk[:, idx]
            
            # Remove NaN values (trajectories are stored until they're computed)
            valid = ~np.isnan(x_traj)
            if np.sum(valid) > 1:  # Need at least 2 points
                ax.plot(x_traj[valid], y_traj[valid], lw=0.4, alpha=0.4, color='steelblue')
        
        # Mark key locations
        ax.axvline(0, color='red', ls='--', lw=2, alpha=0.7, label='Barrier (x=0)')
        ax.axvline(ROI_X1, color='cyan', ls='--', lw=1.5, alpha=0.6, label=f'ROI: [{ROI_X1}, {ROI_X2}]')
        ax.axvline(ROI_X2, color='cyan', ls='--', lw=1.5, alpha=0.6)
        ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
        ax.axhline(y0, color='green', ls=':', lw=1, alpha=0.5, label=f'Wells: y=±{y0}')
        ax.axhline(-y0, color='green', ls=':', lw=1, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        if n_traj > FULL_TRAJ_MAX:
            ax.set_title(f'Bohmian Trajectories (sampled {FULL_TRAJ_MAX}/{n_traj})', 
                        fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'Bohmian Trajectories (all {n_traj})', fontsize=13, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Set reasonable limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'full_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.draw()
        plt.pause(0.001)
        
        print(f"[FULL_TRAJ] Plot saved as 'full_trajectories.png'")

# Penetration analysis (adapts to regime)
if Xhist_bulk is not None:
    print("\n" + "="*60)
    print("PENETRATION ANALYSIS")
    print("="*60)

    max_x_per_traj = np.nanmax(Xhist_bulk, axis=0)
    positive_penetrators = max_x_per_traj[max_x_per_traj > 0]

    if len(positive_penetrators) == 0:
        print("[WARNING] No trajectories penetrated into x>0!")
    else:
        
        if is_evanescent and 'L_decay' in globals():
            # Evanescent regime: show in units of decay length
            print(f"[REGIME] Evanescent - analyzing in units of L_decay = {L_decay:.2f}")
            for x_threshold in [5, 10, 15, 20, 25]:
                n_reach = np.sum(max_x_per_traj > x_threshold)
                n_decays = x_threshold / L_decay
                print(f"x > {x_threshold:2d} ({n_decays:.1f} L_decay): {n_reach:5d} trajectories "
                      f"({100*n_reach/n_traj:.4f}%)")
        else:
            # Propagative or mixed regime: show absolute distances
            print(f"[REGIME] {'Propagative' if is_propagative else 'Mixed'} - analyzing absolute penetration")
            for x_threshold in [50, 100, 200, 400, 600, 800]:
                if x_threshold > float(x_max):
                    break
                n_reach = np.sum(max_x_per_traj > x_threshold)
                print(f"x > {x_threshold:4d}: {n_reach:5d} trajectories ({100*n_reach/n_traj:.2f}%)")

        max_penetration = np.max(positive_penetrators)
        print(f"\nMaximum penetration: x = {max_penetration:.2f}")
        


        # Adaptive histogram
        plt.figure(figsize=(10,5))
        if is_evanescent and max_penetration < 50:
            bins = np.linspace(0, min(30, max_penetration + 2), 60)
            plt.hist(positive_penetrators, bins=bins, alpha=0.7, log=True, 
                    edgecolor='black', linewidth=0.5)
            plt.ylabel("Number of trajectories (log scale)", fontsize=12)
            if 'L_decay' in globals():
                plt.axvline(L_decay, color='r', ls='--', lw=2, label=f'L_decay={L_decay:.2f}')
            plt.title(f"Evanescent Penetration Distribution", fontsize=13)
        else:
            bins = np.linspace(0, min(max_penetration + 50, 1200), 60)
            plt.hist(positive_penetrators, bins=bins, alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
            plt.ylabel("Number of trajectories", fontsize=12)
            if not is_evanescent:
                plt.axvline(x_max_center, color='r', ls='--', lw=1.5, 
                           label=f'Expected: <x>_max ≈ {x_max_center:.0f}')
            plt.title(f"Penetration Distribution ({'Propagative' if is_propagative else 'Mixed'} Regime)", 
                     fontsize=13)
        
        plt.xlabel("Maximum x reached", fontsize=12)
        
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend(loc='upper right')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'penetration_histogram.png'), dpi=150)
        plt.draw()
        plt.pause(0.001)
else:
    print("\n[PENETRATION] Skipped (bulk trajectory storage disabled)")

# Time-averaged density with EXPERIMENTAL-STYLE VISUALIZATION (adaptive to regime)
if DO_TIME_AVG_DENSITY and density_count_roi > 0:
    
    print("\n" + "="*60)
    print(f"TIME-AVERAGED DENSITY IN ROI ({regime_name.upper()} REGIME)")
    print("="*60)
    
    from matplotlib.colors import LogNorm, PowerNorm
    
    if barrier_hit_step is not None:
        t_start = averaging_start_step * dt
        t_end = averaging_end_step * dt
        print(f"[TIME_AVG] Window: t = {t_start:.2f} to {t_end:.2f}")
    else:
        print(f"[TIME_AVG] WARNING: Barrier hit never detected!")
        t_start = (n_steps - density_count_roi) * dt
        t_end = n_steps * dt
    
    print(f"[TIME_AVG] Averaged over {density_count_roi} snapshots")
    
    if PROPAGATOR.upper() == 'FFT':
        density_avg_roi = cp.asnumpy(density_sum_roi / density_count_roi)
    else:
        density_avg_roi = density_sum_roi / density_count_roi
    
    x_roi = x_cpu_full[ix_roi1:ix_roi2]
    y_plot = np.linspace(float(y_min), float(y_max), Ny)
    
    # Create figure with multiple visualizations (like experimental paper)
    fig = plt.figure(figsize=(14, 10))
    
    # ========== PANEL 1: Linear scale (like experimental imaging) ==========
    ax1 = plt.subplot(3, 1, 1)
    
    # Use linear scale with good contrast
    vmax = np.max(density_avg_roi)
    vmin = 0.0  # Start from zero for linear
    
    im1 = ax1.imshow(density_avg_roi, 
                     extent=[x_roi[0], x_roi[-1], y_plot[0], y_plot[-1]],
                     origin='lower', aspect='auto', cmap='hot',  # 'hot' like experimental
                     vmin=vmin, vmax=vmax*0.8)  # Saturate at 80% for contrast
    
    plt.colorbar(im1, ax=ax1, label='Density (linear scale)')
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title(f'Time-Averaged Density - LINEAR SCALE ({regime_name} regime)\nt={t_start:.1f}–{t_end:.1f}',
                  fontsize=12, fontweight='bold')
    ax1.axhline(0, color='cyan', ls='--', lw=0.5, alpha=0.5)  # Mark y=0
    
    # ========== PANEL 2: Power-law scale (best for double-well structure) ==========
    ax2 = plt.subplot(3, 1, 2)
    
    # Power-law normalization (gamma < 1 enhances low values)
    im2 = ax2.imshow(density_avg_roi,
                     extent=[x_roi[0], x_roi[-1], y_plot[0], y_plot[-1]],
                     origin='lower', aspect='auto', cmap='inferno',
                     norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))  # gamma=0.5 is sqrt scale
    
    plt.colorbar(im2, ax=ax2, label='Density (sqrt scale)')
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('SQRT SCALE (enhances double-well structure)', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='cyan', ls='--', lw=0.5, alpha=0.5)
    
    # ========== PANEL 3: Log scale (for exponential decay) ==========
    ax3 = plt.subplot(3, 1, 3)
    
    # Filter noise for log scale
    noise_threshold = vmax * 1e-12
    density_significant = density_avg_roi[density_avg_roi > noise_threshold]
    
    if density_significant.size > 0:
        vmin_log = np.percentile(density_significant, 1.0)
        vmin_log = np.clip(vmin_log, vmax / 1e6, vmax / 1e4)
    else:
        vmin_log = vmax / 1e6
    
    im3 = ax3.imshow(density_avg_roi,
                     extent=[x_roi[0], x_roi[-1], y_plot[0], y_plot[-1]],
                     origin='lower', aspect='auto', cmap='viridis',
                     norm=LogNorm(vmin=vmin_log, vmax=vmax))
    
    plt.colorbar(im3, ax=ax3, label='Density (log scale)', extend='both')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title('LOG SCALE (exponential decay)', fontsize=12, fontweight='bold')
    ax3.axhline(0, color='cyan', ls='--', lw=0.5, alpha=0.5)
    
    print(f"[TIME_AVG] Linear scale range: 0 to {vmax:.3e}")
    print(f"[TIME_AVG] Log scale range: {vmin_log:.3e} to {vmax:.3e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'time_averaged_density_roi_multiview.png'), dpi=300)
    plt.draw()
    plt.pause(0.001)
    
    # ========== Y-integrated density (adaptive analysis) ==========
    fig2, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Integrate over y to see x-dependence
    density_vs_x = np.sum(density_avg_roi, axis=0) * float(dy)
    ax_x.semilogy(x_roi, density_vs_x, lw=2, color='darkblue')
    ax_x.set_xlabel('x', fontsize=12)
    ax_x.set_ylabel('∫|ψ|² dy (log scale)', fontsize=12)
    ax_x.grid(alpha=0.3)
    ax_x.axvline(0, color='red', ls='--', lw=2, alpha=0.7, label='Barrier')
    
    # Regime-adaptive analysis
    x_pos_mask = x_roi > 0
    if np.sum(x_pos_mask) > 10:
        x_pos = x_roi[x_pos_mask]
        rho_pos = density_vs_x[x_pos_mask]
        peak_density = np.max(density_vs_x)
        
        if is_evanescent:
            # Evanescent: fit exponential decay
            ax_x.set_title('Exponential Decay vs Position', fontsize=13, fontweight='bold')
            
            x_fit_all = x_roi[x_pos_mask]
            rho_fit_all = density_vs_x[x_pos_mask]
            
            # Define fitting window - skip boundary region, focus on clean exponential
            if 'L_decay' in globals():
                x_fit_min = 0.5 * L_decay  # Skip first half decay length (boundary effects)
                x_fit_max = 6.0 * L_decay  # Fit over ~5.5 decay lengths
                rho_min = peak_density * 1e-8  # Minimum density threshold to avoid noise
                
                fit_mask = ((x_fit_all >= x_fit_min) & 
                           (x_fit_all <= x_fit_max) & 
                           (rho_fit_all > rho_min))
                
                print(f"\n[DECAY FIT] Theory decay length: L = {L_decay:.4f} (code) = {L_decay*L0*1e6:.3f} μm")
                print(f"[DECAY FIT] Fit window: x ∈ [{x_fit_min:.2f}, {x_fit_max:.2f}]")
            else:
                # Fallback if L_decay not available
                fit_window_low = peak_density * 1e-6
                fit_window_high = peak_density * 1e-2
                fit_mask = ((rho_fit_all > fit_window_low) & 
                           (rho_fit_all < fit_window_high) & 
                           (rho_fit_all > 0))
                print(f"\n[DECAY FIT] Using density-based window (L_decay not available)")
            
            if np.sum(fit_mask) > 10:  # Need at least 10 points for reliable fit
                x_fit = x_fit_all[fit_mask]
                rho_fit = rho_fit_all[fit_mask]
                
                # Fit: log(ρ) = a + b×x  →  ρ = exp(a)×exp(b×x)
                # Decay length: L = -1/b
                coeffs = np.polyfit(x_fit, np.log(rho_fit), 1)
                slope_b = coeffs[0]
                intercept_a = coeffs[1]
                L_decay_measured = -1.0 / slope_b
                
                # Compute R² for fit quality
                log_rho_fit = np.log(rho_fit)
                log_rho_pred = np.polyval(coeffs, x_fit)
                ss_res = np.sum((log_rho_fit - log_rho_pred)**2)
                ss_tot = np.sum((log_rho_fit - np.mean(log_rho_fit))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # Get density at x=0 for normalization
                rho_at_zero = np.interp(0, x_fit_all, rho_fit_all)
                
                # Plot fitted curve: ρ(x) = ρ(0) × exp(-x/L_measured)
                x_plot = np.linspace(0, min(x_fit[-1]*1.2, x_fit_all[-1]), 100)
                rho_fit_curve = rho_at_zero * np.exp(-x_plot / L_decay_measured)
                
                if 'L_decay' in globals():
                    ratio = L_decay_measured / L_decay
                    fit_label = f'Fit: L={L_decay_measured:.3f} (R²={r_squared:.4f})'
                else:
                    fit_label = f'Fit: L={L_decay_measured:.3f} (R²={r_squared:.4f})'
                
                ax_x.plot(x_plot, rho_fit_curve, '--', 
                         color='orange', lw=2.5, label=fit_label, zorder=10)
                
                # Show fit window
                ax_x.axvspan(x_fit[0], x_fit[-1], alpha=0.15, color='orange', 
                            label=f'Fit region')
                
                # Plot theoretical curve if available
                if 'L_decay' in globals():
                    rho_theory_curve = rho_at_zero * np.exp(-x_plot / L_decay)
                    ax_x.plot(x_plot, rho_theory_curve, ':', 
                             color='red', lw=2, alpha=0.7,
                             label=f'Theory: L={L_decay:.3f}', zorder=9)
                    
                    # Mark one decay length
                    if L_decay < x_fit_all[-1]:
                        ax_x.axvline(L_decay, color='gray', ls=':', alpha=0.5, lw=1,
                                   label=f'1×L (theory)')
                
                # Print diagnostics
                print(f"[DECAY FIT] Measured: L = {L_decay_measured:.4f} (code) = {L_decay_measured*L0*1e6:.3f} μm")
                print(f"[DECAY FIT] Fit quality: R² = {r_squared:.6f}")
                print(f"[DECAY FIT] Fit points: {np.sum(fit_mask)}")
                print(f"[DECAY FIT] ρ(0) = {rho_at_zero:.3e}")
                
                if 'L_decay' in globals():
                    ratio = L_decay_measured / L_decay
                    percent_error = abs(ratio - 1.0) * 100
                    print(f"[DECAY FIT] Measured/Theory = {ratio:.4f} ({percent_error:.2f}% error)")
                    
                    if r_squared > 0.99 and abs(ratio - 1.0) < 0.1:
                        print(f"[DECAY FIT] ✓ Excellent: High R² and <10% error")
                    elif r_squared > 0.95 and abs(ratio - 1.0) < 0.2:
                        print(f"[DECAY FIT] ✓ Good: Reasonable agreement")
                    elif r_squared > 0.90:
                        print(f"[DECAY FIT] ~ Acceptable: R² good but some deviation")
                    else:
                        print(f"[DECAY FIT] ⚠ Poor fit quality - possible issues:")
                        if r_squared < 0.90:
                            print(f"             • Low R²: Decay may not be exponential (oscillations?)")
                        if abs(ratio - 1.0) > 0.3:
                            print(f"             • Large error: Check grid resolution or regime")
                        print(f"             • Verify: dx < L_decay/10 = {L_decay/10:.4f}")
                        print(f"             • Current: dx ≈ {np.mean(np.diff(x_fit_all)):.4f}")
                
                # Check for residual structure (non-exponential behavior)
                residuals = log_rho_fit - log_rho_pred
                residual_std = np.std(residuals)
                if residual_std > 0.3:
                    print(f"[DECAY FIT] ⚠ Large residual std dev ({residual_std:.3f}) - may have oscillations")
                
            else:
                print(f"[DECAY FIT] ⚠ Insufficient points for fit ({np.sum(fit_mask)} < 10)")
                print(f"[DECAY FIT]    Increase ROI range or check if density is too low")
                
        else:
            # Propagative or mixed: just show distribution
            ax_x.set_title('Transmitted Wave Distribution vs Position', fontsize=13, fontweight='bold')
            print(f"\n[TRANSMISSION] {regime_name} regime - peak density: {peak_density:.3e}")
            if is_propagative:
                print(f"[TRANSMISSION] Note: Expect oscillatory structure, not exponential decay")
        
        ax_x.legend(fontsize=9, loc='best')
    
    # IMPROVED: Integrate over x > L_equil to see equilibrated double-well structure
    if 'L_equil' in globals() and is_evanescent:
        # Filter x > L_equil for equilibrated region
        x_equil_mask = x_roi > L_equil
        if np.sum(x_equil_mask) > 0:
            density_vs_y = np.sum(density_avg_roi[:, x_equil_mask], axis=1) * float(dx)
            title_suffix = f' (x > L_equil={L_equil:.1f})'
            print(f"[DOUBLE-WELL] Integrating over equilibrated region: x > {L_equil:.2f}")
        else:
            # Fallback if not enough data past L_equil
            density_vs_y = np.sum(density_avg_roi, axis=1) * float(dx)
            title_suffix = ' (full domain)'
            print(f"[DOUBLE-WELL] WARNING: No data past L_equil={L_equil:.2f}, using full domain")
    else:
        # Non-evanescent or L_equil not defined: use full domain
        density_vs_y = np.sum(density_avg_roi, axis=1) * float(dx)
        title_suffix = ''
    
    ax_y.plot(density_vs_y, y_plot, lw=2, color='darkred')
    ax_y.set_ylabel('y', fontsize=12)
    ax_y.set_xlabel('∫|ψ|² dx', fontsize=12)
    ax_y.set_title('Double-Well Population' + title_suffix, fontsize=13, fontweight='bold')
    ax_y.grid(alpha=0.3)
    ax_y.axhline(0, color='blue', ls='--', lw=2, alpha=0.7, label='y=0 (barrier)')
    ax_y.legend()
    
    # Mark the well locations
    ax_y.axhspan(-y0, -y0+2, alpha=0.2, color='green', label='Lower well')
    ax_y.axhspan(y0-2, y0, alpha=0.2, color='purple', label='Upper well')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exponential_decay.png'), dpi=200)
    plt.draw()
    plt.pause(0.001)

if DO_TIME_AVG_DENSITY and density_count_roi > 0 and is_propagative:

    # ============================================================================
    # PUBLICATION DENSITY + TRAJECTORIES PLOT (Two Panels, Matched)
    # ============================================================================
    
    print("\n" + "="*60)
    print("CREATING PUBLICATION DENSITY + TRAJECTORIES PLOT (Two Panels)")
    print("="*60)
    
    # Check if density data exists
    if 'density_sum_roi' not in globals() or density_count_roi == 0:
        print("[ERROR] Time-averaged density data not available!")
        print("        Set DO_TIME_AVG_DENSITY=True and re-run simulation")
    elif len(traj_hist_X_roi) == 0:
        print("[ERROR] No ROI trajectory data available!")
    else:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
        # ========================================================================
        # CONFIGURATION
        # ========================================================================
    
        # Local micrometres-per-code-unit factor. Deliberately NOT named L0: the
        # module-level L0 is metres per code unit (2e-6) and must stay that way for
        # ANALYZE_TUNNELING_SPEED / ANALYZE_RHO_A further down. Same name and value
        # as the CODE_TO_UM used in the later plotting blocks.
        CODE_TO_UM = 2.0  # 1 code unit = 2 micrometers
        TRAJ_SAMPLE_MAX = 300
    
        # Get viridis colors
        viridis = plt.cm.viridis
        VIRIDIS_ZERO = viridis(0.0)       # Dark purple for background
        TRAJ_COLOR = viridis(0.65)        # Cyan-green from high end of viridis
        TRAJ_ALPHA = 0.5
        TRAJ_LW = 0.6
    
        # ========================================================================
        # PREPARE DENSITY DATA
        # ========================================================================
    
        if PROPAGATOR.upper() == 'FFT':
            import cupy as cp
            rho_avg = cp.asnumpy(density_sum_roi / density_count_roi)
        else:
            rho_avg = density_sum_roi / density_count_roi
    
        # Get spatial coordinates
        x_cpu_full = np.linspace(float(x_min), float(x_max), Nx)
        x_roi = x_cpu_full[ix_roi1:ix_roi2]
        y_plot = np.linspace(float(y_min), float(y_max), Ny)
    
        # Convert to physical units (micrometers)
        x_roi_um = x_roi * CODE_TO_UM
        y_plot_um = y_plot * CODE_TO_UM
    
        print(f"[DENSITY] Time-averaged over {density_count_roi} snapshots")
        print(f"[RANGE] x ∈ [{x_roi_um[0]:.1f}, {x_roi_um[-1]:.1f}] μm")
        print(f"[RANGE] y ∈ [{y_plot_um[0]:.1f}, {y_plot_um[-1]:.1f}] μm")
    
        # ========================================================================
        # SAMPLE TRAJECTORIES
        # ========================================================================
    
        traj_indices = list(traj_hist_X_roi.keys())
        n_total = len(traj_indices)
    
        if n_total > TRAJ_SAMPLE_MAX:
            import random
            random.seed(42)
            traj_sample = random.sample(traj_indices, TRAJ_SAMPLE_MAX)
            traj_label = f'{TRAJ_SAMPLE_MAX}/{n_total}'
            print(f"[TRAJ] Sampled {TRAJ_SAMPLE_MAX} of {n_total} trajectories")
        else:
            traj_sample = traj_indices
            traj_label = f'{n_total}'
            print(f"[TRAJ] Using all {n_total} trajectories")
    
        # ========================================================================
        # COMMON LIMITS
        # ========================================================================
    
        y0_um = y0 * CODE_TO_UM
        y_min_plot = -15 * CODE_TO_UM
        y_max_plot = 15 * CODE_TO_UM
        x_min_plot = x_roi_um[0]
        x_max_plot = x_roi_um[-1]
    
        # ========================================================================
        # CREATE FIGURE (Two Panels, Matched)
        # ========================================================================
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    
        # ========================================================================
        # TOP PANEL: DENSITY
        # ========================================================================
    
        vmax = np.max(rho_avg)
        vmin = 0.0
    
        im = ax1.imshow(rho_avg, 
                        extent=[x_roi_um[0], x_roi_um[-1], y_plot_um[0], y_plot_um[-1]],
                        origin='lower', 
                        aspect='auto', 
                        cmap='viridis',
                        vmin=vmin, 
                        vmax=vmax)
    
        # Colorbar using make_axes_locatable for consistent sizing
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax1, label=r'$|\psi|^2$ (time-averaged)')
        cbar.ax.tick_params(labelsize=10)
    
        # Reference lines (white for visibility on viridis)
        ax1.axhline(y0_um, color='white', ls='--', lw=1.5, alpha=0.8)
        ax1.axhline(-y0_um, color='white', ls='--', lw=1.5, alpha=0.8)
        ax1.axhline(0, color='white', ls=':', lw=1.0, alpha=0.6)
    
        ax1.set_xlim(x_min_plot, x_max_plot)
        ax1.set_ylim(y_min_plot, y_max_plot)
        ax1.set_ylabel('y (μm)', fontsize=12)
        ax1.set_title('(a) Time-Averaged Wavepacket Density', fontsize=13, fontweight='bold')
        ax1.tick_params(labelbottom=False)
        ax1.grid(False)
    
        # ========================================================================
        # BOTTOM PANEL: TRAJECTORIES
        # ========================================================================
    
        # Set background to match viridis zero
        ax2.set_facecolor(VIRIDIS_ZERO)
    
        # Plot trajectories
        for traj_idx in traj_sample:
            x_traj = np.array(traj_hist_X_roi[traj_idx])
            y_traj = np.array(traj_hist_Y_roi[traj_idx])
    
            if len(x_traj) < 2:
                continue
    
            # Convert to physical units
            x_um = x_traj * CODE_TO_UM
            y_um = y_traj * CODE_TO_UM
    
            ax2.plot(x_um, y_um, color=TRAJ_COLOR, lw=TRAJ_LW, alpha=TRAJ_ALPHA)
    
        # Reference lines (white for visibility on purple)
        ax2.axhline(y0_um, color='white', ls='--', lw=1.5, alpha=0.8)
        ax2.axhline(-y0_um, color='white', ls='--', lw=1.5, alpha=0.8)
        ax2.axhline(0, color='white', ls=':', lw=1.0, alpha=0.6)
    
        # Match limits exactly
        ax2.set_xlim(x_min_plot, x_max_plot)
        ax2.set_ylim(y_min_plot, y_max_plot)
        ax2.set_xlabel('x (μm)', fontsize=12)
        ax2.set_ylabel('y (μm)', fontsize=12)
        ax2.set_title(f'(b) Bohmian Trajectories ({traj_label})', fontsize=13, fontweight='bold')
        ax2.grid(False)
    
        # Add matching axes space on right (for alignment with colorbar)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="2%", pad=0.1)
        cax2.axis('off')  # Hide the dummy axes
    
        # Legend in the dummy colorbar space area
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=TRAJ_COLOR, lw=1.5, alpha=0.8, label='Trajectories'),
            Line2D([0], [0], color='white', ls='--', lw=1.5, label=f'WG centers (±{y0_um:.0f} μm)'),
            Line2D([0], [0], color='white', ls=':', lw=1.0, label='Barrier center'),
        ]
    
        # Position legend in upper right of the main axes
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                   facecolor=VIRIDIS_ZERO, edgecolor='white', labelcolor='white',
                   framealpha=0.9)
    
        # Adjust layout manually instead of tight_layout
        plt.subplots_adjust(left=0.05, right=0.92, top=0.94, bottom=0.08, hspace=0.15)
    
        # ========================================================================
        # SAVE
        # ========================================================================
    
        # Save PDF (vector)
        save_path_pdf = os.path.join(results_dir, 'density_trajectories_twopanel.pdf')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        
        # Save PNG (preview)
        save_path_png = os.path.join(results_dir, 'density_trajectories_twopanel.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        
        plt.show()
    
        print(f"\n[SAVED] {save_path_pdf}")
        print(f"[SAVED] {save_path_png}")   


if DO_MEAN_TEST:
    plt.figure(figsize=(7,4))
    plt.plot(t_array_bulk, mean_x_psi,  label="<x> ψ", lw=2)
    plt.plot(t_array_bulk, mean_x_trj, "--", label="<x> trajectories", lw=2)
    plt.xlabel("time")
    plt.ylabel("<x>")
    plt.legend(loc='upper right')
    plt.title(f"Mean position ({VELOCITY_METHOD})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mean_position.png'), dpi=150)
    plt.draw()
    plt.pause(0.001)

if DO_RIGHT_IMBALANCE:
    
    # ═══════════════════════════════════════════════════════════════════════
    # POPULATION OSCILLATION ANALYSIS: REGIME-ADAPTIVE
    # ═══════════════════════════════════════════════════════════════════════
    # Propagative: Fit spatial beating pattern D(x) ~ cos(Δk·x + φ)
    # Evanescent: Show temporal evolution D(t)
    # ═══════════════════════════════════════════════════════════════════════
        
    if is_propagative and DO_TIME_AVG_DENSITY and density_count_roi > 0:
        print("\n" + "="*60)
        print("SPATIAL BEATING ANALYSIS (PROPAGATIVE REGIME)")
        print("="*60)
        print(f"[REGIME] Propagative (both channels open)")
        print(f"         E_inj = {E_inj_total:.6f}, V_STEP + E0_R = {V_STEP + E0_R:.6f}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: Compute theoretical Δk
        # ════════════════════════════════════════════════════════════════
        
        print(f"\n[THEORY]")
        
        # Energy defects (kinetic energy in each channel after barrier)
        d0 = E_inj_total - (V_STEP + E0_R)  # Lower channel
        d1 = E_inj_total - (V_STEP + E1_R)  # Upper channel
        
        print(f"  Channel 0 (lower): E_inj - (V + E₀_R) = {d0:.6f}")
        print(f"  Channel 1 (upper): E_inj - (V + E₁_R) = {d1:.6f}")
        
        if d0 > 0 and d1 > 0:
            k0 = np.sqrt(2.0 * d0)
            k1 = np.sqrt(2.0 * d1)
            Delta_k_theory = abs(k1 - k0)
            lambda_beat_theory = 2*np.pi / Delta_k_theory if Delta_k_theory > 0 else np.inf
            
            print(f"  Momentum: k₀ = {k0:.6f}, k₁ = {k1:.6f}")
            print(f"  Spatial beating:")
            print(f"    Δk = |k₁ - k₀| = {Delta_k_theory:.6f}")
            print(f"    λ_beat = 2π/Δk = {lambda_beat_theory:.3f}")
            print(f"  Expected: D(x) ~ cos(Δk·x + φ)")
        else:
            print(f"  WARNING: One or both channels closed!")
            Delta_k_theory = np.nan
            lambda_beat_theory = np.nan
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: Extract D(x) from time-averaged density
        # ════════════════════════════════════════════════════════════════
        
        print(f"\n[EXTRACTING SPATIAL PATTERN]")
        print(f"  Using time-averaged density ({density_count_roi} samples)")
        print(f"  Window: t ∈ [{averaging_start_step*dt:.1f}, {averaging_end_step*dt:.1f}]")
        
        # Get time-averaged density in ROI
        if PROPAGATOR.upper() == 'FFT':
            rho_avg = cp.asnumpy(density_sum_roi / density_count_roi)
        else:
            rho_avg = density_sum_roi / density_count_roi
        
        # Get coordinates
        x_roi = x_cpu_full[ix_roi1:ix_roi2]
        y_cpu_plot = np.linspace(float(y_min), float(y_max), Ny)
        split = int(np.searchsorted(y_cpu_plot, 0.0))
        
        # Compute D(x) at each x-position
        n_x = len(x_roi)
        D_spatial = np.zeros(n_x)
        P_upper_x = np.zeros(n_x)
        P_lower_x = np.zeros(n_x)
        
        dy_val = float(dy if PROPAGATOR.upper() == 'FFT' else dy_np)
        
        for i in range(n_x):
            col = rho_avg[:, i]
            P_upper = np.sum(col[split:]) * dy_val
            P_lower = np.sum(col[:split]) * dy_val
            P_total = P_upper + P_lower
            
            P_upper_x[i] = P_upper
            P_lower_x[i] = P_lower
            
            if P_total > 1e-12:
                D_spatial[i] = (P_upper - P_lower) / P_total
            else:
                D_spatial[i] = 0.0
        
        print(f"  Computed D(x) over {n_x} points: x ∈ [{x_roi[0]:.1f}, {x_roi[-1]:.1f}]")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: Fit spatial beating pattern D(x) = A·cos(Δk·x + φ) + D₀
        # ════════════════════════════════════════════════════════════════
        
        print(f"\n[FITTING SPATIAL PATTERN]")
        
        # Define fitting window (avoid boundary artifacts)
        x_fit_min = 20.0  # Skip near-barrier
        x_fit_max = min(600.0, x_roi[-1])  # Don't go too far
        
        fit_mask = (x_roi >= x_fit_min) & (x_roi <= x_fit_max) & (P_upper_x + P_lower_x > 1e-6)
        
        if np.sum(fit_mask) > 10:
            x_fit = x_roi[fit_mask]
            D_fit = D_spatial[fit_mask]
            
            print(f"  Fit window: x ∈ [{x_fit[0]:.1f}, {x_fit[-1]:.1f}]")
            print(f"  Fit points: {len(x_fit)}")
            
            # Model: D(x) = A·cos(k·x + φ) + D₀
            def spatial_model(x, amplitude, k_spatial, phase, offset):
                return offset + amplitude * np.cos(k_spatial * x + phase)
            
            # Initial guesses
            D_amp_guess = (np.max(D_fit) - np.min(D_fit)) / 2
            D_offset_guess = np.mean(D_fit)
            phase_guess = 0.0
            
            if not np.isnan(Delta_k_theory):
                k_guess = Delta_k_theory
            else:
                # Estimate from FFT
                D_detrend = D_fit - np.mean(D_fit)
                D_fft = np.fft.rfft(D_detrend)
                k_fft = np.fft.rfftfreq(len(D_detrend), d=(x_fit[1] - x_fit[0]))
                power = np.abs(D_fft)**2
                if len(k_fft) > 1:
                    k_guess = k_fft[1:][np.argmax(power[1:])]
                else:
                    k_guess = 0.01
            
            try:
                
                popt, pcov = curve_fit(spatial_model, x_fit, D_fit,
                                      p0=[D_amp_guess, k_guess, phase_guess, D_offset_guess],
                                      bounds=([-1, 0, -2*np.pi, -1], 
                                             [1, 0.5, 2*np.pi, 1]),
                                      maxfev=10000)
                
                A_fit, k_fit, phi_fit, offset_fit = popt
                
                # Compute fitted curve
                D_fitted = spatial_model(x_fit, *popt)
                
                # R² goodness of fit
                residuals = D_fit - D_fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((D_fit - np.mean(D_fit))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                # Extract wavelength
                lambda_beat_measured = 2*np.pi / abs(k_fit) if k_fit != 0 else np.inf
                
                print(f"\n[MEASURED from SPATIAL FIT]")
                print(f"  Spatial frequency:  Δk_measured = {k_fit:.6f}")
                print(f"  Beating wavelength: λ_measured = {lambda_beat_measured:.3f}")
                print(f"  Amplitude:          A = {A_fit:.4f}")
                print(f"  Phase:              φ = {phi_fit:.4f} rad")
                print(f"  Offset:             D₀ = {offset_fit:.4f}")
                print(f"  R² goodness-of-fit: {r2:.4f}")
                
                # Compare to theory
                if not np.isnan(Delta_k_theory):
                    print(f"\n[COMPARISON]")
                    k_ratio = k_fit / Delta_k_theory
                    print(f"  Δk_measured / Δk_theory = {k_ratio:.4f}")
                    
                    if abs(k_ratio - 1.0) < 0.10:
                        print(f"  ✓ Excellent agreement (within 10%)")
                    elif abs(k_ratio - 1.0) < 0.20:
                        print(f"  ✓ Good agreement (within 20%)")
                    else:
                        print(f"  ⚠ Significant discrepancy (>20%)")
                
                if r2 > 0.80:
                    print(f"\n[INTERPRETATION] ✓✓ Excellent fit - clear spatial beating")
                elif r2 > 0.60:
                    print(f"\n[INTERPRETATION] ✓ Good fit - spatial oscillations visible")
                else:
                    print(f"\n[INTERPRETATION] ⚠ Moderate fit - pattern may be complex")
                
                # ════════════════════════════════════════════════════════
                # STEP 4: Create diagnostic plot
                # ════════════════════════════════════════════════════════
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
                
                # Panel 1: D(x) with fit
                ax1.plot(x_roi, D_spatial, 'o', ms=3, alpha=0.5, color='blue', 
                        label='Data: D(x)')
                
                # Fitted curve
                x_plot = np.linspace(x_fit[0], x_fit[-1], 300)
                D_fit_plot = spatial_model(x_plot, *popt)
                ax1.plot(x_plot, D_fit_plot, '-', lw=2.5, color='red',
                        label=f'Fit: Δk={k_fit:.5f}, R²={r2:.3f}')
                
                # Theory curve (if available)
                if not np.isnan(Delta_k_theory):
                    D_theory = offset_fit + A_fit * np.cos(Delta_k_theory * x_plot + phi_fit)
                    ax1.plot(x_plot, D_theory, '--', lw=2, color='green', alpha=0.7,
                            label=f'Theory: Δk={Delta_k_theory:.5f}')
                
                ax1.axhline(0, color='gray', ls=':', alpha=0.5)
                ax1.axvspan(x_fit[0], x_fit[-1], alpha=0.08, color='green', label='Fit region')
                ax1.set_xlabel('Position x', fontsize=12)
                ax1.set_ylabel('D(x) = (P_upper - P_lower) / P_total', fontsize=12)
                
                title_str = 'Spatial Population Beating Pattern\n'
                if not np.isnan(Delta_k_theory):
                    title_str += f'Δk: measured={k_fit:.5f}, theory={Delta_k_theory:.5f} (ratio={k_ratio:.3f})'
                else:
                    title_str += f'Δk_measured = {k_fit:.5f}'
                
                ax1.set_title(title_str, fontsize=12, fontweight='bold')
                ax1.legend(fontsize=10, loc='best')
                ax1.grid(alpha=0.3)
                ax1.set_ylim([-1.1, 1.1])
                
                # Panel 2: Residuals
                ax2.plot(x_fit, residuals, 'o-', ms=3, lw=1, color='purple', alpha=0.7)
                ax2.axhline(0, color='black', ls='-', lw=1, alpha=0.5)
                ax2.fill_between(x_fit, 0, residuals, alpha=0.2, color='purple')
                ax2.set_xlabel('Position x', fontsize=12)
                ax2.set_ylabel('Residuals (Data - Fit)', fontsize=12)
                ax2.set_title(f'Fit Quality: R² = {r2:.4f}', fontsize=12, fontweight='bold')
                ax2.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'spatial_beating_analysis.png'), dpi=200)
                plt.draw()
                plt.pause(0.001)
                
                print(f"\n[SAVED] spatial_beating_analysis.png")
                
            except Exception as e:
                print(f"\n[ERROR] Fitting failed: {e}")
                print(f"        Spatial pattern may be too noisy")
        
        else:
            print(f"\n[ERROR] Insufficient data for fitting ({np.sum(fit_mask)} points)")
    
    else:
        # Evanescent regime or no time-averaged density: show D(t)
        plt.figure(figsize=(8, 4))
        plt.plot(t_array_bulk, D, lw=1.2, color='blue')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("D(t) = (P_R - P_L)/(P_R + P_L)", fontsize=12)
        
        if is_propagative:
            plt.title("Population Imbalance vs Time (propagative regime)\n"
                     "Note: Spatial beating analysis requires time-averaged density", 
                     fontsize=11)
        else:
            plt.title("Population Imbalance vs Time (evanescent regime)", fontsize=12)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'population_imbalance.png'), dpi=200)
        plt.draw()
        plt.pause(0.001)
        
#%% =========== ROI trajectory plot =====================
    
if DO_PLOT_ROI_TURNS and n_roi_entered > 0:  # Removed "and is_evanescent"
        
    print("\n" + "="*60)
    print(f"TRAJECTORY ANALYSIS IN ROI ({regime_name.upper()} REGIME)")
    print("="*60)
    
    from scipy.signal import find_peaks, savgol_filter
    
    # Physical unit conversion
    CODE_TO_UM = 2.0  # 1 code unit = 2 micrometers
    
    def find_turnarounds_hires(x_traj, min_dx_prom=0.10):        
        if len(x_traj) < 10:
            return []
        
        x_arr = np.array(x_traj)
        
        if SMOOTH_X_FOR_TURNS and len(x_arr) > SMOOTH_MIN_WIN_PTS:
            dt_hires = ROI_HIST_STRIDE * dt
            win_pts = max(SMOOTH_MIN_WIN_PTS, int(SMOOTH_SEC_TURNS / dt_hires))
            if win_pts % 2 == 0:
                win_pts += 1
            if win_pts < len(x_arr):
                try:
                    x_smooth = savgol_filter(x_arr, window_length=min(win_pts, len(x_arr)//2*2-1), 
                                            polyorder=2, mode='interp')
                except (ValueError, np.linalg.LinAlgError):
                    x_smooth = x_arr
            else:
                x_smooth = x_arr
        else:
            x_smooth = x_arr
        
        peaks, _ = find_peaks(x_smooth, prominence=min_dx_prom)
        
        if TURN_MODE == 'any_turn_in_ROI':
            valid_peaks = [p for p in peaks if ROI_X1 <= x_arr[p] <= ROI_X2]
        else:
            valid_peaks = [p for p in peaks if x_arr[p] > 0.0]
        
        return valid_peaks
    
    # Regime-aware trajectory filtering
    if is_evanescent:
        # Evanescent: filter to trajectories with turnarounds
        traj_with_turns = []
        for traj_idx in traj_hist_X_roi:
            x_traj = traj_hist_X_roi[traj_idx]
            turns = find_turnarounds_hires(x_traj)
            if len(turns) > 0:
                traj_with_turns.append(traj_idx)
        print(f"[TRAJECTORIES] Found {len(traj_with_turns)} trajectories with turnarounds in ROI")
    else:
        # Propagative/mixed: use all ROI trajectories (no turnarounds expected)
        traj_with_turns = list(traj_hist_X_roi.keys())
        print(f"[TRAJECTORIES] Using all {len(traj_with_turns)} ROI trajectories ({regime_name} regime)")
    
    
    if len(traj_with_turns) > 0:
                if is_evanescent:
                    # Evanescent: migrators are rare. Prioritize all of them and fill the
                    # rest from trajectories with turnarounds.
                    MIGRATION_Y_THRESHOLD = -sigma_well
                    migrating_keys = []
                    for traj_idx in traj_hist_X_roi:
                        y_traj = np.array(traj_hist_Y_roi[traj_idx])
                        if len(y_traj) == 0:
                            continue
                        n_init = min(3, len(y_traj))
                        if np.mean(y_traj[:n_init]) > 0 and np.any(y_traj < MIGRATION_Y_THRESHOLD):
                            migrating_keys.append(traj_idx)
                    
                    print(f"[ROI] {len(migrating_keys)} migrating trajectories identified")
                    
                    MAX_PLOT_TRAJ = 5000
                    must_keep = set(migrating_keys)
                    optional = set(traj_with_turns) - must_keep
                    
                    if len(must_keep) >= MAX_PLOT_TRAJ:
                        import random
                        random.seed(42)
                        traj_roi_subset = random.sample(list(must_keep), MAX_PLOT_TRAJ)
                        print(f"[ROI] All {len(must_keep)} migrating trajectories kept (cap exceeded)")
                    elif len(must_keep) + len(optional) > MAX_PLOT_TRAJ:
                        import random
                        random.seed(42)
                        remaining = MAX_PLOT_TRAJ - len(must_keep)
                        optional_sampled = set(random.sample(list(optional), remaining))
                        traj_roi_subset = list(must_keep | optional_sampled)
                        print(f"[ROI] Saving {len(must_keep)} migrating + "
                              f"{len(optional_sampled)} other = {len(traj_roi_subset)} total")
                    else:
                        traj_roi_subset = list(must_keep | optional)
                        print(f"[ROI] Saving all {len(traj_roi_subset)} trajectories "
                              f"({len(must_keep)} migrating)")
                else:
                    # Propagative / mixed: most trajectories migrate. Prioritizing on a
                    # dip-completion criterion biases the histogram against late
                    # first-crossings whose dips fall past the recording window.
                    # Just sample uniformly from upper-starting trajectories.
                    candidates = []
                    for traj_idx in traj_hist_X_roi:
                        y_traj = np.array(traj_hist_Y_roi[traj_idx])
                        if len(y_traj) == 0:
                            continue
                        n_init = min(3, len(y_traj))
                        if np.mean(y_traj[:n_init]) > 0:
                            candidates.append(traj_idx)
                    
                    MAX_PLOT_TRAJ = 5000
                    if len(candidates) > MAX_PLOT_TRAJ:
                        import random
                        random.seed(42)
                        traj_roi_subset = random.sample(candidates, MAX_PLOT_TRAJ)
                        print(f"[ROI] Sampled {MAX_PLOT_TRAJ} of {len(candidates)} "
                              f"upper-start trajectories (propagative, uniform sample)")
                    else:
                        traj_roi_subset = candidates
                        print(f"[ROI] Saving all {len(traj_roi_subset)} "
                              f"upper-start trajectories (propagative)")
    else:
        traj_roi_subset = []
        
# ============================================================================
# 3-PANEL TRAJECTORY PLOT: x-y (top), x(t) (bottom-left), y(t) migrated (bottom-right)
# ============================================================================

if DO_PLOT_ROI_TURNS and is_evanescent and len(traj_roi_subset) > 0:

    print("\n" + "="*60)
    print("CREATING 3-PANEL TRAJECTORY PLOT")
    print("="*60)

    import matplotlib.gridspec as gridspec

    # Physical unit conversions
    CODE_TO_UM = 2.0  # Spatial

    # Time conversion
    if 'T0' in globals():
        T0_SI = T0
    else:
        # Fallback calculation
        hbar_SI = 1.055e-34
        m_SI = 6.95e-36
        L0_SI = CODE_TO_UM * 1e-6
        E0_SI = hbar_SI**2 / (m_SI * L0_SI**2)
        T0_SI = hbar_SI / E0_SI

    y0_um = y0 * CODE_TO_UM
    NAVY_BLUE = '#003366'

    # Reference line colors (Option 1: Gold/Amber)
    COLOR_WAVEGUIDE = '#DAA520'  # Goldenrod
    COLOR_BARRIER_CENTER = '#FFA500'  # Orange
    COLOR_BARRIER_ENTRANCE = '#FFB300'  # Amber

    # ========================================================================
    # CLASSIFY TRAJECTORIES
    # ========================================================================

    migrated_indices = []
    non_migrated_indices = []

    for traj_idx in traj_roi_subset:
        y_traj = np.array(traj_hist_Y_roi[traj_idx])
        x_traj = np.array(traj_hist_X_roi[traj_idx])

        mask = x_traj >= 0
        if not np.any(mask):
            continue

        y_roi = y_traj[mask]

        if np.any(y_roi < 0):
            migrated_indices.append(traj_idx)
        else:
            non_migrated_indices.append(traj_idx)

    print(f"[CLASSIFICATION]")
    print(f"  Migrated (y<0): {len(migrated_indices)}")
    print(f"  Non-migrated: {len(non_migrated_indices)}")

    # ========================================================================
    # SAMPLING FOR TIME PLOTS
    # ========================================================================

    # For x(t): sample from all (stratified)
    MAX_XT_TRAJ = 500
    max_migrated_xt = int(0.7 * MAX_XT_TRAJ)
    n_sample_migrated_xt = min(len(migrated_indices), max_migrated_xt)
    n_sample_non_migrated_xt = min(MAX_XT_TRAJ - n_sample_migrated_xt, len(non_migrated_indices))

    import random
    random.seed(43)

    sampled_migrated_xt = random.sample(migrated_indices, n_sample_migrated_xt) if len(migrated_indices) > n_sample_migrated_xt else migrated_indices
    sampled_non_migrated_xt = random.sample(non_migrated_indices, n_sample_non_migrated_xt) if len(non_migrated_indices) > n_sample_non_migrated_xt else non_migrated_indices

    traj_xt_sample = sampled_migrated_xt + sampled_non_migrated_xt

    # For y(t): ONLY migrated trajectories
    MAX_YT_TRAJ = 500  # Can show more since filtering to only interesting ones
    if len(migrated_indices) > MAX_YT_TRAJ:
        traj_yt_sample = random.sample(migrated_indices, MAX_YT_TRAJ)
    else:
        traj_yt_sample = migrated_indices

    print(f"[SAMPLING]")
    print(f"  x-y plot: {len(traj_roi_subset)} trajectories")
    print(f"  x(t) plot: {len(traj_xt_sample)} trajectories")
    print(f"  y(t) plot: {len(traj_yt_sample)} trajectories (migrated only)")

    # ========================================================================
    # CREATE FIGURE WITH CUSTOM LAYOUT
    # ========================================================================

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top panel spans full width
    ax_xy = fig.add_subplot(gs[0, :])

    # Bottom panels
    ax_xt = fig.add_subplot(gs[1, 0])
    ax_yt = fig.add_subplot(gs[1, 1])

    # ========================================================================
    # TOP PANEL: x-y trajectories (monochrome navy)
    # ========================================================================

    x_max_reached = ROI_X1

    for traj_idx in traj_roi_subset:
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        y_traj = np.array(traj_hist_Y_roi[traj_idx])

        mask = (x_traj >= ROI_X1) & (x_traj <= ROI_X2)
        if np.sum(mask) < 2:
            continue

        x_plot = x_traj[mask] * CODE_TO_UM
        y_plot = y_traj[mask] * CODE_TO_UM
        x_max_reached = max(x_max_reached, np.max(x_traj[mask]))

        ax_xy.plot(x_plot, y_plot, color=NAVY_BLUE, lw=0.5, alpha=0.5)

    # Reference lines (gold/amber)
    ax_xy.axhline(y0_um, color=COLOR_WAVEGUIDE, ls='--', lw=1.5, alpha=0.9, 
                  label='Waveguide centers', zorder=5)
    ax_xy.axhline(-y0_um, color=COLOR_WAVEGUIDE, ls='--', lw=1.5, alpha=0.9, zorder=5)
    ax_xy.axhline(0, color=COLOR_BARRIER_CENTER, ls='-', lw=2.0, alpha=0.9, 
                  label='Barrier center', zorder=5)

    # Limits
    x_limit = min(x_max_reached * 1.1, ROI_X2)
    x_limit = max(x_limit, ROI_X1 + 2.0)

    ax_xy.set_xlim(ROI_X1 * CODE_TO_UM, x_limit * CODE_TO_UM)
    ax_xy.set_ylim(-y0_um * 1.5, y0_um * 1.5)
    ax_xy.set_xlabel('x (μm)', fontsize=12)
    ax_xy.set_ylabel('y (μm)', fontsize=12)
    ax_xy.set_title(f'(a) Spatial Trajectories in ROI ({len(traj_roi_subset)} trajectories)',
                    fontsize=13, fontweight='bold')
    ax_xy.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax_xy.grid(alpha=0.3)

    # ========================================================================
    # BOTTOM LEFT: x(t) - all trajectories in navy
    # ========================================================================

    for traj_idx in traj_xt_sample:
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        t_traj = np.array(traj_times_roi[traj_idx])

        if len(x_traj) < 2:
            continue

        mask = x_traj >= 0
        if not np.any(mask):
            continue

        x_filtered = x_traj[mask]
        t_filtered = t_traj[mask]

        x_um = x_filtered * CODE_TO_UM
        t_ps = t_filtered * T0_SI * 1e12

        ax_xt.plot(t_ps, x_um, color=NAVY_BLUE, lw=0.5, alpha=0.4)

    # Reference line (amber)
    ax_xt.axhline(0, color=COLOR_BARRIER_ENTRANCE, ls='--', lw=2, alpha=1.0,
                  label='Barrier entrance (x=0)', zorder=5)

    ax_xt.set_xlabel("Time (ps)", fontsize=12)
    ax_xt.set_ylabel("x (μm)", fontsize=12)
    ax_xt.set_title(f"(b) Barrier Penetration vs Time ({len(traj_xt_sample)} trajectories)",
                    fontsize=13, fontweight='bold')
    ax_xt.grid(alpha=0.3)
    ax_xt.legend(fontsize=10, loc='best')

    # ========================================================================
    # BOTTOM RIGHT: y(t) - ONLY migrated trajectories in navy
    # ========================================================================

    for traj_idx in traj_yt_sample:
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        y_traj = np.array(traj_hist_Y_roi[traj_idx])
        t_traj = np.array(traj_times_roi[traj_idx])

        if len(y_traj) < 2:
            continue

        mask = x_traj >= 0
        if not np.any(mask):
            continue

        y_filtered = y_traj[mask]
        t_filtered = t_traj[mask]

        y_um = y_filtered * CODE_TO_UM
        t_ps = t_filtered * T0_SI * 1e12

        # Slightly thicker lines for y(t) plot
        ax_yt.plot(t_ps, y_um, color=NAVY_BLUE, lw=0.7, alpha=0.7)

    # Reference lines (gold/amber)
    ax_yt.axhline(y0_um, color=COLOR_WAVEGUIDE, ls='--', lw=1.5, alpha=0.9,
                  label=f'Waveguide centers (±{y0_um:.1f} μm)', zorder=5)
    ax_yt.axhline(-y0_um, color=COLOR_WAVEGUIDE, ls='--', lw=1.5, alpha=0.9, zorder=5)
    ax_yt.axhline(0, color=COLOR_BARRIER_CENTER, ls='-', lw=2.0, alpha=0.9,
                  label='Barrier center (y=0)', zorder=5)

    ax_yt.set_xlabel("Time (ps)", fontsize=12)
    ax_yt.set_ylabel("y (μm)", fontsize=12)
    ax_yt.set_title(f"(c) Transverse Position vs Time ({len(traj_yt_sample)} migrated)",
                    fontsize=13, fontweight='bold')
    ax_yt.grid(alpha=0.3)

    # Create legend and set zorder separately
    legend = ax_yt.legend(fontsize=10, loc='best', framealpha=0.95)
    legend.set_zorder(10)

    # ========================================================================
    # SAVE
    # ========================================================================

    plt.savefig(os.path.join(results_dir, 'trajectories_3panel.png'), dpi=300, bbox_inches='tight')
    plt.draw()
    plt.pause(0.001)

    print(f"\n[SAVED] trajectories_3panel.png")
    print("="*60)
        
#%% =========== TUNNELING SPEED ANALYSIS (Y-MIGRATION DYNAMICS) =====================

if ANALYZE_TUNNELING_SPEED and COMPUTE_TRAJECTORIES and len(traj_hist_X_roi) > 0:
    print("\n" + "="*70)
    print("TUNNELING SPEED ANALYSIS (Y-MIGRATION)")
    print("="*70)
    
    tunneling_times = []
    x_positions = []
    valid_traj_indices = []
    
    # Also track rejected trajectories for diagnostics
    rejected_turned_around = 0
    
    for traj_idx in traj_hist_X_roi.keys():
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        y_traj = np.array(traj_hist_Y_roi[traj_idx])
        t_traj = np.array(traj_times_roi[traj_idx])
        
        if len(x_traj) < 10:
            continue
        
        # Check 1: Does trajectory start at y > a?
        n_init = min(5, len(y_traj))
        y_initial = np.mean(y_traj[:n_init])
        
        if y_initial <= 0:
            continue  # Doesn't start in upper region
        
        # Check 2: Use first ROI point as barrier crossing (ROI starts at x=0)
        idx_cross = 0
        t_cross = t_traj[0]
        x_cross = x_traj[0]
        
        # Check 3: Find first y-migration (y crosses -a) AFTER barrier crossing
        # Look for points where y < -a
        migrated_mask = y_traj < 0
        
        if not np.any(migrated_mask):
            continue  # Never migrates to lower region
        
        # First migration point
        idx_migrate = np.where(migrated_mask)[0][0]
        t_migrate = t_traj[idx_migrate]
        x_migrate = x_traj[idx_migrate]
        y_migrate = y_traj[idx_migrate]
        
        # Check 4: NEW - Only accept migrations that occur at x > 0
        # Reject trajectories that turned around and migrated during retreat
        if x_migrate <= 0.0:
            rejected_turned_around += 1
            continue  # Migration happened after turning around
        
        # Compute tunneling time and record x-position
        t_tunnel = t_migrate - t_cross
        
        if t_tunnel > 1e-6:  # Sanity check
            tunneling_times.append(t_tunnel)
            x_positions.append(x_migrate)
            valid_traj_indices.append(traj_idx)
    
    # Report statistics
    n_valid = len(tunneling_times)
    print(f"\n[TUNNELING] Found {n_valid} trajectories completing y-migration at x>0")
    print(f"[TUNNELING] Rejected {rejected_turned_around} trajectories (turned around, migrated at x<0)")
    print(f"[TUNNELING] Total trajectories in ROI: {len(traj_hist_X_roi)}")
    
    if n_valid > 3:
        tunneling_times = np.array(tunneling_times)
        x_positions = np.array(x_positions)
        
        print(f"\n[STATISTICS]")
        print(f"  Tunneling time - mean: {np.mean(tunneling_times):.3f}, median: {np.median(tunneling_times):.3f}")
        print(f"  X-position - mean: {np.mean(x_positions):.3f}, median: {np.median(x_positions):.3f}")
        print(f"  X-position - range: [{np.min(x_positions):.1f}, {np.max(x_positions):.1f}]")
        print(f"  Standard deviation - time: {np.std(tunneling_times):.3f}, position: {np.std(x_positions):.3f}")
        
        
        # ============================================================================
        # Y-MIGRATION ANALYSIS (Physical Units with error bar)
        # ============================================================================    
            
        print("\n" + "="*60)
        print("CREATING PUBLICATION Y-MIGRATION PLOT (Physical Units)")
        print("="*60)
        
        # Check if data exists
        if 'tunneling_times' not in globals() or len(tunneling_times) == 0:
            print("[ERROR] Y-migration data not available!")
            print("        Set ANALYZE_TUNNELING_SPEED=True and re-run simulation")
        else:
            # Physical unit conversion (L0, E0, T0 from UNIT SYSTEM block)
            
            # Convert to numpy arrays
            tunneling_times_arr = np.array(tunneling_times)
            x_positions_arr = np.array(x_positions)
            
            # ========================================================================
            # FILTER DATA (remove outliers / invalid entries)
            # ========================================================================
            # Filter: positive times and positions, remove extreme outliers
            valid_mask = (tunneling_times_arr > 0) & (x_positions_arr > 0)
            
            # Optional: remove outliers beyond 3σ
            if np.sum(valid_mask) > 10 and is_propagative:
                velocities_raw = x_positions_arr[valid_mask] / tunneling_times_arr[valid_mask]
                v_median = np.median(velocities_raw)
                v_mad = np.median(np.abs(velocities_raw - v_median))  # Median absolute deviation
                outlier_threshold = 3.0  # MAD units
                velocity_mask = np.abs(velocities_raw - v_median) < outlier_threshold * v_mad * 1.4826
                
                # Apply velocity filter to valid indices
                valid_indices = np.where(valid_mask)[0]
                final_indices = valid_indices[velocity_mask]
                
                tunneling_times_filtered = tunneling_times_arr[final_indices]
                x_positions_filtered = x_positions_arr[final_indices]
                
                n_removed = np.sum(valid_mask) - len(final_indices)
                if n_removed > 0:
                    print(f"[FILTER] Removed {n_removed} outliers ({100*n_removed/len(tunneling_times_arr):.1f}%)")
            else:
                tunneling_times_filtered = tunneling_times_arr[valid_mask]
                x_positions_filtered = x_positions_arr[valid_mask]
            
            print(f"[DATA] {len(tunneling_times_filtered)} trajectories (after filtering)")
            
            # Convert to physical units
            tunneling_times_ps = tunneling_times_filtered * T0 * 1e12  # → ps
            x_positions_um = x_positions_filtered * L0 * 1e6  # → μm
            
            print(f"[RANGE] Time: {tunneling_times_ps.min():.2f} - {tunneling_times_ps.max():.2f} ps")
            print(f"[RANGE] Position: {x_positions_um.min():.2f} - {x_positions_um.max():.2f} μm")
            
            # ========================================================================
            # COMPUTE INDIVIDUAL VELOCITIES AND STATISTICS
            # ========================================================================
            # Compute individual velocities in code units
            velocities_code = x_positions_filtered / tunneling_times_filtered
            
            v_mean_code = np.mean(velocities_code)
            v_std_code = np.std(velocities_code, ddof=1)  # Sample standard deviation
            v_se_code = v_std_code / np.sqrt(len(velocities_code))  # Standard error
            
            # Convert velocity to physical units: v_phys = (ℏ/m) * v_code / L0
            v_mean_physical_ms = (hbar_SI / m_SI) * (v_mean_code / L0)  # → m/s
            v_mean_physical_kms = v_mean_physical_ms * 1e-3  # → km/s
            
            v_se_physical_ms = (hbar_SI / m_SI) * (v_se_code / L0)  # → m/s
            v_se_physical_kms = v_se_physical_ms * 1e-3  # → km/s
            
            print(f"\n[AVERAGE SPEED]")
            print(f"  Code units: {v_mean_code:.6f} ± {v_se_code:.6f} (mean ± SE)")
            print(f"  Physical: {v_mean_physical_kms:.3f} ± {v_se_physical_kms:.3f} km/s")
            print(f"  Velocity spread (1σ): {v_std_code:.6f} (code), {(v_std_code/v_mean_code)*100:.2f}% relative")
            print(f"  Statistical error: {(v_se_code/v_mean_code)*100:.2f}% relative")
            # ========================================================================
            
            # Convert velocity scale to physical units
            if is_evanescent and 'kappa0' in globals() and 'kappa1' in globals():
                # Paper notation: k₂ = κ̄ (average decay rate)
                kappa_bar = (kappa0 + kappa1) / 2.0
                k2_code = kappa_bar
                k2_physical = k2_code / (L0 * 1e6)  # → μm⁻¹
                v_normalized = v_mean_code / kappa_bar
                v_se_normalized = v_se_code / kappa_bar
                scale_label = 'κ_2'  # No $ signs - will be used inside math mode
                
                print(f"\n[VELOCITY SCALE] Evanescent regime")
                print(f"  κ̄ = k₂ = {k2_code:.6f} (code) = {k2_physical:.4f} μm⁻¹")
                print(f"  v/k₂ = {v_normalized:.3f} ± {v_se_normalized:.3f}")
                
            elif is_propagative and 'd0' in globals() and 'd1' in globals():
                # Paper notation: k₂ = k̄ (average momentum)
                k0 = np.sqrt(2.0 * abs(d0))
                k1 = np.sqrt(2.0 * abs(d1))
                k_bar = (k0 + k1) / 2.0
                k2_code = k_bar
                k2_physical = k2_code / (L0 * 1e6)  # → μm⁻¹
                v_normalized = v_mean_code / k_bar
                v_se_normalized = v_se_code / k_bar
                scale_label = 'k_2'  # No $ signs - will be used inside math mode
                
                print(f"\n[VELOCITY SCALE] Propagative regime")
                print(f"  k̄ = k₂ = {k2_code:.6f} (code) = {k2_physical:.4f} μm⁻¹")
                print(f"  v/k₂ = {v_normalized:.3f} ± {v_se_normalized:.3f}")
                
            else:
                v_normalized = v_mean_code
                v_se_normalized = v_se_code
                scale_label = 'code units'
                print(f"\n[VELOCITY SCALE] Using raw units")
            
            # Convert reference scales to physical units
            if 'J0_energy' in globals():
                t_scale_code = 1.0 / J0_energy
                t_scale_ps = t_scale_code * T0 * 1e12
            
            if is_evanescent and 'kappa0' in globals() and 'kappa1' in globals():
                if (kappa1 - kappa0) > 1e-6:
                    # Paper notation: 1/k₁ (instead of 2/Δκ)
                    k1_paper = (kappa1 - kappa0) / 2.0  # Define k₁ for paper
                    L_scale_code = 1.0 / (2*k1_paper)
                    L_scale_um = L_scale_code * L0 * 1e6
                    distance_label = r'1/2κ₁'  # No $ signs - will be used inside math mode
                    
                    print(f"\n[LENGTH SCALE]")
                    print(f"  Δκ = {kappa1 - kappa0:.6f}")
                    print(f"  κ₁ (paper) = Δκ/2 = {k1_paper:.6f} (code) = {k1_paper/(L0*1e6):.4f} μm⁻¹")
                    print(f"  1/2κ₁ = {L_scale_um:.2f} μm")
            
            elif is_propagative and 'd0' in globals() and 'd1' in globals():
                k0 = np.sqrt(2.0 * abs(d0))
                k1 = np.sqrt(2.0 * abs(d1))
                Delta_k = abs(k1 - k0)
                
                if Delta_k > 1e-6:
                    # Paper notation: 1/k₁ (instead of 2/|Δk|)
                    k1_paper = Delta_k / 2.0  # Define k₁ for paper
                    L_scale_code = 1.0 / k1_paper
                    L_scale_um = L_scale_code * L0 * 1e6
                    distance_label = r'1/k_1'  # No $ signs - will be used inside math mode
                    
                    print(f"\n[LENGTH SCALE]")
                    print(f"  Δk = {Delta_k:.6f}")
                    print(f"  k₁ (paper) = Δk/2 = {k1_paper:.6f} (code) = {k1_paper/(L0*1e6):.4f} μm⁻¹")
                    print(f"  1/k₁ = {L_scale_um:.2f} μm")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Scatter plot (now with filtered data)
            ax.scatter(tunneling_times_ps, x_positions_um, alpha=0.6, s=50,
                       label=f'Trajectories (n={len(tunneling_times_ps)})',
                       color='blue', edgecolors='black', linewidth=0.5)
    
    # ========================================================================
    # AVERAGE SPEED LINE WITH ±1 SE CONFIDENCE BAND
    # ========================================================================
    t_line_ps = np.linspace(0, np.max(tunneling_times_ps) * 1.1, 100)
    
    # Central line
    x_line_um = v_mean_physical_kms * t_line_ps * 1e-3  # Convert ps to μs for km/s units
    
    # Upper and lower bounds (±1 SE)
    x_upper_um = (v_mean_physical_kms + v_se_physical_kms) * t_line_ps * 1e-3
    x_lower_um = (v_mean_physical_kms - v_se_physical_kms) * t_line_ps * 1e-3
    
    # Plot confidence band
    if v_se_normalized >= 0.001: 
        ax.fill_between(t_line_ps, x_lower_um, x_upper_um, 
                        color='red', alpha=0.2, zorder=2,
                        label=f'±1 SE confidence')
        
        # Plot central line
        ax.plot(t_line_ps, x_line_um, '--', color='red', lw=2.5, alpha=0.8, zorder=3,
                label=f'Avg speed: $v = ({v_normalized:.3f} \\pm {v_se_normalized:.3f}) \\, {scale_label}$ ({v_mean_physical_kms:.2f} km/s)')
    else: 
        ax.plot(t_line_ps, x_line_um, '--', color='red', lw=2.5, alpha=0.8, zorder=3,
                label=f'Avg speed: $v = {v_normalized:.3f} \\, {scale_label}$ ({v_mean_physical_kms:.2f} km/s)')
    # ========================================================================
    
    # Time scale: 1/J₀
    if 'J0_energy' in globals():
        ax.axvline(t_scale_ps, color='orange', ls='--', lw=2, alpha=0.7,
                   label=f'Time scale: $1/J_0 = {t_scale_ps:.1f}$ ps')
    
    # Distance scale: 1/k₁
    if 'L_scale_um' in locals():
        ax.axhline(L_scale_um, color='green', ls='-.', lw=2, alpha=0.7,
                   label=f'Length scale: ${distance_label} = {L_scale_um:.1f}$ μm')
    
    # Labels with physical units
    ax.set_xlabel('Tunneling Time (ps)', fontsize=13)
    ax.set_ylabel('X-Position of Migration (μm)', fontsize=13)
    
    regime_str = 'Propagative' if is_propagative else ('Evanescent' if is_evanescent else 'Mixed')
    ax.set_title(f'Y-Migration Dynamics ({regime_str} Regime)', 
                fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Set reasonable y-limits, ensuring length scale is visible
    y_padding = 0.1 * (x_positions_um.max() - x_positions_um.min())
    y_min_plot2 = max(-2, x_positions_um.min() - y_padding)
    y_max_plot2 = x_positions_um.max() + y_padding
    
    # Ensure length scale line is visible (if it exists).
    # (Use the LOCAL plot limit; do NOT touch the domain variable y_max.)
    if 'L_scale_um' in locals():
        y_max_plot2 = max(y_max_plot2, L_scale_um * 1.1)  # 10% margin above scale
    
    ax.set_ylim(y_min_plot2, y_max_plot2)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(results_dir, 'ymigration_physical_units.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.draw()
    plt.pause(0.001)
    
    print(f"\n[SAVED] {save_path}")
    print(f"[UNITS] Time in picoseconds, position in micrometers, velocity in km/s")
    print("="*60)
    

#%%=========== ρ_a(x) ANALYSIS: WAVEGUIDE POPULATION RATIO =================


if ANALYZE_RHO_A and COMPUTE_TRAJECTORIES and len(traj_hist_X_roi) > 0 and is_evanescent:

    print("\n" + "="*70)
    print(f"ρ_a(x) ANALYSIS: WAVEGUIDE POPULATION RATIO ({regime_name.upper()} REGIME)")
    print("="*70)

    # Y-threshold: 2σ from well center
    y_threshold_physical = y0 - 2 * sigma_well

    print(f"  Migration 2σ threshold: |y| < {y_threshold_physical:.2f}")

    # Get physical unit conversion
    if 'L0' in locals():
        length_to_um = L0 * 1e6
    else:
        length_to_um = 2.0  # Fallback

    # ========================================================================
    # DETERMINE TIME WINDOW FROM TRAJECTORY STATISTICS
    # ========================================================================

    print(f"\n[DETERMINING TIME WINDOW]")

    # Collect all entry and exit times
    entry_times = []
    exit_times = []

    for traj_idx in traj_hist_X_roi.keys():
        times = traj_times_roi[traj_idx]
        if len(times) > 0:
            entry_times.append(times[0])
            exit_times.append(times[-1])

    if len(entry_times) == 0:
        print("[ERROR] No trajectories in ROI")
        t_window_start = 0
        t_window_end = 0
    else:
        # Start: when first trajectories enter
        t_window_start = np.min(entry_times)

        # End: when 90% of trajectories have exited
        t_window_end = np.percentile(exit_times, 90)

        # Build histogram of occupancy vs time
        time_bins = np.linspace(t_window_start, np.max(exit_times), 100)
        occupancy = np.zeros(len(time_bins) - 1)

        for i, t in enumerate(time_bins[:-1]):
            t_mid = 0.5 * (time_bins[i] + time_bins[i+1])
            n_active = 0
            for traj_idx in traj_hist_X_roi.keys():
                times = traj_times_roi[traj_idx]
                if len(times) > 0 and times[0] <= t_mid <= times[-1]:
                    n_active += 1
            occupancy[i] = n_active

        # Find when occupancy drops to 10%
        peak_occ = np.max(occupancy)
        threshold_occ = 0.1 * peak_occ

        idx_peak = np.argmax(occupancy)
        for i in range(idx_peak, len(occupancy)):
            if occupancy[i] < threshold_occ:
                t_window_end_alt = time_bins[i]
                break
        else:
            t_window_end_alt = time_bins[-1]

        t_window_end = min(t_window_end, t_window_end_alt)

    print(f"  Start: t = {t_window_start:.2f}")
    print(f"  End: t = {t_window_end:.2f}")
    print(f"  Duration: {t_window_end - t_window_start:.2f}")
    print(f"  Number of trajectories: {len(traj_hist_X_roi)}")

    # ========================================================================
    # SETUP SPATIAL BINNING
    # ========================================================================

    spatial_bin_width = RHO_A_SPATIAL_BIN
    x_bins = np.arange(ROI_X1, ROI_X2 + spatial_bin_width, spatial_bin_width)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])

    print(f"\n[SPATIAL BINNING]")
    print(f"  Bin width: Δx = {spatial_bin_width:.2f} ({spatial_bin_width * length_to_um:.2f} μm)")
    print(f"  Number of bins: {len(x_centers)}")
    print(f"  Range: x ∈ [{ROI_X1:.1f}, {ROI_X2:.1f}]")

    # ========================================================================
    # COMPUTE RESIDENCE TIME WITH SEGMENT CROSSING HANDLING
    # ========================================================================

    print(f"\n[COMPUTING RESIDENCE TIMES]")
    print(f"  Using threshold: |y| < {y_threshold_physical:.2f}")
    print(f"  Handling segment crossings with interpolation")

    residence_lower = np.zeros(len(x_centers))
    residence_upper = np.zeros(len(x_centers))
    residence_barrier = np.zeros(len(x_centers))

    # Also track counts for error estimation
    count_lower = np.zeros(len(x_centers))
    count_upper = np.zeros(len(x_centers))

    n_traj_processed = 0

    def classify_y(y, y_thresh):
        """Classify y position: -1=lower, 0=barrier, +1=upper"""
        if y < -y_thresh:
            return -1
        elif y > y_thresh:
            return 1
        else:
            return 0

    def split_segment_time(y1, y2, dt, y_thresh):
        """
        Split segment time between regions if crossing occurs.
        Returns (dt_lower, dt_upper, dt_barrier)
        """
        class1 = classify_y(y1, y_thresh)
        class2 = classify_y(y2, y_thresh)

        # No crossing
        if class1 == class2:
            if class1 == -1:
                return dt, 0, 0
            elif class1 == 1:
                return 0, dt, 0
            else:
                return 0, 0, dt

        # Crossing occurred - interpolate
        dt_lower, dt_upper, dt_barrier = 0, 0, 0

        if y2 == y1:
            # Degenerate case
            if class1 == -1:
                return dt, 0, 0
            elif class1 == 1:
                return 0, dt, 0
            else:
                return 0, 0, dt

        # Find crossing times (as fraction of dt)
        crossings = []

        # Lower threshold crossing
        if (y1 < -y_thresh) != (y2 < -y_thresh):
            t_cross = (-y_thresh - y1) / (y2 - y1)
            if 0 < t_cross < 1:
                crossings.append((t_cross, 'lower'))

        # Upper threshold crossing
        if (y1 > y_thresh) != (y2 > y_thresh):
            t_cross = (y_thresh - y1) / (y2 - y1)
            if 0 < t_cross < 1:
                crossings.append((t_cross, 'upper'))

        # Sort crossings by time
        crossings.sort(key=lambda x: x[0])

        # Add start and end
        t_points = [0.0] + [c[0] for c in crossings] + [1.0]

        # Accumulate time in each region
        for i in range(len(t_points) - 1):
            t_mid_frac = 0.5 * (t_points[i] + t_points[i+1])
            y_mid = y1 + t_mid_frac * (y2 - y1)
            dt_segment = (t_points[i+1] - t_points[i]) * dt

            if y_mid < -y_thresh:
                dt_lower += dt_segment
            elif y_mid > y_thresh:
                dt_upper += dt_segment
            else:
                dt_barrier += dt_segment

        return dt_lower, dt_upper, dt_barrier

    # Process trajectories
    for traj_idx in traj_hist_X_roi.keys():
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        y_traj = np.array(traj_hist_Y_roi[traj_idx])
        t_traj = np.array(traj_times_roi[traj_idx])

        if len(t_traj) < 2:
            continue

        # Filter to time window
        mask = (t_traj >= t_window_start) & (t_traj <= t_window_end)
        x_traj = x_traj[mask]
        y_traj = y_traj[mask]
        t_traj = t_traj[mask]

        if len(t_traj) < 2:
            continue

        n_traj_processed += 1

        # Process each segment with crossing handling
        for i in range(len(t_traj) - 1):
            x_mid = 0.5 * (x_traj[i] + x_traj[i+1])
            y1, y2 = y_traj[i], y_traj[i+1]
            dt = t_traj[i+1] - t_traj[i]

            # Find spatial bin
            bin_idx = np.searchsorted(x_bins, x_mid) - 1
            if bin_idx < 0 or bin_idx >= len(x_centers):
                continue

            # Split time between regions
            dt_lower, dt_upper, dt_barrier = split_segment_time(y1, y2, dt, y_threshold_physical)

            residence_lower[bin_idx] += dt_lower
            residence_upper[bin_idx] += dt_upper
            residence_barrier[bin_idx] += dt_barrier

            # Track counts (for error estimation)
            if dt_lower > 0:
                count_lower[bin_idx] += 1
            if dt_upper > 0:
                count_upper[bin_idx] += 1

    print(f"  Processed {n_traj_processed} trajectories")

    # ========================================================================
    # COMPUTE ρ_a WITH ERROR BARS
    # ========================================================================

    total_residence_localized = residence_lower + residence_upper
    rho_a_raw = np.zeros(len(x_centers))
    rho_a_err = np.zeros(len(x_centers))

    valid_bins = total_residence_localized > 1e-12
    rho_a_raw[valid_bins] = residence_lower[valid_bins] / total_residence_localized[valid_bins]

    # Error estimation using effective counts
    # Effective count based on residence time ratio
    for i in range(len(x_centers)):
        if total_residence_localized[i] > 1e-12:
            # Use count-based binomial error
            n_total = count_lower[i] + count_upper[i]
            if n_total > 0:
                p = rho_a_raw[i]
                rho_a_err[i] = np.sqrt(p * (1 - p) / n_total)

    # Report statistics
    total_res_all = residence_lower + residence_upper + residence_barrier
    frac_lower = np.sum(residence_lower) / np.sum(total_res_all) if np.sum(total_res_all) > 0 else 0
    frac_upper = np.sum(residence_upper) / np.sum(total_res_all) if np.sum(total_res_all) > 0 else 0
    frac_barrier = np.sum(residence_barrier) / np.sum(total_res_all) if np.sum(total_res_all) > 0 else 0

    print(f"\n[RESIDENCE STATISTICS]")
    print(f"  Lower waveguide:  {100*frac_lower:.1f}%")
    print(f"  Upper waveguide:  {100*frac_upper:.1f}%")
    print(f"  Barrier region:   {100*frac_barrier:.1f}% (excluded from ρ_a)")

    # ========================================================================
    # BASELINE CORRECTION
    # ========================================================================

    idx_x0 = np.argmin(np.abs(x_centers - 0.0))
    baseline = rho_a_raw[idx_x0]
    rho_a_corrected = np.clip(rho_a_raw - baseline, 0.0, 1.0)

    print(f"\n[BASELINE CORRECTION]")
    print(f"  Baseline at x≈0: {baseline:.6f}")

    # ========================================================================
    # STATISTICAL CUTOFF (based on data availability)
    # ========================================================================
    
    print(f"\n[STATISTICAL CUTOFF]")
    
    Delta_kappa = kappa1 - kappa0
    
    # Count total samples per bin
    count_total = count_lower + count_upper
    
    # Method 1: Minimum count threshold
    # Require at least N samples per bin for reliable statistics
    min_count_threshold = 30  # Typical rule of thumb for binomial statistics
    sufficient_counts_mask = count_total >= min_count_threshold
    
    if np.any(sufficient_counts_mask):
        # Find the farthest bin with sufficient counts
        idx_sufficient = np.where(sufficient_counts_mask)[0]
        x_cutoff_counts = x_centers[idx_sufficient[-1]]
    else:
        # Fallback: use 10th percentile of counts
        nonzero_counts = count_total[count_total > 0]
        if len(nonzero_counts) > 0:
            count_threshold_fallback = max(5, np.percentile(nonzero_counts, 10))
            fallback_mask = count_total >= count_threshold_fallback
            if np.any(fallback_mask):
                x_cutoff_counts = x_centers[np.where(fallback_mask)[0][-1]]
            else:
                x_cutoff_counts = x_centers[count_total > 0][-1]
        else:
            x_cutoff_counts = x_centers[-1]
    
    # Method 2: Continuous coverage
    # Find where we have continuous data (no large gaps)
    has_data = count_total > 0
    x_cutoff_continuous = x_centers[0]
    
    for i in range(len(x_centers)):
        if x_centers[i] > 0.1:  # Start checking after x > 0.1
            if has_data[i]:
                x_cutoff_continuous = x_centers[i]
            else:
                # Found a gap - stop here
                break
    
    # Method 3: Cumulative trajectory coverage
    # Find where 95% of trajectory-bin contributions occur
    cumsum_counts = np.cumsum(count_total)
    total_counts = cumsum_counts[-1] if cumsum_counts[-1] > 0 else 1
    coverage_95_idx = np.searchsorted(cumsum_counts, 0.95 * total_counts)
    x_cutoff_coverage = x_centers[min(coverage_95_idx, len(x_centers) - 1)]
    
    # Method 4: SNR-based (keep this as secondary check)
    snr_threshold = 2.0
    snr = np.zeros(len(x_centers))
    for i in range(len(x_centers)):
        if rho_a_err[i] > 1e-12:
            snr[i] = rho_a_corrected[i] / rho_a_err[i]
    
    if np.any(snr > snr_threshold):
        idx_good_snr = np.where(snr > snr_threshold)[0]
        x_cutoff_snr = x_centers[idx_good_snr[-1]] if len(idx_good_snr) > 0 else x_centers[-1]
    else:
        x_cutoff_snr = x_centers[-1]
    
    # Theory reference (for comparison only, not used for cutoff)
    x_equil_theory = 2.0 / Delta_kappa if Delta_kappa > 0 else 10.0
    
    # Use the minimum of data-based methods
    x_cutoff = min(x_cutoff_counts, x_cutoff_continuous, x_cutoff_coverage)
    
    print(f"  Min count threshold (N≥{min_count_threshold}): x ≤ {x_cutoff_counts:.2f} ({x_cutoff_counts * length_to_um:.2f} μm)")
    print(f"  Continuous coverage: x ≤ {x_cutoff_continuous:.2f} ({x_cutoff_continuous * length_to_um:.2f} μm)")
    print(f"  95% cumulative coverage: x ≤ {x_cutoff_coverage:.2f} ({x_cutoff_coverage * length_to_um:.2f} μm)")
    print(f"  SNR-based (SNR>{snr_threshold}): x ≤ {x_cutoff_snr:.2f} ({x_cutoff_snr * length_to_um:.2f} μm)")
    print(f"  Theory equilibration (2/Δκ): {x_equil_theory:.2f} ({x_equil_theory * length_to_um:.2f} μm) [reference only]")
    print(f"  → Using cutoff: x ≤ {x_cutoff:.2f} ({x_cutoff * length_to_um:.2f} μm)")
    
    # Report bin statistics at cutoff
    idx_cutoff = np.argmin(np.abs(x_centers - x_cutoff))
    print(f"\n[BIN STATISTICS AT CUTOFF]")
    print(f"  x = {x_centers[idx_cutoff]:.2f}: N_lower={count_lower[idx_cutoff]:.0f}, N_upper={count_upper[idx_cutoff]:.0f}, N_total={count_total[idx_cutoff]:.0f}")
    
    # Show count distribution
    print(f"\n[COUNT DISTRIBUTION]")
    print(f"  Total bins: {len(x_centers)}")
    print(f"  Bins with N≥{min_count_threshold}: {np.sum(sufficient_counts_mask)}")
    print(f"  Bins with any data: {np.sum(has_data)}")
    print(f"  Max counts in any bin: {np.max(count_total):.0f}")
    print(f"  Median counts (non-zero bins): {np.median(count_total[count_total > 0]):.0f}" if np.any(count_total > 0) else "  No data")

    # ========================================================================
    # PREPARE FIT DATA
    # ========================================================================

    fit_mask = (rho_a_corrected > 0) & (x_centers > 0.1) & (x_centers <= x_cutoff) & (rho_a_err > 0)
    x_fit = x_centers[fit_mask]
    rho_fit = rho_a_corrected[fit_mask]
    rho_fit_err = rho_a_err[fit_mask]

    if len(x_fit) < 5:
        print("[ERROR] Insufficient data points for fitting")
        print(f"  Only {len(x_fit)} points found in range [0.1, {x_cutoff:.2f}]")
    else:
        # ========================================================================
        # THEORY FITTING (WEIGHTED)
        # ========================================================================

        B_theory = Delta_kappa / 2.0
        A_theory_quad = B_theory**2

        print(f"\n[THEORY PARAMETERS]")
        print(f"  κ₀ = {kappa0:.6f}")
        print(f"  κ₁ = {kappa1:.6f}")
        print(f"  Δκ = κ₁ - κ₀ = {Delta_kappa:.6f}")
        print(f"  B = Δκ/2 = {B_theory:.6f}")
        print(f"  A = B² = {A_theory_quad:.6f}")

        # Define theory function
        def theory_func(x, B):
            return np.sinh(B * x)**2 / np.cosh(2 * B * x)

        # Evaluate pure theory (no free parameters)
        rho_theory = theory_func(x_fit, B_theory)

        # Weighted R² for theory
        weights = 1.0 / rho_fit_err**2
        ss_res_weighted = np.sum(weights * (rho_fit - rho_theory)**2)
        ss_tot_weighted = np.sum(weights * (rho_fit - np.average(rho_fit, weights=weights))**2)
        r2_theory = 1 - ss_res_weighted / ss_tot_weighted if ss_tot_weighted > 0 else 0

        # Unweighted R² for comparison
        ss_res = np.sum((rho_fit - rho_theory)**2)
        ss_tot = np.sum((rho_fit - np.mean(rho_fit))**2)
        r2_theory_unweighted = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Fit B as free parameter (weighted)
        try:
            popt, pcov = curve_fit(theory_func, x_fit, rho_fit,
                                   p0=[B_theory], bounds=([0], [0.5]),
                                   sigma=rho_fit_err, absolute_sigma=True)
            B_fitted = popt[0]
            B_err = np.sqrt(pcov[0, 0])

            rho_fitted = theory_func(x_fit, B_fitted)
            ss_res_fit_weighted = np.sum(weights * (rho_fit - rho_fitted)**2)
            r2_fitted = 1 - ss_res_fit_weighted / ss_tot_weighted if ss_tot_weighted > 0 else 0

            # Chi-squared
            chi2 = np.sum(((rho_fit - rho_fitted) / rho_fit_err)**2)
            chi2_red = chi2 / (len(x_fit) - 1)
        except Exception as e:
            print(f"  Fitting failed: {e}")
            B_fitted = np.nan
            B_err = np.nan
            r2_fitted = 0
            chi2_red = np.nan

        # Quadratic fit (small x only, weighted)
        small_x_mask = x_fit < min(3.0, x_cutoff)
        if np.sum(small_x_mask) > 3:
            x_quad = x_fit[small_x_mask]
            rho_quad = rho_fit[small_x_mask]
            rho_quad_err = rho_fit_err[small_x_mask]
            weights_quad = 1.0 / rho_quad_err**2

            # Theory x² (no fit)
            rho_theory_quad = A_theory_quad * x_quad**2
            ss_res_quad_w = np.sum(weights_quad * (rho_quad - rho_theory_quad)**2)
            ss_tot_quad_w = np.sum(weights_quad * (rho_quad - np.average(rho_quad, weights=weights_quad))**2)
            r2_theory_quad = 1 - ss_res_quad_w / ss_tot_quad_w if ss_tot_quad_w > 0 else 0
        else:
            r2_theory_quad = 0

        print(f"\n[FIT RESULTS (WEIGHTED)]")
        print(f"  Full theory sinh²(Bx)/cosh(2Bx):")
        print(f"    R² (weighted) = {r2_theory:.4f}")
        print(f"    R² (unweighted) = {r2_theory_unweighted:.4f}")
        if not np.isnan(B_fitted):
            print(f"  Fitted B = {B_fitted:.6f} ± {B_err:.6f}")
            print(f"  B_fit / B_theory = {B_fitted/B_theory:.4f}")
            print(f"  Fitted R² (weighted) = {r2_fitted:.4f}")
            print(f"  χ²/dof = {chi2_red:.3f}")
        print(f"  Small-x theory (B²x²): R² = {r2_theory_quad:.4f}")

        # ========================================================================
        # PLOTTING
        # ========================================================================


        fig, ax = plt.subplots(figsize=(14, 7))

        x_plot = np.linspace(0.01, x_cutoff + 0.5, 200)

        # Data points with error bars
        ax.errorbar(x_fit * length_to_um, rho_fit, yerr=rho_fit_err,
                    fmt='s', ms=7, alpha=0.8, color='b',
                    ecolor='b', capsize=3, capthick=1,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label='Data (2σ threshold, residence-weighted)', zorder=10)

        # Full theory curve (no free parameters)
        ax.plot(x_plot * length_to_um, theory_func(x_plot, B_theory), '-', lw=3, 
                color='#2E7D32', alpha=0.8,
                label=f'Theory: sinh²(κ₁x)/cosh(2κ₁x) (R²={r2_theory:.3f})')

        # Quadratic approximation (small x only)
        if r2_theory_quad > 0:
            x_plot_small = x_plot[x_plot < min(3.0, x_cutoff)]
            ax.plot(x_plot_small * length_to_um, A_theory_quad * x_plot_small**2,
                    '--', lw=2.5, color='#7B1FA2', alpha=0.8,
                    label=f'Small-x: (κ₁x)² (R²={r2_theory_quad:.3f})')

        # Best fit (if different from theory)
        if not np.isnan(B_fitted) and abs(B_fitted - B_theory) / B_theory > 0.1:
            ax.plot(x_plot * length_to_um, theory_func(x_plot, B_fitted), '-.', lw=2.5,
                    color='#E65100', alpha=0.8,
                    label=f'Fit: B={B_fitted:.4f}±{B_err:.4f} (R²={r2_fitted:.3f}, χ²/dof={chi2_red:.2f})')

        # Mark cutoff
        ax.axvline(x_cutoff * length_to_um, color='red', ls=':', lw=2, alpha=0.5,
                   label=f'Cutoff: x = {x_cutoff * length_to_um:.1f} μm')

        # Labels and styling
        ax.set_xlabel('Position x (μm)', fontsize=13)
        ax.set_ylabel(r'$\rho_a(x)$', fontsize=13)

        title_str = f'Waveguide Population Ratio: $\\rho_{{a}}(x)$ ({regime_name} regime)\n'
        title_str += f'y-threshold = {y_threshold_physical:.2f} (2σ from well center)'
        ax.set_title(title_str, fontsize=13, fontweight='bold')

        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.3 * length_to_um, (x_cutoff + 0.8) * length_to_um)
        ax.set_ylim(-0.01, max(0.08, np.max(rho_fit) * 1.3))

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'rho_a_analysis_physical.png'), dpi=200)
        plt.draw()
        plt.pause(0.001)

        print(f"\n[SAVED] {os.path.join(results_dir, 'rho_a_analysis2_physical.png')}")
        

#%%=========== VELOCITY DECELERATION ANALYSIS =================        

if VELOCITY_ANALYSIS and COMPUTE_TRAJECTORIES and is_evanescent and len(traj_hist_X_roi) > 0:
    import os
    
    print("\n" + "="*70)
    print("VELOCITY DECELERATION ANALYSIS WITH THEORY COMPARISON")
    print("(Residence-time corrected: one sample per trajectory per bin)")
    print("="*70)
    
    # ========================================================================
    # EXTRACT SIMULATION PARAMETERS
    # ========================================================================
    sigma_x = sigx
    sigma_k = 1.0 / (np.sqrt(2) * sigma_x)
    k0 = kx0
    kappa_2 = (kappa0 + kappa1) / 2.0
    kappa_1_paper = abs(kappa1 - kappa0) / 2.0
    omega_prime = k0  # In units where ℏ/m = 1
    hbar_over_m = 1.0
    
    print(f"\nSimulation Parameters:")
    print(f"  σ_x = {sigma_x:.4f}")
    print(f"  σ_k = {sigma_k:.4f}")
    print(f"  k₀ = {k0:.4f}")
    print(f"  ω'₀ = {omega_prime:.4f}")
    print(f"  κ₁ (paper) = {kappa_1_paper:.4f}")
    print(f"  κ₂ = {kappa_2:.4f}")
    
    # Theory formulas
    denominator = kappa_2**2 - kappa_1_paper**2
    
    # Simple theory (x-independent, κ₂ >> κ₁ approximation)
    a_simple = hbar_over_m * sigma_k**2 * k0**2 * kappa_2 / denominator
    
    print(f"\nSimple Theory (x-independent):")
    print(f"  a_simple = {a_simple:.6f}")
    
    def v_full_theory(x, t):
        """Full theory with position dependence
        
        v̄_x(x,t) = -(ℏ/m)·σ_k²·k₀·ω'₀·t / (κ₂²-κ₁²) · [κ₂ + κ₁·tanh(...)]
        """
        if t == 0:
            return 0.0
        
        prefactor = -hbar_over_m * sigma_k**2 * k0 * omega_prime * t / denominator
        
        # Tanh argument: 2κ₁x + (2σ_k²k₀²κ₂κ₁)/(κ₂²-κ₁²)² · x²
        tanh_arg = (2 * kappa_1_paper * x + 
                    (2 * sigma_k**2 * k0**2 * kappa_2 * kappa_1_paper) / denominator**2 * x**2)
        
        bracket = kappa_2 + kappa_1_paper * np.tanh(tanh_arg)
        
        return prefactor * bracket
    
    def v_simple_theory(t):
        """Simple theory (x-averaged, κ₂ >> κ₁)"""
        return -a_simple * t
    
    # ========================================================================
    # STEP 1: COLLECT POST-TURNAROUND TRAJECTORY DATA
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 1: COLLECTING POST-TURNAROUND TRAJECTORY DATA")
    print("-"*70)
    
    x_min_threshold = 0.5  # Filter shallow reflections
    
    # Storage for trajectory data (not concatenated!)
    traj_data = []
    
    x_turn_list = []    # Turnaround positions
    t_turn_list = []    # Absolute turnaround times
    
    n_excluded = 0
    
    for traj_idx in traj_hist_X_roi.keys():
        x_traj = np.array(traj_hist_X_roi[traj_idx])
        y_traj = np.array(traj_hist_Y_roi[traj_idx])
        vx_traj = np.array(traj_hist_VX_roi[traj_idx])
        t_traj = np.array(traj_times_roi[traj_idx])
        
        # Skip if never in barrier
        if not np.any(x_traj > 0):
            n_excluded += 1
            continue
        
        # Find turnaround (maximum x in barrier)
        mask_barrier = x_traj > 0
        x_barrier = x_traj[mask_barrier]
        t_barrier = t_traj[mask_barrier]
        
        idx_turn_local = np.argmax(x_barrier)
        x_turn = x_barrier[idx_turn_local]
        t_turn = t_barrier[idx_turn_local]
        
        # Filter by penetration depth
        if x_turn < x_min_threshold:
            n_excluded += 1
            continue
        
        # Get post-turnaround data
        idx_turn_global = np.where(mask_barrier)[0][idx_turn_local]
        
        if idx_turn_global >= len(x_traj) - 2:
            n_excluded += 1
            continue
        
        x_after = x_traj[idx_turn_global:]
        y_after = y_traj[idx_turn_global:]
        vx_after = vx_traj[idx_turn_global:]
        t_after = t_traj[idx_turn_global:]
        
        # Time relative to this trajectory's turnaround
        t_shifted = t_after - t_turn
        
        # Store trajectory data (keep separate, don't concatenate!)
        traj_data.append({
            't_shifted': t_shifted,
            'vx': vx_after,
            'x': x_after,
            'y': y_after,
            'x_turn': x_turn,
            't_turn': t_turn
        })
        
        x_turn_list.append(x_turn)
        t_turn_list.append(t_turn)
    
    x_turn_arr = np.array(x_turn_list)
    t_turn_arr = np.array(t_turn_list)
    
    n_total = len(traj_hist_X_roi)
    n_valid = len(traj_data)
    
    print(f"  Total trajectories: {n_total}")
    print(f"  Valid (x_turn > {x_min_threshold}): {n_valid}")
    print(f"  Excluded: {n_excluded}")
    
    # Check turnaround time spread
    dt_turn_abs = t_turn_arr.max() - t_turn_arr.min()
    print(f"\n  Turnaround time spread (absolute): {dt_turn_abs:.2f}")
    print(f"  Mean turnaround time: {t_turn_arr.mean():.2f}")
    print(f"  Relative spread: {dt_turn_abs/t_turn_arr.mean()*100:.1f}%")
    
    if n_valid == 0:
        print("\n[ERROR] No valid trajectories for analysis!")
    else:
        # ====================================================================
        # STEP 2: SAMPLE ONCE PER TRAJECTORY AT EACH TIME BIN CENTER
        # ====================================================================
        print("\n" + "-"*70)
        print("STEP 2: SAMPLING AT TIME BIN CENTERS")
        print("-"*70)
        
        # Define time bins
        t_max_all = max([traj['t_shifted'].max() for traj in traj_data])
        t_max = min(t_max_all, 100)  # Limit to reasonable duration
        n_bins = 25
        t_bin_centers = np.linspace(0, t_max, n_bins)
        
        print(f"  Time bins: {n_bins} bins from 0 to {t_max:.1f}")
        print(f"  Sampling method: interpolate to bin centers")
        
        # Storage for binned quantities
        vx_means = []
        vx_sems = []
        x_means = []        # Mean x-position at each time
        x_percentiles = []  # (x_25, x_50, x_75) at each time
        y_means = []
        n_samples = []
        
        for t_c in t_bin_centers:
            # Sample from each trajectory at time t_c
            vx_samples = []
            x_samples = []
            y_samples = []
            
            for traj in traj_data:
                t_traj = traj['t_shifted']
                
                # Check if trajectory has data at this time
                if t_c < t_traj.min() or t_c > t_traj.max():
                    continue  # Trajectory doesn't cover this time
                
                # Interpolate to get values at t_c
                vx_interp = np.interp(t_c, t_traj, traj['vx'])
                x_interp = np.interp(t_c, t_traj, traj['x'])
                y_interp = np.interp(t_c, t_traj, traj['y'])
                
                vx_samples.append(vx_interp)
                x_samples.append(x_interp)
                y_samples.append(y_interp)
            
            # Compute statistics (one sample per trajectory = no residence bias!)
            if len(vx_samples) > 20:  # Minimum trajectories per bin
                vx_samples = np.array(vx_samples)
                x_samples = np.array(x_samples)
                y_samples = np.array(y_samples)
                
                vx_means.append(np.mean(vx_samples))
                vx_sems.append(np.std(vx_samples, ddof=1) / np.sqrt(len(vx_samples)))
                x_means.append(np.mean(x_samples))
                x_percentiles.append([
                    np.percentile(x_samples, 25),
                    np.percentile(x_samples, 50),
                    np.percentile(x_samples, 75)
                ])
                y_means.append(np.mean(y_samples))
                n_samples.append(len(vx_samples))
            else:
                vx_means.append(np.nan)
                vx_sems.append(np.nan)
                x_means.append(np.nan)
                x_percentiles.append([np.nan, np.nan, np.nan])
                y_means.append(np.nan)
                n_samples.append(0)
        
        vx_means = np.array(vx_means)
        vx_sems = np.array(vx_sems)
        x_means = np.array(x_means)
        x_percentiles = np.array(x_percentiles)
        y_means = np.array(y_means)
        n_samples = np.array(n_samples)
        
        valid = ~np.isnan(vx_means)
        
        print(f"  Valid time bins: {np.sum(valid)} / {n_bins}")
        print(f"  Mean trajectories/bin: {np.mean(n_samples[valid]):.0f}")
        print(f"  (Each trajectory contributes exactly once per bin)")
        
        # ================================================================
        # STEP 3: FIT MEASURED DECELERATION
        # ================================================================
        print("\n" + "-"*70)
        print("STEP 3: LINEAR FIT TO SIMULATION DATA")
        print("-"*70)
        
        t_fit = t_bin_centers[valid]
        vx_fit = vx_means[valid]
        vx_err_fit = vx_sems[valid]
        
        def linear_through_origin(t, a):
            """v(t) = -a·t, assuming v=0 at turnaround"""
            return -a * t
        
        try:
            popt, pcov = curve_fit(linear_through_origin, t_fit, vx_fit,
                                  sigma=vx_err_fit, absolute_sigma=True)
            
            a_measured = popt[0]
            a_err = np.sqrt(np.diag(pcov))[0]
            
            # Goodness of fit
            vx_pred = linear_through_origin(t_fit, a_measured)
            residuals = vx_fit - vx_pred
            chi2 = np.sum((residuals / vx_err_fit)**2)
            dof = len(t_fit) - 1
            chi2_red = chi2 / dof
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((vx_fit - np.mean(vx_fit))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            print(f"  ✓ Fit successful")
            print(f"  a_measured = {a_measured:.6f} ± {a_err:.6f}")
            print(f"  R² = {r_squared:.4f}")
            print(f"  χ²/dof = {chi2_red:.2f}")
            
            fit_success = True
            
        except Exception as e:
            print(f"  ✗ Fit failed: {e}")
            fit_success = False
            a_measured = np.nan
            a_err = np.nan
        
        # ================================================================
        # STEP 4: THEORY COMPARISON
        # ================================================================
        print("\n" + "-"*70)
        print("STEP 4: THEORY COMPARISON")
        print("-"*70)
        
        # Evaluate theory at measured x(t) positions (for plotting)
        x_mean_fit = x_means[valid]
        x_25_fit = x_percentiles[valid, 0]
        x_75_fit = x_percentiles[valid, 2]
        
        v_theory_simple = v_simple_theory(t_fit)
        v_theory_full_mean = np.array([v_full_theory(x, t) for x, t in zip(x_mean_fit, t_fit)])
        v_theory_full_25 = np.array([v_full_theory(x, t) for x, t in zip(x_25_fit, t_fit)])
        v_theory_full_75 = np.array([v_full_theory(x, t) for x, t in zip(x_75_fit, t_fit)])
        
        # Compute theory deceleration at mean turnaround position (direct calculation)
        x_mean_turn = np.mean(x_turn_list)
        
        tanh_arg_turn = (2 * kappa_1_paper * x_mean_turn + 
                         (2 * sigma_k**2 * k0**2 * kappa_2 * kappa_1_paper) / denominator**2 * x_mean_turn**2)
        
        bracket_turn = kappa_2 + kappa_1_paper * np.tanh(tanh_arg_turn)
        
        a_theory_full = hbar_over_m * sigma_k**2 * k0 * omega_prime / denominator * bracket_turn
        
        print(f"\n  Theory predictions:")
        print(f"    a_simple (x-independent): {a_simple:.6f}")
        print(f"    a_full (at ⟨x_turn⟩={x_mean_turn:.3f}): {a_theory_full:.6f}")
        
        # Quantify x-dependence
        x_dependence = abs(a_theory_full - a_simple) / a_simple * 100
        print(f"\n  X-dependence: {x_dependence:.1f}% correction from simple to full theory")
        
        if x_dependence < 5:
            print(f"    → Weak x-dependence (κ₂>>κ₁ regime valid)")
        elif x_dependence < 20:
            print(f"    → Moderate x-dependence (full theory preferred)")
        else:
            print(f"    → Strong x-dependence (must use full theory)")
        
        if fit_success:
            ratio_simple = a_measured / a_simple
            ratio_full = a_measured / a_theory_full
            
            print(f"\n  Comparison with measurement:")
            print(f"    Measured/Simple: {ratio_simple:.3f}")
            print(f"    Measured/Full:   {ratio_full:.3f}")
            
            # Determine best match
            dev_simple = abs(ratio_simple - 1.0)
            dev_full = abs(ratio_full - 1.0)
            
            print(f"\n  Agreement quality:")
            if dev_full < 0.05:
                print(f"    ✓✓✓ EXCELLENT agreement with full theory (<5% deviation)")
            elif dev_full < 0.10:
                print(f"    ✓✓ VERY GOOD agreement with full theory (<10% deviation)")
            elif dev_full < 0.20:
                print(f"    ✓ GOOD agreement with full theory (<20% deviation)")
            elif dev_simple < 0.20:
                print(f"    ✓ GOOD agreement with simple theory (<20% deviation)")
            else:
                print(f"    ~ Moderate agreement (>20% deviation)")
                print(f"      → Check: parameter extraction, units, approximations")
                
        # ================================================================
        # EXTRACT CONVERSION FACTORS (add this before plotting)
        # ================================================================
        # Length conversion
        L0 = y0_exp / y0  # Should already be defined earlier in your code
        length_to_um = L0 * 1e6  # Code units → micrometers
        
        # Time conversion
        T0 = hbar_SI / E0  # Should already be defined earlier
        time_to_ps = T0 * 1e12  # Code units → picoseconds
        
        print(f"\n[UNIT CONVERSIONS FOR PLOTS]")
        print(f"  Length: 1 code unit = {length_to_um:.4f} μm")
        print(f"  Time:   1 code unit = {time_to_ps:.4f} ps")
        
        # ================================================================
        # COMPUTE AVERAGE DWELL TIME (ALL TRAJECTORIES)
        # ================================================================
        print("\n" + "-"*70)
        print("COMPUTING AVERAGE DWELL TIME (ALL TRAJECTORIES)")
        print("-"*70)
        
        # Trajectories that entered the barrier (from ROI)
        n_entered = len(traj_hist_X_roi)
        
        # Total trajectories launched
        n_total = n_traj  # Defined earlier in simulation
        
        # Trajectories that never entered
        n_never_entered = n_total - n_entered
        
        print(f"  Total trajectories launched: {n_total}")
        print(f"  Entered barrier (x>0): {n_entered}")
        print(f"  Never entered: {n_never_entered}")
        
        # Compute dwell times for those that entered (no x_turn filtering!)
        entered_dwell_times = []
        
        for traj_idx in traj_hist_X_roi.keys():
            x_traj = np.array(traj_hist_X_roi[traj_idx])
            t_traj = np.array(traj_times_roi[traj_idx])
            
            # Entry time (first crossing into barrier)
            idx_entry = np.where(x_traj > 0)[0][0]
            t_entry = t_traj[idx_entry]
            
            # Find last index where in barrier
            in_barrier = x_traj > 0
            idx_last_in = np.where(in_barrier)[0][-1]
            
            # Check if trajectory has exited
            if idx_last_in < len(x_traj) - 1:
                # Trajectory exited - use next point as exit time
                t_exit = t_traj[idx_last_in + 1]
            else:
                # Trajectory still in barrier at end of simulation - use last time
                t_exit = t_traj[-1]
            
            dwell_time = t_exit - t_entry
            entered_dwell_times.append(dwell_time)
        
        entered_dwell_times = np.array(entered_dwell_times)
        
        # Statistics for entered trajectories only
        dwell_mean_entered = np.mean(entered_dwell_times)
        dwell_std_entered = np.std(entered_dwell_times, ddof=1)
        dwell_se_entered = dwell_std_entered / np.sqrt(n_entered)
        
        # Combine: n_entered dwell times + n_never_entered zeros
        all_dwell_times = np.concatenate([entered_dwell_times, np.zeros(n_never_entered)])
        
        # Statistics for all trajectories
        dwell_mean_all = np.mean(all_dwell_times)
        dwell_std_all = np.std(all_dwell_times, ddof=1)
        dwell_se_all = dwell_std_all / np.sqrt(n_total)
        
        entry_fraction = n_entered / n_total
        
        print(f"\n  Entry fraction: {entry_fraction*100:.2f}%")
        print(f"\n  Dwell time (ONLY entered trajectories):")
        print(f"    Mean: {dwell_mean_entered:.2f} ± {dwell_se_entered:.2f}")
        print(f"    Std:  {dwell_std_entered:.2f}")
        print(f"\n  Dwell time (ALL trajectories, zeros for non-entered):")
        print(f"    Mean: {dwell_mean_all:.2f} ± {dwell_se_all:.2f}")
        print(f"    Std:  {dwell_std_all:.2f}")
        print(f"\n  Check: mean_all / mean_entered = {dwell_mean_all/dwell_mean_entered:.4f}")
        print(f"         entry_fraction = {entry_fraction:.4f}")
        
        # Wigner delay comparison
        if 'kappa_2' in locals() and 'k0' in locals():
            tau_wigner = 1.0 / (k0 * kappa_2)
            print(f"\n  Wigner delay τ_W = 1/(k₀·κ₂) = {tau_wigner:.2f}")
            print(f"  Ratio dwell(all)/τ_W = {dwell_mean_all/tau_wigner:.4f}")
            print(f"  Ratio dwell(entered)/τ_W = {dwell_mean_entered/tau_wigner:.4f}")
        
        # Convert to physical units
        if 'time_to_ps' in locals():
            print(f"\n  Physical units:")
            print(f"    Dwell (all):     {dwell_mean_all * time_to_ps:.2f} ps")
            print(f"    Dwell (entered): {dwell_mean_entered * time_to_ps:.2f} ps")
            if 'tau_wigner' in locals():
                print(f"    Wigner delay:    {tau_wigner * time_to_ps:.2f} ps")
        
        # ================================================================
        # PLOT 1: VELOCITY DECELERATION WITH THEORY (PHYSICAL UNITS)
        # ================================================================
        print("\n" + "-"*70)
        print("CREATING VELOCITY DECELERATION PLOT")
        print("-"*70)
        
        fig1, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Convert to physical units
        t_fit_ps = t_fit * time_to_ps
        vx_fit_physical = vx_fit * length_to_um / time_to_ps  # velocity in μm/ps
        vx_err_fit_physical = vx_err_fit * length_to_um / time_to_ps
        a_measured_physical = a_measured * length_to_um / time_to_ps**2  # acceleration in μm/ps²
        
        # Simulation data
        ax.errorbar(t_fit_ps, vx_fit_physical, yerr=vx_err_fit_physical,
                    fmt='o', color='C0', markersize=6, capsize=3,
                    label=f'Simulation (residence-corrected)', alpha=0.7, zorder=3)
        
        # Fitted line
        if fit_success:
            t_plot = np.linspace(0, t_fit.max(), 100)
            t_plot_ps = t_plot * time_to_ps
            v_fit_line = linear_through_origin(t_plot, a_measured)
            v_fit_line_physical = v_fit_line * length_to_um / time_to_ps
            
            ax.plot(t_plot_ps, v_fit_line_physical, '-', color='C0', linewidth=1.5,
                    alpha=0.5, label=f'Fit: a={a_measured_physical:.4f} μm/ps²')
        
        # Convert theory curves
        v_theory_simple_physical = v_theory_simple * length_to_um / time_to_ps
        v_theory_full_mean_physical = v_theory_full_mean * length_to_um / time_to_ps
        v_theory_full_25_physical = v_theory_full_25 * length_to_um / time_to_ps
        v_theory_full_75_physical = v_theory_full_75 * length_to_um / time_to_ps
        
        a_simple_physical = a_simple * length_to_um / time_to_ps**2
        a_theory_full_physical = a_theory_full * length_to_um / time_to_ps**2
        
        # Simple theory
        ax.plot(t_fit_ps, v_theory_simple_physical, '--', color='C1', linewidth=2,
                label=f'Simple theory (a={a_simple_physical:.4f} μm/ps²)', zorder=2)
        
        # Full theory with x-variation band
        ax.plot(t_fit_ps, v_theory_full_mean_physical, '-', color='C2', linewidth=2,
                label=f'Full theory at ⟨x⟩(t) (a={a_theory_full_physical:.4f} μm/ps²)', zorder=2)
        
        ax.fill_between(t_fit_ps, v_theory_full_25_physical, v_theory_full_75_physical,
                        alpha=0.2, color='C2', 
                        label='Full theory (x-variation: 25th-75th %ile)',
                        zorder=1)
        
        # Formatting
        ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time after turnaround (ps)', fontsize=12)
        ax.set_ylabel('Mean velocity $\\langle v_x \\rangle$ (μm/ps)', fontsize=12)
        ax.set_title('Post-Turnaround Velocity Deceleration vs Theory\n(Physical Units)', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'velocity_deceleration_with_theory.png'),
                    dpi=150, bbox_inches='tight')
        print("→ Velocity plot saved")
        
        # ================================================================
        # COLLECT TIMING DATA FOR PLOT 2
        # ================================================================
        print("\n" + "-"*70)
        print("COLLECTING TIMING DATA FOR CORRELATION PLOT")
        print("-"*70)
        
        t_entry_list = []
        t_exit_list = []
        t_turn_timing_list = []
        x_turn_timing = []
        dt_in_list = []
        dt_out_list = []
        
        for traj_idx in traj_hist_X_roi.keys():
            x_traj = np.array(traj_hist_X_roi[traj_idx])
            t_traj = np.array(traj_times_roi[traj_idx])
            
            if not np.any(x_traj > 0):
                continue
            
            # Entry time (first crossing into barrier)
            idx_entry = np.where(x_traj > 0)[0][0]
            t_entry = t_traj[idx_entry]
            
            # Turnaround
            mask_barrier = x_traj > 0
            x_barrier = x_traj[mask_barrier]
            t_barrier = t_traj[mask_barrier]
            idx_turn_local = np.argmax(x_barrier)
            t_turn = t_barrier[idx_turn_local]
            x_turn = x_barrier[idx_turn_local]
            
            # Filter by depth
            if x_turn < x_min_threshold:
                continue
            
            # Exit time
            idx_turn_global = np.where(mask_barrier)[0][idx_turn_local]
            x_after_turn = x_traj[idx_turn_global:]
            t_after_turn = t_traj[idx_turn_global:]
            
            exit_mask = x_after_turn < 0.1
            if np.any(exit_mask):
                idx_exit_local = np.where(exit_mask)[0][0]
                t_exit = t_after_turn[idx_exit_local]
                
                dt_in = t_turn - t_entry
                dt_out = t_exit - t_turn
                
                t_entry_list.append(t_entry)
                t_exit_list.append(t_exit)
                t_turn_timing_list.append(t_turn)
                x_turn_timing.append(x_turn)
                dt_in_list.append(dt_in)
                dt_out_list.append(dt_out)
        
        # Convert to arrays
        t_entry_arr = np.array(t_entry_list)
        t_turn_timing_arr = np.array(t_turn_timing_list)  # Use from earlier collection (STEP 1)
        t_exit_arr = np.array(t_exit_list)
        x_turn_timing_arr = np.array(x_turn_timing)
        dt_in_arr = np.array(dt_in_list)
        dt_out_arr = np.array(dt_out_list)
        
        print(f"Timing data collected for {len(x_turn_timing)} trajectories")
        
        # ================================================================
        # PLOT 2: TIMING CORRELATIONS (PHYSICAL UNITS - 2x3)
        # ================================================================
        print("\n" + "-"*70)
        print("CREATING TIMING CORRELATION PLOTS")
        print("-"*70)
        
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Convert timing arrays to physical units
        t_entry_ps = t_entry_arr * time_to_ps
        t_turn_ps = t_turn_timing_arr * time_to_ps
        t_exit_ps = t_exit_arr * time_to_ps
        x_turn_um = x_turn_timing_arr * length_to_um
        dt_in_ps = dt_in_arr * time_to_ps
        dt_out_ps = dt_out_arr * time_to_ps
        
        # --- Panel 1: Timeline (top-left) ---
        ax1 = axes[0, 0]
        
        timeline_data = [t_entry_ps, t_turn_ps, t_exit_ps]
        positions = [1, 2, 3]
        labels = ['Entry', 'Turnaround', 'Exit']
        
        parts = ax1.violinplot(timeline_data, positions=positions, widths=0.6,
                                showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('C0')
            pc.set_alpha(0.6)
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Absolute time (ps)', fontsize=11)
        ax1.set_title('Timeline: Entry → Turnaround → Exit', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Place means below x-axis labels
        for i, (pos, data) in enumerate(zip(positions, timeline_data)):
            mean_val = np.mean(data)
            ax1.text(pos, ax1.get_ylim()[0] - 0.08*(ax1.get_ylim()[1]-ax1.get_ylim()[0]), 
                     f'μ={mean_val:.1f} ps',
                     ha='center', fontsize=9, fontweight='bold')
        
        # --- Panel 2: Symmetry Check (top-middle) ---
        ax2 = axes[0, 1]
        
        ax2.scatter(dt_in_ps, dt_out_ps, alpha=0.3, s=10, color='C0')
        
        max_dt = max(dt_in_ps.max(), dt_out_ps.max())
        ax2.plot([0, max_dt], [0, max_dt], 'k--', linewidth=2, label='Symmetric')
        
        corr_symmetry = np.corrcoef(dt_in_ps, dt_out_ps)[0,1]
        ax2.text(0.05, 0.95, f'ρ = {corr_symmetry:.3f}', 
                 transform=ax2.transAxes, fontsize=11, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Time in (entry → turnaround) (ps)', fontsize=11)
        ax2.set_ylabel('Time out (turnaround → exit) (ps)', fontsize=11)
        ax2.set_title('Symmetry Check: Δt_in vs Δt_out', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # --- Panel 3: Distribution info (top-right) ---
        ax3 = axes[0, 2]
        ax3.axis('off')
        
        summary_text = "TIMING STATISTICS\n" + "="*30 + "\n\n"
        summary_text += f"Entry time:\n"
        summary_text += f"  Mean:   {t_entry_ps.mean():.1f} ps\n"
        summary_text += f"  Std:    {t_entry_ps.std():.1f} ps\n"
        summary_text += f"  Range:  [{t_entry_ps.min():.1f}, {t_entry_ps.max():.1f}] ps\n\n"
        
        summary_text += f"Turnaround time:\n"
        summary_text += f"  Mean:   {t_turn_ps.mean():.1f} ps\n"
        summary_text += f"  Std:    {t_turn_ps.std():.1f} ps\n"
        summary_text += f"  Range:  [{t_turn_ps.min():.1f}, {t_turn_ps.max():.1f}] ps\n\n"
        
        summary_text += f"Exit time:\n"
        summary_text += f"  Mean:   {t_exit_ps.mean():.1f} ps\n"
        summary_text += f"  Std:    {t_exit_ps.std():.1f} ps\n"
        summary_text += f"  Range:  [{t_exit_ps.min():.1f}, {t_exit_ps.max():.1f}] ps\n\n"
        
        summary_text += f"Durations:\n"
        summary_text += f"  ⟨Δt_in⟩:  {dt_in_ps.mean():.1f} ps\n"
        summary_text += f"  ⟨Δt_out⟩: {dt_out_ps.mean():.1f} ps\n"
        summary_text += f"  Ratio:    {dt_out_ps.mean()/dt_in_ps.mean():.3f}\n\n"
        
        summary_text += f"N_trajectories = {len(x_turn_timing)}"
        
        ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # --- Panel 4: Entry time vs x_turn (bottom-left) ---
        ax4 = axes[1, 0]
        
        ax4.scatter(x_turn_um, t_entry_ps, alpha=0.3, s=10, color='C3')
        
        corr_x_tentry = np.corrcoef(x_turn_um, t_entry_ps)[0,1]
        ax4.text(0.05, 0.95, f'ρ = {corr_x_tentry:.3f}',
                 transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.set_xlabel('Turnaround position $x_{turn}$ (μm)', fontsize=11)
        ax4.set_ylabel('Entry time $t_{entry}$ (ps)', fontsize=11)
        ax4.set_title('When Do Trajectories Enter?', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # --- Panel 5: x_turn vs dt_in (bottom-middle) ---
        ax5 = axes[1, 1]
        
        ax5.scatter(x_turn_um, dt_in_ps, alpha=0.3, s=10, color='C1')
        
        corr_x_dt_in = np.corrcoef(x_turn_um, dt_in_ps)[0,1]
        ax5.text(0.05, 0.95, f'ρ = {corr_x_dt_in:.3f}', 
                 transform=ax5.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax5.set_xlabel('Turnaround position $x_{turn}$ (μm)', fontsize=11)
        ax5.set_ylabel('Time to turnaround (Δt = $t_{turn} - t_{entry}$) (ps)', fontsize=11)
        ax5.set_title('Penetration Depth vs Time to Turnaround', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # --- Panel 6: x_turn vs t_turn_abs (bottom-right) ---
        ax6 = axes[1, 2]
        
        ax6.scatter(x_turn_um, t_turn_ps, alpha=0.3, s=10, color='C2')
        
        corr_x_tturn = np.corrcoef(x_turn_um, t_turn_ps)[0,1]
        ax6.text(0.05, 0.95, f'ρ = {corr_x_tturn:.3f}',
                 transform=ax6.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax6.set_xlabel('Turnaround position $x_{turn}$ (μm)', fontsize=11)
        ax6.set_ylabel('Absolute turnaround time $t_{turn}$ (ps)', fontsize=11)
        ax6.set_title('When Does Turnaround Happen?', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'timing_correlations.png'),
                    dpi=150, bbox_inches='tight')
        print("→ Timing plot saved")
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if fit_success:
            print(f"\nMeasured deceleration (residence-corrected):")
            print(f"  a_measured = {a_measured:.6f} ± {a_err:.6f}")
            
            print(f"\nTheory predictions:")
            print(f"  a_simple = {a_simple:.6f}")
            if not np.isnan(a_theory_full):
                print(f"  a_full = {a_theory_full:.6f}")
            
            print(f"\nComparison:")
            print(f"  Ratio measured/simple: {ratio_simple:.3f}")
            if not np.isnan(a_theory_full):
                print(f"  Ratio measured/full: {ratio_full:.3f}")
        
        print(f"\nTiming statistics:")
        print(f"  Asymmetry ratio (out/in): {np.mean(dt_out_arr/dt_in_arr):.3f}")
        print(f"  Turnaround spread (abs): {dt_turn_abs:.2f} ({dt_turn_abs/t_turn_arr.mean()*100:.1f}%)")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        plt.show()        

elif not COMPUTE_TRAJECTORIES:
    print("\n[ρ_a ANALYSIS] Requires COMPUTE_TRAJECTORIES = True")
elif not ANALYZE_RHO_A:
    print("\n[ρ_a ANALYSIS] Disabled (set ANALYZE_RHO_A = True to enable)")

print("\n[DONE] Simulation complete!")
if COMPUTE_TRAJECTORIES:
    print(f"[METHOD] Velocity: {VELOCITY_METHOD}, Propagator: {PROPAGATOR}")
else:
    print(f"[METHOD] Trajectories: DISABLED, Propagator: {PROPAGATOR}")
print(f"[REGIME] Analysis automatically adapted to detected regime")
print("\n" + "="*70)

# Keep all figures open - Spyder will display them
print("\n[PLOTS] All analysis plots generated")

try:
    import requests 
    requests.post("https://ntfy.sh/dlp_sim_271828", 
        data="Simulation complete. 🤖".encode(encoding='utf-8'),
        timeout=10)
except Exception:
    pass

#%% ======= Save Data ====================================
if not NOLOG:
    # Close log file and restore stdout
    _close_log()
    # Unregister atexit so it doesn't run again on Python exit
    atexit.unregister(_close_log)
if not NOLOG:
    print(f"[DATA] All outputs saved to: {results_dir}/")
print("="*70)

# Any interactive stuff after this point won't be logged

# Add at the very end (after line 3800)
if SAVE_DATA and not NOLOG:
    print("\n[DATA] Preparing to save...")
    
    if COMPUTE_TRAJECTORIES and 'traj_roi_subset' in globals() and len(traj_roi_subset) > 0:
        traj_X_subset = {k: traj_hist_X_roi[k] for k in traj_roi_subset}
        traj_Y_subset = {k: traj_hist_Y_roi[k] for k in traj_roi_subset}
        traj_T_subset = {k: traj_times_roi[k] for k in traj_roi_subset}
    else:
        traj_X_subset = {}
        traj_Y_subset = {}
        traj_T_subset = {}
        
    save_data = {

        # ═══════════════════════════════════════════════════════════
        # METADATA & REGIME
        # ═══════════════════════════════════════════════════════════
        'timestamp': timestamp if not NOLOG else datetime.now().strftime("%Y%m%d_%H%M%S"),
        'regime_name': regime_name,
        'is_evanescent': is_evanescent,
        'is_propagative': is_propagative,
        
        # ═══════════════════════════════════════════════════════════
        # GRID & DOMAIN (needed for integration, binning, axes)
        # ═══════════════════════════════════════════════════════════
        'Ny': Ny,  # For y grid reconstruction
        'dx': float(dx),
        'dy': float(dy),
        'dt': dt,
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'ROI_X1': ROI_X1,
        'ROI_X2': ROI_X2,
        'ROI_HIST_STRIDE': ROI_HIST_STRIDE,  # For trajectory time resolution
        
        # ═══════════════════════════════════════════════════════════
        # KEY PARAMETERS (for theory curves)
        # ═══════════════════════════════════════════════════════════
        'y0': y0,
        'sigma_well': sigma_well,
        'y_threshold_physical': y_threshold_physical,
        'J0_energy': J0_energy,
        'E0_R': E0_R,
        'E1_R': E1_R,
        'V_STEP': V_STEP,
        'E_inj_total': E_inj_total,
        
        # Regime-specific parameters
        'kappa0': kappa0 if is_evanescent else None,
        'kappa1': kappa1 if is_evanescent else None,
        'd0': d0,
        'd1': d1,
        'L_decay': L_decay if is_evanescent and 'L_decay' in globals() else None,
        'L_equil': L_equil if 'L_equil' in globals() else None,
        
     
        # ═══════════════════════════════════════════════════════════
        # RAW DATA: ROI TRAJECTORIES (sampled subset for visualization)
        # ═══════════════════════════════════════════════════════════        
        'traj_hist_X_roi': traj_X_subset,
        'traj_hist_Y_roi': traj_Y_subset,
        'traj_times_roi': traj_T_subset,
        
        # ═══════════════════════════════════════════════════════════
        # RAW DATA: TIME-AVERAGED DENSITY (2D field)
        # ═══════════════════════════════════════════════════════════
        'density_avg_roi': cp.asnumpy(density_sum_roi / density_count_roi) if DO_TIME_AVG_DENSITY and density_count_roi > 0 and PROPAGATOR.upper() == 'FFT' else (density_sum_roi / density_count_roi if DO_TIME_AVG_DENSITY and density_count_roi > 0 else None),
        'x_roi': x_cpu_full[ix_roi1:ix_roi2] if DO_TIME_AVG_DENSITY else None,
        'y_plot': np.linspace(float(y_min), float(y_max), Ny),
        'averaging_start_time': averaging_start_step * dt if DO_TIME_AVG_DENSITY and averaging_start_step is not None else None,
        'averaging_end_time': averaging_end_step * dt if DO_TIME_AVG_DENSITY and averaging_end_step is not None else None,
        'density_count_roi': density_count_roi if DO_TIME_AVG_DENSITY else None,
        
        # ═══════════════════════════════════════════════════════════
        # RAW DATA: Y-MIGRATION (already processed from trajectories)
        # ═══════════════════════════════════════════════════════════
        'tunneling_times': np.array(tunneling_times) if 'tunneling_times' in globals() and len(tunneling_times) > 0 else None,
        'x_positions': np.array(x_positions) if 'x_positions' in globals() and len(x_positions) > 0 else None,
        'valid_traj_indices': valid_traj_indices if 'valid_traj_indices' in globals() else None,
        
        # ═══════════════════════════════════════════════════════════
        # ANALYSIS PARAMETERS (needed to recompute fits)
        # ═══════════════════════════════════════════════════════════
        'RHO_A_SPATIAL_BIN': RHO_A_SPATIAL_BIN,  # Spatial bin width for ρ_a
        't_window_start': t_window_start if 't_window_start' in globals() else None,  # Time window for ρ_a
        't_window_end': t_window_end if 't_window_end' in globals() else None,
        
        # ═══════════════════════════════════════════════════════════
        # FIT RESULTS (optional - can be recomputed from raw data)
        # ═══════════════════════════════════════════════════════════
        
        # Exponential decay fit (evanescent)
        'L_decay_measured': L_decay_measured if 'L_decay_measured' in globals() else None,
        'r2_decay': r_squared if 'r_squared' in globals() and 'L_decay_measured' in globals() else None,
        
        # Spatial beating fit (propagative)
        'D_spatial': D_spatial if 'D_spatial' in globals() else None,
        'P_upper_arr': P_upper_arr if 'P_upper_arr' in globals() else None,
        'P_lower_arr': P_lower_arr if 'P_lower_arr' in globals() else None,
        'k_fit': k_fit if 'k_fit' in globals() else None,
        'A_fit': A_fit if 'A_fit' in globals() else None,
        'phi_fit': phi_fit if 'phi_fit' in globals() else None,
        'offset_fit': offset_fit if 'offset_fit' in globals() else None,
        'r2_spatial': r2 if 'k_fit' in globals() and 'r2' in globals() else None,
        
        # Spatial equilibration (evanescent)
        'L_equil_fit': L_equil_fit if 'L_equil_fit' in globals() else None,
        'D_final_fit': D_final_fit if 'D_final_fit' in globals() else None,
        'D_initial': D_initial if 'D_initial' in globals() else None,
        'r2_equil': r_squared if 'L_equil_fit' in globals() and 'r_squared' in globals() else None,
        
        # ρ_a analysis
        'x_centers': x_centers if 'x_centers' in globals() and 'rho_fit' in globals() else None,
        'rho_a_corrected': rho_a_corrected if 'rho_a_corrected' in globals() else None,
        'x_fit': x_fit if 'x_fit' in globals() and 'rho_fit' in globals() else None,
        'rho_fit': rho_fit if 'rho_fit' in globals() else None,
        'x_cutoff': x_cutoff if 'x_cutoff' in globals() and 'rho_fit' in globals() else None,
        'baseline': baseline if 'baseline' in globals() and 'rho_fit' in globals() else None,
        'B_theory': B_theory if 'B_theory' in globals() and 'rho_fit' in globals() else None,
        'B_fitted': B_fitted if 'B_fitted' in globals() and not np.isnan(B_fitted) else None,
        'r2_theory': r2_theory if 'r2_theory' in globals() and 'rho_fit' in globals() else None,
        'r2_fitted': r2_fitted if 'r2_fitted' in globals() and not np.isnan(B_fitted) else None,

        # ═══════════════════════════════════════════════════════════
        # DWELL TIME (Eq. (3) enforced; paper convention)
        # ═══════════════════════════════════════════════════════════
        'delta_y': delta_y,
        'sigx': sigx,
        'kx0': float(kx0),
        'sigma_k_code': sigma_k_code if 'sigma_k_code' in globals() else 1.0/(2.0*sigx),
        'dwell_time_entered_mean': dwell_time_entered_mean,
        'dwell_time_entered_se': dwell_time_entered_se,
        'dwell_time_all_mean': dwell_time_all_mean,
        'dwell_time_all_se': dwell_time_all_se,
        'dwell_wf_integral': dwell_wf_integral,
        'tau_entered_theory': tau_entered_theory,
        'tau_d2d_theory': tau_d2d_theory,
        'n_still_inside': n_still_inside,
    }
    
    # Save with compression
    print("[DATA] Compressing and writing...")
    save_path = os.path.join(results_dir, 'simulation_data.npz')
    np.savez_compressed(save_path, **save_data)
    
    print(f"\n[DATA SAVED] {save_path}")


#%% ======= BATCH JSON SUMMARY (Option B) ====================================
# Flat VEL_<run_tag>.json consumed by plot_dwell_time_from_batch.py.
# NOTE: Delta_code here is Delta = E_k - V0 (the detuning as reported in both the
# paper and Sharoglazova et al.). The old DWbatch.py instead stored Delta + hbar*J0
# (an offset internal convention); the 'detuning_convention' tag lets the plotter
# detect that offset in legacy files and refuse mixed batches.
if BATCH_MODE:
    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    def _i(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    meta = {
        'run_tag': str(RUN_TAG),
        'detuning_convention': 'paper',          # Delta = E_k - V0
        'eq3_enforced': True,                     # delta_y ADDED to V_right
        'sigma_k_convention': 'std_of_psihat_1_over_2sigx',
        # --- geometry / parameters ---
        'kx0': _f(kx0),
        'sigx': _f(sigx),
        'sigma_k': _f(1.0 / (2.0 * sigx)),        # CORRECTED: std of |psi_hat(k)|^2
        'V_STEP': _f(V_STEP),
        'Delta_code': _f(locals().get('Delta_code', None)),   # paper convention
        'delta_y': _f(locals().get('delta_y', None)),
        'J0_energy': _f(locals().get('J0_energy', None)),
        'E0_R': _f(locals().get('E0_R', None)),   # effective (Eq.3-shifted)
        'E1_R': _f(locals().get('E1_R', None)),
        'kappa0': _f(locals().get('kappa0', None)),
        'kappa1': _f(locals().get('kappa1', None)),
        'y0': _f(y0),
        'ROI': [_f(ROI_X1), _f(ROI_X2)],
        'dt': _f(dt), 'Nx': _i(Nx), 'Ny': _i(Ny),
        'n_traj': _i(n_traj), 'n_steps': _i(n_steps),
        'x_min': _f(x_min), 'x_max': _f(x_max),
        'regime_flags': {
            'd0': _f(locals().get('d0', None)),
            'd1': _f(locals().get('d1', None)),
            'evanescent': bool(locals().get('is_evanescent', False)),
            'propagative': bool(locals().get('is_propagative', False)),
        },
        'unit_conversions': {
            'L0_m': _f(locals().get('L0', None)),
            'T0_s': _f(locals().get('T0', None)),
            'length_to_um': _f(L0 * 1e6) if 'L0' in locals() else None,
            'time_to_ps': _f(T0 * 1e12) if 'T0' in locals() else None,
        },
        # --- dwell time (the quantities plot_dwell_time_from_batch.py reads) ---
        'tunneling_speed': {
            'dwell_time_entered_mean': _f(locals().get('dwell_time_entered_mean', None)),
            'dwell_time_entered_se':   _f(locals().get('dwell_time_entered_se', None)),
            'dwell_time_all_mean':     _f(locals().get('dwell_time_all_mean', None)),
            'dwell_time_all_se':       _f(locals().get('dwell_time_all_se', None)),
            'dwell_wf_integral':       _f(locals().get('dwell_wf_integral', None)),
            'tau_entered_theory':      _f(locals().get('tau_entered_theory', None)),
            'tau_d2d_theory':          _f(locals().get('tau_d2d_theory', None)),
            'n_still_inside':          _i(locals().get('n_still_inside', None)),
            'units': 'code',          # all dwell times above are in code time units
        },
    }

    _outbase = os.path.join(_BATCH_OUTDIR, f"VEL_{RUN_TAG}.json")
    with open(_outbase, 'w', encoding='utf-8') as _jf:
        json.dump(meta, _jf, indent=2)
    print(f"[BATCH] Wrote JSON summary: {_outbase}")
