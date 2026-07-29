"""
plot_dwell_time_from_batch.py  (paper-convention version, v2)

Standalone script to create the publication 2-panel dwell-time plot
(Fig. 7) from the VEL_*.json files written by DWtunneling.py, typically
via batch_sigx_sweep.py.

    python plot_dwell_time_from_batch.py batch_out

Writes dwell_time_analysis.{pdf,png} into the input directory.

NOTE: the tau_d^2D reference line is computed ANALYTICALLY here, from
V_STEP, J0_energy and kx0 in the paper convention (see tau_d_2d_code);
it is NOT read back from the simulation. The measured all-trajectory
mean agreeing with it is therefore an independent check on the run.

CONVENTIONS (this version)
--------------------------
The simulation stores the detuning in the Sharoglazova et al. convention,

    Delta_code (stored) = kx0^2/2 - V_STEP + J0          [Sharoglazova]

while the paper uses

    Delta_paper = E_k - V_0 = kx0^2/2 - V_STEP.          [paper, Eq. (20) frame]

This script converts everything to the PAPER convention:

  * J0 is recovered per run from the stored identity
        J0_code = Delta_code + V_STEP - kx0^2/2
    (exactly the relation the simulation used to set kx0), so no
    eigensolve and no hardcoded J0 are needed.
  * kx0 is recomputed from Delta_paper as a consistency check and must
    agree with the stored kx0 to machine precision.
  * Filtering and all labels use Delta_paper (meV).

sigma_k: recomputed from sigx as 1/(2*sigx). The initial packet is
psi ~ exp(-(x-x0)^2/(4*sigx^2)), so sigx is the std of |psi|^2 and the
std of |psi_hat(k)|^2 is 1/(2*sigx), matching Eq. (39) of the paper.
(The sigma_k field saved in the JSON uses 1/(sqrt(2)*sigx) and is
sqrt(2) too large -- it is bypassed.)

TIME UNITS OF STORED DWELL TIMES
--------------------------------
The 'entered' and 'all' dwell-time statistics in the JSON are not
guaranteed to be in the same units (code time vs ps). This script
therefore resolves the units of each quantity SEPARATELY:

  * DWELL_ENTERED_UNITS / DWELL_ALL_UNITS = 'code', 'ps', or 'auto'.
  * In 'auto' mode, the stored values are compared (in log space)
    against the theoretical expectation in code units
        entered:  tau = sqrt(pi/2)/(kx0*sigma_k_code)     [Eq. (62)]
        all:      tau_d^2D(k0)                            [Eqs. (25),(36)]
    and against the same expectation in ps. Whichever interpretation
    is closer (the two differ by a factor 1/T0_ps ~ 3.8) is chosen,
    and the decision is printed. A per-run debug table shows raw and
    converted values so the choice can be verified by eye.

Theory curves:
  * Conditional Bohmian dwell time, Eq. (62):
        tau = sqrt(pi/2) * m / (hbar * k0 * sigma_k)
  * Unconditional reference: 2D Buettiker dwell time over both
    transverse channels:
        kappa_pm^2 = 2*(V0 -+ J0) - k0^2
        g_pm^2     = 4 k0^2 / (k0^2 + kappa_pm^2)
        tau_d^2D   = (1/(2*k0)) * [ g+^2/(2*kappa+) + g-^2/(2*kappa-) ]
    The previous frame-mixed 1D value sqrt(V0-|Delta_S|)/(V0*sqrt(|Delta_S|))
    is still printed for reference but no longer plotted.

Top panel:    entered trajectories vs sigma_k, 1/sigma_k fit + theory
Bottom panel: all trajectories (zoomed) with mean and tau_d^2D reference

Saves PDF (vector) + PNG preview.

Usage:
    python plot_dwell_time_from_batch.py
    python plot_dwell_time_from_batch.py path/to/batch_out
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from pathlib import Path
import json
import sys
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

# Filter criteria -- PAPER convention (set to None to include all)
# MUST match DETUNING_MEV in batch_sigx_sweep.py, or every run is filtered out
# and the script exits without plotting. -0.09 is the Fig. 7 sweep.
TARGET_DETUNING_PAPER_MEV = -0.09   # Delta_paper = E_k - V0 in meV
DETUNING_TOL_MEV = 0.02             # tolerance for matching runs
REGIME_FILTER = 'evanescent'         # 'evanescent', 'propagative', or None

# Units of the stored dwell-time statistics: 'code', 'ps', or 'auto'
DWELL_ENTERED_UNITS = 'auto'
DWELL_ALL_UNITS = 'auto'
DEBUG_PRINT_RAW = True               # print per-run raw/converted table

# Output settings
OUTPUT_FILENAME = "dwell_time_analysis"   # without extension

# Figure size: 15:10 total, each panel ~15:5
FIGSIZE = (15, 10)

# Physical constants (keep identical to the simulation!)
HBAR_SI = 1.055e-34      # J s
M_SI = 6.95e-36          # kg
MEV_TO_J = 1.602e-22     # 1 meV in Joules
L0_DEFAULT_M = 2.0e-6    # fallback length unit if not in JSON

# ============================================================================
# COLOR SCHEME
# ============================================================================

COLORS = {
    'entered_data': '#4575B4',      # Blue
    'all_data': '#2E7D32',          # Green
    'fit_line': '#D55E00',          # Terracotta red
    'theory_line': '#1565C0',       # Dark blue
    'tau_d': '#7B1FA2',             # Purple
    'mean_line': 'dimgray',         # Gray
}

# ============================================================================
# FILE LOADING
# ============================================================================

def pick_batch_folder():
    """Open dialog to select batch_out folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(
            title="Select batch_out folder containing VEL_*.json files"
        )
        root.destroy()
        return folder
    except ImportError:
        print("[ERROR] tkinter not available.")
        return None


def load_runs(batch_folder):
    """Load all VEL_*.json files from batch folder."""
    pattern = str(Path(batch_folder) / "VEL_*.json")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"[ERROR] No VEL_*.json files found in {batch_folder}")
        return []

    print(f"[LOADING] Found {len(files)} JSON files")

    runs = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            data['_file'] = Path(filepath).name
            runs.append(data)
        except Exception as e:
            print(f"[WARNING] Could not load {filepath}: {e}")

    return runs

# ============================================================================
# UNIT HELPERS
# ============================================================================

def get_units(run):
    """Return (L0_m, T0_s, E0_meV) for a run."""
    unit_conv = run.get('unit_conversions', {})
    L0 = unit_conv.get('L0_m', L0_DEFAULT_M)
    E0_J = HBAR_SI**2 / (M_SI * L0**2)
    T0 = unit_conv.get('T0_s', None)
    if T0 is None:
        T0 = HBAR_SI / E0_J
    E0_meV = E0_J / MEV_TO_J
    return L0, T0, E0_meV

# ============================================================================
# CONVENTION CONVERSION (Sharoglazova -> paper)
# ============================================================================

def derive_conventions(run):
    """
    From the stored (Sharoglazova-convention) quantities, derive all
    paper-convention quantities for one run. Returns a dict or None.

    Stored identity used by the simulation:
        kx0^2/2 = Delta_sharo + V_STEP - J0
    =>  J0_code          = Delta_sharo + V_STEP - kx0^2/2
        Delta_paper_code = kx0^2/2 - V_STEP = Delta_sharo - J0_code
    """
    kx0 = run.get('kx0', None)
    V_STEP = run.get('V_STEP', None)
    Delta_stored = run.get('Delta_code', None)

    if kx0 is None or V_STEP is None or Delta_stored is None:
        return None
    kx0 = float(kx0)
    V_STEP = float(V_STEP)
    Delta_stored = float(Delta_stored)

    Ek_code = 0.5 * kx0**2                      # longitudinal kinetic energy
    Delta_paper_code = Ek_code - V_STEP         # paper convention (always)

    # Convention of the STORED Delta_code:
    #   - DWtunneling.py tags 'detuning_convention': 'paper' and stores
    #     Delta_code = E_k - V0 directly.
    #   - Old DWbatch.py stored the Sharoglazova value Delta + hbar*J0 and left
    #     the tag absent. In that case J0 is recovered from the stored identity
    #     kx0^2/2 = Delta_sharo + V_STEP - J0.
    conv_tag = run.get('detuning_convention', None)
    if conv_tag == 'paper':
        Delta_sharo = None
        J0_code = run.get('J0_energy', None)
        if J0_code is not None:
            J0_code = float(J0_code)
        else:
            # Fall back to the recovered identity only if J0 is not stored.
            J0_code = (Delta_stored + Ek_code + V_STEP) - 2.0 * Ek_code  # = Delta_stored + V_STEP - Ek_code
            J0_code = None if abs(Delta_stored - Delta_paper_code) > 1e-12 else J0_code
            # (For a clean paper-convention run Delta_stored == Delta_paper_code,
            #  so the recovered-identity branch is not applicable; require J0_energy.)
        Delta_sharo_code = (Delta_paper_code + J0_code) if J0_code is not None else None
    else:
        # Legacy Sharoglazova convention: recover J0 from the identity.
        Delta_sharo = Delta_stored
        J0_code = Delta_stored + V_STEP - Ek_code   # recovered coupling
        Delta_sharo_code = Delta_stored

    if J0_code is None or J0_code <= 0:
        print(f"[WARNING] Could not determine a positive J0 "
              f"(kx0={kx0:.4f}, V={V_STEP:.4f}, Delta_stored={Delta_stored:.4f}, "
              f"convention={conv_tag or 'legacy_sharoglazova'}) -- run skipped. "
              "For paper-convention runs, ensure 'J0_energy' is stored.")
        return None

    # Consistency check: recomputing kx0 from Delta_paper must reproduce
    # the stored kx0 (documents the convention in the data files).
    kx0_check = np.sqrt(2.0 * (V_STEP + Delta_paper_code))
    if abs(kx0_check - kx0) > 1e-9 * max(1.0, abs(kx0)):
        print(f"[WARNING] kx0 consistency check failed: "
              f"stored {kx0:.10f} vs recomputed {kx0_check:.10f}")

    L0, T0, E0_meV = get_units(run)

    return {
        'kx0': kx0,
        'V_STEP': V_STEP,
        'J0_code': J0_code,
        'Delta_sharo_code': Delta_sharo_code,
        'Delta_paper_code': Delta_paper_code,
        'Delta_sharo_meV': (Delta_sharo_code * E0_meV) if Delta_sharo_code is not None else None,
        'Delta_paper_meV': Delta_paper_code * E0_meV,
        'J0_meV': J0_code * E0_meV,
        'L0': L0,
        'T0': T0,
        'E0_meV': E0_meV,
    }

# ============================================================================
# FILTERING (paper convention)
# ============================================================================

def filter_runs(runs, target_paper_mev=None, regime='evanescent'):
    """Filter runs by paper-convention detuning and regime."""
    filtered = []

    for run in runs:
        conv = derive_conventions(run)
        if conv is None:
            continue

        # Regime check from first principles (paper convention):
        # fully evanescent  <=>  kx0^2 < 2*(V0 - J0)  <=>  Delta_sharo < 0
        kp2 = 2.0 * (conv['V_STEP'] - conv['J0_code']) - conv['kx0']**2
        km2 = 2.0 * (conv['V_STEP'] + conv['J0_code']) - conv['kx0']**2
        is_evanescent = (kp2 > 0) and (km2 > 0)
        is_propagative = (kp2 < 0) and (km2 < 0)

        if regime == 'evanescent' and not is_evanescent:
            continue
        if regime == 'propagative' and not is_propagative:
            continue

        if target_paper_mev is not None:
            if abs(conv['Delta_paper_meV'] - target_paper_mev) > DETUNING_TOL_MEV:
                continue

        run['_conv'] = conv
        filtered.append(run)

    return filtered

# ============================================================================
# THEORY (code units; conversion to ps happens at the call site)
# ============================================================================

def tau_entered_theory_code(kx0, sigma_k_code):
    """Conditional Bohmian dwell time, Eq. (62), in code units."""
    return np.sqrt(np.pi / 2) / (kx0 * sigma_k_code)


def tau_d_2d_code(kx0, V_STEP, J0):
    """
    2D Buettiker dwell time over both transverse channels (code units):

        kappa_pm^2 = 2*(V0 -+ J0) - k0^2
        g_pm^2     = 4 k0^2 / (k0^2 + kappa_pm^2)
        tau_d      = (1/k0) * (1/2) * [ g+^2/(2 kappa+) + g-^2/(2 kappa-) ]

    The 1/2 is the equal weight of phi_+ / phi_- for a wave incident in
    the main waveguide (phi_m = (phi_+ + phi_-)/sqrt(2)).
    Returns None if not fully evanescent.
    """
    kp2 = 2.0 * (V_STEP - J0) - kx0**2      # kappa_+^2 (symmetric mode)
    km2 = 2.0 * (V_STEP + J0) - kx0**2      # kappa_-^2 (antisymmetric mode)

    if kp2 <= 0 or km2 <= 0:
        return None                          # not fully evanescent

    kp, km = np.sqrt(kp2), np.sqrt(km2)
    gp2 = 4.0 * kx0**2 / (kx0**2 + kp2)
    gm2 = 4.0 * kx0**2 / (kx0**2 + km2)

    return (0.5 / kx0) * (gp2 / (2.0 * kp) + gm2 / (2.0 * km))


def tau_d_sharo_1d_code(V_STEP, Delta_sharo):
    """
    Frame-mixed 1D value used previously (Sharoglazova-convention Delta in
    the single-step formula). Printed for reference only -- NOT plotted.
    """
    if V_STEP <= 0 or Delta_sharo >= 0 or abs(Delta_sharo) >= V_STEP:
        return None
    return (np.sqrt(V_STEP - abs(Delta_sharo))
            / (V_STEP * np.sqrt(abs(Delta_sharo))))

# ============================================================================
# DATA EXTRACTION (raw values; sigma_k fix; unit resolution afterwards)
# ============================================================================

def extract_dwell_data(runs):
    """
    Extract sigma_k and RAW (as-stored) dwell times from runs.

    sigma_k is recomputed from sigx as 1/(2*sigx): the initial packet is
    psi ~ exp(-(x-x0)^2/(4*sigx^2)), so sigx is the std of |psi|^2 and
    1/(2*sigx) is the std of |psi_hat(k)|^2 -- the paper's sigma_k.
    """
    data = {
        'file': [],
        'sigma_k': [],            # um^-1 (TRUE std of |psi_hat|^2)
        'sigma_k_code': [],       # code units
        'dwell_entered_raw': [],  # as stored
        'dwell_entered_se_raw': [],
        'dwell_all_raw': [],      # as stored
        'dwell_all_se_raw': [],
        'kx0': [],                # code units
        'V_STEP': [],             # code units
        'J0_code': [],            # code units (recovered)
        'Delta_paper_meV': [],    # paper convention
        'Delta_sharo_code': [],   # stored convention (code units)
        'Delta_sharo_meV': [],
        'L0': [],                 # m
        'T0': [],                 # s
    }

    for run in runs:
        conv = run['_conv']
        ts = run.get('tunneling_speed', {})

        sigx_code = run.get('sigx', None)
        if sigx_code is None or sigx_code <= 0:
            continue

        # CORRECT formula: sigma_k = 1/(2*sigx)  [std of |psi_hat(k)|^2]
        sigma_k_code = 1.0 / (2.0 * sigx_code)

        dwell_entered = ts.get('dwell_time_entered_mean', None)
        if dwell_entered is None:
            continue

        L0_um = conv['L0'] * 1e6

        data['file'].append(run.get('_file', '?'))
        data['sigma_k'].append(sigma_k_code / L0_um)   # um^-1
        data['sigma_k_code'].append(sigma_k_code)
        data['dwell_entered_raw'].append(dwell_entered)
        data['dwell_entered_se_raw'].append(
            ts.get('dwell_time_entered_se', 0) or 0)
        data['dwell_all_raw'].append(ts.get('dwell_time_all_mean', 0) or 0)
        data['dwell_all_se_raw'].append(ts.get('dwell_time_all_se', 0) or 0)
        data['kx0'].append(conv['kx0'])
        data['V_STEP'].append(conv['V_STEP'])
        data['J0_code'].append(conv['J0_code'])
        data['Delta_paper_meV'].append(conv['Delta_paper_meV'])
        data['Delta_sharo_code'].append(conv['Delta_sharo_code'])
        data['Delta_sharo_meV'].append(conv['Delta_sharo_meV'])
        data['L0'].append(conv['L0'])
        data['T0'].append(conv['T0'])

    for key in data:
        data[key] = np.array(data[key])

    if len(data['sigma_k']) > 0:
        sort_idx = np.argsort(data['sigma_k'])
        for key in data:
            data[key] = data[key][sort_idx]

    return data


def resolve_time_units(raw_values, expected_code, T0_ps, label, mode):
    """
    Decide whether raw_values are in code units or already in ps, and
    return (factor_to_ps, chosen_mode).

    expected_code: array (or scalar) of theoretically expected values in
    CODE units, used only as a magnitude reference. The two hypotheses
    differ by a factor 1/T0_ps (~3.8 for T0 ~ 0.26 ps), so a coarse
    log-distance comparison is robust even if theory is off by 10-20%.
    """
    if mode == 'code':
        return T0_ps, 'code'
    if mode == 'ps':
        return 1.0, 'ps'

    raw = np.asarray(raw_values, dtype=float)
    exp_code = np.asarray(expected_code, dtype=float) * np.ones_like(raw)
    valid = (raw > 0) & (exp_code > 0)
    if not np.any(valid):
        print(f"[UNITS] {label}: no positive values -- assuming code units.")
        return T0_ps, 'code (fallback)'

    # log-distance of the median ratio from each hypothesis
    ratio = np.median(raw[valid] / exp_code[valid])
    d_code = abs(np.log(ratio))             # raw ~ expected_code
    d_ps = abs(np.log(ratio / T0_ps))       # raw ~ expected_code * T0_ps

    if d_code <= d_ps:
        chosen, factor = 'code', T0_ps
    else:
        chosen, factor = 'ps', 1.0

    print(f"[UNITS] {label}: median(raw/expected_code) = {ratio:.3f} "
          f"-> interpreted as {chosen.upper()} units "
          f"(distinguishing factor 1/T0_ps = {1.0/T0_ps:.2f})")
    if min(d_code, d_ps) > np.log(2.0):
        print(f"[UNITS] WARNING: {label} is more than a factor 2 from BOTH "
              f"hypotheses -- check the stored values and theory inputs!")
    return factor, chosen


def fit_inverse_model(sigma_k, dwell, dwell_se):
    """Fit tau = C/sigma_k model. Returns C, C_err."""
    def inverse_model(sk, C):
        return C / sk

    if np.all(dwell_se > 0):
        popt, pcov = curve_fit(inverse_model, sigma_k, dwell,
                               sigma=dwell_se, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(inverse_model, sigma_k, dwell)

    return popt[0], np.sqrt(pcov[0, 0])

# ============================================================================
# PLOTTING
# ============================================================================

def plot_dwell_time(data, output_path):
    """Create publication-quality 2-panel dwell time plot."""

    if len(data['sigma_k']) < 3:
        print("[ERROR] Need at least 3 data points for plotting")
        return

    sigma_k = data['sigma_k']

    # Common parameters (single-detuning batch; verified below)
    k0 = data['kx0'][0]
    L0 = data['L0'][0]
    T0 = data['T0'][0]
    V_STEP = data['V_STEP'][0]
    J0_code = data['J0_code'][0]
    Delta_paper_meV = data['Delta_paper_meV'][0]
    Delta_sharo_meV = data['Delta_sharo_meV'][0]

    # Cross-run consistency of recovered J0 and kx0
    J0_spread = data['J0_code'].std() / max(data['J0_code'].mean(), 1e-30)
    k0_spread = data['kx0'].std() / max(data['kx0'].mean(), 1e-30)
    if J0_spread > 1e-6 or k0_spread > 1e-6:
        print(f"[WARNING] Parameter spread across runs: "
              f"J0 rel. std = {J0_spread:.2e}, kx0 rel. std = {k0_spread:.2e}")

    L0_um = L0 * 1e6
    T0_ps = T0 * 1e12
    hbar_J0_ueV = J0_code * (HBAR_SI**2 / (M_SI * L0**2)) / MEV_TO_J * 1e3

    print(f"\n[PARAMETERS]  (paper convention)")
    if Delta_sharo_meV is not None and np.isfinite(np.asarray(Delta_sharo_meV, dtype=float)):
        print(f"  Delta_paper = {Delta_paper_meV:.4f} meV "
              f"(Sharoglazova Delta + hbar*J0 = {float(Delta_sharo_meV):.4f} meV)")
    else:
        print(f"  Delta_paper = {Delta_paper_meV:.4f} meV "
              f"(stored directly in paper convention)")
    print(f"  hbar*J0 (recovered) = {hbar_J0_ueV:.2f} ueV "
          f"({J0_code:.6f} code)")
    print(f"  kx0 = {k0:.6f} code = {k0 / L0_um:.4f} um^-1")
    print(f"  V_STEP = {V_STEP:.6f} code")
    print(f"  L0 = {L0_um:.2f} um, T0 = {T0_ps:.4f} ps")
    print(f"  sigma_k range: [{sigma_k.min():.4f}, {sigma_k.max():.4f}] um^-1")

    # ========================================================================
    # RESOLVE TIME UNITS OF STORED DWELL VALUES (separately per quantity)
    # ========================================================================

    tau_ent_exp_code = tau_entered_theory_code(k0, data['sigma_k_code'])
    tau_d_code = tau_d_2d_code(k0, V_STEP, J0_code)

    f_ent, mode_ent = resolve_time_units(
        data['dwell_entered_raw'], tau_ent_exp_code, T0_ps,
        "entered dwell times", DWELL_ENTERED_UNITS)
    f_all, mode_all = resolve_time_units(
        data['dwell_all_raw'],
        tau_d_code if tau_d_code is not None else tau_ent_exp_code,
        T0_ps, "all-trajectory dwell times", DWELL_ALL_UNITS)

    dwell_entered = data['dwell_entered_raw'] * f_ent
    dwell_entered_se = data['dwell_entered_se_raw'] * f_ent
    dwell_all = data['dwell_all_raw'] * f_all
    dwell_all_se = data['dwell_all_se_raw'] * f_all

    if DEBUG_PRINT_RAW:
        print(f"\n[DEBUG] Per-run raw -> converted dwell times "
              f"(entered: {mode_ent}, all: {mode_all}):")
        print(f"  {'file':<28} {'sigma_k':>9} "
              f"{'ent_raw':>10} {'ent_ps':>8} {'all_raw':>10} {'all_ps':>8}")
        for i in range(len(sigma_k)):
            print(f"  {str(data['file'][i]):<28} {sigma_k[i]:>9.4f} "
                  f"{data['dwell_entered_raw'][i]:>10.4f} "
                  f"{dwell_entered[i]:>8.3f} "
                  f"{data['dwell_all_raw'][i]:>10.4f} "
                  f"{dwell_all[i]:>8.3f}")

    # ========================================================================
    # FIT TO INVERSE MODEL
    # ========================================================================

    C_fit, C_fit_err = fit_inverse_model(sigma_k, dwell_entered,
                                         dwell_entered_se)
    C_theory = np.sqrt(np.pi / 2) * T0_ps / (k0 * L0_um)

    print(f"\n[FIT] tau = C/sigma_k")
    print(f"  C_fit    = {C_fit:.4f} +/- {C_fit_err:.4f} ps*um^-1")
    print(f"  C_theory = sqrt(pi/2)*T0/(k0*L0_um) = {C_theory:.4f} ps*um^-1")
    print(f"  Agreement: C_fit/C_theory = {C_fit / C_theory:.4f}")
    print(f"  Relative difference: "
          f"{100 * abs(C_fit - C_theory) / C_theory:.1f}%")

    # ========================================================================
    # THEORY CURVES (in ps)
    # ========================================================================

    sigma_k_smooth = np.linspace(sigma_k.min() * 0.9,
                                 sigma_k.max() * 1.05, 200)
    tau_theory_smooth = (tau_entered_theory_code(k0, sigma_k_smooth * L0_um)
                         * T0_ps)
    tau_fit_smooth = C_fit / sigma_k_smooth

    tau_d = tau_d_code * T0_ps if tau_d_code is not None else None
    _ds0 = data['Delta_sharo_code'][0]
    tau_d_old_code = (tau_d_sharo_1d_code(V_STEP, float(_ds0))
                      if (_ds0 is not None and np.isfinite(np.asarray(_ds0, dtype=float)))
                      else None)
    tau_d_old = tau_d_old_code * T0_ps if tau_d_old_code is not None else None

    mean_dwell_all = np.mean(dwell_all)

    print(f"\n[REFERENCES]")
    if tau_d is not None:
        print(f"  tau_d^2D(k0)  = {tau_d:.4f} ps   "
              f"<-- plotted (paper-consistent)")
    else:
        print(f"  tau_d^2D: not computed (not fully evanescent)")
    if tau_d_old is not None:
        print(f"  tau_d (old, frame-mixed 1D) = {tau_d_old:.4f} ps  "
              f"[reference only, NOT plotted]")
    print(f"  Mean of all trajectories: {mean_dwell_all:.4f} ps")
    if tau_d is not None:
        print(f"  Mean / tau_d^2D = {mean_dwell_all / tau_d:.4f}")

    # ========================================================================
    # CREATE FIGURE
    # ========================================================================

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=FIGSIZE,
                                            sharex=True,
                                            gridspec_kw={'hspace': 0.15})

    # ========================================================================
    # TOP PANEL: Entered trajectories
    # ========================================================================

    ax_top.errorbar(sigma_k, dwell_entered, yerr=dwell_entered_se,
                    fmt='o', color=COLORS['entered_data'],
                    markersize=8, capsize=3, lw=1.5, zorder=5)

    ax_top.scatter(sigma_k, dwell_all,
                   marker='s', s=50, color=COLORS['all_data'],
                   alpha=0.8, zorder=4)

    ax_top.plot(sigma_k_smooth, tau_fit_smooth,
                color=COLORS['fit_line'], lw=2.5, ls='-', alpha=0.8,
                zorder=3)

    ax_top.plot(sigma_k_smooth, tau_theory_smooth,
                color=COLORS['theory_line'], lw=2.5, ls='--', alpha=0.8,
                zorder=2)

    if tau_d is not None:
        ax_top.axhline(tau_d, color=COLORS['tau_d'],
                       ls='--', lw=2.0, alpha=0.7, zorder=2)

    ax_top.set_ylabel("Dwell time (ps)", fontsize=12)
    ax_top.set_title(r"(a) Dwell Time vs $\sigma_k$",
                     fontsize=13, fontweight='bold')
    ax_top.grid(alpha=0.25, ls='--')
    ax_top.tick_params(labelbottom=False)

    legend_handles_top = [
        Line2D([0], [0], color=COLORS['entered_data'], marker='o', ms=8,
               ls='', label='Entered trajectories'),
        Line2D([0], [0], color=COLORS['all_data'], marker='s', ms=8,
               ls='', label='All trajectories'),
        Line2D([0], [0], color=COLORS['fit_line'], lw=2.5, ls='-',
               label=f"Fit: {C_fit:.3f}/σ$_k$ ps·μm$^{{-1}}$"),
        Line2D([0], [0], color=COLORS['theory_line'], lw=2.5, ls='--',
               label=f"Theory: {C_theory:.3f}/σ$_k$ ps·μm$^{{-1}}$"),
    ]
    if tau_d is not None:
        legend_handles_top.append(
            Line2D([0], [0], color=COLORS['tau_d'], ls='--', lw=2.0,
                   label=f'τ$_d$(k$_0$) [2D]: {tau_d:.2f} ps')
        )

    ax_top.legend(handles=legend_handles_top, fontsize=10, loc='upper right',
                  framealpha=0.95)

    # ========================================================================
    # BOTTOM PANEL: All trajectories (zoomed)
    # ========================================================================

    ax_bottom.errorbar(sigma_k, dwell_all, yerr=dwell_all_se,
                       fmt='s', color=COLORS['all_data'],
                       markersize=8, capsize=3, lw=1.5, zorder=5)

    ax_bottom.axhline(mean_dwell_all, color=COLORS['mean_line'],
                      ls='-', lw=2.5, alpha=0.8, zorder=3)

    if tau_d is not None:
        ax_bottom.axhline(tau_d, color=COLORS['tau_d'],
                          ls='--', lw=2.0, alpha=0.7, zorder=2)

    ax_bottom.set_xlabel(r"$\sigma_k$ (μm$^{-1}$)", fontsize=12)
    ax_bottom.set_ylabel("Dwell time (ps)", fontsize=12)
    ax_bottom.set_title(r"(b) Dwell Time: All Trajectories (Zoomed View)",
                        fontsize=13, fontweight='bold')
    ax_bottom.grid(alpha=0.25, ls='--')

    legend_handles_bottom = [
        Line2D([0], [0], color=COLORS['all_data'], marker='s', ms=8,
               ls='', label='All trajectories'),
        Line2D([0], [0], color=COLORS['mean_line'], ls='-', lw=2.5,
               label=f'Mean: {mean_dwell_all:.2f} ps'),
    ]
    if tau_d is not None:
        legend_handles_bottom.append(
            Line2D([0], [0], color=COLORS['tau_d'], ls='--', lw=2.0,
                   label=f'τ$_d$(k$_0$) [2D]: {tau_d:.2f} ps')
        )

    ax_bottom.legend(handles=legend_handles_bottom, fontsize=10, loc='best',
                     framealpha=0.95)

    # Zoom y-axis on bottom panel: include both the data and tau_d
    y_vals = list(dwell_all)
    if tau_d is not None:
        y_vals.append(tau_d)
    y_vals = np.array(y_vals)
    y_range = y_vals.max() - y_vals.min()
    y_pad = 0.3 * y_range if y_range > 0 else 0.1
    ax_bottom.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    # ========================================================================
    # SAVE
    # ========================================================================

    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"\n[SAVED] {pdf_path}")

    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {png_path}")

    plt.show()

    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    if len(sys.argv) > 1:
        batch_folder = sys.argv[1]
    else:
        batch_folder = pick_batch_folder()
        if not batch_folder:
            batch_folder = "batch_out"
            print(f"[INFO] Using default: {batch_folder}")

    if not Path(batch_folder).is_dir():
        print(f"[ERROR] Folder not found: {batch_folder}")
        return

    runs = load_runs(batch_folder)
    if len(runs) == 0:
        return

    print(f"\n[FILTERING] Target detuning (PAPER convention): "
          f"{TARGET_DETUNING_PAPER_MEV} meV +/- {DETUNING_TOL_MEV} meV, "
          f"regime: {REGIME_FILTER}")
    filtered_runs = filter_runs(runs, TARGET_DETUNING_PAPER_MEV, REGIME_FILTER)
    print(f"[FILTERING] {len(filtered_runs)} of {len(runs)} runs match criteria")

    if len(filtered_runs) == 0:
        print("[ERROR] No runs match filter criteria!")
        # Report what IS present so the mismatch is obvious (target vs. available).
        print("\n  Runs found in this folder (derived, PAPER convention):")
        print(f"  {'file':<34}{'Delta_paper (meV)':>18}{'regime':>14}")
        present = []
        for run in runs:
            conv = derive_conventions(run)
            fname = run.get('_file', '?')
            if conv is None:
                print(f"  {fname:<34}{'(unreadable)':>18}{'-':>14}")
                continue
            kp2 = 2.0*(conv['V_STEP']-conv['J0_code']) - conv['kx0']**2
            km2 = 2.0*(conv['V_STEP']+conv['J0_code']) - conv['kx0']**2
            reg = 'evanescent' if (kp2>0 and km2>0) else (
                  'propagative' if (kp2<0 and km2<0) else 'mixed')
            present.append((conv['Delta_paper_meV'], reg))
            print(f"  {fname:<34}{conv['Delta_paper_meV']:>18.4f}{reg:>14}")
        if present:
            dets = sorted(set(round(d, 4) for d, _ in present))
            print(f"\n  Target was {TARGET_DETUNING_PAPER_MEV} +/- {DETUNING_TOL_MEV} meV"
                  f", regime '{REGIME_FILTER}'.")
            print(f"  Detunings present: {dets} meV.")
            if len(dets) == 1:
                print(f"  -> Set TARGET_DETUNING_PAPER_MEV = {dets[0]} (or None to include all).")
            else:
                print("  -> Adjust TARGET_DETUNING_PAPER_MEV to one of the above, "
                      "or set it to None.")
        return

    data = extract_dwell_data(filtered_runs)
    print(f"\n[DATA] Extracted {len(data['sigma_k'])} valid data points")
    print(f"[DATA] sigma_k computed from sigx as 1/(2*sigx) [paper convention]")

    if len(data['sigma_k']) < 3:
        print("[ERROR] Need at least 3 data points")
        return

    output_path = Path(batch_folder) / OUTPUT_FILENAME
    plot_dwell_time(data, output_path)


if __name__ == "__main__":
    main()
