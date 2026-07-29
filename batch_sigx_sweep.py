# batch_sigx_sweep.py -- drives the Fig. 7 sigma_k sweep across several GPUs.
#
# One subprocess per GPU, launched concurrently via a ThreadPoolExecutor whose
# worker count == number of GPUs. The threads only supervise the subprocesses;
# the actual compute runs in those separate simulation processes, each pinned to
# its own card with --device.
#
# Each run writes one VEL_<run_tag>.json into OUTDIR; plot the collection with
#     python plot_dwell_time_from_batch.py batch_out
# Keep DETUNING_MEV below in sync with TARGET_DETUNING_PAPER_MEV in that script,
# or it will filter out every run.
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# The simulation this driver invokes. NOTE: earlier revisions of this file named
# it "DWfinal.py"; the batch-capable script is now DWtunneling.py, and the file
# currently called DWfinal.py is an older variant with no CLI, no dwell-time
# analysis and no JSON output -- pointing SIM at it yields zero artifacts.
SIM = "DWtunneling.py"
OUTDIR = "batch_out"
LOGDIR = "batch_logs"

# Fig. 7 sweep: fixed PAPER-convention detuning, vary sigma_x.
#   sigma_k = 1/(2*sigx)   (larger sigx -> smaller sigma_k)
# sigx 60..140 -> sigma_k 0.0042..0.0018 um^-1 at L0 = 2 um/code unit.
DETUNING_MEV = -0.09       # paper convention: Delta = E_k - V0
sigx_values = [60, 70, 80, 90, 100, 110, 120, 130, 140]

# Everything else fixed; sigx is passed separately in run_one(). Take a sigx
# argument anyway so per-case overrides can be added without changing callers.
def per_case_args(sigx):
    return dict(DETUNING_MEV=DETUNING_MEV,
                V_STEP=0.216,
                x_min=-1100.0, x_max=100.0,   # evanescent: absorber at x=25 is
                                              # far beyond the decay length
                n_steps=130000, n_traj=300000)

# GPUs to use, one concurrent run each. Override without editing the file, e.g.
#     DW_GPU_IDS=0,1,2,3 python batch_sigx_sweep.py
gpu_ids = [int(g) for g in os.environ.get("DW_GPU_IDS", "0,1").split(",") if g.strip()]

def run_one(sigx, gpu_id):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    # sigx alone identifies the run; keeping the GPU id out of the tag means the
    # output filenames do not depend on how work happened to be scheduled.
    run_tag = f"sigx={sigx:.3f}"

    cmd = [
        sys.executable, SIM,
        f"--sigx={sigx}",
        f"--outdir={OUTDIR}",
        f"--run_tag={run_tag}",
        "--traj",                 # trajectories needed for dwell statistics
        "--no_anim", "--no_video",
        "--no_save_npz",          # JSON summary is enough for the batch plot
        "--device", str(gpu_id),
    ]
    for k, v in per_case_args(sigx).items():
        cmd.append(f"--{k}={v}")

    log_path = os.path.join(LOGDIR, f"{run_tag}.log")
    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, "w", buffering=1) as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        rc = subprocess.call(cmd, stdout=lf, stderr=lf)
    dur = time.time() - t0

    # The simulation writes exactly VEL_<run_tag>.json into OUTDIR.
    out_json = os.path.join(OUTDIR, f"VEL_{run_tag}.json")
    artifacts = [out_json] if os.path.exists(out_json) else []
    return dict(sigx=sigx, gpu=gpu_id, rc=rc, dur=dur,
                log=log_path, artifacts=artifacts)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # Fail loudly and immediately rather than after N silent subprocess errors.
    if not os.path.exists(SIM):
        print(f"[ERROR] Simulation script not found: {SIM}\n"
              f"        (looked in {here})", file=sys.stderr)
        sys.exit(2)
    if not gpu_ids:
        print("[ERROR] No GPUs selected; set DW_GPU_IDS, e.g. DW_GPU_IDS=0,1",
              file=sys.stderr)
        sys.exit(2)
    print(f"[SETUP] {SIM} -> {OUTDIR}/  |  {len(sigx_values)} runs on GPUs {gpu_ids}",
          flush=True)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = []
        for i, sigx in enumerate(sigx_values):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futs.append(ex.submit(run_one, sigx, gpu_id))
        for f in as_completed(futs):
            r = f.result()
            ok = "OK" if (r["rc"] == 0 and r["artifacts"]) else "FAIL"
            print(f"[{ok}] sigx={r['sigx']:.3f} GPU{r['gpu']} rc={r['rc']} "
                  f"in {r['dur']/60:.1f} min  artifacts={len(r['artifacts'])}  "
                  f"log={r['log']}", flush=True)

    all_json = sorted(glob.glob(os.path.join(OUTDIR, "VEL_*.json")))
    print(f"\n[SUMMARY] Total VEL artifacts: {len(all_json)}")
    if not all_json:
        print("[ERROR] No VEL_*.json artifacts found. Check logs in", LOGDIR)
        sys.exit(2)
    print(f"[NEXT] python plot_dwell_time_from_batch.py {OUTDIR}")
    print(f"[NEXT] ensure TARGET_DETUNING_PAPER_MEV = {DETUNING_MEV} in that script")

if __name__ == "__main__":
    main()
