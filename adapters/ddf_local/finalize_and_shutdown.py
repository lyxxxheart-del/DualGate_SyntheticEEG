"""Wait for UNetLidar training, then convert + validate + save + shutdown.

Launch once in the background after ``train_unetlidar.py`` starts. This script
survives independently of the agent session: it polls the training log for
completion, then

  1. convert_to_npy.py         -> ``[144, 22, n_times]`` data.npy + labels.npy
  2. validate_ddf_synthetic.py -> Baseline / Unfiltered / Cov / Dual table
                                  (saved under ``C:\\BMI\\reports\\``)
  3. shutdown /s               -> power off the machine

If training errors out (``Traceback`` in the log) the script exits WITHOUT
shutting down, so the failure stays visible. Pass ``--no-shutdown`` for a dry run
of the chain, or ``--shutdown-delay N`` to change the power-off grace period.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_DDF = Path(__file__).resolve().parent            # C:\BMI\adapters\ddf_local
_REPO = _DDF.parents[1]                            # C:\BMI
_ADAPTERS = _REPO / "adapters"                     # C:\BMI\adapters

LOG = _DDF / "train_unetlidar.log"
OUT = _DDF / "stage1_out"
_DONE_MARKER = "Done. Next:"
_FAIL_MARKER = "Traceback"


def log(msg):
    print(msg, flush=True)


def wait_for_training():
    """Poll the training log until the success or failure marker appears."""
    log(f"[finalize] waiting for training completion in {LOG}")
    while True:
        txt = LOG.read_text(errors="ignore") if LOG.exists() else ""
        if _DONE_MARKER in txt:
            log(f"[finalize] training finished (saw '{_DONE_MARKER}')")
            return True
        if _FAIL_MARKER in txt:
            log(f"[finalize] training FAILED (saw '{_FAIL_MARKER}'); "
                "NOT shutting down so the error stays visible")
            return False
        time.sleep(30)


def run(cmd):
    log(f"[finalize] $ {' '.join(map(str, cmd))}")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.stdout:
        log(r.stdout.rstrip())
    if r.returncode != 0:
        log(f"[finalize] WARN: exit {r.returncode}")
        if r.stderr:
            log(r.stderr.rstrip())
    return r.returncode


def shutdown(delay):
    exe = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "shutdown.exe"
    cmd = [str(exe), "/s", "/t", str(delay)]
    log(f"[finalize] powering off: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[finalize] shutdown command failed ({r.returncode}): {r.stderr.strip()}")
    return r.returncode


def main(args):
    if not wait_for_training():
        return 1

    py = sys.executable
    run([py, str(_DDF / "convert_to_npy.py"),
         "--merged-dir", str(OUT / "merged"),
         "--labels", str(OUT / "labels.npy"),
         "--out", str(OUT / "ddf_synthetic")])

    run([py, str(_ADAPTERS / "validate_ddf_synthetic.py"),
         "--syn", str(OUT / "ddf_synthetic" / "data.npy"),
         "--syn-labels", str(OUT / "ddf_synthetic" / "labels.npy"),
         "--subject", args.subject, "--normalize", args.normalize])

    if args.no_shutdown:
        log("[finalize] --no-shutdown set: skipping power-off")
        return 0

    log(f"[finalize] results saved; shutting down in {args.shutdown_delay}s")
    return shutdown(args.shutdown_delay)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="A01")
    ap.add_argument("--normalize", default="trace")
    ap.add_argument("--shutdown-delay", type=int, default=60,
                    help="grace period in seconds before power-off")
    ap.add_argument("--no-shutdown", action="store_true",
                    help="run the convert+validate chain but skip power-off")
    args = ap.parse_args()
    sys.exit(main(args))
