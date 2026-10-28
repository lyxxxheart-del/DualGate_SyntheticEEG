"""Convert Stage 1 merged output to the adapter's ``[n_epochs, 22, n_times]``.

Reads the per-trial merged matrices produced by ``run_stage1_pipeline.py`` (field
``merged_data``, shape ``[n_times x 25]`` -- 22 EEG + 3 EOG in BCI IV-2a file
order), stacks them into ``[n_trials, 25, n_times]``, drops the 3 EOG channels to
keep the 22 EEG channels the gate operates on, and saves ``data.npy`` +
``labels.npy`` for ``adapters/load_ddf_net_synthetic.py``.

Run:  py convert_to_npy.py --merged-dir stage1_out/merged \
          --labels stage1_out/labels.npy --out stage1_out/ddf_synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_REPO_ROOT = Path(__file__).resolve().parents[2]  # C:\BMI
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.config import EEG_CHANNELS  # 22 EEG channels in file order

N_EEG = len(EEG_CHANNELS)  # 22
N_TOTAL = N_EEG + 3          # 25 (22 EEG + 3 EOG)


def load_merged(merged_dir):
    """Return ``(data [T,25,N], order)`` stacked from ``{1..K}.mat`` files."""
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob("*.mat"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not files:
        raise FileNotFoundError(f"no merged .mat files under {merged_dir}")

    trials = []
    for f in files:
        arr = loadmat(f)["merged_data"]          # [N x 25]
        if arr.ndim != 2 or arr.shape[1] != N_TOTAL:
            raise ValueError(f"{f.name}: expected [N x {N_TOTAL}], got {arr.shape}")
        trials.append(arr.T)                     # [25 x N]
    n_times = {t.shape[1] for t in trials}
    if len(n_times) != 1:
        raise ValueError(f"inconsistent trial lengths: {n_times}")
    return np.stack(trials, axis=0), [int(f.stem) for f in files]


def main(args):
    data, order = load_merged(args.merged_dir)
    eeg = data[:, :N_EEG, :]                     # drop the 3 EOG channels
    labels = np.load(args.labels).ravel() if args.labels else None

    if labels is not None and len(labels) != len(eeg):
        raise ValueError(f"labels ({len(labels)}) != trials ({len(eeg)})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "data.npy", eeg)
    if labels is not None:
        np.save(out / "labels.npy", labels)

    print(f"[convert] {len(eeg)} trials -> {eeg.shape} (22 EEG channels)")
    print(f"[convert] saved {out / 'data.npy'}"
          + (f" and {out / 'labels.npy'}" if labels is not None else ""))
    if labels is not None:
        uniq, counts = np.unique(labels, return_counts=True)
        print(f"[convert] label distribution: "
              + ", ".join(f"class {u}: {c}" for u, c in zip(uniq, counts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
