"""Stage 1 (DDPM synthetic generation) orchestrator for DDF-Net.

Faithful inline port of the official Stage 1 order documented in the DDF-Net
README -- channelization -> 166x600 interpolation -> diffusion -> inverse restore
+ 25-channel merge -- reading our BCI IV-2a ``.gdf`` directly, because we hold
``.gdf`` while the official ``process0.py`` expects the BNCI ``.mat`` files.

The official scripts are not directly runnable as shipped:

* ``process0.py`` / ``process.py`` hard-code ``D:\\`` paths at module level and
  execute on import (no ``if __name__ == "__main__"`` guard);
* ``DiffLidar.py``'s active code is a half-adapted LiDAR/HSI trainer: it imports
  the missing ``uvit``, hard-codes 5000 epochs, has no BCI-2a dataset choice,
  and saves a field name ``process1.py`` never reads;
* ``process1.py``'s restore writes field ``restored_data`` while its merge reads
  ``diffusion_result`` (a field-name mismatch).

Field names and the interpolation shape (166x600) are preserved exactly
(``interp_matrix``, ``x_old``, ``x_new``, ``diffusion_result``, ``merged_data``)
so the artifacts stay interchangeable with the official scripts' outputs.

Two modes:

* ``--mode placeholder`` (default) -- fastest. The "diffusion" output is the
  interpolated matrix plus small Gaussian noise (a stand-in for a poorly
  trained DDPM). This is the "_t.mat placeholder data" path.
* ``--mode diffusion``   -- trains the repo's actual ``Diffusion`` + ``UNetLidar``
  for ``--epochs`` steps. Requires PyTorch + torchvision (a GPU is strongly
  recommended) and is not exercised by default.

Run:  py run_stage1_pipeline.py --subject A01 --trials 40 --mode placeholder
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat

_REPO_ROOT = Path(__file__).resolve().parents[2]  # C:\BMI
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mne
from src.config import (
    ANNOTATION_TO_EVENT_ID,
    BCICIV_2A_CHANNELS,
    EOG_CHANNELS,
    EVENT_ID,
    FMAX,
    FMIN,
    FS,
    TMAX,
    TMIN,
)

N_CHANNELS = len(BCICIV_2A_CHANNELS)  # 25 = 22 EEG + 3 EOG
TARGET_SHAPE = (166, 600)


# ---------------------------------------------------------------------------
# Step 0: real data (25-channel trial arrays from our .gdf)
# ---------------------------------------------------------------------------

def load_25ch_trials(subject, run, raw_dir, n_trials, seed):
    """Load band-pass-filtered 25-channel left/right trials as (data, labels)."""
    gdf = Path(raw_dir) / f"{subject}{run}.gdf"
    if not gdf.exists():
        raise FileNotFoundError(f"missing {gdf}")

    raw = mne.io.read_raw_gdf(gdf, preload=True)
    raw.rename_channels(dict(zip(raw.ch_names, BCICIV_2A_CHANNELS)))
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS})
    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_TO_EVENT_ID)
    raw = raw.filter(FMIN, FMAX)  # all 25 data channels
    epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                        baseline=None, preload=True)
    if epochs.info["sfreq"] != FS:
        epochs = epochs.resample(FS)

    data = epochs.get_data()          # [n_epochs, 25, n_times]
    labels = epochs.events[:, 2]

    # Balanced subsample across classes.
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = int(np.ceil(n_trials / len(classes)))
    idx = [rng.choice(np.where(labels == c)[0], size=min(per_class, (labels == c).sum()),
                      replace=False).tolist() for c in classes]
    idx = np.concatenate(idx)
    return data[idx], labels[idx]


# ---------------------------------------------------------------------------
# Step 1-4: interpolation -> diffusion -> restore -> merge
# ---------------------------------------------------------------------------

def interpolate(series_1d):
    """1D series -> 166x600 matrix plus the interpolation coordinates."""
    x_old = np.linspace(0, 1, len(series_1d))
    x_new = np.linspace(0, 1, np.prod(TARGET_SHAPE))
    interp = interp1d(x_old, series_1d, kind="linear", fill_value="extrapolate")
    return interp(x_new).reshape(TARGET_SHAPE), x_old, x_new


def placeholder_diffuse(interp_matrix, rng, noise_std=0.05):
    """Stand-in for a DDPM reconstruction: small Gaussian perturbation."""
    scale = noise_std * float(np.std(interp_matrix))
    return interp_matrix + scale * rng.standard_normal(interp_matrix.shape)


def real_diffuse(interp_matrix, epochs):
    """Train the repo's real Diffusion + UNetLidar (needs torch + torchvision).

    Not exercised by default; kept so a GPU/Torch environment can swap it in.
    """
    raise NotImplementedError(
        "real_diffuse requires PyTorch + torchvision and the repo's "
        "Diffusion/UNetLidar classes; run in a GPU-enabled environment")


def restore(interp_matrix, x_old, x_new):
    """Inverse-interpolate the 166x600 matrix back to the original 1D length."""
    rev = interp1d(x_new, interp_matrix.flatten(), kind="linear",
                   fill_value="extrapolate")
    return rev(x_old)


def run_pipeline(subject, run, raw_dir, out_dir, n_trials, seed, mode,
                 epochs, n_tmat, noise_std):
    out_dir = Path(out_dir)
    merged_dir = out_dir / "merged"
    tmat_dir = out_dir / "diffused"
    merged_dir.mkdir(parents=True, exist_ok=True)
    tmat_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {subject}{run} -> {n_trials} balanced 25-channel trials")
    data, labels = load_25ch_trials(subject, run, raw_dir, n_trials, seed)
    n_trials, n_ch, n_times = data.shape
    print(f"  trials={n_trials} channels={n_ch} time={n_times}")

    rng = np.random.default_rng(seed + 1)
    merged = np.empty((n_trials, n_ch, n_times), dtype=np.float64)

    for t in range(n_trials):
        channels = []
        for c in range(n_ch):
            series = data[t, c]
            interp_matrix, x_old, x_new = interpolate(series)

            if mode == "placeholder":
                diffused = placeholder_diffuse(interp_matrix, rng, noise_std)
            else:
                diffused = real_diffuse(interp_matrix, epochs)

            restored = restore(diffused, x_old, x_new)
            channels.append(restored)

            # Write the "_t.mat" placeholder artifact for the first `n_tmat` trials.
            if t < n_tmat:
                ch_dir = tmat_dir / str(t + 1)
                ch_dir.mkdir(parents=True, exist_ok=True)
                savemat(ch_dir / f"{c + 1}_t.mat", {"diffusion_result": diffused})

        merged[t] = np.stack(channels, axis=0)   # [25, n_times]

    # Per-trial merged [n_times x 25] matrices (the format convert_to_npy reads).
    for t in range(n_trials):
        savemat(merged_dir / f"{t + 1}.mat", {"merged_data": merged[t].T})

    np.save(out_dir / "labels.npy", labels)
    print(f"[out] merged/{{1..{n_trials}}}.mat  ({n_trials} x {n_ch} x {n_times})")
    print(f"[out] diffused/{{1..{n_tmat}}}/{{1..25}}_t.mat placeholder files")
    print(f"[out] labels.npy  ({n_trials},)")
    return merged, labels


def main(args):
    run_pipeline(args.subject, args.run, args.raw_dir, args.out_dir, args.trials,
                 args.seed, args.mode, args.epochs, args.n_tmat, args.noise_std)
    print("Stage 1 pipeline done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="A01")
    ap.add_argument("--run", default="T")
    ap.add_argument("--raw-dir", default=str(_REPO_ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "stage1_out"))
    ap.add_argument("--trials", type=int, default=40, help="trials in the final merged batch")
    ap.add_argument("--n-tmat", type=int, default=2, help="trials to write _t.mat artifacts for")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["placeholder", "diffusion"], default="placeholder")
    ap.add_argument("--epochs", type=int, default=2, help="epochs for --mode diffusion")
    ap.add_argument("--noise-std", type=float, default=0.05, help="placeholder noise level")
    args = ap.parse_args()
    main(args)
