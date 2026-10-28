"""Adapter: load DDF-Net generated MI-EEG samples and screen them with the gate.

DDF-Net ("Diffusion enhancement-based dual-domain fusion attention motor
imagery electroencephalography classification network", *Engineering
Applications of Artificial Intelligence*, 2026) pre-trains a DDPM on MI-EEG and
uses it to augment the training set. Code and pretrained weights:
https://github.com/HyperSystemAndImageProc/DDF-Net

DDF-Net's on-disk output format is not pinned by this project, so the loader is
shape-tolerant: it accepts ``[n_epochs, 22, n_times]``,
``[n_epochs, n_times, 22]``, or a flat ``[n_epochs, 22 * n_times]`` array and
normalizes every input to ``[n_epochs, 22, n_times]`` in BCI IV-2a channel
order.

Two entry points are provided:

* :func:`load_ddf_net_synthetic` -- load one array (plus optional labels).
* :func:`load_ddf_net_synthetic_per_class` -- load two per-class files and
  assign left/right labels.

The module is also runnable as a CLI. From the repo root::

    py adapters/load_ddf_net_synthetic.py --syn path/to/ddf_samples.npy \
        --syn-labels path/to/labels.npy --subject A01 --normalize trace

which calibrates the dual gate on real ``A01T`` data, filters the synthetic
batch, and reports pass rates and downstream CSP accuracy (Unfiltered vs
Filtered-Cov vs Filtered-Dual).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import EEG_CHANNELS, EVENT_ID
from src.cov_filter import (
    calibrate_covariance_thresholds,
    calibrate_discriminative_threshold,
    cov_reject_sampling,
    disc_reject_sampling,
    extract_covariance_features,
    fit_discriminative_direction,
)
from src.data_loader import load_epochs
from src.downstream import accuracy, fit_csp_lda, predict_csp_lda
from src.experiment import split_train_test

N_CHANNELS = len(EEG_CHANNELS)  # 22
LEFT_ID = int(EVENT_ID["left_hand"])
RIGHT_ID = int(EVENT_ID["right_hand"])


# ---------------------------------------------------------------------------
# Shape normalization
# ---------------------------------------------------------------------------

def normalize_to_trials_channels_times(arr, n_channels=N_CHANNELS):
    """Return ``arr`` as ``(n_epochs, n_channels, n_times)``.

    Accepts ``(n, C, T)``, ``(n, T, C)`` (when ``T == n_channels``), or a flat
    ``(n, C * T)`` array.
    """
    arr = np.asarray(arr, float)
    if arr.ndim == 3:
        if arr.shape[1] == n_channels:
            return arr
        if arr.shape[2] == n_channels:
            return arr.transpose(0, 2, 1)
        raise ValueError(
            f"3-D array must have {n_channels} channels on axis 1 or 2; "
            f"got shape {arr.shape}")
    if arr.ndim == 2:
        n, m = arr.shape
        if m % n_channels != 0:
            raise ValueError(
                f"flat array's second dim ({m}) is not divisible by "
                f"{n_channels} channels")
        return arr.reshape(n, n_channels, m // n_channels)
    raise ValueError(f"expected 2-D or 3-D array, got shape {arr.shape}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_labels(labels_path, n_epochs, syn_path):
    labels = np.load(labels_path, allow_pickle=False)
    labels = np.asarray(labels).ravel()
    if len(labels) != n_epochs:
        raise ValueError(
            f"labels ({len(labels)}) and samples ({n_epochs}) length mismatch "
            f"for {syn_path}")
    return labels


def load_ddf_net_synthetic(path, labels_path=None, n_channels=N_CHANNELS):
    """Load a DDF-Net synthetic ``.npy`` file as ``(data, labels)``.

    ``labels`` are read from ``labels_path`` or a sibling ``<stem>_labels.npy``
    if present; otherwise ``labels`` is ``None`` (the caller must supply them,
    e.g. via :func:`load_ddf_net_synthetic_per_class`).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    data = normalize_to_trials_channels_times(data, n_channels=n_channels)

    if labels_path is None:
        sibling = path.with_name(path.stem + "_labels.npy")
        labels_path = sibling if sibling.exists() else None

    labels = None
    if labels_path is not None:
        labels = _load_labels(labels_path, len(data), path)

    return data, labels


def load_ddf_net_synthetic_per_class(left_path, right_path, n_channels=N_CHANNELS):
    """Load two per-class DDF-Net files and assign left/right labels.

    Returns ``(data, labels)`` where labels are ``{left: 1, right: 2}``.
    """
    left, _ = load_ddf_net_synthetic(left_path, n_channels=n_channels)
    right, _ = load_ddf_net_synthetic(right_path, n_channels=n_channels)
    data = np.concatenate([left, right], axis=0)
    labels = np.concatenate([np.full(len(left), LEFT_ID),
                             np.full(len(right), RIGHT_ID)])
    return data, labels


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------

def filter_with_dual_gate(syn_data, syn_labels, real_data, real_labels,
                          normalize="trace"):
    """Calibrate the dual gate on real data and keep passing synthetic trials.

    Returns a dict with ``mask_cov``, ``mask_disc``, ``mask_dual`` (boolean
    keep-masks) and the three pass rates (fraction kept) plus the calibrated
    ``(cov_thr, w, disc_thr)`` for reuse.
    """
    syn_data = np.asarray(syn_data, float)
    syn_labels = np.asarray(syn_labels)

    real_cov = extract_covariance_features(real_data, normalize=normalize)
    cov_thr = calibrate_covariance_thresholds(real_cov, real_labels)
    w = fit_discriminative_direction(real_cov, real_labels)
    disc_thr = calibrate_discriminative_threshold(real_cov, real_labels, w)

    syn_cov = extract_covariance_features(syn_data, normalize=normalize)
    mask_cov = cov_reject_sampling(syn_cov, syn_labels, cov_thr)
    mask_disc = disc_reject_sampling(syn_cov, syn_labels, w, disc_thr)
    mask_dual = mask_cov & mask_disc

    n = len(syn_data)
    return {
        "mask_cov": mask_cov,
        "mask_disc": mask_disc,
        "mask_dual": mask_dual,
        "pass_cov": float(mask_cov.sum()) / n,
        "pass_disc": float(mask_disc.sum()) / n,
        "pass_dual": float(mask_dual.sum()) / n,
        "cov_thr": cov_thr,
        "w": w,
        "disc_thr": disc_thr,
    }


# ---------------------------------------------------------------------------
# CLI: load real subject -> calibrate -> filter -> evaluate
# ---------------------------------------------------------------------------

def main(args):
    if args.syn_labels:
        syn_data, syn_labels = load_ddf_net_synthetic(args.syn, args.syn_labels)
    elif args.syn_right:
        syn_data, syn_labels = load_ddf_net_synthetic_per_class(
            args.syn, args.syn_right)
    else:
        syn_data, syn_labels = load_ddf_net_synthetic(args.syn)
    if syn_labels is None:
        print("ERROR: no labels found for the synthetic batch. Provide "
              "--syn-labels, or --syn-right for per-class files "
              "(left/right). The gate needs labels to assign class templates "
              "and score windows.")
        return 1

    print(f"[load] {len(syn_data)} synthetic trials x {syn_data.shape[1]} ch x "
          f"{syn_data.shape[2]} samples")

    epochs = load_epochs(args.subject, "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    pool_idx, test_idx = split_train_test(labels)
    pool_data = data[pool_idx]
    pool_labels = labels[pool_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]

    print(f"[calibrate] gates on {args.subject}T ({len(pool_data)} train-pool "
          f"trials, normalize={args.normalize})")
    out = filter_with_dual_gate(syn_data, syn_labels, pool_data, pool_labels,
                                normalize=args.normalize)
    print(f"[filter] pass rates -> covariance {out['pass_cov']:.3f}, "
          f"discriminative {out['pass_disc']:.3f}, dual {out['pass_dual']:.3f} "
          f"({int(out['mask_dual'].sum())}/{len(syn_data)} kept)")

    # Downstream accuracy: real pool + synthetic (all vs. Filtered-Dual).
    def _acc(extra_data, extra_labels):
        train_data = np.concatenate([pool_data, extra_data])
        train_labels = np.concatenate([pool_labels, extra_labels])
        model = fit_csp_lda(train_data, train_labels)
        return accuracy(test_labels, predict_csp_lda(model, test_data))

    acc_unf = _acc(syn_data, syn_labels)
    acc_fcov = _acc(syn_data[out["mask_cov"]], syn_labels[out["mask_cov"]])
    acc_fdual = _acc(syn_data[out["mask_dual"]], syn_labels[out["mask_dual"]])

    print(f"[downstream] Unfiltered {acc_unf:.3f} | "
          f"Filtered-Cov {acc_fcov:.3f} | Filtered-Dual {acc_fdual:.3f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--syn", required=True,
                    help="path to a DDF-Net .npy file (or the left-class file "
                         "when --syn-right is given)")
    ap.add_argument("--syn-labels", default=None,
                    help="optional .npy file of per-trial labels (1=left, 2=right)")
    ap.add_argument("--syn-right", default=None,
                    help="optional right-class .npy; with --syn as the left-class "
                         "file, labels are assigned automatically")
    ap.add_argument("--subject", default="A01",
                    help="real subject used to calibrate the gates (default A01)")
    ap.add_argument("--normalize", default="trace",
                    choices=["raw", "trace", "corr"])
    args = ap.parse_args()
    sys.exit(main(args))
