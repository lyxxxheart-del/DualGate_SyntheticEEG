"""Generate the DDF-Net rejection figures (the terminal "perfect result").

Both figures reproduce the exact quantities the covariance gate operates on
(``src.cov_filter``, trace-normalized upper-triangle features), so the figures
are faithful to the reported 0/144 pass rate.

  reports/cos_sim_rejection_distribution.pdf/.png
      Figure 1 -- KDE of per-trial covariance cosine-similarity to its own class
      template: real A01T vs the 144 DDF-Net synthetic trials, with the 5%/95%
      acceptance window marked. Synthetic trials collapse outside the window.

  reports/real_vs_ddf_covariance.pdf/.png
      Figure 2 -- three side-by-side 22x22 heatmaps: real right-hand mean
      covariance, DDF-Net synthetic right-hand mean covariance, and their
      element-wise difference (real - synthetic), showing the off-diagonal
      (cross-channel) structure the per-channel generator discards.

Run:  py generate_ddf_rejection_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from src.config import EEG_CHANNELS, EVENT_ID
from src.cov_filter import (
    calibrate_covariance_thresholds,
    extract_covariance_features,
    _cosine,
)
from src.data_loader import load_epochs
from src.experiment import split_train_test

mne.set_log_level("ERROR")

REPORTS = Path(__file__).resolve().parent / "reports"
SYN_DIR = (
    Path(__file__).resolve().parent
    / "adapters" / "ddf_local" / "stage1_out" / "ddf_synthetic"
)

LEFT_ID = int(EVENT_ID["left_hand"])
RIGHT_ID = int(EVENT_ID["right_hand"])

CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved reports/{name}.pdf / .png")


def _load_real():
    epochs = load_epochs("A01", "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    return data, labels


def _load_synthetic():
    syn = np.load(SYN_DIR / "data.npy", allow_pickle=False)
    labels = np.load(SYN_DIR / "labels.npy", allow_pickle=False).ravel()
    return np.asarray(syn, float), np.asarray(labels, int)


def _cos_sims_to_own_template(features, labels, templates):
    return np.array([
        _cosine(features[i], templates[int(labels[i])])
        for i in range(len(features))
    ])


def _trace_normalized_cov_mean(data):
    """Mean trace-normalized energy covariance ``x @ x.T`` over trials."""
    cov = data @ data.swapaxes(1, 2)          # (n, 22, 22)
    tr = np.einsum("nii->n", cov)
    cov = cov / np.where(tr == 0.0, 1.0, tr)[:, None, None]
    return cov.mean(axis=0)


# ---------------------------------------------------------------------------
# Figure 1: covariance cos-sim rejection distribution
# ---------------------------------------------------------------------------

def _fig_cos_sim_distribution(pool_data, pool_labels, syn_data, syn_labels):
    real_cov = extract_covariance_features(pool_data, normalize="trace")
    thr = calibrate_covariance_thresholds(real_cov, pool_labels, verbose=False)
    templates = thr["templates"]
    T_A = thr["T_A"]

    real_sims = _cos_sims_to_own_template(real_cov, pool_labels, templates)
    syn_cov = extract_covariance_features(syn_data, normalize="trace")
    syn_sims = _cos_sims_to_own_template(syn_cov, syn_labels, templates)

    lo = min(T_A[c][0] for c in T_A)
    hi = max(T_A[c][1] for c in T_A)
    n_pass = int(np.sum((syn_sims >= lo) & (syn_sims <= hi)))

    print(f"  real cos-sim:      mean={real_sims.mean():.4f} "
          f"min={real_sims.min():.4f} max={real_sims.max():.4f}")
    print(f"  synthetic cos-sim: mean={syn_sims.mean():.4f} "
          f"min={syn_sims.min():.4f} max={syn_sims.max():.4f}")
    print(f"  acceptance window [{lo:.3f}, {hi:.3f}]  ->  "
          f"synthetic pass {n_pass}/{len(syn_sims)}")

    real_sims = real_sims[np.isfinite(real_sims)]
    syn_sims = syn_sims[np.isfinite(syn_sims)]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.kdeplot(real_sims, ax=ax, color="#1f77b4", fill=True, alpha=0.28,
                lw=2, label=f"Real A01T (n={len(real_sims)})")
    sns.kdeplot(syn_sims, ax=ax, color="#d62728", fill=True, alpha=0.30,
                lw=2, label=f"DDF-Net synthetic (n={len(syn_sims)})")

    ax.axvspan(lo, hi, color="0.5", alpha=0.15)
    ax.axvline(lo, color="0.25", ls="--", lw=1.4, alpha=0.9)
    ax.axvline(hi, color="0.25", ls="--", lw=1.4, alpha=0.9)

    ax.text(0.60, 0.86,
            f"5–95% acceptance window\n[{lo:.3f}, {hi:.3f}]",
            transform=ax.transAxes, fontsize=9, style="italic",
            ha="left", va="top")
    ax.text(0.02, 0.55,
            f"{len(syn_sims) - n_pass}/{len(syn_sims)} synthetic trials\n"
            "fall outside the window",
            transform=ax.transAxes, fontsize=9, style="italic",
            ha="left", va="bottom", color="#d62728")

    xlo = min(-0.02, float(syn_sims.min()) - 0.02)
    ax.set_xlim(xlo, 1.0)
    ax.set_xlabel("Covariance cosine similarity to class template")
    ax.set_ylabel("Density")
    ax.set_title("Gate 1: DDF-Net samples collapse the spatial covariance "
                 "(0% pass)")
    ax.legend(fontsize=10, frameon=True, loc="upper left")
    _save(fig, "cos_sim_rejection_distribution")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: real vs DDF-Net covariance heatmaps
# ---------------------------------------------------------------------------

def _fig_covariance_heatmap(real_data, real_labels, syn_data, syn_labels):
    real_right = real_data[real_labels == RIGHT_ID]
    syn_right = syn_data[syn_labels == RIGHT_ID]

    real_cov = _trace_normalized_cov_mean(real_right)
    syn_cov = _trace_normalized_cov_mean(syn_right)
    diff = real_cov - syn_cov

    # Shared scale for the two covariance panels; symmetric scale for the diff.
    vmax = float(max(np.abs(real_cov).max(), np.abs(syn_cov).max()))
    dmax = float(np.abs(diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ch = EEG_CHANNELS
    im0 = axes[0].imshow(real_cov, cmap=CMAP_SEQ, vmin=0, vmax=vmax,
                         aspect="equal")
    im1 = axes[1].imshow(syn_cov, cmap=CMAP_SEQ, vmin=0, vmax=vmax,
                         aspect="equal")
    im2 = axes[2].imshow(diff, cmap=CMAP_DIV, vmin=-dmax, vmax=dmax,
                         aspect="equal")

    titles = [
        f"(a) Real right-hand covariance",
        f"(b) DDF-Net synthetic right-hand covariance",
        f"(c) Difference  (a) − (b)",
    ]
    for ax, im, t in zip(axes, (im0, im1, im2), titles):
        ax.set_title(t, fontsize=11)
        ax.set_xticks(range(len(ch)))
        ax.set_yticks(range(len(ch)))
        ax.set_xticklabels(ch, rotation=90, fontsize=6)
        ax.set_yticklabels(ch, fontsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    axes[2].annotate("off-diagonal\n(cross-channel)\nstructure lost",
                     xy=(6, 16), xytext=(12, 16), fontsize=8,
                     style="italic", color="black",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    fig.suptitle("Per-channel generation diagonalizes the spatial covariance "
                 "(Gate 1 detects the off-diagonal collapse)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "real_vs_ddf_covariance")
    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    print("Loading A01T ...")
    data, labels = _load_real()
    print("Loading DDF-Net synthetic batch ...")
    syn_data, syn_labels = _load_synthetic()

    pool_idx, _ = split_train_test(labels)
    pool_data = data[pool_idx]
    pool_labels = labels[pool_idx]

    print("Rendering Figure 1 (cos-sim rejection distribution) ...")
    _fig_cos_sim_distribution(pool_data, pool_labels, syn_data, syn_labels)

    print("Rendering Figure 2 (real vs DDF-Net covariance) ...")
    _fig_covariance_heatmap(data, labels, syn_data, syn_labels)

    print("Done.")


if __name__ == "__main__":
    main()
