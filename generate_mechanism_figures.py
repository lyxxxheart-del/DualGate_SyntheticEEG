"""Generate the two mechanism figures for the paper (Task 4).

Both figures are computed live from A01T (sensor-space covariance features; no
source modelling), so they reproduce the exact quantities the gate operates on.

  reports/mechanism_projection_gap.pdf/.png
      Figure 1 -- LDA discriminative-score KDEs for real left, real right, and
      class-mixed synthetic trials, with the 5%/95% rejection windows and the
      gap between the classes. This is where class-ambiguous blends fall and get
      rejected by Gate 2.

  reports/mechanism_covariance_heatmap.pdf/.png
      Figure 2 -- three side-by-side 22x22 heatmaps: the real left-hand
      class-mean covariance, the same covariance after 0 dB white-noise
      corruption, and their element-wise difference. White noise inflates the
      diagonal (self-power) while collapsing the off-diagonal (cross-channel)
      structure -- the "diagonalization" that Gate 1 detects.

Run:  py generate_mechanism_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from src.config import EEG_CHANNELS, EVENT_ID
from src.cov_filter import extract_covariance_features, fit_discriminative_direction
from src.data_loader import load_epochs
from src.experiment import split_train_test

mne.set_log_level("ERROR")

REPORTS = Path(__file__).resolve().parent / "reports"

LEFT_ID = int(EVENT_ID["left_hand"])
RIGHT_ID = int(EVENT_ID["right_hand"])

# Colorblind-safe, perceptually-uniform colormaps (sequential + diverging).
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved reports/{name}.pdf / .png")


# ---------------------------------------------------------------------------
# Shared: load A01T and its covariance quantities
# ---------------------------------------------------------------------------

def _load_a01t():
    epochs = load_epochs("A01", "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    return data, labels


def _trace_normalized_cov_mean(data):
    """Mean trace-normalized energy covariance ``x @ x.T`` over trials."""
    cov = data @ data.swapaxes(1, 2)          # (n, 22, 22)
    tr = np.einsum("nii->n", cov)
    cov = cov / np.where(tr == 0.0, 1.0, tr)[:, None, None]
    return cov.mean(axis=0)


# ---------------------------------------------------------------------------
# Figure 1: projection gap KDE
# ---------------------------------------------------------------------------

def _fig_projection_gap(data, labels):
    pool_idx, _ = split_train_test(labels)
    pool_data = data[pool_idx]
    pool_labels = labels[pool_idx]

    real_cov = extract_covariance_features(pool_data, normalize="trace")
    w = fit_discriminative_direction(real_cov, pool_labels)
    scores = real_cov @ w

    left_scores = scores[pool_labels == LEFT_ID]
    right_scores = scores[pool_labels == RIGHT_ID]
    thr = {
        LEFT_ID: (float(np.percentile(left_scores, 5)),
                  float(np.percentile(left_scores, 95))),
        RIGHT_ID: (float(np.percentile(right_scores, 5)),
                   float(np.percentile(right_scores, 95))),
    }

    # Representative class-mixed batch (same generator as the experiments).
    gen_rng = np.random.default_rng(0)
    left = pool_data[pool_labels == LEFT_ID]
    right = pool_data[pool_labels == RIGHT_ID]
    n = 2 * 72
    l = gen_rng.integers(0, len(left), size=n)
    r = gen_rng.integers(0, len(right), size=n)
    mix = 0.5 * (left[l] + right[r])
    mix_scores = extract_covariance_features(mix, normalize="trace") @ w

    l5, l95 = thr[LEFT_ID]
    r5, r95 = thr[RIGHT_ID]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.kdeplot(left_scores, ax=ax, color="#1f77b4", fill=True, alpha=0.28,
                lw=2, label="Real left-hand")
    sns.kdeplot(right_scores, ax=ax, color="#d62728", fill=True, alpha=0.28,
                lw=2, label="Real right-hand")
    sns.kdeplot(mix_scores, ax=ax, color="#2ca02c", fill=True, alpha=0.35,
                lw=2, label="Class-mixed synthetic")

    for xx, c in ((l5, "#1f77b4"), (l95, "#1f77b4"),
                  (r5, "#d62728"), (r95, "#d62728")):
        ax.axvline(xx, color=c, ls="--", lw=1.3, alpha=0.8)

    ax.axvspan(l95, r5, color="0.5", alpha=0.18)
    mid = (l95 + r5) / 2
    ax.annotate("rejection gap\n(ambiguous class direction)",
                xy=(mid, 0), xytext=(mid, ax.get_ylim()[1] * 0.55),
                ha="center", fontsize=9, style="italic")

    ax.set_xlabel("LDA discriminative score  (features · w)")
    ax.set_ylabel("Density")
    ax.set_title("Gate 2: the discriminative projection gap catches class-mixed trials")
    ax.legend(fontsize=10, frameon=True, loc="upper left")
    _save(fig, "mechanism_projection_gap")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: covariance heatmap comparison
# ---------------------------------------------------------------------------

def _fig_covariance_heatmap(data, labels):
    left_data = data[labels == LEFT_ID]

    real_cov = _trace_normalized_cov_mean(left_data)

    # 0 dB white-noise corruption (same recipe as the Mode-2 generator).
    rng = np.random.default_rng(0)
    std = left_data.std(axis=(1, 2), keepdims=True)
    noisy = left_data + std * rng.standard_normal(left_data.shape)
    noise_cov = _trace_normalized_cov_mean(noisy)

    diff = noise_cov - real_cov

    vmax = float(np.abs(real_cov).max())
    dmax = float(np.abs(diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ch = EEG_CHANNELS
    im0 = axes[0].imshow(real_cov, cmap=CMAP_SEQ, vmin=0, vmax=vmax,
                         aspect="equal")
    im1 = axes[1].imshow(noise_cov, cmap=CMAP_SEQ, vmin=0, vmax=vmax,
                         aspect="equal")
    im2 = axes[2].imshow(diff, cmap=CMAP_DIV, vmin=-dmax, vmax=dmax,
                         aspect="equal")

    titles = [
        "(a) Real left-hand covariance",
        "(b) + 0 dB white noise",
        "(c) Difference  (b) - (a)",
    ]
    for ax, im, t in zip(axes, (im0, im1, im2), titles):
        ax.set_title(t, fontsize=11)
        ax.set_xticks(range(len(ch)))
        ax.set_yticks(range(len(ch)))
        ax.set_xticklabels(ch, rotation=90, fontsize=6)
        ax.set_yticklabels(ch, fontsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    axes[2].annotate("off-diagonal\nstructure lost", xy=(6, 16),
                     xytext=(11, 16), fontsize=8, style="italic", color="black",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    fig.suptitle("White noise diagonalizes the spatial covariance (Gate 1 detects "
                 "off-diagonal collapse)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "mechanism_covariance_heatmap")
    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    print("Loading A01T ...")
    data, labels = _load_a01t()

    print("Rendering Figure 1 (projection gap) ...")
    _fig_projection_gap(data, labels)

    print("Rendering Figure 2 (covariance heatmap) ...")
    _fig_covariance_heatmap(data, labels)

    print("Done.")


if __name__ == "__main__":
    main()
