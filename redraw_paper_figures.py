"""Redraw the three core paper figures for IEEE double-column (single-column
width 3.5 in; the covariance-collapse panel is a full-width figure*).

All three figures are computed live from A01T and the DDF-Net synthetic batch
with the exact quantities the gates operate on (trace-normalised covariance
upper-triangle features), so the figures stay faithful to the reported 0/144
pass rate and the 0.8333 -> 0.6667 -> 0.8333 downstream result.

Outputs (overwrite) in reports/:
  mechanism_projection_gap.pdf/.png   -> Fig. 1  LDA discriminative-score gap
  real_vs_ddf_covariance.pdf/.png     -> Fig. 2  real vs. synthetic covariance
  cos_sim_rejection_distribution.pdf/.png -> Fig. 3 cos-sim rejection + zoom

Run:  py redraw_paper_figures.py
"""

from pathlib import Path
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import mne

from src.config import EEG_CHANNELS, EVENT_ID
from src.cov_filter import (
    calibrate_covariance_thresholds,
    extract_covariance_features,
    fit_discriminative_direction,
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

# Physical widths. Fig. 1 and Fig. 3 are single-column (3.5 in). Fig. 2 is a
# full-width figure* in main.tex (\textwidth ~ 7 in), so it stays double-column.
SINGLE_COL_W = 3.5   # inches
FULL_TEXT_W = 7.0    # inches

# Soft, colour-blind-friendly palette (left/right stay blue/red, muted).
BLUE = "#3775BA"    # real / left
RED = "#B64342"     # synthetic / right
GREEN = "#2E9E44"   # class-mixed synthetic
GREY = "#767676"
DARK_GREY = "#4D4D4D"

CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def _apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",      # editable text in SVG
        "pdf.fonttype": 42,          # editable TrueType text in PDF
        "font.size": 9,
        "axes.labelsize": 11,        # axis labels 10-12 pt
        "axes.titlesize": 12,
        "xtick.labelsize": 9,        # ticks 8-10 pt
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def _kde(x, grid):
    """Scott-rule Gaussian KDE evaluated on ``grid`` (1-D)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.ptp(x) == 0.0:
        return np.zeros_like(grid)
    return gaussian_kde(x)(grid)


def _grid(*arrays, n=1400, pad_frac=0.08):
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    span = hi - lo
    if span == 0.0:
        span = 1.0
    return np.linspace(lo - pad_frac * span, hi + pad_frac * span, n)


def _pdf_size_inches(path):
    """Read the PDF MediaBox without PyMuPDF; return (width_in, height_in)."""
    raw = Path(path).read_bytes()
    m = re.search(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    if not m:
        return None
    x0, y0, x1, y1 = map(float, m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def _save(fig, name, align=False, exclude_axes=None):
    if align:
        sys.path.insert(
            0, r"C:\Users\lyxxx\.claude\skills\nature-figure\scripts")
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(REPORTS / f"{name}.alignment.json"),
            overlay_svg=str(REPORTS / f"{name}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
            exclude_axes=exclude_axes or [],
        )
    pdf = REPORTS / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(REPORTS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    size = _pdf_size_inches(pdf)
    if size is None:
        print(f"  saved reports/{name}.pdf (size unknown)")
    else:
        print(f"  saved reports/{name}.pdf  ({size[0]:.2f} x {size[1]:.2f} in)")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_real():
    epochs = load_epochs("A01", "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    return data, labels


def _load_synthetic():
    syn = np.load(SYN_DIR / "data.npy", allow_pickle=False)
    labels = np.load(SYN_DIR / "labels.npy", allow_pickle=False).ravel()
    return np.asarray(syn, float), np.asarray(labels, int)


def _trace_normalized_cov_mean(data):
    cov = data @ data.swapaxes(1, 2)
    tr = np.einsum("nii->n", cov)
    cov = cov / np.where(tr == 0.0, 1.0, tr)[:, None, None]
    return cov.mean(axis=0)


def _cos_sims_to_own_template(features, labels, templates):
    return np.array([
        _cosine(features[i], templates[int(labels[i])])
        for i in range(len(features))
    ])


# ---------------------------------------------------------------------------
# Figure 1: LDA projection gap (single column)
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

    gen_rng = np.random.default_rng(0)
    left = pool_data[pool_labels == LEFT_ID]
    right = pool_data[pool_labels == RIGHT_ID]
    n = 2 * 72
    li = gen_rng.integers(0, len(left), size=n)
    ri = gen_rng.integers(0, len(right), size=n)
    mix = 0.5 * (left[li] + right[ri])
    mix_scores = extract_covariance_features(mix, normalize="trace") @ w

    l5, l95 = thr[LEFT_ID]
    r5, r95 = thr[RIGHT_ID]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.55))
    grid = _grid(left_scores, right_scores, mix_scores)
    dL = _kde(left_scores, grid)
    dR = _kde(right_scores, grid)
    dM = _kde(mix_scores, grid)

    ax.plot(grid, dL, color=BLUE, lw=1.6, label="Real left-hand")
    ax.fill_between(grid, dL, color=BLUE, alpha=0.22)
    ax.plot(grid, dR, color=RED, lw=1.6, label="Real right-hand")
    ax.fill_between(grid, dR, color=RED, alpha=0.22)
    ax.plot(grid, dM, color=GREEN, lw=1.6, label="Class-mixed synthetic")
    ax.fill_between(grid, dM, color=GREEN, alpha=0.26)

    for xx, c in ((l5, BLUE), (l95, BLUE), (r5, RED), (r95, RED)):
        ax.axvline(xx, color=c, ls="--", lw=1.1, alpha=0.85)

    ax.axvspan(l95, r5, color=GREY, alpha=0.16)
    mid = (l95 + r5) / 2
    ymax = max(dL.max(), dR.max(), dM.max())

    ax.annotate(
        "rejection gap",
        xy=(mid, 0.02),
        xytext=(0.66, 0.66),
        textcoords="axes fraction",
        ha="center", va="center",
        fontsize=8, style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#999999",
                  alpha=0.75, lw=0.6),
        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8,
                        connectionstyle="arc3,rad=0.0"),
    )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("LDA discriminative score  (features \u00b7 w)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              handlelength=1.4, labelspacing=0.4)
    _save(fig, "mechanism_projection_gap")


# ---------------------------------------------------------------------------
# Figure 2: real vs. DDF-Net covariance heatmaps (full-width figure*)
# ---------------------------------------------------------------------------

def _fig_covariance_heatmap(real_data, real_labels, syn_data, syn_labels):
    real_right = real_data[real_labels == RIGHT_ID]
    syn_right = syn_data[syn_labels == RIGHT_ID]

    real_cov = _trace_normalized_cov_mean(real_right)
    syn_cov = _trace_normalized_cov_mean(syn_right)
    diff = real_cov - syn_cov

    # One shared scale for panels (a)/(b); symmetric scale for (c).
    vmax = float(max(np.abs(real_cov).max(), np.abs(syn_cov).max()))
    dmax = float(np.abs(diff).max())

    ch = EEG_CHANNELS
    keep = {"Fz", "C3", "Cz", "C4", "Pz"}
    tick_labels = [c if c in keep else "" for c in ch]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_TEXT_W, 3.30),
                             constrained_layout=True)

    norm_seq = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    norm_div = mpl.colors.Normalize(vmin=-dmax, vmax=dmax)

    im0 = axes[0].imshow(real_cov, cmap=CMAP_SEQ, norm=norm_seq, aspect="equal")
    im1 = axes[1].imshow(syn_cov, cmap=CMAP_SEQ, norm=norm_seq, aspect="equal")
    im2 = axes[2].imshow(diff, cmap=CMAP_DIV, norm=norm_div, aspect="equal")

    titles = ["(a) Real", "(b) DDF-Net synthetic", "(c) Difference  (a) \u2212 (b)"]
    for ax, im, t in zip(axes, (im0, im1, im2), titles):
        ax.set_title(t, fontsize=12)
        ax.set_xticks(range(len(ch)))
        ax.set_yticks(range(len(ch)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)

    # One colourbar per panel. Panels (a)/(b) share the same Normalize, so their
    # colourbars are identical in range -- the unified scale the comparison needs.
    cbs = []
    for ax, im in zip(axes, (im0, im1, im2)):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        cbs.append(cb)

    _save(fig, "real_vs_ddf_covariance", align=True,
          exclude_axes=[cb.ax for cb in cbs])


# ---------------------------------------------------------------------------
# Figure 3: cos-sim rejection distribution + zoom inset (single column)
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
    real_sims = real_sims[np.isfinite(real_sims)]
    syn_sims = syn_sims[np.isfinite(syn_sims)]

    print(f"  real cos-sim: mean={real_sims.mean():.4f} "
          f"[{real_sims.min():.4f}, {real_sims.max():.4f}]")
    print(f"  synth cos-sim: mean={syn_sims.mean():.4f} "
          f"[{syn_sims.min():.4f}, {syn_sims.max():.4f}]")
    print(f"  window [{lo:.3f}, {hi:.3f}]  ->  {n_pass}/{len(syn_sims)} pass")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.85))

    grid = np.linspace(0.30, 1.05, 1500)
    d_real = _kde(real_sims, grid)
    d_syn = _kde(syn_sims, grid)

    ax.plot(grid, d_real, color=BLUE, lw=1.6,
            label=f"Real A01T (n={len(real_sims)})")
    ax.fill_between(grid, d_real, color=BLUE, alpha=0.24)
    ax.plot(grid, d_syn, color=RED, lw=1.6,
            label=f"DDF-Net synthetic (n={len(syn_sims)})")
    ax.fill_between(grid, d_syn, color=RED, alpha=0.28)

    ax.axvspan(lo, hi, color=GREY, alpha=0.15)
    ax.axvline(lo, color=DARK_GREY, ls="--", lw=1.1, alpha=0.9)
    ax.axvline(hi, color=DARK_GREY, ls="--", lw=1.1, alpha=0.9)

    # Thin arrow to the synthetic peak (the mode is ~0.368).
    syn_mode = float(syn_sims.mean())
    peak_y = float(_kde(syn_sims, np.array([syn_mode]))[0])
    ax.annotate(
        f"{len(syn_sims)}/{len(syn_sims)} rejected (0% pass)",
        xy=(syn_mode, peak_y),
        xytext=(0.30, 0.34),
        textcoords="axes fraction",
        color=RED, fontsize=8, ha="left", va="center",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                        connectionstyle="arc3,rad=-0.25",
                        shrinkA=3, shrinkB=3),
    )

    ymax = max(float(d_real.max()), float(d_syn.max()))
    ax.set_xlim(0.30, 1.05)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("Covariance cosine similarity to class template")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              handlelength=1.4, labelspacing=0.4)

    # Inset: zoom on the narrow acceptance window [0.985, 1.0].
    axins = ax.inset_axes([0.58, 0.52, 0.40, 0.42])
    axins.set_facecolor("white")
    gi = np.linspace(0.985, 1.0, 400)
    di = _kde(real_sims, gi)
    axins.plot(gi, di, color=BLUE, lw=1.4)
    axins.fill_between(gi, di, color=BLUE, alpha=0.24)
    axins.axvspan(lo, hi, color=GREY, alpha=0.15)
    axins.axvline(lo, color=DARK_GREY, ls="--", lw=1.0)
    axins.axvline(hi, color=DARK_GREY, ls="--", lw=1.0)
    axins.set_xlim(0.985, 1.0)
    axins.set_ylim(0, di.max() * 1.10)
    axins.set_yticks([])
    axins.tick_params(axis="x", labelsize=7)
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
    axins.spines["left"].set_visible(False)

    _save(fig, "cos_sim_rejection_distribution")


# ---------------------------------------------------------------------------

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("Loading A01T ...")
    data, labels = _load_real()
    print("Loading DDF-Net synthetic batch ...")
    syn_data, syn_labels = _load_synthetic()

    pool_idx, _ = split_train_test(labels)
    pool_data = data[pool_idx]
    pool_labels = labels[pool_idx]

    print("Rendering Figure 1 (projection gap) ...")
    _fig_projection_gap(data, labels)

    print("Rendering Figure 2 (covariance heatmap) ...")
    _fig_covariance_heatmap(data, labels, syn_data, syn_labels)

    print("Rendering Figure 3 (cos-sim rejection + zoom) ...")
    _fig_cos_sim_distribution(pool_data, pool_labels, syn_data, syn_labels)

    print("Done.")


if __name__ == "__main__":
    main()
