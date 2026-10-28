"""Generate publication figures for the covariance-gate methodology report.

Produces three figures in ``reports/`` (each saved as both PDF and 300-dpi PNG):

  fig1  Downstream accuracy vs. real-data proportion for the three defect
        models (Baseline / Unfiltered / Filtered-Dual), with significance stars.
  fig2  Retention (pass) rate of the ERD gate, covariance gate, and dual gate
        across the three defect models.
  fig3  LDA discriminative-score KDEs for real left, real right, and class-mixed
        synthetic trials, with the 5%/95% rejection thresholds and the gap.

Provenance of the plotted values:

* Figure 1 and the covariance/dual bars of Figure 2 are the measured 10-seed
  results from ``test_step5.py`` (normalize="trace", A01T, fixed 108/36
  train/test split). The ERD-gate bars of Figure 2 are the source-space ERD
  gate pass rates from ``test_step4.py`` (mean over the 25%/50%/100% proportions,
  of 144 synthetic trials). They are hard-coded here so the figure script does
  not re-run eLORETA.

* Figure 3 is computed live (covariance features + Fisher LDA only, no source
  modelling) and reproduces the calibration of ``test_step5.py`` exactly.

Run:  py generate_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from src.config import EVENT_ID
from src.cov_filter import extract_covariance_features, fit_discriminative_direction
from src.data_loader import load_epochs

mne.set_log_level("ERROR")

REPORTS = Path(__file__).resolve().parent / "reports"
N_SYN = 144  # synthetic trials per batch

# ---------------------------------------------------------------------------
# Measured results (10 seeds, normalize="trace"), from test_step5.py.
# ---------------------------------------------------------------------------
PROPS = [0.25, 0.50, 1.00]
PROP_LABELS = ["25%", "50%", "100%"]
MODES = {1: "Class mixing", 2: "Covariance noise", 3: "Temporal jitter"}

# mode -> prop -> {"base": (mean, std), "unf": (mean, std), "fdual": (mean, std),
#                  "p": p(F-Dual vs Unfiltered, paired t)}
FIG1 = {
    1: {
        0.25: {"base": (0.758, 0.112), "unf": (0.678, 0.061),
               "fdual": (0.750, 0.053), "p": 0.015},
        0.50: {"base": (0.808, 0.058), "unf": (0.739, 0.092),
               "fdual": (0.800, 0.049), "p": 0.016},
        1.00: {"base": (0.833, 0.000), "unf": (0.803, 0.044),
               "fdual": (0.819, 0.038), "p": 0.370},
    },
    2: {
        0.25: {"base": (0.692, 0.072), "unf": (0.692, 0.116),
               "fdual": (0.692, 0.072), "p": 1.000},
        0.50: {"base": (0.767, 0.038), "unf": (0.717, 0.058),
               "fdual": (0.767, 0.038), "p": 0.029},
        1.00: {"base": (0.833, 0.000), "unf": (0.778, 0.037),
               "fdual": (0.833, 0.000), "p": 0.002},
    },
    3: {
        0.25: {"base": (0.722, 0.060), "unf": (0.711, 0.089),
               "fdual": (0.686, 0.069), "p": 0.271},
        0.50: {"base": (0.811, 0.057), "unf": (0.758, 0.060),
               "fdual": (0.744, 0.052), "p": 0.363},
        1.00: {"base": (0.833, 0.000), "unf": (0.831, 0.015),
               "fdual": (0.853, 0.018), "p": 0.011},
    },
}

# Retention counts (of 144 synthetic trials), mean over the 25%/50%/100%
# proportions. ERD gate from test_step4.py; covariance/dual from test_step5.py.
FIG2 = {
    "ERD gate":        [112.2, 106.2, 115.4],
    "Covariance gate": [123.2,   0.0, 129.4],
    "Dual gate":       [ 50.0,   0.0, 114.5],
}

# Colours (colorblind-safe).
C_BASE = "#222222"
C_UNF = "#d62728"
C_DUAL = "#1f77b4"
C_COV = "#ff7f0e"
C_ERD = "#7f7f7f"


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved reports/{name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 3 data (computed live, reproduces test_step5.py calibration)
# ---------------------------------------------------------------------------
def _compute_fig3_data():
    epochs = load_epochs("A01", "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]

    classes = np.unique(labels)
    rng = np.random.default_rng(42)
    test_idx = []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        test_idx.extend(c_idx[:18].tolist())
    test_idx = np.array(test_idx)
    test_set = set(test_idx.tolist())
    pool_idx = np.array([i for i in range(len(labels)) if i not in test_set])
    pool_labels = labels[pool_idx]
    pool_data = data[pool_idx]

    real_cov = extract_covariance_features(pool_data, normalize="trace")
    w = fit_discriminative_direction(real_cov, pool_labels)
    scores = real_cov @ w

    left_id = int(EVENT_ID["left_hand"])
    right_id = int(EVENT_ID["right_hand"])
    left_scores = scores[pool_labels == left_id]
    right_scores = scores[pool_labels == right_id]

    thr = {
        left_id: (float(np.percentile(left_scores, 5)),
                  float(np.percentile(left_scores, 95))),
        right_id: (float(np.percentile(right_scores, 5)),
                   float(np.percentile(right_scores, 95))),
    }

    # Representative class-mixed batch (same generator as test_step5.py).
    gen_rng = np.random.default_rng(0)
    left = pool_data[pool_labels == left_id]
    right = pool_data[pool_labels == right_id]
    n = 2 * 72
    l = gen_rng.integers(0, len(left), size=n)
    r = gen_rng.integers(0, len(right), size=n)
    mix = 0.5 * (left[l] + right[r])
    mix_scores = (extract_covariance_features(mix, normalize="trace")) @ w

    return left_scores, right_scores, mix_scores, thr


def _fig1():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    x = np.arange(len(PROP_LABELS))

    for ax, (mode, name) in zip(axes, MODES.items()):
        base_m = [FIG1[mode][p]["base"][0] for p in PROPS]
        base_s = [FIG1[mode][p]["base"][1] for p in PROPS]
        unf_m = [FIG1[mode][p]["unf"][0] for p in PROPS]
        unf_s = [FIG1[mode][p]["unf"][1] for p in PROPS]
        dual_m = [FIG1[mode][p]["fdual"][0] for p in PROPS]
        dual_s = [FIG1[mode][p]["fdual"][1] for p in PROPS]

        ax.errorbar(x, base_m, yerr=base_s, color=C_BASE, marker="o", lw=2,
                    ms=6, capsize=3, label="Baseline")
        ax.errorbar(x, unf_m, yerr=unf_s, color=C_UNF, marker="s", lw=2, ls="--",
                    ms=6, capsize=3, label="Unfiltered")
        ax.errorbar(x, dual_m, yerr=dual_s, color=C_DUAL, marker="^", lw=2,
                    ms=6, capsize=3, label="Filtered-Dual")

        for i, p in enumerate(PROPS):
            st = _stars(FIG1[mode][p]["p"])
            if st:
                ax.text(i, dual_m[i] + 0.018, st, ha="center", va="bottom",
                        fontsize=14, fontweight="bold", color=C_DUAL)

        ax.set_title(f"Mode {mode}: {name}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(PROP_LABELS)
        ax.set_xlabel("Real-data proportion")
        ax.set_ylim(0.60, 0.90)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Downstream accuracy")
    axes[0].legend(fontsize=10, frameon=True, loc="lower right")
    fig.suptitle("Filtered-Dual recovers downstream accuracy across defect models",
                 fontsize=13)
    fig.text(0.5, 0.005,
             "* p<0.05, ** p<0.01, *** p<0.001 (Filtered-Dual vs Unfiltered, "
             "paired t-test, 10 seeds). Error bars: ±1 std.",
             ha="center", fontsize=9)
    _save(fig, "fig1_accuracy")
    return fig


def _fig2():
    modes = list(MODES.values())
    gates = list(FIG2.keys())
    vals = np.array([FIG2[g] for g in gates]) / N_SYN * 100.0  # percent
    counts = np.array([FIG2[g] for g in gates])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(modes))
    width = 0.26
    colors = {"ERD gate": C_ERD, "Covariance gate": C_COV, "Dual gate": C_DUAL}

    for i, g in enumerate(gates):
        bars = ax.bar(x + (i - 1) * width, vals[i], width,
                      color=colors[g], label=g, edgecolor="white")
        for j, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{counts[i, j]:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Retention rate (% of 144 synthetic trials)")
    ax.set_ylim(0, 100)
    ax.set_title("Gate retention across defect models (annotated = trials kept / 144)")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig2_retention")
    return fig


def _fig3(left_scores, right_scores, mix_scores, thr):
    left_id = int(EVENT_ID["left_hand"])
    right_id = int(EVENT_ID["right_hand"])
    l5, l95 = thr[left_id]
    r5, r95 = thr[right_id]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.kdeplot(left_scores, ax=ax, color=C_DUAL, fill=True, alpha=0.28,
                lw=2, label="Real left-hand")
    sns.kdeplot(right_scores, ax=ax, color=C_UNF, fill=True, alpha=0.28,
                lw=2, label="Real right-hand")
    sns.kdeplot(mix_scores, ax=ax, color="#2ca02c", fill=True, alpha=0.35,
                lw=2, label="Class-mixed synthetic")

    for xx, c in ((l5, C_DUAL), (l95, C_DUAL), (r5, C_UNF), (r95, C_UNF)):
        ax.axvline(xx, color=c, ls="--", lw=1.3, alpha=0.8)

    # The gap between the left 95th and right 5th percentiles.
    ax.axvspan(l95, r5, color="0.5", alpha=0.18)
    mid = (l95 + r5) / 2
    ax.annotate("rejection gap\n(ambiguous class direction)",
                xy=(mid, 0), xytext=(mid, ax.get_ylim()[1] * 0.55),
                ha="center", fontsize=9, style="italic")

    ax.set_xlabel("LDA discriminative score  (features · w)")
    ax.set_ylabel("Density")
    ax.set_title("The discriminative projection gap catches class-mixed trials")
    ax.legend(fontsize=10, frameon=True, loc="upper left")
    _save(fig, "fig3_projection_gap")
    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    print("Computing Figure 3 data (covariance features + LDA, live) ...")
    left_scores, right_scores, mix_scores, thr = _compute_fig3_data()
    print(f"  left  5%/95% = [{thr[int(EVENT_ID['left_hand'])][0]:.3f}, "
          f"{thr[int(EVENT_ID['left_hand'])][1]:.3f}]")
    print(f"  right 5%/95% = [{thr[int(EVENT_ID['right_hand'])][0]:.3f}, "
          f"{thr[int(EVENT_ID['right_hand'])][1]:.3f}]")

    print("Rendering figures ...")
    _fig1()
    _fig2()
    _fig3(left_scores, right_scores, mix_scores, thr)
    print("Done.")


if __name__ == "__main__":
    main()
