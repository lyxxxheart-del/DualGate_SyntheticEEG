"""Diagnose single-trial ERD topography discrimination (criterion A).

Compares within- vs between-class template correlation on the raw 100-vertex
features, then evaluates three alternative criterion-A formulations (laterality
index, spatial smoothing, PCA reduction) by how well each separates left- vs
right-hand trials.

Run from the project root:  py diagnose_criterion_a.py
"""

import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from scipy.spatial.distance import cdist

from src.data_loader import load_epochs
from src.pipeline import compute_erd_source, get_roi_vertex_positions


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0.0 or b.std() == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0.0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def _gaussian_smooth(features, pos, fwhm):
    """Spatially smooth feature rows with a Gaussian kernel over vertex positions."""
    dist = cdist(pos, pos)  # (n_roi, n_roi), meters
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    W /= W.sum(axis=1, keepdims=True)  # row-normalized
    return features @ W.T


def _report_within_between(name, within, between):
    print(f"\n[{name}] within- vs between-template r")
    for nm, arr in (("within", within), ("between", between)):
        q = np.nanpercentile(arr, [5, 20, 50, 80, 95])
        print(f"  {nm:7s}: mean={np.nanmean(arr):+.4f}  median={np.nanmedian(arr):+.4f}  "
              f"p5/20/50/80/95 = {['%.4f' % v for v in q]}")
    d = _cohens_d(within, between)
    t, p = ttest_rel(within, between)
    print(f"  Cohen's d (within vs between) = {d:+.3f}, paired t-test p = {p:.3e}")


def _report_signed_separation(name, score, labels):
    """Print separation of a signed class score (negative -> left, positive -> right)."""
    s_left = score[labels == 1]
    s_right = score[labels == 2]
    d = _cohens_d(s_right, s_left)  # positive = right scores higher (correct direction)
    t, p = ttest_ind(s_right, s_left)
    correct = int(((labels == 1) & (score < 0)).sum()
                  + ((labels == 2) & (score > 0)).sum())
    print(f"\n[{name}]")
    print(f"  mean score: left={s_left.mean():+.4f}  right={s_right.mean():+.4f}")
    print(f"  correct-side rate: {correct}/{len(labels)} ({correct / len(labels):.0%})")
    print(f"  Cohen's d (right vs left) = {d:+.3f}")
    print(f"  t-test (independent) p = {p:.3e}")
    return d


if __name__ == "__main__":
    print("Loading A01T and computing source-space ERD (mu band) ...")
    epochs = load_epochs(subject="A01", run="T")
    erd_mu, erd_beta, erd_traj = compute_erd_source(epochs)
    labels = epochs.events[:, 2]

    roi, pos = get_roi_vertex_positions(epochs.info)
    x = pos[:, 0]
    left_mask = x < 0.0
    right_mask = x > 0.0
    print(f"\nROI: {len(roi)} vertices | X range [{x.min() * 1000:.1f}, "
          f"{x.max() * 1000:.1f}] mm | left={int(left_mask.sum())}, right={int(right_mask.sum())}")

    tpl = {int(c): erd_mu[labels == c].mean(axis=0) for c in (1, 2)}
    other = {1: 2, 2: 1}

    # ---- Task 1: raw 100-dim within- vs between-class template r ----
    within = np.array([_pearson(erd_mu[i], tpl[int(labels[i])]) for i in range(len(erd_mu))])
    between = np.array([_pearson(erd_mu[i], tpl[other[int(labels[i])]]) for i in range(len(erd_mu))])
    print("\n=== Task 1: raw 100-dim features ===")
    _report_within_between("raw", within, between)

    # ---- Scheme 1: laterality index ----
    li = erd_mu[:, right_mask].mean(axis=1) - erd_mu[:, left_mask].mean(axis=1)
    _report_signed_separation("Scheme 1: Laterality Index (ERD_right - ERD_left)", li, labels)

    # ---- Scheme 2: spatial smoothing + template r ----
    fwhm = 0.015  # 15 mm
    smooth = _gaussian_smooth(erd_mu, pos, fwhm)
    stpl = {int(c): smooth[labels == c].mean(axis=0) for c in (1, 2)}
    within2 = np.array([_pearson(smooth[i], stpl[int(labels[i])]) for i in range(len(smooth))])
    between2 = np.array([_pearson(smooth[i], stpl[other[int(labels[i])]]) for i in range(len(smooth))])
    score2 = np.array([_pearson(smooth[i], stpl[2]) - _pearson(smooth[i], stpl[1])
                       for i in range(len(smooth))])
    print(f"\n=== Scheme 2: spatial smoothing (FWHM={fwhm * 1000:.0f} mm) ===")
    _report_within_between("smoothed", within2, between2)
    _report_signed_separation(f"Scheme 2 (signed r_right - r_left)", score2, labels)

    # ---- Scheme 3: PCA reduction + template r ----
    k = 8
    Xc = erd_mu - erd_mu.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:k].T  # (n_roi, k)
    red = Xc @ P  # (n_trials, k)
    rtpl = {int(c): red[labels == c].mean(axis=0) for c in (1, 2)}
    within3 = np.array([_pearson(red[i], rtpl[int(labels[i])]) for i in range(len(red))])
    between3 = np.array([_pearson(red[i], rtpl[other[int(labels[i])]]) for i in range(len(red))])
    score3 = np.array([_pearson(red[i], rtpl[2]) - _pearson(red[i], rtpl[1])
                       for i in range(len(red))])
    print(f"\n=== Scheme 3: PCA reduction (top {k} PCs) ===")
    _report_within_between("PCA", within3, between3)
    _report_signed_separation(f"Scheme 3 (signed r_right - r_left, {k} PCs)", score3, labels)

    print("\nDone.")
