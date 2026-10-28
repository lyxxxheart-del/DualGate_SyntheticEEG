"""Threshold calibration on real MI-EEG data.

Criterion A (PCA topography, per-class direction): project real feature vectors
onto a PCA subspace and score each trial against the left/right discriminative
direction ``diff_vec = template_right - template_left`` via ``corr(x, diff_vec)``
(left -> negative, right -> positive). A separate 5th/95th-percentile window is
estimated per class, so a synthetic trial must fall in its own class's window and
point the correct way.

Criterion B (onset latency): percentile window of sustained-ERD onset times.
"""

import numpy as np
from mne.filter import filter_data

from .config import (
    BASELINE_TMAX,
    BASELINE_TMIN,
    EEG_CHANNELS,
    EVENT_ID,
    FS,
    MU_BAND,
    TASK_TMAX,
    TASK_TMIN,
    TMIN,
)
from .pipeline import erd_onset_latency

N_PCA_COMPONENTS = 8


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0.0 or b.std() == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def topography_scores(features, criterion_a):
    """Left/right signed topography score (corr with right-minus-left direction)."""
    features = np.asarray(features, float)
    red = (features - criterion_a["mean"]) @ criterion_a["components"]
    return np.array([_pearson(red[i], criterion_a["diff_vec"])
                     for i in range(len(red))])


def calibrate_thresholds(erd_features, erd_trajectory, labels):
    """Calibrate physiological acceptance thresholds from real trials.

    Parameters
    ----------
    erd_features : ndarray, shape (n_trials, n_features)
    erd_trajectory : ndarray, shape (n_trials, n_times)
    labels : ndarray, shape (n_trials,)

    Returns
    -------
    criterion_a : dict
        PCA projection model (``mean``, ``components``, ``diff_vec``).
    thresholds : dict
        ``{"T_A": {label: (lo, hi)}, "T_B_min", "T_B_max"}``.
    """
    features = np.asarray(erd_features, float)
    traj = np.asarray(erd_trajectory, float)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    # --- Criterion A: PCA subspace + left/right signed score ---
    mean = features.mean(axis=0)
    Xc = features - mean
    k = min(N_PCA_COMPONENTS, Xc.shape[1], Xc.shape[0] - 1)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].T
    red = Xc @ components
    templates = {int(c): red[labels == c].mean(axis=0) for c in classes}

    left_id = int(EVENT_ID["left_hand"])
    right_id = int(EVENT_ID["right_hand"])
    diff_vec = templates[right_id] - templates[left_id]

    criterion_a = {
        "mean": mean,
        "components": components,
        "diff_vec": diff_vec,
        "templates": templates,
        "n_components": k,
    }

    scores = topography_scores(features, criterion_a)
    T_A = {
        int(c): (float(np.percentile(scores[labels == c], 5)),
                 float(np.percentile(scores[labels == c], 95)))
        for c in classes
    }

    # --- Criterion B: onset latency window ---
    latencies = np.array([erd_onset_latency(t) for t in traj])
    valid = latencies[~np.isnan(latencies)]
    if len(valid) == 0:
        T_B_min = T_B_max = float("nan")
    else:
        T_B_min = float(np.percentile(valid, 5))
        T_B_max = float(np.percentile(valid, 95))

    thresholds = {"T_A": T_A, "T_B_min": T_B_min, "T_B_max": T_B_max}

    print("Source-space calibration (real data):")
    for c in sorted(T_A):
        name = "left" if c == left_id else "right"
        print(f"  T_A_{name} (class {c}) = [{T_A[c][0]:.3f}, {T_A[c][1]:.3f}]")
    print(f"  T_B = [{T_B_min:.3f}, {T_B_max:.3f}] s "
          f"(sustained ERD in {len(valid)}/{len(latencies)})")

    return criterion_a, thresholds


def sensor_erd_scores(data, sfreq=FS, tmin=TMIN):
    """C3/C4 mu-band ERD lateralization score per trial.

    Returns ``ERD(C4) - ERD(C3)``; left-hand MI -> negative, right-hand ->
    positive. ``data`` has shape ``(n_trials, n_channels, n_times)`` with
    channels in EEG order (see ``EEG_CHANNELS``).
    """
    data = np.asarray(data, float)
    c3 = EEG_CHANNELS.index("C3")
    c4 = EEG_CHANNELS.index("C4")
    mu = filter_data(data[:, [c3, c4]], sfreq, MU_BAND[0], MU_BAND[1], verbose=False)

    times = tmin + np.arange(data.shape[2]) / sfreq
    base_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
    task_mask = (times >= TASK_TMIN) & (times < TASK_TMAX)
    p_base = (mu[:, :, base_mask] ** 2).mean(axis=2)  # (n, 2): [C3, C4]
    p_task = (mu[:, :, task_mask] ** 2).mean(axis=2)

    def _erd(task, base):
        base = np.where(base == 0.0, np.nan, base)
        return (task - base) / base

    erd = _erd(p_task, p_base)
    return erd[:, 1] - erd[:, 0]  # C4 - C3


def calibrate_sensor_erd(epochs):
    """Calibrate per-class C3/C4 mu-ERD thresholds from real epochs."""
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    scores = sensor_erd_scores(data)
    classes = np.unique(labels)
    left_id = int(EVENT_ID["left_hand"])
    T_A = {
        int(c): (float(np.percentile(scores[labels == c], 5)),
                 float(np.percentile(scores[labels == c], 95)))
        for c in classes
    }

    print("Sensor-space calibration (C3/C4 mu ERD):")
    for c in sorted(T_A):
        name = "left" if c == left_id else "right"
        print(f"  T_A_sensor_{name} (class {c}) = [{T_A[c][0]:.3f}, {T_A[c][1]:.3f}]")

    return {"T_A_sensor": T_A}
