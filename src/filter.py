"""Physiological rejection sampling for synthetic MI-EEG trials."""

import numpy as np

from .pipeline import erd_onset_latency
from .calibration import sensor_erd_scores, topography_scores


def physio_reject_sampling(
    erd_features_syn, erd_trajectory_syn, labels_syn, criterion_a, thresholds
):
    """Source-space rejection: per-class topography (A) AND onset latency (B).

    Returns a boolean mask (True = keep) over the synthetic trials.
    """
    features = np.asarray(erd_features_syn, float)
    traj = np.asarray(erd_trajectory_syn, float)
    labels = np.asarray(labels_syn)

    scores = topography_scores(features, criterion_a)
    T_A = thresholds["T_A"]
    T_B_min = thresholds["T_B_min"]
    T_B_max = thresholds["T_B_max"]

    mask = np.empty(len(features), dtype=bool)
    for i in range(len(features)):
        lo, hi = T_A[int(labels[i])]
        ok_a = (not np.isnan(scores[i])) and (lo <= scores[i] <= hi)
        lat = erd_onset_latency(traj[i])
        ok_b = (not np.isnan(lat)) and (T_B_min <= lat <= T_B_max)
        mask[i] = ok_a and ok_b
    return mask


def sensor_erd_reject_sampling(data_syn, labels_syn, sensor_thresholds):
    """Sensor-space rejection: per-class C3/C4 mu-ERD topography only.

    Returns a boolean mask (True = keep) over the synthetic trials.
    """
    data = np.asarray(data_syn, float)
    labels = np.asarray(labels_syn)
    scores = sensor_erd_scores(data)
    T_A = sensor_thresholds["T_A_sensor"]

    mask = np.empty(len(data), dtype=bool)
    for i in range(len(data)):
        lo, hi = T_A[int(labels[i])]
        mask[i] = (not np.isnan(scores[i])) and (lo <= scores[i] <= hi)
    return mask
