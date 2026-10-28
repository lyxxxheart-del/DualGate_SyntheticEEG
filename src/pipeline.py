"""Source-space ERD pipeline (eLORETA) for BCI Competition IV 2a.

Projects band-pass filtered EEG epochs onto the fsaverage cortical surface via
eLORETA and computes, per trial, (1) mu/beta ERD features over a sensory-motor
ROI and (2) the ROI-averaged mu-band ERD time trajectory used for onset-latency
estimation.

Per vertex and band: ``ERD = (P_task - P_base) / P_base`` where ``P_task`` is the
mean band power over the task window and ``P_base`` the mean power over the
pre-cue baseline window.
"""

from pathlib import Path

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.filter import filter_data
from scipy.spatial.distance import cdist

from .config import (
    BASELINE_TMAX,
    BASELINE_TMIN,
    BETA_BAND,
    FS,
    FSAVERAGE_SUBJECT,
    MONTAGE_NAME,
    MU_BAND,
    N_ROI_VERTICES,
    SENSORIMOTOR_ROI_CHANNELS,
    SNR,
    SOURCE_SPACING,
    TASK_TMAX,
    TASK_TMIN,
)

_FORWARD_CACHE = {}

_INVERSE_CACHE = {}


def get_subjects_dir() -> Path:
    """Return the subjects directory containing ``fsaverage`` (downloaded once)."""
    return Path(fetch_fsaverage(verbose=False)).parent


def _ensure_montage(epochs):
    """Attach the template 10-20 montage if the epochs carry no channel positions."""
    if epochs.get_montage() is None:
        epochs.set_montage(mne.channels.make_standard_montage(MONTAGE_NAME))
    return epochs


def _get_forward_solution(info):
    """Build (and cache) the fsaverage EEG forward solution for an info."""
    key = (tuple(info["ch_names"]), MONTAGE_NAME, SOURCE_SPACING)
    if key in _FORWARD_CACHE:
        return _FORWARD_CACHE[key]

    subjects_dir = str(get_subjects_dir())
    subject = FSAVERAGE_SUBJECT

    src = mne.setup_source_space(
        subject, spacing=SOURCE_SPACING, subjects_dir=subjects_dir, add_dist=False
    )

    bem_path = (
        Path(subjects_dir) / subject / "bem" / f"{subject}-5120-5120-5120-bem-sol.fif"
    )
    if bem_path.exists():
        bem_sol = mne.read_bem_solution(bem_path)
    else:
        bem = mne.make_bem_model(
            subject, conductivity=[0.3, 0.006, 0.3], subjects_dir=subjects_dir
        )
        bem_sol = mne.make_bem_solution(bem)

    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=src, bem=bem_sol, eeg=True, meg=False,
    )

    _FORWARD_CACHE[key] = (fwd, src)
    return fwd, src


def _vertex_positions(fwd):
    """Source vertex positions (m), ordered [left vertno, right vertno] like stc.data."""
    return np.concatenate([s["rr"][s["vertno"]] for s in fwd["src"]], axis=0)


def _select_roi_vertices(fwd, info, n_vertices=N_ROI_VERTICES):
    """Select source vertices nearest to the sensory-motor electrodes."""
    picks = mne.pick_channels(info["ch_names"], SENSORIMOTOR_ROI_CHANNELS)
    elec_pos = np.array([info["chs"][p]["loc"][:3] for p in picks])  # (n_elec, 3)

    vert_pos = _vertex_positions(fwd)
    dist = cdist(elec_pos, vert_pos)  # (n_elec, n_vert)
    per_elec = max(1, int(np.ceil(n_vertices / len(elec_pos))))
    nearest = np.argsort(dist, axis=1)[:, :per_elec]
    roi = np.unique(nearest.ravel())
    if len(roi) > n_vertices:
        roi = roi[:n_vertices]
    return roi


def get_roi_vertex_positions(info):
    """Return ``(roi_indices, roi_positions)`` for the sensory-motor ROI.

    ``roi_positions`` has shape ``(n_roi, 3)`` in head coordinates (meters); the
    first column is the left-right axis with ``X > 0`` in the right hemisphere.
    """
    fwd, _ = _get_forward_solution(info)
    roi = _select_roi_vertices(fwd, info)
    return roi, _vertex_positions(fwd)[roi]


def _erd_scalar(power, base_mask, task_mask):
    """Per-vertex scalar ERD from a band-power time series (n_vert, n_times)."""
    p_base = power[:, base_mask].mean(axis=1)
    p_task = power[:, task_mask].mean(axis=1)
    p_base = np.where(p_base == 0.0, np.nan, p_base)  # avoid division by zero
    return (p_task - p_base) / p_base


def erd_onset_latency(traj, sfreq=FS, min_duration=0.1):
    """Onset latency (s from cue) of a sustained ERD in a trial trajectory.

    A desynchronization counts once ``traj`` is below zero for at least
    ``min_duration`` consecutive samples. The trajectory is assumed to start at
    ``TASK_TMIN`` and to be sampled at ``sfreq``.

    Returns the latency in seconds from cue onset, or NaN if no sustained ERD
    is found.
    """
    traj = np.asarray(traj, dtype=float)
    below = traj < 0.0
    min_samples = int(round(min_duration * sfreq))
    run = 0
    for i, is_below in enumerate(below):
        if is_below:
            run += 1
            if run == min_samples:
                onset = i - min_samples + 1
                return TASK_TMIN + onset / sfreq
        else:
            run = 0
    return np.nan


def prepare_source_model(epochs):
    """Build (and cache) the eLORETA inverse operator and ROI for ``epochs``.

    Reused across calls to :func:`compute_erd_source` so that synthetic/repeated
    epoch sets do not rebuild the forward model, noise covariance, and inverse
    operator every time.
    """
    _ensure_montage(epochs)

    # eLORETA source modeling requires an EEG average reference (added as an SSP
    # projector); apply it before building the forward model so data and lead
    # field share the same reference.
    if not epochs.info["projs"]:
        epochs.set_eeg_reference("average", projection=True)

    key = (tuple(epochs.ch_names), MONTAGE_NAME, SOURCE_SPACING)
    if key in _INVERSE_CACHE:
        return _INVERSE_CACHE[key]

    fwd, _ = _get_forward_solution(epochs.info)
    noise_cov = mne.compute_covariance(
        epochs, tmin=BASELINE_TMIN, tmax=BASELINE_TMAX, method="shrunk"
    )
    inv = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
    )
    roi = _select_roi_vertices(fwd, epochs.info)

    model = {"inv": inv, "roi": roi}
    _INVERSE_CACHE[key] = model
    return model


def compute_erd_source(epochs, model=None):
    """Compute per-trial mu/beta ERD features and the mu ERD time trajectory.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG-only epochs spanning at least ``[BASELINE_TMIN, TASK_TMAX]`` and
        band-pass filtered (see ``data_loader.load_epochs``).
    model : dict, optional
        Precomputed source model from :func:`prepare_source_model`; built when
        None.

    Returns
    -------
    erd_mu : ndarray, shape (n_epochs, n_roi_vertices)
    erd_beta : ndarray, shape (n_epochs, n_roi_vertices)
    erd_traj : ndarray, shape (n_epochs, n_task_times)
        ROI-averaged mu-band ERD trajectory over the task window (``TASK_TMIN``
        to ``TASK_TMAX``, sampled at ``FS``; time axis is
        ``TASK_TMIN + arange(n_task_times) / FS``).

    Notes
    -----
    Adds the ``colin27_1020`` template montage and an EEG average-reference
    projector to ``epochs`` in place when they are absent.
    """
    model = prepare_source_model(epochs) if model is None else model
    inv = model["inv"]
    roi = model["roi"]

    times = epochs.times
    base_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
    task_mask = (times >= TASK_TMIN) & (times < TASK_TMAX)
    n_task = int(task_mask.sum())

    n_epochs = len(epochs)
    erd_mu = np.empty((n_epochs, len(roi)))
    erd_beta = np.empty((n_epochs, len(roi)))
    erd_traj = np.empty((n_epochs, n_task))

    # Stream one source estimate at a time to bound memory for large epoch counts.
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / SNR**2, method="eLORETA", pick_ori="normal",
        return_generator=True, verbose=False,
    )
    for i, stc in enumerate(stcs):
        data = stc.data[roi, :]  # (n_roi, n_times)
        mu_pow = filter_data(data, FS, MU_BAND[0], MU_BAND[1], verbose=False) ** 2
        beta_pow = filter_data(data, FS, BETA_BAND[0], BETA_BAND[1], verbose=False) ** 2

        erd_mu[i] = _erd_scalar(mu_pow, base_mask, task_mask)
        erd_beta[i] = _erd_scalar(beta_pow, base_mask, task_mask)

        roi_mu_power = mu_pow.mean(axis=0)  # ROI-averaged mu power (n_times,)
        base_power = roi_mu_power[base_mask].mean()
        base_power = np.nan if base_power == 0.0 else base_power
        erd_traj[i] = (roi_mu_power[task_mask] - base_power) / base_power

    return erd_mu, erd_beta, erd_traj
