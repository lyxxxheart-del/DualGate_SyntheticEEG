"""Sensor-space covariance-quality gate aligned with the CSP downstream task.

The ERD gate measured *temporal* contrast (band power drop between baseline and
task windows), which the CSP+LDA classifier never sees. This module instead gates
on the per-trial spatial covariance matrix -- the exact quantity CSP consumes
(``downstream._csp_filters`` builds class covariances as ``x @ x.T``) -- so the
gate and the downstream classifier operate in the same representational space.

Two complementary gates live here:

* **Plausibility gate** (:func:`cov_reject_sampling`) -- cosine similarity of a
  trial's covariance upper triangle to its class template. Rejects trials whose
  covariance is not a physically plausible EEG covariance (e.g. white-noise
  corrupted), but cannot reject a class-ambiguous blend, which is still a valid
  covariance.

* **Discriminative gate** (:func:`disc_reject_sampling`) -- the LDA direction
  ``w`` estimated on the covariance features separates left/right; a trial's
  projection ``features @ w`` must fall inside its own class's score window. This
  rejects class-ambiguous blends whose projection lands between the classes.

The covariance is reduced to the upper triangle of the symmetric 22x22 matrix
(``22 * 23 / 2 = 253`` dimensions).
"""

import numpy as np

from .config import EVENT_ID
from .downstream import _lda_fit


def _upper_triangle(cov):
    """Flatten the upper triangle (including diagonal) of ``cov`` per trial."""
    tri = np.triu_indices(cov.shape[1])
    return cov[:, tri[0], tri[1]]


def _covariance(data, normalize):
    """Per-trial sensor-space covariance features.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_times)
    normalize : {"raw", "trace", "corr"}
        * ``"raw"``  -- upper triangle of the raw energy covariance ``x @ x.T``.
        * ``"trace"`` -- trace-normalized covariance (scale-invariant), the first
          affine-invariance step of the Riemannian covariance alignment the task
          references. White noise attenuates the off-diagonal spatial structure
          relative to the diagonal, so it shifts the vector's direction.
        * ``"corr"``  -- the correlation matrix (unit diagonal), the purest
          measure of cross-channel spatial structure.

    Returns
    -------
    features : ndarray, shape (n_trials, n_features)
    """
    data = np.asarray(data, float)
    # Uncentered energy covariance, matching downstream._csp_filters exactly.
    cov = data @ data.swapaxes(1, 2)  # (n_trials, n_ch, n_ch)

    if normalize == "trace":
        tr = np.einsum("nii->n", cov)
        cov = cov / np.where(tr == 0.0, 1.0, tr)[:, None, None]
    elif normalize == "corr":
        d = np.sqrt(np.einsum("nii->ni", cov))
        cov = cov / (d[:, :, None] * d[:, None, :] + 1e-12)
    elif normalize != "raw":
        raise ValueError(f"normalize must be 'raw', 'trace', or 'corr', got {normalize!r}")

    return _upper_triangle(cov)


def extract_covariance_features(epochs, normalize="trace"):
    """Extract per-trial covariance features.

    Parameters
    ----------
    epochs : mne.Epochs or ndarray
        Either an ``mne.Epochs`` object or a raw array of shape
        ``(n_trials, n_channels, n_times)``.
    normalize : {"raw", "trace", "corr"}
        See :func:`_covariance`.

    Returns
    -------
    features : ndarray, shape (n_trials, 253)
    """
    data = epochs.get_data() if hasattr(epochs, "get_data") else np.asarray(epochs, float)
    return _covariance(data, normalize)


def _cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# Gate 1: covariance plausibility (cosine similarity to class template)
# ---------------------------------------------------------------------------

def calibrate_covariance_thresholds(real_cov_features, labels, verbose=True):
    """Calibrate per-class covariance-similarity acceptance windows.

    Returns ``{"T_A": {label: (lo, hi)}, "templates": {label: template}}`` where
    each window is the 5th/95th percentile of the cosine similarity of real
    trials to their own class template (the class-mean feature).
    """
    features = np.asarray(real_cov_features, float)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    templates = {int(c): features[labels == c].mean(axis=0) for c in classes}

    T_A = {}
    stats = {}
    for c in classes:
        tpl = templates[int(c)]
        sims = np.array([_cosine(features[i], tpl)
                         for i in np.where(labels == c)[0]])
        T_A[int(c)] = (float(np.percentile(sims, 5)), float(np.percentile(sims, 95)))
        stats[int(c)] = (float(sims.mean()), float(sims.std()))

    thresholds = {"T_A": T_A, "templates": templates}

    if verbose:
        left_id = int(EVENT_ID["left_hand"])
        print("Covariance calibration (real data):")
        for c in sorted(T_A):
            name = "left" if c == left_id else "right"
            mu, sd = stats[c]
            print(f"  T_A_{name} (class {c}) = [{T_A[c][0]:.3f}, {T_A[c][1]:.3f}]  "
                  f"(within-class cos sim {mu:.3f}+-{sd:.3f})")

    return thresholds


def cov_reject_sampling(syn_cov_features, labels, thresholds):
    """Keep synthetic trials whose covariance matches their class template.

    Returns a boolean mask (True = keep) over the synthetic trials.
    """
    features = np.asarray(syn_cov_features, float)
    labels = np.asarray(labels)
    templates = thresholds["templates"]
    T_A = thresholds["T_A"]

    mask = np.empty(len(features), dtype=bool)
    for i in range(len(features)):
        c = int(labels[i])
        lo, hi = T_A[c]
        s = _cosine(features[i], templates[c])
        mask[i] = (not np.isnan(s)) and (lo <= s <= hi)
    return mask


# ---------------------------------------------------------------------------
# Gate 2: discriminative direction (LDA projection on the covariance features)
# ---------------------------------------------------------------------------

def fit_discriminative_direction(real_cov_features, labels, reg=0.01):
    """Fit Fisher LDA on the covariance features; return the weight vector ``w``.

    ``w`` points from the first class (left-hand, id 1) toward the second
    (right-hand, id 2), so ``features @ w`` is negative for left trials and
    positive for right trials.
    """
    features = np.asarray(real_cov_features, float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    w, _ = _lda_fit(features, labels, classes, reg=reg)
    return w


def calibrate_discriminative_threshold(real_cov_features, labels, w, verbose=True):
    """Calibrate per-class windows of the LDA discriminative score.

    Returns ``{"T_D": {label: (lo, hi)}}`` where each window is the 5th/95th
    percentile of ``features @ w`` for that class's real trials.
    """
    features = np.asarray(real_cov_features, float)
    labels = np.asarray(labels)
    w = np.asarray(w, float)
    classes = np.unique(labels)

    scores = features @ w
    T_D = {}
    stats = {}
    for c in classes:
        s = scores[labels == c]
        T_D[int(c)] = (float(np.percentile(s, 5)), float(np.percentile(s, 95)))
        stats[int(c)] = (float(s.mean()), float(s.std()))

    thresholds = {"T_D": T_D}

    if verbose:
        left_id = int(EVENT_ID["left_hand"])
        print("Discriminative calibration (real data, LDA score):")
        for c in sorted(T_D):
            name = "left" if c == left_id else "right"
            mu, sd = stats[c]
            print(f"  T_D_{name} (class {c}) = [{T_D[c][0]:.3f}, {T_D[c][1]:.3f}]  "
                  f"(score {mu:+.3f}+-{sd:.3f})")

    return thresholds


def disc_reject_sampling(syn_cov_features, labels, w, thresholds):
    """Keep synthetic trials whose discriminative score matches their class.

    A trial is kept when ``features @ w`` falls inside its own class's ``[lo, hi]``
    window, which enforces both the correct sign and a plausible magnitude.

    Returns a boolean mask (True = keep) over the synthetic trials.
    """
    features = np.asarray(syn_cov_features, float)
    labels = np.asarray(labels)
    w = np.asarray(w, float)
    scores = features @ w
    T_D = thresholds["T_D"]

    mask = np.empty(len(features), dtype=bool)
    for i in range(len(features)):
        c = int(labels[i])
        lo, hi = T_D[c]
        mask[i] = (not np.isnan(scores[i])) and (lo <= scores[i] <= hi)
    return mask
