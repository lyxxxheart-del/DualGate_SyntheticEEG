"""Shared experimental core for the covariance-gate studies.

This module is the single source of truth for the multi-subject pipeline used by
``run_all_subjects.py``. It extracts the logic previously inlined in
``test_step5.py`` so that a subject can be processed with one call.

Conditions (``CONDITIONS``):

* ``base``   -- real training data only (no synthetic).
* ``unf``    -- real + ALL synthetic (unfiltered).
* ``fcov``   -- real + synthetic passing Gate 1 (covariance plausibility).
* ``fdual``  -- real + synthetic passing Gate 1 AND Gate 2 (the proposed gate).
* ``noise``  -- real + Gaussian-noise augmentation (SNR=10 dB), one copy per
                real trial (the classic "double the training set" baseline).
* ``slide``  -- real + sliding-window (circular time-shift) augmentation, one
                copy per real trial.

The final gate operates on sensor-space covariance only -- no source modelling
(eLORETA) is required, because CSP consumes ``x @ x.T``. Consequently there is
no per-subject source-feature cache to maintain; the only repeated cost is
loading/band-pass-filtering each subject's epochs, which ``run_all_subjects.py``
caches as raw arrays.
"""

import numpy as np

from .config import EVENT_ID
from .cov_filter import (
    calibrate_covariance_thresholds,
    calibrate_discriminative_threshold,
    cov_reject_sampling,
    disc_reject_sampling,
    extract_covariance_features,
    fit_discriminative_direction,
)
from .downstream import (
    accuracy,
    fit_csp_lda,
    gaussian_noise_augment,
    predict_csp_lda,
    sliding_window_augment,
)

LEFT_ID = int(EVENT_ID["left_hand"])
RIGHT_ID = int(EVENT_ID["right_hand"])

N_TEST_PER_CLASS = 18       # held-out test set per class (36 total)
N_SYN_PER_CLASS = 72        # 144 defective synthetic trials total

CONDITIONS = ["base", "unf", "fcov", "fdual", "noise", "slide"]

MODE_NAMES = {1: "class mixing", 2: "covariance noise", 3: "temporal jitter"}


# ---------------------------------------------------------------------------
# Index / split helpers
# ---------------------------------------------------------------------------

def sample_indices(labels, k_per_class, rng, replace=False):
    """Draw ``k_per_class`` indices from each class using ``rng``."""
    classes = np.unique(labels)
    idx = [rng.choice(np.where(labels == c)[0], size=k_per_class, replace=replace)
           for c in classes]
    return np.concatenate(idx)


def split_train_test(labels, n_test_per_class=N_TEST_PER_CLASS, seed=42):
    """Stratified hold-out: return ``(pool_idx, test_idx)``.

    ``pool_idx`` is used to calibrate the gates and to draw the training sets;
    ``test_idx`` is fixed and never sees synthetic data.
    """
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        test_idx.extend(c_idx[:n_test_per_class].tolist())
    test_idx = np.array(test_idx)
    test_set = set(test_idx.tolist())
    pool_idx = np.array([i for i in range(len(labels)) if i not in test_set])
    return pool_idx, test_idx


# ---------------------------------------------------------------------------
# Defect-mode synthetic generators (each returns 144 balanced "bad" trials)
# ---------------------------------------------------------------------------

def gen_class_mixing(real_data, real_lab, rng):
    left = real_data[real_lab == LEFT_ID]
    right = real_data[real_lab == RIGHT_ID]
    n = 2 * N_SYN_PER_CLASS
    l = rng.integers(0, len(left), size=n)
    r = rng.integers(0, len(right), size=n)
    syn = 0.5 * (left[l] + right[r])
    lab = np.concatenate([np.full(N_SYN_PER_CLASS, LEFT_ID),
                          np.full(N_SYN_PER_CLASS, RIGHT_ID)])
    return syn, lab


def gen_covariance_noise(real_data, real_lab, rng):
    idx = sample_indices(real_lab, N_SYN_PER_CLASS, rng, replace=True)
    src = real_data[idx]
    std = src.std(axis=(1, 2), keepdims=True)  # 0 dB: noise std == signal std
    syn = src + std * rng.standard_normal(src.shape)
    return syn, real_lab[idx]


def gen_temporal_jitter(real_data, real_lab, rng, max_shift=50):
    idx = sample_indices(real_lab, N_SYN_PER_CLASS, rng, replace=True)
    src = real_data[idx]
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(src))
    syn = np.empty_like(src)
    for i, s in enumerate(shifts):
        syn[i] = np.roll(src[i], s, axis=-1)
    return syn, real_lab[idx]


GENERATORS = {
    1: gen_class_mixing,
    2: gen_covariance_noise,
    3: gen_temporal_jitter,
}


# ---------------------------------------------------------------------------
# Traditional augmentation baselines (one augmented copy per real trial)
# ---------------------------------------------------------------------------

def gen_gaussian_noise(real_data, real_lab, rng, snr_db=10.0):
    """Gaussian-noise augmentation: returns ``(aug_data, labels)``."""
    return gaussian_noise_augment(real_data, real_lab, rng, snr_db=snr_db)


def gen_sliding_window(real_data, real_lab, rng, max_shift=50):
    """Sliding-window (circular shift) augmentation."""
    return sliding_window_augment(real_data, real_lab, rng, max_shift=max_shift)


# ---------------------------------------------------------------------------
# Calibration / evaluation
# ---------------------------------------------------------------------------

def calibrate_gates(pool_data, pool_labels, normalize="trace", verbose=True):
    """Calibrate both gates on the train pool; return ``(cov_thr, w, disc_thr)``."""
    real_cov = extract_covariance_features(pool_data, normalize=normalize)
    cov_thr = calibrate_covariance_thresholds(real_cov, pool_labels, verbose=verbose)
    w = fit_discriminative_direction(real_cov, pool_labels)
    disc_thr = calibrate_discriminative_threshold(real_cov, pool_labels, w,
                                                  verbose=verbose)
    return cov_thr, w, disc_thr


def evaluate(train_data, train_labels, test_data, test_labels):
    """Fit CSP+LDA on the train set and return held-out accuracy."""
    model = fit_csp_lda(train_data, train_labels)
    pred = predict_csp_lda(model, test_data)
    return accuracy(test_labels, pred)


def _cat(*arrays):
    return np.concatenate(arrays)


# ---------------------------------------------------------------------------
# Per-subject run
# ---------------------------------------------------------------------------

def run_subject(data, labels, seeds=10, modes=(1, 2, 3),
                proportions=(0.25, 0.5, 1.0), normalize="trace", verbose=True):
    """Run the full experiment for one subject's trial array.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_times)
    labels : ndarray, shape (n_trials,)
        Event ids (1=left, 2=right).

    Returns
    -------
    dict with ``"means"`` (per-subject mean accuracy) and ``"per_seed"`` (the
    raw per-seed accuracy arrays), each keyed ``[mode][proportion][condition]``.
    """
    data = np.asarray(data, float)
    labels = np.asarray(labels)

    pool_idx, test_idx = split_train_test(labels, N_TEST_PER_CLASS, seed=42)
    pool_data = data[pool_idx]
    pool_labels = labels[pool_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]

    cov_thr, w, disc_thr = calibrate_gates(pool_data, pool_labels,
                                           normalize=normalize, verbose=verbose)

    classes = np.unique(pool_labels)
    n_per_class = int((pool_labels == classes[0]).sum())

    means = {m: {p: {c: float("nan") for c in CONDITIONS}
                 for p in proportions} for m in modes}
    per_seed = {m: {p: {c: [] for c in CONDITIONS}
                    for p in proportions} for m in modes}

    # Real-training-set sampling is mode-independent, so ``base``/``noise``/
    # ``slide`` are computed once per (proportion, seed) and shared across modes.
    shared = {}

    for p in proportions:
        k_real = int(round(p * n_per_class))
        for seed in range(seeds):
            real_rng = np.random.default_rng(1000 * int(p * 100) + seed)
            real_idx = sample_indices(pool_labels, k_real, real_rng, replace=False)
            real_data = pool_data[real_idx]
            real_lab = pool_labels[real_idx]

            acc_base = evaluate(real_data, real_lab, test_data, test_labels)

            noise_data, noise_lab = gen_gaussian_noise(real_data, real_lab, real_rng)
            slide_data, slide_lab = gen_sliding_window(real_data, real_lab, real_rng)
            acc_noise = evaluate(_cat(real_data, noise_data),
                                 _cat(real_lab, noise_lab),
                                 test_data, test_labels)
            acc_slide = evaluate(_cat(real_data, slide_data),
                                 _cat(real_lab, slide_lab),
                                 test_data, test_labels)

            shared[(p, seed)] = {"base": acc_base, "noise": acc_noise,
                                 "slide": acc_slide}

    for mode in modes:
        gen = GENERATORS[mode]
        for p in proportions:
            k_real = int(round(p * n_per_class))
            for seed in range(seeds):
                real_rng = np.random.default_rng(1000 * int(p * 100) + seed)
                gen_rng = np.random.default_rng(1000 * int(p * 100) + 100 * mode + seed)
                real_idx = sample_indices(pool_labels, k_real, real_rng, replace=False)
                real_data = pool_data[real_idx]
                real_lab = pool_labels[real_idx]

                syn_data, syn_lab = gen(real_data, real_lab, gen_rng)
                syn_cov = extract_covariance_features(syn_data, normalize=normalize)
                mask_cov = cov_reject_sampling(syn_cov, syn_lab, cov_thr)
                mask_disc = disc_reject_sampling(syn_cov, syn_lab, w, disc_thr)
                mask_dual = mask_cov & mask_disc

                acc_unf = evaluate(_cat(real_data, syn_data),
                                   _cat(real_lab, syn_lab),
                                   test_data, test_labels)
                acc_fcov = evaluate(_cat(real_data, syn_data[mask_cov]),
                                    _cat(real_lab, syn_lab[mask_cov]),
                                    test_data, test_labels)
                acc_fdual = evaluate(_cat(real_data, syn_data[mask_dual]),
                                     _cat(real_lab, syn_lab[mask_dual]),
                                     test_data, test_labels)

                seed_accs = {
                    "base": shared[(p, seed)]["base"],
                    "unf": acc_unf,
                    "fcov": acc_fcov,
                    "fdual": acc_fdual,
                    "noise": shared[(p, seed)]["noise"],
                    "slide": shared[(p, seed)]["slide"],
                }
                for c in CONDITIONS:
                    per_seed[mode][p][c].append(seed_accs[c])

    for mode in modes:
        for p in proportions:
            for c in CONDITIONS:
                means[mode][p][c] = float(np.mean(per_seed[mode][p][c]))

    return {"means": means, "per_seed": per_seed}
