"""Test step 5: covariance-quality gate aligned with the CSP downstream task.

Reuses the three defect models from ``test_step4`` (class mixing, covariance
noise, temporal jitter) but filters synthetic trials with sensor-space gates on
the per-trial covariance -- the quantity CSP+LDA is built on -- so gate and
classifier share one representational space.

Conditions compared:

* Baseline      -- real only
* Unfiltered    -- real + all synthetic
* Filtered-Cov  -- real + synthetic passing the covariance plausibility gate
* Filtered-Dual -- real + synthetic passing BOTH the plausibility gate and the
                   discriminative-direction (LDA) gate

Train/test split is the fixed stratified hold-out of A01T (18 trials/class held
out), identical to ``test_step4``; the seed formula matches ``test_step4`` so
the generated synthetic sets are reproducible across both scripts.

Run:  py test_step5.py [--seeds 10] [--modes 1,2,3] [--proportions 0.25,0.5,1.0]
                       [--normalize trace|raw|corr]
"""

import argparse

import numpy as np
from scipy.stats import ttest_rel

from src.cov_filter import (
    calibrate_covariance_thresholds,
    calibrate_discriminative_threshold,
    cov_reject_sampling,
    disc_reject_sampling,
    extract_covariance_features,
    fit_discriminative_direction,
)
from src.downstream import accuracy, fit_csp_lda, predict_csp_lda
from src.config import EVENT_ID

LEFT_ID = int(EVENT_ID["left_hand"])
RIGHT_ID = int(EVENT_ID["right_hand"])

N_TEST_PER_CLASS = 18       # held-out test set per class (36 total)
N_SYN_PER_CLASS = 72        # 144 synthetic trials total

MODE_NAMES = {1: "class mixing", 2: "covariance noise", 3: "temporal jitter"}


def _sample_indices(labels, k_per_class, rng, replace=False):
    classes = np.unique(labels)
    idx = [rng.choice(np.where(labels == c)[0], size=k_per_class, replace=replace)
           for c in classes]
    return np.concatenate(idx)


# --- synthetic generators (each returns 144 balanced "bad" trials) -----------

def _gen_class_mixing(real_data, real_lab, rng):
    left = real_data[real_lab == LEFT_ID]
    right = real_data[real_lab == RIGHT_ID]
    n = 2 * N_SYN_PER_CLASS
    l = rng.integers(0, len(left), size=n)
    r = rng.integers(0, len(right), size=n)
    syn = 0.5 * (left[l] + right[r])
    lab = np.concatenate([np.full(N_SYN_PER_CLASS, LEFT_ID),
                          np.full(N_SYN_PER_CLASS, RIGHT_ID)])
    return syn, lab


def _gen_covariance_noise(real_data, real_lab, rng):
    idx = _sample_indices(real_lab, N_SYN_PER_CLASS, rng, replace=True)
    src = real_data[idx]
    std = src.std(axis=(1, 2), keepdims=True)  # 0 dB: noise std == signal std
    syn = src + std * rng.standard_normal(src.shape)
    return syn, real_lab[idx]


def _gen_temporal_jitter(real_data, real_lab, rng, max_shift=50):
    idx = _sample_indices(real_lab, N_SYN_PER_CLASS, rng, replace=True)
    src = real_data[idx]
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(src))
    syn = np.empty_like(src)
    for i, s in enumerate(shifts):
        syn[i] = np.roll(src[i], s, axis=-1)
    return syn, real_lab[idx]


GENERATORS = {
    1: _gen_class_mixing,
    2: _gen_covariance_noise,
    3: _gen_temporal_jitter,
}


def _evaluate(train_data, train_labels, test_data, test_labels):
    model = fit_csp_lda(train_data, train_labels)
    pred = predict_csp_lda(model, test_data)
    return accuracy(test_labels, pred)


def main(args):
    from src.data_loader import load_epochs

    print("Loading A01T ...")
    epochs = load_epochs("A01", "T")
    data = epochs.get_data()
    labels = epochs.events[:, 2]

    classes = np.unique(labels)
    split_rng = np.random.default_rng(42)
    test_idx = []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        split_rng.shuffle(c_idx)
        test_idx.extend(c_idx[:N_TEST_PER_CLASS].tolist())
    test_idx = np.array(test_idx)
    test_set = set(test_idx.tolist())
    pool_idx = np.array([i for i in range(len(labels)) if i not in test_set])
    pool_labels = labels[pool_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]
    print(f"Split: {len(pool_idx)} train-pool / {len(test_idx)} test trials")

    pool_epochs = epochs[pool_idx]
    print(f"Calibrating gates (train pool, normalize={args.normalize}) ...")
    real_cov = extract_covariance_features(pool_epochs, normalize=args.normalize)
    cov_thresholds = calibrate_covariance_thresholds(real_cov, pool_labels)
    w = fit_discriminative_direction(real_cov, pool_labels)
    disc_thresholds = calibrate_discriminative_threshold(real_cov, pool_labels, w)

    n_per_class = int((pool_labels == classes[0]).sum())

    for mode in args.modes:
        gen = GENERATORS[mode]
        results = {p: {"base": [], "unf": [], "fcov": [], "fdual": [],
                       "pass_cov": [], "pass_disc": [], "pass_dual": []}
                   for p in args.proportions}

        for p in args.proportions:
            k_real = int(round(p * n_per_class))
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * int(p * 100) + 100 * mode + seed)

                real_idx = _sample_indices(pool_labels, k_real, rng, replace=False)
                real_data = data[pool_idx[real_idx]]
                real_lab = labels[pool_idx[real_idx]]

                syn_data, syn_lab = gen(real_data, real_lab, rng)
                syn_cov = extract_covariance_features(syn_data, normalize=args.normalize)

                mask_cov = cov_reject_sampling(syn_cov, syn_lab, cov_thresholds)
                mask_disc = disc_reject_sampling(syn_cov, syn_lab, w, disc_thresholds)
                mask_dual = mask_cov & mask_disc

                acc_base = _evaluate(real_data, real_lab, test_data, test_labels)
                acc_unf = _evaluate(np.concatenate([real_data, syn_data]),
                                    np.concatenate([real_lab, syn_lab]),
                                    test_data, test_labels)
                acc_fcov = _evaluate(np.concatenate([real_data, syn_data[mask_cov]]),
                                     np.concatenate([real_lab, syn_lab[mask_cov]]),
                                     test_data, test_labels)
                acc_fdual = _evaluate(np.concatenate([real_data, syn_data[mask_dual]]),
                                      np.concatenate([real_lab, syn_lab[mask_dual]]),
                                      test_data, test_labels)

                results[p]["base"].append(acc_base)
                results[p]["unf"].append(acc_unf)
                results[p]["fcov"].append(acc_fcov)
                results[p]["fdual"].append(acc_fdual)
                results[p]["pass_cov"].append(int(mask_cov.sum()))
                results[p]["pass_disc"].append(int(mask_disc.sum()))
                results[p]["pass_dual"].append(int(mask_dual.sum()))

        print(f"\n=== Mode {mode}: {MODE_NAMES[mode]} ===")
        hdr = (f"{'real%':>6} | {'Baseline':>18} | {'Unfiltered':>18} | "
               f"{'F-Cov':>18} | {'F-Dual':>18} | {'cov/disc/dual pass':>19}")
        print(hdr)
        for p in args.proportions:
            b = np.array(results[p]["base"])
            u = np.array(results[p]["unf"])
            fc = np.array(results[p]["fcov"])
            fd = np.array(results[p]["fdual"])
            pc = np.mean(results[p]["pass_cov"])
            pd = np.mean(results[p]["pass_disc"])
            pz = np.mean(results[p]["pass_dual"])
            print(f"{int(p * 100):>5}% | {b.mean():>9.3f}+-{b.std():.3f} | "
                  f"{u.mean():>9.3f}+-{u.std():.3f} | {fc.mean():>9.3f}+-{fc.std():.3f} | "
                  f"{fd.mean():>9.3f}+-{fd.std():.3f} | {pc:>6.1f}/{pd:<6.1f}/{pz:<6.1f}")
            t_cov, p_cov = ttest_rel(fc, u)
            t_dual, p_dual = ttest_rel(fd, u)
            print(f"       t(F-Cov  vs Unf)={t_cov:+.2f} p={p_cov:.3f}  |  "
                  f"t(F-Dual vs Unf)={t_dual:+.2f} p={p_dual:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--modes", type=str, default="1,2,3")
    ap.add_argument("--proportions", type=str, default="0.25,0.5,1.0")
    ap.add_argument("--normalize", type=str, default="trace",
                    choices=["raw", "trace", "corr"])
    args = ap.parse_args()
    args.modes = [int(x) for x in args.modes.split(",")]
    args.proportions = [float(x) for x in args.proportions.split(",")]
    main(args)
