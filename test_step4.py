"""Test step 4: downstream CSP+LDA validation of the physiological filter.

Trains CSP+LDA on a low-real-data subset of A01T (optionally augmented with
synthetic trials), tests on a held-out portion of A01T, and compares four
conditions: Baseline, Unfiltered, Filtered-Source (source-space ERD gate) and
Filtered-Sensor (sensor-space C3/C4 ERD gate).

Three "bad synthetic" defect models are simulated separately:

  Mode 1  class mixing      -- 0.5*(left + right), class-boundary-ambiguous trials
  Mode 2  covariance noise  -- real trial + white Gaussian noise at 0 dB SNR
  Mode 3  temporal jitter   -- real trial circularly shifted by +-200 ms

The train/test split is a fixed stratified hold-out of A01T (18 trials/class
held out) because A01E encodes every cue as "783" and ships its labels in a
separate "true labels" .mat that is not part of ``BCICIV_2a_gdf.zip``.

Run:  py test_step4.py [--seeds 10] [--modes 1,2,3] [--proportions 0.25,0.5,1.0]
"""

import argparse

import numpy as np
from scipy.stats import ttest_rel
import mne

from src.data_loader import load_epochs
from src.pipeline import compute_erd_source, prepare_source_model
from src.calibration import calibrate_sensor_erd, calibrate_thresholds
from src.filter import physio_reject_sampling, sensor_erd_reject_sampling
from src.downstream import fit_csp_lda, predict_csp_lda, accuracy
from src.config import EVENT_ID, TMIN

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


def _to_epochs(data, info):
    events = np.column_stack([np.arange(len(data)), np.zeros(len(data), int),
                              np.ones(len(data), int)])
    return mne.EpochsArray(data, info, events=events, event_id={"x": 1},
                           tmin=TMIN, verbose=False)


def _evaluate(train_data, train_labels, test_data, test_labels):
    model = fit_csp_lda(train_data, train_labels)
    pred = predict_csp_lda(model, test_data)
    return accuracy(test_labels, pred)


def main(args):
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
    print("Calibrating source-space + sensor-space thresholds (train pool) ...")
    model = prepare_source_model(pool_epochs)
    erd_mu, _, erd_traj = compute_erd_source(pool_epochs, model=model)
    criterion_a, thresholds = calibrate_thresholds(erd_mu, erd_traj, pool_labels)
    sensor_thresholds = calibrate_sensor_erd(pool_epochs)

    n_per_class = int((pool_labels == classes[0]).sum())

    for mode in args.modes:
        gen = GENERATORS[mode]
        results = {p: {"base": [], "unf": [], "fsrc": [], "fsens": [],
                       "pass_src": [], "pass_sens": []}
                   for p in args.proportions}

        for p in args.proportions:
            k_real = int(round(p * n_per_class))
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * int(p * 100) + 100 * mode + seed)

                real_idx = _sample_indices(pool_labels, k_real, rng, replace=False)
                real_data = data[pool_idx[real_idx]]
                real_lab = labels[pool_idx[real_idx]]

                syn_data, syn_lab = gen(real_data, real_lab, rng)

                syn_mu, _, syn_traj = compute_erd_source(
                    _to_epochs(syn_data, pool_epochs.info), model=model
                )
                mask_src = physio_reject_sampling(syn_mu, syn_traj, syn_lab,
                                                  criterion_a, thresholds)
                mask_sens = sensor_erd_reject_sampling(syn_data, syn_lab,
                                                      sensor_thresholds)

                acc_base = _evaluate(real_data, real_lab, test_data, test_labels)
                acc_unf = _evaluate(np.concatenate([real_data, syn_data]),
                                    np.concatenate([real_lab, syn_lab]),
                                    test_data, test_labels)
                acc_fsrc = _evaluate(np.concatenate([real_data, syn_data[mask_src]]),
                                     np.concatenate([real_lab, syn_lab[mask_src]]),
                                     test_data, test_labels)
                acc_fsens = _evaluate(np.concatenate([real_data, syn_data[mask_sens]]),
                                      np.concatenate([real_lab, syn_lab[mask_sens]]),
                                      test_data, test_labels)

                results[p]["base"].append(acc_base)
                results[p]["unf"].append(acc_unf)
                results[p]["fsrc"].append(acc_fsrc)
                results[p]["fsens"].append(acc_fsens)
                results[p]["pass_src"].append(int(mask_src.sum()))
                results[p]["pass_sens"].append(int(mask_sens.sum()))

        print(f"\n=== Mode {mode}: {MODE_NAMES[mode]} ===")
        hdr = (f"{'real%':>6} | {'Baseline':>18} | {'Unfiltered':>18} | "
               f"{'F-Source':>18} | {'F-Sensor':>18} | {'src/sens pass':>14}")
        print(hdr)
        for p in args.proportions:
            b = np.array(results[p]["base"])
            u = np.array(results[p]["unf"])
            fs = np.array(results[p]["fsrc"])
            fn = np.array(results[p]["fsens"])
            ps = np.mean(results[p]["pass_src"])
            pn = np.mean(results[p]["pass_sens"])
            print(f"{int(p * 100):>5}% | {b.mean():>9.3f}+-{b.std():.3f} | "
                  f"{u.mean():>9.3f}+-{u.std():.3f} | {fs.mean():>9.3f}+-{fs.std():.3f} | "
                  f"{fn.mean():>9.3f}+-{fn.std():.3f} | {ps:>6.1f}/{pn:<6.1f}")
            t_src, p_src = ttest_rel(fs, u)
            t_sens, p_sens = ttest_rel(fn, u)
            print(f"       t(F-Source vs Unf)={t_src:+.2f} p={p_src:.3f}  |  "
                  f"t(F-Sensor vs Unf)={t_sens:+.2f} p={p_sens:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--modes", type=str, default="1,2,3")
    ap.add_argument("--proportions", type=str, default="0.25,0.5,1.0")
    args = ap.parse_args()
    args.modes = [int(x) for x in args.modes.split(",")]
    args.proportions = [float(x) for x in args.proportions.split(",")]
    main(args)
