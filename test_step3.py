"""Test step 3: calibration + physiological rejection on A01T.

Run from the project root:  py test_step3.py
"""

import numpy as np

from src.data_loader import load_epochs
from src.pipeline import compute_erd_source
from src.calibration import calibrate_thresholds
from src.filter import physio_reject_sampling


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("Loading A01T and computing source-space ERD ...")
    epochs = load_epochs(subject="A01", run="T")
    erd_mu, erd_beta, erd_traj = compute_erd_source(epochs)
    labels = epochs.events[:, 2]

    print("erd_mu:", erd_mu.shape, "erd_beta:", erd_beta.shape,
          "erd_traj:", erd_traj.shape)

    templates, thresholds = calibrate_thresholds(erd_mu, erd_traj, labels)

    # Placeholder: simulate low-quality synthetic data by corrupting the first
    # 50 real trials with Gaussian noise (std = the real features' std).
    n_syn = 50
    syn_features = erd_mu[:n_syn] + rng.normal(0.0, np.std(erd_mu), erd_mu[:n_syn].shape)
    syn_traj = erd_traj[:n_syn] + rng.normal(0.0, np.std(erd_traj), erd_traj[:n_syn].shape)
    syn_labels = labels[:n_syn]

    mask = physio_reject_sampling(syn_features, syn_traj, syn_labels, templates, thresholds)

    print(f"\nSynthetic placeholder test: {int(mask.sum())}/{n_syn} trials passed the filter")
    print(f"rejection rate: {1.0 - mask.sum() / n_syn:.0%}")
