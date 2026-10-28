"""Test step 2: source-space ERD on A01T (first 10 epochs).

Run from the project root:  py test_step2.py
"""

import time

import numpy as np

from src.data_loader import load_epochs
from src.pipeline import compute_erd_source

if __name__ == "__main__":
    print("Loading A01T ...")
    epochs = load_epochs(subject="A01", run="T")
    print("full epochs shape:", epochs.get_data().shape)

    epochs = epochs[:10]
    print("using first 10 epochs")

    t0 = time.time()
    erd_mu, erd_beta = compute_erd_source(epochs)
    dt = time.time() - t0

    print("erd_mu_features shape:", erd_mu.shape)
    print("erd_beta_features shape:", erd_beta.shape)
    print(f"elapsed: {dt:.2f} s")
    print("erd_mu finite fraction:", np.isfinite(erd_mu).mean())
    print("erd_beta finite fraction:", np.isfinite(erd_beta).mean())
