"""CSP + LDA classification pipeline for MI-EEG (time-domain epochs)."""

import numpy as np
from scipy.linalg import eigh

N_CSP_PAIRS = 3  # spatial filters per class -> 2 * N_CSP_PAIRS features


def _csp_filters(X, y, n_pairs=N_CSP_PAIRS, reg=0.01):
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("CSP requires exactly two classes")

    covs = {}
    for c in classes:
        Xc = X[y == c]
        C = np.mean([x @ x.T for x in Xc], axis=0)
        C = C + reg * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        covs[int(c)] = C

    c0, c1 = int(classes[0]), int(classes[1])
    eigvals, eigvecs = eigh(covs[c0], covs[c0] + covs[c1])
    # ascending eigenvalues: smallest -> class c1, largest -> class c0.
    W = np.concatenate([eigvecs[:, :n_pairs], eigvecs[:, -n_pairs:]], axis=1)
    return W


def _csp_features(X, W):
    feats = []
    for x in X:
        proj = W.T @ x
        v = np.var(proj, axis=1)
        feats.append(np.log(v / (v.sum() + 1e-12)))
    return np.array(feats)


def _lda_fit(F, y, classes, reg=0.01):
    F = np.asarray(F, float)
    y = np.asarray(y)
    m0 = F[y == classes[0]].mean(axis=0)
    m1 = F[y == classes[1]].mean(axis=0)
    S = (np.cov(F[y == classes[0]].T) + np.cov(F[y == classes[1]].T)) / 2.0
    S = S + reg * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w = np.linalg.solve(S, m1 - m0)
    b = -0.5 * (m1 + m0) @ w
    return w, b


def fit_csp_lda(train_data, train_labels):
    """Fit CSP + LDA.

    Parameters
    ----------
    train_data : ndarray, shape (n_trials, n_channels, n_times)
    train_labels : ndarray, shape (n_trials,)
    """
    classes = np.unique(np.asarray(train_labels))
    W = _csp_filters(train_data, train_labels)
    F = _csp_features(train_data, W)
    w, b = _lda_fit(F, train_labels, classes)
    return {"W": W, "lda_w": w, "lda_b": b, "classes": classes}


def predict_csp_lda(model, data):
    """Predict class labels for time-domain epochs."""
    F = _csp_features(np.asarray(data, float), model["W"])
    scores = F @ model["lda_w"] + model["lda_b"]
    return np.where(scores > 0, model["classes"][1], model["classes"][0])


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def stratified_kfold_accuracy(data, labels, n_splits=5, seed=0):
    """Stratified K-fold cross-validated accuracy of the CSP+LDA pipeline.

    Returns ``(mean_accuracy, fold_accuracies)``.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        for f, part in enumerate(np.array_split(c_idx, n_splits)):
            folds[f].extend(part.tolist())

    accs = []
    for f in range(n_splits):
        test_idx = np.array(folds[f])
        train_idx = np.array([i for g in range(n_splits) if g != f for i in folds[g]])
        model = fit_csp_lda(data[train_idx], labels[train_idx])
        pred = predict_csp_lda(model, data[test_idx])
        accs.append(accuracy(labels[test_idx], pred))
    return float(np.mean(accs)), accs


def gaussian_noise_augment(data, labels, rng, snr_db=10.0):
    """Return a white-noise-augmented copy of ``data`` (one copy per trial).

    Adds zero-mean Gaussian noise at ``snr_db`` (default 10 dB) to each trial
    independently, keeping the labels unchanged. ``rng`` is a
    ``numpy.random.Generator``. Returns ``(augmented_data, labels)``.
    """
    data = np.asarray(data, float)
    labels = np.asarray(labels)
    p_sig = (data ** 2).mean(axis=(1, 2), keepdims=True)
    noise = rng.standard_normal(data.shape) * np.sqrt(p_sig / 10 ** (snr_db / 10.0))
    return data + noise, labels


def sliding_window_augment(data, labels, rng, max_shift=50):
    """Return a time-shifted copy of ``data`` (one copy per trial).

    Rolls each trial's time axis by a random integer offset in
    ``[-max_shift, +max_shift]`` samples (circular). This is a cheap
    sliding-window time augmentation that leaves the spatial covariance
    ``x @ x.T`` invariant. Returns ``(augmented_data, labels)``.
    """
    data = np.asarray(data, float)
    labels = np.asarray(labels)
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(data))
    out = np.empty_like(data)
    for i, s in enumerate(shifts):
        out[i] = np.roll(data[i], s, axis=-1)
    return out, labels
