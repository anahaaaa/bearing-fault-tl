# ==================================================
# Z-Score Normalization Utilities
# Works for:
# - one 1D signal
# - batch of windows (N,1024)
# - segmented domain dictionaries
# ==================================================

import numpy as np


# --------------------------------------------------
# 1. Normalize one 1D signal/window
# --------------------------------------------------
def zscore_1d(signal, eps=1e-8):
    """
    Z-score normalize one 1D array.

    Output:
        mean ~= 0
        std  ~= 1
    """

    signal = np.asarray(signal, dtype=np.float32)

    mean = np.mean(signal)
    std = np.std(signal)

    if std < eps:
        std = eps

    return (signal - mean) / std


# --------------------------------------------------
# 2. Normalize batch of windows independently
# Input shape: (N, window_size)
# Each row normalized separately
# --------------------------------------------------
def zscore_batch(X, eps=1e-8):
    """
    Normalize each sample independently.

    Example:
        X shape = (5000,1024)
    """

    X = np.asarray(X, dtype=np.float32)

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    std[std < eps] = eps

    return (X - mean) / std


# --------------------------------------------------
# 3. Normalize segmented domain dictionary
# Input:
# domains[load] = [
#   {"window":..., "label":...}
# ]
# --------------------------------------------------
def zscore_domains(segmented_domains, eps=1e-8):
    """
    Applies per-window z-score normalization.
    """

    normalized = {}

    for load in segmented_domains:

        normalized[load] = []

        for sample in segmented_domains[load]:

            norm_window = zscore_1d(sample["window"], eps=eps)

            new_sample = sample.copy()
            new_sample["window"] = norm_window

            normalized[load].append(new_sample)

    return normalized


# --------------------------------------------------
# 4. Quick verification helper
# --------------------------------------------------
def check_window_stats(window):
    """
    Prints mean/std of one window
    """

    print("Mean:", np.mean(window))
    print("Std :", np.std(window))
