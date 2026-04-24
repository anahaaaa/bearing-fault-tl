import numpy as np


# ----------------------------------------------------
# Segment one 1D signal into fixed windows
# 1024 samples, 50% overlap
# ----------------------------------------------------
def segment_signal(signal, window_size=1024, overlap=0.5):
    """
    Split a 1D signal into overlapping windows.

    Args:
        signal (np.ndarray): 1D input signal
        window_size (int): number of samples per window
        overlap (float): overlap ratio (0.5 = 50%)

    Returns:
        np.ndarray of shape (num_windows, window_size)
    """

    signal = np.asarray(signal, dtype=np.float32)

    step = int(window_size * (1 - overlap))   # 512 if overlap=0.5

    if step <= 0:
        raise ValueError("Overlap too high. Must be < 1.0")

    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    windows = []

    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end])

    return np.array(windows, dtype=np.float32)


# ----------------------------------------------------
# Segment loaded domains
#
# Input:
# domains[load] = [
#   {
#     "signal": np.ndarray,
#     "label": int,
#     "file": str,
#     "type": str
#   }
# ]
# ----------------------------------------------------
def segment_domains(domains,
                    window_size=1024,
                    overlap=0.5):
    """
    Convert long signals into training windows.

    Returns:
        segmented[load] = [
            {
              "window": np.ndarray (1024,),
              "label": int,
              "file": str,
              "type": str
            }
        ]
    """

    segmented = {}

    for load in domains:

        segmented[load] = []

        for sample in domains[load]:

            signal = sample["signal"]

            windows = segment_signal(
                signal=signal,
                window_size=window_size,
                overlap=overlap
            )

            for w in windows:
                segmented[load].append({
                    "window": w,
                    "label": sample["label"],
                    "file": sample["file"],
                    "type": sample.get("type", "unknown")
                })

    return segmented


# ----------------------------------------------------
# Convert one domain to X, y arrays
# ----------------------------------------------------
def domain_to_xy(segmented_domains, load_id):
    """
    Args:
        segmented_domains (dict)
        load_id (int): 0 / 1 / 2 / 3

    Returns:
        X -> (N,1024)
        y -> (N,)
    """

    samples = segmented_domains[load_id]

    X = np.array(
        [sample["window"] for sample in samples],
        dtype=np.float32
    )

    y = np.array(
        [sample["label"] for sample in samples],
        dtype=np.int64
    )

    return X, y


# ----------------------------------------------------
# Merge multiple domains
# Example:
# X,y = merge_domains(segmented,[0,1])
# ----------------------------------------------------
def merge_domains(segmented_domains, loads):
    """
    Args:
        segmented_domains (dict)
        loads (list): e.g. [0,1]

    Returns:
        X -> (N,1024)
        y -> (N,)
    """

    X_all = []
    y_all = []

    for load in loads:

        X, y = domain_to_xy(segmented_domains, load)

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if len(X_all) == 0:
        return np.empty((0, 1024), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    return X_all, y_all

# Add channel dimension for CNN / ResNet input
# (N,1024) -> (N,1,1024)
# ----------------------------------------------------
def add_channel_dimension(X):
    """
    Adds channel dimension for deep learning models.
    """
    return np.expand_dims(X, axis=1)
