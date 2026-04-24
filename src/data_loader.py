import os
import scipy.io as sio
import numpy as np


# --------------------------------------------------
# 10-Class Label Mapping
# --------------------------------------------------
LABEL_MAP = {
    "Normal": 0,

    "IR_7": 1,
    "IR_14": 2,
    "IR_21": 3,

    "OR6_7": 4,
    "OR6_14": 5,
    "OR6_21": 6,

    "B_7": 7,
    "B_14": 8,
    "B_21": 9
}


# --------------------------------------------------
# Extract class label from filename
# --------------------------------------------------
def get_label(filename):
    """
    Converts filename into class label.
    
    Removes:
        *_28 files
        OR3_* files
        OR12_* files
    """

    name = filename.replace(".mat", "")

    # Remove unwanted files
    if "_28" in name:
        return None

    if name.startswith("OR3") or name.startswith("OR12"):
        return None

    # Exact mapping
    for key, value in LABEL_MAP.items():
        if name.startswith(key):
            return value

    return None


# --------------------------------------------------
# Extract vibration signal from .mat file
# --------------------------------------------------
def extract_signal(mat_dict):
    """
    Finds drive-end signal key (DE_time)
    """

    for key in mat_dict.keys():

        if "DE_time" in key:
            return mat_dict[key].flatten().astype(np.float32)

    return None


# --------------------------------------------------
# Main Loader
# --------------------------------------------------
def load_cwru(base_path, signal_type="raw"):
    """
    Load CWRU dataset from one representation folder.

    Args:
        base_path:
            Example:
            /kaggle/input/raw/CWRU_DATA
            /kaggle/input/envelope/CWRU_DATA

        signal_type:
            raw / tsa / residual / envelope / difference

    Returns:
        domains = {
            0: [sample, sample...],
            1: [...],
            2: [...],
            3: [...]
        }

        sample = {
            "signal": np.ndarray,
            "label": int,
            "file": str,
            "type": str
        }
    """

    domains = {}

    for load in range(4):

        folder = os.path.join(base_path, f"Load_{load}")
        domains[load] = []

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        for file in sorted(os.listdir(folder)):

            if not file.endswith(".mat"):
                continue

            label = get_label(file)

            # Skip unwanted files
            if label is None:
                continue

            path = os.path.join(folder, file)

            mat = sio.loadmat(path)

            signal = extract_signal(mat)

            if signal is None:
                continue

            domains[load].append({
                "signal": signal,
                "label": label,
                "file": file,
                "type": signal_type
            })

    return domains
