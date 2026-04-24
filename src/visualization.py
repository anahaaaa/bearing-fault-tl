import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def extract_signal(mat_dict):
    for key in mat_dict.keys():
        if "DE_time" in key:
            return mat_dict[key].flatten().astype(np.float32)

    raise ValueError("No DE_time key found")


def load_one_signal(base_path, load_id, filename):
    file_path = os.path.join(
        base_path,
        f"Load_{load_id}",
        filename
    )

    mat = sio.loadmat(file_path)
    return extract_signal(mat)


def plot_all_representations(
    raw_path,
    tsa_path,
    residual_path,
    envelope_path,
    difference_path,
    load_id=0,
    filename="IR_7_load0.mat",
    start=0,
    length=3000
):

    reps = {
        "Raw": load_one_signal(raw_path, load_id, filename),
        "TSA": load_one_signal(tsa_path, load_id, filename),
        "Residual": load_one_signal(residual_path, load_id, filename),
        "Envelope": load_one_signal(envelope_path, load_id, filename),
        "Difference": load_one_signal(difference_path, load_id, filename),
    }

    plt.figure(figsize=(16, 12))

    for i, (name, sig) in enumerate(reps.items(), 1):

        segment = sig[start:start + length]

        plt.subplot(5, 1, i)
        plt.plot(segment, linewidth=1)
        plt.title(name)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
