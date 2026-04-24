# src/utils.py

import os
import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed=42):
    """
    Full reproducibility setup
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def print_class_distribution(domains, num_classes=10):
    """
    domains[load] = list of samples

    sample format:
    {
        "window": ...,
        "label": int,
        ...
    }
    """

    for load in sorted(domains.keys()):

        counts = defaultdict(int)

        for sample in domains[load]:
            counts[sample["label"]] += 1

        total = sum(counts.values())

        print(f"\n==============================")
        print(f"Load {load} | Total Samples = {total}")
        print(f"==============================")

        for cls in range(num_classes):
            print(f"Class {cls}: {counts[cls]}")
