import numpy as np
import torch
from collections import defaultdict


def print_class_counts(domains):
    """
    Print counts per class per domain
    """

    for load in sorted(domains.keys()):

        counts = defaultdict(int)

        for sample in domains[load]:
            counts[sample["label"]] += 1

        print(f"\nLoad {load}")

        for c in range(10):
            print(f"Class {c}: {counts[c]}")


def compute_class_weights(samples, num_classes=10):
    """
    Compute inverse-frequency class weights
    for one training dataset.

    samples = list of segmented samples
    """

    counts = np.zeros(num_classes, dtype=np.float32)

    for sample in samples:
        counts[sample["label"]] += 1

    # avoid divide by zero
    counts[counts == 0] = 1.0

    weights = counts.sum() / (num_classes * counts)

    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def merge_domains(domains, loads):
    """
    Merge sample lists from selected domains
    """

    merged = []

    for load in loads:
        merged.extend(domains[load])

    return merged
