# ==================================================
# src/dataset.py
# PyTorch Dataset + DataLoader for CWRU Signals
# Input:
#   X -> (N,1,1024)
#   y -> (N,)
# ==================================================

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np


# --------------------------------------------------
# Dataset Class
# --------------------------------------------------
class BearingDataset(Dataset):
    """
    Dataset for 1D bearing fault signals
    """

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# --------------------------------------------------
# Standard DataLoader
# --------------------------------------------------
def create_dataloader(
    X,
    y,
    batch_size=64,
    shuffle=True,
    num_workers=0
):
    """
    Standard DataLoader
    """

    dataset = BearingDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader


# --------------------------------------------------
# Weighted Sampler DataLoader
# (Alternative to weighted loss)
# --------------------------------------------------
def create_weighted_dataloader(
    X,
    y,
    batch_size=64,
    num_workers=0
):
    """
    Balanced mini-batches using WeightedRandomSampler
    """

    dataset = BearingDataset(X, y)

    y_np = np.array(y)

    class_counts = np.bincount(y_np)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[y_np]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers
    )

    return loader


# --------------------------------------------------
# Quick Test Utility
# --------------------------------------------------
def inspect_batch(loader):
    """
    Prints one batch shape
    """

    X_batch, y_batch = next(iter(loader))

    print("Batch X shape:", X_batch.shape)
    print("Batch y shape:", y_batch.shape)
