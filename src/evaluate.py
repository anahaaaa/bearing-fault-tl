# ==================================================
# Evaluation Utilities for Bearing Fault Diagnosis
# ==================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Run Inference on Test Loader
# --------------------------------------------------
def predict(model, loader, device):
    """
    Returns:
        y_true
        y_pred
    """

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            outputs = model(X)

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


# --------------------------------------------------
# Compute Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    Returns metrics dict
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }

    return metrics


# --------------------------------------------------
# Plot Confusion Matrix
# --------------------------------------------------
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    save_path=None
):
    """
    Plots confusion matrix
    """

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path is not None:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(save_path, dpi=300)

    plt.show()


# --------------------------------------------------
# Full Evaluation
# --------------------------------------------------
def evaluate_model(
    model,
    test_loader,
    device,
    class_names=None,
    cm_save_path=None
):
    """
    Full evaluation pipeline
    """

    y_true, y_pred = predict(
        model,
        test_loader,
        device
    )

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        save_path=cm_save_path
    )

    return metrics


# --------------------------------------------------
# Class Names 
# --------------------------------------------------
DEFAULT_CLASS_NAMES = [
    "Normal",
    "IR_7",
    "IR_14",
    "IR_21",
    "OR_7",
    "OR_14",
    "OR_21",
    "B_7",
    "B_14",
    "B_21"
]
