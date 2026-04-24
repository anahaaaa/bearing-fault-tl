# ==================================================
# src/main.py
# FINAL POLISHED VERSION
# Full Automated Bearing Fault Diagnosis Pipeline
# ==================================================

import os
import traceback
import pandas as pd
import torch

# --------------------------------------------------
# Imports
# --------------------------------------------------

from utils import set_seed, print_class_distribution
from data_loader import load_cwru
from segmentation import (
    segment_domains,
    merge_domains,
    add_channel_dimension
)
from preprocessing import zscore_domains
from class_weights import compute_class_weights
from dataset import create_dataloader
from model import build_model
from train import train_model
from evaluate import evaluate_model, DEFAULT_CLASS_NAMES
from visualization import plot_all_representations


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42

# --------------------------------------------------
# DATASET PATHS
# Update these when separate processed folders exist
# --------------------------------------------------

DATA_ROOT = "data"

REPRESENTATION_PATHS = {
    "raw":        f"{DATA_ROOT}/CWRU_DATA",
    "tsa":        f"{DATA_ROOT}/CWRU_DATA",
    "residual":   f"{DATA_ROOT}/CWRU_DATA",
    "envelope":   f"{DATA_ROOT}/CWRU_DATA",
    "difference": f"{DATA_ROOT}/CWRU_DATA",
}

# If only raw exists now:
SIGNAL_TYPES = list(REPRESENTATION_PATHS.keys())

WINDOW_SIZE = 1024
OVERLAP = 0.5
BATCH_SIZE = 64

SOURCE_LOADS = [0]
TARGET_LOADS = [3]

EPOCHS = 30
LR = 1e-3

VISUALIZE = True
VISUAL_LOAD = 0
VISUAL_FILE = "IR_7_load0.mat"

SAVE_RESULTS = True
RESULTS_FILE = "results/pipeline_summary.csv"


# --------------------------------------------------
# RUN ONE SIGNAL TYPE
# --------------------------------------------------

def run_pipeline(signal_type):

    print("\n" + "=" * 70)
    print(f"RUNNING REPRESENTATION: {signal_type.upper()}")
    print("=" * 70)

    data_path = REPRESENTATION_PATHS[signal_type]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Folder not found: {data_path}")

    # ----------------------------------------------
    # Load Data
    # ----------------------------------------------
    domains = load_cwru(
        base_path=data_path,
        signal_type=signal_type
    )

    print("\nOriginal files before segmentation")
    print_class_distribution(domains)

    # ----------------------------------------------
    # Segment Signals
    # ----------------------------------------------
    segmented_domains = segment_domains(
        domains=domains,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP
    )

    print("\nSamples after segmentation")
    print_class_distribution(segmented_domains)

    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------
    segmented_domains = zscore_domains(segmented_domains)

    # ----------------------------------------------
    # Merge Source / Target
    # ----------------------------------------------
    X_train, y_train = merge_domains(
        segmented_domains,
        SOURCE_LOADS
    )

    X_test, y_test = merge_domains(
        segmented_domains,
        TARGET_LOADS
    )

    # ----------------------------------------------
    # Add Channel Dimension
    # (N,1024) -> (N,1,1024)
    # ----------------------------------------------
    X_train = add_channel_dimension(X_train)
    X_test = add_channel_dimension(X_test)

    # ----------------------------------------------
    # DataLoaders
    # ----------------------------------------------
    print("\nCreating DataLoaders...")

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = create_dataloader(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----------------------------------------------
    # Compute Class Weights
    # ----------------------------------------------
    source_samples = []

    for load in SOURCE_LOADS:
        source_samples.extend(segmented_domains[load])

    class_weights = compute_class_weights(source_samples)

    # ----------------------------------------------
    # Build Model
    # ----------------------------------------------
    print("\nBuilding ResNet1D model...")

    model = build_model(num_classes=10)

    # ----------------------------------------------
    # Train Model
    # ----------------------------------------------
    print("\nTraining model...")

    save_path = f"results/{signal_type}_best_model.pth"

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        epochs=EPOCHS,
        lr=LR,
        save_path=save_path
    )

    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------
    print("\nEvaluating model...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=DEFAULT_CLASS_NAMES,
        cm_save_path=f"results/{signal_type}_cm.png"
    )

    # ----------------------------------------------
    # Final Summary
    # ----------------------------------------------
    print("\nFINAL SUMMARY")
    print("Signal Type :", signal_type)
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)
    print("Accuracy    :", round(metrics["accuracy"], 4))
    print("Macro F1    :", round(metrics["f1_macro"], 4))

    return {
        "signal_type": signal_type,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "accuracy": round(metrics["accuracy"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
        "status": "success"
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    set_seed(SEED)

    print("=" * 70)
    print("AUTOMATED CWRU PIPELINE")
    print("=" * 70)

    # ----------------------------------------------
    # Visualization
    # ----------------------------------------------
    if VISUALIZE:

        print("\nVisualizing signal representations...")

        try:
            plot_all_representations(
                raw_path=REPRESENTATION_PATHS["raw"],
                tsa_path=REPRESENTATION_PATHS["tsa"],
                residual_path=REPRESENTATION_PATHS["residual"],
                envelope_path=REPRESENTATION_PATHS["envelope"],
                difference_path=REPRESENTATION_PATHS["difference"],
                load_id=VISUAL_LOAD,
                filename=VISUAL_FILE,
                start=0,
                length=3000
            )

        except Exception as e:
            print("Visualization skipped:", e)

    # ----------------------------------------------
    # Run Experiments
    # ----------------------------------------------
    results = []

    for signal_type in SIGNAL_TYPES:

        try:
            output = run_pipeline(signal_type)
            results.append(output)

        except Exception as e:

            print(f"\nFAILED: {signal_type}")
            print(e)
            traceback.print_exc()

            results.append({
                "signal_type": signal_type,
                "status": "failed"
            })

    # ----------------------------------------------
    # Final Results
    # ----------------------------------------------
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df)

    # ----------------------------------------------
    # Save CSV
    # ----------------------------------------------
    if SAVE_RESULTS:

        os.makedirs("results", exist_ok=True)
        df.to_csv(RESULTS_FILE, index=False)

        print(f"\nSaved results to: {RESULTS_FILE}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()
