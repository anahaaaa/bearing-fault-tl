# ==================================================
# src/main.py
# Automated Pipeline for ALL Signal Representations
# raw / tsa / residual / envelope / difference
# ==================================================

import os
import traceback
import pandas as pd
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
from visualization import plot_all_representations


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42

DATA_ROOT = "data"

REPRESENTATION_PATHS = {
    "raw":        f"{DATA_ROOT}/raw/CWRU_DATA",
    "tsa":        f"{DATA_ROOT}/tsa/CWRU_DATA",
    "residual":   f"{DATA_ROOT}/residual/CWRU_DATA",
    "envelope":   f"{DATA_ROOT}/envelope/CWRU_DATA",
    "difference": f"{DATA_ROOT}/difference/CWRU_DATA",
}

SIGNAL_TYPES = list(REPRESENTATION_PATHS.keys())

WINDOW_SIZE = 1024
OVERLAP = 0.5

SOURCE_LOADS = [0]
TARGET_LOADS = [3]

VISUAL_LOAD = 0
VISUAL_FILE = "IR_7_load0.mat"

SAVE_RESULTS = True
RESULTS_FILE = "results/pipeline_summary.csv"


# --------------------------------------------------
# RUN ONE REPRESENTATION
# --------------------------------------------------

def run_pipeline(signal_type):

    print("\n" + "=" * 70)
    print(f"RUNNING REPRESENTATION: {signal_type.upper()}")
    print("=" * 70)

    data_path = REPRESENTATION_PATHS[signal_type]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Folder not found: {data_path}")

    # ----------------------------------------------
    # Load Dataset
    # ----------------------------------------------
    domains = load_cwru(
        base_path=data_path,
        signal_type=signal_type
    )

    print("\nOriginal files before segmentation")
    print_class_distribution(domains)

    # ----------------------------------------------
    # Segment
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
    # Merge Domains
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
    # ----------------------------------------------
    X_train = add_channel_dimension(X_train)
    X_test = add_channel_dimension(X_test)
    
    # ----------------------------------------------
    # Creating DataLoaders
    # ----------------------------------------------

    print("\n[8] Creating DataLoaders...")
    
    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=64,
        shuffle=True
    )
    
    test_loader = create_dataloader(
        X_test,
        y_test,
        batch_size=64,
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
        epochs=30,
        lr=1e-3,
        save_path=save_path
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------
    print("\nFINAL SUMMARY")
    print("Signal Type :", signal_type)
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    best_acc = max(history["test_acc"])
    
    return {
        "signal_type": signal_type,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "best_test_acc": round(best_acc, 4),
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
    # Visualize Once
    # ----------------------------------------------
    print("\n[0] Visualizing all signal representations...")

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
    # Run All Representations
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
    # Final Results Table
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
