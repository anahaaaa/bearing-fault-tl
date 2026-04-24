# ==================================================
# src/train.py
# Training Utilities for ResNet1D Bearing Diagnosis
# ==================================================

import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm


# --------------------------------------------------
# Train One Epoch
# --------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)

    for X, y in loop:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        _, preds = torch.max(outputs, 1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# --------------------------------------------------
# Evaluate
# --------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)

            _, preds = torch.max(outputs, 1)

            total += y.size(0)
            correct += (preds == y).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


# --------------------------------------------------
# Full Training Loop
# --------------------------------------------------
def train_model(
    model,
    train_loader,
    test_loader,
    class_weights,
    epochs=30,
    lr=1e-3,
    save_path="results/best_model.pth"
):
    """
    Train model and save best checkpoint
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = model.to(device)

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    patience = 5
    counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(1, epochs + 1):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Test  Loss: {test_loss:.4f} | "
            f"Test  Acc: {test_acc:.4f}"
        )

        # Save best model
        if test_acc > best_acc + 1e-4:

            best_acc = test_acc
            best_model = copy.deepcopy(
                model.state_dict()
            )

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            torch.save(best_model, save_path)

            print("Best model saved.")
            counter = 0
            
        else:
            
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model back
    model.load_state_dict(best_model)

    print("\nTraining Complete.")
    print(f"Best Test Accuracy: {best_acc:.4f}")

    return model, history
