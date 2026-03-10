"""
MLP - Level 5: FashionMNIST Classification

A fully connected MLP trained on the FashionMNIST dataset.
Uses batch normalization, dropout, and learning rate scheduling.

Cross-entropy loss: L = -sum(y_true * log(softmax(logits)))
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_task_metadata():
    return {
        "id": "mlp_lvl5_fashionmnist",
        "series": "MLP",
        "level": 5,
        "algorithm": "MLP with BatchNorm + Dropout on FashionMNIST",
        "description": "Train a deep MLP on FashionMNIST using batch normalization, dropout, and LR scheduling.",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=128, val_split=0.1, seed=42):
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_full = datasets.FashionMNIST(root="/tmp/fashionmnist", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="/tmp/fashionmnist", train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def build_model(input_dim=784, n_classes=10, device=None):
    if device is None:
        device = get_device()

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, n_classes),
    ).to(device)
    return model


def train(model, loader, optimizer, criterion, device, epochs=15, scheduler=None):
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        if scheduler:
            scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return loss_history


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    # R2 placeholder (not meaningful for classification but required by protocol)
    r2 = 2 * accuracy - 1  # maps accuracy [0,1] -> R2 proxy [-1, 1]
    return {"loss": float(avg_loss), "accuracy": float(accuracy), "MSE": float(avg_loss), "R2": float(r2)}


def predict(model, X_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        return logits.argmax(dim=1).cpu()


def save_artifacts(results, path="mlp_lvl5_artifacts.pt"):
    torch.save(results, path)
    print(f"Artifacts saved to {path}")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Task: {get_task_metadata()['algorithm']}\n")

    print("Loading FashionMNIST...")
    train_loader, val_loader, test_loader = make_dataloaders(batch_size=128)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\nTraining MLP...")
    loss_history = train(model, train_loader, optimizer, criterion, device, epochs=15, scheduler=scheduler)

    print("\nEvaluating...")
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Train - Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")

    save_artifacts({
        "train": train_metrics,
        "val": val_metrics,
        "loss_history": loss_history,
    })

    assert val_metrics["accuracy"] > 0.85, f"Val accuracy {val_metrics['accuracy']:.4f} below threshold 0.85"
    assert val_metrics["loss"] < 0.50, f"Val loss {val_metrics['loss']:.4f} above threshold 0.50"

    print("\nAll assertions passed!")
    sys.exit(0)
