"""
Ensemble Learning - Level 1: Bagging

Bagging (Bootstrap Aggregating) trains multiple independent models on
random subsamples of the training data and averages their predictions.

Objective: J = (1/N) * sum((y - mean(h_k(x)))^2)
where h_k(x) is the k-th base learner's prediction.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def get_task_metadata():
    return {
        "id": "ens_lvl1_bagging",
        "series": "Ensemble",
        "level": 1,
        "algorithm": "Bagging (Bootstrap Aggregating)",
        "description": "Train multiple independent linear models on bootstrap samples and average predictions.",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=32, n_samples=500, n_features=10, seed=42):
    set_seed(seed)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10.0, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Normalize
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, X_train, X_val, y_train, y_val, n_features


def build_model(n_features, device):
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    return model


def train(model, loader, optimizer, criterion, device, epochs=50):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / len(loader))
    return loss_history


def evaluate(model_list, loader, device):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            batch_preds = torch.stack([m(X_batch).squeeze(-1) for m in model_list], dim=0)
            ensemble_pred = batch_preds.mean(dim=0).cpu()
            all_preds.append(ensemble_pred)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    r2 = float(r2_score(targets, preds))
    return {"MSE": mse, "R2": r2}


def predict(model_list, X_tensor, device):
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        preds = torch.stack([m(X_tensor).squeeze(-1) for m in model_list], dim=0)
        return preds.mean(dim=0).cpu()


def save_artifacts(results, path="ens_lvl1_artifacts.pt"):
    torch.save(results, path)
    print(f"Artifacts saved to {path}")


def train_bagging_ensemble(X_train, y_train, n_features, n_estimators=5, device=None, epochs=50):
    """Train n_estimators models on bootstrap samples of the training data."""
    if device is None:
        device = get_device()

    n = len(X_train)
    model_list = []
    criterion = nn.MSELoss()

    for i in range(n_estimators):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = torch.tensor(X_train[indices])
        y_boot = torch.tensor(y_train[indices])
        boot_ds = TensorDataset(X_boot, y_boot)
        boot_loader = DataLoader(boot_ds, batch_size=32, shuffle=True)

        model = build_model(n_features, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, boot_loader, optimizer, criterion, device, epochs=epochs)
        model_list.append(model)
        print(f"  Estimator {i+1}/{n_estimators} trained.")

    return model_list


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Task: {get_task_metadata()['algorithm']}\n")

    train_loader, val_loader, X_train, X_val, y_train, y_val, n_features = make_dataloaders()

    print("Training bagging ensemble (5 estimators)...")
    model_list = train_bagging_ensemble(X_train, y_train, n_features, n_estimators=5, device=device, epochs=60)

    print("\nEvaluating...")
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader_eval = DataLoader(train_ds, batch_size=32)

    train_metrics = evaluate(model_list, train_loader_eval, device)
    val_metrics = evaluate(model_list, val_loader, device)

    print(f"Train MSE: {train_metrics['MSE']:.4f} | Train R2: {train_metrics['R2']:.4f}")
    print(f"Val   MSE: {val_metrics['MSE']:.4f} | Val   R2: {val_metrics['R2']:.4f}")

    save_artifacts({"train": train_metrics, "val": val_metrics})

    # Quality thresholds
    assert val_metrics["R2"] > 0.80, f"Val R2 {val_metrics['R2']:.4f} below threshold 0.80"
    assert val_metrics["MSE"] < 0.5, f"Val MSE {val_metrics['MSE']:.4f} above threshold 0.5"

    print("\nAll assertions passed!")
    sys.exit(0)
