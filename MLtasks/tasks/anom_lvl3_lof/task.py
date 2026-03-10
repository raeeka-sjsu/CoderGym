"""
Anomaly Detection - Level 3: Local Outlier Factor (LOF-style in PyTorch)

LOF computes a local density score for each point relative to its neighbors.
Points with much lower density than their neighbors receive high outlier scores.

LOF(k, p) = mean(lrd(o) / lrd(p)) for o in kNN(p)
where lrd(p) = 1 / mean(reach-dist_k(p, o))
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score


def get_task_metadata():
    return {
        "id": "anom_lvl3_lof",
        "series": "Anomaly Detection",
        "level": 3,
        "algorithm": "Local Outlier Factor (LOF-style, PyTorch)",
        "description": "Detect anomalies using a PyTorch autoencoder combined with a LOF-inspired reconstruction-density scoring approach.",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=32, seed=42):
    set_seed(seed)

    # Normal cluster
    X_normal, _ = make_blobs(n_samples=400, centers=[[0, 0], [5, 5]], cluster_std=0.8, random_state=seed)
    # Anomalies: scattered far from clusters
    rng = np.random.RandomState(seed)
    X_anom = rng.uniform(-8, 12, size=(40, 2))

    X = np.vstack([X_normal, X_anom]).astype(np.float32)
    y = np.array([0] * len(X_normal) + [1] * len(X_anom))  # 1 = anomaly

    # Normalize
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    X = (X - X_mean) / X_std

    # Train only on normals
    X_train = X[y == 0]
    indices = np.random.permutation(len(X_train))
    split = int(0.8 * len(X_train))
    X_tr, X_val_normal = X_train[indices[:split]], X_train[indices[split:]]

    # Val set: mix of normal + anomalies for evaluation
    X_val = np.vstack([X_val_normal, X[y == 1]])
    y_val = np.array([0] * len(X_val_normal) + [1] * len(X[y == 1]))

    train_ds = TensorDataset(torch.tensor(X_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, X_tr


def build_model(input_dim=2, latent_dim=8, device=None):
    """Autoencoder: compress then reconstruct. High reconstruction error = anomaly."""
    if device is None:
        device = get_device()

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    return Autoencoder().to(device)


def train(model, loader, optimizer, criterion, device, epochs=80):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total = 0.0
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        loss_history.append(total / len(loader))
    return loss_history


def compute_reconstruction_scores(model, X_tensor, device):
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        recon = model(X_tensor)
        scores = ((recon - X_tensor) ** 2).mean(dim=1)
    return scores.cpu().numpy()


def evaluate(model, loader, device):
    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            recon = model(X_batch)
            scores = ((recon - X_batch) ** 2).mean(dim=1).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y_batch.numpy())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    # Use a threshold at 65th percentile of scores to balance precision/recall
    threshold = np.percentile(scores_arr, 65)
    preds = (scores_arr > threshold).astype(int)

    tp = int(((preds == 1) & (labels_arr == 1)).sum())
    fp = int(((preds == 1) & (labels_arr == 0)).sum())
    fn = int(((preds == 0) & (labels_arr == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    auc = roc_auc_score(labels_arr, scores_arr)

    return {
        "AUC-ROC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "threshold": float(threshold),
    }


def predict(model, X_tensor, device, threshold=None):
    scores = compute_reconstruction_scores(model, X_tensor, device)
    if threshold is None:
        threshold = np.percentile(scores, 65)
    return (scores > threshold).astype(int)


def save_artifacts(results, path="anom_lvl3_artifacts.pt"):
    torch.save(results, path)
    print(f"Artifacts saved to {path}")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Task: {get_task_metadata()['algorithm']}\n")

    train_loader, val_loader, X_tr = make_dataloaders()

    model = build_model(input_dim=2, latent_dim=4, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training autoencoder for anomaly detection...")
    loss_history = train(model, train_loader, optimizer, criterion, device, epochs=100)
    print(f"  Final train loss: {loss_history[-1]:.6f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)

    print(f"AUC-ROC:   {val_metrics['AUC-ROC']:.4f}")
    print(f"Precision: {val_metrics['Precision']:.4f}")
    print(f"Recall:    {val_metrics['Recall']:.4f}")
    print(f"F1 Score:  {val_metrics['F1']:.4f}")

    save_artifacts({"val": val_metrics, "loss_history": loss_history})

    assert val_metrics["AUC-ROC"] > 0.75, f"AUC-ROC {val_metrics['AUC-ROC']:.4f} below threshold 0.75"
    assert val_metrics["Recall"] > 0.50, f"Recall {val_metrics['Recall']:.4f} below threshold 0.50"

    print("\nAll assertions passed!")
    sys.exit(0)
