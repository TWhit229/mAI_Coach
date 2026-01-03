"""Model training logic."""

import json
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from core.metrics import metrics_to_features


class BenchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _extract_training_example(rep: Dict, tag_order: Sequence[str]):
    feats = metrics_to_features(rep)
    if feats is None:
        return None
    tags = set(rep.get("tags") or [])
    labels = [1 if t in tags else 0 for t in tag_order]
    return feats, labels


def preprocess_dataset(
    dataset_dir: Path, output_prefix: Path, tag_order: Sequence[str], log: Callable[[str], None]
):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    json_files = sorted(dataset_dir.glob("*.json"))
    log(f"Found {len(json_files)} JSON files in {dataset_dir}")
    if not json_files:
        return

    X, Y = [], []
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            log(f"Skipping {path.name}: {exc}")
            continue

        ex = _extract_training_example(data, tag_order)
        if ex:
            X.append(ex[0])
            Y.append(ex[1])

    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)
    
    if len(X_np) == 0:
        log("No valid examples extracted.")
        return

    log(f"Extracted dataset: X={X_np.shape}, Y={Y_np.shape}")
    out_x = str(output_prefix) + "_X.npy"
    out_y = str(output_prefix) + "_Y.npy"
    out_meta = str(output_prefix) + "_meta.json"

    np.save(out_x, X_np)
    np.save(out_y, Y_np)

    meta = {
        "feature_names": [
            "load_lbs",
            "grip_ratio_median",
            "grip_ratio_range",
            "grip_uneven_median",
            "grip_uneven_norm",
            "bar_tilt_median_deg",
            "bar_tilt_deg_max",
            "tracking_bad_ratio",
            "tracking_quality",
            "wrist_y_min",
            "wrist_y_max",
            "wrist_y_range",
        ],
        "tags": tag_order,
        "num_examples": int(X_np.shape[0]),
        "num_features": int(X_np.shape[1]),
        "num_tags": int(Y_np.shape[1]),
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    log(f"Saved arrays to {out_x}, {out_y}")
    log(f"Saved metadata to {out_meta}")


def _split_and_scale(X, y, dev_fraction, seed):
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=dev_fraction, random_state=seed
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    return X_train_scaled, y_train, X_dev_scaled, y_dev, scaler


def _build_dataloaders(X_train, y_train, X_dev, y_dev, batch_size):
    t_x_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    t_x_dev = torch.tensor(X_dev, dtype=torch.float32)
    t_y_dev = torch.tensor(y_dev, dtype=torch.float32)

    train_ds = TensorDataset(t_x_train, t_y_train)
    dev_ds = TensorDataset(t_x_dev, t_y_dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


def _evaluate_model(model, X_dev_t, y_dev, tags, device, log: Callable[[str], None]):
    model.eval()
    with torch.no_grad():
        dev_tensor = torch.tensor(X_dev_t, dtype=torch.float32).to(device)
        logits = model(dev_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

    acc = accuracy_score(y_dev, preds)
    f1 = f1_score(y_dev, preds, average="macro", zero_division=0)
    log(f"Dev Accuracy: {acc:.4f}")
    log(f"Dev Macro F1: {f1:.4f}")

    report = classification_report(
        y_dev, preds, target_names=tags, zero_division=0
    )
    log("Classification Report:\n" + report)


def _train_model(model, train_loader, dev_loader, device, epochs, log: Callable[[str], None]):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            model.eval()
            dev_loss = 0.0
            with torch.no_grad():
                for dbx, dby in dev_loader:
                    dbx, dby = dbx.to(device), dby.to(device)
                    dout = model(dbx)
                    dloss = criterion(dout, dby)
                    dev_loss += dloss.item()
            avg_dev = dev_loss / len(dev_loader)
            log(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Dev Loss: {avg_dev:.4f}")


def train_dataset_model(
    data_prefix: Path,
    output_prefix: Path,
    tags: Sequence[str],
    epochs: int,
    batch_size: int,
    dev_fraction: float,
    seed: int,
    log: Callable[[str], None],
):
    import pickle

    x_path = str(data_prefix) + "_X.npy"
    y_path = str(data_prefix) + "_Y.npy"
    if not Path(x_path).exists() or not Path(y_path).exists():
        log(f"Missing data files: {x_path} / {y_path}")
        return

    log(f"Loading data from {x_path}")
    X = np.load(x_path)
    Y = np.load(y_path)
    
    if len(X) == 0:
        log("Dataset is empty.")
        return

    X_train_s, y_train, X_dev_s, y_dev, scaler = _split_and_scale(
        X, Y, dev_fraction, seed
    )
    
    device = torch.device("cpu")
    train_loader, dev_loader = _build_dataloaders(
        X_train_s, y_train, X_dev_s, y_dev, batch_size
    )

    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    log(f"Building MLP: in={in_dim}, out={out_dim}")
    model = BenchMLP(in_dim, out_dim).to(device)

    log("Starting training...")
    _train_model(model, train_loader, dev_loader, device, epochs, log)
    _evaluate_model(model, X_dev_s, y_dev, tags, device, log)

    # Save model
    model_out = str(output_prefix) + ".pth"
    torch.save(model.state_dict(), model_out)
    log(f"Saved model to {model_out}")

    # Save scaler
    scaler_out = str(output_prefix) + "_scaler.pkl"
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    log(f"Saved scaler to {scaler_out}")
