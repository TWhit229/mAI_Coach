#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    simpledialog = None


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label MLP for bench dataset")
    parser.add_argument("--data_prefix", type=str, help="Prefix of *_X.npy / *_y.npy / *_meta.json")
    parser.add_argument("--output_prefix", type=str, help="Prefix for output artifacts")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(prefix: str):
    base = Path(prefix)
    X = np.load(str(base) + "_X.npy")
    y = np.load(str(base) + "_y.npy")
    with open(str(base) + "_meta.json", "r") as f:
        meta = json.load(f)
    return X, y, meta


def split_and_scale(X, y, dev_fraction, seed):
    X_train, X_dev, y_train, y_dev = train_test_split(
        X,
        y,
        test_size=dev_fraction,
        random_state=seed,
        shuffle=True,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    return X_train_scaled, X_dev_scaled, y_train, y_dev, scaler


def build_dataloaders(X_train, y_train, X_dev, y_dev, batch_size):
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_dev_t = torch.from_numpy(X_dev).float()
    y_dev_t = torch.from_numpy(y_dev).float()
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(X_dev_t, y_dev_t), batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, X_dev_t, y_dev_t


def train_model(model, train_loader, dev_loader, device, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_dev_loss = float("inf")
    best_state_dict = None
    patience = 10
    epochs_without_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        model.eval()
        dev_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                dev_loss_sum += loss.item() * xb.size(0)
        avg_dev_loss = dev_loss_sum / len(dev_loader.dataset)
        progress = (epoch + 1) / epochs
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "-" * (bar_len - filled)
        print(
            f"Epoch {epoch + 1}/{epochs} [{bar}] train_loss={avg_train_loss:.4f} dev_loss={avg_dev_loss:.4f}"
        )
        if avg_dev_loss < best_dev_loss - 1e-4:
            best_dev_loss = avg_dev_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return best_dev_loss, epoch + 1


def evaluate(model, X_dev_t, y_dev, tags, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_dev_t.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

    subset_acc = accuracy_score(y_dev, y_pred)
    micro_f1 = f1_score(y_dev, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_dev, y_pred, average="macro", zero_division=0)

    print("\n==== Overall dev metrics ====")
    print(f"Subset accuracy: {subset_acc:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Macro F1: {macro_f1:.3f}\n")
    for idx, tag in enumerate(tags):
        print(f"==== {tag} ====")
        print(
            classification_report(
                y_dev[:, idx],
                y_pred[:, idx],
                digits=3,
                zero_division=0,
            )
        )


def save_outputs(
    model,
    scaler,
    meta,
    output_prefix,
    epochs_trained,
    best_dev_loss,
    dev_fraction,
):
    torch.save(model.state_dict(), f"{output_prefix}_model.pt")
    np.savez(
        f"{output_prefix}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    train_meta = {
        "feature_names": meta.get("feature_names", []),
        "tags": meta.get("tags", []),
        "hidden_layers": [32, 16],
        "threshold": 0.5,
        "epochs_trained": int(epochs_trained),
        "best_dev_loss": float(best_dev_loss),
        "dev_fraction": float(dev_fraction),
    }
    with open(f"{output_prefix}_train_meta.json", "w") as f:
        json.dump(train_meta, f, indent=2)
    print(f"Saved model weights -> {output_prefix}_model.pt")
    print(f"Saved scaler -> {output_prefix}_scaler.npz")
    print(f"Saved training meta -> {output_prefix}_train_meta.json")


def main():
    args = parse_args()
    data_prefix = args.data_prefix
    output_prefix = args.output_prefix

    if not data_prefix:
        if filedialog is not None:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(
                title="Select features file (_X.npy)",
                filetypes=[
                    ("NumPy features", "*_X.npy"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
            if selected:
                sel_path = Path(selected)
                if sel_path.name.endswith("_X.npy"):
                    data_prefix = str(sel_path.with_name(sel_path.name[:-6]))
                else:
                    data_prefix = str(sel_path.with_suffix(""))
        if not data_prefix:
            data_prefix = input("Data prefix (path without _X.npy): ").strip()

    if not data_prefix:
        print("No data prefix provided; exiting.")
        return

    if not output_prefix:
        if simpledialog is not None:
            root = tk.Tk()
            root.withdraw()
            output_prefix = simpledialog.askstring(
                "Output prefix", "Enter prefix for outputs (e.g., bench_mlp_v1)"
            )
            root.destroy()
        if not output_prefix:
            output_prefix = input("Output prefix: ").strip()

    if not output_prefix:
        print("No output prefix provided; exiting.")
        return

    X, y, meta = load_data(data_prefix)
    X_train, X_dev, y_train, y_dev, scaler = split_and_scale(
        X, y, args.dev_fraction, args.seed
    )
    train_loader, dev_loader, X_dev_t, y_dev_t = build_dataloaders(
        X_train, y_train, X_dev, y_dev, args.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    model = BenchMLP(in_dim, out_dim).to(device)
    best_dev_loss, epochs_trained = train_model(
        model, train_loader, dev_loader, device, args.epochs
    )
    evaluate(model, X_dev_t, y_dev, meta.get("tags", []), device)
    save_outputs(
        model,
        scaler,
        meta,
        output_prefix,
        epochs_trained,
        best_dev_loss,
        args.dev_fraction,
    )


if __name__ == "__main__":
    main()
