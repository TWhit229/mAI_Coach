import argparse
import json
import sys
from pathlib import Path

# Add shared core logic to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import label_config
from core.metrics import metrics_to_features

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
    parser = argparse.ArgumentParser(description="Bench dataset helper")
    parser.add_argument("--mode", choices=["preprocess", "train"], help="Which action to run")
    parser.add_argument("--dataset_dir", type=str, help="Folder of JSONs (for preprocess)")
    parser.add_argument("--output_prefix", type=str, help="Prefix for outputs")
    parser.add_argument("--data_prefix", type=str, help="Prefix of *_X.npy/_y.npy/_meta.json (for train)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_example(rep, all_tags):
    features = metrics_to_features(rep)
    if features is None:
        return None
    tags = set(rep.get("tags") or [])
    labels = [1 if tag in tags else 0 for tag in all_tags]
    return features, labels


def run_preprocess(dataset_dir: str | None, prefix: str | None):
    if dataset_dir is None and filedialog is not None:
        root = tk.Tk()
        root.withdraw()
        dataset_dir = filedialog.askdirectory(title="Select dataset folder")
        root.destroy()
    if not dataset_dir:
        dataset_dir = input("Dataset directory: ").strip()
    if prefix is None:
        if simpledialog is not None:
            root = tk.Tk()
            root.withdraw()
            prefix = simpledialog.askstring("Output prefix", "Prefix (e.g., bench_v1)")
            root.destroy()
        if not prefix:
            prefix = input("Output prefix: ").strip()
    if not dataset_dir or not prefix:
        print("Missing dataset directory or output prefix.")
        return
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    json_files = sorted(dataset_dir.glob("*.json"))
    
    # Load dynamic tags
    cfg = label_config.load_label_config()
    all_tags = cfg.get("tags") or []
    
    X, Y = [], []
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"Skipping {path.name}: failed to load JSON ({exc})")
            continue
        example = extract_example(data, all_tags)
        if example is None:
            continue
        feats, labels = example
        X.append(feats)
        Y.append(labels)
    if not X:
        print("No valid examples found.")
        return
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    np.save(prefix + "_X.npy", X)
    np.save(prefix + "_y.npy", Y)
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
        "tags": all_tags,
        "num_examples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "num_tags": int(Y.shape[1]),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved X to {prefix}_X.npy with shape {X.shape}")
    print(f"Saved y to {prefix}_y.npy with shape {Y.shape}")
    print(f"Saved meta to {prefix}_meta.json")


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
    print("\n==== Overall dev stats ====")
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


def run_training(data_prefix, output_prefix, epochs, batch_size, dev_fraction, seed):
    if not data_prefix and filedialog is not None:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select features file (_X.npy)",
            filetypes=[("NumPy features", "*_X.npy"), ("All files", "*.*")],
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
        print("No data prefix, aborting.")
        return
    if not output_prefix:
        if simpledialog is not None:
            root = tk.Tk()
            root.withdraw()
            output_prefix = simpledialog.askstring(
                "Output prefix", "Prefix for outputs (e.g., bench_mlp_v1)"
            )
            root.destroy()
        if not output_prefix:
            output_prefix = input("Output prefix: ").strip()
    if not output_prefix:
        print("No output prefix, aborting.")
        return
    X = np.load(str(data_prefix) + "_X.npy")
    y = np.load(str(data_prefix) + "_y.npy")
    with open(str(data_prefix) + "_meta.json", "r") as f:
        meta = json.load(f)
    X_train, X_dev, y_train, y_dev, scaler = split_and_scale(X, y, dev_fraction, seed)
    train_loader, dev_loader, X_dev_t, y_dev_t = build_dataloaders(
        X_train, y_train, X_dev, y_dev, batch_size
    )
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BenchMLP(in_dim, out_dim).to(device)
    best_dev_loss, epochs_trained = train_model(
        model, train_loader, dev_loader, device, epochs
    )
    evaluate(model, X_dev_t, y_dev, meta.get("tags", []), device)
    torch.save(model.state_dict(), f"{output_prefix}_model.pt")
    np.savez(f"{output_prefix}_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
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
    print(f"Saved model -> {output_prefix}_model.pt")
    print(f"Saved scaler -> {output_prefix}_scaler.npz")
    print(f"Saved meta -> {output_prefix}_train_meta.json")


def main():
    args = parse_args()
    mode = args.mode
    if not mode:
        mode = input("Choose mode: [p]reprocess / [t]rain: ").strip().lower()
        mode = "preprocess" if mode.startswith("p") else "train"
    if mode == "preprocess":
        run_preprocess(args.dataset_dir, args.output_prefix)
    else:
        run_training(
            args.data_prefix,
            args.output_prefix,
            args.epochs,
            args.batch_size,
            args.dev_fraction,
            args.seed,
        )


if __name__ == "__main__":
    main()
