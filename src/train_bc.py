"""Train a behavioral-cloning policy from the tree-branch dataset.

The tree dataset contains, for each query, one branch per valid first action
followed by expert recovery. We select the branch with the highest cumulative
return for each query and train the policy to imitate all (state, action) pairs
in those best branches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tree-branch dataset defaults. CSV is needed to recover query IDs and branch boundaries.
DEFAULT_DATASET_H5 = PROJECT_ROOT / "outputs" / "mdp_dataset_tree_branches_5000_terminal_reward.h5"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "outputs" / "trajectories_tree_branches_5000_terminal_reward.csv"

MODEL_DIR = PROJECT_ROOT / "outputs" / "bc_checkpoints"

# Paths are set inside train() based on the chosen dataset so different datasets
# do not overwrite each other.
SCALER_PATH = MODEL_DIR / "bc_scaler.npz"
META_PATH = MODEL_DIR / "bc_meta.json"

ACTION_NAMES = ["STOP", "QueryReform", "Rerank", "PseudoRelevanceFeedback"]

STATE_DIM = 399
N_ACTIONS = 4

# Same hidden sizes as the CQL encoder so weights can potentially be reused.
HIDDEN_UNITS = [512, 512, 256]
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
N_EPOCHS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_best_branch_transitions(
    h5_path: Path,
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return (observations, actions) from the highest-return branch per query."""
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    # Load CSV and segment into branches.
    df = pd.read_csv(csv_path)
    required_cols = {"query_id", "step", "reward", "terminal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    branches: list[dict] = []
    current_branch_rows: list[int] = []
    current_query: str | int | None = None

    for idx, row in df.iterrows():
        step = int(row["step"])
        query_id = row["query_id"]

        # New branch starts when step resets to 1 (except at the very beginning).
        if step == 1 and current_branch_rows:
            branches.append({
                "query_id": current_query,
                "rows": current_branch_rows,
            })
            current_branch_rows = []

        current_query = query_id
        current_branch_rows.append(idx)

    if current_branch_rows:
        branches.append({
            "query_id": current_query,
            "rows": current_branch_rows,
        })

    # Compute cumulative reward per branch and select the best per query.
    branch_returns: list[tuple[int, float]] = []
    for branch in branches:
        total_reward = float(df.loc[branch["rows"], "reward"].sum())
        branch_returns.append((len(branch_returns), total_reward))

    branch_df = pd.DataFrame(
        {
            "branch_idx": range(len(branches)),
            "query_id": [b["query_id"] for b in branches],
            "total_reward": [r for _, r in branch_returns],
        }
    )
    best_branch_df = branch_df.loc[
        branch_df.groupby("query_id")["total_reward"].idxmax()
    ].reset_index(drop=True)
    best_branch_indices = best_branch_df["branch_idx"].tolist()

    print(
        f"Selected {len(best_branch_indices)} best branches from "
        f"{len(branches)} total branches across {branch_df['query_id'].nunique()} queries"
    )
    print(
        "Mean return of selected branches: "
        f"{best_branch_df['total_reward'].mean():.4f}"
    )

    # Load the d3rlpy replay buffer and pick the episodes matching best branches.
    with h5_path.open("rb") as f:
        rb = ReplayBuffer.load(f, InfiniteBuffer())

    if len(rb.episodes) != len(branches):
        raise ValueError(
            f"Mismatch: CSV has {len(branches)} branches but HDF5 has "
            f"{len(rb.episodes)} episodes"
        )

    observations = []
    actions = []
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    for branch_idx in best_branch_indices:
        ep = rb.episodes[branch_idx]
        observations.append(ep.observations.copy())
        actions.append(ep.actions.flatten().copy())
        for a in ep.actions.flatten():
            action_counts[int(a)] += 1

    X = np.concatenate(observations, axis=0).astype(np.float32)
    y = np.concatenate(actions, axis=0).astype(np.int64)

    action_dist = {ACTION_NAMES[i]: int(action_counts[i]) for i in range(N_ACTIONS)}
    print(f"Total transitions in best branches: {len(X)}")
    print(f"Action distribution in best branches: {action_dist}")

    return X, y, action_dist


class BCPolicy(nn.Module):
    """Simple MLP classifier for discrete actions."""

    def __init__(self, state_dim: int, n_actions: int, hidden_units: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(x).to(DEVICE))
            return logits.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(x).to(DEVICE))
            return F.softmax(logits, dim=-1).cpu().numpy()


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def train(h5_path: Path, csv_path: Path, suffix: str) -> None:
    model_path = MODEL_DIR / f"bc_policy_{suffix}.pt"
    scaler_path = MODEL_DIR / f"bc_scaler_{suffix}.npz"
    meta_path = MODEL_DIR / f"bc_meta_{suffix}.json"

    print(f"Loading dataset: {h5_path}")
    print(f"Loading CSV: {csv_path}")
    X, y, action_dist = load_best_branch_transitions(h5_path, csv_path)

    print("Fitting observation scaler")
    X_scaled, mean, std = fit_scaler(X)

    # Train/validation split by transition.
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BCPolicy(STATE_DIM, N_ACTIONS, HIDDEN_UNITS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()
            train_total += len(xb)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_total += len(xb)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss / train_total:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss / val_total:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model, scaler, and metadata.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=mean, std=std)

    meta = {
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "hidden_units": HIDDEN_UNITS,
        "dataset_h5": str(h5_path),
        "dataset_csv": str(csv_path),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "best_val_accuracy": float(best_val_acc),
        "action_distribution": action_dist,
        "device": str(DEVICE),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved BC policy to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BC on tree-branch dataset")
    parser.add_argument("--h5", type=Path, default=DEFAULT_DATASET_H5)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for output files (defaults to dataset h5 stem).",
    )
    args = parser.parse_args()

    suffix = args.suffix or args.h5.stem.replace("mdp_dataset_tree_branches_5000_", "")
    train(args.h5, args.csv, suffix)
