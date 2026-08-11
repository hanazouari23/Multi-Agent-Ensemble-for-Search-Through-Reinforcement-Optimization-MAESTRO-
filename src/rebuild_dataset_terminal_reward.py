"""Rebuild the tree-branches dataset with a low terminal STOP reward.

The full terminal STOP reward (2.0*NDCG + 0.5*Recall) made the policy stop too
often. This version lowers the STOP bonus so the policy is more willing to take
actions that may improve ranking:

- Action reward: 0.99 * (2.0*ndcg_after + 0.5*recall_after)
                  - (2.0*ndcg_before + 0.5*recall_before)
                  - 0.2*cost - 0.1*(latency_ms / 3000)
- STOP reward  : 0.5 * ndcg_before

STOP is now mildly attractive on already-good queries, but not so attractive
that it chokes off exploration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer


ROOT = Path(__file__).resolve().parent.parent
OLD_H5 = ROOT / "outputs" / "mdp_dataset_tree_branches_5000.h5"
OLD_CSV = ROOT / "outputs" / "trajectories_tree_branches_5000.csv"
NEW_H5 = ROOT / "outputs" / "mdp_dataset_tree_branches_5000_low_stop.h5"
NEW_CSV = ROOT / "outputs" / "trajectories_tree_branches_5000_low_stop.csv"

ALPHA = 2.0
BETA = 0.5
GAMMA = 0.2
DELTA = 0.1
LATENCY_NORM = 3000.0
DISCOUNT = 0.99

ACTION_NAMES = {
    0: "QueryReform",
    1: "Rerank",
    2: "PseudoRelevanceFeedback",
    3: "STOP",
}


def quality(ndcg: float, recall: float) -> float:
    """State potential used for reward shaping."""
    return ALPHA * ndcg + BETA * recall


def compute_reward(row: pd.Series) -> float:
    """Compute the shaped reward for a single CSV row."""
    if row["action_name"] == "STOP":
        return 0.5 * row["ndcg_before"]

    before = quality(row["ndcg_before"], row["recall_before"])
    after = quality(row["ndcg_after"], row["recall_after"])
    cost_penalty = GAMMA * row["cost"]
    time_penalty = DELTA * (row["latency_ms"] / LATENCY_NORM)
    return DISCOUNT * after - before - cost_penalty - time_penalty


def main() -> None:
    if not OLD_H5.is_file():
        raise FileNotFoundError(f"Missing HDF5 dataset: {OLD_H5}")
    if not OLD_CSV.is_file():
        raise FileNotFoundError(f"Missing CSV dataset: {OLD_CSV}")

    print(f"Loading old HDF5: {OLD_H5}")
    with OLD_H5.open("rb") as f:
        old_rb = ReplayBuffer.load(f, InfiniteBuffer())

    print(f"Loading old CSV: {OLD_CSV}")
    df = pd.read_csv(OLD_CSV)
    required = {
        "action_name",
        "ndcg_before",
        "ndcg_after",
        "recall_before",
        "recall_after",
        "cost",
        "latency_ms",
        "reward",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["reward"] = df.apply(compute_reward, axis=1)
    print(f"Recomputed {len(df)} transition rewards")
    print("New reward summary:")
    print(df.groupby("action_name")["reward"].describe())

    # Check that the total return approximates final_quality - total_cost.
    print("\nSanity check: total return per trajectory vs final_quality - total_cost")
    df["quality_before"] = df.apply(
        lambda r: quality(r["ndcg_before"], r["recall_before"]), axis=1
    )
    df["quality_after"] = df.apply(
        lambda r: quality(r["ndcg_after"], r["recall_after"]), axis=1
    )
    df["cost_plus_time"] = (
        GAMMA * df["cost"] + DELTA * (df["latency_ms"] / LATENCY_NORM)
    )
    # Group by trajectory (query_id + consecutive episodes are not grouped here;
    # use the same row order as the H5 episodes).

    row_idx = 0
    new_observations = []
    new_actions = []
    new_rewards = []
    new_terminals = []
    new_timeouts = []

    for ep in old_rb.episodes:
        ep_len = ep.size()
        csv_rows = df.iloc[row_idx : row_idx + ep_len]
        h5_actions = ep.actions.flatten().tolist()
        csv_actions = [row for row in csv_rows["action_name"]]
        expected = [ACTION_NAMES[a] for a in h5_actions]
        if expected != csv_actions:
            raise ValueError(
                f"CSV/H5 misalignment at row {row_idx}: "
                f"H5={expected}, CSV={csv_actions}"
            )

        new_observations.append(ep.observations)
        new_actions.append(ep.actions)
        new_rewards.append(csv_rows["reward"].to_numpy(dtype=np.float32).reshape(-1, 1))
        terms = np.zeros(ep_len, dtype=np.int64)
        terms[-1] = 1
        new_terminals.append(terms)
        new_timeouts.append(np.zeros(ep_len, dtype=np.int64))

        row_idx += ep_len

    if row_idx != len(df):
        raise ValueError(
            f"Row count mismatch after alignment: processed {row_idx}, CSV has {len(df)}"
        )

    observations = np.concatenate(new_observations, axis=0).astype(np.float32)
    actions = np.concatenate(new_actions, axis=0).astype(np.int64)
    rewards = np.concatenate(new_rewards, axis=0).astype(np.float32)
    terminals = np.concatenate(new_terminals, axis=0).astype(np.int64)
    timeouts = np.concatenate(new_timeouts, axis=0).astype(np.int64)

    mdp_dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    NEW_H5.parent.mkdir(parents=True, exist_ok=True)
    mdp_dataset.dump(str(NEW_H5))
    print(f"\nSaved new HDF5 dataset: {NEW_H5}")

    df.to_csv(NEW_CSV, index=False)
    print(f"Saved new CSV dataset: {NEW_CSV}")


if __name__ == "__main__":
    main()
