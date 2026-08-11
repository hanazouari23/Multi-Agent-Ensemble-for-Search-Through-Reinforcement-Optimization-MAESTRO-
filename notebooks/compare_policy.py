import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path("..")
expert_path = root / "outputs" / "trajectories_expert_test_results_dev_50.csv"
cql_path    = root / "outputs" / "cql_stop_penalty_epoch10_dev_50.csv"

expert = pd.read_csv(expert_path)
cql = pd.read_csv(cql_path)

# Convert numeric columns
def to_numeric(df):
    for col in ["reward", "ndcg_before", "ndcg_after", "cost", "latency_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

expert = to_numeric(expert)
cql = to_numeric(cql)

# Aggregate per query
expert_reward = expert.groupby("query_id", as_index=False)["reward"].sum().rename(columns={"reward": "expert_reward"})
cql_reward = cql.groupby("query_id", as_index=False)["reward"].sum().rename(columns={"reward": "cql_reward"})

paired = expert_reward.merge(cql_reward, on="query_id", how="inner")
print("Matched queries:", len(paired))

# NDCG: first ndcg_before and last ndcg_after per query
expert_ndcg = (
    expert.groupby("query_id", as_index=False)
    .agg(expert_initial_ndcg=("ndcg_before", "first"), expert_final_ndcg=("ndcg_after", "last"))
)
cql_ndcg = (
    cql.groupby("query_id", as_index=False)
    .agg(cql_initial_ndcg=("ndcg_before", "first"), cql_final_ndcg=("ndcg_after", "last"))
)

paired = paired.merge(expert_ndcg, on="query_id", how="left").merge(cql_ndcg, on="query_id", how="left")

paired["expert_ndcg_gain"] = paired["expert_final_ndcg"] - paired["expert_initial_ndcg"]
paired["cql_ndcg_gain"] = paired["cql_final_ndcg"] - paired["cql_initial_ndcg"]

# Summary
print("\n=== Reward per query ===")
print(f"Expert mean total reward: {paired['expert_reward'].mean():.4f}")
print(f"CQL    mean total reward: {paired['cql_reward'].mean():.4f}")
print(f"CQL win rate (reward): {(paired['cql_reward'] > paired['expert_reward']).mean():.1%}")
print(f"Mean reward difference (CQL - Expert): {paired['cql_reward'].subtract(paired['expert_reward']).mean():.4f}")

print("\n=== NDCG@50 ===")
print(f"Expert mean initial NDCG: {paired['expert_initial_ndcg'].mean():.4f}")
print(f"Expert mean final   NDCG: {paired['expert_final_ndcg'].mean():.4f}")
print(f"Expert mean NDCG gain:    {paired['expert_ndcg_gain'].mean():.4f}")
print(f"CQL    mean initial NDCG: {paired['cql_initial_ndcg'].mean():.4f}")
print(f"CQL    mean final   NDCG: {paired['cql_final_ndcg'].mean():.4f}")
print(f"CQL    mean NDCG gain:    {paired['cql_ndcg_gain'].mean():.4f}")

# Plot reward scatter
plt.figure(figsize=(8, 8))
win = paired["cql_reward"] > paired["expert_reward"]
colors = np.where(win, "#2a9d8f", "#e76f51")
plt.scatter(paired["expert_reward"], paired["cql_reward"], c=colors, s=48, alpha=0.7, edgecolor="white", linewidth=0.5)

lo = min(paired["expert_reward"].min(), paired["cql_reward"].min())
hi = max(paired["expert_reward"].max(), paired["cql_reward"].max())
pad = max((hi - lo) * 0.05, 0.01)
plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
plt.xlim(lo - pad, hi + pad)
plt.ylim(lo - pad, hi + pad)
plt.xlabel("Expert total reward")
plt.ylabel("CQL total reward")
plt.title("Total reward per query: CQL vs Expert")
plt.text(0.03, 0.97, f"Matched: {len(paired)}\nCQL win rate: {win.mean():.1%}\nMean diff: {paired['cql_reward'].subtract(paired['expert_reward']).mean():.4f}",
         transform=plt.gca().transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
plt.tight_layout()
plt.savefig(r"..\outputs\cql_vs_expert_stop_penalty_dev_reward_scatter.png", dpi=150)
print("Saved plot: outputs/cql_vs_expert_stop_penalty_dev_reward_scatter.png")

paired.to_csv(r"..\outputs\cql_vs_expert_stop_penalty_dev_per_query.csv", index=False)
print("Saved per-query CSV: outputs/cql_vs_expert_stop_penalty_dev_per_query.csv")

# Display head
print("\nTop 10 largest differences (CQL - Expert):")
pair = paired.copy()
pair["diff"] = pair["cql_reward"] - pair["expert_reward"]
print(pair[["query_id", "expert_reward", "cql_reward", "diff", "expert_ndcg_gain", "cql_ndcg_gain"]].sort_values("diff", ascending=False).head(10).to_string(index=False))
