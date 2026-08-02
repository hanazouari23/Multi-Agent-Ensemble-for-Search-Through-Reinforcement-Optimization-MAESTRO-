"""Visualize per-query comparison between the expert baseline and trained CQL policy.

Reads `outputs/cql_vs_expert_trec_dl_combined_per_query.csv` and writes a set of
publication-ready figures to `outputs/figures/`.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "outputs" / "cql_vs_expert_trec_dl_combined_per_query.csv"
OUTPUT_DIR = ROOT / "outputs" / "figures"


def load_paired_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "query_id",
        "expert_reward",
        "cql_reward",
        "expert_initial_ndcg",
        "expert_final_ndcg",
        "cql_initial_ndcg",
        "cql_final_ndcg",
        "expert_ndcg_gain",
        "cql_ndcg_gain",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["reward_diff"] = df["cql_reward"] - df["expert_reward"]
    df["ndcg_gain_diff"] = df["cql_ndcg_gain"] - df["expert_ndcg_gain"]
    df["final_ndcg_diff"] = df["cql_final_ndcg"] - df["expert_final_ndcg"]

    return {
        "n_queries": len(df),
        "expert_reward_mean": df["expert_reward"].mean(),
        "cql_reward_mean": df["cql_reward"].mean(),
        "reward_diff_mean": df["reward_diff"].mean(),
        "reward_diff_median": df["reward_diff"].median(),
        "cql_win_rate_reward": (df["cql_reward"] > df["expert_reward"]).mean(),
        "expert_win_rate_reward": (df["expert_reward"] > df["cql_reward"]).mean(),
        "expert_ndcg_gain_mean": df["expert_ndcg_gain"].mean(),
        "cql_ndcg_gain_mean": df["cql_ndcg_gain"].mean(),
        "ndcg_gain_diff_mean": df["ndcg_gain_diff"].mean(),
        "cql_win_rate_ndcg_gain": (df["cql_ndcg_gain"] > df["expert_ndcg_gain"]).mean(),
        "expert_final_ndcg_mean": df["expert_final_ndcg"].mean(),
        "cql_final_ndcg_mean": df["cql_final_ndcg"].mean(),
        "final_ndcg_diff_mean": df["final_ndcg_diff"].mean(),
    }


def plot_reward_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    wins = df["cql_reward"] > df["expert_reward"]
    colors = np.where(wins, "#2a9d8f", "#e76f51")

    ax.scatter(
        df["expert_reward"],
        df["cql_reward"],
        c=colors,
        s=64,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    lo = min(df["expert_reward"].min(), df["cql_reward"].min())
    hi = max(df["expert_reward"].max(), df["cql_reward"].max())
    pad = max((hi - lo) * 0.05, 0.01)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2, label="Equal reward")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Expert total reward")
    ax.set_ylabel("CQL total reward")
    ax.set_title("Total reward per query")
    ax.legend(loc="lower right")


def plot_reward_diff_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    diff = df["cql_reward"] - df["expert_reward"]
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    sns.histplot(diff, kde=True, bins=20, color="#4c78a8", ax=ax)
    ax.axvline(diff.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean ({diff.mean():.4f})")
    ax.axvline(diff.median(), color="green", linestyle="--", linewidth=1.2, label=f"Median ({diff.median():.4f})")
    ax.set_xlabel("Reward difference (CQL - Expert)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Distribution of reward differences")
    ax.legend()


def plot_ndcg_gain_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    wins = df["cql_ndcg_gain"] > df["expert_ndcg_gain"]
    colors = np.where(wins, "#2a9d8f", "#e76f51")

    ax.scatter(
        df["expert_ndcg_gain"],
        df["cql_ndcg_gain"],
        c=colors,
        s=64,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    lo = min(df["expert_ndcg_gain"].min(), df["cql_ndcg_gain"].min())
    hi = max(df["expert_ndcg_gain"].max(), df["cql_ndcg_gain"].max())
    pad = max((hi - lo) * 0.05, 0.01)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2, label="Equal gain")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Expert NDCG@50 gain")
    ax.set_ylabel("CQL NDCG@50 gain")
    ax.set_title("NDCG@50 gain per query")
    ax.legend(loc="lower right")


def plot_ndcg_gain_diff_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    diff = df["cql_ndcg_gain"] - df["expert_ndcg_gain"]
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    sns.histplot(diff, kde=True, bins=20, color="#4c78a8", ax=ax)
    ax.axvline(diff.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean ({diff.mean():.4f})")
    ax.axvline(diff.median(), color="green", linestyle="--", linewidth=1.2, label=f"Median ({diff.median():.4f})")
    ax.set_xlabel("NDCG@50 gain difference (CQL - Expert)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Distribution of NDCG@50 gain differences")
    ax.legend()


def plot_final_ndcg_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    wins = df["cql_final_ndcg"] > df["expert_final_ndcg"]
    colors = np.where(wins, "#2a9d8f", "#e76f51")

    ax.scatter(
        df["expert_final_ndcg"],
        df["cql_final_ndcg"],
        c=colors,
        s=64,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    lo = min(df["expert_final_ndcg"].min(), df["cql_final_ndcg"].min())
    hi = max(df["expert_final_ndcg"].max(), df["cql_final_ndcg"].max())
    pad = max((hi - lo) * 0.05, 0.01)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2, label="Equal NDCG")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Expert final NDCG@50")
    ax.set_ylabel("CQL final NDCG@50")
    ax.set_title("Final NDCG@50 per query")
    ax.legend(loc="lower right")


def plot_top_differences(df: pd.DataFrame, ax: plt.Axes, top_n: int = 15) -> None:
    diff = df["cql_reward"] - df["expert_reward"]
    top_improved = diff.sort_values(ascending=False).head(top_n)
    top_degraded = diff.sort_values().head(top_n)

    plot_df = pd.concat([
        top_improved.reset_index().rename(columns={"index": "query_id", "cql_reward - expert_reward": "diff"}),
        top_degraded.reset_index().rename(columns={"index": "query_id", "cql_reward - expert_reward": "diff"}),
    ])
    # Rebuild properly
    temp = pd.DataFrame({
        "query_id": df["query_id"].astype(str),
        "diff": diff,
    })
    improved = temp.nlargest(top_n, "diff").copy()
    improved["category"] = "CQL better"
    degraded = temp.nsmallest(top_n, "diff").copy()
    degraded["category"] = "Expert better"
    plot_df = pd.concat([improved, degraded])

    palette = {"CQL better": "#2a9d8f", "Expert better": "#e76f51"}
    sns.barplot(data=plot_df, y="query_id", x="diff", hue="category", palette=palette, ax=ax, dodge=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Reward difference (CQL - Expert)")
    ax.set_ylabel("Query ID")
    ax.set_title(f"Top {top_n} largest reward differences per side")
    ax.legend(loc="lower right")


def plot_summary_table(summary: dict, ax: plt.Axes) -> None:
    ax.axis("off")
    rows = [
        ("Matched queries", f"{summary['n_queries']}"),
        ("", ""),
        ("Reward", ""),
        ("  Expert mean", f"{summary['expert_reward_mean']:.4f}"),
        ("  CQL mean", f"{summary['cql_reward_mean']:.4f}"),
        ("  Mean difference (CQL - Expert)", f"{summary['reward_diff_mean']:.4f}"),
        ("  Median difference", f"{summary['reward_diff_median']:.4f}"),
        ("  CQL win rate", f"{summary['cql_win_rate_reward']:.1%}"),
        ("", ""),
        ("NDCG@50 gain", ""),
        ("  Expert mean", f"{summary['expert_ndcg_gain_mean']:.4f}"),
        ("  CQL mean", f"{summary['cql_ndcg_gain_mean']:.4f}"),
        ("  Mean difference (CQL - Expert)", f"{summary['ndcg_gain_diff_mean']:.4f}"),
        ("  CQL win rate", f"{summary['cql_win_rate_ndcg_gain']:.1%}"),
        ("", ""),
        ("Final NDCG@50", ""),
        ("  Expert mean", f"{summary['expert_final_ndcg_mean']:.4f}"),
        ("  CQL mean", f"{summary['cql_final_ndcg_mean']:.4f}"),
        ("  Mean difference (CQL - Expert)", f"{summary['final_ndcg_diff_mean']:.4f}"),
    ]
    table = ax.table(
        cellText=[[r[0], r[1]] for r in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colColours=["#f0f0f0", "#f0f0f0"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title("Summary statistics", pad=10)


def create_summary_figure(df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    ax_reward_scatter = fig.add_subplot(gs[0, 0])
    ax_reward_hist = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_ndcg_gain_scatter = fig.add_subplot(gs[1, 0])
    ax_ndcg_gain_hist = fig.add_subplot(gs[1, 1])
    ax_final_ndcg = fig.add_subplot(gs[1, 2])
    ax_top_diff = fig.add_subplot(gs[2, :])

    plot_reward_scatter(df, ax_reward_scatter)
    plot_reward_diff_histogram(df, ax_reward_hist)
    plot_summary_table(summary, ax_summary)
    plot_ndcg_gain_scatter(df, ax_ndcg_gain_scatter)
    plot_ndcg_gain_diff_histogram(df, ax_ndcg_gain_hist)
    plot_final_ndcg_scatter(df, ax_final_ndcg)
    plot_top_differences(df, ax_top_diff, top_n=15)

    fig.suptitle("CQL vs Expert baseline: per-query comparison (TREC DL combined)", fontsize=16, y=0.995)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved summary figure: {output_path}")


def create_individual_figures(df: pd.DataFrame, output_dir: Path) -> None:
    # Reward scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_reward_scatter(df, ax)
    win_rate = (df["cql_reward"] > df["expert_reward"]).mean()
    mean_diff = (df["cql_reward"] - df["expert_reward"]).mean()
    ax.text(
        0.03,
        0.97,
        f"Matched: {len(df)}\nCQL win rate: {win_rate:.1%}\nMean diff: {mean_diff:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.savefig(output_dir / "cql_vs_expert_reward_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # NDCG gain scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_ndcg_gain_scatter(df, ax)
    win_rate = (df["cql_ndcg_gain"] > df["expert_ndcg_gain"]).mean()
    mean_diff = (df["cql_ndcg_gain"] - df["expert_ndcg_gain"]).mean()
    ax.text(
        0.03,
        0.97,
        f"Matched: {len(df)}\nCQL win rate: {win_rate:.1%}\nMean gain diff: {mean_diff:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.savefig(output_dir / "cql_vs_expert_ndcg_gain_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Final NDCG scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_final_ndcg_scatter(df, ax)
    win_rate = (df["cql_final_ndcg"] > df["expert_final_ndcg"]).mean()
    mean_diff = (df["cql_final_ndcg"] - df["expert_final_ndcg"]).mean()
    ax.text(
        0.03,
        0.97,
        f"Matched: {len(df)}\nCQL win rate: {win_rate:.1%}\nMean final NDCG diff: {mean_diff:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.savefig(output_dir / "cql_vs_expert_final_ndcg_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Reward difference histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_reward_diff_histogram(df, ax)
    fig.savefig(output_dir / "cql_vs_expert_reward_diff_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # NDCG gain difference histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_ndcg_gain_diff_histogram(df, ax)
    fig.savefig(output_dir / "cql_vs_expert_ndcg_gain_diff_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Top differences
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_top_differences(df, ax, top_n=15)
    fig.savefig(output_dir / "cql_vs_expert_top_reward_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved individual figures to {output_dir}")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Expected per-query CSV not found: {INPUT_CSV}\n"
            "Run notebooks/compare_policy.py first to generate it."
        )

    df = load_paired_csv(INPUT_CSV)
    summary = compute_summary(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== CQL vs Expert per-query summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print()

    create_summary_figure(df, summary, OUTPUT_DIR / "cql_vs_expert_summary.png")
    create_individual_figures(df, OUTPUT_DIR)

    # Save ranked list
    ranked = df.copy()
    ranked["reward_diff"] = ranked["cql_reward"] - ranked["expert_reward"]
    ranked["ndcg_gain_diff"] = ranked["cql_ndcg_gain"] - ranked["expert_ndcg_gain"]
    ranked = ranked.sort_values("reward_diff", ascending=False)
    ranked.to_csv(OUTPUT_DIR / "cql_vs_expert_ranked_by_reward_diff.csv", index=False)
    print(f"Saved ranked CSV: {OUTPUT_DIR / 'cql_vs_expert_ranked_by_reward_diff.csv'}")


if __name__ == "__main__":
    main()
