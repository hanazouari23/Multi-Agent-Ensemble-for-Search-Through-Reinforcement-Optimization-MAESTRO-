"""Add BM25 retrieval-score features to an existing lamer_labels.csv.

This script only calls the OpenSearch BM25 retriever; it does **not** call the
LameR LLM. It recomputes the BM25 result for every query in the labels file,
extracts score-distribution statistics, and writes them back as extra columns.

The script is resumable: progress is saved to a temporary CSV next to the labels
file, so rerunning it after an interruption will skip already-processed queries.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import kurtosis, skew

# Suppress noisy HTTPS warnings from the retriever.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Make ``src`` importable when running from the ``scripts/`` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LABELS_CSV = PROJECT_ROOT / "outputs" / "lamer_labels.csv"
TOP_K = 50

# Columns this script will add (or overwrite) in the CSV.
NEW_FEATURE_COLUMNS = [
    "bm25_num_retrieved",
    "bm25_top1_score",
    "bm25_top5_score",
    "bm25_top10_score",
    "bm25_top20_score",
    "bm25_top50_score",
    "bm25_score_mean",
    "bm25_score_std",
    "bm25_score_skew",
    "bm25_score_kurtosis",
    "bm25_score_sum",
    "bm25_score_range",
    "bm25_gap_1_5",
    "bm25_gap_1_10",
    "bm25_gap_1_20",
    "bm25_gap_1_50",
    "bm25_gap_5_10",
    "bm25_gap_10_50",
    "bm25_ratio_1_5",
    "bm25_ratio_1_10",
    "bm25_ratio_1_50",
    "bm25_ratio_5_10",
    "bm25_score_entropy",
]


def compute_bm25_score_features(scores: List[float], top_k: int = TOP_K) -> Dict[str, float]:
    """Return statistics of the BM25 score distribution for one query."""
    scores_arr = np.asarray(scores, dtype=float)[:top_k]
    n = len(scores_arr)
    feats = {"bm25_num_retrieved": int(n)}

    if n == 0:
        for col in NEW_FEATURE_COLUMNS:
            if col not in feats:
                feats[col] = 0.0
        return feats

    def at(rank: int) -> float:
        return float(scores_arr[min(rank, n - 1)])

    feats["bm25_top1_score"] = at(0)
    feats["bm25_top5_score"] = at(4)
    feats["bm25_top10_score"] = at(9)
    feats["bm25_top20_score"] = at(19)
    feats["bm25_top50_score"] = float(scores_arr[-1])

    feats["bm25_score_mean"] = float(np.mean(scores_arr))
    feats["bm25_score_std"] = float(np.std(scores_arr))
    feats["bm25_score_skew"] = float(skew(scores_arr)) if feats["bm25_score_std"] > 1e-10 else 0.0
    feats["bm25_score_kurtosis"] = float(kurtosis(scores_arr)) if feats["bm25_score_std"] > 1e-10 else 0.0
    feats["bm25_score_sum"] = float(np.sum(scores_arr))
    feats["bm25_score_range"] = float(scores_arr.max() - scores_arr.min())

    feats["bm25_gap_1_5"] = at(0) - at(4)
    feats["bm25_gap_1_10"] = at(0) - at(9)
    feats["bm25_gap_1_20"] = at(0) - at(19)
    feats["bm25_gap_1_50"] = at(0) - float(scores_arr[-1])
    feats["bm25_gap_5_10"] = at(4) - at(9)
    feats["bm25_gap_10_50"] = at(9) - float(scores_arr[-1])

    eps = 1e-10
    feats["bm25_ratio_1_5"] = at(0) / (at(4) + eps)
    feats["bm25_ratio_1_10"] = at(0) / (at(9) + eps)
    feats["bm25_ratio_1_50"] = at(0) / (float(scores_arr[-1]) + eps)
    feats["bm25_ratio_5_10"] = at(4) / (at(9) + eps)

    # Entropy of the softmax distribution over scores: peaked = low entropy.
    exp_scores = np.exp(scores_arr - scores_arr.max())
    probs = exp_scores / exp_scores.sum()
    entropy = -np.sum(probs * np.log(probs + eps))
    feats["bm25_score_entropy"] = float(entropy)

    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Enrich lamer_labels.csv with BM25 score-distribution features."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Path to lamer_labels.csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of BM25 results to use for score statistics.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save progress after this many queries.",
    )
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels_csv}")

    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Loading labels from %s", args.labels_csv)
    df = pd.read_csv(args.labels_csv)
    logger.info("Loaded %d rows", len(df))

    # Back up the original file once.
    backup_path = args.labels_csv.with_suffix(".csv.bak")
    if not backup_path.exists():
        shutil.copy2(args.labels_csv, backup_path)
        logger.info("Created backup: %s", backup_path)

    # Progress file: stores query_id + new features for already-processed rows.
    progress_path = args.labels_csv.with_suffix(".csv.enrich_progress")
    if progress_path.exists():
        progress_df = pd.read_csv(progress_path)
        progress_df["query_id"] = progress_df["query_id"].astype(str)
        processed_ids = set(progress_df["query_id"])
        logger.info("Resuming: %d queries already processed", len(processed_ids))
    else:
        progress_df = pd.DataFrame({"query_id": pd.Series(dtype="object")})
        for col in NEW_FEATURE_COLUMNS:
            progress_df[col] = pd.Series(dtype="float64")
        processed_ids = set()

    logger.info("Initializing BM25 retriever")
    retriever_instance = Retriever(
        endpoint=os.getenv("RETRIEVAL_ENDPOINT"),
        username=os.getenv("MY_USERNAME"),
        password=os.getenv("MY_PASSWORD"),
        index_field="segment",
        top_k=args.top_k,
    )
    retriever_func = create_retriever_callable(retriever_instance)

    new_rows = []
    start_time = time.time()
    remaining = 0
    for _, row in df.iterrows():
        query_id = str(row.get("query_id", ""))
        query_text = row.get("query_text", "")

        if query_id in processed_ids:
            continue

        remaining += 1
        try:
            _, scores, _ = retriever_func(query_text, args.top_k)
            feats = compute_bm25_score_features(scores.tolist(), top_k=args.top_k)
        except Exception as e:
            logger.error("BM25 retrieval failed for query %s: %s", query_id, e)
            feats = {c: 0.0 for c in NEW_FEATURE_COLUMNS}
            feats["bm25_num_retrieved"] = 0

        feats["query_id"] = query_id
        new_rows.append(feats)

        if len(new_rows) % args.checkpoint_every == 0:
            chunk_df = pd.DataFrame(new_rows)
            progress_df = pd.concat([progress_df, chunk_df], ignore_index=True)
            # Keep only the latest record per query_id in the progress file.
            progress_df = progress_df.drop_duplicates(subset=["query_id"], keep="last")
            progress_df.to_csv(progress_path, index=False)
            elapsed = time.time() - start_time
            logger.info(
                "Processed +%d queries (%.1f s, %.2f s/query since start)",
                len(new_rows),
                elapsed,
                elapsed / max(len(new_rows), 1),
            )
            # Reset the in-memory buffer so checkpoints do not double-save rows.
            new_rows = []

    # Final progress save.
    if new_rows:
        chunk_df = pd.DataFrame(new_rows)
        progress_df = pd.concat([progress_df, chunk_df], ignore_index=True)
    progress_df = progress_df.drop_duplicates(subset=["query_id"], keep="last")
    progress_df.to_csv(progress_path, index=False)

    # Merge progress back into the labels dataframe and save.
    df["query_id"] = df["query_id"].astype(str)
    progress_df["query_id"] = progress_df["query_id"].astype(str)
    df = df.drop(columns=[c for c in NEW_FEATURE_COLUMNS if c in df.columns], errors="ignore")
    df = df.merge(progress_df[["query_id"] + NEW_FEATURE_COLUMNS], on="query_id", how="left")
    # If any query failed, fill missing new features with 0.
    for col in NEW_FEATURE_COLUMNS:
        df[col] = df[col].fillna(0.0)

    df.to_csv(args.labels_csv, index=False)
    logger.info("Saved enriched labels to %s", args.labels_csv)
    logger.info("Added columns: %s", ", ".join(NEW_FEATURE_COLUMNS))
    logger.info("Processed %d new queries; progress kept in %s", len(new_rows), progress_path)


if __name__ == "__main__":
    main()
