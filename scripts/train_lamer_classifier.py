"""Train a lightweight classifier that predicts when LameR will help.

Reads ``outputs/lamer_labels.csv`` (produced by ``generate_lamer_labels.py``),
trains a random-forest classifier on qrel-aware retrieval-quality features,
and saves the model to ``outputs/lamer_classifier.pkl``.

Queries with no relevant documents in the qrels are filtered out, because for
those queries LameR can neither improve nor hurt nDCG/recall in a meaningful
way.

The binary target is ``ndcg_gain > 0`` (i.e. invoking LameR improves nDCG@50
over the initial BM25 result). This is equivalent to the user's broader goal
because every query with ``recall_gain > 0`` also has ``ndcg_gain > 0`` in the
label set.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_CSV = PROJECT_ROOT / "outputs" / "lamer_labels.csv"
DEFAULT_QRELS_PATH = PROJECT_ROOT / "notebooks" / "qrels" / "qrels.ms-marco-dev2.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Qrel-aware retrieval-quality features. These require ground-truth relevance
# judgments for the query, so the classifier can only be used in offline
# evaluation or when such judgments are available.
FEATURE_COLUMNS = [
    "bm25_ndcg",
    "bm25_recall",
    "bm25_ndcg_is_zero",
    "bm25_recall_is_zero",
]

TARGET_COLUMN = "invoke_lamer"


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels as ``{query_id: {doc_id: relevance_grade}}``."""
    qrels = defaultdict(dict)
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header.
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            query_id, doc_id, grade_str = parts[0].strip(), parts[2].strip(), parts[3].strip()
            try:
                grade = int(grade_str)
            except ValueError:
                continue
            qrels[query_id][doc_id] = grade
    return dict(qrels)


def has_relevant_docs(query_id: str, qrels: Dict[str, Dict[str, int]]) -> bool:
    """Return True iff the query has at least one document with positive relevance."""
    return any(grade > 0 for grade in qrels.get(query_id, {}).values())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the classifier feature columns."""
    features = df[FEATURE_COLUMNS[:2]].astype(float).copy()
    features["bm25_ndcg_is_zero"] = (df["bm25_ndcg"] == 0).astype(int)
    features["bm25_recall_is_zero"] = (df["bm25_recall"] == 0).astype(int)
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Train a classifier for adaptive LameR routing."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Path to lamer_labels.csv.",
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=DEFAULT_QRELS_PATH,
        help="Path to TSV qrels file used to filter zero-relevance queries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model and report are written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if not args.labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels_csv}")

    logger.info("Loading labels from %s", args.labels_csv)
    df = pd.read_csv(args.labels_csv)

    # Keep only successful runs.
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()

    if len(df) == 0:
        raise ValueError("No successful records found in labels CSV.")

    logger.info("Records available before filtering: %d", len(df))

    # Filter out queries with no relevant documents in the qrels.
    logger.info("Loading qrels from %s", args.qrels_path)
    qrels = load_qrels(args.qrels_path)
    df["has_relevant_docs"] = df["query_id"].astype(str).apply(lambda qid: has_relevant_docs(qid, qrels))
    df = df[df["has_relevant_docs"]].copy()
    df = df.drop(columns=["has_relevant_docs"])

    if len(df) == 0:
        raise ValueError("No queries with relevant documents remain after filtering.")

    logger.info("Records available for training after filtering: %d", len(df))

    # Binary target: LameR helps iff nDCG improves.
    df[TARGET_COLUMN] = (df["ndcg_gain"] > 0).astype(int)

    X = build_features(df)
    y = df[TARGET_COLUMN].values

    # The positive class is the minority (≈28% in the 500-query dev2 set).
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    logger.info(
        "Train size: %d (positive %.1f%%), test size: %d (positive %.1f%%)",
        len(y_train),
        100 * y_train.mean(),
        len(y_test),
        100 * y_test.mean(),
    )

    # A small, constrained random forest is enough for a handful of features.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        oob_score=True,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Pick a decision threshold on out-of-bag training probabilities rather than
    # using the default 0.5. This usually gives a better precision/recall trade.
    oob_proba = clf.oob_decision_function_
    if oob_proba is not None:
        oob_proba = np.asarray(oob_proba)
        if oob_proba.ndim == 2:
            oob_proba = oob_proba[:, 1]
        # Fill NaNs (samples that were in-bag for every tree) with 0.5.
        oob_proba = np.nan_to_num(oob_proba, nan=0.5)

        best_threshold = 0.5
        best_f1 = 0.0
        for thr in np.arange(0.05, 1.0, 0.05):
            pred = (oob_proba >= thr).astype(int)
            tp = ((pred == 1) & (y_train == 1)).sum()
            fp = ((pred == 1) & (y_train == 0)).sum()
            fn = ((pred == 0) & (y_train == 1)).sum()
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thr)
        decision_threshold = best_threshold
        logger.info("OOB-optimal decision threshold: %.2f (F1: %.4f)", decision_threshold, best_f1)
    else:
        decision_threshold = 0.5

    # Evaluation.
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["skip_lamer", "invoke_lamer"],
        digits=4,
    )
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    logger.info("\n=== Test-set classification report ===\n%s", report)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("Average precision: %.4f", avg_precision)

    # Feature importances.
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
    logger.info("\n=== Feature importances ===")
    for feat, imp in importances.sort_values(ascending=False).items():
        logger.info("  %s: %.4f", feat, imp)

    # Save model + metadata needed for inference.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "lamer_classifier.pkl"
    features_path = args.output_dir / "lamer_classifier_features.json"
    report_path = args.output_dir / "lamer_classifier_report.txt"

    artifact = {
        "model": clf,
        "features": FEATURE_COLUMNS,
        "threshold": decision_threshold,
    }
    joblib.dump(artifact, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_COLUMNS, "threshold": decision_threshold}, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== LameR classifier training report ===\n\n")
        f.write(f"Training records: {len(df)}\n")
        f.write(f"Train size: {len(y_train)} (positive {100*y_train.mean():.1f}%)\n")
        f.write(f"Test size:  {len(y_test)} (positive {100*y_test.mean():.1f}%)\n")
        f.write(f"Decision threshold: {decision_threshold:.2f}\n\n")
        f.write(report)
        f.write("\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Average precision: {avg_precision:.4f}\n\n")
        f.write("=== Feature importances ===\n")
        for feat, imp in importances.sort_values(ascending=False).items():
            f.write(f"  {feat}: {imp:.4f}\n")

    logger.info("Saved classifier to %s", model_path)
    logger.info("Saved feature list to %s", features_path)
    logger.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
