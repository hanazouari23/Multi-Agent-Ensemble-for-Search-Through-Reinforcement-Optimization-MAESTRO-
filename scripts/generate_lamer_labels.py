"""Generate a labeled dataset for LameR adaptive routing.

Runs BM25 + LameR on every query (ignoring the adaptive rule) and records
retrieval signals together with the ground-truth nDCG gain. The output CSV can
be used to train/evaluate a classifier that predicts when LameR helps.

The script is resumable: it skips query_ids already present in the output CSV.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Make ``src`` importable when running from the ``scripts/`` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.lamer import LameRAgent
from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config defaults (mirror notebooks/adaptive_lamer.ipynb) ──────────────────
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "notebooks" / "queries" / "topics.ms-marco-dev2.tsv"
DEFAULT_QRELS_PATH = PROJECT_ROOT / "notebooks" / "qrels" / "qrels.ms-marco-dev2.tsv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "lamer_labels.csv"

NDCG_K = 50
RECALL_K = 100
TOP_K_FOR_SIMILARITY = 20
TOP_K_INITIAL = 20
TOP_K_FINAL = 50
N_CANDIDATES = 5
MODERATE_SIM_THRESHOLD = 0.5

LLM_NAME = "google/gemma-4-E4B-it"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Data loading helpers ─────────────────────────────────────────────────────
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


def load_queries(queries_path: Path, num_queries: int = None) -> List[Tuple[str, str]]:
    """Load queries as ``[(query_id, query_text), ...]``."""
    queries = []
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header.
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                query_id, query_text = parts[0].strip(), parts[1].strip()
            else:
                query_id, query_text = str(len(queries)), parts[0].strip()
            queries.append((query_id, query_text))
            if num_queries is not None and len(queries) >= num_queries:
                break
    return queries


# ── Metric helpers ───────────────────────────────────────────────────────────
def _dcg(relevances: np.ndarray, k: int) -> float:
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size == 0:
        return 0.0
    positions = np.arange(2, relevances.size + 2)
    return float(np.sum(relevances / np.log2(positions)))


def normalize_doc_id(doc_id: str) -> str:
    """Strip segment suffix (e.g. 'doc#1' -> 'doc') to match qrels format."""
    return doc_id.split("#", 1)[0] if "#" in doc_id else doc_id


def deduplicate_doc_ids(doc_ids: List[str]) -> List[str]:
    """Normalize then deduplicate doc IDs."""
    deduped = []
    seen = set()
    for doc_id in doc_ids:
        normalized = normalize_doc_id(doc_id)
        if normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def compute_ndcg(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    ranked_docs = deduplicate_doc_ids(ranked_doc_ids)[:k]
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_docs]
    ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
    dcg = _dcg(np.array(gains, dtype=float), k)
    idcg = _dcg(np.array(ideal, dtype=float), k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 100) -> float:
    ranked_docs = deduplicate_doc_ids(ranked_doc_ids)[:k]
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_docs) & relevant) / len(relevant)


# ── Signal helpers ───────────────────────────────────────────────────────────
def query_passage_similarity_signals(
    query_text: str,
    doc_ids: List[str],
    corpus: Dict[str, str],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K_FOR_SIMILARITY,
    moderate_threshold: float = MODERATE_SIM_THRESHOLD,
) -> Dict[str, float]:
    """Return semantic-similarity signals between the query and top-k passages."""
    doc_ids = doc_ids[:top_k]
    passages = [corpus.get(doc_id, "").strip() for doc_id in doc_ids]
    passages = [p for p in passages if p]

    if not passages:
        return {
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
            "std_similarity": 0.0,
            "count_above_moderate": 0,
            "score_gap": 0.0,
        }

    query_embedding = embed_model.encode(query_text, convert_to_numpy=True)
    passage_embeddings = embed_model.encode(passages, convert_to_numpy=True)

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    passage_norms = passage_embeddings / (np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = passage_norms @ query_norm

    return {
        "mean_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities)),
        "count_above_moderate": int(np.sum(similarities >= moderate_threshold)),
        "score_gap": float(np.max(similarities) - np.min(similarities)),
    }


# ── Per-query label generation ───────────────────────────────────────────────
def process_query(
    query_id: str,
    query_text: str,
    retriever_func,
    embed_model: SentenceTransformer,
    lamer_agent: LameRAgent,
    qrels: Dict[str, Dict[str, int]],
) -> Dict:
    """Run BM25 + LameR for one query and return a labeled record."""
    qrels_for_query = qrels.get(query_id, {})

    # Baseline BM25
    bm25_start = time.time()
    bm25_doc_ids, bm25_scores, bm25_corpus = retriever_func(query_text, TOP_K_FINAL)
    bm25_elapsed = time.time() - bm25_start

    bm25_scores_arr = np.asarray(bm25_scores, dtype=float)
    if len(bm25_scores_arr) >= 2:
        last_idx = min(9, len(bm25_scores_arr) - 1)
        bm25_score_gap = float(bm25_scores_arr[0] - bm25_scores_arr[last_idx])
    else:
        bm25_score_gap = 0.0

    # Signals from BM25 retrieval
    signals = query_passage_similarity_signals(
        query_text=query_text,
        doc_ids=bm25_doc_ids,
        corpus=bm25_corpus,
        embed_model=embed_model,
    )

    # Always invoke LameR to collect ground-truth labels
    lamer_start = time.time()
    effects = lamer_agent.compute_effects({
        "query_text": query_text,
        "retriever": retriever_func,
        "top_k": TOP_K_FINAL,
    })
    lamer_elapsed = time.time() - lamer_start

    lamer_doc_ids = effects["new_doc_ids"]
    augmented_query = effects["new_query_text"]

    # Metrics
    bm25_ndcg = compute_ndcg(bm25_doc_ids, qrels_for_query, k=NDCG_K)
    lamer_ndcg = compute_ndcg(lamer_doc_ids, qrels_for_query, k=NDCG_K)
    bm25_recall = compute_recall(bm25_doc_ids, qrels_for_query, k=RECALL_K)
    lamer_recall = compute_recall(lamer_doc_ids, qrels_for_query, k=RECALL_K)

    return {
        "query_id": query_id,
        "query_text": query_text,
        "augmented_query": augmented_query,
        "bm25_doc_ids": ";".join(bm25_doc_ids),
        "lamer_doc_ids": ";".join(lamer_doc_ids),
        "bm25_ndcg": bm25_ndcg,
        "lamer_ndcg": lamer_ndcg,
        "ndcg_gain": lamer_ndcg - bm25_ndcg,
        "bm25_recall": bm25_recall,
        "lamer_recall": lamer_recall,
        "recall_gain": lamer_recall - bm25_recall,
        "bm25_latency_ms": bm25_elapsed * 1000,
        "lamer_latency_ms": lamer_elapsed * 1000,
        "bm25_score_gap": bm25_score_gap,
        "augmented_extra_tokens": len(augmented_query.split()) - len(query_text.split()),
        "lamer_cost": effects["cost"],
        **signals,
        "error": "",
    }


# ── Main batch loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a labeled dataset for LameR adaptive routing."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to TSV file with query_id and query_text.",
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=DEFAULT_QRELS_PATH,
        help="Path to TSV qrels file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the labeled records.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to label (default: all).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write checkpoint after this many new queries.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    logger.info("Initializing retriever")
    retriever_instance = Retriever(
        endpoint=os.getenv("RETRIEVAL_ENDPOINT"),
        username=os.getenv("MY_USERNAME"),
        password=os.getenv("MY_PASSWORD"),
        index_field="segment",
        top_k=TOP_K_FINAL,
    )
    retriever_func = create_retriever_callable(retriever_instance)

    logger.info("Initializing LameR agent: %s", LLM_NAME)
    lamer_agent = LameRAgent(
        embed_model=embed_model,
        n_candidates=N_CANDIDATES,
        top_k_initial=TOP_K_INITIAL,
        top_k_final=TOP_K_FINAL,
        model_name=LLM_NAME,
    )

    logger.info("Loading qrels from %s", args.qrels_path)
    qrels = load_qrels(args.qrels_path)

    logger.info("Loading queries from %s", args.queries_path)
    queries = load_queries(args.queries_path, num_queries=args.num_queries)
    logger.info("Loaded %d queries", len(queries))

    # Resume from existing output
    if args.output_csv.exists():
        logger.info("Found existing output; loading for resume: %s", args.output_csv)
        df_existing = pd.read_csv(args.output_csv)
        # Drop rows that had errors so they can be retried
        df_existing = df_existing[df_existing.get("error", "").astype(str) == ""]
        processed_ids = set(df_existing["query_id"].astype(str))
        records = df_existing.to_dict("records")
        logger.info("Resuming with %d already processed queries", len(records))
    else:
        processed_ids = set()
        records = []

    queries_to_run = [(qid, text) for qid, text in queries if str(qid) not in processed_ids]
    logger.info("Queries remaining: %d", len(queries_to_run))

    start_time = time.time()
    for i, (query_id, query_text) in enumerate(queries_to_run, start=1):
        logger.info(
            "[%d/%d] Query %s: %s...",
            i,
            len(queries_to_run),
            query_id,
            query_text[:60],
        )
        try:
            record = process_query(
                query_id=query_id,
                query_text=query_text,
                retriever_func=retriever_func,
                embed_model=embed_model,
                lamer_agent=lamer_agent,
                qrels=qrels,
            )
            records.append(record)
        except Exception as e:
            logger.error("Failed on query %s: %s", query_id, e)
            traceback.print_exc()
            records.append({
                "query_id": query_id,
                "query_text": query_text,
                "error": str(e),
            })

        # Checkpoint periodically
        if i % args.checkpoint_every == 0 or i == len(queries_to_run):
            df = pd.DataFrame(records)
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            logger.info(
                "Checkpoint saved: %d records -> %s",
                len(df),
                args.output_csv,
            )

    elapsed = time.time() - start_time
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    valid_df = df[df.get("error", "").astype(str) == ""]
    logger.info("=== Done ===")
    logger.info("Total queries: %d", len(df))
    logger.info("Successful: %d", len(valid_df))
    logger.info("Failed: %d", len(df) - len(valid_df))
    logger.info("Elapsed time: %.1f s (%.2f s/query)", elapsed, elapsed / max(len(queries_to_run), 1))

    if len(valid_df) > 0:
        logger.info("BM25  nDCG@%d: %.4f", NDCG_K, valid_df["bm25_ndcg"].mean())
        logger.info("LameR nDCG@%d: %.4f", NDCG_K, valid_df["lamer_ndcg"].mean())
        logger.info("Mean nDCG gain: %+.4f", valid_df["ndcg_gain"].mean())
        logger.info("Win rate (gain > 0): %.1f%%", 100 * (valid_df["ndcg_gain"] > 0).mean())
        logger.info("Hurt rate (gain < 0): %.1f%%", 100 * (valid_df["ndcg_gain"] < 0).mean())


if __name__ == "__main__":
    main()
