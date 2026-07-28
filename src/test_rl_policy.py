from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import d3rlpy
import numpy as np

from .utils.retriever import Retriever, create_retriever_callable
from .simulation import Simulation, ACTION_STOP, ACTION_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "discrete_cql_policy.d3"
# QUERIES_PATH = PROJECT_ROOT / "notebooks" /"queries"/ "combined_trec_dl_queries.tsv"
QUERIES_PATH = PROJECT_ROOT / "notebooks" /"queries"/ "topics.ms-marco-dev2.tsv"
QRELS_PATH = PROJECT_ROOT / "notebooks" / "qrels" / "qrels.ms-marco-dev2.tsv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "cql_test_results2.csv"

TOP_K = 50
NUM_QUERIES_TO_TEST = 50
N_ACTIONS = 4

from .main import (
    load_qrels,
    load_queries,
)

@dataclass
class QueryRetrieval:
    query_id: str
    query_text: str
    doc_ids: list[str]
    doc_scores: np.ndarray
    corpus_data: dict[str, str]


@dataclass
class TestState:
    query_id: str
    query_text: str
    doc_ids: list[str]
    doc_scores: np.ndarray
    corpus_data: dict[str, str]
    state: np.ndarray


# def load_queries(tsv_path: Path) -> list[tuple[str, str]]:
#     """Load TSV rows with headers: query_id and query_text."""
#     if not tsv_path.is_file():
#         raise FileNotFoundError(f"Query TSV does not exist: {tsv_path}")

#     queries = []

#     with tsv_path.open("r", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(f, delimiter="\t")

#         required_columns = {"query_id", "query_text"}
#         if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
#             raise ValueError(
#                 "TSV must contain headers exactly including: "
#                 "'query_id' and 'query_text'. "
#                 f"Found: {reader.fieldnames}"
#             )

#         for row in reader:
#             query_id = row["query_id"].strip()
#             query_text = row["query_text"].strip()

#             if query_id and query_text:
#                 queries.append((query_id, query_text))

#     return queries


def batch_retrieve(
    queries: list[tuple[str, str]],
    num_queries: int,
    retriever_func: callable,
) -> list[QueryRetrieval]:
    """Retrieve the initial top-50 ranking for each selected query."""
    results = []

    for query_id, query_text in queries[:num_queries]:
        doc_ids, doc_scores, corpus_data = retriever_func(query_text, top_k=TOP_K)

        if not doc_ids:
            print(f"Skipping {query_id}: no documents retrieved.")
            continue

        results.append(
            QueryRetrieval(
                query_id=query_id,
                query_text=query_text,
                doc_ids=doc_ids,
                doc_scores=doc_scores,
                corpus_data=corpus_data,
            )
        )

    return results


def generate_test_states(
    retrieved_queries: list[QueryRetrieval],
    simulation: Simulation,
    qrels: dict[str, dict[str, int]]
) -> list[TestState]:
    """Create the initial state, equivalent to state s_0 in an episode."""
    test_states = []

    for item in retrieved_queries:
        qrels_for_query = qrels.get(item.query_id, {})
        query_length = np.float32(len(item.query_text.split()))
        query_emb = simulation.encoder.encode(
            item.query_text, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        initial_ndcg = Simulation.compute_ndcg(
            ranked_doc_ids=item.doc_ids,
            qrels=qrels_for_query,
            k=50
        )
        initial_recall = Simulation.compute_recall(
            ranked_doc_ids=item.doc_ids,
            qrels=qrels_for_query,
            k=50,
        )

        initial_state = simulation.build_state(
            query=item.query_text,
            docids=item.doc_ids,
            docscores=item.doc_scores,
            step=0,
            last_action_agent=[False, False, False],
            previous_docids=None,
            original_query_embedding=query_emb,
            elapsed_ms=0.0,
            cumulative_cost=0.0,
            query_length=query_length,
            query_embedding=query_emb,
        )

        test_states.append(
            TestState(
                query_id=item.query_id,
                query_text=item.query_text,
                doc_ids=item.doc_ids,
                doc_scores=item.doc_scores,
                corpus_data=item.corpus_data,
                state=initial_state,
            )
        )

    return test_states


def select_valid_action(
    cql,
    state: np.ndarray,
    valid_action_mask: np.ndarray,
) -> int:
    """
    Choose the highest-Q valid action.

    This explicitly prevents selecting an invalid previously-used agent.
    The mask is already part of your 399-D state, but applying it again
    at inference guarantees invalid actions cannot be selected.
    """
    state_batch = state.reshape(1, 399).astype(np.float32)

    candidate_actions = np.arange(N_ACTIONS, dtype=np.int64)
    repeated_states = np.repeat(state_batch, N_ACTIONS, axis=0)

    q_values = cql.predict_value(repeated_states, candidate_actions)
    q_values = np.asarray(q_values).reshape(-1)

    masked_q_values = np.where(valid_action_mask.astype(bool), q_values, -np.inf)

    if not np.isfinite(masked_q_values).any():
        return ACTION_STOP

    return int(np.argmax(masked_q_values))


def _state_features(state: np.ndarray) -> dict[str, Any]:
    """Extract the non-embedding, human-readable features from a state vector."""
    return {
        "query_length": float(state[0]),
        "score_spread": float(state[385]),
        "score_entropy": float(state[386]),
        "step": float(state[390]),
        "rank_overlap": float(state[391]),
        "query_drift": float(state[392]),
    }


def test_batch_states(
    initial_states: list[TestState],
    simulation: Simulation,
    cql,
    qrels: dict[str, dict[str, int]],
    output_csv_path: Path,
) -> list[dict[str, Any]]:
    """
    Run a greedy-policy rollout for every initial query state.

    Exports one CSV row per action taken by the policy. The output contains:
    query ID, step, selected action, NDCG/recall changes, action cost,
    elapsed time, reward, and termination status.
    """
    rows: list[dict[str, Any]] = []

    for item in initial_states:
        query_id = item.query_id
        current_query = item.query_text
        doc_ids = item.doc_ids
        doc_scores = item.doc_scores
        corpus_data = item.corpus_data
        state = item.state.astype(np.float32)

        qrels_for_query = qrels.get(query_id, {})

        last_action_agent = [False, False, False]
        cumulative_cost = 0.0
        cumulative_latency_ms = 0.0

        # Cache the original query embedding for query_drift.
        original_query_emb = query_emb.copy()
        # The initial ranking has no predecessor, so rank_overlap will be 0.
        previous_docids: list[str] | None = None

        current_ndcg = Simulation.compute_ndcg(
            ranked_doc_ids=doc_ids,
            qrels=qrels_for_query,
            k=50,
        )
        current_recall = Simulation.compute_recall(
            ranked_doc_ids=doc_ids,
            qrels=qrels_for_query,
            k=50,
        )

        # Optional: cache invariant query properties across all states.
        query_length = np.float32(len(current_query.split()))
        query_emb = simulation.encoder.encode(
            current_query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        for step in range(simulation.cfg.max_steps):
            valid_action_mask = state[-N_ACTIONS:]
            action = select_valid_action(cql, state, valid_action_mask)

            ndcg_before = current_ndcg
            recall_before = current_recall

            # STOP is a valid recorded policy decision, but does not execute an agent.
            if action == ACTION_STOP:
                rows.append(
                    {
                        # Query / action metadata.
                        "query_id": query_id,
                        "query": current_query,
                        "new_query": "",
                        "action_name": ACTION_NAMES[action],
                        # State features.
                        **_state_features(state),
                        # Reward objectives.
                        "ndcg_before": current_ndcg,
                        "ndcg_after": current_ndcg,
                        "recall_before": current_recall,
                        "recall_after": current_recall,
                        "cost": 0.0,
                        "latency_ms": 0.0,
                        "cumulative_cost": cumulative_cost,
                        "cumulative_latency_ms": cumulative_latency_ms,
                        "reward": 0.0,
                        "terminal": True,
                        "timeout": False,
                    }
                )
                break

            # compute_effects returns:
            # new_query, new_doc_ids, new_doc_scores, metrics, elapsed_ms, cost
            (
                next_query,
                next_doc_ids,
                next_doc_scores,
                metrics,
                elapsed_ms,
                action_cost,
            ) = simulation.compute_effects(
                action=action,
                query=current_query,
                doc_ids=doc_ids,
                doc_scores=doc_scores,
                qrels=qrels_for_query,
                corpus_data=corpus_data,
            )

            # Agents may return None as cost; treat it as zero.
            action_cost = float(action_cost) if action_cost is not None else 0.0
            elapsed_ms = float(elapsed_ms)

            # These are computed in compute_effects using the updated ranking.
            current_ndcg = float(metrics["ndcg"])
            current_recall = float(metrics["recall"])

            delta_ndcg = current_ndcg - ndcg_before
            delta_recall = current_recall - recall_before

            reward = simulation._compute_reward(
                ndcg_before=ndcg_before,
                ndcg_after=current_ndcg,
                recall_before=recall_before,
                recall_after=current_recall,
                action=action,
                elapsed_ms=elapsed_ms,
                action_cost=action_cost,
            )

            cumulative_cost += action_cost
            cumulative_latency_ms += elapsed_ms

            # Update the one-hot vector of the last action taken.
            # STOP was already handled above.
            if action == 0:
                last_action_agent = [True, False, False]
            elif action == 1:
                last_action_agent = [False, True, False]
            elif action == 2:
                last_action_agent = [False, False, True]
            else:
                last_action_agent = [False, False, False]

            is_last_allowed_step = step == simulation.cfg.max_steps - 1

            # Only QR and PRF change the query text.
            new_query_text = next_query if action in (0, 2) else ""

            rows.append(
                {
                    # Query / action metadata.
                    "query_id": query_id,
                    "query": current_query,
                    "new_query": new_query_text,
                    "action_name": ACTION_NAMES[action],
                    # State features.
                    **_state_features(state),
                    # Reward objectives.
                    "ndcg_before": ndcg_before,
                    "ndcg_after": current_ndcg,
                    "recall_before": recall_before,
                    "recall_after": current_recall,
                    "cost": action_cost,
                    "latency_ms": elapsed_ms,
                    "cumulative_cost": cumulative_cost,
                    "cumulative_latency_ms": cumulative_latency_ms,
                    "reward": reward,
                    "terminal": False,
                    "timeout": is_last_allowed_step,
                }
            )

            if is_last_allowed_step:
                break

            # Move the rollout forward.
            current_query = next_query
            doc_ids = next_doc_ids
            doc_scores = next_doc_scores

            # This mapping may need updating if an agent retrieves new documents
            # whose text is not already in corpus_data.
            current_corpus_data = corpus_data

            # Query text can change after query reformulation, so recalculate
            # invariant query features for the next state.
            query_length = np.float32(len(current_query.split()))
            query_emb = simulation.encoder.encode(
                current_query,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            next_previous_docids = list(doc_ids)

            state = simulation.build_state(
                query=current_query,
                docids=doc_ids,
                docscores=doc_scores,
                step=step + 1,
                last_action_agent=last_action_agent,
                previous_docids=previous_docids,
                original_query_embedding=original_query_emb,
                elapsed_ms=elapsed_ms,
                cumulative_cost=cumulative_cost,
                query_length=query_length,
                query_embedding=query_emb,
            ).astype(np.float32)

            previous_docids = next_previous_docids
            corpus_data = current_corpus_data

    export_test_results(rows, output_csv_path)
    return rows


def export_test_results(
    rows: list[dict[str, Any]],
    output_csv_path: Path,
) -> None:
    """Write action-level policy evaluation results to CSV."""
    if not rows:
        print("No policy actions were produced; CSV was not written.")
        return

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # Query / action metadata.
        "query_id",
        "query",
        "new_query",
        "action_name",
        # Observable state features (embedding excluded).
        "query_length",
        "score_spread",
        "score_entropy",
        "step",
        "rank_overlap",
        "query_drift",
        # Reward objectives.
        "ndcg_before",
        "ndcg_after",
        "recall_before",
        "recall_after",
        "cost",
        "latency_ms",
        "cumulative_cost",
        "cumulative_latency_ms",
        "reward",
        # Episode markers.
        "terminal",
        "timeout",
    ]

    with output_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} action rows to: {output_csv_path}")


# ---------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from .simulation import Simulation, SimConfig, Transition, ACTION_STOP
from .core.agents import AgentBase
from .agents.reformulate import ReformulationAgent
from .agents.rerank import RerankingAgent
from .agents.prf import PRFAgent

def main() -> None:
    logger = logging.getLogger(__name__)
    # 1. Verify the required input files exist.
    for path in (MODEL_PATH, QUERIES_PATH, QRELS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 2. Load query texts and relevance judgments.
    queries = load_queries(
        queries_path=str(QUERIES_PATH),
        num_queries=NUM_QUERIES_TO_TEST,
    )
    qrels = load_qrels(qrels_path=str(QRELS_PATH))

    if not queries:
        raise RuntimeError("No test queries were loaded.")

    # 3. Initialize dependencies

    logger.info("[Step 2] Initializing components")

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing OpenSearch BM25 retriever")
    retriever_instance = Retriever()
    retriever_func = create_retriever_callable(retriever_instance)

    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)

    agents = [qr_agent, rr_agent, prf_agent]

    config = SimConfig(
        max_steps=3,
        top_k_rerank=50,
        top_k_prf=50,
        ndcg_k=50,
        recall_k=50,
        reward_alpha=2.0,
        reward_beta=1,
        reward_gamma=1.0,
        reward_delta=0.5,
    )

    # 4. Build your Simulation exactly as used during dataset generation.
    # Crucially, use the same config, agents, encoder, and retrieval setup
    # used to create the offline CQL dataset.
    simulation = Simulation(
        encoder=encoder,
        retriever=retriever_func,
        agents=agents,
        config=config,
    )

    # 5. Retrieve the initial top-50 documents for all test queries.
    retrieved_queries = batch_retrieve(
        queries=queries,
        num_queries=NUM_QUERIES_TO_TEST,
        retriever_func=retriever_func,
    )

    if not retrieved_queries:
        raise RuntimeError("No initial retrieval results were produced.")

    # 6. Build the initial 399-D state for every query.
    initial_states = generate_test_states(
        retrieved_queries=retrieved_queries,
        simulation=simulation,
        qrels=qrels,
    )

    if not initial_states:
        raise RuntimeError("No valid test states were generated.")

    # 7. Load the trained d3rlpy policy.
    cql = d3rlpy.load_learnable(
        str(MODEL_PATH),
        device="cpu:0",  # Change to "cuda:0" if appropriate.
    )

    # 8. Run policy rollouts.
    #
    # test_batch_states returns one dictionary per chosen policy action.
    # It should NOT call export_test_results internally.
    rows = test_batch_states(
        initial_states=initial_states,
        simulation=simulation,
        cql=cql,
        qrels=qrels,
        output_csv_path=OUTPUT_CSV_PATH
    )

    # 9. Export all per-action evaluation data once.
    export_test_results(
        rows=rows,
        output_csv_path=OUTPUT_CSV_PATH,
    )

    print(
        f"Evaluation completed: {len(rows)} action rows "
        f"saved to {OUTPUT_CSV_PATH}"
    )


if __name__ == "__main__":
    main()