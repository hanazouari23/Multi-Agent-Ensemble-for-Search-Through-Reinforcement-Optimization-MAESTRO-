"""Run the expert policy on a test query set and export the trajectory CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from sentence_transformers import SentenceTransformer

from .main import load_qrels, load_queries
from .simulation import Simulation, SimConfig, ACTION_NAMES, ACTION_STOP
from .agents.reformulate import ReformulationAgent
from .agents.rerank import RerankingAgent
from .agents.prf import PRFAgent
from .utils.retriever import Retriever, create_retriever_callable
from .test_rl_policy import (
    QueryRetrieval,
    TestState,
    TOP_K,
    DEFAULT_NUM_QUERIES,
    N_ACTIONS,
    batch_retrieve,
    generate_test_states,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "notebooks" / "queries" / "topics.ms-marco-dev.tsv"
DEFAULT_QRELS_PATH = PROJECT_ROOT / "notebooks" / "qrels" / "qrels.ms-marco-dev.tsv"
DEFAULT_OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "expert_current_dev_50.csv"


def select_expert_action(
    simulation: Simulation,
    query: str,
    doc_ids: list[str],
    doc_scores: np.ndarray,
    qrels: dict[str, int],
    valid_action_mask: np.ndarray,
) -> int:
    """Ask the simulation's expert policy for the best valid action."""
    valid = valid_action_mask.astype(bool).tolist()
    return simulation._policy_expert(query, doc_ids, doc_scores, qrels, valid)


def test_batch_states_expert(
    initial_states: list[TestState],
    simulation: Simulation,
    qrels: dict[str, dict[str, int]],
    output_csv_path: Path,
) -> list[dict[str, Any]]:
    """Run the expert policy for every initial query state."""
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

        query_length = np.float32(len(current_query.split()))
        query_emb = simulation.encoder.encode(
            current_query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        original_query_emb = query_emb.copy()
        previous_docids: list[str] | None = None

        current_ndcg = Simulation.compute_ndcg(
            ranked_doc_ids=doc_ids,
            qrels=qrels_for_query,
            k=simulation.cfg.ndcg_k,
        )
        current_recall = Simulation.compute_recall(
            ranked_doc_ids=doc_ids,
            qrels=qrels_for_query,
            k=simulation.cfg.recall_k,
        )

        for step in range(simulation.cfg.max_steps):
            valid_action_mask = state[-N_ACTIONS:]
            action = select_expert_action(
                simulation, current_query, doc_ids, doc_scores, qrels_for_query, valid_action_mask
            )

            ndcg_before = current_ndcg
            recall_before = current_recall

            if action == ACTION_STOP:
                from .test_rl_policy import _state_features
                rows.append(
                    {
                        "query_id": query_id,
                        "query": current_query,
                        "new_query": "",
                        "action_name": ACTION_NAMES[action],
                        **_state_features(state),
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

            action_cost = float(action_cost) if action_cost is not None else 0.0
            elapsed_ms = float(elapsed_ms)

            current_ndcg = float(metrics["ndcg"])
            current_recall = float(metrics["recall"])

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

            if action == 0:
                last_action_agent = [True, False, False]
            elif action == 1:
                last_action_agent = [False, True, False]
            elif action == 2:
                last_action_agent = [False, False, True]
            else:
                last_action_agent = [False, False, False]

            is_last_allowed_step = step == simulation.cfg.max_steps - 1
            new_query_text = next_query if action in (0, 2) else ""

            from .test_rl_policy import _state_features
            rows.append(
                {
                    "query_id": query_id,
                    "query": current_query,
                    "new_query": new_query_text,
                    "action_name": ACTION_NAMES[action],
                    **_state_features(state),
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

            next_previous_docids = list(doc_ids)
            current_query = next_query
            doc_ids = next_doc_ids
            doc_scores = next_doc_scores

            query_length = np.float32(len(current_query.split()))
            query_emb = simulation.encoder.encode(
                current_query,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            state = simulation.build_state(
                query=current_query,
                docids=doc_ids,
                docscores=doc_scores,
                step=step + 1,
                last_action_agent=last_action_agent,
                previous_docids=next_previous_docids,
                original_query_embedding=original_query_emb,
                elapsed_ms=cumulative_latency_ms,
                cumulative_cost=cumulative_cost,
                query_length=query_length,
                query_embedding=query_emb,
            )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} action rows to: {output_csv_path}")
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the expert policy on dev queries")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    args = parser.parse_args()

    queries = load_queries(str(args.queries), num_queries=None)
    qrels = load_qrels(str(args.qrels))
    queries = [
        (query_id, query_text)
        for query_id, query_text in queries
        if query_id in qrels
    ][: args.num_queries]

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    retriever_instance = Retriever()
    retriever_func = create_retriever_callable(retriever_instance)

    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)

    config = SimConfig(
        max_steps=3,
        top_k_rerank=50,
        top_k_prf=10,
        ndcg_k=50,
        recall_k=100,
        reward_alpha=2.0,
        reward_beta=0.5,
        reward_gamma=0.2,
        reward_delta=0.1,
    )

    simulation = Simulation(
        encoder=encoder,
        retriever=retriever_func,
        agents=[qr_agent, rr_agent, prf_agent],
        config=config,
    )

    retrieved = batch_retrieve(queries, args.num_queries, retriever_func)
    initial_states = generate_test_states(retrieved, simulation, qrels)

    test_batch_states_expert(
        initial_states=initial_states,
        simulation=simulation,
        qrels=qrels,
        output_csv_path=args.output,
    )


if __name__ == "__main__":
    main()
