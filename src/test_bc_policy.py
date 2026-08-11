"""Test a behavioral-cloning policy on the dev set."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .main import load_qrels, load_queries
from .simulation import Simulation, ACTION_NAMES, ACTION_STOP
from .test_rl_policy import (
    QueryRetrieval,
    TestState,
    TOP_K,
    DEFAULT_NUM_QUERIES,
    N_ACTIONS,
    batch_retrieve,
    generate_test_states,
    test_batch_states,
)
from .train_bc import BCPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POLICY_PATH = PROJECT_ROOT / "outputs" / "bc_checkpoints" / "bc_policy.pt"
SCALER_PATH = PROJECT_ROOT / "outputs" / "bc_checkpoints" / "bc_scaler.npz"

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "notebooks" / "queries" / "topics.ms-marco-dev.tsv"
DEFAULT_QRELS_PATH = PROJECT_ROOT / "notebooks" / "qrels" / "qrels.ms-marco-dev.tsv"
DEFAULT_OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "bc_policy_dev_50.csv"

HIDDEN_UNITS = [512, 512, 256]
STATE_DIM = 399
N_ACTIONS = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_bc_policy(policy_path: Path, scaler_path: Path) -> tuple[BCPolicy, np.ndarray, np.ndarray]:
    """Load the BC policy and observation scaler."""
    model = BCPolicy(STATE_DIM, N_ACTIONS, HIDDEN_UNITS).to(DEVICE)
    model.load_state_dict(torch.load(policy_path, map_location=DEVICE, weights_only=True))
    model.eval()

    scaler = np.load(scaler_path)
    mean = scaler["mean"]
    std = scaler["std"]
    return model, mean, std


def select_valid_action(
    model: BCPolicy,
    state: np.ndarray,
    valid_action_mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> int:
    """Choose the highest-probability valid action."""
    state_scaled = (state - mean) / std
    state_batch = torch.from_numpy(state_scaled.reshape(1, -1).astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        logits = model(state_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)

    masked_probs = np.where(valid_action_mask.astype(bool), probs, 0.0)

    if masked_probs.sum() <= 0:
        return ACTION_STOP

    return int(np.argmax(masked_probs))


def test_batch_states_bc(
    initial_states: list[TestState],
    simulation: Simulation,
    model: BCPolicy,
    mean: np.ndarray,
    std: np.ndarray,
    qrels: dict[str, dict[str, int]],
    output_csv_path: Path,
) -> list[dict[str, Any]]:
    """Run a greedy-policy rollout using the BC model."""
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
            action = select_valid_action(model, state, valid_action_mask, mean, std)

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
    import logging
    from sentence_transformers import SentenceTransformer

    from .simulation import SimConfig
    from .agents.reformulate import ReformulationAgent
    from .agents.rerank import RerankingAgent
    from .agents.prf import PRFAgent
    from .utils.retriever import Retriever, create_retriever_callable

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Test BC policy on dev queries")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    args = parser.parse_args()

    if args.suffix:
        suffix = args.suffix
        args.policy = args.policy or (PROJECT_ROOT / "outputs" / "bc_checkpoints" / f"bc_policy_{suffix}.pt")
        args.scaler = args.scaler or (PROJECT_ROOT / "outputs" / "bc_checkpoints" / f"bc_scaler_{suffix}.npz")
        args.output = args.output or (PROJECT_ROOT / "outputs" / f"bc_policy_{suffix}_dev_50.csv")
    else:
        args.policy = args.policy or POLICY_PATH
        args.scaler = args.scaler or SCALER_PATH
        args.output = args.output or DEFAULT_OUTPUT_CSV_PATH

    queries = load_queries(str(args.queries), num_queries=None)
    qrels = load_qrels(str(args.qrels))
    queries = [
        (query_id, query_text)
        for query_id, query_text in queries
        if query_id in qrels
    ][: args.num_queries]

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing OpenSearch BM25 retriever")
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

    model, mean, std = load_bc_policy(args.policy, args.scaler)
    test_batch_states_bc(
        initial_states=initial_states,
        simulation=simulation,
        model=model,
        mean=mean,
        std=std,
        qrels=qrels,
        output_csv_path=args.output,
    )


if __name__ == "__main__":
    main()
