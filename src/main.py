#!/usr/bin/env python3
"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

Main entry point for offline RL trajectory collection.
Generates trajectories by simulating multi-agent retrieval optimization,
exports to CSV for offline RL training.
"""

import os
import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from d3rlpy.dataset import MDPDataset
import urllib3
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from src.simulation import ACTION_STOP

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Setup path
src_path = Path(__file__).parent
root_path = src_path.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

# Load environment variables from .env.txt if it exists
def load_env_file():
    """Load environment variables from .env.txt in the repo root."""
    env_file = Path(__file__).parent.parent / ".env.txt"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

load_env_file()

# Support both package and direct execution
try:
    # Relative imports (when run as package)
    from .simulation import Simulation, SimConfig, Transition
    from .core.agents import AgentBase
    from .agents.reformulate import ReformulationAgent
    from .agents.rerank import RerankingAgent
    from .agents.prf import PRFAgent
    from .utils.retriever import Retriever, create_retriever_callable
except ImportError:
    # Absolute imports (when run as script)
    from simulation import Simulation, SimConfig, Transition
    from core.agents import AgentBase
    from agents.reformulate import ReformulationAgent
    from agents.rerank import RerankingAgent
    from agents.prf import PRFAgent
    from utils.retriever import Retriever, create_retriever_callable


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            # Supports: query_id \t doc_id \t grade
            query_id = parts[0].strip()
            doc_id = parts[2].strip()
            try:
                grade = int(parts[3].strip())
            except ValueError:
                continue

            qrels[query_id][doc_id] = grade

    logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_path}")
    return dict(qrels)

def load_queries(queries_path: str, num_queries: Optional[int] = None) -> List[Tuple[str, str]]:
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                query_id = parts[0].strip()
                query_text = parts[1].strip()
            else:
                query_id = str(len(queries))
                query_text = parts[0].strip()

            queries.append((query_id, query_text))
            if num_queries and len(queries) >= num_queries:
                break

    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def load_initial_retrieval(
    query: str,
    retriever_func: callable,
    top_k: int = 50,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """
    Load initial retrieval results using BM25 backend.
    
    Parameters
    ----------
    query : str
        Query string
    retriever_func : callable
        Retriever function returning (doc_ids, scores, corpus_data)
    top_k : int
        Number of documents to retrieve
    
    Returns
    -------
    doc_ids : List[str]
    doc_scores : np.ndarray
        BM25 scores
    corpus_data : Dict[str, str]
        Mapping of doc_id -> document text
    """
    try:
        doc_ids, doc_scores, corpus_data = retriever_func(query, top_k)
        return doc_ids, doc_scores, corpus_data
    except Exception as e:
        logger.error(f"Retrieval failed for query '{query}': {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectories(
    config: SimConfig,
    qrels: Dict[str, int],
    queries: List[Tuple[str, str]],
    encoder: SentenceTransformer,
    agents: List[AgentBase],
    retriever: callable,
    num_trajectories: int = 100,
    policy: str = "random",
) -> Tuple[List[List[Transition]], MDPDataset]:
    """
    Generate offline RL trajectories for a set of queries.
    
    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    qrels : Dict[str, int]
        Query relevance judgments {doc_id: grade}
    queries : List[Tuple[str, str]]
        List of query IDs and strings
    encoder : SentenceTransformer
        Query encoder
    agents : List[AgentBase]
        [ReformulationAgent, RerankingAgent, PRFAgent]
    retriever : callable
        Retrieval function returning (doc_ids, scores, corpus_data)
    num_trajectories : int
        Number of trajectories to generate (uses first N queries)
    policy : str
        Action selection policy: "random", "expert", "stop"
    
    Returns
    -------
    List[List[Transition]]
        All generated trajectories
    MDPDataset
        Constructed MDP dataset
    """
    logger.info(f"\n{'='*70}")
    logger.info("TRAJECTORY GENERATION PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Number of trajectories: {num_trajectories}")
    logger.info(f"Policy: {policy}")
    logger.info(f"Queries available: {len(queries)}")
    logger.info(f"Config: max_steps={config.max_steps}, top_k_rerank={config.top_k_rerank}")
    
    # Initialize simulation
    sim = Simulation(
        encoder=encoder,
        retriever=retriever,
        agents=agents,
        config=config,
    )
    
    all_trajectories = []
    
    #Initialize arrays to construct MDP dataset
    num_episodes = min(num_trajectories, len(queries))
    max_len = config.max_steps * num_episodes  
    observations = np.zeros((max_len , config.state_dim), dtype=np.float32)
    actions      = np.zeros(max_len, dtype=np.int64)    # discrete
    rewards      = np.zeros(max_len, dtype=np.float32)
    terminals    = np.zeros(max_len, dtype=np.int64)
    timeouts = np.zeros(max_len, dtype=np.int64)
    idx = 0

    for traj_id in range(min(num_trajectories, len(queries))):
        query_id,query = queries[traj_id]
        logger.info(f"\n[Trajectory {traj_id+1}/{num_trajectories}] Query: {query[:60]}...")
        # Get initial retrieval (now includes corpus_data)
        doc_ids, doc_scores, corpus_data = load_initial_retrieval(
            query, retriever, config.top_k_rerank
        )
        logger.info(f"  Retrieved {len(doc_ids)} documents")
        qrels_for_query = qrels.get(query_id, {})
        try:
            # Generate trajectory for this query
            trajectory = sim.generate_trajectory(
                query=query,
                doc_ids=doc_ids,
                doc_scores=doc_scores,
                qrels=qrels_for_query,
                policy=policy,
                corpus_data=corpus_data,  # Pass corpus for RerankingAgent
            )
            
            for step, transition in enumerate(trajectory):
                is_terminal = transition.action == ACTION_STOP
                is_timeout = (
                    step == config.max_steps - 1
                    and transition.action != ACTION_STOP
                )
                observations[idx] = transition.state
                actions[idx] = transition.action
                rewards[idx] = transition.reward
                terminals[idx] = int(is_terminal)
                timeouts[idx] = int(is_timeout)
                idx += 1
            
            all_trajectories.append(trajectory)
            logger.info(f"  ✓ Generated trajectory with {len(trajectory)} steps")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to generate trajectory: {e}", exc_info=False)
            continue
    # Trim unused tail
    observations = observations[:idx]
    actions      = actions[:idx]
    rewards      = rewards[:idx]
    terminals    = terminals[:idx]
    timeouts    = timeouts[:idx]
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts
    )    
    logger.info(f"\n{'='*70}")
    logger.info(f"Generated {len(all_trajectories)} trajectories")
    total_steps = sum(len(traj) for traj in all_trajectories)
    logger.info(f"Total transitions: {total_steps}")
    logger.info(f"{'='*70}\n")
    
    return all_trajectories, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    """
    Main entry point for offline RL trajectory generation.
    """
    logger.info("Starting MAESTRO offline RL trajectory collection...")
    
    # ── Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    
    # Load qrels
    qrels_path = Path(args.qrels_path or "notebooks/qrels/trec_rag_2025_qrels.tsv")
    if not qrels_path.is_absolute():
        qrels_path = root_path / qrels_path
    logger.info(f"Loading qrels from: {qrels_path}")
    qrels = load_qrels(str(qrels_path))
    
    # Load queries
    queries_path = Path(args.queries_path or "notebooks/queries/trec_rag_2025_queries.tsv")
    if not queries_path.is_absolute():
        queries_path = root_path / queries_path
    logger.info(f"Loading queries from: {queries_path}")
    queries = load_queries(str(queries_path), num_queries=args.num_queries)
    
    # ── Step 2: Initialize components
    logger.info("\n[Step 2] Initializing components...")
    
    # Load query encoder
    logger.info("Loading query encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize BM25 retriever (OpenSearch backend)
    logger.info("Initializing BM25 retriever (OpenSearch)...")
    try:
        retriever_instance = Retriever()
        retriever = create_retriever_callable(retriever_instance)
        logger.info("✓ OpenSearch retriever initialized")
    except Exception as e:
        logger.warning(f"OpenSearch retriever failed: {e}")
    
    # Initialize agents
    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)
    agents = [qr_agent, rr_agent, prf_agent]
    
    # Create simulation config
    config = SimConfig(
        max_steps=args.max_steps,
        top_k_rerank=args.top_k_rerank,
        top_k_prf=args.top_k_prf,
        ndcg_k=args.ndcg_k,
        recall_k=args.recall_k,
        reward_alpha=2.0,
        reward_beta=0.5,
        reward_gamma=1.0,
        reward_delta=0.5,
    )
    
    # ── Step 3: Generate trajectories
    logger.info("\n[Step 3] Generating trajectories...")
    trajectories, mdp_dataset = generate_trajectories(
        config=config,
        qrels=qrels,
        queries=queries,
        encoder=encoder,
        agents=agents,
        retriever=retriever,
        num_trajectories=args.num_trajectories,
        policy=args.policy,
    )
    
    # ── Step 4: Export to CSV
    logger.info("\n[Step 4] Exporting trajectories to CSV...")
    csv_path = Simulation.export_trajectories_to_csv(
        trajectories,
        f"trajectories_{args.policy}_{args.num_trajectories}.csv",
    )
    logger.info(f"✓ Trajectories exported to {csv_path}")
    
    # ── Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total trajectories:    {len(trajectories)}")
    logger.info(f"Total transitions:     {sum(len(t) for t in trajectories)}")
    logger.info(f"Queries used:          {min(args.num_trajectories, len(queries))}")
    logger.info(f"Qrels loaded:          {len(qrels)}")
    logger.info(f"Output file:           {csv_path}")
    logger.info("="*70 + "\n")
    
    logger.info("✓ Pipeline complete! Ready for offline RL training.")

    #Save the generated MDP dataset to a file for later use 
    mdp_dataset.dump("mdp_dataset.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate offline RL trajectories for MAESTRO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--qrels-path",
        type=str,
        default="notebooks/qrels/trec_rag_2025_qrels.tsv",
        help="Path to qrels file",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="notebooks/queries/trec_rag_2025_queries.tsv",
        help="Path to queries file",
    )
    
    # Simulation config
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Load only first N queries",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum steps per trajectory",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=50,
        help="Top-k for reranking window",
    )
    parser.add_argument(
        "--top-k-prf",
        type=int,
        default=10,
        help="Top-k for PRF term extraction",
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=50,
        help="NDCG evaluation cutoff",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=100,
        help="Recall evaluation cutoff",
    )
    
    # Policy
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "expert", "stop", "prf"],
        help="Action selection policy",
    )
    
    args = parser.parse_args()
    main(args)
