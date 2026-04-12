#!/usr/bin/env python3
"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

Launcher script demonstrating the agent-based simulation architecture.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


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
    from .simulation import Simulation, SimConfig
    from .core.agents import AgentBase
    from .agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
    from .utils.retriever import Retriever, create_retriever_callable
    from .utils.orcas_loader import load_orcas_tsv_sample
    from .utils.trajectory_collector import TrajectoryCollector
except ImportError:
    # Absolute imports (when run as script)
    from simulation import Simulation, SimConfig
    from core.agents import AgentBase
    from agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
    from utils.retriever import Retriever, create_retriever_callable
    from utils.orcas_loader import load_orcas_tsv_sample
    from utils.trajectory_collector import TrajectoryCollector

def get_base_doc_id(segment_id: str) -> str:
    """Strip segment suffix: 'msmarco_v2.1_doc_05_123#3_456' → 'msmarco_v2.1_doc_05_123'"""
    return segment_id.split('#')[0]

def collect_trajectories_batch(
    retriever: callable,
    sim,
    orcas_index: dict,
    n_trajectories: int = 100,
    sampling_strategy: str = "random",
    output_dir: str = "trajectories",
) -> None:
    """
    Collect a batch of trajectories for offline RL training.

    Parameters
    ----------
    sim : Simulation
        The simulation instance
    orcas_index : dict
        Query → [clicked_docs] mapping from ORCAS
    n_trajectories : int
        Number of trajectories to collect
    sampling_strategy : str
        "random", "stratified", or "sequential"
    output_dir : str
        Output directory for saved trajectories
    """
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(exist_ok=True)

    # Extract queries from ORCAS index
    queries = list(orcas_index.keys())[:100]  # Limit to 100 unique queries for faster collection
    
    if not queries:
        print("ERROR: No queries in ORCAS index. Cannot collect trajectories.")
        return

    print(f"\n" + "="*60)
    print("BATCH TRAJECTORY COLLECTION FOR OFFLINE RL")
    print("="*60)
    print(f"Queries available:    {len(queries)}")
    print(f"Target trajectories:  {n_trajectories}")
    print(f"Sampling strategy:    {sampling_strategy}")
    print(f"Output directory:     {output_path}")

    # Prepare query metadata
    query_metadata = {}
    for query_text in queries:
        retrieved_segments, bm25_scores, corpus_data = retriever(query_text, top_k=50)

        if not retrieved_segments:
            continue

        query_metadata[query_text] = {
            "doc_ids": retrieved_segments,
            "doc_scores": bm25_scores,
        }
    # Create collector
    collector = TrajectoryCollector(
        simulation=sim,
        queries=queries,
        query_metadata=query_metadata,
    )

    # Collect trajectories
    trajectories, stats = collector.collect_batch(
        n_trajectories=n_trajectories,
        sampling_strategy=sampling_strategy,
        policies=["random", "expert"] if os.getenv("API_KEY") else ["random"],
    )

    # Save to disk
    print("\nSaving trajectories to disk...")
    
    # Save as NPZ (efficient for ML)
    npz_path = output_path / f"trajectories_{n_trajectories}_{sampling_strategy}.npz"
    collector.save_trajectories_npz(trajectories, npz_path)

    # Save as JSON (human-readable)
    json_path = output_path / f"trajectories_{n_trajectories}_{sampling_strategy}.json"
    collector.save_trajectories_json(trajectories, json_path, compress=True)

    # Save statistics
    stats_path = output_path / f"stats_{n_trajectories}_{sampling_strategy}.json"
    collector.save_stats_json(stats, stats_path)

    # Print report
    report = collector.report_stats(stats)
    print(report)

    print(f"✓ Trajectories saved to: {npz_path}")
    print(f"✓ JSON (compressed) saved to: {json_path}.gz")
    print(f"✓ Statistics saved to: {stats_path}")


def main():
    """Main launcher function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "collect"],
        default="demo",
        help="demo: test individual agents; collect: generate trajectories for offline RL",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=100,
        help="Number of trajectories to collect (for --mode collect)",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "stratified", "sequential"],
        default="random",
        help="Query sampling strategy for trajectory collection",
    )
    parser.add_argument(
        "--output-dir",
        default="trajectories",
        help="Output directory for trajectory files",
    )

    args = parser.parse_args()

    print("Starting MAESTRO Simulation")

    # Initialize encoder
    print("Loading sentence transformer...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize retriever
    print("Setting up retriever...")
    retriever_instance = Retriever()
    retriever_func = create_retriever_callable(retriever_instance)

    # Load data
    print("Loading data...")
    orcas_file = Path(__file__).parent.parent / "data" / "orcas.tsv"
    if orcas_file.exists():
        print(f"Loading ORCAS from {orcas_file}...")
        orcas_index = load_orcas_tsv_sample(str(orcas_file), max_queries=500)
        print(f"Loaded {len(orcas_index)} unique queries from ORCAS")
    else:
        raise FileNotFoundError("ORCAS file not found. Please download and place it in the data/ directory.")

    # Create agents
    print("Creating agents...")
    qr_agent = ReformulationAgent(encoder)
    rr_agent = RerankingAgent(encoder)
    cp_agent = ClickPriorAgent(encoder)

    agents = [qr_agent, rr_agent, cp_agent]

    # Create simulation
    print("Setting up simulation...")
    config = SimConfig()
    sim = Simulation(
        encoder=encoder,
        retriever=retriever_func,
        agents=agents,
        orcas_index=orcas_index,
        config=config
    )

    # Mode: demo or collect
    if args.mode == "demo":
        _run_demo_mode(sim, qr_agent, rr_agent, cp_agent, orcas_index)
    elif args.mode == "collect":
        collect_trajectories_batch(
            retriever_func,
            sim,
            orcas_index,
            n_trajectories=args.n_trajectories,
            sampling_strategy=args.sampling_strategy,
            output_dir=args.output_dir,
        )

    print("\nMAESTRO completed successfully!")


def _run_demo_mode(sim, qr_agent, rr_agent, cp_agent, orcas_index) -> None:
    """Run demo mode: test individual agents and single trajectory."""
    query = "Restaurants in Passau"
    
    # Retrieve documents for demo query
    seg_ids, doc_scores, corpus_data = sim.retriever(query, top_k=5)

    print(f"\nDemo Query: '{query}'")
    print(f"Initial documents: {seg_ids}")
    print(f"Initial scores: {doc_scores}")

    # Test each agent individually
    print("\n" + "="*50)
    print("Testing Individual Agents")
    print("="*50)

    # Test ReformulationAgent
    print("\n[1] Testing ReformulationAgent...")
    if os.getenv("API_KEY"):
        query_features = {
            'query_text': query,
            'retriever': sim.retriever,
        }
        effects = qr_agent.compute_effects(query_features)
        print(f"   Reformulated: '{effects['new_query_text']}'")
        print(f"   New docs: {len(effects['new_doc_ids'])}")
        print(f"   Time: {effects['elapsed_time']:.3f}s")
    else:
        print("   Skipping ReformulationAgent (API_KEY not set)")

    # Test RerankingAgent
    print("\n[2] Testing RerankingAgent...")
    query_features = {
        'query_text': query,
        'doc_ids': seg_ids,
        'doc_scores': doc_scores,
        'corpus': corpus_data,
        'top_k_rerank': 3,
    }
    effects = rr_agent.compute_effects(query_features)
    print(f"   Reranked docs: {effects['new_doc_ids']}")
    print(f"   New scores: {effects['new_doc_scores']}")
    print(f"   Time: {effects['elapsed_time']:.3f}s")

    # Test ClickPriorAgent
    print("\n[3] Testing ClickPriorAgent...")
    query_features = {
        'query_text': query,
        'doc_ids': seg_ids,
        'doc_scores': doc_scores,
        'orcas_index': orcas_index,
    }
    effects = cp_agent.compute_effects(query_features)
    print(f"   Reweighted docs: {effects['new_doc_ids']}")
    print(f"   New scores: {effects['new_doc_scores']}")
    print(f"   Time: {effects['elapsed_time']:.3f}s")

    # Test full simulation trajectory
    print("\n" + "="*50)
    print("Testing Full Simulation Trajectory")
    print("="*50)

    if os.getenv("API_KEY"):
        # Build empty qrels for demo
        qrels = {}
        trajectory = sim.generate_trajectory(
            query=query,
            doc_ids=doc_ids,
            doc_scores=doc_scores,
            qrels=qrels,
            policy="random"
        )

        print(f"Generated trajectory with {len(trajectory)} steps")
        for i, transition in enumerate(trajectory):
            print(f"Step {i+1}: Action={transition.action}, Reward={transition.reward:.3f}")
    else:
        print("Skipping trajectory generation (API_KEY not set)")
        print("Set API_KEY environment variable to enable query reformulation.")

if __name__ == "__main__":
    main()