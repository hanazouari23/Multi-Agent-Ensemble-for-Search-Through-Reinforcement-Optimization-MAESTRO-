#!/usr/bin/env python3
"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

Launcher script demonstrating the agent-based simulation architecture.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
    from .simulation import Simulation, SimConfig
    from .core.agents import AgentBase
    from .agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
    from .utils.retriever import Retriever, create_retriever_callable
    from .utils.orcas_loader import load_orcas_tsv_sample
except ImportError:
    # Fallback for direct script execution
    from simulation import Simulation, SimConfig
    from core.agents import AgentBase
    from agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
    from utils.retriever import Retriever, create_retriever_callable
    from utils.orcas_loader import load_orcas_tsv_sample

def load_orcas_index(filepath: str) -> dict:
    """Load ORCAS click data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_corpus(filepath: str) -> dict:
    """Load document corpus from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_sample_qrels() -> dict:
    """Create sample qrels for testing (doc_id -> relevance_score)."""
    return {
        "doc1": 1,
        "doc2": 0,
        "doc3": 1,
        "doc4": 0,
        "doc5": 1,
    }

def main():
    """Main launcher function."""
    print("Starting MAESTRO Simulation")

    # Initialize encoder
    print("Loading sentence transformer...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize retriever
    print("Setting up retriever...")
    retriever_instance = Retriever()
    retriever_func = create_retriever_callable(retriever_instance)

    # Load data (you would replace these with actual file paths)
    print("Loading data...")
    # Load ORCAS dataset from TSV
    orcas_file = Path(__file__).parent.parent / "data" / "orcas.tsv"
    if orcas_file.exists():
        print(f"Loading ORCAS from {orcas_file}...")
        orcas_index = load_orcas_tsv_sample(str(orcas_file), max_queries=500)
        print(f"Loaded {len(orcas_index)} unique queries from ORCAS")
    else:
        print("ORCAS file not found, using sample data")
        orcas_index = {"doc1": ["query1"], "doc2": ["query2"]}  # Sample
    
    corpus = {"doc1": "Sample document 1", "doc2": "Sample document 2"}  # Sample

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
        corpus=corpus,
        config=config
    )

    # Sample query and documents
    query = "What are the best restaurants in Passau?"
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    doc_scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    qrels = create_sample_qrels()

    print(f"Testing with query: '{query}'")
    print(f"Initial documents: {doc_ids}")
    print(f"Initial scores: {doc_scores}")

    # Test each agent individually
    print("\n" + "="*50)
    print("Testing Individual Agents")
    print("="*50)

    # Test ReformulationAgent
    print("\n[1] Testing ReformulationAgent...")
    if os.getenv("OPENROUTER_API_KEY"):
        query_features = {
            'query_text': query,
            'retriever': retriever_func,
        }
        effects = qr_agent.compute_effects(query_features)
        print(f"   Reformulated: '{effects['new_query_text']}'")
        print(f"   New docs: {len(effects['new_doc_ids'])}")
        print(f"   Time: {effects['elapsed_time']:.3f}s")
    else:
        print("   Skipping ReformulationAgent (OPENROUTER_API_KEY not set)")

    # Test RerankingAgent
    print("\n[2] Testing RerankingAgent...")
    query_features = {
        'query_text': query,
        'doc_ids': doc_ids,
        'doc_scores': doc_scores,
        'corpus': corpus,
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
        'doc_ids': doc_ids,
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

    if os.getenv("OPENROUTER_API_KEY"):
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
        print("Skipping trajectory generation (OPENROUTER_API_KEY not set)")
        print("Set OPENROUTER_API_KEY environment variable to enable query reformulation.")

    print("\nMAESTRO simulation test completed successfully!")

if __name__ == "__main__":
    main()