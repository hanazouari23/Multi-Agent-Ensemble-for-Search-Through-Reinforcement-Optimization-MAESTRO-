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

from src.agents.prf import PRFAgent


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
    from .agents import ReformulationAgent, RerankingAgent, PRFAgent
    from .utils.retriever import Retriever, create_retriever_callable
except ImportError:
    # Absolute imports (when run as script)
    from simulation import Simulation, SimConfig
    from core.agents import AgentBase
    from agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
    from utils.retriever import Retriever, create_retriever_callable


def get_base_doc_id(segment_id: str) -> str:
    """Strip segment suffix: 'msmarco_v2.1_doc_05_123#3_456' → 'msmarco_v2.1_doc_05_123'"""
    return segment_id.split('#')[0]

def collect_trajectories_batch(
    retriever: callable,
    sim,
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

    print(f"\n" + "="*60)
    print("BATCH TRAJECTORY COLLECTION FOR OFFLINE RL")
    print("="*60)
    print(f"Target trajectories:  {n_trajectories}")
    print(f"Sampling strategy:    {sampling_strategy}")
    print(f"Output directory:     {output_path}")

def _run_demo_mode(sim, qr_agent, rr_agent, prf_agent) -> None:
    """Run demo mode: test individual agents and single trajectory."""
    query = "Improve performance python?"
    
    # Retrieve documents for demo query
    seg_ids, doc_scores, corpus_data = sim.retriever(query, top_k=10)

    print(f"\nDemo Query: '{query}'")
    print(f"Initial documents: {seg_ids}")
    print(f"Initial segments: {corpus_data.values()}")
    print(f"Initial scores: {doc_scores}")

    # Test each agent individually
    print("\n" + "="*50)
    print("Testing Individual Agents")
    print("="*50)

    # Test ReformulationAgent
    # print("\n[1] Testing ReformulationAgent...")
    # if os.getenv("API_KEY"):
    #     query_features = {
    #         'query_text': query,
    #         'retriever': sim.retriever,
    #     }
    #     effects = qr_agent.compute_effects(query_features)
    #     print(f"   Reformulated: '{effects['new_query_text']}'")
    #     print(f"   New docs: {len(effects['new_doc_ids'])}")
    #     print(f"   Time: {effects['elapsed_time']:.3f}s")
    # else:
    #     print("   Skipping ReformulationAgent (API_KEY not set)")

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

    # Test PRFAgent
    print("\n[3] Testing PRFAgent...")
    query_features = {
            'query_text': query,
            'retriever': sim.retriever,
        }
    effects = prf_agent.compute_effects(query_features, raw_results=(seg_ids, doc_scores, corpus_data))
    print(f"   Reweighted docs: {effects['new_doc_ids']}")
    print(f"   New scores: {effects['new_doc_scores']}")
    print(f"   Retrieved segments after PRF: '{effects['new_segments']}'")
    print(f"   Time: {effects['elapsed_time']:.3f}s")
    
    # Generate comparative HTML table
    print("\n" + "="*100)
    print("COMPARATIVE TABLE: Initial Segments vs. PRF-Processed Segments")
    print("="*100)
    
    initial_segments = list(corpus_data.values())
    prf_segments = list(effects['new_segments'])
    
    max_rows = max(len(initial_segments), len(prf_segments))
    
    # Generate HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PRF Comparison - Initial vs After PRF</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                border: 1px solid #ddd;
            }
            td {
                padding: 12px;
                border: 1px solid #ddd;
                vertical-align: top;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <h1>PRF Comparison: Initial Segments vs. After PRF Processing</h1>
        <table>
            <tr>
                <th>Initial Segments</th>
                <th>Segments After PRF</th>
            </tr>
    """
    
    for i in range(max_rows):
        initial_seg = initial_segments[i] if i < len(initial_segments) else "N/A"
        prf_seg = prf_segments[i] if i < len(prf_segments) else "N/A"
        
        html_content += f"""
            <tr>
                <td>{initial_seg}</td>
                <td>{prf_seg}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write HTML file
    output_path = Path(__file__).parent.parent / "prf_comparison.html"
    with open(output_path, 'w', encoding='utf-8') as htmlfile:
        htmlfile.write(html_content)
    
    print(f"\nComparative table saved to: {output_path}")
    print(f"Opening in browser...")
    
    import webbrowser
    webbrowser.open(f'file://{output_path.absolute()}')


def main():
    # """Main launcher function."""
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization"
    # )
    # parser.add_argument(
    #     "--mode",
    #     choices=["demo", "collect"],
    #     default="demo",
    #     help="demo: test individual agents; collect: generate trajectories for offline RL",
    # )
    # parser.add_argument(
    #     "--n-trajectories",
    #     type=int,
    #     default=100,
    #     help="Number of trajectories to collect (for --mode collect)",
    # )
    # parser.add_argument(
    #     "--sampling-strategy",
    #     choices=["random", "stratified", "sequential"],
    #     default="random",
    #     help="Query sampling strategy for trajectory collection",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     default="trajectories",
    #     help="Output directory for trajectory files",
    # )

    # args = parser.parse_args()

    # print("Starting MAESTRO Simulation")

    # # Initialize encoder
    # print("Loading sentence transformer...")
    # encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # # Initialize retriever
    # print("Setting up retriever...")
    # retriever_instance = Retriever()
    # retriever_func = create_retriever_callable(retriever_instance)


    # # Create agents
    # print("Creating agents...")
    # qr_agent = ReformulationAgent(encoder)
    # rr_agent = RerankingAgent(encoder)
    # prf_agent = PRFAgent(encoder)
    
    # agents = [qr_agent, rr_agent, prf_agent]

    # # Create simulation
    # print("Setting up simulation...")
    # config = SimConfig()
    # sim = Simulation(
    #     encoder=encoder,
    #     retriever=retriever_func,
    #     agents=agents,
    #     config=config
    # )

    # # Mode: demo or collect
    # if args.mode == "demo":
    #     _run_demo_mode(sim, qr_agent, rr_agent, prf_agent)
    # elif args.mode == "collect":
    #     collect_trajectories_batch(
    #         retriever_func,
    #         sim,
    #         n_trajectories=args.n_trajectories,
    #         sampling_strategy=args.sampling_strategy,
    #         output_dir=args.output_dir,
    #     )

    # print("\nMAESTRO completed successfully!")
    import pandas as pd
    from collections import defaultdict
    import ir_datasets

    dataset = ir_datasets.load("msmarco-passage/train/judged")

    print("Loading query texts...")
    query_text_map = {}
    for query in dataset.queries_iter():
        query_text_map[str(query.query_id)] = query.text
    print(f"✓ Loaded {len(query_text_map)} query texts")

    print("\nLoading qrels...")
    qrels_by_query = defaultdict(list)
    for qrel in dataset.qrels_iter():
        qrels_by_query[qrel.query_id].append(qrel)
    print(f"✓ Loaded {len(qrels_by_query)} queries with qrels")

    queries_with_3plus = {
        qid: qrels for qid, qrels in qrels_by_query.items() if len(qrels) >= 3
    }
    print(f"✓ Found {len(queries_with_3plus)} queries with >= 3 qrels")

    csv_data = []
    for query_id, qrels in queries_with_3plus.items():
        query_text = query_text_map.get(str(query_id), "")
        for qrel in qrels:
            csv_data.append({
                'query_id':   str(qrel.query_id),
                'query_text': query_text,
                'doc_id':     str(qrel.doc_id),
                'relevance':  int(qrel.relevance),
                'iteration':  str(getattr(qrel, 'iteration', '0'))
            })

    df = pd.DataFrame(csv_data)
    df.to_csv('msmarco_queries_3plus_qrels.csv', index=False, encoding='utf-8')
    print(f"\n✓ Saved {len(csv_data)} qrels from {len(queries_with_3plus)} queries")
    print(f"\nQrels per query stats:")
    print(df.groupby('query_id').size().describe())



if __name__ == "__main__":
    main()