# MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

A framework for multi-agent retrieval optimization using offline reinforcement learning. The system learns a policy that decides which specialized agents to invoke at each step to improve search result quality while managing computational costs.

## Overview

**Goal:** Build an intelligent system that learns which agents to invoke at each step to improve search results. The policy is trained via offline reinforcement learning (Discrete CQL) on collected MDP trajectories, using MS MARCO relevance judgments.

**Key Innovation:** Multi-agent ensemble approach where:
- **ReformulationAgent** refines queries using LLM-powered rewriting
- **RerankingAgent** re-scores results using cross-encoder models
- **PRFAgent** expands queries with pseudo-relevance feedback terms from top-ranked documents

The retrieval pipeline is modeled as an MDP with 4 actions (QueryReform, Rerank, PRF, STOP) and a multi-objective reward:

```
r = α·ΔNDCG + β·ΔRecall − γ·cost − δ·latency
```

The system evaluates impact on metrics like NDCG and Recall.

---

## Architecture

```
src/
├── core/
│   ├── agents.py           # AgentBase abstract class defining agent interface
│   └── __init__.py
├── agents/                 # Concrete agent implementations
│   ├── reformulate.py      # Query rewriting agent (OpenRouter LLM)
│   ├── reformulate_with_feedback.py      # Reformulation using retrieval feedback
│   ├── reformulate_ctx_free_exp_terms.py # Context-free expansion-term variant
│   ├── rerank.py           # CrossEncoder reranking agent
│   ├── prf.py              # Pseudo-relevance feedback expansion agent
│   ├── intent.py           # Multi-intent query expansion (experimental)
│   └── __init__.py         # Agent exports
├── utils/
│   ├── retriever.py        # OpenSearch BM25 retrieval interface
│   └── __init__.py
├── main.py                 # Trajectory collection entry point (resumable checkpoints)
├── simulation.py           # MDP orchestrator, state builder & metrics
├── train_cql.py            # Offline RL training (Discrete CQL via d3rlpy)
├── test_rl_policy.py       # Evaluate the trained policy on held-out queries
└── __init__.py             # Package root
```

### Agents

| Agent | Purpose | Key Implementation |
|-------|---------|-------------------|
| **ReformulationAgent** (action 0) | Query optimization | LLM-based rewriting via OpenRouter |
| **RerankingAgent** (action 1) | Result re-ranking | CrossEncoder (MS-MARCO-MiniLM-L-6-v2) |
| **PRFAgent** (action 2) | Query expansion | Expansion terms mined from top-k documents |
| **STOP** (action 3) | End episode | — |

### Core Classes

- **AgentBase**: Abstract base class with `compute_effects()` method defining agent interface
- **Simulation**: MDP framework orchestrating agents, managing state (399-dim vector), and computing metrics
- **Retriever**: OpenSearch BM25 wrapper for initial retrieval

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Poetry (dependency management)
- OpenRouter API key (for LLM integration)
- OpenSearch instance (for retrieval)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Multi-Agent-Ensemble-for-Search-Through-Reinforcement-Optimization-MAESTRO-

# Install dependencies via Poetry
poetry install

# Set environment variables
echo "OPENROUTER_API_KEY=sk-or-v1-xxx..." > .env.txt
```

**Environment Configuration:**
- Create `.env.txt` in project root with API keys:
  ```
  OPENROUTER_API_KEY=<your-key>
  ```
- `src/main.py` auto-loads `.env.txt` at runtime; agents also support standard `.env` via `python-dotenv`.

### Dependencies

Core packages:
- `sentence-transformers` - Embeddings and cross-encoder models
- `openai` - LLM access via OpenRouter
- `d3rlpy` - Offline RL (Discrete CQL)
- `torch` (cu126) - Policy network backend
- `numpy`, `scipy`, `pandas` - Numerical operations
- `requests` - HTTP client for OpenSearch
- `ir-datasets` - MS MARCO access in notebooks

---

## Quick Start

### 1. Collect Trajectories (Offline RL Dataset)

```bash
# From project root
poetry run python -m src.main \
    --num-trajectories 5000 \
    --policy random \
    --run-name random_5k
```

Key CLI options:
- `--policy {random,expert,stop,prf,rerank}` — action selection behavior
- `--qrels-path`, `--queries-path` — defaults point to MS MARCO dev2 files under `notebooks/`
- `--run-name` — groups checkpoints/outputs; rerun the same command to resume

Each completed trajectory is checkpointed to `checkpoints/<run-name>/<hash>.json`. The final exports are:
- `outputs/trajectories_<run-name>.csv`
- `outputs/mdp_dataset_<run-name>.h5` (d3rlpy MDPDataset)

### 2. Train the Policy

```bash
poetry run python -m src.train_cql
```

Trains Discrete CQL on `outputs/mdp_dataset_trajectories_two_policies_5k.h5` and saves `outputs/discrete_cql_policy.d3` (edit paths at the top of `src/train_cql.py` as needed).

### 3. Evaluate the Policy

```bash
poetry run python -m src.test_rl_policy
```

Runs the trained policy on held-out queries and writes `outputs/cql_test_results2.csv`.

### Using Agents Directly

```python
from src.agents import ReformulationAgent, RerankingAgent, PRFAgent
from src.utils.retriever import create_retriever_callable

# Load embedding model
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize agents
reformulate_agent = ReformulationAgent(embed_model=embed_model)
rerank_agent = RerankingAgent(embed_model=embed_model)
prf_agent = PRFAgent(embed_model=embed_model, num_expansion_terms=5)

# Each agent exposes compute_effects(query_features) -> effects_dict
results = reformulate_agent.compute_effects({
    'query_text': 'machine learning optimization',
    ...
})
```

---

## Data

### MS MARCO Queries & Qrels

The pipeline uses MS MARCO relevance judgments:

- **Queries:** `notebooks/queries/topics.ms-marco-dev2.tsv` (query_id, query_text)
- **Qrels:** `notebooks/qrels/qrels.ms-marco-dev2.tsv` (query_id, iteration, doc_id, relevance_grade)

Additional query/qrels variants (TREC DL, combined sets) live in the same folders for experimentation.

---

## Metrics & Evaluation

The simulation computes:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **NDCG** | ∑ (2^rel - 1) / log(rank+1) | Ranking quality (0-1, higher better) |
| **Recall@k** | relevant_retrieved / all_relevant | Fraction of relevant docs in top-k |
| **Reward** | α·ΔNDCG + β·ΔRecall − γ·cost − δ·latency | Multi-objective RL signal |

---

## Development

### Project Layout Rationale

- **`src/core/`**: Base framework (AgentBase, abstract interfaces)
- **`src/agents/`**: Domain-specific implementations (all agents here)
- **`src/utils/`**: Reusable utilities (retriever)
- **`src/main.py`**: Resumable trajectory collection with per-query checkpoints
- **`src/simulation.py`**: MDP state machine and metrics
- **`notebooks/`**: Experiments — retrieval endpoint, agent comparisons, dataset inspection, policy training/comparison

### Adding New Agents

1. Create `src/agents/my_agent.py`:
```python
from ..core.agents import AgentBase

class MyCustomAgent(AgentBase):
    def __init__(self, embed_model, ...):
        super().__init__(agent_id=4, embed_model=embed_model)

    def compute_effects(self, query_features):
        # Implementation
        return {
            'new_doc_ids': [...],
            'new_doc_scores': [...],
            'elapsed_time': ...
        }
```

2. Export in `src/agents/__init__.py`

3. Register the action in `src/simulation.py` (action constant, name, cost) and integrate in the state builder

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'openai'` | Run `poetry install` to sync dependencies |
| `OPENROUTER_API_KEY not found` | Create `.env.txt` in project root with API key |
| Import errors from agents | Ensure running with `poetry run` from project root |
| Trajectory run interrupted | Rerun the exact same command — completed checkpoints are skipped automatically |

---

## References

- **MS MARCO**: [Microsoft](https://microsoft.github.io/msmarco/)
- **d3rlpy**: [Offline RL library](https://d3rlpy.readthedocs.io/)
- **CrossEncoder**: [Hugging Face Sentence Transformers](https://www.sbert.net/docs/pretrained-models/cross-encoders.html)
- **SentenceTransformers**: [SBERT Documentation](https://www.sbert.net/)

---

## License



## Authors

Hana Zouari
