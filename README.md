# MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

A framework for multi-agent retrieval optimization using reinforcement learning. The system learns policies to orchestrate multiple specialized agents that improve search result quality while managing computational costs.

## Overview

**Goal:** Build an intelligent system that learns which agents to invoke at each step to improve search results. The policy is trained via reinforcement learning on user satisfaction metrics.

**Key Innovation:** Multi-agent ensemble approach where:
- **ReformulationAgent** refines queries using LLM-powered rewriting
- **RerankingAgent** re-scores results using cross-encoder models
- **ClickPriorAgent** boosts clicked documents using ORCAS dataset priors

The system evaluates impact on metrics like NDCG, Recall, and User Satisfaction.

---

## Architecture

```
src/
├── core/
│   ├── agents.py           # AgentBase abstract class defining agent interface
│   └── __init__.py
├── agents/                 # Concrete agent implementations
│   ├── reformulate.py      # Query rewriting agent (OpenRouter LLM)
│   ├── rerank.py           # CrossEncoder reranking agent
│   ├── click_prior.py      # ORCAS click-prior boosting agent
│   └── __init__.py         # Agent exports
├── utils/
│   ├── retriever.py        # OpenSearch BM25 retrieval interface
│   ├── orcas_loader.py     # ORCAS TSV dataset loaders
│   └── __init__.py
├── main.py                 # Entry point & simulation launcher
├── simulation.py           # MDP orchestrator & metrics
└── __init__.py             # Package root
```

### Agents

| Agent | Purpose | Key Implementation |
|-------|---------|-------------------|
| **ReformulationAgent** (id=0) | Query optimization | LLM-based rewriting via OpenRouter (deepseek-v3.2) |
| **RerankingAgent** (id=1) | Result re-ranking | CrossEncoder (MS-MARCO-MiniLM-L-6-v2) |
| **ClickPriorAgent** (id=2) | Click bias incorporation | Binary click signals from ORCAS dataset |

### Core Classes

- **AgentBase**: Abstract base class with `compute_effects()` method defining agent interface
- **Simulation**: MDP framework orchestrating agents, managing state, and computing metrics
- **Retriever**: OpenSearch BM25 wrapper for initial retrieval

---

## Setup & Installation

### Prerequisites
- Python 3.10+
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
- The system auto-loads `.env.txt` at runtime.

### Dependencies

Core packages:
- `sentence-transformers` (5.2.3) - Embeddings and cross-encoder models
- `openai` (2.24.0) - LLM access via OpenRouter
- `numpy` (2.0) - Numerical operations
- `scipy` (1.13+) - Statistical utilities (entropy, softmax)
- `requests` (2.32+) - HTTP client for OpenSearch

---

## Quick Start

### Running the Simulation

```bash
# From project root
poetry run python -m src.main
```

This launches an interactive simulation that:
1. Loads the ORCAS dataset (click priors)
2. Creates agents and configuration
3. Runs retrieval simulation with agent orchestration
4. Reports metrics (NDCG, Recall, Satisfaction)

### Using Agents Directly

```python
from src.agents import ReformulationAgent, RerankingAgent, ClickPriorAgent
from src.core.agents import AgentBase
from src.utils.retriever import create_retriever_callable

# Load embedding model
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize agents
reformulate_agent = ReformulationAgent(embed_model=embed_model)
rerank_agent = RerankingAgent(embed_model=embed_model)
click_agent = ClickPriorAgent(embed_model=embed_model, beta=0.1)

# Each agent exposes compute_effects(query_features) -> effects_dict
results = reformulate_agent.compute_effects({
    'query_text': 'machine learning optimization',
    ...
})
```

### Loading ORCAS Dataset

```python
from src.utils.orcas_loader import load_orcas_tsv_sample

# Load full dataset
orcas_index = load_orcas_tsv_sample('data/orcas.tsv', sample_size=None)

# Load sample (faster for testing)
sample = load_orcas_tsv_sample('data/orcas.tsv', sample_size=1000)
```

---

## Data

### ORCAS Dataset

The system uses ORCAS (Open Relevance Click Analysis Set) for click priors:
- **File:** `data/orcas.tsv` (50MB+)
- **Format:** TSV with columns: query, clicked_doc_ids
- **Purpose:** ClickPriorAgent uses click signals to boost document scores

Expected structure:
```
query	                     clicked_docs
python machine learning    doc123,doc456,doc789
deep learning course       doc111,doc222
...
```

---

## Metrics & Evaluation

The simulation computes:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **NDCG** | ∑ (2^rel - 1) / log(rank+1) | Ranking quality (0-1, higher better) |
| **Recall@k** | relevant_retrieved / all_relevant | Fraction of relevant docs in top-k |
| **User Satisfaction** | Weighted combination of NDCG + click signals | Overall user experience score |

---

## Development

### Project Layout Rationale

- **`src/core/`**: Base framework (AgentBase, abstract interfaces)
- **`src/agents/`**: Domain-specific implementations (all agents here)
- **`src/utils/`**: Reusable utilities (retriever, data loaders)
- **`src/main.py`**: Entry point orchestrating simulation
- **`src/simulation.py`**: MDP state machine and metrics

### Adding New Agents

1. Create `src/agents/my_agent.py`:
```python
from ..core.agents import AgentBase

class MyCustomAgent(AgentBase):
    def __init__(self, embed_model, ...):
        super().__init__(agent_id=3, embed_model=embed_model)
    
    def compute_effects(self, query_features):
        # Implementation
        return {
            'new_doc_ids': [...],
            'new_doc_scores': [...],
            'elapsed_time': ...
        }
```

2. Export in `src/agents/__init__.py`

3. Integrate in simulation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'openai'` | Run `poetry install` to sync dependencies |
| `OPENROUTER_API_KEY not found` | Create `.env.txt` in project root with API key |
| Import errors from agents | Ensure running with `poetry run` or from project root with Python path set |
| ORCAS file not found | Download/place `data/orcas.tsv` in project |

---

## References

- **ORCAS Dataset**: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/orcas-open-source-click-annotations-for-search-evaluation/)
- **CrossEncoder**: [Hugging Face Sentence Transformers](https://www.sbert.net/docs/pretrained-models/cross-encoders.html)
- **SentenceTransformers**: [SBERT Documentation](https://www.sbert.net/)

---

## License



## Authors

Hana Zouari
