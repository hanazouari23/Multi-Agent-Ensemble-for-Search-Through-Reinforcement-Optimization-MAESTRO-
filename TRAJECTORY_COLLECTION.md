# Batch Trajectory Collection for Offline RL Training

The MAESTRO project now has complete support for generating trajectories at scale for offline reinforcement learning training.

## Overview

The trajectory collection system provides:

1. **Batch Generation**: Collect hundreds/thousands of trajectories across multiple queries
2. **Flexible Sampling**: Support for random, stratified, and sequential query sampling
3. **Multiple Policies**: Generate trajectories with different exploration strategies (random, expert oracle)
4. **Efficient Storage**: Export to NPZ (efficient binary) and JSON (human-readable) formats
5. **Statistics Reporting**: Detailed summaries of trajectory quality and composition

## Core Components

### `TrajectoryCollector` (src/utils/trajectory_collector.py)

Main class orchestrating batch collection:

```python
from src.utils.trajectory_collector import TrajectoryCollector

collector = TrajectoryCollector(
    simulation=sim,           # Your Simulation instance
    queries=query_list,       # List of query strings
    query_metadata=metadata   # Optional: per-query doc_ids, scores, qrels
)

# Collect 1000 trajectories
trajectories, stats = collector.collect_batch(
    n_trajectories=1000,
    sampling_strategy="random",    # or "stratified", "sequential"
    policies=["random", "expert"],  # Policies to use
    filter_fn=None                 # Optional: filter transitions
)

# Save to disk
collector.save_trajectories_npz(trajectories, "output/trajectories.npz")
collector.save_trajectories_json(trajectories, "output/trajectories.json")
collector.save_stats_json(stats, "output/stats.json")

# Print report
print(TrajectoryCollector.report_stats(stats))
```

### `Transition` Data Structure (src/simulation.py)

Each trajectory is a list of `Transition` objects:

```python
@dataclass
class Transition:
    state:      np.ndarray           # (786,) - policy network input
    action:     int                  # 0-3 (QR, RR, CP, STOP)
    reward:     float                # Multi-objective reward
    next_state: np.ndarray           # (786,) - next state
    done:       bool                 # Terminal flag
    info:       Dict[str, Any]       # Diagnostics & logging
```

## Usage

### Command Line

#### Demo Mode (Single Query Test)
```bash
poetry run maestro --mode demo
```

Tests individual agents and generates one trajectory.

#### Batch Collection Mode

```bash
# Collect 1000 trajectories with random query sampling
poetry run maestro --mode collect \
    --n-trajectories 1000 \
    --sampling-strategy random \
    --output-dir trajectories

# Stratified sampling (equal distribution across queries)
poetry run maestro --mode collect \
    --n-trajectories 1000 \
    --sampling-strategy stratified \
    --output-dir trajectories

# Sequential sampling (iterate through queries)
poetry run maestro --mode collect \
    --n-trajectories 1000 \
    --sampling-strategy sequential \
    --output-dir trajectories
```

### Python API

```python
from pathlib import Path
from src.simulation import Simulation, SimConfig
from src.utils.trajectory_collector import TrajectoryCollector
from sentence_transformers import SentenceTransformer

# Setup
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sim = Simulation(encoder=encoder, ...)

# Create collector
queries = ["machine learning", "deep learning", "NLP applications", ...]
collector = TrajectoryCollector(sim, queries)

# Collect & save
trajectories, stats = collector.collect_batch(
    n_trajectories=5000,
    sampling_strategy="stratified",
    policies=["random", "expert"]
)

output_dir = Path("offline_rl_data")
output_dir.mkdir(exist_ok=True)

collector.save_trajectories_npz(trajectories, output_dir / "train.npz")
collector.save_stats_json(stats, output_dir / "stats.json")

print(collector.report_stats(stats))
```

## Output Format

### NPZ Format (Optimized for ML)
```
train.npz contains:
├── states        (N, 786) float32  - State vectors
├── actions       (N,)     int32    - Actions taken
├── rewards       (N,)     float32  - Rewards received
├── next_states   (N, 786) float32  - Next states
└── dones         (N,)     bool     - Terminal flags

Load with:
data = np.load("train.npz")
states = data['states']
actions = data['actions']
...
```

### JSON Format (Human-Readable)
```json
[
  {
    "state": [0.12, -0.45, ..., 0.89],     // 786-d state vector
    "action": 1,                            // Action ID
    "reward": 0.25,                         // Reward
    "next_state": [0.14, -0.43, ..., 0.91],
    "done": false,                          // Episode ended?
    "info": {
      "query": "original query",
      "new_query": "reformulated query",
      "step": 0,
      "action_name": "Rerank",
      "ndcg_before": 0.45,
      "ndcg_after": 0.52,
      "recall_before": 0.30,
      "recall_after": 0.35,
      ...
    }
  },
  ...
]
```

### Statistics Report

```
╔══ Trajectory Collection Statistics ══════════════════╗
║ Trajectories:        5000                            ║
║ Total Transitions:   9875                            ║
║ Avg Traj Length:     1.98                            ║
║ Traj Length Range:   [1, 2]                          ║
║                                                      ║
║ Reward Statistics:                                   ║
║   Mean:                0.1234                        ║
║   Std:                 0.4567                        ║
║   Range:              [-1.2345, 2.3456]              ║
║                                                      ║
║ Actions:  QueryReform: 2500, Rerank: 2000, ...      ║
║ Policies: {'random': 2500, 'expert': 2500}          ║
║ Unique Queries:     100                              ║
║ Collection Time:    342.5s                           ║
╚══════════════════════════════════════════════════════╝
```

## State Vector Layout (786 dimensions)

The state representation combines query semantics, retrieval state, and MDP context:

```
[0]       query_length        - Token count
[1:769]   query_embedding     - 768-d SBERT embedding
[769]     score_spread        - Std dev of doc scores
[770]     score_entropy       - Entropy of score distribution
[771:774] agents_used         - Binary: [qr, rr, cp] used?
[774]     step                - Normalized step (step/max_steps)
[775]     ndcg_change         - Δ NDCG@10 since episode start
[776]     recall_change       - Δ Recall@100 since episode start
[777]     elapsed_time        - Cumulative agent time (normalized ms)
[778]     cost                - Cumulative action cost
[779]     prior_coverage      - % of top-50 docs with clicks
[780]     max_prior           - Max click-prior in top-50
[781]     mean_prior          - Mean click-prior in top-50
[782:786] valid_actions       - Binary: [qr, rr, cp, stop] available?
```

## Reward Composition

Multi-objective reward function:
```
r = α·ΔNDCG@10 + β·ΔRecall@100 + ζ·Δsatisfaction - γ·action_cost - δ·step_penalty

Default weights:
  α (NDCG):         2.0     # Ranking quality importance
  β (Recall):       0.5     # Coverage importance  
  ζ (Satisfaction): 0.5     # Click signal importance
  γ (Cost):         1.0     # Penalize expensive actions
  δ (Step):         0.05    # Penalize non-terminal steps
```

## Required Data

For full functionality, provide:

1. **ORCAS Dataset** (`data/orcas.tsv`)
   - Format: TSV with "query" and "clicked_docs" columns
   - Used by ClickPriorAgent for click signals
   - Loaded automatically on startup

2. **Queries**
   - Extracted from ORCAS or provided manually
   - System uses ~100 unique queries per batch for efficiency

3. **Optional Query Metadata**
   - Document lists per query
   - BM25 scores
   - qrels (relevance labels) for evaluation

## Scaling Recommendations

- **Small (1-100 trajectories)**: ~1-5 min, 10-50 MB
- **Medium (100-1000 trajectories)**: ~10-50 min, 50-500 MB
- **Large (1000-10000 trajectories)**: ~2-8 hours, 500MB-5GB
- **XL (10000+ trajectories)**: Run overnight, partition results

### Memory Efficiency Tips

1. Use NPZ format (4-10x smaller than JSON)
2. Batch collection by policy (separate random from expert)
3. Filter low-quality transitions before saving
4. Stream to disk if running >50k trajectories

## Next Steps: Training an Offline RL Policy

Once you have trajectories saved:

```python
# Load for training
import numpy as np

data = np.load("trajectory_data/train.npz")
states = data['states']          # (N, 786)
actions = data['actions']        # (N,)
rewards = data['rewards']        # (N,)
next_states = data['next_states'] # (N, 786)
dones = data['dones']            # (N,)

# Feed into offline RL algorithm:
# - CQL (Conservative Q-Learning)
# - IQL (Implicit Q-Learning)
# - AWR (Advantage Weighted Regression)
# - AWAC (Action-weighted Actor-Critic)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No queries in ORCAS index" | Provide `data/orcas.tsv` or use sample queries |
| Slow collection | Use "random" policy, reduce n_trajectories, use fewer queries |
| Memory error | Switch to NPZ format, partition into batches |
| Low reward trajectories | Adjust MultiObjective weights in SimConfig |
| STOP action dominating | Reduce max_steps or adjust reward_delta penalty |

## References

- **D4RL** (Trajectory Format): https://github.com/rail-berkeley/d4rl
- **Offline RL Algorithms**: https://arxiv.org/abs/2005.01643 (CQL paper)
- **MDP State Design**: Follows standard Deep RL conventions from Schulman et al.
