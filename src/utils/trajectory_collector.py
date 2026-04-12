"""
Batch trajectory collection and serialization for offline RL training.

This module orchestrates large-scale trajectory generation across multiple
queries and exports trajectories to disk in standard formats (JSON, NPZ).
"""

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStats:
    """Summary statistics for a batch of trajectories."""
    total_trajectories: int
    total_transitions: int
    avg_trajectory_length: float
    min_trajectory_length: int
    max_trajectory_length: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    action_distribution: Dict[str, int]
    policy_distribution: Dict[str, int]
    unique_queries: int
    total_collection_time_s: float


class TrajectoryCollector:
    """
    Batch trajectory collector for offline RL training.

    Parameters
    ----------
    simulation : Simulation
        The MDP simulator instance
    queries : List[str]
        Query strings to sample from
    query_metadata : Dict[str, Dict[str, Any]], optional
        Per-query metadata: query → {doc_ids, doc_scores, qrels}
    """

    def __init__(
        self,
        simulation,
        queries: List[str],
        query_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.simulation = simulation
        self.queries = queries
        self.query_metadata = query_metadata or {}

    def collect_batch(
        self,
        n_trajectories: int,
        sampling_strategy: str = "random",
        policies: List[str] = None,
        filter_fn: Optional[Callable] = None,
    ) -> Tuple[List[Any], TrajectoryStats]:
        """
        Generate a batch of trajectories using specified sampling strategy.

        Parameters
        ----------
        n_trajectories : int
            Total number of trajectories to collect
        sampling_strategy : str
            "random"         – uniformly sample queries (replacement OK)
            "stratified"     – sample equally from each query
            "sequential"     – iterate through queries in order
        policies : List[str], optional
            Policies to use: ["random", "expert", "stop"]. Default: ["random"]
        filter_fn : Callable[[Transition], bool], optional
            Optional filter: only keep transitions where filter_fn(transition) == True.
            If None, keep all transitions.

        Returns
        -------
        trajectories : List[Transition]
            All collected transitions (may be shorter than n_trajectories if filtered)
        stats : TrajectoryStats
            Summary statistics of the batch
        """
        if policies is None:
            policies = ["random"]

        t0 = time.perf_counter()
        trajectories = []
        trajectory_lengths = []
        all_rewards = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # QR, RR, CP, STOP
        policy_counts = {p: 0 for p in policies}
        queries_used = set()

        logger.info(
            f"Collecting {n_trajectories} trajectories "
            f"(strategy={sampling_strategy}, policies={policies})"
        )

        collected = 0
        while collected < n_trajectories:
            # Sample query
            query = self._sample_query(sampling_strategy, collected, n_trajectories)
            queries_used.add(query)
            policy = random.choice(policies)
            policy_counts[policy] += 1

            # Get query metadata or use defaults
            metadata = self.query_metadata.get(query, {})
            doc_ids = metadata.get("doc_ids", [f"doc_{i}" for i in range(5)])
            doc_scores = metadata.get(
                "doc_scores", np.array([0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
            )
            qrels = metadata.get("qrels", {doc_ids[0]: 1, doc_ids[2]: 1})

            # Generate trajectory
            try:
                trajectory = self.simulation.generate_trajectory(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=np.asarray(doc_scores, dtype=np.float32),
                    qrels=qrels,
                    policy=policy,
                )
            except Exception as e:
                logger.warning(f"Trajectory generation failed for query '{query}': {e}")
                continue

            # Apply filter if provided
            if filter_fn is not None:
                trajectory = [t for t in trajectory if filter_fn(t)]

            if not trajectory:
                continue

            # Accumulate statistics
            trajectories.extend(trajectory)
            trajectory_lengths.append(len(trajectory))
            rewards = [t.reward for t in trajectory]
            all_rewards.extend(rewards)

            for t in trajectory:
                action_counts[t.action] += 1

            collected += 1

            if collected % 10 == 0:
                logger.info(f"  Collected {collected}/{n_trajectories} trajectories")

        # Compute statistics
        elapsed = time.perf_counter() - t0
        stats = self._compute_stats(
            trajectories,
            trajectory_lengths,
            all_rewards,
            action_counts,
            policy_counts,
            len(queries_used),
            elapsed,
        )

        logger.info(
            f"Collection complete: {len(trajectories)} transitions from "
            f"{collected} trajectories in {elapsed:.1f}s"
        )

        return trajectories, stats

    def _sample_query(self, strategy: str, collected: int, total: int) -> str:
        """Sample a query according to the specified strategy."""
        if strategy == "random":
            return random.choice(self.queries)
        elif strategy == "stratified":
            idx = (collected * len(self.queries)) // total
            return self.queries[idx % len(self.queries)]
        elif strategy == "sequential":
            return self.queries[collected % len(self.queries)]
        else:
            raise ValueError(
                f"Unknown sampling strategy: {strategy!r}. "
                "Choose 'random', 'stratified', or 'sequential'."
            )

    @staticmethod
    def _compute_stats(
        trajectories: List[Any],
        trajectory_lengths: List[int],
        all_rewards: List[float],
        action_counts: Dict[int, int],
        policy_counts: Dict[str, int],
        n_unique_queries: int,
        elapsed_s: float,
    ) -> TrajectoryStats:
        """Compute summary statistics."""
        n_traj = len(trajectory_lengths)

        return TrajectoryStats(
            total_trajectories=n_traj,
            total_transitions=len(trajectories),
            avg_trajectory_length=(
                sum(trajectory_lengths) / n_traj if n_traj > 0 else 0.0
            ),
            min_trajectory_length=min(trajectory_lengths) if trajectory_lengths else 0,
            max_trajectory_length=max(trajectory_lengths) if trajectory_lengths else 0,
            reward_mean=float(np.mean(all_rewards)) if all_rewards else 0.0,
            reward_std=float(np.std(all_rewards)) if all_rewards else 0.0,
            reward_min=float(np.min(all_rewards)) if all_rewards else 0.0,
            reward_max=float(np.max(all_rewards)) if all_rewards else 0.0,
            action_distribution=action_counts,
            policy_distribution=policy_counts,
            unique_queries=n_unique_queries,
            total_collection_time_s=elapsed_s,
        )

    def save_trajectories_json(
        self,
        trajectories: List[Any],
        filepath: Path,
        compress: bool = False,
    ) -> None:
        """
        Save trajectories to JSON (human-readable).

        Parameters
        ----------
        trajectories : List[Transition]
        filepath : Path
            Output file path
        compress : bool
            If True, save as .json.gz
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = []
        for t in trajectories:
            state_list = t.state.tolist() if isinstance(t.state, np.ndarray) else t.state
            next_state_list = (
                t.next_state.tolist() if isinstance(t.next_state, np.ndarray) else t.next_state
            )
            data.append({
                "state": state_list,
                "action": int(t.action),
                "reward": float(t.reward),
                "next_state": next_state_list,
                "done": bool(t.done),
                "info": t.info or {},
            })

        # Write
        if compress:
            import gzip

            with gzip.open(str(filepath) + ".gz", "wt") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(trajectories)} transitions to {filepath}.gz")
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(trajectories)} transitions to {filepath}")

    def save_trajectories_npz(
        self,
        trajectories: List[Any],
        filepath: Path,
    ) -> None:
        """
        Save trajectories to NPZ (efficient binary format for ML).

        Parameters
        ----------
        trajectories : List[Transition]
        filepath : Path
            Output file path (.npz extension will be added if missing)
        """
        filepath = Path(filepath)
        if filepath.suffix != ".npz":
            filepath = filepath.with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Stack arrays
        states = np.array([t.state for t in trajectories], dtype=np.float32)
        actions = np.array([t.action for t in trajectories], dtype=np.int32)
        rewards = np.array([t.reward for t in trajectories], dtype=np.float32)
        next_states = np.array([t.next_state for t in trajectories], dtype=np.float32)
        dones = np.array([t.done for t in trajectories], dtype=bool)

        # Save
        np.savez(
            filepath,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        logger.info(
            f"Saved {len(trajectories)} transitions to {filepath} "
            f"({states.nbytes / 1e6:.1f} MB)"
        )

    def save_stats_json(
        self,
        stats: TrajectoryStats,
        filepath: Path,
    ) -> None:
        """Save collection statistics to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(asdict(stats), f, indent=2)
        logger.info(f"Saved statistics to {filepath}")

    @staticmethod
    def report_stats(stats: TrajectoryStats) -> str:
        """Generate human-readable statistics report."""
        action_names = {0: "QueryReform", 1: "Rerank", 2: "ClickPrior", 3: "STOP"}
        action_str = ", ".join(
            f"{action_names.get(k, f'unknown_{k}')}: {v}"
            for k, v in sorted(stats.action_distribution.items())
        )

        return f"""
╔══ Trajectory Collection Statistics ══════════════════╗
║ Trajectories:        {stats.total_trajectories:6d}                         ║
║ Total Transitions:   {stats.total_transitions:6d}                         ║
║ Avg Traj Length:     {stats.avg_trajectory_length:6.1f}                      ║
║ Traj Length Range:   [{stats.min_trajectory_length}, {stats.max_trajectory_length}]                       ║
║                                                     ║
║ Reward Statistics:                                  ║
║   Mean:             {stats.reward_mean:8.4f}                        ║
║   Std:              {stats.reward_std:8.4f}                        ║
║   Range:            [{stats.reward_min:.4f}, {stats.reward_max:.4f}]          ║
║                                                     ║
║ Actions:            {action_str:<27}║
║ Policies:           {str(stats.policy_distribution):<27}║
║ Unique Queries:     {stats.unique_queries:6d}                         ║
║ Collection Time:    {stats.total_collection_time_s:6.1f}s                      ║
╚═════════════════════════════════════════════════════╝
"""
