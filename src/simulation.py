from __future__ import annotations

import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy

from .core.agents import AgentBase

logger = logging.getLogger(__name__)

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_QR   = 0
ACTION_RR   = 1
ACTION_PRF   = 2
ACTION_STOP = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_QR:   "QueryReform",
    ACTION_RR:   "Rerank",
    ACTION_PRF:   "PseudoRelevanceFeedback",
    ACTION_STOP: "STOP",
}
ACTION_COSTS: Dict[int, float] = {
    ACTION_QR:   0.0,  # Reformulate cost is based on token count, handled separately
    ACTION_RR:   0.0,
    ACTION_PRF:  0.0,
    ACTION_STOP: 0.0,
}
N_AGENTS = 3  # QR, RR, PRF — excludes STOP


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Hyper-parameters for the simulation environment."""

    # MDP
    max_steps:    int   = 3
    n_actions:    int   = 4

    # Retrieval windows
    top_k_rerank: int   = 50
    top_k_prf:    int   = 10   # top docs to extract PRF expansion terms from
    ndcg_k:       int   = 50
    recall_k:     int   = 100

    # Reward weights:   r = α·ΔNDCG + β·ΔRecall − γ·cost − δ·step_penalty
    reward_alpha: float = 2.0   # ΔNDCG@k weight
    reward_beta:  float = 0.5   # ΔRecall@k weight
    reward_gamma: float = 0.2   # cost penalty weight
    reward_delta: float = 0.1   # latency penalty 

    # Elapsed-time normalisation divisor (ms). Divide raw ms by this.
    elapsed_time_norm: float = 3000.0

    # Data paths
    qrels_path: Optional[str] = None      # Path to qrels file {doc_id: relevance_grade}
    queries_path: Optional[str] = None    # Path to queries file or list of query strings

    # State dimensions
    query_emb_dim: int  = 384  # all-MiniLM-L6-v2 returns 384-d embeddings
    # ┌─ Layout ──────────────────────────────────────────────────────────────┐
    # │ [0]       query_length     1                                          │
    # │ [1:385]   query_embedding  384                                        │
    # │ [385]     score_spread     1                                          │
    # │ [386]     score_entropy    1                                          │
    # │ [387:390] agents_used      3                                          │
    # │ [390]     step             1                                          │
    # │ [391]     ndcg_change      1                                          │
    # │ [392]     recall_change    1                                          │
    # │ [393]     elapsed_time     1  (normalised)                            │
    # │ [394]     cost             1  (cumulative)                            │
    # │ [395:399] valid_actions    4                                          │
    # │ Total = 1+384+1+1+3+1+1+1+1+1+4 = 399                              │
    # └──────────────────────────────────────────────────────────────────────┘
    state_dim: int = 399


# ── Transition data structure ─────────────────────────────────────────────────
@dataclass
class Transition:
    """
    Single (s, a, r, s′, done) experience tuple for offline RL training.

    Attributes
    ----------
    state      : float32 ndarray (state_dim,)
    action     : int in {0, 1, 2, 3}
    reward     : scalar float
    next_state : float32 ndarray (state_dim,)
    done       : True when episode ends (STOP chosen or max_steps reached)
    info       : auxiliary diagnostics for logging / analysis
    """
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool
    info:       Dict[str, Any] = field(default_factory=dict)




# ── Main class ────────────────────────────────────────────────────────────────
class Simulation:
    """
    MDP trajectory simulator for multi-agent retrieval offline RL.

    Parameters
    ----------
    encoder       : SentenceTransformer
                    Query encoder (msmarco-distilbert-base-v2 → 768-d).
    retriever     : Callable[[str], List[Tuple[str, float]]]
                    Base retriever: query → [(doc_id, score), …].
    agents        : List[AgentBase]
                    The three agents: [ReformulationAgent, RerankingAgent]
   
    config        : SimConfig  (defaults applied if None)
    """

    def __init__(
        self,
        encoder,
        retriever:    Callable[[str], List[Tuple[str, float]]],
        agents:       List[AgentBase],
        config:       Optional[SimConfig] = None,
    ) -> None:
        self.encoder       = encoder
        self.retriever     = retriever
        self.agents        = agents  # [qr_agent, rr_agent, cp_agent]
        self.cfg           = config or SimConfig()

    # ── Metrics (MDP-level evaluation) ────────────────────────────────────────
    @staticmethod
    def normalize_doc_id(doc_id: str) -> str:
        return doc_id.split('#', 1)[0] if '#' in doc_id else doc_id

    @staticmethod
    def deduplicate_doc_ids(doc_ids: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for doc_id in doc_ids:
            normalized = Simulation.normalize_doc_id(doc_id)
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped
    @staticmethod
    def _dcg(gains: np.ndarray, k: int) -> float:
        """Discounted cumulative gain at rank k."""
        r = np.asarray(gains[:k], dtype=float)
        if r.size == 0:
            return 0.0
        positions = np.arange(2, r.size + 2)
        return float(np.sum(r / np.log2(positions)))
    
    @staticmethod
    def compute_ndcg(
        ranked_doc_ids: List[str],
        qrels: Dict[str, int],
        k: int = 10,
    ) -> float:
        ranked_docs = Simulation.deduplicate_doc_ids(ranked_doc_ids)[:k]
        gains = [qrels.get(doc_id, 0) for doc_id in ranked_docs]
        ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
        dcg = Simulation._dcg(np.array(gains, dtype=float), k)
        idcg = Simulation._dcg(np.array(ideal, dtype=float), k)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_recall(
        ranked_doc_ids: List[str],
        qrels: Dict[str, int],
        k: int = 100,
    ) -> float:
        ranked_docs = Simulation.deduplicate_doc_ids(ranked_doc_ids)[:k]
        relevant = {d for d, r in qrels.items() if r > 0}
        if not relevant:
            return 0.0
        return len(set(ranked_docs) & relevant) / len(relevant)
 
    @staticmethod
    def _valid_actions_mask(agents_used: List[bool], current_ndcg: float) -> List[int]:
        """
        Compute which actions are currently available.

        Returns
        -------
        List[int]  – binary mask of length 4: [qr, rr, prf, stop]
        """
        return [
            int(current_ndcg < 1),   # QR valid if ndcg < 1
            int(current_ndcg > 0 and current_ndcg < 1),   # RR valid if ndcg in [0, 1]
            int(current_ndcg < 1),   # PRF valid if ndcg < 1
            1,                          # STOP always available
        ]

    @staticmethod
    def _score_spread(scores: np.ndarray) -> float:
        """Compute standard deviation of scores."""
        return float(np.std(scores)) if len(scores) > 1 else 0.0

    @staticmethod
    def _score_entropy(scores: np.ndarray) -> float:
        """Compute entropy of softmax-normalized scores."""
        probs = softmax(scores.astype(float))
        return float(scipy_entropy(probs))

    # ── 1. build_state ────────────────────────────────────────────────────────
    def build_state(
        self,
        query:        str,
        doc_ids:      List[str],
        doc_scores:   np.ndarray,
        step:         int,
        agents_used:  List[bool],
        current_ndcg: float,
        ndcg_change:  float = 0.0,
        recall_change: float = 0.0,
        elapsed_ms:   float = 0.0,
        cum_cost:     float = 0.0,
        query_length: Optional[np.ndarray] = None,
        query_emb:    Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Assemble the full state vector for the current MDP step.

        State layout (dim = 399):
            [0]       query_length     word token count (invariant across episode)
            [1:385]   query_embedding  384-d float32 (invariant across episode)
            [385]     score_spread     std(top-k scores)
            [386]     score_entropy    H(softmax(scores))
            [387:390] agents_used      one-hot [last_was_qr, last_was_rr, last_was_prf]
            [390]     step             normalised  step / max_steps
            [391]     ndcg_change      ΔNDCG@k vs episode baseline
            [392]     recall_change    ΔRecall@k vs episode baseline
            [393]     elapsed_time     cumulative elapsed_ms / elapsed_time_norm
            [394]     cost             cumulative action cost this episode
            [395:399] valid_actions    binary [qr_valid, rr_valid, prf_valid, stop_valid]

        Parameters
        ----------
        query         : original query string (for reference)
        doc_ids       : current ranked doc IDs
        doc_scores    : retrieval scores aligned with doc_ids
        step          : 0-based step index
        agents_used   : one-hot encoding of last action taken [qr, rr, prf]
        ndcg_change   : cumulative ΔNDCG since episode start
        recall_change : cumulative ΔRecall since episode start
        elapsed_ms    : cumulative wall-clock time of agent calls (ms)
        cum_cost      : cumulative cost of actions taken this episode
        query_length  : precomputed query length (optional, computed if None)
        query_emb     : precomputed query embedding (optional, computed if None)

        Returns
        -------
        np.ndarray of shape (399,), dtype float32
        """
        cfg = self.cfg

        # ── Scalar query feature (invariant across episode)
        if query_length is None:
            query_length = np.float32(len(query.split()))

        # ── Query embedding (384-d, invariant across episode)
        if query_emb is None:
            query_emb = self.encoder.encode(
                query, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)

        # ── Retrieval distribution features
        spread = np.float32(self._score_spread(doc_scores))
        ent    = np.float32(self._score_entropy(doc_scores))

        # ── Agent & step metadata
        used_vec  = np.array(agents_used[:N_AGENTS], dtype=np.float32)
        norm_step = np.float32(step / max(cfg.max_steps, 1))

        # ── Metric deltas (vs episode start)
        nd = np.float32(ndcg_change)
        rd = np.float32(recall_change)

        # ── Cost & time tracking
        elapsed_norm = np.float32(elapsed_ms / cfg.elapsed_time_norm)
        cost_val     = np.float32(cum_cost)

        # ── Valid-action mask
        valid_mask = np.array(
            self._valid_actions_mask(agents_used, current_ndcg), dtype=np.float32
        )

        state = np.concatenate([
            [query_length],     # 1
            query_emb,          # 384
            [spread],           # 1
            [ent],              # 1
            used_vec,           # 3
            [norm_step],        # 1
            [nd],               # 1
            [rd],               # 1
            [elapsed_norm],     # 1
            [cost_val],         # 1
            valid_mask,         # 4
        ])                      # total: 399

        assert state.shape[0] == cfg.state_dim, (
            f"State dim mismatch: expected {cfg.state_dim}, got {state.shape[0]}"
        )
        return state

    # ── 2. compute_effects ────────────────────────────────────────────────────
    def compute_effects(
        self,
        action:     int,
        query:      str,
        doc_ids:    List[str],
        doc_scores: np.ndarray,
        qrels:      Dict[str, int],
        corpus_data: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str], np.ndarray, Dict[str, float], float, float]:
        """
        Apply the chosen agent action; return updated retrieval state + metrics.

        Parameters
        ----------
        action     : ACTION_QR / ACTION_RR / ACTION_PRF / ACTION_STOP
        query      : current query string
        doc_ids    : current ranked document IDs
        doc_scores : retrieval scores aligned with doc_ids
        qrels      : {doc_id: relevance_grade}

        Returns
        -------
        new_query      : (possibly reformulated) query string
        new_doc_ids    : updated ranked document list
        new_doc_scores : updated scores
        metrics        : {"ndcg": float, "recall": float}
        elapsed_ms     : wall-clock time taken by this agent call (ms)
        """
        t0 = time.perf_counter()

        if action == ACTION_STOP:
            metrics = {
                "ndcg":         Simulation.compute_ndcg(doc_ids, qrels, self.cfg.ndcg_k),
                "recall":       Simulation.compute_recall(doc_ids, qrels, self.cfg.recall_k),
            }
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            return query, list(doc_ids), doc_scores.copy(), metrics, elapsed_ms, 0.0

        # Get the appropriate agent
        if action not in [ACTION_QR, ACTION_RR, ACTION_PRF]:
            raise ValueError(f"Unknown action id: {action}")
        
        agent = self.agents[action]  # ACTION_QR=0, ACTION_RR=1, ACTION_PRF=2
    
        # Prepare query features for the agent
        query_features = {
            'query_text': query,
            'embedding': self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False),
            'doc_ids': doc_ids,
            'doc_scores': doc_scores,
            'retriever': self.retriever,
            'corpus': corpus_data or {},  # Document text mapping for RerankingAgent
            'top_k': self.cfg.top_k_rerank,           # ReformulationAgent expects 'top_k'
            'top_k_rerank': self.cfg.top_k_rerank,   # RerankingAgent expects 'top_k_rerank'
            'top_k_prf': self.cfg.top_k_prf,         # PRFAgent expects 'top_k_prf'
        }
        
        # Call agent
        effects = agent.compute_effects(query_features)
        
        # Extract results
        new_query = effects.get('new_query_text', query)
        new_doc_ids = effects.get('new_doc_ids', doc_ids)
        new_doc_scores = effects.get('new_doc_scores', doc_scores)
        elapsed_ms = effects.get('elapsed_time', 0.0) * 1000.0  # Convert to ms
        cost = effects.get('cost')  # Agent-calculated cost
        
        # Compute metrics on the new results
        metrics = {
            "ndcg":         Simulation.compute_ndcg(new_doc_ids, qrels, self.cfg.ndcg_k),
            "recall":       Simulation.compute_recall(new_doc_ids, qrels, self.cfg.recall_k),
        }
        
        return new_query, new_doc_ids, new_doc_scores, metrics, elapsed_ms, cost

    # ── 3. Reward ─────────────────────────────────────────────────────────────
    def _compute_reward(
        self,
        ndcg_before:   float,
        ndcg_after:    float,
        recall_before: float,
        recall_after:  float,
        action:        int,
        elapsed_ms:    float,
        action_cost:   float = 0.0,
    ) -> float:
        """
        Multi-objective reward:

            r = α·ΔNDCG + β·ΔRecall − γ·cost(a) − δ·step_penalty

        where:
            Δx           = x_after − x_before  (change due to this action)
            cost(a)      = cost returned by agent (e.g., token count for QueryReform)
        """
        cfg = self.cfg
        ndcg_gain = ndcg_after - ndcg_before
        recall_gain = recall_after - recall_before
        quality_reward = cfg.reward_alpha * ndcg_gain
        recall_reward = cfg.reward_beta * recall_gain
        cost_penalty = 0.2 * action_cost
        time_penalty = 0.1 * (elapsed_ms / cfg.elapsed_time_norm)
        print(
            "reward action=%s: ndcg=%+.6f, recall=%+.6f, "
            "quality=%+.6f, recall_term=%+.6f, "
            "cost=-%.6f, time=-%.6f, total=%+.6f","elapsed_ms=%.2f", "action_cost=%.6f",
            ACTION_NAMES[action],
            ndcg_gain,
            recall_gain,
            quality_reward,
            recall_reward,
            time_penalty,
            cost_penalty,
            elapsed_ms/cfg.elapsed_time_norm,
            action_cost,
       )
        if action == ACTION_STOP:
             return float(0.0)
        else:
              return float(
             2 * (ndcg_after   - ndcg_before)
            + 1  * (recall_after - recall_before)
            - (0.2 * action_cost)
            - 0.1 * (elapsed_ms / cfg.elapsed_time_norm)
            )
        

    # ── 4. generate_trajectory ────────────────────────────────────────────────
    def generate_trajectory(
        self,
        query: str,
        doc_ids: List[str],
        doc_scores: np.ndarray,
        qrels: Dict[str, int],
        policy: str = "random",
        corpus_data: Optional[Dict[str, str]] = None,
    ) -> List[Transition]:
        cfg = self.cfg
        trajectory: List[Transition] = []
        agents_used = [False, False, False]
        cur_query = query
        cur_ids = list(doc_ids)
        cur_scores = np.asarray(doc_scores, dtype=np.float32)

        if corpus_data is None:
            raise ValueError("corpus_data must be provided with actual document text for agents")
        cur_corpus = corpus_data.copy()

        query_length = np.float32(len(query.split()))
        query_emb = self.encoder.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        baseline_ndcg = Simulation.compute_ndcg(cur_ids, qrels, cfg.ndcg_k)
        baseline_recall = Simulation.compute_recall(cur_ids, qrels, cfg.recall_k)

        cum_ndcg_change = 0.0
        cum_recall_change = 0.0
        cum_elapsed_ms = 0.0
        cum_cost = 0.0

        for step in range(cfg.max_steps):
            state = self.build_state(
                cur_query, cur_ids, cur_scores,
                step, agents_used, baseline_ndcg,
                cum_ndcg_change, cum_recall_change,
                cum_elapsed_ms, cum_cost,
                query_length=query_length, query_emb=query_emb)

            ndcg_before = Simulation.compute_ndcg(cur_ids, qrels, cfg.ndcg_k)
            recall_before = Simulation.compute_recall(cur_ids, qrels, cfg.recall_k)

            valid = self._valid_actions_mask(agents_used, ndcg_before)

            if policy == "expert":
                action, effects = self._policy_expert_two_step(
                    cur_query,
                    cur_ids,
                    cur_scores,
                    qrels,
                    valid,
                    corpus_data=cur_corpus,
                )
                new_query, new_ids, new_scores, metrics, elapsed_ms, action_cost = effects
            else:
                action = self._select_action(
                    policy, valid,
                    cur_query, cur_ids, cur_scores, qrels,
                )
                new_query, new_ids, new_scores, metrics, elapsed_ms, action_cost = \
                    self.compute_effects(
                        action,
                        cur_query,
                        cur_ids,
                        cur_scores,
                        qrels,
                        corpus_data=cur_corpus,
                    )

            done = (action == ACTION_STOP) or (step == cfg.max_steps - 1)

            ndcg_after = metrics["ndcg"]
            recall_after = metrics["recall"]

            reward = self._compute_reward(
                ndcg_before, ndcg_after,
                recall_before, recall_after,
                action, elapsed_ms, action_cost
            )

            cum_elapsed_ms += elapsed_ms
            cum_cost += action_cost
            cum_ndcg_change = ndcg_after - baseline_ndcg
            cum_recall_change = recall_after - baseline_recall

            if action == ACTION_QR:
                agents_used = [True, False, False]
            elif action == ACTION_RR:
                agents_used = [False, True, False]
            elif action == ACTION_PRF:
                agents_used = [False, False, True]
            else:
                agents_used = [False, False, False]

            next_state = self.build_state(
                new_query, new_ids, new_scores,
                step + 1, agents_used, ndcg_after,
                cum_ndcg_change, cum_recall_change,
                cum_elapsed_ms, cum_cost,
                query_length=query_length, query_emb=query_emb,
            )

            trajectory.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info={
                    "query": cur_query,
                    "new_query": new_query,
                    "step": step,
                    "action_name": ACTION_NAMES[action],
                    "cost": action_cost,
                    "cum_cost": cum_cost,
                    "elapsed_ms": elapsed_ms,
                    "cum_elapsed_ms": cum_elapsed_ms,
                    "ndcg_before": ndcg_before,
                    "ndcg_after": ndcg_after,
                    "recall_before": recall_before,
                    "recall_after": recall_after,
                    "valid_actions": valid,
                    "agents_used": list(agents_used),
                },
            ))

            cur_query = new_query
            cur_ids = new_ids
            cur_scores = np.asarray(new_scores, dtype=np.float32)

            if done:
                break

        return trajectory

    # ── 5. Action-selection policies ─────────────────────────────────────────
    def _select_action(
            self,
            policy:     str,
            valid:      List[int],
            query:      str,
            doc_ids:    List[str],
            doc_scores: np.ndarray,
            qrels:      Dict[str, int],
        ) -> int:
            if policy == "random":
                return self._policy_random(valid)
            if policy == "expert":
                return self._policy_expert(query, doc_ids, doc_scores, qrels, valid)
            if policy == "stop":
                return ACTION_STOP
            if policy == "prf":
                return ACTION_PRF
            raise ValueError(f"Unknown policy: {policy!r}. Choose 'random', 'expert', or 'stop'.")
            

    def _policy_random(self, valid: List[int]) -> int:
            """
            Uniformly sample a valid action.

            Parameters
            ----------
            valid : List[int]
                Binary mask of valid actions [qr_valid, rr_valid, prf_valid, stop_valid]

            Returns
            -------
            int
                Action index (0-3)
            """
            valid_indices = [i for i, v in enumerate(valid) if v == 1]
            return random.choice(valid_indices)

    def _policy_expert(
        self,
        query:      str,
        doc_ids:    List[str],
        doc_scores: np.ndarray,
        qrels:      Dict[str, int],
        valid:      List[int],
    ) -> int:
        """
        Greedy oracle policy: try each valid action and pick the one with best ΔNDCG.

        If multiple actions tie for best ΔNDCG, prefer STOP (to terminate early and save cost).

        Parameters
        ----------
        query      : query string
        doc_ids    : current ranked doc IDs
        doc_scores : retrieval scores
        qrels      : relevance judgments {doc_id: grade}
        valid      : binary mask of valid actions

        Returns
        -------
        int
            Action with best ΔNDCG improvement (or STOP if tied)
        """
        cfg = self.cfg

        # Current NDCG baseline
        ndcg_current = Simulation.compute_ndcg(doc_ids, qrels, cfg.ndcg_k)

        # Try each action and compute ΔNDCG
        action_deltas = {}
        effects = {}
        for action in range(N_AGENTS):  # QR, RR, PRF (excluding STOP)
            if not valid[action]:
                continue  # Skip invalid actions

            try:
                new_query, new_ids, new_scores, metrics, elapsed_ms, cost = \
                    self.compute_effects(action, query, doc_ids, doc_scores, qrels)
                ndcg_new = metrics["ndcg"]
                action_deltas[action] = ndcg_new - ndcg_current
                effects[action] = (new_query, new_ids, new_scores, metrics, elapsed_ms, cost)
            except Exception as e:
                # If an agent fails, penalize that action heavily
                logger.warning(f"Expert policy: action {ACTION_NAMES[action]} failed: {e}")
                action_deltas[action] = float('-inf')

        # STOP is always an option (ΔNDCG = 0, no change)
        if valid[ACTION_STOP]:
            action_deltas[ACTION_STOP] = 0.0
            effects[ACTION_STOP] = (
                query,
                doc_ids,
                doc_scores,
                {
                    "ndcg": ndcg_current,
                    "recall": Simulation.compute_recall(doc_ids, qrels, cfg.recall_k),
                },
                0.0,
                0.0,
            )

        # Pick action with best ΔNDCG; if tied, prefer STOP
        if not action_deltas:
            return ACTION_STOP  # Fallback: no valid actions

        best_delta = max(action_deltas.values())
        best_actions = [a for a, d in action_deltas.items() if d == best_delta]

        # Prefer STOP if it's tied for best
        if ACTION_STOP in best_actions:
            best_action = ACTION_STOP
        else:
            best_action = best_actions[0]

        return best_action, effects.get(best_action)  # Pick the first best action (deterministic)


    def _policy_expert_two_step(
        self,
        query: str,
        doc_ids: List[str],
        doc_scores: np.ndarray,
        qrels: Dict[str, int],
        valid: List[int],
        corpus_data: Optional[Dict[str, str]] = None,
    ):
        cfg = self.cfg
        ndcg_current = Simulation.compute_ndcg(doc_ids, qrels, cfg.ndcg_k)

        stop_effect = (
            query,
            list(doc_ids),
            np.asarray(doc_scores, dtype=np.float32).copy(),
            {
                "ndcg": ndcg_current,
                "recall": Simulation.compute_recall(doc_ids, qrels, cfg.recall_k),
            },
            0.0,
            0.0,
        )

        best_nonstop_action = None
        best_nonstop_value = float("-inf")
        best_nonstop_effect = None
        eps = 1e-12

        for action1 in range(N_AGENTS):
            if not valid[action1]:
                continue

            try:
                query1, ids1, scores1, met1, time1, cost1 = self.compute_effects(
                    action1,
                    query,
                    doc_ids,
                    doc_scores,
                    qrels,
                    corpus_data=corpus_data,
                )

                best_after_action1 = met1["ndcg"]

                if action1 == ACTION_QR:
                    agents_used_2 = [True, False, False]
                elif action1 == ACTION_RR:
                    agents_used_2 = [False, True, False]
                else:
                    agents_used_2 = [False, False, True]

                valid2 = self._valid_actions_mask(agents_used_2, best_after_action1)

                for action2 in range(N_AGENTS):
                    if not valid2[action2]:
                        continue

                    try:
                        query2, ids2, scores2, met2, time2, cost2 = self.compute_effects(
                            action2,
                            query1,
                            ids1,
                            scores1,
                            qrels,
                            corpus_data=corpus_data,
                        )
                        best_after_action1 = max(best_after_action1, met2["ndcg"])
                    except Exception as exc:
                        logger.warning(
                            "Expert two-step: second-step action %s failed after %s: %s",
                            ACTION_NAMES[action2],
                            ACTION_NAMES[action1],
                            exc,
                        )

                if best_after_action1 > best_nonstop_value + eps:
                    best_nonstop_value = best_after_action1
                    best_nonstop_action = action1
                    best_nonstop_effect = (query1, ids1, scores1, met1, time1, cost1)

            except Exception as exc:
                logger.warning(
                    "Expert two-step: first-step action %s failed: %s",
                    ACTION_NAMES[action1],
                    exc,
                )

        if best_nonstop_action is None:
            return ACTION_STOP, stop_effect

        if best_nonstop_value > ndcg_current + eps:
            return best_nonstop_action, best_nonstop_effect

        return ACTION_STOP, stop_effect


    # ── 7. Serialisation ──────────────────────────────────────────────────────
    @staticmethod
    def to_arrays(
        dataset: List[Transition],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack a list of Transition objects into numpy arrays for training.

        Returns
        -------
        states      : float32  (N, state_dim)
        actions     : int64    (N,)
        rewards     : float32  (N,)
        next_states : float32  (N, state_dim)
        dones       : bool     (N,)
        """
        states      = np.stack([t.state      for t in dataset]).astype(np.float32)
        actions     = np.array([t.action     for t in dataset], dtype=np.int64)
        rewards     = np.array([t.reward     for t in dataset], dtype=np.float32)
        next_states = np.stack([t.next_state for t in dataset]).astype(np.float32)
        dones       = np.array([t.done       for t in dataset], dtype=bool)
        return states, actions, rewards, next_states, dones

    @staticmethod
    def export_trajectories_to_csv(
        trajectories: List[List[Transition]],
        filename: str = "trajectories.csv",
    ) -> Path:
        """
        Export trajectories to a CSV file in the trajectories folder.
        Includes classic offline RL dataset fields: state, action, reward, next_state, done.

        Parameters
        ----------
        trajectories : List[List[Transition]]
            List of trajectories, where each trajectory is a list of Transition objects.
        filename : str
            Name of the CSV file to create (default: "trajectories.csv")

        Returns
        -------
        Path
            Absolute path to the created CSV file
        """
        # Determine trajectories folder (relative to this file)
        sim_file = Path(__file__).resolve()
        traj_folder = sim_file.parent.parent / "trajectories"
        traj_folder.mkdir(exist_ok=True)
        
        csv_path = traj_folder / filename
        
        # Flatten all transitions across all trajectories
        all_transitions = []
        for traj_idx, traj in enumerate(trajectories):
            for step_idx, trans in enumerate(traj):
                all_transitions.append((traj_idx, step_idx, trans))
        
        # Write to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                # Trajectory tracking
                'trajectory_id',
                'step',
                # Query information
                'query',
                'new_query',
                # Action tracking
                'action',
                'action_name',
                # Classic offline RL fields
                #'state',
                'reward',
                #'next_state',
                'done',
                # Metrics and diagnostics
                'ndcg_before',
                'ndcg_after',
                'recall_before',
                'recall_after',
                'cost',
                'cum_cost',
                'elapsed_ms',
                'cum_elapsed_ms',
            ])
            writer.writeheader()
            
            for traj_id, step, trans in all_transitions:
                info = trans.info
                # Serialize state vectors as comma-separated values (within the CSV cell)
                state_str = ','.join([f"{x:.6f}" for x in trans.state])
                next_state_str = ','.join([f"{x:.6f}" for x in trans.next_state])
                
                writer.writerow({
                    # Trajectory tracking
                    'trajectory_id': traj_id,
                    'step': step,
                    # Query information
                    'query': info.get('query', ''),
                    'new_query': info.get('new_query', ''),
                    # Action tracking
                    'action': int(trans.action),
                    'action_name': info.get('action_name', ''),
                    # Classic offline RL fields
                    #'state': state_str,
                    'reward': f"{trans.reward:.6f}",
                    #'next_state': next_state_str,
                    'done': trans.done,
                    # Metrics and diagnostics
                    'ndcg_before': f"{info.get('ndcg_before', 0.0):.6f}",
                    'ndcg_after': f"{info.get('ndcg_after', 0.0):.6f}",
                    'recall_before': f"{info.get('recall_before', 0.0):.6f}",
                    'recall_after': f"{info.get('recall_after', 0.0):.6f}",
                    'cost': f"{info.get('cost', 0.0):.6f}",
                    'cum_cost': f"{info.get('cum_cost', 0.0):.6f}",
                    'elapsed_ms': f"{info.get('elapsed_ms', 0.0):.2f}",
                    'cum_elapsed_ms': f"{info.get('cum_elapsed_ms', 0.0):.2f}",
                })
        
        logger.info(f"Trajectories exported to {csv_path}")
        return csv_path
