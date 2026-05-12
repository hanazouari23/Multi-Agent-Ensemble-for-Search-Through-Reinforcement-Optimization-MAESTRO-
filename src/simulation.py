from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy

from .core.agents import AgentBase

logger = logging.getLogger(__name__)

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_QR   = 0
ACTION_RR   = 1
ACTION_CP   = 2
ACTION_STOP = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_QR:   "QueryReform",
    ACTION_RR:   "Rerank",
    ACTION_CP:   "ClickPrior",
    ACTION_STOP: "STOP",
}
ACTION_COSTS: Dict[int, float] = {
    ACTION_QR:   0.80,
    ACTION_RR:   0.10,
    ACTION_CP:   0.01,
    ACTION_STOP: 0.00,
}
N_AGENTS = 3  # QR, RR, CP — excludes STOP


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Hyper-parameters for the simulation environment."""

    # MDP
    max_steps:    int   = 2
    n_actions:    int   = 4

    # Retrieval windows
    top_k_rerank: int   = 50
    ndcg_k:       int   = 10
    recall_k:     int   = 100

    # Reward weights:   r = α·ΔNDCG + β·ΔRecall + ζ·Δsat − γ·cost − δ·step
    reward_alpha: float = 2.0   # ΔNDCG@10 weight
    reward_beta:  float = 0.5   # ΔRecall@100 weight
    reward_gamma: float = 1.0   # cost penalty weight
    reward_delta: float = 0.05  # per non-terminal step penalty

    # Elapsed-time normalisation divisor (ms). Divide raw ms by this.
    elapsed_time_norm: float = 10_000.0

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
    # │ [394]     cost             1  (cumulative)                            │                                         │
    # │ [395:399] valid_actions    4                                          │
    # │ Total = 1+384+1+1+3+1+1+1+1+1+1+1+1+4 = 402                        │
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
        """
        NDCG@k with graded relevance (TREC formulation).
        
        Parameters
        ----------
        ranked_doc_ids : List[str]
            Ranked list of document IDs
        qrels : Dict[str, int]
            Relevance labels: {doc_id: relevance_grade}
        k : int
            Evaluation cutoff rank
        
        Returns
        -------
        float
            NDCG@k score in [0, 1]
        """
        gains = [qrels.get(d, 0) for d in ranked_doc_ids]
        ideal = sorted(qrels.values(), reverse=True)
        idcg = Simulation._dcg(np.array(ideal, dtype=float), k)
        return (
            Simulation._dcg(np.array(gains, dtype=float), k) / idcg
            if idcg > 0
            else 0.0
        )
    
    @staticmethod
    def compute_recall(
        ranked_doc_ids: List[str],
        qrels: Dict[str, int],
        k: int = 100,
    ) -> float:
        """
        Recall@k (binary: relevant if relevance > 0).
        
        Parameters
        ----------
        ranked_doc_ids : List[str]
            Ranked list of document IDs
        qrels : Dict[str, int]
            Relevance labels: {doc_id: relevance_grade}
        k : int
            Evaluation cutoff rank
        
        Returns
        -------
        float
            Recall@k score in [0, 1]
        """
        relevant = {d for d, r in qrels.items() if r > 0}
        if not relevant:
            return 0.0
        retrieved = set(ranked_doc_ids[:k])
        return len(retrieved & relevant) / len(relevant)
 
    @staticmethod
    def _valid_actions_mask(agents_used: List[bool]) -> List[int]:
        """
        Compute which actions are currently available.

        An agent action is invalid if it has already been used this episode.
        STOP is always valid (action index 3).

        Returns
        -------
        List[int]  – binary mask of length 4: [qr, rr, cp, stop]
        """
        return [
            int(not agents_used[0]),   # QR
            int(not agents_used[1]),   # RR
            int(not agents_used[2]),   # PRF
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
        ndcg_change:  float = 0.0,
        recall_change: float = 0.0,
        elapsed_ms:   float = 0.0,
        cum_cost:     float = 0.0,
    ) -> np.ndarray:
        """
        Assemble the full state vector for the current MDP step.

        State layout (dim = 783):
            [0]       query_length     whitespace token count
            [1:769]   query_embedding  768-d float32
            [769]     score_spread     std(top-k scores)
            [770]     score_entropy    H(softmax(scores))
            [771:774] agents_used      binary [qr_used, rr_used, cp_used]
            [774]     step             normalised  step / max_steps
            [775]     ndcg_change      ΔNDCG@10 vs episode baseline
            [776]     recall_change    ΔRecall@100 vs episode baseline
            [777]     elapsed_time     cumulative elapsed_ms / elapsed_time_norm
            [778]     cost             cumulative action cost this episode
            [779:783] valid_actions    binary [qr_valid, rr_valid, stop_valid]

        Parameters
        ----------
        query         : current query string (possibly reformulated)
        doc_ids       : current ranked doc IDs
        doc_scores    : retrieval scores aligned with doc_ids
        step          : 0-based step index

        
        agents_used   : which non-STOP agents have fired [qr, rr, cp]
        ndcg_change   : cumulative ΔNDCG@10 since episode start
        recall_change : cumulative ΔRecall@100 since episode start
        elapsed_ms    : cumulative wall-clock time of agent calls (ms)
        cum_cost      : cumulative cost of actions taken this episode

        Returns
        -------
        np.ndarray of shape (786,), dtype float32
        """
        cfg = self.cfg

        # ── Scalar query feature
        query_length = np.float32(len(query.split()))

        # ── Query embedding (768-d)
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
            self._valid_actions_mask(agents_used), dtype=np.float32
        )

        state = np.concatenate([
            [query_length],     # 1
            query_emb,          # 768
            [spread],           # 1
            [ent],              # 1
            used_vec,           # 3
            [norm_step],        # 1
            [nd],               # 1
            [rd],               # 1
            [elapsed_norm],     # 1
            [cost_val],         # 1
            valid_mask,              # 4
        ])                           # total: 786

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
    ) -> Tuple[str, List[str], np.ndarray, Dict[str, float], float]:
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
        metrics        : {"ndcg": float, "recall": float, "satisfaction": float}
        elapsed_ms     : wall-clock time taken by this agent call (ms)
        """
        t0 = time.perf_counter()

        if action == ACTION_STOP:
            metrics = {
                "ndcg":         Simulation.compute_ndcg(doc_ids, qrels, self.cfg.ndcg_k),
                "recall":       Simulation.compute_recall(doc_ids, qrels, self.cfg.recall_k),
            }
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            return query, list(doc_ids), doc_scores.copy(), metrics, elapsed_ms

        # Get the appropriate agent
        if action not in [ACTION_QR, ACTION_RR, ACTION_CP]:
            raise ValueError(f"Unknown action id: {action}")
        
        agent = self.agents[action]  # ACTION_QR=0, ACTION_RR=1, ACTION_PRF=2
    
        # Prepare query features for the agent
        query_features = {
            'query_text': query,
            'embedding': self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False),
            'doc_ids': doc_ids,
            'doc_scores': doc_scores,
            'retriever': self.retriever,
            'top_k_rerank': self.cfg.top_k_rerank,
        }
        
        # Call agent
        effects = agent.compute_effects(query_features)
        
        # Extract results
        new_query = effects.get('new_query_text', query)
        new_doc_ids = effects.get('new_doc_ids', doc_ids)
        new_doc_scores = effects.get('new_doc_scores', doc_scores)
        elapsed_ms = effects.get('elapsed_time', 0.0) * 1_000.0  # Convert to ms
        
        # Compute metrics on the new results
        metrics = {
            "ndcg":         Simulation.compute_ndcg(new_doc_ids, qrels, self.cfg.ndcg_k),
            "recall":       Simulation.compute_recall(new_doc_ids, qrels, self.cfg.recall_k),
        }
        
        return new_query, new_doc_ids, new_doc_scores, metrics, elapsed_ms

    # ── 3. Reward ─────────────────────────────────────────────────────────────
    def _compute_reward(
            
            
        self,
        ndcg_before:   float,
        ndcg_after:    float,
        recall_before: float,
        recall_after:  float,
        action:        int,
        done:          bool,
    ) -> float:
        """
        Multi-objective reward:

            r = α·ΔNDCG + β·ΔRecall + ζ·Δsat − γ·cost(a) − δ·step_penalty

        where:
            Δx           = x_after − x_before  (change due to this action)
            sat          = click-DCG satisfaction proxy from ORCAS
            step_penalty = reward_delta on non-terminal steps, 0 on terminal
        """
        cfg = self.cfg
        return float(
            cfg.reward_alpha * (ndcg_after   - ndcg_before)
            + cfg.reward_beta  * (recall_after - recall_before)
            - cfg.reward_gamma * ACTION_COSTS[action]
            - (0.0 if done else cfg.reward_delta)
        )

    # ── 4. generate_trajectory ────────────────────────────────────────────────
    def generate_trajectory(
        self,
        query:      str,
        doc_ids:    List[str],
        doc_scores: np.ndarray,
        qrels:      Dict[str, int],
        policy:     str = "random",
    ) -> List[Transition]:
        """
        Simulate one MDP episode for a single query.

        For each step (up to max_steps = 2):
            1. Compute valid_actions mask from agents_used
            2. Build state  s_t
            3. Select action  a_t  via policy (respects valid_actions)
            4. Call compute_effects → updated list + metrics + elapsed_ms
            5. Compute reward  r_t
            6. Advance cumulative trackers (cost, elapsed_time, metric deltas)
            7. Build next state  s_{t+1}
            8. Append Transition(s_t, a_t, r_t, s_{t+1}, done)

        Parameters
        ----------
        query      : query string
        doc_ids    : initial ranked doc IDs from base retrieval
        doc_scores : retrieval scores aligned with doc_ids (float32)
        qrels      : {doc_id: relevance_grade}  MS MARCO-style
        policy     : "random"  – uniform over valid actions
                     "expert"  – greedy ΔNDCG oracle (respects valid_actions)
                     "stop"    – always STOP immediately

        Returns
        -------
        List[Transition]  – length in [1, max_steps]
        """
        cfg          = self.cfg
        trajectory:  List[Transition] = []
        agents_used  = [False, False, False]
        cur_query    = query
        cur_ids      = list(doc_ids)
        cur_scores   = np.asarray(doc_scores, dtype=np.float32)

        # ── Episode-start baselines for cumulative delta tracking
        baseline_ndcg   = Simulation.compute_ndcg(cur_ids, qrels, cfg.ndcg_k)
        baseline_recall = Simulation.compute_recall(cur_ids, qrels, cfg.recall_k)

        cum_ndcg_change   = 0.0
        cum_recall_change = 0.0
        cum_elapsed_ms    = 0.0
        cum_cost          = 0.0

        for step in range(cfg.max_steps):
            # ── Build current state
            state = self.build_state(
                cur_query, cur_ids, cur_scores,
                step, agents_used,
                cum_ndcg_change, cum_recall_change,
                cum_elapsed_ms, cum_cost,
            )

            # ── Select action (policy respects valid_actions mask)
            valid = self._valid_actions_mask(agents_used)
            action = self._select_action(
                policy, valid,
                cur_query, cur_ids, cur_scores, qrels,
            )

            # ── Pre-action metrics snapshot
            ndcg_before   = Simulation.compute_ndcg(cur_ids, qrels, cfg.ndcg_k)
            recall_before = Simulation.compute_recall(cur_ids, qrels, cfg.recall_k)

            # ── Terminal flag
            done = (action == ACTION_STOP) or (step == cfg.max_steps - 1)

            # ── Apply action
            new_query, new_ids, new_scores, metrics, elapsed_ms = \
                self.compute_effects(action, cur_query, cur_ids, cur_scores, qrels)

            ndcg_after   = metrics["ndcg"]
            recall_after = metrics["recall"]

            # ── Reward
            reward = self._compute_reward(
                ndcg_before, ndcg_after,
                recall_before, recall_after,
                action, done,
            )

            # ── Advance cumulative trackers
            cum_elapsed_ms    += elapsed_ms
            cum_cost          += ACTION_COSTS[action]
            cum_ndcg_change    = ndcg_after   - baseline_ndcg
            cum_recall_change  = recall_after - baseline_recall

            # ── Update agents_used
            if   action == ACTION_QR: agents_used[0] = True
            elif action == ACTION_RR: agents_used[1] = True
            elif action == ACTION_CP: agents_used[2] = True

            # ── Build next state
            next_state = self.build_state(
                new_query, new_ids, new_scores,
                step + 1, agents_used,
                cum_ndcg_change, cum_recall_change,
                cum_elapsed_ms, cum_cost,
            )

            trajectory.append(Transition(
                state      = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                done       = done,
                info       = {
                    "query":          cur_query,
                    "new_query":      new_query,
                    "step":           step,
                    "action_name":    ACTION_NAMES[action],
                    "cost":           ACTION_COSTS[action],
                    "cum_cost":       cum_cost,
                    "elapsed_ms":     elapsed_ms,
                    "cum_elapsed_ms": cum_elapsed_ms,
                    "ndcg_before":    ndcg_before,
                    "ndcg_after":     ndcg_after,
                    "recall_before":  recall_before,
                    "recall_after":   recall_after,
                    "valid_actions":  valid,
                    "agents_used":    list(agents_used),
                },
            ))

            cur_query  = new_query
            cur_ids    = new_ids
            cur_scores = new_scores

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
        raise ValueError(f"Unknown policy: {policy!r}. Choose 'random', 'expert', or 'stop'.")


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
