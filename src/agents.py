from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class AgentBase(ABC):
    def __init__(self, agent_id: int, embed_model: SentenceTransformer):
        self.agent_id = agent_id
        self.embed_model = embed_model  # Shared across agents
    
    @abstractmethod
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Core agent logic: given query features, return effects.
        MUST update/recompute embedding if query text changes.
        """
        pass
    
    def update_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional: mutate agents_used and valid_actions after this agent runs.
        """
        # Default: mark self.agent_id as used, mask it
        agents_used_slice = slice(..., ...)  # define based on your state layout
        valid_actions_slice = state[-4:]
        
        new_agents_used = state.copy()
        new_agents_used[agents_used_slice][self.agent_id] = 1.0
        new_valid_actions = valid_actions_slice.copy()
        new_valid_actions[self.agent_id] = 0.0  # can't reuse
        
        return new_agents_used, new_valid_actions
