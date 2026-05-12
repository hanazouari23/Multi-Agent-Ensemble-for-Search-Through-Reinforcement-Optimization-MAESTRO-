"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

A modular framework for multi-agent retrieval optimization using reinforcement learning.
"""

__version__ = "0.1.0"

from .core.agents import AgentBase
from .simulation import Simulation, SimConfig, Transition
from .agents import ReformulationAgent, RerankingAgent, PRFAgent
from .utils.retriever import Retriever, create_retriever_callable

__all__ = [
    "AgentBase",
    "Simulation",
    "SimConfig",
    "Transition",
    "ReformulationAgent",
    "RerankingAgent",
    "PRFAgent",
    "Retriever",
    "create_retriever_callable",
]
