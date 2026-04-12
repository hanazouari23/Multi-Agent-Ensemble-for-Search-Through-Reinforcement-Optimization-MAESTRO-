"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

A modular framework for multi-agent retrieval optimization using reinforcement learning.
"""

__version__ = "0.1.0"

from .agents import AgentBase
from .simulation import Simulation, SimConfig, Transition
from .Reformulate import ReformulationAgent
from .rerank import RerankingAgent
from .click_reweight import ClickPriorAgent
from .retriever import Retriever, create_retriever_callable

__all__ = [
    "AgentBase",
    "Simulation",
    "SimConfig",
    "Transition",
    "ReformulationAgent",
    "RerankingAgent",
    "ClickPriorAgent",
    "Retriever",
    "create_retriever_callable",
]
