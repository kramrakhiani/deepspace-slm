"""
DeepSpace-SLM Inference
========================
Multi-agent inference framework with domain-specific agents
coordinated through a typed message bus.

Components:
    - InferenceEngine:    Core autoregressive generation engine
    - HabitatAgent:       Backward-compatible facade over agent coordinator
    - AgentCoordinator:   Central message router and orchestrator
    - Domain Agents:      InventoryAgent, RadiationAgent, MaintenanceAgent,
                          NavigationAgent, InferenceAgent
"""

from inference.engine import InferenceEngine
from inference.habitat_agent import HabitatAgent
from inference.agents.coordinator import AgentCoordinator

__all__ = ["InferenceEngine", "HabitatAgent", "AgentCoordinator"]
