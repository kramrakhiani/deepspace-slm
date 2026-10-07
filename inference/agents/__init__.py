"""
DeepSpace-SLM Multi-Agent Framework
=====================================
Domain-specific agents coordinated through a typed message bus.

Agents:
    - InventoryAgent:    Supply chain & stock management
    - RadiationAgent:    RAP-G radiation defense coordination
    - MaintenanceAgent:  Audit logging & procedure manuals
    - NavigationAgent:   DTN mesh communications & CRDT sync
    - InferenceAgent:    Pure SLM neural text generation
"""

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)
from inference.agents.coordinator import AgentCoordinator
from inference.agents.inventory_agent import InventoryAgent
from inference.agents.radiation_agent import RadiationAgent
from inference.agents.maintenance_agent import MaintenanceAgent
from inference.agents.navigation_agent import NavigationAgent
from inference.agents.inference_agent import InferenceAgent

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentResponse",
    "AgentCapability",
    "MessagePriority",
    "AgentCoordinator",
    "InventoryAgent",
    "RadiationAgent",
    "MaintenanceAgent",
    "NavigationAgent",
    "InferenceAgent",
]
