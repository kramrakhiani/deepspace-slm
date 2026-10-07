"""
Agent Base Classes & Message Protocol
=======================================
Defines the abstract agent contract, typed message bus primitives,
and capability registration used by all domain agents.
"""

import enum
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MessagePriority(enum.IntEnum):
    """Message priority levels for agent routing."""
    ROUTINE = 0       # Standard queries (inventory lookups, manual searches)
    ELEVATED = 1      # Degraded-system queries (low stock warnings)
    CRITICAL = 2      # Life-safety events (radiation spikes, O2 depletion)
    EMERGENCY = 3     # Immediate crew-action required


@dataclass
class AgentCapability:
    """Declares a single capability an agent can handle.

    Attributes:
        intent: The intent string this capability handles (e.g. "check_stock").
        description: Human-readable description for discovery.
        priority_floor: Minimum priority level this capability will accept.
    """
    intent: str
    description: str
    priority_floor: MessagePriority = MessagePriority.ROUTINE


@dataclass
class AgentMessage:
    """Typed message passed between agents via the coordinator.

    Attributes:
        intent: Identifies the action requested (e.g. "check_stock", "radiation_telemetry").
        payload: Arbitrary key-value data for the receiving agent.
        source: Agent ID or "user" indicating message origin.
        priority: Urgency level for routing decisions.
        correlation_id: Links related request/response pairs.
        timestamp: Unix epoch time the message was created.
    """
    intent: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    priority: MessagePriority = MessagePriority.ROUTINE
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Typed response returned by an agent after processing a message.

    Attributes:
        agent_id: Which agent produced this response.
        status: Outcome — "success", "error", "not_handled", "delegated".
        data: Structured response payload.
        messages: Human-readable text lines for display.
        broadcast: Optional message to broadcast to all other agents.
        priority: Inherited or escalated priority.
    """
    agent_id: str
    status: str = "success"
    data: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    broadcast: Optional[AgentMessage] = None
    priority: MessagePriority = MessagePriority.ROUTINE


class BaseAgent(ABC):
    """Abstract base class for all DeepSpace-SLM domain agents.

    Subclasses must implement:
        - agent_id: Unique identifier string.
        - capabilities: Set of AgentCapability declarations.
        - handle_message(): Process an incoming AgentMessage and return AgentResponse.

    Optionally override:
        - on_broadcast(): React to broadcast messages from other agents.
        - startup() / shutdown(): Lifecycle hooks.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent (e.g. 'inventory', 'radiation')."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent can handle."""
        ...

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a typed response.

        Args:
            message: The incoming AgentMessage to process.

        Returns:
            AgentResponse with status, data, and optional broadcast.
        """
        ...

    def on_broadcast(self, message: AgentMessage) -> None:
        """React to a broadcast message from another agent.

        Default implementation logs and ignores. Override in subclasses
        that need cross-agent event handling (e.g. radiation alerts).

        Args:
            message: The broadcast AgentMessage.
        """
        logger.debug(
            "[%s] Received broadcast intent=%s from=%s",
            self.agent_id, message.intent, message.source,
        )

    def can_handle(self, intent: str) -> bool:
        """Check if this agent declares a capability matching the given intent.

        Args:
            intent: The intent string to match.

        Returns:
            True if this agent can handle the intent.
        """
        return any(cap.intent == intent for cap in self.capabilities)

    def get_handled_intents(self) -> Set[str]:
        """Return the set of intent strings this agent handles."""
        return {cap.intent for cap in self.capabilities}

    def startup(self) -> None:
        """Called when the agent is registered with the coordinator. Override for setup."""
        logger.info("[%s] Agent started", self.agent_id)

    def shutdown(self) -> None:
        """Called when the agent is unregistered. Override for cleanup."""
        logger.info("[%s] Agent shutdown", self.agent_id)

    def __repr__(self) -> str:
        intents = ", ".join(self.get_handled_intents())
        return f"{self.__class__.__name__}(id={self.agent_id!r}, intents=[{intents}])"
