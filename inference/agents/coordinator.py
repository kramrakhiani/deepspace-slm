"""
Agent Coordinator
==================
Central orchestrator that routes messages to domain agents,
handles broadcast events, and supports multi-agent fan-out.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Routes AgentMessages to the appropriate domain agent(s).

    Features:
        - Intent-based routing: maps intent strings to registered agents.
        - Priority escalation: critical messages bypass normal ordering.
        - Broadcast propagation: relays broadcast messages to all agents.
        - Fan-out queries: can dispatch to multiple agents and merge responses.
        - Agent lifecycle: manages startup/shutdown of registered agents.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._intent_map: Dict[str, List[str]] = defaultdict(list)
        self._event_log: List[Dict[str, Any]] = []

    # ── Agent Registration ────────────────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """Register an agent and index its capabilities.

        Args:
            agent: The agent instance to register.

        Raises:
            ValueError: If an agent with the same ID is already registered.
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent '{agent.agent_id}' is already registered")

        self._agents[agent.agent_id] = agent

        for cap in agent.capabilities:
            self._intent_map[cap.intent].append(agent.agent_id)

        agent.startup()
        logger.info(
            "Registered agent '%s' with %d capabilities",
            agent.agent_id, len(agent.capabilities),
        )

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent and remove its capability index entries.

        Args:
            agent_id: ID of the agent to remove.
        """
        agent = self._agents.pop(agent_id, None)
        if agent is None:
            return

        for intent, agent_ids in self._intent_map.items():
            if agent_id in agent_ids:
                agent_ids.remove(agent_id)

        agent.shutdown()
        logger.info("Unregistered agent '%s'", agent_id)

    # ── Message Routing ───────────────────────────────────────────────

    def dispatch(self, message: AgentMessage) -> AgentResponse:
        """Route a message to the first capable agent and return its response.

        If the responding agent includes a broadcast message in its response,
        that broadcast is propagated to all other agents.

        Args:
            message: The incoming AgentMessage.

        Returns:
            AgentResponse from the handling agent, or an error response if
            no agent can handle the intent.
        """
        self._log_event("dispatch", message)

        # Find agents that handle this intent
        target_ids = self._intent_map.get(message.intent, [])

        if not target_ids:
            # Fallback: try each agent's can_handle method
            target_ids = [
                aid for aid, agent in self._agents.items()
                if agent.can_handle(message.intent)
            ]

        if not target_ids:
            return AgentResponse(
                agent_id="coordinator",
                status="not_handled",
                messages=[f"No agent registered for intent '{message.intent}'"],
            )

        # Route to the first capable agent
        target_agent = self._agents[target_ids[0]]
        response = target_agent.handle_message(message)

        # Propagate any broadcast message to all other agents
        if response.broadcast is not None:
            self._propagate_broadcast(response.broadcast, exclude=target_agent.agent_id)

        self._log_event("response", message, response)
        return response

    def dispatch_fanout(self, message: AgentMessage) -> List[AgentResponse]:
        """Route a message to ALL agents that handle the intent.

        Useful for status reports that aggregate data from multiple domains.

        Args:
            message: The incoming AgentMessage.

        Returns:
            List of AgentResponse from all handling agents.
        """
        self._log_event("fanout", message)

        target_ids = self._intent_map.get(message.intent, [])
        if not target_ids:
            target_ids = [
                aid for aid, agent in self._agents.items()
                if agent.can_handle(message.intent)
            ]

        responses = []
        for agent_id in target_ids:
            agent = self._agents[agent_id]
            response = agent.handle_message(message)
            responses.append(response)

            if response.broadcast is not None:
                self._propagate_broadcast(response.broadcast, exclude=agent_id)

        return responses

    def broadcast(self, message: AgentMessage) -> None:
        """Send a message to ALL registered agents (event notification).

        Args:
            message: The broadcast AgentMessage.
        """
        self._log_event("broadcast", message)
        self._propagate_broadcast(message)

    # ── Query Helpers ─────────────────────────────────────────────────

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a registered agent by ID.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent instance, or None if not found.
        """
        return self._agents.get(agent_id)

    @property
    def registered_agents(self) -> Dict[str, BaseAgent]:
        """Return a copy of the registered agents dictionary."""
        return dict(self._agents)

    @property
    def intent_registry(self) -> Dict[str, List[str]]:
        """Return a copy of the intent-to-agent routing table."""
        return dict(self._intent_map)

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        """Return the event log for diagnostics."""
        return list(self._event_log)

    # ── Internal ──────────────────────────────────────────────────────

    def _propagate_broadcast(self, message: AgentMessage, exclude: Optional[str] = None) -> None:
        """Deliver a broadcast message to all agents except the excluded one."""
        for agent_id, agent in self._agents.items():
            if agent_id != exclude:
                try:
                    agent.on_broadcast(message)
                except Exception as e:
                    logger.error(
                        "Broadcast to '%s' failed: %s", agent_id, e,
                    )

    def _log_event(
        self,
        event_type: str,
        message: AgentMessage,
        response: Optional[AgentResponse] = None,
    ) -> None:
        """Record a routing event for diagnostics."""
        entry = {
            "type": event_type,
            "intent": message.intent,
            "source": message.source,
            "priority": message.priority.name,
            "timestamp": time.time(),
        }
        if response is not None:
            entry["response_agent"] = response.agent_id
            entry["response_status"] = response.status
        self._event_log.append(entry)

    def __repr__(self) -> str:
        agent_names = list(self._agents.keys())
        intent_count = sum(len(v) for v in self._intent_map.values())
        return (
            f"AgentCoordinator(agents={agent_names}, "
            f"intent_routes={intent_count}, "
            f"events_logged={len(self._event_log)})"
        )
