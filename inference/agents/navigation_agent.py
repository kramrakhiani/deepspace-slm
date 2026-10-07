"""
Navigation Agent
=================
Manages Delay-Tolerant Networking (DTN) mesh communications,
CRDT state synchronization, and inter-node packet routing.
"""

import logging
from typing import List, Optional, Dict, Any

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)
from data.dtn_mesh import InventoryCRDTMesh, DTNBundleQueue

logger = logging.getLogger(__name__)


class NavigationAgent(BaseAgent):
    """Domain agent for DTN mesh networking and inter-habitat communication.

    Owns:
        - CRDT state updates (mesh_update)
        - Remote state merging (mesh_merge)
        - DTN bundle packet creation and transmission (mesh_transmit)
        - Mesh network status queries (mesh_status)
    """

    def __init__(self, node_id: str = "habitat-alpha"):
        self.node_id = node_id
        self.crdt_mesh = InventoryCRDTMesh(node_id)
        self.bundle_queue = DTNBundleQueue(node_id)

    # ── BaseAgent Protocol ────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return "navigation"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                intent="mesh_update",
                description="Update an item's state in the CRDT mesh",
            ),
            AgentCapability(
                intent="mesh_merge",
                description="Merge remote node state into the local CRDT",
            ),
            AgentCapability(
                intent="mesh_transmit",
                description="Create and transmit a DTN bundle packet",
            ),
            AgentCapability(
                intent="mesh_status",
                description="Query current mesh node state and queue status",
            ),
        ]

    def handle_message(self, message: AgentMessage) -> AgentResponse:
        intent = message.intent
        payload = message.payload

        if intent == "mesh_update":
            return self._mesh_update(
                item_name=payload.get("item_name", ""),
                quantity=payload.get("quantity", 0),
                status=payload.get("status", "nominal"),
            )

        elif intent == "mesh_merge":
            return self._mesh_merge(payload.get("remote_state", {}))

        elif intent == "mesh_transmit":
            return self._mesh_transmit(payload)

        elif intent == "mesh_status":
            return self._mesh_status()

        return AgentResponse(
            agent_id=self.agent_id,
            status="not_handled",
            messages=[f"Navigation agent does not handle intent '{intent}'"],
        )

    def on_broadcast(self, message: AgentMessage) -> None:
        """React to radiation events by queuing a DTN alert bundle."""
        if message.intent == "radiation_threat_change":
            threat_level = message.payload.get("threat_level", "NOMINAL")
            if threat_level in ("STORMY", "CRITICAL"):
                alert_payload = {
                    "type": "radiation_alert",
                    "source_node": self.node_id,
                    "threat_level": threat_level,
                    "radiation_ugy_h": message.payload.get("radiation_ugy_h", 0),
                }
                self.bundle_queue.create_space_packet(alert_payload)
                logger.warning(
                    "[navigation] Queued radiation alert DTN bundle for '%s'",
                    threat_level,
                )

    # ── Domain Logic ──────────────────────────────────────────────────

    def _mesh_update(
        self, item_name: str, quantity: int, status: str
    ) -> AgentResponse:
        entry = self.crdt_mesh.update_item(item_name, quantity, status)

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data=entry,
            messages=[
                f"CRDT mesh updated: {item_name} = {quantity} ({status}) "
                f"[v{entry['version']}]"
            ],
        )

    def _mesh_merge(self, remote_state: Dict[str, Dict[str, Any]]) -> AgentResponse:
        merged_items = self.crdt_mesh.merge_remote_state(remote_state)

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "merged_items": merged_items,
                "count": len(merged_items),
                "total_state_size": len(self.crdt_mesh.state),
            },
            messages=[f"Merged {len(merged_items)} items from remote node"],
        )

    def _mesh_transmit(self, payload: Dict[str, Any]) -> AgentResponse:
        packet = self.bundle_queue.create_space_packet(payload)

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "packet_size_bytes": len(packet),
                "queue_depth": len(self.bundle_queue.queue),
            },
            messages=[f"DTN bundle queued ({len(packet)} bytes), queue depth: {len(self.bundle_queue.queue)}"],
        )

    def _mesh_status(self) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "node_id": self.node_id,
                "crdt_items": len(self.crdt_mesh.state),
                "vector_clock": dict(self.crdt_mesh.vector_clock),
                "queue_depth": len(self.bundle_queue.queue),
            },
            messages=[
                f"Node: {self.node_id}",
                f"CRDT items: {len(self.crdt_mesh.state)}",
                f"DTN queue depth: {len(self.bundle_queue.queue)}",
            ],
        )
