"""
Maintenance Agent
==================
Handles maintenance logging, Merkle DAG audit trail verification,
and habitat procedure manual lookups.
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
from data.inventory_db import HabitatDatabase
from data.manuals import HabitatManualRetriever

logger = logging.getLogger(__name__)


class MaintenanceAgent(BaseAgent):
    """Domain agent for maintenance operations and audit trail management.

    Owns:
        - Maintenance action logging (log_maintenance)
        - Merkle DAG audit verification (verify_audit)
        - Procedure manual search (query_manual)
    """

    def __init__(
        self,
        db: Optional[HabitatDatabase] = None,
        manual_retriever: Optional[HabitatManualRetriever] = None,
        mission_day: int = 1,
        prose_mode: bool = False,
    ):
        self.db = db or HabitatDatabase(":memory:")
        self.manual_retriever = manual_retriever or HabitatManualRetriever()
        self.mission_day = mission_day
        self.prose_mode = prose_mode

    # ── BaseAgent Protocol ────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return "maintenance"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                intent="log_maintenance",
                description="Log a maintenance action with Merkle DAG audit trail",
            ),
            AgentCapability(
                intent="verify_audit",
                description="Verify the cryptographic integrity of the Merkle audit chain",
            ),
            AgentCapability(
                intent="query_manual",
                description="Search habitat procedure manuals for a topic or keyword",
            ),
        ]

    def handle_message(self, message: AgentMessage) -> AgentResponse:
        intent = message.intent
        payload = message.payload

        if intent == "log_maintenance":
            return self._log_maintenance(
                action=payload.get("action", ""),
                item_name=payload.get("item_name", ""),
                location=payload.get("location", ""),
            )

        elif intent == "verify_audit":
            return self._verify_audit()

        elif intent == "query_manual":
            return self._query_manual(payload.get("query", ""))

        return AgentResponse(
            agent_id=self.agent_id,
            status="not_handled",
            messages=[f"Maintenance agent does not handle intent '{intent}'"],
        )

    def on_broadcast(self, message: AgentMessage) -> None:
        """React to radiation events by logging them as maintenance events."""
        if message.intent == "radiation_threat_change":
            threat_level = message.payload.get("threat_level", "NOMINAL")
            if threat_level in ("STORMY", "CRITICAL"):
                logger.warning(
                    "[maintenance] Radiation event '%s' logged as maintenance record",
                    threat_level,
                )
                self.db.log_maintenance(
                    action=f"radiation_event_{threat_level.lower()}",
                    item_name="habitat_systems",
                    location="all_modules",
                    performed_by="rapg_engine",
                    mission_day=self.mission_day,
                )

    # ── Domain Logic ──────────────────────────────────────────────────

    def _log_maintenance(
        self, action: str, item_name: str, location: str
    ) -> AgentResponse:
        result = self.db.log_maintenance(
            action=action,
            item_name=item_name,
            location=location,
            mission_day=self.mission_day,
        )

        if self.prose_mode:
            text = (
                f"Maintenance action logged successfully: {item_name} was {action} "
                f"at {location} on Mission Day {self.mission_day}."
            )
        else:
            text = (
                f"<RESPONSE> logged: {item_name} {action} at {location} "
                f"on mission day {self.mission_day}"
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "item": item_name,
                "action": action,
                "location": location,
                "mission_day": self.mission_day,
                "merkle_hash": result.get("merkle_hash", ""),
            },
            messages=[text],
        )

    def _verify_audit(self) -> AgentResponse:
        is_valid = self.db.verify_merkle_chain()
        status_text = "VALID" if is_valid else "CORRUPTED"

        if self.prose_mode:
            text = (
                f"Merkle DAG audit chain verification result: {status_text}. "
                f"{'All transaction hashes verified successfully.' if is_valid else 'WARNING: Audit chain integrity compromised!'}"
            )
        else:
            text = f"<RESPONSE> Merkle audit chain: {status_text}"

        priority = MessagePriority.ROUTINE if is_valid else MessagePriority.CRITICAL

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={"audit_valid": is_valid},
            messages=[text],
            priority=priority,
        )

    def _query_manual(self, query: str) -> AgentResponse:
        results = self.manual_retriever.search(query)

        if not results:
            return AgentResponse(
                agent_id=self.agent_id,
                status="not_found",
                messages=["No matching procedure manual entry found."],
            )

        messages = [f"Found {len(results)} matching procedure(s):"]
        for r in results:
            messages.append(f"  • {r['title']} ({r['topic']}): {r['content']}")

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "results": results,
                "count": len(results),
            },
            messages=messages,
        )
