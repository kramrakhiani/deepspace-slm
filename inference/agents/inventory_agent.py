"""
Inventory Agent
================
Handles all habitat supply chain operations: stock checks,
item location, consumption forecasting, and alert monitoring.
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
from data.inventory_db import HabitatDatabase, ItemRecord

logger = logging.getLogger(__name__)


class InventoryAgent(BaseAgent):
    """Domain agent for habitat inventory and supply chain management.

    Owns:
        - Stock queries (check_stock)
        - Item location lookups (locate)
        - Consumption rate forecasting (forecast)
        - Low/critical stock alert generation (get_alerts)
        - Secure quantity updates (update_quantity)
    """

    def __init__(
        self,
        db: Optional[HabitatDatabase] = None,
        mission_day: int = 1,
        crew_size: int = 4,
        critical_threshold: float = 0.10,
        low_threshold: float = 0.25,
        prose_mode: bool = False,
    ):
        self.db = db or HabitatDatabase(":memory:")
        if db is None:
            self.db.seed_initial_data()
        self.mission_day = mission_day
        self.crew_size = crew_size
        self.critical_threshold = critical_threshold
        self.low_threshold = low_threshold
        self.prose_mode = prose_mode

    # ── BaseAgent Protocol ────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return "inventory"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                intent="check_stock",
                description="Query current stock levels for an inventory item",
            ),
            AgentCapability(
                intent="locate",
                description="Find the storage location of an inventory item",
            ),
            AgentCapability(
                intent="forecast",
                description="Forecast supply duration based on consumption rates",
            ),
            AgentCapability(
                intent="get_alerts",
                description="Retrieve items below low or critical stock thresholds",
            ),
            AgentCapability(
                intent="update_quantity",
                description="Adjust item quantity (restock or consume)",
                priority_floor=MessagePriority.ELEVATED,
            ),
        ]

    def handle_message(self, message: AgentMessage) -> AgentResponse:
        intent = message.intent
        payload = message.payload

        if intent == "check_stock":
            return self._check_stock(payload.get("item_name", ""))

        elif intent == "locate":
            return self._locate(payload.get("item_name", ""))

        elif intent == "forecast":
            return self._forecast(
                payload.get("item_name", ""),
                payload.get("days", 30),
            )

        elif intent == "get_alerts":
            return self._get_alerts()

        elif intent == "update_quantity":
            return self._update_quantity(
                payload.get("item_name", ""),
                payload.get("delta", 0),
                payload.get("performed_by", "system"),
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="not_handled",
            messages=[f"Inventory agent does not handle intent '{intent}'"],
        )

    def on_broadcast(self, message: AgentMessage) -> None:
        """React to radiation threat broadcasts by checking critical supplies."""
        if message.intent == "radiation_threat_change":
            threat_level = message.payload.get("threat_level", "NOMINAL")
            if threat_level in ("STORMY", "CRITICAL"):
                logger.warning(
                    "[inventory] Radiation event '%s' — running critical supply audit",
                    threat_level,
                )

    # ── Domain Logic ──────────────────────────────────────────────────

    def _check_stock(self, item_name: str) -> AgentResponse:
        record = self.db.query_item(item_name)
        if record is None:
            return AgentResponse(
                agent_id=self.agent_id,
                status="not_found",
                messages=[f"Item '{item_name}' not found in inventory"],
            )

        if self.prose_mode:
            text = (
                f"We currently have {record.quantity} {record.unit} of {record.name} "
                f"stored in {record.location}. System status is {record.status}, "
                f"with approximately {int(record.days_remaining)} days of supply remaining."
            )
        else:
            text = (
                f"<RESPONSE> <ITEM> {record.name} <QTY> {record.quantity} "
                f"{record.unit} <STATUS> {record.status} <LOC> {record.location}"
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data=self._record_to_dict(record),
            messages=[text],
        )

    def _locate(self, item_name: str) -> AgentResponse:
        record = self.db.query_item(item_name)
        if record is None:
            return AgentResponse(
                agent_id=self.agent_id,
                status="not_found",
                messages=[f"Item '{item_name}' not found in inventory"],
            )

        if self.prose_mode:
            text = (
                f"The {record.name} is located in {record.location}. "
                f"There are {record.quantity} {record.unit} available for immediate crew deployment."
            )
        else:
            text = (
                f"<RESPONSE> <ITEM> {record.name} <LOC> {record.location} "
                f"<QTY> {record.quantity} {record.unit} available"
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data=self._record_to_dict(record),
            messages=[text],
        )

    def _forecast(self, item_name: str, days: int = 30) -> AgentResponse:
        record = self.db.query_item(item_name)
        if record is None:
            return AgentResponse(
                agent_id=self.agent_id,
                status="not_found",
                messages=[f"Item '{item_name}' not found in inventory"],
            )

        if self.prose_mode:
            text = (
                f"At a daily consumption rate of {record.consumption_rate} {record.unit}/day, "
                f"the current stock of {record.name} ({record.quantity} {record.unit}) "
                f"is estimated to last {int(record.days_remaining)} days."
            )
        else:
            text = (
                f"<RESPONSE> <ITEM> {record.name} at rate of {record.consumption_rate} "
                f"{record.unit}/day, estimated {int(record.days_remaining)} days remaining"
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data=self._record_to_dict(record),
            messages=[text],
        )

    def _get_alerts(self) -> AgentResponse:
        critical_records = self.db.get_alerts(threshold_ratio=self.low_threshold)
        if critical_records:
            alerts = [f"{r.name}: {r.quantity} {r.unit} ({r.status})" for r in critical_records]
            if self.prose_mode:
                text = (
                    "Attention Crew: The following items are critically low "
                    "or reaching supply thresholds:\n"
                    + "\n".join([f" • {a}" for a in alerts])
                )
            else:
                text = "<RESPONSE> " + " <SEP> ".join([f"<ALERT> {a}" for a in alerts])

            return AgentResponse(
                agent_id=self.agent_id,
                status="success",
                data={"alerts": alerts, "count": len(alerts)},
                messages=[text],
            )

        if self.prose_mode:
            text = "All habitat supply levels are nominal. No active inventory alerts."
        else:
            text = "<RESPONSE> all inventory levels are nominal. no alerts"

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={"alerts": [], "count": 0},
            messages=[text],
        )

    def _update_quantity(
        self, item_name: str, delta: int, performed_by: str = "system"
    ) -> AgentResponse:
        record = self.db.update_quantity(
            item_name, delta, performed_by=performed_by, mission_day=self.mission_day,
        )
        if record is None:
            return AgentResponse(
                agent_id=self.agent_id,
                status="not_found",
                messages=[f"Item '{item_name}' not found for quantity update"],
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data=self._record_to_dict(record),
            messages=[f"Updated {item_name}: quantity now {record.quantity} {record.unit}"],
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(record: ItemRecord) -> Dict[str, Any]:
        return {
            "item": record.name,
            "quantity": record.quantity,
            "unit": record.unit,
            "location": record.location,
            "status": record.status,
            "days_remaining": int(record.days_remaining),
            "consumption_rate": record.consumption_rate,
            "category": record.category,
            "criticality": record.criticality,
        }
