"""
Habitat Agent (Facade)
=======================
Backward-compatible facade that delegates all operations to the
multi-agent coordinator. Preserves the original HabitatAgent public API
so that existing tests and consumers continue to work unchanged.
"""

import re
import hmac
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from inference.engine import InferenceEngine
from inference.grammar import SchemaMasker
from data.inventory_db import HabitatDatabase, ItemRecord
from data.manuals import HabitatManualRetriever
from quantization.rapg_engine import RAPGuardEngine, RadiationThreatLevel, RAPGuardStatus

from inference.agents.base import AgentMessage, AgentResponse, MessagePriority
from inference.agents.coordinator import AgentCoordinator
from inference.agents.inventory_agent import InventoryAgent
from inference.agents.radiation_agent import RadiationAgent
from inference.agents.maintenance_agent import MaintenanceAgent
from inference.agents.navigation_agent import NavigationAgent
from inference.agents.inference_agent import InferenceAgent


@dataclass
class InventoryResponse:
    item: Optional[str] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    days_remaining: Optional[int] = None
    consumption_rate: Optional[float] = None
    raw_text: str = ""
    confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)


class SafetyBoundaryValidator:
    def __init__(self, min_o2_quantity: int = 5, min_water_filter_quantity: int = 2):
        self.min_o2_quantity = min_o2_quantity
        self.min_water_filter_quantity = min_water_filter_quantity

    def validate_action(self, item_name: str, current_qty: int, delta: int) -> bool:
        new_qty = current_qty + delta
        item_lower = item_name.lower()
        if "o2" in item_lower or "oxygen" in item_lower:
            if new_qty < self.min_o2_quantity:
                return False
        if "water filter" in item_lower:
            if new_qty < self.min_water_filter_quantity:
                return False
        return True


class TwoKeySecurityManager:
    def __init__(self, commander_secret: str = "COMMANDER_SECRET_KEY", engineer_secret: str = "ENGINEER_SECRET_KEY"):
        self.commander_secret = commander_secret.encode("utf-8")
        self.engineer_secret = engineer_secret.encode("utf-8")

    def generate_signature(self, role: str, action_payload: str) -> str:
        secret = self.commander_secret if role.lower() == "commander" else self.engineer_secret
        return hmac.new(secret, action_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_two_keys(self, action_payload: str, sig_commander: str, sig_engineer: str) -> bool:
        expected_cmd = self.generate_signature("commander", action_payload)
        expected_eng = self.generate_signature("engineer", action_payload)
        return hmac.compare_digest(expected_cmd, sig_commander) and hmac.compare_digest(expected_eng, sig_engineer)


class HabitatAgent:
    """Backward-compatible facade over the multi-agent coordinator.

    Preserves the full public API of the original monolithic HabitatAgent
    while delegating all domain work to specialized agents via message passing.

    The coordinator manages:
        - InventoryAgent:    stock, locate, forecast, alerts
        - RadiationAgent:    RAP-G telemetry and defense adaptation
        - MaintenanceAgent:  maintenance logging, audit, manuals
        - NavigationAgent:   DTN mesh networking
        - InferenceAgent:    SLM neural text generation
    """

    def __init__(
        self,
        engine: InferenceEngine,
        db: Optional[HabitatDatabase] = None,
        mission_day: int = 1,
        crew_size: int = 4,
        critical_threshold: float = 0.1,
        low_threshold: float = 0.25,
        prose_mode: bool = False,
    ):
        self.engine = engine
        self.db = db or HabitatDatabase(":memory:")
        if db is None:
            self.db.seed_initial_data()

        self.mission_day = mission_day
        self.crew_size = crew_size
        self.critical_threshold = critical_threshold
        self.low_threshold = low_threshold
        self.prose_mode = prose_mode
        self.schema_masker = SchemaMasker(self.engine.tokenizer)
        self.manual_retriever = HabitatManualRetriever()
        self.security_manager = TwoKeySecurityManager()
        self.safety_validator = SafetyBoundaryValidator()
        self.rapg_engine = RAPGuardEngine()

        # ── Multi-Agent Coordinator Setup ─────────────────────────────
        self.coordinator = AgentCoordinator()

        self._inventory_agent = InventoryAgent(
            db=self.db,
            mission_day=mission_day,
            crew_size=crew_size,
            critical_threshold=critical_threshold,
            low_threshold=low_threshold,
            prose_mode=prose_mode,
        )
        self._radiation_agent = RadiationAgent(rapg_engine=self.rapg_engine)
        self._maintenance_agent = MaintenanceAgent(
            db=self.db,
            manual_retriever=self.manual_retriever,
            mission_day=mission_day,
            prose_mode=prose_mode,
        )
        self._navigation_agent = NavigationAgent(node_id="habitat-alpha")
        self._inference_agent = InferenceAgent(engine=engine)

        self.coordinator.register(self._inventory_agent)
        self.coordinator.register(self._radiation_agent)
        self.coordinator.register(self._maintenance_agent)
        self.coordinator.register(self._navigation_agent)
        self.coordinator.register(self._inference_agent)

    # ── Mode Toggling ─────────────────────────────────────────────────

    def set_prose_mode(self, enabled: bool):
        self.prose_mode = enabled
        self._inventory_agent.prose_mode = enabled
        self._maintenance_agent.prose_mode = enabled

    # ── Delegated to RadiationAgent ───────────────────────────────────

    def process_environmental_telemetry(self, radiation_ugy_h: float) -> RAPGuardStatus:
        response = self.coordinator.dispatch(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": radiation_ugy_h, "model": self.engine.model},
            source="habitat_agent",
        ))
        return self.rapg_engine.get_status()

    # ── Delegated to InventoryAgent ───────────────────────────────────

    def check_stock(self, item_name: str) -> InventoryResponse:
        response = self.coordinator.dispatch(AgentMessage(
            intent="check_stock",
            payload={"item_name": item_name},
            source="habitat_agent",
        ))
        return self._agent_response_to_inventory(response, item_name)

    def locate(self, item_name: str) -> InventoryResponse:
        response = self.coordinator.dispatch(AgentMessage(
            intent="locate",
            payload={"item_name": item_name},
            source="habitat_agent",
        ))
        return self._agent_response_to_inventory(response, item_name)

    def forecast_usage(self, item_name: str, days: int = 30) -> InventoryResponse:
        response = self.coordinator.dispatch(AgentMessage(
            intent="forecast",
            payload={"item_name": item_name, "days": days},
            source="habitat_agent",
        ))
        return self._agent_response_to_inventory(response, item_name)

    def get_alerts(self) -> InventoryResponse:
        response = self.coordinator.dispatch(AgentMessage(
            intent="get_alerts",
            payload={},
            source="habitat_agent",
        ))

        raw_text = response.messages[0] if response.messages else ""
        alerts = response.data.get("alerts", [])

        return InventoryResponse(
            alerts=alerts,
            raw_text=raw_text,
            confidence=1.0,
        )

    # ── Delegated to MaintenanceAgent ─────────────────────────────────

    def log_maintenance(self, action: str, item_name: str, location: str) -> InventoryResponse:
        response = self.coordinator.dispatch(AgentMessage(
            intent="log_maintenance",
            payload={"action": action, "item_name": item_name, "location": location},
            source="habitat_agent",
        ))

        raw_text = response.messages[0] if response.messages else ""

        return InventoryResponse(
            item=item_name,
            location=location,
            status="logged",
            raw_text=raw_text,
            confidence=1.0,
        )

    def query_procedure_manual(self, query: str) -> List[Dict[str, str]]:
        response = self.coordinator.dispatch(AgentMessage(
            intent="query_manual",
            payload={"query": query},
            source="habitat_agent",
        ))
        return response.data.get("results", [])

    # ── Security (kept on the facade — cross-cutting concern) ─────────

    def secure_update_quantity(
        self,
        item_name: str,
        delta: int,
        sig_commander: str,
        sig_engineer: str,
    ) -> Optional[ItemRecord]:
        action_payload = f"UPDATE_QTY:{item_name.lower()}:{delta}"
        if not self.security_manager.verify_two_keys(action_payload, sig_commander, sig_engineer):
            raise PermissionError("Security Authorization Failed: Valid Commander and Engineer signatures required.")

        record = self.db.query_item(item_name)
        if record and not self.safety_validator.validate_action(item_name, record.quantity, delta):
            raise ValueError(f"Safety Violation: Action would drop {item_name} below critical life-support safety limits.")

        return self.db.update_quantity(item_name, delta, performed_by="authenticated_crew", mission_day=self.mission_day)

    # ── Delegated to InferenceAgent ───────────────────────────────────

    def free_query(self, query: str) -> str:
        """
        Route a natural-language query to the appropriate domain agent.

        If a domain agent can answer the query, its tool result is returned.
        Otherwise, fall back to the SLM's free-form generation.
        """

        intent, payload = self._route_query(query)

        # ---------------------------------------------------------
        # 1. Domain-agent path
        # ---------------------------------------------------------
        if intent != "free_query":
            response = self.coordinator.dispatch(AgentMessage(
                intent=intent,
                payload=payload,
                source="natural_language_router",
            ))

            if response.status == "success":
                return self._format_agent_response(response)

        # ---------------------------------------------------------
        # 2. Fallback to pure SLM generation
        # ---------------------------------------------------------
        response = self.coordinator.dispatch(AgentMessage(
            intent="free_query",
            payload={
                "query": query,
                "max_new_tokens": 128,
                "greedy": True,
            },
            source="habitat_agent",
        ))

        return response.data.get("generated_text", "")
    
    def _route_query(self, query: str):
        """
        Lightweight natural-language intent router.

        Converts an astronaut's natural-language query into:
            intent + payload

        The coordinator then sends that intent to the appropriate agent.
        """

        q = query.lower().strip()

        # =========================================================
        # INVENTORY DOMAIN
        # =========================================================

        inventory_words = [
            "oxygen", "o2", "tank", "canister",
            "water", "filter", "food", "ration",
            "inventory", "stock", "supply",
            "medical", "medicine", "kit",
            "antibiotic", "wrench", "tools",
            "scrubber", "nitrogen", "n2"
        ]

        if any(word in q for word in inventory_words):

            # Determine the item
            item = self._extract_inventory_item(q)

            # Location query
            if any(word in q for word in [
                "where", "location", "located", "stored"
            ]):
                return "locate", {"item_name": item}

            # Forecast / duration query
            if any(word in q for word in [
                "how long", "how many days",
                "days remaining", "last", "duration"
            ]):
                return "forecast", {
                    "item_name": item,
                    "days": 30
                }

            # Alert query
            if any(word in q for word in [
                "alert", "warning", "low stock",
                "running low", "critical"
            ]):
                return "get_alerts", {}

            # Default inventory operation = stock check
            return "check_stock", {"item_name": item}

        # =========================================================
        # RADIATION DOMAIN
        # =========================================================

        radiation_words = [
            "radiation",
            "radiation level",
            "solar flare",
            "radiation threat",
            "radiation status",
            "radiation exposure"
        ]

        if any(word in q for word in radiation_words):

            return "radiation_status", {}

        # =========================================================
        # MAINTENANCE DOMAIN
        # =========================================================

        maintenance_words = [
            "maintenance",
            "repair",
            "procedure",
            "manual",
            "maintenance manual",
            "how do i repair",
            "how to repair"
        ]

        if any(word in q for word in maintenance_words):

            return "query_manual", {
                "query": query
            }

        # =========================================================
        # AUDIT DOMAIN
        # =========================================================

        if any(word in q for word in [
            "audit",
            "integrity",
            "merkle",
            "transaction history"
        ]):
            return "verify_audit", {}

        # =========================================================
        # NAVIGATION / NETWORK DOMAIN
        # =========================================================

        navigation_words = [
            "navigation",
            "network",
            "mesh",
            "connection",
            "connectivity",
            "antenna",
            "dtn"
        ]

        if any(word in q for word in navigation_words):

            return "mesh_status", {}

        # =========================================================
        # NO KNOWN DOMAIN
        # =========================================================

        return "free_query", {}
    
    def _extract_inventory_item(self, query: str) -> str:
        """
        Convert natural-language references into database item names.
        """

        q = query.lower()

        # Oxygen
        if "oxygen" in q or "o2" in q:
            return "o2 canister"

        # Nitrogen
        if "nitrogen" in q or "n2" in q:
            return "n2 tank"

        # Water
        if "water" in q:
            return "water filter"

        # Food
        if "food" in q or "ration" in q:
            return "food ration"

        # CO2
        if "co2" in q or "carbon dioxide" in q:
            return "co2 scrubber"

        # Medical
        if "medical" in q or "medicine" in q or "med kit" in q:
            return "medical kit"

        # Antibiotics
        if "antibiotic" in q:
            return "antibiotic pack"

        # Tools
        if "wrench" in q or "tools" in q:
            return "wrench set"

        # Default
        return query
    
    def _format_agent_response(self, response: AgentResponse) -> str:
        """
        Convert structured agent output into a readable response.

        The SLM can later be used here for natural-language synthesis.
        """

        data = response.data

        if response.agent_id == "inventory":

            item = data.get("item", "item")
            quantity = data.get("quantity")
            unit = data.get("unit", "units")
            location = data.get("location")
            days = data.get("days_remaining")

            parts = []

            if quantity is not None:
                parts.append(
                    f"There are {quantity} {unit} of {item} remaining."
                )

            if location:
                parts.append(
                    f"They are stored in {location}."
                )

            if days is not None:
                parts.append(
                    f"The estimated remaining supply is {days} days."
                )

            return " ".join(parts)

        # Other agents
        if response.messages:
            return " ".join(response.messages)

        return str(data)

    # ── Response Parsing (legacy compatibility) ───────────────────────

    def _parse_response(self, raw_text: str) -> InventoryResponse:
        response = InventoryResponse(raw_text=raw_text)

        item_match = re.search(r"<ITEM>\s*(\S+(?:\s+\S+)*?)(?:\s*<|\s*$)", raw_text)
        if item_match:
            response.item = item_match.group(1).strip()

        qty_match = re.search(r"<QTY>\s*(\d+)", raw_text)
        if qty_match:
            response.quantity = int(qty_match.group(1))

        loc_match = re.search(r"<LOC>\s*(\S+(?:\s+\S+)*?)(?:\s*<|\s*$)", raw_text)
        if loc_match:
            response.location = loc_match.group(1).strip()

        status_match = re.search(r"<STATUS>\s*(\w+)", raw_text)
        if status_match:
            response.status = status_match.group(1).strip()

        alert_matches = re.findall(r"<ALERT>\s*(.*?)(?=<ALERT>|<SEP>|$)", raw_text)
        response.alerts = [a.strip() for a in alert_matches if a.strip()]

        if response.quantity is None:
            qty_fallback = re.search(r"(\d+)\s+(?:units?|packets?|kits?|bottles?)", raw_text)
            if qty_fallback:
                response.quantity = int(qty_fallback.group(1))

        if response.location is None:
            loc_fallback = re.search(r"(?:in|at|stored in)\s+([\w\s]+(?:bay|locker|shelf|rack|cabinet)[\w\s]*\d*)", raw_text, re.IGNORECASE)
            if loc_fallback:
                response.location = loc_fallback.group(1).strip()

        rate_match = re.search(r"(\d+\.?\d*)\s+\w+\s+per\s+day", raw_text)
        if rate_match:
            response.consumption_rate = float(rate_match.group(1))

        days_match = re.search(r"(\d+)\s+days?\s+remaining", raw_text)
        if days_match:
            response.days_remaining = int(days_match.group(1))

        fields_found = sum([
            response.item is not None,
            response.quantity is not None,
            response.location is not None,
            response.status is not None,
        ])
        response.confidence = fields_found / 4.0
        return response

    # ── Status Report ─────────────────────────────────────────────────

    def status_report(self) -> str:
        lines = [
            f"╔══════════════════════════════════════════╗",
            f"║  HABITAT INVENTORY STATUS REPORT         ║",
            f"║  Mission Day: {self.mission_day:<26d}║",
            f"║  Crew Size:   {self.crew_size:<26d}║",
            f"║  Mode:        {('Conversational Prose' if self.prose_mode else 'Machine Structured'):<26s}║",
            f"╠══════════════════════════════════════════╣",
        ]

        # Radiation status via coordinator
        rad_response = self.coordinator.dispatch(AgentMessage(
            intent="radiation_status", payload={}, source="habitat_agent",
        ))
        rad_data = rad_response.data
        lines.append(f"║  🛡️ RAP-G Threat Level: {rad_data.get('threat_level', 'NOMINAL'):<16s}║")
        lines.append(f"║     Radiation: {rad_data.get('radiation_ugy_h', 0.0):<5.1f} uGy/h             ║")
        lines.append(f"╠══════════════════════════════════════════╣")

        # Agents summary
        agents = self.coordinator.registered_agents
        lines.append(f"║  🤖 Active Agents: {len(agents):<22d}║")
        for aid, agent in agents.items():
            cap_count = len(agent.capabilities)
            lines.append(f"║    • {aid:<15s} ({cap_count} capabilities) ║")
        lines.append(f"╠══════════════════════════════════════════╣")

        # Alerts via coordinator
        alert_response = self.get_alerts()
        if alert_response.alerts:
            lines.append(f"║  ⚠️  ALERTS ({len(alert_response.alerts)}):")
            for alert in alert_response.alerts:
                lines.append(f"║    • {alert[:38]:<38s}║")
        else:
            lines.append(f"║  ✅ All systems nominal                 ║")
        lines.append(f"╚══════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── Internal Helpers ──────────────────────────────────────────────

    def _agent_response_to_inventory(
        self, response: AgentResponse, item_name: str
    ) -> InventoryResponse:
        """Convert an AgentResponse from the inventory agent to InventoryResponse."""
        data = response.data
        raw_text = response.messages[0] if response.messages else ""

        if response.status == "not_found":
            # Fallback to SLM generation for unknown items
            prompt = f"<QUERY> check stock {item_name}"
            raw = self.engine.generate(prompt, max_new_tokens=64, greedy=True)
            return self._parse_response(raw)

        return InventoryResponse(
            item=data.get("item"),
            quantity=data.get("quantity"),
            unit=data.get("unit"),
            location=data.get("location"),
            status=data.get("status"),
            days_remaining=data.get("days_remaining"),
            consumption_rate=data.get("consumption_rate"),
            raw_text=raw_text,
            confidence=1.0,
        )
