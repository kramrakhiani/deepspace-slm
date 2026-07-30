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

    def set_prose_mode(self, enabled: bool):
        self.prose_mode = enabled

    def process_environmental_telemetry(self, radiation_ugy_h: float) -> RAPGuardStatus:
        level = self.rapg_engine.evaluate_radiation_telemetry(radiation_ugy_h)
        self.rapg_engine.adapt_model_defenses(self.engine.model)
        return self.rapg_engine.get_status()

    def check_stock(self, item_name: str) -> InventoryResponse:
        record = self.db.query_item(item_name)
        if record:
            if self.prose_mode:
                raw_text = (
                    f"We currently have {record.quantity} {record.unit} of {record.name} stored in {record.location}. "
                    f"System status is {record.status}, with approximately {int(record.days_remaining)} days of supply remaining."
                )
            else:
                raw_text = f"<RESPONSE> <ITEM> {record.name} <QTY> {record.quantity} {record.unit} <STATUS> {record.status} <LOC> {record.location}"

            return InventoryResponse(
                item=record.name,
                quantity=record.quantity,
                unit=record.unit,
                location=record.location,
                status=record.status,
                days_remaining=int(record.days_remaining),
                consumption_rate=record.consumption_rate,
                raw_text=raw_text,
                confidence=1.0,
            )

        prompt = f"<QUERY> check stock {item_name}"
        raw = self.engine.generate(prompt, max_new_tokens=64, greedy=True)
        return self._parse_response(raw)

    def locate(self, item_name: str) -> InventoryResponse:
        record = self.db.query_item(item_name)
        if record:
            if self.prose_mode:
                raw_text = f"The {record.name} is located in {record.location}. There are {record.quantity} {record.unit} available for immediate crew deployment."
            else:
                raw_text = f"<RESPONSE> <ITEM> {record.name} <LOC> {record.location} <QTY> {record.quantity} {record.unit} available"

            return InventoryResponse(
                item=record.name,
                quantity=record.quantity,
                unit=record.unit,
                location=record.location,
                status=record.status,
                raw_text=raw_text,
                confidence=1.0,
            )

        prompt = f"<QUERY> locate {item_name}"
        raw = self.engine.generate(prompt, max_new_tokens=64, greedy=True)
        return self._parse_response(raw)

    def forecast_usage(self, item_name: str, days: int = 30) -> InventoryResponse:
        record = self.db.query_item(item_name)
        if record:
            if self.prose_mode:
                raw_text = (
                    f"At a daily consumption rate of {record.consumption_rate} {record.unit}/day, "
                    f"the current stock of {record.name} ({record.quantity} {record.unit}) is estimated to last {int(record.days_remaining)} days."
                )
            else:
                raw_text = f"<RESPONSE> <ITEM> {record.name} at rate of {record.consumption_rate} {record.unit}/day, estimated {int(record.days_remaining)} days remaining"

            return InventoryResponse(
                item=record.name,
                quantity=record.quantity,
                unit=record.unit,
                location=record.location,
                status=record.status,
                days_remaining=int(record.days_remaining),
                consumption_rate=record.consumption_rate,
                raw_text=raw_text,
                confidence=1.0,
            )

        prompt = f"<QUERY> forecast {item_name} consumption"
        raw = self.engine.generate(prompt, max_new_tokens=96, greedy=True)
        return self._parse_response(raw)

    def get_alerts(self) -> InventoryResponse:
        critical_records = self.db.get_alerts(threshold_ratio=self.low_threshold)
        if critical_records:
            alerts = [f"{r.name}: {r.quantity} {r.unit} ({r.status})" for r in critical_records]
            if self.prose_mode:
                raw_text = "Attention Crew: The following items are critically low or reaching supply thresholds:\n" + "\n".join([f" • {a}" for a in alerts])
            else:
                raw_text = "<RESPONSE> " + " <SEP> ".join([f"<ALERT> {a}" for a in alerts])

            return InventoryResponse(
                alerts=alerts,
                raw_text=raw_text,
                confidence=1.0,
            )

        if self.prose_mode:
            raw_text = "All habitat supply levels are nominal. No active inventory alerts."
        else:
            raw_text = "<RESPONSE> all inventory levels are nominal. no alerts"

        return InventoryResponse(alerts=[], raw_text=raw_text, confidence=1.0)

    def log_maintenance(self, action: str, item_name: str, location: str) -> InventoryResponse:
        res = self.db.log_maintenance(action, item_name, location, mission_day=self.mission_day)
        if self.prose_mode:
            raw_text = f"Maintenance action logged successfully: {item_name} was {action} at {location} on Mission Day {self.mission_day}."
        else:
            raw_text = f"<RESPONSE> logged: {item_name} {action} at {location} on mission day {self.mission_day}"

        return InventoryResponse(
            item=item_name,
            location=location,
            status="logged",
            raw_text=raw_text,
            confidence=1.0,
        )

    def query_procedure_manual(self, query: str) -> List[Dict[str, str]]:
        return self.manual_retriever.search(query)

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

    def free_query(self, query: str) -> str:
        prompt = f"<QUERY> {query}"
        return self.engine.generate(prompt, max_new_tokens=128, greedy=True)

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

    def status_report(self) -> str:
        lines = [
            f"╔══════════════════════════════════════════╗",
            f"║  HABITAT INVENTORY STATUS REPORT         ║",
            f"║  Mission Day: {self.mission_day:<26d}║",
            f"║  Crew Size:   {self.crew_size:<26d}║",
            f"║  Mode:        {('Conversational Prose' if self.prose_mode else 'Machine Structured'):<26s}║",
            f"╠══════════════════════════════════════════╣",
        ]
        rapg_status = self.rapg_engine.get_status()
        lines.append(f"║  🛡️ RAP-G Threat Level: {rapg_status.threat_level.name:<16s}║")
        lines.append(f"║     Radiation: {rapg_status.radiation_value_ugy_h:<5.1f} uGy/h             ║")
        lines.append(f"╠══════════════════════════════════════════╣")

        alert_response = self.get_alerts()
        if alert_response.alerts:
            lines.append(f"║  ⚠️  ALERTS ({len(alert_response.alerts)}):")
            for alert in alert_response.alerts:
                lines.append(f"║    • {alert[:38]:<38s}║")
        else:
            lines.append(f"║  ✅ All systems nominal                 ║")
        lines.append(f"╚══════════════════════════════════════════╝")
        return "\n".join(lines)
