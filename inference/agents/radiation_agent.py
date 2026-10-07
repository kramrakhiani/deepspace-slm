"""
Radiation Agent
================
Manages RAP-G radiation threat assessment, model defense adaptation,
and broadcasts radiation events to all other agents.
"""

import logging
from typing import List, Optional

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)
from quantization.rapg_engine import RAPGuardEngine, RadiationThreatLevel, RAPGuardStatus

logger = logging.getLogger(__name__)


class RadiationAgent(BaseAgent):
    """Domain agent for radiation monitoring and RAP-G defense coordination.

    Owns:
        - Radiation telemetry evaluation (radiation_telemetry)
        - Model defense adaptation (adapt_defenses)
        - RAP-G status queries (radiation_status)
        - Broadcasts threat-level changes to all agents
    """

    def __init__(self, rapg_engine: Optional[RAPGuardEngine] = None):
        self.rapg_engine = rapg_engine or RAPGuardEngine()
        self._model_ref = None  # Set when adapt_defenses is called

    # ── BaseAgent Protocol ────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return "radiation"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                intent="radiation_telemetry",
                description="Process radiation sensor telemetry and evaluate threat level",
            ),
            AgentCapability(
                intent="adapt_defenses",
                description="Adapt model quantization and TMR defenses to current radiation",
            ),
            AgentCapability(
                intent="radiation_status",
                description="Query current RAP-G radiation defense status",
            ),
        ]

    def handle_message(self, message: AgentMessage) -> AgentResponse:
        intent = message.intent
        payload = message.payload

        if intent == "radiation_telemetry":
            return self._evaluate_telemetry(
                payload.get("radiation_ugy_h", 0.0),
                model=payload.get("model"),
            )

        elif intent == "adapt_defenses":
            return self._adapt_defenses(payload.get("model"))

        elif intent == "radiation_status":
            return self._get_status()

        return AgentResponse(
            agent_id=self.agent_id,
            status="not_handled",
            messages=[f"Radiation agent does not handle intent '{intent}'"],
        )

    # ── Domain Logic ──────────────────────────────────────────────────

    def _evaluate_telemetry(self, radiation_ugy_h: float, model=None) -> AgentResponse:
        previous_level = self.rapg_engine.current_level
        new_level = self.rapg_engine.evaluate_radiation_telemetry(radiation_ugy_h)

        # Adapt model defenses if model reference is available
        if model is not None:
            self._model_ref = model
            self.rapg_engine.adapt_model_defenses(model)

        status = self.rapg_engine.get_status()

        # Determine priority based on threat level
        if new_level in (RadiationThreatLevel.STORMY, RadiationThreatLevel.CRITICAL):
            priority = MessagePriority.CRITICAL
        elif new_level == RadiationThreatLevel.ELEVATED:
            priority = MessagePriority.ELEVATED
        else:
            priority = MessagePriority.ROUTINE

        # Build broadcast if threat level changed
        broadcast = None
        if new_level != previous_level:
            broadcast = AgentMessage(
                intent="radiation_threat_change",
                payload={
                    "threat_level": new_level.name,
                    "previous_level": previous_level.name,
                    "radiation_ugy_h": radiation_ugy_h,
                    "active_defenses": status.active_defenses,
                },
                source=self.agent_id,
                priority=priority,
            )
            logger.info(
                "[radiation] Threat level changed: %s → %s (%.1f uGy/h)",
                previous_level.name, new_level.name, radiation_ugy_h,
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "threat_level": new_level.name,
                "radiation_ugy_h": radiation_ugy_h,
                "active_defenses": status.active_defenses,
                "tmr_enabled": status.tmr_enabled,
                "scrubber_interval_sec": status.scrubber_interval_sec,
            },
            messages=[
                f"Radiation: {radiation_ugy_h:.1f} uGy/h",
                f"Threat Level: {new_level.name}",
                f"Active Defenses: {', '.join(status.active_defenses)}",
            ],
            broadcast=broadcast,
            priority=priority,
        )

    def _adapt_defenses(self, model=None) -> AgentResponse:
        if model is not None:
            self._model_ref = model

        if self._model_ref is None:
            return AgentResponse(
                agent_id=self.agent_id,
                status="error",
                messages=["No model reference available for defense adaptation"],
            )

        self.rapg_engine.adapt_model_defenses(self._model_ref)
        status = self.rapg_engine.get_status()

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "threat_level": status.threat_level.name,
                "active_defenses": status.active_defenses,
                "tmr_enabled": status.tmr_enabled,
            },
            messages=[f"Model defenses adapted for {status.threat_level.name}"],
        )

    def _get_status(self) -> AgentResponse:
        status = self.rapg_engine.get_status()

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={
                "threat_level": status.threat_level.name,
                "radiation_ugy_h": status.radiation_value_ugy_h,
                "active_defenses": status.active_defenses,
                "tmr_enabled": status.tmr_enabled,
                "scrubber_interval_sec": status.scrubber_interval_sec,
            },
            messages=[
                f"RAP-G Threat Level: {status.threat_level.name}",
                f"Radiation: {status.radiation_value_ugy_h:.1f} uGy/h",
                f"Defenses: {', '.join(status.active_defenses)}",
            ],
        )
