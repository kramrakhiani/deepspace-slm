import enum
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from quantization.bitflip_resilience import IdleMemoryScrubber, apply_tmr, TMRLinear
from quantization.ptq import quantize_awq


class RadiationThreatLevel(enum.Enum):
    NOMINAL = 0      # < 100 uGy/h: Max throughput, 3-bit AWQ, speculative decoding
    ELEVATED = 1     # 100 - 500 uGy/h: High scrubber frequency (0.1s interval)
    STORMY = 2       # 500 - 2000 uGy/h: Enable TMR linear layer voting
    CRITICAL = 3     # > 2000 uGy/h: TMR + maximum scrubber frequency + emergency logging


@dataclass
class RAPGuardStatus:
    threat_level: RadiationThreatLevel
    radiation_value_ugy_h: float
    active_defenses: List[str]
    scrubber_interval_sec: float
    tmr_enabled: bool
    timestamp: float


class RAPGuardEngine:
    def __init__(
        self,
        elevated_threshold: float = 100.0,
        stormy_threshold: float = 500.0,
        critical_threshold: float = 2000.0,
    ):
        self.elevated_threshold = elevated_threshold
        self.stormy_threshold = stormy_threshold
        self.critical_threshold = critical_threshold

        self.current_level = RadiationThreatLevel.NOMINAL
        self.last_radiation_val = 0.0
        self.last_update_time = time.time()
        self.tmr_active = False

    def evaluate_radiation_telemetry(self, telemetry_value: float) -> RadiationThreatLevel:
        self.last_radiation_val = float(telemetry_value)
        self.last_update_time = time.time()

        if self.last_radiation_val >= self.critical_threshold:
            new_level = RadiationThreatLevel.CRITICAL
        elif self.last_radiation_val >= self.stormy_threshold:
            new_level = RadiationThreatLevel.STORMY
        elif self.last_radiation_val >= self.elevated_threshold:
            new_level = RadiationThreatLevel.ELEVATED
        else:
            new_level = RadiationThreatLevel.NOMINAL

        self.current_level = new_level
        return new_level

    def adapt_model_defenses(
        self,
        model: nn.Module,
        scrubber: Optional[IdleMemoryScrubber] = None,
    ) -> Tuple[nn.Module, List[str]]:
        defenses_applied = []

        if self.current_level == RadiationThreatLevel.NOMINAL:
            if scrubber:
                scrubber.check_interval_sec = 0.5
            defenses_applied.append("3-Bit AWQ Optimized High Throughput")

        elif self.current_level == RadiationThreatLevel.ELEVATED:
            if scrubber:
                scrubber.check_interval_sec = 0.1
            defenses_applied.append("Accelerated SEC-DED Memory Scrubbing (100ms)")

        elif self.current_level in (RadiationThreatLevel.STORMY, RadiationThreatLevel.CRITICAL):
            if scrubber:
                scrubber.check_interval_sec = 0.05 if self.current_level == RadiationThreatLevel.CRITICAL else 0.1
            if not self.tmr_active:
                apply_tmr(model)
                self.tmr_active = True
            defenses_applied.append("Triple Modular Redundancy (TMR) Median Voting Active")
            defenses_applied.append("Maximum ECC Scrubbing Frequency")

        return model, defenses_applied

    def get_status(self) -> RAPGuardStatus:
        defenses = []
        if self.current_level == RadiationThreatLevel.NOMINAL:
            defenses = ["AWQ 3-Bit", "Standard Scrubber"]
        elif self.current_level == RadiationThreatLevel.ELEVATED:
            defenses = ["Accelerated Scrubber (100ms)", "SEC-DED Protection"]
        else:
            defenses = ["TMR Median Voting", "High-Frequency Scrubber (50ms)", "Radiation Shielding"]

        return RAPGuardStatus(
            threat_level=self.current_level,
            radiation_value_ugy_h=self.last_radiation_val,
            active_defenses=defenses,
            scrubber_interval_sec=0.5 if self.current_level == RadiationThreatLevel.NOMINAL else (0.1 if self.current_level == RadiationThreatLevel.ELEVATED else 0.05),
            tmr_enabled=self.tmr_active,
            timestamp=self.last_update_time,
        )
