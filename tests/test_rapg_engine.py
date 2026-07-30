import pytest
import torch
import torch.nn as nn

from config import tiny_config
from model.transformer import DeepSpaceSLM
from quantization.rapg_engine import RAPGuardEngine, RadiationThreatLevel
from inference.engine import InferenceEngine
from data.tokenizer import HabitatTokenizer
from inference.habitat_agent import HabitatAgent


class TestRAPGuardEngine:
    def test_threat_level_evaluation(self):
        engine = RAPGuardEngine()

        level = engine.evaluate_radiation_telemetry(45.0)
        assert level == RadiationThreatLevel.NOMINAL

        level = engine.evaluate_radiation_telemetry(250.0)
        assert level == RadiationThreatLevel.ELEVATED

        level = engine.evaluate_radiation_telemetry(800.0)
        assert level == RadiationThreatLevel.STORMY

        level = engine.evaluate_radiation_telemetry(3500.0)
        assert level == RadiationThreatLevel.CRITICAL

    def test_model_defense_adaptation(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        rapg = RAPGuardEngine()

        rapg.evaluate_radiation_telemetry(1200.0)
        adapted_model, defenses = rapg.adapt_model_defenses(model)

        assert rapg.tmr_active is True
        assert "Triple Modular Redundancy (TMR) Median Voting Active" in defenses

    def test_habitat_agent_telemetry_integration(self):
        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)
        engine = InferenceEngine(model, tok)
        agent = HabitatAgent(engine)

        status = agent.process_environmental_telemetry(1500.0)
        assert status.threat_level == RadiationThreatLevel.STORMY
        assert status.tmr_enabled is True

        report = agent.status_report()
        assert "RAP-G Threat Level: STORMY" in report
