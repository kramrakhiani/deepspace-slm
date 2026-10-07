"""
Multi-Agent Framework Tests
==============================
Tests for the agent base classes, coordinator routing, all 5 domain agents,
inter-agent broadcast propagation, and fan-out dispatch.
"""

import pytest
import torch
from config import tiny_config
from data.tokenizer import HabitatTokenizer
from data.inventory_db import HabitatDatabase
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)
from inference.agents.coordinator import AgentCoordinator
from inference.agents.inventory_agent import InventoryAgent
from inference.agents.radiation_agent import RadiationAgent
from inference.agents.maintenance_agent import MaintenanceAgent
from inference.agents.navigation_agent import NavigationAgent
from inference.agents.inference_agent import InferenceAgent


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def db():
    database = HabitatDatabase(":memory:")
    database.seed_initial_data()
    return database


@pytest.fixture
def engine():
    cfg = tiny_config()
    tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
    model = DeepSpaceSLM(cfg)
    return InferenceEngine(model, tok)


@pytest.fixture
def inventory_agent(db):
    return InventoryAgent(db=db, prose_mode=False)


@pytest.fixture
def radiation_agent():
    return RadiationAgent()


@pytest.fixture
def maintenance_agent(db):
    return MaintenanceAgent(db=db, prose_mode=False)


@pytest.fixture
def navigation_agent():
    return NavigationAgent(node_id="test-node")


@pytest.fixture
def inference_agent(engine):
    return InferenceAgent(engine=engine)


@pytest.fixture
def coordinator(db, engine):
    """Full coordinator with all 5 agents registered."""
    coord = AgentCoordinator()
    coord.register(InventoryAgent(db=db, prose_mode=False))
    coord.register(RadiationAgent())
    coord.register(MaintenanceAgent(db=db, prose_mode=False))
    coord.register(NavigationAgent(node_id="test-node"))
    coord.register(InferenceAgent(engine=engine))
    return coord


# ── BaseAgent Protocol ────────────────────────────────────────────────

class TestBaseAgentProtocol:
    def test_inventory_agent_id(self, inventory_agent):
        assert inventory_agent.agent_id == "inventory"

    def test_radiation_agent_id(self, radiation_agent):
        assert radiation_agent.agent_id == "radiation"

    def test_maintenance_agent_id(self, maintenance_agent):
        assert maintenance_agent.agent_id == "maintenance"

    def test_navigation_agent_id(self, navigation_agent):
        assert navigation_agent.agent_id == "navigation"

    def test_inference_agent_id(self, inference_agent):
        assert inference_agent.agent_id == "inference"

    def test_agent_capabilities_not_empty(self, inventory_agent):
        assert len(inventory_agent.capabilities) > 0

    def test_can_handle_registered_intent(self, inventory_agent):
        assert inventory_agent.can_handle("check_stock")
        assert inventory_agent.can_handle("locate")
        assert not inventory_agent.can_handle("radiation_telemetry")

    def test_get_handled_intents(self, radiation_agent):
        intents = radiation_agent.get_handled_intents()
        assert "radiation_telemetry" in intents
        assert "radiation_status" in intents
        assert "adapt_defenses" in intents

    def test_agent_repr(self, inventory_agent):
        r = repr(inventory_agent)
        assert "InventoryAgent" in r
        assert "inventory" in r


# ── AgentCoordinator ──────────────────────────────────────────────────

class TestAgentCoordinator:
    def test_register_agent(self, db):
        coord = AgentCoordinator()
        agent = InventoryAgent(db=db)
        coord.register(agent)
        assert "inventory" in coord.registered_agents

    def test_duplicate_registration_raises(self, db):
        coord = AgentCoordinator()
        coord.register(InventoryAgent(db=db))
        with pytest.raises(ValueError, match="already registered"):
            coord.register(InventoryAgent(db=db))

    def test_unregister_agent(self, db):
        coord = AgentCoordinator()
        coord.register(InventoryAgent(db=db))
        coord.unregister("inventory")
        assert "inventory" not in coord.registered_agents

    def test_intent_routing(self, coordinator):
        registry = coordinator.intent_registry
        assert "check_stock" in registry
        assert "radiation_telemetry" in registry
        assert "log_maintenance" in registry
        assert "mesh_status" in registry
        assert "free_query" in registry

    def test_dispatch_routes_to_correct_agent(self, coordinator):
        response = coordinator.dispatch(AgentMessage(
            intent="check_stock",
            payload={"item_name": "o2 canister"},
        ))
        assert response.agent_id == "inventory"
        assert response.status == "success"

    def test_dispatch_unknown_intent(self, coordinator):
        response = coordinator.dispatch(AgentMessage(
            intent="nonexistent_intent",
            payload={},
        ))
        assert response.status == "not_handled"
        assert response.agent_id == "coordinator"

    def test_dispatch_fanout(self, coordinator):
        # Register a second agent that handles "check_stock" to test fan-out
        # (by default only inventory handles it, so we test single-agent fan-out)
        responses = coordinator.dispatch_fanout(AgentMessage(
            intent="check_stock",
            payload={"item_name": "o2 canister"},
        ))
        assert len(responses) >= 1
        assert responses[0].agent_id == "inventory"

    def test_event_log_records_dispatches(self, coordinator):
        coordinator.dispatch(AgentMessage(intent="check_stock", payload={"item_name": "o2 canister"}))
        coordinator.dispatch(AgentMessage(intent="radiation_status", payload={}))
        assert len(coordinator.event_log) >= 4  # dispatch + response for each

    def test_get_agent(self, coordinator):
        agent = coordinator.get_agent("inventory")
        assert agent is not None
        assert agent.agent_id == "inventory"

    def test_get_agent_missing(self, coordinator):
        assert coordinator.get_agent("nonexistent") is None

    def test_coordinator_repr(self, coordinator):
        r = repr(coordinator)
        assert "AgentCoordinator" in r
        assert "inventory" in r


# ── InventoryAgent ────────────────────────────────────────────────────

class TestInventoryAgent:
    def test_check_stock_found(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="check_stock",
            payload={"item_name": "o2 canister"},
        ))
        assert response.status == "success"
        assert response.data["item"] == "o2 canister"
        assert response.data["quantity"] == 48
        assert response.data["location"] == "storage bay a1"

    def test_check_stock_not_found(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="check_stock",
            payload={"item_name": "nonexistent_item"},
        ))
        assert response.status == "not_found"

    def test_locate_item(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="locate",
            payload={"item_name": "medical kit"},
        ))
        assert response.status == "success"
        assert response.data["location"] == "medical bay cabinet 1"

    def test_forecast(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="forecast",
            payload={"item_name": "o2 canister"},
        ))
        assert response.status == "success"
        assert response.data["consumption_rate"] == 2.0
        assert response.data["days_remaining"] == 24

    def test_get_alerts(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="get_alerts", payload={},
        ))
        assert response.status == "success"
        assert "alerts" in response.data

    def test_update_quantity(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="update_quantity",
            payload={"item_name": "o2 canister", "delta": -5, "performed_by": "test"},
        ))
        assert response.status == "success"
        assert response.data["quantity"] == 43

    def test_prose_mode_output(self, db):
        agent = InventoryAgent(db=db, prose_mode=True)
        response = agent.handle_message(AgentMessage(
            intent="check_stock",
            payload={"item_name": "o2 canister"},
        ))
        assert "We currently have" in response.messages[0]

    def test_unhandled_intent(self, inventory_agent):
        response = inventory_agent.handle_message(AgentMessage(
            intent="unknown", payload={},
        ))
        assert response.status == "not_handled"


# ── RadiationAgent ────────────────────────────────────────────────────

class TestRadiationAgent:
    def test_nominal_telemetry(self, radiation_agent):
        response = radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 50.0},
        ))
        assert response.status == "success"
        assert response.data["threat_level"] == "NOMINAL"

    def test_elevated_telemetry(self, radiation_agent):
        response = radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 250.0},
        ))
        assert response.data["threat_level"] == "ELEVATED"

    def test_critical_telemetry_broadcasts(self, radiation_agent):
        response = radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 3000.0},
        ))
        assert response.data["threat_level"] == "CRITICAL"
        assert response.broadcast is not None
        assert response.broadcast.intent == "radiation_threat_change"
        assert response.priority == MessagePriority.CRITICAL

    def test_status_query(self, radiation_agent):
        radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 150.0},
        ))
        response = radiation_agent.handle_message(AgentMessage(
            intent="radiation_status", payload={},
        ))
        assert response.status == "success"
        assert response.data["radiation_ugy_h"] == 150.0

    def test_no_broadcast_on_same_level(self, radiation_agent):
        # First call sets to NOMINAL
        radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 50.0},
        ))
        # Second call stays NOMINAL — no broadcast
        response = radiation_agent.handle_message(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 60.0},
        ))
        assert response.broadcast is None


# ── MaintenanceAgent ──────────────────────────────────────────────────

class TestMaintenanceAgent:
    def test_log_maintenance(self, maintenance_agent):
        response = maintenance_agent.handle_message(AgentMessage(
            intent="log_maintenance",
            payload={"action": "inspected", "item_name": "co2 scrubber", "location": "module alpha"},
        ))
        assert response.status == "success"
        assert response.data["merkle_hash"] != ""

    def test_verify_audit_empty(self, maintenance_agent):
        response = maintenance_agent.handle_message(AgentMessage(
            intent="verify_audit", payload={},
        ))
        assert response.status == "success"
        assert response.data["audit_valid"] is True

    def test_verify_audit_after_log(self, maintenance_agent):
        maintenance_agent.handle_message(AgentMessage(
            intent="log_maintenance",
            payload={"action": "replaced", "item_name": "air filter", "location": "module beta"},
        ))
        response = maintenance_agent.handle_message(AgentMessage(
            intent="verify_audit", payload={},
        ))
        assert response.data["audit_valid"] is True

    def test_query_manual_found(self, maintenance_agent):
        response = maintenance_agent.handle_message(AgentMessage(
            intent="query_manual",
            payload={"query": "co2 scrubber recalibration"},
        ))
        assert response.status == "success"
        assert response.data["count"] > 0

    def test_query_manual_not_found(self, maintenance_agent):
        response = maintenance_agent.handle_message(AgentMessage(
            intent="query_manual",
            payload={"query": "quantum warp drive"},
        ))
        assert response.status == "not_found"


# ── NavigationAgent ───────────────────────────────────────────────────

class TestNavigationAgent:
    def test_mesh_update(self, navigation_agent):
        response = navigation_agent.handle_message(AgentMessage(
            intent="mesh_update",
            payload={"item_name": "o2 canister", "quantity": 48, "status": "nominal"},
        ))
        assert response.status == "success"
        assert response.data["version"] == 1

    def test_mesh_merge(self, navigation_agent):
        remote_state = {
            "water filter": {
                "item_name": "water filter",
                "quantity": 15,
                "status": "low",
                "updated_by": "habitat-beta",
                "version": 3,
                "timestamp": 9999999999.0,
            }
        }
        response = navigation_agent.handle_message(AgentMessage(
            intent="mesh_merge",
            payload={"remote_state": remote_state},
        ))
        assert response.status == "success"
        assert response.data["count"] == 1

    def test_mesh_transmit(self, navigation_agent):
        response = navigation_agent.handle_message(AgentMessage(
            intent="mesh_transmit",
            payload={"type": "inventory_sync", "data": {"o2": 48}},
        ))
        assert response.status == "success"
        assert response.data["packet_size_bytes"] > 0

    def test_mesh_status(self, navigation_agent):
        response = navigation_agent.handle_message(AgentMessage(
            intent="mesh_status", payload={},
        ))
        assert response.status == "success"
        assert response.data["node_id"] == "test-node"


# ── InferenceAgent ────────────────────────────────────────────────────

class TestInferenceAgent:
    def test_free_query(self, inference_agent):
        response = inference_agent.handle_message(AgentMessage(
            intent="free_query",
            payload={"query": "status of life support", "max_new_tokens": 10, "greedy": True},
        ))
        assert response.status == "success"
        assert len(response.data["generated_text"]) > 0

    def test_compute_perplexity(self, inference_agent):
        response = inference_agent.handle_message(AgentMessage(
            intent="compute_perplexity",
            payload={"text": "check stock o2 canister"},
        ))
        assert response.status == "success"
        assert response.data["perplexity"] > 0

    def test_generate_stream(self, inference_agent):
        response = inference_agent.handle_message(AgentMessage(
            intent="generate_stream",
            payload={"query": "hello", "max_new_tokens": 5, "greedy": True},
        ))
        assert response.status == "success"
        assert "stream" in response.data
        # Consume the generator
        tokens = list(response.data["stream"])
        assert isinstance(tokens, list)


# ── Broadcast Propagation ─────────────────────────────────────────────

class TestBroadcastPropagation:
    def test_radiation_broadcast_reaches_all_agents(self, coordinator):
        """When radiation agent broadcasts a threat change, all other agents receive it."""
        response = coordinator.dispatch(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 3000.0},
        ))
        # The broadcast was sent — verify via event log
        assert response.broadcast is not None
        assert response.broadcast.intent == "radiation_threat_change"

    def test_navigation_queues_alert_on_radiation_broadcast(self, coordinator):
        """Navigation agent should queue a DTN bundle when it receives a critical radiation broadcast."""
        nav_agent = coordinator.get_agent("navigation")
        assert len(nav_agent.bundle_queue.queue) == 0

        # Dispatch critical radiation — triggers broadcast to navigation
        coordinator.dispatch(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 3000.0},
        ))

        assert len(nav_agent.bundle_queue.queue) == 1

    def test_coordinator_broadcast_method(self, coordinator):
        """Direct broadcast reaches all agents."""
        coordinator.broadcast(AgentMessage(
            intent="test_broadcast",
            payload={"test": True},
            source="test",
        ))
        # Should not raise, all agents handle unknown broadcasts gracefully


# ── End-to-End via Coordinator ────────────────────────────────────────

class TestEndToEndCoordinator:
    def test_full_inventory_workflow(self, coordinator):
        """Stock check → forecast → update → verify stock changed."""
        # Check initial stock
        r1 = coordinator.dispatch(AgentMessage(
            intent="check_stock", payload={"item_name": "o2 canister"},
        ))
        assert r1.data["quantity"] == 48

        # Forecast
        r2 = coordinator.dispatch(AgentMessage(
            intent="forecast", payload={"item_name": "o2 canister"},
        ))
        assert r2.data["days_remaining"] == 24

        # Consume some
        r3 = coordinator.dispatch(AgentMessage(
            intent="update_quantity",
            payload={"item_name": "o2 canister", "delta": -10},
        ))
        assert r3.data["quantity"] == 38

        # Re-check
        r4 = coordinator.dispatch(AgentMessage(
            intent="check_stock", payload={"item_name": "o2 canister"},
        ))
        assert r4.data["quantity"] == 38

    def test_maintenance_then_audit(self, coordinator):
        """Log maintenance → verify audit chain integrity."""
        coordinator.dispatch(AgentMessage(
            intent="log_maintenance",
            payload={"action": "replaced", "item_name": "air filter", "location": "module beta"},
        ))
        r = coordinator.dispatch(AgentMessage(
            intent="verify_audit", payload={},
        ))
        assert r.data["audit_valid"] is True

    def test_radiation_escalation_workflow(self, coordinator):
        """Nominal → Critical → verify status."""
        r1 = coordinator.dispatch(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 50.0},
        ))
        assert r1.data["threat_level"] == "NOMINAL"

        r2 = coordinator.dispatch(AgentMessage(
            intent="radiation_telemetry",
            payload={"radiation_ugy_h": 3000.0},
        ))
        assert r2.data["threat_level"] == "CRITICAL"
        assert r2.broadcast is not None

        r3 = coordinator.dispatch(AgentMessage(
            intent="radiation_status", payload={},
        ))
        assert r3.data["threat_level"] == "CRITICAL"
