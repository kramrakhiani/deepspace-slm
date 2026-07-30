import pytest
import torch
import torch.nn as nn

from config import tiny_config
from model.feedforward import MoESwiGLUFeedForward
from model.attention import DifferentialCausalAttention
from model.sensor_encoder import SensorTelemetryEncoder
from model.vision_encoder import CompactVisionEncoder
from data.dtn_mesh import InventoryCRDTMesh, DTNBundleQueue
from data.inventory_db import HabitatDatabase
from inference.habitat_agent import HabitatAgent, SafetyBoundaryValidator
from inference.engine import InferenceEngine
from data.tokenizer import HabitatTokenizer
from model.transformer import DeepSpaceSLM


class TestNextGenFeatures:
    def test_moe_feedforward(self):
        moe = MoESwiGLUFeedForward(hidden_dim=128, ffn_dim=344, num_experts=4, top_k=1)
        x = torch.randn(2, 8, 128)
        out = moe(x)
        assert out.shape == (2, 8, 128)

    def test_differential_attention(self):
        diff_attn = DifferentialCausalAttention(hidden_dim=128, num_heads=4, head_dim=32)
        x = torch.randn(2, 8, 128)
        attn_out, new_kv = diff_attn(x)
        assert attn_out.shape == (2, 8, 128)

    def test_sensor_telemetry_encoder(self):
        encoder = SensorTelemetryEncoder(num_channels=5, embed_dim=128)
        telemetry = torch.randn(2, 10, 5)
        out = encoder(telemetry)
        assert out.shape == (2, 10, 128)

    def test_compact_vision_encoder(self):
        encoder = CompactVisionEncoder(in_channels=3, patch_size=16, embed_dim=128)
        images = torch.randn(2, 3, 64, 64)
        out = encoder(images)
        assert out.shape == (2, 16, 128)

    def test_crdt_mesh_and_dtn(self):
        mesh_a = InventoryCRDTMesh(node_id="hab_alpha")
        mesh_b = InventoryCRDTMesh(node_id="hab_beta")

        mesh_a.update_item("o2 canister", 40, "nominal")
        mesh_b.merge_remote_state(mesh_a.state)
        assert "o2 canister" in mesh_b.state

        dtn = DTNBundleQueue(node_id="hab_alpha")
        dtn.create_space_packet({"event": "inventory_sync", "item": "o2 canister"})
        rec = dtn.transmit_next()
        assert rec["event"] == "inventory_sync"

    def test_merkle_dag_audit_log(self):
        db = HabitatDatabase(":memory:")
        db.seed_initial_data()
        db.update_quantity("o2 canister", -2)
        db.log_maintenance("inspected", "air filter", "module_a")
        assert db.verify_merkle_chain() is True

    def test_safety_boundary_validator(self):
        validator = SafetyBoundaryValidator(min_o2_quantity=5)
        assert validator.validate_action("o2 canister", current_qty=10, delta=-3) is True
        assert validator.validate_action("o2 canister", current_qty=10, delta=-8) is False

        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)
        engine = InferenceEngine(model, tok)
        agent = HabitatAgent(engine)

        payload = "UPDATE_QTY:o2 canister:-45"
        sig_cmd = agent.security_manager.generate_signature("commander", payload)
        sig_eng = agent.security_manager.generate_signature("engineer", payload)

        with pytest.raises(ValueError):
            agent.secure_update_quantity("o2 canister", -45, sig_cmd, sig_eng)
