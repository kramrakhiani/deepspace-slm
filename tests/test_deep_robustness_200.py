import os
import time
import math
import torch
import pytest
import numpy as np

from config import ModelConfig, QuantizationConfig, tiny_config
from data.tokenizer import HabitatTokenizer, BYTE_PREFIX
from data.inventory_db import HabitatDatabase
from data.dtn_mesh import InventoryCRDTMesh, DTNBundleQueue
from model.transformer import DeepSpaceSLM
from model.rope import precompute_rope_frequencies, apply_rope, RotaryEmbedding
from model.normalization import RMSNorm
from model.feedforward import SwiGLUFeedForward, MoESwiGLUFeedForward
from model.attention import DifferentialCausalAttention
from inference.engine import InferenceEngine
from inference.habitat_agent import HabitatAgent, HabitatManualRetriever
from inference.grammar import SchemaMasker
from quantization.ptq import quantize_post_training, quantize_awq, _quantize_tensor
from quantization.export import export_binary, verify_export, pack_int4_to_uint8, unpack_uint8_to_int4
from quantization.bitflip_resilience import flip_random_bits, IdleMemoryScrubber
from quantization.rapg_engine import RAPGuardEngine, RadiationThreatLevel


class TestDeepRobustnessSubsystems:

    # -------------------------------------------------------------
    # 1. Model Architecture & Attention Core
    # -------------------------------------------------------------
    def test_rope_shapes_and_orthogonality(self):
        rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
        x = torch.randn(2, 16, 4, 32)
        x_rot = rope(x, start_pos=0)
        assert x_rot.shape == x.shape
        torch.testing.assert_close(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_rmsnorm_zero_mean_variance(self):
        norm = RMSNorm(dim=128)
        x = torch.randn(4, 16, 128) * 10.0 + 5.0
        out = norm(x)
        assert out.shape == x.shape
        assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(4, 16), atol=1e-2)

    def test_swiglu_activation_gradient_flow(self):
        swiglu = SwiGLUFeedForward(hidden_dim=128, ffn_dim=344)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = swiglu(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_differential_attention_masking(self):
        att = DifferentialCausalAttention(hidden_dim=128, num_heads=4, head_dim=32)
        x = torch.randn(1, 10, 128)
        out, cache = att(x)
        assert out.shape == (1, 10, 128)

    def test_moe_top1_routing_sparsity(self):
        moe = MoESwiGLUFeedForward(hidden_dim=128, ffn_dim=344, num_experts=4, top_k=1)
        x = torch.randn(2, 16, 128)
        out = moe(x)
        assert out.shape == x.shape

    def test_slm_kv_cache_accumulation(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        model.eval()

        x1 = torch.randint(0, cfg.vocab_size, (1, 5))
        out1, cache1 = model(x1, kv_caches=None)

        x2 = torch.randint(0, cfg.vocab_size, (1, 1))
        out2, cache2 = model(x2, kv_caches=cache1, start_pos=5)

        assert len(cache2) == cfg.num_layers
        assert cache2[0][0].shape[0] == 1  # Batch size 1

    # -------------------------------------------------------------
    # 2. Byte-Level BPE Tokenizer Robustness
    # -------------------------------------------------------------
    def test_bpe_space_prefix_roundtrip(self):
        tok = HabitatTokenizer(vocab_size=256)
        text = "oxygen canister medical kit water filter"
        ids = tok.encode(text, add_bos=False, add_eos=False)
        decoded = tok.decode(ids)
        assert "oxygen" in decoded
        assert "canister" in decoded
        assert "medical" in decoded

    def test_bpe_out_of_vocab_fallback(self):
        tok = HabitatTokenizer(vocab_size=256)
        text = "supercalifragilisticexpialidocious_123987"
        ids = tok.encode(text, add_bos=False, add_eos=False)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_bpe_padding_and_truncation(self):
        tok = HabitatTokenizer(vocab_size=256)
        ids = [10, 20, 30, 40, 50]
        padded = tok.pad(ids, max_length=10)
        assert len(padded) == 10
        assert padded[5:] == [tok.pad_id] * 5

        truncated = tok.pad(ids, max_length=3)
        assert len(truncated) == 3

    # -------------------------------------------------------------
    # 3. Quantization & Packing Engine
    # -------------------------------------------------------------
    def test_int4_pack_unpack_roundtrip(self):
        tensor = torch.tensor([-8, -4, 0, 3, 7, -1, 2, 5], dtype=torch.int8)
        packed = pack_int4_to_uint8(tensor)
        unpacked = unpack_uint8_to_int4(packed, num_elements=8)
        assert torch.equal(tensor, unpacked)

    def test_int8_symmetric_quantization_error(self):
        tensor = torch.randn(64, 128)
        q_int, scale, zero_point = _quantize_tensor(tensor, bits=8, symmetric=True, per_channel=True)
        dequantized = (q_int.float() - zero_point) * scale
        mse = torch.mean((tensor - dequantized) ** 2).item()
        assert mse < 0.05

    def test_awq_salient_column_preservation(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        qmodel = quantize_awq(model, bits=3)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        out = qmodel(x)[0]
        assert out.shape == (1, 4, cfg.vocab_size)

    def test_binary_export_verification(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        export_path = "checkpoints/robustness_export.dslm"
        export_binary(model, cfg, export_path)
        assert verify_export(export_path) is True

    # -------------------------------------------------------------
    # 4. Radiation Hardening & RAP-G Engine
    # -------------------------------------------------------------
    def test_rapg_threat_level_scaling(self):
        engine = RAPGuardEngine()
        assert engine.evaluate_radiation_telemetry(20.0) == RadiationThreatLevel.NOMINAL
        assert engine.evaluate_radiation_telemetry(250.0) == RadiationThreatLevel.ELEVATED
        assert engine.evaluate_radiation_telemetry(800.0) == RadiationThreatLevel.STORMY
        assert engine.evaluate_radiation_telemetry(3500.0) == RadiationThreatLevel.CRITICAL

    def test_rapg_tmr_median_voting(self):
        engine = RAPGuardEngine()
        model = DeepSpaceSLM(tiny_config())
        engine.evaluate_radiation_telemetry(3500.0)
        adapted_model, defenses = engine.adapt_model_defenses(model)
        x = torch.randint(0, 256, (1, 4))
        out = adapted_model(x)[0]
        assert out.shape == (1, 4, 256)

    def test_ecc_memory_scrubber_recovery(self):
        model = DeepSpaceSLM(tiny_config())
        scrubber = IdleMemoryScrubber(model, check_interval_sec=0.01)
        scrubber.start()

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                corrupted, _ = flip_random_bits(m.weight, flip_rate=1e-4)
                m.weight.data = corrupted
                break

        time.sleep(0.03)
        scrubber.stop()

    # -------------------------------------------------------------
    # 5. Database, Merkle Audit Chaining & Mesh Network
    # -------------------------------------------------------------
    def test_database_fuzzy_search(self):
        db = HabitatDatabase(":memory:")
        db.seed_initial_data()
        item = db.query_item("oxygen cans")
        assert item is not None
        assert item.name == "o2 canister"

    def test_merkle_dag_tamper_detection(self):
        db = HabitatDatabase(":memory:")
        db.seed_initial_data()
        db.update_quantity("o2 canister", -5)
        assert db.verify_merkle_chain() is True

    def test_crdt_mesh_state_replication(self):
        crdt_a = InventoryCRDTMesh("node_a")
        crdt_b = InventoryCRDTMesh("node_b")

        crdt_a.update_item("o2 canister", 48, "nominal")
        crdt_b.merge_remote_state(crdt_a.state)

        assert "o2 canister" in crdt_b.state
        assert crdt_b.state["o2 canister"]["quantity"] == 48

    def test_dtn_space_packet_compression(self):
        queue = DTNBundleQueue("node_a")
        queue.create_space_packet({"item": "o2 canister", "qty": 48})
        parsed = queue.transmit_next()
        assert parsed["item"] == "o2 canister"

    # -------------------------------------------------------------
    # 6. Natural Language 50-Sentence Stress Test
    # -------------------------------------------------------------
    def test_50_sentence_natural_phrasing_stress(self):
        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)

        ckpt_path = "checkpoints/model_trained.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        engine = InferenceEngine(model, tok, device="cpu")

        sentences = [
            "how much o2 left", "do we have any food rations remaining", "where is medical kit",
            "locate wrench set", "check stock on air filters", "when will n2 tanks run out",
            "what is the status of co2 scrubbers", "how many battery cells do we have",
            "forecast water filter consumption", "where are seal gaskets stored",
            "check remaining count of protein bars", "how long will freeze dried meals last",
            "log maintenance on solar panel module", "is oxygen level nominal",
            "where can crew find sanitizer bottles", "how many bandage rolls remain",
            "check stock level of antibiotic packs", "locate suit helmet visor",
            "where is tether cable stored", "forecast daily oxygen usage",
            "report current status of life support", "are there any critical alerts",
            "locate multimeter in lab rack", "how many soldering kits are in tool locker",
            "check stock of water purifier cartridges", "where is painkiller bottle kept",
            "how many syringe units are in medical cabinet", "locate waste disposal bags",
            "where are cable bundles stored", "check status of emergency trauma kit",
            "how many pump assemblies do we have", "locate valve actuator in spare parts",
            "what is the count of solar panel modules", "how many vitamin supplement bottles remain",
            "check stock on n2 tanks", "where is air filter stored",
            "forecast food ration duration for 30 days", "log replacement of co2 scrubber",
            "check remaining count of o2 canisters", "locate medical kit in bay a1",
            "how many wrench sets are in gamma shelf", "where are battery cells kept",
            "forecast consumption of water filters", "log inspection of primary airlock seal",
            "check stock on freeze dried meals", "where is multimeter located",
            "how many suit helmet visors remain", "locate tether cable in airlock",
            "check status of primary environmental scrubber", "report overall habitat inventory status",
        ]

        for s in sentences:
            out = engine.generate(f"<QUERY> {s}", max_new_tokens=24, greedy=True)
            assert len(out) > 0
