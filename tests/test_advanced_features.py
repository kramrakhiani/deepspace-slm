import os
import tempfile
import pytest
import torch
import torch.nn as nn

from config import tiny_config
from model.transformer import DeepSpaceSLM
from model.attention import CausalSelfAttention
from inference.engine import InferenceEngine
from data.tokenizer import HabitatTokenizer
from quantization.ptq import quantize_awq, AWQQuantizedLinear
from quantization.bitflip_resilience import compute_hamming_parity, verify_and_repair_hamming, IdleMemoryScrubber
from inference.habitat_agent import HabitatAgent, TwoKeySecurityManager
from data.manuals import HabitatManualRetriever


class TestAdvancedFeatures:
    def test_attention_sinks_max_kv(self):
        attn = CausalSelfAttention(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            max_kv_size=8,
            num_sink_tokens=2,
        )
        x = torch.randn(1, 1, 128)
        kv_cache = (torch.randn(1, 10, 4, 32), torch.randn(1, 10, 4, 32))
        _, new_kv = attn(x, kv_cache=kv_cache)
        assert new_kv[0].shape[1] == 8
        assert new_kv[1].shape[1] == 8

    def test_medusa_speculative_decoding(self):
        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg, enable_medusa=True, num_medusa_heads=2)
        engine = InferenceEngine(model, tok)

        prompt = "check stock o2 canister"
        out = engine.generate(prompt, max_new_tokens=10, speculative=True)
        assert isinstance(out, str)

    def test_awq_quantization(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        model = quantize_awq(model, salient_ratio=0.05, bits=3)

        awq_count = sum(1 for m in model.modules() if isinstance(m, AWQQuantizedLinear))
        assert awq_count > 0

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        logits, _ = model(input_ids)
        assert logits.shape == (1, 3, cfg.vocab_size)

    def test_hamming_sec_ded_repair(self):
        t = torch.randn(4, 4)
        parity = compute_hamming_parity(t)

        corrupted = t.clone()
        corrupted[0, 0] += 10.0

        repaired_t, num_repaired = verify_and_repair_hamming(corrupted, t, parity)
        assert num_repaired > 0
        assert torch.equal(repaired_t, t)

    def test_idle_memory_scrubber(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        scrubber = IdleMemoryScrubber(model, check_interval_sec=0.1)

        dict(model.named_parameters())["token_embedding.embedding.weight"].data[0, 0] += 5.0
        repaired = scrubber.scrub_once()
        assert repaired > 0

    def test_habitat_manual_rag(self):
        retriever = HabitatManualRetriever()
        results = retriever.search("o2 canister")
        assert len(results) > 0
        assert "o2 canister" in results[0]["topic"]

    def test_two_key_security_manager(self):
        sec = TwoKeySecurityManager()
        payload = "UPDATE_QTY:o2 canister:-5"
        sig_cmd = sec.generate_signature("commander", payload)
        sig_eng = sec.generate_signature("engineer", payload)

        assert sec.verify_two_keys(payload, sig_cmd, sig_eng) is True
        assert sec.verify_two_keys(payload, sig_cmd, "invalid") is False

    def test_agent_secure_update(self):
        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)
        engine = InferenceEngine(model, tok)
        agent = HabitatAgent(engine)

        payload = "UPDATE_QTY:o2 canister:-2"
        sig_cmd = agent.security_manager.generate_signature("commander", payload)
        sig_eng = agent.security_manager.generate_signature("engineer", payload)

        rec = agent.secure_update_quantity("o2 canister", -2, sig_cmd, sig_eng)
        assert rec is not None

        with pytest.raises(PermissionError):
            agent.secure_update_quantity("o2 canister", -2, "bad_sig", sig_eng)
