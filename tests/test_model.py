"""
Model Architecture Tests
==========================
Verify transformer forward pass, parameter counts, causal masking,
and KV-cache consistency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from config import ModelConfig, tiny_config, base_config
from model.transformer import DeepSpaceSLM
from model.rope import precompute_rope_frequencies, apply_rope, RotaryEmbedding
from model.normalization import RMSNorm
from model.feedforward import SwiGLUFeedForward
from model.embeddings import TokenEmbedding
from model.attention import CausalSelfAttention


# ─── RoPE Tests ────────────────────────────────────────────────────────

class TestRoPE:
    def test_frequency_shape(self):
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)  # max_seq_len, head_dim // 2

    def test_rotation_preserves_norm(self):
        """RoPE should preserve vector norms (it's a rotation)."""
        rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
        x = torch.randn(2, 16, 4, 64)
        x_rotated = rope(x, start_pos=0)

        norms_before = x.norm(dim=-1)
        norms_after = x_rotated.norm(dim=-1)
        torch.testing.assert_close(norms_before, norms_after, atol=1e-5, rtol=1e-5)

    def test_different_positions_different_embeddings(self):
        rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
        x = torch.randn(1, 1, 4, 64)

        rot_0 = rope(x, start_pos=0)
        rot_10 = rope(x, start_pos=10)

        # Different positions should produce different rotations
        assert not torch.allclose(rot_0, rot_10)


# ─── RMSNorm Tests ─────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == (2, 16, 64)

    def test_normalized_rms(self):
        """After RMSNorm (with unit scale), RMS should be ~1."""
        norm = RMSNorm(dim=128)
        x = torch.randn(4, 32, 128) * 5.0  # Large scale
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.1)


# ─── SwiGLU FFN Tests ──────────────────────────────────────────────────

class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLUFeedForward(hidden_dim=64, ffn_dim=172)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self):
        ffn = SwiGLUFeedForward(hidden_dim=64, ffn_dim=172)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 16, 64)


# ─── Embeddings Tests ──────────────────────────────────────────────────

class TestEmbeddings:
    def test_embedding_shape(self):
        emb = TokenEmbedding(vocab_size=256, hidden_dim=64)
        ids = torch.randint(0, 256, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 64)

    def test_weight_tying(self):
        emb = TokenEmbedding(vocab_size=256, hidden_dim=64)
        ids = torch.randint(0, 256, (2, 16))
        hidden = emb(ids)
        logits = emb.output_projection(hidden)
        assert logits.shape == (2, 16, 256)


# ─── Attention Tests ───────────────────────────────────────────────────

class TestAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(
            hidden_dim=64, num_heads=4, head_dim=16, max_seq_len=128
        )
        x = torch.randn(2, 16, 64)
        out, kv_cache = attn(x)
        assert out.shape == (2, 16, 64)

    def test_kv_cache_shape(self):
        attn = CausalSelfAttention(
            hidden_dim=64, num_heads=4, head_dim=16, max_seq_len=128
        )
        x = torch.randn(2, 16, 64)
        _, kv_cache = attn(x)
        k, v = kv_cache
        assert k.shape == (2, 16, 4, 16)  # (batch, seq_len, num_heads, head_dim)
        assert v.shape == (2, 16, 4, 16)

    def test_causal_masking(self):
        """Verify that attention is causal: future tokens don't affect past outputs."""
        attn = CausalSelfAttention(
            hidden_dim=64, num_heads=4, head_dim=16, max_seq_len=128
        )
        torch.manual_seed(42)

        # Create input
        x = torch.randn(1, 8, 64)

        # Get output for full sequence
        out_full, _ = attn(x)

        # Modify token at position 7 (last)
        x_modified = x.clone()
        x_modified[0, 7, :] = torch.randn(64)

        out_modified, _ = attn(x_modified)

        # Tokens 0-6 should produce the same output regardless of token 7
        torch.testing.assert_close(
            out_full[0, :7, :], out_modified[0, :7, :], atol=1e-5, rtol=1e-5
        )


# ─── Full Model Tests ─────────────────────────────────────────────────

class TestDeepSpaceSLM:
    @pytest.fixture
    def tiny_model(self):
        config = tiny_config()
        return DeepSpaceSLM(config)

    def test_forward_shape(self, tiny_model):
        config = tiny_model.config
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        logits, kv_caches = tiny_model(input_ids)
        assert logits.shape == (2, 32, config.vocab_size)
        assert len(kv_caches) == config.num_layers

    def test_parameter_count_tiny(self, tiny_model):
        """Tiny config should have ~400-500K parameters."""
        num_params = tiny_model.count_parameters()
        assert 200_000 < num_params < 2_000_000, f"Got {num_params:,}"

    def test_parameter_count_base(self):
        """Base config should have ~20-30M parameters."""
        model = DeepSpaceSLM(base_config())
        num_params = model.count_parameters()
        assert 15_000_000 < num_params < 40_000_000, f"Got {num_params:,}"

    def test_size_estimation(self, tiny_model):
        size_fp32 = tiny_model.estimate_size_mb(32)
        size_int8 = tiny_model.estimate_size_mb(8)
        size_int4 = tiny_model.estimate_size_mb(4)

        assert size_int8 == pytest.approx(size_fp32 / 4, abs=0.1)
        assert size_int4 == pytest.approx(size_fp32 / 8, abs=0.1)

    def test_kv_cache_consistency(self, tiny_model):
        """
        KV-cache incremental decoding should produce the same logits
        as full-sequence decoding.
        """
        tiny_model.eval()
        config = tiny_model.config
        torch.manual_seed(42)

        # Full sequence
        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        with torch.no_grad():
            logits_full, _ = tiny_model(input_ids)

            # Incremental: process 6 tokens, then 2 more with cache
            logits_prefix, kv_caches = tiny_model(input_ids[:, :6])
            logits_suffix, _ = tiny_model(
                input_ids[:, 6:], kv_caches=kv_caches, start_pos=6
            )

        # Last 2 tokens should match
        torch.testing.assert_close(
            logits_full[:, 6:, :], logits_suffix, atol=1e-4, rtol=1e-4
        )

    def test_repr(self, tiny_model):
        """Model repr should include key info."""
        repr_str = repr(tiny_model)
        assert "DeepSpaceSLM" in repr_str
        assert "parameters" in repr_str
        assert "FP32" in repr_str

    def test_gradient_flow_full_model(self, tiny_model):
        """Gradients should flow through the entire model."""
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 8))
        logits, _ = tiny_model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, param in tiny_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
