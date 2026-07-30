"""
Quantization Tests
====================
Verify INT8/INT4 quantization, model size reduction, QAT prepare/convert,
and bit-flip resilience scoring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from config import tiny_config, QuantizationConfig
from model.transformer import DeepSpaceSLM
from quantization.ptq import quantize_post_training, compute_model_size, _quantize_tensor
from quantization.qat import FakeQuantize, MinMaxObserver, prepare_qat, convert_qat
from quantization.bitflip_resilience import (
    flip_random_bits, inject_bitflips, evaluate_resilience,
    compute_weight_checksums, verify_weight_integrity,
)


# ─── Tensor Quantization Tests ─────────────────────────────────────────

class TestTensorQuantization:
    def test_int8_symmetric(self):
        t = torch.randn(64, 32)
        q_int, scale, zp = _quantize_tensor(t, bits=8, symmetric=True)
        assert q_int.dtype == torch.int8
        assert q_int.shape == t.shape
        assert zp.sum().item() == 0  # Symmetric → zero_point = 0

    def test_int4_range(self):
        t = torch.randn(64, 32)
        q_int, scale, zp = _quantize_tensor(t, bits=4, symmetric=True)
        assert q_int.min() >= -7
        assert q_int.max() <= 7

    def test_int8_reconstruction_error(self):
        """INT8 quantization should have small reconstruction error."""
        t = torch.randn(128, 64)
        q_int, scale, zp = _quantize_tensor(t, bits=8, symmetric=True)
        reconstructed = (q_int.float() - zp) * scale
        error = (t - reconstructed).abs().mean()
        assert error < 0.05, f"INT8 reconstruction error too high: {error:.4f}"

    def test_int4_reconstruction_error(self):
        """INT4 has more error than INT8 but should still be bounded."""
        t = torch.randn(128, 64)
        q_int, scale, zp = _quantize_tensor(t, bits=4, symmetric=True)
        reconstructed = (q_int.float() - zp) * scale
        error = (t - reconstructed).abs().mean()
        assert error < 0.3, f"INT4 reconstruction error too high: {error:.4f}"


# ─── Post-Training Quantization Tests ──────────────────────────────────

class TestPTQ:
    @pytest.fixture
    def model(self):
        return DeepSpaceSLM(tiny_config())

    def test_int8_quantization(self, model):
        config = QuantizationConfig(weight_bits=8)
        original_size = compute_model_size(model)

        quantized = quantize_post_training(model, config)
        quantized_size = compute_model_size(quantized)

        # Quantized model should be significantly smaller
        # (buffers store int8 + float32 scale, so not exactly 4x)
        assert quantized_size["total_mb"] < original_size["total_mb"]

    def test_int4_quantization(self, model):
        config = QuantizationConfig(weight_bits=4)
        quantize_post_training(model, config)

    def test_quantized_forward_pass(self, model):
        """Quantized model should still produce valid outputs."""
        config = QuantizationConfig(weight_bits=8)
        quantized = quantize_post_training(model, config)

        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        logits, _ = quantized(input_ids)
        assert logits.shape == (1, 8, model.config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


# ─── QAT Tests ─────────────────────────────────────────────────────────

class TestQAT:
    def test_fake_quantize(self):
        fq = FakeQuantize(bits=8)
        fq.train()
        x = torch.randn(32, 64)

        # First pass: observer collects stats
        out = fq(x)
        assert out.shape == x.shape
        assert fq.observer.num_batches_tracked == 1

    def test_fake_quantize_disabled(self):
        fq = FakeQuantize(bits=8)
        fq.enabled = False
        x = torch.randn(32, 64)
        out = fq(x)
        torch.testing.assert_close(out, x)

    def test_observer_tracks_range(self):
        observer = MinMaxObserver()
        observer.train()

        for _ in range(10):
            x = torch.randn(32, 64)
            observer(x)

        assert observer.min_val < 0
        assert observer.max_val > 0
        assert observer.num_batches_tracked == 10

    def test_prepare_convert_qat(self):
        model = DeepSpaceSLM(tiny_config())
        config = QuantizationConfig(weight_bits=8)

        # Prepare
        model = prepare_qat(model, config)

        # Run a few forward passes to calibrate and verify gradients
        model.train()
        for _ in range(3):
            input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
            logits, _ = model(input_ids)
            loss = logits.sum()
            loss.backward()

            # Verify weight gradients are non-None and non-zero
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    assert module.weight.grad is not None
                    assert not torch.all(module.weight.grad == 0)

        # Convert
        model = convert_qat(model, config)


# ─── Bit-Flip Resilience Tests ─────────────────────────────────────────

class TestBitFlipResilience:
    @pytest.fixture
    def model(self):
        return DeepSpaceSLM(tiny_config())

    def test_flip_random_bits(self):
        t = torch.randn(64, 32)
        corrupted, num_flips = flip_random_bits(t, flip_rate=1e-3, seed=42)
        assert corrupted.shape == t.shape
        assert num_flips > 0
        # At least some values should have changed
        assert not torch.allclose(t, corrupted)

    def test_inject_bitflips(self, model):
        corrupted, flip_counts = inject_bitflips(model, flip_rate=1e-4, seed=42)
        assert len(flip_counts) > 0
        assert all(v > 0 for v in flip_counts.values())

    def test_resilience_scoring(self, model):
        model.eval()
        test_input = torch.randint(0, model.config.vocab_size, (1, 8))
        reports = evaluate_resilience(
            model, test_input,
            flip_rates=[1e-6, 1e-4],
            num_trials=2,
        )
        assert len(reports) == 2
        # Higher flip rate should cause more divergence
        assert reports[1].output_divergence >= reports[0].output_divergence * 0.5

    def test_weight_checksums(self, model):
        checksums = compute_weight_checksums(model)
        assert len(checksums) > 0

        # Verify against self should pass
        results = verify_weight_integrity(model, checksums)
        assert all(results.values())

    def test_checksum_detects_corruption(self, model):
        checksums = compute_weight_checksums(model)

        # Corrupt the model
        corrupted, _ = inject_bitflips(model, flip_rate=1e-3, seed=42)

        # Verification should fail
        results = verify_weight_integrity(corrupted, checksums)
        assert not all(results.values())

    def test_quantized_model_checksums_and_bitflips(self, model):
        """Verify checksums and bitflip injection work on quantized models with weight buffers."""
        config = QuantizationConfig(weight_bits=4)
        quantized = quantize_post_training(model, config)

        checksums = compute_weight_checksums(quantized)
        assert len(checksums) > 0
        assert verify_weight_integrity(quantized, checksums)

        corrupted, counts = inject_bitflips(quantized, flip_rate=1e-3, seed=42)
        assert sum(counts.values()) > 0
        assert not all(verify_weight_integrity(corrupted, checksums).values())


# ─── Export Tests ──────────────────────────────────────────────────────

class TestExport:
    def test_binary_export_and_verification(self, tmp_path):
        from quantization.export import export_binary, verify_export
        from quantization.ptq import quantize_post_training

        model = DeepSpaceSLM(tiny_config())
        config = QuantizationConfig(weight_bits=4)
        quantized = quantize_post_training(model, config)

        output_file = str(tmp_path / "model_int4.bin")
        manifest = export_binary(quantized, model.config, output_file, quant_config=config)

        assert os.path.exists(output_file)
        assert verify_export(output_file)
        assert manifest["quantization"]["weight_bits"] == 4
        # Verify 4-bit weights are flagged as packed in manifest
        packed_buffers = [v for k, v in manifest["parameters"].items() if v.get("packed_int4")]
        assert len(packed_buffers) > 0
