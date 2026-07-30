import pytest
import os
import time
import torch
import numpy as np

from config import ModelConfig, QuantizationConfig, tiny_config
from data.tokenizer import HabitatTokenizer
from data.inventory_db import HabitatDatabase
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine
from quantization.ptq import quantize_post_training, quantize_awq
from quantization.bitflip_resilience import flip_random_bits, IdleMemoryScrubber
from quantization.export import export_binary, verify_export


class TestBenchmarkSuite:
    def test_inference_benchmark(self):
        cfg = tiny_config()
        tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)
        engine = InferenceEngine(model, tok, device="cpu")

        t0 = time.time()
        output = engine.generate("<QUERY> check stock o2 canister", max_new_tokens=16, greedy=True)
        dt_ms = (time.time() - t0) * 1000

        assert len(output) > 0
        assert dt_ms < 500.0

    def test_quantization_benchmark(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)

        qconfig = QuantizationConfig(weight_bits=8, symmetric=True, per_channel=True)
        qmodel = quantize_post_training(model, qconfig)

        x = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
        out = qmodel(x)[0]
        assert out.shape == (1, 4, cfg.vocab_size)

    def test_radiation_resilience_benchmark(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        scrubber = IdleMemoryScrubber(model, check_interval_sec=0.01)

        scrubber.start()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                corrupted, _ = flip_random_bits(module.weight, flip_rate=1e-5)
                module.weight.data = corrupted
                break
        time.sleep(0.02)
        scrubber.stop()

    def test_database_merkle_benchmark(self):
        db = HabitatDatabase(":memory:")
        db.seed_initial_data()
        for i in range(10):
            db.update_quantity("o2 canister", 1)
        assert db.verify_merkle_chain() is True

    def test_export_binary_benchmark(self):
        cfg = tiny_config()
        model = DeepSpaceSLM(cfg)
        export_path = "checkpoints/test_suite_export.dslm"
        os.makedirs("checkpoints", exist_ok=True)

        meta = export_binary(model, cfg, export_path)
        assert len(meta) > 0
        assert verify_export(export_path) is True
