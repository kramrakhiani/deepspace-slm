#!/usr/bin/env python3
"""
DeepSpace-SLM Pure Neural SLM Chat REPL
100% Direct Autoregressive Neural Network Text Generation from DeepSpaceSLM PyTorch Weights.
Zero Python if/else intent routing or hardcoded database overrides.
"""

import sys
import os
import time
import torch
from config import ModelConfig
from data.tokenizer import HabitatTokenizer
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine


def main():
    print("\033[36m=== DeepSpace-SLM Pure Neural Network Chat REPL ===\033[0m")
    print("100% Pure Neural SLM Generation — Every response is generated directly by PyTorch weight matrices.\nType 'exit' to quit.\n")

    cfg = ModelConfig(
        vocab_size=256,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        ffn_dim=344,
        max_seq_len=64,
    )
    tokenizer = HabitatTokenizer(vocab_size=cfg.vocab_size)
    model = DeepSpaceSLM(cfg)

    ckpt_path = "checkpoints/model_trained.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\033[32m[PURE NEURAL MODE] Loaded trained PyTorch SLM weights from {ckpt_path}\033[0m\n")
    else:
        print("\033[33m[NOTE] Running with initial model weights. Run 'python3 train.py' to train.\033[0m\n")

    engine = InferenceEngine(model, tokenizer)

    while True:
        try:
            prompt = input("\033[1;32mUser > \033[0m").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break

            formatted_prompt = f"<QUERY> {prompt}"
            sys.stdout.write("\033[1;34mSLM > \033[0m")
            sys.stdout.flush()

            # 100% Direct Neural Network Autoregressive Generation
            for chunk in engine.generate_stream(formatted_prompt, max_new_tokens=40, greedy=True):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                time.sleep(0.02)
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    main()
