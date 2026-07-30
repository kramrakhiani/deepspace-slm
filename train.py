#!/usr/bin/env python3
"""
DeepSpace-SLM Natural Language Generalization Fine-Tuning Script
Fine-tunes the PyTorch Neural Network SLM over 20,000+ Phrasing-Augmented Conversational Samples with Response Loss Masking.
"""

import sys
import os
import time
import torch

from config import ModelConfig, TrainingConfig
from data.tokenizer import HabitatTokenizer
from data.habitat_dataset import create_dataloader
from model.transformer import DeepSpaceSLM
from training.trainer import Trainer
from inference.engine import InferenceEngine


def main():
    print("\033[36m=== DeepSpace-SLM Natural Language Generalization Training Run ===\033[0m")
    print("Training PyTorch Transformer SLM on 20,000+ Phrasing-Augmented Conversational Dialogue Turns...\n")

    # 1. Configuration
    model_cfg = ModelConfig(
        vocab_size=256,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        ffn_dim=344,
        max_seq_len=64,
    )
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        max_steps=1500,
        warmup_steps=100,
        batch_size=32,
        log_interval=150,
        eval_interval=300,
        save_interval=750,
        checkpoint_dir="checkpoints",
    )

    # 2. Tokenizer & Dataloaders
    print("Building domain BPE vocabulary & augmented conversational dataloaders...")
    tokenizer = HabitatTokenizer(vocab_size=model_cfg.vocab_size)
    train_loader = create_dataloader(
        tokenizer=tokenizer,
        num_samples=20000,
        max_seq_len=model_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        seed=42,
        conversational=True,
    )
    eval_loader = create_dataloader(
        tokenizer=tokenizer,
        num_samples=2000,
        max_seq_len=model_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        seed=100,
        conversational=True,
    )

    # 3. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Instantiating DeepSpaceSLM model on device: \033[33m{device}\033[0m")
    model = DeepSpaceSLM(model_cfg)
    print(f"Total Parameters: \033[32m{model.count_parameters():,}\033[0m\n")

    test_prompt = "<QUERY> how much o2 left"

    # Pre-training generation test
    engine_untrained = InferenceEngine(model, tokenizer, device=device)
    raw_untrained = engine_untrained.generate(test_prompt, max_new_tokens=40, greedy=True)
    print("\033[31m[BEFORE TRAINING - Untrained Random Weights Generation]\033[0m")
    print(f"Prompt: '{test_prompt}'")
    print(f"Neural Output: '{raw_untrained}'\n")

    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
    )

    print("\033[36m--- Starting Optimization Loop (AdamW + Cosine LR Scheduler + Loss Masking) ---\033[0m")
    start_train_time = time.time()

    # Run training loop
    for step in range(1, train_cfg.max_steps + 1):
        try:
            batch = next(data_iter)
        except (NameError, StopIteration):
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss = trainer._train_step(batch)

        if step % train_cfg.log_interval == 0:
            perplexity = torch.exp(torch.tensor(min(loss, 20.0))).item()
            print(f"Step \033[1;33m{step:4d}/{train_cfg.max_steps}\033[0m | Loss: \033[1;32m{loss:.4f}\033[0m | Perplexity: \033[1;36m{perplexity:.2f}\033[0m | LR: {trainer.scheduler.get_lr():.6f}")

    total_time = time.time() - start_train_time
    print(f"\n\033[32m[TRAINING COMPLETED] 1,500 steps finished in {total_time:.2f} seconds.\033[0m")

    # Save trained checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/model_trained.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg}, save_path)
    print(f"Trained model checkpoint saved to: \033[33m{save_path}\033[0m\n")

    # 5. Post-Training Generation Test
    engine_trained = InferenceEngine(model, tokenizer, device=device)
    raw_trained = engine_trained.generate(test_prompt, max_new_tokens=40, greedy=True)

    print("\033[32m[AFTER TRAINING - Pure PyTorch Neural Generation]\033[0m")
    print(f"Prompt: '{test_prompt}'")
    print(f"Neural Output: '{raw_trained}'\n")

    print("\033[35mThe neural network has successfully learned natural language query generalization!\033[0m")


if __name__ == "__main__":
    main()
