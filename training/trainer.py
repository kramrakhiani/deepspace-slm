import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path

from config import ModelConfig, TrainingConfig, QuantizationConfig
from model.transformer import DeepSpaceSLM
from training.loss import LabelSmoothingCrossEntropy
from training.scheduler import CosineWarmupScheduler
from quantization.qat import prepare_qat, convert_qat


class Trainer:
    def __init__(
        self,
        model: DeepSpaceSLM,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        model_config: Optional[ModelConfig] = None,
        train_config: Optional[TrainingConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model_config = model_config or ModelConfig()
        self.train_config = train_config or TrainingConfig()
        self.quant_config = quant_config

        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.train_config.label_smoothing,
            ignore_index=0,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            weight_decay=self.train_config.weight_decay,
            eps=self.train_config.eps,
        )
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.train_config.warmup_steps,
            max_steps=self.train_config.max_steps,
            max_lr=self.train_config.learning_rate,
            min_lr=self.train_config.learning_rate * self.train_config.min_lr_ratio,
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.qat_active = False

        self.step = 0
        self.best_eval_loss = float("inf")
        self.log_history = []

    def train(self) -> Dict[str, Any]:
        config = self.train_config
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)

        running_loss = 0.0
        running_tokens = 0
        start_time = time.time()

        for self.step in range(1, config.max_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            if (config.enable_qat and not self.qat_active
                    and self.step >= config.qat_start_step):
                self._enable_qat()

            loss = self._train_step(batch)
            running_loss += loss
            running_tokens += batch["attention_mask"].sum().item()

            if self.step % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = running_tokens / elapsed
                lr = self.scheduler.get_lr()
                perplexity = math.exp(min(avg_loss, 100))

                log_entry = {
                    "step": self.step,
                    "loss": avg_loss,
                    "perplexity": perplexity,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed": elapsed,
                }
                self.log_history.append(log_entry)

                running_loss = 0.0
                running_tokens = 0
                start_time = time.time()

            if self.eval_loader and self.step % config.eval_interval == 0:
                eval_loss = self._evaluate()
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(ckpt_dir / "best.pt", is_best=True)
                self.model.train()

            if self.step % config.save_interval == 0:
                self._save_checkpoint(ckpt_dir / f"step_{self.step}.pt")

        self._save_checkpoint(ckpt_dir / "final.pt")
        return {
            "final_loss": self.log_history[-1]["loss"] if self.log_history else 0.0,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.step,
            "log_history": self.log_history,
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, target_ids) / self.train_config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if self.step % self.train_config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.train_config.gradient_accumulation_steps

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, target_ids)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _enable_qat(self):
        if self.quant_config is None:
            self.quant_config = QuantizationConfig()
        prepare_qat(self.model, self.quant_config)
        self.qat_active = True

    def _save_checkpoint(self, path: Path, is_best: bool = False):
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
            "model_config": self.model_config.__dict__,
            "train_config": self.train_config.__dict__,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ) -> "Trainer":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_config = ModelConfig(**checkpoint["model_config"])
        train_config = TrainingConfig(**checkpoint["train_config"])
        model = DeepSpaceSLM(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            model_config=model_config,
            train_config=train_config,
            device=device,
        )
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.step = checkpoint["step"]
        trainer.best_eval_loss = checkpoint["best_eval_loss"]
        return trainer
