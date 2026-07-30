from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    ffn_dim: int = 1376
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    num_kv_heads: Optional[int] = None
    enable_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 1
    enable_differential_attn: bool = False

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0
        assert self.head_dim == self.hidden_dim // self.num_heads, (
            f"head_dim ({self.head_dim}) must equal hidden_dim // num_heads "
            f"({self.hidden_dim // self.num_heads})"
        )
        if self.num_kv_heads is not None:
            assert self.num_heads % self.num_kv_heads == 0


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eps: float = 1e-8
    warmup_steps: int = 100
    max_steps: int = 5000
    min_lr_ratio: float = 0.1
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 500
    enable_qat: bool = False
    qat_start_step: int = 1000
    label_smoothing: float = 0.1
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class QuantizationConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    num_calibration_batches: int = 32
    fake_quant_enabled: bool = True
    observer_enabled: bool = True
    export_format: str = "binary"


@dataclass
class MissionConfig:
    enable_tmr: bool = False
    checksum_weights: bool = True
    bitflip_rate: float = 1e-6
    max_memory_mb: int = 64
    max_power_watts: float = 5.0
    max_inference_latency_ms: float = 500.0
    critical_stock_threshold: float = 0.1
    low_stock_threshold: float = 0.25
    forecast_horizon_days: int = 30
    device: str = "cpu"


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        ffn_dim=344,
        max_seq_len=64,
    )


def base_config() -> ModelConfig:
    return ModelConfig()


def mission_config():
    return {
        "model": base_config(),
        "training": TrainingConfig(enable_qat=True),
        "quantization": QuantizationConfig(weight_bits=4),
        "mission": MissionConfig(enable_tmr=True),
    }
