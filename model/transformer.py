import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import ModelConfig
from model.attention import CausalSelfAttention, DifferentialCausalAttention
from model.feedforward import SwiGLUFeedForward, MoESwiGLUFeedForward
from model.normalization import RMSNorm
from model.embeddings import TokenEmbedding


class MedusaHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [head(x) for head in self.heads]


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)

        if getattr(config, "enable_differential_attn", False):
            self.attention = DifferentialCausalAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )
        else:
            self.attention = CausalSelfAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                num_kv_heads=config.num_kv_heads,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
            )

        if getattr(config, "enable_moe", False):
            self.feed_forward = MoESwiGLUFeedForward(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.ffn_dim,
                num_experts=getattr(config, "num_experts", 4),
                top_k=getattr(config, "top_k_experts", 1),
                dropout=config.dropout,
            )
        else:
            self.feed_forward = SwiGLUFeedForward(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attention(
            self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos, mask=mask
        )
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv_cache


class DeepSpaceSLM(nn.Module):
    def __init__(self, config: ModelConfig, enable_medusa: bool = False, num_medusa_heads: int = 3):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False) if not config.tie_embeddings else None

        self.medusa = MedusaHead(config.hidden_dim, config.vocab_size, num_heads=num_medusa_heads) if enable_medusa else None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        start_pos: int = 0,
        return_medusa: bool = False,
    ):
        x = self.token_embedding(token_ids)
        if kv_caches is None:
            kv_caches = [None] * self.config.num_layers

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(
                x,
                kv_cache=kv_caches[i],
                start_pos=start_pos,
                mask=None,
            )
            new_kv_caches.append(new_kv_cache)

        normed_x = self.final_norm(x)
        logits = self.lm_head(normed_x) if self.lm_head is not None else self.token_embedding.output_projection(normed_x)

        if return_medusa and self.medusa is not None:
            medusa_logits = self.medusa(normed_x)
            return logits, new_kv_caches, medusa_logits

        return logits, new_kv_caches

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimate_size_mb(self, bits: int = 32) -> float:
        return (self.count_parameters(trainable_only=False) * bits) / (8 * 1024 * 1024)

    def __repr__(self) -> str:
        num_params = self.count_parameters()
        return (
            f"DeepSpaceSLM(\n"
            f"  layers={self.config.num_layers}, "
            f"heads={self.config.num_heads}, "
            f"dim={self.config.hidden_dim}, "
            f"vocab={self.config.vocab_size}\n"
            f"  parameters={num_params:,}\n"
            f"  size: FP32={self.estimate_size_mb(32):.1f}MB, "
            f"INT8={self.estimate_size_mb(8):.1f}MB, "
            f"INT4={self.estimate_size_mb(4):.1f}MB\n"
            f")"
        )
