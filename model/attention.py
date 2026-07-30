import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        max_kv_size: Optional[int] = None,
        num_sink_tokens: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.max_kv_size = max_kv_size
        self.num_sink_tokens = num_sink_tokens

        self.q_dim = num_heads * head_dim
        self.kv_dim = self.num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden_dim, self.q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, hidden_dim, bias=False)

        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

            if self.max_kv_size is not None and k.shape[1] > self.max_kv_size:
                sink_k = k[:, :self.num_sink_tokens]
                recent_k = k[:, -(self.max_kv_size - self.num_sink_tokens):]
                k = torch.cat([sink_k, recent_k], dim=1)

                sink_v = v[:, :self.num_sink_tokens]
                recent_v = v[:, -(self.max_kv_size - self.num_sink_tokens):]
                v = torch.cat([sink_v, recent_v], dim=1)

        new_kv_cache = (k, v)

        if self.num_groups > 1:
            k = k.unsqueeze(3).expand(
                batch, k.shape[1], self.num_kv_heads, self.num_groups, self.head_dim
            ).reshape(batch, k.shape[1], self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(
                batch, v.shape[1], self.num_kv_heads, self.num_groups, self.head_dim
            ).reshape(batch, v.shape[1], self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is None:
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=x.device),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_weights = attn_weights + mask

        attn_weights = self.attn_dropout(F.softmax(attn_weights, dim=-1))
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.q_dim)

        return self.o_proj(attn_output), new_kv_cache


class DifferentialCausalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, lambda_init: float = 0.8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lambda_param = nn.Parameter(torch.tensor([lambda_init]))

        self.q1_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.q2_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k1_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k2_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q1 = self.q1_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.q2_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k2 = self.k2_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        if kv_cache is not None:
            cached_k1, cached_k2, cached_v = kv_cache
            k1 = torch.cat([cached_k1, k1], dim=1)
            k2 = torch.cat([cached_k2, k2], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        new_kv_cache = (k1, k2, v)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v = v.transpose(1, 2)

        w1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        w2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        if mask is None:
            kv_len = k1.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=x.device),
                diagonal=kv_len - seq_len + 1,
            ).unsqueeze(0).unsqueeze(0)
            w1 = w1 + causal_mask
            w2 = w2 + causal_mask
        else:
            w1 = w1 + mask
            w2 = w2 + mask

        a1 = F.softmax(w1, dim=-1)
        a2 = F.softmax(w2, dim=-1)

        diff_attn = a1 - self.lambda_param * a2
        out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out), new_kv_cache
