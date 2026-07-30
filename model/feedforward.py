import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(hidden))


class MoESwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int = 4, top_k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(hidden_dim, ffn_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        router_logits = self.router(x_flat)
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)

        out = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            top_k_indices = indices[:, k_idx]
            top_k_weights = weights[:, k_idx]

            for i in range(self.num_experts):
                mask = (top_k_indices == i)
                if mask.any():
                    expert_in = x_flat[mask]
                    expert_out = self.experts[i](expert_in)
                    out[mask] += top_k_weights[mask].unsqueeze(-1) * expert_out

        return out.view(batch, seq_len, dim)
