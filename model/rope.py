import torch
import torch.nn as nn


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> torch.Tensor:
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    freq_indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (freq_indices / head_dim))
    positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(
    x: torch.Tensor,
    rope_freqs: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = x.shape
    x_pairs = x.float().reshape(batch, seq_len, num_heads, -1, 2)
    x_complex = torch.view_as_complex(x_pairs)

    freqs = rope_freqs[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs

    x_out = torch.view_as_real(x_rotated).reshape(batch, seq_len, num_heads, head_dim)
    return x_out.type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        freqs = precompute_rope_frequencies(head_dim, max_seq_len, theta)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        return apply_rope(x, self.freqs, start_pos)
