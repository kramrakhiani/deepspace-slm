import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    def output_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.embedding.weight)
