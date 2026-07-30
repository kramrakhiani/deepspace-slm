import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)

        logits_flat = logits.contiguous().view(-1, vocab_size)
        targets_flat = targets.contiguous().view(-1)

        mask = targets_flat != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll_loss = F.nll_loss(log_probs, targets_flat, reduction="none")
        smooth_loss = -log_probs.sum(dim=-1) / vocab_size

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return (loss * mask.float()).sum() / mask.float().sum()
