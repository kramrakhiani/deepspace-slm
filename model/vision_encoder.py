import torch
import torch.nn as nn


class CompactVisionEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.patch_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        x = self.patch_conv(images)
        batch, dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
