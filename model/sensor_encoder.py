import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorTelemetryEncoder(nn.Module):
    def __init__(self, num_channels: int = 5, embed_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, telemetry: torch.Tensor) -> torch.Tensor:
        if telemetry.dim() == 2:
            telemetry = telemetry.unsqueeze(0)
        x = telemetry.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        return self.proj(x)
