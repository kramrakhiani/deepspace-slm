import torch
import torch.nn as nn
from typing import Optional, Tuple
from config import QuantizationConfig


class MinMaxObserver(nn.Module):
    def __init__(self, averaging_constant: float = 0.01):
        super().__init__()
        self.averaging_constant = averaging_constant
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_min = x.detach().min()
            x_max = x.detach().max()

            if self.num_batches_tracked == 0:
                self.min_val.copy_(x_min)
                self.max_val.copy_(x_max)
            else:
                self.min_val.copy_(
                    self.min_val + self.averaging_constant * (x_min - self.min_val)
                )
                self.max_val.copy_(
                    self.max_val + self.averaging_constant * (x_max - self.max_val)
                )
            self.num_batches_tracked += 1

        return x

    def calculate_qparams(self, bits: int, symmetric: bool = True) -> Tuple[float, int]:
        if symmetric:
            max_abs = torch.max(self.min_val.abs(), self.max_val.abs())
            qmax = (1 << (bits - 1)) - 1
            scale = max_abs / qmax
            zero_point = 0
        else:
            qmin = 0
            qmax = (1 << bits) - 1
            scale = (self.max_val - self.min_val) / (qmax - qmin)
            zero_point = int(torch.round(-self.min_val / scale))

        scale = max(scale.item(), 1e-8)
        return scale, zero_point


class FakeQuantize(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.enabled = True
        self.observer = MinMaxObserver()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        self.observer(x)
        if not self.training and self.observer.num_batches_tracked == 0:
            return x

        scale, zero_point = self.observer.calculate_qparams(
            self.bits, self.symmetric
        )
        return _fake_quantize_ste(x, scale, zero_point, self.bits, self.symmetric)


class _FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, scale: float, zero_point: int,
        bits: int, symmetric: bool
    ) -> torch.Tensor:
        if symmetric:
            qmax = (1 << (bits - 1)) - 1
            qmin = -qmax
        else:
            qmin = 0
            qmax = (1 << bits) - 1

        x_quant = torch.clamp(
            torch.round(x / scale) + zero_point, qmin, qmax
        )
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def _fake_quantize_ste(
    x: torch.Tensor, scale: float, zero_point: int,
    bits: int, symmetric: bool
) -> torch.Tensor:
    return _FakeQuantizeSTE.apply(x, scale, zero_point, bits, symmetric)


def prepare_qat(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_fq = FakeQuantize(
                bits=config.weight_bits,
                symmetric=config.symmetric,
                per_channel=config.per_channel,
            )
            module.register_module("weight_fake_quant", weight_fq)

            def make_qat_forward(mod, fq):
                def qat_forward(x):
                    q_weight = fq(mod.weight)
                    return nn.functional.linear(x, q_weight, mod.bias)
                return qat_forward

            module.forward = make_qat_forward(module, weight_fq)

    return model


def convert_qat(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight_fake_quant"):
                fq = module.weight_fake_quant
                if fq.observer.num_batches_tracked > 0:
                    scale, zp = fq.observer.calculate_qparams(
                        config.weight_bits, config.symmetric
                    )
                    qmax = (1 << (config.weight_bits - 1)) - 1
                    qmin = -qmax
                    w_quant = torch.clamp(
                        torch.round(module.weight.data / scale) + zp, qmin, qmax
                    )
                    module.weight.data = (w_quant - zp) * scale

                delattr(module, "weight_fake_quant")

    return model
