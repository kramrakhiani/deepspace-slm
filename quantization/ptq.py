import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from config import QuantizationConfig


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_int: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bits: int = 8,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.register_buffer("weight_int", weight_int)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_float = (self.weight_int.float() - self.zero_point) * self.scale
        return nn.functional.linear(x, weight_float, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, bias={self.bias is not None}"
        )


class AWQQuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        salient_indices: torch.Tensor,
        salient_weights: torch.Tensor,
        non_salient_int: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bits: int = 3,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.register_buffer("salient_indices", salient_indices)
        self.register_buffer("salient_weights", salient_weights)
        self.register_buffer("non_salient_int", non_salient_int)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_float = (self.non_salient_int.float() - self.zero_point) * self.scale
        weight_float[:, self.salient_indices] = self.salient_weights
        return nn.functional.linear(x, weight_float, self.bias)


def _quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = True,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax

        if per_channel:
            reduce_dims = [i for i in range(tensor.dim()) if i != channel_dim]
            max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_abs = tensor.abs().max()

        scale = max_abs / qmax
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.zeros_like(scale)
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    else:
        qmin = 0
        qmax = (1 << bits) - 1

        if per_channel:
            reduce_dims = [i for i in range(tensor.dim()) if i != channel_dim]
            t_min = tensor.amin(dim=reduce_dims, keepdim=True)
            t_max = tensor.amax(dim=reduce_dims, keepdim=True)
        else:
            t_min = tensor.min()
            t_max = tensor.max()

        scale = (t_max - t_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-t_min / scale)
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)

    return quantized.to(torch.int8), scale, zero_point


def quantize_awq(
    model: nn.Module,
    salient_ratio: float = 0.01,
    bits: int = 3,
) -> nn.Module:
    model.eval()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            out_features, in_features = weight.shape

            col_norms = weight.abs().mean(dim=0)
            num_salient = max(1, int(in_features * salient_ratio))
            salient_indices = torch.topk(col_norms, num_salient).indices

            salient_weights = weight[:, salient_indices].clone()
            non_salient_mask = torch.ones(in_features, dtype=torch.bool, device=weight.device)
            non_salient_mask[salient_indices] = False

            w_non_salient = weight.clone()

            q_int, scale, zp = _quantize_tensor(w_non_salient, bits=bits, symmetric=True, per_channel=True)

            awq_module = AWQQuantizedLinear(
                in_features=in_features,
                out_features=out_features,
                salient_indices=salient_indices,
                salient_weights=salient_weights,
                non_salient_int=q_int,
                scale=scale,
                zero_point=zp,
                bits=bits,
                bias=module.bias.data if module.bias is not None else None,
            )

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], awq_module)

    return model


def quantize_post_training(
    model: nn.Module,
    config: QuantizationConfig,
    calibration_data: Optional[List[torch.Tensor]] = None,
) -> nn.Module:
    model.eval()
    stats = {"original_params": 0, "quantized_params": 0, "layers_quantized": 0}

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            original_bytes = weight.numel() * 4
            stats["original_params"] += original_bytes

            w_int, scale, zero_point = _quantize_tensor(
                weight,
                bits=config.weight_bits,
                symmetric=config.symmetric,
                per_channel=config.per_channel,
            )

            quantized_bytes = weight.numel() * (config.weight_bits / 8) + scale.numel() * 4
            stats["quantized_params"] += quantized_bytes
            stats["layers_quantized"] += 1

            q_linear = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                weight_int=w_int,
                scale=scale,
                zero_point=zero_point,
                bits=config.weight_bits,
                bias=module.bias.data if module.bias is not None else None,
            )

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], q_linear)

    return model


def compute_model_size(model: nn.Module) -> Dict[str, float]:
    sizes = {"parameters_mb": 0, "buffers_mb": 0, "total_mb": 0}

    for name, param in model.named_parameters():
        sizes["parameters_mb"] += param.numel() * param.element_size() / (1024 * 1024)

    for name, buf in model.named_buffers():
        sizes["buffers_mb"] += buf.numel() * buf.element_size() / (1024 * 1024)

    sizes["total_mb"] = sizes["parameters_mb"] + sizes["buffers_mb"]
    return sizes
