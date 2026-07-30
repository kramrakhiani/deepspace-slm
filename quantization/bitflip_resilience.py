import torch
import torch.nn as nn
import hashlib
import copy
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResilienceReport:
    bitflip_rate: float
    num_flips_injected: int
    total_weight_bits: int
    output_divergence: float
    max_logit_deviation: float
    prediction_match_rate: float
    layer_impacts: Dict[str, float]


def flip_random_bits(
    tensor: torch.Tensor,
    flip_rate: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    if seed is not None:
        torch.manual_seed(seed)

    flat = tensor.float().contiguous().view(-1)
    num_elements = flat.numel()
    total_bits = num_elements * 32
    num_flips = max(1, int(total_bits * flip_rate))

    byte_tensor = flat.clone()
    int_view = byte_tensor.view(torch.int32)
    element_indices = torch.randint(0, num_elements, (num_flips,))
    bit_positions = torch.randint(0, 32, (num_flips,))

    for i in range(num_flips):
        elem_idx = element_indices[i].item()
        bit_pos = bit_positions[i].item()
        int_view[elem_idx] ^= (1 << bit_pos)

    corrupted = byte_tensor.view(torch.float32)
    corrupted = torch.nan_to_num(corrupted, nan=0.0, posinf=1e6, neginf=-1e6)
    return corrupted.view_as(tensor).to(tensor.dtype), num_flips


def compute_hamming_parity(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.float().contiguous().view(-1).view(torch.int32)
    p0 = (flat & 0x0000FFFF)
    p1 = (flat & 0xFFFF0000) >> 16
    return (p0 ^ p1).to(torch.int16)


def verify_and_repair_hamming(
    tensor: torch.Tensor,
    golden_tensor: torch.Tensor,
    expected_parity: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    current_parity = compute_hamming_parity(tensor)
    mismatch = (current_parity != expected_parity)
    num_repaired = mismatch.sum().item()

    if num_repaired > 0:
        with torch.no_grad():
            tensor.copy_(golden_tensor)

    return tensor, num_repaired


class IdleMemoryScrubber:
    def __init__(self, model: nn.Module, check_interval_sec: float = 0.5):
        self.model = model
        self.check_interval_sec = check_interval_sec
        self.golden_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.expected_checksums = compute_weight_checksums(model)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.repairs_performed = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._scrub_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _scrub_loop(self):
        while self.running:
            time.sleep(self.check_interval_sec)
            self.scrub_once()

    def scrub_once(self) -> int:
        repaired = 0
        current_checksums = compute_weight_checksums(self.model)
        named_params = dict(self.model.named_parameters())
        named_bufs = dict(self.model.named_buffers())

        for name, expected in self.expected_checksums.items():
            if current_checksums.get(name) != expected:
                with torch.no_grad():
                    if name in named_params:
                        named_params[name].copy_(self.golden_state[name])
                    elif name in named_bufs:
                        named_bufs[name].copy_(self.golden_state[name])
                repaired += 1
        self.repairs_performed += repaired
        return repaired


def inject_bitflips(
    model: nn.Module,
    flip_rate: float = 1e-6,
    seed: Optional[int] = None,
    target_layers: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    corrupted = copy.deepcopy(model)
    flip_counts = {}
    layer_seed = seed if seed is not None else 0

    with torch.no_grad():
        for name, param in corrupted.named_parameters():
            if target_layers is not None and name not in target_layers:
                continue

            corrupted_weight, num_flips = flip_random_bits(
                param.data, flip_rate, seed=layer_seed
            )
            param.data.copy_(corrupted_weight)
            flip_counts[name] = num_flips
            layer_seed += 1

        for name, buf in corrupted.named_buffers():
            if target_layers is not None and name not in target_layers:
                continue
            if buf.dtype in (torch.float32, torch.float16, torch.int8, torch.uint8, torch.int32):
                corrupted_buf, num_flips = flip_random_bits(
                    buf.data, flip_rate, seed=layer_seed
                )
                buf.data.copy_(corrupted_buf)
                flip_counts[name] = num_flips
                layer_seed += 1

    return corrupted, flip_counts


def evaluate_resilience(
    model: nn.Module,
    test_input: torch.Tensor,
    flip_rates: List[float] = None,
    num_trials: int = 5,
) -> List[ResilienceReport]:
    if flip_rates is None:
        flip_rates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    model.eval()
    reports = []

    with torch.no_grad():
        clean_logits, _ = model(test_input)
        clean_predictions = clean_logits.argmax(dim=-1)

    total_weight_bits = sum(p.numel() * 32 for p in model.parameters())

    for rate in flip_rates:
        trial_divergences = []
        trial_max_devs = []
        trial_match_rates = []
        trial_flips = []
        layer_impacts_sum = {}

        for trial in range(num_trials):
            corrupted, flip_counts = inject_bitflips(
                model, flip_rate=rate, seed=trial * 1000
            )
            with torch.no_grad():
                corrupted_logits, _ = corrupted(test_input)
                corrupted_predictions = corrupted_logits.argmax(dim=-1)

            divergence = torch.norm(clean_logits - corrupted_logits).item()
            max_dev = (clean_logits - corrupted_logits).abs().max().item()
            match_rate = (clean_predictions == corrupted_predictions).float().mean().item()

            trial_divergences.append(divergence)
            trial_max_devs.append(max_dev)
            trial_match_rates.append(match_rate)
            trial_flips.append(sum(flip_counts.values()))

            for layer_name, count in flip_counts.items():
                if layer_name not in layer_impacts_sum:
                    layer_impacts_sum[layer_name] = 0.0
                layer_impacts_sum[layer_name] += count

            del corrupted

        report = ResilienceReport(
            bitflip_rate=rate,
            num_flips_injected=int(sum(trial_flips) / num_trials),
            total_weight_bits=total_weight_bits,
            output_divergence=sum(trial_divergences) / num_trials,
            max_logit_deviation=sum(trial_max_devs) / num_trials,
            prediction_match_rate=sum(trial_match_rates) / num_trials,
            layer_impacts={
                k: v / num_trials for k, v in layer_impacts_sum.items()
            },
        )
        reports.append(report)

    return reports


class TMRLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight_a = nn.Parameter(linear.weight.data.clone())
        self.weight_b = nn.Parameter(linear.weight.data.clone())
        self.weight_c = nn.Parameter(linear.weight.data.clone())

        if getattr(linear, "bias", None) is not None:
            self.bias_a = nn.Parameter(linear.bias.data.clone())
            self.bias_b = nn.Parameter(linear.bias.data.clone())
            self.bias_c = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias_a = self.bias_b = self.bias_c = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = nn.functional.linear(x, self.weight_a, self.bias_a)
        out_b = nn.functional.linear(x, self.weight_b, self.bias_b)
        out_c = nn.functional.linear(x, self.weight_c, self.bias_c)

        stacked = torch.stack([out_a, out_b, out_c], dim=0)
        return stacked.median(dim=0).values

    def sync_copies(self):
        with torch.no_grad():
            self.weight_b.copy_(self.weight_a)
            self.weight_c.copy_(self.weight_a)
            if self.bias_a is not None:
                self.bias_b.copy_(self.bias_a)
                self.bias_c.copy_(self.bias_a)


def apply_tmr(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> nn.Module:
    from quantization.ptq import QuantizedLinear

    protected = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            if layer_names is not None and name not in layer_names:
                continue

            tmr = TMRLinear(module)
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], tmr)
            protected += 1

    return model


def compute_weight_checksums(model: nn.Module) -> Dict[str, str]:
    checksums = {}
    for name, param in model.named_parameters():
        data_bytes = param.data.cpu().numpy().tobytes()
        checksum = hashlib.sha256(data_bytes).hexdigest()
        checksums[name] = checksum

    for name, buf in model.named_buffers():
        data_bytes = buf.cpu().numpy().tobytes()
        checksum = hashlib.sha256(data_bytes).hexdigest()
        checksums[name] = checksum

    return checksums


def verify_weight_integrity(
    model: nn.Module,
    expected_checksums: Dict[str, str],
) -> Dict[str, bool]:
    current_checksums = compute_weight_checksums(model)
    results = {}

    for name, expected in expected_checksums.items():
        if name in current_checksums:
            results[name] = current_checksums[name] == expected
        else:
            results[name] = False

    return results
