import torch
import torch.nn as nn
import json
import struct
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any

from config import ModelConfig, QuantizationConfig
from quantization.ptq import QuantizedLinear, compute_model_size

MAGIC = b"DSLM"
FORMAT_VERSION = 1


def pack_int4_to_uint8(tensor: torch.Tensor) -> bytes:
    flat = tensor.cpu().contiguous().view(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])

    unsigned = (flat + 8).to(torch.uint8) & 0x0F
    even = unsigned[0::2]
    odd = unsigned[1::2]
    packed = even | (odd << 4)
    return packed.numpy().tobytes()


def unpack_uint8_to_int4(packed_bytes: bytes, num_elements: int) -> torch.Tensor:
    packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8)
    even = (packed & 0x0F).to(torch.int8) - 8
    odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    interleaved = torch.zeros(len(packed) * 2, dtype=torch.int8)
    interleaved[0::2] = even
    interleaved[1::2] = odd

    return interleaved[:num_elements]


def export_binary(
    model: nn.Module,
    config: ModelConfig,
    output_path: str,
    quant_config: Optional[QuantizationConfig] = None,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weight_bits = quant_config.weight_bits if quant_config else 32

    all_params = []
    param_manifest = {}
    offset = 0

    for name, param in model.named_parameters():
        data = param.data.cpu().contiguous()
        param_bytes = data.numpy().tobytes()
        all_params.append(param_bytes)
        param_manifest[name] = {
            "offset": offset,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_bytes": len(param_bytes),
        }
        offset += len(param_bytes)

    for name, buf in model.named_buffers():
        data = buf.cpu().contiguous()
        if weight_bits == 4 and "weight_int" in name:
            buf_bytes = pack_int4_to_uint8(data)
        else:
            buf_bytes = data.numpy().tobytes()

        all_params.append(buf_bytes)
        param_manifest[name] = {
            "offset": offset,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_bytes": len(buf_bytes),
            "packed_int4": (weight_bits == 4 and "weight_int" in name),
        }
        offset += len(buf_bytes)

    weight_data = b"".join(all_params)
    checksum = hashlib.sha256(weight_data).digest()
    total_params = sum(p.numel() for p in model.parameters())

    header_size = 96
    data_offset = header_size

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", config.num_layers))
        f.write(struct.pack("<I", config.hidden_dim))
        f.write(struct.pack("<I", config.num_heads))
        f.write(struct.pack("<I", config.vocab_size))
        f.write(struct.pack("<I", config.max_seq_len))
        f.write(struct.pack("<I", weight_bits))
        f.write(struct.pack("<Q", total_params))
        f.write(struct.pack("<Q", data_offset))
        f.write(checksum)

        current_pos = f.tell()
        if current_pos < header_size:
            f.write(b"\x00" * (header_size - current_pos))

        f.write(weight_data)

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "format": "DSLM",
        "version": FORMAT_VERSION,
        "model_config": {
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
        },
        "quantization": {
            "weight_bits": weight_bits,
            "symmetric": quant_config.symmetric if quant_config else None,
            "per_channel": quant_config.per_channel if quant_config else None,
        },
        "checksum_sha256": hashlib.sha256(weight_data).hexdigest(),
        "total_params": total_params,
        "file_size_bytes": header_size + len(weight_data),
        "file_size_mb": (header_size + len(weight_data)) / (1024 * 1024),
        "parameters": param_manifest,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_export(path: str) -> bool:
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            return False

        version = struct.unpack("<I", f.read(4))[0]
        if version != FORMAT_VERSION:
            return False

        f.read(24)
        total_params = struct.unpack("<Q", f.read(8))[0]
        data_offset = struct.unpack("<Q", f.read(8))[0]
        stored_checksum = f.read(32)

        f.seek(data_offset)
        weight_data = f.read()
        actual_checksum = hashlib.sha256(weight_data).digest()

        if stored_checksum != actual_checksum:
            return False

    return True
