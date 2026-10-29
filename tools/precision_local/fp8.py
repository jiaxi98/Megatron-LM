"""Original FP8 numerical emulation: FP32 storage/compute, never native FP8 GEMM.

E4M3FN/E5M2 finite encodings follow https://arxiv.org/abs/2209.05433.
The current power-of-two scaling policy is specified in fp8_protocol.json.
CPU native float8 casts are used only by independent tests, not this implementation.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class FloatFormat:
    mantissa_bits: int
    minimum_exponent: int
    maximum: float


FORMATS = {"e4m3fn": FloatFormat(3, -6, 448.0), "e5m2": FloatFormat(2, -14, 57344.0)}


def round_finite(value: torch.Tensor, format_name: str) -> torch.Tensor:
    """Saturating round-to-nearest-even, including subnormals and signed zero."""
    if value.dtype != torch.float32 or value.numel() == 0:
        raise ValueError("Quantization requires a nonempty FP32 tensor")
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("Nonfinite FP8 input")
    return _round_finite_unchecked(value, format_name)


def _round_finite_unchecked(value, format_name):
    fmt = FORMATS[format_name]
    magnitude = value.detach().abs().clamp(max=fmt.maximum).contiguous()
    # Exact FP32 exponent extraction avoids log2 rounding at binade boundaries.
    exponent = ((magnitude.view(torch.int32) >> 23) & 255) - 127
    quantum_exponent = (exponent - fmt.mantissa_bits).clamp(
        min=fmt.minimum_exponent - fmt.mantissa_bits
    )
    quantum = torch.pow(2.0, quantum_exponent.float())
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=fmt.maximum)
    return torch.copysign(rounded, value)


def current_quantize(value: torch.Tensor, format_name: str, collect_stats: bool = True):
    """Return dequantized FP32 values and diagnostics; no autograd through scale."""
    if value.dtype != torch.float32 or value.numel() == 0:
        raise ValueError("Quantization requires a nonempty FP32 tensor")
    value = value.detach()
    amax = value.abs().max().item()
    if not math.isfinite(amax):
        raise FloatingPointError("Nonfinite amax")
    fmt = FORMATS[format_name]
    if amax:
        mantissa, exponent = math.frexp(amax / fmt.maximum)
        scale = math.ldexp(1.0, exponent - (mantissa == 0.5))
    else:
        scale = 1.0
    if not torch.finfo(torch.float32).tiny <= scale <= torch.finfo(torch.float32).max:
        raise FloatingPointError("Current scale is outside normal FP32 range")
    scaled = value / scale
    rounded = _round_finite_unchecked(scaled, format_name)
    output = rounded * scale
    if (
        scale * fmt.maximum > torch.finfo(torch.float32).max
        and not torch.isfinite(output).all().item()
    ):
        raise FloatingPointError("FP8 dequantization overflow")
    stats = {"format": format_name, "elements": value.numel(), "amax": amax, "dequant_scale": scale}
    if collect_stats:
        nonzero = value != 0
        stats.update(
            saturated=(scaled.abs() > fmt.maximum).sum().item(),
            underflowed=(nonzero & (rounded == 0)).sum().item(),
            nonzero_inputs=nonzero.sum().item(),
        )
    return output, stats


class QuantizationStats:
    """Optional per-module/operand diagnostics, aggregated without retaining tensors."""

    def __init__(self):
        self.enabled = False
        self.rows = {}

    def record(self, name, values):
        if not self.enabled:
            return
        row = self.rows.setdefault(
            name,
            dict(
                format=values["format"],
                calls=0,
                elements=0,
                saturated=0,
                underflowed=0,
                nonzero_inputs=0,
                amax_max=0.0,
                scale_min=float("inf"),
                scale_max=0.0,
            ),
        )
        row["calls"] += 1
        for key in ("elements", "saturated", "underflowed", "nonzero_inputs"):
            row[key] += values[key]
        row["amax_max"] = max(row["amax_max"], values["amax"])
        row["scale_min"] = min(row["scale_min"], values["dequant_scale"])
        row["scale_max"] = max(row["scale_max"], values["dequant_scale"])


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, recorder, name):
        with torch.autocast(inputs.device.type, enabled=False):
            x, xstats = current_quantize(inputs.float(), "e4m3fn", recorder.enabled)
            w, wstats = current_quantize(weight.float(), "e4m3fn", recorder.enabled)
            recorder.record(name + "/input", xstats)
            recorder.record(name + "/weight", wstats)
            ctx.save_for_backward(x, w)
            ctx.recorder, ctx.name, ctx.has_bias = recorder, name, bias is not None
            result = x @ w.T
            return result if bias is None else result + bias.float()

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with torch.autocast(gradient.device.type, enabled=False):
            x, w = ctx.saved_tensors
            dy, stats = current_quantize(gradient.float(), "e5m2", ctx.recorder.enabled)
            ctx.recorder.record(ctx.name + "/output_gradient", stats)
            dx = dy @ w if ctx.needs_input_grad[0] else None
            dw = dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
            db = gradient.float().reshape(-1, gradient.shape[-1]).sum(0) if ctx.has_bias else None
        return dx, dw, db, None, None


class EmulatedFP8Linear(nn.Module):
    """Same parameter objects/state keys as a Linear; quantized training surrogate."""

    def __init__(self, source: nn.Linear, recorder: QuantizationStats, name: str):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias
        self.recorder, self.name = recorder, name
        self.enabled = True

    def forward(self, inputs):
        if not self.enabled:
            return nn.functional.linear(inputs, self.weight, self.bias)
        return _FP8Linear.apply(inputs, self.weight, self.bias, self.recorder, self.name)


def install_fp8(model: nn.Module):
    """Replace decoder-block projections only; exclude head, embedding and attention."""
    recorder = QuantizationStats()
    names = []
    for index, block in enumerate(model.blocks):
        for name in ("qkv", "projection", "gate", "up", "down"):
            source = getattr(block, name)
            if not isinstance(source, nn.Linear):
                raise TypeError("FP8 installation requires original Linear modules")
            path = f"blocks.{index}.{name}"
            setattr(block, name, EmulatedFP8Linear(source, recorder, path))
            names.append(path)
    return recorder, names


@contextmanager
def disable_fp8(model):
    modules = [module for module in model.modules() if isinstance(module, EmulatedFP8Linear)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled
