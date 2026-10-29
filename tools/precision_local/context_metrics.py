"""Constant-token context partitioning and gradient comparison primitives."""

import hashlib
import math

import torch
from baseline import autocast
from model import token_loss


def split_window(window: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve every next-byte target when changing independent context boundaries."""
    if window.ndim != 1 or context <= 0 or (window.numel() - 1) % context:
        raise ValueError("A one-dimensional span must contain N*context+1 tokens")
    if window.numel() <= 1:
        raise ValueError("Window must contain input/target pairs")
    return window[:-1].reshape(-1, context), window[1:].reshape(-1, context)


def tensor_map_digest(tensors: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """Use FP64 reductions on CPU; explicitly represent undefined zero-reference ratios."""
    if candidate.shape != reference.shape:
        raise ValueError("Comparison shapes differ")
    candidate, reference = (
        value.detach().cpu().double().flatten() for value in (candidate, reference)
    )
    if not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Non-finite gradient comparison")
    delta = candidate - reference
    ref_norm, candidate_norm, delta_norm = (
        torch.linalg.vector_norm(value).item() for value in (reference, candidate, delta)
    )
    dot = torch.dot(candidate, reference).item()
    both_zero = ref_norm == 0 and candidate_norm == 0
    cosine = (
        max(-1.0, min(1.0, dot / (ref_norm * candidate_norm)))
        if ref_norm and candidate_norm
        else (1.0 if both_zero else None)
    )
    flips = ((candidate * reference) < 0).sum().item()
    return {
        "reference_norm": ref_norm,
        "candidate_norm": candidate_norm,
        "error_norm": delta_norm,
        "rel_l2": delta_norm / ref_norm if ref_norm else (0.0 if both_zero else None),
        "cosine": cosine,
        "max_abs": delta.abs().max().item(),
        "sign_flips": flips,
        "sign_flip_fraction": flips / reference.numel(),
        "elements": reference.numel(),
    }


def compare_gradients(candidate: dict, reference: dict) -> dict:
    if candidate.keys() != reference.keys():
        raise ValueError("Gradient parameter names differ")
    per_parameter = {name: tensor_metrics(candidate[name], reference[name]) for name in reference}
    global_metrics = tensor_metrics(
        torch.cat([candidate[name].flatten() for name in reference]),
        torch.cat([value.flatten() for value in reference.values()]),
    )
    return {"global": global_metrics, "per_parameter": per_parameter}


def sampled_memory(device: str) -> dict:
    if device != "mps":
        return {"live_bytes": 0, "driver_bytes": 0}
    return {
        "live_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
    }


def accumulate_gradients(model, window: torch.Tensor, context: int, dtype: str, device: str):
    """Return pre-clipping gradients after averaging over the same total token count."""
    model.zero_grad(set_to_none=True)
    inputs, targets = split_window(window, context)
    count = inputs.shape[0]
    average_loss = 0.0
    memory = sampled_memory(device)
    for x, y in zip(inputs, targets):
        with autocast(device, dtype):
            loss = token_loss(model(x[None].to(device)), y[None].to(device)) / count
        current = sampled_memory(device)
        memory = {key: max(memory[key], current[key]) for key in memory}
        loss.backward()
        average_loss += loss.item()
        current = sampled_memory(device)
        memory = {key: max(memory[key], current[key]) for key in memory}
    if not math.isfinite(average_loss):
        raise FloatingPointError("Non-finite training loss")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise AssertionError(f"Missing or non-FP32 parameter gradient: {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    return {"loss": average_loss, "gradients": gradients, "sampled_memory": memory}


@torch.no_grad()
def evaluate_windows(model, windows: list, context: int, dtype: str, device: str) -> float:
    model.eval()
    total = 0.0
    for window in windows:
        inputs, targets = split_window(window, context)
        for x, y in zip(inputs, targets):
            with autocast(device, dtype):
                total += (
                    token_loss(model(x[None].to(device)), y[None].to(device)).item()
                    / inputs.shape[0]
                )
    model.train()
    value = total / len(windows)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation loss")
    return value
