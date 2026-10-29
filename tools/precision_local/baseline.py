"""Run a bounded MPS baseline or a CPU/MPS single-step audit."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import torch
from model import TinyDecoder, token_loss


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tree_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree_cpu(item) for item in value)
    return value


def compare_trees(left: object, right: object, path: str = "root") -> list[str]:
    """Check every tensor, dtype and structural field, including optimizer/RNG state."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        equal = left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        return [] if equal else [path]
    if type(left) is not type(right):
        return [path + ":type"]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ":keys"]
        return [
            error for key in left for error in compare_trees(left[key], right[key], f"{path}.{key}")
        ]
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [path + ":length"]
        return [
            error
            for i, (a, b) in enumerate(zip(left, right))
            for error in compare_trees(a, b, f"{path}[{i}]")
        ]
    return [] if left == right else [path]


def configure(config: dict, device: str, seed: int) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU is required; CPU fallback is not allowed for this run")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("Disable MPS CPU fallback to preserve the meaning of the experiment")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("MPS fast math must be disabled for the baseline")
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def autocast(device: str, dtype: str):
    return torch.autocast(device, dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load_data(config: dict, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = urllib.request.urlopen(config["dataset"]["url"], timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != config["dataset"]["sha256"]:
            raise ValueError("Downloaded data SHA256 mismatch")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config["dataset"]["sha256"]:
        raise ValueError("Local data SHA256 mismatch")
    if len(payload) != config["dataset"]["bytes"]:
        raise ValueError("Data length mismatch")
    tokens = torch.frombuffer(bytearray(payload), dtype=torch.uint8).long()
    split = int(len(tokens) * config["dataset"]["train_fraction"])
    return tokens[:split], tokens[split:]


def batch(data: torch.Tensor, generator: torch.Generator, config: dict, device: str):
    length = config["model"]["sequence_length"]
    starts = torch.randint(
        len(data) - length, (config["training"]["batch_size"],), generator=generator
    )
    positions = starts[:, None] + torch.arange(length + 1)
    tokens = data[positions]
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device), starts.tolist()


def make_optimizer(model: TinyDecoder, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["epsilon"],
        weight_decay=training["weight_decay"],
        foreach=False,
        fused=False,
    )


def learning_rate(step: int, config: dict) -> float:
    cfg = config["training"]
    if step <= cfg["warmup_steps"]:
        ratio = step / cfg["warmup_steps"]
    else:
        fraction = (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
        ratio = (
            cfg["minimum_lr_ratio"]
            + (1 - cfg["minimum_lr_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2
        )
    return cfg["learning_rate"] * ratio


@torch.no_grad()
def evaluate(
    model: TinyDecoder, data: torch.Tensor, config: dict, device: str, dtype: str
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(99173)
    losses = []
    for _ in range(config["training"]["eval_batches"]):
        inputs, targets, _ = batch(data, generator, config, device)
        with autocast(device, dtype):
            losses.append(token_loss(model(inputs), targets).item())
    model.train()
    return sum(losses) / len(losses)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
    }


def snapshot(
    model, optimizer, generator, device, config_hash, dtype, seed, history, validation, indices
):
    return tree_cpu(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "device_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "device": device,
            "config_hash": config_hash,
            "dtype": dtype,
            "seed": seed,
            "step": len(history),
            "history": history,
            "validation": validation,
            "sample_indices": indices,
        }
    )


def restore(checkpoint: dict, model, optimizer, generator, device: str) -> None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["data_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if device == "mps":
        torch.mps.set_rng_state(checkpoint["device_rng"])


def train(args, config: dict, config_hash: str) -> None:
    configure(config, args.device, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    train_data, validation_data = load_data(config, args.data)
    model = TinyDecoder(config["model"]).to(args.device)
    initial_digest = hashlib.sha256()
    for parameter in model.parameters():
        initial_digest.update(parameter.detach().cpu().numpy().tobytes())
    optimizer = make_optimizer(model, config)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    history, validation, indices = [], [], []
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = {
            "config_hash": config_hash,
            "dtype": args.dtype,
            "seed": args.seed,
            "device": args.device,
        }
        if any(checkpoint[key] != value for key, value in expected.items()):
            raise ValueError("Checkpoint protocol, precision, seed or device mismatch")
        restore(checkpoint, model, optimizer, generator, args.device)
        start_step = checkpoint["step"]
        history, validation, indices = (
            checkpoint["history"],
            checkpoint["validation"],
            checkpoint["sample_indices"],
        )
    else:
        validation.append(
            {"step": 0, "loss": evaluate(model, validation_data, config, args.device, args.dtype)}
        )
    dtypes = set()
    handle = model.blocks[0].qkv.register_forward_hook(
        lambda module, inputs, output: dtypes.add(str(output.dtype))
    )
    synchronize(args.device)
    started = time.perf_counter()
    peak_live_bytes = 0
    model.train()
    for step in range(start_step + 1, training["steps"] + 1):
        inputs, targets, sample_indices = batch(train_data, generator, config, args.device)
        indices.append(sample_indices)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast(args.device, args.dtype):
            loss = token_loss(model(inputs), targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training["clip_grad_norm"], error_if_nonfinite=True
        )
        loss_value, norm_value = loss.item(), grad_norm.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        optimizer.step()
        history.append({"step": step, "loss": loss_value, "grad_norm": norm_value, "lr": lr})
        if args.device == "mps":
            peak_live_bytes = max(peak_live_bytes, torch.mps.current_allocated_memory())
        if step % training["eval_every"] == 0:
            val = evaluate(model, validation_data, config, args.device, args.dtype)
            validation.append({"step": step, "loss": val})
            print(
                json.dumps(
                    {
                        "run": args.out.name,
                        "step": step,
                        "train_loss": loss_value,
                        "validation_loss": val,
                    }
                ),
                flush=True,
            )
        if step == training["checkpoint_step"] or step == training["steps"]:
            state = snapshot(
                model,
                optimizer,
                generator,
                args.device,
                config_hash,
                args.dtype,
                args.seed,
                history,
                validation,
                indices,
            )
            torch.save(state, args.out / f"step-{step}.pt")
    synchronize(args.device)
    elapsed = time.perf_counter() - started
    handle.remove()
    expected_dtype = "torch.bfloat16" if args.dtype == "bf16" else "torch.float32"
    if dtypes != {expected_dtype}:
        raise AssertionError(f"Actual projection dtypes {dtypes} != {expected_dtype}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("Persistent model parameters must remain FP32")
    if any(
        value.dtype != torch.float32
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise AssertionError("Optimizer tensor states must remain FP32")
    summary = {
        "environment": environment(),
        "protocol_sha256": config_hash,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "actual_projection_dtypes": sorted(dtypes),
        "persistent_dtype": "torch.float32",
        "initial_parameters_sha256": initial_digest.hexdigest(),
        "sample_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "history": history,
        "validation": validation,
        "validation_fp32_eval": evaluate(model, validation_data, config, args.device, "fp32"),
        "start_step": start_step,
        "elapsed_seconds": elapsed,
        "executed_tokens": (training["steps"] - start_step)
        * training["batch_size"]
        * config["model"]["sequence_length"],
        "sampled_peak_live_mps_bytes": peak_live_bytes,
        "timing_scope": "Includes training, in-loop evaluation, checkpoint writes and diagnostic synchronizations; excludes setup/initial validation/final fp32 evaluation. Not a throughput benchmark.",
    }
    write_json(args.out / "result.json", summary)


def error_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict:
    value, reference = (
        value.detach().cpu().double().flatten(),
        reference.detach().cpu().double().flatten(),
    )
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Non-finite comparison input")
    difference = value - reference
    reference_norm, value_norm = (
        torch.linalg.vector_norm(reference).item(),
        torch.linalg.vector_norm(value).item(),
    )
    if reference_norm == 0:
        raise ValueError("Relative error needs a nonzero reference")
    return {
        "rel_l2": torch.linalg.vector_norm(difference).item() / reference_norm,
        "max_abs": difference.abs().max().item(),
        "cosine": (
            torch.dot(value, reference).item() / (value_norm * reference_norm)
            if value_norm
            else 0.0
        ),
    }


def one_step(config: dict, data: torch.Tensor, device: str, dtype: str) -> dict:
    configure(config, device, 17)
    model = TinyDecoder(config["model"]).to(device).eval()
    optimizer = make_optimizer(model, config)
    inputs, targets, _ = batch(data, torch.Generator().manual_seed(901), config, device)
    before = [p.detach().cpu().clone() for p in model.parameters()]
    with autocast(device, dtype):
        logits = model(inputs)
        loss = token_loss(logits, targets)
    loss.backward()
    gradients = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()])
    optimizer.step()
    updates = torch.cat(
        [(p.detach().cpu() - old).flatten() for p, old in zip(model.parameters(), before)]
    )
    return {
        "logits": logits.detach().cpu(),
        "grad": gradients,
        "update": updates,
        "loss": loss.item(),
    }


def audit(args, config: dict, config_hash: str) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    data, _ = load_data(config, args.data)
    reference = one_step(config, data, "cpu", "fp32")
    results = {}
    for dtype in ("fp32", "bf16"):
        candidate = one_step(config, data, "mps", dtype)
        metrics = {
            name: error_metrics(candidate[name], reference[name])
            for name in ("logits", "grad", "update")
        }
        thresholds = config["acceptance"][f"{dtype}_vs_cpu"]
        results[dtype] = {
            "loss": candidate["loss"],
            "cpu_loss": reference["loss"],
            "metrics": metrics,
            "checks": {
                f"{name}_rel_l2": metrics[name]["rel_l2"] <= thresholds[f"{name}_rel_l2"]
                for name in metrics
            },
        }
    write_json(
        args.out / "result.json",
        {"protocol_sha256": config_hash, "environment": environment(), "comparisons": results},
    )
    print(json.dumps(results), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument(
        "--data", type=Path, default=Path("local/precision/data/tinyshakespeare.txt")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config, config_hash = json.loads(raw), hashlib.sha256(raw).hexdigest()
    (train if args.mode == "train" else audit)(args, config, config_hash)


if __name__ == "__main__":
    main()
