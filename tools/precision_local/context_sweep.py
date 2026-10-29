"""Train paired MPS trajectories and probe gradient precision up to 4K context."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from baseline import (
    configure,
    environment,
    learning_rate,
    load_data,
    make_optimizer,
    synchronize,
    tree_cpu,
    write_json,
)
from context_metrics import (
    accumulate_gradients,
    compare_gradients,
    evaluate_windows,
    tensor_map_digest,
)
from model import TinyDecoder
from run_suite import file_sha256, launch


def load_protocol(path: Path) -> tuple[dict, dict]:
    protocol = json.loads(path.read_text())
    base_path = path.parent / protocol["base_protocol"]
    if file_sha256(base_path) != protocol["base_protocol_sha256"]:
        raise ValueError("Base protocol hash changed")
    if protocol["microbatch_sequences"] != 1 or protocol["dropout"] != 0:
        raise ValueError("This study requires one-sequence microbatches and disabled dropout")
    for context in protocol["contexts"]:
        if (
            context <= 0
            or protocol["tokens_per_update"] % context
            or protocol["validation_tokens_per_window"] % context
        ):
            raise ValueError("Context must divide both train and validation token budgets")
    if not all(1 <= step <= protocol["steps"] for step in protocol["matched_weight_probe_steps"]):
        raise ValueError("Probe step outside training range")
    if not set(protocol["save_raw_gradient_steps"]).issubset(
        protocol["matched_weight_probe_steps"]
    ):
        raise ValueError("Raw gradient captures require a matched-weight probe")
    return protocol, json.loads(base_path.read_text())


def window_starts(data: torch.Tensor, tokens: int, count: int, seed: int) -> list[int]:
    return torch.randint(
        len(data) - tokens, (count,), generator=torch.Generator().manual_seed(seed)
    ).tolist()


def matched_probe(reference_model, probe_model, window, context, device):
    probe_model.load_state_dict(reference_model.state_dict())
    reference_hash = tensor_map_digest(dict(reference_model.named_parameters()))
    if tensor_map_digest(dict(probe_model.named_parameters())) != reference_hash:
        raise AssertionError("Matched probe does not have identical weights")
    result = accumulate_gradients(probe_model, window, context, "bf16", device)
    result["weight_sha256"] = reference_hash
    return result


def paired_run(args, protocol: dict, base: dict) -> None:
    if args.context not in protocol["contexts"] or args.seed not in protocol["seeds"]:
        raise ValueError("Run is outside the declared context/seed matrix")
    config = copy.deepcopy(base)
    config["model"].update(sequence_length=args.context, dropout=protocol["dropout"])
    config["training"]["steps"] = protocol["steps"]
    configure(config, "mps", args.seed)
    torch.mps.set_per_process_memory_fraction(protocol["mps_memory_fraction"])
    args.out.mkdir(parents=True, exist_ok=False)
    train_data, validation_data = load_data(
        config, Path("local/precision/data/tinyshakespeare.txt")
    )
    tokens = protocol["tokens_per_update"]
    starts = window_starts(train_data, tokens, protocol["steps"], args.seed + 1000)
    val_tokens = protocol["validation_tokens_per_window"]
    val_starts = window_starts(validation_data, val_tokens, protocol["validation_windows"], 99173)
    val_windows = [validation_data[start : start + val_tokens + 1] for start in val_starts]
    models = {
        "fp32": TinyDecoder(config["model"]).to("mps"),
        "bf16": TinyDecoder(config["model"]).to("mps"),
    }
    models["bf16"].load_state_dict(models["fp32"].state_dict())
    probe = TinyDecoder(config["model"]).to("mps")
    initial_hash = tensor_map_digest(dict(models["fp32"].named_parameters()))
    if initial_hash != tensor_map_digest(dict(models["bf16"].named_parameters())):
        raise AssertionError("Paired initializations differ")
    optimizers = {dtype: make_optimizer(model, config) for dtype, model in models.items()}
    observed_dtypes = {name: set() for name in ("fp32", "bf16", "probe")}
    handles = []
    for name, model in {**models, "probe": probe}.items():
        handles.append(
            model.blocks[0].qkv.register_forward_hook(
                lambda module, inputs, output, name=name: observed_dtypes[name].add(
                    str(output.dtype)
                )
            )
        )
    metadata = {
        "environment": json.loads(json.dumps(environment())),
        "context": args.context,
        "seed": args.seed,
        "base_protocol_sha256": protocol["base_protocol_sha256"],
        "context_protocol_sha256": file_sha256(args.protocol),
        "parameter_count": sum(p.numel() for p in models["fp32"].parameters()),
        "initial_parameters_sha256": initial_hash,
        "train_window_starts": starts,
        "validation_window_starts": val_starts,
        "train_window_starts_sha256": hashlib.sha256(json.dumps(starts).encode()).hexdigest(),
        "tokens_per_update": tokens,
        "sequences_per_update": tokens // args.context,
        "processed_tokens_per_precision": tokens * protocol["steps"],
        "mps_recommended_bytes": torch.mps.recommended_max_memory(),
        "mps_memory_fraction": protocol["mps_memory_fraction"],
        "scope": "Same-weight arithmetic probes and separate evolving trajectories on MPS; byte contexts only.",
    }
    write_json(args.out / "metadata.json", metadata)
    history, validation, raw_files = [], [], []
    peak = {"live_bytes": 0, "driver_bytes": 0}

    def evaluate(step):
        validation.append(
            {
                "step": step,
                "processed_tokens": step * tokens,
                **{
                    dtype: evaluate_windows(model, val_windows, args.context, dtype, "mps")
                    for dtype, model in models.items()
                },
            }
        )

    evaluate(0)
    synchronize("mps")
    started = time.perf_counter()
    with (args.out / "gradients.jsonl").open("w") as log:
        for step in range(1, protocol["steps"] + 1):
            start = starts[step - 1]
            window = train_data[start : start + tokens + 1]
            observations = {
                dtype: accumulate_gradients(model, window, args.context, dtype, "mps")
                for dtype, model in models.items()
            }
            reference, candidate = observations["fp32"], observations["bf16"]
            row = {
                "step": step,
                "processed_tokens_before_update": (step - 1) * tokens,
                "fp32_loss": reference["loss"],
                "bf16_loss": candidate["loss"],
                "trajectory": compare_gradients(candidate["gradients"], reference["gradients"]),
            }
            raw_probe = None
            if step in protocol["matched_weight_probe_steps"]:
                raw_probe = matched_probe(models["fp32"], probe, window, args.context, "mps")
                row["matched_weights"] = compare_gradients(
                    raw_probe["gradients"], reference["gradients"]
                )
                row["matched_weight_sha256"] = raw_probe["weight_sha256"]
                row["matched_weight_bf16_loss"] = raw_probe["loss"]
                observations["probe"] = raw_probe
            for observation in observations.values():
                peak = {key: max(peak[key], observation["sampled_memory"][key]) for key in peak}
            if (
                args.seed == protocol["save_raw_gradient_seed"]
                and step in protocol["save_raw_gradient_steps"]
            ):
                raw_name = f"gradients-step-{step}.pt"
                torch.save(
                    tree_cpu(
                        {
                            "context": args.context,
                            "step": step,
                            "window": window,
                            "fp32_weights": models["fp32"].state_dict(),
                            "bf16_trajectory_weights": models["bf16"].state_dict(),
                            "fp32_gradients": reference["gradients"],
                            "bf16_trajectory_gradients": candidate["gradients"],
                            "bf16_matched_gradients": raw_probe["gradients"] if raw_probe else None,
                        }
                    ),
                    args.out / raw_name,
                )
                raw_files.append(raw_name)
            lr = learning_rate(step, config)
            row["learning_rate"] = lr
            for dtype, model in models.items():
                row[f"{dtype}_preclip_norm"] = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["clip_grad_norm"],
                    error_if_nonfinite=True,
                ).item()
                for group in optimizers[dtype].param_groups:
                    group["lr"] = lr
                optimizers[dtype].step()
            history.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if step in protocol["eval_steps"]:
                evaluate(step)
            if step in protocol["matched_weight_probe_steps"]:
                print(
                    json.dumps(
                        {
                            "context": args.context,
                            "seed": args.seed,
                            "step": step,
                            "trajectory_grad_rel_l2": row["trajectory"]["global"]["rel_l2"],
                            "matched_grad_rel_l2": row["matched_weights"]["global"]["rel_l2"],
                        }
                    ),
                    flush=True,
                )
    synchronize("mps")
    elapsed = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    expected = {"fp32": {"torch.float32"}, "bf16": {"torch.bfloat16"}, "probe": {"torch.bfloat16"}}
    if observed_dtypes != expected:
        raise AssertionError(f"Unexpected projection dtypes: {observed_dtypes}")
    for dtype, model in models.items():
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise AssertionError("Non-FP32 persistent parameter")
        if any(
            t.dtype != torch.float32
            for state in optimizers[dtype].state.values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        ):
            raise AssertionError("Non-FP32 optimizer state")
    final_fp32_eval = {
        dtype: evaluate_windows(model, val_windows, args.context, "fp32", "mps")
        for dtype, model in models.items()
    }
    checkpoint = tree_cpu(
        {
            "models": {dtype: model.state_dict() for dtype, model in models.items()},
            "optimizers": {
                dtype: optimizer.state_dict() for dtype, optimizer in optimizers.items()
            },
            "step": protocol["steps"],
            "metadata": metadata,
            "cpu_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state(),
        }
    )
    torch.save(checkpoint, args.out / "final.pt")
    result = {
        **metadata,
        "history": history,
        "validation": validation,
        "final_common_fp32_validation": final_fp32_eval,
        "observed_projection_dtypes": {
            name: sorted(values) for name, values in observed_dtypes.items()
        },
        "sampled_memory": peak,
        "instrumented_training_seconds": elapsed,
        "timing_scope": "Includes gradient transfers/metrics, probes, evaluations and selected raw writes; excludes setup/initial validation/final common evaluation and final checkpoint. Not a performance benchmark.",
        "artifacts": [
            {
                "file": name,
                "sha256": file_sha256(args.out / name),
                "bytes": (args.out / name).stat().st_size,
            }
            for name in [*raw_files, "final.pt", "gradients.jsonl"]
        ],
    }
    write_json(args.out / "result.json", result)
    print(
        json.dumps(
            {
                "completed_context": args.context,
                "seed": args.seed,
                "final_validation": final_fp32_eval,
            }
        ),
        flush=True,
    )


def suite(args, protocol):
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    manifest = {
        "environment": environment(),
        "protocol_sha256": file_sha256(args.protocol),
        "dependencies_sha256": file_sha256(args.protocol.with_name("requirements.lock")),
        "source_files": {
            path.name: file_sha256(path) for path in args.protocol.parent.glob("*.py")
        },
        "commands": [],
    }
    for context in protocol["contexts"]:
        for seed in protocol["seeds"]:
            label = f"context-{context}-seed-{seed}"
            command = [
                sys.executable,
                str(Path(__file__)),
                "run",
                "--protocol",
                str(args.protocol),
                "--context",
                str(context),
                "--seed",
                str(seed),
                "--out",
                str(args.out / label),
            ]
            manifest["commands"].append({"label": label, "argv": command})
            write_json(args.out / "manifest.json", manifest)
            launch(command, args.out, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "suite"))
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("context_protocol.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol, base = load_protocol(args.protocol)
    if args.mode == "run":
        paired_run(args, protocol, base)
    else:
        suite(args, protocol)


if __name__ == "__main__":
    main()
