"""Constant-token FP32/BF16/FP8-emulated training with probes and exact restart."""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import torch
from baseline import (
    compare_trees,
    configure,
    environment,
    learning_rate,
    load_data,
    make_optimizer,
    tree_cpu,
    write_json,
)
from context_metrics import (
    accumulate_gradients,
    compare_gradients,
    evaluate_windows,
    tensor_map_digest,
)
from context_sweep import window_starts
from fp8 import disable_fp8, install_fp8
from model import TinyDecoder
from run_suite import file_sha256, launch


def load_protocol(path):
    protocol = json.loads(path.read_text())
    base_path = path.parent / protocol["base_protocol"]
    if file_sha256(base_path) != protocol["base_protocol_sha256"]:
        raise ValueError("Base protocol changed")
    cfg = protocol["training"]
    if (
        protocol["recipe"]["forward_format"] != "e4m3fn"
        or protocol["recipe"]["gradient_format"] != "e5m2"
    ):
        raise ValueError("This implementation supports only the declared hybrid recipe")
    if cfg["microbatch_sequences"] != 1 or cfg["dropout"] != 0:
        raise ValueError("Only the predeclared accumulation/dropout policy is supported")
    if cfg["arms"] != ["fp32", "bf16", "fp8_emulated"]:
        raise ValueError("Unexpected precision arms")
    for length in cfg["contexts"]:
        if (
            length <= 0
            or cfg["tokens_per_update"] % length
            or cfg["validation_tokens_per_window"] % length
        ):
            raise ValueError("Context must partition both token budgets exactly")
    if not 0 < cfg["resume_step"] < cfg["steps"]:
        raise ValueError("Invalid checkpoint step")
    return protocol, json.loads(base_path.read_text())


def state_is_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(state_is_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(state_is_finite(v) for v in value)
    return True


def validate_resume(saved, metadata):
    for key in (
        "protocol_sha256",
        "context",
        "seed",
        "initial_weights_sha256",
        "train_starts",
        "validation_starts",
    ):
        if saved["metadata"][key] != metadata[key]:
            raise ValueError(f"Resume mismatch: {key}")
    if (
        saved["metadata"]["environment"]["source_commit"]
        != metadata["environment"]["source_commit"]
    ):
        raise ValueError("Resume source revision changed")


def snapshot(models, optimizers, metadata, history, validation):
    result = tree_cpu(
        {
            "models": {k: model.state_dict() for k, model in models.items()},
            "optimizers": {k: optimizer.state_dict() for k, optimizer in optimizers.items()},
            "metadata": metadata,
            "step": len(history),
            "history": history,
            "validation": validation,
            "cpu_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state(),
        }
    )
    if not state_is_finite(result["models"]) or not state_is_finite(result["optimizers"]):
        raise FloatingPointError("Nonfinite model or optimizer state")
    for model in models.values():
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise AssertionError("Non-FP32 persistent parameter")
    for optimizer in optimizers.values():
        if any(
            v.dtype != torch.float32
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ):
            raise AssertionError("Non-FP32 optimizer state")
    return result


def run(args, protocol, base):
    cfg = protocol["training"]
    if args.context not in cfg["contexts"] or args.seed not in cfg["seeds"]:
        raise ValueError("Run outside the declared matrix")
    configure(base, "mps", args.seed)
    torch.mps.set_per_process_memory_fraction(cfg["mps_memory_fraction"])
    args.out.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(base)
    config["model"].update(sequence_length=args.context, dropout=0.0)
    config["training"]["steps"] = cfg["steps"]
    train, valid = load_data(base, Path("local/precision/data/tinyshakespeare.txt"))
    starts = window_starts(train, cfg["tokens_per_update"], cfg["steps"], args.seed + 1000)
    val_starts = window_starts(
        valid, cfg["validation_tokens_per_window"], cfg["validation_windows"], 99173
    )
    val_windows = [
        valid[start : start + cfg["validation_tokens_per_window"] + 1] for start in val_starts
    ]
    models = {name: TinyDecoder(config["model"]).to("mps") for name in cfg["arms"]}
    initial = tree_cpu(models["fp32"].state_dict())
    for model in models.values():
        model.load_state_dict(initial)
    recorder, modules = install_fp8(models["fp8_emulated"])
    optimizers = {name: make_optimizer(model, config) for name, model in models.items()}
    probe = TinyDecoder(config["model"]).to("mps")
    probe_recorder, _ = install_fp8(probe)
    probe_optimizer = make_optimizer(probe, config)
    metadata = {
        "environment": json.loads(json.dumps(environment())),
        "protocol_sha256": file_sha256(args.protocol),
        "context": args.context,
        "seed": args.seed,
        "initial_weights_sha256": {
            k: tensor_map_digest(dict(m.named_parameters())) for k, m in models.items()
        },
        "train_starts": starts,
        "validation_starts": val_starts,
        "tokens_per_update": cfg["tokens_per_update"],
        "processed_tokens_per_arm": cfg["steps"] * cfg["tokens_per_update"],
        "parameter_count": sum(p.numel() for p in models["fp32"].parameters()),
        "fp8_modules": modules,
        "scope": protocol["recipe"],
    }
    observed = {name: set() for name in models}
    handles = [
        model.blocks[0].qkv.register_forward_hook(
            lambda module, inputs, output, name=name: observed[name].add(str(output.dtype))
        )
        for name, model in models.items()
    ]
    history, validation, start_step = [], [], 0
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location="cpu")
        validate_resume(saved, metadata)
        if saved["step"] != cfg["resume_step"]:
            raise ValueError("Resume must use the declared checkpoint step")
        for name, model in models.items():
            model.load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        history, validation, start_step = saved["history"], saved["validation"], saved["step"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])

    def evaluate(step):
        recorder.enabled = False
        values = {
            name: evaluate_windows(
                model, val_windows, args.context, "bf16" if name == "bf16" else "fp32", "mps"
            )
            for name, model in models.items()
        }
        validation.append({"step": step, **values})

    if start_step == 0:
        evaluate(0)
    write_json(args.out / "metadata.json", metadata)
    for step in range(start_step + 1, cfg["steps"] + 1):
        capture = step in cfg["matched_probe_steps"]
        recorder.enabled, recorder.rows = capture, {}
        start = starts[step - 1]
        window = train[start : start + cfg["tokens_per_update"] + 1]
        observations = {
            name: accumulate_gradients(
                model, window, args.context, "bf16" if name == "bf16" else "fp32", "mps"
            )
            for name, model in models.items()
        }
        reference = observations["fp32"]["gradients"]
        row = {
            "step": step,
            "losses": {name: v["loss"] for name, v in observations.items()},
            "trajectory": {
                name: compare_gradients(observations[name]["gradients"], reference)
                for name in ("bf16", "fp8_emulated")
            },
        }
        lr = learning_rate(step, config)
        row["learning_rate"] = lr
        before = None
        if capture:
            before = tree_cpu(dict(models["fp32"].named_parameters()))
            probe.load_state_dict(models["fp32"].state_dict())
            if tensor_map_digest(dict(probe.named_parameters())) != tensor_map_digest(before):
                raise AssertionError("Probe weights differ")
            probe_recorder.enabled, probe_recorder.rows = True, {}
            probe_result = accumulate_gradients(probe, window, args.context, "fp32", "mps")
            row["matched_weights"] = compare_gradients(probe_result["gradients"], reference)
            row["matched_loss"] = probe_result["loss"]
            row["quantization_stats"] = {
                "trajectory": copy.deepcopy(recorder.rows),
                "matched": copy.deepcopy(probe_recorder.rows),
            }
            probe_optimizer.load_state_dict(copy.deepcopy(optimizers["fp32"].state_dict()))
            for group in probe_optimizer.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(), base["training"]["clip_grad_norm"], error_if_nonfinite=True
            )
            probe_optimizer.step()
        row["preclip_norm"] = {}
        for name, model in models.items():
            row["preclip_norm"][name] = torch.nn.utils.clip_grad_norm_(
                model.parameters(), base["training"]["clip_grad_norm"], error_if_nonfinite=True
            ).item()
            for group in optimizers[name].param_groups:
                group["lr"] = lr
            optimizers[name].step()
        if capture:
            actual_delta = {k: p.detach().cpu() - before[k] for k, p in probe.named_parameters()}
            ref_delta = {
                k: p.detach().cpu() - before[k] for k, p in models["fp32"].named_parameters()
            }
            row["matched_optimizer_update"] = compare_gradients(actual_delta, ref_delta)
        history.append(row)
        if step in cfg["eval_steps"]:
            evaluate(step)
        if step == cfg["resume_step"]:
            torch.save(
                snapshot(models, optimizers, metadata, history, validation),
                args.out / f"step-{step}.pt",
            )
        if capture:
            print(
                json.dumps(
                    {
                        "context": args.context,
                        "seed": args.seed,
                        "step": step,
                        "matched_gradient": row["matched_weights"]["global"]["rel_l2"],
                    }
                ),
                flush=True,
            )
    for handle in handles:
        handle.remove()
    if observed != {
        "fp32": {"torch.float32"},
        "bf16": {"torch.bfloat16"},
        "fp8_emulated": {"torch.float32"},
    }:
        raise AssertionError(f"Unexpected projection compute types: {observed}")
    common = {}
    recorder.enabled = False
    for name, model in models.items():
        with disable_fp8(model):
            common[name] = evaluate_windows(model, val_windows, args.context, "fp32", "mps")
    torch.save(snapshot(models, optimizers, metadata, history, validation), args.out / "final.pt")
    result = {
        **metadata,
        "history": history,
        "validation": validation,
        "common_fp32_validation": common,
        "final_checkpoint_sha256": file_sha256(args.out / "final.pt"),
        "final_state_finite_fp32": True,
    }
    write_json(args.out / "result.json", result)
    print(json.dumps({"completed": [args.context, args.seed], "validation": common}), flush=True)


def suite(args, protocol):
    cfg = protocol["training"]
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.protocol, args.out / "protocol.json")
    manifest = {
        "environment": json.loads(json.dumps(environment())),
        "protocol_sha256": file_sha256(args.protocol),
        "commands": [],
    }
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--protocol",
        str(args.protocol.resolve()),
    ]

    def execute(label, context, seed, resume=None):
        command = common + [
            "--context",
            str(context),
            "--seed",
            str(seed),
            "--out",
            str(args.out / label),
        ]
        if resume:
            command += ["--resume", str(resume)]
        manifest["commands"].append(command)
        write_json(args.out / "manifest.json", manifest)
        launch(command, args.out, label)

    for context in cfg["contexts"]:
        for seed in cfg["seeds"]:
            execute(f"context-{context}-seed-{seed}", context, seed)
    context, seed = cfg["replay_context"], cfg["replay_seed"]
    original = args.out / f"context-{context}-seed-{seed}"
    differences = {}
    expected = torch.load(original / "final.pt", weights_only=True, map_location="cpu")
    for mode in ("repeat", "resume"):
        execute(
            mode,
            context,
            seed,
            original / f"step-{cfg['resume_step']}.pt" if mode == "resume" else None,
        )
        actual = torch.load(args.out / mode / "final.pt", weights_only=True, map_location="cpu")
        differences[mode] = compare_trees(expected, actual)
    write_json(args.out / "replay.json", differences)
    if any(differences.values()):
        raise RuntimeError("Training repeat/resume differs; original evidence retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "suite"))
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("fp8_protocol.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    protocol, base = load_protocol(args.protocol)
    run(args, protocol, base) if args.mode == "run" else suite(args, protocol)


if __name__ == "__main__":
    main()
