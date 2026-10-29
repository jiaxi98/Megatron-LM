"""Execute the fixed local protocol in fresh processes and collect gate evidence."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from baseline import compare_trees, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launch(command: list[str], root: Path, label: str) -> None:
    print(f"START {label}", flush=True)
    environment = os.environ.copy()
    environment.update(PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0")
    with (root / f"{label}.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        write_json(
            root / "failure.json", {"label": label, "command": command, "returncode": returncode}
        )
        raise RuntimeError(f"{label} failed; inspect {root / (label + '.log')}")


def assess(config: dict, results: dict, audit: dict, replay: dict) -> dict:
    """Apply the predeclared gates; never promote automated success to acceptance."""
    checks = {
        "audit/complete": set(audit["comparisons"]) == {"fp32", "bf16"},
        "replay/complete": set(replay)
        == {"fp32-repeat", "fp32-resume", "bf16-repeat", "bf16-resume"},
    }
    for dtype, comparison in audit["comparisons"].items():
        checks[f"audit/{dtype}/complete"] = set(comparison["checks"]) == {
            "logits_rel_l2",
            "grad_rel_l2",
            "update_rel_l2",
        }
        for name, passed in comparison["checks"].items():
            checks[f"audit/{dtype}/{name}"] = passed
    pairs = []
    for seed in config["training"]["seeds"]:
        fp32, bf16 = (results[f"{dtype}-{seed}"] for dtype in ("fp32", "bf16"))
        checks[f"pair/{seed}/initial_weights"] = (
            fp32["initial_parameters_sha256"] == bf16["initial_parameters_sha256"]
        )
        checks[f"pair/{seed}/sample_order"] = (
            fp32["sample_indices_sha256"] == bf16["sample_indices_sha256"]
        )
        for dtype, run in (("fp32", fp32), ("bf16", bf16)):
            checks[f"train/{dtype}/{seed}/complete"] = (
                len(run["history"]) == config["training"]["steps"]
                and run["validation"][-1]["step"] == config["training"]["steps"]
            )
            checks[f"train/{dtype}/{seed}/finite"] = all(
                math.isfinite(row[key]) for row in run["history"] for key in ("loss", "grad_norm")
            ) and all(math.isfinite(row["loss"]) for row in run["validation"])
            checks[f"train/{dtype}/{seed}/loss_drop"] = (
                run["validation"][0]["loss"] - run["validation"][-1]["loss"]
                >= config["acceptance"]["minimum_validation_loss_drop"]
            )
        execution_gap = abs(bf16["validation"][-1]["loss"] - fp32["validation"][-1]["loss"])
        fp32_eval_gap = abs(bf16["validation_fp32_eval"] - fp32["validation_fp32_eval"])
        checks[f"pair/{seed}/validation_gap"] = (
            max(execution_gap, fp32_eval_gap)
            <= config["acceptance"]["maximum_paired_final_validation_loss_gap"]
        )
        pairs.append(
            {
                "seed": seed,
                "fp32_validation_loss": fp32["validation"][-1]["loss"],
                "bf16_validation_loss": bf16["validation"][-1]["loss"],
                "bf16_weights_fp32_eval_loss": bf16["validation_fp32_eval"],
                "absolute_execution_loss_gap": execution_gap,
                "absolute_fp32_eval_loss_gap": fp32_eval_gap,
            }
        )
    for name, differences in replay.items():
        checks[f"replay/{name}/exact"] = not differences
    return {
        "gate": "G0-local",
        "status": "awaiting_review",
        "automated_checks_passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "paired_results": pairs,
        "replay_differences": replay,
        "scope": "Local reference only. H20 Megatron/TE baseline and G1-G4 remain pending.",
    }


def plot_curves(config: dict, results: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    tokens_per_step = config["training"]["batch_size"] * config["model"]["sequence_length"]
    for axis, seed in zip(axes, config["training"]["seeds"]):
        for dtype, style in (("fp32", "-"), ("bf16", "--")):
            rows = results[f"{dtype}-{seed}"]["validation"]
            axis.plot(
                [row["step"] * tokens_per_step / 1000 for row in rows],
                [row["loss"] for row in rows],
                style,
                marker="o",
                label=dtype.upper(),
            )
        axis.set(title=f"Seed {seed}", xlabel="Training tokens (thousands)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Held-out byte cross-entropy (nats)")
    figure.suptitle("MPS local reference: short-run baseline, not full-pretraining convergence")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def summarize(root: Path, config: dict) -> dict:
    results = {
        f"{dtype}-{seed}": json.loads((root / f"{dtype}-{seed}" / "result.json").read_text())
        for seed in config["training"]["seeds"]
        for dtype in ("fp32", "bf16")
    }
    audit = json.loads((root / "audit" / "result.json").read_text())
    replay = {}
    steps = config["training"]["steps"]
    first_seed = config["training"]["seeds"][0]
    for dtype in ("fp32", "bf16"):
        original = torch.load(
            root / f"{dtype}-{first_seed}" / f"step-{steps}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for mode in ("repeat", "resume"):
            candidate = torch.load(
                root / f"{dtype}-{first_seed}-{mode}" / f"step-{steps}.pt",
                map_location="cpu",
                weights_only=True,
            )
            replay[f"{dtype}-{mode}"] = compare_trees(original, candidate)
    summary = assess(config, results, audit, replay)
    summary["audit"] = audit
    summary["runs"] = {
        name: {key: value for key, value in run.items() if key != "history"}
        for name, run in results.items()
    }
    summary["protocol_sha256"] = file_sha256(root / "protocol.json")
    artifacts = sorted(root.glob("*/*.pt")) + sorted(root.glob("*/result.json"))
    summary["artifacts"] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifacts
    ]
    write_json(root / "summary.json", summary)
    plot_curves(config, results, root / "validation_curves.png")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "automated_checks_passed", "failed_checks", "paired_results")
            }
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    script = Path(__file__).with_name("baseline.py")
    protocol = Path(__file__).with_name("protocol.json")
    if args.summarize_only:
        summarize(args.out, json.loads((args.out / "protocol.json").read_text()))
        return
    config = json.loads(protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol, args.out / "protocol.json")
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "protocol_sha256": file_sha256(protocol),
        "dependencies_sha256": file_sha256(protocol.with_name("requirements.lock")),
        "source_files": {path.name: file_sha256(path) for path in script.parent.glob("*.py")},
        "hardware": {
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        },
        "commands": [],
    }
    (args.out / "environment.txt").write_text(
        subprocess.check_output(
            ["uv", "pip", "freeze", "--no-config", "--python", sys.executable], text=True
        )
    )

    def execute(label, mode="train", extra=()):
        command = [
            sys.executable,
            str(script),
            mode,
            "--config",
            str(args.out / "protocol.json"),
            "--out",
            str(args.out / label),
            *extra,
        ]
        manifest["commands"].append({"label": label, "argv": command})
        write_json(args.out / "manifest.json", manifest)
        launch(command, args.out, label)

    execute("audit", mode="audit")
    for seed in config["training"]["seeds"]:
        for dtype in ("fp32", "bf16"):
            execute(f"{dtype}-{seed}", extra=("--dtype", dtype, "--seed", str(seed)))
    first_seed = config["training"]["seeds"][0]
    for dtype in ("fp32", "bf16"):
        base = ("--dtype", dtype, "--seed", str(first_seed))
        execute(f"{dtype}-{first_seed}-repeat", extra=base)
        checkpoint = (
            args.out / f"{dtype}-{first_seed}" / f"step-{config['training']['checkpoint_step']}.pt"
        )
        execute(f"{dtype}-{first_seed}-resume", extra=(*base, "--resume", str(checkpoint)))
    summarize(args.out, config)


if __name__ == "__main__":
    main()
