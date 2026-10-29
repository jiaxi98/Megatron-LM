"""Apply unchanged FP8 checks, export evidence, and defer user review in a batch."""

import argparse
import csv
import json
import math
from pathlib import Path

from baseline import write_json
from run_suite import file_sha256


def assess_training(protocol, runs, replay):
    cfg, bounds = protocol["training"], protocol["training_acceptance"]
    expected = {(context, seed) for context in cfg["contexts"] for seed in cfg["seeds"]}
    checks = {
        "matrix_complete": {(r["context"], r["seed"]) for r in runs} == expected
        and len(runs) == len(expected),
        "source_consistent": len({r["environment"]["source_commit"] for r in runs}) == 1,
        "replay_complete": set(replay) == {"repeat", "resume"},
    }
    for name, errors in replay.items():
        checks[f"{name}/exact"] = errors == []
    for seed in cfg["seeds"]:
        selected = [r for r in runs if r["seed"] == seed]
        checks[f"seed-{seed}/common_windows"] = (
            len({tuple(r["train_starts"]) for r in selected}) == 1
        )
        checks[f"seed-{seed}/common_initialization"] = (
            len({r["initial_weights_sha256"]["fp32"] for r in selected}) == 1
        )
    checks["common_validation_windows"] = len({tuple(r["validation_starts"]) for r in runs}) == 1
    for run in runs:
        label = f"context-{run['context']}/seed-{run['seed']}"
        env = run["environment"]
        checks[label + "/execution"] = (
            env["working_tree"] == ""
            and env["deterministic_algorithms"]
            and env["mps_fallback"] == "0"
            and env["mps_fast_math"] == "0"
        )
        checks[label + "/initial_weights"] = (
            set(run["initial_weights_sha256"]) == set(cfg["arms"])
            and len(set(run["initial_weights_sha256"].values())) == 1
        )
        checks[label + "/token_budget"] = (
            run["tokens_per_update"] == cfg["tokens_per_update"]
            and run["processed_tokens_per_arm"] == cfg["steps"] * cfg["tokens_per_update"]
            and len(run["train_starts"]) == cfg["steps"]
        )
        history = run["history"]
        checks[label + "/step_coverage"] = [row["step"] for row in history] == list(
            range(1, cfg["steps"] + 1)
        )
        probes = [row for row in history if "matched_weights" in row]
        checks[label + "/probe_coverage"] = [row["step"] for row in probes] == cfg[
            "matched_probe_steps"
        ]
        checks[label + "/validation_coverage"] = [row["step"] for row in run["validation"]] == cfg[
            "eval_steps"
        ]
        checks[label + "/finite_losses_and_norms"] = all(
            math.isfinite(row[key][arm])
            for row in history
            for key in ("losses", "preclip_norm")
            for arm in cfg["arms"]
        )
        checks[label + "/finite_fp32_state"] = run["final_state_finite_fp32"]
        for row in probes:
            prefix = f"{label}/probe-{row['step']}"
            error = row["matched_weights"]["global"]["rel_l2"]
            checks[prefix + "/gradient"] = (
                error is not None
                and math.isfinite(error)
                and error <= bounds["same_weight_global_gradient_rel_l2_max"]
            )
            names = {
                name + "/" + operand
                for name in run["fp8_modules"]
                for operand in ("input", "weight", "output_gradient")
            }
            checks[prefix + "/quantization_coverage"] = all(
                set(row["quantization_stats"][kind]) == names for kind in ("trajectory", "matched")
            )
            checks[prefix + "/scaling_invariants"] = all(
                v["saturated"] == 0
                and 0 < v["scale_min"] <= v["scale_max"]
                and 0 <= v["underflowed"] <= v["nonzero_inputs"] <= v["elements"]
                for kind in ("trajectory", "matched")
                for v in row["quantization_stats"][kind].values()
            )
        for arm in cfg["arms"]:
            checks[label + f"/{arm}/loss_drop"] = (
                run["validation"][0][arm] - run["validation"][-1][arm]
                >= bounds["minimum_validation_loss_drop"]
            )
            gap = max(
                abs(run["validation"][-1][arm] - run["validation"][-1]["fp32"]),
                abs(run["common_fp32_validation"][arm] - run["common_fp32_validation"]["fp32"]),
            )
            checks[label + f"/{arm}/loss_gap"] = (
                gap <= bounds["maximum_final_validation_gap_vs_fp32"]
            )
    return checks


def export_csv(runs, root):
    gradients, parameters, stats = [], [], []
    for run in runs:
        for row in run["history"]:
            tags = {"context": run["context"], "seed": run["seed"], "step": row["step"]}
            for arm in ("bf16", "fp8_emulated"):
                gradients.append(
                    {**tags, "comparison": arm + "/trajectory", **row["trajectory"][arm]["global"]}
                )
            if "matched_weights" in row:
                for field in ("matched_weights", "matched_optimizer_update"):
                    gradients.append(
                        {**tags, "comparison": "fp8_emulated/" + field, **row[field]["global"]}
                    )
                    for name, values in row[field]["per_parameter"].items():
                        parameters.append(
                            {**tags, "comparison": field, "parameter": name, **values}
                        )
                for kind, rows in row["quantization_stats"].items():
                    for name, values in rows.items():
                        stats.append({**tags, "comparison": kind, "operand": name, **values})
    for name, rows in (
        ("gradient_history", gradients),
        ("parameter_probes", parameters),
        ("quantization_stats", stats),
    ):
        with (root / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plots(protocol, runs, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    contexts = protocol["training"]["contexts"]
    gradient_values = [
        100 * row["trajectory"]["fp8_emulated"]["global"]["rel_l2"]
        for run in runs
        for row in run["history"]
    ]
    gradient_values += [
        100 * row["matched_weights"]["global"]["rel_l2"]
        for run in runs
        for row in run["history"]
        if "matched_weights" in row
    ]
    gradient_limits = (
        min(value for value in gradient_values if value > 0) * 0.8,
        max(gradient_values) * 1.25,
    )
    for mode in ("validation", "gradients"):
        fig, axes = plt.subplots(1, len(contexts), figsize=(12, 3.9), squeeze=False)
        for axis, context in zip(axes[0], contexts):
            selected = [r for r in runs if r["context"] == context]
            series = (
                protocol["training"]["arms"]
                if mode == "validation"
                else ("trajectory", "matched_weights")
            )
            for kind in series:
                rows = [
                    (
                        r["validation"]
                        if mode == "validation"
                        else [
                            row
                            for row in r["history"]
                            if kind == "trajectory" or "matched_weights" in row
                        ]
                    )
                    for r in selected
                ]
                steps = [row["step"] for row in rows[0]]
                values = np.array(
                    [
                        [
                            (
                                row[kind]
                                if mode == "validation"
                                else 100
                                * (
                                    row["trajectory"]["fp8_emulated"]["global"]["rel_l2"]
                                    if kind == "trajectory"
                                    else row["matched_weights"]["global"]["rel_l2"]
                                )
                            )
                            for row in entries
                        ]
                        for entries in rows
                    ]
                )
                axis.plot(
                    steps,
                    values.mean(0),
                    "-o" if mode == "validation" else "--o" if kind == "matched_weights" else "-",
                    markersize=3,
                    label=kind,
                )
                axis.fill_between(steps, values.min(0), values.max(0), alpha=0.12)
            axis.set(
                title=f"Context {context} bytes",
                xlabel="Optimizer step",
                ylabel=(
                    "Validation CE (nats/byte)"
                    if mode == "validation"
                    else "FP8 gradient relative L2 (%)"
                ),
            )
            if mode == "gradients":
                axis.set_yscale("log")
                axis.set_ylim(*gradient_limits)
            axis.legend(fontsize=7)
            axis.grid(alpha=0.2)
        fig.suptitle(
            "Numerical emulation; 4,096 targets/update; mean and range across 3 seeds"
            + ("; common log scale" if mode == "gradients" else "")
        )
        fig.tight_layout()
        fig.savefig(root / (mode + ".png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--operators", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    paths = [
        args.root / f"context-{c}-seed-{s}" / "result.json"
        for c in protocol["training"]["contexts"]
        for s in protocol["training"]["seeds"]
    ]
    runs = [json.loads(path.read_text()) for path in paths]
    replay = json.loads((args.root / "replay.json").read_text())
    checks = assess_training(protocol, runs, replay)
    audit = json.loads(args.operators.read_text())
    expected_cases = {
        (tuple(shape), seed)
        for shape in protocol["operator_shapes_m_k_n"]
        for seed in protocol["operator_seeds"]
    }
    checks["operators/identities"] = {
        (tuple(row["shape"]), row["seed"]) for row in audit["cases"]
    } == expected_cases
    checks["operators/check_coverage"] = len(audit["checks"]) == len(expected_cases) * 10
    base_path = Path(__file__).with_name(protocol["base_protocol"])
    checks["base_protocol_identity"] = file_sha256(base_path) == protocol["base_protocol_sha256"]
    base = json.loads(base_path.read_text())
    expected_modules = {
        f"blocks.{index}.{name}"
        for index in range(base["model"]["layers"])
        for name in ("qkv", "projection", "gate", "up", "down")
    }
    for run in runs:
        checks[f"modules/{run['context']}/{run['seed']}/coverage"] = (
            set(run["fp8_modules"]) == expected_modules
        )
    checks["operators/complete"] = len(audit["cases"]) == len(protocol["operator_seeds"]) * len(
        protocol["operator_shapes_m_k_n"]
    )
    checks.update({"operators/" + k: v for k, v in audit["checks"].items()})
    digest = file_sha256(args.root / "protocol.json")
    checks["protocol_identity"] = audit["protocol_sha256"] == digest and all(
        run["protocol_sha256"] == digest for run in runs
    )
    for run, path in zip(runs, paths):
        checks[f"checkpoint/{run['context']}/{run['seed']}/hash"] = (
            file_sha256(path.parent / "final.pt") == run["final_checkpoint_sha256"]
        )
    summary = {
        "status": "passed_pending_batch_review" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "protocol_sha256": digest,
        "source_commits": sorted({r["environment"]["source_commit"] for r in runs}),
        "operator_result_sha256": file_sha256(args.operators),
        "replay": replay,
        "runs": [
            {
                "context": r["context"],
                "seed": r["seed"],
                "validation": r["validation"],
                "common_fp32_validation": r["common_fp32_validation"],
                "matched_probes": [
                    {
                        "step": row["step"],
                        "gradient": row["matched_weights"]["global"],
                        "update": row["matched_optimizer_update"]["global"],
                    }
                    for row in r["history"]
                    if "matched_weights" in row
                ],
                "result_sha256": file_sha256(path),
                "final_checkpoint_sha256": r["final_checkpoint_sha256"],
            }
            for r, path in zip(runs, paths)
        ],
        "scope": "Local FP8 operand emulation with FP32 GEMMs; not native FP8 or H20/TE acceptance. Same-optimizer update errors are diagnostics without a new acceptance bound.",
    }
    write_json(args.root / "summary.json", summary)
    export_csv(runs, args.root)
    plots(protocol, runs, args.root)
    print(
        json.dumps(
            {"status": summary["status"], "checks": len(checks), "failed": summary["failed_checks"]}
        )
    )


if __name__ == "__main__":
    main()
