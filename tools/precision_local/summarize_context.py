"""Aggregate context-length gradient evidence without granting numerical acceptance."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from baseline import write_json
from run_suite import file_sha256


def spread(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def check_controls(results, protocol):
    expected = {(length, seed) for length in protocol["contexts"] for seed in protocol["seeds"]}
    if {(run["context"], run["seed"]) for run in results} != expected or len(results) != len(
        expected
    ):
        raise ValueError("Missing or duplicate context/seed result")
    checks = {}
    for seed in protocol["seeds"]:
        runs = [run for run in results if run["seed"] == seed]
        checks[f"seed-{seed}/same_initial_weights"] = (
            len({r["initial_parameters_sha256"] for r in runs}) == 1
        )
        checks[f"seed-{seed}/same_training_windows"] = all(
            r["train_window_starts"] == runs[0]["train_window_starts"] for r in runs
        )
    checks["same_validation_windows"] = all(
        r["validation_window_starts"] == results[0]["validation_window_starts"] for r in results
    )
    for run in results:
        label = f"context-{run['context']}/seed-{run['seed']}"
        checks[f"{label}/steps"] = [row["step"] for row in run["history"]] == list(
            range(1, protocol["steps"] + 1)
        )
        checks[f"{label}/probes"] = [
            row["step"] for row in run["history"] if "matched_weights" in row
        ] == protocol["matched_weight_probe_steps"]
        checks[f"{label}/tokens"] = (
            run["processed_tokens_per_precision"]
            == protocol["tokens_per_update"] * protocol["steps"]
        )
        checks[f"{label}/finite_positive_reference"] = all(
            math.isfinite(row[kind]["global"]["rel_l2"])
            and row[kind]["global"]["reference_norm"] > 0
            for row in run["history"]
            for kind in ("trajectory", "matched_weights")
            if kind in row
        )
        checks[f"{label}/initial_probe_matches_trajectory"] = (
            run["history"][0]["matched_weights"] == run["history"][0]["trajectory"]
        )
        checks[f"{label}/clean_source"] = run["environment"]["working_tree"] == ""
    return checks


def aggregate(results, protocol):
    controls = check_controls(results, protocol)
    rows = []
    for context in protocol["contexts"]:
        runs = [run for run in results if run["context"] == context]
        rows.append(
            {
                "context": context,
                "initial_matched_gradient_rel_l2": spread(
                    [r["history"][0]["matched_weights"]["global"]["rel_l2"] for r in runs]
                ),
                "final_matched_gradient_rel_l2": spread(
                    [r["history"][-1]["matched_weights"]["global"]["rel_l2"] for r in runs]
                ),
                "final_trajectory_gradient_rel_l2": spread(
                    [r["history"][-1]["trajectory"]["global"]["rel_l2"] for r in runs]
                ),
                "final_matched_gradient_cosine": spread(
                    [r["history"][-1]["matched_weights"]["global"]["cosine"] for r in runs]
                ),
                "final_matched_sign_flip_fraction": spread(
                    [
                        r["history"][-1]["matched_weights"]["global"]["sign_flip_fraction"]
                        for r in runs
                    ]
                ),
                "final_fp32_validation": spread(
                    [r["final_common_fp32_validation"]["fp32"] for r in runs]
                ),
                "final_bf16_weights_fp32_validation": spread(
                    [r["final_common_fp32_validation"]["bf16"] for r in runs]
                ),
                "max_absolute_common_fp32_validation_gap": max(
                    abs(
                        r["final_common_fp32_validation"]["bf16"]
                        - r["final_common_fp32_validation"]["fp32"]
                    )
                    for r in runs
                ),
                "max_sampled_live_bytes": max(r["sampled_memory"]["live_bytes"] for r in runs),
                "max_sampled_driver_bytes": max(r["sampled_memory"]["driver_bytes"] for r in runs),
            }
        )
    return {
        "status": "awaiting_review",
        "control_checks": controls,
        "control_checks_passed": all(controls.values()),
        "numerical_acceptance": "Not assigned; exploratory measurements with no new bound",
        "by_context": rows,
    }


def export_csv(results, root):
    global_rows, parameter_rows = [], []
    for run in results:
        for row in run["history"]:
            for kind in ("trajectory", "matched_weights"):
                if kind not in row:
                    continue
                prefix = {
                    "context": run["context"],
                    "seed": run["seed"],
                    "step": row["step"],
                    "kind": kind,
                }
                global_rows.append({**prefix, **row[kind]["global"]})
                if kind == "matched_weights":
                    parameter_rows.extend(
                        {**prefix, "parameter": name, **metrics}
                        for name, metrics in row[kind]["per_parameter"].items()
                    )
    for name, rows in (
        ("gradient_history.csv", global_rows),
        ("parameter_probes.csv", parameter_rows),
    ):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_plots(results, protocol, summary, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    contexts = protocol["contexts"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for panel, key, label, marker in (
        (0, "initial_matched_gradient_rel_l2", "Step 1", "o"),
        (0, "final_matched_gradient_rel_l2", "Step 64", "s"),
        (1, "final_trajectory_gradient_rel_l2", "Step 64", "^"),
    ):
        axis = axes[panel]
        means = np.array([row[key]["mean"] for row in summary["by_context"]]) * 100
        lower = np.array([row[key]["min"] for row in summary["by_context"]]) * 100
        upper = np.array([row[key]["max"] for row in summary["by_context"]]) * 100
        axis.errorbar(
            contexts,
            means,
            yerr=[means - lower, upper - means],
            marker=marker,
            capsize=3,
            label=label,
        )
    for axis, title in zip(
        axes,
        ("Shared weights: arithmetic difference", "Separate trajectories: combined difference"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks(contexts, [str(n) for n in contexts])
        axis.set(
            xlabel="Context length (byte tokens)",
            ylabel="Gradient relative L2 error (%)",
            title=title,
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("MPS BF16 vs FP32: mean and min-max across 3 seeds; separate y scales")
    figure.tight_layout()
    figure.savefig(root / "gradient_vs_context.png", dpi=160)
    plt.close(figure)

    for mode in ("gradients", "validation"):
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for axis, context in zip(axes.flat, contexts):
            runs = [run for run in results if run["context"] == context]
            if mode == "gradients":
                series = []
                for kind, label, style in (
                    ("trajectory", "Separate trajectories", "-"),
                    ("matched_weights", "Shared weights", "--o"),
                ):
                    rows = [row for row in runs[0]["history"] if kind in row]
                    steps = [row["step"] for row in rows]
                    values = np.array(
                        [
                            [
                                row[kind]["global"]["rel_l2"] * 100
                                for row in run["history"]
                                if kind in row
                            ]
                            for run in runs
                        ]
                    )
                    series.append((steps, values, label, style))
            else:
                series = [
                    (
                        [row["step"] for row in runs[0]["validation"]],
                        np.array([[row[dtype] for row in run["validation"]] for run in runs]),
                        dtype.upper(),
                        style,
                    )
                    for dtype, style in (("fp32", "-o"), ("bf16", "--s"))
                ]
            for steps, values, label, style in series:
                axis.plot(steps, values.mean(0), style, label=label, markersize=3)
                axis.fill_between(steps, values.min(0), values.max(0), alpha=0.12)
            axis.set(
                title=f"Context {context}",
                xlabel="Optimizer step",
                ylabel=(
                    "Gradient relative L2 (%)" if mode == "gradients" else "Held-out byte CE (nats)"
                ),
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            if mode == "gradients":
                axis.set_yscale("log")
                axis.set_ylim(0.1, 130)
        figure.suptitle(
            "4,096 targets/update; three seeds; shading is seed range"
            + ("; common log y scale" if mode == "gradients" else "")
        )
        figure.tight_layout()
        figure.savefig(root / f"{mode}_curves.png", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    paths = [
        args.root / f"context-{context}-seed-{seed}" / "result.json"
        for context in protocol["contexts"]
        for seed in protocol["seeds"]
    ]
    results = [json.loads(path.read_text()) for path in paths]
    manifest = json.loads((args.root / "manifest.json").read_text())
    if file_sha256(args.root / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Stored protocol bytes do not match the run manifest")
    summary = aggregate(results, protocol)
    summary["protocol"] = protocol
    summary["source_commits"] = sorted({r["environment"]["source_commit"] for r in results})
    summary["protocol_sha256"] = manifest["protocol_sha256"]
    summary["per_run"] = [
        {
            "context": r["context"],
            "seed": r["seed"],
            "validation": r["validation"],
            "final_common_fp32_validation": r["final_common_fp32_validation"],
            "sampled_memory": r["sampled_memory"],
            "instrumented_training_seconds": r["instrumented_training_seconds"],
            "matched_weight_probes": [
                {"step": row["step"], "global": row["matched_weights"]["global"]}
                for row in r["history"]
                if "matched_weights" in row
            ],
            "result_sha256": file_sha256(path),
            "artifacts": r["artifacts"],
        }
        for r, path in zip(results, paths)
    ]
    write_json(args.root / "summary.json", summary)
    export_csv(results, args.root)
    make_plots(results, protocol, summary, args.root)
    print(
        json.dumps(
            {
                "control_checks_passed": summary["control_checks_passed"],
                "status": summary["status"],
                "by_context": summary["by_context"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
