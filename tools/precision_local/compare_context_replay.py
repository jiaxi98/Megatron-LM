"""Require exact original training replay before attributing an extra gradient probe."""

import argparse
import json
from pathlib import Path

import torch
from baseline import compare_trees, write_json
from context_metrics import compare_gradients, tensor_metrics
from run_suite import file_sha256


def replay_mismatches(original, replay, original_checkpoint, replay_checkpoint):
    """Allow additional probe fields, but no changed original numerical evidence."""
    fields = (
        "context",
        "seed",
        "base_protocol_sha256",
        "parameter_count",
        "initial_parameters_sha256",
        "train_window_starts",
        "validation_window_starts",
        "tokens_per_update",
        "sequences_per_update",
        "processed_tokens_per_precision",
        "validation",
        "final_common_fp32_validation",
        "observed_projection_dtypes",
    )
    errors = []
    for field in fields:
        errors.extend(compare_trees(original[field], replay[field], field))
    if len(original["history"]) != len(replay["history"]):
        errors.append("history:length")
    for before, after in zip(original["history"], replay["history"]):
        for field, value in before.items():
            errors.extend(compare_trees(value, after.get(field), f"step-{before['step']}.{field}"))
    for field in ("models", "optimizers", "step", "cpu_rng", "mps_rng"):
        errors.extend(
            compare_trees(
                original_checkpoint[field], replay_checkpoint[field], f"checkpoint.{field}"
            )
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads((path / "result.json").read_text()) for path in (args.original, args.replay)]
    checkpoints = [
        torch.load(path / "final.pt", map_location="cpu", weights_only=True)
        for path in (args.original, args.replay)
    ]
    errors = replay_mismatches(*runs, *checkpoints)
    if errors:
        raise AssertionError(f"Original training did not reproduce exactly: {errors}")
    raw_path = args.replay / f"gradients-step-{args.step}.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    row = next(row for row in runs[1]["history"] if row["step"] == args.step)
    reference = raw["fp32_gradients"]
    matched = raw["bf16_matched_gradients"]
    trajectory = raw["bf16_trajectory_gradients"]
    for name, candidate in (("trajectory", trajectory), ("matched_weights", matched)):
        if compare_trees(compare_gradients(candidate, reference), row[name]):
            raise AssertionError(f"Raw gradients disagree with the stored {name} metrics")
    flatten = lambda values: torch.cat([values[key].double().flatten() for key in reference])
    ref, same, actual = map(flatten, (reference, matched, trajectory))
    arithmetic = same - ref
    weight_effect = actual - same
    total = actual - ref
    total_energy = torch.dot(total, total).item()
    attribution = {
        "definition": "total = (g_bf16(w_bf16)-g_bf16(w_fp32)) + (g_bf16(w_fp32)-g_fp32(w_fp32)); all on the same batch before clipping",
        "arithmetic_rel_to_reference_norm": arithmetic.norm().item() / ref.norm().item(),
        "weight_effect_rel_to_reference_norm": weight_effect.norm().item() / ref.norm().item(),
        "total_rel_to_reference_norm": total.norm().item() / ref.norm().item(),
        "arithmetic_squared_norm_over_total_squared_norm": arithmetic.square().sum().item()
        / total_energy,
        "weight_effect_squared_norm_over_total_squared_norm": weight_effect.square().sum().item()
        / total_energy,
        "cross_term_over_total_squared_norm": 2
        * torch.dot(arithmetic, weight_effect).item()
        / total_energy,
        "note": "Squared terms plus their signed cross term sum to one; they are not independent additive percentages.",
    }
    weights = tensor_metrics(flatten(raw["bf16_trajectory_weights"]), flatten(raw["fp32_weights"]))
    payload = {
        "status": "awaiting_review",
        "exact_original_replay": True,
        "checked": "All original history fields including per-parameter metrics/existing probes; initial weights, input windows, validation; final models, optimizer states, step and CPU/MPS RNG states",
        "mismatches": errors,
        "context": runs[0]["context"],
        "seed": runs[0]["seed"],
        "step": args.step,
        "original_source": runs[0]["environment"],
        "replay_source": runs[1]["environment"],
        "attribution": attribution,
        "weight_difference": weights,
        "probes": [
            {key: value for key, value in item.items() if key != "processed_tokens_before_update"}
            for item in runs[1]["history"]
            if "matched_weights" in item
        ],
        "evidence": {
            str(path): file_sha256(path)
            for path in [
                args.original / "result.json",
                args.original / "final.pt",
                args.replay / "result.json",
                args.replay / "final.pt",
                raw_path,
            ]
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"exact_original_replay": True, "attribution": attribution}))


if __name__ == "__main__":
    main()
