"""Verify extra FP8 probes preserved the original trajectory, then bound drift."""

import argparse
import json
from pathlib import Path

import torch
from baseline import compare_trees, write_json
from run_suite import file_sha256


def added_probe_errors(original, replay, old_checkpoint, new_checkpoint):
    errors = []
    for field in (
        "context",
        "seed",
        "initial_weights_sha256",
        "train_starts",
        "validation_starts",
        "tokens_per_update",
        "processed_tokens_per_arm",
        "parameter_count",
        "fp8_modules",
        "scope",
        "validation",
        "common_fp32_validation",
        "final_state_finite_fp32",
    ):
        errors.extend(compare_trees(original[field], replay[field], field))
    if len(original["history"]) != len(replay["history"]):
        errors.append("history:length")
    for old, new in zip(original["history"], replay["history"]):
        for field in old:
            errors.extend(compare_trees(old[field], new.get(field), f"step-{old['step']}.{field}"))
    for field in ("models", "optimizers", "step", "cpu_rng", "mps_rng"):
        errors.extend(
            compare_trees(old_checkpoint[field], new_checkpoint[field], "checkpoint." + field)
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads((root / "result.json").read_text()) for root in (args.original, args.replay)]
    checkpoints = [
        torch.load(root / "final.pt", weights_only=True, map_location="cpu")
        for root in (args.original, args.replay)
    ]
    errors = added_probe_errors(*runs, *checkpoints)
    if errors:
        raise AssertionError(f"Original trajectory changed: {errors}")
    peak = next(row for row in runs[1]["history"] if row["step"] == args.step)
    trajectory = peak["trajectory"]["fp8_emulated"]["global"]
    matched = peak["matched_weights"]["global"]
    assert trajectory["reference_norm"] == matched["reference_norm"]
    low = abs(trajectory["rel_l2"] - matched["rel_l2"])
    high = trajectory["rel_l2"] + matched["rel_l2"]
    result = {
        "exact_original_replay": True,
        "context": runs[0]["context"],
        "seed": runs[0]["seed"],
        "step": args.step,
        "peak": peak,
        "mismatches": errors,
        "original_environment": runs[0]["environment"],
        "replay_environment": runs[1]["environment"],
        "original_protocol_sha256": runs[0]["protocol_sha256"],
        "replay_protocol_sha256": runs[1]["protocol_sha256"],
        "weight_effect_relative_norm_bounds": [low, high],
        "bound_definition": "By triangle/reverse-triangle inequalities, ||g8(w8)-g8(w32)||/||g32(w32)|| lies between |trajectory_rel_l2-matched_rel_l2| and their sum. This is a bound, not a measured vector decomposition.",
        "evidence_sha256": {
            str(root / name): file_sha256(root / name)
            for root in (args.original, args.replay)
            for name in ("result.json", "final.pt")
        },
        "status": "diagnostic_recorded_for_batch_review",
    }
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "exact_original_replay": True,
                "trajectory": trajectory["rel_l2"],
                "same_weights": matched["rel_l2"],
                "weight_effect_bounds": [low, high],
            }
        )
    )


if __name__ == "__main__":
    main()
