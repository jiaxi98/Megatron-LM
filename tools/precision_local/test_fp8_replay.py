"""Added diagnostics cannot silently change original FP8 evidence."""

import copy

import torch
from fp8_replay import added_probe_errors


def fixture():
    run = dict.fromkeys(
        (
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
        ),
        1,
    )
    run["history"] = [{"step": 1, "losses": {"fp8_emulated": 3.0}}]
    checkpoint = {
        key: torch.ones(2) for key in ("models", "optimizers", "step", "cpu_rng", "mps_rng")
    }
    return run, checkpoint


def test_extra_probe_is_allowed_but_changed_original_value_is_not():
    original, checkpoint = fixture()
    replay = copy.deepcopy(original)
    replay["history"][0]["matched_weights"] = {"rel_l2": 0.1}
    assert added_probe_errors(original, replay, checkpoint, checkpoint) == []
    replay["history"][0]["losses"]["fp8_emulated"] += 1e-12
    assert added_probe_errors(original, replay, checkpoint, checkpoint)


def test_mutated_optimizer_and_missing_steps_are_rejected():
    original, checkpoint = fixture()
    changed = copy.deepcopy(checkpoint)
    changed["optimizers"][0] = 2
    assert added_probe_errors(original, original, checkpoint, changed)
    replay = copy.deepcopy(original)
    replay["history"] = []
    assert added_probe_errors(original, replay, checkpoint, checkpoint)
