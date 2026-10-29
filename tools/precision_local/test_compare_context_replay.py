"""A diagnostic probe must not hide a changed training trajectory or checkpoint."""

import copy

import pytest
import torch
from compare_context_replay import replay_mismatches


def evidence():
    run = dict.fromkeys(
        (
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
        ),
        1,
    )
    run["history"] = [
        {"step": 1, "trajectory": {"rel_l2": 0.1}, "matched_weights": {"rel_l2": 0.1}},
        {"step": 2, "trajectory": {"rel_l2": 0.2}},
    ]
    checkpoint = {
        key: {"tensor": torch.arange(3)} for key in ("models", "optimizers", "cpu_rng", "mps_rng")
    }
    checkpoint["step"] = 2
    return run, checkpoint


def test_accepts_added_probe_and_provenance_only():
    original, checkpoint = evidence()
    replay = copy.deepcopy(original)
    replay["history"][1]["matched_weights"] = {"rel_l2": 0.05}
    replay["environment"] = {"source_commit": "new"}
    assert replay_mismatches(original, replay, checkpoint, copy.deepcopy(checkpoint)) == []


@pytest.mark.parametrize("target", ["trajectory", "matched_weights"])
def test_rejects_changed_original_measurements(target):
    original, checkpoint = evidence()
    replay = copy.deepcopy(original)
    replay["history"][0][target]["rel_l2"] += 1e-12
    assert replay_mismatches(original, replay, checkpoint, checkpoint)


@pytest.mark.parametrize("target", ["models", "optimizers", "cpu_rng", "mps_rng"])
def test_rejects_changed_checkpoint_tensors(target):
    original, checkpoint = evidence()
    replay_checkpoint = copy.deepcopy(checkpoint)
    replay_checkpoint[target]["tensor"][0] += 1
    assert replay_mismatches(original, original, checkpoint, replay_checkpoint)


def test_rejects_missing_training_step():
    original, checkpoint = evidence()
    replay = copy.deepcopy(original)
    replay["history"].pop()
    assert replay_mismatches(original, replay, checkpoint, checkpoint)
