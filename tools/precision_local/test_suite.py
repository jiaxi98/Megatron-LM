"""Ensure acceptance reports cannot silently pass incomplete or failed evidence."""

import copy

import pytest
from run_suite import assess


def evidence():
    config = {
        "training": {"seeds": [17], "steps": 2},
        "acceptance": {
            "minimum_validation_loss_drop": 0.3,
            "maximum_paired_final_validation_loss_gap": 0.1,
        },
    }
    run = {
        "initial_parameters_sha256": "same-initial-weights",
        "sample_indices_sha256": "same-data-order",
        "history": [{"loss": 3.5, "grad_norm": 1.0}, {"loss": 3.0, "grad_norm": 0.5}],
        "validation": [{"step": 0, "loss": 4.0}, {"step": 2, "loss": 3.0}],
        "validation_fp32_eval": 3.0,
    }
    results = {f"{dtype}-17": copy.deepcopy(run) for dtype in ("fp32", "bf16")}
    audit = {
        "comparisons": {
            dtype: {
                "checks": {name: True for name in ("logits_rel_l2", "grad_rel_l2", "update_rel_l2")}
            }
            for dtype in ("fp32", "bf16")
        }
    }
    replay = {f"{dtype}-{mode}": [] for dtype in ("fp32", "bf16") for mode in ("repeat", "resume")}
    return config, results, audit, replay


def test_automated_pass_is_still_awaiting_user_review():
    result = assess(*evidence())
    assert result["automated_checks_passed"]
    assert result["status"] == "awaiting_review"


@pytest.mark.parametrize(
    "failure",
    [
        "weights",
        "data",
        "nan",
        "short_run",
        "loss_gap",
        "missing_audit",
        "missing_replay",
        "optimizer",
    ],
)
def test_failed_or_missing_evidence_blocks_automated_pass(failure):
    config, results, audit, replay = evidence()
    candidate = results["bf16-17"]
    if failure == "weights":
        candidate["initial_parameters_sha256"] = "different"
    elif failure == "data":
        candidate["sample_indices_sha256"] = "different"
    elif failure == "nan":
        candidate["history"][0]["grad_norm"] = float("nan")
    elif failure == "short_run":
        candidate["history"].pop()
    elif failure == "loss_gap":
        candidate["validation_fp32_eval"] += 0.11
    elif failure == "missing_audit":
        audit["comparisons"].pop("bf16")
    elif failure == "missing_replay":
        replay.clear()
    else:
        replay["bf16-resume"] = ["root.optimizer.exp_avg"]
    result = assess(config, results, audit, replay)
    assert not result["automated_checks_passed"]
    assert result["failed_checks"]
