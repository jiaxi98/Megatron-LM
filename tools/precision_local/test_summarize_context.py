"""Prevent incomplete or uncontrolled context runs from passing control checks."""

import copy

import pytest
from summarize_context import check_controls


def evidence():
    protocol = {
        "contexts": [128, 256],
        "seeds": [17],
        "steps": 2,
        "tokens_per_update": 4096,
        "matched_weight_probe_steps": [1, 2],
    }
    comparison = {"global": {"rel_l2": 0.01, "reference_norm": 1.0}}
    runs = [
        {
            "context": length,
            "seed": 17,
            "initial_parameters_sha256": "same",
            "train_window_starts": [7, 11],
            "validation_window_starts": [19],
            "processed_tokens_per_precision": 8192,
            "environment": {"working_tree": ""},
            "history": [
                {
                    "step": step,
                    "trajectory": copy.deepcopy(comparison),
                    "matched_weights": copy.deepcopy(comparison),
                }
                for step in (1, 2)
            ],
        }
        for length in (128, 256)
    ]
    return runs, protocol


def test_complete_controlled_matrix_passes_controls():
    assert all(check_controls(*evidence()).values())


def test_missing_context_cannot_be_summarized_as_complete():
    runs, protocol = evidence()
    with pytest.raises(ValueError):
        check_controls(runs[:1], protocol)


@pytest.mark.parametrize("failure", ["weights", "windows", "step", "nan", "initial_probe"])
def test_control_mismatches_are_retained(failure):
    runs, protocol = evidence()
    run = runs[1]
    if failure == "weights":
        run["initial_parameters_sha256"] = "different"
    elif failure == "windows":
        run["train_window_starts"] = [1, 2]
    elif failure == "step":
        run["history"].pop()
    elif failure == "nan":
        run["history"][1]["trajectory"]["global"]["rel_l2"] = float("nan")
    else:
        run["history"][0]["matched_weights"]["global"]["rel_l2"] = 0.02
    assert not all(check_controls(runs, protocol).values())
