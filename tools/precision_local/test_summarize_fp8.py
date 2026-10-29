"""Acceptance must reject failures, missing coverage and changed paired inputs."""

import copy
import json
from pathlib import Path

import pytest
from summarize_fp8 import assess_training


def fixture():
    protocol = json.loads(Path(__file__).with_name("fp8_protocol.json").read_text())
    protocol["training"].update(
        contexts=[128, 1024], seeds=[17], steps=2, matched_probe_steps=[1, 2], eval_steps=[0, 2]
    )
    modules = ["blocks.0.qkv"]
    telemetry = {
        "format": "e4m3fn",
        "saturated": 0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "underflowed": 0,
        "nonzero_inputs": 2,
        "elements": 2,
    }
    stats = {
        "blocks.0.qkv/" + name: copy.deepcopy(telemetry)
        for name in ("input", "weight", "output_gradient")
    }
    history = [
        {
            "step": step,
            "losses": dict.fromkeys(protocol["training"]["arms"], 3.0),
            "preclip_norm": dict.fromkeys(protocol["training"]["arms"], 1.0),
            "matched_weights": {"global": {"rel_l2": 0.05}},
            "quantization_stats": {
                "trajectory": copy.deepcopy(stats),
                "matched": copy.deepcopy(stats),
            },
        }
        for step in (1, 2)
    ]
    base = {
        "context": 128,
        "seed": 17,
        "environment": {
            "source_commit": "source",
            "working_tree": "",
            "deterministic_algorithms": True,
            "mps_fallback": "0",
            "mps_fast_math": "0",
        },
        "initial_weights_sha256": dict.fromkeys(protocol["training"]["arms"], "weights"),
        "train_starts": [1, 2],
        "validation_starts": [3],
        "tokens_per_update": 4096,
        "processed_tokens_per_arm": 8192,
        "history": history,
        "fp8_modules": modules,
        "validation": [
            {"step": 0, **dict.fromkeys(protocol["training"]["arms"], 5.0)},
            {"step": 2, **dict.fromkeys(protocol["training"]["arms"], 3.0)},
        ],
        "common_fp32_validation": dict.fromkeys(protocol["training"]["arms"], 3.0),
        "final_state_finite_fp32": True,
    }
    second = copy.deepcopy(base)
    second["context"] = 1024
    return protocol, [base, second], {"repeat": [], "resume": []}


def test_all_declared_controls_are_required():
    protocol, runs, replay = fixture()
    assert all(assess_training(protocol, runs, replay).values())
    assert not all(assess_training(protocol, runs[:1], replay).values())
    assert not all(assess_training(protocol, runs, {}).values())
    replay["resume"] = ["optimizer.exp_avg"]
    assert not all(assess_training(protocol, runs, replay).values())


@pytest.mark.parametrize(
    "mutation", ["gradient", "loss", "window", "probe", "saturation", "telemetry", "dirty"]
)
def test_failures_are_not_reclassified_as_success(mutation):
    protocol, runs, replay = fixture()
    run = runs[0]
    if mutation == "gradient":
        run["history"][0]["matched_weights"]["global"]["rel_l2"] = 0.200001
    elif mutation == "loss":
        run["common_fp32_validation"]["fp8_emulated"] += 0.100001
    elif mutation == "window":
        run["train_starts"][0] += 1
    elif mutation == "probe":
        del run["history"][0]["matched_weights"]
    elif mutation == "saturation":
        run["history"][0]["quantization_stats"]["matched"]["blocks.0.qkv/input"]["saturated"] = 1
    elif mutation == "telemetry":
        del run["history"][0]["quantization_stats"]["trajectory"]["blocks.0.qkv/weight"]
    elif mutation == "dirty":
        run["environment"]["working_tree"] = " M fp8.py"
    assert not all(assess_training(protocol, runs, replay).values())
