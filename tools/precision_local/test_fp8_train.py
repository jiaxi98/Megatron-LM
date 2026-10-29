"""Reject incompatible restarts and ensure matched optimizer probes are isolated."""

import copy

import pytest
import torch
from baseline import compare_trees, make_optimizer, tree_cpu
from context_metrics import accumulate_gradients
from fp8 import install_fp8
from fp8_train import state_is_finite, validate_resume
from model import TinyDecoder
from test_baseline import config


@pytest.mark.parametrize(
    "key",
    [
        "protocol_sha256",
        "context",
        "seed",
        "initial_weights_sha256",
        "train_starts",
        "validation_starts",
    ],
)
def test_resume_rejects_wrong_identity_or_data(key):
    metadata = {
        name: 1
        for name in (
            "protocol_sha256",
            "context",
            "seed",
            "initial_weights_sha256",
            "train_starts",
            "validation_starts",
        )
    }
    metadata["environment"] = {"source_commit": "original"}
    saved = {"metadata": copy.deepcopy(metadata)}
    validate_resume(saved, metadata)
    metadata[key] = 2
    with pytest.raises(ValueError):
        validate_resume(saved, metadata)


def test_optimizer_probe_does_not_mutate_reference_state_or_gradients():
    cfg = config()
    cfg["model"].update(dropout=0.0, sequence_length=8)
    torch.manual_seed(51)
    reference, probe = TinyDecoder(cfg["model"]), TinyDecoder(cfg["model"])
    install_fp8(probe)
    ref_opt, probe_opt = make_optimizer(reference, cfg), make_optimizer(probe, cfg)
    window = torch.randint(32, (17,))
    accumulate_gradients(reference, window, 8, "fp32", "cpu")
    ref_opt.step()
    gradients = accumulate_gradients(reference, window, 8, "fp32", "cpu")["gradients"]
    state_before = tree_cpu(ref_opt.state_dict())
    weights_before = tree_cpu(reference.state_dict())
    probe.load_state_dict(reference.state_dict())
    probe_opt.load_state_dict(copy.deepcopy(ref_opt.state_dict()))
    accumulate_gradients(probe, window, 8, "fp32", "cpu")
    probe_opt.step()
    assert compare_trees(state_before, tree_cpu(ref_opt.state_dict())) == []
    assert compare_trees(weights_before, tree_cpu(reference.state_dict())) == []
    assert compare_trees(gradients, {k: p.grad for k, p in reference.named_parameters()}) == []
    assert compare_trees(weights_before, tree_cpu(probe.state_dict()))


def test_finite_check_rejects_nonfinite_nested_optimizer_state():
    assert state_is_finite({"state": [{"m": torch.zeros(3)}]})
    assert not state_is_finite({"state": [{"v": torch.tensor([float("inf")])}]})
