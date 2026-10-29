"""Verify same-weight probes do not contaminate the training reference."""

import copy
from pathlib import Path

import torch
from baseline import compare_trees, make_optimizer, tree_cpu
from context_metrics import accumulate_gradients, tensor_map_digest
from context_sweep import load_protocol, matched_probe, window_starts
from model import TinyDecoder
from test_baseline import config


def test_matched_probe_preserves_reference_weights_gradients_and_optimizer():
    cfg = config()
    cfg["model"].update(dropout=0.0, sequence_length=4)
    torch.manual_seed(3)
    reference = TinyDecoder(cfg["model"])
    probe = TinyDecoder(cfg["model"])
    optimizer = make_optimizer(reference, cfg)
    window = torch.randint(32, (17,))
    accumulate_gradients(reference, window, 4, "fp32", "cpu")
    optimizer.step()
    accumulated = accumulate_gradients(reference, window, 4, "fp32", "cpu")
    weights_before = tree_cpu(reference.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    result = matched_probe(reference, probe, window, 4, "cpu")
    assert result["weight_sha256"] == tensor_map_digest(dict(reference.named_parameters()))
    assert compare_trees(weights_before, tree_cpu(reference.state_dict())) == []
    assert compare_trees(optimizer_before, optimizer.state_dict()) == []
    assert (
        compare_trees(
            accumulated["gradients"], {name: p.grad for name, p in reference.named_parameters()}
        )
        == []
    )


def test_protocol_and_sampling_cover_declared_budget_without_boundary_crossing():
    protocol, _ = load_protocol(Path(__file__).with_name("context_protocol.json"))
    assert protocol["contexts"][-1] == 4096
    data = torch.arange(10000)
    first = window_starts(data, protocol["tokens_per_update"], 64, 17)
    second = window_starts(data, protocol["tokens_per_update"], 64, 17)
    assert first == second
    assert min(first) >= 0
    assert max(first) + protocol["tokens_per_update"] < len(data)
