"""Adversarial data, comparison and replay checks for the local runner."""

import json
from pathlib import Path

import pytest
import torch
from baseline import (
    batch,
    compare_trees,
    configure,
    error_metrics,
    make_optimizer,
    restore,
    snapshot,
)
from model import TinyDecoder, token_loss
from test_model import tiny_config


def config():
    value = json.loads(Path(__file__).with_name("protocol.json").read_text())
    value["model"] = tiny_config()
    value["training"]["batch_size"] = 2
    return value


def test_batches_have_shifted_targets_and_stay_inside_their_split():
    cfg = config()
    data = torch.arange(80)
    generator = torch.Generator().manual_seed(19)
    for _ in range(10):
        inputs, targets, starts = batch(data, generator, cfg, "cpu")
        assert torch.equal(targets, inputs + 1)
        assert targets.max() < len(data)
        assert inputs[:, 0].tolist() == starts


def test_metric_rejects_nonfinite_and_zero_references():
    with pytest.raises(FloatingPointError):
        error_metrics(torch.tensor([float("nan")]), torch.ones(1))
    with pytest.raises(ValueError):
        error_metrics(torch.ones(1), torch.zeros(1))
    result = error_metrics(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert result == {"rel_l2": 1.0, "max_abs": 1.0, "cosine": 1.0}


def test_comparator_detects_optimizer_rng_and_dtype_corruption():
    reference = {
        "optimizer": {"exp_avg": torch.ones(2)},
        "rng": torch.tensor([1, 2], dtype=torch.uint8),
    }
    changed = {
        "optimizer": {"exp_avg": torch.zeros(2)},
        "rng": torch.tensor([1, 3], dtype=torch.uint8),
    }
    assert compare_trees(reference, changed) == ["root.optimizer.exp_avg", "root.rng"]
    assert compare_trees(torch.ones(1), torch.ones(1, dtype=torch.float64)) == ["root"]


def test_serialized_checkpoint_restores_optimizer_dropout_and_data_rng(tmp_path):
    cfg = config()
    configure(cfg, "cpu", 17)
    model = TinyDecoder(cfg["model"])
    optimizer = make_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(91)
    data = torch.arange(200) % cfg["model"]["vocab_size"]

    def step(active_model, active_optimizer, active_generator):
        inputs, targets, indices = batch(data, active_generator, cfg, "cpu")
        active_optimizer.zero_grad(set_to_none=True)
        loss = token_loss(active_model(inputs), targets)
        loss.backward()
        active_optimizer.step()
        return [{"step": 1, "loss": loss.item()}], indices

    history, indices = step(model, optimizer, generator)
    saved = snapshot(model, optimizer, generator, "cpu", "test", "fp32", 17, history, [], indices)
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    expected_history, expected_indices = step(model, optimizer, generator)
    expected = snapshot(
        model,
        optimizer,
        generator,
        "cpu",
        "test",
        "fp32",
        17,
        expected_history,
        [],
        expected_indices,
    )

    restarted = TinyDecoder(cfg["model"])
    restarted_optimizer = make_optimizer(restarted, cfg)
    restarted_generator = torch.Generator().manual_seed(999)
    loaded = torch.load(path, weights_only=True)
    restore(loaded, restarted, restarted_optimizer, restarted_generator, "cpu")
    actual_history, actual_indices = step(restarted, restarted_optimizer, restarted_generator)
    actual = snapshot(
        restarted,
        restarted_optimizer,
        restarted_generator,
        "cpu",
        "test",
        "fp32",
        17,
        actual_history,
        [],
        actual_indices,
    )
    assert compare_trees(expected, actual) == []
