"""Check context controls and gradient reductions independently of MPS hardware."""

import pytest
import torch
from context_metrics import (
    accumulate_gradients,
    compare_gradients,
    split_window,
    tensor_map_digest,
    tensor_metrics,
)
from model import TinyDecoder, token_loss
from test_model import tiny_config


def test_all_contexts_preserve_identical_input_and_target_tokens():
    window = torch.arange(17)
    for length in (2, 4, 8, 16):
        inputs, targets = split_window(window, length)
        assert torch.equal(inputs.flatten(), window[:-1])
        assert torch.equal(targets.flatten(), window[1:])
        assert torch.equal(targets, inputs + 1)
    with pytest.raises(ValueError):
        split_window(window, 3)


def test_accumulated_gradient_matches_direct_effective_batch():
    config = tiny_config()
    config.update(dropout=0.0, sequence_length=4)
    torch.manual_seed(17)
    model = TinyDecoder(config)
    window = torch.randint(32, (17,))
    accumulated = accumulate_gradients(model, window, 4, "fp32", "cpu")
    model.zero_grad(set_to_none=True)
    inputs, targets = split_window(window, 4)
    direct_loss = token_loss(model(inputs), targets)
    direct_loss.backward()
    assert accumulated["loss"] == pytest.approx(direct_loss.item(), abs=1e-6)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            accumulated["gradients"][name], parameter.grad, rtol=2e-5, atol=1e-6
        )


def test_context_buffer_size_does_not_change_initialized_weights():
    hashes = []
    for length in (4, 8, 16):
        config = tiny_config()
        config["sequence_length"] = length
        torch.manual_seed(7)
        model = TinyDecoder(config)
        hashes.append(tensor_map_digest(dict(model.named_parameters())))
    assert len(set(hashes)) == 1


def test_zero_reference_and_sign_flip_metrics_are_explicit():
    result = tensor_metrics(torch.tensor([-1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert result["sign_flips"] == 1
    assert result["rel_l2"] == pytest.approx(2 / 5**0.5)
    assert tensor_metrics(torch.zeros(2), torch.zeros(2))["rel_l2"] == 0
    assert tensor_metrics(torch.ones(2), torch.zeros(2))["rel_l2"] is None
    with pytest.raises(FloatingPointError):
        tensor_metrics(torch.tensor([float("nan")]), torch.ones(1))


def test_global_error_uses_vector_norm_not_mean_of_parameter_ratios():
    reference = {"small": torch.ones(1), "large": torch.tensor([100.0])}
    candidate = {"small": torch.zeros(1), "large": torch.tensor([100.0])}
    result = compare_gradients(candidate, reference)
    assert result["per_parameter"]["small"]["rel_l2"] == 1
    assert result["global"]["rel_l2"] == pytest.approx(1 / 10001**0.5)
