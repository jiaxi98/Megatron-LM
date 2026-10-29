"""Independent float8-cast oracle and adversarial checks of the FP8 surrogate."""

import copy

import pytest
import torch
from fp8 import (
    FORMATS,
    EmulatedFP8Linear,
    QuantizationStats,
    current_quantize,
    disable_fp8,
    install_fp8,
    round_finite,
)
from model import TinyDecoder, token_loss
from test_model import tiny_config


def boundary_values(format_name):
    dtype = getattr(torch, "float8_" + format_name)
    decoded = torch.arange(256, dtype=torch.uint8).view(dtype).float()
    finite = decoded[torch.isfinite(decoded)].unique().sort().values
    midpoints = (finite[1:] + finite[:-1]) / 2
    generator = torch.Generator().manual_seed(512)
    random = torch.randn(10000, generator=generator) * FORMATS[format_name].maximum
    return torch.cat(
        [
            finite,
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
            random,
            torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1e30, -1e30]),
        ]
    )


@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_all_encodings_midpoints_neighbors_and_random_match_native_cast(format_name, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    values = boundary_values(format_name)
    maximum = FORMATS[format_name].maximum
    expected = values.clamp(-maximum, maximum).to(getattr(torch, "float8_" + format_name)).float()
    actual = round_finite(values.to(device), format_name).cpu()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("format_name", FORMATS)
def test_scale_zero_underflow_nonfinite_and_representability(format_name):
    output, stats = current_quantize(torch.tensor([0.0, -0.0]), format_name)
    assert stats["dequant_scale"] == 1 and stats["underflowed"] == 0
    assert torch.signbit(output)[1]
    values = torch.tensor([1.0, -3.0, 1e-30, 0.0])
    output, stats = current_quantize(values, format_name)
    scale = stats["dequant_scale"]
    expected = (values / scale).to(getattr(torch, "float8_" + format_name)).float() * scale
    assert torch.equal(output, expected)
    assert stats["underflowed"] == 1 and stats["saturated"] == 0
    assert math_is_power_of_two(scale)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(FloatingPointError):
            current_quantize(torch.tensor([bad]), format_name)
    with pytest.raises(ValueError):
        current_quantize(torch.empty(0), format_name)
    with pytest.raises(FloatingPointError):
        current_quantize(torch.tensor([1e-40]), format_name)


def math_is_power_of_two(value):
    import math

    return math.frexp(value)[0] == 0.5


def oracle(value, format_name):
    import math

    maximum = torch.finfo(getattr(torch, "float8_" + format_name)).max
    amax = value.abs().max().item()
    scale = 2.0 ** math.ceil(math.log2(amax / maximum)) if amax else 1.0
    return (value / scale).to(getattr(torch, "float8_" + format_name)).float() * scale


def test_linear_forward_dgrad_wgrad_and_bias_use_independent_cast_oracle():
    torch.manual_seed(8)
    linear = torch.nn.Linear(32, 48)
    inputs = torch.randn(2, 7, 32, requires_grad=True)
    dy = torch.randn(2, 7, 48)
    recorder = QuantizationStats()
    recorder.enabled = True
    candidate = EmulatedFP8Linear(linear, recorder, "test")
    output = candidate(inputs)
    output.backward(dy)
    x, w, g = (
        oracle(inputs.detach(), "e4m3fn"),
        oracle(linear.weight.detach(), "e4m3fn"),
        oracle(dy, "e5m2"),
    )
    torch.testing.assert_close(output, x @ w.T + linear.bias, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, g @ w, rtol=0, atol=0)
    torch.testing.assert_close(
        linear.weight.grad, g.flatten(0, 1).T @ x.flatten(0, 1), rtol=0, atol=0
    )
    torch.testing.assert_close(linear.bias.grad, dy.flatten(0, 1).sum(0), rtol=0, atol=0)
    assert set(recorder.rows) == {"test/input", "test/weight", "test/output_gradient"}


def test_install_preserves_parameters_optimizer_and_disabled_model_exactly():
    cfg = tiny_config()
    cfg["dropout"] = 0
    model = TinyDecoder(cfg)
    reference = copy.deepcopy(model)
    original_ids = [id(p) for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters())
    _, names = install_fp8(model)
    assert len(names) == cfg["layers"] * 5
    assert [id(p) for p in model.parameters()] == original_ids
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == original_ids
    tokens = torch.randint(32, (2, 16))
    with disable_fp8(model):
        assert torch.equal(reference(tokens), model(tokens))
    loss = token_loss(model(tokens), tokens)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    state = copy.deepcopy(model.state_dict())
    reference.load_state_dict(state)
    with disable_fp8(model):
        assert torch.equal(reference(tokens), model(tokens))


def test_disable_restores_flags_after_exception():
    model = TinyDecoder(tiny_config())
    install_fp8(model)
    with pytest.raises(RuntimeError), disable_fp8(model):
        raise RuntimeError("test")
    assert model.blocks[0].qkv.enabled
