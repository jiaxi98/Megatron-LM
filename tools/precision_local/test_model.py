"""CPU correctness checks for the independent reference decoder."""

import torch
from model import RMSNorm, TinyDecoder, token_loss


def tiny_config():
    return {
        "vocab_size": 32,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "ffn_width": 64,
        "sequence_length": 16,
        "dropout": 0.1,
        "norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
    }


def test_future_tokens_cannot_change_prefix_logits():
    torch.manual_seed(3)
    model = TinyDecoder(tiny_config()).eval()
    inputs = torch.randint(32, (2, 16))
    changed = inputs.clone()
    changed[:, 8:] = (changed[:, 8:] + 11) % 32
    with torch.no_grad():
        original, modified = model(inputs), model(changed)
    torch.testing.assert_close(original[:, :8], modified[:, :8], rtol=0, atol=0)
    assert not torch.equal(original[:, 8:], modified[:, 8:])


def test_normalization_matches_fp64_reference_with_large_dynamic_range():
    inputs = torch.tensor([[1e-6, -1e-3, 1.0, 1e4], [0.0, 0.0, 0.0, 0.0]])
    norm = RMSNorm(4, 1e-5)
    reference = inputs.double() / torch.sqrt(inputs.double().square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(inputs).double(), reference, rtol=2e-7, atol=1e-12)


def test_backward_reaches_every_parameter():
    torch.manual_seed(7)
    model = TinyDecoder(tiny_config()).train()
    tokens = torch.randint(32, (2, 17))
    loss = token_loss(model(tokens[:, :-1]), tokens[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_bf16_autocast_retains_fp32_parameters_and_logits():
    model = TinyDecoder(tiny_config()).eval()
    observed = []
    handle = model.blocks[0].qkv.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype)
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randint(32, (2, 16)))
    handle.remove()
    assert observed == [torch.bfloat16]
    assert output.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
