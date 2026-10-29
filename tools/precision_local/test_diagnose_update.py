"""Validate attribution against an independently constructed first Adam step."""

import pytest
import torch
from diagnose_update import attribute_update


def test_sign_flip_attribution_and_first_adam_formula():
    reference_gradient = torch.tensor([1e-9, 1.0], dtype=torch.float64)
    candidate_gradient = torch.tensor([-1e-9, 1.0], dtype=torch.float64)
    epsilon, lr = 1e-8, 1e-3
    updates = []
    for gradient in (reference_gradient, candidate_gradient):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=lr, eps=epsilon, foreach=False, fused=False)
        parameter.grad = gradient.clone()
        optimizer.step()
        updates.append(parameter.detach() - 1)
    result = attribute_update(reference_gradient, candidate_gradient, *updates, lr, epsilon)
    assert result["sign_flip_count"] == 1
    assert result["sign_flip_fraction"] == 0.5
    assert result["sign_flip_update_error_energy_fraction"] == pytest.approx(1.0)
    assert result["analytic_first_adam_residual_rel_l2"] < 1e-10


def test_identical_updates_have_zero_error_attribution():
    gradient, update = torch.ones(4), torch.full((4,), -1e-3)
    result = attribute_update(gradient, gradient, update, update, 1e-3, 1e-8)
    assert result["sign_flip_count"] == 0
    assert result["sign_flip_update_error_energy_fraction"] == 0
    assert result["analytic_first_adam_residual_rel_l2"] == 0
