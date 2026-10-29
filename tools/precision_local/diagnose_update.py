"""Attribute first-step AdamW update differences without changing acceptance gates."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from baseline import environment, error_metrics, load_data, one_step, write_json
from model import TinyDecoder


def attribute_update(
    reference_gradient, candidate_gradient, reference_update, candidate_update, lr, epsilon
):
    """Separate gradient sign flips and compare against the analytic first Adam step."""
    reference_gradient, candidate_gradient, reference_update, candidate_update = (
        tensor.detach().cpu().double().flatten()
        for tensor in (reference_gradient, candidate_gradient, reference_update, candidate_update)
    )
    difference = candidate_update - reference_update
    energy = difference.square().sum().item()
    flips = reference_gradient * candidate_gradient < 0
    prediction = -lr * (
        candidate_gradient / (candidate_gradient.abs() + epsilon)
        - reference_gradient / (reference_gradient.abs() + epsilon)
    )
    residual = (difference - prediction).square().sum().item()
    return {
        "elements": reference_gradient.numel(),
        "sign_flip_count": flips.sum().item(),
        "sign_flip_fraction": flips.double().mean().item(),
        "sign_flip_update_error_energy_fraction": (
            difference[flips].square().sum().item() / energy if energy else 0.0
        ),
        "analytic_first_adam_residual_rel_l2": (residual / energy) ** 0.5 if energy else 0.0,
        "median_abs_reference_gradient_on_flips": (
            reference_gradient[flips].abs().median().item() if flips.any() else None
        ),
        "median_abs_reference_gradient_all": reference_gradient.abs().median().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = Path(__file__).with_name("protocol.json").read_bytes()
    config = json.loads(raw)
    data, _ = load_data(config, Path("local/precision/data/tinyshakespeare.txt"))
    reference = one_step(config, data, "cpu", "fp32")
    candidate = one_step(config, data, "mps", "bf16")
    attribution = attribute_update(
        reference["grad"],
        candidate["grad"],
        reference["update"],
        candidate["update"],
        config["training"]["learning_rate"],
        config["training"]["epsilon"],
    )
    total_energy = (
        (candidate["update"].double() - reference["update"].double()).square().sum().item()
    )
    rows, offset = [], 0
    for name, parameter in TinyDecoder(config["model"]).named_parameters():
        end = offset + parameter.numel()
        ref_grad, grad = reference["grad"][offset:end], candidate["grad"][offset:end]
        ref_update, update = reference["update"][offset:end], candidate["update"][offset:end]
        rows.append(
            {
                "parameter": name,
                "gradient_rel_l2": error_metrics(grad, ref_grad)["rel_l2"],
                "update_rel_l2": error_metrics(update, ref_update)["rel_l2"],
                "update_error_energy_fraction": (update.double() - ref_update.double())
                .square()
                .sum()
                .item()
                / total_energy,
                "gradient_sign_flip_fraction": (grad.double() * ref_grad.double() < 0)
                .double()
                .mean()
                .item(),
            }
        )
        offset = end
    if offset != reference["grad"].numel():
        raise AssertionError("Parameter ordering/size does not cover the flattened gradient")
    report = {
        "environment": environment(),
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": 17,
        "purpose": "Post-run diagnosis only; original 0.10 update-error threshold remains failed and unchanged.",
        "update_metrics": error_metrics(candidate["update"], reference["update"]),
        "attribution": attribution,
        "parameters": sorted(
            rows, key=lambda row: row["update_error_energy_fraction"], reverse=True
        ),
    }
    write_json(args.out / "result.json", report)
    print(
        json.dumps({"update_metrics": report["update_metrics"], "attribution": attribution}),
        flush=True,
    )


if __name__ == "__main__":
    main()
