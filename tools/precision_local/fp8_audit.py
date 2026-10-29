"""Score local FP8 operators against independent CPU casts and FP64 GEMMs."""

import argparse
import json
import math
from pathlib import Path

import torch
from baseline import configure, environment, write_json
from context_metrics import tensor_metrics
from fp8 import EmulatedFP8Linear, QuantizationStats, current_quantize
from run_suite import file_sha256


def cast_reference(value, format_name):
    """Independent oracle: CPU native cast; double-precision scale calculation."""
    value = value.detach().cpu()
    dtype = getattr(torch, "float8_" + format_name)
    amax = value.abs().max().item()
    scale = 2.0 ** math.ceil(math.log2(amax / torch.finfo(dtype).max)) if amax else 1.0
    return (value / scale).to(dtype).float() * scale


def audit(protocol, base):
    configure(base, "mps", 17)
    rows, checks = [], {}
    limits = protocol["operator_acceptance"]
    for shape in protocol["operator_shapes_m_k_n"]:
        m, k, n = shape
        for seed in protocol["operator_seeds"]:
            generator = torch.Generator().manual_seed(seed)
            x = torch.randn(m, k, generator=generator)
            w = torch.randn(n, k, generator=generator) / math.sqrt(k)
            dy = torch.randn(m, n, generator=generator) / m
            xq, wq, gq = (
                cast_reference(v, fmt) for v, fmt in ((x, "e4m3fn"), (w, "e4m3fn"), (dy, "e5m2"))
            )
            key = f"{m}-{k}-{n}/seed-{seed}"
            for label, value, expected, fmt in (
                ("input", x, xq, "e4m3fn"),
                ("weight", w, wq, "e4m3fn"),
                ("gradient", dy, gq, "e5m2"),
            ):
                actual, stats = current_quantize(value.to("mps"), fmt)
                checks[f"{key}/quantizer/{label}"] = torch.equal(actual.cpu(), expected)
            recorder = QuantizationStats()
            recorder.enabled = True
            layer = torch.nn.Linear(k, n, bias=False, device="mps")
            with torch.no_grad():
                layer.weight.copy_(w)
            candidate = EmulatedFP8Linear(layer, recorder, "linear")
            xx = x.to("mps").requires_grad_()
            yy = candidate(xx)
            yy.backward(dy.to("mps"))
            outputs = {
                "forward": yy.detach().cpu(),
                "dgrad": xx.grad.cpu(),
                "wgrad": layer.weight.grad.cpu(),
            }
            quantized = {
                "forward": xq.double() @ wq.double().T,
                "dgrad": gq.double() @ wq.double(),
                "wgrad": gq.double().T @ xq.double(),
            }
            unquantized = {
                "forward": x.double() @ w.double().T,
                "dgrad": dy.double() @ w.double(),
                "wgrad": dy.double().T @ x.double(),
            }
            metrics = {}
            for name, actual in outputs.items():
                implementation = tensor_metrics(actual, quantized[name])
                quantization = tensor_metrics(quantized[name], unquantized[name])
                checks[f"{key}/{name}/implementation"] = (
                    implementation["rel_l2"] <= limits["quantized_operand_gemm_vs_fp64_rel_l2"]
                )
                checks[f"{key}/{name}/quantization"] = (
                    quantization["rel_l2"]
                    <= limits["quantization_vs_unquantized_fp64_rel_l2"][name]
                )
                metrics[name] = {"implementation": implementation, "quantization": quantization}
            layer.weight.grad = None
            xx.grad = None
            repeated = candidate(xx)
            repeated.backward(dy.to("mps"))
            checks[f"{key}/exact_repeat"] = all(
                torch.equal(a, b)
                for a, b in zip(
                    outputs.values(),
                    (repeated.detach().cpu(), xx.grad.cpu(), layer.weight.grad.cpu()),
                )
            )
            rows.append(
                {
                    "shape": shape,
                    "seed": seed,
                    "metrics": metrics,
                    "quantization_stats": recorder.rows,
                }
            )
    return {
        "status": "passed_pending_batch_review" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "cases": rows,
        "environment": json.loads(json.dumps(environment())),
        "scope": "Numerical emulation only; no native FP8/TE/H20 acceptance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("fp8_protocol.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    base_path = args.protocol.parent / protocol["base_protocol"]
    if file_sha256(base_path) != protocol["base_protocol_sha256"]:
        raise ValueError("Base protocol changed")
    args.out.mkdir(parents=True, exist_ok=False)
    result = audit(protocol, json.loads(base_path.read_text()))
    result["protocol_sha256"] = file_sha256(args.protocol)
    write_json(args.out / "result.json", result)
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}))
    if result["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
