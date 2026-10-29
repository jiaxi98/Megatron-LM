# Local precision baseline

This independent PyTorch reference harness runs preliminary experiments on the
user's Apple GPU. It is not a port of Megatron/Transformer Engine to MPS and does
not validate CUDA kernels, distributed training, native FP8, or native FP4.
Its purpose is to establish the data, comparison and replay protocol before the
H20 experiments. All implementation code in this directory is original.

## Environment

The user explicitly authorized local GPU experiments. Use a separate host MPS
environment; the upstream container workflow remains applicable to CUDA work.
The upstream `pyproject.toml` suppresses installing torch because the NVIDIA
container supplies it. Consequently, `--no-config` is required here.

From the repository root:

```bash
uv venv --no-config --python 3.12 local/precision/.venv
uv pip sync --no-config --require-hashes \
  --python local/precision/.venv/bin/python \
  tools/precision_local/requirements.lock
```

Python is 3.12.14 for the initial experiment. `requirements.in` defines direct
dependencies; `requirements.lock` pins the resolved macOS environment with hashes.
Datasets, raw tensors and checkpoints live under the already-ignored `local/`
directory. Compact results and the acceptance report belong in `docs/precision`.

## Predeclared protocol

`protocol.json` fixes the dataset revision/hash, byte tokenizer, held-out suffix,
model configuration, optimizer/schedule, seeds and preliminary tolerances before
any scored run. These tolerances are engineering checks for the local reference;
they are not general LLM convergence guarantees.

The dataset is the Tiny Shakespeare text distributed in Karpathy's `char-rnn`
repository. Only the dataset is downloaded, from the exact revision in the
protocol; no training implementation is copied. The first 90% of bytes are used
for training and the remaining 10% for validation, with no sampled sequence
crossing the boundary. All 256 byte values are valid tokens; there is no padding.

The local gate is **G0-local**. Its evidence includes CPU FP32 comparisons,
paired MPS FP32/BF16 short training, repeatability, and checkpoint replay in fresh
processes. Results must be reviewed with the user before advancing to low-precision
experiments. The H20 G0 baseline remains pending independently of this gate.

## Reference model

`model.py` implements a small causal decoder with RMSNorm, rotary embeddings,
explicit attention, SwiGLU, residual dropout and untied byte embeddings/output
head. Normalization, softmax, residuals, loss and persistent parameters use FP32.
BF16 autocast affects linear layers and attention matrix multiplications. No
claim of bitwise equivalence to a Megatron model is made at this stage.

```bash
local/precision/.venv/bin/python -m pytest -q -o addopts= \
  tools/precision_local/test_model.py
```

These standalone CPU tests run outside the upstream CUDA/distributed test tree.
