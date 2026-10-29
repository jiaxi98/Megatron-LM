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

The byte embedding uses FP32 one-hot GEMM because PyTorch 2.15 MPS rejects
`embedding_dense_backward_mps` with strict deterministic algorithms enabled.
The forward and repeated-token weight gradients are checked against CPU embedding
lookup. This equivalent but more expensive reference implementation is practical
for 256 tokens; it is not intended for production-sized vocabularies. The failed
initial attempt is retained separately, and acceptance thresholds are unchanged.

```bash
local/precision/.venv/bin/python -m pytest -q -o addopts= \
  tools/precision_local/test_model.py
```

These standalone CPU tests run outside the upstream CUDA/distributed test tree.

## Training and replay

`baseline.py` runs FP32/BF16 training with FP32 AdamW states, an explicit warmup
and cosine schedule, fixed validation samples, loss/gradient diagnostics and
serialized model/optimizer/CPU/MPS/data RNG state. Unsupported MPS operations
must fail rather than silently fall back to the CPU. The runner checks that
projection outputs actually use the requested dtype.

```bash
local/precision/.venv/bin/python tools/precision_local/baseline.py train \
  --dtype bf16 --seed 17 --out local/precision/manual-bf16-17
local/precision/.venv/bin/python tools/precision_local/baseline.py train \
  --dtype bf16 --seed 17 --resume local/precision/manual-bf16-17/step-64.pt \
  --out local/precision/manual-bf16-17-resume
local/precision/.venv/bin/python -m pytest -q -o addopts= \
  tools/precision_local/test_baseline.py
```

`baseline.py audit --out <new-directory>` compares a deterministic evaluation-mode
step on CPU FP32, MPS FP32 and MPS BF16, including gradients and AdamW parameter
updates. This validates the local baseline, not a new low-precision recipe.

All outputs require new directories. Checkpoints validate the exact protocol
hash, seed, dtype and device before restoration. The same saved sample indices
and initial parameter hashes establish paired-run equivalence. Final FP32
evaluation of both learned models separates weight quality from evaluation
precision. Both the execution-precision and FP32-evaluation loss gaps must meet
the protocol's paired-loss bound.

Elapsed time includes compilation, diagnostic synchronization, in-loop validation
and checkpoint I/O. It is a diagnostic runtime, not a steady-state performance
benchmark. MPS memory is sampled live allocation, not a true allocator peak.

## Complete local experiment

```bash
local/precision/.venv/bin/python -m pytest -q -o addopts= tools/precision_local
local/precision/.venv/bin/python tools/precision_local/run_suite.py \
  --out local/precision/runs/g0-local-v1
```

The suite launches a fresh process for the audit, each of six paired training
runs, two same-seed repeats, and two restarts from step 64. It compares complete
final checkpoints (including numerical history) for exact replay. Manifest,
environment, per-step results, checkpoint hashes, summary and validation curves
are retained under the output directory. Only compact evidence is committed.

The summary requires all expected audit/replay evidence and reports failed
criteria without weakening them. Even a complete automated pass is marked
`awaiting_review`; it never grants user acceptance. Full-run wall time is also
affected by thermal state and experiment order, so no FP32/BF16 speed ranking is
derived from this preliminary suite.

## Diagnosing an update-error failure

```bash
local/precision/.venv/bin/python tools/precision_local/diagnose_update.py \
  --out local/precision/runs/g0-local-update-diagnosis
```

This post-run diagnostic reruns the same single-step comparison, attributes
update-error energy to gradient sign flips, and checks the analytic first AdamW
update difference. Common weight decay cancels for identical initial weights.
It reports per-parameter contributions and never alters or reclassifies the
original acceptance threshold. Its formula is tested against FP64 AdamW.

## Context-length study

The user authorized extending context to 4K while recording gradient differences.
`context_protocol.json` predeclares 128/256/512/1024/2048/4096 byte-token contexts,
three seeds and 64 optimizer steps. Each update processes exactly 4,096 targets
via one-sequence microbatches and gradient accumulation. For a given seed/step,
all contexts use the same 4,097-byte source span: only its partition into causal
contexts changes. Positions reset within each independent sequence.

Dropout is disabled for both arms to keep its shape-dependent RNG behavior out
of this comparison. Both arms run on MPS, with unchanged architecture, FP32
parameters/optimizer states and the base precision boundaries. This is a new
controlled experiment, not an extension of the previous dropout-enabled curves.

Record two distinct gradient comparisons before clipping: actual FP32/BF16
training trajectories at every step, and BF16 recomputation at the same FP32
weights at steps 1/8/16/32/64. The latter isolates arithmetic differences from
evolving-weight differences. Store global and per-parameter metrics, fixed-window
validation curves, and selected raw gradients. No numerical threshold is chosen
after seeing the data. The earlier 10% update criterion remains unchanged and
unaccepted; completion of this additional study requires another user review.
