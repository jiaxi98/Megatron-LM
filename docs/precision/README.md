# H20 FP8/FP4 training research

## Project baseline

| Item | Value |
| --- | --- |
| Personal fork | https://github.com/jiaxi98/Megatron-LM |
| Upstream | https://github.com/NVIDIA/Megatron-LM |
| Upstream snapshot | `53944a33c6e2b440b1ef67ca2ac12f8017cf77ea` |
| Snapshot branch | `main` |
| Setup date | 2026-10-29 |
| Development branch | `research/fp8-h20` |
| Commit identity | `Jiaxi Zhao <jiaxi.zhao@u.nus.edu>` |
| Target hardware | 64 NVIDIA H20 GPUs; cluster topology not yet inspected |
| Training stack | Megatron Core + Transformer Engine |
| Current stage | G0-local awaiting user review: 40/41 checks passed; BF16 first-step update error exceeded its bound. H20 experiments have not started |

The snapshot is an immutable source reference, not a claim that this upstream
development version and its dependencies have been validated on the cluster.
Select and record the exact Megatron, Transformer Engine, PyTorch, CUDA, cuDNN,
NCCL, driver, and container versions before running the baseline. Record any
later source-base change and rerun the affected validation gates.

## Scope

1. Establish a reproducible BF16 training baseline on H20.
2. Reproduce an official Hopper-compatible TE FP8 recipe in the same trainer.
3. Develop and compare custom quantization policies or kernels against both
   baselines, including forward, input-gradient, and weight-gradient paths.
4. Investigate FP4 quantization numerics and, separately, FP4 activation storage
   or communication with FP8/BF16 compute on H20.
5. Validate native FP4 compute performance separately on suitable Blackwell
   hardware when it becomes available.

H20 supports native FP8 compute, not native NVFP4/MXFP4 Tensor Core compute.
FP4 simulation requires an appropriate emulation implementation; selecting a
Blackwell-only TE recipe does not make it executable on H20. Simulated rounding
and accumulation must be described explicitly when relating results to hardware.

Start with a roughly 1B dense model for diagnosis and repeated experiments, then
confirm the selected recipe at a larger scale such as 7B-8B. These sizes are a
proposed starting point, not finalized run configurations. Inspect cluster
topology, memory use, and representative GEMM shapes before selecting parallel
degrees. Use smaller GPU allocations for ablations and scale up to 64 GPUs for
distributed validation and performance measurements.

## Acceptance gates

The user requested discussion and acceptance at every substantial precision
milestone on 2026-10-29. Complete each stage's evidence before requesting review.
Then mark it `awaiting_review` and pause dependent next-stage work until the user
explicitly accepts it. No gate is accepted by creating this document.

| Gate | Evidence required for discussion | Status |
| --- | --- | --- |
| G0-local: Apple GPU preliminary reference | [Report and evidence](reports/2026-10-29-local-g0/README.md): fixed protocol, CPU/MPS comparisons, paired short training and fresh-process replay; does not replace H20 G0 | awaiting_review; 1 failed criterion |
| G0: BF16 baseline | Pinned environment, exact data/model/run configuration, repeatability, loss and gradient sanity, checkpoint/resume evidence | pending |
| G1: Operator numerics | Quantize/dequantize reference; forward, input-gradient and weight-gradient GEMM comparisons; rounding, scaling, layout and accumulation checks | pending |
| G2: Modules and complete step | Linear, MLP, attention, block and full-step comparisons; activations, logits, loss, gradients and parameter-update differences | pending |
| G3: Distributed and resume | Increasing GPU counts; accumulation, collectives, recomputation, scaling state and RNG restoration; matched effective token batch | pending |
| G4: Training convergence and efficiency | Controlled BF16/official-FP8/candidate runs; validation loss, downstream quality, seed variation and time-to-quality | pending |

Each new FP4 recipe repeats the applicable gates. A short smoke run does not
establish full-pretraining convergence. A change in the model, precision policy,
kernel backend or parallel configuration must identify which previous evidence
still applies and which gates need renewed validation.

## Comparison protocol

- Use independent FP32 references with TF32 disabled, and FP64 where needed for
  small operator cases. Compare hardware kernels using the same quantized
  operands to distinguish quantization error from implementation/accumulation
  error. BF16 is the practical full-training control.
- Measure relative L2 error, cosine similarity, quantization error, saturation,
  underflow and non-finite values as appropriate. Define tensor-specific
  tolerances before scoring a candidate; there is no universal error threshold.
- Test both replay of identical reference inputs and propagation of errors
  through the actual model. Include parameter-update differences, not only loss
  and gradient norms.
- Initially keep FP32 master weights and optimizer states, with appropriate
  higher precision for reductions and sensitive operations. Record operand,
  accumulation, persistent-state and communication dtypes separately.
- Match initialization, tokenizer, data order, effective valid-token batch,
  token budget, optimizer and learning-rate schedule across comparison arms.
  Use multiple seeds for small-model convergence studies and inspect warmup,
  later training and learning-rate decay. Document unavoidable nondeterminism.
- Introduce parallelism, recomputation and quantized communication incrementally.
  Check loss-mask normalization, scaling synchronization, delayed-scaling history
  where applicable, and stochastic-rounding RNG state across checkpoint resume.
- Report tokens/s, peak allocated/reserved memory, total wall time and GPU-hours
  to an agreed quality target. Include conversion and communication costs;
  disable precision-debug instrumentation for timing and report warmup/compile
  time separately. Distinguish matched-configuration and independently tuned
  performance comparisons. Do not extrapolate H100/Blackwell results to H20.

## Review report and decision record

Each report must include:

1. Gate, hypothesis, source commits, environment and hardware topology.
2. Exact commands/configurations, data identity, seeds and effective token counts.
3. References, predeclared tolerances, aggregate results and worst-case failures.
4. Loss/quality/performance evidence, with durable locations for raw artifacts.
5. Limitations, proposed acceptance decision and the next stage it would enable.

Keep raw tensors, checkpoints and datasets in appropriate artifact storage;
commit compact reports, configurations and artifact identifiers in this fork.
Record the user's actual decision, date and conditions below. Do not infer
acceptance from silence or from an automated pass.

| Date | Gate | User decision | Conditions / evidence |
| --- | --- | --- | --- |
| 2026-10-29 | G0-local | Awaiting the user's decision; not accepted | [40/41 criteria passed; update error 0.10269984 exceeds 0.10](reports/2026-10-29-local-g0/README.md). Dependent next-stage work is paused. |

The user authorized preliminary local GPU experiments on 2026-10-29 and requested
atomic feature commits. The [standalone MPS harness](../../tools/precision_local/README.md)
uses its own hash-locked environment and predeclared protocol. Native CUDA/TE and
H20 validation remain separate; local results cannot establish their correctness
or performance.

## Initial source-reading map

- [Megatron low-precision integration](../user-guide/features/low_precision_training.md)
- [FP8 state and utilities](../../megatron/core/fp8_utils.py)
- [FP4 state and utilities](../../megatron/core/fp4_utils.py)
- [Transformer Engine integration](../../megatron/core/extensions/transformer_engine.py)
- [Existing FP8 unit tests](../../tests/unit_tests/test_fp8_utils.py)
- [TE hardware support matrix](https://nvidia.github.io/TransformerEngine/support_matrix.html)
- [TE precision debug tools](https://nvidia.github.io/TransformerEngine/debug/3_api_features.html)
- [TorchTitan NVFP4 reference](https://github.com/pytorch/torchtitan/blob/main/torchtitan/quantization/nvfp4/README.md)
- [Hopper FP4 research reference](https://arxiv.org/abs/2603.02731)

External documentation and main branches change over time. Pin the relevant TE
and comparison-implementation commits when they become experiment dependencies.
