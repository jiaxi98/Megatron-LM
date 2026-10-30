# H20 FP8/FP4 training research

## Experiment reports after cloning

All six numerical-alignment experiment categories are documented in this branch.
They are grouped into **three report directories**, with links to the relevant
sections for each comparison. Reports, configurations, figures, compact JSON/CSV results,
source revisions and artifact hashes are committed to Git and can be read
without installing PyTorch or downloading a dataset.

Clone the research branch of the personal fork:

```bash
git clone --branch research/fp8-h20 https://github.com/jiaxi98/Megatron-LM.git
cd Megatron-LM
```

Start with this file, `docs/precision/README.md`. The upstream snapshot on `main`
does not contain this research work. On GitHub, select `research/fp8-h20` when
browsing the fork.

| # | Experiment category | Setup | Detailed report and evidence |
| --- | --- | --- | --- |
| 1 | FP32/BF16 single-step alignment | Identical weights/input; CPU FP32 versus MPS FP32/BF16; logits, gradients and AdamW updates | [Report section](reports/2026-10-29-local-g0/README.md#numerical-results), [metrics and checks](reports/2026-10-29-local-g0/metrics.json) |
| 2 | FP32/BF16 short-training alignment | Context 128; 3 seeds; 128 steps per arm; validation loss and exact repeat/resume | [Report section](reports/2026-10-29-local-g0/README.md#short-training-and-exact-replay), [validation curves](reports/2026-10-29-local-g0/validation_curves.png) |
| 3 | BF16 context-length alignment through 4K | Contexts 128/256/512/1024/2048/4096; 3 seeds; 64 steps; 4096 targets/update; same-weight and trajectory gradients | [Report section](reports/2026-10-29-context-4k/README.md#results), [summary](reports/2026-10-29-context-4k/summary.json), [gradient CSV](reports/2026-10-29-context-4k/gradient_history.csv) |
| 4 | FP8 quantizer alignment | E4M3FN/E5M2 rounding versus independent native CPU float8 casts; encodings, midpoints, neighbors, subnormals, signed zero and extremes | [Report section](reports/2026-10-29-local-fp8/README.md#operator-and-quantizer-evidence), [test cases](../../tools/precision_local/test_fp8.py), [test execution record](reports/2026-10-29-local-fp8/unit-tests.txt) |
| 5 | FP8 Linear operator alignment | Forward, input gradient and weight gradient; 3 matrix shapes × 3 seeds; separate quantization error from GEMM implementation error against FP64 | [Report section](reports/2026-10-29-local-fp8/README.md#operator-and-quantizer-evidence), [all operator measurements](reports/2026-10-29-local-fp8/operators.json) |
| 6 | FP8 full-model short-training alignment | FP32/BF16/FP8-emulated arms; contexts 128/1024/4096; 3 seeds; 64 steps; same-weight gradients and same-state AdamW updates | [Report section](reports/2026-10-29-local-fp8/README.md#full-model-and-training-evidence), [summary](reports/2026-10-29-local-fp8/summary.json), [per-parameter CSV](reports/2026-10-29-local-fp8/parameter_probes.csv), [quantization statistics](reports/2026-10-29-local-fp8/quantization_stats.csv) |

These are local CPU/MPS reference experiments on a small byte-level decoder.
FP8 is numerical emulation with dequantized FP32 GEMMs; native H20/TE, FP4,
distributed training and long-horizon convergence remain separate work.

Supporting diagnostics are also committed:

- [BF16 first-step AdamW update attribution](reports/2026-10-29-local-g0/README.md#failed-update-criterion-attribution)
  and [per-parameter measurements](reports/2026-10-29-local-g0/update_diagnosis.json).
- [BF16 2K gradient-peak replay](reports/2026-10-29-context-4k/README.md#the-2k-trajectory-outlier)
  and [exact replay evidence](reports/2026-10-29-context-4k/peak_replay.json).
- [FP8 4K gradient-peak replay](reports/2026-10-29-local-fp8/README.md#trajectory-peak-diagnostic),
  [repeat/resume checks](reports/2026-10-29-local-fp8/replay.json), and
  [unchanged FP32/BF16 baseline checks](reports/2026-10-29-local-fp8/baseline_crosscheck.json).

Each report includes experimental conditions, reference definitions, thresholds,
results, failures or limitations, provenance and reproduction instructions.
Historical report statuses describe the decisions at the time of measurement;
the [current review policy](#acceptance-gates) supersedes their earlier pause
instructions without changing failed numerical criteria.

**Reading versus reproducing:** all committed reports, figures and compact
results are available immediately after cloning. Multi-GB raw checkpoints and
some raw gradient tensors remain in the original machine's gitignored `local/`
directory; their hashes are recorded, but those files are not downloaded by
`git clone`. Dataset content and the Python environment are also not committed.
To rerun experiments, follow the [isolated harness setup and commands](../../tools/precision_local/README.md)
on compatible hardware, using new output directories. Reading reports requires
no local GPU, and the index/report links are repository-relative.

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
| Current stage | Local FP8 numerical emulation and training integration completed through 4K, with evidence retained for batch review. H20 experiments have not started |

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

The user superseded the earlier milestone-pause instruction on 2026-10-29:
continue implementation, record passing experiments, and defer their review
until a later batch discussion. Mark new passing experiments
`passed_pending_batch_review` and continue without individual success reports
or approval stops. Failures remain failures and must be investigated without
relaxing thresholds. Historical reports retain their original decisions;
continued research does not retroactively accept the earlier failed criterion.

| Gate | Evidence required for discussion | Status |
| --- | --- | --- |
| G0-local: Apple GPU preliminary reference | [Report and evidence](reports/2026-10-29-local-g0/README.md): fixed protocol, CPU/MPS comparisons, paired short training and fresh-process replay; does not replace H20 G0 | awaiting_review; 1 failed criterion |
| G0-local-context: context and gradients | [Report and evidence](reports/2026-10-29-context-4k/README.md): 18 constant-token MPS pairs through 4096 byte tokens, same-weight/trajectory gradients, exact replay of the 2K outlier | awaiting_review; 115/115 controls passed; numerical acceptance not granted |
| G1/G2-local-FP8: numerical emulation | [Report and evidence](reports/2026-10-29-local-fp8/README.md): E4M3/E5M2 quantizer/operator oracles, 27 training arms through 4K, matched gradient/AdamW probes and exact repeat/resume | passed_pending_batch_review; 359/359 checks; native H20/TE acceptance remains pending |
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
| 2026-10-29 | G0-local-context | User requested increasing context to 4K and recording gradient differences | Authorizes this additional local study; does not accept or waive the preceding failed criterion, or authorize native FP8 claims. |
| 2026-10-29 | G0-local-context | Awaiting the user's decision; not accepted | [Completed 4K report](reports/2026-10-29-context-4k/README.md). Final 4K same-weight gradient difference 0.593–1.097%; a 2K trajectory peak of 107.3% has 0.584% same-weight difference on exact replay. Dependent next-stage work is paused. |
| 2026-10-29 | Review policy, superseding earlier pause | User instructed continued implementation and later batch review of passing precision experiments | Record complete evidence, continue passing stages without per-experiment approval/reporting. Earlier failures and numerical thresholds remain unchanged. Historical pause statements above no longer control continued research. |
| 2026-10-29 | G1/G2-local-FP8 | Automated pass recorded for later batch review; no user acceptance inferred | [Local FP8 report](reports/2026-10-29-local-fp8/README.md). Thresholds unchanged; update/trajectory diagnostics retained. No individual approval stop under the updated instruction. |

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
