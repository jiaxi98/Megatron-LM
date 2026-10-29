# Repository Guidelines

## H20 precision research fork

This personal fork develops and validates FP8 training on 64 NVIDIA H20 GPUs,
with FP4 numerical simulation and storage/communication research as subsequent
work. Read [the experiment plan](docs/precision/README.md) before working on this
project. The upstream guidelines below continue to apply within this scope.

- Use `Jiaxi Zhao <jiaxi.zhao@u.nus.edu>` for the author and committer identity of
  new project commits. Configure this per repository; do not rewrite upstream
  history. Keep the upstream sign-off and cryptographic signing requirements.
- Develop on `research/fp8-h20` or task branches based on it. Push project work to
  the personal fork, `jiaxi98/Megatron-LM`. Upstream contribution instructions
  apply when an upstream contribution is requested; this research setup does
  not request an upstream PR.
- **The user explicitly requires a pause at every substantial precision
  acceptance milestone.** Prepare the completed experiments and a reviewable
  report, mark the gate `awaiting_review`, and stop dependent next-stage work.
  Discuss the evidence with the user and obtain their explicit acceptance
  before advancing. Record the decision and any conditions in the experiment
  plan. Passing automated checks alone does not constitute acceptance.
- Gates cover the BF16 baseline, operator numerics, module/full-step numerics,
  distributed/resume correctness, and training convergence. A new FP4 recipe
  goes through the applicable gates again. Routine fixes and experiments within
  the current stage may proceed without repeated permission requests.
- Establish numerical tolerances and comparison conditions before evaluating a
  candidate. Never relax thresholds or replace reference results just to make
  a candidate pass. Show failures, uncertainty, and missing evidence in reviews.
- Distinguish H20-native FP8 compute, FP4 simulation, and FP4 storage/communication
  with higher-precision compute. Do not describe H20 experiments as native FP4
  Tensor Core performance measurements.

## Skills

The `skills/` directory contains structured guides for common tasks (running
tests, building containers, managing dependencies, submitting SLURM jobs, etc.).
**Always read the relevant `SKILL.md` before starting any task it covers —
skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- When rebasing, cherry-picking or committing, sign the commits with both `-s` and `-S`: `-s` adds the required
  `Signed-off-by` trailer, and `-S` signs the commit so copy-pr-bot and
  `/ok to test` can verify the pushed commit without manually specifying the SHA.
  `/ok to test` is not required for PR authors in `NVIDIA/mcore-engineers` as long as
  they sign their commits with `-S`.
- Authors in `NVIDIA/mcore-engineers` should sign using their NVIDIA emails so they
  are automatically added to the right user groups on the internal Slack
  workspace.
- Before merging any PR, check for open PRs whose GitHub base/diffbase is
  `pull-request/<this PR number>`. Retarget every such dependent PR to `main`
  before merging; otherwise GitHub may close it and discard review discussion
  or approvals.
- Read [docs/developer/contribute.md](./docs/developer/contribute.md) for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Provenance

- Best effort to avoid copying code from other repositories. Implement changes
  independently.
- If copying code is necessary, add the source URL, revision, and license to
 all headers of files that contain such code. In addition, ensure the
 "Code provenance" section is filled in within the PR description.
- Preserve all required license and copyright notices and attribution for any
  external code in the proposed changes.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.

### Megatron Core Process Groups

- In `megatron/core` production code, avoid adding new direct reads of global
  process groups from `parallel_state` (for example,
  `parallel_state.get_tensor_model_parallel_group()` or directly imported
  `get_*_group()` helpers). Prefer accepting a `ProcessGroupCollection` or an
  explicit `torch.distributed.ProcessGroup` from the caller and passing that
  through.
- Allowed compatibility points include `megatron/core/parallel_state.py`,
  `megatron/core/process_groups_config.py`, initialization/bootstrap code that
  materializes a `ProcessGroupCollection` from MPU globals, tests, docs, and
  migration fallbacks with an explicit comment.
- This guidance targets Megatron Core library code. Do not apply it to
  `megatron/training` or other training-loop code unless the PR explicitly
  opts into that migration.
- In reviews, flag new direct `parallel_state.get_*_group()` usage in
  `megatron/core` unless it is one of the compatibility points above. This is
  advisory guidance, not a CI gate.
