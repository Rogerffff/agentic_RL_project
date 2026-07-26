# Repository Agent Handoff

## CaRR DeepSearch Is the Custom Project

This repository is a `verl` fork whose main custom project is under
`examples/carr_deepsearch/`. When a task concerns CaRR, DeepSearch, agentic
reinforcement learning, training history, evaluation, resume writing, or
interview preparation, read these files first:

1. `examples/carr_deepsearch/AGENT.md` for the current project status and asset index.
2. `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md` for claims and numbers that may be used externally.
3. `examples/carr_deepsearch/docs/training_full_history_20260404.md` for the complete training chronology and actual configurations.
4. `examples/carr_deepsearch/docs/eval_analysis_20260415.md` for the final measured evaluation results.
5. `examples/carr_deepsearch/docs/agent_loop_run_explained.md` and
   `examples/carr_deepsearch/docs/eval_result_deep_analysis_followup.md` for
   corrected metric semantics and sample-level caveats.

## Current Project State

- The implementation, supervised fine-tuning, synchronous reinforcement learning,
  asynchronous reinforcement learning, final internal evaluation, and
  `BrowseComp subset256 64k` external evaluation are complete.
- There is no required GPU run remaining for the current resume-ready project.
  Full BrowseComp and 128K-context evaluations are optional future extensions.
- The authoritative branch is `feature/carr-deepsearch`. It contains the complete
  tracked `verl` fork, the CaRR project code, asynchronous infrastructure changes,
  and committed documentation.
- Do not describe the project as a paper reproduction. Do not present
  `BrowseComp subset256` as the full benchmark.

## Repository and Local-Only Assets

- A Git clone does not include ignored local artifacts such as
  `examples/carr_deepsearch/eval_results/`, most raw logs, Parquet data, API
  secrets, model checkpoints, or `examples/carr_deepsearch/CaRR_repo/`.
- Before assuming that an artifact was lost, read
  `examples/carr_deepsearch/docs/new_mac_migration_20260726.md`.
- `examples/carr_deepsearch/CaRR_repo/` is a historical remote-machine snapshot,
  not the current source of truth. Edit the primary worktree instead.
- Never commit `.env`, `examples/carr_deepsearch/api.md`, API keys, or model
  checkpoints to the public repository.

## Historical Documents

Training plans and GPU runbooks document how the project was developed, but many
of them were already executed. Prefer the final-state documents listed above
before treating an older plan as current work.
