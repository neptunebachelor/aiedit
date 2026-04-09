# Backlog

## Requested Changes

### 1. Make provider routing order configurable

Current behavior:

- provider auto-routing order is hard-coded in `pipeline.py`
- current order is `local -> gemini -> qwen -> api`

Requested change:

- allow route priority to be configured from project config and/or CLI
- avoid hard-coding the fallback order in code

### 2. Store per-video metadata beside the source video

Current behavior:

- processing metadata and derived files are written under the code workspace output directory

Requested change:

- do not write video processing metadata into the code directory by default
- place all per-video metadata and derived artifacts in the same directory as the source video
- use a sibling folder named exactly after the source video stem

Desired layout example:

- source video:
  `C:\path\to\videos\VID_20260408_172214_010.mp4`
- artifact folder:
  `C:\path\to\videos\VID_20260408_172214_010\`

This should include:

- extract metadata such as `extract/index.json`
- extracted frame images
- checkpoints
- analysis outputs
- review/edit plans
- render plans and rendered videos

### 3. Add Qwen Batch support for coarse infer

Current behavior:

- `qwen` inference currently sends one synchronous `chat/completions` request per candidate frame
- async batch support is only implemented for `gemini`
- this makes long-video coarse infer slow and more expensive than necessary

Project fit:

- the current coarse infer stage is order-independent per frame
- requests do not carry rolling model state from prior frames
- outputs are merged later by `frame_number` / timestamp, so backend execution order does not need to match submission order

Requested change:

- add OpenAI-compatible Batch support for `qwen3.5-plus` and `qwen3.6-plus`
- keep request/result mapping keyed by frame identity, not response order
- preserve existing output contracts so downstream temporal/review/render stages do not need to change
- preserve resume/restart semantics

Operational notes:

- explicitly set `enable_thinking = false` for coarse filtering workloads unless there is a strong reason not to
- prefer Batch for long-running coarse infer jobs where latency is less important than throughput and cost
- Batch pricing is expected to be lower than real-time synchronous calls, but final implementation should verify current platform billing behavior before rollout

Temporal constraint:

- unordered backend batch execution is acceptable only for order-independent coarse infer requests
- do not use unordered batch execution for future rolling-state or strictly sequential temporal reasoning stages

### 4. Add usable progress reporting for review

Current behavior:

- `review` can run for a long time when generating multiple highlight variants and preview videos
- current output does not provide a clear high-level progress view for the review stage
- users mostly see raw `ffmpeg` encoder logs instead of stage progress

Requested change:

- add explicit review progress reporting similar to `infer.progress.json`
- show which variant is being processed and which sub-step is active
- include preview rendering progress when `--preview` is enabled
- prefer readable stage progress over raw encoder-only logs

Suggested visibility:

- total variants
- current variant index
- current sub-step such as `build_review_outputs`, `render_preview`, or `completed`
- optional machine-readable progress snapshot file for UI or resume diagnostics
