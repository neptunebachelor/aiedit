# Backlog

This file tracks active, user-facing work that is not already complete. Detailed implementation plans live under `plans/todo/`.

## Priority

### P0. Validate one-command background viral run on a real ride video

Why: the CLI path now exists, but the long-term goal is not just plumbing. We need evidence that one command can run unattended, resume cleanly, and produce a usable 30s short-video candidate from real riding footage.

Target command:

```bash
python pipeline.py run \
  --video ./input/lap01.mp4 \
  --provider gemini \
  --viral \
  --background
```

Current state:

- `pipeline.py run` chains extract -> infer -> temporal -> review -> render.
- `--background` writes per-video `runs/<run_id>/job.json`, `pid`, `stdout.log`, and `stderr.log`.
- `pipeline.py status --video ...` reports latest job state and infer/review progress.
- rerunning without `--restart` reuses `extract/index.json`, skips infer when `analysis.json` exists, and otherwise resumes from `infer/frame_decisions.checkpoint.jsonl`.
- `--viral` fills in 30s riding short-video defaults without overriding explicit user options.
- Gemini CLI packed infer is wired into `pipeline.py` for personal-use runs when the `gemini` CLI is available and API credentials are absent.

Remaining work:

- run the target command on real riding footage and confirm `job.json` reaches `completed`.
- verify the final MP4, review JSON, editable plan, and SRT outputs are created under the video artifact directory.
- interrupt and rerun at least once to confirm extract/infer resume behavior on a real artifact tree.
- inspect the resulting 30s candidate for short-video quality; tune `--viral` defaults, prompt labels, or review settings based on the sample.

### P1. Finish generic OpenAI-compatible async batch

Why: Gemini and Qwen batch are in place, but generic `api` still falls back to sync even when the endpoint supports the Batch File API.

Source plan: `plans/todo/PLAN_multi_provider_batch_infer.md`

Current state:

- `GeminiBatchVisionProvider` exists.
- `OpenAICompatibleBatchVisionProvider` exists and is wired for Qwen.
- Qwen batch has focused tests.

Remaining work:

- allow `route == "api"` + `submission_mode == "async"` to use `OpenAICompatibleBatchVisionProvider`
- add `supports_async_batch` / `prefer_async_batch` / optional `extra_body` config for `openai_compatible`
- make collect/cancel generic instead of Gemini/Qwen-only
- document verified providers versus opt-in user-verified endpoints

### P1. Provider-aware packed image prompts for faster sync infer

Why: single-frame sync infer is too slow on real footage. In one observed Gemini run on `VID_20260405_133154_006.mp4`, progress was about 141 / 691 frames after 26m42s, roughly 18s per frame. We need the sync path to send multiple frame images in one prompt whenever the selected provider/model can safely handle it.

User requirement:

- default to 8 images per prompt
- dynamically adjust images-per-prompt by provider and model capability
- read each model vendor's current documentation for context window, image limits, request-size limits, token accounting, and rate-limit behavior before setting defaults

Remaining work:

- add provider/model capability metadata for Gemini, Qwen, and generic OpenAI-compatible providers
- implement packed sync requests for official API paths, not only Gemini CLI
- choose pack size from documented limits, configured defaults, and optional CLI override, with `8` as the default target
- preserve per-frame decisions in checkpoint and final `analysis.json` even when one prompt covers multiple images
- add retry logic that splits a failed pack into smaller packs or single frames on context/request-size errors
- record observed token/request usage when providers expose it, so future runs can tune pack size safely
- update docs with the verified model IDs and pack-size guidance

### P2. Make provider auto-routing order configurable

Why: current auto-routing is hard-coded as `local -> gemini -> qwen -> api`, which makes fallback policy a code edit instead of a config choice.

Remaining work:

- add a config field for route priority, for example `provider.route_order = ["local", "gemini", "qwen", "api"]`
- optionally add a CLI override
- validate route names and preserve the current order as the default
- update CLI help and README docs

### P2. Instrument Gemini CLI image input calibration

Why: byte-identical staged files prove local preparation, but not how Gemini CLI/backend actually attaches, resizes, counts, or batches image inputs.

Remaining work:

- add a calibration path using Gemini API `count_tokens` / usage metadata where available, or Gemini CLI debug logs if usable
- report number of images attached per prompt
- report prompt/image token usage and media-resolution behavior when available
- keep calibration prompts to one pack per prompt by default

### P3. Connect the frontend/backend scaffold to real pipeline data

Why: the React review workspace and FastAPI backend exist, but they are still demo/mock surfaces.

Current state:

- `frontend/src/lib/api.ts` returns mock review data.
- `backend/tasks.py` simulates pipeline progress with sleep calls.

Remaining work:

- replace the demo worker with real extract/infer/review/render invocation
- expose review outputs through backend endpoints
- make the frontend load/save real editable plans
- stream real progress from pipeline progress JSON or backend job state

### Triggered Backlog. Gemini API `response_schema`

Source plan: `plans/todo/PLAN_gemini_api_response_schema.md`

Do this only when one of the plan's pickup triggers fires, such as CLI adherence regression, need for `finish_reason` / `safety_ratings`, or moving from OAuth-bound CLI use to official API use.

## Removed From Active Backlog

- Artifact path rules and repo-local video data root: completed in `plans/done/PLAN_artifact_path_rules.md`.
- Qwen Batch support for coarse infer: implemented; remaining generic batch work is tracked as P1 above.
- Gemini CLI packed infer is wired into `pipeline.py` for the personal-use path, including default pack size 8 and retry-on-missing-frame behavior.
- One-command background run plumbing is in place: `run --background`, resumable `run`, `status`, and `--viral`.

### Section D. Review progress reporting

Original request: add usable progress reporting for `review`, especially for multi-variant review runs and preview rendering.

Completion state:

- implemented `review.progress.json`
- added readable `review.progress` stage logging in `pipeline.py`
- reports variant counts and current sub-steps such as `build_review_outputs`, `render_preview`, and `variant_completed`

This is no longer active backlog work unless a new UI or resume-diagnostics requirement appears.
