# Plan: Multi-provider async batch infer (half-price)

## Goal

Run the coarse infer stage through each provider's official **async batch API** instead of per-frame sync requests. Batch APIs typically cost ~50% of sync at the price of 24h turnaround, which fits coarse wide-search fine.

**Scope**: coarse single-frame infer only. One frame = one request inside the batch file. Results come back unordered and are reassembled by `frame_number` via `custom_id`. This plan does NOT move rolling-state or sequential temporal reasoning into batch — see `PLAN_packed_multi_frame_infer.md` for the orthogonal "pack N frames into one request" track.

## Current state

Already implemented:

- `GeminiBatchVisionProvider` (`pipeline.py:722`) — official google-genai Batch API, with submit/collect/cancel.
- `OpenAICompatibleBatchVisionProvider` (`pipeline.py:520`) — OpenAI-compatible Batch File API, used by Qwen via DashScope.
- Routing: `local` / `gemini` / `qwen` / `api`, with `submission_mode` = `sync` / `async` / `auto` (`pipeline.py:1938`).
- `build_provider` dispatches `gemini`+`async_batch` → `GeminiBatchVisionProvider`, and `qwen`+`async_batch` → `OpenAICompatibleBatchVisionProvider` (`pipeline.py:2025`).
- Unit test: `tests/test_qwen_batch.py`.

Gap: the generic `api` route (OpenAI-compatible, non-Qwen endpoints) has no async-batch branch — it falls through to sync.

## Phase 1. Generalize OpenAI-compatible batch beyond Qwen

Goal: any OpenAI-compatible endpoint that speaks the Batch File API (OpenAI, DeepSeek, etc.) should be reachable through the existing `OpenAICompatibleBatchVisionProvider`, not just Qwen.

- Extend `build_provider` (`pipeline.py:2031`) to also construct the batch provider for `route == "api"` when `execution_mode == "async_batch"`.
- Mirror the Qwen config surface under `openai_compatible`: `supports_async_batch`, `prefer_async_batch`, `extra_body` (optional).
- Honor `resolve_execution_mode` (`pipeline.py:1938`) unchanged — it already gates on `supports_async_batch`.
- Keep `extra_body` inlining generic (Qwen's `enable_thinking=false` is already handled via inline_extra_body; no new Qwen-specific code paths).

## Phase 2. Provider capability matrix in config

Instead of hard-coded route-name checks, drive batch capability from per-route config so adding a new provider is a config change, not a code change:

```toml
[provider.qwen]
supports_async_batch = true
batch_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
completion_window = "24h"
extra_body = { enable_thinking = false }

[provider.openai_compatible]
supports_async_batch = false   # opt in per-endpoint
```

- Document which providers have been end-to-end verified (Gemini ✅, Qwen ✅, others = opt-in + user verifies).
- Keep `auto` submission mode conservative: only pick `async_batch` when `prefer_async_batch = true`, never silently.

## Phase 3. Operational polish

- CLI parity: `--provider` + `--submission-mode async` work for all routes (already does for gemini/qwen; verify after Phase 1).
- `collect` / `cancel` subcommands (already exist at `pipeline.py:3732` / `pipeline.py:3758`) work generically — no per-route branching.
- Manifest schema (`analysis.batch.json`) is provider-agnostic; confirm Gemini and OpenAI-compatible manifests share the same shape so `status` tooling doesn't need to special-case.
- Document `batch` cost/latency tradeoff in the CLI help and in `pipeline.py` module docstring.

## Phase 4. Safe rollout

- Opt-in via config (`prefer_async_batch = true`) or CLI (`--submission-mode async`). Never flip the default without user action.
- Regression: compare sync vs batch output on a representative video for each route; `analysis.json` must stay schema-compatible.
- Record pricing snapshot (per provider) and regional availability before recommending batch as default for a given route.

## Non-goals

- Parallelism / temporal reasoning across frames → covered by `PLAN_packed_multi_frame_infer.md`.
- Wrapping any CLI (Gemini CLI, Claude CLI, etc.) as an API — rejected due to ToS and account-ban risk.
- Supporting batch APIs that don't expose a file-based submit model.

## Deliverables

- `build_provider` handles `route=api` + `execution_mode=async_batch` (Phase 1).
- Config-driven batch capability flags documented (Phase 2).
- CLI `submit` / `collect` / `cancel` verified across gemini / qwen / generic api (Phase 3).
- Parity regression test on one sample video per newly enabled route (Phase 4).
