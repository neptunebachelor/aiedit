# Qwen Batch Plan

## Scope

Implement Qwen Batch support for the current coarse infer stage only.

Important constraint:

- Batch execution order does not need to match submission order for the current coarse infer stage
- results must be mapped back by `frame_number`
- this plan does not move rolling-state or strongly sequential temporal reasoning into unordered batch execution

## Phase 1. Match current sync infer behavior

- implement a Qwen batch provider using the OpenAI-compatible Batch File API
- support `qwen3.5-plus` and `qwen3.6-plus`
- reuse stable frame request keys so results can be reassembled by `frame_number`
- keep the current single-frame prompt and decision schema unchanged
- add manifest / collect / cancel flows parallel to the existing Gemini batch workflow
- explicitly set `enable_thinking = false` for coarse filtering unless intentionally overridden

## Phase 2. Safe rollout

- make Qwen batch opt-in first via config or CLI
- validate that batch-collected `analysis.json` stays schema-compatible with current sync outputs
- compare a sample video in sync vs batch mode for keep/discard stability
- verify current pricing, free-tier behavior, and regional support before defaulting to batch

## Phase 3. Better temporal understanding on fewer requests

- keep coarse wide search cheap
- move stronger temporal reasoning to later stages on fewer candidate windows
- prefer contact sheets or short temporal-window analysis for later-stage model calls
- avoid introducing rolling-state dependencies into unordered batch execution by default

## Deliverables

- Qwen batch submission support
- Qwen batch collection support
- Qwen batch cancellation support
- config or CLI switch for sync vs batch
- regression check comparing sync and batch outputs on a representative video
