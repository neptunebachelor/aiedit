# Plan: Packed multi-frame infer via official APIs

## Goal

Pack **N frames into a single vision chat request** so the model can use within-pack temporal context (motion continuity, duplicate suppression, before/after framing) and so we spend 1 request per N frames instead of N. This is orthogonal to batch pricing — packed + batch compose for ~2x savings on top of ~2x from batch.

## What this replaces

Supersedes the Gemini CLI packed prototype at `skills/ride-video-infer/scripts/run_gemini_packed.py`.

**The CLI-wrapping route is dropped.** Wrapping `gemini` CLI (OAuth-bound free tier) into an API and feeding it pipeline traffic violates Gemini CLI / Google Generative AI Additional Terms: the free tier is for interactive personal use, not automated batch. Detection signals are obvious (request cadence, prompt fingerprinting, CLI telemetry, single-account volume) and the worst case is a full Google account suspension — not just API quota loss. Not worth the savings when official Batch APIs already give half-price.

The packed *idea* stays. The CLI *transport* goes. All packed inference must go through the official API of whichever provider we route to.

## Scope

- Pack up to N frames (tune per model, probably 5–15) into one chat completion whose user message contains all N images + a list of `(frame_number, timestamp_seconds)` tuples.
- Model returns a JSON array of decisions, one per frame, keyed by `frame_number`.
- Works with any vision-capable chat API: Gemini, Qwen-VL, GPT-4o, Claude, local vision models via Ollama.
- Composes with sync OR async-batch submission (see `PLAN_multi_provider_batch_infer.md`).

Out of scope:

- Rolling-state across packs (each pack is self-contained).
- Replacing single-frame coarse infer globally — packed is opt-in for users who want better duplicate/continuity judgment or lower request count.

## Design

### 1. New provider capability

Add `PackedVisionProvider` protocol alongside `VisionProvider` / `AsyncBatchVisionProvider` in `pipeline.py:246`:

```python
class PackedVisionProvider(Protocol):
    def infer_pack(
        self,
        frames: list[dict[str, Any]],   # each has frame_number, timestamp_seconds, image bytes/path
        *,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:            # list of decision dicts keyed by frame_number
        ...
```

Implementations:

- `OpenAICompatibleVisionProvider` gets `infer_pack` using the chat-completions multi-image content array (each frame as an `image_url` or inline base64 part).
- `GeminiVisionProvider` gets `infer_pack` using `contents[].parts[]` with multiple `inline_data` parts.
- `OllamaVisionProvider` gets `infer_pack` for local vision models that accept multiple images.

Frames that don't come back in the response get a `provider_error: missing from output` fallback decision (port the recovery logic from `run_gemini_packed.py:287`).

### 2. Prompt contract

A shared prompt builder in `analyze_video.py` (next to `build_user_prompt`) produces:

- System prompt: same decision schema as single-frame, but instructs that the output is a JSON array with one entry per input frame, keyed by `frame_number`.
- User prompt: lists `Frame <n>, t=<s>s, image #<k>` tuples in order, followed by the N image parts in matching order.
- Strict output: `[{"frame_number", "keep", "score", "labels", "reason", "discard_reason"}, ...]`.

Keep the single-frame prompt untouched; packed is a new builder.

### 3. Config

```toml
[infer]
pack_size = 1        # 1 = single-frame (current behavior), >1 = packed
pack_max_pixels = 1280   # downscale knob; packed prompts hit context limits fast
```

- `pack_size = 1` must stay the default so existing runs are unchanged.
- Validate against a per-provider `max_images_per_request` hint (optional config) to fail fast instead of at the API.

### 4. Composition with batch

When `pack_size > 1` AND `submission_mode = async`, each batch-file request body contains one **pack** (not one frame). `custom_id` becomes `pack_<first_frame>_<last_frame>` or similar, and `parse_openai_batch_results` must return a flat dict of per-frame decisions (expand packs at parse time).

This is the biggest implementation subtlety — the batch provider currently assumes one `custom_id` = one `frame_number`.

### 5. Rollout phases

1. **Phase 1 — sync single-provider**: implement `infer_pack` for the OpenAI-compatible route (Qwen-VL is a good first target — multi-image input, cheap, official API). `pack_size` configurable, default 1.
2. **Phase 2 — cross-provider**: add `infer_pack` for Gemini and Ollama, verify on a sample video that `pack_size = 5` gives comparable keep/discard to `pack_size = 1`.
3. **Phase 3 — packed + batch**: extend `OpenAICompatibleBatchVisionProvider._build_request_record` + `parse_openai_batch_results` to handle pack bodies. Gate behind opt-in config.
4. **Phase 4 — delete the CLI prototype**: remove `skills/ride-video-infer/scripts/run_gemini_packed.py` once the API-based path matches or beats it on a representative video.

## Validation

- Output parity: on a sample video, `pack_size = 1` vs `pack_size = N` should not change the kept-frame set by more than ~X% (define tolerance).
- Missing-frame recovery: force a model response that omits a frame; verify the fallback `provider_error` decision is emitted and downstream stages handle it.
- Context-limit regression: a pack that overflows the model's image/token budget must fail cleanly with a typed error (not a cryptic provider stack trace).

## Non-goals

- Multi-turn conversations per video.
- Passing previous-pack decisions into the next pack (stateful rolling) — out of scope; batch order is not guaranteed.
- Wrapping any vendor CLI. See top of document.

## Deliverables

- `PackedVisionProvider` protocol + `infer_pack` on at least one OpenAI-compatible route (Phase 1).
- Config `infer.pack_size` wired end-to-end.
- Cross-provider packed support (Phase 2).
- Packed-inside-batch request/response handling (Phase 3).
- CLI prototype removed (Phase 4).
