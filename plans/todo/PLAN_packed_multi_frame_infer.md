# Plan: Packed multi-frame infer (CLI + official APIs)

## Goal

Pack **N frames into a single vision chat request** so the model can use within-pack temporal context (motion continuity, duplicate suppression, before/after framing) and so we spend 1 request per N frames instead of N. This is orthogonal to batch pricing — packed + batch compose for ~2x savings on top of ~2x from batch.

## Two transports, same shape

This plan covers **two parallel transports** for the same packed contract:

1. **Gemini CLI** (`skills/ride-video-infer/scripts/run_gemini_packed.py`) — the **production path for single-user personal use** (current aiedit case, Douyin-style 30s segment selection). Validated by the Phase 0+1 MVP under `plans/done/PLAN_gemini_schema_harness.md`: at `pack_size=8` + fence-strip + retry-on-truncation, mild-harness `validated_ok = 100%` over 90 calls; per-frame keep drift (~16%) washes out at the 30s window aggregation that downstream review uses. **Currently a standalone script — not yet wired into `pipeline.py`.** Wiring it in (as a `PackedVisionProvider` implementation) is part of this plan.
2. **Official APIs** (Gemini API, OpenAI-compatible, Ollama) — the **path for production / shared / commercial / multi-tenant use**. The CLI is OAuth-bound to one user and intended for interactive personal use; sustained automated traffic from a service-style wrapper isn't its intended shape and risks server-side throttling. Anything beyond one user's personal pipeline must go through an official API.

Phase 0+1 retired the "CLI is technically inadequate" framing: at `pack_size=8` with the implemented retry harness, the CLI is the right tool for the personal use case. The API path's marginal value over CLI-at-p8 is mostly observability (`finish_reason`, `safety_ratings`) and ToS shape for multi-tenant deployment, not adherence — see `plans/todo/PLAN_gemini_api_response_schema.md`.

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

- `OpenAICompatibleVisionProvider` gets `infer_pack` (**done** — `pipeline.py:564`) using the chat-completions multi-image content array (each frame as an `image_url` or inline base64 part).
- **`GeminiCLIPackedVisionProvider` (NEW)** wraps `run_gemini_packed.py`'s pack-call logic (`build_prompt` / `stage_pack_images` / `call_and_parse`) so `pipeline.py` can drive CLI inference end-to-end. Same retry-on-missing-frames semantics as the standalone script.
- `GeminiVisionProvider` gets `infer_pack` using `contents[].parts[]` with multiple `inline_data` parts (official API path).
- `OllamaVisionProvider` gets `infer_pack` for local vision models that accept multiple images.

Frames that don't come back in the response get the recovery the standalone CLI script already does (retry as a smaller pack; remaining missing → `provider_error: missing after retry`). Port that logic when wrapping the CLI; mirror it for the API providers.

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

1. **Phase 1 — sync OpenAI-compatible (DONE)**: `infer_pack` implemented at `pipeline.py:564`. `pack_size` configurable, default 1.
2. **Phase 2 — wire CLI into pipeline (NEXT, personal-use unblock)**: add `GeminiCLIPackedVisionProvider` that wraps `run_gemini_packed.py`'s pack-call logic and surfaces it as a selectable provider in `pipeline.py infer`. Default `pack_size=8` per Phase 0+1 outcomes. End-to-end target: `pipeline.py infer` on the test video produces `frame_decisions.jsonl` via CLI, and downstream `analyze` / `render` consume it unchanged.
3. **Phase 3 — cross-provider API packed**: add `infer_pack` for Gemini official API and Ollama. Verify on the same sample video that `pack_size = 8` gives a kept-frame set within tolerance of the CLI run.
4. **Phase 4 — packed + batch**: extend `OpenAICompatibleBatchVisionProvider._build_request_record` + `parse_openai_batch_results` to handle pack bodies. Gate behind opt-in config.

**Removed:** the previous "Phase 4 — delete the CLI prototype" step. The CLI is the production path for personal use; it stays.

## Validation

- Output parity: on a sample video, `pack_size = 1` vs `pack_size = N` should not change the kept-frame set by more than ~X% (define tolerance).
- Missing-frame recovery: force a model response that omits a frame; verify the fallback `provider_error` decision is emitted and downstream stages handle it.
- Context-limit regression: a pack that overflows the model's image/token budget must fail cleanly with a typed error (not a cryptic provider stack trace).

## Non-goals

- Multi-turn conversations per video.
- Passing previous-pack decisions into the next pack (stateful rolling) — out of scope; batch order is not guaranteed.
- Wrapping a vendor CLI as a multi-tenant / production transport. The CLI provider here is single-user only; multi-tenant routes go through official APIs (Phase 3).

## Deliverables

- `PackedVisionProvider` protocol + `infer_pack` on the OpenAI-compatible route (Phase 1, **done**).
- `GeminiCLIPackedVisionProvider` wrapping `run_gemini_packed.py`, selectable in `pipeline.py infer` (Phase 2).
- Config `infer.pack_size` wired end-to-end (Phase 2).
- Cross-provider API packed support — Gemini API + Ollama (Phase 3).
- Packed-inside-batch request/response handling (Phase 4).
