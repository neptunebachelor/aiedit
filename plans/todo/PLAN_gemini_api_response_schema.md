# Plan: Gemini API `response_schema` (backlog)

## Status

**Backlog.** Originally Phase 2 of `plans/done/PLAN_gemini_schema_harness.md`. Deferred because the CLI path at `pack_size=8` reaches 100% `validated_ok` with a one-line fence-stripping harness — see Outcomes section of that plan. Picking this back up only when one of the triggers below fires.

## Pickup triggers

Do this when **any** of the following becomes true:

- CLI route's rolling `validated_ok` regresses below ~95% on a representative video (e.g. after a Gemini model upgrade silently changes output verbosity).
- We need `finish_reason` / `safety_ratings` for production debugging — i.e. a failure mode shows up that we can't attribute from response shape alone.
- We need controllable `temperature=0` / `seed` / `max_output_tokens` (CLI exposes none of these).
- We move off the OAuth-bound CLI for ToS / multi-tenant reasons — the API path is the natural replacement transport.
- We want decoder-level enum enforcement for fields with closed value sets (e.g. `best_frame_id: Literal[...]`).

## Scope when picked up

Promote `pipeline.py:836` from `response_mime_type="application/json"` to `response_schema=<Pydantic model>`:

- Define Pydantic models per inference kind (coarse single-frame, packed multi-frame).
- Pass the Pydantic class directly as `response_schema` in `generation_config` (current `google-genai` SDK supports this — verify version pin first).
- Use `Literal[...]` / `Enum` for any closed choice space; that's the constraint Gemini enforces strongest.
- Order fields so reasoning-style fields precede conclusion fields (`reason` before `keep`) — implicit CoT improves quality.
- Keep Pydantic validation on `resp.parsed` for soft constraints Gemini drops (`ge`/`le`, `max_length`, `pattern`, `additionalProperties:false`).
- Consume `finish_reason` from `candidates[0]` and persist alongside decisions for monitoring.

## Open questions to resolve at pickup time

- Current `google-genai` SDK version in `requirements.txt` — does it accept Pydantic class as `response_schema` directly, or do we hand-build the JSON Schema dict?
- Does `response_schema` with nested arrays of per-frame objects hold up at `pack_size = 8–12`, or does adherence collapse with array length? (Open question from the original Phase 0 plan — re-test on this path.)

## Out of scope

- Replacing the CLI route for personal-use single-user deployment. CLI at `pack_size=8` is good enough; this plan is purely additive.
- Multi-provider batch / packed inference — see `PLAN_multi_provider_batch_infer.md` and `PLAN_packed_multi_frame_infer.md`.
