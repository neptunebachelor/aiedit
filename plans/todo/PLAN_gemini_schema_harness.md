# Plan: Gemini structured-output harness

## Goal

Decide how thick a validation/repair layer to wrap around Gemini inference so that downstream code can trust the returned JSON (field names, enums, numeric ranges, no extra fields). The decision must be driven by **measured adherence rates**, not guesswork.

## Background: what each layer actually guarantees

| Layer | What it constrains |
|---|---|
| `gemini` CLI `--output-format json` | Only the **outer** CLI envelope (`{response, stats, error}`). `.response` is still free-form model text — may be Markdown, fenced ```json blocks, extra prose, or invalid JSON. |
| Gemini API `response_mime_type="application/json"` (already used at `pipeline.py:836`) | Model will emit JSON-shaped text, but field names, enums, and ranges are **not** enforced. |
| Gemini API `response_schema=<Pydantic class>` | Decoder-level constraint on **structure and enums**. Still does **not** enforce soft constraints (`ge/le`, `max_length`, `pattern`) — those are stripped from the schema Gemini accepts. |

Consequence: **a harness is mandatory on any path**. The only variable is its thickness.

## Hard constraints Gemini's schema layer ignores

Must be enforced in Pydantic on our side, even when using `response_schema`:

- Numeric bounds: `ge`, `le`, `gt`, `lt`
- String length: `min_length`, `max_length`
- Regex: `pattern`
- `additionalProperties: false` (Gemini may add extra keys)
- Recursive types / `$ref`
- Complex `oneOf` / discriminated unions (support is weak)

Gemini **does** respect: `Literal[...]` / enum, `required`, object nesting, simple arrays, basic types. So candidate-ID style fields (`best_frame_id: Literal["f_3a7b", ...]`) are the most robust pattern.

## Phase 0. MVP: measure adherence before designing the harness

Goal: produce a real number for "schema adherence rate" under each path, then size the harness to the gap.

- Write a throwaway `mvp.py` that:
  1. Runs a fixed set of ~30–50 tasks × 3 repeats.
  2. Calls the target path (CLI or API) with `temperature=0`.
  3. Logs per-run JSONL with: `finish_reason`, `raw_text`, `had_markdown_fence`, outer-parse ok, inner-parse ok, Pydantic `validated_ok`, per-field validation errors, latency.
  4. Does **no** retry, **no** repair, **no** prompt babysitting — we want the naked failure rate.
- Stats to compute from `runs.jsonl`:
  - `parsed_ok / total` — JSON-level success
  - `validated_ok / total` — schema-level success (the number that matters)
  - `had_markdown_fence / total` — CLI-specific noise
  - `finish_reason` distribution (watch `SAFETY`, `MAX_TOKENS`, `RECITATION`)
  - Same-task 3-run `best_frame_id` consistency
- Keep raw failing outputs on disk. Future prompt/model changes must be regression-tested against this set.

## Phase 1. Size the harness by the measured gap

Red lines, applied per path:

| `validated_ok / total` | Action |
|---|---|
| ≥ 98% | Naked call + single `try/except`. No repair needed. |
| 90–98% | Parse + 1 repair pass (temp=0, restated schema, previous failure reason). |
| 80–90% | Two-pass: (a) free-form reasoning, (b) separate formatter call that only turns reasoning into schema JSON. Isolate contexts between passes. |
| < 80% | Don't thicken harness — fix prompt, enum the choice space, or change model. Harness cannot rescue a weak task definition. |

## Phase 2. Path choice for aiedit

`pipeline.py:836` already uses `response_mime_type=application/json` via the API. Promote to `response_schema` there:

- Define Pydantic models per inference kind (coarse single-frame, packed multi-frame, rolling-state).
- Pass the model class directly as `response_schema` in `generation_config` (new `google-genai` SDK supports Pydantic class → schema conversion).
- Prefer `Literal[...]` / `Enum` for any fixed choice (frame IDs, decision categories), since that's the strongest constraint Gemini enforces.
- Keep Pydantic validation on `resp.parsed` for the soft constraints Gemini drops (see list above).
- Order fields so reasoning-style fields precede conclusion fields (e.g. `reason` before `best_frame_id`) — improves quality via implicit CoT.

For the CLI-based path under `skills/ride-video-infer/scripts/run_gemini_packed.py`:

- No schema enforcement available. Minimum harness: outer `json.loads` → strip ```json fence → inner `json.loads` → Pydantic validate → 1 repair retry on failure.
- If Phase 0 shows CLI path `validated_ok` < 95%, migrate the skill to API like `pipeline.py`.

## Phase 3. Repair-pass design (when needed)

Anti-pattern: "please fix your previous output." The model tends to copy the bad output and tweak one field.

Correct form: **restart the task** with:
- Original task context
- Explicit list of what failed (e.g. `score must be int 0..100`, `best_frame_id must be in [...]`)
- Schema restated
- `"Return only JSON. No markdown."`
- `temperature=0`

Cap at 1 repair. Second failure → log + return structured error, don't retry forever.

## Phase 4. Observability

- Persist every failed raw output under a dated directory for later analysis.
- Track `validated_ok` rate as a rolling metric per model version + per task kind. Model upgrades and prompt edits can silently regress adherence; without a baseline we won't notice.
- Alert if rolling rate drops >5pp from the Phase-0 baseline.

## Non-goals

- Not building a general-purpose LLM validation framework. Harness should stay <150 lines.
- Not adding retry loops that hide failure rates. Every retry must be counted in the metrics.
- Not babysitting prompts to squeeze the last 1–2%. If the measured gap is small, accept it and handle failures as errors.

## Open questions

- Does the current `google-genai` SDK version in `requirements.txt` accept Pydantic classes as `response_schema` directly, or do we need to hand-build the JSON Schema dict?
- For packed multi-frame (`PLAN_packed_multi_frame_infer.md`), does `response_schema` with nested arrays of per-frame objects hold up, or does adherence collapse with array length?

## References

- Gemini structured output schema support: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini CLI headless JSON envelope: https://google-gemini.github.io/gemini-cli/docs/cli/headless.html
- Existing API JSON call site: `pipeline.py:834`
- Existing CLI call site: `skills/ride-video-infer/scripts/run_gemini_packed.py:304`
