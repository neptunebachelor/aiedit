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

---

## Outcomes (Phase 0 + 1, executed 2026-04-26)

MVP: `skills/ride-video-infer/scripts/mvp_gemini_cli_adherence.py`
Subject: 30 min ride video, drm-hwaccel CPU extract @ 1fps → 360 candidate frames.
Two CLI runs, 90 calls each, `gemini-2.5-flash`, `--yolo --output-format json`, no retry / no repair / no babysitting.

### Headline numbers

| Pack size | `validated_ok` (mild harness: fence strip + brute `[...]` extract) | `validated_ok` (strict naked) | Truncated runs | Hallucinated frame_numbers | Pydantic field errors | p50 latency |
|---|---|---|---|---|---|---|
| 12 | 87.8% | 80.0% | 10/90 | 1/90 | 0 | 35.4s |
| 8  | 100%  | 84.4% | 0/90  | 0/90 | 0 | 32.7s |

`mean_per_pack_keep_consistency` (3-repeat agreement): 0.84 (p12), 0.86 (p8). 16% per-frame keep flip rate.

### Findings that contradict the plan above

1. **Red-line table is not applicable here.** Table assumed failures = format drift. In our data, failures are 100% output-budget exhaustion (truncation) — re-prompt or two-pass cannot recover lost content. The right knob is pack size, not harness thickness.
2. **CLI hidden output cap ≈ 2200 chars (~500-600 tokens).** All p12 truncations land below the largest p12 success (3029 chars). p8 has comfortable headroom and never trips.
3. **Per-frame keep drift (16%) is a non-issue for this pipeline.** Downstream review aggregates ~30 frames into a 30s window; flip noise washes out at window scale. 3-way voting is over-engineering.
4. **CLI 0.27 envelope drops `finish_reason`, `safety_ratings`, all `candidates[*]` diagnostics.** Detection of truncation is heuristic-only (response doesn't end with `]` or emitted_count < expected_count). This is a permanent CLI limitation, not a config issue.
5. **`validated_ok` reported as a single number is misleading** — strict-naked vs. fence-stripped vs. brute-force-extracted give 80 / 88 / 88 at p12. Always report which harness layer each number assumes.

### Decisions taken

- **CLI minimum harness = pack_size 8 + fence strip + retry-on-truncation.** No two-pass, no 3-way voting, no repair-prompt. Implemented in `run_gemini_packed.py` via separate PR.
- **Phase 2 API path (`response_schema` on `pipeline.py:836`) deferred to backlog.** See `plans/todo/PLAN_gemini_api_response_schema.md`. Marginal value over CLI-at-p8 is now mostly observability, not adherence.
- **Phase 3 (repair-pass design) dropped.** Not needed at p8.
- **Phase 4 (observability) reduced to: synthesize a `finish_reason`-equivalent enum from response shape** (`TRUNCATED`, `INCOMPLETE_ARRAY`, `HALLUCINATED_FRAMES`, `SCHEMA_DRIFT`, `OK`). For monitoring only, not for control flow. Deferred until we see a regression.

### Reproducibility

- Raw `runs.jsonl` + `stats.json` + per-call `raw/*.stdout.txt` live at:
  - p12: `<root>/videos/VID_20260413_160844_002/infer/mvp_runs/20260426T045718Z/`
  - p8:  `<root>/videos/VID_20260413_160844_002/infer/mvp_runs/20260426T062342Z/`
- These are **not** committed — they live under `RIDE_VIDEO_DATA_ROOT` (off-repo).
- Re-running needs: working `gemini` CLI (OAuth-bound), 360+ extracted frames, `python skills/ride-video-infer/scripts/mvp_gemini_cli_adherence.py --index <index.json> --pack-size N --num-packs 30 --repeats 3`.
