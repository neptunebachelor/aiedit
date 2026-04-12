---
name: ride-video-infer
description: Run motorcycle ride video highlight inference for this project using Codex visual analysis or Gemini CLI instead of the normal API/local provider. Use when the user asks to use a skill, Codex, or Gemini CLI to inspect extracted frame images, score highlight moments, create frame decisions, resume inference from extract/index.json, or produce pipeline-compatible analysis.json artifacts.
---

# Ride Video Infer

## Purpose

Use this skill to turn extracted motorcycle video frames into the same decision artifacts that `pipeline.py infer` normally creates. Prefer the existing project pipeline for extraction, review, and render; use Codex or Gemini CLI only for the visual decision step.

Default to comparative packed inference: send multiple candidate frame images as separate image parts in one model call, keep each image at the extracted resolution, and ask the model to compare frames within the group while returning one JSON decision per frame. Do not create contact sheets or downscale beyond the project's extracted frame size.

## Decision Path

1. If the user wants normal API/local inference, run `python pipeline.py infer ...` with the requested provider.
2. If the user explicitly wants Codex visual inference, use the Codex-in-the-loop workflow below.
3. If the user is in an interactive Gemini CLI session, use the **In-Session Gemini CLI Workflow** below. This is the primary path to leverage Gemini's visual capabilities while avoiding API isolation and File API limitations.
4. If `analysis.json` already exists and the user wants review/render, skip visual inference and continue with `review` or `render`.

## Shared Workflow

1. Work from the project root containing `pipeline.py`.
2. Treat the repo-local data root as canonical: `<repo_root>/.video_data`. For a source video, pipeline artifacts live under `<repo_root>/.video_data/videos/<video_slug>/`, extracted frame images live under `<repo_root>/.video_data/frames/<video_slug>/`, and skill inference details live under `<repo_root>/.video_data/videos/<video_slug>/infer/`.
3. Ensure candidate frames exist:

```powershell
python pipeline.py extract --video <video> --config <config>
```

   If the user gives an existing `extract/index.json`, use it directly. The apply/materialization scripts will still write derived outputs to the canonical `.video_data` artifact directory for that video.

4. Read the extract index and inspect only frames where `candidate` is true and `image_path` is non-empty. If an older index points at the original external video tree, resolve frame basenames from `.video_data/frames/<video_slug>/` or `.video_data/frames/` before failing.
5. Process frames in deterministic order by `frame_number` unless resuming from an existing decisions file. Skip frames already present in the decisions file/checkpoint. In comparative mode, pack consecutive candidate frames into groups of **20 to 26 frames**.
6. Produce one decision per candidate frame in JSONL. Each line must include `frame_number` and the decision schema:

```json
{"frame_number": 0, "keep": false, "score": 0.1, "labels": [], "reason": "low editorial value", "discard_reason": "pit or waiting area"}
```

7. Materialize project outputs with:

```powershell
python skills/ride-video-infer/scripts/apply_decisions.py --index <extract/index.json> --decisions <decisions.jsonl> --config <config> --provider codex
```

Use `--provider gemini_cli` when decisions came from Gemini CLI.

## Comparative Packing Policy

Default mode is multi-image comparative inference. Send each frame as its own image input with nearby text metadata, not as a contact sheet. The model should see full extracted-frame detail and compare frames inside the group.

Do not create contact sheets. Contact sheets reduce per-frame detail and make frame IDs harder to audit.

Use one-frame strict mode only for calibration, retries, uncertain frames, or final review of top segments.

When the user wants to maximize a fixed Gemini CLI request quota, calculate a packed size before running:

```powershell
python skills/ride-video-infer/scripts/recommend_pack_size.py --index <extract/index.json> --daily-requests 1500
```

Use the recommended packed size for a coarse pass, then optionally run one-frame review on the highest-score and uncertain frames.

Recommended defaults:

- `compare-20x`: default. Good request efficiency while staying well below inline image and context limits for the project's extracted JPGs.
- `compare-32x`: aggressive. Use after 2-3 calibration packs return complete, correctly keyed JSON.
- `strict-1x`: review/retry mode.

For each packed request, instruct the model to compare all images in the group and output one decision for every `frame_number`. It should keep only the strongest 1-4 frames in an ordinary 20-frame group unless many frames are genuinely strong.

## Comparative Prompt Contract

Every packed request must include:

- A text header with the video type, positive labels, negative labels, and scoring rubric.
- For each image, a small text item immediately before that image: `Frame <frame_number>, timestamp <timestamp_seconds>s`.
- The image as an independent image input at the extracted resolution.
- A final instruction to return a JSON array only.

Require this output shape:

```json
[
  {
    "frame_number": 123,
    "keep": true,
    "score": 0.82,
    "labels": ["apex", "high_speed"],
    "reason": "strong lean and track tension",
    "discard_reason": ""
  }
]
```

The response must contain exactly one object for each requested `frame_number`, with no missing or duplicate frame numbers. If a response is missing frames, duplicated, malformed, or contains commentary outside JSON, retry the pack once with a stricter prompt; if it still fails, write `skill_error:` reject decisions for the missing frames and continue.

## Scoring Rules

Use the prompt settings from the config when available. The default road labels prioritize bends, scenery, traffic, overtakes, group riding, tunnel transitions, water views, mountain views, and sunset. Track configs prioritize corner entry, apex, corner exit, full throttle, high speed, late braking, handlebar wobble, and near-barrier moments.

Score from `0.0` to `1.0`.

- `0.85-1.0`: obvious short-form hook, strong motion, near pass, dramatic bend, apex, transition, scenery reveal, or high-speed tension.
- `0.65-0.84`: usable highlight, visually distinct, worth keeping if it fits duration.
- `0.35-0.64`: maybe useful context but not a highlight.
- `0.0-0.34`: reject.

Set `keep` true only when `score` is at or above the config's `selection.min_keep_score` unless the user asks for looser recall.

Keep `reason` and `discard_reason` short. Avoid Markdown in decision files.

## Codex-In-The-Loop Workflow

Use this when the user says to let Codex infer the frames.

1. Read `extract/index.json`.
2. Iterate over candidate frames in ascending `frame_number` order, using comparative packs by default.
3. Inspect images as separate image inputs. Do not combine them into a single contact sheet.
4. Append one JSONL decision per frame under the video's canonical infer directory, for example `<repo_root>/.video_data/videos/<video_slug>/infer/codex.frame_decisions.jsonl`.
5. Run `apply_decisions.py` to generate `analysis.json`, `segments.raw.json`, SRTs, and `frame_decisions.jsonl`.

For large jobs, keep the packed comparative loop but checkpoint after every pack. Do not spend visual effort on non-candidate frames unless the user asks for audit/debugging.

## In-Session Gemini CLI Workflow

This is the standard workflow when running inside a Gemini CLI session. It avoids API isolation and File API issues by performing visual inference directly within the current chat session.

### Execution Steps

1. **Prepare Packs**: Group candidate frames into packs of **20 to 26 frames**. This size maximizes model context usage while maintaining high accuracy for comparative scoring.
   ```powershell
   python skills/ride-video-infer/scripts/prepare_packs.py --index manifest/index.json --pack-size 26 --end-pack 1
   ```
2. **Build In-Session Prompt**: Generate a deterministic prompt that includes local image references (`@path`) and frame metadata (number, timestamp).
   ```powershell
   python skills/ride-video-infer/scripts/build_gemini_in_session_prompt.py --packs-dir .video_data/videos/<slug>/infer/packs --start-pack 1 --end-pack 1
   ```
3. **Model Inference**: The current Gemini session inspects the provided images and executes the **In-Session Pack Inference Contract**.
4. **Result Capture**: Write the model's JSON array output to `response.json` in the pack directory. Ensure no markdown or commentary is included.
5. **Standardized Cleanup**: Run the BOM cleanup script to ensure the JSON is parseable.
   ```powershell
   python skills/ride-video-infer/scripts/strip_response_bom.py --packs-dir .video_data/videos/<slug>/infer/packs --start-pack 1 --end-pack 1
   ```
6. **Validation & Append**: Validate that the response contains the exact set of requested frames, then append to the main `.jsonl` decisions file.
   ```powershell
   python skills/ride-video-infer/scripts/validate_pack_response.py --packs-dir .video_data/videos/<slug>/infer/packs --start-pack 1 --end-pack 1 --append-decisions .video_data/videos/<slug>/infer/gemini_cli.frame_decisions.jsonl
   ```
7. **Materialization**: Convert the decision list into the final `analysis.json` required by downstream pipeline stages.
   ```powershell
   python skills/ride-video-infer/scripts/apply_decisions.py --index manifest/index.json --decisions .video_data/videos/<slug>/infer/gemini_cli.frame_decisions.jsonl --provider gemini_cli
   ```

### In-Session Pack Inference Contract

This contract is the mandatory prompt structure for in-session visual inference.

- **Source of Truth**: Always use the pack's `manifest.json` for frame numbers and timestamps.
- **Image Input**: Each frame must be provided as a separate image input. Never use contact sheets.
- **Output Format**: Return a **strictly plain JSON array**. Do not use markdown code fences, headers, or conversational text.
- **Scoring Logic**:
  - `keep` must be `true` ONLY if `score >= 0.65`.
  - In a 20-26 frame group, keep only the strongest 1-4 frames unless multiple frames are exceptionally high quality.
  - Score range: `0.0` (reject) to `1.0` (perfect highlight).
- **Schema**: Each object in the array must contain: `frame_number`, `keep`, `score`, `labels`, `reason`, `discard_reason`.

### Model Selection And Output Rules

1. Select the model deliberately:
   - Use `--model flash-lite` for the cheapest/highest-throughput rough highlight scoring when it is available in the user's plan.
   - Use `--model flash` when flash-lite is too noisy on a short calibration sample.
   - Avoid `--model pro` for bulk frame scoring unless the user asks for quality over quota.
   - Use `--model auto` only when the user wants Gemini CLI to spend the plan quota opportunistically across available models.
2. For each pack, ask for strict JSON only. Include separate image inputs plus per-image `frame_number` and timestamp metadata. Never use a contact sheet unless the user explicitly asks.
3. Parse stdout or `response.json` carefully. If the CLI wraps output in a JSON event or response envelope, extract the model text first, then parse the inner decision JSON.
4. Write normalized JSONL and run `apply_decisions.py --provider gemini_cli`.

For plan-quota runs, count one packed group as one model request. Check current usage with Gemini CLI stats when possible and stop gracefully before exhausting the user's preferred model quota.

## Progress Discipline

When running inside Gemini CLI:

- Work in bounded ranges such as 5-10 packs at a time.
- After each range, run `validate_pack_response.py` and stop if any pack is invalid.
- Never claim a pack is complete unless `response.json` exists and validation passes.
- Prepared packs with only `manifest.json` and `images/` are not inferred yet.
- Keep all business artifacts under `<repo_root>/.video_data/videos/<video_slug>/infer`, not under the repository root, the original external video directory, or `.gemini/tmp`.
- Do not create one-off files such as `write_batch_responses.py` in the project root.

## Output Contract

The final user-visible result should name the generated `analysis.json` path and summarize how many candidate frames were decided and how many were kept.

If any frames fail inference, write a valid reject decision with `discard_reason` beginning with `provider_error:` or `skill_error:` so downstream stages remain usable.
