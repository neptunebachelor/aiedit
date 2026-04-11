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
3. If the user explicitly wants Gemini CLI inference, use the Gemini CLI workflow below.
4. If `analysis.json` already exists and the user wants review/render, skip visual inference and continue with `review` or `render`.

## Shared Workflow

1. Work from the project root containing `pipeline.py`.
2. Ensure candidate frames exist:

```powershell
python pipeline.py extract --video <video> --config <config>
```

   If the user gives an existing `extract/index.json`, use it directly.

3. Read the extract index and inspect only frames where `candidate` is true and `image_path` is non-empty.
4. Process frames in deterministic order by `frame_number` unless resuming from an existing decisions file. Skip frames already present in the decisions file/checkpoint. In comparative mode, pack consecutive candidate frames into groups.
5. Produce one decision per candidate frame in JSONL. Each line must include `frame_number` and the decision schema:

```json
{"frame_number": 0, "keep": false, "score": 0.1, "labels": [], "reason": "low editorial value", "discard_reason": "pit or waiting area"}
```

6. Materialize project outputs with:

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
4. Append one JSONL decision per frame immediately under the video's output directory, for example `codex.frame_decisions.jsonl`.
5. Run `apply_decisions.py` to generate `analysis.json`, `segments.raw.json`, SRTs, and `frame_decisions.jsonl`.

For large jobs, keep the packed comparative loop but checkpoint after every pack. Do not spend visual effort on non-candidate frames unless the user asks for audit/debugging.

## Gemini CLI Workflow

Use this when the user wants Gemini CLI or wants to spend Gemini plan quota through the CLI.

First decide where the skill is running:

- If you are already inside an interactive Gemini CLI chat, do not start another `gemini` process from shell or Python. A nested Gemini CLI commonly opens an interactive approval UI and stalls the run.
- If you are in a normal shell, Codex, or another non-Gemini host, use the external runner path below. Do not invoke `gemini <image-path> <prompt>` directly; that starts the Gemini CLI TUI instead of a bounded non-interactive request.

### External Runner Path

Use this path only from a normal shell or Codex, not from inside Gemini CLI.

1. Check whether `gemini` is installed and authenticated:

```powershell
Get-Command gemini -ErrorAction SilentlyContinue
gemini --version
```

2. If unavailable, say so and fall back to Codex-in-the-loop or the existing Gemini API provider.
3. Run the checkpointed non-interactive runner:

```powershell
python <skill_dir>/scripts/run_gemini_packed.py --index <extract/index.json> --pack-size 20 --model gemini-2.5-flash-lite --apply
```

The runner copies pack images into an ASCII workspace, references them with Gemini CLI `@images/...` file attachments, requests strict JSON, writes raw Gemini output beside each pack for debugging, appends normalized JSONL decisions, and can call `apply_decisions.py` with `--apply`.

For calibration, start small:

```powershell
python <skill_dir>/scripts/run_gemini_packed.py --index <extract/index.json> --pack-size 5 --max-frames 20 --model gemini-2.5-flash-lite
```

Do not use a bare manual test such as:

```powershell
gemini "C:\path\to\frame.jpg" "Describe this image."
```

Use non-interactive `-p` plus an `@` file reference instead:

```powershell
gemini --yolo --output-format json -p "Describe this image briefly: @C:/path/to/frame.jpg"
```

### In-Session Gemini CLI Path

Use this path only when you are already inside Gemini CLI and the current chat session itself can inspect image inputs.

1. Prefer the non-nested pack workflow. The current Gemini CLI session must perform the visual inference directly. Do not launch another `gemini` process from Python or shell.
2. Keep the user prompt short and stable. The detailed scoring, validation, and file-writing rules belong in this skill, not in each test prompt.
3. Generate repeatable in-session prompts with:

```powershell
python <skill_dir>/scripts/build_gemini_in_session_prompt.py --packs-dir <video_dir>/infer/packs --start-pack 1 --end-pack 3
```

The generated prompt is the only text that should be pasted into Gemini CLI for that range. For the same arguments and paths, it must be byte-for-byte stable.

For pack-size calibration, generate and paste one pack per prompt by setting `--start-pack` and `--end-pack` to the same pack number. A range prompt includes every image in the requested range, so `--start-pack 1 --end-pack 2` for 26-frame packs attaches 52 images and does not test a 26-image single-request pack.

When Gemini receives the generated prompt, it must execute this skill's `Gemini In-Session Pack Inference Contract`.

### Gemini In-Session Pack Inference Contract

This contract is the canonical prompt body for in-session Gemini CLI pack inference. Do not duplicate it into ad hoc test prompts.

Prepare packs:

```powershell
python <skill_dir>/scripts/prepare_packs.py --index <extract/index.json> --pack-size 20 --start-pack 1 --end-pack 5
```

For each prepared pack:

- Read `manifest.json`.
- Use the pack's `manifest.json` as the source of truth for expected frame numbers, timestamps, image paths, and output location.
- Read every file under the pack's `images/` directory as separate image inputs.
- Compare the frames in the current Gemini CLI session.
- Write `response.json` directly in that pack directory as UTF-8 without BOM.
- Write a top-level JSON array only. Do not include Markdown, commentary, code fences, or response envelopes.
- Include exactly one object for every `frame_number` in the pack manifest, with no missing, extra, or duplicate frame numbers.
- Use this object schema: `frame_number`, `keep`, `score`, `labels`, `reason`, `discard_reason`.
- `score` must be a number from `0.0` to `1.0`; `labels` must be an array of short strings.
- Set `keep` to exactly match the score threshold: `keep` must be `true` when `score >= 0.65`, and `false` when `score < 0.65`.
- In an ordinary 20-frame pack, keep only the strongest 1-4 frames unless many frames are genuinely strong.
- Keep `reason` and `discard_reason` short and plain.
- After writing all `response.json` files for the requested range, run the BOM cleanup script for that exact range before validation:

```powershell
python <skill_dir>/scripts/strip_response_bom.py --packs-dir <video_dir>/infer/packs --start-pack 1 --end-pack 5
```

  This script only removes the leading UTF-8 BOM bytes (`EF BB BF`) from `response.json` files; it does not change the JSON decisions. Running this Python script is allowed inside Gemini CLI because it does not launch another Gemini model process.
- Do not write temporary Python scripts containing decisions.
- Do not write decisions through PowerShell string literals when paths contain non-ASCII characters; use Python `json.dump(..., ensure_ascii=False, indent=2)` when a file write is needed.

Validate and optionally append valid decisions:

```powershell
python <skill_dir>/scripts/validate_pack_response.py --packs-dir <video_dir>/infer/packs --start-pack 1 --end-pack 5 --append-decisions <video_dir>/infer/gemini_cli.frame_decisions.jsonl
```

After all packs are valid, materialize project outputs:

```powershell
python <skill_dir>/scripts/apply_decisions.py --index <extract/index.json> --decisions <video_dir>/infer/gemini_cli.frame_decisions.jsonl --provider gemini_cli
```

The older `run_gemini_packed.py` runner is only for running from a normal shell outside Gemini CLI. Do not run it inside Gemini CLI because it launches another Gemini CLI process.

If the current Gemini CLI session cannot inspect local image files directly, stop and tell the user to run the External Runner Path from a normal shell instead of attempting a nested `gemini` command.

## File API Packed Inference

Use this path when the user wants Codex or Gemini tooling to control tests while the model reads previously uploaded image files by provider file reference. This is different from the Gemini CLI `@local/image.jpg` workflow and different from the pipeline's async JSONL batch upload. Here, every extracted screenshot is uploaded first, then comparative packed inference references provider file IDs or URIs.

Default behavior:

- Upload frames from `extract/index.json` with `candidate=true` and `image_path` by default.
- Use `--include all` during upload when the user explicitly asks to upload every screenshot file with `image_path`.
- Inference still uses candidate frames by default, even when the upload manifest includes all frames. Pass `--include-all-frames` to `run_file_api_packed.py` only for an explicit all-frame audit.
- Keep one file upload manifest per provider under `<video_dir>/infer/file_uploads.<provider>.json`.
- Keep one inference decisions file per provider under `<video_dir>/infer/<provider>_file_api.frame_decisions.jsonl`.

Upload file references:

```powershell
python <skill_dir>/scripts/upload_frame_files.py --index <extract/index.json> --provider openai --include all --reuse
python <skill_dir>/scripts/upload_frame_files.py --index <extract/index.json> --provider gemini --include all --reuse
```

For OpenAI, upload images with the Files API using `purpose="vision"` and reference each image in the Responses API as an `input_image` with `file_id`. For Gemini, upload images with the Gemini Files API and reference each image by `file_uri` plus MIME type in `generateContent`.

Run controlled packed inference:

```powershell
python <skill_dir>/scripts/run_file_api_packed.py --upload-manifest <video_dir>/infer/file_uploads.openai.json --provider openai --model gpt-4.1-mini --pack-size 5 --max-frames 20 --prompt-variant calibration --dry-run
python <skill_dir>/scripts/run_file_api_packed.py --upload-manifest <video_dir>/infer/file_uploads.openai.json --provider openai --model gpt-4.1-mini --pack-size 5 --max-frames 20 --prompt-variant calibration --apply
```

Use the same shape for Gemini:

```powershell
python <skill_dir>/scripts/run_file_api_packed.py --upload-manifest <video_dir>/infer/file_uploads.gemini.json --provider gemini --model gemini-2.5-flash-lite --pack-size 5 --max-frames 20 --prompt-variant calibration --apply
```

The runner writes `run_manifest.json` and per-pack raw responses under `<video_dir>/infer/file_api_runs/`. The run manifest must record all controlled variables: provider, model, pack size, pack range, max frames, prompt variant, temperature, minimum keep score, retry policy, candidate frame numbers, `upload_manifest_hash`, and `prompt_hash`.

File API prompt discipline:

- Start with calibration: `--pack-size 5 --max-frames 20 --prompt-variant calibration`.
- Then run the normal pass: `--pack-size 20` with the same model, temperature, and prompt variant.
- Use a larger pack size only after dry-run planning and calibration responses validate.
- Change one variable per test round. If testing model quality, keep pack size, prompt variant, temperature, frame range, and upload manifest fixed. If testing pack size, keep model, prompt variant, temperature, frame range, and upload manifest fixed.
- The prompt must require one JSON object for every listed `frame_number`, no markdown, no comments, no code fences, and keys `frame_number`, `keep`, `score`, `labels`, `reason`, `discard_reason`.
- If a pack response is malformed, missing frames, or duplicates frames, retry once with the strict retry prompt. If still invalid, write valid reject decisions with `discard_reason` beginning `skill_error:` so downstream stages remain usable.

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
- Keep all business artifacts under `<video_dir>/infer`, not under the repository root or `.gemini/tmp`.
- Do not create one-off files such as `write_batch_responses.py` in the project root.

## Output Contract

The final user-visible result should name the generated `analysis.json` path and summarize how many candidate frames were decided and how many were kept.

If any frames fail inference, write a valid reject decision with `discard_reason` beginning with `provider_error:` or `skill_error:` so downstream stages remain usable.
