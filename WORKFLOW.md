# Workflow

This project now uses a cross-platform Python pipeline instead of a Windows-only workflow wrapper.

The main entry point is:

```bash
python pipeline.py <stage> ...
```

For one-shot processing you can also run the full pipeline:

```bash
python pipeline.py run --video ./input/ride01.mp4
```

To generate multiple 30-second continuous highlight variants with a stronger short-video bias:

```bash
python pipeline.py run \
  --video ./input/ride01.mp4 \
  --selection-mode single_continuous \
  --top-highlights 5 \
  --prompt-preset douyin_riding \
  --caption-style douyin \
  --caption-detail-prefix "Hook: "
```

## Overview

There are four primary stages:

1. `extract`
2. `infer`
3. `temporal`
4. `review`
5. `render`

And one utility stage:

6. `edit`

## Stage 1: Extract

Goal:

- turn a video into LLM-readable image files
- keep only candidate frames after the coarse blur / motion / duplicate gates

Primary inputs:

- `--video`
- `--frame-interval-seconds`
- `--sample-fps`
- `--max-frames`
- `--resize-for-llm`
- `--config`

Examples:

```bash
python pipeline.py extract --video ./input/ride01.mp4 --frame-interval-seconds 1.0
```

```bash
python pipeline.py extract --video ./input/lap01.mp4 --config ./config.track.toml --frame-interval-seconds 0.5
```

Outputs:

```text
<source_video_dir>/<video_stem>/
`-- extract/
    |-- frames/
    `-- index.json
```

`index.json` contains:

- sampled frame timestamps
- blur / diff / duplicate metrics
- whether each sampled frame is a candidate
- the saved image path for each candidate frame

## Stage 2: Infer

Goal:

- call a vision model on extracted frames
- route automatically between local Ollama and API inference when configured
- turn frame-level decisions into merged source-time segments

Supported provider types:

- `ollama`
- `gemini`
- `openai_compatible`

Primary inputs:

- `--video`
- `--extract-index`
- `--provider`
- `--provider-type`
- `--api-base`
- `--api-key`
- `--api-key-env`
- `--model`
- `--config`

Examples:

```bash
python pipeline.py infer --video ./input/lap01.mp4 --config ./config.track.toml
```

By default this prefers local Ollama, then Gemini 3 Flash, then Qwen, and finally a generic OpenAI-compatible API only if that provider is configured for vision support.

For local development, the pipeline also loads a workspace `.env` file automatically. `GEMINI_BASE_URL` and `DEEPSEEK_BASE_URL` can override provider endpoints without editing TOML.

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider gemini
```

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider api \
  --api-base https://api.deepseek.com \
  --model deepseek-chat \
  --api-key-env DEEPSEEK_API_KEY
```

Async batch is opt-in. Submit a Gemini batch with:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider gemini \
  --submission-mode async
```

Then collect it later:

```bash
python pipeline.py collect \
  --manifest ./input/lap01/analysis.batch.json
```

Outputs:

```text
<source_video_dir>/<video_stem>/
|-- analysis.json
|-- frame_decisions.jsonl
|-- segments.raw.json
|-- segments.raw.srt
|-- highlights.json
`-- highlights.srt
```

When async batch mode is used, `infer` writes `analysis.batch.json` first and `collect` materializes the final `analysis.json`.

`segments.raw.srt` stays on the original source-video timeline.

## Stage 3: Temporal

Goal:

- convert coarse frame-level infer outputs into structured temporal-window analysis
- emit candidate segments and window-level scores
- generate contact sheets for top windows
- produce a refined 30s final highlight proposal (`highlight.final.json`)

Example:

```bash
python pipeline.py temporal \
  --input ./input/lap01/analysis.json \
  --top-k 5 \
  --window-seconds 3 \
  --window-stride 1.5 \
  --contact-sheet-frames 6 \
  --final-duration-seconds 30
```

Outputs:

```text
<source_video_dir>/<video_stem>/
|-- candidate_segments.json
|-- temporal_windows.json
|-- highlight.final.json
`-- contact_sheets/
```

## Stage 4: Review

Goal:

- compress the raw highlight segments into a shorter candidate edit
- emit files that humans can review and modify
- optionally render a low-resolution preview video

Primary inputs:

- `--input`
- `--target-seconds`
- `--caption-mode`
- `--preview`
- `--preview-resolution`
- `--config`

Caption modes:

- `score`
- `reason`
- `human`

Preview resolutions:

- `540p`
- `720p`
- `1080p`
- `source`

Example:

```bash
python pipeline.py review \
  --input ./input/lap01/analysis.json \
  --target-seconds 30 \
  --caption-mode human \
  --preview \
  --preview-resolution 540p
```

Outputs:

```text
<source_video_dir>/<video_stem>/
|-- highlights_30s.review.json
|-- highlights_30s.editable.json
|-- highlights_30s.final.srt
|-- highlights_30s.source.srt
`-- highlights_30s.preview.mp4
```

Semantics:

- `*.final.srt` is rebased to the compact final timeline
- `*.source.srt` stays on the original source timeline
- `*.editable.json` is the file meant for user intervention

## Stage 5: Render

Goal:

- cut and concatenate the final MP4 from the reviewed edit plan

Primary inputs:

- `--input`
- `--stem`
- `--resolution`
- `--ffmpeg`
- `--config`

Render resolutions:

- `source`
- `540p`
- `720p`
- `1080p`

Examples:

```bash
python pipeline.py render \
  --input ./input/lap01/highlights_30s.editable.json \
  --stem lap01_final \
  --resolution source
```

```bash
python pipeline.py render \
  --input ./input/lap01/highlights_30s.editable.json \
  --stem lap01_final_720p \
  --resolution 720p
```

Outputs:

```text
<source_video_dir>/<video_stem>/
|-- lap01_final.json
|-- lap01_final.final.srt
|-- lap01_final.source.srt
|-- lap01_final.mp4
`-- lap01_final_parts/
```

## Stage 6: Edit

Goal:

- patch the editable plan without opening the JSON manually

Supported actions:

- `update-segment`
- `update-caption`

Update segment timing:

```bash
python pipeline.py edit update-segment \
  --plan ./input/lap01/highlights_30s.editable.json \
  --rank 3 \
  --source-start-seconds 150.2 \
  --source-end-seconds 153.2
```

Update caption text:

```bash
python pipeline.py edit update-caption \
  --plan ./input/lap01/highlights_30s.editable.json \
  --rank 3 \
  --caption "Handlebar wobble starts here." \
  --caption-detail "Close to the tyre wall."
```

The editable plan is designed to support the common review loop:

- rerun the whole review stage
- modify a segment
- modify subtitles

## Recommended Review Loop

1. Run `extract`
2. Run `infer`
3. Run `review` with preview enabled
4. Inspect `*.source.srt` and `*.preview.mp4`
5. Patch `*.editable.json` with `edit`
6. Run `render`

Use `pipeline.py` for anything meant to be portable across Windows, macOS, Linux, or server environments.
