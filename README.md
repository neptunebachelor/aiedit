# Local AI Highlight Pipeline

This project builds short highlight videos from forward-facing riding footage.

The product-facing entry point is now cross-platform:

```bash
python pipeline.py <stage> ...
```

Windows-only wrappers such as `run.ps1` and `run.cmd` are kept only as legacy compatibility helpers.

## Stages

The pipeline is split into four explicit stages:

1. `extract`
   Turn a video into LLM-ready images.
2. `infer`
   Send extracted frames to either a local Ollama model or a third-party API.
3. `review`
   Build a reviewable edit plan, source/final SRT files, and an optional preview video.
4. `render`
   Cut and concatenate the final MP4 with ffmpeg.

There is also an `edit` utility stage for patching an editable review plan:

- `edit update-segment`
- `edit update-caption`

## Setup

1. Install Python 3.12+.
2. Install the dependencies in `requirements.txt`.
3. Make sure Ollama is running if you use the local provider.
4. Put exported videos into `input/`.
5. Copy `config.example.toml` to `config.toml` if needed.

## Quick Start

### 1. Extract frames

Road riding:

```bash
python pipeline.py extract --video ./input/ride01.mp4 --frame-interval-seconds 1.0
```

Track laps:

```bash
python pipeline.py extract --video ./input/lap01.mp4 --config ./config.track.toml --frame-interval-seconds 0.5
```

### 2. Run inference

Local Ollama:

```bash
python pipeline.py infer --video ./input/lap01.mp4 --config ./config.track.toml
```

OpenAI-compatible provider:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider-type openai_compatible \
  --api-base https://your-api.example/v1 \
  --model your-vision-model \
  --api-key-env OPENAI_API_KEY
```

### 3. Review and preview

```bash
python pipeline.py review \
  --input ./output/lap01/analysis.json \
  --target-seconds 30 \
  --caption-mode human \
  --preview \
  --preview-resolution 720p
```

This writes:

```text
output/lap01/
|-- highlights_30s.review.json
|-- highlights_30s.editable.json
|-- highlights_30s.final.srt
|-- highlights_30s.source.srt
`-- highlights_30s.preview.mp4
```

Preview resolutions:

- `540p`
- `720p`
- `1080p`
- `source`

### 4. Patch the plan if needed

Change a source cut:

```bash
python pipeline.py edit update-segment \
  --plan ./output/lap01/highlights_30s.editable.json \
  --rank 3 \
  --source-start-seconds 150.2 \
  --source-end-seconds 153.2
```

Change subtitle text:

```bash
python pipeline.py edit update-caption \
  --plan ./output/lap01/highlights_30s.editable.json \
  --rank 3 \
  --caption "Handlebar wobble starts here." \
  --caption-detail "Close to the tyre wall."
```

### 5. Render the final video

```bash
python pipeline.py render \
  --input ./output/lap01/highlights_30s.editable.json \
  --stem lap01_final \
  --resolution source
```

Or render a lower-resolution deliverable:

```bash
python pipeline.py render \
  --input ./output/lap01/highlights_30s.editable.json \
  --stem lap01_final_720p \
  --resolution 720p
```

## Configuration

The config now supports both the original legacy sections and the new pipeline sections:

- `project`
- `sampling` and `extract`
- `filters`
- `ollama` and `provider`
- `prompt`
- `decision` and `selection`
- `review`
- `preview`
- `render`

Key user-facing knobs:

- frame interval or sample FPS
- provider type and model
- API base URL and API key
- target highlight duration
- clip length limits
- preview resolution
- final render resolution

## Outputs

Typical stage outputs look like this:

```text
output/lap01/
|-- extract/
|   |-- frames/
|   `-- index.json
|-- analysis.json
|-- frame_decisions.jsonl
|-- segments.raw.json
|-- segments.raw.srt
|-- highlights_30s.review.json
|-- highlights_30s.editable.json
|-- highlights_30s.final.srt
|-- highlights_30s.source.srt
|-- highlights_30s.preview.mp4
`-- lap01_final.mp4
```

## More

See `WORKFLOW.md` for the full stage-by-stage workflow.
