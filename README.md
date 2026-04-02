# Local AI Highlight Pipeline

This project builds short highlight videos from forward-facing riding footage.

Chinese version: `README.zh-CN.md`

The product-facing entry point is cross-platform:

```bash
python pipeline.py <stage> ...
```

## Stages

The pipeline is split into four explicit stages:

1. `extract`
   Turn a video into LLM-ready images.
2. `infer`
   Send extracted frames to either a local Ollama model or an API model with automatic routing.
3. `review`
   Build a reviewable edit plan, source/final SRT files, and an optional preview video.
4. `render`
   Cut and concatenate the final MP4 with ffmpeg.

There is also an `edit` utility stage for patching an editable review plan:

- `edit update-segment`
- `edit update-caption`

And a one-shot stage:

- `run` (extract -> infer -> review -> render)

## Setup

1. Install Python 3.12+.
2. Install the dependencies in `requirements.txt`.
3. For local development, create a `.env` file if you want Gemini or other API fallback.
4. Make sure Ollama is running if you want local inference.
5. Put exported videos into `input/`.
6. Copy `config.example.toml` to `config.toml` if needed.

Example `.env`:

```ini
GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

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

Automatic routing:

```bash
python pipeline.py infer --video ./input/lap01.mp4 --config ./config.track.toml
```

The default `infer` route is:

- local Ollama first
- Gemini 3 Flash next if local inference is unavailable and `GEMINI_API_KEY` is set
- Qwen next if Gemini is unavailable and `DASHSCOPE_API_KEY` is set
- generic OpenAI-compatible API last if it is explicitly configured for vision support

Force Gemini:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider gemini
```

Force the generic API route:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider api
```

Force a specific OpenAI-compatible API model:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider api \
  --api-base https://api.deepseek.com \
  --model deepseek-chat \
  --api-key-env DEEPSEEK_API_KEY
```

Opt into Gemini async batch submission:

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider gemini \
  --submission-mode async
```

Then collect the completed batch into `analysis.json`:

```bash
python pipeline.py collect \
  --manifest ./output/lap01/analysis.batch.json
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

### One command for the full CLI flow

```bash
python pipeline.py run \
  --video ./input/lap01.mp4 \
  --provider auto \
  --sample-fps 1 \
  --target-seconds 30 \
  --caption-mode human \
  --resolution source
```

`--sample-fps` accepts values from `0.1` to `24` (default `1`).

## Configuration

The config now supports both the original legacy sections and the new pipeline sections:

- `project`
- `sampling` and `extract`
- `filters`
- `ollama` and `provider`
- `provider.routing`
- `provider.ollama`
- `provider.gemini`
- `provider.openai_compatible`
- `prompt`
- `decision` and `selection`
- `review`
- `preview`
- `render`

Key user-facing knobs:

- frame interval or sample FPS
- provider routing and model
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

## FastAPI backend (realtime job API)

The repo now includes a minimal backend scaffold in `backend/`:

- **FastAPI** for REST + WebSocket + SSE.
- **Redis + RQ** for long-running task execution.

### Run locally

1. Start Redis (`redis://localhost:6379/0` by default).
2. Start API server:

   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Start an RQ worker in another terminal:

   ```bash
   python -m backend.worker
   ```

### API endpoints

- `POST /jobs` create a background job.
- `GET /jobs/{job_id}` poll current status.
- `GET /sse/jobs/{job_id}` stream status via SSE.
- `WS /ws/jobs/{job_id}` stream status via WebSocket.

## More

See `WORKFLOW.md` for the full stage-by-stage workflow.
