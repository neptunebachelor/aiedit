# Local AI Highlight Workflow

This project is a Windows-first MVP for front-facing riding footage.

## What it does

1. Reads forward-fixed `.mp4` footage exported from Insta360 Studio.
2. Samples frames with OpenCV.
3. Applies a coarse filter for blur, low-motion, and near-duplicate frames.
4. Sends candidate frames to a local Ollama vision model.
5. Writes:
   - `analysis.json`
   - `highlights.json`
   - `highlights.srt`

## Folder layout

```text
.
|-- analyze_video.py
|-- generate_srt.py
|-- run.ps1
|-- config.example.toml
|-- requirements.txt
|-- input/
`-- output/
```

## Setup

1. Install Python 3.12+.
2. Install Ollama for Windows.
3. Make sure the Ollama app is running locally.
4. Pull a vision model, for example `ollama pull qwen3-vl:8b`.
   If the CLI is not in `PATH`, the analyzer still works as long as the local Ollama API is reachable on `http://127.0.0.1:11434`.
5. Copy `config.example.toml` to `config.toml`.
6. Put exported videos into `input/`.

## Run

Preferred on Windows:

```cmd
.\run.cmd
```

If you want to call the PowerShell script directly, use a one-off execution policy bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or process a specific file:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -InputPath .\input\ride01.mp4
```

Run extraction only, without Ollama:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -ExtractOnly
```

## Output

Each video gets its own folder:

```text
output/
`-- ride01/
    |-- analysis.json
    |-- highlights.json
    |-- highlights.srt
    `-- thumbnails/
```

Use `highlights.srt` in CapCut/Jianying as timeline anchors.

## Trim To 30 Seconds

After analysis, you can build a tighter 30-second cut plan from `analysis.json`:

```powershell
python .\render_highlights.py `
  --input .\output\ride01\analysis.json `
  --target-seconds 30 `
  --stem highlights_30s
```

This writes:

```text
output/ride01/
|-- highlights_30s.json
|-- highlights_30s.srt
`-- highlights_30s_source.srt
```

The script splits long highlight runs into shorter clip candidates, ranks them, and keeps the best non-overlapping clips until it reaches the target total duration.

`analysis.json` is the preferred input because it contains `source_timestamps`, which lets the script split long highlight runs more intelligently than plain SRT.
`highlights_30s.srt` is rebased to the final 30-second timeline, while `highlights_30s_source.srt` keeps the original source-video timestamps for manual cutting on the full clip.

## Render Final MP4

If `ffmpeg.exe` is available, the same script can cut and concatenate the final video automatically:

```powershell
python .\render_highlights.py `
  --input .\output\ride01\analysis.json `
  --target-seconds 30 `
  --stem highlights_30s `
  --render `
  --ffmpeg C:\path\to\ffmpeg.exe
```

If `ffmpeg` is already in `PATH`, you can omit `--ffmpeg`.
Use `--max-clips-per-source-segment 0` if you want to disable the built-in diversity limit and allow more clips from the same long highlight section.
