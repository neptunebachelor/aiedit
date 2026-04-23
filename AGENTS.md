# Artifact Path Rules for AI Coding Tools

This file is the authoritative rule set for where video-processing artifacts (extracted frames, inference JSON, analysis metadata, concat staging, etc.) may live in this repository. `CLAUDE.md` and `GEMINI.md` are symlinks to this file so Claude Code, Codex, and Gemini CLI all read the same rules.

**Scope:** Any script, test, or one-off experiment that produces derived files from a video.

---

## 1. The one root

All video-derived artifacts MUST live under the directory returned by `resolve_video_data_root()` in `video_data_paths.py`.

- Default: `<repo_root>/.video_data/`
- Override: set environment variable `RIDE_VIDEO_DATA_ROOT=/absolute/path` to relocate the whole tree (e.g. onto an external drive)
- Function-arg override: `resolve_video_data_root(override=...)` beats the env var

Resolution priority: `override` arg > `RIDE_VIDEO_DATA_ROOT` > `<repo_root>/.video_data/`.

---

## 2. Hard bans (what you MUST NOT do)

These are the patterns that historically leaked into the repo and the `.gitignore`. Do not re-create any of them:

**No ad-hoc directories in the repo root or CWD:**
- `tmp_*/`, `temp_*/`, `*_tmp/`, `tmpo*/`
- `inference_tmp/`, `tmp_batch_infer/`, `tmp_verify*/`, `temp_visualize/`
- `.gemini_temp_bulk/`, `.gemini_temp_pack/`, `.gemini_temp_*/`
- `.ride-video-infer-tmp/`, `.ride-video-infer-test/`, `.ride-video-infer-*/`
- `.codex_backlog*/`
- `infer_calib_*/`

**No loose files in the repo root or CWD:**
- `temp.json`, `temp_index.json`
- `gemini_prompt*.txt`
- `*.frame_decisions.jsonl`

**No `tempfile.mkdtemp()` for cross-step data.** `/tmp` is fine for single-process throwaways inside one function. Anything another step reads back MUST go under the data root. Unit tests may use `tempfile` for isolation.

**No artifacts next to the source video.** The source video directory (e.g. `C:\videos\ride01.mp4`) is read-only from the pipeline's perspective. Do not write `ride01_frames/` or `ride01.analysis.json` there.

---

## 3. The API you MUST use

All helpers live in `video_data_paths.py`. Do not reinvent these — import them.

| Function | Purpose |
|---|---|
| `resolve_video_data_root(repo_root=None, override=None)` | The one root (see §1) |
| `video_artifact_dir(video_path, *, data_root=None)` | `<root>/videos/<slug>/` — per-video artifact dir |
| `video_frames_dir(video_path, *, data_root=None)` | `<root>/frames/<slug>/` — extracted frame images |
| `artifact_dir_from_index(index_path, *, data_root=None)` | Recover artifact dir from an `extract/index.json` path |
| `artifact_dir_from_payload(payload, *, fallback=None, data_root=None)` | Same, from a loaded index payload |
| `infer_dir_from_index(index_path, *, data_root=None)` | `<root>/videos/<slug>/infer/` — inference JSON, packs, CLI runs |
| `resolve_frame_image_path(frame, *, index_path=None, payload=None, data_root=None)` | Locate a frame's PNG/JPG |
| `safe_video_slug(value)` | Sanitise a filename/path into a slug (strips extension, unsafe chars) |
| `safe_existing_slug(value)` | Same, but preserves dots (use for already-slugged input) |

Scripts under `skills/ride-video-infer/scripts/` import these via a `sys.path.insert(REPO_ROOT, ...)` shim — see `upload_frame_files.py:17-27` for the canonical pattern.

---

## 4. Directory layout

Under `resolve_video_data_root()`:

```
<root>/
├── videos/<slug>/
│   ├── extract/
│   │   └── index.json            # authoritative per-video extract manifest
│   ├── infer/                    # coarse/temporal inference outputs
│   │   ├── packs/                # prepared pack manifests + images
│   │   ├── gemini_cli_runs/      # Gemini CLI per-run logs
│   │   ├── file_uploads/         # provider File API upload manifests
│   │   └── analysis*.json
│   ├── analysis.json             # merged analysis
│   ├── highlights.json           # review/edit plan
│   ├── debug/                    # prepare_temp_index.py output
│   └── staging/                  # optional per-video one-shot staging
└── frames/<slug>/
    └── frame_*.jpg               # extracted frames
```

Per-run ffmpeg concat staging for `render_highlights.py` is the one explicit exception — see §5.

---

## 5. Whitelisted exceptions

These paths are allowed despite looking like the banned patterns:

1. **`render_highlights.py` concat staging** — `<output_video.parent>/<output_video.stem>_parts/` is created as ffmpeg concat input and deleted as soon as the mux finishes. It lives next to the final mp4 (user-chosen output dir), not inside the repo, and is not a cross-step artifact.
2. **`tests/_tmp/`** — test isolation. Unit tests may write here.

Anything else matching the banned patterns will be rejected by `tests/test_no_forbidden_paths.py`.

---

## 6. Checklist for new code

Before committing code that writes files, confirm:

- [ ] Output path comes from a `video_data_paths.py` helper, not a hardcoded string
- [ ] No `tmp_`, `temp_`, `.gemini_temp_`, `.ride-video-infer-` style directories under repo root or CWD
- [ ] If you use `tempfile.mkdtemp()`, the data does not need to persist past this function
- [ ] Test files go under `tests/_tmp/` (or `tmp_path` pytest fixture)
- [ ] `pytest tests/test_no_forbidden_paths.py` passes

If you genuinely need a new artifact category, add a subdirectory under `<root>/videos/<slug>/` and document it here — do not add a new top-level bucket.
