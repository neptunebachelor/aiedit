# Plan: Trigger extract/infer on remote files surfaced by `ls`

## Context

The remote CLI (`remote_infer.py`) currently supports `ls` (raw videos under the server's `--videos-dir`) and `ls-local` (extracted videos in the caller's local `.video_data`). But the `extract` and `infer` subcommands still require the user to paste absolute paths on the PC. The goal is to close that loop: after running `ls`, the user should be able to trigger `extract` / `infer` directly by **filename** or **slug**, without knowing the absolute path layout on the server.

Out of scope: bulk operations (one job per invocation), auth, file upload. All operations execute on the server against files the server already sees on its filesystem.

## Branch

Build on top of `feature/remote-infer-ls-commands` (already contains `ls` / `ls-local`). Not yet merged to main — extend it with the new commits.

## Design

### 1. Server: `infer_server.py`

**a) Extend `ExtractJobRequest` to accept filename (in addition to absolute path)** (`infer_server.py:42`)

```python
class ExtractJobRequest(BaseModel):
    video_path: str | None = None   # existing: absolute path on PC
    video_name: str | None = None   # new: filename inside --videos-dir
```

In the extract handler (`infer_server.py:180`), resolve `video_name` to an absolute path under `_videos_dir`, reject if neither/both provided, reject if not a video extension, reject if outside `_videos_dir` (resolve then check `is_relative_to`).

**b) Extend `InferJobRequest` to accept slug** (`infer_server.py:46`)

```python
class InferJobRequest(BaseModel):
    index_path: str | None = None   # existing
    slug: str | None = None         # new
    shutdown: bool = False
```

Slug resolves to `<data_root>/videos/<slug>/extract/index.json` via existing helpers in `video_data_paths.py` (`resolve_video_data_root`, `safe_video_slug`). 404 if the index doesn't exist.

**c) New endpoint `GET /ls/extracted`** (mirror of client-side `ls-local`)

Scans `<data_root>/videos/*/extract/index.json`, returns:
```json
{
  "data_root": "/path/to/.video_data",
  "extracted": [
    {"slug": "ride01", "frames": 842, "has_infer": true,
     "source_path": "/mnt/videos/ride01.mp4", "index_path": "/abs/.../index.json"}
  ]
}
```
Reuse the exact scan logic already written in `remote_infer.py:cmd_ls_local` (lines ~180–218 of the ls-commands branch) — extract it into a small helper in `infer_server.py` that takes a `Path` and returns the list.

### 2. Client: `remote_infer.py`

**a) `extract` subcommand** — make `--video` optional, add `--name`:

```
remote_infer.py extract --host H --name ride01.mp4
remote_infer.py extract --host H --video /abs/path.mp4   # existing, still works
```
Exactly one of `--name` / `--video` required (argparse mutually exclusive group). Body posted to server picks the matching field.

**b) `infer` subcommand** — make `--index` optional, add `--slug`:

```
remote_infer.py infer --host H --slug ride01 [--shutdown]
remote_infer.py infer --host H --index /abs/index.json [--shutdown]   # existing
```

**c) `ls` subcommand** — add `--extracted` flag:

```
remote_infer.py ls --host H                # raw videos (existing)
remote_infer.py ls --host H --extracted    # already-extracted videos on server
```
When `--extracted`, GET `/ls/extracted`; print a table with columns `SLUG / FRAMES / INFER / SOURCE` (same format as existing `ls-local`).

### 3. Intended workflow

```
# 1. See what's available on the PC
$ remote_infer.py ls --host pc:8765
NAME          SIZE      PATH
ride01.mp4    850 MB    /mnt/videos/ride01.mp4
ride02.mp4    1.2 GB    /mnt/videos/ride02.mp4

# 2. Extract one of them (no abs path needed)
$ remote_infer.py extract --host pc:8765 --name ride01.mp4
# ...SSE progress... result.index_path returned

# 3. See what's been extracted on the PC
$ remote_infer.py ls --host pc:8765 --extracted
SLUG      FRAMES  INFER  SOURCE
ride01       842     no  /mnt/videos/ride01.mp4

# 4. Infer by slug, then shut the PC down
$ remote_infer.py infer --host pc:8765 --slug ride01 --shutdown
```

## Files to modify

- `infer_server.py`
  - L42 `ExtractJobRequest`: add `video_name` (optional)
  - L46 `InferJobRequest`: add `slug` (optional), make `index_path` optional
  - L180 extract handler: resolve `video_name` → path (with sandboxing under `_videos_dir`), validate XOR with `video_path`
  - L200 infer handler: resolve `slug` → `index_path` via `resolve_video_data_root` + `safe_video_slug`
  - New `GET /ls/extracted` endpoint near existing `GET /ls/videos` (L261)
  - Small helper `_scan_extracted(data_root: Path) -> list[dict]` used by the new endpoint
- `remote_infer.py`
  - `cmd_extract`: accept `--name`, send `video_name` in body; mutually exclusive with `--video`
  - `cmd_infer`: accept `--slug`, send `slug` in body; mutually exclusive with `--index`
  - `cmd_ls`: add `--extracted` flag that calls `/ls/extracted` and reuses `ls-local`-style table
  - Update top-of-file docstring examples
  - argparse wiring at bottom

## Reuse

- `video_data_paths.resolve_video_data_root`, `safe_video_slug` — already imported by `infer_server.py:28`
- SSE streaming helper `_stream_sse` / `_submit_and_stream` in `remote_infer.py` — unchanged
- `VIDEO_EXTS` constant in `infer_server.py:33` — reuse for `video_name` validation
- `ls-local` scan logic — port into server as shared helper; the client's `cmd_ls_local` stays as-is (still useful for local callers who don't want to hit the server)

## Validation rules (server-side)

- `video_name`: must not contain `/` or `..`; resolved path must be inside `_videos_dir`; must have a video extension; file must exist. Return 400 with a clear message otherwise.
- `slug`: pass through `safe_video_slug`; resolved `index.json` must exist. Return 404 otherwise.
- `extract-jobs` / `infer-jobs`: reject if both identifiers provided or neither provided (400).
- `/ls/extracted`: if no videos extracted yet, return `{"extracted": [], "data_root": "..."}` (same shape as `/ls/videos`).

## Verification

1. Start server with a videos dir containing one sample mp4:
   `python infer_server.py --videos-dir /tmp/test-videos --port 8765`
2. `python remote_infer.py ls --host 127.0.0.1:8765` — lists the mp4.
3. `python remote_infer.py extract --host 127.0.0.1:8765 --name sample.mp4` — SSE progress streams; prints returned `index_path`.
4. `python remote_infer.py ls --host 127.0.0.1:8765 --extracted` — shows the new slug.
5. `python remote_infer.py infer --host 127.0.0.1:8765 --slug sample` — SSE progress; completes.
6. Negative cases (manual curl or CLI):
   - `--name ../../etc/passwd` → 400
   - `--name nonexistent.mp4` → 400
   - `--slug nope` → 404
   - Both `--name` and `--video` → argparse error client-side; both fields in body → 400 server-side.
7. Backward compat: existing `--video /abs/path` and `--index /abs/index.json` still work unchanged.
