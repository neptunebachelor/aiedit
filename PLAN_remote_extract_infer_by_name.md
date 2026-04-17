# Plan: Trigger extract/infer on remote files surfaced by `ls`

## Context

The remote CLI (`remote_infer.py`) currently supports `ls` (raw videos under the server's
`--videos-dir`) and `ls-local` (extracted videos in the caller's local `.video_data`). But
`extract` and `infer` still require the user to paste absolute paths on the PC. The goal is to
close that loop: after running `ls`, the user should be able to trigger `extract` / `infer`
directly by **filename** or **slug**, without knowing the absolute path layout on the server.

Out of scope: bulk operations (one job per invocation), auth, file upload. All operations execute
on the server against files the server already sees on its filesystem.

---

## Current code state

| File | Relevant symbols | Actual line |
|---|---|---|
| `infer_server.py` | `_videos_dir` global | L39 |
| `infer_server.py` | `ExtractJobRequest` | L46 |
| `infer_server.py` | `InferJobRequest` | L50 |
| `infer_server.py` | `POST /extract-jobs` handler | L184 |
| `infer_server.py` | `POST /infer-jobs` handler | L204 |
| `infer_server.py` | `GET /ls/videos` | L264 |
| `infer_server.py` | `main()` | L284 |
| `remote_infer.py` | `cmd_extract` | L122 |
| `remote_infer.py` | `cmd_infer` | L131 |
| `remote_infer.py` | `cmd_ls` | L142 |
| `remote_infer.py` | `cmd_ls_local` (scan loop) | L171–213 |
| `remote_infer.py` | argparse wiring | L220 |

---

## Design

### 1. Server: `infer_server.py`

#### a) Add `_data_root` global and `--data-root` startup arg

`/ls/extracted` and slug resolution both need to know where `.video_data` lives.
Add a global alongside `_videos_dir`:

```python
_data_root: Path | None = None   # set at startup
```

In `main()`, add an optional `--data-root` arg. If omitted, auto-detect via
`resolve_video_data_root()`:

```python
parser.add_argument("--data-root", default=None,
                    help="Path to .video_data root (default: auto-detect from repo)")
# ...
_data_root = Path(args.data_root).expanduser().resolve() if args.data_root \
             else resolve_video_data_root()
```

#### b) Extend `ExtractJobRequest` (L46)

```python
class ExtractJobRequest(BaseModel):
    video_path: str | None = None   # existing: absolute path on PC
    video_name: str | None = None   # new: filename inside --videos-dir
```

Both optional; the handler enforces XOR.

#### c) Extend `InferJobRequest` (L50)

```python
class InferJobRequest(BaseModel):
    index_path: str | None = None   # existing
    slug: str | None = None         # new: slug of an already-extracted video
    shutdown: bool = False
```

`index_path` becomes optional; the handler enforces XOR with `slug`.

#### d) Slug → index path resolution

`safe_video_slug` is a **sanitiser** (strips unsafe chars from a filename stem), not a
reverse lookup. To go from user slug to `index.json`:

```python
clean = safe_video_slug(payload.slug)   # sanitise — prevents "../.." traversal
index_path = _data_root / "videos" / clean / "extract" / "index.json"
if not index_path.is_file():
    raise HTTPException(status_code=404, detail=f"No extracted video with slug: {clean!r}")
```

`safe_video_slug("ride01")` → `"ride01"` (no-op on clean input).
`safe_video_slug("ride01.mp4")` → `"ride01"` (strips extension — UX convenience).

#### e) `_scan_extracted` helper + `GET /ls/extracted` (add near L264)

Port the scan loop from `cmd_ls_local` (L171–213) into the server.
Client's `cmd_ls_local` keeps its own copy.

```python
def _scan_extracted(data_root: Path) -> list[dict]:
    videos_dir = data_root / "videos"
    if not videos_dir.is_dir():
        return []
    entries = []
    for slug_dir in sorted(videos_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        index_path = slug_dir / "extract" / "index.json"
        if not index_path.is_file():
            continue
        source_path, frame_count = "-", 0
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            source_path = (payload.get("video") or {}).get("source_path") or "-"
            frame_count = len(payload.get("frames") or [])
        except Exception:
            pass
        has_infer = (slug_dir / "infer").is_dir() and any((slug_dir / "infer").iterdir())
        entries.append({
            "slug": slug_dir.name,
            "frames": frame_count,
            "has_infer": has_infer,
            "source_path": source_path,
            "index_path": str(index_path),
        })
    return entries


@app.get("/ls/extracted")
def list_extracted() -> dict:
    return {"data_root": str(_data_root), "extracted": _scan_extracted(_data_root)}
```

---

### 2. Client: `remote_infer.py`

#### a) `extract` — add `--name`, mutually exclusive with `--video`

Both mean "which video to process" — only one can be given per job.

```python
mx = p_extract.add_mutually_exclusive_group(required=True)
mx.add_argument("--video", help="Absolute path to video ON THE PC (existing)")
mx.add_argument("--name",  help="Filename under the server's --videos-dir (new)")
```

```python
def cmd_extract(args: argparse.Namespace) -> None:
    body = {"video_name": args.name} if args.name else {"video_path": args.video}
    _submit_and_stream(base=f"http://{args.host}", endpoint="extract-jobs",
                       body=body, label="Extract")
```

#### b) `infer` — add `--slug`, mutually exclusive with `--index`

```python
mx = p_infer.add_mutually_exclusive_group(required=True)
mx.add_argument("--index", help="Absolute path to extract/index.json ON THE PC (existing)")
mx.add_argument("--slug",  help="Slug of an already-extracted video on the PC (new)")
```

```python
def cmd_infer(args: argparse.Namespace) -> None:
    if args.shutdown:
        print("NOTE: PC will shut down after infer completes.")
    body = {"slug": args.slug} if args.slug else {"index_path": args.index}
    body["shutdown"] = args.shutdown
    _submit_and_stream(base=f"http://{args.host}", endpoint="infer-jobs",
                       body=body, label="Infer")
```

#### c) `ls` — add `--extracted` flag

```python
p_ls.add_argument("--extracted", action="store_true",
                   help="Show already-extracted videos instead of raw files")
```

In `cmd_ls`, branch on the flag: when set, `GET /ls/extracted` and print in the same
`SLUG / FRAMES / INFER / SOURCE` format as `cmd_ls_local`.

---

### 3. Intended workflow (end state)

```
# 1. See what raw videos are on the PC
$ remote_infer.py ls --host pc:8765
NAME          SIZE      PATH
ride01.mp4    850 MB    /mnt/videos/ride01.mp4

# 2. Extract by name — no abs path needed
$ remote_infer.py extract --host pc:8765 --name ride01.mp4
# ...SSE progress...  index_path: /home/user/.video_data/videos/ride01/extract/index.json

# 3. Confirm extraction on PC
$ remote_infer.py ls --host pc:8765 --extracted
SLUG      FRAMES  INFER  SOURCE
ride01       842     no  /mnt/videos/ride01.mp4

# 4. Infer by slug, then shut the PC down
$ remote_infer.py infer --host pc:8765 --slug ride01 --shutdown
```

---

## Validation rules (server-side)

- **`video_name`**: must not contain `/` or `..`; resolved path must be inside `_videos_dir`
  (`resolved.is_relative_to(_videos_dir)`); must have a video extension (`VIDEO_EXTS`); file
  must exist → 400
- **`slug`**: sanitise with `safe_video_slug`; resolved `index.json` must exist → 404
- **XOR rule**: both or neither identifier provided → 400
- **`_videos_dir` not set**: `video_name` used but server started without `--videos-dir` → 400
- **`/ls/extracted`**: `_data_root/videos/` doesn't exist yet → `{"extracted": [], "data_root": "..."}`

---

## Verification checklist

1. `python infer_server.py --videos-dir /tmp/test-videos --port 8765`
2. `remote_infer.py ls --host 127.0.0.1:8765` — lists mp4.
3. `remote_infer.py extract --host 127.0.0.1:8765 --name sample.mp4` — SSE streams; prints `index_path`.
4. `remote_infer.py ls --host 127.0.0.1:8765 --extracted` — shows slug, `has_infer: false`.
5. `remote_infer.py infer --host 127.0.0.1:8765 --slug sample` — SSE streams; completes.
6. `remote_infer.py ls --host 127.0.0.1:8765 --extracted` — `has_infer: true`.
7. Negative cases:
   - `--name ../../etc/passwd` → 400
   - `--name nonexistent.mp4` → 400
   - `--slug nope` → 404
   - `--name ride01.mp4 --video /abs/path` → argparse error (client-side)
   - Both fields in body → 400 (server-side)
8. Backward compat: `--video /abs/path` and `--index /abs/index.json` still work unchanged.
