# Plan: 统一临时 / 衍生产物路径规则（三端一致）

## Context

项目 `aiedit` 在视频 extract / infer 流程中会产出大量临时文件：抽帧图像、推理 JSON、分析元数据、concat staging 等。历史上这些产物被散落到 12 种 ad-hoc 目录（见 `.gitignore` 里的 `.gemini_temp_bulk/`、`inference_tmp/`、`tmp_batch_infer/`、`tmp_verify*/`、`infer_calib_*/`、`tmpo*/`、`temp.json`、`temp_index.json` 等），每次靠往 `.gitignore` 打补丁兜底。

实际勘察后结论比预期好：当前活跃代码绝大多数已经走 `video_data_paths.py` 的规范路径（`.video_data/{videos,frames}/<slug>/...`），只有 2 处历史违规。但问题是：**没有任何文档告诉 AI 编码工具"只能用这个根"**，所以每次新对话都可能再造一个 `tmp_xxx/`。

用户同时使用 Claude Code、Codex、Gemini CLI 三个 AI 编码工具，规则文件必须三端都认。目标：

1. 一份权威规则文档，三端都能读到
2. 把仅存的两处违规修掉，让规则立住
3. 加一个环境变量 `RIDE_VIDEO_DATA_ROOT`，让想把 artifact 放外置盘的用户能重定向整棵树
4. 清理 `.gitignore` 里已废弃的历史模式

---

## 设计决策

- **规则正本**：`AGENTS.md`（Codex 原生读取，也是 de facto 标准）
- **Claude Code / Gemini CLI 接入**：`CLAUDE.md`、`GEMINI.md` 作为 symlink 指向 `AGENTS.md`（Linux 原生稳，不依赖各 CLI 的 `@import` 支持）
- **ffmpeg concat staging**：`render_highlights.py` 在 `output_video.parent / "{stem}_parts"` 创建的 staging 目录**算合法例外**，写进规则白名单 —— 它是一次性 concat 输入，运行完立即删，和长期 artifact 不是一回事
- **env override**：`resolve_video_data_root()` 已支持 `override=` 参数，本次追加读取 `RIDE_VIDEO_DATA_ROOT` 环境变量作为 fallback

---

## 需要修改的文件

### 1. 新建 `AGENTS.md`（仓库根）

规则内容大纲：

- **唯一根目录**：所有视频衍生产物必须位于 `resolve_video_data_root()` 返回的目录下（默认 `<repo>/.video_data/`，可被 `RIDE_VIDEO_DATA_ROOT` 覆盖）
- **不许做的事**（硬禁令，给 AI 看的）：
  - 禁止在仓库根或 CWD 创建 `tmp_*/`、`temp_*/`、`*_tmp/`、`inference_tmp/`、`.gemini_temp_*/`、`.ride-video-infer-*/`、`infer_calib_*/`、`tmpo*/` 等目录
  - 禁止写 `temp.json`、`temp_index.json`、`gemini_prompt*.txt`、`*.frame_decisions.jsonl` 到 CWD
  - 禁止用 `tempfile.mkdtemp()` 在系统 `/tmp` 下放跨步骤数据（单元测试除外）
  - 禁止在输入视频旁边创建帧/JSON 产物
- **该用的 API**（引用 `video_data_paths.py` 里的函数）：
  - `resolve_video_data_root(repo_root=None, override=None)` — 根目录
  - `video_artifact_dir(video_path)` — 单视频 artifact 目录（包含 analysis.json、highlights.json 等）
  - `video_frames_dir(video_path)` — 帧图目录
  - `artifact_dir_from_index(index_path)` / `artifact_dir_from_payload(payload)` — 从 index 或 payload 反推 artifact 目录
  - `infer_dir_from_index(index_path)` — 推理产物目录（`.../infer/`）
- **目录约定**：
  - `videos/<slug>/` — 元数据、analysis、highlights
  - `frames/<slug>/` — 抽帧 PNG/JPG
  - `videos/<slug>/infer/` — 推理 JSON、packs、gemini_cli_runs
  - `videos/<slug>/debug/` — 临时 index（见 `prepare_temp_index.py:47`）
  - `videos/<slug>/staging/` — 如需一次性 staging，放这里（不要放到仓库根）
- **合法例外白名单**：
  - `render_highlights.py` 的 `{output_video.stem}_parts/` ffmpeg concat staging（一次性、与最终 mp4 相邻、run 完即删）
  - 单元测试用的 `tests/_tmp/`（测试隔离）
- **env 变量**：`RIDE_VIDEO_DATA_ROOT` 指向替代根目录（绝对路径）

### 2. 创建两个 symlink（仓库根）

```
CLAUDE.md  -> AGENTS.md
GEMINI.md  -> AGENTS.md
```

用 `ln -s AGENTS.md CLAUDE.md` 和 `ln -s AGENTS.md GEMINI.md`。git 会把它们作为 symlink 跟踪。

### 3. 修改 `video_data_paths.py:19-22`

在 `resolve_video_data_root` 里加入环境变量读取：

```python
def resolve_video_data_root(repo_root: Path | None = None, override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("RIDE_VIDEO_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (repo_root or find_repo_root(Path(__file__))).resolve() / ".video_data"
```

需要 `import os`。优先级：函数参数 `override` > 环境变量 > 默认 `<repo>/.video_data/`。

### 4. 修复 `skills/ride-video-infer/scripts/prepare_packs.py:51-57`

原代码（手写路径）：
```python
infer_dir: Path
if args.output_dir:
    infer_dir = Path(args.output_dir).expanduser().resolve()
else:
    # Default to the canonical .video_data video infer directory
    # which is index_path.parent.parent / "infer"
    infer_dir = index_path.parent.parent / "infer"
```

改用 helper：
```python
from video_data_paths import infer_dir_from_index
...
if args.output_dir:
    infer_dir = Path(args.output_dir).expanduser().resolve()
else:
    infer_dir = infer_dir_from_index(index_path)
```

注意 import 路径要能从该脚本所在位置解析（`skills/ride-video-infer/scripts/` 下其他脚本已有类似 import，参照它们的做法，`sys.path` 注入或相对 import）。

### 5. `render_highlights.py:582` — 不改代码，加注释

在 `temp_dir = output_video.parent / f"{output_video.stem}_parts"` 上方加一行短注释，说明这是 `AGENTS.md` 白名单里的 ffmpeg concat staging 合法例外、用完即删。只是给后续 AI 编码器看，不改路径。

### 6. 清理 `.gitignore:13-37`

保留：`.video_data/`、`workspace/jobs/`、`deepseek_v3_tokenizer.zip`

删除已废弃的历史模式：
- `.gemini_temp_bulk/`、`.gemini_temp_pack/`
- `.ride-video-infer-tmp/`、`.ride-video-infer-test/`
- `.codex_backlog*/`
- `inference_tmp/`（出现两次，删除两次）
- `tmp_batch_infer/`（出现两次）
- `temp_visualize/`（出现两次）
- `tmp_verify*/`、`infer_calib_*/`、`tmpo*/`
- `temp.json`、`temp_index.json`、`gemini_prompt*.txt`、`*.frame_decisions.jsonl`

理由：这些路径在 AGENTS.md 里明确禁止创建，没必要再用 `.gitignore` 兜底；兜底会让"写了也没事"的心理继续存在。

---

## 关键文件一览

| 文件 | 操作 |
|---|---|
| `AGENTS.md` | 新建（仓库根） |
| `CLAUDE.md` | 新建 symlink → `AGENTS.md` |
| `GEMINI.md` | 新建 symlink → `AGENTS.md` |
| `video_data_paths.py` | 修改 `resolve_video_data_root`（L19-22），加 `import os` |
| `skills/ride-video-infer/scripts/prepare_packs.py` | 修改 L51-57，用 `infer_dir_from_index` |
| `render_highlights.py` | L582 上方加单行注释 |
| `.gitignore` | 删除 L15-33 的历史模式 |

---

## 已存在可复用的工具

全部在 `video_data_paths.py`，**不要新建函数**：

- `find_repo_root(start=None)` — L9
- `resolve_video_data_root(repo_root, override)` — L19（本次修改加 env 支持）
- `safe_video_slug(value)` — L25
- `video_artifact_dir(video_path, data_root=None)` — L37
- `video_frames_dir(video_path, data_root=None)` — L41
- `artifact_dir_from_payload(payload, fallback, data_root)` — L67
- `artifact_dir_from_index(index_path, data_root)` — L77
- `infer_dir_from_index(index_path, data_root)` — L90
- `resolve_frame_image_path(frame, index_path, payload, data_root)` — L94

正确用法示例可对照：
- `pipeline.py`（写 analysis.json，搜索 `resolve_video_dir_for_index`）
- `skills/ride-video-infer/scripts/run_gemini_packed.py`（写 frame_decisions.jsonl）
- `skills/ride-video-infer/scripts/upload_frame_files.py`（写 file_uploads manifest）

---

## Verification

1. **Symlink 正确**：
   ```bash
   ls -la CLAUDE.md GEMINI.md AGENTS.md
   # CLAUDE.md 和 GEMINI.md 应显示 -> AGENTS.md
   readlink CLAUDE.md  # AGENTS.md
   readlink GEMINI.md  # AGENTS.md
   ```

2. **env 变量生效**：
   ```bash
   RIDE_VIDEO_DATA_ROOT=/tmp/vd_test python3 -c \
     "from video_data_paths import resolve_video_data_root; print(resolve_video_data_root())"
   # 期望输出 /tmp/vd_test
   python3 -c "from video_data_paths import resolve_video_data_root; print(resolve_video_data_root())"
   # 期望输出 <repo>/.video_data
   python3 -c "from video_data_paths import resolve_video_data_root; print(resolve_video_data_root(override='/custom'))"
   # 期望输出 /custom（override 优先）
   ```

3. **prepare_packs 仍走规范路径**：
   ```bash
   python3 -m pytest tests/ -x
   ```
   全部通过。如果本脚本有 dry-run 入口，在已有 `.video_data/videos/<slug>/extract/index.json` 的样本上跑一次，确认产物出现在 `.video_data/videos/<slug>/infer/packs/`。

4. **render_highlights 路径不变**：只加了注释，`git diff` 验证只是 `+` 一行注释，行为无变化。

5. **AGENTS.md 可读性**：自己通读一遍，确保硬禁令部分清晰、函数引用精确到文件路径和函数名，使其他 AI 工具能立即定位。

6. **.gitignore 无误删**：`.video_data/`、`workspace/jobs/`、`__pycache__/`、`.venv/`、`.env`、`output/`、`output_gemini/` 等仍在。
