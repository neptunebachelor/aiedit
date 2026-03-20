# 本地 AI 高光视频流水线

这个项目用于从前向视角的骑行视频中自动生成短高光视频。

正式入口是跨平台的：

```bash
python pipeline.py <stage> ...
```

## 阶段说明

整个流水线拆分为 4 个明确阶段：

1. `extract`
   将视频转换为适合 LLM 识别的图片。
2. `infer`
   将抽出的帧发送给本地 Ollama 模型或第三方 API 做识别。
3. `review`
   生成可审阅的剪辑方案、source/final 两套 SRT，以及可选的预览视频。
4. `render`
   使用 ffmpeg 裁剪并拼接最终 MP4。

另外还提供一个 `edit` 工具阶段，用于修改可编辑的审阅方案：

- `edit update-segment`
- `edit update-caption`

## 环境准备

1. 安装 Python 3.12 及以上版本。
2. 安装 `requirements.txt` 中的依赖。
3. 如果你使用本地 provider，请确保 Ollama 正在运行。
4. 将导出的视频放入 `input/` 目录。
5. 如果需要，先复制 `config.example.toml` 为 `config.toml`。

## 快速开始

### 1. 抽帧

普通道路骑行：

```bash
python pipeline.py extract --video ./input/ride01.mp4 --frame-interval-seconds 1.0
```

赛道视频：

```bash
python pipeline.py extract --video ./input/lap01.mp4 --config ./config.track.toml --frame-interval-seconds 0.5
```

### 2. 运行识别

本地 Ollama：

```bash
python pipeline.py infer --video ./input/lap01.mp4 --config ./config.track.toml
```

兼容 OpenAI 的第三方 provider：

```bash
python pipeline.py infer \
  --video ./input/lap01.mp4 \
  --provider-type openai_compatible \
  --api-base https://your-api.example/v1 \
  --model your-vision-model \
  --api-key-env OPENAI_API_KEY
```

### 3. 审阅并生成预览

```bash
python pipeline.py review \
  --input ./output/lap01/analysis.json \
  --target-seconds 30 \
  --caption-mode human \
  --preview \
  --preview-resolution 720p
```

这一步会输出：

```text
output/lap01/
|-- highlights_30s.review.json
|-- highlights_30s.editable.json
|-- highlights_30s.final.srt
|-- highlights_30s.source.srt
`-- highlights_30s.preview.mp4
```

预览分辨率支持：

- `540p`
- `720p`
- `1080p`
- `source`

### 4. 按需修正方案

修改原视频中的切点：

```bash
python pipeline.py edit update-segment \
  --plan ./output/lap01/highlights_30s.editable.json \
  --rank 3 \
  --source-start-seconds 150.2 \
  --source-end-seconds 153.2
```

修改字幕文本：

```bash
python pipeline.py edit update-caption \
  --plan ./output/lap01/highlights_30s.editable.json \
  --rank 3 \
  --caption "Handlebar wobble starts here." \
  --caption-detail "Close to the tyre wall."
```

### 5. 渲染最终视频

输出原始分辨率成片：

```bash
python pipeline.py render \
  --input ./output/lap01/highlights_30s.editable.json \
  --stem lap01_final \
  --resolution source
```

也可以输出较低分辨率的交付版本：

```bash
python pipeline.py render \
  --input ./output/lap01/highlights_30s.editable.json \
  --stem lap01_final_720p \
  --resolution 720p
```

## 配置说明

配置文件现在同时兼容原来的 legacy 段落和新的 pipeline 段落：

- `project`
- `sampling` 和 `extract`
- `filters`
- `ollama` 和 `provider`
- `prompt`
- `decision` 和 `selection`
- `review`
- `preview`
- `render`

用户最常调整的参数包括：

- 抽帧间隔或 sample FPS
- provider 类型和模型名
- API Base URL 和 API Key
- 目标高光总时长
- 单段片段时长限制
- 预览分辨率
- 最终渲染分辨率

## 输出文件

典型输出目录如下：

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

## 更多说明

完整的分阶段工作流请参考 `WORKFLOW.md`。
