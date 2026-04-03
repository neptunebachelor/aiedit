# Temporal Highlight Extraction Spec

## Goal

Build a highlight selection pipeline that can reliably find a strong 30-second highlight from a longer riding video without losing temporal context.

This spec focuses on the gap between:

- directly using provider-side video understanding
- manually extracting frames and feeding them to an image-capable LLM

The main design target is a practical hybrid pipeline that keeps costs under control while preserving motion, rhythm, and event continuity.

---

## Problem Statement

A naive frame-by-frame captioning pipeline often loses the information that actually makes a moment feel exciting.

Examples of what gets lost when a video is reduced to isolated frames:

- motion progression
- rhythm changes
- cause and effect
- event duration
- buildup -> peak -> release structure

A single frame may show a rider leaning, but it does not necessarily preserve:

- whether the lean is increasing or decreasing
- whether it is a sudden save or a smooth corner
- whether the segment is the peak of the action

This makes "describe each frame, then summarize later" weaker than it first appears.

---

## Core Design Principles

1. Do not treat a long video as a flat list of unrelated images.
2. Preserve local temporal windows whenever possible.
3. Spend more compute on promising candidate segments, not on the whole video.
4. Prefer structured outputs over verbose natural-language descriptions.
5. When an LLM only supports one image per request, encode temporal information into either:
   - the image itself
   - the request text
   - the sequence state carried across requests

---

## Recommended Pipeline

```text
Raw Video
  -> Shot Detection / Segmentation
  -> Lightweight Candidate Scoring
  -> Top-K Candidate Segments
  -> Temporal Window Analysis
  -> Highlight Ranking
  -> Boundary Refinement
  -> Final 30s Highlight
  -> Subtitle / Title / Description
```

### Stage 1: Shot Detection / Segmentation

Split the video into natural segments rather than blindly sampling one frame every fixed interval.

Possible signals:

- shot boundary changes
- large frame difference
- camera motion spikes
- object motion changes
- subtitle or OCR changes
- audio energy changes

Output:

- short time segments
- each segment has start/end timestamps

### Stage 2: Lightweight Candidate Scoring

Quickly score each segment using cheap signals before using an LLM heavily.

Suggested features:

- motion intensity
- scene change intensity
- camera shake / speed change
- audio peaks
- OCR / ASR keyword spikes
- domain heuristics (e.g. overtakes, heavy lean, near-miss moments)

Output:

- candidate segments ranked by coarse excitement score

### Stage 3: Temporal Window Analysis

Analyze only the top candidate segments with temporal context preserved.

Instead of asking the model to explain unrelated single frames, analyze short windows such as:

- 2 seconds
- 3 seconds
- 5 seconds

For each window, keep frame ordering explicit.

Output per window should be structured:

```json
{
  "window_start": 15.0,
  "window_end": 18.0,
  "event": "rider leans sharply into a turn and exits fast",
  "peak_time": [16.5, 17.2],
  "score": 8.8,
  "recommended": true
}
```

### Stage 4: Highlight Ranking

Rank candidate windows using both cheap features and LLM analysis.

Ranking dimensions:

- action intensity
- rhythm increase
- clarity of subject
- completeness of event
- visual readability
- suitability as a standalone highlight

### Stage 5: Boundary Refinement

Take the top 1-3 candidates and refine the exact highlight boundaries.

Typical refinement approach:

- expand the candidate by a few seconds on both sides
- analyze a denser frame sequence near the peak
- choose better start/end boundaries so the clip does not start too late or end too abruptly

### Stage 6: Subtitle and Metadata Generation

Generate subtitles, title, and short description only after the final highlight window is fixed.

This avoids paying for expensive generation on segments that will not survive ranking.

---

## Why Temporal Information Gets Lost

Temporal information is lost when continuous motion is compressed into disconnected visual snapshots.

Key failure modes:

### 1. Motion Progression Loss

A sequence like:

- approach turn
- initiate lean
- maximum lean
- recovery

can collapse into a vague summary such as "motorcycle on road" if frames are treated independently.

### 2. Rhythm Loss

Excitement often comes from change, not from any single image.

Examples:

- sudden acceleration
- abrupt camera shake
- surprise obstacle
- overtake
- near miss

Sparse or badly aligned sampling can skip the exact peak moment.

### 3. Causality Loss

Isolated frames do not reliably preserve why a later frame matters.

### 4. Duration Loss

One frame cannot tell whether an event lasted 0.5 seconds or 5 seconds.

---

## Single-Image LLM Feeding Strategies

Many image-capable LLM APIs work best with a single image per request. In that case, temporal context must be reintroduced manually.

### Strategy A: Multi-Frame Contact Sheet

Build one image from several consecutive frames arranged in time order.

Example window:

- 12.0s
- 12.3s
- 12.6s
- 12.9s
- 13.2s
- 13.5s

Compose them into a labeled grid or strip and feed that one image to the model.

Prompt should explicitly ask for change over time, not isolated captioning.

Advantages:

- easiest to implement
- one request covers a short time window
- preserves local temporal pattern better than single-frame captioning

Tradeoff:

- if too many frames are packed into one image, each frame becomes too small
- small frames reduce detail visibility
- this is good for trend detection, but weaker for tiny visual details

**Important rule:** do not put so many frames into one sheet that each cell becomes visually tiny.

Recommended starting point:

- 4 to 6 frames per sheet for coarse ranking
- denser sampling only for later refinement

### Strategy B: Sequential Single-Frame Analysis with Rolling State

Feed one frame at a time, but carry forward a compact structured summary from prior frames.

Each request contains:

- current frame
- current timestamp
- short prior-state summary

The model should describe changes relative to the previous state, not just caption the current image.

Advantages:

- preserves more detail per frame
- works with strict single-image APIs

Tradeoffs:

- more requests
- higher cost and latency
- early errors in summaries can propagate forward

### Strategy C: Pairwise Difference Analysis

Instead of asking "what is in this image?", ask "what changed from the previous frame?"

Implementation options:

- feed previous state as text and current frame as image
- or compose a left/right two-frame comparison image

Advantages:

- directly targets change detection
- reduces useless static descriptions

Tradeoffs:

- limited long-range temporal memory
- still needs a later aggregation step

---

## Recommended Practical Combination

For the current project, the default recommended design is:

1. coarse segment scoring on the full video
2. temporal window analysis on top candidate segments using contact sheets
3. denser re-analysis near the highest scoring window
4. final boundary refinement
5. subtitle generation on the final selected clip only

This gives a better cost / quality tradeoff than:

- running provider video understanding on the full video multiple times
- or captioning every extracted frame independently

---

## Provider Video Understanding vs Manual Frame Pipeline

### Direct Provider Video Understanding

Pros:

- fastest to ship
- simpler API integration
- better native handling of temporal continuity

Cons:

- more black-box behavior
- less control over sampling and reasoning
- long videos may be compressed internally and lose detail
- harder to debug why a segment was chosen

### Manual Frame Extraction + LLM Reasoning

Pros:

- maximum control
- easier to integrate domain heuristics
- easier to debug and tune
- better long-term architecture for productization

Cons:

- time can be lost if frames are treated independently
- much more pipeline work
- more orchestration complexity
- bad sampling strategy can make results both worse and more expensive

### Project Position

Use a hybrid design:

- cheap local or heuristic scoring for broad search
- LLM analysis only on candidate windows
- preserve temporal context explicitly during candidate analysis

---

## Output Contracts

### Candidate Segment

```json
{
  "segment_id": "seg_0012",
  "start": 194.2,
  "end": 201.8,
  "coarse_score": 7.4,
  "signals": {
    "motion": 8.2,
    "audio_peak": 6.8,
    "scene_change": 4.3
  }
}
```

### Temporal Window Analysis

```json
{
  "segment_id": "seg_0012",
  "window_start": 196.0,
  "window_end": 199.0,
  "event": "rider leans into the corner, stabilizes, and accelerates out",
  "change_summary": "motion intensity rises sharply in the middle of the window",
  "peak_time": [197.1, 198.0],
  "score": 8.9,
  "recommended": true
}
```

### Final Highlight

```json
{
  "highlight_id": "hl_0001",
  "source_start": 196.6,
  "source_end": 226.6,
  "duration": 30.0,
  "score": 9.1,
  "reason": "strongest action buildup and clean exit sequence",
  "subtitle_mode": "human"
}
```

---

## Non-Goals

This spec does not yet define:

- exact subtitle generation style
- UI review workflow details
- storage schema for all intermediate artifacts
- training or fine-tuning strategy
- audio-only understanding pipeline

---

## Implementation Notes

Short-term implementation priority:

1. add temporal window contact-sheet analysis for candidate segments
2. add structured scoring outputs
3. add boundary refinement pass near the peak window
4. keep single-frame inference available as a fallback, not as the main ranking path

---

## Summary

The key idea is simple:

- do not flatten a video into unrelated frames
- keep time visible to the model
- analyze windows, not just images
- reserve dense and expensive reasoning for the best candidate regions

For single-image LLMs, the first practical default should be:

- contact sheets with timestamps
- explicit prompts about change over time
- structured outputs for ranking and later refinement
