import { useDeferredValue } from "react";

import { selectActiveClip, useReviewStore } from "../../../store/reviewStore";

export function Timeline() {
  const job = useReviewStore((state) => state.job);
  const activeClip = useReviewStore(selectActiveClip);
  const selectClip = useReviewStore((state) => state.selectClip);
  const deferredActiveClipId = useDeferredValue(activeClip?.id ?? null);

  if (!job || !activeClip) {
    return null;
  }

  const playheadPosition = activeClip.left + activeClip.width / 2;

  return (
    <section className="timeline-shell">
      <div className="timeline-grid">
        <div className="track-labels">
          <div className="track-label ruler-label">&nbsp;</div>
          <div className="track-label">AI Tags</div>
          <div className="track-label">Subtitles</div>
          <div className="track-label">Highlight Clips</div>
          <div className="track-label">Original Video</div>
        </div>

        <div className="track-area">
          <div className="ruler">
            <div className="tick-labels">
              <span>0:00</span>
              <span>0:30</span>
              <span>1:00</span>
              <span>1:30</span>
              <span>2:00</span>
              <span>2:30</span>
              <span>3:00</span>
              <span>3:30</span>
            </div>
          </div>

          <div className="track">
            {job.clips.map((clip) => {
              const result = job.results.find((item) => item.id === clip.resultId)!;
              return (
                <button
                  key={`tag-${clip.id}`}
                  className={`track-clip ${result.trackClass} ${deferredActiveClipId === clip.id ? "active" : ""} ${clip.decision === "delete" ? "deleted" : ""}`}
                  type="button"
                  style={{ left: `${clip.left}%`, width: `${clip.width}%` }}
                  onClick={() => selectClip(clip.id)}
                >
                  {result.shortLabel}
                </button>
              );
            })}
          </div>

          <div className="track">
            {job.clips.map((clip) => (
              <button
                key={`subtitle-${clip.id}`}
                className={`track-clip subtitle-track-clip ${deferredActiveClipId === clip.id ? "active" : ""} ${clip.decision === "delete" ? "deleted" : ""}`}
                type="button"
                style={{ left: `${clip.left}%`, width: `${clip.width}%` }}
                onClick={() => selectClip(clip.id)}
                title={clip.subtitle}
              >
                {clip.label}
              </button>
            ))}
          </div>

          <div className="track">
            {job.clips.map((clip) => (
              <button
                key={`highlight-${clip.id}`}
                className={`track-clip ${clip.highlightClass} ${deferredActiveClipId === clip.id ? "active" : ""} ${clip.decision === "delete" ? "deleted" : ""}`}
                type="button"
                style={{ left: `${clip.left}%`, width: `${clip.width}%` }}
                onClick={() => selectClip(clip.id)}
              >
                {clip.label}
              </button>
            ))}
          </div>

          <div className="track waveform-track">
            <div className="waveform" />
          </div>

          <div className="playhead" style={{ left: `${playheadPosition}%` }} />
        </div>
      </div>
    </section>
  );
}
