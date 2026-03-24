import { selectActiveClip, selectActiveResult, useReviewStore } from "../../../store/reviewStore";

export function ViewerPanel() {
  const job = useReviewStore((state) => state.job);
  const activeClip = useReviewStore(selectActiveClip);
  const activeResult = useReviewStore(selectActiveResult);
  const isPlaying = useReviewStore((state) => state.isPlaying);
  const togglePlayback = useReviewStore((state) => state.togglePlayback);
  const selectClip = useReviewStore((state) => state.selectClip);

  if (!job || !activeClip || !activeResult) {
    return null;
  }

  const currentIndex = job.clips.findIndex((clip) => clip.id === activeClip.id);
  const previousClip = currentIndex > 0 ? job.clips[currentIndex - 1] : null;
  const nextClip = currentIndex < job.clips.length - 1 ? job.clips[currentIndex + 1] : null;

  return (
    <section className="viewer panel-card">
      <div className="viewer-header">
        <div>
          <div className="eyebrow">Review Sequence</div>
          <div className="viewer-title">{activeClip.title}</div>
        </div>
        <div className="viewer-meta">
          <span className={`pill ${activeClip.decision === "delete" ? "delete" : "keep"}`}>
            {activeClip.decision === "delete" ? "DELETE" : "KEEP"}
          </span>
          <span className="pill neutral">
            {activeClip.start} - {activeClip.end}
          </span>
        </div>
      </div>

      <div
        className={`video-frame ${activeClip.decision === "delete" ? "deleted" : ""}`}
        data-result={activeResult.id}
      >
        <div className="video-badge">
          {activeClip.decision === "delete" ? `${activeResult.name} - Pending Delete` : activeResult.name}
        </div>
        <div className="road" />
        <div className="lane" />
        <div className="scrim" />
        <div className="hud">
          <div className="subtitle-preview">{activeClip.subtitle}</div>
        </div>
      </div>

      <div className="progress-bar-shell">
        <div className="progress-bar-fill" style={{ width: `${activeClip.progress}%` }} />
      </div>

      <div className="player-controls">
        <button className="control-btn" type="button" onClick={() => selectClip(job.clips[0].id)}>
          |&lt;
        </button>
        <button
          className="control-btn"
          type="button"
          disabled={!previousClip}
          onClick={() => previousClip && selectClip(previousClip.id)}
        >
          &lt;
        </button>
        <button className="control-btn" type="button" onClick={togglePlayback}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          className="control-btn"
          type="button"
          disabled={!nextClip}
          onClick={() => nextClip && selectClip(nextClip.id)}
        >
          &gt;
        </button>
        <button
          className="control-btn"
          type="button"
          onClick={() => selectClip(job.clips[job.clips.length - 1].id)}
        >
          &gt;|
        </button>
        <div className="rate-pill">1.0x</div>
      </div>
    </section>
  );
}

