import { selectActiveClip, useReviewStore } from "../../../store/reviewStore";

export function AssetPanel() {
  const job = useReviewStore((state) => state.job);
  const activeClip = useReviewStore(selectActiveClip);
  const selectClip = useReviewStore((state) => state.selectClip);

  if (!job || !activeClip) {
    return null;
  }

  return (
    <aside className="sidebar">
      <section className="panel-block">
        <h3>Source Media</h3>
        <div className="item-list">
          {job.sourceMedia.map((item, index) => (
            <button
              key={item.id}
              className={`item-card ${index === 0 ? "active" : ""}`}
              type="button"
            >
              <div className="item-left">
                <span className="icon-chip">{item.icon}</span>
                <span className={item.muted ? "muted" : ""}>{item.label}</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="panel-block">
        <h3>AI Results</h3>
        <div className="item-list">
          {job.results.map((result) => {
            const relatedClips = job.clips.filter((clip) => clip.resultId === result.id);
            return (
              <button
                key={result.id}
                className={`item-card ${activeClip.resultId === result.id ? "active" : ""}`}
                type="button"
                onClick={() => selectClip(relatedClips[0].id)}
              >
                <div className="item-left">
                  <span className="icon-chip">AI</span>
                  <div className="item-copy">
                    <span className="item-title">{result.name}</span>
                    <span className="item-subtitle">{relatedClips.map((clip) => clip.label).join(" / ")}</span>
                  </div>
                </div>
                <span className="pill neutral">{relatedClips.length}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="panel-block">
        <h3>Highlight Clips</h3>
        <div className="item-list">
          {job.clips.map((clip) => (
            <button
              key={clip.id}
              className={`item-card ${clip.id === activeClip.id ? "active" : ""} ${clip.decision === "delete" ? "deleted" : ""}`}
              type="button"
              onClick={() => selectClip(clip.id)}
            >
              <div className="item-left">
                <span className="icon-chip">V</span>
                <div className="item-copy">
                  <span className="item-title">{clip.label}</span>
                  <span className="item-subtitle">
                    {clip.start} - {clip.end}
                  </span>
                </div>
              </div>
              <span className={`pill ${clip.decision === "delete" ? "delete" : "keep"}`}>
                {clip.decision === "delete" ? "DROP" : "KEEP"}
              </span>
            </button>
          ))}
        </div>
      </section>
    </aside>
  );
}

