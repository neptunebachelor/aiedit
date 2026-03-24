import { selectActiveClip, selectActiveResult, useReviewStore } from "../../../store/reviewStore";

export function InspectorPanel() {
  const activeClip = useReviewStore(selectActiveClip);
  const activeResult = useReviewStore(selectActiveResult);
  const updateSubtitle = useReviewStore((state) => state.updateSubtitle);
  const updateStart = useReviewStore((state) => state.updateStart);
  const updateEnd = useReviewStore((state) => state.updateEnd);
  const setDecision = useReviewStore((state) => state.setDecision);

  if (!activeClip || !activeResult) {
    return null;
  }

  return (
    <aside className="inspector">
      <section className="panel-block">
        <h3>Clip Info</h3>
        <div className="panel-body">
          <div className="headline">{activeClip.title}</div>
          <div className="reason-copy">{activeClip.note}</div>
          <div className="info-grid">
            <div className="info-row">
              <span>Result</span>
              <strong>{activeResult.name}</strong>
            </div>
            <div className="info-row">
              <span>Window</span>
              <strong>
                {activeClip.start} - {activeClip.end}
              </strong>
            </div>
          </div>
        </div>
      </section>

      <section className="panel-block">
        <h3>AI Tags</h3>
        <div className="panel-body">
          <div className="tag-group">
            {activeClip.tags.map((tag) => (
              <span key={tag} className="tag-chip">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="panel-block">
        <div className="panel-body score-card">
          <div>
            <div className="eyebrow">Model Confidence</div>
            <strong>{activeClip.score.toFixed(1)}</strong>
          </div>
          <div className="score-bar">
            <div className="score-marker" style={{ left: `${Math.max(10, Math.min(95, activeClip.score * 10))}%` }} />
          </div>
        </div>
      </section>

      <section className="panel-block">
        <h3>Subtitle</h3>
        <div className="panel-body">
          <textarea
            className="textarea"
            value={activeClip.subtitle}
            onChange={(event) => updateSubtitle(event.target.value)}
          />
          <div className="field-row">
            <label htmlFor="start-time">Start Time</label>
            <input
              id="start-time"
              className="field-input"
              value={activeClip.start}
              onChange={(event) => updateStart(event.target.value)}
            />
          </div>
          <div className="field-row">
            <label htmlFor="end-time">End Time</label>
            <input
              id="end-time"
              className="field-input"
              value={activeClip.end}
              onChange={(event) => updateEnd(event.target.value)}
            />
          </div>
          <div className="field-row review-note-row">
            <label>Review Note</label>
            <div className="note-box">{activeClip.note}</div>
          </div>
          <div className="decision-row">
            <button
              className={`btn decision-btn ${activeClip.decision === "keep" ? "active keep-active" : ""}`}
              type="button"
              onClick={() => setDecision("keep")}
            >
              Keep
            </button>
            <button
              className={`btn decision-btn ${activeClip.decision === "delete" ? "active delete-active" : ""}`}
              type="button"
              onClick={() => setDecision("delete")}
            >
              Delete
            </button>
          </div>
        </div>
      </section>
    </aside>
  );
}

