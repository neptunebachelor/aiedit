import { saveReviewJob } from "../../../lib/api";
import { selectActiveClip, useReviewStore } from "../../../store/reviewStore";

export function Topbar() {
  const job = useReviewStore((state) => state.job);
  const activeClip = useReviewStore(selectActiveClip);
  const status = useReviewStore((state) => state.status);
  const cycleSubtitle = useReviewStore((state) => state.cycleSubtitle);
  const markSaved = useReviewStore((state) => state.markSaved);
  const setStatus = useReviewStore((state) => state.setStatus);

  if (!job || !activeClip) {
    return null;
  }

  async function handleSave() {
    await saveReviewJob(useReviewStore.getState().job!);
    markSaved();
  }

  function handleExport() {
    const previousStatus = useReviewStore.getState().status;
    setStatus("export", "Ready to export");

    window.setTimeout(() => {
      const current = useReviewStore.getState().status;
      if (current.kind !== "export") {
        return;
      }

      if (previousStatus.kind === "dirty") {
        useReviewStore.getState().setStatus(previousStatus.kind, previousStatus.text);
        return;
      }

      useReviewStore.getState().markSaved();
    }, 900);
  }

  return (
    <header className="topbar">
      <div className="topbar-title">
        <strong>{job.projectTitle}</strong>
        <div className="divider" />
        <span>Reviewing: {activeClip.title}</span>
      </div>
      <div className="topbar-actions">
        <div className={`status status-${status.kind}`}>{status.text}</div>
        <button className="btn" type="button" onClick={cycleSubtitle}>
          Regenerate
        </button>
        <button className="btn" type="button" onClick={handleSave}>
          Save Changes
        </button>
        <button className="btn btn-primary" type="button" onClick={handleExport}>
          Export
        </button>
      </div>
    </header>
  );
}

