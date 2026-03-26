import { startTransition, useEffect, useState } from "react";

import { loadReviewJob } from "./lib/api";
import { ReviewWorkspace } from "./features/review/components/ReviewWorkspace";
import { useReviewStore } from "./store/reviewStore";

type BootStatus = "loading" | "ready" | "error";

export default function App() {
  const [bootStatus, setBootStatus] = useState<BootStatus>("loading");
  const loadJob = useReviewStore((state) => state.loadJob);

  useEffect(() => {
    let cancelled = false;

    loadReviewJob()
      .then((job) => {
        if (cancelled) {
          return;
        }

        startTransition(() => {
          loadJob(job);
          setBootStatus("ready");
        });
      })
      .catch(() => {
        if (!cancelled) {
          setBootStatus("error");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [loadJob]);

  if (bootStatus === "loading") {
    return (
      <div className="boot-screen">
        <div className="boot-panel">
          <div className="boot-kicker">Review Workspace</div>
          <h1>Loading review workspace...</h1>
          <p>Bootstrapping review data into the React store.</p>
        </div>
      </div>
    );
  }

  if (bootStatus === "error") {
    return (
      <div className="boot-screen">
        <div className="boot-panel">
          <div className="boot-kicker">Review Workspace</div>
          <h1>Workspace failed to load.</h1>
          <p>Check <code>VITE_API_BASE_URL</code> or the fallback mock API layer in <code>src/lib/api.ts</code>.</p>
        </div>
      </div>
    );
  }

  return <ReviewWorkspace />;
}
