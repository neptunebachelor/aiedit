import { AssetPanel } from "./AssetPanel";
import { InspectorPanel } from "./InspectorPanel";
import { Timeline } from "./Timeline";
import { Topbar } from "./Topbar";
import { ViewerPanel } from "./ViewerPanel";

export function ReviewWorkspace() {
  return (
    <div className="app-shell">
      <Topbar />
      <main className="workspace-main">
        <AssetPanel />
        <ViewerPanel />
        <InspectorPanel />
      </main>
      <Timeline />
    </div>
  );
}

