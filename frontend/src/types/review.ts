export type ReviewStatusKind = "saved" | "dirty" | "export";
export type ReviewDecision = "keep" | "delete";

export interface SourceMediaItem {
  id: string;
  icon: string;
  label: string;
  muted?: boolean;
}

export interface ReviewResult {
  id: string;
  name: string;
  shortLabel: string;
  trackClass: string;
}

export interface ReviewClip {
  id: string;
  label: string;
  title: string;
  resultId: string;
  subtitle: string;
  alternatives: string[];
  start: string;
  end: string;
  score: number;
  tags: string[];
  note: string;
  decision: ReviewDecision;
  left: number;
  width: number;
  highlightClass: string;
  progress: number;
}

export interface ReviewJob {
  projectTitle: string;
  sourceMedia: SourceMediaItem[];
  results: ReviewResult[];
  clips: ReviewClip[];
}

export interface StatusBadge {
  kind: ReviewStatusKind;
  text: string;
}

