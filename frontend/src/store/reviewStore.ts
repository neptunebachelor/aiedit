import { create } from "zustand";

import type {
  ReviewClip,
  ReviewDecision,
  ReviewJob,
  ReviewStatusKind,
  StatusBadge
} from "../types/review";

interface ReviewStoreState {
  job: ReviewJob | null;
  activeClipId: string | null;
  isPlaying: boolean;
  dirty: boolean;
  status: StatusBadge;
  loadJob: (job: ReviewJob) => void;
  selectClip: (clipId: string) => void;
  updateSubtitle: (subtitle: string) => void;
  updateStart: (start: string) => void;
  updateEnd: (end: string) => void;
  setDecision: (decision: ReviewDecision) => void;
  cycleSubtitle: () => void;
  togglePlayback: () => void;
  setStatus: (kind: ReviewStatusKind, text: string) => void;
  markSaved: () => void;
}

const savedStatus: StatusBadge = {
  kind: "saved",
  text: "Autosaved"
};

function updateClipCollection(
  job: ReviewJob,
  activeClipId: string | null,
  updater: (clip: ReviewClip) => ReviewClip
) {
  if (!activeClipId) {
    return job;
  }

  return {
    ...job,
    clips: job.clips.map((clip) => (clip.id === activeClipId ? updater(clip) : clip))
  };
}

export const useReviewStore = create<ReviewStoreState>((set, get) => ({
  job: null,
  activeClipId: null,
  isPlaying: false,
  dirty: false,
  status: savedStatus,
  loadJob: (job) =>
    set({
      job,
      activeClipId: job.clips.at(-1)?.id ?? null,
      isPlaying: false,
      dirty: false,
      status: savedStatus
    }),
  selectClip: (clipId) =>
    set((state) => ({
      activeClipId: clipId,
      isPlaying: false,
      status: state.dirty
        ? {
            kind: "dirty",
            text: "Unsaved changes"
          }
        : {
            kind: "saved",
            text: "Autosaved"
          }
    })),
  updateSubtitle: (subtitle) =>
    set((state) => {
      if (!state.job) {
        return state;
      }

      return {
        job: updateClipCollection(state.job, state.activeClipId, (clip) => ({
          ...clip,
          subtitle
        })),
        dirty: true,
        status: {
          kind: "dirty",
          text: "Subtitle edited"
        }
      };
    }),
  updateStart: (start) =>
    set((state) => {
      if (!state.job) {
        return state;
      }

      return {
        job: updateClipCollection(state.job, state.activeClipId, (clip) => ({
          ...clip,
          start
        })),
        dirty: true,
        status: {
          kind: "dirty",
          text: "Start time updated"
        }
      };
    }),
  updateEnd: (end) =>
    set((state) => {
      if (!state.job) {
        return state;
      }

      return {
        job: updateClipCollection(state.job, state.activeClipId, (clip) => ({
          ...clip,
          end
        })),
        dirty: true,
        status: {
          kind: "dirty",
          text: "End time updated"
        }
      };
    }),
  setDecision: (decision) =>
    set((state) => {
      if (!state.job) {
        return state;
      }

      return {
        job: updateClipCollection(state.job, state.activeClipId, (clip) => ({
          ...clip,
          decision
        })),
        dirty: true,
        status: {
          kind: "dirty",
          text: decision === "delete" ? "Clip marked for removal" : "Clip kept"
        }
      };
    }),
  cycleSubtitle: () =>
    set((state) => {
      if (!state.job || !state.activeClipId) {
        return state;
      }

      return {
        job: updateClipCollection(state.job, state.activeClipId, (clip) => {
          const currentIndex = clip.alternatives.indexOf(clip.subtitle);
          const nextIndex = currentIndex === -1 ? 0 : (currentIndex + 1) % clip.alternatives.length;
          return {
            ...clip,
            subtitle: clip.alternatives[nextIndex]
          };
        }),
        dirty: true,
        status: {
          kind: "dirty",
          text: "Alternative subtitle loaded"
        }
      };
    }),
  togglePlayback: () =>
    set((state) => {
      const isPlaying = !state.isPlaying;
      return {
        isPlaying,
        status: isPlaying
          ? {
              kind: "export",
              text: "Preview playing"
            }
          : state.dirty
            ? {
                kind: "dirty",
                text: "Unsaved changes"
              }
            : savedStatus
      };
    }),
  setStatus: (kind, text) =>
    set({
      status: {
        kind,
        text
      }
    }),
  markSaved: () =>
    set({
      dirty: false,
      status: savedStatus
    })
}));

export function selectActiveClip(state: ReviewStoreState) {
  if (!state.job || !state.activeClipId) {
    return null;
  }

  return state.job.clips.find((clip) => clip.id === state.activeClipId) ?? null;
}

export function selectActiveResult(state: ReviewStoreState) {
  const clip = selectActiveClip(state);
  if (!clip || !state.job) {
    return null;
  }

  return state.job.results.find((result) => result.id === clip.resultId) ?? null;
}
