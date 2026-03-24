import type { ReviewJob } from "../../../types/review";

export const mockReviewJob: ReviewJob = {
  projectTitle: "Project: Racing Highlights",
  sourceMedia: [
    { id: "source-lap", icon: "V", label: "lap01.mp4" },
    { id: "source-preview", icon: "P", label: "preview.mp4", muted: true },
    { id: "source-srt", icon: "S", label: "final.srt" },
    { id: "source-editable", icon: "J", label: "editable.json" }
  ],
  results: [
    { id: "overtake", name: "Overtake Clip", shortLabel: "OVERTAKE", trackClass: "purple" },
    { id: "action", name: "Action Moment", shortLabel: "ACTION", trackClass: "blue" }
  ],
  clips: [
    {
      id: "clip-01",
      label: "Clip 01",
      title: "Highlight Clip 01",
      resultId: "overtake",
      subtitle: "I send it down the inside and come out ahead.",
      alternatives: [
        "I send it down the inside and come out ahead.",
        "Late on the brakes, I squeeze through and take the spot.",
        "One clean dive, one clean pass, and the corner is mine."
      ],
      start: "00:00:42",
      end: "00:00:55",
      score: 9.1,
      tags: ["Late Brake", "Position Gain", "Crowd Pop"],
      note: "Best opening clip. Strong positional change and a clean caption beat.",
      decision: "keep",
      left: 16,
      width: 17,
      highlightClass: "yellow",
      progress: 24
    },
    {
      id: "clip-02",
      label: "Clip 02",
      title: "Highlight Clip 02",
      resultId: "action",
      subtitle: "The rear steps out, but I catch it and stay flat.",
      alternatives: [
        "The rear steps out, but I catch it and stay flat.",
        "The car snaps loose for a second and I hold onto it.",
        "It starts to slide, I catch the drift, and the run survives."
      ],
      start: "00:01:21",
      end: "00:01:38",
      score: 8.3,
      tags: ["Action Moment", "Car Control", "Onboard"],
      note: "Visually dramatic, but the subtitle needs tighter timing at the back end.",
      decision: "keep",
      left: 34,
      width: 20,
      highlightClass: "red",
      progress: 46
    },
    {
      id: "clip-03",
      label: "Clip 03",
      title: "Highlight Clip 03",
      resultId: "overtake",
      subtitle: "I lean into the turn and accelerate past him!",
      alternatives: [
        "I lean into the turn and accelerate past him!",
        "The exit hooks up perfectly and I drive by on power.",
        "I stay planted through the apex and blast straight past."
      ],
      start: "00:02:15",
      end: "00:02:30",
      score: 8.7,
      tags: ["Overtake", "High Speed", "Onboard View"],
      note: "Strong finish clip with stable framing. Good candidate for the hero beat.",
      decision: "keep",
      left: 55,
      width: 14,
      highlightClass: "pink",
      progress: 62
    }
  ]
};

