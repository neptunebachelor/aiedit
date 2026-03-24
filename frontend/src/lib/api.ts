import { mockReviewJob } from "../features/review/mock/reviewJob";
import type { ReviewJob } from "../types/review";

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function loadReviewJob(): Promise<ReviewJob> {
  await sleep(120);
  return mockReviewJob;
}

export async function saveReviewJob(_job: ReviewJob): Promise<void> {
  await sleep(120);
}

