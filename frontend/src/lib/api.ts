import { mockReviewJob } from "../features/review/mock/reviewJob";
import type { ReviewJob } from "../types/review";

const apiBaseUrl = String(import.meta.env.VITE_API_BASE_URL ?? "")
  .trim()
  .replace(/\/+$/, "");
const useRemoteApi = apiBaseUrl.length > 0;

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function loadReviewJob(): Promise<ReviewJob> {
  if (useRemoteApi) {
    return requestJson<ReviewJob>("/api/review-job");
  }

  await sleep(120);
  return mockReviewJob;
}

export async function saveReviewJob(job: ReviewJob): Promise<void> {
  if (useRemoteApi) {
    await requestJson("/api/review-job", {
      method: "PUT",
      body: JSON.stringify(job)
    });
    return;
  }

  await sleep(120);
}
