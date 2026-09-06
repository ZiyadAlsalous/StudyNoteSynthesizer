import type { ChapterRange, Course, NotePage, Progress, RunRecord } from "./types";

const base = "/api";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export const api = {
  courses: () => json<Course[]>("/courses"),

  createCourse: (id: string, title: string) =>
    json<{ id: string }>("/courses", { method: "POST", body: JSON.stringify({ id, title }) }),

  uploadTextbook: async (course: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${base}/courses/${course}/textbook`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) throw new Error(await response.text());
    return (await response.json()) as { chunks: number };
  },

  chapters: (course: string) => json<ChapterRange[]>(`/courses/${course}/chapters`),

  saveChapters: (course: string, chapters: ChapterRange[]) =>
    json<{ chunks: number }>(`/courses/${course}/chapters`, {
      method: "PUT",
      body: JSON.stringify({ chapters }),
    }),

  startRun: (course: string, chapter: string, slidesDir: string, notesDir: string) =>
    json<{ run_id: string }>(`/courses/${course}/chapters/${chapter}/runs`, {
      method: "POST",
      body: JSON.stringify({ slides_dir: slidesDir, notes_dir: notesDir }),
    }),

  run: (runId: string) => json<RunRecord>(`/runs/${runId}`),

  notes: (runId: string) => json<NotePage[]>(`/runs/${runId}/notes`),

  noteImage: (runId: string, page: number) => `${base}/runs/${runId}/notes/${page}/image`,

  approve: (runId: string, notes?: unknown[]) =>
    json<{ run_id: string }>(`/runs/${runId}/approve`, {
      method: "POST",
      body: JSON.stringify({ notes: notes ?? null }),
    }),

  document: (runId: string, highlight: boolean) =>
    json<{ markdown: string; html: string }>(`/runs/${runId}/document?highlight=${highlight}`),

  provenance: (runId: string) => json<{ report: string }>(`/runs/${runId}/provenance`),
};

/** Server-sent progress. The caller owns closing the stream. */
export function streamProgress(runId: string, onEvent: (event: Progress) => void): () => void {
  const source = new EventSource(`${base}/runs/${runId}/events`);
  source.onmessage = (message) => onEvent(JSON.parse(message.data) as Progress);
  source.onerror = () => source.close();
  return () => source.close();
}
