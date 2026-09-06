import { useEffect, useRef, useState } from "react";
import { api, streamProgress } from "../api";
import type { ChapterRange, Progress } from "../types";

const LABELS: Record<string, string> = {
  ingest_slides: "Reading the slides",
  ingest_notes: "Transcribing your notes",
  awaiting_review: "Waiting for you to check the transcript",
  extract_concepts: "Finding the concepts",
  find_gaps: "Finding what the lecture leaves unresolved",
  draft_concepts: "Drafting from slides and notes",
  retrieve_textbook: "Consulting the textbook",
  synthesize: "Writing the document",
  verify: "Checking every claim",
  done: "Finished",
};

/** Screen 2: pick a chapter, point at the sources, watch the graph run. */
export default function ChapterRun({
  course,
  onStarted,
}: {
  course: string;
  onStarted: (runId: string) => void;
}) {
  const [chapters, setChapters] = useState<ChapterRange[]>([]);
  const [chapter, setChapter] = useState("");
  const [slides, setSlides] = useState("");
  const [notes, setNotes] = useState("");
  const [events, setEvents] = useState<Progress[]>([]);
  const [error, setError] = useState("");
  const close = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (course) api.chapters(course).then(setChapters).catch(() => setChapters([]));
    return () => close.current?.();
  }, [course]);

  async function start() {
    setEvents([]);
    setError("");
    try {
      const { run_id } = await api.startRun(course, chapter, slides, notes);
      close.current = streamProgress(run_id, (event) => {
        setEvents((rows) => [...rows, event]);
        if (event.node === "awaiting_review") onStarted(run_id);
        if (event.error) setError(event.error);
      });
    } catch (e) {
      setError(String(e));
    }
  }

  const ready = course && chapter && slides && notes;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border bg-white p-5">
        <h2 className="mb-3 font-semibold">Start a chapter</h2>
        {!course && <p className="text-sm text-slate-500">Pick a course first.</p>}
        <div className="grid gap-3 sm:grid-cols-3">
          <select
            value={chapter}
            onChange={(e) => setChapter(e.target.value)}
            className="rounded border px-2 py-1.5 text-sm"
          >
            <option value="">Chapter…</option>
            {chapters.map((row) => (
              <option key={row.chapter} value={row.chapter}>
                {row.title || row.chapter}
              </option>
            ))}
          </select>
          <input
            value={slides}
            onChange={(e) => setSlides(e.target.value)}
            placeholder="slides folder"
            className="rounded border px-2 py-1.5 text-sm"
          />
          <input
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="note images folder"
            className="rounded border px-2 py-1.5 text-sm"
          />
        </div>
        <button
          disabled={!ready}
          onClick={() => void start()}
          className="mt-3 rounded bg-slate-900 px-3 py-1.5 text-sm text-white disabled:opacity-40"
        >
          Run chapter
        </button>
        {error && <pre className="mt-3 overflow-x-auto rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>}
      </section>

      {events.length > 0 && (
        <section className="rounded-lg border bg-white p-5">
          <h2 className="mb-3 font-semibold">Progress</h2>
          <ol className="space-y-1.5 text-sm">
            {events.map((event, index) => (
              <li key={index} className="flex gap-3">
                <span className="w-64 shrink-0">{LABELS[event.node] ?? event.node}</span>
                <span className="text-slate-500">{event.detail}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
