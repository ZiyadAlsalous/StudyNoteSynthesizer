import { useEffect, useState } from "react";
import { api } from "./api";
import type { ChapterRange, Course } from "./types";

/**
 * Everything you do before a run: pick a course, ingest the textbook, correct
 * the chapter page ranges, point at the slides and notes.
 *
 * Collapses into a sidebar once a run is in flight, because none of it is
 * editable mid-run anyway.
 */
export default function Setup({
  course,
  onCourse,
  onStart,
  compact,
}: {
  course: string;
  onCourse: (id: string) => void;
  onStart: (runId: string) => void;
  compact: boolean;
}) {
  const [courses, setCourses] = useState<Course[]>([]);
  const [chapters, setChapters] = useState<ChapterRange[]>([]);
  const [chapter, setChapter] = useState("");
  const [slides, setSlides] = useState("");
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    api.courses().then(setCourses).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!course) return;
    api.chapters(course).then(setChapters).catch(() => setChapters([]));
  }, [course]);

  async function addCourse(form: FormData) {
    const id = String(form.get("id") ?? "").trim();
    if (!id) return;
    await api.createCourse(id, String(form.get("title") ?? "").trim() || id);
    setCourses(await api.courses());
    onCourse(id);
  }

  async function uploadTextbook(file: File) {
    setStatus("Parsing the textbook and building the index…");
    setError("");
    try {
      const { chunks } = await api.uploadTextbook(course, file);
      setStatus(`${chunks} chunks indexed.`);
      setChapters(await api.chapters(course));
    } catch (e) {
      setError(String(e));
      setStatus("");
    }
  }

  function editRange(index: number, field: "page_start" | "page_end", value: number) {
    setChapters((rows) =>
      rows.map((row, i) => (i === index ? { ...row, [field]: value, manual_override: true } : row)),
    );
  }

  async function start() {
    setError("");
    try {
      const { run_id } = await api.startRun(course, chapter, slides, notes);
      onStart(run_id);
    } catch (e) {
      setError(String(e));
    }
  }

  const ready = Boolean(course && chapter && slides && notes);
  const field = "w-full rounded border px-2 py-1.5 text-sm";

  if (compact) {
    return (
      <div className="space-y-1 text-sm">
        <p className="font-medium">{courses.find((c) => c.id === course)?.title ?? course}</p>
        <p className="text-slate-500">{chapters.find((c) => c.chapter === chapter)?.title ?? chapter}</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-slate-500">
          Course
        </label>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {courses.map((row) => (
            <button
              key={row.id}
              onClick={() => onCourse(row.id)}
              className={`rounded border px-2.5 py-1 text-sm ${
                course === row.id ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
              }`}
            >
              {row.title}
            </button>
          ))}
        </div>
        <form
          className="flex gap-1.5"
          onSubmit={(event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            event.currentTarget.reset();
            void addCourse(form);
          }}
        >
          <input name="id" placeholder="cs3340" className={`${field} w-28`} />
          <input name="title" placeholder="Analysis of Algorithms" className={field} />
          <button className="rounded border px-2.5 py-1 text-sm hover:bg-slate-50">Add</button>
        </form>
      </div>

      {course && (
        <div>
          <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-slate-500">
            Textbook
          </label>
          <input
            type="file"
            accept="application/pdf"
            className="w-full text-sm"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void uploadTextbook(file);
            }}
          />
          {chapters.length > 0 && (
            <button
              onClick={() => setEditing((open) => !open)}
              className="mt-2 text-xs text-slate-500 underline"
            >
              {editing ? "Hide" : `${chapters.length} chapters detected — check the page ranges`}
            </button>
          )}
          {editing && (
            <div className="mt-2 max-h-56 space-y-1 overflow-y-auto rounded border p-2">
              {chapters.map((row, index) => (
                <div key={row.chapter} className="flex items-center gap-1.5 text-sm">
                  <span className="flex-1 truncate">{row.title || row.chapter}</span>
                  <input
                    type="number"
                    value={row.page_start}
                    onChange={(e) => editRange(index, "page_start", Number(e.target.value))}
                    className="w-16 rounded border px-1 py-0.5"
                  />
                  <input
                    type="number"
                    value={row.page_end}
                    onChange={(e) => editRange(index, "page_end", Number(e.target.value))}
                    className="w-16 rounded border px-1 py-0.5"
                  />
                </div>
              ))}
              <button
                onClick={() =>
                  void api
                    .saveChapters(course, chapters)
                    .then(({ chunks }) => setStatus(`Re-indexed, ${chunks} chunks.`))
                    .catch((e) => setError(String(e)))
                }
                className="mt-1 rounded border px-2.5 py-1 text-xs hover:bg-slate-50"
              >
                Save and re-index
              </button>
            </div>
          )}
        </div>
      )}

      {course && (
        <div className="space-y-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            This chapter
          </label>
          <select value={chapter} onChange={(e) => setChapter(e.target.value)} className={field}>
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
            className={field}
          />
          <input
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="note photos folder"
            className={field}
          />
          <button
            disabled={!ready}
            onClick={() => void start()}
            className="w-full rounded bg-slate-900 py-2 text-sm text-white disabled:opacity-40"
          >
            Build this chapter
          </button>
        </div>
      )}

      {status && <p className="text-xs text-slate-600">{status}</p>}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
