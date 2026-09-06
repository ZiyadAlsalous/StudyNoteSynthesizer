import { useEffect, useState } from "react";
import { api } from "../api";
import type { ChapterRange, Course } from "../types";

/** Screen 1: create a course, ingest a textbook, correct the detected chapters. */
export default function Courses({
  course,
  onCourse,
}: {
  course: string;
  onCourse: (id: string) => void;
}) {
  const [courses, setCourses] = useState<Course[]>([]);
  const [chapters, setChapters] = useState<ChapterRange[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.courses().then(setCourses).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (course) api.chapters(course).then(setChapters).catch(() => setChapters([]));
  }, [course]);

  async function create(form: FormData) {
    const id = String(form.get("id") ?? "");
    if (!id) return;
    await api.createCourse(id, String(form.get("title") ?? id));
    setCourses(await api.courses());
    onCourse(id);
  }

  async function upload(file: File) {
    setBusy("Parsing the textbook and building the index. This takes a while.");
    setError("");
    try {
      const { chunks } = await api.uploadTextbook(course, file);
      setBusy(`${chunks} chunks indexed.`);
      setChapters(await api.chapters(course));
    } catch (e) {
      setError(String(e));
      setBusy("");
    }
  }

  function edit(index: number, field: "page_start" | "page_end", value: number) {
    setChapters((rows) =>
      rows.map((row, i) =>
        i === index ? { ...row, [field]: value, manual_override: true } : row,
      ),
    );
  }

  return (
    <div className="space-y-8">
      <section className="rounded-lg border bg-white p-5">
        <h2 className="mb-3 font-semibold">Courses</h2>
        <div className="mb-4 flex flex-wrap gap-2">
          {courses.map((row) => (
            <button
              key={row.id}
              onClick={() => onCourse(row.id)}
              className={`rounded border px-3 py-1.5 text-sm ${
                course === row.id ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
              }`}
            >
              {row.title}
            </button>
          ))}
          {courses.length === 0 && <p className="text-sm text-slate-500">No courses yet.</p>}
        </div>
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void create(new FormData(event.currentTarget));
            event.currentTarget.reset();
          }}
        >
          <input name="id" placeholder="cs3340" className="rounded border px-2 py-1.5 text-sm" />
          <input
            name="title"
            placeholder="Analysis of Algorithms"
            className="flex-1 rounded border px-2 py-1.5 text-sm"
          />
          <button className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white">Add</button>
        </form>
      </section>

      {course && (
        <section className="rounded-lg border bg-white p-5">
          <h2 className="mb-1 font-semibold">Textbook</h2>
          <p className="mb-3 text-sm text-slate-500">
            Chapter ranges come from the PDF outline. Where the outline is missing or wrong, fix
            the pages here: a wrong range silently poisons chapter scoping.
          </p>
          <input
            type="file"
            accept="application/pdf"
            className="mb-3 text-sm"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void upload(file);
            }}
          />
          {busy && <p className="mb-3 text-sm text-slate-600">{busy}</p>}
          {error && <p className="mb-3 text-sm text-red-600">{error}</p>}

          {chapters.length > 0 && (
            <>
              <table className="w-full text-sm">
                <thead className="text-left text-slate-500">
                  <tr>
                    <th className="py-1">Chapter</th>
                    <th className="w-24">First page</th>
                    <th className="w-24">Last page</th>
                    <th className="w-20">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {chapters.map((row, index) => (
                    <tr key={row.chapter} className="border-t">
                      <td className="py-1.5">{row.title || row.chapter}</td>
                      <td>
                        <input
                          type="number"
                          value={row.page_start}
                          onChange={(e) => edit(index, "page_start", Number(e.target.value))}
                          className="w-20 rounded border px-1.5 py-1"
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          value={row.page_end}
                          onChange={(e) => edit(index, "page_end", Number(e.target.value))}
                          className="w-20 rounded border px-1.5 py-1"
                        />
                      </td>
                      <td className="text-slate-500">{row.manual_override ? "manual" : "outline"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button
                onClick={() => void api.saveChapters(course, chapters).then(() => setBusy("Re-indexed."))}
                className="mt-3 rounded bg-slate-900 px-3 py-1.5 text-sm text-white"
              >
                Save and re-index
              </button>
            </>
          )}
        </section>
      )}
    </div>
  );
}
