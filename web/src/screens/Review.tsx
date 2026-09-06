import { useEffect, useState } from "react";
import { api } from "../api";
import type { NotePage } from "../types";

/**
 * Screen 3: the interrupt. OCR on handwriting is the least reliable step in the
 * system, so the student corrects it before anything is built on top of it.
 */
export default function Review({ runId, onResumed }: { runId: string; onResumed: () => void }) {
  const [pages, setPages] = useState<NotePage[]>([]);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!runId) return;
    setError("");
    api
      .notes(runId)
      .then(setPages)
      .catch((e) => setError(String(e)));
  }, [runId]);

  if (!runId) return <p className="text-sm text-slate-500">No run in progress.</p>;

  const page = pages[index];

  function edit(markdown: string) {
    setPages((rows) =>
      rows.map((row, i) => (i === index ? { ...row, markdown, edited_by_student: true } : row)),
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="font-semibold">Check the transcript</h2>
        <span className="text-sm text-slate-500">
          Anything wrong here is wrong in the finished document.
        </span>
      </div>

      {pages.length > 1 && (
        <div className="flex gap-1">
          {pages.map((row, i) => (
            <button
              key={row.page}
              onClick={() => setIndex(i)}
              className={`rounded border px-2.5 py-1 text-sm ${
                i === index ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
              }`}
            >
              {row.page}
              {row.edited_by_student ? " ·" : ""}
            </button>
          ))}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border bg-white p-3">
          {page ? (
            <img src={api.noteImage(runId, page.page)} alt={`Note page ${page.page}`} className="w-full" />
          ) : (
            <p className="text-sm text-slate-500">Waiting for the transcription to finish.</p>
          )}
        </div>
        <textarea
          value={page?.markdown ?? ""}
          onChange={(e) => edit(e.target.value)}
          className="min-h-[28rem] rounded-lg border bg-white p-3 font-mono text-sm"
          placeholder="The transcript appears here once OCR finishes."
        />
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      <button
        onClick={() => void api.approve(runId, pages).then(onResumed).catch((e) => setError(String(e)))}
        className="rounded bg-slate-900 px-4 py-2 text-sm text-white"
      >
        Looks right, carry on
      </button>
    </div>
  );
}
