import { useEffect, useState } from "react";
import { api } from "../api";

/** Screen 4: the finished document, provenance highlighting, rejection report. */
export default function Library({
  runId,
  onRunId,
}: {
  runId: string;
  onRunId: (id: string) => void;
}) {
  const [html, setHtml] = useState("");
  const [report, setReport] = useState("");
  const [highlight, setHighlight] = useState(true);
  const [error, setError] = useState("");
  const [lookup, setLookup] = useState(runId);

  useEffect(() => setLookup(runId), [runId]);

  useEffect(() => {
    if (!runId) return;
    setError("");
    api
      .document(runId, highlight)
      .then((payload) => setHtml(payload.html))
      .catch((e) => setError(String(e)));
    api
      .provenance(runId)
      .then((payload) => setReport(payload.report))
      .catch(() => setReport(""));
  }, [runId, highlight]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={lookup}
          onChange={(e) => setLookup(e.target.value)}
          placeholder="run id"
          className="rounded border px-2 py-1.5 text-sm"
        />
        <button onClick={() => onRunId(lookup)} className="rounded border px-3 py-1.5 text-sm">
          Open
        </button>
        <label className="ml-auto flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={highlight}
            onChange={(e) => setHighlight(e.target.checked)}
          />
          Show where each passage came from
        </label>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {highlight && (
        <p className="text-xs text-slate-500">
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-amber-800">textbook</span>{" "}
          passages carry a page reference.{" "}
          <span className="rounded bg-red-50 px-1.5 py-0.5 text-red-700">Check this</span> marks a
          place where your notes and the slides disagree.
        </p>
      )}

      <article
        className={`prose-doc rounded-lg border bg-white p-6 ${highlight ? "" : "plain"}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />

      {report && (
        <details className="rounded-lg border bg-white p-4">
          <summary className="cursor-pointer text-sm font-semibold">
            What the textbook gate rejected
          </summary>
          <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs">{report}</pre>
        </details>
      )}
    </div>
  );
}
