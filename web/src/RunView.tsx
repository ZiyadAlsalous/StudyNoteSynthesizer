import { useEffect, useMemo, useRef, useState } from "react";
import { api, streamProgress } from "./api";
import type { NotePage, Progress } from "./types";

const STEPS = [
  ["ingest_slides", "Reading the slides"],
  ["ingest_notes", "Transcribing your notes"],
  ["awaiting_review", "Waiting for you to check the transcript"],
  ["extract_concepts", "Finding the concepts"],
  ["find_gaps", "Finding what the lecture leaves unresolved"],
  ["draft_concepts", "Drafting from slides and notes"],
  ["retrieve_textbook", "Consulting the textbook"],
  ["synthesize", "Writing the document"],
  ["verify", "Checking every claim"],
] as const;

type Phase = "running" | "review" | "done";

/**
 * One view for the whole life of a run: progress, the transcript interrupt, and
 * the finished document. They are the same run at three moments, so switching
 * between them is a state change and never a page load.
 */
export default function RunView({ runId }: { runId: string }) {
  const [events, setEvents] = useState<Progress[]>([]);
  const [phase, setPhase] = useState<Phase>("running");
  const [pages, setPages] = useState<NotePage[]>([]);
  const [page, setPage] = useState(0);
  const [document, setDocument] = useState("");
  const [report, setReport] = useState("");
  const [highlight, setHighlight] = useState(true);
  const [error, setError] = useState("");
  const close = useRef<(() => void) | null>(null);

  useEffect(() => {
    setEvents([]);
    setPhase("running");
    setDocument("");
    setError("");
    close.current?.();
    close.current = streamProgress(runId, (event) => {
      setEvents((rows) => [...rows, event]);
      if (event.error) setError(event.error);
      if (event.node === "awaiting_review") setPhase("review");
      if (event.node === "done") setPhase("done");
    });
    return () => close.current?.();
  }, [runId]);

  useEffect(() => {
    if (phase !== "review") return;
    api.notes(runId).then(setPages).catch((e) => setError(String(e)));
  }, [phase, runId]);

  // Fetched once. Toggling provenance is a CSS class, not a round trip.
  useEffect(() => {
    if (phase !== "done") return;
    api.document(runId, true).then((d) => setDocument(d.html)).catch((e) => setError(String(e)));
    api.provenance(runId).then((p) => setReport(p.report)).catch(() => setReport(""));
  }, [phase, runId]);

  const reached = useMemo(() => new Set(events.map((e) => e.node)), [events]);
  const detail = useMemo(
    () => Object.fromEntries(events.map((e) => [e.node, e.detail])),
    [events],
  );

  async function resume() {
    setPhase("running");
    try {
      await api.approve(runId, pages);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="space-y-6">
      <ol className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {STEPS.map(([key, label]) => (
          <li
            key={key}
            className={reached.has(key) ? "text-slate-900" : "text-slate-300"}
            title={detail[key] ?? ""}
          >
            <span className={reached.has(key) ? "text-green-600" : ""}>●</span> {label}
            {detail[key] ? <span className="text-slate-400"> · {detail[key]}</span> : null}
          </li>
        ))}
      </ol>

      {error && (
        <pre className="overflow-x-auto rounded bg-red-50 p-3 text-xs text-red-700">{error}</pre>
      )}

      {phase === "review" && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">Check the transcript</h2>
            <p className="text-sm text-slate-500">
              Anything wrong here is wrong in the finished document. Handwriting is the least
              reliable thing this system reads.
            </p>
          </div>
          {pages.length > 1 && (
            <div className="flex gap-1">
              {pages.map((row, index) => (
                <button
                  key={row.page}
                  onClick={() => setPage(index)}
                  className={`rounded border px-2.5 py-1 text-sm ${
                    index === page ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
                  }`}
                >
                  {row.page}
                  {row.edited_by_student ? " ·" : ""}
                </button>
              ))}
            </div>
          )}
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-lg border bg-white p-2">
              {pages[page] && (
                <img
                  src={api.noteImage(runId, pages[page].page)}
                  alt={`Note page ${pages[page].page}`}
                  className="w-full"
                />
              )}
            </div>
            <textarea
              value={pages[page]?.markdown ?? ""}
              onChange={(e) =>
                setPages((rows) =>
                  rows.map((row, i) =>
                    i === page ? { ...row, markdown: e.target.value, edited_by_student: true } : row,
                  ),
                )
              }
              className="min-h-[26rem] rounded-lg border bg-white p-3 font-mono text-sm"
            />
          </div>
          <button
            onClick={() => void resume()}
            className="rounded bg-slate-900 px-4 py-2 text-sm text-white"
          >
            Looks right, carry on
          </button>
        </section>
      )}

      {phase === "done" && (
        <section className="space-y-3">
          <div className="flex items-center gap-3">
            <h2 className="font-semibold">Your study document</h2>
            <label className="ml-auto flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={highlight}
                onChange={(e) => setHighlight(e.target.checked)}
              />
              Show where each passage came from
            </label>
          </div>
          {highlight && (
            <p className="text-xs text-slate-500">
              <span className="rounded bg-amber-50 px-1.5 py-0.5 text-amber-800">textbook</span>{" "}
              passages carry a page reference.{" "}
              <span className="rounded bg-red-50 px-1.5 py-0.5 text-red-700">Check this</span> marks
              where your notes and the slides disagree.
            </p>
          )}
          <article
            className={`prose-doc rounded-lg border bg-white p-6 ${highlight ? "" : "plain"}`}
            dangerouslySetInnerHTML={{ __html: document }}
          />
          {report && (
            <details className="rounded-lg border bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold">
                What the textbook gate rejected
              </summary>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs">{report}</pre>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
