import { useState } from "react";
import RunView from "./RunView";
import Setup from "./Setup";

/**
 * One page. There are no tabs because you never choose a screen: the run's own
 * state decides what you should be looking at.
 */
export default function App() {
  const [course, setCourse] = useState("");
  const [runId, setRunId] = useState("");

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-6 py-3">
          <span className="font-semibold">Study Note Synthesizer</span>
          {runId && (
            <button
              onClick={() => setRunId("")}
              className="ml-auto rounded border px-2.5 py-1 text-sm hover:bg-slate-50"
            >
              New chapter
            </button>
          )}
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-8 px-6 py-8 lg:grid-cols-[20rem_1fr]">
        <aside className="rounded-lg border bg-white p-5">
          <Setup
            course={course}
            onCourse={setCourse}
            onStart={setRunId}
            compact={Boolean(runId)}
          />
        </aside>
        <section>
          {runId ? (
            <RunView runId={runId} />
          ) : (
            <div className="rounded-lg border border-dashed bg-white p-12 text-center text-sm text-slate-500">
              Pick a course and a chapter, point at your slides and notes, and press
              <span className="font-medium text-slate-700"> Build this chapter</span>.
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
