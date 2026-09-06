import { useState } from "react";
import Courses from "./screens/Courses";
import ChapterRun from "./screens/ChapterRun";
import Review from "./screens/Review";
import Library from "./screens/Library";
import type { Screen } from "./types";

const TABS: { key: Screen; label: string }[] = [
  { key: "courses", label: "Courses" },
  { key: "run", label: "Chapter run" },
  { key: "review", label: "Review" },
  { key: "library", label: "Library" },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("courses");
  const [course, setCourse] = useState("");
  const [runId, setRunId] = useState("");

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-3">
          <span className="font-semibold">Study Note Synthesizer</span>
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setScreen(tab.key)}
                className={`rounded px-3 py-1.5 text-sm ${
                  screen === tab.key ? "bg-slate-900 text-white" : "hover:bg-slate-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          {course && <span className="ml-auto text-sm text-slate-500">{course}</span>}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {screen === "courses" && <Courses course={course} onCourse={setCourse} />}
        {screen === "run" && (
          <ChapterRun
            course={course}
            onStarted={(id) => {
              setRunId(id);
              setScreen("review");
            }}
          />
        )}
        {screen === "review" && <Review runId={runId} onResumed={() => setScreen("library")} />}
        {screen === "library" && <Library runId={runId} onRunId={setRunId} />}
      </main>
    </div>
  );
}
