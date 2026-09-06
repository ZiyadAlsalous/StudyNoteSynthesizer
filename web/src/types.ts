export type ChapterRange = {
  chapter: string;
  title: string;
  page_start: number;
  page_end: number;
  manual_override: boolean;
};

export type Course = { id: string; title: string };

export type RunRecord = {
  id: string;
  course: string;
  chapter: string;
  status: "pending" | "running" | "done" | "failed";
  created_at: string;
  updated_at: string;
  document_path: string | null;
  error: string | null;
};

export type Progress = {
  run_id: string;
  node: string;
  detail: string;
  done: boolean;
  error: string | null;
};

export type NotePage = {
  page: number;
  image_path: string;
  content_hash: string;
  markdown: string;
  edited_by_student: boolean;
};
