"use client";

import { TOP_K_OPTIONS } from "@/lib/config";
import { useSearch } from "@/features/search/SearchContext";

export function TopKSelect() {
  const { topK, setTopK } = useSearch();

  return (
    <label className="flex shrink-0 items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
      <span className="whitespace-nowrap font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Top n</span>
      <select
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
        className="rounded-xl border border-slate-200/90 bg-white/85 px-3 py-2 text-sm font-semibold text-slate-800 shadow-[0_8px_20px_rgba(15,23,42,0.08)] outline-none ring-sky-300/25 focus:border-sky-400 focus:ring-2 dark:border-slate-700 dark:bg-slate-900/90 dark:text-slate-100 dark:shadow-[0_12px_28px_rgba(2,6,23,0.35)] dark:[color-scheme:dark]"
      >
        {TOP_K_OPTIONS.map((k) => (
          <option key={k} value={k}>
            Top {k}
          </option>
        ))}
      </select>
    </label>
  );
}