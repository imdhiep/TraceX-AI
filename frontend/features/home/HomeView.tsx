"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import { useToast } from "@/components/ui/ToastProvider";
import { VideoGrid } from "@/components/video/VideoGrid";
import { VideoList } from "@/components/video/VideoList";
import { useSearch } from "@/features/search/SearchContext";
import { getSearchHistory, getVideoDetail, triggerIngest, type SearchHistoryItem } from "@/lib/api";
import { loadSessionUser, type AuthUser } from "@/lib/auth";
import { GRID_BATCH_SIZE } from "@/lib/config";
import {
  HOME_GUIDE_FEATURES,
  HOME_GUIDE_LINES,
  HOME_GUIDE_SUPPORTED_CRITERIA,
  HOME_GUIDE_TITLE,
  HOME_GUIDE_UNSUPPORTED_EXAMPLES,
} from "@/lib/content/homeGuide";
import type { VideoClip } from "@/lib/types";

const PAGE_SIZE = GRID_BATCH_SIZE;

const guideCards = [
  {
    tone: "blue" as const,
    title: "Nhập mô tả hoặc tải ảnh",
    body: HOME_GUIDE_LINES[0],
    icon: (
      <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 16V4" />
        <path d="m7 9 5-5 5 5" />
        <path d="M20 16v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3" />
      </svg>
    ),
  },
  {
    tone: "teal" as const,
    title: "Tìm kiếm thông minh",
    body: HOME_GUIDE_LINES[1],
    icon: (
      <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="6" />
        <path d="m16 16 4 4" />
      </svg>
    ),
  },
  {
    tone: "amber" as const,
    title: "Xem kết quả chi tiết",
    body: "Hệ thống hỗ trợ tìm kiếm dựa trên đặc điểm ngoại hình, hành vi và nhiều tiêu chí khác.",
    icon: (
      <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
];

function iconClass(tone: "blue" | "teal" | "amber") {
  if (tone === "teal") return "bg-teal-500 text-white shadow-[0_14px_26px_rgba(20,184,166,0.28)]";
  if (tone === "amber") return "bg-amber-500 text-white shadow-[0_14px_26px_rgba(245,158,11,0.28)]";
  return "bg-blue-600 text-white shadow-[0_14px_26px_rgba(37,99,235,0.28)]";
}

function HomeViewInner() {
  const { showToast } = useToast();
  const searchParams = useSearchParams();
  const view = searchParams.get("view");
  const { hasSearched, isLoading, error, hasMore, results, gridPage, setGridPage, topK, loadMore, setQuery } = useSearch();
  const lastErrorRef = useRef<string | null>(null);

  // History state
  const [historyItems, setHistoryItems] = useState<SearchHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedHistoryClips, setSelectedHistoryClips] = useState<VideoClip[]>([]);

  // Admin import state
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const isAdmin = currentUser?.role === "SUPER_ADMIN" || currentUser?.role === "ADMIN";

  useEffect(() => {
    setCurrentUser(loadSessionUser());
  }, []);

  // Results pagination
  const pageItems = useMemo(
    () => results.slice(gridPage * PAGE_SIZE, gridPage * PAGE_SIZE + PAGE_SIZE),
    [results, gridPage],
  );
  const totalPages = Math.ceil(results.length / PAGE_SIZE) || 1;
  const isLastLoadedPage = gridPage >= totalPages - 1 || pageItems.length === 0;
  const isLastPage = isLastLoadedPage && !hasMore;
  const startRank = gridPage * PAGE_SIZE + 1;
  const endRank = Math.min((gridPage + 1) * PAGE_SIZE, results.length);

  useEffect(() => {
    if (error && error !== lastErrorRef.current) {
      showToast(error, "error");
      lastErrorRef.current = error;
    }
  }, [error, showToast]);

  // Load history when view=history
  useEffect(() => {
    if (view !== "history") return;
    let cancelled = false;
    setHistoryLoading(true);
    void getSearchHistory()
      .then((items) => {
        if (cancelled) return;
        setHistoryItems(items);
      })
      .catch((err) => {
        if (cancelled) return;
        showToast(err instanceof Error ? err.message : "Không thể tải lịch sử.", "error");
      })
      .finally(() => {
        if (cancelled) return;
        setHistoryLoading(false);
      });
    return () => { cancelled = true; };
  }, [showToast, view]);

  const handleImport = useCallback(async () => {
    if (isImporting) return;
    setIsImporting(true);
    setImportMessage("Đang di chuyển videos từ Temp → Storage...");
    try {
      const result = await triggerIngest({ action: "move_and_ingest" });
      setImportMessage(result.message);
      showToast(result.message, "success");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Import thất bại.";
      setImportMessage(msg);
      showToast(msg, "error");
    } finally {
      setIsImporting(false);
      setTimeout(() => setImportMessage(null), 6000);
    }
  }, [isImporting, showToast]);

  // ── History view ────────────────────────────────────────────────────────────
  if (view === "history") {
    return (
      <div className="mx-auto w-full max-w-4xl text-ink dark:text-slate-100">
        <div className="mb-8 text-center">
          <p className="text-xs font-black uppercase tracking-[0.16em] text-blue-700 dark:text-blue-400">Search Intelligence</p>
          <h1 className="mt-3 text-3xl font-black tracking-normal text-slate-950 dark:text-white md:text-5xl">Lịch sử truy vấn</h1>
          <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300">
            Chỉ hiển thị 20 truy vấn gần nhất. Bấm vào một dòng để lặp lại query.
          </p>
        </div>
        {historyLoading ? <p className="mb-4 text-sm text-slate-500 dark:text-slate-400">Đang tải lịch sử...</p> : null}
        <ul className="space-y-3">
          {historyItems.map((row) => (
            <li
              key={row.queryId}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-[0_10px_24px_rgba(15,23,42,0.05)] transition hover:border-blue-200 hover:shadow-[0_16px_30px_rgba(37,99,235,0.1)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)] dark:hover:border-blue-500/60 dark:hover:shadow-[0_24px_44px_rgba(30,64,175,0.24)]"
            >
              <button
                type="button"
                className="w-full text-left"
                onClick={() => {
                  setQuery(row.queryText);
                  if (!row.videoId) {
                    setSelectedHistoryClips([]);
                    return;
                  }
                  void getVideoDetail(row.videoId)
                    .then((payload) => setSelectedHistoryClips(payload.segments))
                    .catch((err) => showToast(err instanceof Error ? err.message : "Không thể tải video.", "error"));
                }}
              >
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  {new Date(row.updatedAt).toLocaleString("vi-VN")}
                </span>
                <p className="mt-1 text-sm font-bold text-slate-900 dark:text-slate-100 md:text-base">{row.queryText}</p>
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  {row.videoId ? `Video: ${row.videoId}` : "Chưa có video được chọn"}
                </p>
              </button>
            </li>
          ))}
        </ul>
        {selectedHistoryClips.length > 0 ? (
          <div className="mt-8">
            <VideoList clips={selectedHistoryClips} />
          </div>
        ) : null}
      </div>
    );
  }

  // ── Pre-search guide ────────────────────────────────────────────────────────
  if (!hasSearched) {
    return (
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center pb-24 text-ink dark:text-slate-100 md:pb-0">

        {/* Admin import panel */}
        {isAdmin ? (
          <div className="mb-6 flex w-full items-center justify-between gap-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-5 py-3 dark:border-emerald-900/70 dark:bg-emerald-950/45">
            <div>
              <p className="text-sm font-semibold text-emerald-800 dark:text-emerald-300">Import video mới</p>
              <p className="text-xs text-emerald-600 dark:text-emerald-400">Di chuyển từ Drive → Storage và chạy pipeline</p>
            </div>
            <button
              type="button"
              onClick={() => void handleImport()}
              disabled={isImporting}
              className={[
                "shrink-0 rounded-xl border px-4 py-2 text-sm font-medium transition",
                isImporting
                  ? "cursor-not-allowed border-emerald-200 bg-emerald-100 text-emerald-400 dark:border-emerald-900/70 dark:bg-emerald-950/50 dark:text-emerald-700"
                  : "border-emerald-300 bg-white text-emerald-700 hover:bg-emerald-100 dark:border-emerald-800 dark:bg-slate-900 dark:text-emerald-300 dark:hover:bg-emerald-950/60",
              ].join(" ")}
            >
              {isImporting ? "Đang xử lý..." : "Import Videos"}
            </button>
          </div>
        ) : null}
        {importMessage ? (
          <div className={[
            "mb-4 w-full rounded-xl border px-4 py-3 text-sm",
            importMessage.toLowerCase().includes("thất bại") || importMessage.toLowerCase().includes("error")
              ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/70 dark:bg-red-950/45 dark:text-red-300"
              : "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/70 dark:bg-emerald-950/45 dark:text-emerald-300",
          ].join(" ")}>
            {importMessage}
          </div>
        ) : null}

        <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-4 py-2 text-xs font-black uppercase tracking-wide text-blue-700 shadow-[0_8px_18px_rgba(37,99,235,0.12)] dark:border-blue-900/70 dark:bg-blue-950/45 dark:text-blue-300 dark:shadow-[0_14px_28px_rgba(30,64,175,0.22)]">
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2v5" /><path d="M12 17v5" />
            <path d="M2 12h5" /><path d="M17 12h5" />
            <path d="m4.9 4.9 3.5 3.5" /><path d="m15.6 15.6 3.5 3.5" />
            <path d="m19.1 4.9-3.5 3.5" /><path d="m8.4 15.6-3.5 3.5" />
          </svg>
          AI-powered intelligence
        </div>

        <section className="mt-8 w-full text-center">
          <h1 className="text-4xl font-black tracking-normal text-slate-950 dark:text-white md:text-6xl">{HOME_GUIDE_TITLE}</h1>
          <p className="mx-auto mt-7 max-w-3xl text-lg leading-8 text-slate-600 dark:text-slate-300">
            Tập trung mô tả ngắn gọn và chính xác để hệ thống phân tích nhanh hơn và trả về kết quả phù hợp.
          </p>
        </section>

        <section className="mt-16 grid w-full grid-cols-1 gap-6 md:grid-cols-3">
          {guideCards.map((card, index) => (
            <article
              key={card.title}
              className="relative min-h-[236px] overflow-visible rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-[0_12px_28px_rgba(15,23,42,0.05)] transition duration-200 hover:-translate-y-1 hover:border-blue-200 hover:shadow-[0_18px_36px_rgba(15,23,42,0.1)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)] dark:hover:border-blue-500/60 dark:hover:shadow-[0_24px_44px_rgba(2,6,23,0.44)]"
            >
              <span className="pointer-events-none absolute right-5 top-1 text-6xl font-black leading-none text-slate-50 dark:text-slate-800">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className={["relative flex h-14 w-14 items-center justify-center rounded-2xl", iconClass(card.tone)].join(" ")}>
                {card.icon}
              </div>
              <h2 className="relative mt-7 text-xl font-black tracking-normal text-slate-950 dark:text-white">{card.title}</h2>
              <p className="relative mt-4 text-sm leading-7 text-slate-600 dark:text-slate-300">{card.body}</p>
              {index < guideCards.length - 1 ? (
                <span className="absolute right-[-13px] top-1/2 hidden h-7 w-7 -translate-y-1/2 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-400 shadow-sm dark:border-slate-700 dark:bg-slate-900 dark:text-slate-500 md:flex">
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
                  </svg>
                </span>
              ) : null}
            </article>
          ))}
        </section>

        <section className="mt-14 grid w-full grid-cols-1 gap-4 md:grid-cols-3">
          {HOME_GUIDE_FEATURES.map((feature, index) => (
            <article
              key={feature.title}
              className={[
                "flex min-h-[116px] items-center gap-4 rounded-2xl border bg-white p-5 text-left shadow-[0_12px_28px_rgba(15,23,42,0.06)] dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]",
                index === 0
                  ? "border-blue-300 shadow-[0_14px_30px_rgba(37,99,235,0.14)] dark:border-blue-500/60 dark:shadow-[0_18px_36px_rgba(30,64,175,0.24)]"
                  : "border-slate-200 dark:border-slate-800",
              ].join(" ")}
            >
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-50 text-blue-600 dark:bg-blue-950/55 dark:text-blue-300">
                <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2">
                  {index === 0 ? <><path d="M15 10 20 7v10l-5-3" /><rect x="4" y="6" width="11" height="12" rx="2" /></> : null}
                  {index === 1 ? <><path d="M16 19c0-2.2-1.8-4-4-4s-4 1.8-4 4" /><circle cx="12" cy="9" r="3" /></> : null}
                  {index === 2 ? <path d="m13 2-9 13h8l-1 7 9-13h-8l1-7Z" /> : null}
                </svg>
              </div>
              <div>
                <h2 className="text-base font-black tracking-normal text-slate-950 dark:text-white">{feature.title}</h2>
                <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">{feature.body}</p>
              </div>
            </article>
          ))}
        </section>

        <section className="mt-12 w-full rounded-3xl border border-slate-200 bg-white p-6 text-left shadow-[0_14px_34px_rgba(15,23,42,0.08)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_22px_44px_rgba(2,6,23,0.36)] md:p-8">
          <h2 className="text-xl font-black tracking-normal text-slate-950 dark:text-white">Các tiêu chí tìm kiếm được hỗ trợ</h2>
          <div className="mt-7 flex flex-wrap gap-3">
            {HOME_GUIDE_SUPPORTED_CRITERIA.map((criterion) => (
              <span
                key={criterion}
                className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-800 shadow-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              >
                {criterion}
              </span>
            ))}
          </div>
        </section>

        <section className="mt-6 w-full rounded-3xl border border-amber-200 bg-amber-50/70 p-6 text-left shadow-[0_14px_34px_rgba(146,64,14,0.08)] dark:border-amber-900/70 dark:bg-amber-950/35 dark:shadow-[0_22px_44px_rgba(120,53,15,0.22)] md:p-8">
          <h2 className="text-xl font-black tracking-normal text-slate-950 dark:text-white">Ví dụ chưa hỗ trợ</h2>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {HOME_GUIDE_UNSUPPORTED_EXAMPLES.map((example) => (
              <p
                key={example}
                className="rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm font-semibold leading-6 text-slate-700 dark:border-amber-900/70 dark:bg-slate-900/85 dark:text-slate-200"
              >
                {example}
              </p>
            ))}
          </div>
        </section>
      </div>
    );
  }

  // ── Search results ──────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-8 text-ink dark:text-slate-100">
      {isLoading ? (
        <p className="rounded-2xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-semibold text-blue-900 dark:border-blue-900/70 dark:bg-blue-950/45 dark:text-blue-200">
          Đang tải kết quả...
        </p>
      ) : null}
      {!isLoading && !error && results.length === 0 ? (
        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-200">
          Không tìm thấy kết quả phù hợp.
        </p>
      ) : null}
      <VideoGrid items={pageItems} />

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_10px_28px_rgba(15,23,42,0.06)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)]">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-sm font-bold text-slate-700 dark:text-slate-200">
            {isLastPage ? "Hết kết quả" : `Top ${startRank}–${endRank}`}
          </p>
          <button
            type="button"
            disabled={isLastPage || isLoading}
            onClick={async () => {
              if (!isLastLoadedPage) {
                setGridPage(gridPage + 1);
                return;
              }
              if (hasMore) {
                const added = await loadMore();
                if (added) setGridPage(gridPage + 1);
              }
            }}
            className="rounded-xl bg-slate-950 px-5 py-2.5 text-sm font-bold text-white shadow-[0_12px_28px_rgba(15,23,42,0.22)] transition duration-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-blue-600 dark:shadow-[0_16px_32px_rgba(37,99,235,0.28)] dark:hover:bg-blue-500"
          >
            Tiếp theo
          </button>
        </div>
      </div>

      <p className="text-center text-xs font-semibold tracking-wide text-slate-500 dark:text-slate-400">
        Mỗi trang {PAGE_SIZE} kết quả · Top n hiện tại = {topK}
      </p>
    </div>
  );
}

export function HomeView() {
  return (
    <Suspense fallback={<div className="flex flex-1 items-center justify-center py-24 text-sm text-ink-secondary dark:text-slate-400">Đang tải...</div>}>
      <HomeViewInner />
    </Suspense>
  );
}
