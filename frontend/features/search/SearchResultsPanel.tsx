"use client";

import { useState } from "react";

import { VideoCard } from "@/components/video/VideoCard";
import { CandidateDetailModal } from "@/features/candidate/CandidateDetailModal";
import { useSearch } from "@/features/search/SearchContext";
import type { VideoItem } from "@/lib/types";

export function SearchResultsPanel() {
  const { results, isLoading, error, hasSearched, hasMore, loadMore, query } = useSearch();
  const [selected, setSelected] = useState<{ queryId: string; candidateId: string } | null>(null);

  const handleCardClick = (item: VideoItem) => {
    if (!item.queryId) return;
    setSelected({ queryId: item.queryId, candidateId: item.id });
  };

  if (!hasSearched) return null;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">
          {isLoading
            ? "Đang tìm kiếm..."
            : error
              ? null
              : `${results.length} kết quả cho "${query}"`}
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      ) : isLoading && results.length === 0 ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
          {Array.from({ length: 10 }).map((_, i) => (
            <div
              key={i}
              className="aspect-video animate-pulse rounded-2xl bg-slate-200"
            />
          ))}
        </div>
      ) : results.length === 0 ? (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <p className="text-base font-semibold text-slate-700">Không tìm thấy kết quả</p>
          <p className="text-sm text-slate-400">Thử thay đổi từ khóa hoặc bộ lọc và tìm lại.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {results.map((item) => (
              <VideoCard key={item.id} video={item} onClick={handleCardClick} />
            ))}
          </div>

          {hasMore && (
            <div className="flex justify-center pt-2">
              <button
                type="button"
                onClick={() => void loadMore()}
                disabled={isLoading}
                className="rounded-xl border border-slate-200 bg-white px-6 py-2.5 text-sm font-medium text-slate-700 shadow-card transition hover:border-sky-300 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLoading ? "Đang tải..." : "Tải thêm"}
              </button>
            </div>
          )}
        </>
      )}

      <CandidateDetailModal
        open={selected !== null}
        queryId={selected?.queryId ?? null}
        candidateId={selected?.candidateId ?? null}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
