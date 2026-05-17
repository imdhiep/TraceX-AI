"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { GRID_BATCH_SIZE } from "@/lib/config";
import type { VideoItem } from "@/lib/types";
import { mapLocationIdsToCameraIds } from "@/lib/config";
import { resolveMediaUrl, searchVideos, selectHistoryVideo } from "@/lib/api";

export type SearchImageState = {
  file: File;
  name: string;
  size: number;
  type: string;
  previewUrl: string;
};

type SearchFiltersState = {
  locationIds: string[];
  timeFrom: string;
  timeTo: string;
};

type SearchContextValue = {
  query: string;
  setQuery: (value: string) => void;
  image: SearchImageState | null;
  setImage: (value: SearchImageState | null) => void;
  topK: number;
  setTopK: (value: number) => void;
  hasSearched: boolean;
  isLoading: boolean;
  error: string | null;
  hasMore: boolean;
  results: VideoItem[];
  gridPage: number;
  setGridPage: (page: number) => void;
  filters: SearchFiltersState;
  setLocationIds: (ids: string[]) => void;
  setTimeFrom: (value: string) => void;
  setTimeTo: (value: string) => void;
  clearFilters: () => void;
  filterError: string | null;
  runSearch: () => Promise<void>;
  loadMore: () => Promise<boolean>;
  resetSearch: () => void;
  updateCandidateTrackletRemoval: (
    candidateId: string,
    trackletId: string,
    remainingTrackletCount: number,
    /** Backend-provided cache-busted URL for the new representative crop.
     *  Passed through when the removed tracklet was the representative; nullish
     *  otherwise. See _preview_url_for() in query-service candidates.py. */
    newPreviewUrl?: string | null,
  ) => void;
};

const SearchContext = createContext<SearchContextValue | null>(null);

function toDatasetClockIso(value: string): string | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const [datePart, timePart = "00:00"] = trimmed.split("T");
  const [hour = "00", minute = "00", second = "00"] = timePart.split(":");
  return `${datePart}T${hour.padStart(2, "0")}:${minute.padStart(2, "0")}:${second.padStart(2, "0")}.000Z`;
}

export function SearchProvider({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState("");
  const [image, setImageState] = useState<SearchImageState | null>(null);
  const [topK, setTopKState] = useState(50);
  const [hasSearched, setHasSearched] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [results, setResults] = useState<VideoItem[]>([]);
  const [currentQueryId, setCurrentQueryId] = useState<string | null>(null);
  const [gridPage, setGridPage] = useState(0);
  const [locationIds, setLocationIds] = useState<string[]>([]);
  const [timeFrom, setTimeFrom] = useState("");
  const [timeTo, setTimeTo] = useState("");
  const [filterError, setFilterError] = useState<string | null>(null);

  const setTopK = useCallback((value: number) => {
    setTopKState(value);
    if (hasSearched && query.trim()) {
      void selectHistoryVideo(query, value - 1).catch(() => { /* no-op */ });
    }
  }, [hasSearched, query]);

  const setImage = useCallback((value: SearchImageState | null) => {
    setImageState(value);
  }, []);

  const filters = useMemo(
    () => ({ locationIds, timeFrom, timeTo }),
    [locationIds, timeFrom, timeTo],
  );

  const buildSearchFilters = useCallback(() => {
    const cameraIds = mapLocationIdsToCameraIds(locationIds);
    const timeFromIso = toDatasetClockIso(timeFrom);
    const timeToIso = toDatasetClockIso(timeTo);
    return {
      camera_ids: cameraIds.length ? cameraIds : undefined,
      time_from: timeFromIso,
      time_to: timeToIso,
    };
  }, [locationIds, timeFrom, timeTo]);

  const validateTimeRange = useCallback((): boolean => {
    if (!timeFrom || !timeTo) {
      setFilterError(null);
      return true;
    }
    if (new Date(timeFrom).getTime() <= new Date(timeTo).getTime()) {
      setFilterError(null);
      return true;
    }
    setFilterError("Thời gian bắt đầu không được lớn hơn thời gian kết thúc.");
    return false;
  }, [timeFrom, timeTo]);

  const runSearch = useCallback(async () => {
    const trimmed = query.trim();
    const hasImage = image !== null;
    if (!trimmed && !hasImage) {
      setError("Vui lòng nhập mô tả hoặc tải ảnh để tìm kiếm.");
      setResults([]);
      setHasSearched(false);
      return;
    }
    if (!validateTimeRange()) {
      return;
    }

    setError(null);
    setHasSearched(true);
    setIsLoading(true);
    setCurrentQueryId(null);
    setResults([]);
    setGridPage(0);
    setHasMore(false);
    try {
      const requestLimit = Math.min(GRID_BATCH_SIZE, topK);
      const page = await searchVideos(trimmed, requestLimit, 0, buildSearchFilters(), {
        queryImage: image?.file ?? null,
      });
      setResults(page.items);
      setCurrentQueryId(page.queryId);
      setGridPage(0);
      setHasMore(page.items.length >= requestLimit && page.items.length < topK);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể tải kết quả tìm kiếm.";
      setError(message);
      setResults([]);
      setHasMore(false);
    } finally {
      setIsLoading(false);
    }
  }, [query, image, topK, validateTimeRange, buildSearchFilters]);

  const loadMore = useCallback(async (): Promise<boolean> => {
    const trimmed = query.trim();
    if ((!trimmed && !image && !currentQueryId) || isLoading || !hasSearched || !hasMore || !validateTimeRange()) {
      return false;
    }

    const remaining = topK - results.length;
    if (remaining <= 0) {
      setHasMore(false);
      return false;
    }

    setIsLoading(true);
    setError(null);
    try {
      const requestLimit = Math.min(GRID_BATCH_SIZE, remaining);
      const page = await searchVideos(
        trimmed,
        requestLimit,
        results.length,
        buildSearchFilters(),
        {
          reuseQueryId: currentQueryId,
          queryImage: currentQueryId ? null : image?.file ?? null,
        },
      );
      if (!page.items.length) {
        setHasMore(false);
        return false;
      }
      setResults((prev) => [...prev, ...page.items]);
      if (page.queryId && page.queryId !== currentQueryId) {
        setCurrentQueryId(page.queryId);
      }
      setHasMore(page.items.length >= requestLimit && results.length + page.items.length < topK);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể tải thêm kết quả.";
      setError(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [query, image, isLoading, hasSearched, hasMore, topK, results.length, currentQueryId, validateTimeRange, buildSearchFilters]);

  const clearFilters = useCallback(() => {
    setLocationIds([]);
    setTimeFrom("");
    setTimeTo("");
    setFilterError(null);
  }, []);

  const resetSearch = useCallback(() => {
    setQuery("");
    setImageState(null);
    setResults([]);
    setHasSearched(false);
    setError(null);
    setIsLoading(false);
    setHasMore(false);
    setGridPage(0);
    setCurrentQueryId(null);
    clearFilters();
  }, [clearFilters]);

  const updateCandidateTrackletRemoval = useCallback((
    candidateId: string,
    trackletId: string,
    remainingTrackletCount: number,
    newPreviewUrl?: string | null,
  ) => {
    setResults((current) =>
      current.map((item) => {
        if (item.id !== candidateId) return item;
        // Overwrite thumbnail when backend gave us a new representative crop —
        // otherwise the card keeps showing the removed tracklet's image.
        // Must go through resolveMediaUrl so the relative `/candidates/.../preview?v=...`
        // path is rebased onto the API host instead of the frontend origin.
        const next = {
          ...item,
          trackletCount: remainingTrackletCount,
          tracklets: item.tracklets?.filter((t) => t.trackletId !== trackletId),
        };
        if (newPreviewUrl) {
          (next as { thumbnailUrl?: string | null }).thumbnailUrl = resolveMediaUrl(newPreviewUrl);
        }
        return next;
      }),
    );
  }, []);

  const isFirstTopKRender = useRef(true);
  useEffect(() => {
    if (isFirstTopKRender.current) {
      isFirstTopKRender.current = false;
      return;
    }
    if (!hasSearched) {
      return;
    }
    const trimmed = query.trim();
    if (!trimmed && !image) {
      return;
    }
    if (!validateTimeRange()) {
      return;
    }
    setIsLoading(true);
    setError(null);
    // Top-N change starts a fresh search session, so don't pass the previous qid.
    setCurrentQueryId(null);
    void searchVideos(trimmed, Math.min(GRID_BATCH_SIZE, topK), 0, buildSearchFilters(), {
      queryImage: image?.file ?? null,
    })
      .then((page) => {
        setResults(page.items);
        setCurrentQueryId(page.queryId);
        setGridPage(0);
        setHasMore(page.items.length >= Math.min(GRID_BATCH_SIZE, topK) && page.items.length < topK);
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Không thể tải kết quả tìm kiếm.";
        setError(message);
        setResults([]);
        setHasMore(false);
      })
      .finally(() => {
        setIsLoading(false);
      });
    // Only refetch on Top-N change; intentionally exclude hasSearched/query from deps
    // so clicking "Gửi" doesn't double-fire (runSearch already handles that path).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topK]);


  const value = useMemo(
    () => ({
      query,
      setQuery,
      image,
      setImage,
      topK,
      setTopK,
      hasSearched,
      isLoading,
      error,
      hasMore,
      results,
      gridPage,
      setGridPage,
      filters,
      setLocationIds,
      setTimeFrom,
      setTimeTo,
      clearFilters,
      filterError,
      runSearch,
      loadMore,
      resetSearch,
      updateCandidateTrackletRemoval,
    }),
    [
      query,
      image,
      setImage,
      topK,
      hasSearched,
      isLoading,
      error,
      hasMore,
      results,
      gridPage,
      filters,
      clearFilters,
      filterError,
      runSearch,
      loadMore,
      resetSearch,
      updateCandidateTrackletRemoval,
    ],
  );

  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
}

export function useSearch() {
  const ctx = useContext(SearchContext);
  if (!ctx) {
    throw new Error("useSearch must be used within SearchProvider");
  }
  return ctx;
}
