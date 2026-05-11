import type { VideoClip, VideoItem } from "@/lib/types";
import { loadAccessToken } from "@/lib/auth";
import { parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";

type SearchApiResponse = {
  results: Array<{
    id: string;
    thumbnail_url: string;
    description: string;
  }>;
};

type SearchHistoryApiItem = {
  query_id: string;
  query_text: string;
  video_id: string | null;
  storage_path: string | null;
  created_at: string;
  updated_at: string;
};

type SearchHistoryApiResponse = {
  count: number;
  items: SearchHistoryApiItem[];
};

export type SearchHistoryItem = {
  queryId: string;
  queryText: string;
  videoId: string | null;
  storagePath: string | null;
  createdAt: string;
  updatedAt: string;
};

type VideoDetailApiResponse = {
  id: string;
  segments: Array<{
    id: string;
    video_url: string;
    title: string;
    description: string;
  }>;
};

export type SearchFilters = {
  camera_ids?: string[];
  time_from?: string;
  time_to?: string;
};

function isLikelyImageUrl(url: string): boolean {
  const value = url.toLowerCase();
  if (!value) return false;
  if (value.includes("/candidates/") && value.endsWith("/preview")) {
    return true;
  }
  return /\.(png|jpg|jpeg|webp|gif|avif)(\?.*)?$/.test(value);
}

function placeholderThumbnail(seed: string): string {
  const safe = encodeURIComponent(seed || "mcpt");
  return `https://picsum.photos/seed/${safe}/400/225`;
}

/** Base URL cho fetch từ browser hoặc SSR/server. */
export function getApiBaseUrl(): string {
  const envBase = (process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_GATEWAY_URL ?? "").trim();
  if (envBase.startsWith("/")) {
    return envBase.replace(/\/$/, "");
  }
  if (envBase.startsWith("http://") || envBase.startsWith("https://")) {
    return envBase.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    return "";
  }
  return "http://metadata-service:8000";
}

function resolveMediaUrl(url: string, apiBaseUrl: string): string {
  const raw = (url ?? "").trim();
  if (!raw) {
    return "";
  }
  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }
  if (raw.startsWith("/")) {
    return `${apiBaseUrl}${raw}`;
  }
  return `${apiBaseUrl}/${raw}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBaseUrl = getApiBaseUrl();
  const authHeaders: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = loadAccessToken();
    if (token) {
      authHeaders.Authorization = `Bearer ${token}`;
    }
  }

  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response));
  }
  return parseJsonOrThrow<T>(response);
}

export async function searchVideos(query: string, topK: number, offset = 0, filters?: SearchFilters, persistQuery = false): Promise<VideoItem[]> {
  const apiBaseUrl = getApiBaseUrl();
  const payloadBody: {
    query: string;
    top_k: number;
    offset: number;
    persist_query: boolean;
    camera_ids?: string[];
    time_from?: string;
    time_to?: string;
  } = { query, top_k: topK, offset, persist_query: persistQuery };
  if (filters?.camera_ids?.length) {
    payloadBody.camera_ids = filters.camera_ids;
  }
  if (filters?.time_from) {
    payloadBody.time_from = filters.time_from;
  }
  if (filters?.time_to) {
    payloadBody.time_to = filters.time_to;
  }
  const payload = await apiFetch<SearchApiResponse>("/search", {
    method: "POST",
    body: JSON.stringify(payloadBody),
  });

  return payload.results.map((item) => ({
    id: item.id,
    title: `Candidate ${item.id}`,
    description: item.description,
    thumbnailUrl: isLikelyImageUrl(item.thumbnail_url)
      ? resolveMediaUrl(item.thumbnail_url, apiBaseUrl)
      : placeholderThumbnail(item.id),
  }));
}

export async function getVideoDetail(videoId: string): Promise<{ id: string; segments: VideoClip[] }> {
  const apiBaseUrl = getApiBaseUrl();
  const payload = await apiFetch<VideoDetailApiResponse>(`/videos/${videoId}`);
  return {
    id: payload.id,
    segments: payload.segments.map((segment) => ({
      id: segment.id,
      title: segment.title,
      description: segment.description,
      thumbnailUrl: "https://picsum.photos/seed/mcpt-segment/320/180",
      previewUrl: resolveMediaUrl(segment.video_url, apiBaseUrl),
    })),
  };
}

type ListVideosResponse = {
  count: number;
  items: Array<{
    video_id: string;
    title: string;
    description: string | null;
    storage_path: string;
    storage_backend: string;
    source_filename: string | null;
    content_type: string | null;
    created_at: string;
  }>;
};

export async function listVideos(params: { page?: number; pageSize?: number }): Promise<{ videos: VideoClip[] }> {
  const payload = await apiFetch<ListVideosResponse>("/videos");
  return {
    videos: payload.items.map((item) => ({
      id: item.video_id,
      title: item.title,
      description: item.description ?? "",
      thumbnailUrl: "https://picsum.photos/seed/mcpt-video/400/225",
      previewUrl: resolveMediaUrl(item.storage_path, getApiBaseUrl()),
    })),
  };
}

// ---------------------------------------------------------------------------
// Ingest API
// ---------------------------------------------------------------------------

export type IngestResponse = {
  status: string;
  moved_count: number;
  ingested_count: number;
  skipped_count: number;
  total_videos_in_db: number;
  total_tracklets_saved?: number;
  processed_videos?: number;
  errors: string[];
  message: string;
  finetune?: { scenes_processed: number; errors: string[] };
};

export type IngestStatsResponse = {
  total_videos_in_db: number;
  total_cameras_in_db: number;
  total_tracklets_in_db: number;
  total_embeddings_in_db: number;
};

export async function triggerIngest(
  body: { action: string; dry_run?: boolean } = { action: "move_and_ingest" }
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>("/ingest/move-and-process", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getIngestStats(): Promise<IngestStatsResponse> {
  return apiFetch<IngestStatsResponse>("/ingest/stats");
}

export async function triggerFullPipeline(
  body: { action: string; dry_run?: boolean } = { action: "move_and_ingest" }
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>("/ingest/full-pipeline", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getSearchHistory(): Promise<SearchHistoryItem[]> {
  const payload = await apiFetch<SearchHistoryApiResponse>("/history");
  return payload.items.map((item) => ({
    queryId: item.query_id,
    queryText: item.query_text,
    videoId: item.video_id,
    storagePath: item.storage_path,
    createdAt: item.created_at,
    updatedAt: item.updated_at,
  }));
}

export async function selectHistoryVideo(query: string, selectedIndex: number): Promise<void> {
  await apiFetch<Record<string, unknown>>("/history/select", {
    method: "POST",
    body: JSON.stringify({ query, selectedIndex }),
  });
}

export async function getAdminUserQueries(userId: number): Promise<SearchHistoryItem[]> {
  const payload = await apiFetch<SearchHistoryApiResponse>(`/admin/users/${userId}/queries`);
  return payload.items.map((item) => ({
    queryId: item.query_id,
    queryText: item.query_text,
    videoId: item.video_id,
    storagePath: item.storage_path,
    createdAt: item.created_at,
    updatedAt: item.updated_at,
  }));
}
