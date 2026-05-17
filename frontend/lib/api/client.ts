import type { VideoClip, VideoItem } from "@/lib/types";
import { loadAccessToken } from "@/lib/auth";
import { parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";

type TrackletSummaryApi = {
  tracklet_id: string;
  camera_id: string | null;
  time_start: string | null;
  time_end: string | null;
};

type SearchApiResponse = {
  results: Array<{
    id: string;
    thumbnail_url: string;
    description: string;
    query_id?: string;
    tracklet_count?: number;
    tracklets?: TrackletSummaryApi[];
  }>;
  query_id?: string;
};

type SearchHistoryApiItem = {
  query_id: string;
  query_text: string;
  query_image_url?: string | null;
  status?: string | null;
  selected_candidate_id?: string | null;
  candidate_count?: number;
  evidence_video_id?: number | null;
  has_evidence?: boolean;
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
  queryImageUrl: string | null;
  status: string | null;
  selectedCandidateId: string | null;
  candidateCount: number;
  evidenceVideoId: number | null;
  hasEvidence: boolean;
  videoId: string | null;
  storagePath: string | null;
  createdAt: string;
  updatedAt: string;
};

type HistoryCandidateApi = {
  id: string;
  thumbnail_url: string;
  description: string;
  query_id: string;
  is_selected: boolean;
  rank_position: number;
  fusion_score: number;
  tracklet_count?: number;
  tracklets?: TrackletSummaryApi[];
};

type HistoryCandidatesApiResponse = {
  results: HistoryCandidateApi[];
  query_id: string;
  selected_candidate_id: string | null;
  selected_candidate_ids?: string[];
  total_count?: number;
  offset?: number;
  limit?: number;
  has_more?: boolean;
};

export type HistoryCandidatesResult = {
  queryId: string;
  selectedCandidateId: string | null;
  selectedCandidateIds: string[];
  totalCount: number;
  offset: number;
  limit: number;
  hasMore: boolean;
  items: VideoItem[];
};

export type HistoryEvidenceMeta = {
  evidenceId: number;
  queryId: string;
  candidateId: string;
  traceConfidence: number;
  segmentCount: number;
  totalDuration: number;
  createdAt: string;
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

export type SearchVideosOptions = {
  persistQuery?: boolean;
  reuseQueryId?: string | null;
  queryImage?: File | null;
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

export function resolveMediaUrl(url: string, apiBaseUrl: string = getApiBaseUrl()): string {
  const raw = (url ?? "").trim();
  if (!raw) {
    return "";
  }
  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }
  if (raw.startsWith("/static/")) {
    const mediaBaseUrl = apiBaseUrl.replace(/\/$/, "").replace(/\/api\/v\d+$/i, "");
    if (!mediaBaseUrl || mediaBaseUrl.startsWith("/")) {
      return raw;
    }
    return `${mediaBaseUrl}${raw}`;
  }
  if (raw.startsWith("/")) {
    return `${apiBaseUrl}${raw}`;
  }
  return `${apiBaseUrl}/${raw}`;
}

async function apiFetchFromBase<T>(apiBaseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const authHeaders: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = loadAccessToken();
    if (token) {
      authHeaders.Authorization = `Bearer ${token}`;
    }
  }

  const isFormDataBody =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  const headers: Record<string, string> = {
    ...authHeaders,
    ...((init?.headers ?? {}) as Record<string, string>),
  };
  if (!isFormDataBody && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response));
  }
  return parseJsonOrThrow<T>(response);
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  return apiFetchFromBase<T>(getApiBaseUrl(), path, init);
}

export type SearchPage = {
  queryId: string | null;
  items: VideoItem[];
};

export async function searchVideos(
  query: string,
  topK: number,
  offset = 0,
  filters?: SearchFilters,
  optionsOrPersistQuery: SearchVideosOptions | boolean = false,
  reuseQueryId?: string | null,
): Promise<SearchPage> {
  // Keep a single browser-facing API entrypoint. metadata-service forwards
  // /search to query-service, which then chooses text_only/image_only/image_text
  // internally without making the frontend depend on query-service exposure.
  const apiBaseUrl = getApiBaseUrl();
  const options: SearchVideosOptions =
    typeof optionsOrPersistQuery === "boolean"
      ? { persistQuery: optionsOrPersistQuery, reuseQueryId }
      : optionsOrPersistQuery;
  const payloadBody: {
    query: string;
    top_k: number;
    offset: number;
    persist_query: boolean;
    camera_ids?: string[];
    time_from?: string;
    time_to?: string;
    query_id?: string;
  } = {
    query,
    top_k: topK,
    offset,
    persist_query: Boolean(options.persistQuery),
  };
  if (filters?.camera_ids?.length) {
    payloadBody.camera_ids = filters.camera_ids;
  }
  if (filters?.time_from) {
    payloadBody.time_from = filters.time_from;
  }
  if (filters?.time_to) {
    payloadBody.time_to = filters.time_to;
  }
  if (options.reuseQueryId) {
    payloadBody.query_id = options.reuseQueryId;
  }
  const requestInit: RequestInit = { method: "POST" };
  if (options.queryImage) {
    const form = new FormData();
    form.append("query", payloadBody.query);
    form.append("top_k", String(payloadBody.top_k));
    form.append("offset", String(payloadBody.offset));
    form.append("persist_query", String(payloadBody.persist_query));
    if (payloadBody.query_id) {
      form.append("query_id", payloadBody.query_id);
    }
    for (const cameraId of payloadBody.camera_ids ?? []) {
      form.append("camera_ids", cameraId);
    }
    if (payloadBody.time_from) {
      form.append("time_from", payloadBody.time_from);
    }
    if (payloadBody.time_to) {
      form.append("time_to", payloadBody.time_to);
    }
    form.append("query_image", options.queryImage, options.queryImage.name);
    requestInit.body = form;
  } else {
    requestInit.body = JSON.stringify(payloadBody);
  }
  const payload = await apiFetchFromBase<SearchApiResponse>(apiBaseUrl, "/search", {
    ...requestInit,
  });

  const queryId = payload.query_id ?? null;
  const items = payload.results.map((item, index) => ({
    id: item.id,
    title: "",
    description: item.description,
    thumbnailUrl: isLikelyImageUrl(item.thumbnail_url)
      ? resolveMediaUrl(item.thumbnail_url, apiBaseUrl)
      : placeholderThumbnail(item.id),
    queryId: item.query_id ?? queryId ?? undefined,
    rank: offset + index + 1,
    trackletCount: item.tracklet_count,
    tracklets: item.tracklets?.map((t) => ({
      trackletId: t.tracklet_id,
      cameraId: t.camera_id,
      timeStart: t.time_start,
      timeEnd: t.time_end,
    })),
  }));
  return { queryId, items };
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

function mapHistoryItem(item: SearchHistoryApiItem): SearchHistoryItem {
  return {
    queryId: item.query_id,
    queryText: item.query_text,
    queryImageUrl: item.query_image_url ? resolveMediaUrl(item.query_image_url) : null,
    status: item.status ?? null,
    selectedCandidateId: item.selected_candidate_id ?? null,
    candidateCount: item.candidate_count ?? 0,
    evidenceVideoId: item.evidence_video_id ?? null,
    hasEvidence: Boolean(item.has_evidence),
    videoId: item.video_id,
    storagePath: item.storage_path,
    createdAt: item.created_at,
    updatedAt: item.updated_at,
  };
}

export async function getSearchHistory(): Promise<SearchHistoryItem[]> {
  const payload = await apiFetch<SearchHistoryApiResponse>("/history");
  return payload.items.map(mapHistoryItem);
}

export async function getHistoryCandidates(
  queryId: string,
  offset = 0,
  limit = 10,
): Promise<HistoryCandidatesResult> {
  const apiBaseUrl = getApiBaseUrl();
  const payload = await apiFetch<HistoryCandidatesApiResponse>(
    `/history/${encodeURIComponent(queryId)}/candidates?offset=${offset}&limit=${limit}`,
  );
  const items: VideoItem[] = payload.results.map((row) => ({
    id: row.id,
    title: "",
    description: row.description,
    thumbnailUrl: isLikelyImageUrl(row.thumbnail_url)
      ? resolveMediaUrl(row.thumbnail_url, apiBaseUrl)
      : "",
    queryId: row.query_id,
    rank: row.rank_position,
    trackletCount: row.tracklet_count,
    tracklets: row.tracklets?.map((t) => ({
      trackletId: t.tracklet_id,
      cameraId: t.camera_id,
      timeStart: t.time_start,
      timeEnd: t.time_end,
    })),
  }));
  return {
    queryId: payload.query_id,
    selectedCandidateId: payload.selected_candidate_id,
    selectedCandidateIds:
      payload.selected_candidate_ids ??
      payload.results.filter((row) => row.is_selected).map((row) => row.id),
    totalCount: payload.total_count ?? items.length,
    offset: payload.offset ?? offset,
    limit: payload.limit ?? limit,
    hasMore: Boolean(payload.has_more),
    items,
  };
}

export async function getHistoryEvidence(queryId: string): Promise<HistoryEvidenceMeta> {
  type EvidenceApi = {
    evidence_id: number;
    query_id: string;
    candidate_id: string;
    trace_confidence: number;
    segment_count: number;
    total_duration: number;
    created_at: string;
  };
  const payload = await apiFetch<EvidenceApi>(
    `/history/${encodeURIComponent(queryId)}/evidence`,
  );
  return {
    evidenceId: payload.evidence_id,
    queryId: payload.query_id,
    candidateId: payload.candidate_id,
    traceConfidence: payload.trace_confidence,
    segmentCount: payload.segment_count,
    totalDuration: payload.total_duration,
    createdAt: payload.created_at,
  };
}

export async function selectHistoryVideo(query: string, selectedIndex: number): Promise<void> {
  await apiFetch<Record<string, unknown>>("/history/select", {
    method: "POST",
    body: JSON.stringify({ query, selectedIndex }),
  });
}

export async function getAdminUserQueries(userId: number): Promise<SearchHistoryItem[]> {
  const payload = await apiFetch<SearchHistoryApiResponse>(`/admin/users/${userId}/queries`);
  return payload.items.map(mapHistoryItem);
}

// ---------------------------------------------------------------------------
// Trace API — candidate detail, select, build
// ---------------------------------------------------------------------------

import type {
  BuildTraceResult,
  CandidateDetail,
  CandidateTracklet,
  TraceSegment,
} from "@/lib/types";

type CandidateTrackletApi = {
  tracklet_id: string;
  video_id: string | null;
  camera_id: string | null;
  track_id: string | null;
  time_start: string | null;
  time_end: string | null;
  start_offset_seconds: number | null;
  end_offset_seconds: number | null;
  duration_seconds: number | null;
  crop_url: string | null;
  representative_bbox: number[] | null;
  quality_score: number | null;
  confidence: number | null;
  gender: string | null;
  gender_conf: number | null;
  age_range: string | null;
  age_range_conf: number | null;
  upper_color: string | null;
  upper_type: string | null;
  upper_desc: string | null;
  upper_conf: number | null;
  lower_color: string | null;
  lower_type: string | null;
  lower_desc: string | null;
  lower_conf: number | null;
  shoes_color: string | null;
  shoes_type: string | null;
  shoes_desc: string | null;
  shoes_conf: number | null;
  bag_presence: string | null;
  bag_type: string | null;
  bag_desc: string | null;
  bag_conf: number | null;
  hat_presence: string | null;
  hat_color: string | null;
  hat_type: string | null;
  hat_desc: string | null;
  hat_conf: number | null;
  mask_presence: string | null;
  mask_conf: number | null;
  hair_style: string | null;
  hair_style_conf: number | null;
  hair_color: string | null;
  hair_color_conf: number | null;
  appearance_summary: string | null;
  appearance_summary_conf: number | null;
  bev_x: number | null;
  bev_y: number | null;
  actions: Array<{
    action_label: string;
    action_label_vi?: string | null;
    kinetics_label: string | null;
    kinetics_label_vi?: string | null;
    confidence: number;
  }>;
  embedding: { has_embedding: boolean; dim: number | null; model: string };
  display?: {
    gender?: string | null;
    age_range?: string | null;
    hair?: string | null;
    upper?: string | null;
    lower?: string | null;
    shoes?: string | null;
    bag?: string | null;
    hat?: string | null;
    mask?: string | null;
  };
};

type CandidateDetailApi = {
  candidate_id: string;
  candidate_key: string | null;
  fusion_score: number | null;
  vector_score: number | null;
  text_score: number | null;
  appearance_summary: string | null;
  preview_url: string | null;
  rank_position: number | null;
  total_tracklets_in_window: number;
  tracklets: CandidateTrackletApi[];
  camera_path: string[];
};

function mapTracklet(t: CandidateTrackletApi): CandidateTracklet {
  return {
    trackletId: t.tracklet_id,
    videoId: t.video_id,
    cameraId: t.camera_id,
    trackId: t.track_id,
    timeStart: t.time_start,
    timeEnd: t.time_end,
    startOffsetSeconds: t.start_offset_seconds,
    endOffsetSeconds: t.end_offset_seconds,
    durationSeconds: t.duration_seconds,
    cropUrl: t.crop_url,
    representativeBbox: t.representative_bbox,
    qualityScore: t.quality_score,
    confidence: t.confidence,
    gender: t.gender,
    genderConf: t.gender_conf,
    ageRange: t.age_range,
    ageRangeConf: t.age_range_conf,
    upperColor: t.upper_color,
    upperType: t.upper_type,
    upperDesc: t.upper_desc,
    upperConf: t.upper_conf,
    lowerColor: t.lower_color,
    lowerType: t.lower_type,
    lowerDesc: t.lower_desc,
    lowerConf: t.lower_conf,
    shoesColor: t.shoes_color,
    shoesType: t.shoes_type,
    shoesDesc: t.shoes_desc,
    shoesConf: t.shoes_conf,
    bagPresence: t.bag_presence,
    bagType: t.bag_type,
    bagDesc: t.bag_desc,
    bagConf: t.bag_conf,
    hatPresence: t.hat_presence,
    hatColor: t.hat_color,
    hatType: t.hat_type,
    hatDesc: t.hat_desc,
    hatConf: t.hat_conf,
    maskPresence: t.mask_presence,
    maskConf: t.mask_conf,
    hairStyle: t.hair_style,
    hairStyleConf: t.hair_style_conf,
    hairColor: t.hair_color,
    hairColorConf: t.hair_color_conf,
    appearanceSummary: t.appearance_summary,
    appearanceSummaryConf: t.appearance_summary_conf,
    bevX: t.bev_x,
    bevY: t.bev_y,
    actions: (t.actions ?? []).map((a) => ({
      actionLabel: a.action_label,
      actionLabelVi: a.action_label_vi ?? null,
      kineticsLabel: a.kinetics_label,
      kineticsLabelVi: a.kinetics_label_vi ?? null,
      confidence: a.confidence,
    })),
    embedding: {
      hasEmbedding: t.embedding?.has_embedding ?? false,
      dim: t.embedding?.dim ?? null,
      model: t.embedding?.model ?? "SigLIP2-So400m",
    },
    display: t.display
      ? {
          gender: t.display.gender ?? null,
          ageRange: t.display.age_range ?? null,
          hair: t.display.hair ?? null,
          upper: t.display.upper ?? null,
          lower: t.display.lower ?? null,
          shoes: t.display.shoes ?? null,
          bag: t.display.bag ?? null,
          hat: t.display.hat ?? null,
          mask: t.display.mask ?? null,
        }
      : undefined,
  };
}

export async function getCandidateDetail(
  queryId: string,
  candidateId: string,
): Promise<CandidateDetail> {
  const payload = await apiFetch<CandidateDetailApi>("/trace/candidate-detail", {
    method: "POST",
    body: JSON.stringify({ query_id: queryId, candidate_id: candidateId }),
  });
  return {
    candidateId: payload.candidate_id,
    candidateKey: payload.candidate_key,
    fusionScore: payload.fusion_score,
    vectorScore: payload.vector_score,
    textScore: payload.text_score,
    appearanceSummary: payload.appearance_summary,
    previewUrl: payload.preview_url,
    rankPosition: payload.rank_position,
    totalTrackletsInWindow: payload.total_tracklets_in_window,
    tracklets: payload.tracklets.map(mapTracklet),
    cameraPath: payload.camera_path,
  };
}

type BuildTraceApi = {
  success: boolean;
  query_id: string;
  candidate_id: string;
  candidate_ids?: string[] | null;
  evidence_id: number;
  trace_duration_ms: number | null;
  trace_confidence: number | null;
  segment_count: number;
  total_duration_seconds: number | null;
  segments: Array<{
    segment_order: number;
    tracklet_id: string;
    camera_id: string | null;
    time_start: string | null;
    time_end: string | null;
    duration_seconds: number | null;
    thumbnail_url: string | null;
    video_clip_url: string | null;
    confidence: number | null;
  }>;
  merged_video_url: string | null;
  time_window_start: string | null;
  time_window_end: string | null;
};

function mapSegment(s: BuildTraceApi["segments"][number]): TraceSegment {
  return {
    segmentOrder: s.segment_order,
    trackletId: s.tracklet_id,
    cameraId: s.camera_id,
    timeStart: s.time_start,
    timeEnd: s.time_end,
    durationSeconds: s.duration_seconds,
    thumbnailUrl: s.thumbnail_url,
    videoClipUrl: s.video_clip_url,
    confidence: s.confidence,
  };
}

export async function selectCandidate(queryId: string, candidateId: string): Promise<void> {
  await apiFetch<Record<string, unknown>>("/trace/select", {
    method: "POST",
    body: JSON.stringify({ query_id: queryId, candidate_id: candidateId }),
  });
}

export type RemoveCandidateTrackletResult = {
  candidateId: string;
  trackletId: string;
  remainingTrackletCount: number;
  newPreviewUrl: string | null;
  newRepresentativeTrackletId: string | null;
};

export async function removeCandidateTracklet(
  queryId: string,
  candidateId: string,
  trackletId: string,
): Promise<RemoveCandidateTrackletResult> {
  const payload = await apiFetch<{
    candidate_id: string;
    tracklet_id: string;
    remaining_tracklet_count: number;
    new_preview_url?: string | null;
    new_representative_tracklet_id?: string | null;
  }>("/trace/candidate-tracklet/remove", {
    method: "POST",
    body: JSON.stringify({
      query_id: queryId,
      candidate_id: candidateId,
      tracklet_id: trackletId,
    }),
  });
  return {
    candidateId: payload.candidate_id,
    trackletId: payload.tracklet_id,
    remainingTrackletCount: payload.remaining_tracklet_count,
    newPreviewUrl: payload.new_preview_url ?? null,
    newRepresentativeTrackletId: payload.new_representative_tracklet_id ?? null,
  };
}

export async function buildTrace(
  queryId: string,
  candidateIds: string | string[],
  options?: { timeWindowHours?: number; mergeVideos?: boolean; excludedTrackletIds?: string[] },
): Promise<BuildTraceResult> {
  const ids = (Array.isArray(candidateIds) ? candidateIds : [candidateIds])
    .map((id) => id.trim())
    .filter(Boolean);
  if (!ids.length) {
    throw new Error("Chưa chọn candidate để truy vết.");
  }
  const payload = await apiFetch<BuildTraceApi>("/trace/build", {
    method: "POST",
    body: JSON.stringify({
      query_id: queryId,
      candidate_id: ids[0],
      candidate_ids: ids,
      excluded_tracklet_ids: options?.excludedTrackletIds ?? [],
      time_window_hours: options?.timeWindowHours ?? 24,
      merge_videos: options?.mergeVideos ?? false,
    }),
  });
  return {
    evidenceId: payload.evidence_id,
    queryId: payload.query_id,
    candidateId: payload.candidate_id,
    candidateIds: payload.candidate_ids ?? undefined,
    traceConfidence: payload.trace_confidence,
    segmentCount: payload.segment_count,
    totalDurationSeconds: payload.total_duration_seconds,
    segments: (payload.segments ?? []).map(mapSegment),
    mergedVideoUrl: payload.merged_video_url,
    timeWindowStart: payload.time_window_start,
    timeWindowEnd: payload.time_window_end,
  };
}

export async function getTraceTimeline(evidenceId: number): Promise<BuildTraceResult> {
  type TimelineApi = {
    segments: BuildTraceApi["segments"];
    camera_path: string[];
    time_window_start: string | null;
    time_window_end: string | null;
  };
  const payload = await apiFetch<TimelineApi>(`/trace/timeline/${evidenceId}`);
  return {
    evidenceId,
    queryId: "",
    candidateId: "",
    traceConfidence: null,
    segmentCount: payload.segments.length,
    totalDurationSeconds: null,
    segments: payload.segments.map(mapSegment),
    mergedVideoUrl: null,
    timeWindowStart: payload.time_window_start,
    timeWindowEnd: payload.time_window_end,
  };
}
