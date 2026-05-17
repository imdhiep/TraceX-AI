import Image from "next/image";
import Link from "next/link";

import { cameraIdToLocationLabel, cameraIdsToLocationLabels } from "@/lib/config";
import type { TrackletSummary, VideoItem } from "@/lib/types";

type VideoCardProps = {
  video: VideoItem;
  onClick?: (video: VideoItem) => void;
  onToggleSelect?: (video: VideoItem) => void;
  isSelected?: boolean;
  rank?: number;
};

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function parseTime(iso: string | null): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d;
}

function formatUtcDate(d: Date): string {
  return `${pad2(d.getUTCDate())}/${pad2(d.getUTCMonth() + 1)}/${d.getUTCFullYear()}`;
}

function formatUtcClock(d: Date): string {
  return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`;
}

function sameUtcDate(a: Date, b: Date): boolean {
  return (
    a.getUTCFullYear() === b.getUTCFullYear() &&
    a.getUTCMonth() === b.getUTCMonth() &&
    a.getUTCDate() === b.getUTCDate()
  );
}

function formatUtcDateTime(d: Date): string {
  return `${formatUtcDate(d)} ${formatUtcClock(d)}`;
}

function formatTimeRange(start: string | null, end: string | null): string | null {
  const a = parseTime(start);
  const b = parseTime(end);
  if (a && b) {
    if (sameUtcDate(a, b)) {
      return `${formatUtcDate(a)} ${formatUtcClock(a)}–${formatUtcClock(b)}`;
    }
    return `${formatUtcDateTime(a)} – ${formatUtcDateTime(b)}`;
  }
  return a ? formatUtcDateTime(a) : b ? formatUtcDateTime(b) : null;
}

function uniqueLocations(tracklets: TrackletSummary[]): string[] {
  return cameraIdsToLocationLabels(tracklets.map((t) => t.cameraId));
}

function overallTimeRange(tracklets: TrackletSummary[]): string | null {
  let earliest: number | null = null;
  let latest: number | null = null;
  let earliestIso: string | null = null;
  let latestIso: string | null = null;
  for (const t of tracklets) {
    if (t.timeStart) {
      const ts = new Date(t.timeStart).getTime();
      if (!Number.isNaN(ts) && (earliest === null || ts < earliest)) {
        earliest = ts;
        earliestIso = t.timeStart;
      }
    }
    if (t.timeEnd) {
      const ts = new Date(t.timeEnd).getTime();
      if (!Number.isNaN(ts) && (latest === null || ts > latest)) {
        latest = ts;
        latestIso = t.timeEnd;
      }
    }
  }
  return formatTimeRange(earliestIso, latestIso);
}

export function VideoCard({ video, onClick, onToggleSelect, isSelected = false, rank }: VideoCardProps) {
  const className =
    `group relative flex flex-col overflow-hidden rounded-2xl border bg-white text-left shadow-card transition hover:-translate-y-0.5 hover:shadow-elevated ${
      isSelected ? "border-blue-500 ring-2 ring-blue-200" : "border-surface-muted"
    }`;

  const displayRank = video.rank ?? rank;
  const label = displayRank !== undefined ? `Candidate ${displayRank}` : video.title;

  const tracklets = video.tracklets ?? [];
  const trackletCount = video.trackletCount ?? tracklets.length;
  const locations = uniqueLocations(tracklets);
  const isMulti = trackletCount > 1;

  let locationLine: string | null = null;
  let timeLine: string | null = null;
  let countLine: string | null = null;
  if (tracklets.length > 0) {
    if (isMulti) {
      countLine = `${trackletCount} tracklets`;
      locationLine = locations.length ? `Vị trí: ${locations.join(", ")}` : null;
      timeLine = overallTimeRange(tracklets);
    } else {
      const t = tracklets[0];
      const loc = cameraIdToLocationLabel(t.cameraId);
      const time = formatTimeRange(t.timeStart, t.timeEnd);
      locationLine = loc ? `Vị trí: ${loc}` : null;
      timeLine = time;
    }
  }

  const body = (
    <>
      <div className="relative aspect-video w-full overflow-hidden bg-surface-muted">
        {video.thumbnailUrl ? (
          <Image
            src={video.thumbnailUrl}
            alt={label || "Candidate"}
            fill
            unoptimized
            className="object-cover transition duration-300 group-hover:scale-105"
            sizes="(max-width: 768px) 50vw, 20vw"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs font-medium text-ink-secondary">
            Ảnh đại diện hiện không tồn tại
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-1 p-3">
        <p className="line-clamp-2 text-sm font-semibold text-ink">{label}</p>
        {video.description ? (
          <p className={`${isMulti ? "line-clamp-2" : "line-clamp-3"} text-xs leading-relaxed text-ink-secondary`}>
            {video.description}
          </p>
        ) : null}
        {countLine ? (
          <p className="text-xs font-semibold text-blue-700 dark:text-blue-300">{countLine}</p>
        ) : null}
        {locationLine ? (
          <p className="line-clamp-2 text-xs leading-snug text-ink-secondary dark:text-slate-400">{locationLine}</p>
        ) : null}
        {timeLine ? (
          <p className="text-xs font-medium leading-snug text-ink-secondary dark:text-slate-400">{timeLine}</p>
        ) : null}
      </div>
    </>
  );

  if (onClick || onToggleSelect) {
    return (
      <article className={className}>
        {onToggleSelect ? (
          <button
            type="button"
            aria-pressed={isSelected}
            onClick={() => onToggleSelect(video)}
            className={`absolute right-2 top-2 z-10 rounded-full border px-3 py-1 text-xs font-bold shadow-sm transition ${
              isSelected
                ? "border-blue-600 bg-blue-600 text-white"
                : "border-white/80 bg-white/95 text-slate-700 hover:border-blue-300 hover:text-blue-700"
            }`}
          >
            {isSelected ? "Đã chọn" : "Chọn"}
          </button>
        ) : null}
        {onClick ? (
          <button type="button" className="flex flex-1 flex-col text-left" onClick={() => onClick(video)}>
            {body}
          </button>
        ) : (
          <div className="flex flex-1 flex-col">{body}</div>
        )}
      </article>
    );
  }

  return (
    <Link href={`/detail/${video.id}`} className={className}>
      {body}
    </Link>
  );
}
