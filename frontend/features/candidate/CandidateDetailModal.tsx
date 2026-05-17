"use client";

import { useEffect, useMemo, useState } from "react";

import { getCandidateDetail, removeCandidateTracklet, resolveMediaUrl } from "@/lib/api";
import type { CandidateDetail, CandidateTracklet } from "@/lib/types";

type Props = {
  open: boolean;
  queryId: string | null;
  candidateId: string | null;
  candidateLabel?: string | null;
  onClose: () => void;
  onTrackletRemoved?: (
    candidateId: string,
    trackletId: string,
    remainingTrackletCount: number,
    /** When the removed tracklet was the candidate's representative, the
     *  backend picks a new one and returns this cache-busted URL. Card
     *  thumbnails should swap to this immediately — otherwise the stale
     *  crop of the deleted tracklet keeps showing. */
    newPreviewUrl?: string | null,
  ) => void;
};

export function CandidateDetailModal({
  open,
  queryId,
  candidateId,
  candidateLabel,
  onClose,
  onTrackletRemoved,
}: Props) {
  const [detail, setDetail] = useState<CandidateDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [removingTrackletId, setRemovingTrackletId] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !queryId || !candidateId) {
      setDetail(null);
      setError(null);
      setRemovingTrackletId(null);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setDetail(null);
    void getCandidateDetail(queryId, candidateId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Không thể tải thông tin candidate.");
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, queryId, candidateId]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const handleRemoveTracklet = async (trackletId: string) => {
    if (!queryId || !candidateId || removingTrackletId) return;
    if (detail && detail.tracklets.length <= 1) {
      setError("Candidate phải giữ lại ít nhất 1 tracklet.");
      return;
    }
    setRemovingTrackletId(trackletId);
    setError(null);
    try {
      const result = await removeCandidateTracklet(queryId, candidateId, trackletId);
      setDetail((current) => {
        if (!current) return current;
        const tracklets = current.tracklets.filter((t) => t.trackletId !== trackletId);
        const cameraPath = Array.from(
          new Set(tracklets.map((t) => t.cameraId).filter((id): id is string => Boolean(id))),
        );
        return {
          ...current,
          tracklets,
          cameraPath,
          totalTrackletsInWindow: result.remainingTrackletCount,
        };
      });
      onTrackletRemoved?.(
        candidateId,
        trackletId,
        result.remainingTrackletCount,
        result.newPreviewUrl,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể loại tracklet khỏi candidate.");
    } finally {
      setRemovingTrackletId(null);
    }
  };

  if (!open) return null;
  const title = candidateLabel ?? (detail?.rankPosition ? `#${detail.rankPosition}` : null) ?? "—";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative flex h-[90vh] w-[min(1100px,96vw)] flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Candidate</p>
            <h2 className="text-lg font-semibold text-slate-900">
              {title}
            </h2>
            {detail?.appearanceSummary ? (
              <p className="mt-1 max-w-2xl text-sm text-slate-600">{detail.appearanceSummary}</p>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
            aria-label="Đóng"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-4">
          {isLoading ? (
            <div className="flex items-center gap-3 py-8 text-sm text-slate-500">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700" />
              Đang tải thông tin candidate…
            </div>
          ) : null}

          {error ? (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          {!isLoading && detail ? (
            <CandidateBody
              detail={detail}
              removingTrackletId={removingTrackletId}
              onRemoveTracklet={handleRemoveTracklet}
            />
          ) : null}
        </div>

        <footer className="flex items-center justify-between gap-4 border-t border-slate-200 bg-slate-50 px-6 py-4">
          <div className="text-xs text-slate-500">
            {detail
              ? `${detail.totalTrackletsInWindow} tracklets · path: ${detail.cameraPath.join(" → ") || "—"}`
              : null}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
            >
              Đóng
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function CandidateBody({
  detail,
  removingTrackletId,
  onRemoveTracklet,
}: {
  detail: CandidateDetail;
  removingTrackletId: string | null;
  onRemoveTracklet: (trackletId: string) => void;
}) {
  const canRemoveTracklets = detail.tracklets.length > 1;

  return (
    <div className="flex flex-col gap-4">
      <ScoreStrip detail={detail} />
      <ol className="flex flex-col gap-4">
        {detail.tracklets.map((t, idx) => (
          <li key={t.trackletId}>
            <TrackletRow
              tracklet={t}
              index={idx}
              isRemoving={removingTrackletId === t.trackletId}
              canRemove={canRemoveTracklets}
              onRemove={() => onRemoveTracklet(t.trackletId)}
            />
          </li>
        ))}
        {detail.tracklets.length === 0 ? (
          <p className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500">
            Candidate này không còn tracklet nào.
          </p>
        ) : null}
      </ol>
    </div>
  );
}

function ScoreStrip({ detail }: { detail: CandidateDetail }) {
  const items: Array<[string, string | number | null]> = [
    ["Rank", detail.rankPosition],
    ["Fusion", fmtScore(detail.fusionScore)],
    ["Vector", fmtScore(detail.vectorScore)],
    ["Text", fmtScore(detail.textScore)],
  ];
  return (
    <div className="grid grid-cols-2 gap-2 rounded-xl bg-slate-50 px-4 py-3 sm:grid-cols-4">
      {items.map(([k, v]) => (
        <div key={k}>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">{k}</p>
          <p className="text-sm font-mono text-slate-800">{v ?? "—"}</p>
        </div>
      ))}
    </div>
  );
}

function TrackletRow({
  tracklet,
  index,
  isRemoving,
  canRemove,
  onRemove,
}: {
  tracklet: CandidateTracklet;
  index: number;
  isRemoving: boolean;
  canRemove: boolean;
  onRemove: () => void;
}) {
  const cropSrc = useMemo(() => {
    if (!tracklet.cropUrl) return null;
    return resolveMediaUrl(tracklet.cropUrl);
  }, [tracklet.cropUrl]);

  const timeLabel = fmtTimeRange(tracklet.timeStart, tracklet.timeEnd, tracklet.durationSeconds);
  const topActions = tracklet.actions.slice(0, 3);
  const bagValue =
    tracklet.display?.bag ??
    (tracklet.bagPresence === "yes"
      ? joinNonNull([tracklet.bagType, tracklet.bagDesc]) || "yes"
      : tracklet.bagPresence);
  const hatValue =
    tracklet.display?.hat ??
    (tracklet.hatPresence === "yes"
      ? joinNonNull([tracklet.hatColor, tracklet.hatType]) || "yes"
      : tracklet.hatPresence);

  return (
    <article className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:flex-row">
      <div className="flex shrink-0 flex-col items-center gap-2">
        <span className="rounded-full bg-slate-900/90 px-2 py-0.5 text-[10px] font-semibold text-white">
          #{index + 1}
        </span>
        <div className="relative h-40 w-32 overflow-hidden rounded-xl bg-slate-100">
          {cropSrc ? (
            // Use plain <img> — these are arbitrary backend URLs not in next/image config.
            // eslint-disable-next-line @next/next/no-img-element
            <img src={cropSrc} alt={tracklet.trackletId} className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-xs text-slate-400">no crop</div>
          )}
        </div>
        <div className="text-center text-[10px] text-slate-500">
          <p className="font-mono">{tracklet.cameraId ?? "—"}</p>
          {timeLabel ? <p>{timeLabel}</p> : null}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-3">
        <header className="flex flex-wrap items-start justify-between gap-2">
          <p className="font-mono text-xs text-slate-500" title={tracklet.trackletId}>
            {tracklet.trackletId}
          </p>
          <div className="flex flex-wrap items-center justify-end gap-2 text-[10px] text-slate-500">
            <div className="flex gap-2">
              <Badge>quality {fmtScore(tracklet.qualityScore)}</Badge>
              {tracklet.embedding.hasEmbedding ? (
                <Badge>{tracklet.embedding.model} · {tracklet.embedding.dim ?? "?"}d</Badge>
              ) : (
                <Badge>no embedding</Badge>
              )}
            </div>
            {canRemove ? (
              <button
                type="button"
                onClick={onRemove}
                disabled={isRemoving}
                title="Loại tracklet khỏi candidate"
                className="rounded-full border border-red-200 bg-red-50 px-3 py-1 text-[11px] font-semibold text-red-700 transition hover:border-red-300 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isRemoving ? "Đang loại..." : "Loại khỏi candidate"}
              </button>
            ) : null}
          </div>
        </header>

        {tracklet.appearanceSummary ? (
          <p className="text-sm text-slate-700">{tracklet.appearanceSummary}</p>
        ) : null}

        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-600 sm:grid-cols-3">
          <Attr label="Giới tính" value={tracklet.display?.gender ?? tracklet.gender} conf={tracklet.genderConf} />
          <Attr label="Tuổi" value={tracklet.display?.ageRange ?? tracklet.ageRange} conf={tracklet.ageRangeConf} />
          <Attr label="Tóc" value={tracklet.display?.hair ?? joinNonNull([tracklet.hairStyle, tracklet.hairColor])} conf={tracklet.hairStyleConf} />
          <Attr label="Áo" value={tracklet.display?.upper ?? joinNonNull([tracklet.upperColor, tracklet.upperType])} conf={tracklet.upperConf} />
          <Attr label="Quần/váy" value={tracklet.display?.lower ?? joinNonNull([tracklet.lowerColor, tracklet.lowerType])} conf={tracklet.lowerConf} />
          <Attr label="Giày" value={tracklet.display?.shoes ?? joinNonNull([tracklet.shoesColor, tracklet.shoesType])} conf={tracklet.shoesConf} />
          <Attr label="Túi" value={bagValue} conf={tracklet.bagConf} />
          <Attr label="Mũ" value={hatValue} conf={tracklet.hatConf} />
          <Attr label="Khẩu trang" value={tracklet.display?.mask ?? tracklet.maskPresence} conf={tracklet.maskConf} />
          <Attr
            label="BEV"
            value={
              tracklet.bevX !== null && tracklet.bevY !== null
                ? `${tracklet.bevX.toFixed(2)}, ${tracklet.bevY.toFixed(2)}`
                : null
            }
            conf={null}
          />
        </div>

        {topActions.length ? (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Hành động</span>
            {topActions.map((a) => (
              <span
                key={`${tracklet.trackletId}-${a.actionLabel}`}
                className="rounded-full bg-sky-50 px-2 py-0.5 text-[11px] text-sky-700"
                title={a.kineticsLabelVi ?? a.kineticsLabel ?? ""}
              >
                {a.actionLabelVi ?? a.actionLabel} · {(a.confidence * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-slate-100 px-2 py-0.5 font-mono text-[10px] text-slate-600">
      {children}
    </span>
  );
}

function Attr({ label, value, conf }: { label: string; value: string | null | undefined; conf: number | null | undefined }) {
  if (!value || value === "unknown") return null;
  return (
    <div>
      <span className="font-semibold text-slate-500">{label}: </span>
      <span className="text-slate-700">{value}</span>
      {typeof conf === "number" ? (
        <span className="ml-1 text-[10px] text-slate-400">({Math.round(conf * 100)}%)</span>
      ) : null}
    </div>
  );
}

function joinNonNull(parts: Array<string | null | undefined>): string {
  return parts.filter(Boolean).join(" · ");
}

function fmtScore(v: number | null | undefined): string | null {
  if (v === null || v === undefined) return null;
  return v.toFixed(3);
}

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

function fmtTimeRange(start: string | null, end: string | null, duration: number | null): string | null {
  const a = parseTime(start);
  const b = parseTime(end);
  const dur = duration ? ` · ${duration.toFixed(1)}s` : "";
  if (a && b) {
    const range = sameUtcDate(a, b)
      ? `${formatUtcDate(a)} ${formatUtcClock(a)}–${formatUtcClock(b)}`
      : `${formatUtcDateTime(a)} – ${formatUtcDateTime(b)}`;
    return `${range}${dur}`;
  }
  if (a) return `${formatUtcDateTime(a)}${dur}`;
  if (b) return `${formatUtcDateTime(b)}${dur}`;
  return duration ? `${duration.toFixed(1)}s` : null;
}
