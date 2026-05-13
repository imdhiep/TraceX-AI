"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getTraceTimeline, resolveMediaUrl } from "@/lib/api";
import type { BuildTraceResult, TraceSegment } from "@/lib/types";

type Props = {
  evidenceId: number;
  queryId: string | null;
  candidateId: string | null;
};

export function TraceView({ evidenceId, queryId, candidateId }: Props) {
  const [trace, setTrace] = useState<BuildTraceResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    void getTraceTimeline(evidenceId)
      .then((t) => {
        if (cancelled) return;
        setTrace(t);
        setActiveIdx(0);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Không thể tải timeline.");
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [evidenceId]);

  const segments = trace?.segments ?? [];
  const active = segments[activeIdx];

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Thông tin truy vết</p>
          <h1 className="text-2xl font-semibold text-ink">
            Evidence #{evidenceId}
            {candidateId ? <span className="ml-2 font-mono text-sm text-slate-500">· {candidateId}</span> : null}
          </h1>
        </div>
        <Link
          href={queryId ? "/home" : "/home"}
          className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
        >
          ← Quay lại
        </Link>
      </header>

      {isLoading ? <p className="text-sm text-ink-secondary">Đang tải timeline…</p> : null}
      {error ? (
        <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>
      ) : null}

      {!isLoading && !error && trace ? (
        <>
          <CameraTimeline segments={segments} activeIdx={activeIdx} onSelect={setActiveIdx} />
          {active ? <SegmentPlayer segment={active} /> : (
            <p className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-sm text-slate-500">
              Trace không có segment nào.
            </p>
          )}
          <SegmentList segments={segments} activeIdx={activeIdx} onSelect={setActiveIdx} />
        </>
      ) : null}
    </div>
  );
}

function CameraTimeline({
  segments,
  activeIdx,
  onSelect,
}: {
  segments: TraceSegment[];
  activeIdx: number;
  onSelect: (i: number) => void;
}) {
  if (!segments.length) return null;
  return (
    <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <ol className="flex items-stretch gap-3">
        {segments.map((s, i) => {
          const isActive = i === activeIdx;
          return (
            <li key={`${s.segmentOrder}-${s.trackletId}`} className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => onSelect(i)}
                className={`flex w-44 shrink-0 flex-col items-start gap-1 rounded-xl border px-3 py-2 text-left transition ${
                  isActive
                    ? "border-sky-500 bg-sky-50 shadow-sm"
                    : "border-slate-200 bg-slate-50 hover:border-slate-300"
                }`}
              >
                <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  Step {s.segmentOrder}
                </span>
                <span className="font-mono text-sm text-slate-900">{s.cameraId ?? "—"}</span>
                <span className="text-[11px] text-slate-500">{fmtTime(s.timeStart)}</span>
                {s.durationSeconds != null ? (
                  <span className="text-[11px] text-slate-500">{s.durationSeconds.toFixed(1)}s</span>
                ) : null}
              </button>
              {i < segments.length - 1 ? <span className="text-slate-300">→</span> : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function SegmentPlayer({ segment }: { segment: TraceSegment }) {
  const url = segment.videoClipUrl ? resolveMediaUrl(segment.videoClipUrl) : null;
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-black/90 shadow-card">
      {url ? (
        <video
          key={url}
          src={url}
          controls
          preload="metadata"
          className="aspect-video w-full bg-black"
        />
      ) : (
        <div className="flex aspect-video w-full items-center justify-center text-sm text-white/70">
          Không có video cho segment này.
        </div>
      )}
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 bg-white/95 px-4 py-3 text-xs text-slate-600">
        <span className="font-mono text-sm text-slate-900">{segment.cameraId ?? "—"}</span>
        <span>{fmtRange(segment.timeStart, segment.timeEnd)}</span>
        {segment.durationSeconds != null ? <span>· {segment.durationSeconds.toFixed(1)}s</span> : null}
        {typeof segment.confidence === "number" ? (
          <span>· conf {Math.round(segment.confidence * 100)}%</span>
        ) : null}
        <span className="ml-auto font-mono text-[11px] text-slate-500" title={segment.trackletId}>
          {segment.trackletId}
        </span>
      </div>
    </div>
  );
}

function SegmentList({
  segments,
  activeIdx,
  onSelect,
}: {
  segments: TraceSegment[];
  activeIdx: number;
  onSelect: (i: number) => void;
}) {
  if (!segments.length) return null;
  return (
    <ol className="flex flex-col gap-3">
      {segments.map((s, i) => {
        const isActive = i === activeIdx;
        const thumb = s.thumbnailUrl ? resolveMediaUrl(s.thumbnailUrl) : null;
        return (
          <li key={`${s.segmentOrder}-${s.trackletId}-row`}>
            <button
              type="button"
              onClick={() => onSelect(i)}
              className={`flex w-full items-center gap-4 rounded-2xl border bg-white p-3 text-left shadow-sm transition ${
                isActive ? "border-sky-500 ring-2 ring-sky-200" : "border-slate-200 hover:border-slate-300"
              }`}
            >
              <span className="w-8 shrink-0 text-center text-xs font-semibold text-slate-500">#{s.segmentOrder}</span>
              <div className="h-16 w-24 shrink-0 overflow-hidden rounded-lg bg-slate-100">
                {thumb ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={thumb} alt={s.trackletId} className="h-full w-full object-cover" />
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-0.5">
                <p className="font-mono text-sm text-slate-900">{s.cameraId ?? "—"}</p>
                <p className="text-xs text-slate-500">{fmtRange(s.timeStart, s.timeEnd)}</p>
                <p className="font-mono text-[11px] text-slate-400">{s.trackletId}</p>
              </div>
              <div className="text-right text-xs text-slate-600">
                {s.durationSeconds != null ? <p>{s.durationSeconds.toFixed(1)}s</p> : null}
                {typeof s.confidence === "number" ? (
                  <p className="text-slate-400">conf {Math.round(s.confidence * 100)}%</p>
                ) : null}
              </div>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString(undefined, { hour12: false });
}

function fmtRange(start: string | null, end: string | null): string {
  if (!start && !end) return "—";
  return `${fmtTime(start)} → ${fmtTime(end)}`;
}
