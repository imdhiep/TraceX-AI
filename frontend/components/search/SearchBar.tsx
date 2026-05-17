"use client";

import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DayPicker, type DateRange } from "react-day-picker";
import { vi } from "date-fns/locale";
import { CalendarRange, ImagePlus, MapPin, X } from "lucide-react";
import "react-day-picker/style.css";

import { LOCATION_OPTIONS, summarizeSelectedLocations } from "@/lib/config";
import { useSearch } from "@/features/search/SearchContext";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

function parseDateTimeLocal(value: string): Date | undefined {
  if (!value) return undefined;
  const [datePart, timePart = "00:00"] = value.split("T");
  const [y, m, d] = datePart.split("-").map(Number);
  const [hh, mm] = timePart.split(":").map(Number);
  if (!y || !m || !d) return undefined;
  return new Date(y, m - 1, d, hh || 0, mm || 0);
}

function formatDateTimeLocal(date: Date | undefined, timeHHmm: string): string {
  if (!date) return "";
  const [hh, mm] = (timeHHmm || "00:00").split(":");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${hh}:${mm}`;
}

function extractTime(value: string): string {
  if (!value) return "";
  const parts = value.split("T");
  return parts[1] ?? "";
}

function formatRangeLabel(from: string, to: string): string {
  const f = parseDateTimeLocal(from);
  const t = parseDateTimeLocal(to);
  if (!f && !t) return "Thời gian";
  const fmt = (d: Date) =>
    `${pad(d.getDate())}/${pad(d.getMonth() + 1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (f && t) return `${fmt(f)} → ${fmt(t)}`;
  if (f) return `Từ ${fmt(f)}`;
  return `Đến ${fmt(t!)}`;
}

type SearchBarProps = {
  className?: string;
};

export function SearchBar({ className = "" }: SearchBarProps) {
  const {
    query,
    setQuery,
    image,
    setImage,
    runSearch,
    isLoading,
    filters,
    setLocationIds,
    setTimeFrom,
    setTimeTo,
    clearFilters,
    filterError,
  } = useSearch();
  const [locationOpen, setLocationOpen] = useState(false);
  const [locationKeyword, setLocationKeyword] = useState("");
  const [imageError, setImageError] = useState<string | null>(null);
  const [imagePreviewOpen, setImagePreviewOpen] = useState(false);
  const [dateOpen, setDateOpen] = useState(false);
  const [datePos, setDatePos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const [mounted, setMounted] = useState(false);
  const locationRef = useRef<HTMLDivElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const imageButtonRef = useRef<HTMLButtonElement | null>(null);
  const queryInputRef = useRef<HTMLInputElement | null>(null);
  const dateRef = useRef<HTMLDivElement | null>(null);
  const datePopoverRef = useRef<HTMLDivElement | null>(null);
  const dateButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  function recomputeDatePos() {
    if (!dateButtonRef.current) return;
    const rect = dateButtonRef.current.getBoundingClientRect();
    setDatePos({ top: rect.bottom + 8, left: rect.left });
  }

  function toggleDateOpen() {
    if (!dateOpen) recomputeDatePos();
    setDateOpen((prev) => !prev);
  }

  useEffect(() => {
    if (!dateOpen) return;
    function onScrollOrResize() {
      recomputeDatePos();
    }
    window.addEventListener("scroll", onScrollOrResize, true);
    window.addEventListener("resize", onScrollOrResize);
    return () => {
      window.removeEventListener("scroll", onScrollOrResize, true);
      window.removeEventListener("resize", onScrollOrResize);
    };
  }, [dateOpen]);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (locationRef.current && !locationRef.current.contains(target)) {
        setLocationOpen(false);
      }
      const insideDateTrigger = dateRef.current?.contains(target) ?? false;
      const insideDatePopover = datePopoverRef.current?.contains(target) ?? false;
      if (!insideDateTrigger && !insideDatePopover) {
        setDateOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const dateRange: DateRange | undefined = useMemo(() => {
    const from = parseDateTimeLocal(filters.timeFrom);
    const to = parseDateTimeLocal(filters.timeTo);
    if (!from && !to) return undefined;
    return { from, to };
  }, [filters.timeFrom, filters.timeTo]);

  const fromTime = extractTime(filters.timeFrom) || "00:00";
  const toTime = extractTime(filters.timeTo) || "23:59";

  function handleRangeSelect(range: DateRange | undefined) {
    if (!range) {
      setTimeFrom("");
      setTimeTo("");
      return;
    }
    setTimeFrom(range.from ? formatDateTimeLocal(range.from, fromTime) : "");
    setTimeTo(range.to ? formatDateTimeLocal(range.to, toTime) : "");
  }

  function handleFromTimeChange(value: string) {
    const current = parseDateTimeLocal(filters.timeFrom) ?? dateRange?.from;
    if (!current) {
      const today = new Date();
      setTimeFrom(formatDateTimeLocal(today, value));
      return;
    }
    setTimeFrom(formatDateTimeLocal(current, value));
  }

  function handleToTimeChange(value: string) {
    const current = parseDateTimeLocal(filters.timeTo) ?? dateRange?.to;
    if (!current) {
      const today = new Date();
      setTimeTo(formatDateTimeLocal(today, value));
      return;
    }
    setTimeTo(formatDateTimeLocal(current, value));
  }

  useEffect(() => {
    return () => {
      if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    };
  }, [image]);

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setImageError("Chỉ hỗ trợ file ảnh.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setImageError("Ảnh không được vượt quá 5MB.");
      return;
    }
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    setImageError(null);
    setImage({
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      previewUrl: URL.createObjectURL(file),
    });
    imageButtonRef.current?.blur();
    queryInputRef.current?.focus();
  }

  function handleRemoveImage() {
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    setImage(null);
    setImageError(null);
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  const filteredLocations = useMemo(() => {
    const keyword = locationKeyword.trim().toLowerCase();
    if (!keyword) {
      return LOCATION_OPTIONS;
    }
    return LOCATION_OPTIONS.filter((option) => {
      const labels = [option.label, ...option.keywords].join(" ").toLowerCase();
      return labels.includes(keyword);
    });
  }, [locationKeyword]);

  const selectedSummary = summarizeSelectedLocations(filters.locationIds);
  const selectedCount = filters.locationIds.length;

  function toggleLocation(locationId: string) {
    const selected = new Set(filters.locationIds);
    if (selected.has(locationId)) {
      selected.delete(locationId);
    } else {
      selected.add(locationId);
    }
    setLocationIds(Array.from(selected));
  }

  return (
    <form
      className={["flex w-full flex-col gap-3", className].filter(Boolean).join(" ")}
      onSubmit={(e) => {
        e.preventDefault();
        void runSearch();
      }}
    >
      <div className="flex w-full flex-wrap gap-2">
        <input
          ref={queryInputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Nhập mô tả người hoặc hành vi cần tìm..."
          className="min-h-[52px] flex-1 rounded-2xl border border-slate-200/90 bg-gradient-to-br from-white/95 to-blue-50/70 px-5 text-base font-medium text-slate-900 shadow-[0_12px_30px_rgba(15,23,42,0.08)] outline-none ring-sky-300/35 transition-[border-color,box-shadow] duration-200 placeholder:text-slate-400 focus:border-sky-400 focus:ring-2 dark:border-slate-700/90 dark:bg-gradient-to-br dark:from-slate-900 dark:to-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500 dark:shadow-[0_16px_34px_rgba(2,6,23,0.35)]"
          aria-label="Ô tìm kiếm"
        />
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleImageChange}
        />
        <button
          ref={imageButtonRef}
          type="button"
          onClick={() => imageInputRef.current?.click()}
          title="Tải ảnh mẫu (tối đa 5MB)"
          className="min-h-[52px] shrink-0 cursor-pointer rounded-2xl border border-slate-200/90 bg-white/85 px-4 text-sm font-medium text-slate-700 shadow-[0_12px_30px_rgba(15,23,42,0.08)] transition duration-200 hover:border-sky-300 hover:text-slate-900 dark:border-slate-700/90 dark:bg-slate-900/85 dark:text-slate-200 dark:shadow-[0_16px_34px_rgba(2,6,23,0.35)] dark:hover:border-sky-500 dark:hover:text-white"
        >
          <ImagePlus className="h-5 w-5" strokeWidth={1.9} />
        </button>
        <button
          type="submit"
          disabled={isLoading}
          className="min-h-[52px] shrink-0 cursor-pointer rounded-2xl bg-[#0F172A] px-6 text-sm font-semibold text-white shadow-[0_14px_28px_rgba(15,23,42,0.24)] transition duration-200 hover:bg-[#1E293B] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-600 dark:shadow-[0_16px_32px_rgba(37,99,235,0.3)] dark:hover:bg-blue-500"
        >
          {isLoading ? "Đang tìm..." : "Gửi"}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative" ref={locationRef}>
          <button
            type="button"
            className="inline-flex min-h-[44px] cursor-pointer items-center gap-2 rounded-xl border border-slate-200/90 bg-white/85 px-4 text-sm font-medium text-slate-700 shadow-[0_8px_20px_rgba(15,23,42,0.06)] transition duration-200 hover:border-sky-300 hover:text-slate-900 dark:border-slate-700/90 dark:bg-slate-900/85 dark:text-slate-200 dark:shadow-[0_12px_26px_rgba(2,6,23,0.32)] dark:hover:border-sky-500 dark:hover:text-white"
            onClick={() => setLocationOpen((prev) => !prev)}
          >
            <MapPin className="h-4 w-4" strokeWidth={1.9} />
            {selectedCount ? `${selectedSummary} (${selectedCount})` : "Vị trí"}
          </button>
          {locationOpen ? (
            <div className="absolute left-0 top-[calc(100%+8px)] z-40 w-[320px] rounded-2xl border border-slate-200/90 bg-white/95 p-3 shadow-[0_18px_36px_rgba(15,23,42,0.16)] backdrop-blur-md dark:border-slate-700/90 dark:bg-slate-950/95 dark:shadow-[0_24px_50px_rgba(2,6,23,0.45)]">
              <input
                type="text"
                value={locationKeyword}
                onChange={(event) => setLocationKeyword(event.target.value)}
                placeholder="Tìm vị trí..."
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none ring-sky-300/35 focus:border-sky-400 focus:ring-2 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500"
              />
              <div className="mt-2 max-h-56 space-y-1 overflow-y-auto pr-1">
                {filteredLocations.map((option) => {
                  const checked = filters.locationIds.includes(option.id);
                  return (
                    <label
                      key={option.id}
                      className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleLocation(option.id)}
                        className="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500 dark:border-slate-600 dark:bg-slate-900"
                      />
                      <span className="font-medium">{option.label}</span>
                    </label>
                  );
                })}
              </div>
              <button
                type="button"
                onClick={() => setLocationIds([])}
                className="mt-2 text-xs font-medium text-sky-700 hover:text-sky-800 dark:text-sky-400 dark:hover:text-sky-300"
              >
                Bỏ chọn vị trí
              </button>
            </div>
          ) : null}
        </div>

        <div className="relative" ref={dateRef}>
          <button
            ref={dateButtonRef}
            type="button"
            onClick={toggleDateOpen}
            className="inline-flex min-h-[44px] cursor-pointer items-center gap-2 rounded-xl border border-slate-200/90 bg-white/85 px-4 text-sm font-medium text-slate-700 shadow-[0_8px_20px_rgba(15,23,42,0.06)] transition duration-200 hover:border-sky-300 hover:text-slate-900 dark:border-slate-700/90 dark:bg-slate-900/85 dark:text-slate-200 dark:shadow-[0_12px_26px_rgba(2,6,23,0.32)] dark:hover:border-sky-500 dark:hover:text-white"
            aria-label="Chọn khoảng thời gian"
          >
            <CalendarRange className="h-4 w-4" strokeWidth={1.9} />
            {formatRangeLabel(filters.timeFrom, filters.timeTo)}
          </button>
        </div>
        <button
          type="button"
          onClick={clearFilters}
          className="min-h-[44px] cursor-pointer rounded-xl border border-slate-200/90 bg-white/85 px-4 text-sm font-medium text-slate-600 transition duration-200 hover:border-slate-300 hover:text-slate-800 dark:border-slate-700/90 dark:bg-slate-900/85 dark:text-slate-300 dark:hover:border-slate-600 dark:hover:text-white"
        >
          Xóa lọc
        </button>
      </div>
      {image ? (
        <div className="flex flex-wrap items-center gap-2">
          <div className="group relative inline-flex items-center gap-2 rounded-2xl border border-slate-200/90 bg-white/85 p-1.5 pr-3 shadow-[0_8px_20px_rgba(15,23,42,0.06)] dark:border-slate-700/90 dark:bg-slate-900/85 dark:shadow-[0_12px_26px_rgba(2,6,23,0.32)]">
            <button
              type="button"
              onClick={() => setImagePreviewOpen(true)}
              title="Xem ảnh phóng to"
              className="relative h-12 w-12 shrink-0 overflow-hidden rounded-xl border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-800"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={image.previewUrl} alt={image.name} className="h-full w-full object-cover" />
            </button>
            <div className="flex min-w-0 flex-col">
              <span className="max-w-[180px] truncate text-sm font-medium text-slate-800 dark:text-slate-100">{image.name}</span>
              <span className="text-[11px] text-slate-500 dark:text-slate-400">{(image.size / 1024).toFixed(0)} KB</span>
            </div>
            <button
              type="button"
              onClick={handleRemoveImage}
              title="Xóa ảnh"
              className="ml-1 flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition hover:border-red-300 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-400 dark:hover:border-red-500/60 dark:hover:bg-red-950/40 dark:hover:text-red-300"
            >
              <X className="h-3.5 w-3.5" strokeWidth={2.5} />
            </button>
          </div>
        </div>
      ) : null}
      {filterError ? <p className="text-xs font-medium text-red-600 dark:text-red-400">{filterError}</p> : null}
      {imageError ? <p className="text-xs font-medium text-red-600 dark:text-red-400">{imageError}</p> : null}

      {mounted && dateOpen
        ? createPortal(
            <div
              ref={datePopoverRef}
              style={{ position: "fixed", top: datePos.top, left: datePos.left, zIndex: 9999 }}
              className="w-max max-w-[calc(100vw-2rem)] rounded-2xl border border-slate-200/90 bg-white/95 p-3 shadow-[0_18px_36px_rgba(15,23,42,0.16)] backdrop-blur-md dark:border-slate-700/90 dark:bg-slate-950/95 dark:shadow-[0_24px_50px_rgba(2,6,23,0.45)]"
            >
              <DayPicker
                mode="range"
                numberOfMonths={2}
                locale={vi}
                selected={dateRange}
                onSelect={handleRangeSelect}
                weekStartsOn={1}
                showOutsideDays
                className="rdp-tracex"
              />
              <div className="mt-2 flex flex-wrap items-center gap-3 border-t border-slate-200 pt-3 dark:border-slate-800">
                <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
                  Từ
                  <input
                    type="time"
                    value={fromTime}
                    onChange={(e) => handleFromTimeChange(e.target.value)}
                    className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm text-slate-800 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-300/35 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:[color-scheme:dark]"
                  />
                </label>
                <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
                  Đến
                  <input
                    type="time"
                    value={toTime}
                    onChange={(e) => handleToTimeChange(e.target.value)}
                    className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm text-slate-800 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-300/35 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:[color-scheme:dark]"
                  />
                </label>
                <div className="ml-auto flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setTimeFrom("");
                      setTimeTo("");
                    }}
                    className="text-xs font-medium text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                  >
                    Xoá
                  </button>
                  <button
                    type="button"
                    onClick={() => setDateOpen(false)}
                    className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-500"
                  >
                    Xong
                  </button>
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {imagePreviewOpen && image ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm"
          onClick={() => setImagePreviewOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <div className="relative max-h-full max-w-4xl" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              onClick={() => setImagePreviewOpen(false)}
              className="absolute -right-3 -top-3 flex h-9 w-9 cursor-pointer items-center justify-center rounded-full bg-white text-slate-700 shadow-lg transition hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              title="Đóng"
            >
              <X className="h-4 w-4" strokeWidth={2.5} />
            </button>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={image.previewUrl}
              alt={image.name}
              className="max-h-[85vh] max-w-full rounded-2xl object-contain shadow-2xl"
            />
            <p className="mt-3 text-center text-sm text-white/90">{image.name}</p>
          </div>
        </div>
      ) : null}
    </form>
  );
}
