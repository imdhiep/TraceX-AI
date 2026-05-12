"use client";

import Link from "next/link";
import { type FormEvent, useEffect, useMemo, useState } from "react";

import { clearSession, loadAccessToken, loadSessionUser, type AuthUser } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";
import { useToast } from "@/components/ui/ToastProvider";

type MeResponse = {
  id: number;
  email: string;
  full_name: string;
  role: "SUPER_ADMIN" | "ADMIN" | "USER";
  is_active: boolean;
  last_login?: string | null;
  created_at?: string;
};

type OverviewResponse = {
  metrics?: {
    total_users?: number;
    total_managed_videos?: number;
    total_queries?: number;
    total_candidates?: number;
    total_cameras?: number;
    total_candidate_videos?: number;
    total_queue_videos?: number;
  };
};

function formatDateTime(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString("vi-VN");
}

export function SettingsPage() {
  const { showToast } = useToast();
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);
  const [me, setMe] = useState<MeResponse | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);

  const isAdmin = useMemo(
    () => ["ADMIN", "SUPER_ADMIN"].includes((me?.role || sessionUser?.role || "").toUpperCase()),
    [me?.role, sessionUser?.role],
  );

  async function authFetch(path: string, init?: RequestInit): Promise<Response> {
    const token = loadAccessToken();
    return fetch(`${getApiBaseUrl()}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token ?? ""}`,
        ...(init?.headers ?? {}),
      },
      cache: "no-store",
    });
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!currentPassword.trim() || !newPassword.trim() || !confirmPassword.trim()) {
      showToast("Vui lòng nhập đầy đủ thông tin.", "error");
      return;
    }
    if (newPassword.length < 8) {
      showToast("Mật khẩu mới phải có ít nhất 8 ký tự.", "error");
      return;
    }
    if (newPassword !== confirmPassword) {
      showToast("Xác nhận mật khẩu không khớp.", "error");
      return;
    }
    setChangingPassword(true);
    try {
      const res = await authFetch("/auth/change-password", {
        method: "POST",
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      });
      if (!res.ok) throw new Error(await readApiErrorMessage(res));
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      showToast("Đổi mật khẩu thành công.", "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể đổi mật khẩu.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setChangingPassword(false);
    }
  }

  async function loadSettingsData() {
    setLoading(true);
    try {
      const meRes = await authFetch("/auth/me");
      if (!meRes.ok) throw new Error(await readApiErrorMessage(meRes));
      const mePayload = await parseJsonOrThrow<MeResponse>(meRes);
      setMe(mePayload);

      // Overview is optional for settings UX; don't block page on downstream AI auth failures.
      const overviewRes = await authFetch("/search/overview");
      if (overviewRes.ok) {
        const overviewPayload = await parseJsonOrThrow<OverviewResponse>(overviewRes);
        setOverview(overviewPayload);
      } else {
        setOverview(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Khong the tai settings.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setSessionUser(loadSessionUser());
    void loadSettingsData();
  }, []);

  return (
    <div className="space-y-6 text-ink dark:text-slate-100">
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]">
        <h1 className="text-xl font-semibold text-slate-950 dark:text-white">Settings</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Quan ly tai khoan, phien dang nhap va thong tin he thong.</p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]">
          <p className="text-sm font-semibold text-slate-950 dark:text-white">Thong tin tai khoan</p>
          {loading ? <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Dang tai...</p> : null}
          {!loading ? (
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500 dark:text-slate-400">Email</dt>
                <dd className="font-medium text-slate-950 dark:text-slate-100">{me?.email || sessionUser?.email || "--"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500 dark:text-slate-400">Role</dt>
                <dd className="font-medium text-slate-950 dark:text-slate-100">{me?.role || sessionUser?.role || "--"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500 dark:text-slate-400">Trang thai</dt>
                <dd className="font-medium text-slate-950 dark:text-slate-100">{me?.is_active === false ? "Inactive" : "Active"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500 dark:text-slate-400">Lan dang nhap cuoi</dt>
                <dd className="font-medium text-slate-950 dark:text-slate-100">{formatDateTime(me?.last_login)}</dd>
              </div>
            </dl>
          ) : null}
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]">
          <p className="text-sm font-semibold text-slate-950 dark:text-white">Tac vu nhanh</p>
          <div className="mt-3 flex flex-col gap-2">
            <button
              type="button"
              onClick={async () => {
                await loadSettingsData();
                showToast("Da refresh du lieu settings.", "success");
              }}
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-sm text-slate-800 transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Refresh du lieu
            </button>
            {isAdmin ? (
              <Link
                href="/admin/users"
                className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                Mo User management
              </Link>
            ) : null}
            <button
              type="button"
              onClick={() => {
                clearSession();
                window.location.href = "/login";
              }}
              className="rounded-xl border border-red-200 bg-white px-3 py-2 text-left text-sm text-red-600 transition hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/20 dark:text-red-300 dark:hover:bg-red-950/40"
            >
              Dang xuat tai khoan
            </button>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]">
        <p className="text-sm font-semibold text-slate-950 dark:text-white">Đổi mật khẩu</p>
        <form className="mt-3 flex flex-col gap-3" onSubmit={handleChangePassword}>
          <input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            placeholder="Mật khẩu hiện tại"
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-300/25 placeholder:text-slate-400 focus:border-accent focus:ring-2 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="Mật khẩu mới (tối thiểu 8 ký tự)"
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-300/25 placeholder:text-slate-400 focus:border-accent focus:ring-2 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Xác nhận mật khẩu mới"
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-300/25 placeholder:text-slate-400 focus:border-accent focus:ring-2 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
          <button
            type="submit"
            disabled={changingPassword}
            className="rounded-xl bg-accent px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-600 dark:hover:bg-blue-500"
          >
            {changingPassword ? "Đang xử lý..." : "Đổi mật khẩu"}
          </button>
        </form>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]">
        <p className="text-sm font-semibold text-slate-950 dark:text-white">Thong tin he thong</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-950/45">
            <p className="text-xs text-slate-500 dark:text-slate-400">Users</p>
            <p className="text-lg font-semibold text-slate-950 dark:text-slate-100">{overview?.metrics?.total_users ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-950/45">
            <p className="text-xs text-slate-500 dark:text-slate-400">Managed videos</p>
            <p className="text-lg font-semibold text-slate-950 dark:text-slate-100">{overview?.metrics?.total_managed_videos ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-950/45">
            <p className="text-xs text-slate-500 dark:text-slate-400">Queries</p>
            <p className="text-lg font-semibold text-slate-950 dark:text-slate-100">{overview?.metrics?.total_queries ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-950/45">
            <p className="text-xs text-slate-500 dark:text-slate-400">Queue videos</p>
            <p className="text-lg font-semibold text-slate-950 dark:text-slate-100">{overview?.metrics?.total_queue_videos ?? "--"}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
