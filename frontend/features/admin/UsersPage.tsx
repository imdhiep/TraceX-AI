"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { loadAccessToken } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";
import { useToast } from "@/components/ui/ToastProvider";

type UserItem = {
  id: number;
  email: string;
  full_name: string;
  role: "SUPER_ADMIN" | "ADMIN" | "USER";
  is_active: boolean;
  last_login?: string | null;
};

const ROLE_RANK: Record<UserItem["role"], number> = {
  USER: 1,
  ADMIN: 2,
  SUPER_ADMIN: 3,
};
const ROLE_OPTIONS: UserItem["role"][] = ["SUPER_ADMIN", "ADMIN", "USER"];
const ROLE_BADGE_CLASS: Record<UserItem["role"], string> = {
  SUPER_ADMIN: "border-violet-200 bg-violet-100 text-violet-700 dark:border-violet-800/70 dark:bg-violet-950/55 dark:text-violet-300",
  ADMIN: "border-blue-200 bg-blue-100 text-blue-700 dark:border-blue-800/70 dark:bg-blue-950/55 dark:text-blue-300",
  USER: "border-slate-200 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200",
};

const PANEL_CLASS =
  "rounded-2xl border border-slate-200 bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.34)]";
const FIELD_CLASS =
  "rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-300/25 placeholder:text-slate-400 focus:border-accent focus:ring-2 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:placeholder:text-slate-500 dark:[color-scheme:dark]";
const SMALL_FIELD_CLASS =
  "rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm text-slate-900 outline-none ring-sky-300/25 focus:border-accent focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:[color-scheme:dark]";

export function UsersPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [currentUser, setCurrentUser] = useState<UserItem | null>(null);
  const [items, setItems] = useState<UserItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserItem["role"]>("USER");
  const [isActive, setIsActive] = useState(true);
  const currentUserRole = (currentUser?.role ?? "USER") as UserItem["role"];
  const currentUserRank = ROLE_RANK[currentUserRole] ?? 1;
  const currentUserId = currentUser?.id ?? 0;

  async function authFetch(path: string, init?: RequestInit): Promise<Response> {
    const token = loadAccessToken();
    return fetch(`${getApiBaseUrl()}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token ?? ""}`,
        ...(init?.headers ?? {}),
      },
    });
  }

  async function loadUsers() {
    setLoading(true);
    try {
      const response = await authFetch("/users");
      if (response.status === 401) {
        showToast("Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.", "error");
        router.replace("/login");
        return;
      }
      if (response.status === 403) {
        showToast("Bạn không có quyền admin.", "error");
        return;
      }
      if (!response.ok) throw new Error(await readApiErrorMessage(response));
      const payload = await parseJsonOrThrow<{ items: UserItem[] }>(response);
      setItems(payload.items ?? []);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Tải danh sách user thất bại.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setLoading(false);
    }
  }

  async function loadCurrentUser() {
    const response = await authFetch("/auth/me");
    if (response.status === 401) {
      showToast("Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.", "error");
      router.replace("/login");
      return null;
    }
    if (!response.ok) {
      throw new Error(await readApiErrorMessage(response));
    }
    return await parseJsonOrThrow<UserItem>(response);
  }

  useEffect(() => {
    void (async () => {
      try {
        const me = await loadCurrentUser();
        if (!me) return;
        setCurrentUser(me);
        if (!["ADMIN", "SUPER_ADMIN"].includes(me.role)) {
          showToast("Bạn không có quyền vào trang quản lý user.", "error");
          router.replace("/home");
          return;
        }
        await loadUsers();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Không thể tải thông tin người dùng.";
        showToast(mapBackendErrorMessage(message), "error");
      }
    })();
  }, [router]);

  async function handleCreateUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const response = await authFetch("/users", {
      method: "POST",
      body: JSON.stringify({ email, full_name: fullName, password, role, is_active: isActive }),
    });
    if (!response.ok) {
      showToast(mapBackendErrorMessage(await readApiErrorMessage(response)), "error");
      return;
    }
    setEmail("");
    setFullName("");
    setPassword("");
    setRole("USER");
    setIsActive(true);
    showToast("Tạo user thành công.", "success");
    await loadUsers();
  }

  async function handleUpdateUser(userId: number, patch: { role?: UserItem["role"]; is_active?: boolean }) {
    const response = await authFetch(`/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    if (!response.ok) {
      showToast(mapBackendErrorMessage(await readApiErrorMessage(response)), "error");
      return;
    }
    showToast("Cập nhật user thành công.", "success");
    await loadUsers();
  }

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <section className={PANEL_CLASS}>
        <h1 className="text-xl font-semibold text-slate-950 dark:text-white">Users</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Quản lý user (ADMIN và SUPER_ADMIN).</p>
      </section>

      <section className={PANEL_CLASS}>
        <form className="grid grid-cols-1 gap-3 md:grid-cols-12" onSubmit={handleCreateUser}>
          <input
            className={`${FIELD_CLASS} md:col-span-3`}
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            className={`${FIELD_CLASS} md:col-span-3`}
            placeholder="Full name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            required
          />
          <input
            className={`${FIELD_CLASS} md:col-span-2`}
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <select
            className={`${FIELD_CLASS} md:col-span-2`}
            value={isActive ? "active" : "inactive"}
            onChange={(e) => setIsActive(e.target.value === "active")}
          >
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
          <div className="flex min-w-0 gap-2 md:col-span-2">
            <select
              className={`${FIELD_CLASS} min-w-0 flex-1`}
              value={role}
              onChange={(e) => setRole(e.target.value as UserItem["role"])}
            >
              {ROLE_OPTIONS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <button className="shrink-0 rounded-xl bg-accent px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 dark:bg-blue-600 dark:hover:bg-blue-500" type="submit">
              Create
            </button>
          </div>
        </form>
      </section>

      <section className={PANEL_CLASS}>
        {loading ? <p className="text-sm text-slate-500 dark:text-slate-400">Đang tải users...</p> : null}
        <div className="space-y-2">
          {items
            .filter((user) => (ROLE_RANK[user.role] ?? 1) < currentUserRank && user.id !== currentUserId)
            .map((user) => {
            const isSelf = user.id === currentUserId;
            const targetRank = ROLE_RANK[user.role] ?? 1;
            const canManageTarget = targetRank < currentUserRank;
            return (
              <div
                key={user.id}
                className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-3 py-2 transition-colors dark:border-slate-700 dark:bg-slate-950/45"
              >
                <div>
                  <p className="flex items-center gap-2 text-sm font-medium text-slate-950 dark:text-slate-100">
                    <span>{user.email}</span>
                    <span className={["rounded-full border px-2 py-0.5 text-[11px] font-semibold", ROLE_BADGE_CLASS[user.role]].join(" ")}>
                      {user.role}
                    </span>
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {user.full_name} · {user.is_active ? "Active" : "Inactive"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    className={SMALL_FIELD_CLASS}
                    value={user.role}
                    disabled={isSelf || !canManageTarget}
                    onChange={(e) => void handleUpdateUser(user.id, { role: e.target.value as UserItem["role"] })}
                  >
                    {ROLE_OPTIONS.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={isSelf || !canManageTarget}
                    onClick={() => void handleUpdateUser(user.id, { is_active: !user.is_active })}
                    className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 transition hover:bg-slate-50 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
                  >
                    {user.is_active ? "Disable" : "Enable"}
                  </button>
                  <Link
                    href={`/admin/users/${user.id}/queries`}
                    className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 transition hover:bg-slate-50 hover:text-slate-900 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
                  >
                    Queries
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
