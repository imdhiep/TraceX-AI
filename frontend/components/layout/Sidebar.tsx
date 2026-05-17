"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  History,
  LayoutDashboard,
  Moon,
  ScanSearch,
  Settings,
  SquarePlus,
  Sun,
  Users,
} from "lucide-react";

import { loadSessionUser, type AuthUser } from "@/lib/auth";
import { useSearch } from "@/features/search/SearchContext";

function NavIcon({ active, children }: { active: boolean; children: ReactNode }) {
  return (
    <span
      className={[
        "flex h-9 w-9 items-center justify-center rounded-xl transition-colors",
        active ? "bg-blue-600 text-white" : "text-slate-500 group-hover:text-blue-600 dark:text-slate-400 dark:group-hover:text-blue-300",
      ].join(" ")}
    >
      {children}
    </span>
  );
}

function navItemClass(active: boolean, collapsed = false) {
  return [
    "group relative flex items-center rounded-2xl text-sm font-semibold transition-[background-color,color,box-shadow] duration-200 cursor-pointer",
    collapsed ? "justify-center px-2 py-2.5" : "gap-3 px-3 py-2.5",
    active
      ? "bg-blue-600 text-white shadow-[0_14px_28px_rgba(37,99,235,0.28)]"
      : "text-slate-600 hover:bg-blue-50 hover:text-slate-950 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white",
  ].join(" ");
}

function NavLabel({ collapsed, children }: { collapsed: boolean; children: ReactNode }) {
  return (
    <>
      <span className={collapsed ? "sr-only" : "truncate"}>{children}</span>
      {collapsed ? (
        <span className="pointer-events-none absolute left-[calc(100%+10px)] top-1/2 z-50 -translate-y-1/2 whitespace-nowrap rounded-xl bg-slate-950 px-3 py-2 text-xs font-bold text-white opacity-0 shadow-[0_12px_28px_rgba(15,23,42,0.24)] transition group-hover:opacity-100 dark:bg-white dark:text-slate-950">
          {children}
        </span>
      ) : null}
    </>
  );
}

type NavItemProps = {
  active: boolean;
  collapsed: boolean;
  href?: string;
  label: string;
  onClick?: () => void;
  children: ReactNode;
};

function NavItem({ active, collapsed, href, label, onClick, children }: NavItemProps) {
  const content = (
    <>
      <NavIcon active={active}>{children}</NavIcon>
      <NavLabel collapsed={collapsed}>{label}</NavLabel>
    </>
  );
  if (href) {
    return (
      <Link href={href} onClick={onClick} className={navItemClass(active, collapsed)} aria-label={label}>
        {content}
      </Link>
    );
  }
  return (
    <button type="button" onClick={onClick} className={navItemClass(active, collapsed)} aria-label={label}>
      {content}
    </button>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const view = searchParams.get("view");
  const { resetSearch } = useSearch();
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);
  const [darkMode, setDarkMode] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    setSessionUser(loadSessionUser());
  }, []);

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("mcpt-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const shouldUseDark = savedTheme ? savedTheme === "dark" : prefersDark;
    document.documentElement.classList.toggle("dark", shouldUseDark);
    setDarkMode(shouldUseDark);

    const savedCollapsed = window.localStorage.getItem("mcpt-sidebar-collapsed");
    setCollapsed(savedCollapsed === "true");
  }, []);

  const isHome = pathname === "/home";
  const isNew = isHome && view !== "history";
  const isHistory = pathname === "/history" || pathname.startsWith("/history/");
  const isDetail = pathname.startsWith("/detail/");
  const isUsers = pathname === "/admin/users";
  const isGuide = pathname === "/guide";
  const isDashboard = pathname === "/dashboard";
  const isSettings = pathname === "/settings";
  const role = sessionUser?.role ?? "USER";
  const canAccessHome = role !== "SUPER_ADMIN";
  const canAccessUsers = ["ADMIN", "SUPER_ADMIN"].includes(role);
  const canAccessDashboard = role === "SUPER_ADMIN";

  function toggleTheme() {
    const nextDarkMode = !darkMode;
    document.documentElement.classList.toggle("dark", nextDarkMode);
    window.localStorage.setItem("mcpt-theme", nextDarkMode ? "dark" : "light");
    setDarkMode(nextDarkMode);
  }

  function toggleCollapsed() {
    const nextCollapsed = !collapsed;
    window.localStorage.setItem("mcpt-sidebar-collapsed", String(nextCollapsed));
    setCollapsed(nextCollapsed);
  }

  const iconClass = "h-[18px] w-[18px]";

  return (
    <>
      {/* Desktop sidebar */}
      <aside
        className={[
          "relative hidden h-screen shrink-0 flex-col border-r border-slate-200 bg-white px-3 py-4 shadow-[10px_0_28px_rgba(15,23,42,0.05)] transition-[width,background-color,border-color] duration-200 dark:border-slate-800 dark:bg-slate-950 md:flex",
          collapsed ? "w-[64px]" : "w-[264px]",
        ].join(" ")}
      >
        {/* Collapse toggle */}
        <button
          type="button"
          onClick={toggleCollapsed}
          className="absolute -right-3 top-24 z-40 flex h-7 w-7 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-[0_8px_20px_rgba(15,23,42,0.14)] transition hover:border-blue-300 hover:text-blue-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500"
          aria-label={collapsed ? "Mở rộng sidebar" : "Thu gọn sidebar"}
          aria-expanded={!collapsed}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>

        {/* Logo */}
        <div className={["mb-7 flex items-center gap-3", collapsed ? "justify-center px-0" : "px-1"].join(" ")}>
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-[0_12px_24px_rgba(37,99,235,0.28)]">
            <ScanSearch className="h-5 w-5" strokeWidth={2.2} />
          </div>
          {!collapsed ? (
            <div className="min-w-0">
              <p className="truncate text-sm font-black uppercase leading-none tracking-tight text-slate-950 dark:text-white">TraceX-AI</p>
              <p className="mt-1 truncate text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">
                AI-Powered CCTV
              </p>
            </div>
          ) : null}
        </div>

        {/* Nav */}
        <nav className="flex flex-1 flex-col gap-2">
          {canAccessHome ? (
            <>
              <NavItem href="/home" onClick={resetSearch} label="Tạo mới" active={isNew} collapsed={collapsed}>
                <SquarePlus className={iconClass} strokeWidth={1.9} />
              </NavItem>
              <NavItem href="/history" label="Lịch sử" active={isHistory} collapsed={collapsed}>
                <History className={iconClass} strokeWidth={1.9} />
              </NavItem>
            </>
          ) : null}
          {canAccessUsers ? (
            <NavItem href="/admin/users" label="Người dùng" active={isUsers} collapsed={collapsed}>
              <Users className={iconClass} strokeWidth={1.9} />
            </NavItem>
          ) : null}
          <NavItem href="/guide" label="Hướng dẫn" active={isGuide} collapsed={collapsed}>
            <BookOpen className={iconClass} strokeWidth={1.9} />
          </NavItem>
          {canAccessDashboard ? (
            <NavItem href="/dashboard" label="Dashboard" active={isDashboard} collapsed={collapsed}>
              <LayoutDashboard className={iconClass} strokeWidth={1.9} />
            </NavItem>
          ) : null}
          {isDetail && !collapsed ? (
            <p className="mt-4 px-3 text-xs font-medium text-slate-500 dark:text-slate-400">Đang xem chi tiết video</p>
          ) : null}
        </nav>

        {/* Dark mode toggle */}
        <div className="mb-4 border-t border-slate-200 pt-4 dark:border-slate-800">
          <NavItem onClick={toggleTheme} label={darkMode ? "Chế độ sáng" : "Chế độ tối"} active={false} collapsed={collapsed}>
            {darkMode ? <Sun className={iconClass} strokeWidth={1.9} /> : <Moon className={iconClass} strokeWidth={1.9} />}
          </NavItem>
        </div>

        {/* Settings */}
        <NavItem href="/settings" label="Cài đặt" active={isSettings} collapsed={collapsed}>
          <Settings className={iconClass} strokeWidth={1.9} />
        </NavItem>
      </aside>

      {/* Mobile bottom nav */}
      <nav className="fixed inset-x-3 bottom-3 z-50 flex items-center justify-around rounded-3xl border border-slate-200 bg-white/95 p-2 shadow-[0_18px_40px_rgba(15,23,42,0.18)] backdrop-blur transition-colors dark:border-slate-800 dark:bg-slate-950/95 md:hidden">
        {canAccessHome ? (
          <>
            <NavItem href="/home" onClick={resetSearch} label="Tạo mới" active={isNew} collapsed>
              <SquarePlus className={iconClass} strokeWidth={1.9} />
            </NavItem>
            <NavItem href="/history" label="Lịch sử" active={isHistory} collapsed>
              <History className={iconClass} strokeWidth={1.9} />
            </NavItem>
          </>
        ) : null}
        {canAccessUsers ? (
          <NavItem href="/admin/users" label="Người dùng" active={isUsers} collapsed>
            <Users className={iconClass} strokeWidth={1.9} />
          </NavItem>
        ) : null}
        <NavItem href="/guide" label="Hướng dẫn" active={isGuide} collapsed>
          <BookOpen className={iconClass} strokeWidth={1.9} />
        </NavItem>
        <NavItem href="/settings" label="Cài đặt" active={isSettings} collapsed>
          <Settings className={iconClass} strokeWidth={1.9} />
        </NavItem>
        <NavItem label="Đổi chế độ" active={false} collapsed onClick={toggleTheme}>
          {darkMode ? <Sun className={iconClass} strokeWidth={1.9} /> : <Moon className={iconClass} strokeWidth={1.9} />}
        </NavItem>
      </nav>
    </>
  );
}
