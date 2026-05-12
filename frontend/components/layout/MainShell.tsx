"use client";

import { usePathname } from "next/navigation";
import { Suspense, type ReactNode } from "react";

import { SearchBar } from "@/components/search/SearchBar";
import { TopKSelect } from "@/components/search/TopKSelect";
import { useSearch } from "@/features/search/SearchContext";

import { AccountMenu } from "./AccountMenu";
import { Sidebar } from "./Sidebar";

function SidebarFallback() {
  return (
    <aside className="flex h-screen w-[264px] shrink-0 animate-pulse border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950" />
  );
}

export function MainShell({ children }: { children: ReactNode }) {
  const { hasSearched } = useSearch();
  const pathname = usePathname();
  const showSearchControls = hasSearched && pathname === "/home";

  return (
    <div className="flex h-screen overflow-hidden bg-gradient-to-br from-slate-50 via-slate-100 to-blue-50 dark:from-slate-950 dark:via-slate-950 dark:to-slate-900">
      <Suspense fallback={<SidebarFallback />}>
        <Sidebar />
      </Suspense>
      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <header
          className={
            showSearchControls
              ? "sticky top-0 z-30 border-b border-slate-200/70 bg-white/75 px-6 pb-4 pt-5 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-950/80"
              : "px-6 pb-2 pt-5"
          }
        >
          <div className="mx-auto flex max-w-6xl flex-col gap-4">
            <div className="rounded-2xl border border-slate-200/85 bg-white/65 p-3 shadow-[0_12px_30px_rgba(15,23,42,0.07)] backdrop-blur-md dark:border-slate-800/85 dark:bg-slate-900/70 dark:shadow-[0_18px_40px_rgba(2,6,23,0.38)]">
              <div className="flex flex-wrap items-start gap-4">
                <SearchBar className="min-w-[min(100%,280px)] flex-1" />
                <div className="ml-auto flex shrink-0 items-start gap-3 pt-1">
                  {showSearchControls ? <TopKSelect /> : null}
                  <AccountMenu />
                </div>
              </div>
            </div>
          </div>
        </header>
        <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-8">{children}</main>
      </div>
    </div>
  );
}