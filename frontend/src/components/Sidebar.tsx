"use client";

import { useEffect, useState } from "react";
import {
  Search,
  Code2,
  PenTool,
  Settings,
  Network,
  Activity,
  Plus,
  RefreshCw,
  Info,
  Clock,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";

interface IndexStatus {
  status: string;
  indexed_files: number;
  total_files: number;
  current_file: string;
}

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  onAddContext: () => void;
  onReindex: () => void;
  showHistory: boolean;
  onToggleHistory: () => void;
  historyCount: number;
  onClearHistory: () => void;
}

const TABS = [
  { id: "search", label: "Search", icon: Search, shortcut: "⌘1" },
  { id: "analyze", label: "Analyze", icon: Code2, shortcut: "⌘2" },
  { id: "modify", label: "Modify", icon: PenTool, shortcut: "⌘3" },
  { id: "workspace", label: "Workspace", icon: Settings, shortcut: "⌘4" },
  { id: "lsp", label: "LSP", icon: Network, shortcut: "⌘5" },
];

export function Sidebar({
  activeTab,
  onTabChange,
  onAddContext,
  onReindex,
  showHistory,
  onToggleHistory,
  historyCount,
  onClearHistory,
}: SidebarProps) {
  const [status, setStatus] = useState<IndexStatus | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/tools/status");
        if (res.ok && !cancelled) {
          const data = await res.json();
          setStatus(data);
        }
      } catch {
        // Silent
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="w-72 glass border-r flex flex-col h-full shrink-0">
      <div className="p-5 border-b border-white/5">
        <div className="flex items-center gap-3 mb-1">
          <div className="p-2 bg-cta/20 rounded-lg cta-glow">
            <Code2 className="w-5 h-5 text-cta" />
          </div>
          <h1 className="text-lg font-bold tracking-tight">VectorMCP</h1>
        </div>
        <p className="text-xs text-foreground/50">Code Intelligence Dashboard</p>
      </div>

      <nav className="p-3 space-y-1">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all cursor-pointer group",
                activeTab === tab.id
                  ? "bg-cta/15 text-cta border border-cta/20"
                  : "text-foreground/60 hover:text-foreground hover:bg-white/5 border border-transparent"
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span className="flex-1 text-left">{tab.label}</span>
              <span className="text-[10px] font-mono text-foreground/20 group-hover:text-foreground/40 transition-colors">
                {tab.shortcut}
              </span>
            </button>
          );
        })}
      </nav>

      <div className="flex-1 overflow-y-auto p-3 space-y-5">
        <section>
          <button
            onClick={onToggleHistory}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all cursor-pointer",
              showHistory
                ? "bg-cta/15 text-cta border border-cta/20"
                : "text-foreground/60 hover:text-foreground hover:bg-white/5 border border-transparent"
            )}
          >
            <Clock className="w-4 h-4 shrink-0" />
            <span className="flex-1 text-left">History</span>
            {historyCount > 0 && (
              <span className="px-1.5 py-0.5 rounded-full bg-cta/20 text-[10px] font-bold text-cta">
                {historyCount}
              </span>
            )}
          </button>
        </section>

        <section>
          <h2 className="text-[10px] font-bold text-foreground/40 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
            <Activity className="w-3 h-3" /> Index Status
          </h2>
          <div className="glass-light p-3 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-foreground/70">Status</span>
              <span
                className={cn(
                  "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
                  status?.status === "Ready"
                    ? "bg-cta/20 text-cta"
                    : "bg-blue-500/20 text-blue-400"
                )}
              >
                {status?.status || "Idle"}
              </span>
            </div>
            {status && status.total_files > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] font-mono text-foreground/40">
                  <span>{status.indexed_files}/{status.total_files}</span>
                  <span>{Math.round((status.indexed_files / status.total_files) * 100)}%</span>
                </div>
                <div className="h-1 w-full bg-primary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cta transition-all duration-700 ease-out"
                    style={{ width: `${(status.indexed_files / status.total_files) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </section>

        <section>
          <h2 className="text-[10px] font-bold text-foreground/40 uppercase tracking-[0.2em] mb-3">
            Quick Actions
          </h2>
          <div className="space-y-1">
            <button
              onClick={onAddContext}
              className="w-full flex items-center gap-3 p-2.5 rounded-xl hover:bg-white/5 transition-all text-sm group cursor-pointer"
            >
              <div className="p-1.5 bg-primary rounded-lg group-hover:bg-secondary transition-all">
                <Plus className="w-3.5 h-3.5 text-cta" />
              </div>
              <span className="text-foreground/70 group-hover:text-foreground">Add Context</span>
            </button>
            <button
              onClick={onReindex}
              className="w-full flex items-center gap-3 p-2.5 rounded-xl hover:bg-white/5 transition-all text-sm group cursor-pointer"
            >
              <div className="p-1.5 bg-primary rounded-lg group-hover:bg-secondary transition-all">
                <RefreshCw
                  className={cn(
                    "w-3.5 h-3.5",
                    status?.status !== "Ready" ? "animate-spin text-cta" : "text-foreground/50"
                  )}
                />
              </div>
              <span className="text-foreground/70 group-hover:text-foreground">Re-index</span>
            </button>
            {historyCount > 0 && (
              <button
                onClick={onClearHistory}
                className="w-full flex items-center gap-3 p-2.5 rounded-xl hover:bg-white/5 transition-all text-sm group cursor-pointer"
              >
                <div className="p-1.5 bg-primary rounded-lg group-hover:bg-secondary transition-all">
                  <Trash2 className="w-3.5 h-3.5 text-foreground/50" />
                </div>
                <span className="text-foreground/70 group-hover:text-foreground">Clear History</span>
              </button>
            )}
          </div>
        </section>
      </div>

      <div className="p-3 border-t border-white/5 glass">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[10px] text-foreground/30 font-mono">
            <Info className="w-3 h-3" />
            <span>v0.1.0</span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
