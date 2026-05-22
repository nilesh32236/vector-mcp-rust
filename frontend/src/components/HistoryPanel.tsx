"use client";

import { useState, useEffect } from "react";
import { Clock, Search, Code2, PenTool, Settings, Network, Trash2, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { ResultCopyButton } from "./ResultCopyButton";
import { getHistory, clearHistory, HistoryEntry } from "@/lib/api";

const TOOL_ICONS: Record<string, typeof Search> = {
  search_workspace: Search,
  analyze_code: Code2,
  modify_workspace: PenTool,
  workspace_manager: Settings,
  lsp_query: Network,
};

function formatTime(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 60000) return "Just now";
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function HistoryPanel() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  const refresh = () => setEntries(getHistory());

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleClear = () => {
    clearHistory();
    setEntries([]);
    setExpanded(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-foreground/90 flex items-center gap-2">
          <Clock className="w-4 h-4 text-cta" />
          History
        </h2>
        {entries.length > 0 && (
          <button
            onClick={handleClear}
            className="flex items-center gap-1.5 text-xs text-foreground/40 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
            Clear all
          </button>
        )}
      </div>

      {entries.length === 0 && (
        <div className="text-center py-12">
          <Clock className="w-8 h-8 mx-auto text-foreground/20 mb-3" />
          <p className="text-xs text-foreground/40">No history yet</p>
          <p className="text-[10px] text-foreground/30 mt-1">Run a tool to see results here</p>
        </div>
      )}

      <div className="space-y-2">
        {entries.map((entry) => {
          const Icon = TOOL_ICONS[entry.tool] ?? Search;
          const isExpanded = expanded === entry.id;

          return (
            <div
              key={entry.id}
              className="bg-[#0d1117] rounded-xl overflow-hidden border border-white/5"
            >
              <button
                onClick={() => setExpanded(isExpanded ? null : entry.id)}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/[0.02] transition-colors cursor-pointer"
              >
                <Icon className="w-3.5 h-3.5 text-cta/60 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-foreground/70 truncate">
                      {entry.label || entry.action}
                    </span>
                    <span className="text-[10px] text-foreground/30 px-1.5 py-0.5 rounded bg-white/5">
                      {entry.action}
                    </span>
                  </div>
                  <p className="text-[10px] text-foreground/30 mt-0.5">
                    {formatTime(entry.timestamp)}
                  </p>
                </div>
                {isExpanded ? (
                  <ChevronDown className="w-3 h-3 text-foreground/30 shrink-0" />
                ) : (
                  <ChevronRight className="w-3 h-3 text-foreground/30 shrink-0" />
                )}
              </button>

              {isExpanded && (
                <div className="px-4 pb-3">
                  <div className="flex items-center justify-end mb-2">
                    <ResultCopyButton content={entry.result} label="Copy" />
                  </div>
                  <MarkdownRenderer content={entry.result} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
