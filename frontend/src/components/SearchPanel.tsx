"use client";

import { useState, useCallback } from "react";
import { Search, FileSearch, GitBranch, Activity, Play, Loader2, AlertCircle, Sparkles, Terminal } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { mcpCall, addHistory } from "@/lib/api";
import { ResultCopyButton } from "./ResultCopyButton";

type SearchMode = "vector" | "regex" | "graph" | "index_status";

interface SearchPanelProps {
  onResult?: () => void;
}

export function SearchPanel({ onResult }: SearchPanelProps) {
  const [mode, setMode] = useState<SearchMode>("vector");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [pathFilter, setPathFilter] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const modes: { value: SearchMode; label: string; icon: typeof Search }[] = [
    { value: "vector", label: "Vector Search", icon: Search },
    { value: "regex", label: "Regex Search", icon: FileSearch },
    { value: "graph", label: "Graph Query", icon: GitBranch },
    { value: "index_status", label: "Index Status", icon: Activity },
  ];

  const handleSearch = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setResult(null);

    try {
      const args: Record<string, unknown> = { action: mode };

      if (mode === "vector") {
        args.query = query;
        args.limit = limit;
      } else if (mode === "regex") {
        args.query = query;
        if (pathFilter) args.path = pathFilter;
      } else if (mode === "graph") {
        args.query = query;
      }

      const text = await mcpCall("search_workspace", args);
      setResult(text);
      setHasSearched(true);
      addHistory({ tool: "search_workspace", action: mode, label: query, args: args, result: text });
      onResult?.();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Search failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, [mode, query, limit, pathFilter]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const suggestions = [
    "API authentication",
    "SSE implementation",
    "Vector storage",
    "Database schema",
  ];

  const canSearch = mode === "index_status" || query.trim().length > 0;

  return (
    <div className="glass rounded-2xl p-6 space-y-5">
      <div className="flex items-center gap-2">
        <div className="p-2 bg-cta/20 rounded-lg">
          <Search className="w-5 h-5 text-cta" />
        </div>
        <h2 className="text-lg font-bold tracking-tight">Search Workspace</h2>
      </div>

      <div className="flex gap-1 bg-primary/30 rounded-xl p-1 border border-white/5">
        {modes.map((m) => {
          const Icon = m.icon;
          const isActive = mode === m.value;
          return (
            <button
              key={m.value}
              onClick={() => setMode(m.value)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer",
                isActive
                  ? "bg-cta/20 text-cta shadow-sm"
                  : "text-foreground/50 hover:text-foreground/80"
              )}
            >
              <Icon className="w-4 h-4" />
              {m.label}
            </button>
          );
        })}
      </div>

      <div className="space-y-4">
        {mode === "vector" && (
          <>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search your codebase..."
              className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-foreground/50 font-medium">Limit:</label>
              <input
                type="number"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                min={1}
                max={100}
                className="w-20 bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
              />
            </div>
          </>
        )}

        {mode === "regex" && (
          <>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Regex pattern..."
              className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
            />
            <input
              type="text"
              value={pathFilter}
              onChange={(e) => setPathFilter(e.target.value)}
              placeholder="Path filter (optional)"
              className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
            />
          </>
        )}

        {mode === "graph" && (
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Symbol name (e.g. handleSearch, Config struct)"
            className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
          />
        )}

        {mode === "index_status" && (
          <p className="text-sm text-foreground/40">
            Click Run to check the current indexing status of your workspace.
          </p>
        )}

        <button
          onClick={handleSearch}
          disabled={loading || !canSearch}
          className="bg-cta text-black font-bold rounded-xl py-2.5 px-6 hover:brightness-110 transition-all flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? "Searching..." : "Run"}
        </button>
      </div>

      {loading && (
        <div className="bg-[#0d1117] rounded-xl p-5 overflow-x-auto space-y-3 animate-pulse">
          <div className="h-3 bg-white/8 rounded w-3/4" />
          <div className="h-3 bg-white/8 rounded w-1/2" />
          <div className="h-3 bg-white/8 rounded w-2/3" />
          <div className="h-3 bg-white/8 rounded w-1/2" />
          <div className="h-3 bg-white/8 rounded w-3/4" />
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-red-400 mb-1">Error</p>
              <p className="text-xs text-red-300/70 whitespace-pre-wrap break-all">{error}</p>
              <button
                onClick={handleSearch}
                className="mt-2 text-xs text-red-400 hover:text-red-300 underline underline-offset-2"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {!loading && result && (
        <div className="bg-[#0d1117] rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
            <span className="text-[11px] font-medium text-foreground/40">Result</span>
            <ResultCopyButton content={result} />
          </div>
          <div className="p-5 overflow-x-auto">
            <MarkdownRenderer content={result} />
          </div>
        </div>
      )}

      {!loading && !result && hasSearched && !error && (
        <div className="glass rounded-2xl p-12 text-center">
          <div className="w-16 h-16 bg-primary/40 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-white/10">
            <Terminal className="w-8 h-8 text-foreground/20" />
          </div>
          <h3 className="text-lg font-bold text-foreground/60">No results found</h3>
          <p className="text-sm text-foreground/30 mt-2 max-w-xs mx-auto">
            Try a different search term or ensure the repository is indexed.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => {
                  setQuery(s);
                  setMode("vector");
                }}
                className="px-4 py-2 rounded-full bg-white/5 border border-white/5 text-xs text-foreground/40 hover:border-cta/30 hover:text-cta transition-all cursor-pointer"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {!loading && !result && !hasSearched && (
        <div className="glass rounded-2xl p-8 text-center">
          <Sparkles className="w-8 h-8 text-cta/30 mx-auto mb-3" />
          <p className="text-sm text-foreground/40">
            Select a search mode and enter your query above to search the workspace.
          </p>
        </div>
      )}
    </div>
  );
}
