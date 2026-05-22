"use client";

import { useState } from "react";
import { Network, Play, Loader2, AlertCircle } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { mcpCall, addHistory } from "@/lib/api";
import { ResultCopyButton } from "./ResultCopyButton";

type LspAction = "definition" | "references" | "type_hierarchy" | "impact_analysis";

const actions: { value: LspAction; label: string }[] = [
  { value: "definition", label: "Definition" },
  { value: "references", label: "References" },
  { value: "type_hierarchy", label: "Type Hierarchy" },
  { value: "impact_analysis", label: "Impact Analysis" },
];

interface LspPanelProps {
  onResult?: () => void;
}

export function LspPanel({ onResult }: LspPanelProps) {
  const [action, setAction] = useState<LspAction>("definition");
  const [filePath, setFilePath] = useState("");
  const [line, setLine] = useState(0);
  const [character, setCharacter] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const args: Record<string, unknown> = {
        action,
        path: filePath,
        line,
        character,
      };

      const data = await mcpCall("lsp_query", args);
      setResult(data);
      addHistory({ tool: "lsp_query", action, label: action, args, result: data });
      onResult?.();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "LSP query failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm";

  return (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 space-y-5">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-cta/20 rounded-lg">
            <Network className="w-5 h-5 text-cta" />
          </div>
          <h2 className="text-lg font-bold tracking-tight">LSP Query</h2>
        </div>

        <div>
          <label className="block text-xs text-foreground/50 font-medium mb-1.5">Action</label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value as LspAction)}
            className={inputClass + " appearance-none cursor-pointer"}
          >
            {actions.map((a) => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-foreground/50 font-medium mb-1.5">File Path</label>
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            className={inputClass}
            placeholder="e.g. /home/user/project/src/main.rs"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-foreground/50 font-medium mb-1.5">Line (0-indexed)</label>
            <input
              type="number"
              value={line}
              onChange={(e) => setLine(Number(e.target.value))}
              min={0}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-foreground/50 font-medium mb-1.5">Character (0-indexed)</label>
            <input
              type="number"
              value={character}
              onChange={(e) => setCharacter(Number(e.target.value))}
              min={0}
              className={inputClass}
            />
          </div>
        </div>

        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-cta text-black font-bold rounded-xl py-2.5 px-6 hover:brightness-110 transition-all flex items-center gap-2 disabled:opacity-40"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {loading ? "Querying..." : "Run"}
        </button>
      </div>

      {loading && (
        <div className="bg-[#0d1117] rounded-xl p-5 overflow-x-auto space-y-3 animate-pulse">
          <div className="h-3 bg-white/8 rounded w-3/4" />
          <div className="h-3 bg-white/8 rounded w-1/2" />
          <div className="h-3 bg-white/8 rounded w-2/3" />
        </div>
      )}

      {error && !loading && (
        <div className="glass rounded-2xl p-4 border border-red-500/30 bg-red-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <p className="text-sm text-red-300 font-mono whitespace-pre-wrap">{error}</p>
          </div>
          <button
            onClick={handleRun}
            className="mt-3 bg-red-500/20 text-red-300 text-sm font-medium rounded-lg py-1.5 px-4 hover:bg-red-500/30 transition-all"
          >
            Retry
          </button>
        </div>
      )}

      {result && !loading && (
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

      {!loading && !result && !error && (
        <div className="glass rounded-2xl p-8 text-center">
          <Network className="w-8 h-8 text-cta/30 mx-auto mb-3" />
          <p className="text-sm text-foreground/40">Query LSP for definitions, references, type hierarchy, or impact analysis.</p>
        </div>
      )}
    </div>
  );
}
