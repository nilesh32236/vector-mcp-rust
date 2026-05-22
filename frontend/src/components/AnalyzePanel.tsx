"use client";

import { useState } from "react";
import { toast } from "sonner";
import {
  Code2,
  Play,
  Loader2,
  AlertCircle,
  ListTree,
  FileType,
  Braces,
  GitBranch,
  ScanSearch,
  Package,
  BookOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { mcpCall, addHistory } from "@/lib/api";
import { ResultCopyButton } from "./ResultCopyButton";

type AnalyzeAction =
  | "architecture"
  | "api_list"
  | "dead_code"
  | "docstring"
  | "ast_skeleton"
  | "duplicate_code"
  | "dependencies"
  | "distill_package";

interface AnalyzeOption {
  value: AnalyzeAction;
  label: string;
  icon: React.ReactNode;
  inputs: { key: string; label: string; type: "text" | "checkbox"; placeholder?: string }[];
}

const ANALYZE_OPTIONS: AnalyzeOption[] = [
  { value: "architecture", label: "Architecture Graph", icon: <GitBranch className="w-4 h-4" />, inputs: [{ key: "monorepo_prefix", label: "Monorepo Prefix", type: "text", placeholder: "e.g. vector-mcp-rust" }] },
  { value: "api_list", label: "API List", icon: <ListTree className="w-4 h-4" />, inputs: [] },
  { value: "dead_code", label: "Dead Code", icon: <ScanSearch className="w-4 h-4" />, inputs: [{ key: "is_library", label: "Is Library", type: "checkbox" }] },
  { value: "docstring", label: "Docstring", icon: <BookOpen className="w-4 h-4" />, inputs: [
    { key: "entity_name", label: "Entity Name", type: "text", placeholder: "e.g. handle_ping" },
    { key: "file_path", label: "File Path", type: "text", placeholder: "e.g. /path/to/file.rs" },
  ]},
  { value: "ast_skeleton", label: "AST Skeleton", icon: <FileType className="w-4 h-4" />, inputs: [{ key: "path", label: "Path", type: "text", placeholder: "e.g. src/" }] },
  { value: "duplicate_code", label: "Duplicate Code", icon: <Braces className="w-4 h-4" />, inputs: [] },
  { value: "dependencies", label: "Dependencies", icon: <Package className="w-4 h-4" />, inputs: [] },
  { value: "distill_package", label: "Distill Package", icon: <Code2 className="w-4 h-4" />, inputs: [{ key: "path", label: "Path", type: "text", placeholder: "e.g. crates/mcp-core" }] },
];

interface AnalyzePanelProps {
  onResult?: () => void;
}

export function AnalyzePanel({ onResult }: AnalyzePanelProps) {
  const [action, setAction] = useState<AnalyzeAction>("api_list");
  const [inputs, setInputs] = useState<Record<string, string | boolean>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const currentOption = ANALYZE_OPTIONS.find((o) => o.value === action)!;

  const handleRun = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const args: Record<string, unknown> = { action };
      for (const input of currentOption.inputs) {
        const val = inputs[input.key];
        if (val !== undefined && val !== "") {
          args[input.key] = val;
        }
      }
      const text = await mcpCall("analyze_code", args);
      setResult(text);
      addHistory({ tool: "analyze_code", action, label: action, args, result: text });
      onResult?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-cta/20 rounded-lg">
          <Code2 className="w-5 h-5 text-cta" />
        </div>
        <div>
          <h2 className="text-lg font-bold">Analyze Code</h2>
          <p className="text-sm text-foreground/40">Run codebase analysis tools</p>
        </div>
      </div>

      <div className="glass rounded-2xl p-6 space-y-5">
        <select
          value={action}
          onChange={(e) => {
            setAction(e.target.value as AnalyzeAction);
            setResult(null);
            setError(null);
          }}
          className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all text-sm"
        >
          {ANALYZE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        {currentOption.inputs.length > 0 && (
          <div className="space-y-3">
            {currentOption.inputs.map((input) => (
              <div key={input.key} className="space-y-1.5">
                <label className="text-[10px] font-bold text-foreground/60 uppercase tracking-[0.2em]">
                  {input.label}
                </label>
                {input.type === "checkbox" ? (
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={!!inputs[input.key]}
                      onChange={(e) => setInputs((prev) => ({ ...prev, [input.key]: e.target.checked }))}
                      className="w-4 h-4 rounded border-white/20 bg-primary/40 accent-cta"
                    />
                    <span className="text-sm text-foreground/60">{input.label}</span>
                  </label>
                ) : (
                  <input
                    type="text"
                    value={(inputs[input.key] as string) ?? ""}
                    onChange={(e) => setInputs((prev) => ({ ...prev, [input.key]: e.target.value }))}
                    placeholder={input.placeholder}
                    className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm"
                  />
                )}
              </div>
            ))}
          </div>
        )}

        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-cta text-black font-bold rounded-xl py-2.5 px-6 hover:brightness-110 transition-all flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? "Running..." : `Run ${currentOption.label}`}
        </button>
      </div>

      {loading && (
        <div className="bg-[#0d1117] p-5 overflow-x-auto space-y-3 animate-pulse">
          <div className="h-3 bg-white/8 rounded w-3/4" />
          <div className="h-3 bg-white/8 rounded w-1/2" />
          <div className="h-3 bg-white/8 rounded w-2/3" />
          <div className="h-3 bg-white/8 rounded w-1/2" />
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
                onClick={handleRun}
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

      {!loading && !result && !error && (
        <div className="text-center py-16">
          <Code2 className="w-12 h-12 text-foreground/10 mx-auto mb-4" />
          <p className="text-sm text-foreground/30">Select an analysis tool and click Run</p>
        </div>
      )}
    </div>
  );
}
