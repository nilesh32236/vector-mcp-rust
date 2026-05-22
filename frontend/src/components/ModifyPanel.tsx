"use client";

import { useState } from "react";
import { PenTool, Play, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { mcpCall, addHistory } from "@/lib/api";
import { ResultCopyButton } from "./ResultCopyButton";

type ModifyAction = "create_file" | "apply_patch" | "verify_patch" | "run_linter";

interface ActionConfig {
  value: ModifyAction;
  label: string;
}

const actions: ActionConfig[] = [
  { value: "create_file", label: "Create File" },
  { value: "apply_patch", label: "Apply Patch" },
  { value: "verify_patch", label: "Verify Patch" },
  { value: "run_linter", label: "Run Linter" },
];

const linterTools = ["rustfmt", "prettier"] as const;

const inputClass =
  "w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm";

const textareaClass =
  "w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm resize-none min-h-[120px]";

interface ModifyPanelProps {
  onResult?: () => void;
}

export function ModifyPanel({ onResult }: ModifyPanelProps) {
  const [action, setAction] = useState<ModifyAction>("create_file");
  const [path, setPath] = useState("");
  const [content, setContent] = useState("");
  const [search, setSearch] = useState("");
  const [replace, setReplace] = useState("");
  const [linterTool, setLinterTool] = useState<string>("prettier");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const args: Record<string, unknown> = { action };

      switch (action) {
        case "create_file":
          args.path = path;
          args.content = content;
          break;
        case "apply_patch":
          args.search = search;
          args.replace = replace;
          args.path = path;
          break;
        case "verify_patch":
          args.search = search;
          args.replace = replace;
          args.path = path;
          break;
        case "run_linter":
          args.tool = linterTool;
          args.path = path;
          break;
      }

      const text = await mcpCall("modify_workspace", args);
      setResult(text);
      addHistory({ tool: "modify_workspace", action, label: action, args, result: text });
      onResult?.();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Modification failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const renderForm = () => {
    switch (action) {
      case "create_file":
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Path</label>
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g. src/components/NewFile.tsx"
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Content</label>
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                rows={10}
                placeholder="File content..."
                className={`${textareaClass} min-h-[240px]`}
              />
            </div>
          </div>
        );
      case "apply_patch":
      case "verify_patch":
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Search (exact text to find)</label>
              <textarea
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Text to search for..."
                className={textareaClass}
              />
            </div>
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Replace (new text)</label>
              <textarea
                value={replace}
                onChange={(e) => setReplace(e.target.value)}
                placeholder="Replacement text..."
                className={textareaClass}
              />
            </div>
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">File Path</label>
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g. src/components/SomeFile.tsx"
                className={inputClass}
              />
            </div>
          </div>
        );
      case "run_linter":
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Tool</label>
              <select
                value={linterTool}
                onChange={(e) => setLinterTool(e.target.value)}
                className={`${inputClass} appearance-none cursor-pointer`}
              >
                {linterTools.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">File Path</label>
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g. src/components/SomeFile.tsx"
                className={inputClass}
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 space-y-5">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-cta/20 rounded-lg">
            <PenTool className="w-5 h-5 text-cta" />
          </div>
          <h2 className="text-lg font-bold tracking-tight">Modify Workspace</h2>
        </div>

        <div>
          <label className="block text-xs text-foreground/50 font-medium mb-1.5">Action</label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value as ModifyAction)}
            className={`${inputClass} appearance-none cursor-pointer`}
          >
            {actions.map((a) => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </div>

        {renderForm()}

        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-cta text-black font-bold rounded-xl py-2.5 px-6 hover:brightness-110 transition-all flex items-center gap-2 disabled:opacity-40"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {loading && (
        <div className="glass rounded-2xl p-6 space-y-3">
          <div className="h-4 bg-white/5 rounded animate-pulse w-3/4" />
          <div className="h-4 bg-white/5 rounded animate-pulse w-1/2" />
          <div className="h-4 bg-white/5 rounded animate-pulse w-5/6" />
        </div>
      )}

      {error && !loading && (
        <div className="glass rounded-2xl p-4 border border-red-500/30 bg-red-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-red-300 font-mono whitespace-pre-wrap mb-3">{error}</p>
              <button
                onClick={handleRun}
                className="text-xs font-medium text-red-400 hover:text-red-300 transition-colors flex items-center gap-1.5"
              >
                <Play className="w-3 h-3" />
                Retry
              </button>
            </div>
          </div>
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
          <PenTool className="w-8 h-8 text-cta/30 mx-auto mb-3" />
          <p className="text-sm text-foreground/40">
            Select an action and fill in the required fields.
          </p>
        </div>
      )}
    </div>
  );
}
