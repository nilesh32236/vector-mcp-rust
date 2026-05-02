"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  FileCode,
  ExternalLink,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Hash,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { CodeBlock } from "./CodeBlock";

interface SearchResult {
  id: string;
  text: string;
  similarity: number;
  path: string;
  summary?: string;
  start_line?: number;
  end_line?: number;
}

interface SearchResultCardProps {
  result: SearchResult;
}

/** Map a file extension to a Shiki language id. */
function extToLang(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    rs: "rust",
    ts: "typescript",
    tsx: "tsx",
    js: "javascript",
    jsx: "jsx",
    go: "go",
    py: "python",
    md: "markdown",
    toml: "toml",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    sh: "bash",
  };
  return map[ext] ?? "plaintext";
}

/**
 * Shorten an absolute path to a project-relative form.
 * e.g. /home/nilesh/Documents/vector-mcp-rust/src/mcp/sse.rs → src/mcp/sse.rs
 */
function shortenPath(path: string): string {
  if (!path) return path;
  const markers = ["/src/", "/frontend/", "/docs/", "/scripts/", "/design-system/"];
  for (const marker of markers) {
    const idx = path.indexOf(marker);
    if (idx !== -1) {
      return path.slice(idx + 1);
    }
  }
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.slice(-3).join("/");
}

/** Relevance score → human label + colour class */
function relevanceLabel(score: number): { label: string; cls: string } {
  if (score >= 0.85) return { label: "High", cls: "text-cta" };
  if (score >= 0.65) return { label: "Good", cls: "text-blue-400" };
  return { label: "Low", cls: "text-foreground/40" };
}

// Lines of code to show when collapsed
const COLLAPSED_LINES = 12;

export function SearchResultCard({ result }: SearchResultCardProps) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard permission denied
    }
  };

  const hasSummary = Boolean(result.summary?.trim());
  const hasLines =
    result.start_line != null &&
    result.end_line != null &&
    result.end_line > 0 &&
    result.start_line > 0;

  const shortPath = shortenPath(result.path);
  const lang = extToLang(result.path);
  const similarity = result.similarity ?? 0;
  const { label: relLabel, cls: relCls } = relevanceLabel(similarity);

  // Count lines to decide whether truncation is needed
  const lines = result.text.split("\n");
  const needsTruncation = lines.length > COLLAPSED_LINES;

  // When collapsed, show only the first N lines
  const displayText = needsTruncation && !isExpanded
    ? lines.slice(0, COLLAPSED_LINES).join("\n")
    : result.text;

  const markdownContent = `\`\`\`${lang}\n${displayText}\n\`\`\``;

  return (
    <div className="glass rounded-2xl group hover:border-cta/30 transition-all duration-300">
      {/* ── Header ── */}
      <div className="flex items-start justify-between px-5 pt-5 pb-4 gap-4">
        {/* File info */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="shrink-0 p-2 bg-primary rounded-lg border border-white/5 group-hover:bg-secondary transition-all">
            <FileCode className="w-4 h-4 text-foreground/60" />
          </div>
          <div className="min-w-0">
            <h3
              className="font-mono text-sm font-semibold text-foreground/90 truncate"
              title={result.path}
            >
              {shortPath || result.path || "Manual Context"}
            </h3>
            <div className="flex items-center gap-2 mt-0.5">
              {hasLines ? (
                <span className="flex items-center gap-1 text-[10px] font-mono text-foreground/35">
                  <Hash className="w-2.5 h-2.5" />
                  {result.start_line}–{result.end_line}
                </span>
              ) : (
                <span className="text-[10px] font-mono text-foreground/25">
                  {result.id.substring(0, 10)}…
                </span>
              )}
              <span className="text-[10px] font-mono text-foreground/20 uppercase tracking-wider">
                {lang}
              </span>
            </div>
          </div>
        </div>

        {/* Right side: relevance + actions */}
        <div className="flex items-center gap-4 shrink-0">
          {/* Relevance */}
          <div className="hidden sm:flex flex-col items-end gap-1">
            <div className="flex items-center gap-1.5">
              <span className={cn("text-[10px] font-bold uppercase tracking-widest", relCls)}>
                {relLabel}
              </span>
              <span className="text-[10px] font-mono text-foreground/30">
                {Math.round(similarity * 100)}%
              </span>
            </div>
            <div className="h-1 w-20 bg-primary rounded-full overflow-hidden">
              <div
                className="h-full bg-cta transition-all duration-700 ease-out"
                style={{ width: `${similarity * 100}%` }}
              />
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={handleCopy}
              className="p-1.5 hover:bg-white/8 rounded-lg transition-all cursor-pointer group/btn"
              title="Copy code"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-cta" />
              ) : (
                <Copy className="w-3.5 h-3.5 text-foreground/30 group-hover/btn:text-foreground/70" />
              )}
            </button>
            <button
              onClick={() => {
                if (!result.path) return;
                let url: string;
                if (result.path.startsWith("/")) {
                  url = `file://${result.path}`;
                } else {
                  try {
                    const parsed = new URL(result.path);
                    if (!["https:", "http:", "file:"].includes(parsed.protocol)) return;
                    url = parsed.href;
                  } catch {
                    return;
                  }
                }
                window.open(url, "_blank", "noopener,noreferrer");
              }}
              disabled={!result.path}
              className="p-1.5 hover:bg-white/8 rounded-lg transition-all cursor-pointer group/btn disabled:opacity-20 disabled:cursor-not-allowed"
              title="Open file"
            >
              <ExternalLink className="w-3.5 h-3.5 text-foreground/30 group-hover/btn:text-foreground/70" />
            </button>
          </div>
        </div>
      </div>

      {/* ── AI Summary ── */}
      {hasSummary && (
        <div className="mx-5 mb-3 flex items-start gap-2.5 px-3.5 py-2.5 bg-cta/5 border border-cta/15 rounded-xl">
          <Sparkles className="w-3 h-3 text-cta mt-0.5 shrink-0" />
          <p className="text-xs text-foreground/65 leading-relaxed italic">
            {result.summary}
          </p>
        </div>
      )}

      {/* ── Code content ── */}
      <div className="px-5 pb-1">
        <ReactMarkdown
          components={{
            code({ className, children, ...props }) {
              const isBlock =
                Boolean(className?.startsWith("language-")) ||
                String(children).includes("\n");
              if (isBlock) {
                return (
                  <CodeBlock className={className}>
                    {String(children).replace(/\n$/, "")}
                  </CodeBlock>
                );
              }
              return (
                <code
                  className="bg-white/10 px-1.5 py-0.5 rounded text-cta font-mono text-xs"
                  {...props}
                >
                  {children}
                </code>
              );
            },
            p({ children }) {
              return (
                <p className="mb-2 last:mb-0 text-xs leading-relaxed font-mono text-foreground/75 whitespace-pre-wrap">
                  {children}
                </p>
              );
            },
            pre({ children }) {
              return <>{children}</>;
            },
          }}
        >
          {markdownContent}
        </ReactMarkdown>
      </div>

      {/* ── Expand / Collapse ── */}
      {needsTruncation && (
        <div className="px-5 pb-4 pt-1 flex justify-center">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-foreground/40 hover:text-cta transition-all cursor-pointer px-4 py-2 rounded-full hover:bg-white/5 border border-white/5 hover:border-cta/20"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                Show full code ({lines.length} lines)
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
