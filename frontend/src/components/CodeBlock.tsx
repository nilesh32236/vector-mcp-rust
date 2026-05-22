"use client";

import { useEffect, useState } from "react";
import {
  Copy,
  Check,
  FileCode,
  FileJson,
  FileText,
  Braces,
  Terminal,
  File,
} from "lucide-react";

const LANG_META: Record<string, { label: string; icon: typeof FileCode }> = {
  rust: { label: "Rust", icon: FileCode },
  typescript: { label: "TypeScript", icon: FileCode },
  tsx: { label: "TSX", icon: FileCode },
  javascript: { label: "JavaScript", icon: Braces },
  jsx: { label: "JSX", icon: Braces },
  python: { label: "Python", icon: FileCode },
  go: { label: "Go", icon: FileCode },
  bash: { label: "Terminal", icon: Terminal },
  sh: { label: "Shell", icon: Terminal },
  shell: { label: "Shell", icon: Terminal },
  json: { label: "JSON", icon: FileJson },
  toml: { label: "TOML", icon: Braces },
  yaml: { label: "YAML", icon: Braces },
  yml: { label: "YAML", icon: Braces },
  markdown: { label: "Markdown", icon: FileText },
  md: { label: "Markdown", icon: FileText },
  dockerfile: { label: "Dockerfile", icon: Terminal },
  css: { label: "CSS", icon: FileCode },
  html: { label: "HTML", icon: FileCode },
  plaintext: { label: "Text", icon: File },
  text: { label: "Text", icon: File },
};

function detectLanguage(className?: string): string {
  const match = className?.match(/language-(\w+)/);
  return match?.[1] ?? "text";
}

const LANG_MAP: Record<string, string> = {
  js: "javascript",
  ts: "typescript",
  sh: "bash",
  shell: "bash",
  text: "plaintext",
  txt: "plaintext",
};

let highlighterPromise: Promise<import("shiki").Highlighter> | null = null;

function getHighlighter(): Promise<import("shiki").Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = import("shiki")
      .then(({ createHighlighter }) =>
        createHighlighter({
          themes: ["github-dark"],
          langs: [
            "rust", "typescript", "tsx", "javascript", "jsx",
            "python", "go", "bash", "json", "toml", "yaml",
            "markdown", "plaintext",
          ],
        })
      )
      .catch((err) => {
        highlighterPromise = null;
        throw err;
      });
  }
  return highlighterPromise;
}

interface CodeBlockProps {
  children: string;
  className?: string;
}

export function CodeBlock({ children, className }: CodeBlockProps) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const rawLang = detectLanguage(className);
  const resolvedLang = LANG_MAP[rawLang] ?? rawLang;
  const meta = LANG_META[resolvedLang] ?? LANG_META.plaintext!;
  const Icon = meta.icon;

  useEffect(() => {
    let cancelled = false;

    getHighlighter()
      .then(async (highlighter) => {
        try {
          const loadedLangs = highlighter.getLoadedLanguages();
          let fallbackLang = resolvedLang;
          if (!loadedLangs.includes(resolvedLang as never)) {
            try {
              await highlighter.loadLanguage(resolvedLang as never);
            } catch {
              fallbackLang = "plaintext";
            }
          }
          if (!cancelled) {
            setHtml(
              highlighter.codeToHtml(children, {
                lang: fallbackLang,
                theme: "github-dark",
              })
            );
          }
        } catch {
          if (!cancelled) setHtml("");
        }
      })
      .catch(() => {
        if (!cancelled) setHtml("");
      });

    return () => {
      cancelled = true;
    };
  }, [children, resolvedLang]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard permission denied
    }
  };

  const lineCount = children.split("\n").length;

  return (
    <div
      className="relative group/code my-2 rounded-xl border overflow-hidden"
      style={{ borderColor: "rgba(255,255,255,0.08)", background: "#0d1117" }}
    >
      {/* ── Window chrome / title bar ── */}
      <div
        className="flex items-center justify-between px-3 py-1.5 border-b select-none"
        style={{
          background: "rgba(255,255,255,0.03)",
          borderColor: "rgba(255,255,255,0.06)",
        }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: "rgba(255,255,255,0.35)" }} />
          <span className="text-[11px] font-medium tracking-wide truncate"
            style={{ color: "rgba(255,255,255,0.45)" }}>
            {meta.label}
          </span>
          {lineCount > 1 && (
            <span className="text-[10px] hidden sm:inline" style={{ color: "rgba(255,255,255,0.2)" }}>
              {lineCount} lines
            </span>
          )}
        </div>

        <button
          onClick={handleCopy}
          aria-label="Copy code"
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-all cursor-pointer"
          style={{
            color: copied ? "#2f9e44" : "rgba(255,255,255,0.45)",
            background: copied
              ? "rgba(47,158,68,0.1)"
              : "rgba(255,255,255,0.04)",
          }}
          onMouseEnter={(e) => {
            if (!copied)
              e.currentTarget.style.background = "rgba(255,255,255,0.08)";
          }}
          onMouseLeave={(e) => {
            if (!copied)
              e.currentTarget.style.background = "rgba(255,255,255,0.04)";
          }}
        >
          {copied ? (
            <Check className="w-3 h-3" />
          ) : (
            <Copy className="w-3 h-3" />
          )}
          <span className="hidden sm:inline">{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>

      {/* ── Loading skeleton ── */}
      {html === null && (
        <div
          className="p-4 font-mono text-xs animate-pulse overflow-x-auto"
          style={{ minHeight: "3rem" }}
          aria-busy="true"
          aria-label="Loading syntax highlighting"
        >
          <div className="h-3 bg-white/8 rounded w-3/4 mb-2" />
          <div className="h-3 bg-white/8 rounded w-1/2 mb-2" />
          <div className="h-3 bg-white/8 rounded w-2/3" />
        </div>
      )}

      {/* ── Shiki highlighted output ── */}
      {html !== null && html !== "" && (
        <div
          className="overflow-x-auto text-xs [&>pre]:p-4 [&>pre]:m-0 [&>pre]:leading-relaxed [&>pre]:bg-transparent"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}

      {/* ── Plain-text fallback ── */}
      {html === "" && (
        <pre className="p-4 font-mono text-xs whitespace-pre overflow-x-auto leading-relaxed"
          style={{ color: "rgba(255,255,255,0.7)" }}>
          {children}
        </pre>
      )}
    </div>
  );
}
