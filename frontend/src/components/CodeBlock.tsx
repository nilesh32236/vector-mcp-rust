"use client";

import { useEffect, useState } from "react";
import { Copy, Check } from "lucide-react";

/** Extract language identifier from a react-markdown className like "language-rust" */
function detectLanguage(className?: string): string {
  const match = className?.match(/language-(\w+)/);
  return match?.[1] ?? "text";
}

// ---------------------------------------------------------------------------
// Module-level singleton — createHighlighter is called exactly once.
// ---------------------------------------------------------------------------

// Map common aliases to Shiki-supported language IDs.
const LANG_MAP: Record<string, string> = {
  js: "javascript",
  ts: "typescript",
  tsx: "tsx",
  jsx: "jsx",
  sh: "bash",
  shell: "bash",
  text: "plaintext",
  txt: "plaintext",
};

let highlighterPromise: Promise<import("shiki").Highlighter> | null = null;

function getHighlighter(): Promise<import("shiki").Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = import("shiki").then(({ createHighlighter }) =>
      createHighlighter({
        themes: ["github-dark"],
        // Pre-load the most common languages; others are loaded on demand.
        langs: [
          "rust",
          "typescript",
          "tsx",
          "javascript",
          "jsx",
          "python",
          "go",
          "bash",
          "json",
          "toml",
          "yaml",
          "markdown",
          "plaintext",
        ],
      })
    );
  }
  return highlighterPromise;
}

// ---------------------------------------------------------------------------

interface CodeBlockProps {
  children: string;
  className?: string;
}

export function CodeBlock({ children, className }: CodeBlockProps) {
  const [html, setHtml] = useState<string | null>(null); // null = loading, "" = error/fallback
  const [copied, setCopied] = useState(false);
  const lang = detectLanguage(className);

  useEffect(() => {
    let cancelled = false;

    getHighlighter()
      .then(async (highlighter) => {
        try {
          const resolvedLang = LANG_MAP[lang] ?? lang;

          // Load the language dynamically if it wasn't pre-loaded.
          const loadedLangs = highlighter.getLoadedLanguages();
          if (!loadedLangs.includes(resolvedLang as never)) {
            try {
              await highlighter.loadLanguage(resolvedLang as never);
            } catch {
              // Unknown language — fall back to plaintext.
            }
          }

          if (!cancelled) {
            setHtml(
              highlighter.codeToHtml(children, {
                lang: resolvedLang,
                theme: "github-dark",
              })
            );
          }
        } catch {
          if (!cancelled) setHtml(""); // signal fallback
        }
      })
      .catch(() => {
        if (!cancelled) setHtml(""); // signal fallback
      });

    return () => {
      cancelled = true;
    };
  }, [children, lang]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Silent — clipboard permission denied
    }
  };

  return (
    <div className="relative group/code my-3 rounded-xl overflow-hidden border border-white/5">
      {/* Copy button — visible on hover */}
      <button
        onClick={handleCopy}
        aria-label="Copy code"
        className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-white/5 opacity-0 group-hover/code:opacity-100 transition-opacity cursor-pointer hover:bg-white/10"
      >
        {copied ? (
          <Check className="w-3.5 h-3.5 text-cta" />
        ) : (
          <Copy className="w-3.5 h-3.5 text-foreground/50" />
        )}
      </button>

      {/* Loading skeleton while Shiki initialises */}
      {html === null && (
        <div
          className="p-4 font-mono text-xs animate-pulse bg-black/30 overflow-x-auto"
          style={{ minHeight: "3rem" }}
          aria-busy="true"
          aria-label="Loading syntax highlighting"
        >
          <div className="h-3 bg-white/10 rounded w-3/4 mb-2" />
          <div className="h-3 bg-white/10 rounded w-1/2" />
        </div>
      )}

      {/* Highlighted output */}
      {html !== null && html !== "" && (
        <div
          className="text-xs overflow-x-auto [&>pre]:p-4 [&>pre]:m-0 [&>pre]:rounded-none [&>pre]:bg-transparent"
          style={{ background: "#0d1117" }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}

      {/* Plain fallback — only shown after Shiki has finished (and failed) */}
      {html === "" && (
        <pre className="p-4 font-mono text-xs text-foreground/80 whitespace-pre-wrap bg-black/30 overflow-x-auto">
          {children}
        </pre>
      )}
    </div>
  );
}
