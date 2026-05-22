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
    highlighterPromise = import("shiki")
      .then(({ createHighlighter }) =>
        createHighlighter({
          themes: ["github-dark"],
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
      )
      .catch((err) => {
        highlighterPromise = null;
        throw err;
      });
  }
  return highlighterPromise;
}

// ---------------------------------------------------------------------------

interface CodeBlockProps {
  children: string;
  className?: string;
}

export function CodeBlock({ children, className }: CodeBlockProps) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const lang = detectLanguage(className);

  useEffect(() => {
    let cancelled = false;

    getHighlighter()
      .then(async (highlighter) => {
        try {
          const resolvedLang = LANG_MAP[lang] ?? lang;
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
    // No overflow-hidden here — the parent card controls clipping.
    // rounded-xl + border give visual containment without hiding content.
    <div className="relative group/code my-2 rounded-xl border border-white/8" style={{ background: "#0d1117" }}>
      {/* Copy button — shown at low opacity, full opacity on hover */}
      <button
        onClick={handleCopy}
        aria-label="Copy code"
        className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-white/5 opacity-40 group-hover/code:opacity-100 transition-opacity cursor-pointer hover:bg-white/15"
      >
        {copied ? (
          <Check className="w-3.5 h-3.5 text-cta" />
        ) : (
          <Copy className="w-3.5 h-3.5 text-foreground/60" />
        )}
      </button>

      {/* Loading skeleton */}
      {html === null && (
        <div
          className="p-4 font-mono text-xs animate-pulse overflow-x-auto"
          style={{ minHeight: "3rem" }}
          aria-busy="true"
          aria-label="Loading syntax highlighting"
        >
          <div className="h-3 bg-white/10 rounded w-3/4 mb-2" />
          <div className="h-3 bg-white/10 rounded w-1/2" />
        </div>
      )}

      {/* Shiki highlighted output */}
      {html !== null && html !== "" && (
        <div
          className="overflow-x-auto text-xs [&>pre]:p-4 [&>pre]:m-0 [&>pre]:rounded-xl [&>pre]:bg-transparent [&>pre]:leading-relaxed"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}

      {/* Plain-text fallback when Shiki fails */}
      {html === "" && (
        <pre className="p-4 font-mono text-xs text-foreground/80 whitespace-pre overflow-x-auto leading-relaxed">
          {children}
        </pre>
      )}
    </div>
  );
}
