"use client";

import ReactMarkdown from "react-markdown";
import { CodeBlock } from "./CodeBlock";

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="max-w-none space-y-2">
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
          h1({ children }) {
            return (
              <h1 className="text-base font-bold text-foreground/90 mt-6 mb-3 first:mt-0">
                {children}
              </h1>
            );
          },
          h2({ children }) {
            return (
              <h2 className="text-sm font-bold text-cta mt-5 mb-2 first:mt-0 uppercase tracking-wider">
                {children}
              </h2>
            );
          },
          h3({ children }) {
            return (
              <h3 className="text-sm font-semibold text-foreground/80 mt-4 mb-2 first:mt-0">
                {children}
              </h3>
            );
          },
          h4({ children }) {
            return (
              <h4 className="text-xs font-mono font-semibold text-foreground/60 mt-3 mb-1 border-b border-white/5 pb-1">
                {children}
              </h4>
            );
          },
          p({ children }) {
            return (
              <p className="text-xs leading-relaxed text-foreground/60 mb-2 last:mb-0">
                {children}
              </p>
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
          ul({ children }) {
            return <ul className="space-y-1 my-2 ml-2">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="space-y-1 my-2 ml-2 list-decimal">{children}</ol>;
          },
          li({ children }) {
            return (
              <li className="text-xs text-foreground/60 flex items-start gap-2">
                <span className="text-cta/40 mt-1 shrink-0">•</span>
                <span className="min-w-0">{children}</span>
              </li>
            );
          },
          strong({ children }) {
            return (
              <strong className="text-foreground/80 font-semibold">
                {children}
              </strong>
            );
          },
          em({ children }) {
            return <em className="text-foreground/50 italic">{children}</em>;
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cta hover:underline underline-offset-2"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
