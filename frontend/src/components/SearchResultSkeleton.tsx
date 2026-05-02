/**
 * Skeleton loader for SearchResultCard.
 * Mirrors the card layout with animate-pulse placeholders.
 * Respects prefers-reduced-motion via globals.css.
 */
export function SearchResultSkeleton() {
  return (
    <div
      className="glass p-6 rounded-2xl"
      aria-busy="true"
      aria-label="Loading search result"
    >
      {/* Header row */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          {/* Icon placeholder */}
          <div className="w-10 h-10 rounded-xl bg-white/5 animate-pulse" />
          <div className="space-y-2">
            {/* File path placeholder */}
            <div className="h-3 w-48 rounded bg-white/5 animate-pulse" />
            {/* Line range placeholder */}
            <div className="h-2 w-24 rounded bg-white/5 animate-pulse" />
          </div>
        </div>
        {/* Relevance bar placeholder */}
        <div className="hidden sm:flex flex-col items-end gap-1.5">
          <div className="h-2 w-16 rounded bg-white/5 animate-pulse" />
          <div className="h-1.5 w-24 rounded-full bg-white/5 animate-pulse" />
        </div>
      </div>

      {/* Summary placeholder (occasionally shown) */}
      <div className="mb-4 h-8 rounded-xl bg-white/5 animate-pulse" />

      {/* Code block placeholder */}
      <div className="h-32 rounded-xl bg-white/5 animate-pulse" />
    </div>
  );
}
