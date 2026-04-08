## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.
