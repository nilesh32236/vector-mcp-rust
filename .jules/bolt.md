## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.

## 2026-04-08 - [Avoid intermediate heap allocations when collecting Map/Set keys]
**Learning:** `serde_json::to_string()` accepts and serializes arrays of string references `Vec<&String>` just as easily as arrays of owned strings `Vec<String>`. By using `.collect()` directly on `unique.keys()` or `set.iter()` instead of chaining `.cloned()`, we skip allocating memory entirely for a new set of Strings when generating JSON arrays or extracting sorted key lists.
**Action:** Prefer collecting `Vec<&String>` and `sort_unstable()` only for same-scope, short-lived operations (e.g., immediate serialization with `serde_json::to_string()`, as with `unique.keys()` or `set.iter()`), but retain `.cloned().collect()` and `sort()` (or `sort_unstable()`) when the keys must be moved, returned, or sent across threads — borrowed references cannot outlive the backing collection.

## 2026-04-17 - [N+1 IO bottlenecks in IndexWriter deletions]
**Learning:** Iterating over a collection of documents and calling `delete_term()` followed by `commit()` on a Tantivy `IndexWriter` inside the loop creates an N+1 IO bottleneck. Each `commit()` forces disk synchronization, which is severely slow for bulk operations.
**Action:** Batch document deletions by exposing a `remove_no_commit()` method that only deletes the term. Call this method in the loop, and execute a single `commit()` on the writer after the loop finishes. Be careful not to hold a mutable reference (`let mut writer`) if the underlying `RwLockWriteGuard` allows mutation through `DerefMut` without it, to avoid unused mut warnings.
