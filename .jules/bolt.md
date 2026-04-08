## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.

## 2026-04-08 - [Avoid intermediate heap allocations when collecting Map/Set keys]
**Learning:** `serde_json::to_string()` accepts and serializes arrays of string references `Vec<&String>` just as easily as arrays of owned strings `Vec<String>`. By using `.collect()` directly on `unique.keys()` or `set.iter()` instead of chaining `.cloned()`, we skip allocating memory entirely for a new set of Strings when generating JSON arrays or extracting sorted key lists.
**Action:** When extracting lists of keys from HashMaps or elements from HashSets for sorting, referencing, or serializing in a vector, use references (`Vec<&String>`) and `sort_unstable()` instead of deep cloning `.cloned().collect()` and `sort()`.
