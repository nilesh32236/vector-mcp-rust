## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.

## 2026-04-08 - [Avoid intermediate heap allocations when collecting Map/Set keys]
**Learning:** `serde_json::to_string()` accepts and serializes arrays of string references `Vec<&String>` just as easily as arrays of owned strings `Vec<String>`. By using `.collect()` directly on `unique.keys()` or `set.iter()` instead of chaining `.cloned()`, we skip allocating memory entirely for a new set of Strings when generating JSON arrays or extracting sorted key lists.
**Action:** Prefer collecting `Vec<&String>` and `sort_unstable()` only for same-scope, short-lived operations (e.g., immediate serialization with `serde_json::to_string()`, as with `unique.keys()` or `set.iter()`), but retain `.cloned().collect()` and `sort()` (or `sort_unstable()`) when the keys must be moved, returned, or sent across threads — borrowed references cannot outlive the backing collection.

## 2024-05-18 - [Avoid intermediate string allocations in loops with writeln!]
**Learning:** Using `out.push_str(&format!(...))` inside loops, such as when building a directory tree recursively or iterating over many items, causes intermediate String allocations on the heap for every single iteration. For a directory tree with 1,000 items, this is 1,000 unnecessary allocations. Using the `write!` or `writeln!` macros from `std::fmt::Write` allows appending formatted content directly to the existing buffer, skipping the intermediate allocation entirely.
**Action:** Always prefer `writeln!(&mut out, ...)` or `write!(&mut out, ...)` over `out.push_str(&format!(...))` inside loops or performance-sensitive areas to minimize heap allocations.
