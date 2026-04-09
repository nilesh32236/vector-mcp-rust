## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.

## 2026-04-08 - [Avoid intermediate heap allocations when collecting Map/Set keys]
**Learning:** `serde_json::to_string()` accepts and serializes arrays of string references `Vec<&String>` just as easily as arrays of owned strings `Vec<String>`. By using `.collect()` directly on `unique.keys()` or `set.iter()` instead of chaining `.cloned()`, we skip allocating memory entirely for a new set of Strings when generating JSON arrays or extracting sorted key lists.
**Action:** Prefer collecting `Vec<&String>` and `sort_unstable()` only for same-scope, short-lived operations (e.g., immediate serialization with `serde_json::to_string()`, as with `unique.keys()` or `set.iter()`), but retain `.cloned().collect()` and `sort()` (or `sort_unstable()`) when the keys must be moved, returned, or sent across threads — borrowed references cannot outlive the backing collection.

## 2026-04-09 - [Avoid intermediate heap allocations when appending formatted strings in a loop]
**Learning:** `out.push_str(&format!(...))` inside loops creates an intermediate `String` allocation for every formatted line before pushing it to the main string buffer, leading to high heap allocation overhead.
**Action:** Replace `out.push_str(&format!(...))` with the `std::fmt::Write` macro (`write!(&mut out, ...)` or `writeln!(&mut out, ...)`), which appends formatted data directly into the existing buffer, eliminating intermediate allocations.
