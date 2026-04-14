## 2026-04-06 - [Avoid repeated prefix allocation for string iteration]

**Learning:** In the `fast_chunk` fallback function, allocating a new string `let prefix: String = chars[..i].iter().collect();` just to count the number of new lines with `prefix.matches('\n').count()` produces O(N^2) time complexity. We should track `current_line` incrementally.
**Action:** Always maintain incremental running counts for string metadata (like line numbers) when iterating through character arrays or byte slices in O(N) operations, rather than copying expanding prefixes.

## 2026-04-08 - [Avoid intermediate heap allocations when collecting Map/Set keys]
**Learning:** `serde_json::to_string()` accepts and serializes arrays of string references `Vec<&String>` just as easily as arrays of owned strings `Vec<String>`. By using `.collect()` directly on `unique.keys()` or `set.iter()` instead of chaining `.cloned()`, we skip allocating memory entirely for a new set of Strings when generating JSON arrays or extracting sorted key lists.
**Action:** Prefer collecting `Vec<&String>` and `sort_unstable()` only for same-scope, short-lived operations (e.g., immediate serialization with `serde_json::to_string()`, as with `unique.keys()` or `set.iter()`), but retain `.cloned().collect()` and `sort()` (or `sort_unstable()`) when the keys must be moved, returned, or sent across threads — borrowed references cannot outlive the backing collection.

## 2026-04-14 - [Avoid intermediate string allocations using fmt::Write]
**Learning:** Using `string_buffer.push_str(&format!(...))` inside loops creates intermediate `String` allocations for every iteration on the heap before being copied and dropped. We can use `std::fmt::Write` to let the formatting macros (`write!`, `writeln!`) push chunks directly into the existing string buffer, eliminating temporary heap allocations.
**Action:** When building large strings dynamically within tight loops, prefer using `write!(&mut string_buffer, ...)` or `writeln!(&mut string_buffer, ...)` instead of `string_buffer.push_str(&format!(...))`. Remember to use `writeln!` instead of `write!` for strings ending with a newline to avoid `clippy::write_with_newline` CI build failures.
