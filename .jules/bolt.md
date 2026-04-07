# Bolt's Journal

## 2024-05-14 - Initial Setup
**Learning:** Just starting out here. I'll document critical learnings as I go.
**Action:** Let's look for optimization opportunities in the Rust codebase.
## 2024-05-14 - Rust String Indexing O(N^2) to O(N) Optimization
**Learning:** In Rust, iterating over `.chars()` and collecting to a `Vec<char>` can cause extreme performance degradation (O(N^2) and high memory usage) for large strings when used in sliding window algorithms. Using `char_indices` to map character indexes to byte offsets, and storing newline positions to quickly calculate line numbers via `partition_point` (binary search) is a significantly faster approach. This reduced test processing time from ~182ms to ~2.8ms in my benchmarks.
**Action:** Always prefer `char_indices` over collecting `char`s for sliding window slicing in Rust.
