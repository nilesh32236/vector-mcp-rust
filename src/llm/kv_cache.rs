//! KV-cache persistence for `LlamaEngine::summarize_code`.
//!
//! Saves the llama.cpp KV cache (the model's "short-term memory") to disk
//! after each full prompt evaluation.  On subsequent calls with the same
//! prompt, the cached state is loaded instead of re-evaluating the full
//! prompt, dropping time-to-first-token from seconds to milliseconds.
//!
//! ## Eviction
//!
//! A simple LRU policy is used: when the number of cached entries exceeds
//! `max_entries`, the least-recently-used entry is deleted from disk.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// KvCacheStore
// ---------------------------------------------------------------------------

/// Manages on-disk KV-cache files with LRU eviction.
pub struct KvCacheStore {
    dir: PathBuf,
    max_entries: usize,
    /// LRU order: front = most recently used, back = least recently used.
    lru: Mutex<VecDeque<String>>,
}

impl KvCacheStore {
    /// Create a new store backed by `dir`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(dir: PathBuf, max_entries: usize) -> Self {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "KvCacheStore: failed to create cache directory"
            );
        }
        Self {
            dir,
            max_entries: max_entries.max(1),
            lru: Mutex::new(VecDeque::new()),
        }
    }

    /// Return the path for the cache file corresponding to `hash`.
    pub fn cache_path(&self, hash: &str) -> PathBuf {
        self.dir.join(format!("{hash}.kvcache"))
    }

    /// Check whether a cache file exists for `hash`.
    pub fn exists(&self, hash: &str) -> bool {
        self.cache_path(hash).exists()
    }

    /// Record a cache hit or new entry — moves `hash` to the front of the LRU
    /// and immediately evicts the oldest entry if the cache is over capacity.
    ///
    /// Combining touch + eviction in a single lock acquisition prevents the
    /// transient off-by-one where `lru.len() == max_entries + 1` between calls.
    pub fn touch(&self, hash: &str) {
        let mut lru = self.lru.lock().unwrap();
        lru.retain(|h| h != hash);
        lru.push_front(hash.to_string());
        // Evict inline while we hold the lock.
        while lru.len() > self.max_entries {
            if let Some(oldest) = lru.pop_back() {
                let path = self.dir.join(format!("{oldest}.kvcache"));
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "KvCacheStore: failed to evict cache file"
                    );
                } else {
                    tracing::debug!(hash = %oldest, "KvCacheStore: evicted LRU entry");
                }
            }
        }
    }

    /// Evict the least-recently-used entry if the cache is over capacity.
    ///
    /// Kept for backwards compatibility; `touch` now handles eviction inline.
    pub fn evict_if_needed(&self) {
        // No-op: eviction is now handled atomically inside `touch`.
    }

    /// Remove a specific cache file (e.g. when it is found to be corrupt).
    pub fn remove(&self, hash: &str) {
        let path = self.cache_path(hash);
        let _ = std::fs::remove_file(&path);
        let mut lru = self.lru.lock().unwrap();
        lru.retain(|h| h != hash);
    }

    /// Number of entries currently tracked in the LRU.
    pub fn entry_count(&self) -> usize {
        self.lru.lock().unwrap().len()
    }

    /// Maximum number of entries before eviction kicks in.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// The directory where cache files are stored.
    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    /// Remove all `.kvcache` files from disk and reset the LRU state.
    pub fn clear(&self) {
        let mut lru = self.lru.lock().unwrap();
        for hash in lru.iter() {
            let path = self.dir.join(format!("{hash}.kvcache"));
            let _ = std::fs::remove_file(&path);
        }
        lru.clear();
        tracing::info!(dir = %self.dir.display(), "KV cache cleared");
    }

    /// Purge all `.kvcache` files from the directory on startup.
    ///
    /// KV-cache files are only valid within the process that created them —
    /// they encode the exact `n_ctx` and model weights used at save time.
    /// Loading a file saved by a previous run (potentially with a different
    /// context size) causes `GGML_ASSERT(logits != nullptr)` → SIGABRT.
    /// Clearing on startup is the safest policy.
    pub fn clear_on_startup(&self) {
        let dir = &self.dir;
        let count = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x == "kvcache")
                    .unwrap_or(false)
            })
            .filter_map(|e| std::fs::remove_file(e.path()).ok())
            .count();
        let mut lru = self.lru.lock().unwrap();
        lru.clear();
        if count > 0 {
            tracing::info!(
                dir = %dir.display(),
                count,
                "KV-cache: purged stale files from previous run"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_store(dir: &std::path::Path, max: usize) -> KvCacheStore {
        KvCacheStore::new(dir.to_path_buf(), max)
    }

    #[test]
    fn test_eviction_removes_oldest() {
        // Use an isolated tempdir to avoid flakes in parallel test runs.
        let tmp = tempdir().expect("tempdir");
        let store = make_store(tmp.path(), 2);

        // Simulate writing 3 cache files
        for hash in &["aaa", "bbb", "ccc"] {
            let path = store.cache_path(hash);
            fs::write(&path, b"data").unwrap();
            store.touch(hash); // touch now evicts inline
        }

        // "aaa" should have been evicted (oldest), "bbb" and "ccc" remain
        assert!(!store.cache_path("aaa").exists(), "aaa should be evicted");
        assert!(store.cache_path("bbb").exists(), "bbb should remain");
        assert!(store.cache_path("ccc").exists(), "ccc should remain");
        // TempDir cleans up automatically on drop.
    }

    #[test]
    fn test_touch_promotes_entry() {
        let tmp = tempdir().expect("tempdir");
        let store = make_store(tmp.path(), 2);

        // Write two entries
        for hash in &["aaa", "bbb"] {
            let path = store.cache_path(hash);
            fs::write(&path, b"data").unwrap();
            store.touch(hash);
        }

        // Touch "aaa" to promote it — "bbb" becomes the LRU candidate
        store.touch("aaa");

        // Write a third entry — "bbb" should be evicted, not "aaa"
        let path = store.cache_path("ccc");
        fs::write(&path, b"data").unwrap();
        store.touch("ccc"); // eviction happens inside touch

        assert!(store.cache_path("aaa").exists(), "aaa should remain (recently touched)");
        assert!(!store.cache_path("bbb").exists(), "bbb should be evicted");
        assert!(store.cache_path("ccc").exists(), "ccc should remain");
    }
}
