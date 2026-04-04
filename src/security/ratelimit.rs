//! Per-client token-bucket rate limiter, mirrors Go's ratelimit package.

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

struct Bucket {
    tokens: f64,
    last_update: Instant,
}

pub struct RateLimiter {
    buckets: Arc<DashMap<String, Bucket>>,
    rate: f64,  // tokens / second
    burst: f64, // max tokens
}

impl RateLimiter {
    pub fn new(rate: f64, burst: f64) -> Self {
        Self {
            buckets: Arc::new(DashMap::new()),
            rate,
            burst,
        }
    }

    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn allow(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut entry = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| Bucket {
                tokens: self.burst,
                last_update: now,
            });

        let elapsed = now.duration_since(entry.last_update).as_secs_f64();
        entry.tokens = (entry.tokens + elapsed * self.rate).min(self.burst);
        entry.last_update = now;

        if entry.tokens >= 1.0 {
            entry.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Evict stale buckets older than `max_age_secs` to prevent unbounded growth.
    #[allow(dead_code)]
    pub fn cleanup(&self, max_age_secs: f64) {
        let now = Instant::now();
        self.buckets
            .retain(|_, b| now.duration_since(b.last_update).as_secs_f64() < max_age_secs);
    }
}
