//! Path guard — prevents directory traversal, mirrors Go's pathguard package.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

pub struct PathGuard {
    base: PathBuf,
}

impl PathGuard {
    /// Create a guard rooted at `base`. The directory must exist.
    pub fn new(base: impl AsRef<Path>) -> Result<Self> {
        let base = std::fs::canonicalize(base.as_ref())
            .map_err(|e| anyhow::anyhow!("base path invalid: {e}"))?;
        if !base.is_dir() {
            bail!("base path is not a directory");
        }
        Ok(Self { base })
    }

    /// Validate `target` and return its absolute path, guaranteed to be inside `base`.
    pub fn validate(&self, target: impl AsRef<Path>) -> Result<PathBuf> {
        let target = target.as_ref();
        if target.as_os_str().is_empty() {
            bail!("empty path");
        }

        let joined = if target.is_absolute() {
            target.to_path_buf()
        } else {
            self.base.join(target)
        };

        // Resolve without requiring the path to exist (for create operations).
        let resolved = normalize_path(&joined);

        if !resolved.starts_with(&self.base) {
            bail!("path traversal attempt detected");
        }

        // Depth check (max 20 components below base).
        let rel = resolved.strip_prefix(&self.base).unwrap();
        if rel.components().count() > 20 {
            bail!("path exceeds maximum depth");
        }

        Ok(resolved)
    }

    #[allow(dead_code)]
    pub fn base(&self) -> &Path {
        &self.base
    }
}

/// Resolve `..` and `.` without hitting the filesystem (path may not exist yet).
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}
