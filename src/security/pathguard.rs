//! Path guard — prevents directory traversal, hardens against symlink escapes.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Blocked path segments for write operations.
const WRITE_BLOCKED_SEGMENTS: &[&str] = &[".git", "node_modules", "target"];

/// Allowed file extensions for write operations.
const WRITE_ALLOWED_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "go", "py", "md", "toml", "json", "yaml", "yml",
];

/// Path operations — Read for existing files, Create for potentially new files,
/// Write for overwriting existing files with safety checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOp {
    /// Read/Inspect an existing path. Resolves target with `canonicalize`.
    Read,
    /// Create a new path. Resolves the parent directory with `canonicalize`.
    Create,
    /// Write/overwrite an existing or new path. Applies blocked-segment and
    /// extension allowlist checks in addition to the base containment check.
    Write,
}

pub struct PathGuard {
    base: PathBuf,
    /// Optional explicit allowlist of absolute paths that are permitted for writes.
    /// When `None`, any path passing the other checks is allowed.
    write_allowlist: Option<Vec<PathBuf>>,
}

impl PathGuard {
    /// Create a guard rooted at `base`. The directory must exist.
    pub fn new(base: impl AsRef<Path>) -> Result<Self> {
        let base = std::fs::canonicalize(base.as_ref())
            .map_err(|e| anyhow::anyhow!("base path invalid: {e}"))?;
        if !base.is_dir() {
            bail!("base path is not a directory");
        }

        // Read optional write allowlist from env using platform-aware path splitting.
        let write_allowlist = std::env::var_os("WRITE_ALLOWLIST").map(|val| {
            std::env::split_paths(&val)
                .filter(|p| !p.as_os_str().is_empty())
                .collect::<Vec<_>>()
        });

        Ok(Self {
            base,
            write_allowlist,
        })
    }

    /// Validate `target` and return its absolute path, guaranteed to be inside `base`.
    /// Resolves symlinks before prefix checks to prevent symlink escape.
    /// Use `op=Read` for existing files, `op=Create` for new files (validates parent),
    /// `op=Write` for overwriting files (applies blocked-segment + extension checks).
    pub fn validate(&self, target: impl AsRef<Path>, op: PathOp) -> Result<PathBuf> {
        let target = target.as_ref();
        if target.as_os_str().is_empty() {
            bail!("empty path");
        }

        let joined = if target.is_absolute() {
            target.to_path_buf()
        } else {
            self.base.join(target)
        };

        let resolved = match op {
            PathOp::Read => {
                // For read operations, target must exist so we can canonicalize it directly.
                std::fs::canonicalize(&joined).with_context(|| {
                    format!("failed to canonicalize target: {}", joined.display())
                })?
            }
            PathOp::Create => {
                // For create operations, the target might not exist. Canonicalize parent.
                let parent = joined.parent().ok_or_else(|| {
                    anyhow::anyhow!("invalid target parent: {}", joined.display())
                })?;

                // If parent doesn't exist (deep create), create_dir_all happens later but
                // we should at least validate the closest existing ancestor starts with base.
                let mut current = parent;
                while !current.exists() && current != self.base {
                    if let Some(p) = current.parent() {
                        current = p;
                    } else {
                        break;
                    }
                }

                let resolved_parent = std::fs::canonicalize(current).with_context(|| {
                    format!("failed to canonicalize ancestor: {}", current.display())
                })?;

                if !resolved_parent.starts_with(&self.base) {
                    bail!("path traversal attempt via ancestor detected");
                }

                // Now that we know the ancestor is safe, we use lexical normalization for the rest.
                normalize_path(&joined)
            }
            PathOp::Write => {
                // Write: lexical normalisation first (file may not exist yet for atomic writes).
                let normalized = normalize_path(&joined);

                // If the file already exists, canonicalize to catch symlink escapes.
                let resolved = if normalized.exists() {
                    std::fs::canonicalize(&normalized).with_context(|| {
                        format!("failed to canonicalize write target: {}", normalized.display())
                    })?
                } else {
                    // File doesn't exist yet — validate the parent exists and is safe.
                    let parent = normalized.parent().ok_or_else(|| {
                        anyhow::anyhow!("invalid write target parent: {}", normalized.display())
                    })?;
                    let mut current = parent;
                    while !current.exists() && current != self.base.as_path() {
                        if let Some(p) = current.parent() {
                            current = p;
                        } else {
                            break;
                        }
                    }
                    let resolved_parent = std::fs::canonicalize(current).with_context(|| {
                        format!("failed to canonicalize write ancestor: {}", current.display())
                    })?;
                    if !resolved_parent.starts_with(&self.base) {
                        bail!("path traversal attempt via ancestor detected (write)");
                    }
                    normalized
                };

                // Check blocked segments.
                for component in resolved.components() {
                    let seg = component.as_os_str().to_string_lossy();
                    if WRITE_BLOCKED_SEGMENTS.iter().any(|b| *b == seg.as_ref()) {
                        bail!("write blocked: path contains blocked segment '{seg}'");
                    }
                }

                // Check extension allowlist.
                let ext = resolved
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if !WRITE_ALLOWED_EXTENSIONS.contains(&ext) {
                    bail!(
                        "write blocked: extension '.{ext}' is not in the write allowlist \
                         (allowed: {})",
                        WRITE_ALLOWED_EXTENSIONS.join(", ")
                    );
                }

                // Check explicit allowlist if configured.
                if let Some(ref allowlist) = self.write_allowlist {
                    let permitted = allowlist.iter().any(|allowed| resolved.starts_with(allowed));
                    if !permitted {
                        bail!(
                            "write blocked: path is not in the configured WRITE_ALLOWLIST"
                        );
                    }
                }

                resolved
            }
        };

        if !resolved.starts_with(&self.base) {
            bail!(
                "path traversal attempt detected: resolved path {} is outside base {}",
                resolved.display(),
                self.base.display()
            );
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

/// Resolve `..` and `.` without hitting the filesystem.
/// Note: This is lexical only. For security, `std::fs::canonicalize` MUST be used
/// on existing components before prefix checks.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_path_guard_basics() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        let guard = PathGuard::new(base)?;

        let file = base.join("test.txt");
        fs::write(&file, "hello")?;

        // Read existing file
        let validated = guard.validate("test.txt", PathOp::Read)?;
        assert_eq!(validated, fs::canonicalize(file)?);

        // Create new file
        let new_file = guard.validate("new.txt", PathOp::Create)?;
        assert_eq!(new_file, base.join("new.txt"));

        Ok(())
    }

    #[test]
    fn test_path_guard_traversal() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        let guard = PathGuard::new(base)?;

        // Lexical traversal
        assert!(guard.validate("../outside.txt", PathOp::Read).is_err());
        assert!(guard.validate("../outside.txt", PathOp::Create).is_err());

        Ok(())
    }

    #[test]
    fn test_path_guard_symlink_escape() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        let guard = PathGuard::new(base)?;

        let outside_dir = tempdir()?;
        let secret = outside_dir.path().join("secret.txt");
        fs::write(&secret, "confidential")?;

        // Create a symlink inside base pointing outside
        #[cfg(unix)]
        {
            let link = base.join("link_to_outside");
            std::os::unix::fs::symlink(outside_dir.path(), &link)?;

            // Try to read through the link
            let malicious = "link_to_outside/secret.txt";
            assert!(guard.validate(malicious, PathOp::Read).is_err());
        }

        Ok(())
    }

    #[test]
    fn test_write_blocked_segments() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        let guard = PathGuard::new(base)?;

        // Blocked segments should be rejected
        assert!(guard.validate(".git/config", PathOp::Write).is_err());
        assert!(guard.validate("node_modules/pkg/index.js", PathOp::Write).is_err());
        assert!(guard.validate("target/debug/build.rs", PathOp::Write).is_err());

        Ok(())
    }

    #[test]
    fn test_write_extension_allowlist() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();
        let guard = PathGuard::new(base)?;

        // Disallowed extensions
        assert!(guard.validate("binary.exe", PathOp::Write).is_err());
        assert!(guard.validate("archive.zip", PathOp::Write).is_err());
        assert!(guard.validate("image.png", PathOp::Write).is_err());

        // Allowed extensions (file doesn't need to exist for Write validation)
        assert!(guard.validate("src/main.rs", PathOp::Write).is_ok());
        assert!(guard.validate("README.md", PathOp::Write).is_ok());
        assert!(guard.validate("config.toml", PathOp::Write).is_ok());

        Ok(())
    }

    #[test]
    fn test_write_allowlist_env_var() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path();

        // Create a sub-directory that will be in the allowlist.
        let allowed_sub = base.join("allowed");
        fs::create_dir_all(&allowed_sub)?;

        // Set WRITE_ALLOWLIST to the allowed sub-directory.
        // Use std::env::join_paths to be platform-safe.
        let allowlist_val = std::env::join_paths([&allowed_sub]).unwrap();
        // SAFETY: test-only; single-threaded test runner.
        unsafe { std::env::set_var("WRITE_ALLOWLIST", &allowlist_val) };

        let guard = PathGuard::new(base)?;

        // A path inside the allowed sub-directory should be accepted.
        assert!(
            guard.validate("allowed/main.rs", PathOp::Write).is_ok(),
            "path inside WRITE_ALLOWLIST should be accepted"
        );

        // A path outside the allowed sub-directory should be rejected.
        assert!(
            guard.validate("other/main.rs", PathOp::Write).is_err(),
            "path outside WRITE_ALLOWLIST should be rejected"
        );

        // Clean up env var so other tests are not affected.
        unsafe { std::env::remove_var("WRITE_ALLOWLIST") };

        Ok(())
    }
}
