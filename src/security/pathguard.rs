//! Path guard — prevents directory traversal, hardens against symlink escapes.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Path operations — Read for existing files, Create for potentially new files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOp {
    /// Read/Inspect an existing path. Resolves target with `canonicalize`.
    Read,
    /// Create a new path. Resolves the parent directory with `canonicalize`.
    Create,
}

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
    /// Resolves symlinks before prefix checks to prevent symlink escape.
    /// Use `op=Read` for existing files, `op=Create` for new files (validates parent).
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
}
