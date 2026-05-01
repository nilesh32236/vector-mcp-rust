//! Append-only write operation log.
//!
//! Every file mutation performed by the MCP server is recorded as an NDJSON
//! entry in `$DATA_DIR/write_ops.log`. The log is never truncated — entries
//! are only appended. This provides an audit trail and enables `restore_backup`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// A single write operation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteLogEntry {
    /// RFC 3339 timestamp of the operation.
    pub timestamp: String,
    /// Relative or absolute path of the file that was written.
    pub path: String,
    /// Action performed: `"write_file"`, `"generate_inline_docs"`,
    /// `"propose_refactor"`, or `"restore"`.
    pub action: String,
    /// Path to the backup file created before the write, if any.
    pub backup_path: Option<String>,
    /// Hex-encoded SHA-256 hash of the content that was written.
    pub content_hash: String,
}

/// Append-only NDJSON write log.
pub struct WriteLog {
    log_path: PathBuf,
}

impl WriteLog {
    /// Create a `WriteLog` whose log file lives at `data_dir/write_ops.log`.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            log_path: data_dir.join("write_ops.log"),
        }
    }

    /// Append a single entry to the log atomically.
    ///
    /// The log file is opened in append mode and created if it does not exist.
    /// An exclusive advisory lock is held for the duration of the write to
    /// prevent interleaved entries from concurrent processes.
    /// `sync_data()` is called before releasing the lock to ensure durability.
    pub fn append(&self, entry: &WriteLogEntry) -> Result<()> {
        use std::fs::OpenOptions;

        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.log_path)?;

        // Acquire an exclusive advisory lock.
        lock_exclusive(&file)?;

        let line = serde_json::to_string(entry)?;
        let result = (|| -> Result<()> {
            writeln!(file, "{line}")?;
            file.sync_data()?;
            Ok(())
        })();

        // Always release the lock, even on error.
        let _ = unlock(&file);
        result
    }

    /// Return the last `n` entries from the log efficiently.
    ///
    /// Reads the file from the end in chunks, collecting lines until `n`
    /// complete JSON entries have been found. Returns an empty `Vec` when the
    /// log file does not exist or cannot be read.
    pub fn last_n(&self, n: usize) -> Vec<WriteLogEntry> {
        if n == 0 {
            return Vec::new();
        }

        let mut file = match std::fs::File::open(&self.log_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let file_len = match file.seek(SeekFrom::End(0)) {
            Ok(len) => len,
            Err(_) => return Vec::new(),
        };

        if file_len == 0 {
            return Vec::new();
        }

        // Read backwards in 8 KiB chunks, collecting complete lines.
        const CHUNK: u64 = 8192;
        let mut collected_lines: Vec<String> = Vec::new();
        let mut pos = file_len;
        // Accumulates bytes from the current read position to the end of the
        // last incomplete line boundary.
        let mut tail_buf: Vec<u8> = Vec::new();

        while pos > 0 && collected_lines.len() < n {
            let read_size = CHUNK.min(pos);
            pos -= read_size;

            if file.seek(SeekFrom::Start(pos)).is_err() {
                break;
            }

            let mut chunk = vec![0u8; read_size as usize];
            let bytes_read = match std::io::Read::read(&mut file, &mut chunk) {
                Ok(b) => b,
                Err(_) => break,
            };
            chunk.truncate(bytes_read);

            // Prepend chunk to tail_buf so we process [chunk | previous_tail].
            chunk.extend_from_slice(&tail_buf);
            // chunk now holds: new bytes + previously unprocessed tail
            let combined = chunk;

            // Split on newlines — collect complete lines (all but possibly the first).
            // The first segment may be a partial line if pos > 0.
            let mut split_iter = combined.splitn(usize::MAX, |&b| b == b'\n');
            let first = split_iter.next().unwrap_or(&[]).to_vec();

            // Collect the rest as complete lines (in reverse for chronological order).
            let rest: Vec<Vec<u8>> = split_iter.map(|s| s.to_vec()).collect();

            // The first segment is a partial line; save it for the next iteration.
            tail_buf = first;

            for line_bytes in rest.into_iter().rev() {
                let trimmed = line_bytes
                    .iter()
                    .copied()
                    .skip_while(|&b| b == b' ' || b == b'\r')
                    .collect::<Vec<_>>();
                let trimmed = {
                    let mut v = trimmed;
                    while v.last() == Some(&b' ') || v.last() == Some(&b'\r') {
                        v.pop();
                    }
                    v
                };
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(s) = std::str::from_utf8(&trimmed) {
                    collected_lines.push(s.to_string());
                    if collected_lines.len() >= n {
                        break;
                    }
                }
            }
        }

        // Process any leftover bytes before the first newline (the file's first line).
        if !tail_buf.is_empty() && collected_lines.len() < n {
            let trimmed = tail_buf
                .iter()
                .copied()
                .skip_while(|&b| b == b' ' || b == b'\r')
                .collect::<Vec<_>>();
            let trimmed = {
                let mut v = trimmed;
                while v.last() == Some(&b' ') || v.last() == Some(&b'\r') {
                    v.pop();
                }
                v
            };
            if !trimmed.is_empty() {
                if let Ok(s) = std::str::from_utf8(&trimmed) {
                    collected_lines.push(s.to_string());
                }
            }
        }

        // Parse and return in chronological order (oldest first).
        let mut entries: Vec<WriteLogEntry> = collected_lines
            .iter()
            .rev()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        entries.truncate(n);
        entries
    }
}

// ---------------------------------------------------------------------------
// Platform-specific file locking helpers
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn lock_exclusive(file: &std::fs::File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: flock is a standard POSIX syscall; fd is valid for the lifetime of the call.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if ret != 0 {
        Err(anyhow::anyhow!(
            "flock(LOCK_EX) failed: {}",
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn unlock(file: &std::fs::File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: flock is a standard POSIX syscall; fd is valid for the lifetime of the call.
    let ret = unsafe { libc::flock(fd, libc::LOCK_UN) };
    if ret != 0 {
        Err(anyhow::anyhow!(
            "flock(LOCK_UN) failed: {}",
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

// On non-Unix platforms use fs2 for cross-platform advisory file locking.
#[cfg(not(unix))]
fn lock_exclusive(file: &std::fs::File) -> Result<()> {
    use fs2::FileExt;
    file.lock_exclusive()
        .map_err(|e| anyhow::anyhow!("lock_exclusive failed: {e}"))
}

#[cfg(not(unix))]
fn unlock(file: &std::fs::File) -> Result<()> {
    use fs2::FileExt;
    file.unlock()
        .map_err(|e| anyhow::anyhow!("unlock failed: {e}"))
}
