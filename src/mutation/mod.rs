//! Mutation safety — LSP-verified patch integrity checking.
//!
//! Mirrors Go's `internal/mutation/safety.go`.
//!
//! Before any search-and-replace patch is written to disk, [`SafetyChecker`]
//! applies it **in-memory**, sends a `textDocument/didOpen` notification to the
//! language server, and waits up to 5 seconds for `textDocument/publishDiagnostics`.
//! If the LSP reports severity-1 (Error) diagnostics, the patch is rejected.

pub mod write_log;

use std::time::Duration;

use anyhow::{Result, bail};

use crate::lsp::{Diagnostic, LspManager};

/// Timeout waiting for `publishDiagnostics` after a `didOpen` notification.
const DIAG_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// SafetyChecker
// ---------------------------------------------------------------------------

/// Verifies proposed code patches using LSP diagnostics before writing to disk.
pub struct SafetyChecker;

impl SafetyChecker {
    /// Verify that applying `search → replace` in `path` does not introduce
    /// compiler errors according to the language server.
    ///
    /// # Returns
    /// - `Ok(vec![])` — patch is safe (no errors).
    /// - `Ok(diags)` — patch introduces diagnostics; caller should inspect severity.
    /// - `Err(_)` — LSP unavailable, file unreadable, or search string not found.
    pub async fn verify_patch(
        lsp: &LspManager,
        path: &str,
        search: &str,
        replace: &str,
    ) -> Result<Vec<Diagnostic>> {
        // 1. Read original file content.
        let original = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read {path}: {e}"))?;

        // 2. Apply patch in-memory (never touches disk here).
        if !original.contains(search) {
            bail!("search string not found in {path}");
        }
        let patched = original.replacen(search, replace, 1);

        // 3. Register a one-shot diagnostics waiter for this URI.
        let uri = format!("file://{path}");
        let diag_rx = lsp.wait_for_diagnostics(&uri).await;

        // 4. Notify the LSP with the patched content via `textDocument/didOpen`.
        //    Even if the file is already open, gopls / tsserver will re-analyse it.
        let lang_id = language_id_for(path);
        lsp.notify(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri":        uri,
                    "languageId": lang_id,
                    "version":    1,
                    "text":       patched,
                }
            }),
        )
        .await?;

        // 5. Wait for diagnostics with timeout.
        let diags = tokio::time::timeout(DIAG_TIMEOUT, diag_rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timeout waiting for LSP diagnostics for {path}"))?
            .map_err(|_| anyhow::anyhow!("Diagnostics channel closed"))?;

        Ok(diags)
    }

    /// Format a list of diagnostics into a human-readable summary.
    pub fn format_diagnostics(diags: &[Diagnostic]) -> String {
        if diags.is_empty() {
            return "✅ No compiler errors introduced.".to_string();
        }
        let mut out = format!("⚠️ {} diagnostic(s) found:\n", diags.len());
        for d in diags {
            let severity = match d.severity {
                Some(1) => "Error",
                Some(2) => "Warning",
                Some(3) => "Info",
                _ => "Hint",
            };
            out.push_str(&format!(
                "  [{severity}] line {}: {}\n",
                d.range.start.line + 1,
                d.message
            ));
        }
        out
    }

    /// Returns `true` if any diagnostic has severity 1 (Error).
    pub fn has_errors(diags: &[Diagnostic]) -> bool {
        diags.iter().any(|d| d.severity == Some(1))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a file path to an LSP `languageId` string.
fn language_id_for(path: &str) -> &'static str {
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
    {
        "go" => "go",
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "py" => "python",
        _ => "plaintext",
    }
}
