//! LSP client — fully async Tokio implementation.
//!
//! Mirrors Go's `internal/lsp/client.go` with the following improvements:
//! - Fully async I/O via `tokio::process::Command` (no `blocking_recv`)
//! - 10-minute idle TTL: child process is killed automatically when unused
//! - `publishDiagnostics` notification routing for mutation safety verification
//! - Per-(extension, root) session pool via [`LspPool`]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use dashmap::DashMap;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, oneshot};
use tokio::time::Instant;
use tracing::info;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Idle TTL before the language server process is killed.
const IDLE_TTL: Duration = Duration::from_secs(600); // 10 minutes

/// Maximum time to wait for a single LSP response.
const CALL_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Language server command map (mirrors Go's LanguageServerMapping)
// ---------------------------------------------------------------------------

/// Returns the command + args for the LSP server associated with a file extension.
pub fn server_command(ext: &str) -> Option<Vec<&'static str>> {
    match ext.to_lowercase().trim_start_matches('.') {
        "go" => Some(vec!["gopls"]),
        "js" | "jsx" => Some(vec!["typescript-language-server", "--stdio"]),
        "ts" | "tsx" => Some(vec!["typescript-language-server", "--stdio"]),
        "rs" => Some(vec!["rust-analyzer"]),
        "py" => Some(vec!["pylsp"]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Diagnostic types (used by mutation safety)
// ---------------------------------------------------------------------------

/// An LSP diagnostic message returned by `textDocument/publishDiagnostics`.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Diagnostic {
    pub range: DiagnosticRange,
    /// 1 = Error, 2 = Warning, 3 = Information, 4 = Hint
    pub severity: Option<u8>,
    pub message: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DiagnosticRange {
    pub start: DiagnosticPosition,
    pub end: DiagnosticPosition,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DiagnosticPosition {
    pub line: u32,
    pub character: u32,
}

// ---------------------------------------------------------------------------
// Inner mutable state (behind an async Mutex)
// ---------------------------------------------------------------------------

struct Inner {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    id_counter: i64,
    /// Pending request id → oneshot sender waiting for the JSON-RPC response.
    pending: HashMap<i64, oneshot::Sender<Value>>,
    /// URI → oneshot sender waiting for `publishDiagnostics` for that file.
    diag_waiters: HashMap<String, oneshot::Sender<Vec<Diagnostic>>>,
    last_used: Instant,
}

impl Inner {
    fn new() -> Self {
        Self {
            child: None,
            stdin: None,
            id_counter: 0,
            pending: HashMap::new(),
            diag_waiters: HashMap::new(),
            last_used: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// LspManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of a single language server process.
///
/// - Lazily starts the server on first use.
/// - Kills the process after [`IDLE_TTL`] of inactivity.
/// - Routes `publishDiagnostics` notifications to registered one-shot waiters.
pub struct LspManager {
    cmd: Vec<String>,
    root: String,
    inner: Arc<Mutex<Inner>>,
}

impl LspManager {
    /// Create a new manager. The server process is **not** started until the first call.
    pub fn new(cmd: Vec<String>, root: String) -> Self {
        Self {
            cmd,
            root,
            inner: Arc::new(Mutex::new(Inner::new())),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Send a JSON-RPC request and await the response (up to [`CALL_TIMEOUT`]).
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.ensure_started().await?;
        self.call_inner(method, params).await
    }

    /// Internal call — does NOT call `ensure_started` (avoids recursion from `initialize`).
    async fn call_inner(&self, method: &str, params: Value) -> Result<Value> {
        let (tx, rx) = oneshot::channel::<Value>();
        let id = {
            let mut g = self.inner.lock().await;
            g.id_counter += 1;
            let id = g.id_counter;
            g.pending.insert(id, tx);
            g.last_used = Instant::now();

            let msg = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            });
            write_message(g.stdin.as_mut().unwrap(), &msg).await?;
            id
        };

        let result = tokio::time::timeout(CALL_TIMEOUT, rx)
            .await
            .map_err(|_| anyhow::anyhow!("LSP call '{method}' timed out (id={id})"))?
            .map_err(|_| anyhow::anyhow!("LSP response channel closed"))?;

        if let Some(err) = result.get("error") {
            bail!("LSP error for '{method}': {err}");
        }
        Ok(result["result"].clone())
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let mut g = self.inner.lock().await;
        if g.stdin.is_none() {
            return Ok(());
        }
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        write_message(g.stdin.as_mut().unwrap(), &msg).await
    }

    /// Register a one-shot waiter for `textDocument/publishDiagnostics` on `uri`.
    ///
    /// The sender is consumed the first time diagnostics arrive for that URI.
    /// Used by mutation safety to verify patches before writing to disk.
    pub async fn wait_for_diagnostics(&self, uri: &str) -> oneshot::Receiver<Vec<Diagnostic>> {
        let (tx, rx) = oneshot::channel();
        let mut g = self.inner.lock().await;
        g.diag_waiters.insert(uri.to_string(), tx);
        rx
    }

    /// Ensure the language server is running, starting it if necessary.
    pub async fn ensure_started(&self) -> Result<()> {
        let mut g = self.inner.lock().await;
        if g.child.is_some() {
            g.last_used = Instant::now();
            return Ok(());
        }
        if self.cmd.is_empty() {
            bail!("no LSP command configured");
        }

        info!(cmd = ?self.cmd, root = %self.root, "Starting LSP server");

        let mut child = Command::new(&self.cmd[0])
            .args(&self.cmd[1..])
            .current_dir(&self.root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let stdin: ChildStdin = child.stdin.take().unwrap();
        let stdout: ChildStdout = child.stdout.take().unwrap();

        g.child = Some(child);
        g.stdin = Some(stdin);
        g.last_used = Instant::now();

        // Spawn background reader task.
        let inner = Arc::clone(&self.inner);
        tokio::spawn(read_loop(stdout, inner));

        // Spawn idle TTL monitor.
        let inner_ttl = Arc::clone(&self.inner);
        tokio::spawn(ttl_monitor(inner_ttl));

        // Release lock before calling initialize (which re-acquires it).
        drop(g);
        self.initialize().await
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    async fn initialize(&self) -> Result<()> {
        let params = serde_json::json!({
            "processId": std::process::id(),
            "rootUri": format!("file://{}", self.root),
            "capabilities": {
                "textDocument": {
                    "definition":  { "dynamicRegistration": true },
                    "references":  { "dynamicRegistration": true },
                    "publishDiagnostics": {}
                }
            }
        });
        self.call_inner("initialize", params).await?;
        self.notify("initialized", serde_json::json!({})).await
    }
}

impl Drop for LspManager {
    fn drop(&mut self) {
        // Best-effort kill on drop — the async TTL monitor handles the normal path.
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            let mut g = inner.lock().await;
            kill_child(&mut g);
        });
    }
}

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

/// Reads LSP messages from stdout and routes them to pending request channels
/// or registered notification handlers.
async fn read_loop(stdout: ChildStdout, inner: Arc<Mutex<Inner>>) {
    let mut reader = BufReader::new(stdout);

    loop {
        // --- Parse Content-Length header ---
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => return, // EOF or error
                _ => {}
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length: ") {
                content_length = rest.trim().parse().unwrap_or(0);
            }
        }
        if content_length == 0 {
            continue;
        }

        // --- Read body ---
        let mut body = vec![0u8; content_length];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }

        let Ok(msg) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };

        let mut g = inner.lock().await;

        // Response to a pending request.
        if let Some(id) = msg.get("id").and_then(|v| v.as_i64()) {
            if let Some(tx) = g.pending.remove(&id) {
                let _ = tx.send(msg);
            }
            continue;
        }

        // Server-initiated notification.
        let method = msg["method"].as_str().unwrap_or("");
        if method == "textDocument/publishDiagnostics" {
            let uri = msg["params"]["uri"].as_str().unwrap_or("").to_string();
            if let Some(tx) = g.diag_waiters.remove(&uri) {
                let diags: Vec<Diagnostic> =
                    serde_json::from_value(msg["params"]["diagnostics"].clone())
                        .unwrap_or_default();
                let _ = tx.send(diags);
            }
        }
    }
}

/// Kills the language server process if it has been idle for [`IDLE_TTL`].
/// Mirrors Go's `monitorTTL()`.
async fn ttl_monitor(inner: Arc<Mutex<Inner>>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        let mut g = inner.lock().await;
        if g.child.is_none() {
            return; // Already stopped.
        }
        if g.last_used.elapsed() >= IDLE_TTL {
            info!("LSP server idle TTL reached — shutting down");
            kill_child(&mut g);
            return;
        }
    }
}

fn kill_child(g: &mut Inner) {
    if let Some(mut child) = g.child.take() {
        let _ = child.start_kill();
    }
    g.stdin = None;
    g.pending.clear();
    g.diag_waiters.clear();
}

// ---------------------------------------------------------------------------
// Wire-format helper
// ---------------------------------------------------------------------------

async fn write_message(stdin: &mut ChildStdin, msg: &Value) -> Result<()> {
    let body = serde_json::to_vec(msg)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin.write_all(header.as_bytes()).await?;
    stdin.write_all(&body).await?;
    stdin.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session pool
// ---------------------------------------------------------------------------

/// Per-(extension, root) pool of [`LspManager`] instances.
///
/// Lazily creates one manager per file extension. Managers are kept alive
/// until their idle TTL expires.
pub struct LspPool {
    sessions: DashMap<String, Arc<LspManager>>,
    root: String,
}

impl LspPool {
    /// Create a new pool rooted at `root`.
    pub fn new(root: String) -> Self {
        Self {
            sessions: DashMap::new(),
            root,
        }
    }

    /// Get or create an [`LspManager`] for the given file extension.
    ///
    /// Returns `None` if no LSP server is configured for that extension.
    pub fn get(&self, ext: &str) -> Option<Arc<LspManager>> {
        let cmd = server_command(ext)?;
        Some(
            self.sessions
                .entry(ext.to_string())
                .or_insert_with(|| {
                    Arc::new(LspManager::new(
                        cmd.iter().map(|s| s.to_string()).collect(),
                        self.root.clone(),
                    ))
                })
                .clone(),
        )
    }

    /// Derive the file extension from a path and return the matching manager.
    pub fn get_for_path(&self, path: &str) -> Option<Arc<LspManager>> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))?;
        self.get(&ext)
    }
}
