//! MCP server — reads JSON-RPC from stdin, writes responses to stdout.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info};

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;
use crate::llm::worker::LlmWorker;
use crate::lsp::LspPool;
use crate::mcp::router::SemanticRouter;
use crate::mutation::write_log::WriteLog;
use crate::security::pathguard::PathGuard;
use crate::security::ratelimit::RateLimiter;

use super::handlers;
use super::protocol::{
    CallToolParams, CallToolResult, INTERNAL_ERROR, INVALID_PARAMS, InitializeResult,
    JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND, ServerCapabilities,
    ServerInfo, ToolCapability,
};
use super::tools;

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The MCP server holding shared state.
pub struct Server {
    pub store: Arc<Store>,
    pub config: Arc<Config>,
    pub embedder: Arc<Embedder>,
    pub summarizer: Arc<Summarizer>,
    pub llm_worker: Option<Arc<LlmWorker>>,
    pub semantic_router: Option<Arc<SemanticRouter>>,
    pub reload_watcher_tx: tokio::sync::mpsc::Sender<String>,
    pub indexing_progress: Arc<std::sync::RwLock<crate::indexer::scanner::ProgressState>>,
    /// session_id → SSE sender for `$/progress` notifications.
    pub progress_senders:
        Arc<dashmap::DashMap<String, tokio::sync::mpsc::UnboundedSender<serde_json::Value>>>,
    pub path_guard: Arc<PathGuard>,
    pub rate_limiter: Arc<RateLimiter>,
    pub lsp_pool: Arc<LspPool>,
    /// Append-only log of all file write operations.
    pub write_log: Arc<WriteLog>,
}

impl Server {
    pub fn new(
        store: Arc<Store>,
        config: Arc<Config>,
        embedder: Arc<Embedder>,
        summarizer: Arc<Summarizer>,
        llm_worker: Option<Arc<LlmWorker>>,
        semantic_router: Option<Arc<SemanticRouter>>,
        reload_watcher_tx: tokio::sync::mpsc::Sender<String>,
        write_log: Arc<WriteLog>,
    ) -> Self {
        let indexing_progress = Arc::new(std::sync::RwLock::new(
            crate::indexer::scanner::ProgressState::default(),
        ));
        let root = config.project_root.read().unwrap().clone();
        let path_guard =
            PathGuard::new(&root).unwrap_or_else(|_| PathGuard::new(std::env::temp_dir()).unwrap());
        Self {
            store,
            config,
            embedder,
            summarizer,
            llm_worker,
            semantic_router,
            reload_watcher_tx,
            indexing_progress,
            progress_senders: Arc::new(dashmap::DashMap::new()),
            path_guard: Arc::new(path_guard),
            rate_limiter: Arc::new(RateLimiter::new(30.0, 60.0)),
            lsp_pool: Arc::new(LspPool::new(root)),
            write_log,
        }
    }

    /// Run the JSON-RPC read loop on stdin/stdout until EOF.
    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line_buf = String::new();

        info!("MCP server listening on stdio");

        loop {
            line_buf.clear();
            let bytes_read = reader
                .read_line(&mut line_buf)
                .await
                .context("reading from stdin")?;

            if bytes_read == 0 {
                // EOF — client disconnected.
                info!("stdin closed, shutting down MCP server");
                break;
            }

            let trimmed = line_buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            debug!(raw = trimmed, "incoming request");

            let response = self.process_message(trimmed).await;
            let response_bytes = serde_json::to_vec(&response).unwrap_or_else(|e| {
                format!(r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"Serialization failed: {e}"}}}}"#)
                    .into_bytes()
            });

            stdout
                .write_all(&response_bytes)
                .await
                .context("writing response to stdout")?;
            stdout
                .write_all(b"\n")
                .await
                .context("writing newline to stdout")?;
            stdout.flush().await.context("flushing stdout")?;
        }

        Ok(())
    }

    /// Parse one JSON-RPC line and produce the response object.
    pub async fn process_message(&self, line: &str) -> serde_json::Value {
        self.process_message_with_session(line, None).await
    }

    /// Parse one JSON-RPC line, optionally associated with an SSE session.
    pub async fn process_message_with_session(
        &self,
        line: &str,
        session_id: Option<&str>,
    ) -> serde_json::Value {
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                error!(err = %e, "failed to parse JSON-RPC request");
                let resp =
                    JsonRpcErrorResponse::new(None, INVALID_PARAMS, format!("Invalid JSON: {e}"));
                return to_json_rpc_value(&resp, None);
            }
        };

        let id = request.id.clone();
        let response = match request.method.as_str() {
            "initialize" => {
                let result = self.handle_initialize();
                to_json_rpc_value(&JsonRpcResponse::new(id.clone(), json_value(&result)), id)
            }
            "initialized" | "notifications/initialized" => serde_json::Value::Null,
            "tools/list" => {
                let tools = tools::tool_definitions();
                to_json_rpc_value(
                    &JsonRpcResponse::new(id.clone(), json!({ "tools": tools })),
                    id,
                )
            }
            "tools/call" => {
                self.handle_tools_call(id, &request.params, session_id)
                    .await
            }
            "notifications/cancelled" | "notifications/progress" => {
                to_json_rpc_value(&JsonRpcResponse::new(id.clone(), json!({})), id)
            }
            _ => {
                let resp = JsonRpcErrorResponse::new(
                    id.clone(),
                    METHOD_NOT_FOUND,
                    format!("Method not found: {}", request.method),
                );
                to_json_rpc_value(&resp, id)
            }
        };

        info!(method = %request.method, "Processed MCP message");
        debug!(response = %serde_json::to_string(&response).unwrap_or_default(), "Outgoing response");
        response
    }

    /// `initialize` — handshake.
    fn handle_initialize(&self) -> InitializeResult {
        InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: ToolCapability {
                    list_changed: false,
                },
            },
            server_info: ServerInfo {
                name: "vector-mcp-rust".into(),
                version: "0.1.0".into(),
            },
        }
    }

    /// `tools/call` — parse params, dispatch to handler, wrap result.
    async fn handle_tools_call(
        &self,
        id: Option<serde_json::Value>,
        raw_params: &serde_json::Value,
        session_id: Option<&str>,
    ) -> serde_json::Value {
        // --- Rate Limiting ---
        let key = session_id.unwrap_or("stdio");
        if !self.rate_limiter.allow(key) {
            let resp = JsonRpcErrorResponse::new(
                id.clone(),
                -32001, // Custom code for rate limiting
                "Rate limit exceeded (30 requests/min). Please slow down tool calls.".to_string(),
            );
            return to_json_rpc_value(&resp, id);
        }

        let params: CallToolParams = match serde_json::from_value(raw_params.clone()) {
            Ok(p) => p,
            Err(e) => {
                let resp = JsonRpcErrorResponse::new(
                    id.clone(),
                    INVALID_PARAMS,
                    format!("Invalid tools/call params: {e}"),
                );
                return to_json_rpc_value(&resp, id);
            }
        };

        let result: CallToolResult = match handlers::dispatch(self, &params, session_id).await {
            Ok(r) => r,
            Err(e) => {
                error!(tool = %params.name, err = %e, "handler error");
                let resp = JsonRpcErrorResponse::new(
                    id.clone(),
                    INTERNAL_ERROR,
                    format!("Tool execution error: {e}"),
                );
                return to_json_rpc_value(&resp, id);
            }
        };

        to_json_rpc_value(&JsonRpcResponse::new(id.clone(), json_value(&result)), id)
    }

    /// Emit a `$/progress` JSON-RPC notification to the SSE session (fire-and-forget).
    #[allow(dead_code)]
    pub fn emit_progress(&self, session_id: &str, token: &str, progress: u64, total: u64) {
        if let Some(tx) = self.progress_senders.get(session_id) {
            let notification = serde_json::json!({
                "jsonrpc": "2.0",
                "method": "$/progress",
                "params": {
                    "progressToken": token,
                    "progress": progress,
                    "total": total,
                }
            });
            let _ = tx.send(notification);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn json_value<T: serde::Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or_else(|e| json!({"error": e.to_string()}))
}

/// Fallback for JSON-RPC serialization errors.
fn to_json_rpc_value<T: serde::Serialize>(
    v: &T,
    id: Option<serde_json::Value>,
) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or_else(|e| {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": INTERNAL_ERROR,
                "message": format!("Serialization failed: {e}")
            }
        })
    })
}
