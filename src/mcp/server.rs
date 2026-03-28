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
    pub reload_watcher_tx: tokio::sync::mpsc::Sender<String>,
    pub indexing_progress: Arc<dashmap::DashMap<String, serde_json::Value>>,
}

impl Server {
    pub fn new(
        store: Arc<Store>,
        config: Arc<Config>,
        embedder: Arc<Embedder>,
        summarizer: Arc<Summarizer>,
        reload_watcher_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Self {
        let indexing_progress = Arc::new(dashmap::DashMap::new());
        Self {
            store,
            config,
            embedder,
            summarizer,
            reload_watcher_tx,
            indexing_progress,
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
        // 1. Parse JSON
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                error!(err = %e, "failed to parse JSON-RPC request");
                let resp =
                    JsonRpcErrorResponse::new(None, INVALID_PARAMS, format!("Invalid JSON: {e}"));
                return to_json_rpc_value(&resp, None);
            }
        };

        // 2. Route by method
        let id = request.id.clone();
        match request.method.as_str() {
            "initialize" => {
                let result = self.handle_initialize();
                to_json_rpc_value(&JsonRpcResponse::new(id.clone(), json_value(&result)), id)
            }
            "initialized" => to_json_rpc_value(&JsonRpcResponse::new(id.clone(), json!({})), id),
            "tools/list" => {
                let tools = tools::tool_definitions();
                to_json_rpc_value(
                    &JsonRpcResponse::new(id.clone(), json!({ "tools": tools })),
                    id,
                )
            }
            "tools/call" => self.handle_tools_call(id, &request.params).await,
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
        }
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
    ) -> serde_json::Value {
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

        let result: CallToolResult = match handlers::dispatch(self, &params).await {
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
