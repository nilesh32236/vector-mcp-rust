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
    CallToolParams, CallToolResult, InitializeResult, JsonRpcErrorResponse,
    JsonRpcRequest, JsonRpcResponse, ServerCapabilities, ServerInfo,
    ToolCapability, INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND,
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
}

impl Server {
    pub fn new(
        store: Arc<Store>,
        config: Arc<Config>,
        embedder: Arc<Embedder>,
        summarizer: Arc<Summarizer>,
    ) -> Self {
        Self {
            store,
            config,
            embedder,
            summarizer,
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

            let response_bytes = self.handle_line(trimmed).await;

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

    /// Parse one JSON-RPC line and produce the response bytes.
    async fn handle_line(&self, line: &str) -> Vec<u8> {
        // 1. Parse JSON
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                error!(err = %e, "failed to parse JSON-RPC request");
                let resp = JsonRpcErrorResponse::new(
                    None,
                    INVALID_PARAMS,
                    format!("Invalid JSON: {e}"),
                );
                return serialize_response(&resp);
            }
        };

        // 2. Route by method
        let id = request.id.clone();
        match request.method.as_str() {
            "initialize" => {
                let result = self.handle_initialize();
                serialize_response(&JsonRpcResponse::new(id, json_value(&result)))
            }
            "initialized" => {
                serialize_response(&JsonRpcResponse::new(id, json!({})))
            }
            "tools/list" => {
                let tools = tools::tool_definitions();
                serialize_response(&JsonRpcResponse::new(id, json!({ "tools": tools })))
            }
            "tools/call" => self.handle_tools_call(id, &request.params).await,
            "notifications/cancelled" | "notifications/progress" => {
                serialize_response(&JsonRpcResponse::new(id, json!({})))
            }
            _ => {
                let resp = JsonRpcErrorResponse::new(
                    id,
                    METHOD_NOT_FOUND,
                    format!("Method not found: {}", request.method),
                );
                serialize_response(&resp)
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
    ) -> Vec<u8> {
        let params: CallToolParams = match serde_json::from_value(raw_params.clone()) {
            Ok(p) => p,
            Err(e) => {
                let resp = JsonRpcErrorResponse::new(
                    id,
                    INVALID_PARAMS,
                    format!("Invalid tools/call params: {e}"),
                );
                return serialize_response(&resp);
            }
        };

        let result: CallToolResult = match handlers::dispatch(self, &params).await {
            Ok(r) => r,
            Err(e) => {
                error!(tool = %params.name, err = %e, "handler error");
                let resp = JsonRpcErrorResponse::new(
                    id,
                    INTERNAL_ERROR,
                    format!("Tool execution error: {e}"),
                );
                return serialize_response(&resp);
            }
        };

        serialize_response(&JsonRpcResponse::new(id, json_value(&result)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn json_value<T: serde::Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or_else(|e| json!({"error": e.to_string()}))
}

fn serialize_response<T: serde::Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_else(|e| {
        format!(r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"Serialization failed: {e}"}}}}"#)
            .into_bytes()
    })
}
