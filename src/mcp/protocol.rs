//! JSON-RPC 2.0 and MCP protocol types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 core types
// ---------------------------------------------------------------------------

/// Incoming JSON-RPC 2.0 request envelope.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Outgoing JSON-RPC 2.0 success response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    pub result: serde_json::Value,
}

/// Outgoing JSON-RPC 2.0 error response.
#[derive(Debug, Serialize)]
pub struct JsonRpcErrorResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    pub error: JsonRpcError,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// -- Standard JSON-RPC error codes -----------------------------------------

/// Method not found.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// Invalid params.
pub const INVALID_PARAMS: i64 = -32602;
/// Internal error.
pub const INTERNAL_ERROR: i64 = -32603;

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl JsonRpcResponse {
    pub fn new(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result,
        }
    }
}

impl JsonRpcErrorResponse {
    pub fn new(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            error: JsonRpcError {
                code,
                message: message.into(),
                data: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MCP-specific payload types
// ---------------------------------------------------------------------------

/// Single tool definition returned by `tools/list`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: ToolInputSchema,
}

/// JSON Schema fragment describing a tool's expected input.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    pub schema_type: &'static str,
    pub properties: serde_json::Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// Parameters for `tools/call`.
#[derive(Debug, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Result payload for `tools/call`.
#[derive(Debug, Serialize)]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

/// Content block inside a `CallToolResult`.
#[derive(Debug, Serialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

impl CallToolResult {
    /// Convenience: success text result.
    pub fn text(msg: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text",
                text: msg.into(),
            }],
            is_error: false,
        }
    }

    /// Convenience: error text result (still a valid tool response, not a
    /// JSON-RPC error).
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text",
                text: msg.into(),
            }],
            is_error: true,
        }
    }
}

/// Server capabilities advertised during `initialize`.
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    pub tools: ToolCapability,
}

#[derive(Debug, Serialize)]
pub struct ToolCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Response payload for `initialize`.
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}
