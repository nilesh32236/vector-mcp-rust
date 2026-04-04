//! Slave client — delegates embedding and store operations to the master daemon.
//!
//! When a slave process starts and detects a live master on the Unix socket,
//! it uses [`RemoteEmbedder`] and [`RemoteStore`] instead of local instances.
//! This keeps slave processes lightweight (no ONNX model loaded, no LanceDB open).

#![allow(dead_code)]

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::db::Record;

/// Timeout for a single RPC call to the master.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Low-level RPC helper
// ---------------------------------------------------------------------------

/// Send one JSON request line to the master and read one JSON response line.
async fn rpc_call(socket_path: &str, request: &Value) -> Result<Value> {
    let stream = tokio::time::timeout(
        Duration::from_secs(2),
        UnixStream::connect(socket_path),
    )
    .await
    .context("connect timeout")?
    .with_context(|| format!("failed to connect to master at {socket_path}"))?;

    let (reader, mut writer) = stream.into_split();

    let mut line = serde_json::to_string(request)?;
    line.push('\n');

    tokio::time::timeout(RPC_TIMEOUT, async {
        writer.write_all(line.as_bytes()).await?;
        writer.flush().await?;

        let mut response_line = String::new();
        BufReader::new(reader).read_line(&mut response_line).await?;
        Ok::<String, anyhow::Error>(response_line)
    })
    .await
    .context("RPC call timed out")?
    .context("RPC I/O error")
    .and_then(|s| serde_json::from_str(&s).context("invalid JSON response from master"))
}

/// Extract the `result` field or return the `error` as an `Err`.
fn unwrap_response(resp: Value) -> Result<Value> {
    if resp["ok"].as_bool().unwrap_or(false) {
        Ok(resp["result"].clone())
    } else {
        bail!(
            "master error: {}",
            resp["error"].as_str().unwrap_or("unknown")
        )
    }
}

// ---------------------------------------------------------------------------
// RemoteEmbedder
// ---------------------------------------------------------------------------

/// Delegates embedding calls to the master daemon over the Unix socket.
///
/// Drop-in replacement for [`crate::llm::embedding::Embedder`] in slave mode.
pub struct RemoteEmbedder {
    socket_path: String,
}

impl RemoteEmbedder {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self { socket_path: socket_path.into() }
    }

    /// Embed a document chunk.
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({ "method": "embed", "params": { "text": text } }),
        )
        .await?;
        let result = unwrap_response(resp)?;
        serde_json::from_value(result["embedding"].clone()).context("invalid embedding response")
    }

    /// Embed a search query (applies Nomic prefix on the master side).
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({ "method": "embed_query", "params": { "text": text } }),
        )
        .await?;
        let result = unwrap_response(resp)?;
        serde_json::from_value(result["embedding"].clone()).context("invalid embedding response")
    }
}

// ---------------------------------------------------------------------------
// RemoteStore
// ---------------------------------------------------------------------------

/// Delegates vector store operations to the master daemon over the Unix socket.
///
/// Drop-in replacement for [`crate::db::Store`] in slave mode.
pub struct RemoteStore {
    socket_path: String,
}

impl RemoteStore {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self { socket_path: socket_path.into() }
    }

    /// Hybrid search (vector + BM25 RRF) delegated to the master.
    pub async fn hybrid_search(
        &self,
        vector: Vec<f32>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Record>> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({
                "method": "hybrid_search",
                "params": { "query": query, "vector": vector, "limit": limit }
            }),
        )
        .await?;
        let result = unwrap_response(resp)?;
        serde_json::from_value(result["records"].clone()).context("invalid records response")
    }

    /// Upsert records into the master's store.
    pub async fn upsert_records(&self, records: Vec<Record>) -> Result<()> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({
                "method": "insert",
                "params": { "records": records }
            }),
        )
        .await?;
        unwrap_response(resp).map(|_| ())
    }

    /// Delete records by file path on the master.
    pub async fn delete_by_path(&self, path: &str) -> Result<()> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({
                "method": "delete_by_path",
                "params": { "path": path }
            }),
        )
        .await?;
        unwrap_response(resp).map(|_| ())
    }

    /// Retrieve all records from the master (used for full-codebase analysis).
    pub async fn get_all_records(&self) -> Result<Vec<Record>> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({ "method": "get_all_records", "params": {} }),
        )
        .await?;
        let result = unwrap_response(resp)?;
        serde_json::from_value(result["records"].clone()).context("invalid records response")
    }

    /// Enqueue a path for background indexing on the master.
    pub async fn index_project(&self, path: &str) -> Result<()> {
        let resp = rpc_call(
            &self.socket_path,
            &serde_json::json!({
                "method": "index_project",
                "params": { "path": path }
            }),
        )
        .await?;
        unwrap_response(resp).map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// Detection helper
// ---------------------------------------------------------------------------

/// Returns `true` if a master daemon is currently listening on `socket_path`.
pub async fn master_is_running(socket_path: &str) -> bool {
    tokio::net::UnixStream::connect(socket_path).await.is_ok()
}
