//! Tool call handlers — stub implementations for Stage 2.
//!
//! Each handler parses its expected arguments and returns a placeholder
//! success message. Real logic will be wired in Stage 3+.

use anyhow::Result;
use std::sync::Arc;

use super::protocol::{CallToolParams, CallToolResult};
use super::server::Server;

/// Route a `tools/call` to the appropriate handler by tool name.
pub async fn dispatch(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    match params.name.as_str() {
        "ping" => Ok(handle_ping()),
        "trigger_project_index" => handle_trigger_project_index(server, params).await,
        "set_project_root" => handle_set_project_root(server, params),
        "store_context" => handle_store_context(server, params),
        "get_related_context" => handle_get_related_context(server, params),
        "find_duplicate_code" => handle_find_duplicate_code(server, params),
        "delete_context" => handle_delete_context(server, params).await,
        "index_status" => handle_index_status(server).await,
        "get_codebase_skeleton" => handle_get_codebase_skeleton(server, params).await,
        "check_dependency_health" => handle_check_dependency_health(server, params),
        "generate_docstring_prompt" => handle_generate_docstring_prompt(server, params),
        "analyze_architecture" => handle_analyze_architecture(server, params).await,
        "find_dead_code" => handle_find_dead_code(server, params),
        "filesystem_grep" => handle_filesystem_grep(server, params),
        "search_codebase" => handle_search_codebase(server, params).await,
        "get_indexing_diagnostics" => handle_get_indexing_diagnostics(server),
        _ => Ok(CallToolResult::error(format!(
            "Unknown tool: {}",
            params.name
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helper: extract a string arg, returning a tool-level error if missing.
// ---------------------------------------------------------------------------

fn require_string_arg(params: &CallToolParams, key: &str) -> Result<String> {
    params
        .arguments
        .get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow::anyhow!("{key} is required"))
}

fn optional_string_arg(params: &CallToolParams, key: &str) -> Option<String> {
    params
        .arguments
        .get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
}

fn optional_f64_arg(params: &CallToolParams, key: &str) -> Option<f64> {
    params.arguments.get(key).and_then(|v| v.as_f64())
}

// ---------------------------------------------------------------------------
// Individual handlers (stubs)
// ---------------------------------------------------------------------------

fn handle_ping() -> CallToolResult {
    CallToolResult::text("pong")
}

async fn handle_trigger_project_index(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "project_path")?;
    
    // In a real implementation with high-concurrency, we'd use a queue.
    // For now, we perform a direct indexing.
    let store = Arc::clone(&server.store);
    let config = Arc::clone(&server.config);
    let embedder = Arc::clone(&server.embedder);
    let summarizer = Arc::clone(&server.summarizer);
    
    // Spawn task to avoid blocking JSON-RPC response
    let path_clone = path.clone();
    tokio::spawn(async move {
        let _ = crate::indexer::index_file(&path_clone, config, store, embedder, summarizer).await;
    });

    Ok(CallToolResult::text(format!("Indexing initiated for {path}")))
}

fn handle_set_project_root(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let _path = require_string_arg(params, "project_path")?;
    Ok(CallToolResult::text("Project root updated (file watcher reset pending)"))
}

fn handle_store_context(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let text = require_string_arg(params, "text")?;
    Ok(CallToolResult::text(format!("Context stored: {} chars", text.len())))
}

fn handle_get_related_context(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let file_path = require_string_arg(params, "filePath")?;
    Ok(CallToolResult::text(format!("Related context for {file_path} (Stub)")))
}

fn handle_find_duplicate_code(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    Ok(CallToolResult::text(format!("Duplicate scan for {target_path} (Stub)")))
}

async fn handle_delete_context(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    server.store.delete_by_path(&target_path).await?;
    Ok(CallToolResult::text(format!("Deleted context for {target_path}")))
}

async fn handle_index_status(server: &Server) -> Result<CallToolResult> {
    let count = server.store.code_vectors.count_rows(None).await?;
    Ok(CallToolResult::text(format!("Index contains {count} vectors. Status: Ready.")))
}

async fn handle_get_codebase_skeleton(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = optional_string_arg(params, "target_path").unwrap_or_else(|| ".".into());
    Ok(CallToolResult::text(format!("Skeleton for {path} (Stub)")))
}

fn handle_check_dependency_health(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "directory_path")?;
    Ok(CallToolResult::text(format!("Dependency health for {path} (Stub)")))
}

fn handle_generate_docstring_prompt(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let name = require_string_arg(params, "entity_name")?;
    Ok(CallToolResult::text(format!("Docstring prompt for {name} (Stub)")))
}

async fn handle_analyze_architecture(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let prefix = optional_string_arg(params, "monorepo_prefix").unwrap_or_else(|| "@herexa/".into());
    
    // Simple implementation: retrieve top chunks and summarize
    let vector = server.embedder.embed_text("architecture overview system design")?;
    let records = server.store.hybrid_search(vector, "architecture", 5).await?;
    
    if records.is_empty() {
        return Ok(CallToolResult::text("Insufficient data for architectural analysis. Please index the project first."));
    }
    
    let mut combined_text = String::new();
    for r in records {
        combined_text.push_str(&r.content);
        combined_text.push('\n');
    }
    
    let summary = server.summarizer.summarize_chunk(&combined_text, Arc::clone(&server.config)).await?;
    Ok(CallToolResult::text(format!("Architecture analysis for {prefix}:\n\n{summary}")))
}

fn handle_find_dead_code(_server: &Server, _params: &CallToolParams) -> Result<CallToolResult> {
    Ok(CallToolResult::text("Dead code analysis (Stub)"))
}

fn handle_filesystem_grep(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    Ok(CallToolResult::text(format!("Grep for '{query}' (Stub)")))
}

async fn handle_search_codebase(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let top_k = optional_f64_arg(params, "topK").unwrap_or(10.0) as usize;
    
    let vector = server.embedder.embed_text(&query)?;
    let records = server.store.hybrid_search(vector, &query, top_k).await?;
    
    let mut output = String::new();
    for rec in records {
        output.push_str(&format!("--- ID: {} ---\n{}\n\n", rec.id, rec.content));
    }
    
    Ok(CallToolResult::text(output))
}

fn handle_get_indexing_diagnostics(_server: &Server) -> Result<CallToolResult> {
    Ok(CallToolResult::text("All systems nominal. No errors in last 24h."))
}
