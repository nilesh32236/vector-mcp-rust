#![allow(dead_code)]
//! Tool call handlers for the Rust MCP server.
//! Provides high-performance implementations for codebase analysis,
//! semantic search, and local AI summarization.

use anyhow::Result;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::protocol::{CallToolParams, CallToolResult};
use super::server::Server;

/// Route a `tools/call` to the appropriate handler by tool name.
///
/// Only the 5 Fat Tool names are matched here. All legacy sub-handlers are
/// called internally by the Fat Tool dispatchers — they are never reachable
/// directly from the MCP protocol, preventing accidental invocation.
pub async fn dispatch(
    server: &Server,
    params: &CallToolParams,
    session_id: Option<&str>,
) -> Result<CallToolResult> {
    match params.name.as_str() {
        "search_workspace" => handle_search_workspace(server, params).await,
        "workspace_manager" => handle_workspace_manager(server, params, session_id).await,
        "analyze_code" => handle_analyze_code(server, params).await,
        "modify_workspace" => handle_modify_workspace(server, params).await,
        "lsp_query" => handle_lsp_query(server, params).await,
        _ => Ok(CallToolResult::error(format!(
            "Unknown tool '{}'. Available: search_workspace, workspace_manager, analyze_code, modify_workspace, lsp_query",
            params.name
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
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

fn optional_string_array_arg(params: &CallToolParams, key: &str) -> Option<Vec<String>> {
    params.arguments.get(key).and_then(|v| {
        if let Some(arr) = v.as_array() {
            Some(
                arr.iter()
                    .filter_map(|i| i.as_str().map(|s| s.to_string()))
                    .collect(),
            )
        } else {
            v.as_str().map(|s| {
                s.split(',')
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty())
                    .collect()
            })
        }
    })
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn handle_ping() -> CallToolResult {
    CallToolResult::text("pong")
}

async fn handle_trigger_project_index(
    server: &Server,
    params: &CallToolParams,
    session_id: Option<&str>,
) -> Result<CallToolResult> {
    let path = require_string_arg(params, "project_path")?;
    let store = Arc::clone(&server.store);
    let config = Arc::clone(&server.config);
    let embedder = Arc::clone(&server.embedder);
    let summarizer = Arc::clone(&server.summarizer);
    let progress = Arc::clone(&server.indexing_progress);

    // Build a progress channel if we have an SSE session to push to.
    let progress_tx = session_id.and_then(|sid| {
        server.progress_senders.get(sid).map(|tx| {
            let (scan_tx, mut scan_rx) =
                tokio::sync::mpsc::channel::<crate::indexer::scanner::ScanProgress>(32);
            let sse_tx = tx.clone();
            let sid = sid.to_string();
            tokio::spawn(async move {
                while let Some(p) = scan_rx.recv().await {
                    let notification = serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "$/progress",
                        "params": {
                            "progressToken": p.progress_token,
                            "progress": p.progress,
                            "total": p.total,
                            "currentFile": p.current_file,
                            "status": p.status,
                        }
                    });
                    if sse_tx.send(notification).is_err() {
                        tracing::warn!(session = %sid, "SSE client disconnected during indexing");
                        break;
                    }
                }
            });
            scan_tx
        })
    });

    let path_clone = path.clone();
    tokio::spawn(async move {
        let _ = crate::indexer::scanner::scan_project(
            config,
            store,
            embedder,
            summarizer,
            progress,
            progress_tx,
        )
        .await;
        tracing::info!("Indexing complete for {}", path_clone);
    });

    Ok(CallToolResult::text(format!(
        "Indexing initiated for {path}"
    )))
}

async fn handle_set_project_root(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let path = require_string_arg(params, "project_path")?;

    {
        let mut root = server.config.project_root.write().unwrap();
        *root = path.clone();
    }

    // Send signal to reload the watcher
    if let Err(e) = server.reload_watcher_tx.send(path.clone()).await {
        tracing::warn!("Failed to send watcher reload signal: {}", e);
    }

    Ok(CallToolResult::text(format!(
        "Project root updated to {}",
        path
    )))
}

async fn handle_store_context(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let text = require_string_arg(params, "text")?;
    let project_id = optional_string_arg(params, "project_id")
        .unwrap_or_else(|| server.config.project_root.read().unwrap().clone());

    server
        .store
        .store_project_context(&project_id, &text)
        .await?;

    Ok(CallToolResult::text(format!(
        "Context stored successfully for project {}: {} chars",
        project_id,
        text.len()
    )))
}

async fn handle_get_related_context(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let file_path = require_string_arg(params, "filePath")?;

    // 0. Fetch stored project context
    let project_id = server.config.project_root.read().unwrap().clone();
    let project_contexts = server
        .store
        .get_project_context(&project_id)
        .await
        .unwrap_or_default();

    // 1. Fetch records for the target file
    let records = server.store.get_all_records().await?;
    let target_records: Vec<_> = records
        .iter()
        .filter(|r| r.metadata_str("path") == file_path)
        .collect();

    if target_records.is_empty() {
        return Ok(CallToolResult::text(format!(
            "No indexed context found for {}",
            file_path
        )));
    }

    let mut out = format!("<context>\n  <file path=\"{}\">\n", file_path);
    let mut symbols = HashSet::new();
    let mut relations = HashSet::new();

    for r in &target_records {
        let meta = r.metadata_json();
        if let Some(s) = meta["symbols"].as_array() {
            for sym in s {
                if let Some(st) = sym.as_str() {
                    symbols.insert(st.to_string());
                }
            }
        }
        if let Some(rel) = meta["relationships"].as_array() {
            for rpath in rel {
                if let Some(rt) = rpath.as_str() {
                    relations.insert(rt.to_string());
                }
            }
        }
        out.push_str(&format!(
            "    <code_chunk>\n{}\n    </code_chunk>\n",
            r.content
        ));
    }
    out.push_str("  </file>\n");

    // 2. Resolve and Fetch Dependencies
    if !relations.is_empty() {
        out.push_str("  <dependencies>\n");
        for rel in relations {
            let dep_records: Vec<_> = records
                .iter()
                .filter(|r| r.metadata_str("path").contains(&rel))
                .collect();
            if !dep_records.is_empty() {
                out.push_str(&format!(
                    "    <file path=\"{}\" resolved_from=\"{}\">\n",
                    rel, rel
                ));
                for dr in dep_records {
                    out.push_str(&format!(
                        "      <code_chunk>\n{}\n      </code_chunk>\n",
                        dr.content
                    ));
                }
                out.push_str("    </file>\n");
            }
        }
        out.push_str("  </dependencies>\n");
    }

    // 3. Usage Samples for Symbols
    if !symbols.is_empty() {
        out.push_str("  <usage_samples>\n");
        for sym in symbols {
            let mut found = 0;
            for r in &records {
                if r.metadata_str("path") == file_path {
                    continue;
                }
                if r.content.contains(&sym) {
                    out.push_str(&format!(
                        "    <sample symbol=\"{}\" used_in=\"{}\">\n{}\n    </sample>\n",
                        sym,
                        r.metadata_str("path"),
                        r.content
                    ));
                    found += 1;
                }
                if found >= 2 {
                    break;
                }
            }
        }
        out.push_str("  </usage_samples>\n");
    }

    if !project_contexts.is_empty() {
        out.push_str("  <project_instructions>\n");
        for ctx in project_contexts {
            out.push_str(&format!("    <instruction>\n{}\n    </instruction>\n", ctx));
        }
        out.push_str("  </project_instructions>\n");
    }

    out.push_str("</context>");
    Ok(CallToolResult::text(out))
}

async fn handle_find_duplicate_code(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    let records = server.store.get_all_records().await?;
    let mut found = false;
    let mut out = format!("## Duplicate Code Analysis for {}\n\n", target_path);

    for r in records {
        if r.metadata_str("path") == target_path {
            let vector = r.vector.clone();
            let matches = server
                .store
                .hybrid_search(vector, &r.content, 3, None)
                .await?;
            for m in matches {
                if m.metadata_str("path") != target_path {
                    out.push_str(&format!(
                        "- Possible duplicate in `{}`\n",
                        m.metadata_str("path")
                    ));
                    found = true;
                }
            }
        }
    }

    if !found {
        out.push_str("No duplicates found.");
    }
    Ok(CallToolResult::text(out))
}

async fn handle_delete_context(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    server.store.delete_by_path(&target_path).await?;
    Ok(CallToolResult::text(format!(
        "Deleted context for {target_path}"
    )))
}

async fn handle_index_status(server: &Server) -> Result<CallToolResult> {
    let count = server.store.code_vectors.count_rows(None).await?;
    let records = server.store.get_all_records().await?;

    let root = server.config.project_root.read().unwrap().clone();
    let walker = ignore::WalkBuilder::new(&root)
        .standard_filters(true)
        .hidden(true)
        .add_custom_ignore_filename(".vector-ignore")
        .build();

    let mut disk_files = std::collections::HashSet::new();
    let mut modified = 0;
    let mut missing = 0;

    // Build map of indexed paths and their update timestamps
    let mut indexed_files: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for r in &records {
        let meta = r.metadata_json();
        if let Some(path) = meta["path"].as_str() {
            let updated_at_str = meta["updated_at"].as_str().unwrap_or("0");
            let updated_at = updated_at_str.parse::<u64>().unwrap_or(0);

            // Keep the newest timestamp for a given file
            let entry = indexed_files.entry(path.to_string()).or_insert(updated_at);
            if updated_at > *entry {
                *entry = updated_at;
            }
        }
    }

    for result in walker {
        if let Ok(entry) = result
            && entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
        {
            let path_buf = entry.path();
            let path_str = path_buf.to_string_lossy().to_string();

            // Only consider files we actually try to index
            let ext = path_buf.extension().and_then(|e| e.to_str()).unwrap_or("");
            let is_source = matches!(
                ext,
                "go" | "rs" | "js" | "ts" | "tsx" | "php" | "py" | "md" | "pdf"
            );

            if !is_source {
                continue;
            }

            disk_files.insert(path_str.clone());

            let file_mtime = if let Ok(meta) = std::fs::metadata(path_buf) {
                if let Ok(mtime) = meta.modified() {
                    mtime
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                } else {
                    0
                }
            } else {
                0
            };

            if let Some(indexed_time) = indexed_files.get(&path_str) {
                if file_mtime > *indexed_time {
                    modified += 1;
                }
            } else {
                missing += 1;
            }
        }
    }

    let mut deleted = 0;
    for path in indexed_files.keys() {
        if !disk_files.contains(path) {
            deleted += 1;
        }
    }

    let out = format!(
        "## Index Sync Status\n\n        - **Total Indexed**: {} chunks\n        - **Modified**: {} files\n        - **Missing/New**: {} files\n        - **Deleted**: {} records correspond to removed files",
        count, modified, missing, deleted
    );

    Ok(CallToolResult::text(out))
}

async fn handle_get_codebase_skeleton(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let root = server.config.project_root.read().unwrap().clone();
    let target_path = optional_string_arg(params, "target_path").unwrap_or_else(|| ".".into());
    let max_depth = optional_f64_arg(params, "max_depth").unwrap_or(3.0) as usize;
    let max_items = optional_f64_arg(params, "max_items").unwrap_or(1000.0) as usize;

    let abs_path = if std::path::Path::new(&target_path).is_absolute() {
        std::path::PathBuf::from(target_path)
    } else {
        std::path::Path::new(&root).join(target_path)
    };

    if !abs_path.exists() {
        return Ok(CallToolResult::error("Invalid path"));
    }

    let mut out = format!("Directory Tree: {:?} (Depth: {})\n", abs_path, max_depth);
    let mut item_count = 0;

    fn walk_dir(
        path: &std::path::Path,
        depth: usize,
        max_depth: usize,
        item_count: &mut usize,
        max_items: usize,
        out: &mut String,
    ) {
        if depth > max_depth || *item_count >= max_items {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.file_name());
            for entry in entries {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name == "node_modules" || name == ".git" || name == "target" {
                    continue;
                }
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                out.push_str(&format!("{}├── {}\n", "│   ".repeat(depth), name));
                *item_count += 1;
                if is_dir && depth < max_depth && *item_count < max_items {
                    walk_dir(
                        &entry.path(),
                        depth + 1,
                        max_depth,
                        item_count,
                        max_items,
                        out,
                    );
                }
            }
        }
    }
    walk_dir(
        &abs_path,
        0,
        max_depth,
        &mut item_count,
        max_items,
        &mut out,
    );
    Ok(CallToolResult::text(out))
}

async fn handle_check_dependency_health(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let dir_path = require_string_arg(params, "directory_path")?;
    let abs_path =
        std::path::Path::new(&*server.config.project_root.read().unwrap()).join(&dir_path);
    let mut declared_deps = HashSet::new();
    let mut project_type = "unknown";

    // 1. Detect project type and load declared dependencies
    if abs_path.join("package.json").exists() {
        project_type = "npm";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("package.json"))
            && let Ok(v) = serde_json::from_str::<Value>(&content)
        {
            if let Some(deps) = v["dependencies"].as_object() {
                for d in deps.keys() {
                    declared_deps.insert(d.clone());
                }
            }
            if let Some(dev) = v["devDependencies"].as_object() {
                for d in dev.keys() {
                    declared_deps.insert(d.clone());
                }
            }
        }
    } else if abs_path.join("go.mod").exists() {
        project_type = "go";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("go.mod")) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("require ") {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 {
                        declared_deps.insert(parts[1].to_string());
                    }
                }
            }
        }
    } else if abs_path.join("requirements.txt").exists() {
        project_type = "python";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("requirements.txt")) {
            let re = regex::Regex::new(r"^([a-zA-Z0-9_\-]+)")?;
            for line in content.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty()
                    && !trimmed.starts_with('#')
                    && let Some(cap) = re.captures(trimmed)
                {
                    declared_deps.insert(cap[1].to_string());
                }
            }
        }
    }

    if project_type == "unknown" {
        return Ok(CallToolResult::error(
            "Could not identify project type (no package.json, go.mod, or requirements.txt)",
        ));
    }

    // 2. Scan for used dependencies in indexed records
    let records = server.store.get_all_records().await?;
    let mut used_deps = HashSet::new();
    let dir_str = dir_path.replace('\\', "/");

    for r in records {
        let meta: Value = serde_json::from_str(&r.metadata).unwrap_or(Value::Null);
        let path = meta["path"].as_str().unwrap_or("");
        if !path.contains(&dir_str) {
            continue;
        }

        if let Some(rels) = meta["relationships"].as_array() {
            for rel in rels {
                if let Some(import_path) = rel.as_str() {
                    if project_type == "npm" {
                        if !import_path.starts_with('.') && !import_path.starts_with('/') {
                            let parts: Vec<&str> = import_path.split('/').collect();
                            if !parts.is_empty() {
                                let pkg = if parts[0].starts_with('@') && parts.len() > 1 {
                                    format!("{}/{}", parts[0], parts[1])
                                } else {
                                    parts[0].to_string()
                                };
                                used_deps.insert(pkg);
                            }
                        }
                    } else if project_type == "go" {
                        // Skip stdlib (heuristic: no dot in first path component)
                        if import_path.contains('.')
                            && !import_path.contains(&*server.config.project_root.read().unwrap())
                        {
                            used_deps.insert(import_path.to_string());
                        }
                    } else if project_type == "python" && !import_path.starts_with('.') {
                        used_deps.insert(import_path.to_string());
                    }
                }
            }
        }
    }

    // 3. Compare and report
    let missing: Vec<_> = used_deps
        .iter()
        .filter(|d| !declared_deps.contains(*d))
        .collect();
    let unused: Vec<_> = declared_deps
        .iter()
        .filter(|d| !used_deps.contains(*d) && !d.contains("types") && !d.contains("eslint"))
        .collect();

    let mut out = format!("## Dependency Health Report ({})\n\n", project_type);
    out.push_str(&format!("- Directory: `{}`\n", dir_path));
    out.push_str(&format!("- Declared: {}\n", declared_deps.len()));
    out.push_str(&format!("- Actually Used: {}\n\n", used_deps.len()));

    if !missing.is_empty() {
        out.push_str("### ⚠️ Missing in Manifest (Used but not declared)\n");
        for m in &missing {
            out.push_str(&format!("- `{}`\n", m));
        }
        out.push('\n');
    }

    if !unused.is_empty() {
        out.push_str("### ℹ️ Potentially Unused (Declared but not seen in imports)\n");
        for u in &unused {
            out.push_str(&format!("- `{}`\n", u));
        }
    }

    if missing.is_empty() && unused.is_empty() {
        out.push_str(
            "✅ All declared dependencies are used, and no missing dependencies were found.",
        );
    }

    Ok(CallToolResult::text(out))
}

async fn handle_generate_docstring_prompt(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let name = require_string_arg(params, "entity_name")?;
    let vector = server.embedder.embed_text(&name)?;
    let records = server.store.hybrid_search(vector, &name, 1, None).await?;
    if let Some(r) = records.first() {
        Ok(CallToolResult::text(format!(
            "Generate documentation for:\n\n{}",
            r.content
        )))
    } else {
        Ok(CallToolResult::error("Entity not found"))
    }
}

async fn handle_analyze_architecture(
    server: &Server,
    _params: &CallToolParams,
) -> Result<CallToolResult> {
    let vector = server.embedder.embed_text("system architecture design")?;
    let records = server
        .store
        .hybrid_search(vector, "architecture", 5, None)
        .await?;
    let mut combined = String::new();
    for r in records {
        combined.push_str(&r.content);
        combined.push('\n');
    }
    let summary = server
        .summarizer
        .summarize_chunk(&combined, Arc::clone(&server.config))
        .await?;
    Ok(CallToolResult::text(format!(
        "## Architecture Overview\n\n{}",
        summary
    )))
}

async fn handle_find_dead_code(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let is_library = params
        .arguments
        .get("is_library")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let records = server.store.get_all_records().await?;
    let mut exports = HashMap::new();
    let mut usages = HashSet::new();

    for r in records {
        let path = r.metadata_str("path");
        if path.contains("test") || path.contains("spec") {
            continue;
        }
        if is_library && !path.contains("internal") && !path.contains("private") {
            continue;
        }

        let meta = r.metadata_json();
        if let Some(syms) = meta["symbols"].as_array() {
            for s in syms {
                if let Some(st) = s.as_str() {
                    exports.insert(st.to_string(), path.to_string());
                }
            }
        }
        if let Some(calls) = meta["calls"].as_array() {
            for c in calls {
                if let Some(st) = c.as_str() {
                    usages.insert(st.to_string());
                }
            }
        }
    }

    let mut dead = Vec::new();
    for (name, path) in exports {
        let base_name = name.split('.').next_back().unwrap_or(&name);
        if !usages.contains(&name) && !usages.contains(base_name) {
            dead.push((name, path));
        }
    }

    if dead.is_empty() {
        return Ok(CallToolResult::text("✅ No dead code found."));
    }
    let mut out = String::from("## 🔎 Potential Dead Code Report\n\n");
    for (n, p) in dead {
        out.push_str(&format!("- `{}` in `{}`\n", n, p));
    }
    Ok(CallToolResult::text(out))
}

fn handle_filesystem_grep(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?.to_lowercase();
    let root = server.config.project_root.read().unwrap().clone();
    let mut results = Vec::new();

    let walker = ignore::WalkBuilder::new(&root)
        .standard_filters(true)
        .hidden(true)
        .add_custom_ignore_filename(".vector-ignore")
        .build();

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(entry.path()) {
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&query) {
                    results.push(format!(
                        "{}:{}: {}",
                        entry.path().display(),
                        i + 1,
                        line.trim()
                    ));
                }
            }
        }
        if results.len() > 50 {
            break;
        }
    }
    Ok(CallToolResult::text(results.join("\n")))
}

async fn handle_search_codebase(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let top_k = optional_f64_arg(params, "topK").unwrap_or(10.0) as usize;

    let vector = server.embedder.embed_text(&query)?;

    // 1. Fetch more results than needed for reranking (e.g. top_k * 3)
    let records = server
        .store
        .hybrid_search(vector, &query, top_k * 3, None)
        .await?;
    if records.is_empty() {
        return Ok(CallToolResult::text("No matches found."));
    }

    let mut final_records = records;

    // 2. Perform Reranking if enabled
    if server.embedder.rerank_session.is_some() {
        let docs: Vec<String> = final_records.iter().map(|r| r.content.clone()).collect();
        if let Ok(scores) = server.embedder.rerank(&query, docs) {
            let mut scored_records: Vec<_> = final_records.into_iter().zip(scores).collect();
            // Higher score is better
            scored_records
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            final_records = scored_records
                .into_iter()
                .map(|(r, _)| r)
                .take(top_k)
                .collect();
        }
    } else {
        final_records.truncate(top_k);
    }

    let max_tokens = optional_f64_arg(params, "max_tokens").unwrap_or(10000.0) as usize;
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    let mut total_tokens = 0;
    let mut out = String::new();

    for r in final_records {
        let meta = r.metadata_json();
        let path = meta["path"].as_str().unwrap_or("unknown_path");
        let start_line = meta["start_line"].as_u64().unwrap_or(0);
        let end_line = meta["end_line"].as_u64().unwrap_or(0);

        let mut symbols = String::new();
        if let Some(syms) = meta["symbols"].as_array() {
            let names: Vec<&str> = syms.iter().filter_map(|s| s.as_str()).collect();
            symbols = names.join(", ");
        }
        if symbols.is_empty() {
            symbols = "None".to_string();
        }

        let summary = r.metadata_str("summary");

        let chunk_text = format!(
            "#### [{}] (Lines {}-{})
- **Entities**: `{}`
- **Summary**: {}
{}

",
            path, start_line, end_line, symbols, summary, r.content
        );

        let tokens = bpe.encode_with_special_tokens(&chunk_text).len();
        if total_tokens + tokens > max_tokens && total_tokens > 0 {
            out.push_str(
                "... (truncating further results to stay within context window)
",
            );
            break;
        }

        out.push_str(&chunk_text);
        total_tokens += tokens;
    }

    if out.is_empty() {
        out = "No matches found within the token limit.".to_string();
    }

    Ok(CallToolResult::text(out))
}

async fn handle_get_indexing_diagnostics(server: &Server) -> Result<CallToolResult> {
    let p = server.indexing_progress.read().unwrap();
    let status = if p.status.is_empty() {
        "idle".to_string()
    } else {
        p.status.clone()
    };
    let current_file = if p.current_file.is_empty() {
        "None".to_string()
    } else {
        p.current_file.clone()
    };
    let errors_str = if p.errors.is_empty() {
        "None".to_string()
    } else {
        format!("\n- {}", p.errors.join("\n- "))
    };

    let out = format!(
        "## Indexing Diagnostics\n\n- **Status**: {}\n- **Current File**: {}\n- **Files Indexed**: {}\n- **Recent Errors**: {}",
        status, current_file, p.indexed_files, errors_str
    );
    Ok(CallToolResult::text(out))
}

async fn handle_get_summarized_context(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let vector = server.embedder.embed_text(&query)?;
    let records = server.store.hybrid_search(vector, &query, 5, None).await?;
    let mut text = String::new();
    for r in records {
        text.push_str(&r.content);
        text.push('\n');
    }
    let summary = server
        .summarizer
        .summarize_chunk(&text, Arc::clone(&server.config))
        .await?;
    Ok(CallToolResult::text(format!("### Summary\n\n{}", summary)))
}

async fn handle_verify_implementation_gap(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let vector = server.embedder.embed_text(&query)?;
    let records = server.store.hybrid_search(vector, &query, 10, None).await?;
    let mut out = format!("## Implementation Gap Analysis for '{}'\n\n", query);
    for r in records {
        let cat = r.metadata_str("category");
        let cat_display = if cat.is_empty() { "code" } else { &cat };
        out.push_str(&format!(
            "- [{}] `{}`\n",
            cat_display,
            r.metadata_str("path")
        ));
    }
    Ok(CallToolResult::text(out))
}

async fn handle_find_missing_tests(
    server: &Server,
    _params: &CallToolParams,
) -> Result<CallToolResult> {
    let records = server.store.get_all_records().await?;
    let mut exports = HashMap::new();
    let mut usages = HashSet::new();

    for r in records {
        let path = r.metadata_str("path");
        if path.contains("test") {
            for word in r.content.split_whitespace() {
                usages.insert(word.to_string());
            }
        } else {
            let meta = r.metadata_json();
            if let Some(syms) = meta["symbols"].as_array() {
                for s in syms {
                    if let Some(st) = s.as_str() {
                        exports.insert(st.to_string(), path.to_string());
                    }
                }
            }
        }
    }

    let mut missing = Vec::new();
    for (name, path) in exports {
        if !usages.contains(&name) {
            missing.push((name, path));
        }
    }
    if missing.is_empty() {
        return Ok(CallToolResult::text("All exports tested."));
    }
    let mut out = String::from("## Missing Tests\n\n");
    for (n, p) in missing {
        out.push_str(&format!("- `{}` in `{}`\n", n, p));
    }
    Ok(CallToolResult::text(out))
}

async fn handle_list_api_endpoints(
    server: &Server,
    _params: &CallToolParams,
) -> Result<CallToolResult> {
    let keywords = ["HandleFunc", "app.GET", "app.POST", "Route("];
    let mut out = String::from("## API Endpoints\n\n");
    for kw in keywords {
        let vector = server.embedder.embed_text(kw)?;
        let records = server.store.hybrid_search(vector, kw, 5, None).await?;
        for r in records {
            out.push_str(&format!(
                "- `{}`: `{}`\n",
                r.metadata_str("path"),
                r.content.lines().next().unwrap_or("")
            ));
        }
    }
    Ok(CallToolResult::text(out))
}

async fn handle_get_code_history(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let path = require_string_arg(params, "file_path")?;
    let output = std::process::Command::new("git")
        .args(["log", "-n", "5", "--pretty=format:%h - %s", "--", &path])
        .current_dir(&*server.config.project_root.read().unwrap())
        .output()?;
    Ok(CallToolResult::text(
        String::from_utf8_lossy(&output.stdout).to_string(),
    ))
}

async fn handle_reindex_all(
    server: &Server,
    _params: &CallToolParams,
    session_id: Option<&str>,
) -> Result<CallToolResult> {
    let config = Arc::clone(&server.config);
    let store = Arc::clone(&server.store);
    let embedder = Arc::clone(&server.embedder);
    let summarizer = Arc::clone(&server.summarizer);
    let progress = Arc::clone(&server.indexing_progress);

    let progress_tx = session_id.and_then(|sid| {
        server.progress_senders.get(sid).map(|tx| {
            let (scan_tx, mut scan_rx) =
                tokio::sync::mpsc::channel::<crate::indexer::scanner::ScanProgress>(32);
            let sse_tx = tx.clone();
            let sid = sid.to_string();
            tokio::spawn(async move {
                while let Some(p) = scan_rx.recv().await {
                    let notification = serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "$/progress",
                        "params": {
                            "progressToken": p.progress_token,
                            "progress": p.progress,
                            "total": p.total,
                            "status": p.status,
                        }
                    });
                    if sse_tx.send(notification).is_err() {
                        tracing::warn!(session = %sid, "SSE client disconnected during reindex");
                        break;
                    }
                }
            });
            scan_tx
        })
    });

    tokio::spawn(async move {
        let _ = crate::indexer::scanner::scan_project(
            config,
            store,
            embedder,
            summarizer,
            progress,
            progress_tx,
        )
        .await;
    });

    Ok(CallToolResult::text(
        "Full project re-indexing initiated in background.",
    ))
}
async fn handle_check_llm_connectivity(server: &Server) -> Result<CallToolResult> {
    let client = crate::llm::gemini::GeminiClient::new(&server.config);
    match client.list_models().await {
        Ok(models) => {
            let mut out = String::from(
                "### ✅ LLM Connectivity Status\n\n**Status**: Connected Successfully\n",
            );
            out.push_str(&format!(
                "**Configured Default**: `{}`\n\n",
                server.config.default_gemini_model
            ));
            out.push_str("**Available Models**:\n");
            for m in models {
                out.push_str(&format!(
                    "- `{}` ({}): {}\n",
                    m.name, m.display_name, m.description
                ));
            }
            Ok(CallToolResult::text(out))
        }
        Err(e) => Ok(CallToolResult::text(format!(
            "### ❌ LLM Connectivity Status\n\n**Status**: API Error\n**Error**: {e}\n\n**Troubleshooting**:\n1. Verify your GEMINI_API_KEY is correct.\n2. Ensure your key has permissions for the Generative Language API."
        ))),
    }
}

async fn handle_distill_knowledge(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let path_prefix = require_string_arg(params, "path")?;
    let records = server.store.get_records_by_path(&path_prefix).await?;

    if records.is_empty() {
        return Ok(CallToolResult::error(format!(
            "No indexed content found for path: {path_prefix}"
        )));
    }

    let mut relevant_content = String::new();
    let mut count = 0;
    for r in records {
        if r.metadata_str("path").starts_with(&path_prefix) {
            relevant_content.push_str(&format!(
                "File: {}\nContent:\n{}\n---\n",
                r.metadata_str("path"),
                r.content
            ));
            count += 1;
        }
    }

    if count == 0 {
        return Ok(CallToolResult::error(format!(
            "No indexed content found starting with: {path_prefix}"
        )));
    }

    let system_prompt = "You are a Senior Software Architect. Your task is to analyze the provided source code and \"distill\" it into a set of architectural patterns, coding standards, and business rules.\nFormat the output as a high-quality Markdown Knowledge Item (KI) including:\n1. Overview: What is this component/module for?\n2. Key Architectural Decisions: Why was it built this way?\n3. Implementation Rules: Mandatory patterns for anyone modifying this code.\n4. Security/Compliance: PHI handling, encryption rules, etc. (if applicable).\nKeep it concise and actionable.";
    let user_prompt = format!(
        "Analyze these files from path '{}':\n\n{}",
        path_prefix, relevant_content
    );

    let client = crate::llm::gemini::GeminiClient::new(&server.config);
    let distilled = client
        .generate_completion(
            &server.config.default_gemini_model,
            system_prompt,
            &user_prompt,
        )
        .await?;

    // Store distilled knowledge
    let project_id = server.config.project_root.read().unwrap().clone();
    server
        .store
        .store_project_context(&project_id, &distilled)
        .await?;

    Ok(CallToolResult::text(format!(
        "### ✅ Knowledge Distilled & Indexed\n\n{distilled}"
    )))
}

async fn handle_verify_proposed_change(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let proposed_change = require_string_arg(params, "proposed_change")?;
    let project_id = server.config.project_root.read().unwrap().clone();

    // Retrieve distilled knowledge (KIs)
    let contexts = server.store.get_project_context(&project_id).await?;
    let ki_context = if contexts.is_empty() {
        "No Architectural Decisions or Knowledge Items found for this project.".to_string()
    } else {
        contexts.join("\n\n---\n\n")
    };

    let system_prompt = "You are a Senior Software Engineer performing a code review. Your task is to verify if a proposed change complies with the existing architectural decisions and knowledge items provided below.\n\nReply with a structured verification report identifying potential violations or confirming compliance.";
    let user_prompt = format!(
        "### Existing Knowledge Items:\n{}\n\n### Proposed Change:\n{}",
        ki_context, proposed_change
    );

    let client = crate::llm::gemini::GeminiClient::new(&server.config);
    let review = client
        .generate_completion(
            &server.config.default_gemini_model,
            system_prompt,
            &user_prompt,
        )
        .await?;

    Ok(CallToolResult::text(format!(
        "### 🔎 Proposed Change Verification\n\n{review}\n\n---\n*Verified against indexed Knowledge Items using {}.*",
        server.config.default_gemini_model
    )))
}

// ---------------------------------------------------------------------------
// Task 1: Action-based super-tools
// ---------------------------------------------------------------------------

/// `search_workspace` — unified vector / regex / graph / index_status dispatcher.
async fn handle_search_workspace(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let action = optional_string_arg(params, "action").unwrap_or_default();
    let query = optional_string_arg(params, "query").unwrap_or_default();
    let limit = optional_f64_arg(params, "limit").unwrap_or(10.0) as usize;
    let path_filter = optional_string_arg(params, "path");
    let cross_reference = optional_string_array_arg(params, "cross_reference_projects");

    match action.as_str() {
        "vector" | "" => {
            // Semantic search
            if query.is_empty() {
                return Ok(CallToolResult::error("query is required for vector search"));
            }
            let vector = server.embedder.embed_query(&query)?;
            let mut results = server
                .store
                .hybrid_search(vector, &query, limit * 3, cross_reference.as_deref())
                .await?;

            // Path filter
            if let Some(ref pf) = path_filter {
                results.retain(|r| r.metadata_str("path").contains(pf.as_str()));
            }
            results.truncate(limit);

            if results.is_empty() {
                return Ok(CallToolResult::text("No matches found."));
            }
            let mut out = format!("### Search Results for '{query}':\n\n");
            for r in results {
                let meta = r.metadata_json();
                let path = meta["path"].as_str().unwrap_or("?");
                let start = meta["start_line"].as_u64().unwrap_or(0);
                let end = meta["end_line"].as_u64().unwrap_or(0);
                out.push_str(&format!(
                    "#### {path} (Lines {start}-{end})\n```\n{}\n```\n\n",
                    r.content
                ));
            }
            Ok(CallToolResult::text(out))
        }
        "regex" => {
            // Delegate to filesystem_grep
            let mut grep_params = params.arguments.clone();
            grep_params["is_regex"] = serde_json::json!(true);
            if let Some(pf) = path_filter {
                grep_params["include_pattern"] = serde_json::json!(pf);
            }
            let synthetic = CallToolParams {
                name: "filesystem_grep".into(),
                arguments: grep_params,
            };
            handle_filesystem_grep(server, &synthetic)
        }
        "graph" => {
            // Interface implementations or symbol search via knowledge graph.
            if query.is_empty() {
                return Ok(CallToolResult::error("query is required for graph search"));
            }
            let impls = server.store.graph.get_implementations(&query);
            if !impls.is_empty() {
                let mut out = format!("Implementations of '{query}':\n");
                for n in impls {
                    out.push_str(&format!("- {} ({}) in {}\n", n.name, n.node_type, n.path));
                }
                return Ok(CallToolResult::text(out));
            }
            let nodes = server.store.graph.search_by_name(&query);
            if nodes.is_empty() {
                return Ok(CallToolResult::text(format!(
                    "No graph entries found for '{query}'."
                )));
            }
            let mut out = format!("Graph results for '{query}':\n");
            for n in nodes.iter().take(limit) {
                out.push_str(&format!("- {} ({}) in {}\n", n.name, n.node_type, n.path));
            }
            Ok(CallToolResult::text(out))
        }
        "index_status" => handle_index_status(server).await,
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: vector, regex, index_status"
        ))),
    }
}

/// `workspace_manager` — set_project_root / trigger_index / get_indexing_diagnostics.
async fn handle_workspace_manager(
    server: &Server,
    params: &CallToolParams,
    session_id: Option<&str>,
) -> Result<CallToolResult> {
    let action = optional_string_arg(params, "action").unwrap_or_default();
    let path = optional_string_arg(params, "path");

    match action.as_str() {
        "set_project_root" => {
            let p = path.ok_or_else(|| anyhow::anyhow!("path is required"))?;
            let synthetic = CallToolParams {
                name: "set_project_root".into(),
                arguments: serde_json::json!({ "project_path": p }),
            };
            handle_set_project_root(server, &synthetic).await
        }
        "trigger_index" => {
            let p = path.unwrap_or_else(|| server.config.project_root.read().unwrap().clone());
            let synthetic = CallToolParams {
                name: "trigger_project_index".into(),
                arguments: serde_json::json!({ "project_path": p }),
            };
            handle_trigger_project_index(server, &synthetic, session_id).await
        }
        "get_indexing_diagnostics" => handle_get_indexing_diagnostics(server).await,
        "store_context" => handle_store_context(server, params).await,
        "delete_context" => handle_delete_context(server, params).await,
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: set_project_root, trigger_index, get_indexing_diagnostics, store_context, delete_context"
        ))),
    }
}

/// `analyze_code` — ast_skeleton / dead_code / duplicate_code / dependencies.
async fn handle_analyze_code(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let action = optional_string_arg(params, "action").unwrap_or_default();
    let path = optional_string_arg(params, "path").unwrap_or_else(|| ".".into());

    match action.as_str() {
        "ast_skeleton" => {
            let synthetic = CallToolParams {
                name: "get_codebase_skeleton".into(),
                arguments: serde_json::json!({ "target_path": path }),
            };
            handle_get_codebase_skeleton(server, &synthetic).await
        }
        "dead_code" => handle_find_dead_code(server, params).await,
        "duplicate_code" => {
            let synthetic = CallToolParams {
                name: "find_duplicate_code".into(),
                arguments: serde_json::json!({ "target_path": path }),
            };
            handle_find_duplicate_code(server, &synthetic).await
        }
        "dependencies" => {
            let synthetic = CallToolParams {
                name: "check_dependency_health".into(),
                arguments: serde_json::json!({ "directory_path": path }),
            };
            handle_check_dependency_health(server, &synthetic).await
        }
        "distill_package" => {
            let synthetic = CallToolParams {
                name: "distill_package_purpose".into(),
                arguments: serde_json::json!({ "path": path }),
            };
            handle_distill_package_purpose(server, &synthetic).await
        }
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: ast_skeleton, dead_code, duplicate_code, dependencies, distill_package"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Task 4: Mutation tools (modify_workspace)
// ---------------------------------------------------------------------------

/// `modify_workspace` — apply_patch / create_file / run_linter / verify_patch.
async fn handle_modify_workspace(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let action = optional_string_arg(params, "action").unwrap_or_default();

    match action.as_str() {
        "apply_patch" => handle_apply_patch(server, params).await,
        "create_file" => handle_create_file(server, params),
        "run_linter" => handle_run_linter(server, params).await,
        "verify_patch" => handle_verify_patch(server, params),
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: apply_patch, create_file, run_linter, verify_patch"
        ))),
    }
}

/// Apply a search-and-replace patch with LSP diagnostic verification.
///
/// If an LSP server is available for the file's language, the patch is first
/// applied in-memory and verified via `textDocument/publishDiagnostics`. The
/// patch is rejected if the LSP reports severity-1 (Error) diagnostics.
async fn handle_apply_patch(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "path")?;
    let search = require_string_arg(params, "search")?;
    let replace = optional_string_arg(params, "replace").unwrap_or_default();

    let abs = server
        .path_guard
        .validate(&path)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    let abs_str = abs.to_string_lossy().to_string();

    // --- LSP safety verification (if a server is available for this language) ---
    if let Some(lsp) = server.lsp_pool.get_for_path(&abs_str) {
        match crate::mutation::SafetyChecker::verify_patch(&lsp, &abs_str, &search, &replace).await
        {
            Ok(diags) if crate::mutation::SafetyChecker::has_errors(&diags) => {
                let report = crate::mutation::SafetyChecker::format_diagnostics(&diags);
                return Ok(CallToolResult::error(format!(
                    "Patch rejected — LSP reported compiler errors:\n{report}"
                )));
            }
            Ok(diags) if !diags.is_empty() => {
                // Warnings only — proceed but surface them.
                let report = crate::mutation::SafetyChecker::format_diagnostics(&diags);
                tracing::warn!(path = %path, "LSP warnings on patch: {report}");
            }
            Err(e) => {
                // LSP unavailable or timed out — fall through and apply anyway.
                tracing::warn!(path = %path, "LSP verification skipped: {e}");
            }
            _ => {} // No diagnostics — safe to proceed.
        }
    }

    // --- Apply patch to disk ---
    let content = tokio::fs::read_to_string(&abs)
        .await
        .map_err(|e| anyhow::anyhow!("read failed: {e}"))?;

    if !content.contains(&search) {
        return Ok(CallToolResult::error("search string not found in file"));
    }

    let new_content = content.replacen(&search, &replace, 1);
    tokio::fs::write(&abs, new_content)
        .await
        .map_err(|e| anyhow::anyhow!("write failed: {e}"))?;

    Ok(CallToolResult::text(format!("✅ Patched {path}")))
}

fn handle_create_file(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "path")?;
    let content = optional_string_arg(params, "content").unwrap_or_default();

    let abs = server
        .path_guard
        .validate(&path)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| anyhow::anyhow!("mkdir failed: {e}"))?;
    }

    std::fs::write(&abs, content).map_err(|e| anyhow::anyhow!("write failed: {e}"))?;

    Ok(CallToolResult::text(format!("Created {path}")))
}

async fn handle_run_linter(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "path")?;
    let tool = optional_string_arg(params, "tool").unwrap_or_else(|| "fmt".into());

    let abs = server
        .path_guard
        .validate(&path)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    let (cmd, args): (&str, Vec<&str>) = match tool.as_str() {
        "go fmt" | "gofmt" => ("gofmt", vec!["-w", abs.to_str().unwrap_or("")]),
        "rustfmt" => ("rustfmt", vec![abs.to_str().unwrap_or("")]),
        "prettier" => ("prettier", vec!["--write", abs.to_str().unwrap_or("")]),
        other => return Ok(CallToolResult::error(format!("Unsupported tool: {other}"))),
    };

    let output = tokio::process::Command::new(cmd)
        .args(&args)
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("{cmd} not found: {e}"))?;

    if output.status.success() {
        Ok(CallToolResult::text(format!("{tool} applied to {path}")))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(CallToolResult::error(format!("{tool} failed: {stderr}")))
    }
}

fn handle_verify_patch(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "path")?;
    let search = require_string_arg(params, "search")?;

    let abs = server
        .path_guard
        .validate(&path)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    let content = std::fs::read_to_string(&abs).map_err(|e| anyhow::anyhow!("read failed: {e}"))?;

    if content.contains(&search) {
        Ok(CallToolResult::text(format!(
            "✅ Search string found in {path} — patch is applicable"
        )))
    } else {
        Ok(CallToolResult::text(format!(
            "❌ Search string NOT found in {path} — patch would fail"
        )))
    }
}

// ---------------------------------------------------------------------------
// lsp_query — definition / references / type_hierarchy / impact_analysis
// ---------------------------------------------------------------------------

pub async fn handle_lsp_query(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let action = optional_string_arg(params, "action").unwrap_or_default();
    let path = optional_string_arg(params, "path").unwrap_or_default();
    let line = optional_f64_arg(params, "line").unwrap_or(0.0) as u32;
    let character = optional_f64_arg(params, "character").unwrap_or(0.0) as u32;

    if path.is_empty() {
        return Ok(CallToolResult::error("path is required"));
    }

    let ext = std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();

    let lsp = match server.lsp_pool.get(&ext) {
        Some(l) => l,
        None => {
            return Ok(CallToolResult::error(format!(
                "No LSP server for extension '{ext}'"
            )));
        }
    };

    let lsp_method = match action.as_str() {
        "definition" => "textDocument/definition",
        "references" => "textDocument/references",
        "type_hierarchy" => "textDocument/prepareTypeHierarchy",
        "impact_analysis" => "textDocument/references",
        _ => {
            return Ok(CallToolResult::error(
                "Invalid action. Use: definition, references, type_hierarchy, impact_analysis",
            ));
        }
    };

    let mut lsp_params = serde_json::json!({
        "textDocument": { "uri": format!("file://{path}") },
        "position": { "line": line, "character": character },
    });

    if action == "references" || action == "impact_analysis" {
        lsp_params["context"] = serde_json::json!({ "includeDeclaration": true });
    }

    // LSP calls are async — await directly.
    let result = lsp.call(lsp_method, lsp_params).await?;

    if action == "impact_analysis" {
        // Summarise blast radius from references list.
        let refs: usize = result.as_array().map(|a| a.len()).unwrap_or(0);
        let risk = match refs {
            0..=3 => "Low",
            4..=10 => "Medium",
            _ => "High",
        };
        return Ok(CallToolResult::text(format!(
            "### Impact Analysis\n- **Risk**: {risk}\n- **References**: {refs}\n\nRaw: {result}"
        )));
    }

    Ok(CallToolResult::text(format!("{result}")))
}

// ---------------------------------------------------------------------------
// trace_data_flow — graph-based symbol usage tracing
// ---------------------------------------------------------------------------

pub async fn handle_trace_data_flow(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let field = require_string_arg(params, "field_name")?;
    let nodes = server.store.graph.find_usage(&field);
    if nodes.is_empty() {
        return Ok(CallToolResult::text(format!(
            "No entities found using symbol '{field}'."
        )));
    }
    let mut out = format!("Entities using '{field}':\n");
    for n in nodes {
        out.push_str(&format!("- {} ({}) in {}\n", n.name, n.node_type, n.path));
        if !n.docstring.is_empty() {
            out.push_str(&format!("  Doc: {}\n", n.docstring));
        }
    }
    Ok(CallToolResult::text(out))
}

// ---------------------------------------------------------------------------
// distill_package_purpose — summarise a package via Gemini and re-index
// ---------------------------------------------------------------------------

pub async fn handle_distill_package_purpose(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let pkg_path = require_string_arg(params, "path")?;
    let records = server.store.get_records_by_path(&pkg_path).await?;

    if records.is_empty() {
        return Ok(CallToolResult::error(format!(
            "No indexed content for path: {pkg_path}"
        )));
    }

    let mut content = String::new();
    for r in &records {
        content.push_str(&format!(
            "File: {}\n{}\n---\n",
            r.metadata_str("path"),
            r.content
        ));
    }

    let system_prompt = "You are a Senior Software Architect. Analyse the provided source code and produce a concise Markdown summary covering: 1) Purpose, 2) Key architectural decisions, 3) Implementation rules. Be brief and actionable.";
    let user_prompt = format!("Package path: {pkg_path}\n\n{content}");

    let client = crate::llm::gemini::GeminiClient::new(&server.config);
    let summary = client
        .generate_completion(
            &server.config.default_gemini_model,
            system_prompt,
            &user_prompt,
        )
        .await?;

    // Re-index the distilled summary with high-priority metadata.
    let vector = server.embedder.embed_text(&summary)?;
    let metadata = serde_json::json!({
        "path": pkg_path,
        "type": "distilled_summary",
        "priority": "2.0",
        "project_id": server.config.project_root.read().unwrap().clone(),
    });
    let record = crate::db::Record {
        id: format!("distill-{}", uuid::Uuid::new_v4()),
        content: summary.clone(),
        vector,
        metadata: metadata.to_string(),
    };
    server.store.upsert_records(vec![record]).await?;

    Ok(CallToolResult::text(format!(
        "### ✅ Package Distilled\n\n**Path**: {pkg_path}\n\n{summary}\n\n*Re-indexed with 2.0x priority.*"
    )))
}
