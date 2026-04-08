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
use crate::security::pathguard::PathOp;

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
    // Update project_root so scan_project indexes the requested path.
    {
        let mut root = config.project_root.write().unwrap();
        *root = path.clone();
    }
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

async fn handle_find_duplicate_code(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    let records = server.store.get_records_by_path(&target_path).await?;

    if records.is_empty() {
        return Ok(CallToolResult::text(format!(
            "No indexed content found for: {target_path}"
        )));
    }

    // Combine all chunk content and embed once instead of N separate searches.
    let combined: String = records
        .iter()
        .map(|r| r.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let vector = server.embedder.embed_query(&combined)?;
    let matches = server
        .store
        .hybrid_search(vector, &combined, 10, None)
        .await?;

    let mut out = format!("## Duplicate Code Analysis for {target_path}\n\n");
    let mut found = false;
    for m in matches {
        if m.metadata_str("path") != target_path {
            out.push_str(&format!(
                "- Possible duplicate in `{}`\n",
                m.metadata_str("path")
            ));
            found = true;
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
    let target_path = optional_string_arg(params, "target_path").unwrap_or_else(|| ".".into());
    let max_depth = optional_f64_arg(params, "max_depth").unwrap_or(3.0) as usize;
    let max_items = optional_f64_arg(params, "max_items").unwrap_or(1000.0) as usize;
    let include_pattern = optional_string_arg(params, "include_pattern");
    let exclude_pattern = optional_string_arg(params, "exclude_pattern");

    // Security enhancement: Prevent path traversal by using path_guard
    let abs_path = match server.path_guard.validate(&target_path, PathOp::Read) {
        Ok(p) => p,
        Err(e) => return Ok(CallToolResult::error(format!("Invalid path: {e}"))),
    };

    let mut out = format!("Directory Tree: {:?} (Depth: {})\n", abs_path, max_depth);

    // Build a map of path -> depth using WalkBuilder (respects .gitignore / .vector-ignore).
    let mut walker_builder = ignore::WalkBuilder::new(&abs_path);
    walker_builder
        .standard_filters(true)
        .hidden(true)
        .add_custom_ignore_filename(".vector-ignore")
        .max_depth(Some(max_depth + 1));

    let walker = walker_builder.build();

    let mut entries: Vec<(std::path::PathBuf, usize)> = Vec::new();
    for entry in walker.flatten() {
        let depth = entry.depth();
        if depth == 0 {
            continue; // skip root itself
        }
        let path = entry.path().to_path_buf();

        // Apply optional include/exclude glob patterns.
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        if let Some(ref pat) = exclude_pattern {
            if name.contains(pat.as_str()) {
                continue;
            }
        }
        if let Some(ref pat) = include_pattern {
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if !is_dir && !name.contains(pat.as_str()) {
                continue;
            }
        }

        entries.push((path, depth - 1));
        if entries.len() >= max_items {
            break;
        }
    }

    for (path, depth) in &entries {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        out.push_str(&format!("{}├── {}\n", "│   ".repeat(*depth), name));
    }

    Ok(CallToolResult::text(out))
}

async fn handle_check_dependency_health(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let dir_path = require_string_arg(params, "directory_path")?;
    let root = server.config.project_root.read().unwrap().clone();
    // Security enhancement: Prevent path traversal by using path_guard
    let abs_path = match server.path_guard.validate(&dir_path, PathOp::Read) {
        Ok(p) => p,
        Err(e) => return Ok(CallToolResult::error(format!("Invalid path: {e}"))),
    };

    // 1. Detect project type and parse manifest
    let (project_type, declared_deps) = {
        let npm_manifest = abs_path.join("package.json");
        let go_manifest = abs_path.join("go.mod");
        let py_manifest = abs_path.join("requirements.txt");

        if npm_manifest.exists() {
            let mut deps = HashSet::new();
            if let Ok(content) = std::fs::read_to_string(&npm_manifest)
                && let Ok(v) = serde_json::from_str::<Value>(&content)
            {
                for key in ["dependencies", "devDependencies"] {
                    if let Some(obj) = v[key].as_object() {
                        deps.extend(obj.keys().cloned());
                    }
                }
            }
            ("npm", deps)
        } else if go_manifest.exists() {
            let mut deps = HashSet::new();
            if let Ok(content) = std::fs::read_to_string(&go_manifest) {
                for line in content.lines() {
                    let t = line.trim();
                    if t.is_empty() || t.starts_with("//") {
                        continue;
                    }
                    let parts: Vec<&str> = t.split_whitespace().collect();
                    if parts.len() >= 2 && parts[0] != "module" && parts[0] != "go" {
                        deps.insert(parts[0].to_string());
                    }
                }
            }
            ("go", deps)
        } else if py_manifest.exists() {
            let mut deps = HashSet::new();
            if let Ok(content) = std::fs::read_to_string(&py_manifest) {
                for line in content.lines() {
                    let t = line.trim();
                    if t.is_empty() || t.starts_with('#') {
                        continue;
                    }
                    deps.insert(t.split("==").next().unwrap_or(t).to_string());
                }
            }
            ("python", deps)
        } else {
            return Ok(CallToolResult::error(
                "No supported manifest found (package.json, go.mod, requirements.txt)",
            ));
        }
    };

    // 2. Scan indexed records for imports in this directory
    let records = server.store.get_all_records().await?;
    let rel_dir = abs_path
        .strip_prefix(&root)
        .unwrap_or(&abs_path)
        .to_string_lossy()
        .replace('\\', "/");
    let mut missing_deps: HashMap<String, Vec<String>> = HashMap::new();

    for r in records {
        let meta = r.metadata_json();
        let file_path = meta["path"].as_str().unwrap_or("").to_string();
        if !rel_dir.is_empty() && !file_path.starts_with(&rel_dir) {
            continue;
        }
        let rels: Vec<String> = meta["relationships"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for dep in rels {
            let pkg = match project_type {
                "npm" => {
                    if dep.starts_with('.') || dep.starts_with('/') {
                        continue;
                    }
                    // Skip monorepo-local packages that resolve to a local path
                    if resolve_monorepo_import(&dep, &root).is_some() {
                        continue;
                    }
                    // Skip Node.js builtins and non-module strings (named imports, etc.)
                    // A valid npm package import contains only lowercase, digits, hyphens, @, /
                    // Named imports like "ApiError", "Request" contain uppercase — skip them.
                    if dep
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    // Skip Node.js built-in modules
                    if matches!(
                        dep.as_str(),
                        "fs" | "path"
                            | "http"
                            | "https"
                            | "crypto"
                            | "os"
                            | "url"
                            | "util"
                            | "stream"
                            | "events"
                            | "buffer"
                            | "child_process"
                            | "async_hooks"
                            | "net"
                            | "tls"
                            | "dns"
                            | "readline"
                            | "vm"
                            | "worker_threads"
                            | "zlib"
                            | "console"
                            | "constants"
                            | "module"
                            | "process"
                            | "punycode"
                            | "querystring"
                            | "string_decoder"
                            | "timers"
                            | "tty"
                            | "v8"
                    ) {
                        continue;
                    }
                    if dep
                        .chars()
                        .next()
                        .map(|c| c.is_lowercase())
                        .unwrap_or(false)
                        && !dep.contains("/")
                        && !dep.contains(".")
                        && !["react", "next", "lucide-react"].contains(&dep.as_str())
                    {
                        // It looks like a single-word named import (e.g., "useState") but lowercase
                        // npm packages usually have more structure or are explicitly in the manifest.
                        // If it's not in declared_deps, we'll check if it looks like a package.
                        // For now, let's be more strict.
                        if !declared_deps.contains(&dep) {
                            continue;
                        }
                    }
                    let parts: Vec<&str> = dep.splitn(3, '/').collect();
                    if dep.starts_with('@') && parts.len() > 1 {
                        format!("{}/{}", parts[0], parts[1])
                    } else {
                        parts[0].to_string()
                    }
                }
                "go" => {
                    if !dep.contains('.') || dep.starts_with(&root) {
                        continue;
                    }
                    dep.clone()
                }
                "python" => {
                    if dep.starts_with('.') {
                        continue;
                    }
                    dep.clone()
                }
                _ => continue,
            };
            if !declared_deps.contains(&pkg) {
                missing_deps.entry(pkg).or_default().push(file_path.clone());
            }
        }
    }

    if missing_deps.is_empty() {
        return Ok(CallToolResult::text(format!(
            "✅ Dependency Health Check ({project_type}): All external imports are correctly declared."
        )));
    }

    let mut out = format!(
        "## ⚠️ Dependency Health Report ({project_type})\n\nThe following external dependencies are imported but missing from your manifest:\n\n"
    );
    let mut sorted_deps: Vec<_> = missing_deps.iter().collect();
    sorted_deps.sort_by_key(|(k, _)| k.as_str());
    for (dep, files) in sorted_deps {
        let mut unique: Vec<_> = files.iter().collect::<HashSet<_>>().into_iter().collect();
        unique.sort();
        out.push_str(&format!("### `{dep}`\nImported in:\n"));
        for f in unique {
            out.push_str(&format!("- {f}\n"));
        }
        out.push('\n');
    }
    Ok(CallToolResult::text(out))
}

async fn handle_generate_docstring_prompt(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let file_path = require_string_arg(params, "file_path")?;
    let entity_name = require_string_arg(params, "entity_name")?;
    let language = optional_string_arg(params, "language").unwrap_or_else(|| {
        match std::path::Path::new(&file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
        {
            "go" => "Go",
            "ts" | "tsx" | "js" | "jsx" => "TypeScript/JavaScript",
            "py" => "Python",
            "rs" => "Rust",
            _ => "unknown",
        }
        .to_string()
    });

    let doc_style = match language.to_lowercase().as_str() {
        "go" => "Godoc comments",
        "typescript/javascript" | "typescript" | "javascript" => "JSDoc comments",
        "python" => "Python docstrings (PEP 257 format)",
        "rust" => "Rust doc comments (/// style)",
        _ => "professional documentation comment",
    };

    let records = server.store.get_records_by_path(&file_path).await?;
    let match_record = records.iter().find(|r| {
        let meta = r.metadata_json();
        meta["symbols"]
            .as_array()
            .map(|syms| syms.iter().any(|s| s.as_str() == Some(&entity_name)))
            .unwrap_or(false)
    });

    let Some(r) = match_record else {
        return Ok(CallToolResult::error(format!(
            "Entity '{entity_name}' not found in file '{file_path}'"
        )));
    };

    let meta = r.metadata_json();
    let calls = meta["calls"].to_string();
    let symbols = meta["symbols"].to_string();
    let relationships = meta["relationships"].to_string();

    let prompt = format!(
        "Please write a professional {doc_style} for the following code.\n\
        Architecture Context:\n\
        - Entity: {symbols}\n\
        - Internal Calls made: {calls}\n\
        - File Imports: {relationships}\n\n\
        Code:\n{}",
        r.content
    );
    Ok(CallToolResult::text(prompt))
}

async fn handle_analyze_architecture(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let monorepo_prefix = optional_string_arg(params, "monorepo_prefix").unwrap_or_default();
    let records = server.store.get_all_records().await?;

    // adjacency: src_pkg -> set of target_pkgs
    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();

    for r in records {
        let meta = r.metadata_json();
        let path = meta["path"].as_str().unwrap_or("").to_string();
        if path.is_empty() {
            continue;
        }
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 2 {
            continue;
        }
        let src_pkg = if parts.len() > 2 && (parts[0] == "apps" || parts[0] == "packages") {
            format!("{}/{}", parts[0], parts[1])
        } else {
            parts[0].to_string()
        };

        let rels: Vec<String> = meta["relationships"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for rel in rels {
            if monorepo_prefix.is_empty() || rel.starts_with(&monorepo_prefix) {
                adj.entry(src_pkg.clone()).or_default().insert(rel);
            }
        }
    }

    if adj.is_empty() {
        return Ok(CallToolResult::text(
            "No inter-package dependencies found in the indexed codebase.",
        ));
    }

    let mut out = String::from("graph TD\n");
    let mut sources: Vec<_> = adj.keys().collect();
    sources.sort();
    for src in sources {
        let mut targets: Vec<_> = adj[src].iter().collect();
        targets.sort();
        for tgt in targets {
            out.push_str(&format!("    \"{src}\" --> \"{tgt}\"\n"));
        }
    }
    Ok(CallToolResult::text(out))
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

async fn handle_filesystem_grep(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?.to_lowercase();
    let root = server.config.project_root.read().unwrap().clone();

    // Collect file paths in a blocking task to avoid blocking the async runtime.
    let paths = tokio::task::spawn_blocking(move || {
        let walker = ignore::WalkBuilder::new(&root)
            .standard_filters(true)
            .hidden(true)
            .add_custom_ignore_filename(".vector-ignore")
            .build();
        walker
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
            .map(|e| e.path().to_path_buf())
            .collect::<Vec<_>>()
    })
    .await?;

    let mut results = Vec::new();
    'outer: for path in paths {
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&query) {
                    results.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                    if results.len() >= 100 {
                        break 'outer;
                    }
                }
            }
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
    let mut exports: HashMap<String, String> = HashMap::new();
    let mut tested_symbols: HashSet<String> = HashSet::new();

    for r in records {
        let path = r.metadata_str("path");
        if path.contains("test") || path.contains("spec") {
            // Extract symbols from test files (functions/classes being tested).
            let meta = r.metadata_json();
            if let Some(syms) = meta["symbols"].as_array() {
                for s in syms {
                    if let Some(st) = s.as_str() {
                        tested_symbols.insert(st.to_string());
                    }
                }
            }
            // Also extract calls — test files call the functions they test.
            if let Some(calls) = meta["calls"].as_array() {
                for c in calls {
                    if let Some(st) = c.as_str() {
                        tested_symbols.insert(st.to_string());
                    }
                }
            }
        } else {
            let meta = r.metadata_json();
            if let Some(syms) = meta["symbols"].as_array() {
                for s in syms {
                    if let Some(st) = s.as_str() {
                        exports.insert(st.to_string(), path.clone());
                    }
                }
            }
        }
    }

    let mut missing = Vec::new();
    for (name, path) in exports {
        if !tested_symbols.contains(&name) {
            missing.push((name, path));
        }
    }
    if missing.is_empty() {
        return Ok(CallToolResult::text(
            "✅ All exported symbols appear in test files.",
        ));
    }
    let mut out = String::from("## ⚠️ Potentially Untested Exports\n\n");
    for (n, p) in missing {
        out.push_str(&format!("- `{}` in `{}`\n", n, p));
    }
    Ok(CallToolResult::text(out))
}

async fn handle_list_api_endpoints(
    server: &Server,
    _params: &CallToolParams,
) -> Result<CallToolResult> {
    let keywords = [
        "HandleFunc",
        "mux.Handle",
        "app.GET",
        "app.POST",
        "router.Register",
        "Route(",
        "@app.route",
    ];

    let mut unique: HashMap<String, crate::db::Record> = HashMap::new();
    for kw in keywords {
        let matches = server.store.lexical_search(kw, 20, None).await?;
        for m in matches {
            let meta = m.metadata_json();
            let key = format!(
                "{}:{}",
                meta["path"].as_str().unwrap_or(""),
                meta["start_line"].as_u64().unwrap_or(0)
            );
            unique.insert(key, m);
        }
    }

    if unique.is_empty() {
        return Ok(CallToolResult::text("No API routing patterns detected."));
    }

    let mut keys: Vec<_> = unique.keys().cloned().collect();
    keys.sort();

    let mut out = String::from("## 🌐 Detected API Endpoints / Routes\n\n");
    for k in keys {
        let r = &unique[&k];
        let meta = r.metadata_json();
        out.push_str(&format!(
            "### {} (Line {})\n```\n{}\n```\n\n",
            meta["path"].as_str().unwrap_or("?"),
            meta["start_line"].as_u64().unwrap_or(0),
            r.content.trim()
        ));
    }
    Ok(CallToolResult::text(out))
}

async fn handle_get_code_history(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let path = require_string_arg(params, "file_path")?;

    // Security enhancement: Prevent path traversal by using path_guard
    let abs = server
        .path_guard
        .validate(&path, PathOp::Read)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    let root = server.config.project_root.read().unwrap().clone();
    let output = tokio::process::Command::new("git")
        .args([
            "log",
            "-n",
            "5",
            "--pretty=format:%h - %s",
            "--",
            abs.to_str().unwrap_or(&path),
        ])
        .current_dir(&root)
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);
        return Ok(CallToolResult::error(format!(
            "git log failed (exit {code}): {stderr}"
        )));
    }

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
    // Gemini removed — report local embedder status instead.
    let model = &server.config.model_name;
    let reranker = &server.config.reranker_model_name;
    let has_reranker = server.embedder.rerank_session.is_some();
    Ok(CallToolResult::text(format!(
        "### ✅ Local LLM Status\n\n- **Embedder**: `{model}`\n- **Reranker**: `{reranker}` (loaded: {has_reranker})\n- **Local summarizer**: {}\n",
        if server.config.feature_toggles.enable_local_llm {
            "enabled"
        } else {
            "disabled"
        }
    )))
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

    // Store the raw indexed content as a knowledge item — no external LLM needed.
    let project_id = server.config.project_root.read().unwrap().clone();
    let distilled = relevant_content.trim().to_string();
    server
        .store
        .store_project_context(&project_id, &distilled)
        .await?;

    Ok(CallToolResult::text(format!(
        "### ✅ Knowledge Indexed\n\n{distilled}"
    )))
}

async fn handle_verify_proposed_change(
    server: &Server,
    params: &CallToolParams,
) -> Result<CallToolResult> {
    let proposed_change = require_string_arg(params, "proposed_change")?;
    let project_id = server.config.project_root.read().unwrap().clone();

    let contexts = server.store.get_project_context(&project_id).await?;
    if contexts.is_empty() {
        return Ok(CallToolResult::text(
            "No Knowledge Items found for this project. Use `store_context` to add architectural rules first.\n\nProposed change:\n".to_string() + &proposed_change
        ));
    }

    // Return the KIs alongside the proposed change for the agent to reason about.
    let ki_context = contexts.join("\n\n---\n\n");
    Ok(CallToolResult::text(format!(
        "### 🛡️ Knowledge Items for Review\n\n{ki_context}\n\n---\n\n### Proposed Change\n\n{proposed_change}\n\n*Agent: compare the proposed change against the Knowledge Items above and report any violations.*"
    )))
}

// ---------------------------------------------------------------------------
// Task 1: Action-based super-tools
// ---------------------------------------------------------------------------

/// Resolve a monorepo package alias (e.g. `@org/shared`) to a local path by
/// reading `package.json` workspaces and `tsconfig.json` `paths` from the project root.
fn resolve_monorepo_import(import: &str, project_root: &str) -> Option<String> {
    // 1. Try tsconfig.json paths
    let tsconfig_path = std::path::Path::new(project_root).join("tsconfig.json");
    if let Ok(content) = std::fs::read_to_string(&tsconfig_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(paths) = v["compilerOptions"]["paths"].as_object() {
                for (alias, targets) in paths {
                    // alias may be "@org/pkg" or "@org/pkg/*"
                    let alias_base = alias.trim_end_matches("/*");
                    let import_base = import.trim_end_matches("/*");
                    if import_base == alias_base || import.starts_with(alias_base) {
                        if let Some(first) = targets.as_array().and_then(|a| a.first()) {
                            if let Some(target) = first.as_str() {
                                let resolved =
                                    target.trim_end_matches("/*").trim_start_matches("./");
                                let suffix = import.strip_prefix(alias_base).unwrap_or("");
                                let full = format!("{}/{}{}", project_root, resolved, suffix);
                                return Some(full);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Try package.json workspaces — scan for a package whose "name" matches
    let pkg_json = std::path::Path::new(project_root).join("package.json");
    if let Ok(content) = std::fs::read_to_string(&pkg_json) {
        if let Ok(root_pkg) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(workspaces) = root_pkg["workspaces"].as_array() {
                for ws in workspaces {
                    let pattern = ws.as_str().unwrap_or("").trim_end_matches("/*");
                    let ws_dir = std::path::Path::new(project_root).join(pattern);
                    if let Ok(entries) = std::fs::read_dir(&ws_dir) {
                        for entry in entries.flatten() {
                            let child_pkg = entry.path().join("package.json");
                            if let Ok(c) = std::fs::read_to_string(&child_pkg) {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c) {
                                    if v["name"].as_str() == Some(import) {
                                        return Some(entry.path().to_string_lossy().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Recursive related context: file chunks + dependency chunks + usage samples.
async fn handle_get_related_context(server: &Server, file_path: &str) -> Result<CallToolResult> {
    let records = server.store.get_records_by_path(file_path).await?;
    if records.is_empty() {
        return Ok(CallToolResult::text(format!(
            "No context found for file: {file_path}"
        )));
    }

    let project_root = server.config.project_root.read().unwrap().clone();

    // Collect symbols and relationships from all chunks of the target file
    let mut all_symbols: HashSet<String> = HashSet::new();
    let mut unique_deps: HashMap<String, String> = HashMap::new(); // import_str -> resolved path

    for r in &records {
        let meta = r.metadata_json();
        if let Some(rels) = meta["relationships"].as_array() {
            for rel in rels.iter().filter_map(|v| v.as_str()) {
                if rel.starts_with("./") || rel.starts_with("../") {
                    let resolved = std::path::Path::new(
                        std::path::Path::new(file_path)
                            .parent()
                            .unwrap_or(std::path::Path::new("")),
                    )
                    .join(rel)
                    .to_string_lossy()
                    .to_string();
                    unique_deps.insert(rel.to_string(), resolved);
                } else if let Some(local) = resolve_monorepo_import(rel, &project_root) {
                    unique_deps.insert(rel.to_string(), local);
                } else {
                    unique_deps.insert(rel.to_string(), rel.to_string());
                }
            }
        }
        if let Some(syms) = meta["symbols"].as_array() {
            for s in syms.iter().filter_map(|v| v.as_str()) {
                all_symbols.insert(s.to_string());
            }
        }
    }

    let mut out = String::from("<context>\n");

    // 1. Target file chunks
    let dep_list: Vec<_> = unique_deps.keys().cloned().collect();
    let sym_list: Vec<_> = all_symbols.iter().cloned().collect();
    out.push_str(&format!(
        "  <file path=\"{file_path}\">\n    <metadata>\n      <dependencies>{}</dependencies>\n      <symbols>{}</symbols>\n    </metadata>\n",
        serde_json::to_string(&dep_list).unwrap_or_default(),
        serde_json::to_string(&sym_list).unwrap_or_default(),
    ));
    for r in &records {
        out.push_str(&format!(
            "    <code_chunk>\n{}\n    </code_chunk>\n",
            r.content
        ));
    }
    out.push_str("  </file>\n");

    // 2. Dependency chunks (fetch indexed chunks for each import)
    if !unique_deps.is_empty() {
        let all_records = server.store.get_all_records().await?;
        let mut path_map: HashMap<String, Vec<&crate::db::Record>> = HashMap::new();
        for r in &all_records {
            path_map.entry(r.metadata_str("path")).or_default().push(r);
        }

        for (import_str, resolved) in &unique_deps {
            let chunks: Vec<_> = path_map
                .iter()
                .filter(|(p, _)| p.contains(resolved.as_str()) || p.contains(import_str.as_str()))
                .flat_map(|(_, v)| v.iter().copied())
                .collect();

            out.push_str(&format!(
                "  <file path=\"{resolved}\" resolved_from=\"{import_str}\">\n"
            ));
            if chunks.is_empty() {
                out.push_str("    <error>No indexed chunks found.</error>\n");
            } else {
                for chunk in chunks {
                    out.push_str(&format!(
                        "    <code_chunk>\n{}\n    </code_chunk>\n",
                        chunk.content
                    ));
                }
            }
            out.push_str("  </file>\n");
        }
    }

    // 3. Usage samples: find where symbols from this file are used elsewhere
    if !all_symbols.is_empty() {
        out.push_str("  <usage_samples>\n");
        let mut found_any = false;
        for sym in &all_symbols {
            let usages = server.store.lexical_search(sym, 5, None).await?;
            for u in usages {
                if u.metadata_str("path") == file_path {
                    continue;
                }
                out.push_str(&format!(
                    "    <sample symbol=\"{sym}\" used_in=\"{}\">\n{}\n    </sample>\n",
                    u.metadata_str("path"),
                    u.content
                ));
                found_any = true;
            }
        }
        if !found_any {
            out.push_str("    <info>No external usage samples found.</info>\n");
        }
        out.push_str("  </usage_samples>\n");
    }

    out.push_str("</context>");
    Ok(CallToolResult::text(out))
}

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
            if query.is_empty() {
                return Ok(CallToolResult::error("query is required for vector search"));
            }
            let vector = server.embedder.embed_query(&query)?;
            let mut results = server
                .store
                .hybrid_search(vector, &query, limit * 3, cross_reference.as_deref())
                .await?;

            if let Some(ref pf) = path_filter {
                results.retain(|r| r.metadata_str("path").contains(pf.as_str()));
            }

            // Rerank if cross-encoder is available
            if server.embedder.rerank_session.is_some() && results.len() > 1 {
                let docs: Vec<String> = results.iter().map(|r| r.content.clone()).collect();
                if let Ok(scores) = server.embedder.rerank(&query, docs) {
                    let mut scored: Vec<_> = results.into_iter().zip(scores).collect();
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    results = scored.into_iter().map(|(r, _)| r).take(limit).collect();
                } else {
                    results.truncate(limit);
                }
            } else {
                results.truncate(limit);
            }

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
            let mut grep_params = params.arguments.clone();
            grep_params["is_regex"] = serde_json::json!(true);
            if let Some(pf) = path_filter {
                grep_params["include_pattern"] = serde_json::json!(pf);
            }
            let synthetic = CallToolParams {
                name: "filesystem_grep".into(),
                arguments: grep_params,
            };
            handle_filesystem_grep(server, &synthetic).await
        }
        "graph" => {
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
        "related_context" => {
            let file_path = if !query.is_empty() {
                query.clone()
            } else {
                return Ok(CallToolResult::error(
                    "query (file path) is required for related_context",
                ));
            };
            handle_get_related_context(server, &file_path).await
        }
        "index_status" => handle_index_status(server).await,
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: vector, regex, graph, related_context, index_status"
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
        "architecture" => handle_analyze_architecture(server, params).await,
        "api_list" => handle_list_api_endpoints(server, params).await,
        "docstring" => handle_generate_docstring_prompt(server, params).await,
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: ast_skeleton, dead_code, duplicate_code, dependencies, distill_package, architecture, api_list, docstring"
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
        "auto_fix" => handle_auto_fix(server, params).await,
        _ => Ok(CallToolResult::error(format!(
            "Invalid action '{action}'. Use: apply_patch, create_file, run_linter, verify_patch, auto_fix"
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
        .validate(&path, PathOp::Read)
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
        .validate(&path, PathOp::Create)
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
        .validate(&path, PathOp::Read)
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
        .validate(&path, PathOp::Read)
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

async fn handle_auto_fix(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let diag_json = match optional_string_arg(params, "diagnostic_json") {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(CallToolResult::error("diagnostic_json is required")),
    };

    // Parse the diagnostic — expect { "path": "...", "message": "...", "range": { "start": { "line": N } } }
    let diag: serde_json::Value = serde_json::from_str(&diag_json)
        .map_err(|e| anyhow::anyhow!("invalid diagnostic JSON: {e}"))?;

    let path = diag["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("diagnostic_json must contain 'path'"))?;
    let message = diag["message"].as_str().unwrap_or("unknown error");
    let line = diag["range"]["start"]["line"].as_u64().unwrap_or(0) as usize;

    let abs = server
        .path_guard
        .validate(path, PathOp::Read)
        .map_err(|e| anyhow::anyhow!("path guard: {e}"))?;

    let content = tokio::fs::read_to_string(&abs)
        .await
        .map_err(|e| anyhow::anyhow!("read failed: {e}"))?;

    let lines: Vec<&str> = content.lines().collect();
    let context_start = line.saturating_sub(2);
    let context_end = (line + 3).min(lines.len());
    let snippet = lines[context_start..context_end].join("\n");

    // If an LSP server is available, request code actions for the diagnostic range
    let abs_str = abs.to_string_lossy().to_string();
    if let Some(lsp) = server.lsp_pool.get_for_path(&abs_str) {
        let code_actions_result = lsp
            .call(
                "textDocument/codeAction",
                serde_json::json!({
                    "textDocument": { "uri": format!("file://{abs_str}") },
                    "range": {
                        "start": { "line": line, "character": 0 },
                        "end":   { "line": line, "character": 0 }
                    },
                    "context": {
                        "diagnostics": [diag],
                        "only": ["quickfix"]
                    }
                }),
            )
            .await;

        if let Ok(actions) = code_actions_result {
            if let Some(arr) = actions.as_array()
                && !arr.is_empty()
            {
                let titles: Vec<_> = arr.iter().filter_map(|a| a["title"].as_str()).collect();
                return Ok(CallToolResult::text(format!(
                    "### LSP Code Actions for `{path}` line {}\n\nDiagnostic: {message}\n\nAvailable fixes:\n{}",
                    line + 1,
                    titles
                        .iter()
                        .map(|t| format!("- {t}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )));
            }
        }
    }

    // Fallback: return the diagnostic + snippet for the agent to fix manually.
    Ok(CallToolResult::text(format!(
        "### Auto-Fix: No LSP Action Available\n\n**File**: `{path}` line {}\n**Diagnostic**: {message}\n\n**Context**:\n```\n{snippet}\n```\n\nLSP has no automatic fix. Agent: please analyze this snippet and provide a manual fix using `apply_patch`.",
        line + 1
    )))
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

    // Build a symbol-based summary without an external LLM.
    let mut symbols: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for r in &records {
        let meta = r.metadata_json();
        files.push(r.metadata_str("path"));
        if let Some(syms) = meta["symbols"].as_array() {
            symbols.extend(syms.iter().filter_map(|v| v.as_str().map(String::from)));
        }
    }
    symbols.sort();
    symbols.dedup();
    files.sort();
    files.dedup();

    let summary = format!(
        "## Package: {pkg_path}\n\n**Files** ({}):\n{}\n\n**Exported Symbols** ({}):\n{}",
        files.len(),
        files
            .iter()
            .map(|f| format!("- {f}"))
            .collect::<Vec<_>>()
            .join("\n"),
        symbols.len(),
        symbols
            .iter()
            .map(|s| format!("- `{s}`"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

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
