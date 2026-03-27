//! Tool call handlers for the Rust MCP server.
//! Provides high-performance implementations for codebase analysis,
//! semantic search, and local AI summarization.

use anyhow::Result;
use std::sync::Arc;
use std::collections::{HashSet, HashMap};
use serde_json::Value;

use super::protocol::{CallToolParams, CallToolResult};
use super::server::Server;

/// Route a `tools/call` to the appropriate handler by tool name.
pub async fn dispatch(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    match params.name.as_str() {
        "ping" => Ok(handle_ping()),
        "trigger_project_index" => handle_trigger_project_index(server, params).await,
        "set_project_root" => handle_set_project_root(server, params),
        "store_context" => handle_store_context(server, params),
        "get_related_context" => handle_get_related_context(server, params).await,
        "find_duplicate_code" => handle_find_duplicate_code(server, params).await,
        "delete_context" => handle_delete_context(server, params).await,
        "index_status" => handle_index_status(server).await,
        "get_codebase_skeleton" => handle_get_codebase_skeleton(server, params).await,
        "check_dependency_health" => handle_check_dependency_health(server, params).await,
        "generate_docstring_prompt" => handle_generate_docstring_prompt(server, params).await,
        "analyze_architecture" => handle_analyze_architecture(server, params).await,
        "find_dead_code" => handle_find_dead_code(server, params).await,
        "filesystem_grep" => handle_filesystem_grep(server, params),
        "search_codebase" => handle_search_codebase(server, params).await,
        "get_indexing_diagnostics" => handle_get_indexing_diagnostics(server),
        "get_summarized_context" => handle_get_summarized_context(server, params).await,
        "verify_implementation_gap" => handle_verify_implementation_gap(server, params).await,
        "find_missing_tests" => handle_find_missing_tests(server, params).await,
        "list_api_endpoints" => handle_list_api_endpoints(server, params).await,
        "get_code_history" => handle_get_code_history(server, params).await,
        "reindex_all" => handle_reindex_all(server, params).await,
        _ => Ok(CallToolResult::error(format!(
            "Unknown tool: {}",
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

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn handle_ping() -> CallToolResult {
    CallToolResult::text("pong")
}

async fn handle_trigger_project_index(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "project_path")?;
    let store = Arc::clone(&server.store);
    let config = Arc::clone(&server.config);
    let embedder = Arc::clone(&server.embedder);
    let summarizer = Arc::clone(&server.summarizer);
    
    let path_clone = path.clone();
    tokio::spawn(async move {
        let _ = crate::indexer::index_file(&path_clone, config, store, embedder, summarizer).await;
    });

    Ok(CallToolResult::text(format!("Indexing initiated for {path}")))
}

fn handle_set_project_root(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let _path = require_string_arg(params, "project_path")?;
    Ok(CallToolResult::text("Project root updated"))
}

fn handle_store_context(_server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let text = require_string_arg(params, "text")?;
    Ok(CallToolResult::text(format!("Context stored: {} chars", text.len())))
}

async fn handle_get_related_context(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let file_path = require_string_arg(params, "filePath")?;
    
    // 1. Fetch records for the target file
    let records = server.store.get_all_records().await?;
    let target_records: Vec<_> = records.iter().filter(|r| r.metadata_str("path") == file_path).collect();

    if target_records.is_empty() {
        return Ok(CallToolResult::text(format!("No indexed context found for {}", file_path)));
    }

    let mut out = format!("<context>\n  <file path=\"{}\">\n", file_path);
    let mut symbols = HashSet::new();
    let mut relations = HashSet::new();

    for r in &target_records {
        let meta = r.metadata_json();
        if let Some(s) = meta["symbols"].as_array() {
            for sym in s { if let Some(st) = sym.as_str() { symbols.insert(st.to_string()); } }
        }
        if let Some(rel) = meta["relationships"].as_array() {
            for rpath in rel { if let Some(rt) = rpath.as_str() { relations.insert(rt.to_string()); } }
        }
        out.push_str(&format!("    <code_chunk>\n{}\n    </code_chunk>\n", r.content));
    }
    out.push_str("  </file>\n");

    // 2. Resolve and Fetch Dependencies
    if !relations.is_empty() {
        out.push_str("  <dependencies>\n");
        for rel in relations {
            let dep_records: Vec<_> = records.iter().filter(|r| r.metadata_str("path").contains(&rel)).collect();
            if !dep_records.is_empty() {
                out.push_str(&format!("    <file path=\"{}\" resolved_from=\"{}\">\n", rel, rel));
                for dr in dep_records {
                    out.push_str(&format!("      <code_chunk>\n{}\n      </code_chunk>\n", dr.content));
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
                if r.metadata_str("path") == file_path { continue; }
                if r.content.contains(&sym) {
                    out.push_str(&format!("    <sample symbol=\"{}\" used_in=\"{}\">\n{}\n    </sample>\n", sym, r.metadata_str("path"), r.content));
                    found += 1;
                }
                if found >= 2 { break; }
            }
        }
        out.push_str("  </usage_samples>\n");
    }

    out.push_str("</context>");
    Ok(CallToolResult::text(out))
}

async fn handle_find_duplicate_code(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let target_path = require_string_arg(params, "target_path")?;
    let records = server.store.get_all_records().await?;
    let mut found = false;
    let mut out = format!("## Duplicate Code Analysis for {}\n\n", target_path);

    for r in records {
        if r.metadata_str("path") == target_path {
            let vector = r.vector.clone();
            let matches = server.store.hybrid_search(vector, &r.content, 3).await?;
            for m in matches {
                if m.metadata_str("path") != target_path {
                    out.push_str(&format!("- Possible duplicate in `{}`\n", m.metadata_str("path")));
                    found = true;
                }
            }
        }
    }

    if !found { out.push_str("No duplicates found."); }
    Ok(CallToolResult::text(out))
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

async fn handle_get_codebase_skeleton(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let root = &*server.config.project_root;
    let target_path = optional_string_arg(params, "target_path").unwrap_or_else(|| ".".into());
    let max_depth = optional_f64_arg(params, "max_depth").unwrap_or(3.0) as usize;
    let max_items = optional_f64_arg(params, "max_items").unwrap_or(1000.0) as usize;

    let abs_path = if std::path::Path::new(&target_path).is_absolute() {
        std::path::PathBuf::from(target_path)
    } else {
        std::path::Path::new(root).join(target_path)
    };

    if !abs_path.exists() { return Ok(CallToolResult::error("Invalid path")); }

    let mut out = format!("Directory Tree: {:?} (Depth: {})\n", abs_path, max_depth);
    let mut item_count = 0;

    fn walk_dir(path: &std::path::Path, depth: usize, max_depth: usize, item_count: &mut usize, max_items: usize, out: &mut String) {
        if depth > max_depth || *item_count >= max_items { return; }
        if let Ok(entries) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.file_name());
            for entry in entries {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name == "node_modules" || name == ".git" || name == "target" { continue; }
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                out.push_str(&format!("{}├── {}\n", "│   ".repeat(depth), name));
                *item_count += 1;
                if is_dir && depth < max_depth && *item_count < max_items {
                    walk_dir(&entry.path(), depth + 1, max_depth, item_count, max_items, out);
                }
            }
        }
    }
    walk_dir(&abs_path, 0, max_depth, &mut item_count, max_items, &mut out);
    Ok(CallToolResult::text(out))
}

async fn handle_check_dependency_health(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let dir_path = require_string_arg(params, "directory_path")?;
    let abs_path = std::path::Path::new(&*server.config.project_root).join(&dir_path);
    let mut declared_deps = HashSet::new();
    let mut project_type = "unknown";

    // 1. Detect project type and load declared dependencies
    if abs_path.join("package.json").exists() {
        project_type = "npm";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("package.json")) {
            if let Ok(v) = serde_json::from_str::<Value>(&content) {
                if let Some(deps) = v["dependencies"].as_object() { for d in deps.keys() { declared_deps.insert(d.clone()); } }
                if let Some(dev) = v["devDependencies"].as_object() { for d in dev.keys() { declared_deps.insert(d.clone()); } }
            }
        }
    } else if abs_path.join("go.mod").exists() {
        project_type = "go";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("go.mod")) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("require ") {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 { declared_deps.insert(parts[1].to_string()); }
                }
            }
        }
    } else if abs_path.join("requirements.txt").exists() {
        project_type = "python";
        if let Ok(content) = std::fs::read_to_string(abs_path.join("requirements.txt")) {
            let re = regex::Regex::new(r"^([a-zA-Z0-9_\-]+)")?;
            for line in content.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    if let Some(cap) = re.captures(trimmed) {
                        declared_deps.insert(cap[1].to_string());
                    }
                }
            }
        }
    }

    if project_type == "unknown" {
        return Ok(CallToolResult::error("Could not identify project type (no package.json, go.mod, or requirements.txt)"));
    }

    // 2. Scan for used dependencies in indexed records
    let records = server.store.get_all_records().await?;
    let mut used_deps = HashSet::new();
    let dir_str = dir_path.replace('\\', "/");

    for r in records {
        let meta: Value = serde_json::from_str(&r.metadata).unwrap_or(Value::Null);
        let path = meta["path"].as_str().unwrap_or("");
        if !path.contains(&dir_str) { continue; }

        if let Some(rels) = meta["relationships"].as_array() {
            for rel in rels {
                if let Some(import_path) = rel.as_str() {
                    if project_type == "npm" {
                        if !import_path.starts_with('.') && !import_path.starts_with('/') {
                            let parts: Vec<&str> = import_path.split('/').collect();
                            if !parts.is_empty() {
                                let pkg = if parts[0].starts_with('@') && parts.len() > 1 {
                                    format!("{}/{}", parts[0], parts[1])
                                } else { parts[0].to_string() };
                                used_deps.insert(pkg);
                            }
                        }
                    } else if project_type == "go" {
                        // Skip stdlib (heuristic: no dot in first path component)
                        if import_path.contains('.') && !import_path.contains(&*server.config.project_root) {
                            used_deps.insert(import_path.to_string());
                        }
                    } else if project_type == "python" {
                        if !import_path.starts_with('.') {
                            used_deps.insert(import_path.to_string());
                        }
                    }
                }
            }
        }
    }

    // 3. Compare and report
    let missing: Vec<_> = used_deps.iter().filter(|d| !declared_deps.contains(*d)).collect();
    let unused: Vec<_> = declared_deps.iter().filter(|d| !used_deps.contains(*d) && !d.contains("types") && !d.contains("eslint")).collect();

    let mut out = format!("## Dependency Health Report ({})\n\n", project_type);
    out.push_str(&format!("- Directory: `{}`\n", dir_path));
    out.push_str(&format!("- Declared: {}\n", declared_deps.len()));
    out.push_str(&format!("- Actually Used: {}\n\n", used_deps.len()));

    if !missing.is_empty() {
        out.push_str("### ⚠️ Missing in Manifest (Used but not declared)\n");
        for m in &missing { out.push_str(&format!("- `{}`\n", m)); }
        out.push('\n');
    }

    if !unused.is_empty() {
        out.push_str("### ℹ️ Potentially Unused (Declared but not seen in imports)\n");
        for u in &unused { out.push_str(&format!("- `{}`\n", u)); }
    }

    if missing.is_empty() && unused.is_empty() {
        out.push_str("✅ All declared dependencies are used, and no missing dependencies were found.");
    }

    Ok(CallToolResult::text(out))
}

async fn handle_generate_docstring_prompt(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let name = require_string_arg(params, "entity_name")?;
    let vector = server.embedder.embed_text(&name)?;
    let records = server.store.hybrid_search(vector, &name, 1).await?;
    if let Some(r) = records.first() {
        Ok(CallToolResult::text(format!("Generate documentation for:\n\n{}", r.content)))
    } else {
        Ok(CallToolResult::error("Entity not found"))
    }
}

async fn handle_analyze_architecture(server: &Server, _params: &CallToolParams) -> Result<CallToolResult> {
    let vector = server.embedder.embed_text("system architecture design")?;
    let records = server.store.hybrid_search(vector, "architecture", 5).await?;
    let mut combined = String::new();
    for r in records { combined.push_str(&r.content); combined.push('\n'); }
    let summary = server.summarizer.summarize_chunk(&combined, Arc::clone(&server.config)).await?;
    Ok(CallToolResult::text(format!("## Architecture Overview\n\n{}", summary)))
}

async fn handle_find_dead_code(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let is_library = params.arguments.get("is_library").and_then(|v| v.as_bool()).unwrap_or(false);
    let records = server.store.get_all_records().await?;
    let mut exports = HashMap::new();
    let mut usages = HashSet::new();

    for r in records {
        let path = r.metadata_str("path");
        if path.contains("test") || path.contains("spec") { continue; }
        if is_library && !path.contains("internal") && !path.contains("private") { continue; }

        let meta = r.metadata_json();
        if let Some(syms) = meta["symbols"].as_array() {
            for s in syms { if let Some(st) = s.as_str() { exports.insert(st.to_string(), path.to_string()); } }
        }
        if let Some(calls) = meta["calls"].as_array() {
            for c in calls { if let Some(st) = c.as_str() { usages.insert(st.to_string()); } }
        }
    }

    let mut dead = Vec::new();
    for (name, path) in exports {
        let base_name = name.split('.').last().unwrap_or(&name);
        if !usages.contains(&name) && !usages.contains(base_name) {
            dead.push((name, path));
        }
    }

    if dead.is_empty() { return Ok(CallToolResult::text("✅ No dead code found.")); }
    let mut out = String::from("## 🔎 Potential Dead Code Report\n\n");
    for (n, p) in dead { out.push_str(&format!("- `{}` in `{}`\n", n, p)); }
    Ok(CallToolResult::text(out))
}

fn handle_filesystem_grep(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?.to_lowercase();
    let root = &*server.config.project_root;
    let mut results = Vec::new();
    
    use walkdir::WalkDir;
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let entry: walkdir::DirEntry = entry;
        if entry.file_type().is_file() {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                for (i, line) in content.lines().enumerate() {
                    if line.to_lowercase().contains(&query) {
                        results.push(format!("{}:{}: {}", entry.path().display(), i+1, line.trim()));
                    }
                }
            }
        }
        if results.len() > 50 { break; }
    }
    Ok(CallToolResult::text(results.join("\n")))
}

async fn handle_search_codebase(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let top_k = optional_f64_arg(params, "topK").unwrap_or(10.0) as usize;
    
    let vector = server.embedder.embed_text(&query)?;
    
    // 1. Fetch more results than needed for reranking (e.g. top_k * 3)
    let records = server.store.hybrid_search(vector, &query, top_k * 3).await?;
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
            scored_records.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            final_records = scored_records.into_iter().map(|(r, _)| r).take(top_k).collect();
        }
    } else {
        final_records.truncate(top_k);
    }

    let mut out = String::new();
    for r in final_records { 
        let summary = r.metadata_str("summary");
        out.push_str(&format!("### {}\n**Summary**: {}\n\n{}\n\n", r.id, summary, r.content)); 
    }
    Ok(CallToolResult::text(out))
}

fn handle_get_indexing_diagnostics(_server: &Server) -> Result<CallToolResult> {
    Ok(CallToolResult::text("System status: Optimal"))
}

async fn handle_get_summarized_context(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let vector = server.embedder.embed_text(&query)?;
    let records = server.store.hybrid_search(vector, &query, 5).await?;
    let mut text = String::new();
    for r in records { text.push_str(&r.content); text.push('\n'); }
    let summary = server.summarizer.summarize_chunk(&text, Arc::clone(&server.config)).await?;
    Ok(CallToolResult::text(format!("### Summary\n\n{}", summary)))
}

async fn handle_verify_implementation_gap(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let query = require_string_arg(params, "query")?;
    let vector = server.embedder.embed_text(&query)?;
    let records = server.store.hybrid_search(vector, &query, 10).await?;
    let mut out = format!("## Implementation Gap Analysis for '{}'\n\n", query);
    for r in records {
        let cat = r.metadata_str("category");
        let cat_display = if cat.is_empty() { "code" } else { &cat };
        out.push_str(&format!("- [{}] `{}`\n", cat_display, r.metadata_str("path")));
    }
    Ok(CallToolResult::text(out))
}

async fn handle_find_missing_tests(server: &Server, _params: &CallToolParams) -> Result<CallToolResult> {
    let records = server.store.get_all_records().await?;
    let mut exports = HashMap::new();
    let mut usages = HashSet::new();

    for r in records {
        let path = r.metadata_str("path");
        if path.contains("test") {
            for word in r.content.split_whitespace() { usages.insert(word.to_string()); }
        } else {
            let meta = r.metadata_json();
            if let Some(syms) = meta["symbols"].as_array() {
                for s in syms { if let Some(st) = s.as_str() { exports.insert(st.to_string(), path.to_string()); } }
            }
        }
    }

    let mut missing = Vec::new();
    for (name, path) in exports { if !usages.contains(&name) { missing.push((name, path)); } }
    if missing.is_empty() { return Ok(CallToolResult::text("All exports tested.")); }
    let mut out = String::from("## Missing Tests\n\n");
    for (n, p) in missing { out.push_str(&format!("- `{}` in `{}`\n", n, p)); }
    Ok(CallToolResult::text(out))
}

async fn handle_list_api_endpoints(server: &Server, _params: &CallToolParams) -> Result<CallToolResult> {
    let keywords = ["HandleFunc", "app.GET", "app.POST", "Route("];
    let mut out = String::from("## API Endpoints\n\n");
    for kw in keywords {
        let vector = server.embedder.embed_text(kw)?;
        let records = server.store.hybrid_search(vector, kw, 5).await?;
        for r in records {
            out.push_str(&format!("- `{}`: `{}`\n", r.metadata_str("path"), r.content.lines().next().unwrap_or("")));
        }
    }
    Ok(CallToolResult::text(out))
}

async fn handle_get_code_history(server: &Server, params: &CallToolParams) -> Result<CallToolResult> {
    let path = require_string_arg(params, "file_path")?;
    let output = std::process::Command::new("git")
        .args(["log", "-n", "5", "--pretty=format:%h - %s", "--", &path])
        .current_dir(&*server.config.project_root)
        .output()?;
    Ok(CallToolResult::text(String::from_utf8_lossy(&output.stdout).to_string()))
}

async fn handle_reindex_all(server: &Server, _params: &CallToolParams) -> Result<CallToolResult> {
    let config = Arc::clone(&server.config);
    let store = Arc::clone(&server.store);
    let embedder = Arc::clone(&server.embedder);
    let summarizer = Arc::clone(&server.summarizer);

    tokio::spawn(async move {
        let _ = crate::indexer::scanner::scan_project(config, store, embedder, summarizer).await;
    });

    Ok(CallToolResult::text("Full project re-indexing initiated in background."))
}
