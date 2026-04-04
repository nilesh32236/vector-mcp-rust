use anyhow::Result;
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{error, info, warn};

use crate::config::Config;
use crate::db::Store;
use crate::indexer;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

/// Starts the background CDC file watcher.
pub async fn start_watcher(
    config: Arc<Config>,
    db: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
) -> Result<Option<notify::RecommendedWatcher>> {
    if !config.feature_toggles.enable_live_indexing {
        info!("Live indexing disabled via config");
        return Ok(None);
    }

    info!(
        "Initializing background CDC watcher on: {}",
        config.project_root.read().unwrap().clone()
    );

    let (tx, mut rx) = mpsc::channel::<PathBuf>(100);

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if is_interesting_event(&event) {
                for path in event.paths {
                    let _ = tx.blocking_send(path);
                }
            }
        }
        Err(e) => error!("watch error: {:?}", e),
    })?;

    watcher.watch(
        std::path::Path::new(&config.project_root.read().unwrap().clone()),
        RecursiveMode::Recursive,
    )?;

    tokio::spawn(async move {
        let mut pending: HashMap<PathBuf, Instant> = HashMap::new();
        let debounce_duration = Duration::from_millis(500);

        loop {
            tokio::select! {
                Some(path) = rx.recv() => {
                    pending.insert(path, Instant::now() + debounce_duration);
                }

                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    let now = Instant::now();
                    let ready_paths: Vec<PathBuf> = pending
                        .iter()
                        .filter(|(_, deadline)| now >= **deadline)
                        .map(|(path, _)| path.clone())
                        .collect();

                    for path in ready_paths {
                        pending.remove(&path);

                        let path_str = path.to_string_lossy().to_string();
                        info!("CDC: Re-indexing changed file: {}", path_str);

                        let config = Arc::clone(&config);
                        let db = Arc::clone(&db);
                        let embedder = Arc::clone(&embedder);
                        let summarizer = Arc::clone(&summarizer);

                        tokio::spawn(async move {
                            if let Err(e) = indexer::index_file(&path_str, Arc::clone(&config), Arc::clone(&db), Arc::clone(&embedder), Arc::clone(&summarizer)).await {
                                error!("Failed to re-index {}: {:?}", path_str, e);
                                return;
                            }

                            // --- Architectural Guardrails ---
                            let rel_path = get_relative_path(&path_str, &config.project_root.read().unwrap());
                            check_architectural_compliance(&rel_path, Arc::clone(&config), Arc::clone(&db), Arc::clone(&embedder)).await;

                            // --- Autonomous Re-Distillation ---
                            redistill_dependents(&rel_path, Arc::clone(&config), Arc::clone(&db), Arc::clone(&embedder), Arc::clone(&summarizer)).await;
                        });
                    }
                }
            }
        }
    });

    Ok(Some(watcher))
}

fn get_relative_path(path: &str, root: &str) -> String {
    if let Ok(rel) = std::path::Path::new(path).strip_prefix(root) {
        rel.to_string_lossy().to_string()
    } else {
        path.to_string()
    }
}

async fn check_architectural_compliance(
    rel_path: &str,
    config: Arc<Config>,
    db: Arc<Store>,
    embedder: Arc<Embedder>,
) {
    let _project_root = config.project_root.read().unwrap().clone();

    // 1. Fetch the records for the file we just indexed
    let records = match db.get_records_by_path(rel_path).await {
        Ok(r) if !r.is_empty() => r,
        _ => return,
    };

    // 2. Search for relevant ADRs and Distilled Summaries
    let vector = match embedder.embed_query("architecture dependency rules constraints ADR") {
        Ok(v) => v,
        _ => return,
    };

    let relevant_rules = match db.hybrid_search(vector, "architecture", 5, None).await {
        Ok(r) => r,
        _ => return,
    };

    for r in records {
        let meta = r.metadata_json();
        let current_deps: Vec<String> = meta["relationships"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        for rule in &relevant_rules {
            let rule_meta = rule.metadata_json();
            let cat = rule_meta["category"].as_str().unwrap_or("");
            let rule_type = rule_meta["type"].as_str().unwrap_or("");

            if cat != "adr" && rule_type != "distilled_summary" {
                continue;
            }

            let content_lower = rule.content.to_lowercase();
            for dep in &current_deps {
                let dep_lower = dep.to_lowercase();
                if content_lower.contains(&format!("no {}", dep_lower))
                    || content_lower.contains(&format!("forbidden: {}", dep_lower))
                {
                    let msg = format!(
                        "🛡️ Architectural Alert: File `{}` might violate rule in `{}`. Found forbidden dependency: `{}`",
                        rel_path,
                        rule_meta["path"].as_str().unwrap_or("unknown"),
                        dep
                    );
                    warn!("{}", msg);
                }
            }
        }
    }
}

async fn redistill_dependents(
    rel_path: &str,
    config: Arc<Config>,
    db: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
) {
    let pkg = std::path::Path::new(rel_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());

    let query = format!("pkg:{}", pkg);
    let vector = match embedder.embed_query(&query) {
        Ok(v) => v,
        _ => return,
    };

    let records = match db.hybrid_search(vector, &query, 20, None).await {
        Ok(r) => r,
        _ => return,
    };

    let mut dependent_pkgs = std::collections::HashSet::new();
    for r in records {
        let path = r.metadata_str("path");
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let dir = parent.to_string_lossy().to_string();
            if dir != pkg {
                dependent_pkgs.insert(dir);
            }
        }
    }

    for d_pkg in dependent_pkgs {
        info!(
            "Triggering autonomous re-distillation for dependent package: {} (reason: {})",
            d_pkg, rel_path
        );
        let config_clone = Arc::clone(&config);
        let db_clone = Arc::clone(&db);
        let embedder_clone = Arc::clone(&embedder);
        let summarizer_clone = Arc::clone(&summarizer);
        tokio::spawn(async move {
            let _ = distill_package_internal(
                &d_pkg,
                config_clone,
                db_clone,
                embedder_clone,
                summarizer_clone,
            )
            .await;
        });
    }
}

async fn distill_package_internal(
    pkg_path: &str,
    config: Arc<Config>,
    db: Arc<Store>,
    embedder: Arc<Embedder>,
    _summarizer: Arc<Summarizer>,
) -> Result<()> {
    let records = db.get_records_by_path(pkg_path).await?;
    if records.is_empty() {
        return Ok(());
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

    let client = crate::llm::gemini::GeminiClient::new(&config);
    let summary = client
        .generate_completion(&config.default_gemini_model, system_prompt, &user_prompt)
        .await?;

    let vector = embedder.embed_text(&summary)?;
    let metadata = serde_json::json!({
        "path": pkg_path,
        "type": "distilled_summary",
        "priority": "2.0",
        "project_id": config.project_root.read().unwrap().clone(),
    });
    let record = crate::db::Record {
        id: format!("distill-{}", uuid::Uuid::new_v4()),
        content: summary.clone(),
        vector,
        metadata: metadata.to_string(),
    };
    db.upsert_records(vec![record]).await?;

    Ok(())
}

fn is_interesting_event(event: &Event) -> bool {
    match event.kind {
        EventKind::Modify(_) | EventKind::Create(_) => event.paths.iter().any(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "go" | "rs" | "js" | "ts" | "tsx" | "php" | "py" | "md") {
                return false;
            }
            // Reject paths containing ignored directories
            let path_str = p.to_string_lossy();
            !path_str.contains("/node_modules/")
                && !path_str.contains("/dist/")
                && !path_str.contains("/build/")
                && !path_str.contains("/.next/")
                && !path_str.contains("/generated/prisma/")
                && !path_str.contains("/target/")
        }),
        _ => false,
    }
}
