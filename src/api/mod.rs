//! HTTP API server — mirrors Go's internal/api/server.go.
//! Exposes health/ready/live probes + REST search/context endpoints.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::{
    cors::CorsLayer,
    services::{ServeDir, ServeFile},
};
use tracing::info;

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
use crate::llm::kv_cache::KvCacheStore;
use crate::llm::summarizer::Summarizer;
use crate::security::ratelimit::RateLimiter;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub struct ApiState {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    #[allow(dead_code)]
    pub summarizer: Arc<Summarizer>,
    pub config: Arc<Config>,
    pub index_tx: tokio::sync::mpsc::Sender<String>,
    #[allow(dead_code)]
    pub rate_limiter: Arc<RateLimiter>,
    pub progress: Arc<std::sync::RwLock<crate::indexer::scanner::ProgressState>>,
    pub version: &'static str,
    /// Optional KV-cache store reference for the /api/cache/status endpoint.
    pub kv_cache: Option<Arc<KvCacheStore>>,
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    docs_only: bool,
}
fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
struct SearchResponse {
    id: String,
    text: String,
    similarity: f32,
    path: String,
    summary: String,
    start_line: u64,
    end_line: u64,
}

#[derive(Deserialize)]
struct ContextRequest {
    text: String,
    #[serde(default)]
    source: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn embed_query_blocking(embedder: Arc<Embedder>, query: String) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || embedder.embed_query(&query))
        .await
        .map_err(|e| format!("Embedding task failed: {e}"))?
        .map_err(|e| e.to_string())
}

async fn embed_text_blocking(embedder: Arc<Embedder>, text: String) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || embedder.embed_text(&text))
        .await
        .map_err(|e| format!("Embedding task failed: {e}"))?
        .map_err(|e| e.to_string())
}

async fn handle_health(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let db_ok = s.store.code_vectors.count_rows(None).await.is_ok();
    let status = if db_ok { "ok" } else { "degraded" };
    let code = if db_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(serde_json::json!({
            "status": status,
            "version": s.version,
            "checks": { "database": { "status": if db_ok { "ok" } else { "unhealthy" } } }
        })),
    )
}

async fn handle_ready(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let ready = s.store.code_vectors.count_rows(None).await.is_ok();
    let code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(serde_json::json!({ "ready": ready })))
}

async fn handle_live() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "alive" }))
}

async fn handle_stats(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let p = s.progress.read().unwrap();
    Json(serde_json::to_value(&*p).unwrap_or_default())
}

// --- UI Compatibility Handlers ---

async fn handle_tools_repos(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let root = s.config.project_root.read().unwrap().clone();
    let status = s.progress.read().unwrap().status.clone();
    let status = if status.is_empty() {
        "Ready".to_string()
    } else {
        status
    };
    Json(serde_json::json!([{
        "path": root,
        "status": status
    }]))
}

async fn handle_tools_status(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let p = s.progress.read().unwrap();
    Json(serde_json::json!({
        "status": p.status,
        "indexed_files": p.indexed_files,
        "total_files": p.total_files,
        "current_file": p.current_file,
    }))
}

#[derive(Deserialize)]
struct IndexRequest {
    path: Option<String>,
}

async fn handle_tools_index(
    State(s): State<Arc<ApiState>>,
    Json(req): Json<IndexRequest>,
) -> impl IntoResponse {
    let path = req
        .path
        .unwrap_or_else(|| s.config.project_root.read().unwrap().clone());
    let _ = s.index_tx.send(path).await;
    Json(serde_json::json!({ "status": "triggered" }))
}

async fn handle_tools_skeleton(
    State(s): State<Arc<ApiState>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let path = params
        .get("path")
        .cloned()
        .unwrap_or_else(|| s.config.project_root.read().unwrap().clone());

    // Use the analyze_code logic or a simplified version
    let mut tree = String::new();
    if let Ok(walker) = crate::indexer::get_directory_tree(&std::path::PathBuf::from(path), 3) {
        tree = walker;
    }

    Json(serde_json::json!(tree))
}

async fn handle_search(
    State(s): State<Arc<ApiState>>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let top_k = req.top_k.clamp(1, 100);
    let query = req.query.clone();

    let vector = match embed_query_blocking(Arc::clone(&s.embedder), query.clone()).await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };

    let results = match s
        .store
        .hybrid_search_scored(vector, &query, top_k, None)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let resp: Vec<SearchResponse> = results
        .into_iter()
        .filter(|(r, _)| {
            if req.docs_only {
                r.metadata_str("category") == "document"
            } else {
                true
            }
        })
        .map(|(r, score)| {
            let path = r.metadata_str("path");
            let meta = r.metadata_json();
            let summary = meta["summary"].as_str().unwrap_or("").to_string();
            let start_line = meta["start_line"].as_u64().unwrap_or(0);
            let end_line = meta["end_line"].as_u64().unwrap_or(0);
            SearchResponse {
                id: r.id,
                text: r.content,
                similarity: score,
                path,
                summary,
                start_line,
                end_line,
            }
        })
        .collect();

    Json(resp).into_response()
}

async fn handle_cache_status(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    match &s.kv_cache {
        Some(kv) => Json(serde_json::json!({
            "enabled": true,
            "entries": kv.entry_count(),
            "max_entries": kv.max_entries(),
            "cache_dir": kv.dir().display().to_string(),
        })),
        None => Json(serde_json::json!({ "enabled": false })),
    }
}

async fn handle_graph_stats(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "nodes": s.store.graph.node_count(),
        "call_edges": s.store.graph.call_edge_count(),
        "impl_edges": s.store.graph.impl_edge_count(),
    }))
}

async fn handle_context(
    State(s): State<Arc<ApiState>>,
    Json(req): Json<ContextRequest>,
) -> impl IntoResponse {
    let vector = match embed_text_blocking(Arc::clone(&s.embedder), req.text.clone()).await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };

    let id = format!("manual_{}", uuid::Uuid::new_v4());
    let metadata = serde_json::json!({
        "type": "manual_context",
        "source": req.source,
        "path": "",
    });

    let record = crate::db::Record {
        id,
        content: req.text,
        vector,
        metadata: metadata.to_string(),
    };

    match s.store.upsert_records(vec![record]).await {
        Ok(_) => Json(serde_json::json!({ "status": "ok" })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router(state: Arc<ApiState>) -> Router {
    let api_routes = Router::new()
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/live", get(handle_live))
        .route("/stats", get(handle_stats))
        .route("/tools/repos", get(handle_tools_repos))
        .route("/tools/status", get(handle_tools_status))
        .route("/tools/index", post(handle_tools_index))
        .route("/tools/skeleton", get(handle_tools_skeleton))
        .route("/cache/status", get(handle_cache_status))
        .route("/graph/stats", get(handle_graph_stats))
        .route("/search", post(handle_search))
        .route("/context", post(handle_context));

    Router::new()
        .nest("/api", api_routes)
        .fallback_service(ServeDir::new("public").fallback(ServeFile::new("public/index.html")))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn start_api_server(state: Arc<ApiState>, port: u16) -> anyhow::Result<()> {
    let addr = format!("[::]:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            // Configured port is taken — bind on any free port instead.
            let fallback = tokio::net::TcpListener::bind("[::]:0").await?;
            let actual = fallback.local_addr()?;
            tracing::warn!(
                configured = port,
                actual = actual.port(),
                "API port in use, falling back to random free port"
            );
            fallback
        }
        Err(e) => return Err(e.into()),
    };
    info!(
        "HTTP API server listening on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, router(state)).await?;
    Ok(())
}
