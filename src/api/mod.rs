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
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
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

async fn handle_health(State(s): State<Arc<ApiState>>) -> impl IntoResponse {
    let db_ok = s.store.get_all_records().await.is_ok();
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
    let ready = s.store.get_all_records().await.is_ok();
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

    let vector = match s.embedder.embed_query(&req.query) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let results = match s.store.hybrid_search(vector, &req.query, top_k, None).await {
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
        .filter(|r| {
            if req.docs_only {
                r.metadata_str("category") == "document"
            } else {
                true
            }
        })
        .map(|r| {
            let path = r.metadata_str("path");
            SearchResponse {
                id: r.id,
                text: r.content,
                similarity: 0.0,
                path,
            }
        })
        .collect();

    Json(resp).into_response()
}

async fn handle_context(
    State(s): State<Arc<ApiState>>,
    Json(req): Json<ContextRequest>,
) -> impl IntoResponse {
    let vector = match s.embedder.embed_text(&req.text) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
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
    Router::new()
        .route("/api/health", get(handle_health))
        .route("/api/ready", get(handle_ready))
        .route("/api/live", get(handle_live))
        .route("/api/stats", get(handle_stats))
        .route("/api/tools/repos", get(handle_tools_repos))
        .route("/api/tools/status", get(handle_tools_status))
        .route("/api/tools/index", post(handle_tools_index))
        .route("/api/tools/skeleton", get(handle_tools_skeleton))
        .route("/api/search", post(handle_search))
        .route("/api/context", post(handle_context))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn start_api_server(state: Arc<ApiState>, port: u16) -> anyhow::Result<()> {
    let addr = format!("[::]:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("HTTP API server listening on http://{addr}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}
