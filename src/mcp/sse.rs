//! SSE transport for MCP.

use axum::{
    Json, Router,
    extract::{Host, Query, State},
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, post},
};
use dashmap::DashMap;
use futures::stream::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tracing::info;
use uuid::Uuid;

use super::server::Server;

// ---------------------------------------------------------------------------
// SSE Session Manager
// ---------------------------------------------------------------------------

type SessionSender = mpsc::UnboundedSender<Event>;

pub struct SseManager {
    /// session_id -> SSE event sender (populated when GET /sse opens the stream)
    sessions: DashMap<String, SessionSender>,
    server: Arc<Server>,
    /// Most recently active session (fallback for clients that don't send mcp-session-id)
    last_session: Arc<std::sync::RwLock<Option<String>>>,
}

impl SseManager {
    pub fn new(server: Arc<Server>) -> Self {
        Self {
            sessions: DashMap::new(),
            server,
            last_session: Arc::new(std::sync::RwLock::new(None)),
        }
    }
}

// ---------------------------------------------------------------------------
// SSE Session Cleanup Wrapper
// ---------------------------------------------------------------------------

struct SessionCleanupStream<S> {
    inner: Pin<Box<S>>,
    session_id: String,
    manager: Arc<SseManager>,
}

impl<S> Stream for SessionCleanupStream<S>
where
    S: Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for SessionCleanupStream<S> {
    fn drop(&mut self) {
        info!(session = %self.session_id, "SSE session disconnected, cleaning up");
        self.manager.sessions.remove(&self.session_id);
        self.manager
            .server
            .progress_senders
            .remove(&self.session_id);
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /sse or POST /message — Receive JSON-RPC messages.
///
/// MCP 2025 streamable-HTTP flow:
///   1. Client POSTs `initialize` → server creates session, returns response + `mcp-session-id` header.
///   2. Client GETs `/sse?session_id=<id>` → opens the SSE stream for server-push.
///   3. All subsequent POSTs carry `mcp-session-id` header or `session_id` query param.
async fn message_handler(
    State(manager): State<Arc<SseManager>>,
    Query(params): Query<HashMap<String, String>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let body_str = String::from_utf8_lossy(&body).to_string();

    // Extract session ID from header or query param
    let provided_sid = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| params.get("session_id").cloned());

    let method = serde_json::from_str::<serde_json::Value>(&body_str)
        .ok()
        .and_then(|v| v["method"].as_str().map(|s| s.to_string()))
        .unwrap_or_default();

    // `initialize` creates a new session
    if method == "initialize" {
        let sid = Uuid::new_v4().to_string();
        // Pre-register a placeholder progress sender (replaced when GET /sse opens the stream)
        let (val_tx, _) = mpsc::unbounded_channel::<serde_json::Value>();
        manager.server.progress_senders.insert(sid.clone(), val_tx);
        if let Ok(mut last) = manager.last_session.write() {
            *last = Some(sid.clone());
        }
        let response = manager
            .server
            .process_message_with_session(&body_str, Some(&sid))
            .await;
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/json".to_string()),
                (header::HeaderName::from_static("mcp-session-id"), sid),
            ],
            Json(response),
        )
            .into_response();
    }

    // All other requests require a valid session ID — fall back to last known session
    let sid = provided_sid
        .or_else(|| manager.last_session.read().ok().and_then(|g| g.clone()))
        .unwrap_or_default();

    let server = Arc::clone(&manager.server);

    // Synchronous methods: respond directly in the HTTP body
    if matches!(
        method.as_str(),
        "tools/list" | "tools/call" | "notifications/initialized" | "initialized"
    ) {
        let response = server
            .process_message_with_session(&body_str, Some(&sid))
            .await;
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/json".to_string()),
                (header::HeaderName::from_static("mcp-session-id"), sid),
            ],
            Json(response),
        )
            .into_response();
    }

    // Async methods: push response over SSE stream
    let sender = manager.sessions.get(&sid).map(|s| s.clone());
    let sid_for_task = sid.clone();
    tokio::spawn(async move {
        let response = server
            .process_message_with_session(&body_str, Some(&sid_for_task))
            .await;
        if response.get("error").is_some() {
            tracing::error!(session = %sid_for_task, response = %response, "process_message_with_session returned error");
        }
        if !response.is_null()
            && let Some(tx) = sender
        {
            let event = Event::default()
                .event("message")
                .data(serde_json::to_string(&response).unwrap_or_default());
            let _ = tx.send(event);
        }
    });

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (header::HeaderName::from_static("mcp-session-id"), sid),
        ],
        Json(serde_json::json!({})),
    )
        .into_response()
}

/// GET /sse?session_id=<id> — Open the SSE stream for an existing session.
async fn sse_handler(
    Host(host): Host,
    State(manager): State<Arc<SseManager>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let session_id = match params.get("session_id").cloned() {
        Some(sid) => sid,
        // Legacy clients that open SSE before initialize: create a new session
        None => Uuid::new_v4().to_string(),
    };

    let (tx, rx) = mpsc::unbounded_channel::<Event>();
    manager.sessions.insert(session_id.clone(), tx.clone());
    if let Ok(mut last) = manager.last_session.write() {
        *last = Some(session_id.clone());
    }

    // Wire up progress notifications for this session
    let (val_tx, mut val_rx) = mpsc::unbounded_channel::<serde_json::Value>();
    let event_tx = tx.clone();
    tokio::spawn(async move {
        while let Some(v) = val_rx.recv().await {
            let data = serde_json::to_string(&v).unwrap_or_default();
            if event_tx
                .send(Event::default().event("message").data(data))
                .is_err()
            {
                break;
            }
        }
    });
    manager
        .server
        .progress_senders
        .insert(session_id.clone(), val_tx);

    info!(session = %session_id, "SSE stream opened");

    // Send the endpoint URL so legacy SSE clients know where to POST
    let endpoint_url = format!("http://{}/message?session_id={}", host, session_id);
    let _ = tx.send(Event::default().event("endpoint").data(endpoint_url));

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(Ok::<Event, std::convert::Infallible>);
    let cleanup = SessionCleanupStream {
        inner: Box::pin(stream),
        session_id: session_id.clone(),
        manager: Arc::clone(&manager),
    };

    let mut response = Sse::new(cleanup).into_response();
    response.headers_mut().insert(
        header::HeaderName::from_static("mcp-session-id"),
        header::HeaderValue::from_str(&session_id).unwrap(),
    );
    response
}

// ---------------------------------------------------------------------------
// Server Bootstrap
// ---------------------------------------------------------------------------

pub async fn start_sse_server(server: Arc<Server>, port: u16) -> anyhow::Result<()> {
    let manager = Arc::new(SseManager::new(server));

    let app = Router::new()
        .route(
            "/sse",
            get(sse_handler)
                .post(message_handler)
                .delete(delete_session_handler),
        )
        .route("/message", post(message_handler))
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_not_required),
        )
        .route(
            "/.well-known/oauth-protected-resource/sse",
            get(oauth_not_required),
        )
        .layer(middleware::from_fn(logging_middleware))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .with_state(manager);

    let addr = format!("[::]:{}", port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            let fallback = tokio::net::TcpListener::bind("[::]:0").await?;
            let actual = fallback.local_addr()?;
            tracing::warn!(
                configured = port,
                actual = actual.port(),
                "MCP port in use, falling back to random free port"
            );
            fallback
        }
        Err(e) => return Err(e.into()),
    };
    info!("SSE server listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn delete_session_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}

/// MCP 2025: respond to OAuth discovery with 200 + no auth required
async fn oauth_not_required() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        r#"{"resource":"vector-mcp-rust","authorization_servers":[]}"#,
    )
}

async fn logging_middleware(req: Request<axum::body::Body>, next: Next) -> impl IntoResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    info!(method = %method, uri = %uri, "Incoming request");
    let response = next.run(req).await;
    info!(method = %method, uri = %uri, status = %response.status(), "Response sent");
    response
}
