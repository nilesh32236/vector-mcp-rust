//! SSE transport for MCP.

use axum::{
    Router,
    extract::{Query, State, Host},
    http::StatusCode,
    response::{Sse, sse::Event},
    routing::{get, post},
};
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

use super::server::Server;

// ---------------------------------------------------------------------------
// SSE Session Manager
// ---------------------------------------------------------------------------

type SessionSender = mpsc::UnboundedSender<Event>;

pub struct SseManager {
    sessions: DashMap<String, SessionSender>,
    server: Arc<Server>,
}

impl SseManager {
    pub fn new(server: Arc<Server>) -> Self {
        Self {
            sessions: DashMap::new(),
            server,
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
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------


#[derive(serde::Deserialize)]
struct MessageQuery {
    session_id: String,
}

/// GET /sse — Start the SSE connection.
async fn sse_handler(
    Host(host): Host,
    State(manager): State<Arc<SseManager>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::unbounded_channel();

    manager.sessions.insert(session_id.clone(), tx.clone());

    info!(session = %session_id, "new SSE session");

    // Send the endpoint event as per MCP spec.
    // The client will use this URL to POST messages.
    // Use an absolute URL as some clients are sensitive to relative paths.
    // Dynamically build the absolute URL based on the incoming Host header.
    let endpoint_url = format!(
        "http://{}/message?session_id={}",
        host, session_id
    );
    let _ = tx.send(Event::default().event("endpoint").data(endpoint_url));

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok);
    let session_cleanup = SessionCleanupStream {
        inner: Box::pin(stream),
        session_id: session_id.clone(),
        manager: Arc::clone(&manager),
    };

    Sse::new(session_cleanup)
}

/// POST /message — Receive JSON-RPC messages.
async fn message_handler(
    State(manager): State<Arc<SseManager>>,
    Query(query): Query<MessageQuery>,
    body: String,
) -> StatusCode {
    let sender = manager.sessions.get(&query.session_id).map(|s| s.clone());

    let tx = match sender {
        Some(t) => t,
        None => {
            warn!(session = %query.session_id, "message posted to invalid session");
            return StatusCode::NOT_FOUND;
        }
    };

    let server = Arc::clone(&manager.server);
    let session_id = query.session_id.clone();

    // Process the message asynchronously and send the response back via SSE.
    tokio::spawn(async move {
        let response = server.process_message(&body).await;
        let response_json = serde_json::to_string(&response).unwrap_or_default();

        let event = Event::default().event("message").data(response_json);
        if tx.send(event).is_err() {
            warn!(session = %session_id, "failed to send response to SSE stream (client disconnected)");
        }
    });

    StatusCode::ACCEPTED
}

// ---------------------------------------------------------------------------
// Server Bootstrap
// ---------------------------------------------------------------------------

pub async fn start_sse_server(server: Arc<Server>, port: u16) -> anyhow::Result<()> {
    let manager = Arc::new(SseManager::new(server));

    // Handle both /sse and /message with a unified approach to avoid 405 Method Not Allowed.
    // Some clients might POST to /sse or GET /message depending on implementation details.
    let app = Router::new()
        .route("/sse", get(sse_handler).post(message_handler_permissive))
        .route("/message", post(message_handler).get(not_found_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
        )
        .with_state(manager);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("SSE server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

/// A version of message_handler that can handle session_id from query or body if needed.
async fn message_handler_permissive(
    state: State<Arc<SseManager>>,
    query: Query<HashMap<String, String>>,
    body: String,
) -> StatusCode {
    let session_id = query.get("session_id").cloned();
    match session_id {
        Some(sid) => message_handler(state, Query(MessageQuery { session_id: sid }), body).await,
        None => {
            warn!("POST to SSE without session_id");
            StatusCode::METHOD_NOT_ALLOWED
        }
    }
}

async fn not_found_handler() -> StatusCode {
    StatusCode::NOT_FOUND
}
