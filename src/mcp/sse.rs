//! SSE transport for MCP.

use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{Sse, sse::Event},
    routing::{get, post},
};
use futures::stream::Stream;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
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
    sessions: Mutex<HashMap<String, SessionSender>>,
    server: Arc<Server>,
    port: u16,
}

impl SseManager {
    pub fn new(server: Arc<Server>, port: u16) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            server,
            port,
        }
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
    State(manager): State<Arc<SseManager>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::unbounded_channel();

    {
        let mut sessions = manager.sessions.lock().await;
        sessions.insert(session_id.clone(), tx.clone());
    }

    info!(session = %session_id, "new SSE session");

    // Send the endpoint event as per MCP spec.
    // The client will use this URL to POST messages.
    // Use an absolute URL as some clients are sensitive to relative paths.
    let endpoint_url = format!(
        "http://localhost:{}/message?session_id={}",
        manager.port, session_id
    );
    let _ = tx.send(Event::default().event("endpoint").data(endpoint_url));

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok);

    Sse::new(stream)
}

/// POST /message — Receive JSON-RPC messages.
async fn message_handler(
    State(manager): State<Arc<SseManager>>,
    Query(query): Query<MessageQuery>,
    body: String,
) -> StatusCode {
    let sender = {
        let sessions = manager.sessions.lock().await;
        sessions.get(&query.session_id).cloned()
    };

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
        if let Err(_) = tx.send(event) {
            warn!(session = %session_id, "failed to send response to SSE stream (client disconnected)");
        }
    });

    StatusCode::ACCEPTED
}

// ---------------------------------------------------------------------------
// Server Bootstrap
// ---------------------------------------------------------------------------

pub async fn start_sse_server(server: Arc<Server>, port: u16) -> anyhow::Result<()> {
    let manager = Arc::new(SseManager::new(server, port));

    // Handle both /sse and /message with a unified approach to avoid 405 Method Not Allowed.
    // Some clients might POST to /sse or GET /message depending on implementation details.
    let app = Router::new()
        .route("/sse", get(sse_handler).post(message_handler_permissive))
        .route("/message", post(message_handler).get(not_found_handler))
        .layer(CorsLayer::permissive())
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
