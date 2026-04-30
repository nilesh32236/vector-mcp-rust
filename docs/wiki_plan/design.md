# Local Code Wiki Design

## Architecture
The "Local Code Wiki" follows a classic SPA (Single Page Application) architecture where the Rust backend serves static assets and provides RESTful APIs for data.

### Backend Integration
-   **Static Asset Serving**: We will use `tower-http` with the `fs` feature to serve the `public/` directory.
-   **Router Modification**: In `src/api/mod.rs`, we will add a fallback route to serve `index.html` for any non-API routes, and a `ServeDir` for the `/static` or root path.
-   **Endpoint Reuse**:
    -   `POST /api/search`: Used for semantic queries.
    -   `POST /api/context`: Used for manual context injection.
    -   `GET /api/tools/status`: Used for displaying indexing progress.
    -   `GET /api/tools/repos`: Used for identifying the current project root.

### Frontend Architecture
-   **Structure**: A single `index.html` acting as the shell.
-   **Styling**: A modern `style.css` using:
    -   CSS Variables for easy theme management.
    -   Flexbox/Grid for layouts.
    -   Glassmorphism effects (backdrop-filter) for a premium feel.
    -   Animations for smooth transitions between states.
-   **Logic**: A modular `app.js` using:
    -   `fetch` API for network requests.
    -   `EventTarget` or simple state objects for reactivity.
    -   DOM Templates for rendering search results.

## UI Layout
1.  **Sidebar**:
    -   System Status (Indexing %, Total Files).
    -   Recent Searches (stored in `localStorage`).
    -   Quick Links (Docs, GitHub).
2.  **Main Content**:
    -   **Hero Section**: Search bar with auto-suggestions or "Trending" queries.
    -   **Results Area**: Cards showing:
        -   File Path (clickable).
        -   Similarity Score (visual gauge).
        -   Code Snippet (syntax highlighted using `Prism.js` or similar lightweight library, or just pre-tags).
3.  **Context Modal**: A popup to paste text and add it as manual context.

## Implementation Details

### Axum Integration (`src/api/mod.rs`)
```rust
use tower_http::services::ServeDir;

// Inside router function:
let router = Router::new()
    .route("/api/search", post(handle_search))
    // ... other routes ...
    .fallback_service(ServeDir::new("public").fallback(ServeFile::new("public/index.html")));
```

### JSON Parsing
The frontend will handle the `SearchResponse` array:
```javascript
// Example Parsing
const results = await response.json();
results.forEach(res => {
    // res.id, res.text, res.similarity, res.path
    renderResultCard(res);
});
```

## Security
-   **CORS**: Already handled by `CorsLayer::permissive()`.
-   **Path Traversal**: `ServeDir` handles basic path traversal protection.
-   **Input Sanitization**: Ensure user input in the search bar is treated as plain text when making API calls.
