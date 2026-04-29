# Local Code Wiki Requirements

## Overview
The goal is to integrate a lightweight, high-performance "Local Code Wiki" directly into the `vector-mcp-rust` repository. This UI will provide a visual interface for searching and exploring the codebase, leveraging existing vector search and context capabilities.

## Core Features
1.  **Semantic Search Interface**: A search bar to query the codebase using the existing `/api/search` endpoint.
2.  **Result Visualization**: Display search results with snippets of code, file paths, and similarity scores.
3.  **Code Exploration**: Ability to view code context and potentially navigate through the indexed files.
4.  **Real-time Indexing Status**: Display the current status of the indexing process using the `/api/tools/status` endpoint.
5.  **Context Injection**: A way to manually add context to the vector store via the `/api/context` endpoint.
6.  **Responsive Design**: A modern, responsive UI that works across different screen sizes.

## Technical Constraints
1.  **No External Backend**: The frontend must be served directly by the existing Rust Axum server. No Node.js, Python, or other external runtimes are allowed for the production UI.
2.  **Lightweight SPA**: The UI should be a Single Page Application (SPA) built with vanilla HTML, CSS, and JavaScript to ensure minimal footprint and fast load times.
3.  **Minimal Dependencies**: Avoid heavy frontend frameworks. Use modern browser APIs for state management and UI updates.
4.  **Non-Interference**: The UI integration must not interfere with the existing MCP (Model Context Protocol) SSE functionality or other API endpoints.
5.  **Static File Serving**: Use `tower-http`'s `ServeDir` to efficiently serve files from a `public/` directory.

## Expected Behavior
-   **Landing Page**: A clean, modern dashboard showing indexing stats and a prominent search bar.
-   **Search Flow**: As the user types or submits a query, the UI calls `/api/search` and displays results dynamically.
-   **Detail View**: Clicking on a search result shows more context or the full file content (if available via API).
-   **Error Handling**: Graceful handling of network errors or server-side failures with user-friendly notifications.
