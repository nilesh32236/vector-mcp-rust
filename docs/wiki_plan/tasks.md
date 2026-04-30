# Local Code Wiki Tasks

## Phase 1: Backend Infrastructure
- [x] **Update `Cargo.toml`**:
    - [x] Add `fs` feature to `tower-http`.
- [x] **Modify `src/api/mod.rs`**:
    - [x] Import `tower_http::services::{ServeDir, ServeFile}`.
    - [x] Update `router` function to include `ServeDir` for the `public` directory.
    - [x] Implement a fallback to `index.html` for client-side routing support.
- [x] **Create Directory Structure**:
    - [x] Create `public/` directory in the root.
    - [x] Create `frontend/` for Next.js app.

## Phase 2: Frontend Foundation (Next.js)
- [x] **Initialize Next.js App**:
    - [x] Use `npx create-next-app` in `frontend/`.
    - [x] Configure for Static Export (`output: 'export'`).
- [x] **Implement Design System**:
    - [x] Apply colors (#0F172A background, #22C55E CTA).
    - [x] Setup fonts (JetBrains Mono, IBM Plex Sans).
    - [x] Implement Tailwind config based on the design system.
- [x] **Create Core Components**:
    - [x] Hero section with Search Bar.
    - [x] Result Card component.
    - [x] Sidebar for stats and navigation.

## Phase 3: Integration & Features
- [x] **Connect to Rust API**:
    - [x] Implement API client using `fetch`.
    - [x] Handle Search and Context endpoints.
    - [x] Real-time polling for Status.
- [x] **Implement Advanced Features**:
    - [x] Syntax highlighting for code snippets.
    - [x] Similarity score visualizations.
    - [x] Markdown rendering for documentation.
- [x] **Build & Deploy Pipeline**:
    - [x] Script to build Next.js and sync to `public/`.
    - [x] Verify Rust serves the exported files.

## Phase 4: Verification
- [ ] **Manual Testing**:
    - Verify `http://localhost:port/` serves the UI.
    - Verify search results match API expectations.
    - Verify MCP SSE functionality remains unaffected.
- [ ] **Build Check**:
    - Ensure `cargo build` and `cargo run` work as expected with the new assets.
