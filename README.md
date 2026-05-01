# Vector MCP Rust Server 🦀

A high-performance, purely deterministic Model Context Protocol (MCP) server written in Rust. This server provides advanced semantic search, architectural analysis, and codebase mutation capabilities directly to your LLM clients (like Claude Desktop or Cursor).

## Why Rust?

This Rust implementation is the successor to the original Go version (`vector-mcp-go`). It offers significant architectural advantages:

1.  **Local AI Privacy**: Built-in support for Hugging Face `candle-core`. It downloads and runs quantized GGUF models (like `Qwen2.5-0.5B-Instruct`) locally on your CPU for code summarization. No internet connection required for codebase indexing!
2.  **Absolute Memory Safety**: Uses Rust's `Drop` trait to perfectly manage C-allocated Tree-sitter AST memory. Zero garbage collection pauses and zero memory leaks during massive repository scans.
3.  **True Hybrid Search**: Native integration with `lancedb` for concurrent Vector ANN (IvfPq) and Lexical Full-Text Search (Tantivy BM25), fused with Reciprocal Rank Fusion (RRF).
4.  **Autonomous Reasoning**: Live CDC (Change Data Capture) watcher that automatically checks your Architectural Decision Records (ADRs) against new code, ensuring compliance on every save.
5.  **Context Efficiency**: Implements the **Fat Tool Pattern**, exposing only 5 super-tools to the LLM to prevent context exhaustion and hallucination.

## Features

-   **Blazing Fast Indexing**: Semantic AST chunking for Go, Rust, TypeScript, JavaScript, PHP, and Python.
-   **Mutation Safety**: Every `modify_workspace` patch is applied in-memory and verified against a real Language Server (LSP) like `tsserver` or `rust-analyzer`. If the patch causes a compiler error, it is rejected.
-   **Cross-Project Context**: Search across multiple indexed repositories simultaneously.

## Getting Started

Please see the [Usage Guide](usage_guide.md) for full instructions on building, configuring, and attaching the server to your favorite MCP client.

### Build Dependencies

For optimal build performance, this project is configured to use the `mold` linker and `sccache`.

#### Linux
```bash
sudo apt install mold
```

#### macOS
```bash
brew install mold
```

#### Cross-platform (Linux & macOS)
```bash
cargo install sccache
```

> **Alternative:** If you prefer not to install `mold` or `sccache`, comment out the `rustflags` and `rustc-wrapper` entries in `.cargo/config.toml` to use the default Rust toolchain without any additional linker or caching configuration.

## License
MIT
