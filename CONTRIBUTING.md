# Contributing to vector-mcp-rust 🦀

We welcome contributions! Whether it's a bug fix, new feature, or documentation improvement, here's how you can help:

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nilesh32236/vector-mcp-rust.git
   ```
2. **Setup your environment**:
   Copy `.env.example` to `.env` and adjust as needed.
3. **Build and test**:
   ```bash
   cargo build
   cargo test
   ```

## Pull Request Process

1. Create a new branch for your changes: `git checkout -b feature/my-new-feature`
2. Commit your changes with clear messages.
3. Ensure the project builds without warnings: `cargo check`
4. Push your branch and open a Pull Request.

## Coding Standards

- Follow idiomatic Rust patterns.
- Ensure all public functions are documented.
- No `unwrap()` or `panic!`—always use `Result` and `anyhow`.
- Avoid `any` types or unsafe code where possible.

Thank you for helping improve vector-mcp-rust!
