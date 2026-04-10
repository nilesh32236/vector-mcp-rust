## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.
## 2025-02-14 - Argument Injection and Unsafe Path Conversions
**Vulnerability:** Invoking external tools with filenames starting with dashes (e.g. `-foo.rs`) interpreted as arguments, as well as unsafe string conversion of `PathBuf` returning an empty string.
**Learning:** Shelling out to Git, GoFmt, RustFmt, and Prettier by directly converting paths to strings (`to_str().unwrap_or("")`) poses risks of CLI arguments injection or executing on arbitrary directories / entering hanging states.
**Prevention:** Always append `--` to distinguish flags from file paths. Feed `PathBuf` directly via `tokio::process::Command::arg(&abs)` to bypass unreliable utf-8 string fallback.
