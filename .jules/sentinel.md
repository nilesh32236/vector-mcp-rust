## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.

## 2026-04-06 - [Argument Injection and non-UTF-8 Path Risks in Subprocesses]

**Vulnerability:** External tools (e.g. `gofmt`, `git log`) invoked via `tokio::process::Command::new` passed user-supplied file paths using `abs.to_str().unwrap_or("")` without `--` argument separation, exposing the system to argument injection and failing to correctly handle non-UTF-8 file paths.
**Learning:** Argument injection occurs when an attacker can craft a filename that starts with `-` (like `-w`), tricking the external command into treating it as a flag instead of a file. Using `.unwrap_or("")` defaults non-UTF-8 paths to empty strings, which could execute against unintended directory scopes or fail silently.
**Prevention:** Always use `--` to clearly separate positional file arguments from flags, and pass `PathBuf` directly via `Command::arg(&abs)` to safely encode paths regardless of UTF-8 correctness.
