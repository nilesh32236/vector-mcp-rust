## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.

## 2026-04-16 - [Fix Command Argument Injection & UTF-8 conversion in process invocations]
**Vulnerability:** External tools like `git`, `rustfmt`, and `gofmt` were invoked using arguments generated from user-supplied paths via `abs.to_str().unwrap_or("")`. This could allow command argument injection if a file path started with `-`, and would fail on invalid UTF-8 paths.
**Learning:** `tokio::process::Command` safely escapes options, but tools themselves might interpret positional arguments that start with `-` as flags. Also, paths on Unix are arbitrary byte sequences, not necessarily valid UTF-8.
**Prevention:** Always use the `--` flag before passing user-supplied paths to command line tools. Furthermore, pass `PathBuf` references directly to `.arg()` rather than converting them to strings, avoiding unsafe `.to_str()` conversions that can drop data or use default values on failure.
