## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.

## 2026-04-06 - [Argument Injection in External Command Handlers]

**Vulnerability:** User-supplied paths were converted to string and placed alongside command arguments for commands like `gofmt`, `rustfmt`, and `git`. A malicious path beginning with `-` could be parsed as a command argument, causing command injection issues.
**Learning:** Argument injection can happen even if file paths are strictly bounded inside the project root workspace because file names themselves can be malicious command-line flags. Furthermore, lossy UTF-8 conversion via `.to_str().unwrap_or("")` strips away information and provides a weak security layer against specially crafted inputs.
**Prevention:** Always use the `--` flag before passing variable arguments like file paths to external commands. Furthermore, pass the raw `PathBuf` object via `tokio::process::Command::arg` instead of converting to standard UTF-8 strings.
