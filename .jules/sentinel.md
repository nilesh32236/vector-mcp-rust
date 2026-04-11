## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.
## 2024-04-11 - [Argument Injection in Command Invocations]
**Vulnerability:** External command invocations in `handle_run_linter` (`tokio::process::Command`) were formatting arguments directly with file paths strings that could be interpreted as command line flags (e.g., `-e` or `--check`). Additionally, `.to_str().unwrap_or(&path)` was used, mapping non-UTF8 paths unreliably.
**Learning:** Even after validation through `path_guard` to prevent directory traversal, valid filenames starting with `-` or `--` within the workspace can be processed as arbitrary options by tools like `gofmt` or `rustfmt`, leading to potential command execution or failures.
**Prevention:** Always use the `--` flag before passing positional file arguments to signal the end of command options. Additionally, pass `PathBuf` references directly to `tokio::process::Command::arg()` rather than converting them to strings to handle invalid UTF-8 paths safely.
