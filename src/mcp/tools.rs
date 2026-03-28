//! Tool definitions — exact mirror of the Go `registerTools()` schemas.

use serde_json::json;

use super::protocol::{ToolInfo, ToolInputSchema};

/// Returns all tool definitions matching the Go MCP server's tool registry.
pub fn tool_definitions() -> Vec<ToolInfo> {
    vec![
        tool_ping(),
        tool_trigger_project_index(),
        tool_set_project_root(),
        tool_store_context(),
        tool_get_related_context(),
        tool_find_duplicate_code(),
        tool_delete_context(),
        tool_index_status(),
        tool_get_codebase_skeleton(),
        tool_check_dependency_health(),
        tool_generate_docstring_prompt(),
        tool_analyze_architecture(),
        tool_find_dead_code(),
        tool_filesystem_grep(),
        tool_search_codebase(),
        tool_get_indexing_diagnostics(),
        tool_get_summarized_context(),
        tool_verify_implementation_gap(),
        tool_find_missing_tests(),
        tool_list_api_endpoints(),
        tool_get_code_history(),
        tool_reindex_all(),
        tool_verify_proposed_change(),
        tool_check_llm_connectivity(),
        tool_distill_knowledge(),
    ]
}

fn tool_reindex_all() -> ToolInfo {
    ToolInfo {
        name: "reindex_all".into(),
        description:
            "Force a full re-index of the entire project to refresh metadata and AI summaries."
                .into(),
        input_schema: empty_schema(),
    }
}

fn tool_get_summarized_context() -> ToolInfo {
    ToolInfo {
        name: "get_summarized_context".into(),
        description: "Retrieves context for a query and uses a local LLM to provide a concise summary instead of raw chunks.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "query": { "type": "string", "description": "The search query to summarize" },
                "topK": { "type": "number", "description": "Optional: Number of chunks to include in summary (default 5)" }
            }),
            required: vec!["query".into()],
        },
    }
}

fn tool_verify_implementation_gap() -> ToolInfo {
    ToolInfo {
        name: "verify_implementation_gap".into(),
        description:
            "Verifies if things from docs and client feedback are actually implemented in the code."
                .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "query": { "type": "string", "description": "The requirement or feedback query to verify (e.g. 'user authentication')" }
            }),
            required: vec!["query".into()],
        },
    }
}

fn tool_find_missing_tests() -> ToolInfo {
    ToolInfo {
        name: "find_missing_tests".into(),
        description: "Identifies exported symbols that lack corresponding test coverage by mapping source to tests.".into(),
        input_schema: empty_schema(),
    }
}

fn tool_list_api_endpoints() -> ToolInfo {
    ToolInfo {
        name: "list_api_endpoints".into(),
        description:
            "Identifies potential API route definitions in the codebase across various frameworks."
                .into(),
        input_schema: empty_schema(),
    }
}

fn tool_get_code_history() -> ToolInfo {
    ToolInfo {
        name: "get_code_history".into(),
        description: "Retrieves recent git history (last 10 commits) for a specific file to understand its evolution.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "file_path": { "type": "string", "description": "The relative path of the file to check history for" }
            }),
            required: vec!["file_path".into()],
        },
    }
}

// ---------------------------------------------------------------------------
// Individual tool builders
// ---------------------------------------------------------------------------

fn tool_ping() -> ToolInfo {
    ToolInfo {
        name: "ping".into(),
        description: "Check server connectivity".into(),
        input_schema: empty_schema(),
    }
}

fn tool_trigger_project_index() -> ToolInfo {
    ToolInfo {
        name: "trigger_project_index".into(),
        description: "Trigger a full background index of a project. Use this when you first open a project or after major changes to ensure the vector index is up to date.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "project_path": { "type": "string", "description": "Absolute path to the project root" }
            }),
            required: vec![],
        },
    }
}

fn tool_set_project_root() -> ToolInfo {
    ToolInfo {
        name: "set_project_root".into(),
        description: "Dynamically switch the active project root and update the file watcher. Use this when moving between different codebases or monorepo packages.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "project_path": { "type": "string", "description": "Absolute path to the new project root" }
            }),
            required: vec![],
        },
    }
}

fn tool_store_context() -> ToolInfo {
    ToolInfo {
        name: "store_context".into(),
        description: "Store general project rules, architectural decisions, or shared context for other agents to read. This helps maintain consistency across different AI sessions.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "text": { "type": "string", "description": "The text context to store." },
                "project_id": { "type": "string", "description": "The project this context belongs to. Defaults to the current project." }
            }),
            required: vec![],
        },
    }
}

fn tool_get_related_context() -> ToolInfo {
    ToolInfo {
        name: "get_related_context".into(),
        description: "Retrieve context for a file and its local dependencies, optionally cross-referencing other projects. Use this to understand how a specific file fits into the larger codebase.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "filePath": { "type": "string", "description": "The relative path of the file to analyze" },
                "max_tokens": { "type": "number", "description": "Optional: Maximum total tokens to include in the context (default 10,000)" }
            }),
            required: vec![],
        },
    }
}

fn tool_find_duplicate_code() -> ToolInfo {
    ToolInfo {
        name: "find_duplicate_code".into(),
        description: "Scans a specific path to find duplicated logic across namespaces. Use this during refactoring to identify consolidation opportunities.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "target_path": { "type": "string", "description": "The relative path to check" }
            }),
            required: vec![],
        },
    }
}

fn tool_delete_context() -> ToolInfo {
    ToolInfo {
        name: "delete_context".into(),
        description:
            "Delete specific shared memory context, or completely wipe a project's vector index."
                .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "target_path": { "type": "string", "description": "The exact file path, context ID, or 'ALL' to clear the whole project." },
                "project_id": { "type": "string", "description": "The project ID to target. Defaults to the current project." },
                "dry_run": { "type": "boolean", "description": "Optional: If true, returns the list of files that would be deleted without actually deleting them." }
            }),
            required: vec![],
        },
    }
}

fn tool_index_status() -> ToolInfo {
    ToolInfo {
        name: "index_status".into(),
        description: "Check index status and background progress. Use this to verify if the server is still indexing or if it's ready for queries.".into(),
        input_schema: empty_schema(),
    }
}

fn tool_get_codebase_skeleton() -> ToolInfo {
    ToolInfo {
        name: "get_codebase_skeleton".into(),
        description: "Returns a topological tree map of the directory structure. Use this to progressively explore large codebases by specifying sub-directories and depths.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "target_path": { "type": "string", "description": "Relative or absolute path to the directory to map (optional, defaults to project root)." },
                "max_depth": { "type": "number", "description": "Maximum depth of the tree to generate (optional, defaults to 3)." },
                "include_pattern": { "type": "string", "description": "Optional: Only include files matching this glob pattern" },
                "exclude_pattern": { "type": "string", "description": "Optional: Exclude files matching this glob pattern" },
                "max_items": { "type": "number", "description": "Optional: Maximum number of items to return (default 1000)" }
            }),
            required: vec![],
        },
    }
}

fn tool_check_dependency_health() -> ToolInfo {
    ToolInfo {
        name: "check_dependency_health".into(),
        description: "Analyzes a directory's package.json against its indexed imports to identify missing dependencies in the manifest.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "directory_path": { "type": "string", "description": "The path to the directory containing package.json and source files" }
            }),
            required: vec![],
        },
    }
}

fn tool_generate_docstring_prompt() -> ToolInfo {
    ToolInfo {
        name: "generate_docstring_prompt".into(),
        description: "Generates a highly contextual prompt for an LLM to write professional documentation for a specific entity.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "file_path": { "type": "string", "description": "The relative path of the file" },
                "entity_name": { "type": "string", "description": "The name of the function or class to document" },
                "language": { "type": "string", "description": "Optional: The language of the file (e.g., 'Go', 'TypeScript', 'Python'). Extracted from file extension if omitted." }
            }),
            required: vec![],
        },
    }
}

fn tool_analyze_architecture() -> ToolInfo {
    ToolInfo {
        name: "analyze_architecture".into(),
        description: "Generates a Mermaid.js dependency graph between packages in a monorepo."
            .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "monorepo_prefix": { "type": "string", "description": "Optional prefix for monorepo packages (e.g., '@herexa/')" }
            }),
            required: vec![],
        },
    }
}

fn tool_find_dead_code() -> ToolInfo {
    ToolInfo {
        name: "find_dead_code".into(),
        description: "Identifies potentially dead code by finding exported symbols that are never imported or called.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "exclude_paths": {
                    "type": "array",
                    "description": "Optional list of file paths or patterns to exclude from dead code analysis.",
                    "items": { "type": "string" }
                },
                "is_library": { "type": "boolean", "description": "Optional: Set to true if analyzing a library where public exports are expected. Only flags unused symbols inside internal/ or marked as private." }
            }),
            required: vec![],
        },
    }
}

fn tool_filesystem_grep() -> ToolInfo {
    ToolInfo {
        name: "filesystem_grep".into(),
        description: "Exact string or regex search across the project files.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "query": { "type": "string", "description": "The search query (string or regex)" },
                "include_pattern": { "type": "string", "description": "Optional: Glob pattern to filter files (e.g. '*.go')" },
                "is_regex": { "type": "boolean", "description": "Whether the query is a regular expression" }
            }),
            required: vec![],
        },
    }
}

fn tool_search_codebase() -> ToolInfo {
    ToolInfo {
        name: "search_codebase".into(),
        description: "Unified semantic and lexical search across the codebase. Replaces retrieve_context and retrieve_docs.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "query": { "type": "string", "description": "The natural language search query" },
                "category": { "type": "string", "description": "Optional: 'code' or 'document'. Handles both 'code' and 'document' retrieval. Defaults to searching both." },
                "topK": { "type": "number", "description": "Number of results to return (default 10)" },
                "path_filter": { "type": "string", "description": "Optional: Only search files whose path contains this string" },
                "min_score": { "type": "number", "description": "Optional: Minimum similarity score (0.0 to 1.0) to include a result" },
                "max_tokens": { "type": "number", "description": "Optional: Maximum total tokens to include in the context (default 10,000)" },
                "cross_reference_projects": {
                    "type": "array",
                    "description": "Optional list of project IDs to search across",
                    "items": { "type": "string" }
                }
            }),
            required: vec![],
        },
    }
}

fn tool_verify_proposed_change() -> ToolInfo {
    ToolInfo {
        name: "verify_proposed_change".into(),
        description: "Checks a proposed code change or task against stored Knowledge Items and Architectural Decisions to ensure pattern compliance.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "proposed_change": {
                    "type": "string",
                    "description": "The description or diff of the proposed change"
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional: Filter by project ID"
                }
            }),
            required: vec!["proposed_change".into()],
        },
    }
}
fn tool_get_indexing_diagnostics() -> ToolInfo {
    ToolInfo {
        name: "get_indexing_diagnostics".into(),
        description: "Provides detailed diagnostics on the indexing process, including recent errors and queue status.".into(),
        input_schema: empty_schema(),
    }
}

fn tool_check_llm_connectivity() -> ToolInfo {
    ToolInfo {
        name: "check_llm_connectivity".into(),
        description: "Checks the Gemini API connectivity and lists available models for the current API key.".into(),
        input_schema: empty_schema(),
    }
}

fn tool_distill_knowledge() -> ToolInfo {
    ToolInfo {
        name: "distill_knowledge".into(),
        description: "Analyzes a directory and automatically generates a Knowledge Item summary.".into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "path": { "type": "string", "description": "The relative path to analyze" }
            }),
            required: vec!["path".into()],
        },
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_schema() -> ToolInputSchema {
    ToolInputSchema {
        schema_type: "object",
        properties: json!({}),
        required: vec![],
    }
}
