//! Crate-level error types.

use thiserror::Error;

/// Unified error type for all qmd operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// SQLite / rusqlite error.
    ///
    /// Formatted with the SQLite extended error code in brackets when
    /// available, so opaque failures like a bare `constraint failed`
    /// surface as `database: constraint failed [SQLITE_CONSTRAINT_PRIMARYKEY]`.
    #[error("database: {}", format_sqlite_error(.0))]
    Database(#[from] rusqlite::Error),

    /// Filesystem I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML serialization error.
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yml::Error),

    /// Embedding model error.
    #[error("embedding: {0}")]
    Embedding(String),

    /// Reranking model error.
    #[error("rerank: {0}")]
    Rerank(String),

    /// Configuration or path resolution error.
    #[error("config: {0}")]
    Config(String),

    /// Document not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Collection already exists.
    #[error("collection already exists: {0}")]
    CollectionExists(String),
}

/// Crate-level result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Format a `rusqlite::Error`, appending the extended SQLite error code in
/// brackets for `SqliteFailure` so opaque messages like `constraint failed`
/// gain enough context to be diagnosable.
///
/// Codes from <https://www.sqlite.org/rescode.html> — primary code 19
/// (`SQLITE_CONSTRAINT`) plus a sub-class shifted left 8 bits.
fn format_sqlite_error(err: &rusqlite::Error) -> String {
    let rusqlite::Error::SqliteFailure(code, _) = err else {
        return err.to_string();
    };
    let label = match code.extended_code {
        275 => "SQLITE_CONSTRAINT_CHECK",
        531 => "SQLITE_CONSTRAINT_COMMITHOOK",
        787 => "SQLITE_CONSTRAINT_FOREIGNKEY",
        1043 => "SQLITE_CONSTRAINT_FUNCTION",
        1299 => "SQLITE_CONSTRAINT_NOTNULL",
        1555 => "SQLITE_CONSTRAINT_PRIMARYKEY",
        1811 => "SQLITE_CONSTRAINT_TRIGGER",
        2067 => "SQLITE_CONSTRAINT_UNIQUE",
        2323 => "SQLITE_CONSTRAINT_VTAB",
        2579 => "SQLITE_CONSTRAINT_ROWID",
        _ => return format!("{err} [extended code {}]", code.extended_code),
    };
    format!("{err} [{label} ({})]", code.extended_code)
}
