//! Crate-level error types.

use thiserror::Error;

/// Unified error type for all qmd operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// SQLite / rusqlite error.
    #[error("database: {0}")]
    Database(#[from] rusqlite::Error),

    /// Filesystem I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// YAML serialization error.
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP request error.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    /// Configuration or path resolution error.
    #[error("config: {0}")]
    Config(String),

    /// Collection not found.
    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    /// Document not found.
    #[error("document not found: {0}")]
    DocumentNotFound(String),

    /// Model download, loading, or inference error.
    #[error("model: {0}")]
    Model(String),
}

/// Crate-level result alias.
pub type Result<T> = std::result::Result<T, Error>;
