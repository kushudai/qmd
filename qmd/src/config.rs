//! Platform-aware path resolution (XDG-style).
//!
//! All qmd data lives under standard platform directories:
//! - **Database**: `~/.cache/qmd/{index}.sqlite`
//! - **Config**:   `~/.config/qmd/{index}.yml` (override: `$QMD_CONFIG_DIR`)
//! - **Models**:   `~/.cache/qmd/models/`

use std::path::PathBuf;

use crate::error::{Error, Result};

/// Default glob pattern for markdown file discovery.
pub const DEFAULT_GLOB: &str = "**/*.md";

/// Directories always excluded from indexing.
pub const EXCLUDE_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    ".cache",
    "vendor",
    "dist",
    "build",
    "target",
];

/// Resolve the database path for a named index.
///
/// Returns `~/.cache/qmd/{index_name}.sqlite`, creating parent dirs.
pub fn db_path(index_name: &str) -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .ok_or_else(|| Error::Config("cannot determine cache directory".into()))?
        .join("qmd");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join(format!("{index_name}.sqlite")))
}

/// Resolve the config directory.
///
/// Uses `$QMD_CONFIG_DIR` if set, otherwise `~/.config/qmd`.
pub fn config_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("QMD_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    dirs::config_dir()
        .map(|d| d.join("qmd"))
        .ok_or_else(|| Error::Config("cannot determine config directory".into()))
}

/// Resolve the config file path for a named index.
///
/// Returns `~/.config/qmd/{index_name}.yml`.
pub fn config_path(index_name: &str) -> Result<PathBuf> {
    Ok(config_dir()?.join(format!("{index_name}.yml")))
}

/// Resolve the model cache directory.
///
/// Returns `~/.cache/qmd/models/`, creating it if needed.
pub fn model_cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .ok_or_else(|| Error::Config("cannot determine cache directory".into()))?
        .join("qmd")
        .join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}
