//! Path utilities for virtual paths and indexing exclusions.

use std::path::Path;

use crate::config::EXCLUDE_DIRS;

/// Check if a path component should be excluded from indexing.
///
/// Excludes hidden directories (starting with `.`) and common build
/// artifact directories defined in [`EXCLUDE_DIRS`](crate::config::EXCLUDE_DIRS).
#[must_use]
pub fn should_exclude(path: &Path) -> bool {
    path.components().any(|c| {
        if let std::path::Component::Normal(name) = c {
            let s = name.to_string_lossy();
            s.starts_with('.') || EXCLUDE_DIRS.contains(&s.as_ref())
        } else {
            false
        }
    })
}

/// Check if a string is a virtual `qmd://` path.
#[must_use]
pub fn is_virtual(path: &str) -> bool {
    let t = path.trim();
    t.starts_with("qmd:") || t.starts_with("//")
}

/// Parse a virtual path into `(collection, file_path)`.
///
/// Accepts `qmd://collection/path` and `//collection/path`.
#[must_use]
pub fn parse_virtual(path: &str) -> Option<(String, String)> {
    let trimmed = path.trim();
    let rest = trimmed
        .strip_prefix("qmd://")
        .or_else(|| trimmed.strip_prefix("qmd:"))
        .or_else(|| trimmed.strip_prefix("//"))?;
    let clean = rest.trim_start_matches('/');
    let mut parts = clean.splitn(2, '/');
    let collection = parts.next()?.to_string();
    let file_path = parts.next().unwrap_or("").to_string();
    Some((collection, file_path))
}

/// Build a virtual path: `qmd://{collection}/{path}`.
#[must_use]
pub fn build_virtual(collection: &str, path: &str) -> String {
    format!("qmd://{collection}/{path}")
}

/// Check if a string looks like a docid (`#` prefix + 6 hex chars).
#[must_use]
pub fn is_docid(s: &str) -> bool {
    let clean = s.trim_start_matches('#');
    clean.len() == 6 && clean.chars().all(|c| c.is_ascii_hexdigit())
}
