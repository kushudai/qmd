//! Core document types.

/// An indexed document in the store.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Document {
    /// Parent collection name.
    pub collection: String,
    /// Relative path within the collection.
    pub path: String,
    /// Document title (extracted from markdown headings).
    pub title: String,
    /// Content SHA-256 hash.
    pub hash: String,
    /// Last modification timestamp (RFC 3339).
    pub modified_at: String,
    /// Body length in bytes.
    pub body_len: usize,
    /// Body text (loaded on demand).
    pub body: Option<String>,
    /// Folder context description (from config).
    pub context: Option<String>,
}

impl Document {
    /// Short document id (first 6 hex chars of the hash).
    #[must_use]
    pub fn docid(&self) -> &str {
        &self.hash[..6.min(self.hash.len())]
    }

    /// Virtual path: `qmd://collection/path`.
    #[must_use]
    pub fn virtual_path(&self) -> String {
        crate::path::build_virtual(&self.collection, &self.path)
    }

    /// Display path: `collection/path`.
    #[must_use]
    pub fn display_path(&self) -> String {
        format!("{}/{}", self.collection, self.path)
    }
}

/// Search result with relevance score.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SearchResult {
    /// The matched document.
    pub doc: Document,
    /// Relevance score (higher is better).
    pub score: f64,
    /// Search backend that produced this result.
    pub source: SearchSource,
    /// Chunk position for vector results (char offset).
    pub chunk_pos: Option<usize>,
}

/// Which search backend produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchSource {
    /// Full-text search (BM25).
    Fts,
    /// Vector similarity search.
    Vec,
}
