//! Embedding engine backed by [`fastembed`].

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::{Error, Result};

/// Text embedding engine.
pub struct Embedder {
    /// Underlying fastembed model.
    model: TextEmbedding,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    /// Create an embedder with the default model ([`AllMiniLML6V2`]).
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create an embedder with a specific model.
    pub fn with_model(kind: EmbeddingModel) -> Result<Self> {
        let opts = InitOptions::new(kind).with_show_download_progress(true);
        let model = TextEmbedding::try_new(opts).map_err(|e| Error::Embedding(e.to_string()))?;
        Ok(Self { model })
    }

    /// Embed a single query string.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let results = self
            .model
            .embed(vec![query], None)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding result".into()))
    }

    /// Embed a batch of documents.
    pub fn embed_documents(&mut self, docs: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(docs, None)
            .map_err(|e| Error::Embedding(e.to_string()))
    }
}
