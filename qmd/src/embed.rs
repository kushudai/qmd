//! Embedding engine for generating document vectors.

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::chunk::Tokenizer;
use crate::error::{Error, Result};
use crate::model;

/// Embedding engine backed by a GGUF model.
pub struct EmbeddingEngine {
    /// Llama backend handle.
    backend: LlamaBackend,
    /// Loaded GGUF model.
    model: Arc<LlamaModel>,
    /// Embedding dimensionality (populated after first call).
    dims: Option<usize>,
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngine")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

impl EmbeddingEngine {
    /// Load a model from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF model cannot be loaded.
    pub fn new(path: &Path) -> Result<Self> {
        let loaded = model::load_gguf(path)?;
        Ok(Self {
            backend: loaded.backend,
            model: loaded.model,
            dims: None,
        })
    }

    /// Load the default embedding model from cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the model path cannot be resolved or the model fails to load.
    pub fn load_default() -> Result<Self> {
        Self::new(&model::get_path(model::EMBED.filename)?)
    }

    /// Embed a document with an optional title.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn embed_document(&mut self, text: &str, title: Option<&str>) -> Result<Vec<f32>> {
        self.embed_formatted(&fmt_document(text, title))
    }

    /// Embed a search query.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        self.embed_formatted(&fmt_query(query))
    }

    /// Embedding dimensionality (available after the first call).
    #[must_use]
    pub const fn dims(&self) -> Option<usize> {
        self.dims
    }

    /// Embed pre-formatted text.
    fn embed_formatted(&mut self, text: &str) -> Result<Vec<f32>> {
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(Error::Model("empty token sequence".into()));
        }

        let n_ctx = (tokens.len() + 64).max(512);
        #[allow(clippy::cast_possible_truncation)]
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_n_ctx(std::num::NonZero::new(n_ctx as u32))
            .with_n_batch(n_ctx as u32)
            .with_n_ubatch(n_ctx as u32);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Model(format!("context: {e}")))?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| Error::Model(format!("batch: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Model(format!("decode: {e}")))?;

        let emb = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| Error::Model(format!("embeddings: {e}")))?;

        if self.dims.is_none() {
            self.dims = Some(emb.len());
        }

        Ok(emb.to_vec())
    }
}

impl Tokenizer for EmbeddingEngine {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        let tokens = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;
        Ok(tokens.len())
    }
}

/// Cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Format document text for embedding (nomic-style prefix).
pub(crate) fn fmt_document(text: &str, title: Option<&str>) -> String {
    let t = title.unwrap_or("none");
    format!("title: {t} | text: {text}")
}

/// Format a query for embedding.
pub(crate) fn fmt_query(query: &str) -> String {
    format!("task: search result | query: {query}")
}
