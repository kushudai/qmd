//! Cross-encoder reranking engine.

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::embed::{cosine_similarity, fmt_document, fmt_query};
use crate::error::{Error, Result};
use crate::model;

/// Scored rerank result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Scored {
    /// Identifier (file path or key).
    pub key: String,
    /// Relevance score (higher is better).
    pub score: f32,
    /// Original index in the input list.
    pub index: usize,
}

/// Reranking engine backed by a GGUF model.
pub struct RerankEngine {
    /// Llama backend handle.
    backend: LlamaBackend,
    /// Loaded GGUF model.
    model: Arc<LlamaModel>,
}

impl std::fmt::Debug for RerankEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerankEngine").finish_non_exhaustive()
    }
}

impl RerankEngine {
    /// Load a model from a file path.
    pub fn new(path: &Path) -> Result<Self> {
        let loaded = model::load_gguf(path)?;
        Ok(Self {
            backend: loaded.backend,
            model: loaded.model,
        })
    }

    /// Load the default rerank model from cache.
    pub fn load_default() -> Result<Self> {
        Self::new(&model::get_path(model::RERANK.filename)?)
    }

    /// Whether the default rerank model is cached locally.
    #[must_use]
    pub fn is_available() -> bool {
        model::exists(model::RERANK.filename).unwrap_or(false)
    }

    /// Rerank candidates by relevance to `query`.
    ///
    /// Each candidate is `(key, text, optional_title)`. Returns results
    /// sorted by descending score.
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[(String, String, Option<String>)],
    ) -> Result<Vec<Scored>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let ctx_params = LlamaContextParams::default().with_embeddings(true);
        let q_emb = self.embed_text(&fmt_query(query), &ctx_params)?;

        let mut results: Vec<Scored> = candidates
            .iter()
            .enumerate()
            .map(|(i, (key, text, title))| {
                let formatted = fmt_document(text, title.as_deref());
                let score = self
                    .embed_text(&formatted, &ctx_params)
                    .map(|emb| cosine_similarity(&q_emb, &emb))
                    .unwrap_or(0.0);
                Scored {
                    key: key.clone(),
                    score,
                    index: i,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Compute an embedding vector for formatted text.
    fn embed_text(&self, text: &str, ctx_params: &LlamaContextParams) -> Result<Vec<f32>> {
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params.clone())
            .map_err(|e| Error::Model(format!("context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(Error::Model("empty token sequence".into()));
        }

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
        Ok(emb.to_vec())
    }
}
