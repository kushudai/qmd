//! Text generation engine for query expansion.

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{Error, Result};
use crate::model;
use crate::search::{Query, QueryType};

/// Text generation engine backed by a GGUF model.
pub struct GenerationEngine {
    /// Llama backend handle.
    backend: LlamaBackend,
    /// Loaded GGUF model.
    model: Arc<LlamaModel>,
}

impl std::fmt::Debug for GenerationEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationEngine").finish_non_exhaustive()
    }
}

impl GenerationEngine {
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
        })
    }

    /// Load the default generation model from cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the model path cannot be resolved or the model fails to load.
    pub fn load_default() -> Result<Self> {
        Self::new(&model::get_path(model::GENERATE.filename)?)
    }

    /// Whether the default generation model is cached locally.
    #[must_use]
    pub fn is_available() -> bool {
        model::exists(model::GENERATE.filename).unwrap_or(false)
    }

    /// Generate text from a prompt (up to `max_tokens`).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, context creation, or inference fails.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZero::new(4096));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Model(format!("context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| Error::Model(format!("batch: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Model(format!("decode: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(42),
        ]);

        let mut output = String::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(new_token) {
                break;
            }
            if let Ok(piece) = self
                .model
                .token_to_str(new_token, llama_cpp_2::model::Special::Tokenize)
            {
                output.push_str(&piece);
            }
            batch.clear();
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                batch
                    .add(new_token, n_cur as i32, &[0], true)
                    .map_err(|e| Error::Model(format!("batch: {e}")))?;
            }
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| Error::Model(format!("decode: {e}")))?;
        }

        Ok(output)
    }

    /// Expand a query into multiple search variations using LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying text generation fails.
    pub fn expand_query(&self, query: &str, include_lexical: bool) -> Result<Vec<Query>> {
        let prompt = format!(
            "/no_think Expand this search query into different forms for retrieval.\n\
             Output format (one per line):\n\
             lex: keyword terms for BM25 search\n\
             vec: semantic query for vector search\n\
             hyde: hypothetical document that would answer the query\n\n\
             Query: {query}\n"
        );

        let text = self.generate(&prompt, 300)?;
        let mut queries = Query::from_llm_output(&text, query);

        if !include_lexical {
            queries.retain(|q| q.kind != QueryType::Lex);
        }

        Ok(queries)
    }
}
