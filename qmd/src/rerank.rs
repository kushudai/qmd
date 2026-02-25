//! True cross-encoder reranking engine.
//!
//! Uses the Qwen3-reranker chat template to jointly encode query + document,
//! then extracts logits for "yes" / "no" tokens to compute a continuous
//! relevance probability.  This is significantly more accurate than
//! dual-encoder (separate embedding + cosine) approaches because the model
//! sees both query and document simultaneously via full cross-attention.

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::error::{Error, Result};
use crate::model;

/// Maximum document tokens to feed the reranker before truncation.
const MAX_DOC_TOKENS: usize = 1024;

/// Overhead tokens reserved for the chat template and query.
const TEMPLATE_OVERHEAD: usize = 200;

/// Scored rerank result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Scored {
    /// Identifier (file path or key).
    pub key: String,
    /// Relevance probability in `[0, 1]` (higher is better).
    pub score: f32,
    /// Original index in the input list.
    pub index: usize,
}

/// Cross-encoder reranking engine backed by a GGUF model.
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

    /// Load the default rerank model from cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the model path cannot be resolved or the model fails to load.
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
    /// Each candidate is `(key, text, optional_title)`.  Returns results
    /// sorted by descending relevance probability.
    ///
    /// Internally uses the Qwen3-reranker cross-encoder: the model sees
    /// query + document together and produces a "yes" / "no" judgment
    /// whose softmax probability becomes the score.
    ///
    /// # Errors
    ///
    /// Returns an error if token resolution, context creation, or inference fails.
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[(String, String, Option<String>)],
    ) -> Result<Vec<Scored>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve "yes" / "no" token IDs once.
        let yes_id = self.token_id("yes")?;
        let no_id = self.token_id("no")?;

        let mut results: Vec<Scored> = candidates
            .iter()
            .enumerate()
            .map(|(i, (key, text, title))| {
                let doc = title
                    .as_ref()
                    .map_or_else(|| text.clone(), |t| format!("{t}\n\n{text}"));
                let score = self.score_pair(query, &doc, yes_id, no_id).unwrap_or(0.0);
                Scored {
                    key: key.clone(),
                    score,
                    index: i,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(results)
    }

    /// Compute the relevance probability for a single (query, document) pair.
    ///
    /// Formats the pair with the Qwen3-reranker chat template, runs
    /// a forward pass, and returns `softmax(logit_yes, logit_no).yes`.
    fn score_pair(&self, query: &str, document: &str, yes_id: i32, no_id: i32) -> Result<f32> {
        let prompt = format_reranker_prompt(query, document);

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(Error::Model("empty token sequence".into()));
        }

        // Truncate document tokens to stay within budget.
        let max_tokens = MAX_DOC_TOKENS + TEMPLATE_OVERHEAD;
        let toks = if tokens.len() > max_tokens {
            &tokens[..max_tokens]
        } else {
            &tokens
        };

        #[allow(clippy::cast_possible_truncation)]
        let n_ctx = (toks.len() + 64).max(512);
        #[allow(clippy::cast_possible_truncation)]
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZero::new(n_ctx as u32))
            .with_n_batch(n_ctx as u32);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Model(format!("context: {e}")))?;

        let mut batch = LlamaBatch::new(toks.len(), 1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for (i, token) in toks.iter().enumerate() {
            // Request logits only for the last token.
            batch
                .add(*token, i as i32, &[0], i == toks.len() - 1)
                .map_err(|e| Error::Model(format!("batch: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Model(format!("decode: {e}")))?;

        // Extract logits for the last token position.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let logits = ctx.get_logits_ith((toks.len() - 1) as i32);

        #[allow(clippy::cast_sign_loss)]
        let yes_idx = yes_id as usize;
        #[allow(clippy::cast_sign_loss)]
        let no_idx = no_id as usize;
        let n_vocab = logits.len();
        let logit_yes = if yes_idx < n_vocab {
            logits[yes_idx]
        } else {
            0.0
        };
        let logit_no = if no_idx < n_vocab {
            logits[no_idx]
        } else {
            0.0
        };

        // softmax over [yes, no]
        Ok(softmax_pair(logit_yes, logit_no))
    }

    /// Resolve a word to its first token ID.
    fn token_id(&self, word: &str) -> Result<i32> {
        let tokens = self
            .model
            .str_to_token(word, AddBos::Never)
            .map_err(|e| Error::Model(format!("tokenize '{word}': {e}")))?;
        tokens
            .first()
            .map(|t| t.0)
            .ok_or_else(|| Error::Model(format!("no token for '{word}'")))
    }
}

/// Format a query + document pair using the Qwen3-reranker chat template.
fn format_reranker_prompt(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Judge whether the Document is relevant to the Query. Answer only \"yes\" or \"no\".\
         <|im_end|>\n\
         <|im_start|>user\n\
         <Query>{query}</Query>\n\
         <Document>{document}</Document>\n\
         Is the document relevant to the query?<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Softmax over two logits, returning P(a).
fn softmax_pair(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let ea = (a - max).exp();
    let eb = (b - max).exp();
    ea / (ea + eb)
}
