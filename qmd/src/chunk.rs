//! Document chunking for embedding pipelines.
//!
//! Provides [`Chunker`] for splitting documents into overlapping chunks
//! suitable for vector embedding, using token-aware or character-based
//! strategies.

use crate::error::Result;

/// Default chunk size in tokens.
pub const DEFAULT_CHUNK_TOKENS: usize = 800;

/// Default overlap in tokens (~15% of chunk size).
pub const DEFAULT_OVERLAP_TOKENS: usize = DEFAULT_CHUNK_TOKENS * 15 / 100;

/// A text chunk produced by [`Chunker`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Chunk {
    /// Chunk text.
    pub text: String,
    /// Character offset in the source document.
    pub pos: usize,
    /// Token count (`Some` when produced by token-aware chunking).
    pub token_count: Option<usize>,
}

/// Trait for token counting, decoupling chunking from a specific model.
pub trait Tokenizer {
    /// Count tokens in `text`.
    fn count_tokens(&self, text: &str) -> Result<usize>;
}

/// Configurable document chunker.
///
/// # Examples
///
/// ```rust,no_run
/// use qmd::chunk::Chunker;
///
/// // Character-based (no tokenizer needed)
/// let chunks = Chunker::default().split_chars("some long document...");
///
/// // Token-aware (requires a Tokenizer implementation)
/// // let chunks = Chunker::default().split(&engine, "some long document...")?;
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Chunker {
    /// Max tokens per chunk.
    pub max_tokens: usize,
    /// Overlap in tokens.
    pub overlap_tokens: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_CHUNK_TOKENS,
            overlap_tokens: DEFAULT_OVERLAP_TOKENS,
        }
    }
}

impl Chunker {
    /// Create a chunker with custom token limits.
    #[must_use]
    pub const fn new(max_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens,
        }
    }

    /// Split a document into token-aware chunks.
    ///
    /// Keeps paragraphs intact when possible.
    pub fn split(&self, tok: &impl Tokenizer, content: &str) -> Result<Vec<Chunk>> {
        let total = tok.count_tokens(content)?;
        if total <= self.max_tokens {
            return Ok(vec![Chunk {
                text: content.to_string(),
                pos: 0,
                token_count: Some(total),
            }]);
        }

        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut cur_text = String::new();
        let mut cur_tokens = 0usize;
        let mut chunk_start = 0usize;
        let mut char_pos = 0usize;

        for (i, para) in paragraphs.iter().enumerate() {
            let para_tokens = tok.count_tokens(para)?;
            let sep_tokens = if i > 0 { 2 } else { 0 };
            let para_with_sep = if i > 0 {
                format!("\n\n{para}")
            } else {
                (*para).to_string()
            };

            if cur_tokens + para_tokens + sep_tokens > self.max_tokens && !cur_text.is_empty() {
                chunks.push(Chunk {
                    text: cur_text.clone(),
                    pos: chunk_start,
                    token_count: Some(cur_tokens),
                });

                let overlap = tail_overlap(&cur_text, self.overlap_tokens, tok)?;
                cur_text = overlap;
                cur_tokens = tok.count_tokens(&cur_text)?;
                chunk_start = char_pos.saturating_sub(cur_text.len());
            }

            if !cur_text.is_empty() {
                cur_text.push_str("\n\n");
            }
            cur_text.push_str(para);
            cur_tokens += para_tokens + sep_tokens;
            char_pos += para_with_sep.len();
        }

        if !cur_text.is_empty() {
            chunks.push(Chunk {
                text: cur_text.clone(),
                pos: chunk_start,
                token_count: Some(cur_tokens),
            });
        }

        Ok(chunks)
    }

    /// Split a document using character-based heuristics (no tokenizer needed).
    #[must_use]
    pub fn split_chars(&self, content: &str) -> Vec<Chunk> {
        let max_chars = self.max_tokens * 4;
        let overlap_chars = self.overlap_tokens * 4;

        if content.len() <= max_chars {
            return vec![Chunk {
                text: content.to_string(),
                pos: 0,
                token_count: None,
            }];
        }

        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            let end = (pos + max_chars).min(content.len());
            let mut actual_end = if end < content.len() {
                find_break_point(content, pos, end)
            } else {
                end
            };
            if actual_end <= pos {
                actual_end = (pos + max_chars).min(content.len());
            }

            chunks.push(Chunk {
                text: content[pos..actual_end].to_string(),
                pos,
                token_count: None,
            });

            if actual_end >= content.len() {
                break;
            }

            pos = actual_end.saturating_sub(overlap_chars);
            if chunks.last().is_some_and(|last| pos <= last.pos) {
                pos = actual_end;
            }
        }

        chunks
    }
}

/// Extract overlap text from the tail of a chunk.
fn tail_overlap(text: &str, target_tokens: usize, tok: &impl Tokenizer) -> Result<String> {
    let start = text.len() * 4 / 5;
    let candidate = &text[start..];

    if let Some(pos) = candidate.find("\n\n") {
        let overlap = &candidate[pos + 2..];
        if tok.count_tokens(overlap)? <= target_tokens * 2 {
            return Ok(overlap.to_string());
        }
    }

    let words: Vec<&str> = candidate.split_whitespace().collect();
    let mut result = String::new();
    for word in words.iter().rev().take(target_tokens / 2) {
        if result.is_empty() {
            result = (*word).to_string();
        } else {
            result = format!("{word} {result}");
        }
    }

    Ok(result)
}

/// Find a good break point (paragraph > sentence > line > word).
fn find_break_point(content: &str, start: usize, end: usize) -> usize {
    let slice = &content[start..end];
    let search_start = slice.len() * 7 / 10;
    let tail = &slice[search_start..];

    if let Some(pos) = tail.rfind("\n\n") {
        return start + search_start + pos + 2;
    }
    for pat in &[". ", ".\n", "? ", "?\n", "! ", "!\n"] {
        if let Some(pos) = tail.rfind(pat) {
            return start + search_start + pos + 2;
        }
    }
    if let Some(pos) = tail.rfind('\n') {
        return start + search_start + pos + 1;
    }
    if let Some(pos) = tail.rfind(' ') {
        return start + search_start + pos + 1;
    }

    end
}
