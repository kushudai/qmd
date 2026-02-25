//! Smart document chunking for embedding pipelines.
//!
//! Splits documents into overlapping chunks using scored break points,
//! code fence protection, and distance decay — ensuring chunks never
//! split inside fenced code blocks and preferring structural boundaries.
//!
//! # Algorithm
//!
//! 1. Pre-scan the document for all break points (headings, blank lines,
//!    list items, sentences, etc.) and code fence ranges.
//! 2. For each chunk boundary, find the best break point within a search
//!    window using a score × proximity formula.
//! 3. Reject any break point that falls inside a fenced code block.
//!
//! # Examples
//!
//! ```rust,no_run
//! use qmd::chunk::Chunker;
//!
//! let chunks = Chunker::default().split_chars("some long document...");
//! // or with a tokenizer:
//! // let chunks = Chunker::default().split(&engine, "some long document...")?;
//! ```

use crate::error::Result;

/// Default chunk size in tokens.
pub const DEFAULT_CHUNK_TOKENS: usize = 800;

/// Default overlap in tokens (~15 % of chunk size).
pub const DEFAULT_OVERLAP_TOKENS: usize = DEFAULT_CHUNK_TOKENS * 15 / 100;

/// Chars-per-token estimate (prose ≈ 4, code ≈ 2, mixed ≈ 3).
const CHARS_PER_TOKEN: usize = 4;

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
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to process the input.
    fn count_tokens(&self, text: &str) -> Result<usize>;
}

/// Configurable document chunker.
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
    /// Uses the same scored-break-point algorithm as [`split_chars`](Self::split_chars),
    /// but verifies each chunk against the tokenizer and re-splits oversized
    /// chunks with a tighter character budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to count tokens.
    pub fn split(&self, tok: &impl Tokenizer, content: &str) -> Result<Vec<Chunk>> {
        let total = tok.count_tokens(content)?;
        if total <= self.max_tokens {
            return Ok(vec![Chunk {
                text: content.to_string(),
                pos: 0,
                token_count: Some(total),
            }]);
        }

        // First pass: character-based smart chunking.
        let char_chunks = self.split_chars(content);

        // Second pass: verify token counts, re-split oversized chunks.
        let mut result = Vec::with_capacity(char_chunks.len());
        for chunk in char_chunks {
            let tokens = tok.count_tokens(&chunk.text)?;
            if tokens <= self.max_tokens {
                result.push(Chunk {
                    token_count: Some(tokens),
                    ..chunk
                });
            } else {
                // Re-split with tighter budget based on actual chars/token ratio.
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss
                )]
                let safe_chars = {
                    let ratio = chunk.text.len() as f64 / tokens as f64;
                    (self.max_tokens as f64 * ratio * 0.95) as usize
                };
                let sub = Self::new(safe_chars / CHARS_PER_TOKEN, self.overlap_tokens / 2);
                for sc in sub.split_chars(&chunk.text) {
                    let t = tok.count_tokens(&sc.text)?;
                    result.push(Chunk {
                        text: sc.text,
                        pos: chunk.pos + sc.pos,
                        token_count: Some(t),
                    });
                }
            }
        }
        Ok(result)
    }

    /// Split a document using smart character-based heuristics.
    ///
    /// Uses scored break points with code fence protection.
    #[must_use]
    pub fn split_chars(&self, content: &str) -> Vec<Chunk> {
        let max_chars = self.max_tokens * CHARS_PER_TOKEN;
        let overlap_chars = self.overlap_tokens * CHARS_PER_TOKEN;
        let window_chars = self.max_tokens; // ~25 % of max_chars

        if content.len() <= max_chars {
            return vec![Chunk {
                text: content.to_string(),
                pos: 0,
                token_count: None,
            }];
        }

        // Pre-scan once for the whole document.
        let breaks = scan_break_points(content);
        let fences = scan_code_fences(content);

        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            let target_end = (pos + max_chars).min(content.len());
            let mut end = target_end;

            // Find best scored break point (only if not at document end).
            if end < content.len()
                && let Some(bp) = find_best_cutoff(&breaks, target_end, window_chars, &fences)
                && bp > pos
                && bp <= target_end
            {
                end = bp;
            }

            // Guarantee forward progress.
            if end <= pos {
                end = (pos + max_chars).min(content.len());
            }

            chunks.push(Chunk {
                text: content[pos..end].to_string(),
                pos,
                token_count: None,
            });

            if end >= content.len() {
                break;
            }

            // Advance with overlap.
            let next = end.saturating_sub(overlap_chars);
            pos = if chunks.last().is_some_and(|c| next <= c.pos) {
                end // prevent infinite loop
            } else {
                next
            };
        }

        chunks
    }
}

/// A scored position where the document *may* be cut.
#[derive(Debug, Clone, Copy)]
struct BreakPoint {
    /// Byte offset in the document (cut happens *before* this position).
    pos: usize,
    /// Base score — higher means better cut point.
    score: u32,
}

/// Ranked break-point patterns (descending importance).
const PATTERNS: &[(&str, u32)] = &[
    ("\n# ", 100),   // H1
    ("\n## ", 90),   // H2
    ("\n### ", 85),  // H3
    ("\n#### ", 80), // H4
    ("\n```", 75),   // Code fence boundary
    ("\n---", 70),   // Horizontal rule / front-matter
    ("\n\n", 60),    // Blank line (paragraph break)
    ("\n- ", 45),    // Unordered list item
    ("\n* ", 45),    // Unordered list item (alt)
    ("\n1. ", 45),   // Ordered list item
    (". ", 30),      // Sentence end
    (".\n", 30),     // Sentence end at EOL
    ("? ", 25),      // Question
    ("! ", 25),      // Exclamation
    ("\n", 10),      // Any line break
];

/// Scan the entire document for break points, returning them sorted by position.
fn scan_break_points(content: &str) -> Vec<BreakPoint> {
    let mut points = Vec::new();
    for &(pat, score) in PATTERNS {
        let mut start = 0;
        while let Some(idx) = content[start..].find(pat) {
            let abs = start + idx;
            // Cut position is right after the pattern for line-oriented patterns,
            // or at the pattern start for inline sentence breaks.
            let cut = if pat.starts_with('\n') {
                abs + 1 // cut at the start of the new structural element
            } else {
                abs + pat.len() // cut after ". " etc.
            };
            if cut > 0 && cut < content.len() {
                points.push(BreakPoint { pos: cut, score });
            }
            start = abs + pat.len().max(1);
        }
    }
    points.sort_by_key(|bp| bp.pos);
    points.dedup_by_key(|bp| bp.pos);
    points
}

/// A byte range [start, end) inside a fenced code block.
#[derive(Debug, Clone, Copy)]
struct FenceRange {
    /// Start of the opening fence line.
    start: usize,
    /// End of the closing fence line (exclusive).
    end: usize,
}

/// Scan for fenced code blocks (triple-backtick or triple-tilde).
fn scan_code_fences(content: &str) -> Vec<FenceRange> {
    let mut ranges = Vec::new();
    let mut open: Option<usize> = None;

    for (idx, line) in content.split('\n').scan(0usize, |pos, line| {
        let start = *pos;
        *pos += line.len() + 1; // +1 for the '\n'
        Some((start, line))
    }) {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if let Some(start) = open {
                ranges.push(FenceRange {
                    start,
                    end: idx + line.len(),
                });
                open = None;
            } else {
                open = Some(idx);
            }
        }
    }
    ranges
}

/// Check if a position falls inside any fenced code block.
fn inside_fence(pos: usize, fences: &[FenceRange]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Find the best break point near `target` within `±window` chars.
///
/// Uses `score × proximity` ranking and rejects points inside code fences.
fn find_best_cutoff(
    breaks: &[BreakPoint],
    target: usize,
    window: usize,
    fences: &[FenceRange],
) -> Option<usize> {
    let lo = target.saturating_sub(window);
    let hi = target + window / 4; // asymmetric: prefer cutting early

    breaks
        .iter()
        .filter(|bp| bp.pos >= lo && bp.pos <= hi)
        .filter(|bp| !inside_fence(bp.pos, fences))
        .max_by(|a, b| {
            let sa = weighted_score(a, target, window);
            let sb = weighted_score(b, target, window);
            sa.total_cmp(&sb)
        })
        .map(|bp| bp.pos)
}

/// Score a break point factoring in distance decay from `target`.
///
/// `weighted = base_score × (1 - (distance / window)²)`
#[allow(clippy::cast_precision_loss)]
fn weighted_score(bp: &BreakPoint, target: usize, window: usize) -> f64 {
    let dist = bp.pos.abs_diff(target) as f64;
    let w = window as f64;
    let ratio = dist / w;
    let proximity = ratio.mul_add(-ratio, 1.0).max(0.0);
    f64::from(bp.score) * proximity
}
