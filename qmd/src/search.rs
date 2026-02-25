//! Search utilities: query expansion, FTS5 query parsing, RRF fusion,
//! snippet extraction, and structured search pipeline.

use std::collections::HashMap;

use regex::Regex;

/// Search backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryType {
    /// Lexical (BM25).
    Lex,
    /// Vector (semantic).
    Vec,
    /// `HyDE` — Hypothetical Document Embedding.
    Hyde,
}

/// A typed search query destined for a specific backend.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Query {
    /// Backend kind.
    pub kind: QueryType,
    /// Query text.
    pub text: String,
}

impl Query {
    /// Create a query with the given kind and text.
    #[must_use]
    pub fn new(kind: QueryType, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
        }
    }

    /// Create a lexical (BM25) query.
    #[must_use]
    pub fn lex(text: impl Into<String>) -> Self {
        Self::new(QueryType::Lex, text)
    }

    /// Create a vector (semantic) query.
    #[must_use]
    pub fn vec(text: impl Into<String>) -> Self {
        Self::new(QueryType::Vec, text)
    }

    /// Create a `HyDE` query.
    #[must_use]
    pub fn hyde(text: impl Into<String>) -> Self {
        Self::new(QueryType::Hyde, text)
    }

    /// Simple (non-LLM) expansion into lex + vec + hyde.
    #[must_use]
    pub fn expand_simple(query: &str) -> Vec<Self> {
        vec![
            Self::lex(query),
            Self::vec(query),
            Self::hyde(format!("Information about {query}")),
        ]
    }

    /// Parse structured LLM output into typed queries.
    ///
    /// Expected format (one per line): `lex:`, `vec:`, `hyde:`.
    /// Falls back to [`Self::expand_simple`] if no valid lines found.
    #[must_use]
    pub fn from_llm_output(output: &str, original_query: &str) -> Vec<Self> {
        let query_lower = original_query.to_lowercase();
        let line_re = Regex::new(r"^(lex|vec|hyde):\s*(.+)$").ok();
        let mut queries = Vec::new();

        for raw in output.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(ref re) = line_re
                && let Some(caps) = re.captures(line)
            {
                let kind = match &caps[1] {
                    "lex" => QueryType::Lex,
                    "vec" => QueryType::Vec,
                    "hyde" => QueryType::Hyde,
                    _ => continue,
                };
                let text = caps[2].trim();
                let text_lower = text.to_lowercase();

                let has_overlap = query_lower
                    .split_whitespace()
                    .any(|t| t.len() >= 3 && text_lower.contains(t));

                if has_overlap || query_lower.len() < 3 {
                    queries.push(Self::new(kind, text));
                }
            }
        }

        if queries.is_empty() {
            return Self::expand_simple(original_query);
        }
        queries
    }
}

/// Sanitize a single term for FTS5 (keep only letters, digits, apostrophes).
fn sanitize_fts5_term(term: &str) -> String {
    term.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .collect::<String>()
        .to_lowercase()
}

/// Build an FTS5 query from user-facing search syntax.
///
/// Supports:
/// - **Quoted phrases**: `"exact phrase"` → exact match
/// - **Negation**: `-term` or `-"phrase"` → FTS5 NOT operator
/// - **Plain terms**: `term` → `"term"*` (prefix match)
///
/// Returns `None` if the query contains no usable terms.
///
/// # Examples
///
/// ```
/// use qmd::search::build_fts5_query;
///
/// assert_eq!(
///     build_fts5_query("performance -sports"),
///     Some(r#""performance"* NOT "sports"*"#.to_string()),
/// );
/// assert_eq!(
///     build_fts5_query(r#""machine learning""#),
///     Some(r#""machine learning""#.to_string()),
/// );
/// ```
#[must_use]
pub fn build_fts5_query(query: &str) -> Option<String> {
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();

    let s = query.trim();
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        // Check negation prefix.
        let negated = bytes[i] == b'-';
        if negated {
            i += 1;
            if i >= bytes.len() {
                break;
            }
        }

        // Quoted phrase.
        if bytes[i] == b'"' {
            i += 1; // skip opening quote
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let phrase = &s[start..i];
            if i < bytes.len() {
                i += 1; // skip closing quote
            }
            let sanitized: String = phrase
                .split_whitespace()
                .map(sanitize_fts5_term)
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !sanitized.is_empty() {
                let fts = format!("\"{sanitized}\"");
                if negated {
                    negative.push(fts);
                } else {
                    positive.push(fts);
                }
            }
        } else {
            // Plain term (until whitespace or quote).
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'"' {
                i += 1;
            }
            let term = &s[start..i];
            let sanitized = sanitize_fts5_term(term);
            if !sanitized.is_empty() {
                let fts = format!("\"{sanitized}\"*");
                if negated {
                    negative.push(fts);
                } else {
                    positive.push(fts);
                }
            }
        }
    }

    if positive.is_empty() && negative.is_empty() {
        return None;
    }

    // If only negative terms, nothing to match against.
    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" ");
    for neg in &negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

/// Minimum BM25 score to consider a "strong signal" — high enough that
/// LLM query expansion can be skipped for latency savings.
pub const STRONG_SIGNAL_THRESHOLD: f64 = 12.0;

/// Minimum number of strong-signal results before skipping expansion.
pub const STRONG_SIGNAL_MIN_RESULTS: usize = 3;

/// Check whether FTS results are strong enough to skip LLM expansion.
#[must_use]
pub fn has_strong_signal(fts_scores: &[f64]) -> bool {
    fts_scores
        .iter()
        .filter(|&&s| s >= STRONG_SIGNAL_THRESHOLD)
        .count()
        >= STRONG_SIGNAL_MIN_RESULTS
}

/// A fused RRF result carrying a key and merged score.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RrfHit {
    /// Document key (typically `collection/path`).
    pub key: String,
    /// Merged RRF score.
    pub score: f64,
    /// Best rank across all input lists (0-indexed).
    pub best_rank: usize,
}

/// Reciprocal Rank Fusion over multiple ranked key lists.
///
/// `RRF(d) = Σ weight_i / (k + rank_i + 1)` with position bonuses.
#[must_use]
pub fn rrf(lists: &[&[String]], weights: Option<&[f64]>, k: usize) -> Vec<RrfHit> {
    let mut scores: HashMap<&str, (f64, usize)> = HashMap::new();

    for (list_idx, keys) in lists.iter().enumerate() {
        let w = weights
            .and_then(|ws| ws.get(list_idx))
            .copied()
            .unwrap_or(1.0);

        for (rank, key) in keys.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let s = w / (k + rank + 1) as f64;
            scores
                .entry(key.as_str())
                .and_modify(|(acc, best)| {
                    *acc += s;
                    *best = (*best).min(rank);
                })
                .or_insert((s, rank));
        }
    }

    let mut hits: Vec<RrfHit> = scores
        .into_iter()
        .map(|(key, (score, best_rank))| {
            let bonus = match best_rank {
                0..=2 => 0.08,
                3..=9 => 0.04,
                10..=19 => 0.01,
                _ => 0.0,
            };
            RrfHit {
                key: key.to_string(),
                score: score + bonus,
                best_rank,
            }
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// Fuse two ranked key lists with equal weights.
#[must_use]
pub fn hybrid_rrf(fts_keys: &[String], vec_keys: &[String], k: usize) -> Vec<RrfHit> {
    rrf(&[fts_keys, vec_keys], Some(&[1.0, 1.0]), k)
}

/// Normalize a slice of scores to `[0, 1]` using min-max scaling.
#[must_use]
pub fn normalize_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < f64::EPSILON {
        return vec![1.0; scores.len()];
    }
    scores.iter().map(|s| (s - min) / range).collect()
}

/// An extracted snippet with its starting line number.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Snippet {
    /// Snippet text.
    pub text: String,
    /// 1-indexed line number.
    pub line: usize,
}

/// Extract a relevant snippet from `body` around query terms or `chunk_pos`.
#[must_use]
pub fn extract_snippet(
    body: &str,
    query: &str,
    max_chars: usize,
    chunk_pos: Option<usize>,
) -> Snippet {
    if body.len() <= max_chars {
        return Snippet {
            text: body.to_string(),
            line: 1,
        };
    }

    let body_lower = body.to_lowercase();
    let start_pos = chunk_pos.map_or_else(
        || {
            query
                .split_whitespace()
                .filter(|t| t.len() >= 3)
                .find_map(|t| body_lower.find(&t.to_lowercase()))
                .map_or(0, |p| p.saturating_sub(50))
        },
        |pos| pos.min(body.len().saturating_sub(max_chars)),
    );

    let line_start = body[..start_pos].rfind('\n').map_or(0, |p| p + 1);
    let end_pos = (line_start + max_chars).min(body.len());
    let line_end = body[end_pos..]
        .find('\n')
        .map_or(body.len(), |p| end_pos + p);
    let line = body[..line_start].matches('\n').count() + 1;

    Snippet {
        text: body[line_start..line_end].to_string(),
        line,
    }
}

/// Add line numbers to text content (1-indexed).
#[must_use]
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {line}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}
