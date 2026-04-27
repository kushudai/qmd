//! Search utilities: FTS5 query building, RRF fusion, snippet extraction.

use std::collections::HashMap;

use regex::Regex;

/// Search backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum QueryType {
    /// Lexical (BM25).
    Lex,
    /// Vector (semantic).
    Vec,
    /// HyDE — Hypothetical Document Embedding.
    Hyde,
}

/// A typed search query destined for a specific backend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Query {
    /// Backend kind.
    pub kind: QueryType,
    /// Query text.
    pub text: String,
}

impl Query {
    /// Create a lexical (BM25) query.
    #[must_use]
    pub fn lex(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Lex,
            text: text.into(),
        }
    }

    /// Create a vector (semantic) query.
    #[must_use]
    pub fn vec(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Vec,
            text: text.into(),
        }
    }

    /// Create a HyDE query.
    #[must_use]
    pub fn hyde(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Hyde,
            text: text.into(),
        }
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
    /// Falls back to [`expand_simple`](Self::expand_simple) if no valid lines found.
    #[must_use]
    pub fn from_llm_output(output: &str, original: &str) -> Vec<Self> {
        let query_lower = original.to_lowercase();
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
                    queries.push(Self {
                        kind,
                        text: text.to_string(),
                    });
                }
            }
        }

        if queries.is_empty() {
            Self::expand_simple(original)
        } else {
            queries
        }
    }
}

impl<'de> serde::Deserialize<'de> for QueryType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "lex" => Ok(Self::Lex),
            "vec" => Ok(Self::Vec),
            "hyde" => Ok(Self::Hyde),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &["lex", "vec", "hyde"],
            )),
        }
    }
}

/// Sanitize a term into FTS5-safe sub-tokens.
///
/// The FTS5 virtual table uses `tokenize='porter unicode61'`, which splits the
/// indexed text on every non-alphanumeric character — including `_` and `-`.
/// So an identifier like `should_notify` is indexed as the two tokens
/// `should` and `notify`. Stripping the underscore and concatenating the
/// halves into `shouldnotify` (the previous behaviour) produced a token
/// that never appears in the index, returning zero matches for any
/// snake_case or kebab-case query.
///
/// Instead, split on disallowed characters and return each non-empty
/// alphanumeric run as its own sub-token. Callers decide how to combine
/// them (as a phrase or as ANDed prefix terms).
fn sanitize_fts5_term(term: &str) -> Vec<String> {
    term.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Build an FTS5 query from user-facing search syntax.
///
/// Supports quoted phrases, negation (`-term`), and prefix matching. Bare
/// terms become prefix searches (`tokio` → `"tokio"*`). Punctuated
/// identifiers like `should_notify` and `tokio-rt` are split on the
/// punctuation and emitted as a phrase (`"should notify"`) so they match
/// the sub-tokens that FTS5's `unicode61` tokenizer actually stored.
/// Returns `None` if no usable terms.
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
///     build_fts5_query("should_notify"),
///     Some(r#""should notify""#.to_string()),
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
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let negated = bytes[i] == b'-';
        if negated {
            i += 1;
            if i >= bytes.len() {
                break;
            }
        }

        if bytes[i] == b'"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let phrase = &s[start..i];
            if i < bytes.len() {
                i += 1;
            }
            // Quoted phrase: flatten every word into its sub-tokens and emit
            // them as a single phrase. e.g. `"some_func returns"` becomes
            // `"some func returns"` so FTS5 matches the indexed token sequence.
            let tokens: Vec<String> = phrase
                .split_whitespace()
                .flat_map(sanitize_fts5_term)
                .collect();
            if !tokens.is_empty() {
                let fts = format!("\"{}\"", tokens.join(" "));
                if negated {
                    &mut negative
                } else {
                    &mut positive
                }
                .push(fts);
            }
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'"' {
                i += 1;
            }
            // Bare term: a single alphanumeric run gets prefix-matched
            // (`tok` -> `"tok"*` matches `tokio`); a punctuated identifier
            // like `should_notify` becomes the phrase `"should notify"` so
            // it matches the underlying tokens FTS5 actually stored, without
            // a prefix star (the user typed two complete words).
            let tokens = sanitize_fts5_term(&s[start..i]);
            let fts = match tokens.as_slice() {
                [] => None,
                [tok] => Some(format!("\"{tok}\"*")),
                toks => Some(format!("\"{}\"", toks.join(" "))),
            };
            if let Some(clause) = fts {
                if negated {
                    &mut negative
                } else {
                    &mut positive
                }
                .push(clause);
            }
        }
    }

    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" ");
    for neg in &negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

/// Monotonic mapping of raw BM25 to `[0, 1)`: `x / (1 + x)`.
#[must_use]
pub fn normalize_bm25(score: f64) -> f64 {
    let s = score.abs();
    s / (1.0 + s)
}

/// A fused RRF result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RrfHit {
    /// Document key.
    pub key: String,
    /// Merged RRF score.
    pub score: f64,
}

/// Reciprocal Rank Fusion over multiple ranked key lists.
///
/// `RRF(d) = Σ weight_i / (k + rank_i + 1)`.
#[must_use]
pub fn rrf(lists: &[&[String]], weights: Option<&[f64]>, k: usize) -> Vec<RrfHit> {
    let mut scores: HashMap<&str, f64> = HashMap::new();

    for (list_idx, keys) in lists.iter().enumerate() {
        let w = weights
            .and_then(|ws| ws.get(list_idx))
            .copied()
            .unwrap_or(1.0);
        for (rank, key) in keys.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let s = w / (k + rank + 1) as f64;
            *scores.entry(key.as_str()).or_default() += s;
        }
    }

    let mut hits: Vec<RrfHit> = scores
        .into_iter()
        .map(|(key, score)| RrfHit {
            key: key.to_string(),
            score,
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits
}

/// Extract a relevant snippet from `body` around query terms.
///
/// All offsets are snapped to UTF-8 char boundaries before slicing, so a
/// multi-byte character (`→`, `×`, em-dash, …) at a snippet boundary cannot
/// panic the search pipeline.
#[must_use]
pub fn extract_snippet(body: &str, query: &str, max_chars: usize) -> String {
    if body.len() <= max_chars {
        return body.to_string();
    }

    let body_lower = body.to_lowercase();
    let raw_start = query
        .split_whitespace()
        .filter(|t| t.len() >= 3)
        .find_map(|t| body_lower.find(&t.to_lowercase()))
        .map_or(0, |p| p.saturating_sub(50));
    let start_pos = crate::chunk::floor_char_boundary(body, raw_start);

    let line_start = body[..start_pos].rfind('\n').map_or(0, |p| p + 1);
    let raw_end = (line_start + max_chars).min(body.len());
    let end_pos = crate::chunk::floor_char_boundary(body, raw_end);
    let line_end = body[end_pos..]
        .find('\n')
        .map_or(body.len(), |p| end_pos + p);

    body[line_start..line_end].to_string()
}

#[cfg(test)]
mod fts_query_tests {
    use super::build_fts5_query;

    #[test]
    fn bare_word_becomes_prefix_match() {
        assert_eq!(build_fts5_query("tokio"), Some(r#""tokio"*"#.to_string()));
    }

    #[test]
    fn snake_case_becomes_phrase() {
        // The previous behaviour stripped the underscore and produced
        // "shouldnotify"*, which matches nothing in the index because the
        // unicode61 tokenizer split the indexed text on the underscore.
        assert_eq!(
            build_fts5_query("should_notify"),
            Some(r#""should notify""#.to_string()),
        );
    }

    #[test]
    fn kebab_case_becomes_phrase() {
        assert_eq!(
            build_fts5_query("tokio-rt"),
            Some(r#""tokio rt""#.to_string())
        );
    }

    #[test]
    fn dotted_identifier_becomes_phrase() {
        assert_eq!(
            build_fts5_query("std::sync::Mutex"),
            Some(r#""std sync mutex""#.to_string()),
        );
    }

    #[test]
    fn negation_with_punctuation() {
        assert_eq!(
            build_fts5_query("perf -should_notify"),
            Some(r#""perf"* NOT "should notify""#.to_string()),
        );
    }

    #[test]
    fn quoted_phrase_with_punctuation_flattens() {
        // Quoted phrase: every word's sub-tokens are flattened into the
        // phrase so the FTS5 phrase matches the indexed token sequence.
        assert_eq!(
            build_fts5_query(r#""should_notify regression""#),
            Some(r#""should notify regression""#.to_string()),
        );
    }

    #[test]
    fn pure_punctuation_returns_none() {
        assert_eq!(build_fts5_query("---"), None);
        assert_eq!(build_fts5_query(""), None);
        assert_eq!(build_fts5_query("   "), None);
    }

    #[test]
    fn apostrophes_are_preserved() {
        assert_eq!(
            build_fts5_query("don't worry"),
            Some(r#""don't"* "worry"*"#.to_string()),
        );
    }
}

#[cfg(test)]
mod snippet_tests {
    use super::extract_snippet;

    #[test]
    fn short_body_returned_verbatim() {
        let body = "tiny — body";
        let s = extract_snippet(body, "body", 4096);
        assert_eq!(s, body);
    }

    #[test]
    fn multibyte_char_at_end_boundary_does_not_panic() {
        // Reproduces the `→` (U+2192, 3 bytes) panic at `search.rs:294`:
        // a multi-byte char straddling `line_start + max_chars`.
        //
        // Layout: 3998 ASCII bytes, then `→` (3 bytes) at byte 3998..4001.
        // With max_chars = 4000 the naive end_pos lands inside `→`.
        let mut body = String::with_capacity(5000);
        body.push_str(&"a".repeat(3998));
        body.push('→');
        body.push_str(&"b".repeat(1000));

        let s = extract_snippet(&body, "search", 4000);
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    }

    #[test]
    fn multibyte_char_near_query_match_does_not_panic() {
        // The 50-byte left-pad before the query match can also land inside
        // a multi-byte char.
        let body = format!("{}→needle and the rest", "x".repeat(60));
        let s = extract_snippet(&body, "needle", 50);
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
        assert!(s.contains("needle"));
    }
}
