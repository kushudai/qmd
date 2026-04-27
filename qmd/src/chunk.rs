//! Smart document chunking for embedding pipelines.
//!
//! Splits documents into overlapping chunks using scored break points
//! and code fence protection.

/// Default chunk size in characters (~200 tokens).
pub const DEFAULT_CHUNK_CHARS: usize = 3200;

/// Default overlap in characters (~15%).
pub const DEFAULT_OVERLAP_CHARS: usize = DEFAULT_CHUNK_CHARS * 15 / 100;

/// A text chunk produced by [`Chunker`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Chunk {
    /// Chunk text.
    pub text: String,
    /// Byte offset in the source document.
    pub pos: usize,
}

/// Configurable document chunker.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Chunker {
    /// Max characters per chunk.
    pub max_chars: usize,
    /// Overlap in characters.
    pub overlap_chars: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_CHUNK_CHARS,
            overlap_chars: DEFAULT_OVERLAP_CHARS,
        }
    }
}

impl Chunker {
    /// Create a chunker with custom limits.
    #[must_use]
    pub const fn new(max_chars: usize, overlap_chars: usize) -> Self {
        Self {
            max_chars,
            overlap_chars,
        }
    }

    /// Split a document into chunks using scored break points.
    #[must_use]
    pub fn split(&self, content: &str) -> Vec<Chunk> {
        if content.len() <= self.max_chars {
            return vec![Chunk {
                text: content.to_string(),
                pos: 0,
            }];
        }

        let breaks = scan_break_points(content);
        let fences = scan_code_fences(content);
        let window = self.max_chars / 4;

        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            let target_end = (pos + self.max_chars).min(content.len());
            let mut end = target_end;

            if end < content.len()
                && let Some(bp) = find_best_cutoff(&breaks, target_end, window, &fences)
                && bp > pos
                && bp <= target_end
            {
                end = bp;
            }

            if end <= pos {
                end = (pos + self.max_chars).min(content.len());
            }

            // Snap `end` down to a UTF-8 char boundary so the slice below cannot
            // panic on multi-byte codepoints. Break-point cutoffs are ASCII and
            // already boundary-aligned, but the fallback `pos + max_chars` is not.
            end = floor_char_boundary(content, end);
            if end <= pos {
                end = ceil_char_boundary(content, pos + 1).min(content.len());
            }

            chunks.push(Chunk {
                text: content[pos..end].to_string(),
                pos,
            });

            if end >= content.len() {
                break;
            }

            let next = floor_char_boundary(content, end.saturating_sub(self.overlap_chars));
            pos = if chunks.last().is_some_and(|c| next <= c.pos) {
                end
            } else {
                next
            };
        }

        chunks
    }
}

/// Return the largest index `<= idx` that lies on a UTF-8 char boundary.
///
/// Equivalent to the unstable [`str::floor_char_boundary`]; reimplemented here
/// to stay on stable Rust.
pub(crate) fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Return the smallest index `>= idx` that lies on a UTF-8 char boundary.
pub(crate) fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// A scored break point position.
#[derive(Debug, Clone, Copy)]
struct BreakPoint {
    /// Byte offset in the document.
    pos: usize,
    /// Base score — higher means better cut point.
    score: u32,
}

/// Ranked break-point patterns.
const PATTERNS: &[(&str, u32)] = &[
    ("\n# ", 100),
    ("\n## ", 90),
    ("\n### ", 85),
    ("\n#### ", 80),
    ("\n```", 75),
    ("\n---", 70),
    ("\n\n", 60),
    ("\n- ", 45),
    ("\n* ", 45),
    ("\n1. ", 45),
    (". ", 30),
    (".\n", 30),
    ("\n", 10),
];

/// Scan the document for break points, sorted by position.
fn scan_break_points(content: &str) -> Vec<BreakPoint> {
    let mut points = Vec::new();
    for &(pat, score) in PATTERNS {
        let mut start = 0;
        while let Some(idx) = content[start..].find(pat) {
            let abs = start + idx;
            let cut = if pat.starts_with('\n') {
                abs + 1
            } else {
                abs + pat.len()
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

/// A byte range `[start, end)` inside a fenced code block.
#[derive(Debug, Clone, Copy)]
struct FenceRange {
    /// Start of the opening fence line.
    start: usize,
    /// End of the closing fence line (exclusive).
    end: usize,
}

/// Scan for fenced code blocks.
fn scan_code_fences(content: &str) -> Vec<FenceRange> {
    let mut ranges = Vec::new();
    let mut open: Option<usize> = None;

    for (idx, line) in content.split('\n').scan(0usize, |pos, line| {
        let start = *pos;
        *pos += line.len() + 1;
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

/// Check if a position falls inside a code fence.
fn inside_fence(pos: usize, fences: &[FenceRange]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Find the best break point near `target` within `±window`.
fn find_best_cutoff(
    breaks: &[BreakPoint],
    target: usize,
    window: usize,
    fences: &[FenceRange],
) -> Option<usize> {
    let lo = target.saturating_sub(window);
    let hi = target + window / 4;

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

/// Score a break point with distance decay: `score × (1 - (d/w)²)`.
#[allow(clippy::cast_precision_loss)]
fn weighted_score(bp: &BreakPoint, target: usize, window: usize) -> f64 {
    let dist = bp.pos.abs_diff(target) as f64;
    let w = window as f64;
    let ratio = dist / w;
    let proximity = ratio.mul_add(-ratio, 1.0).max(0.0);
    f64::from(bp.score) * proximity
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reassembling overlapping chunks must reproduce the original document.
    fn assert_chunks_cover(content: &str, chunks: &[Chunk]) {
        assert!(!chunks.is_empty(), "expected at least one chunk");
        assert_eq!(chunks[0].pos, 0, "first chunk must start at 0");

        // Each chunk's text must equal the slice at its `pos`.
        for c in chunks {
            assert!(
                content.is_char_boundary(c.pos),
                "chunk pos {} not on char boundary",
                c.pos
            );
            assert!(content[c.pos..].starts_with(c.text.as_str()));
        }

        // Walk forward across chunks, making sure every byte is covered.
        let mut covered = 0usize;
        for c in chunks {
            assert!(c.pos <= covered, "gap before chunk at pos {}", c.pos);
            covered = c.pos + c.text.len();
        }
        assert_eq!(covered, content.len(), "tail uncovered");
    }

    #[test]
    fn splits_pure_ascii() {
        let body = "abcdefghij".repeat(1000); // 10_000 bytes
        let chunks = Chunker::default().split(&body);
        assert!(chunks.len() > 1);
        assert_chunks_cover(&body, &chunks);
    }

    #[test]
    fn handles_multibyte_codepoint_at_chunk_boundary() {
        // Reproduces the panic at qmd-0.3.2/src/llm.rs:542 and the equivalent
        // 0.5.0 chunker bug: a multi-byte char (`×`, U+00D7, 2 bytes in UTF-8)
        // straddling `pos + max_chars`.
        let chunker = Chunker::new(16, 4);

        // 15 ASCII bytes, then `×` (2 bytes) starting at byte 15. With
        // max_chars = 16, the naive cutoff lands at byte 16 — inside the `×`.
        let body = "aaaaaaaaaaaaaaa×bbbbbbbbbbbbbbb";
        let chunks = chunker.split(body);
        assert_chunks_cover(body, &chunks);
        for c in &chunks {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn handles_multibyte_overlap() {
        // overlap_chars subtraction can also land mid-codepoint.
        let chunker = Chunker::new(20, 5);
        // Pack `×` chars densely around the overlap zone.
        let body = "abcdefghij×××××klmnopqrstuvwxyz×××××123456789";
        let chunks = chunker.split(body);
        assert_chunks_cover(body, &chunks);
    }

    #[test]
    fn small_content_returns_single_chunk() {
        let body = "tiny — with em dash";
        let chunks = Chunker::default().split(body);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, body);
    }

    #[test]
    fn original_panic_repro_string() {
        // Mirrors the exact substring from the production panic.
        let snippet = "at the cost of giving up `1.51`/`1.52` Tokio features for as long as the pin holds. Root-cause identification";
        // Repeat enough times to exceed default max_chars and force chunking.
        let body = snippet.repeat(40);
        let chunks = Chunker::default().split(&body);
        assert_chunks_cover(&body, &chunks);
    }
}
