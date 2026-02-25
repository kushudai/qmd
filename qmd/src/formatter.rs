//! Output formatting for search results and documents.
//!
//! Supports JSON, CSV, XML, Markdown, file-list, and plain CLI formats.

use std::fmt::Write;

use serde::Serialize;

use crate::document::{Document, SearchResult};
use crate::search;

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
    /// Machine-readable JSON.
    Json,
    /// Comma-separated values.
    Csv,
    /// XML document.
    Xml,
    /// Markdown table / sections.
    Md,
    /// One file path per line.
    Files,
    /// Human-readable CLI output.
    Cli,
}

impl Format {
    /// Parse a format string (case-insensitive).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "xml" => Some(Self::Xml),
            "md" | "markdown" => Some(Self::Md),
            "files" | "list" => Some(Self::Files),
            "cli" | "text" => Some(Self::Cli),
            _ => None,
        }
    }
}

/// Options controlling output details.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct FormatOptions {
    /// Show full document body instead of snippet.
    pub full: bool,
    /// Original query (for snippet extraction and highlighting).
    pub query: String,
    /// Add line numbers to body output.
    pub line_numbers: bool,
}

/// Serializable search result for JSON/CSV/XML output.
#[derive(Debug, Serialize)]
struct ResultRow {
    /// Short document identifier.
    docid: String,
    /// Virtual file path.
    file: String,
    /// Document title.
    title: String,
    /// Relevance score.
    score: f64,
    /// Optional context annotation.
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    /// Text snippet.
    snippet: String,
}

/// Format a list of search results.
#[must_use]
pub fn format_results(results: &[SearchResult], fmt: Format, opts: &FormatOptions) -> String {
    let rows: Vec<ResultRow> = results
        .iter()
        .map(|r| {
            let snippet = if opts.full {
                r.doc.body.as_deref().unwrap_or("").to_string()
            } else {
                r.doc
                    .body
                    .as_deref()
                    .map(|b| search::extract_snippet(b, &opts.query, 300, r.chunk_pos).text)
                    .unwrap_or_default()
            };
            ResultRow {
                docid: format!("#{}", r.doc.docid()),
                file: r.doc.display_path(),
                title: r.doc.title.clone(),
                score: r.score,
                context: r.doc.context.clone(),
                snippet,
            }
        })
        .collect();

    match fmt {
        Format::Json => serde_json::to_string_pretty(&rows).unwrap_or_default(),
        Format::Csv => format_csv(&rows),
        Format::Xml => format_xml_results(&rows),
        Format::Md => format_md_results(&rows),
        Format::Files => rows
            .iter()
            .map(|r| r.file.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Format::Cli => format_cli_results(&rows),
    }
}

/// Format a single document for output.
#[must_use]
pub fn format_document(doc: &Document, fmt: Format, opts: &FormatOptions) -> String {
    let body = doc.body.as_deref().unwrap_or("");
    let body_out = if opts.line_numbers {
        search::add_line_numbers(body, 1)
    } else {
        body.to_string()
    };

    match fmt {
        Format::Json => {
            #[derive(Serialize)]
            struct DocOut<'a> {
                docid: &'a str,
                file: String,
                title: &'a str,
                #[serde(skip_serializing_if = "Option::is_none")]
                context: &'a Option<String>,
                body: String,
            }
            let out = DocOut {
                docid: doc.docid(),
                file: doc.display_path(),
                title: &doc.title,
                context: &doc.context,
                body: body_out,
            };
            serde_json::to_string_pretty(&out).unwrap_or_default()
        }
        Format::Xml => {
            let mut s = String::from("<document>\n");
            let _ = writeln!(s, "  <docid>{}</docid>", escape_xml(doc.docid()));
            let _ = writeln!(s, "  <file>{}</file>", escape_xml(&doc.display_path()));
            let _ = writeln!(s, "  <title>{}</title>", escape_xml(&doc.title));
            if let Some(ref ctx) = doc.context {
                let _ = writeln!(s, "  <context>{}</context>", escape_xml(ctx));
            }
            let _ = writeln!(s, "  <body>{}</body>", escape_xml(&body_out));
            s.push_str("</document>");
            s
        }
        Format::Md => {
            let mut s = format!("# {}\n\n", doc.title);
            let _ = writeln!(s, "**File:** `{}`  ", doc.display_path());
            let _ = writeln!(s, "**Docid:** `#{}`\n", doc.docid());
            if let Some(ref ctx) = doc.context {
                let _ = writeln!(s, "> {ctx}\n");
            }
            s.push_str(&body_out);
            s
        }
        _ => body_out,
    }
}

/// Escape special XML characters.
#[must_use]
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Escape a value for CSV (RFC 4180).
#[must_use]
pub fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Render search results as CSV.
fn format_csv(rows: &[ResultRow]) -> String {
    let mut out = String::from("docid,file,title,score,context,snippet\n");
    for r in rows {
        let _ = writeln!(
            out,
            "{},{},{},{:.4},{},{}",
            escape_csv(&r.docid),
            escape_csv(&r.file),
            escape_csv(&r.title),
            r.score,
            escape_csv(r.context.as_deref().unwrap_or("")),
            escape_csv(&r.snippet),
        );
    }
    out
}

/// Render search results as XML.
fn format_xml_results(rows: &[ResultRow]) -> String {
    let mut s = String::from("<results>\n");
    for r in rows {
        s.push_str("  <result>\n");
        let _ = writeln!(s, "    <docid>{}</docid>", escape_xml(&r.docid));
        let _ = writeln!(s, "    <file>{}</file>", escape_xml(&r.file));
        let _ = writeln!(s, "    <title>{}</title>", escape_xml(&r.title));
        let _ = writeln!(s, "    <score>{:.4}</score>", r.score);
        if let Some(ref ctx) = r.context {
            let _ = writeln!(s, "    <context>{}</context>", escape_xml(ctx));
        }
        let _ = writeln!(s, "    <snippet>{}</snippet>", escape_xml(&r.snippet));
        s.push_str("  </result>\n");
    }
    s.push_str("</results>");
    s
}

/// Render search results as a Markdown table.
fn format_md_results(rows: &[ResultRow]) -> String {
    if rows.is_empty() {
        return String::from("*No results found.*\n");
    }
    let mut s = String::from("| # | Score | File | Title |\n|---|-------|------|-------|\n");
    for (i, r) in rows.iter().enumerate() {
        let _ = writeln!(
            s,
            "| {} | {:.0}% | `{}` | {} |",
            i + 1,
            r.score * 100.0,
            r.file,
            r.title,
        );
    }
    s
}

/// Render search results as human-readable CLI output.
fn format_cli_results(rows: &[ResultRow]) -> String {
    if rows.is_empty() {
        return String::from("No results found.\n");
    }
    let mut s = String::new();
    for (i, r) in rows.iter().enumerate() {
        let _ = writeln!(
            s,
            "{:>2}. {} {:.0}% {} \u{2014} {}",
            i + 1,
            r.docid,
            r.score * 100.0,
            r.file,
            r.title,
        );
    }
    s
}
