//! QMD CLI — local search engine for markdown files.

#![allow(
    clippy::missing_docs_in_private_items,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

mod glibc_compat;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qmd::{Collection, Qmd};

/// QMD — local search engine for markdown files.
#[derive(Parser)]
#[command(name = "qmd", version, about)]
struct Cli {
    /// Path to the SQLite index file.
    #[arg(long, default_value = "index.sqlite")]
    index: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Manage collections.
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    /// Re-index all (or specific) collections.
    Update {
        /// Only index these collections.
        #[arg(short, long)]
        collection: Vec<String>,
    },
    /// Generate vector embeddings for unembedded documents.
    Embed {
        /// Clear all embeddings first.
        #[arg(long)]
        force: bool,
    },
    /// Hybrid search (FTS + vector + RRF + rerank).
    Search {
        /// Search query.
        query: String,
        /// Max results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Full-text keyword search (BM25 only).
    Fts {
        /// Search query.
        query: String,
        /// Max results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Get a document by collection/path or #docid.
    Get {
        /// Path (collection/file.md) or docid (#abc123).
        path: String,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show index status.
    Status {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Manage contexts.
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },
    /// Clean up inactive documents and orphaned data.
    Cleanup,
    /// Vacuum the database to reclaim space.
    Vacuum,
}

#[derive(Subcommand)]
enum CollectionAction {
    /// Register a new collection.
    Add {
        /// Absolute path to the directory.
        path: PathBuf,
        /// Collection name.
        #[arg(long)]
        name: String,
        /// Glob pattern for files (default: **/*.md).
        #[arg(long, default_value = "**/*.md")]
        pattern: String,
    },
    /// List all collections.
    List {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove a collection.
    Remove {
        /// Collection name.
        name: String,
    },
    /// Rename a collection.
    Rename {
        /// Current name.
        old: String,
        /// New name.
        new: String,
    },
}

#[derive(Subcommand)]
enum ContextAction {
    /// Add context to a collection path.
    Add {
        /// Collection name.
        collection: String,
        /// Path prefix (e.g. "/" or "/api").
        path: String,
        /// Context description text.
        text: String,
    },
    /// List all contexts.
    List {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove context from a collection path.
    #[command(name = "rm")]
    Remove {
        /// Collection name.
        collection: String,
        /// Path prefix to remove.
        path: String,
    },
    /// Set or clear global context.
    Global {
        /// Context text (omit to clear).
        text: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = run(cli);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> qmd::Result<()> {
    match cli.command {
        Command::Collection { action } => cmd_collection(&cli.index, action),
        Command::Update { collection } => cmd_update(&cli.index, &collection),
        Command::Embed { force } => cmd_embed(&cli.index, force),
        Command::Search { query, limit, json } => cmd_search(&cli.index, &query, limit, json),
        Command::Fts { query, limit, json } => cmd_fts(&cli.index, &query, limit, json),
        Command::Get { path, json } => cmd_get(&cli.index, &path, json),
        Command::Status { json } => cmd_status(&cli.index, json),
        Command::Context { action } => cmd_context(&cli.index, action),
        Command::Cleanup => cmd_cleanup(&cli.index),
        Command::Vacuum => cmd_vacuum(&cli.index),
    }
}

fn cmd_collection(index: &PathBuf, action: CollectionAction) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    match action {
        CollectionAction::Add {
            path,
            name,
            pattern,
        } => {
            let abs = std::fs::canonicalize(&path)
                .map_err(|e| qmd::Error::Config(format!("{}: {e}", path.display())))?;
            let coll = Collection::new(&name, abs.to_string_lossy().as_ref()).with_pattern(pattern);
            qmd.register_collection(&coll)?;
            println!("registered collection '{name}' at {}", abs.display());
        }
        CollectionAction::List { json } => {
            let colls = qmd.list_collections()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&colls)?);
            } else if colls.is_empty() {
                println!("no collections registered");
            } else {
                for c in &colls {
                    println!(
                        "{:<16} {:<40} {} docs",
                        c.collection.name, c.collection.path, c.doc_count
                    );
                }
            }
        }
        CollectionAction::Remove { name } => {
            let n = qmd.remove_collection(&name)?;
            println!("removed '{name}' ({n} documents)");
        }
        CollectionAction::Rename { old, new } => {
            qmd.rename_collection(&old, &new)?;
            println!("renamed '{old}' → '{new}'");
        }
    }
    Ok(())
}

fn cmd_update(index: &PathBuf, collections: &[String]) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    let filter: Option<Vec<&str>> = if collections.is_empty() {
        None
    } else {
        Some(collections.iter().map(String::as_str).collect())
    };
    let r = qmd.update(filter.as_deref())?;
    println!(
        "{} collections: {} indexed, {} updated, {} unchanged, {} removed",
        r.collections, r.indexed, r.updated, r.unchanged, r.removed
    );
    Ok(())
}

fn cmd_embed(index: &PathBuf, force: bool) -> qmd::Result<()> {
    let mut qmd = Qmd::open(index)?;
    if force {
        qmd.clear_embeddings()?;
        println!("cleared all embeddings");
    }
    let r = qmd.embed()?;
    println!("{} documents embedded, {} chunks", r.embedded, r.chunks);
    Ok(())
}

fn cmd_search(index: &PathBuf, query: &str, limit: usize, json: bool) -> qmd::Result<()> {
    let mut qmd = Qmd::open(index)?;
    let results = qmd.search(query, limit)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("no results");
    } else {
        for r in &results {
            println!(
                "{:.3}  #{} {} — {}",
                r.score,
                r.doc.docid(),
                r.doc.display_path(),
                r.doc.title,
            );
        }
    }
    Ok(())
}

fn cmd_fts(index: &PathBuf, query: &str, limit: usize, json: bool) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    let results = qmd.search_fts(query, limit)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("no results");
    } else {
        for r in &results {
            println!(
                "{:.3}  #{} {} — {}",
                r.score,
                r.doc.docid(),
                r.doc.display_path(),
                r.doc.title,
            );
        }
    }
    Ok(())
}

fn cmd_get(index: &PathBuf, path: &str, json: bool) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    let doc = qmd.get(path)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        println!("# {}\n", doc.title);
        if let Some(body) = &doc.body {
            println!("{body}");
        }
    }
    Ok(())
}

fn cmd_status(index: &PathBuf, json: bool) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    let s = qmd.status()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        println!("documents:        {}", s.total_documents);
        println!("needs embedding:  {}", s.needs_embedding);
        println!(
            "vector index:     {}",
            if s.has_vector_index { "yes" } else { "no" }
        );
        if !s.collections.is_empty() {
            println!("\ncollections:");
            for c in &s.collections {
                println!(
                    "  {:<16} {} docs  {}",
                    c.collection.name, c.doc_count, c.collection.path,
                );
            }
        }
    }
    Ok(())
}

fn cmd_context(index: &PathBuf, action: ContextAction) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    match action {
        ContextAction::Add {
            collection,
            path,
            text,
        } => {
            if qmd.set_context(&collection, &path, &text)? {
                println!("context set for {collection}:{path}");
            } else {
                eprintln!("collection '{collection}' not found");
            }
        }
        ContextAction::List { json } => {
            let colls = qmd.db().list_collections()?;
            let global = qmd.global_context()?;
            if json {
                let mut all = Vec::new();
                if let Some(g) = &global {
                    all.push(serde_json::json!({"collection": "*", "path": "/", "context": g}));
                }
                for c in &colls {
                    for (p, t) in &c.context {
                        all.push(
                            serde_json::json!({"collection": c.name, "path": p, "context": t}),
                        );
                    }
                }
                println!("{}", serde_json::to_string_pretty(&all)?);
            } else {
                if let Some(g) = &global {
                    println!("*  /  {g}");
                }
                for c in &colls {
                    for (p, t) in &c.context {
                        println!("{}  {}  {t}", c.name, p);
                    }
                }
            }
        }
        ContextAction::Remove { collection, path } => {
            if qmd.remove_context(&collection, &path)? {
                println!("context removed for {collection}:{path}");
            } else {
                eprintln!("context not found");
            }
        }
        ContextAction::Global { text } => {
            qmd.set_global_context(text.as_deref())?;
            if text.is_some() {
                println!("global context set");
            } else {
                println!("global context cleared");
            }
        }
    }
    Ok(())
}

fn cmd_cleanup(index: &PathBuf) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    let n = qmd.cleanup()?;
    println!("{n} items cleaned up");
    Ok(())
}

fn cmd_vacuum(index: &PathBuf) -> qmd::Result<()> {
    let qmd = Qmd::open(index)?;
    qmd.vacuum()?;
    println!("database vacuumed");
    Ok(())
}
