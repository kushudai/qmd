//! QMD — Query Markdown Documents.
//!
//! Local search engine for markdown files combining full-text search (BM25),
//! vector semantic search, and LLM-powered query expansion.
//!
//! # Architecture
//!
//! | Layer | Modules |
//! |-------|---------|
//! | **Foundation** | [`config`] · [`error`] · [`path`] |
//! | **Data**       | [`document`] · [`collections`] · [`store`] |
//! | **ML**         | [`model`] · [`embed`] · [`generate`] · [`rerank`] |
//! | **Search**     | [`search`] · [`chunk`] |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use qmd::Store;
//! use qmd::embed::EmbeddingEngine;
//!
//! let store = Store::new()?;
//! let results = store.search_fts("rust ownership", 10, None)?;
//!
//! let mut engine = EmbeddingEngine::load_default()?;
//! let emb = engine.embed_query("how does borrowing work?")?;
//! let vec_results = store.search_vec(&emb, 10, None)?;
//! # Ok::<(), qmd::Error>(())
//! ```

pub mod chunk;
pub mod collections;
pub mod config;
pub mod document;
pub mod embed;
pub mod error;
pub mod generate;
pub mod model;
pub mod path;
pub mod rerank;
pub mod search;
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub mod store;

pub use collections::ConfigManager;
pub use document::{Document, SearchResult, SearchSource};
pub use error::{Error, Result};
pub use search::{Query, QueryType};
pub use store::Store;
