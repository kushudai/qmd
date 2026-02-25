//! Database store for document indexing and retrieval.
//!
//! This module provides all database operations, search functions, and document
//! retrieval for QMD.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::collections::ConfigManager;
use crate::config;
use crate::document::{Document, SearchResult, SearchSource};
use crate::error::{Error, Result};
use crate::search;

/// RFC 3339 UTC timestamp from system clock.
fn now_rfc3339() -> String {
    humantime::format_rfc3339_seconds(SystemTime::now()).to_string()
}

/// Collection info from database.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Working directory path.
    pub pwd: String,
    /// Glob pattern.
    pub glob_pattern: String,
    /// Number of active documents.
    pub active_count: usize,
    /// Last modification timestamp.
    pub last_modified: Option<String>,
}

/// Unified index statistics and health.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexStats {
    /// Total active documents.
    pub total_docs: usize,
    /// Documents needing embedding.
    pub needs_embedding: usize,
    /// Whether the vector table exists.
    pub has_vectors: bool,
    /// Days since last update (`None` if never indexed).
    pub days_stale: Option<u64>,
    /// Per-collection info.
    pub collections: Vec<CollectionInfo>,
}

impl IndexStats {
    /// Whether the index is considered healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let embedding_ok = self.needs_embedding == 0
            || (self.needs_embedding as f64 / self.total_docs.max(1) as f64) < 0.1;
        let fresh = self.days_stale.is_none() || self.days_stale < Some(14);
        embedding_ok && fresh
    }
}

/// A file entry returned by [`Store::list_files`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileEntry {
    /// Relative path within the collection.
    pub path: String,
    /// Document title.
    pub title: String,
    /// Last modification timestamp.
    pub modified_at: String,
    /// Body size in bytes.
    pub size: usize,
}

/// A document that needs embedding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UnembeddedDoc {
    /// Content hash.
    pub hash: String,
    /// File path.
    pub path: String,
    /// Full document body.
    pub body: String,
}

/// A fuzzy-matched file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FuzzyMatch {
    /// Virtual display path.
    pub display: String,
    /// Raw file path.
    pub path: String,
    /// Match score.
    pub score: i64,
}

/// An active document record (used during indexing).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ActiveDoc {
    /// Database row id.
    pub id: i64,
    /// Content hash.
    pub hash: String,
    /// Document title.
    pub title: String,
}

/// The database store.
#[derive(Debug)]
pub struct Store {
    /// Database connection.
    conn: Connection,
    /// Database file path.
    db_path: PathBuf,
    /// Collection configuration manager.
    config: ConfigManager,
}

impl Store {
    /// Create a new store with the default index name (`"index"`).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or initialized.
    pub fn new() -> Result<Self> {
        Self::with_index("index")
    }

    /// Create a store for a named index.
    ///
    /// # Errors
    ///
    /// Returns an error if the database path cannot be resolved or the store cannot be opened.
    pub fn with_index(index_name: &str) -> Result<Self> {
        let db_path = config::db_path(index_name)?;
        Self::open(&db_path, ConfigManager::new(index_name))
    }

    /// Open a store at an explicit path with default configuration.
    ///
    /// Convenient for examples and tests. For production use, prefer
    /// [`Store::open`] with an explicit [`ConfigManager`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub fn open_path(db_path: &Path) -> Result<Self> {
        Self::open(db_path, ConfigManager::default())
    }

    /// Open a store at an explicit database path with a specific config.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or the schema cannot be initialized.
    pub fn open(db_path: &Path, config: ConfigManager) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;
        let store = Self {
            conn,
            db_path: db_path.to_path_buf(),
            config,
        };
        store.initialize()?;
        Ok(store)
    }

    /// Get a reference to the underlying `ConfigManager`.
    #[must_use]
    pub const fn config(&self) -> &ConfigManager {
        &self.config
    }

    /// Get the database path.
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Initialize database schema.
    fn initialize(&self) -> Result<()> {
        self.conn.execute_batch(
            r"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            -- Content-addressable storage
            CREATE TABLE IF NOT EXISTS content (
                hash TEXT PRIMARY KEY,
                doc TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Documents table
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
                UNIQUE(collection, path)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);

            -- FTS index
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                filepath, title, body,
                tokenize='porter unicode61'
            );

            -- LLM cache
            CREATE TABLE IF NOT EXISTS llm_cache (
                hash TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Content vectors metadata
            CREATE TABLE IF NOT EXISTS content_vectors (
                hash TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                pos INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                PRIMARY KEY (hash, seq)
            );
            ",
        )?;

        // Create FTS triggers.
        self.create_fts_triggers()?;

        Ok(())
    }

    /// Create FTS synchronization triggers.
    fn create_fts_triggers(&self) -> Result<()> {
        // Check if triggers exist.
        let trigger_exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='documents_ai'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !trigger_exists {
            self.conn.execute_batch(
                r"
                CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
                WHEN new.active = 1
                BEGIN
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    SELECT
                        new.id,
                        new.collection || '/' || new.path,
                        new.title,
                        (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
                BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;
                    INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
                    SELECT
                        new.id,
                        new.collection || '/' || new.path,
                        new.title,
                        (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;
                ",
            )?;
        }

        Ok(())
    }

    /// Hash content using SHA-256, returning a lowercase hex string.
    #[must_use]
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Extract a title from document content.
    ///
    /// Supports markdown (`# Heading`, `## Heading`) and Org-mode
    /// (`#+TITLE:`, `* Heading`).  Falls back to a cleaned-up `filename`
    /// (extension stripped, last path component).
    #[must_use]
    pub fn extract_title(content: &str, filename: &str) -> String {
        let ext = filename
            .rfind('.')
            .map_or("", |i| &filename[i..])
            .to_ascii_lowercase();

        let title = match ext.as_str() {
            ".org" => Self::title_from_org(content),
            _ => Self::title_from_markdown(content),
        };

        if let Some(t) = title {
            return t;
        }

        // Fallback: derive title from filename.
        filename
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(filename)
            .rfind('.')
            .map_or_else(
                || filename.to_string(),
                |i| filename.rsplit(['/', '\\']).next().unwrap_or(filename)[..i].to_string(),
            )
    }

    /// Extract title from markdown content (H1 or H2).
    fn title_from_markdown(content: &str) -> Option<String> {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("# ") {
                let t = rest.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
            if let Some(rest) = trimmed.strip_prefix("## ") {
                let t = rest.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
        None
    }

    /// Extract title from Org-mode content (`#+TITLE:` or first heading).
    fn title_from_org(content: &str) -> Option<String> {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("#+TITLE:")
                .or_else(|| trimmed.strip_prefix("#+title:"))
            {
                let t = rest.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
        // Fallback to first org heading.
        for line in content.lines() {
            if let Some(rest) = line.trim().strip_prefix("* ") {
                let t = rest.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
        None
    }

    /// Attach folder context descriptions to search results.
    fn attach_context(&self, results: &mut [SearchResult]) {
        for r in results.iter_mut() {
            r.doc.context = self
                .config
                .find_context(&r.doc.collection, &r.doc.path)
                .unwrap_or(None);
        }
    }

    /// Insert content into content-addressable storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn insert_content(&self, hash: &str, content: &str, created_at: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            params![hash, content, created_at],
        )?;
        Ok(())
    }

    /// Insert a document record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn insert_document(
        &self,
        collection: &str,
        path: &str,
        title: &str,
        hash: &str,
        created_at: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            r"
            INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
            ON CONFLICT(collection, path) DO UPDATE SET
                title = excluded.title,
                hash = excluded.hash,
                modified_at = excluded.modified_at,
                active = 1
            ",
            params![collection, path, title, hash, created_at, modified_at],
        )?;
        Ok(())
    }

    /// Find an active document by collection and path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_active_document(&self, collection: &str, path: &str) -> Result<Option<ActiveDoc>> {
        let result = self
            .conn
            .query_row(
                "SELECT id, hash, title FROM documents WHERE collection = ?1 AND path = ?2 AND active = 1",
                params![collection, path],
                |row| {
                    Ok(ActiveDoc {
                        id: row.get(0)?,
                        hash: row.get(1)?,
                        title: row.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Update document hash, title, and timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn update_document(
        &self,
        document_id: i64,
        title: &str,
        hash: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET title = ?1, hash = ?2, modified_at = ?3 WHERE id = ?4",
            params![title, hash, modified_at, document_id],
        )?;
        Ok(())
    }

    /// Deactivate a document.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn deactivate_document(&self, collection: &str, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2",
            params![collection, path],
        )?;
        Ok(())
    }

    /// Get all active document paths for a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_active_document_paths(&self, collection: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?;
        let paths = stmt
            .query_map(params![collection], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        Ok(paths)
    }

    /// Full-text search using FTS5.
    ///
    /// The `query` is parsed through [`search::build_fts5_query`] which
    /// supports quoted phrases, negation (`-term`), and prefix matching.
    ///
    /// # Errors
    ///
    /// Returns an error if the FTS query execution fails.
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let fts_query = search::build_fts5_query(query).unwrap_or_else(|| query.to_string());

        let coll_filter = if collection.is_some() {
            "AND d.collection = ?2"
        } else {
            ""
        };
        let limit_param = if collection.is_some() { "?3" } else { "?2" };

        // Field weights: filepath 10×, title 1×, body 1×.
        let sql = format!(
            r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     bm25(documents_fts, 10.0, 1.0, 1.0) as score, LENGTH(c.doc)
              FROM documents_fts fts
              JOIN documents d ON d.id = fts.rowid
              JOIN content c ON c.hash = d.hash
              WHERE documents_fts MATCH ?1 {coll_filter} AND d.active = 1
              ORDER BY score
              LIMIT {limit_param}"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            let body_len: i64 = row.get(6)?;
            let raw_bm25: f64 = row.get(5)?;
            Ok(SearchResult {
                doc: Document {
                    collection: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    hash: row.get(3)?,
                    modified_at: row.get(4)?,
                    body_len: body_len as usize,
                    body: None,
                    context: None,
                },
                score: search::normalize_bm25(-raw_bm25),
                source: SearchSource::Fts,
                chunk_pos: None,
            })
        };

        let mut results: Vec<SearchResult> = if let Some(coll) = collection {
            stmt.query_map(params![&fts_query, coll, limit as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![&fts_query, limit as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        self.attach_context(&mut results);
        Ok(results)
    }

    /// Get document by collection and path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_document(&self, collection: &str, path: &str) -> Result<Option<Document>> {
        let result = self
            .conn
            .query_row(
                r"
                SELECT d.title, d.hash, d.modified_at, c.doc, LENGTH(c.doc)
                FROM documents d
                JOIN content c ON c.hash = d.hash
                WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1
                ",
                params![collection, path],
                |row| {
                    let body: String = row.get(3)?;
                    let body_len: i64 = row.get(4)?;
                    Ok(Document {
                        collection: collection.to_string(),
                        path: path.to_string(),
                        title: row.get(0)?,
                        hash: row.get(1)?,
                        modified_at: row.get(2)?,
                        body_len: body_len as usize,
                        body: Some(body),
                        context: None,
                    })
                },
            )
            .optional()?;

        // Attach context.
        Ok(result.map(|mut doc| {
            doc.context = self.config.find_context(collection, path).unwrap_or(None);
            doc
        }))
    }

    /// Get document by docid (first 6 chars of hash).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_document_by_docid(&self, docid: &str) -> Result<Option<(String, String)>> {
        let clean_docid = docid.trim_start_matches('#');
        let result = self
            .conn
            .query_row(
                r"
                SELECT d.collection, d.path
                FROM documents d
                WHERE d.hash LIKE ?1 || '%' AND d.active = 1
                LIMIT 1
                ",
                params![clean_docid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        Ok(result)
    }

    /// List collections with stats from database.
    ///
    /// # Errors
    ///
    /// Returns an error if the config or database cannot be read.
    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let yaml_collections = self.config.list()?;

        let mut collections = Vec::new();

        for (name, coll) in yaml_collections {
            let stats: (i64, Option<String>) = self
                .conn
                .query_row(
                    r"
                    SELECT COUNT(*) as count, MAX(modified_at) as last_modified
                    FROM documents
                    WHERE collection = ?1 AND active = 1
                    ",
                    params![name],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap_or((0, None));

            collections.push(CollectionInfo {
                name,
                pwd: coll.path,
                glob_pattern: coll.pattern,
                active_count: stats.0 as usize,
                last_modified: stats.1,
            });
        }

        Ok(collections)
    }

    /// Unified index statistics and health diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error if the database queries fail.
    pub fn stats(&self) -> Result<IndexStats> {
        let total_docs = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE active = 1",
            [],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?;

        let needs_embedding = self.conn.query_row(
            r"SELECT COUNT(DISTINCT d.hash)
              FROM documents d
              LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
              WHERE d.active = 1 AND v.hash IS NULL",
            [],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?;

        let has_vectors: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        let days_stale: Option<u64> = self
            .conn
            .query_row(
                r"SELECT CAST(julianday('now') - julianday(MAX(modified_at)) AS INTEGER)
                  FROM documents WHERE active = 1",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .ok()
            .flatten()
            .map(|d| d.max(0) as u64);

        let collections = self.list_collections()?;

        Ok(IndexStats {
            total_docs,
            needs_embedding,
            has_vectors,
            days_stale,
            collections,
        })
    }

    /// Remove a collection and its documents from the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operations fail.
    pub fn remove_collection_documents(&self, name: &str) -> Result<(usize, usize)> {
        // Get count before deletion.
        let doc_count = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE collection = ?1",
            params![name],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?;

        // Delete documents.
        self.conn
            .execute("DELETE FROM documents WHERE collection = ?1", params![name])?;

        // Cleanup orphaned content.
        let cleaned = self.cleanup_orphaned_content()?;

        Ok((doc_count, cleaned))
    }

    /// Rename collection in database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn rename_collection_documents(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET collection = ?1 WHERE collection = ?2",
            params![new_name, old_name],
        )?;
        Ok(())
    }

    /// Cleanup orphaned content (not referenced by any active document).
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn cleanup_orphaned_content(&self) -> Result<usize> {
        let changes = self.conn.execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        Ok(changes)
    }

    /// Cleanup orphaned vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn cleanup_orphaned_vectors(&self) -> Result<usize> {
        let changes = self.conn.execute(
            r"
            DELETE FROM content_vectors
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            ",
            [],
        )?;
        Ok(changes)
    }

    /// Delete inactive documents.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn delete_inactive_documents(&self) -> Result<usize> {
        let changes = self
            .conn
            .execute("DELETE FROM documents WHERE active = 0", [])?;
        Ok(changes)
    }

    /// Look up a cached LLM result by cache key.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn cache_get(&self, key: &str) -> Result<Option<String>> {
        let result = self
            .conn
            .query_row(
                "SELECT result FROM llm_cache WHERE hash = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Store an LLM result in the cache.
    ///
    /// Probabilistically evicts old entries (~1 % of writes) to keep
    /// the cache bounded at 1 000 entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn cache_set(&self, key: &str, result: &str) -> Result<()> {
        let now = now_rfc3339();
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?1, ?2, ?3)",
            params![key, result, now],
        )?;

        // Probabilistic eviction: ~1 % of writes.
        if fastrand::u8(..) < 3 {
            self.conn.execute(
                "DELETE FROM llm_cache WHERE hash NOT IN \
                 (SELECT hash FROM llm_cache ORDER BY created_at DESC LIMIT 1000)",
                [],
            )?;
        }
        Ok(())
    }

    /// Build a cache key from an operation name and input body.
    #[must_use]
    pub fn cache_key(operation: &str, body: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(operation.as_bytes());
        hasher.update(body.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Clear the entire LLM cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn clear_cache(&self) -> Result<usize> {
        let changes = self.conn.execute("DELETE FROM llm_cache", [])?;
        Ok(changes)
    }

    /// Vacuum database.
    ///
    /// # Errors
    ///
    /// Returns an error if the vacuum operation fails.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }

    /// Ensure the vector table exists with the correct dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the table creation fails.
    pub fn ensure_vector_table(&self, _dimensions: usize) -> Result<()> {
        // Create vectors_vec table for storing embeddings
        self.conn.execute(
            r"
                CREATE TABLE IF NOT EXISTS vectors_vec (
                    hash_seq TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                ",
            [],
        )?;
        Ok(())
    }

    /// Insert an embedding for a content hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub fn insert_embedding(
        &self,
        hash: &str,
        seq: usize,
        pos: usize,
        embedding: &[f32],
        model: &str,
        embedded_at: &str,
    ) -> Result<()> {
        // Insert metadata
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ",
            params![hash, seq as i64, pos as i64, model, embedded_at],
        )?;

        // Insert vector data
        let hash_seq = format!("{hash}_{seq}");
        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
            params![hash_seq, embedding_bytes],
        )?;

        Ok(())
    }

    /// Documents that need embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn unembedded_docs(&self) -> Result<Vec<UnembeddedDoc>> {
        let mut stmt = self.conn.prepare(
            r"SELECT DISTINCT d.hash, d.path, c.doc
              FROM documents d
              JOIN content c ON c.hash = d.hash
              LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
              WHERE d.active = 1 AND v.hash IS NULL",
        )?;

        let results = stmt
            .query_map([], |row| {
                Ok(UnembeddedDoc {
                    hash: row.get(0)?,
                    path: row.get(1)?,
                    body: row.get(2)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Get embedding for a hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_embedding(&self, hash: &str, seq: usize) -> Result<Option<Vec<f32>>> {
        let hash_seq = format!("{hash}_{seq}");
        let result: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT embedding FROM vectors_vec WHERE hash_seq = ?1",
                params![hash_seq],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result.map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }))
    }

    /// Vector similarity search.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query or embedding retrieval fails.
    pub fn search_vec(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let coll_filter = if collection.is_some() {
            "AND d.collection = ?1"
        } else {
            ""
        };

        let sql = format!(
            r"SELECT DISTINCT d.collection, d.path, d.title, d.hash,
                     d.modified_at, LENGTH(c.doc)
              FROM documents d
              JOIN content c ON c.hash = d.hash
              JOIN vectors_vec v ON v.hash_seq = d.hash || '_0'
              WHERE d.active = 1 {coll_filter}"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            let body_len: i64 = row.get(5)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                body_len as usize,
            ))
        };

        let rows: Vec<_> = if let Some(coll) = collection {
            stmt.query_map(params![coll], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut results: Vec<SearchResult> = Vec::new();
        for (coll_name, path, title, hash, modified_at, body_len) in rows {
            if let Some(emb) = self.get_embedding(&hash, 0)? {
                let sim = crate::embed::cosine_similarity(query_embedding, &emb);
                results.push(SearchResult {
                    doc: Document {
                        collection: coll_name,
                        path,
                        title,
                        hash,
                        modified_at,
                        body_len,
                        body: None,
                        context: None,
                    },
                    score: f64::from(sim),
                    source: SearchSource::Vec,
                    chunk_pos: Some(0),
                });
            }
        }

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(limit);
        self.attach_context(&mut results);
        Ok(results)
    }

    /// Clear all embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn clear_embeddings(&self) -> Result<usize> {
        let changes1 = self.conn.execute("DELETE FROM content_vectors", [])?;
        let _ = self.conn.execute("DELETE FROM vectors_vec", []);
        Ok(changes1)
    }

    /// List files in a collection, optionally filtered by path prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn list_files(
        &self,
        collection: &str,
        path_prefix: Option<&str>,
    ) -> Result<Vec<FileEntry>> {
        let prefix_filter = if path_prefix.is_some() {
            "AND d.path LIKE ?2"
        } else {
            ""
        };

        let sql = format!(
            r"SELECT d.path, d.title, d.modified_at, LENGTH(c.doc)
              FROM documents d
              JOIN content c ON d.hash = c.hash
              WHERE d.collection = ?1 {prefix_filter} AND d.active = 1
              ORDER BY d.path"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            let size: i64 = row.get(3)?;
            Ok(FileEntry {
                path: row.get(0)?,
                title: row.get(1)?,
                modified_at: row.get(2)?,
                size: size as usize,
            })
        };

        let files = if let Some(prefix) = path_prefix {
            let pattern = format!("{prefix}%");
            stmt.query_map(params![collection, pattern], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![collection], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(files)
    }

    /// Find files similar to `query` using fuzzy matching.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_similar_files(&self, query: &str, limit: usize) -> Result<Vec<FuzzyMatch>> {
        use fuzzy_matcher::FuzzyMatcher;
        use fuzzy_matcher::skim::SkimMatcherV2;

        let matcher = SkimMatcherV2::default();
        let query_lower = query.to_lowercase();

        let mut stmt = self
            .conn
            .prepare("SELECT collection, path FROM documents WHERE active = 1")?;

        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(std::result::Result::ok)
            .collect();

        let mut hits: Vec<FuzzyMatch> = rows
            .into_iter()
            .filter_map(|(collection, path)| {
                let display = crate::path::build_virtual(&collection, &path);
                matcher
                    .fuzzy_match(&path.to_lowercase(), &query_lower)
                    .map(|score| FuzzyMatch {
                        display,
                        path,
                        score,
                    })
            })
            .collect();

        hits.sort_by_key(|b| std::cmp::Reverse(b.score));
        hits.truncate(limit);
        Ok(hits)
    }

    /// Match files using a glob pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the glob pattern is invalid or the database query fails.
    pub fn match_files_by_glob(&self, pattern: &str) -> Result<Vec<Document>> {
        let glob_pattern = glob::Pattern::new(pattern).map_err(|e| Error::Config(e.to_string()))?;

        let mut stmt = self.conn.prepare(
            r"
            SELECT d.collection, d.path, d.title, d.hash, d.modified_at, LENGTH(c.doc)
            FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.active = 1
            ",
        )?;

        let results: Vec<Document> = stmt
            .query_map([], |row| {
                let collection: String = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let hash: String = row.get(3)?;
                let modified_at: String = row.get(4)?;
                let body_length: i64 = row.get(5)?;
                Ok((collection, path, title, hash, modified_at, body_length))
            })?
            .filter_map(std::result::Result::ok)
            .filter(|(_, path, _, _, _, _)| glob_pattern.matches(path))
            .map(
                |(collection, path, title, hash, modified_at, body_length)| {
                    let context = self.config.find_context(&collection, &path).ok().flatten();

                    Document {
                        collection,
                        path,
                        title,
                        hash,
                        modified_at,
                        body_len: body_length as usize,
                        body: None,
                        context,
                    }
                },
            )
            .collect();

        Ok(results)
    }

    /// Get document body content by hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_body_by_hash(&self, hash: &str) -> Result<Option<String>> {
        let result = self
            .conn
            .query_row(
                "SELECT doc FROM content WHERE hash = ?1",
                params![hash],
                |row| row.get(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Get document body with optional line range.
    ///
    /// Returns `(text, total_lines)`.  If `from_line` or `max_lines` are
    /// provided, only that slice is returned.  Lines are 1-indexed.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails or the document is not found.
    pub fn get_document_body(
        &self,
        collection: &str,
        path: &str,
        from_line: Option<usize>,
        max_lines: Option<usize>,
    ) -> Result<Option<(String, usize)>> {
        let Some(doc) = self.get_document(collection, path)? else {
            return Ok(None);
        };
        let body = doc.body.as_deref().unwrap_or("");
        let lines: Vec<&str> = body.lines().collect();
        let total = lines.len();

        let start = from_line.unwrap_or(1).saturating_sub(1).min(total);
        let count = max_lines.unwrap_or(total - start);
        let end = (start + count).min(total);

        let text = lines[start..end].join("\n");
        Ok(Some((text, total)))
    }

    /// Serialize expanded queries into a cacheable string format.
    fn serialize_queries(queries: &[search::Query]) -> String {
        queries
            .iter()
            .map(|q| {
                let kind = match q.kind {
                    search::QueryType::Lex => "lex",
                    search::QueryType::Vec => "vec",
                    search::QueryType::Hyde => "hyde",
                };
                format!("{kind}: {}", q.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Expand the user query into multiple sub-queries using LLM or cache.
    fn expand_queries(
        &self,
        query: &str,
        fts_scores: &[f64],
        generator: Option<&crate::generate::GenerationEngine>,
    ) -> Vec<search::Query> {
        if search::has_strong_signal(fts_scores) || generator.is_none() {
            return search::Query::expand_simple(query);
        }

        let cache_key = Self::cache_key("expand", query);
        if let Ok(Some(cached)) = self.cache_get(&cache_key) {
            return search::Query::from_llm_output(&cached, query);
        }

        // Safety: generator.is_none() is handled by early return above.
        let engine = generator.unwrap_or_else(|| unreachable!());
        let Ok(expanded) = engine.expand_query(query, true) else {
            return search::Query::expand_simple(query);
        };

        // Cache the raw LLM output for next time.
        let raw = Self::serialize_queries(&expanded);
        let _ = self.cache_set(&cache_key, &raw);
        expanded
    }

    /// Apply reranking to the top results using the cross-encoder.
    ///
    /// Instead of sending full document bodies, extracts a focused snippet
    /// around query terms (~4000 chars ≈ 1024 tokens) so the reranker sees
    /// the most relevant portion.  After reranking, blends the reranker
    /// score with the original RRF positional rank to avoid the reranker
    /// completely overriding good fusion results.
    fn apply_reranking(
        &self,
        query: &str,
        results: &mut Vec<SearchResult>,
        limit: usize,
        ranker: &crate::rerank::RerankEngine,
    ) {
        /// Max snippet chars for the reranker (~1024 tokens at 4 chars/token).
        const RERANK_SNIPPET_CHARS: usize = 4000;

        let top_n = results.len().min(limit * 2);
        let candidates: Vec<(String, String, Option<String>)> = results[..top_n]
            .iter()
            .filter_map(|r| {
                let body = self.get_body_by_hash(&r.doc.hash).ok()??;
                // Extract the most relevant snippet instead of full body.
                let snippet =
                    search::extract_snippet(&body, query, RERANK_SNIPPET_CHARS, r.chunk_pos);
                Some((
                    r.doc.display_path(),
                    snippet.text,
                    Some(r.doc.title.clone()),
                ))
            })
            .collect();

        if candidates.is_empty() {
            return;
        }
        let Ok(scored) = ranker.rerank(query, &candidates) else {
            return;
        };

        // Position-aware score blending: merge reranker score with RRF rank.
        let mut ranked_output: Vec<SearchResult> = Vec::with_capacity(scored.len());
        for s in &scored {
            if let Some(pos) = results.iter().position(|r| r.doc.display_path() == s.key) {
                let mut r = results.swap_remove(pos);
                let rrf_score = r.score;
                let rerank_score = f64::from(s.score);
                // Blend weight based on original RRF rank position.
                let blend = match pos {
                    0..=2 => 0.75,
                    3..=9 => 0.60,
                    _ => 0.40,
                };
                r.score = blend * rerank_score + (1.0 - blend) * rrf_score;
                ranked_output.push(r);
            }
        }
        // Re-sort by blended score.
        ranked_output.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranked_output.append(results);
        *results = ranked_output;
    }

    /// Structured hybrid search with automatic LLM query expansion.
    ///
    /// 1. Run initial FTS with the raw query.
    /// 2. If a [`GenerationEngine`](crate::generate::GenerationEngine) is
    ///    provided *and* the initial FTS lacks a strong signal, expand the
    ///    query into multiple sub-queries (lex / vec / hyde).
    /// 3. Run FTS + vector search for each sub-query, collecting ranked
    ///    key lists.
    /// 4. Fuse all lists with Reciprocal Rank Fusion (RRF).
    /// 5. Optionally rerank the top candidates with a
    ///    [`RerankEngine`](crate::rerank::RerankEngine).
    /// 6. Return the final `SearchResult` list, scored and sorted.
    ///
    /// # Errors
    ///
    /// Returns an error if any database query or embedding operation fails.
    pub fn structured_search(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
        embed: &mut crate::embed::EmbeddingEngine,
        generator: Option<&crate::generate::GenerationEngine>,
        reranker: Option<&crate::rerank::RerankEngine>,
    ) -> Result<Vec<SearchResult>> {
        let fetch_limit = limit * 3;

        let initial_fts = self
            .search_fts(query, fetch_limit, collection)
            .ok()
            .unwrap_or_default();
        let fts_scores: Vec<f64> = initial_fts.iter().map(|r| r.score).collect();
        let queries = self.expand_queries(query, &fts_scores, generator);

        Ok(self.execute_search_pipeline(
            query,
            &queries,
            initial_fts,
            limit,
            collection,
            embed,
            reranker,
        ))
    }

    /// Search with pre-expanded queries, skipping LLM expansion.
    ///
    /// Use this when the caller (e.g. an MCP tool or external LLM) has
    /// already generated query variations.  Accepts a raw query for the
    /// initial FTS probe plus a list of typed sub-queries.
    ///
    /// # Errors
    ///
    /// Returns an error if any database query or embedding operation fails.
    pub fn search_with_queries(
        &self,
        query: &str,
        queries: &[search::Query],
        limit: usize,
        collection: Option<&str>,
        embed: &mut crate::embed::EmbeddingEngine,
        reranker: Option<&crate::rerank::RerankEngine>,
    ) -> Result<Vec<SearchResult>> {
        let fetch_limit = limit * 3;
        let initial_fts = self
            .search_fts(query, fetch_limit, collection)
            .ok()
            .unwrap_or_default();

        Ok(self.execute_search_pipeline(
            query,
            queries,
            initial_fts,
            limit,
            collection,
            embed,
            reranker,
        ))
    }

    /// Shared search pipeline: FTS + sub-queries → RRF → rerank.
    #[allow(clippy::too_many_arguments)]
    fn execute_search_pipeline(
        &self,
        query: &str,
        queries: &[search::Query],
        initial_fts: Vec<SearchResult>,
        limit: usize,
        collection: Option<&str>,
        embed: &mut crate::embed::EmbeddingEngine,
        reranker: Option<&crate::rerank::RerankEngine>,
    ) -> Vec<SearchResult> {
        let fetch_limit = limit * 3;

        let mut all_lists: Vec<Vec<String>> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        // Helper: collect search results into the shared map and ranked list.
        let mut collect = |results: Vec<SearchResult>, weight: f64| {
            let keys: Vec<String> = results.iter().map(|r| r.doc.display_path()).collect();
            for r in results {
                result_map.entry(r.doc.display_path()).or_insert(r);
            }
            if !keys.is_empty() {
                all_lists.push(keys);
                all_weights.push(weight);
            }
        };

        collect(initial_fts, 1.0);

        for q in queries {
            match q.kind {
                search::QueryType::Lex => {
                    if let Ok(hits) = self.search_fts(&q.text, fetch_limit, collection) {
                        collect(hits, 0.8);
                    }
                }
                search::QueryType::Vec | search::QueryType::Hyde => {
                    if let Ok(emb) = embed.embed_query(&q.text)
                        && let Ok(hits) = self.search_vec(&emb, fetch_limit, collection)
                    {
                        collect(hits, 1.0);
                    }
                }
            }
        }

        // RRF fusion.
        let list_refs: Vec<&[String]> = all_lists.iter().map(Vec::as_slice).collect();
        let fused = search::rrf(&list_refs, Some(&all_weights), 60);

        let mut results: Vec<SearchResult> = fused
            .iter()
            .filter_map(|hit| {
                result_map.remove(&hit.key).map(|mut r| {
                    r.score = hit.score;
                    r
                })
            })
            .collect();

        // Optional reranking.
        if let Some(ranker) = reranker {
            self.apply_reranking(query, &mut results, limit, ranker);
        }

        results.truncate(limit);
        self.attach_context(&mut results);
        results
    }
}
