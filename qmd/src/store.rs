//! Database store for document indexing and retrieval.
//!
//! This module provides all database operations, search functions, and document
//! retrieval for QMD.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::collections::ConfigManager;
use crate::config;
use crate::document::{Document, SearchResult, SearchSource};
use crate::error::{Error, Result};

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
    pub fn new() -> Result<Self> {
        Self::with_index("index")
    }

    /// Create a store for a named index.
    pub fn with_index(index_name: &str) -> Result<Self> {
        let db_path = config::db_path(index_name)?;
        Self::open(&db_path, ConfigManager::new(index_name))
    }

    /// Open a store at an explicit path with default configuration.
    ///
    /// Convenient for examples and tests. For production use, prefer
    /// [`Store::open`] with an explicit [`ConfigManager`].
    pub fn open_path(db_path: &Path) -> Result<Self> {
        Self::open(db_path, ConfigManager::default())
    }

    /// Open a store at an explicit database path with a specific config.
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

    /// Extract the first heading from markdown content as a title.
    #[must_use]
    pub fn extract_title(content: &str) -> String {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("# ") {
                return rest.trim().to_string();
            }
            if let Some(rest) = trimmed.strip_prefix("## ") {
                return rest.trim().to_string();
            }
        }
        String::new()
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
    pub fn insert_content(&self, hash: &str, content: &str, created_at: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            params![hash, content, created_at],
        )?;
        Ok(())
    }

    /// Insert a document record.
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
    pub fn find_active_document(
        &self,
        collection: &str,
        path: &str,
    ) -> Result<Option<ActiveDoc>> {
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
    pub fn deactivate_document(&self, collection: &str, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2",
            params![collection, path],
        )?;
        Ok(())
    }

    /// Get all active document paths for a collection.
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
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let coll_filter = if collection.is_some() {
            "AND d.collection = ?2"
        } else {
            ""
        };
        let limit_param = if collection.is_some() { "?3" } else { "?2" };

        let sql = format!(
            r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     bm25(documents_fts) as score, LENGTH(c.doc)
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
                score: -row.get::<_, f64>(5)?,
                source: SearchSource::Fts,
                chunk_pos: None,
            })
        };

        let mut results: Vec<SearchResult> = if let Some(coll) = collection {
            stmt.query_map(params![query, coll, limit as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![query, limit as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        self.attach_context(&mut results);
        Ok(results)
    }

    /// Get document by collection and path.
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
    pub fn rename_collection_documents(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET collection = ?1 WHERE collection = ?2",
            params![new_name, old_name],
        )?;
        Ok(())
    }

    /// Cleanup orphaned content (not referenced by any active document).
    pub fn cleanup_orphaned_content(&self) -> Result<usize> {
        let changes = self.conn.execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        Ok(changes)
    }

    /// Cleanup orphaned vectors.
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
    pub fn delete_inactive_documents(&self) -> Result<usize> {
        let changes = self
            .conn
            .execute("DELETE FROM documents WHERE active = 0", [])?;
        Ok(changes)
    }

    /// Clear LLM cache.
    pub fn clear_cache(&self) -> Result<usize> {
        let changes = self.conn.execute("DELETE FROM llm_cache", [])?;
        Ok(changes)
    }

    /// Vacuum database.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }

    /// Ensure the vector table exists with the correct dimensions.
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

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        self.attach_context(&mut results);
        Ok(results)
    }

    /// Clear all embeddings.
    pub fn clear_embeddings(&self) -> Result<usize> {
        let changes1 = self.conn.execute("DELETE FROM content_vectors", [])?;
        let _ = self.conn.execute("DELETE FROM vectors_vec", []);
        Ok(changes1)
    }

    /// List files in a collection, optionally filtered by path prefix.
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

        hits.sort_by(|a, b| b.score.cmp(&a.score));
        hits.truncate(limit);
        Ok(hits)
    }

    /// Match files using a glob pattern.
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
}
