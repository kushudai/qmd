//! YAML-based collection configuration.
//!
//! A collection maps a filesystem directory + glob pattern to an index.
//! Config lives at `~/.config/qmd/{index}.yml`.

use std::collections::BTreeMap;
use std::fs;

use serde::{Deserialize, Serialize};

use crate::config;
use crate::error::{Error, Result};

/// Default index name.
const DEFAULT_INDEX: &str = "index";

/// A single collection: directory + glob pattern + optional context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Collection {
    /// Absolute filesystem path to index.
    pub path: String,
    /// Glob pattern (e.g. `**/*.md`).
    pub pattern: String,
    /// Per-path context descriptions (prefix → text).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<BTreeMap<String, String>>,
    /// Shell command to run on `qmd update`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
}

/// Root config file: global context + named collections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Context applied to all collections.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
    /// Collection name → configuration.
    #[serde(default)]
    pub collections: BTreeMap<String, Collection>,
}

/// Manages collection config for a named index.
///
/// Each operation reads/writes `~/.config/qmd/{index}.yml` directly,
/// guaranteeing consistency with no stale cache.
#[derive(Debug, Clone)]
pub struct ConfigManager {
    /// Index name this manager operates on.
    index_name: String,
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self {
            index_name: DEFAULT_INDEX.to_string(),
        }
    }
}

impl ConfigManager {
    /// Create a manager for the given index name.
    #[must_use]
    pub fn new(index_name: &str) -> Self {
        Self {
            index_name: index_name.to_string(),
        }
    }

    /// The index name.
    #[must_use]
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Load config from disk. Returns empty config if absent.
    pub fn load(&self) -> Result<Config> {
        let path = config::config_path(&self.index_name)?;
        if !path.exists() {
            return Ok(Config::default());
        }
        let content = fs::read_to_string(&path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    /// Persist config to disk, creating directories as needed.
    pub fn save(&self, cfg: &Config) -> Result<()> {
        let dir = config::config_dir()?;
        fs::create_dir_all(&dir)?;
        let path = config::config_path(&self.index_name)?;
        fs::write(&path, serde_yaml::to_string(cfg)?)?;
        Ok(())
    }

    /// Get a collection by name.
    pub fn get(&self, name: &str) -> Result<Option<Collection>> {
        Ok(self.load()?.collections.get(name).cloned())
    }

    /// List all collections as `(name, collection)` pairs.
    pub fn list(&self) -> Result<BTreeMap<String, Collection>> {
        Ok(self.load()?.collections)
    }

    /// Add or update a collection (preserves existing context).
    pub fn add(&self, name: &str, path: &str, pattern: &str) -> Result<()> {
        let mut cfg = self.load()?;
        let existing_ctx = cfg.collections.get(name).and_then(|c| c.context.clone());
        cfg.collections.insert(
            name.to_string(),
            Collection {
                path: path.to_string(),
                pattern: pattern.to_string(),
                context: existing_ctx,
                update: None,
            },
        );
        self.save(&cfg)
    }

    /// Remove a collection. Returns `true` if it existed.
    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut cfg = self.load()?;
        let removed = cfg.collections.remove(name).is_some();
        if removed {
            self.save(&cfg)?;
        }
        Ok(removed)
    }

    /// Rename a collection. Returns `false` if `old_name` was not found.
    pub fn rename(&self, old_name: &str, new_name: &str) -> Result<bool> {
        let mut cfg = self.load()?;
        let Some(coll) = cfg.collections.remove(old_name) else {
            return Ok(false);
        };
        if cfg.collections.contains_key(new_name) {
            return Err(Error::Config(format!(
                "collection '{new_name}' already exists"
            )));
        }
        cfg.collections.insert(new_name.to_string(), coll);
        self.save(&cfg)?;
        Ok(true)
    }

    /// Get the global context.
    pub fn global_context(&self) -> Result<Option<String>> {
        Ok(self.load()?.global_context)
    }

    /// Set (or clear) the global context.
    pub fn set_global_context(&self, context: Option<&str>) -> Result<()> {
        let mut cfg = self.load()?;
        cfg.global_context = context.map(str::to_string);
        self.save(&cfg)
    }

    /// Add or update a context entry. Returns `false` if collection not found.
    pub fn add_context(&self, collection: &str, prefix: &str, text: &str) -> Result<bool> {
        let mut cfg = self.load()?;
        let Some(coll) = cfg.collections.get_mut(collection) else {
            return Ok(false);
        };
        coll.context
            .get_or_insert_with(BTreeMap::new)
            .insert(prefix.to_string(), text.to_string());
        self.save(&cfg)?;
        Ok(true)
    }

    /// Remove a context entry. Returns `true` if it existed.
    pub fn remove_context(&self, collection: &str, prefix: &str) -> Result<bool> {
        let mut cfg = self.load()?;
        let Some(coll) = cfg.collections.get_mut(collection) else {
            return Ok(false);
        };
        let Some(ref mut map) = coll.context else {
            return Ok(false);
        };
        let removed = map.remove(prefix).is_some();
        if map.is_empty() {
            coll.context = None;
        }
        if removed {
            self.save(&cfg)?;
        }
        Ok(removed)
    }

    /// Find the best matching context for a file path.
    ///
    /// Uses longest-prefix matching. Falls back to global context.
    pub fn find_context(&self, collection: &str, file_path: &str) -> Result<Option<String>> {
        let cfg = self.load()?;

        let Some(coll) = cfg.collections.get(collection) else {
            return Ok(cfg.global_context);
        };
        let Some(ref map) = coll.context else {
            return Ok(cfg.global_context);
        };

        let norm = if file_path.starts_with('/') {
            file_path.to_string()
        } else {
            format!("/{file_path}")
        };

        let best = map
            .iter()
            .filter(|(prefix, _)| {
                let p = if prefix.starts_with('/') {
                    (*prefix).clone()
                } else {
                    format!("/{prefix}")
                };
                norm.starts_with(&p)
            })
            .max_by_key(|(prefix, _)| prefix.len());

        match best {
            Some((_, ctx)) => Ok(Some(ctx.clone())),
            None => Ok(cfg.global_context),
        }
    }
}

/// Validate a collection name (alphanumeric, `-`, `_`).
#[must_use]
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}
