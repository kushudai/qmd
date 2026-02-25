//! GGUF model management: download, cache, and resolution.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};

use crate::config;
use crate::error::{Error, Result};

/// A named model with its local filename and remote URI.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ModelSpec {
    /// Local cache filename.
    pub filename: &'static str,
    /// HuggingFace download URI (`hf:user/repo/file`).
    pub uri: &'static str,
}

/// Default embedding model.
pub const EMBED: ModelSpec = ModelSpec {
    filename: "embeddinggemma-300M-Q8_0.gguf",
    uri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
};

/// Default rerank model.
pub const RERANK: ModelSpec = ModelSpec {
    filename: "qwen3-reranker-0.6b-q8_0.gguf",
    uri: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf",
};

/// Default generation model.
pub const GENERATE: ModelSpec = ModelSpec {
    filename: "qmd-query-expansion-1.7B-q4_k_m.gguf",
    uri: "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf",
};

/// Local path of a cached model. Errors if not found.
pub fn get_path(name: &str) -> Result<PathBuf> {
    let path = config::model_cache_dir()?.join(name);
    if !path.exists() {
        return Err(Error::Model(format!("model not found: {}", path.display())));
    }
    Ok(path)
}

/// Whether a model is cached locally.
pub fn exists(name: &str) -> Result<bool> {
    Ok(config::model_cache_dir()?.join(name).exists())
}

/// List cached `.gguf` model filenames.
pub fn list_cached() -> Result<Vec<String>> {
    let dir = config::model_cache_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    Ok(fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default())
}

/// Download or verify a model. Returns `(local_path, freshly_downloaded)`.
///
/// Supports `hf:user/repo/file.gguf` URIs and plain filenames.
pub fn pull(uri: &str, refresh: bool) -> Result<(PathBuf, bool)> {
    let cache_dir = config::model_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let hf_ref = parse_hf_uri(uri);
    let filename = hf_ref
        .as_ref()
        .map_or_else(|| uri.to_string(), |hf| hf.file.clone());

    let local_path = cache_dir.join(&filename);
    let etag_path = cache_dir.join(format!("{filename}.etag"));

    let needs_download = refresh
        || !local_path.exists()
        || hf_ref.as_ref().is_some_and(|hf| {
            let remote = remote_etag(hf);
            let local = fs::read_to_string(&etag_path).ok();
            remote.is_some() && remote != local
        });

    if needs_download {
        let hf = hf_ref
            .as_ref()
            .ok_or_else(|| Error::Model(format!("no HuggingFace URI for: {uri}")))?;
        download(hf, &local_path, &etag_path)?;
    }

    Ok((local_path, needs_download))
}

/// Resolve a model URI to a local path, downloading if needed.
pub fn resolve(uri: &str) -> Result<PathBuf> {
    let (path, _) = pull(uri, false)?;
    Ok(path)
}

/// A loaded GGUF model ready for inference.
#[non_exhaustive]
#[allow(missing_debug_implementations)]
pub struct LoadedModel {
    /// Llama backend handle.
    pub backend: llama_cpp_2::llama_backend::LlamaBackend,
    /// Shared model handle.
    pub model: std::sync::Arc<llama_cpp_2::model::LlamaModel>,
}

/// Load a GGUF model from disk.
pub fn load_gguf(path: &Path) -> Result<LoadedModel> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let backend = LlamaBackend::init().map_err(|e| Error::Model(format!("backend init: {e}")))?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, path, &params)
        .map_err(|e| Error::Model(format!("load {}: {e}", path.display())))?;
    Ok(LoadedModel {
        backend,
        model: std::sync::Arc::new(model),
    })
}

/// Parsed HuggingFace model reference.
#[derive(Debug, Clone)]
struct HfRef {
    /// Repository in `user/repo` format.
    repo: String,
    /// Filename within the repository.
    file: String,
}

/// Parse `hf:user/repo/file.gguf` into components.
fn parse_hf_uri(uri: &str) -> Option<HfRef> {
    let rest = uri.strip_prefix("hf:")?;
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() < 3 {
        return None;
    }
    Some(HfRef {
        repo: format!("{}/{}", parts[0], parts[1]),
        file: parts[2].to_string(),
    })
}

/// Build the HuggingFace download URL.
fn hf_url(hf: &HfRef) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", hf.repo, hf.file)
}

/// Fetch remote ETag for cache validation.
fn remote_etag(hf: &HfRef) -> Option<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client.head(hf_url(hf)).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
}

/// Download a model file with progress bar.
fn download(hf: &HfRef, local_path: &Path, etag_path: &Path) -> Result<()> {
    let url = hf_url(hf);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_hours(1))
        .build()?;

    let mut resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        return Err(Error::Model(format!("HTTP {} for {url}", resp.status())));
    }

    let pb = ProgressBar::new(resp.content_length().unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .map_err(|e| Error::Model(format!("progress template: {e}")))?
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {}", hf.file));

    let mut file = File::create(local_path)?;
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 8192];

    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message(format!("Downloaded {}", hf.file));

    if let Some(etag) = resp.headers().get("etag")
        && let Ok(s) = etag.to_str()
    {
        let _ = fs::write(etag_path, s.trim_matches('"'));
    }

    Ok(())
}
