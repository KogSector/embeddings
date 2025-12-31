//! Embedding orchestrator service.
//!
//! Manages the local embedding model and provides embedding generation.
//! Simplified from multi-provider architecture to single local model.

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::clients::{LocalEmbeddingClient, LocalModelConfig};
use crate::config::Config;
use crate::services::EmbeddingCache;
use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse,
};

/// The main orchestrator that manages the local embedding model.
pub struct EmbeddingOrchestrator {
    client: Arc<LocalEmbeddingClient>,
    cache: EmbeddingCache,
    config: Config,
}

impl EmbeddingOrchestrator {
    /// Create a new embedding orchestrator with local model.
    pub async fn new(config: &Config) -> Result<Self> {
        info!("Initializing embedding orchestrator with local model");

        // Validate model files exist
        if let Err(e) = config.validate_model_files() {
            return Err(anyhow::anyhow!(e));
        }

        // Initialize local embedding client
        let local_config = config.to_local_model_config();
        let client = LocalEmbeddingClient::new(local_config)?;

        info!(
            "✓ Local embedding model initialized: {} ({}D)",
            config.model_name, config.model_dimension
        );

        let cache = EmbeddingCache::new(config.cache_size);

        Ok(Self {
            client: Arc::new(client),
            cache,
            config: config.clone(),
        })
    }

    /// Get list of available providers (always "local").
    pub fn available_providers(&self) -> Vec<String> {
        vec!["local".to_string()]
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> u32 {
        self.config.model_dimension
    }

    /// Generate embedding for a single text.
    pub async fn embed(
        &self,
        text: String,
        _source_type: Option<&str>,
        _provider: Option<&str>,
        _model: Option<String>,
    ) -> Result<EmbeddingResponse> {
        // Check cache first
        let cache_key = EmbeddingCache::generate_key(&text, &self.config.model_name);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(EmbeddingResponse {
                embedding: cached.clone(),
                dimension: self.config.model_dimension,
                model: self.config.model_name.clone(),
                usage: None,
            });
        }

        // Generate embedding
        let request = EmbeddingRequest {
            text,
            model: None,
            output_dimension: None,
            task_type: None,
        };

        let response = self.client.embed(request).await?;

        // Cache the result
        self.cache.insert(cache_key, response.embedding.clone());

        Ok(response)
    }

    /// Generate embeddings for multiple texts.
    pub async fn embed_batch(
        &self,
        texts: Vec<String>,
        _source_type: Option<&str>,
        _provider: Option<&str>,
        _model: Option<String>,
    ) -> Result<BatchEmbeddingResponse> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResponse {
                embeddings: vec![],
                dimension: self.config.model_dimension,
                model: self.config.model_name.clone(),
                usage: None,
            });
        }

        // Check cache and separate cached/uncached
        let mut cached_embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut uncached_texts: Vec<(usize, String)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let cache_key = EmbeddingCache::generate_key(text, &self.config.model_name);
            if let Some(cached) = self.cache.get(&cache_key) {
                cached_embeddings.push((i, cached));
            } else {
                uncached_texts.push((i, text.clone()));
            }
        }

        // If all are cached, return immediately
        if uncached_texts.is_empty() {
            cached_embeddings.sort_by_key(|(i, _)| *i);
            let embeddings: Vec<Vec<f32>> = cached_embeddings.into_iter().map(|(_, e)| e).collect();
            return Ok(BatchEmbeddingResponse {
                embeddings,
                dimension: self.config.model_dimension,
                model: self.config.model_name.clone(),
                usage: None,
            });
        }

        // Generate embeddings for uncached texts
        let request = BatchEmbeddingRequest {
            texts: uncached_texts.iter().map(|(_, t)| t.clone()).collect(),
            model: None,
            output_dimension: None,
            task_type: None,
        };

        let response = self.client.embed_batch(request).await?;

        // Cache new embeddings
        for ((_, text), embedding) in uncached_texts.iter().zip(response.embeddings.iter()) {
            let cache_key = EmbeddingCache::generate_key(text, &self.config.model_name);
            self.cache.insert(cache_key, embedding.clone());
        }

        // Combine cached and new embeddings in correct order
        let mut all_embeddings: Vec<(usize, Vec<f32>)> = cached_embeddings;
        for ((i, _), embedding) in uncached_texts.into_iter().zip(response.embeddings.into_iter()) {
            all_embeddings.push((i, embedding));
        }
        all_embeddings.sort_by_key(|(i, _)| *i);

        let embeddings: Vec<Vec<f32>> = all_embeddings.into_iter().map(|(_, e)| e).collect();

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: self.config.model_dimension,
            model: self.config.model_name.clone(),
            usage: response.usage,
        })
    }

    /// Generate embeddings (simplified version, same as embed_batch).
    pub async fn embed_with_fusion(
        &self,
        texts: Vec<String>,
        source_type: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let response = self.embed_batch(texts, Some(source_type), None, None).await?;
        Ok(response.embeddings)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        self.cache.stats()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}
