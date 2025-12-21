//! Embedding orchestrator service.
//!
//! Manages multiple embedding clients and provides routing based on source type.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::{Config, FusionConfig, RoutingRule};
use crate::clients::{OpenAIClient, CohereClient, VoyageClient, JinaClient, HuggingFaceClient};
use crate::services::{EmbeddingCache, VectorOps};
use crate::traits::{
    EmbeddingClient, EmbeddingRequest, EmbeddingResponse,
    BatchEmbeddingRequest, BatchEmbeddingResponse,
    RerankClient, RerankRequest, RerankResponse,
};

/// The main orchestrator that manages embedding clients and routing.
pub struct EmbeddingOrchestrator {
    clients: HashMap<String, Arc<dyn EmbeddingClient>>,
    rerank_clients: HashMap<String, Arc<dyn RerankClient>>,
    fusion_config: Option<FusionConfig>,
    cache: EmbeddingCache,
    config: Config,
}

impl EmbeddingOrchestrator {
    /// Create a new embedding orchestrator.
    pub async fn new(config: &Config) -> Result<Self> {
        let mut clients: HashMap<String, Arc<dyn EmbeddingClient>> = HashMap::new();
        let mut rerank_clients: HashMap<String, Arc<dyn RerankClient>> = HashMap::new();

        // Initialize OpenAI client
        if let Some(ref api_key) = config.openai_api_key {
            let client = OpenAIClient::new(api_key.clone());
            if client.is_available() {
                info!("✓ OpenAI client initialized");
                clients.insert("openai".to_string(), Arc::new(client));
            }
        }

        // Initialize Cohere client
        if let Some(ref api_key) = config.cohere_api_key {
            let client = CohereClient::new(api_key.clone());
            if client.is_available() {
                info!("✓ Cohere client initialized");
                clients.insert("cohere".to_string(), Arc::new(client));
            }
        }

        // Initialize Voyage client
        if let Some(ref api_key) = config.voyage_api_key {
            let client = VoyageClient::new(api_key.clone());
            if client.is_available() {
                info!("✓ Voyage client initialized");
                clients.insert("voyage".to_string(), Arc::new(client));
            }
        }

        // Initialize Jina client (also supports reranking)
        if let Some(ref api_key) = config.jina_api_key {
            let client = JinaClient::new(api_key.clone());
            // Use EmbeddingClient trait explicitly to disambiguate
            if EmbeddingClient::is_available(&client) {
                info!("✓ Jina client initialized");
                let client_arc: Arc<JinaClient> = Arc::new(client);
                clients.insert("jina".to_string(), client_arc.clone());
                rerank_clients.insert("jina".to_string(), client_arc);
            }
        }


        // Initialize HuggingFace client
        if let Some(ref api_token) = config.huggingface_api_token {
            let client = HuggingFaceClient::new(api_token.clone());
            if client.is_available() {
                info!("✓ HuggingFace client initialized");
                clients.insert("huggingface".to_string(), Arc::new(client));
            }
        }

        if clients.is_empty() {
            return Err(anyhow!(
                "No embedding clients could be initialized. Please set at least one API key."
            ));
        }

        // Load fusion config
        let fusion_config = match FusionConfig::from_file(&config.fusion_config_path) {
            Ok(fc) => {
                info!("✓ Loaded fusion config from {}", config.fusion_config_path);
                Some(fc)
            }
            Err(e) => {
                warn!("Could not load fusion config: {}. Using defaults.", e);
                None
            }
        };

        let cache = EmbeddingCache::new(config.cache_size);

        Ok(Self {
            clients,
            rerank_clients,
            fusion_config,
            cache,
            config: config.clone(),
        })
    }

    /// Get list of available providers.
    pub fn available_providers(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Get the default client based on configuration.
    fn get_default_client(&self) -> Result<&Arc<dyn EmbeddingClient>> {
        // Try to get the configured default provider
        if let Some(client) = self.clients.get(&self.config.default_provider) {
            return Ok(client);
        }

        // Fallback to first available client
        self.clients
            .values()
            .next()
            .ok_or_else(|| anyhow!("No embedding clients available"))
    }

    /// Get a client by provider name.
    fn get_client(&self, provider: &str) -> Result<&Arc<dyn EmbeddingClient>> {
        self.clients
            .get(provider)
            .ok_or_else(|| anyhow!("Provider '{}' not available", provider))
    }

    /// Get client for a source type based on routing configuration.
    fn get_client_for_source(&self, source_type: &str) -> Result<(&Arc<dyn EmbeddingClient>, Option<&RoutingRule>)> {
        if let Some(ref fusion_config) = self.fusion_config {
            if let Some(routing) = fusion_config.get_routing_for_source(source_type) {
                // Get the first model in the routing rule
                if let Some(model_name) = routing.models.first() {
                    if let Some(model_config) = fusion_config.get_model_config(model_name) {
                        if let Some(client) = self.clients.get(&model_config.client) {
                            return Ok((client, Some(routing)));
                        }
                    }
                }
            }

            // Try fallback model
            if let Some(fallback) = fusion_config.get_fallback_model() {
                if let Some(client) = self.clients.get(&fallback.client) {
                    return Ok((client, None));
                }
            }
        }

        // Use default client
        Ok((self.get_default_client()?, None))
    }

    /// Generate embedding for a single text.
    pub async fn embed(
        &self,
        text: String,
        source_type: Option<&str>,
        provider: Option<&str>,
        model: Option<String>,
    ) -> Result<EmbeddingResponse> {
        // Determine which client to use
        let client = if let Some(p) = provider {
            self.get_client(p)?
        } else if let Some(st) = source_type {
            self.get_client_for_source(st)?.0
        } else {
            self.get_default_client()?
        };

        let model_to_use = model.clone().unwrap_or_else(|| client.default_model().to_string());

        // Check cache
        let cache_key = EmbeddingCache::generate_key(&text, &model_to_use);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(EmbeddingResponse {
                embedding: cached.clone(),
                dimension: cached.len() as u32,
                model: model_to_use,
                usage: None,
            });
        }

        // Generate embedding
        let request = EmbeddingRequest {
            text,
            model,
            output_dimension: None,
            task_type: None,
        };

        let response = client.embed(request).await?;

        // Cache the result
        self.cache.insert(cache_key, response.embedding.clone());

        Ok(response)
    }

    /// Generate embeddings for multiple texts.
    pub async fn embed_batch(
        &self,
        texts: Vec<String>,
        source_type: Option<&str>,
        provider: Option<&str>,
        model: Option<String>,
    ) -> Result<BatchEmbeddingResponse> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResponse {
                embeddings: vec![],
                dimension: 0,
                model: String::new(),
                usage: None,
            });
        }

        // Determine which client to use
        let client = if let Some(p) = provider {
            self.get_client(p)?
        } else if let Some(st) = source_type {
            self.get_client_for_source(st)?.0
        } else {
            self.get_default_client()?
        };

        let model_to_use = model.clone().unwrap_or_else(|| client.default_model().to_string());

        // Check cache and separate cached/uncached
        let mut cached_embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut uncached_texts: Vec<(usize, String)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let cache_key = EmbeddingCache::generate_key(text, &model_to_use);
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
            let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(0);
            return Ok(BatchEmbeddingResponse {
                embeddings,
                dimension: dim,
                model: model_to_use,
                usage: None,
            });
        }

        // Generate embeddings for uncached texts
        let request = BatchEmbeddingRequest {
            texts: uncached_texts.iter().map(|(_, t)| t.clone()).collect(),
            model: model.clone(),
            output_dimension: None,
            task_type: None,
        };

        let response = client.embed_batch(request).await?;

        // Cache new embeddings
        for ((_, text), embedding) in uncached_texts.iter().zip(response.embeddings.iter()) {
            let cache_key = EmbeddingCache::generate_key(text, &model_to_use);
            self.cache.insert(cache_key, embedding.clone());
        }

        // Combine cached and new embeddings in correct order
        let mut all_embeddings: Vec<(usize, Vec<f32>)> = cached_embeddings;
        for ((i, _), embedding) in uncached_texts.into_iter().zip(response.embeddings.into_iter()) {
            all_embeddings.push((i, embedding));
        }
        all_embeddings.sort_by_key(|(i, _)| *i);

        let embeddings: Vec<Vec<f32>> = all_embeddings.into_iter().map(|(_, e)| e).collect();
        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(0);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model_to_use,
            usage: response.usage,
        })
    }

    /// Generate embeddings with fusion from multiple models.
    pub async fn embed_with_fusion(
        &self,
        texts: Vec<String>,
        source_type: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let fusion_config = self.fusion_config.as_ref()
            .ok_or_else(|| anyhow!("Fusion config not loaded"))?;

        let routing = fusion_config.get_routing_for_source(source_type);

        if routing.is_none() {
            // No specific routing, use default
            let response = self.embed_batch(texts, Some(source_type), None, None).await?;
            return Ok(response.embeddings);
        }

        let routing = routing.unwrap();

        if routing.models.len() == 1 || routing.fusion_strategy == "single" {
            // Single model, no fusion needed
            let model_name = &routing.models[0];
            if let Some(model_config) = fusion_config.get_model_config(model_name) {
                let response = self.embed_batch(
                    texts,
                    None,
                    Some(&model_config.client),
                    Some(model_config.model.clone()),
                ).await?;
                return Ok(response.embeddings);
            }
        }

        // Multi-model fusion
        let mut all_model_embeddings: Vec<Vec<Vec<f32>>> = Vec::new();

        for model_name in &routing.models {
            if let Some(model_config) = fusion_config.get_model_config(model_name) {
                if let Some(client) = self.clients.get(&model_config.client) {
                    let request = BatchEmbeddingRequest {
                        texts: texts.clone(),
                        model: Some(model_config.model.clone()),
                        output_dimension: Some(model_config.dimension),
                        task_type: None,
                    };

                    match client.embed_batch(request).await {
                        Ok(response) => {
                            all_model_embeddings.push(response.embeddings);
                        }
                        Err(e) => {
                            warn!("Failed to get embeddings from {}: {}", model_name, e);
                        }
                    }
                }
            }
        }

        if all_model_embeddings.is_empty() {
            return Err(anyhow!("All models failed to generate embeddings"));
        }

        // Apply fusion strategy
        let fused = self.apply_fusion_strategy(
            &all_model_embeddings,
            &routing.weights,
            &routing.fusion_strategy,
            texts.len(),
        )?;

        Ok(fused)
    }

    /// Apply fusion strategy to combine embeddings from multiple models.
    fn apply_fusion_strategy(
        &self,
        all_embeddings: &[Vec<Vec<f32>>],
        weights: &[f32],
        strategy: &str,
        num_texts: usize,
    ) -> Result<Vec<Vec<f32>>> {
        match strategy {
            "weighted_average" => {
                let mut result = Vec::new();
                for text_idx in 0..num_texts {
                    let text_embeddings: Vec<Vec<f32>> = all_embeddings
                        .iter()
                        .filter_map(|model_embs| model_embs.get(text_idx).cloned())
                        .collect();

                    if let Some(fused) = VectorOps::weighted_average(&text_embeddings, weights) {
                        result.push(fused);
                    } else {
                        // Fallback to first available
                        if let Some(first) = text_embeddings.into_iter().next() {
                            result.push(first);
                        }
                    }
                }
                Ok(result)
            }
            "concatenate" => {
                let mut result = Vec::new();
                for text_idx in 0..num_texts {
                    let text_embeddings: Vec<Vec<f32>> = all_embeddings
                        .iter()
                        .filter_map(|model_embs| model_embs.get(text_idx).cloned())
                        .collect();

                    let concatenated = VectorOps::concatenate(&text_embeddings);
                    result.push(concatenated);
                }
                Ok(result)
            }
            "max_pooling" => {
                let mut result = Vec::new();
                for text_idx in 0..num_texts {
                    let text_embeddings: Vec<Vec<f32>> = all_embeddings
                        .iter()
                        .filter_map(|model_embs| model_embs.get(text_idx).cloned())
                        .collect();

                    if let Some(pooled) = VectorOps::max_pool(&text_embeddings) {
                        result.push(pooled);
                    } else if let Some(first) = text_embeddings.into_iter().next() {
                        result.push(first);
                    }
                }
                Ok(result)
            }
            _ => {
                warn!("Unknown fusion strategy '{}', using first model", strategy);
                Ok(all_embeddings
                    .first()
                    .cloned()
                    .unwrap_or_default())
            }
        }
    }

    /// Rerank documents by relevance to a query.
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> {
        // Try to find a rerank client
        let client = self.rerank_clients
            .values()
            .next()
            .ok_or_else(|| anyhow!("No reranking clients available. Jina API key required for reranking."))?;

        client.rerank(request).await
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
