//! Local embedding model client using ONNX Runtime.
//!
//! This module provides a local embedding model that runs entirely on-device
//! without requiring external API calls. Uses ONNX Runtime for efficient inference.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, TokenUsage,
};

/// Configuration for the local embedding model.
#[derive(Debug, Clone)]
pub struct LocalModelConfig {
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Path to the tokenizer.json file.
    pub tokenizer_path: String,
    /// Output embedding dimension.
    pub dimension: u32,
    /// Maximum sequence length.
    pub max_length: usize,
    /// Model name for identification.
    pub model_name: String,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/all-MiniLM-L6-v2.onnx".to_string(),
            tokenizer_path: "./models/tokenizer.json".to_string(),
            dimension: 384,
            max_length: 512,
            model_name: "all-MiniLM-L6-v2".to_string(),
        }
    }
}

/// Local embedding client using ONNX Runtime for inference.
pub struct LocalEmbeddingClient {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    config: LocalModelConfig,
    // Cache for frequently used embeddings
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
}

impl LocalEmbeddingClient {
    /// Create a new local embedding client.
    ///
    /// # Arguments
    /// * `config` - Configuration for the model
    ///
    /// # Returns
    /// A new LocalEmbeddingClient instance
    pub fn new(config: LocalModelConfig) -> Result<Self> {
        info!(
            "Initializing local embedding model: {}",
            config.model_name
        );

        // Check if model file exists
        if !Path::new(&config.model_path).exists() {
            return Err(anyhow!(
                "Model file not found: {}. Please download the model first.",
                config.model_path
            ));
        }

        // Check if tokenizer file exists
        if !Path::new(&config.tokenizer_path).exists() {
            return Err(anyhow!(
                "Tokenizer file not found: {}. Please download the tokenizer first.",
                config.tokenizer_path
            ));
        }

        // Initialize ONNX Runtime session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&config.model_path)?;

        info!("✓ ONNX session created for {}", config.model_name);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        info!("✓ Tokenizer loaded");

        // Initialize LRU cache (10000 entries)
        let cache = lru::LruCache::new(std::num::NonZeroUsize::new(10000).unwrap());

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            config,
            cache: Arc::new(RwLock::new(cache)),
        })
    }

    /// Create with default configuration (all-MiniLM-L6-v2).
    pub fn with_defaults() -> Result<Self> {
        Self::new(LocalModelConfig::default())
    }

    /// Tokenize and encode text for the model.
    fn encode_text(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        // Truncate to max length
        let max_len = self.config.max_length;
        let input_ids = if input_ids.len() > max_len {
            input_ids[..max_len].to_vec()
        } else {
            input_ids
        };
        let attention_mask = if attention_mask.len() > max_len {
            attention_mask[..max_len].to_vec()
        } else {
            attention_mask
        };

        Ok((input_ids, attention_mask))
    }

    /// Run inference on a single text.
    fn run_inference(&self, text: &str) -> Result<Vec<f32>> {
        let (input_ids, attention_mask) = self.encode_text(text)?;
        let seq_len = input_ids.len();

        // Create tensors using ort 2.0 API
        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids.into_boxed_slice()))?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))?;
        let token_type_ids: Vec<i64> = vec![0i64; seq_len];
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids.into_boxed_slice()))?;

        // Run inference using ort::inputs! macro
        let mut session = self.session.lock()
            .map_err(|e| anyhow!("Failed to lock session: {}", e))?;
        let outputs = session.run(ort::inputs![
            input_ids_tensor,
            attention_mask_tensor,
            token_type_ids_tensor,
        ])?;

        // Extract embeddings from first output
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("No output tensor found"))?
            .1;

        let (shape, data) = output.try_extract_tensor::<f32>()?;

        // Shape is typically [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        let embedding = if shape.len() == 3 {
            // [batch, seq_len, hidden_size] - need to pool
            let hidden_size = shape[2] as usize;
            let seq_len = shape[1] as usize;
            
            let mut pooled = vec![0.0f32; hidden_size];
            for i in 0..hidden_size {
                let mut sum = 0.0;
                for j in 0..seq_len {
                    sum += data[j * hidden_size + i];
                }
                pooled[i] = sum / seq_len as f32;
            }
            
            // Normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                pooled.iter_mut().for_each(|x| *x /= norm);
            }
            
            pooled
        } else if shape.len() == 2 {
            // [batch, hidden_size] - already pooled
            let embedding: Vec<f32> = data.to_vec();
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            }
        } else {
            return Err(anyhow!("Unexpected output tensor shape: {:?}", shape));
        };

        Ok(embedding)
    }

    /// Run batch inference.
    fn run_batch_inference(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // For simplicity, run sequentially. Can be parallelized with rayon.
        texts
            .iter()
            .map(|text| self.run_inference(text))
            .collect()
    }

    /// Generate cache key for text.
    fn cache_key(text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[async_trait]
impl EmbeddingClient for LocalEmbeddingClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let cache_key = Self::cache_key(&request.text);

        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.peek(&cache_key) {
                debug!("Cache hit for embedding");
                return Ok(EmbeddingResponse {
                    embedding: cached.clone(),
                    dimension: self.config.dimension,
                    model: self.config.model_name.clone(),
                    usage: None,
                });
            }
        }

        // Generate embedding
        let embedding = self.run_inference(&request.text)?;

        // Store in cache
        {
            let mut cache = self.cache.write().await;
            cache.put(cache_key, embedding.clone());
        }

        Ok(EmbeddingResponse {
            embedding,
            dimension: self.config.dimension,
            model: self.config.model_name.clone(),
            usage: Some(TokenUsage {
                prompt_tokens: request.text.split_whitespace().count(),
                total_tokens: request.text.split_whitespace().count(),
            }),
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        if request.texts.is_empty() {
            return Ok(BatchEmbeddingResponse {
                embeddings: vec![],
                dimension: self.config.dimension,
                model: self.config.model_name.clone(),
                usage: None,
            });
        }

        let mut embeddings = Vec::with_capacity(request.texts.len());
        let mut cached_count = 0;
        let mut texts_to_embed: Vec<(usize, String)> = Vec::new();

        // Check cache for each text
        {
            let cache = self.cache.read().await;
            for (i, text) in request.texts.iter().enumerate() {
                let cache_key = Self::cache_key(text);
                if let Some(cached) = cache.peek(&cache_key) {
                    embeddings.push((i, cached.clone()));
                    cached_count += 1;
                } else {
                    texts_to_embed.push((i, text.clone()));
                }
            }
        }

        debug!(
            "Batch embedding: {} cached, {} to generate",
            cached_count,
            texts_to_embed.len()
        );

        // Generate embeddings for uncached texts
        if !texts_to_embed.is_empty() {
            let texts: Vec<String> = texts_to_embed.iter().map(|(_, t)| t.clone()).collect();
            let new_embeddings = self.run_batch_inference(&texts)?;

            // Store in cache and collect results
            {
                let mut cache = self.cache.write().await;
                for ((i, text), embedding) in texts_to_embed.into_iter().zip(new_embeddings.into_iter()) {
                    let cache_key = Self::cache_key(&text);
                    cache.put(cache_key, embedding.clone());
                    embeddings.push((i, embedding));
                }
            }
        }

        // Sort by original index
        embeddings.sort_by_key(|(i, _)| *i);
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, e)| e).collect();

        let total_tokens: usize = request.texts.iter().map(|t| t.split_whitespace().count()).sum();

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: self.config.dimension,
            model: self.config.model_name.clone(),
            usage: Some(TokenUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            }),
        })
    }

    fn get_dimension(&self, _model: &str) -> Option<u32> {
        Some(self.config.dimension)
    }

    fn provider_name(&self) -> &str {
        "local"
    }

    fn default_model(&self) -> &str {
        &self.config.model_name
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires model files to be present
    async fn test_local_embedding() {
        let client = LocalEmbeddingClient::with_defaults().unwrap();
        
        let response = client
            .embed(EmbeddingRequest {
                text: "Hello, world!".to_string(),
                model: None,
                output_dimension: None,
                task_type: None,
            })
            .await
            .unwrap();

        assert_eq!(response.dimension, 384);
        assert_eq!(response.embedding.len(), 384);
    }

    #[tokio::test]
    #[ignore] // Requires model files to be present
    async fn test_batch_embedding() {
        let client = LocalEmbeddingClient::with_defaults().unwrap();
        
        let response = client
            .embed_batch(BatchEmbeddingRequest {
                texts: vec![
                    "Hello".to_string(),
                    "World".to_string(),
                ],
                model: None,
                output_dimension: None,
                task_type: None,
            })
            .await
            .unwrap();

        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.dimension, 384);
    }
}
