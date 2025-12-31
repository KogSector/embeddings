//! Configuration module for the embeddings service.
//!
//! Provides configuration for the local embedding model.

use serde::{Deserialize, Serialize};

/// Main service configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// HTTP server port.
    pub port: u16,
    /// HTTP server host.
    pub host: String,
    
    // Local Model Configuration
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Path to the tokenizer.json file.
    pub tokenizer_path: String,
    /// Model name for identification.
    pub model_name: String,
    /// Output embedding dimension.
    pub model_dimension: u32,
    /// Maximum sequence length for tokenization.
    pub max_sequence_length: usize,
    
    // Performance Configuration
    /// Size of the embedding cache (number of entries).
    pub cache_size: usize,
    /// Maximum concurrent embedding requests.
    pub max_concurrency: usize,
    /// Batch size for batch embedding operations.
    pub batch_size: usize,
    /// Request timeout in seconds.
    pub timeout_seconds: u64,
    
    // Feature Flags
    /// Whether to normalize embeddings to unit vectors.
    pub normalize_embeddings: bool,
}

impl Config {
    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(3021),
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            
            // Local model configuration
            model_path: std::env::var("MODEL_PATH")
                .unwrap_or_else(|_| "./models/all-MiniLM-L6-v2.onnx".to_string()),
            tokenizer_path: std::env::var("TOKENIZER_PATH")
                .unwrap_or_else(|_| "./models/tokenizer.json".to_string()),
            model_name: std::env::var("MODEL_NAME")
                .unwrap_or_else(|_| "all-MiniLM-L6-v2".to_string()),
            model_dimension: std::env::var("MODEL_DIMENSION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(384),
            max_sequence_length: std::env::var("MAX_SEQUENCE_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            
            // Performance configuration
            cache_size: std::env::var("EMBEDDING_CACHE_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10000),
            max_concurrency: std::env::var("EMBEDDING_MAX_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            batch_size: std::env::var("EMBEDDING_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            timeout_seconds: std::env::var("REQUEST_TIMEOUT_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            
            // Feature flags
            normalize_embeddings: std::env::var("NORMALIZE_EMBEDDINGS")
                .map(|v| v.to_lowercase() == "true" || v == "1")
                .unwrap_or(true),
        }
    }

    /// Check if the model files exist.
    pub fn validate_model_files(&self) -> Result<(), String> {
        use std::path::Path;
        
        if !Path::new(&self.model_path).exists() {
            return Err(format!(
                "Model file not found: {}. Please download the model.",
                self.model_path
            ));
        }
        
        if !Path::new(&self.tokenizer_path).exists() {
            return Err(format!(
                "Tokenizer file not found: {}. Please download the tokenizer.",
                self.tokenizer_path
            ));
        }
        
        Ok(())
    }

    /// Get the local model configuration for the embedding client.
    pub fn to_local_model_config(&self) -> crate::clients::LocalModelConfig {
        crate::clients::LocalModelConfig {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            dimension: self.model_dimension,
            max_length: self.max_sequence_length,
            model_name: self.model_name.clone(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 3021,
            host: "0.0.0.0".to_string(),
            model_path: "./models/all-MiniLM-L6-v2.onnx".to_string(),
            tokenizer_path: "./models/tokenizer.json".to_string(),
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_dimension: 384,
            max_sequence_length: 512,
            cache_size: 10000,
            max_concurrency: 10,
            batch_size: 100,
            timeout_seconds: 30,
            normalize_embeddings: true,
        }
    }
}
