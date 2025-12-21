//! Configuration module for the embeddings service.

use serde::{Deserialize, Serialize};
use std::fs;
use anyhow::Result;

/// Main service configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub default_provider: String,
    pub default_model: String,
    pub fusion_config_path: String,
    pub cache_size: usize,
    pub max_concurrency: usize,
    pub batch_size: usize,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    // API Keys
    pub openai_api_key: Option<String>,
    pub cohere_api_key: Option<String>,
    pub voyage_api_key: Option<String>,
    pub jina_api_key: Option<String>,
    pub huggingface_api_token: Option<String>,
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
            default_provider: std::env::var("DEFAULT_PROVIDER")
                .unwrap_or_else(|_| "openai".to_string()),
            default_model: std::env::var("DEFAULT_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string()),
            fusion_config_path: std::env::var("FUSION_CONFIG_PATH")
                .unwrap_or_else(|_| "config/fusion_config.json".to_string()),
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
            max_retries: std::env::var("MAX_RETRIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            cohere_api_key: std::env::var("COHERE_API_KEY").ok(),
            voyage_api_key: std::env::var("VOYAGE_API_KEY").ok(),
            jina_api_key: std::env::var("JINA_API_KEY").ok(),
            huggingface_api_token: std::env::var("HUGGINGFACE_API_TOKEN").ok(),
        }
    }

    /// Check if at least one provider is configured.
    pub fn has_any_provider(&self) -> bool {
        self.openai_api_key.is_some()
            || self.cohere_api_key.is_some()
            || self.voyage_api_key.is_some()
            || self.jina_api_key.is_some()
            || self.huggingface_api_token.is_some()
    }

    /// Get list of configured providers.
    pub fn configured_providers(&self) -> Vec<&str> {
        let mut providers = Vec::new();
        if self.openai_api_key.is_some() {
            providers.push("openai");
        }
        if self.cohere_api_key.is_some() {
            providers.push("cohere");
        }
        if self.voyage_api_key.is_some() {
            providers.push("voyage");
        }
        if self.jina_api_key.is_some() {
            providers.push("jina");
        }
        if self.huggingface_api_token.is_some() {
            providers.push("huggingface");
        }
        providers
    }
}

/// Fusion configuration loaded from JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub models: Vec<ModelConfig>,
    pub routing: Vec<RoutingRule>,
    pub fallback_model: String,
    pub cache_embeddings: bool,
    pub normalize_embeddings: bool,
    pub batch_size: usize,
    pub max_retries: u32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub client: String,
    pub model: String,
    pub dimension: u32,
    pub strengths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    pub source: String,
    pub models: Vec<String>,
    pub weights: Vec<f32>,
    pub fusion_strategy: String,
}

impl FusionConfig {
    /// Load fusion configuration from a JSON file.
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: FusionConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Get routing rule for a given source type.
    pub fn get_routing_for_source(&self, source: &str) -> Option<&RoutingRule> {
        self.routing.iter().find(|r| r.source == source)
    }

    /// Get model configuration by name.
    pub fn get_model_config(&self, name: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Get the fallback model configuration.
    pub fn get_fallback_model(&self) -> Option<&ModelConfig> {
        self.get_model_config(&self.fallback_model)
    }
}
