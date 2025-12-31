//! Embedding clients module.
//!
//! This module provides embedding clients for generating vector embeddings.
//! Uses a local ONNX-based model for on-device inference.

pub mod local;

pub use local::{LocalEmbeddingClient, LocalModelConfig};
