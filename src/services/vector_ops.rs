//! Vector operations with SIMD-optimized parallel processing.

use rayon::prelude::*;

/// Utility struct for vector operations.
pub struct VectorOps;

impl VectorOps {
    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.par_iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.par_iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Compute dot product between two vectors.
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute L2 (Euclidean) distance between two vectors.
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize a vector in-place to unit length.
    pub fn normalize_inplace(vector: &mut [f32]) {
        let norm: f32 = vector.par_iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.par_iter_mut().for_each(|x| *x /= norm);
        }
    }

    /// Normalize a vector and return a new vector.
    pub fn normalize(vector: &[f32]) -> Vec<f32> {
        let norm: f32 = vector.par_iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.par_iter().map(|x| x / norm).collect()
        } else {
            vector.to_vec()
        }
    }

    /// Batch normalize vectors in-place.
    pub fn batch_normalize(vectors: &mut [Vec<f32>]) {
        vectors.par_iter_mut().for_each(|vector| {
            Self::normalize_inplace(vector);
        });
    }

    /// Compute weighted average of multiple embeddings.
    pub fn weighted_average(embeddings: &[Vec<f32>], weights: &[f32]) -> Option<Vec<f32>> {
        if embeddings.is_empty() || weights.is_empty() {
            return None;
        }

        let dim = embeddings[0].len();
        if embeddings.iter().any(|e| e.len() != dim) {
            return None;
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight == 0.0 {
            return None;
        }

        let mut result = vec![0.0f32; dim];

        for (embedding, &weight) in embeddings.iter().zip(weights.iter()) {
            if weight > 0.0 {
                for (i, &val) in embedding.iter().enumerate() {
                    result[i] += val * weight;
                }
            }
        }

        for val in &mut result {
            *val /= total_weight;
        }

        Some(result)
    }

    /// Concatenate multiple embeddings into a single vector.
    pub fn concatenate(embeddings: &[Vec<f32>]) -> Vec<f32> {
        let total_dim: usize = embeddings.iter().map(|e| e.len()).sum();
        let mut result = Vec::with_capacity(total_dim);
        
        for embedding in embeddings {
            result.extend_from_slice(embedding);
        }
        
        result
    }

    /// Max pooling across multiple embeddings.
    pub fn max_pool(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
        if embeddings.is_empty() {
            return None;
        }

        let dim = embeddings[0].len();
        if embeddings.iter().any(|e| e.len() != dim) {
            return None;
        }

        let mut result = vec![f32::MIN; dim];

        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] = result[i].max(val);
            }
        }

        Some(result)
    }

    /// Resize embedding dimension using interpolation.
    pub fn resize_dimension(embedding: &[f32], target_dim: usize) -> Vec<f32> {
        let source_dim = embedding.len();
        if source_dim == target_dim {
            return embedding.to_vec();
        }

        let mut resized = vec![0.0f32; target_dim];
        let ratio = source_dim as f32 / target_dim as f32;

        for i in 0..target_dim {
            let source_idx = (i as f32 * ratio) as usize;
            resized[i] = embedding.get(source_idx).copied().unwrap_or(0.0);
        }

        resized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((VectorOps::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((VectorOps::cosine_similarity(&a, &c)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = VectorOps::normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let weights = vec![0.5, 0.5];
        let result = VectorOps::weighted_average(&embeddings, &weights).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_concatenate() {
        let embeddings = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let result = VectorOps::concatenate(&embeddings);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
