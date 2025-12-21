//! Embedding cache with LFU eviction.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use sha2::{Sha256, Digest};

/// A thread-safe LFU-based embedding cache.
#[derive(Clone)]
pub struct EmbeddingCache {
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    access_count: Arc<RwLock<HashMap<String, u64>>>,
    max_size: usize,
}

impl EmbeddingCache {
    /// Create a new cache with the specified maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_count: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Generate a cache key from text and model.
    pub fn generate_key(text: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}", model, text));
        format!("{:x}", hasher.finalize())
    }

    /// Get an embedding from the cache.
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read().unwrap();
        if let Some(embedding) = cache.get(key) {
            // Update access count
            let mut access = self.access_count.write().unwrap();
            *access.entry(key.to_string()).or_insert(0) += 1;
            Some(embedding.clone())
        } else {
            None
        }
    }

    /// Insert an embedding into the cache.
    pub fn insert(&self, key: String, value: Vec<f32>) {
        let mut cache = self.cache.write().unwrap();

        // Evict if cache is full
        if cache.len() >= self.max_size {
            self.evict_lfu(&mut cache);
        }

        cache.insert(key.clone(), value);
        let mut access = self.access_count.write().unwrap();
        access.insert(key, 1);
    }

    /// Evict the least frequently used item.
    fn evict_lfu(&self, cache: &mut HashMap<String, Vec<f32>>) {
        let access = self.access_count.read().unwrap();
        if let Some((lfu_key, _)) = access.iter().min_by_key(|(_, &count)| count) {
            let lfu_key = lfu_key.clone();
            drop(access);
            cache.remove(&lfu_key);
            let mut access = self.access_count.write().unwrap();
            access.remove(&lfu_key);
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (usize, usize) {
        let cache = self.cache.read().unwrap();
        (cache.len(), self.max_size)
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut access = self.access_count.write().unwrap();
        cache.clear();
        access.clear();
    }

    /// Check if an entry exists in the cache.
    pub fn contains(&self, key: &str) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = EmbeddingCache::new(2);
        
        let key1 = EmbeddingCache::generate_key("hello", "model1");
        let key2 = EmbeddingCache::generate_key("world", "model1");
        
        cache.insert(key1.clone(), vec![1.0, 2.0, 3.0]);
        cache.insert(key2.clone(), vec![4.0, 5.0, 6.0]);
        
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());
        
        let (size, max) = cache.stats();
        assert_eq!(size, 2);
        assert_eq!(max, 2);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = EmbeddingCache::new(2);
        
        let key1 = EmbeddingCache::generate_key("a", "model");
        let key2 = EmbeddingCache::generate_key("b", "model");
        let key3 = EmbeddingCache::generate_key("c", "model");
        
        cache.insert(key1.clone(), vec![1.0]);
        cache.insert(key2.clone(), vec![2.0]);
        
        // Access key1 to increase its count
        cache.get(&key1);
        cache.get(&key1);
        
        // Insert key3, should evict key2 (lower access count)
        cache.insert(key3.clone(), vec![3.0]);
        
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key3).is_some());
        // key2 should be evicted
        assert!(cache.get(&key2).is_none());
    }
}
