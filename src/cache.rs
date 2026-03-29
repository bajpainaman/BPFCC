use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::types::BpfCompileResult;

/// Thread-safe cache for compiled BPF programs.
pub struct CompilationCache {
    cache: RwLock<HashMap<[u8; 32], Arc<BpfCompileResult>>>,
}

impl CompilationCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get a cached compilation result.
    pub fn get(&self, program_hash: &[u8; 32]) -> Option<Arc<BpfCompileResult>> {
        self.cache.read().ok()?.get(program_hash).cloned()
    }

    /// Insert a compilation result.
    pub fn insert(&self, program_hash: [u8; 32], result: BpfCompileResult) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(program_hash, Arc::new(result));
        }
    }

    /// Remove a cached entry (e.g., on program upgrade).
    pub fn invalidate(&self, program_hash: &[u8; 32]) {
        if let Ok(mut cache) = self.cache.write() {
            cache.remove(program_hash);
        }
    }

    /// Mark a program as CPU-only (failed too many GPU executions).
    pub fn mark_cpu_only(&self, program_hash: &[u8; 32]) {
        if let Ok(mut cache) = self.cache.write() {
            if let Some(entry) = cache.get_mut(program_hash) {
                let mut result = (**entry).clone();
                result.gpu_eligible = false;
                result.reject_reason = Some("GPU failure rate exceeded 10%".into());
                *entry = Arc::new(result);
            }
        }
    }

    /// Number of cached programs.
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::new()
    }
}
