use std::collections::HashMap;
use crate::types::*;

/// Hints returned to the JIT compiler to guide recompilation.
#[derive(Debug, Clone, PartialEq)]
pub enum RecompileHint {
    /// High branch divergence detected — use aggressive predication.
    AggressivePredication,
    /// Low cache hit rate — enable Structure-of-Arrays transposition.
    EnableSoaTransposition,
    /// Register spills detected — reduce register pressure.
    ReduceRegisterPressure { max_registers: u32 },
    /// Low occupancy — limit register count to raise active warps.
    LimitRegisters { max_registers: u32 },
}

/// Adaptive JIT profiler and recompilation manager.
pub struct AdaptiveJit {
    profiles: HashMap<[u8; 32], ProgramProfile>,
    warmup_threshold: u64,
    recompile_interval: u64,
}

impl AdaptiveJit {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            warmup_threshold: 100,
            recompile_interval: 10_000,
        }
    }

    /// Record execution metrics for a program.
    pub fn record_execution(
        &mut self,
        program_hash: &[u8; 32],
        gpu_time_us: u64,
        occupancy: f64,
        divergence_rate: f64,
        cache_hit_rate: f64,
        cu_exceeded: bool,
    ) {
        let profile = self.profiles.entry(*program_hash).or_default();

        profile.execution_count += 1;
        profile.total_gpu_time_us += gpu_time_us;

        // Exponential moving average for occupancy (alpha = 0.1).
        const ALPHA: f64 = 0.1;
        if profile.execution_count == 1 {
            profile.avg_occupancy = occupancy;
            profile.divergence_rate = divergence_rate;
            profile.cache_hit_rate = cache_hit_rate;
        } else {
            profile.avg_occupancy =
                (1.0 - ALPHA) * profile.avg_occupancy + ALPHA * occupancy;
            profile.divergence_rate =
                (1.0 - ALPHA) * profile.divergence_rate + ALPHA * divergence_rate;
            profile.cache_hit_rate =
                (1.0 - ALPHA) * profile.cache_hit_rate + ALPHA * cache_hit_rate;
        }

        if cu_exceeded {
            profile.cu_exceeded_count += 1;
        }
    }

    /// Check if a program should be recompiled with different hints.
    ///
    /// Returns the highest-priority `RecompileHint` if conditions are met,
    /// or `None` if the program is performing within acceptable bounds.
    pub fn should_recompile(&self, program_hash: &[u8; 32]) -> Option<RecompileHint> {
        let profile = self.profiles.get(program_hash)?;

        // Must have passed the warmup threshold before considering recompile.
        if profile.execution_count < self.warmup_threshold {
            return None;
        }

        // Check conditions in priority order.

        // 1. Register spills → most urgent, fix first.
        if profile.register_spills > 0 {
            return Some(RecompileHint::ReduceRegisterPressure { max_registers: 32 });
        }

        // 2. High divergence → aggressive predication.
        if profile.execution_count > 1_000 && profile.divergence_rate > 0.5 {
            return Some(RecompileHint::AggressivePredication);
        }

        // 3. Low cache hit rate → enable SoA transposition.
        if profile.execution_count > 1_000 && profile.cache_hit_rate < 0.3 {
            return Some(RecompileHint::EnableSoaTransposition);
        }

        // 4. Low occupancy → limit register count to increase active warps.
        if profile.avg_occupancy < 0.25 {
            return Some(RecompileHint::LimitRegisters { max_registers: 48 });
        }

        None
    }

    /// Get profile for a program.
    pub fn get_profile(&self, program_hash: &[u8; 32]) -> Option<&ProgramProfile> {
        self.profiles.get(program_hash)
    }

    /// Detect whether recent invocations consistently use the same first argument.
    ///
    /// If more than 90% of `recent_args` are identical, returns that argument
    /// value so the compiler can specialize for it (e.g., a fixed token mint).
    pub fn detect_constant_args(
        &self,
        program_hash: &[u8; 32],
        recent_args: &[[u8; 32]],
    ) -> Option<[u8; 32]> {
        // Program must exist and have warmed up.
        let profile = self.profiles.get(program_hash)?;
        if profile.execution_count < self.warmup_threshold {
            return None;
        }

        if recent_args.is_empty() {
            return None;
        }

        // Count occurrences of each distinct argument value.
        let mut counts: HashMap<[u8; 32], usize> = HashMap::new();
        for arg in recent_args {
            *counts.entry(*arg).or_insert(0) += 1;
        }

        // Find the most frequent argument.
        let (most_common, &freq) = counts
            .iter()
            .max_by_key(|(_, &cnt)| cnt)?;

        // Specialise only when frequency exceeds the 90% threshold.
        let ratio = freq as f64 / recent_args.len() as f64;
        if ratio > 0.9 {
            Some(*most_common)
        } else {
            None
        }
    }

    /// Inform the profiler that a program has been recompiled; resets the
    /// execution counter so the new version goes through the warmup phase.
    pub fn on_recompile(&mut self, program_hash: &[u8; 32]) {
        if let Some(profile) = self.profiles.get_mut(program_hash) {
            profile.execution_count = 0;
            profile.register_spills = 0;
        }
    }

    /// Record register spill events for a program (can be called from the
    /// CUDA runtime via the profiling counter).
    pub fn record_register_spills(&mut self, program_hash: &[u8; 32], spill_count: u32) {
        let profile = self.profiles.entry(*program_hash).or_default();
        profile.register_spills = profile.register_spills.saturating_add(spill_count);
    }

    /// Return true if the program has completed the warmup phase and is a
    /// candidate for profiling-driven optimizations.
    pub fn is_warmed_up(&self, program_hash: &[u8; 32]) -> bool {
        self.profiles
            .get(program_hash)
            .map(|p| p.execution_count >= self.warmup_threshold)
            .unwrap_or(false)
    }

    /// Return the average GPU time (µs) for a program across all executions.
    pub fn avg_gpu_time_us(&self, program_hash: &[u8; 32]) -> Option<u64> {
        let profile = self.profiles.get(program_hash)?;
        if profile.execution_count == 0 {
            return None;
        }
        Some(profile.total_gpu_time_us / profile.execution_count)
    }

    /// Return whether a program should be scheduled for recompilation based
    /// on the recompile interval (every `recompile_interval` executions).
    pub fn is_recompile_due(&self, program_hash: &[u8; 32]) -> bool {
        self.profiles
            .get(program_hash)
            .map(|p| {
                p.execution_count > 0
                    && p.execution_count % self.recompile_interval == 0
            })
            .unwrap_or(false)
    }
}

impl Default for AdaptiveJit {
    fn default() -> Self {
        Self::new()
    }
}
