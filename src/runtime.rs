use sha2::{Digest, Sha256};
use crate::types::{BpfCompileResult, KernelConfig};
use crate::cache::CompilationCache;
use crate::tier8_adaptive::AdaptiveJit;

/// The main BPFCC compiler — orchestrates the 9-tier pipeline.
pub struct BpfCompiler {
    cache: CompilationCache,
    adaptive: AdaptiveJit,
}

impl BpfCompiler {
    pub fn new() -> Self {
        Self {
            cache: CompilationCache::new(),
            adaptive: AdaptiveJit::new(),
        }
    }

    /// Compile BPF bytecode through the full pipeline.
    /// Returns cached result if available.
    pub fn compile(&mut self, bytecode: &[u8]) -> BpfCompileResult {
        // 1. Hash the bytecode
        let program_hash = Self::hash_bytecode(bytecode);

        // 2. Check cache
        if let Some(cached) = self.cache.get(&program_hash) {
            return (*cached).clone();
        }

        // 3. Run pipeline
        let result = self.run_pipeline(bytecode, program_hash);

        // 4. Cache result
        self.cache.insert(program_hash, result.clone());

        result
    }

    fn run_pipeline(&self, bytecode: &[u8], program_hash: [u8; 32]) -> BpfCompileResult {
        // Tier 0: Static Analysis
        let analysis = match crate::tier0_analysis::analyze(bytecode) {
            Ok(a) => a,
            Err(e) => {
                return BpfCompileResult {
                    gpu_eligible: false,
                    reject_reason: Some(format!("Tier 0 analysis failed: {}", e)),
                    ptx: None,
                    ir_program: None,
                    analysis: None,
                    program_hash,
                }
            }
        };

        if !analysis.gpu_eligible {
            return BpfCompileResult {
                gpu_eligible: false,
                reject_reason: analysis.reject_reason.clone(),
                ptx: None,
                ir_program: None,
                analysis: Some(analysis),
                program_hash,
            };
        }

        // Tier 1: BPF -> IR Lowering
        let mut ir_program = match crate::tier1_lowering::lower(bytecode, &analysis) {
            Ok(p) => p,
            Err(e) => {
                return BpfCompileResult {
                    gpu_eligible: false,
                    reject_reason: Some(format!("Tier 1 lowering failed: {}", e)),
                    ptx: None,
                    ir_program: None,
                    analysis: Some(analysis),
                    program_hash,
                }
            }
        };

        // Tier 2: SSA Construction & Optimization
        crate::tier2_optimization::optimize(&mut ir_program);

        // Tier 3: Compute Unit Metering
        let block_costs = crate::tier3_metering::calculate_block_costs(&ir_program);
        crate::tier3_metering::insert_metering(&mut ir_program, &block_costs);

        // Tier 4: PTX Emission
        let config = KernelConfig {
            block_size: 128,
            grid_size: 1, // Will be set at launch time
            shared_mem_bytes: 0,
            max_registers: 64,
        };

        let ptx = match crate::tier4_ptx_emit::emit_ptx(&ir_program, &program_hash, &config) {
            Ok(p) => p,
            Err(e) => {
                return BpfCompileResult {
                    gpu_eligible: false,
                    reject_reason: Some(format!("Tier 4 PTX emission failed: {}", e)),
                    ptx: None,
                    ir_program: Some(ir_program),
                    analysis: Some(analysis),
                    program_hash,
                }
            }
        };

        // Tier 5: Warp-Level Optimization
        let ptx = crate::tier5_warp_opt::optimize_warps(&ptx, &ir_program);

        // Tier 6: Memory Coalescing
        let (ptx, _transposition_plan) =
            crate::tier6_memory::optimize_memory(&ptx, &ir_program);

        // Tiers 7-8 are applied at batch execution time, not compilation time

        // Prepend syscall device functions
        let syscall_ptx = crate::syscalls::syscall_device_functions();
        let full_ptx = format!("{}\n{}", syscall_ptx, ptx);

        BpfCompileResult {
            gpu_eligible: true,
            reject_reason: None,
            ptx: Some(full_ptx),
            ir_program: Some(ir_program),
            analysis: Some(analysis),
            program_hash,
        }
    }

    fn hash_bytecode(bytecode: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(bytecode);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Get the compilation cache.
    pub fn cache(&self) -> &CompilationCache {
        &self.cache
    }

    /// Get the adaptive JIT profiler.
    pub fn adaptive(&mut self) -> &mut AdaptiveJit {
        &mut self.adaptive
    }
}

impl Default for BpfCompiler {
    fn default() -> Self {
        Self::new()
    }
}
