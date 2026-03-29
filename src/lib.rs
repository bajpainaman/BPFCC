//! BPFCC — BPF-to-GPU JIT Compiler
//!
//! Translates Solana BPF/SBF smart contract bytecode into NVIDIA PTX assembly
//! for parallel GPU execution. 9-tier compilation pipeline.

pub mod bpf;
pub mod types;
pub mod syscall_ids;
pub mod tier0_analysis;
pub mod tier1_lowering;
pub mod tier2_optimization;
pub mod tier3_metering;
pub mod tier4_ptx_emit;
pub mod tier5_warp_opt;
pub mod tier6_memory;
pub mod tier7_fusion;
pub mod tier8_adaptive;
pub mod syscalls;
pub mod cache;
pub mod runtime;

pub use types::*;
pub use tier0_analysis::analyze;
pub use tier1_lowering::lower;
pub use runtime::BpfCompiler;
pub use cache::CompilationCache;
