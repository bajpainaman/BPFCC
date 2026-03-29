//! Tier 3: Compute Unit Metering
//!
//! Inserts per-block CU metering ops at block boundaries so the runtime can
//! enforce compute limits without checking every instruction individually.

use crate::types::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Insert CU metering at the start of each basic block.
///
/// `block_cu_costs[i]` is the pre-computed cost for block `i`.
/// A `MeterCU` op is prepended to each block's ops vector so the runtime
/// deducts the cost before executing any instruction in that block.
pub fn insert_metering(program: &mut IrProgram, block_cu_costs: &[u64]) {
    for (idx, block) in program.blocks.iter_mut().enumerate() {
        let cost = block_cu_costs.get(idx).copied().unwrap_or(0);
        block.ops.insert(0, IrOp::MeterCU { cost });
    }
}

/// Calculate per-block CU costs from an IR program.
///
/// Returns a `Vec<u64>` with one entry per block.  The cost is the sum of
/// the CU weights of every instruction in the block according to the table
/// defined in the BPFCC specification.
pub fn calculate_block_costs(program: &IrProgram) -> Vec<u64> {
    program
        .blocks
        .iter()
        .map(|block| block.ops.iter().map(op_cu_cost).sum())
        .collect()
}

// ---------------------------------------------------------------------------
// Per-instruction CU cost table
// ---------------------------------------------------------------------------

fn op_cu_cost(op: &IrOp) -> u64 {
    match op {
        // ALU 64-bit: 1 CU each
        IrOp::Add64 { .. }
        | IrOp::Sub64 { .. }
        | IrOp::Mul64 { .. }
        | IrOp::And64 { .. }
        | IrOp::Or64  { .. }
        | IrOp::Xor64 { .. }
        | IrOp::Lsh64 { .. }
        | IrOp::Rsh64 { .. }
        | IrOp::Arsh64 { .. }
        | IrOp::Neg64 { .. } => 1,

        // Div/Mod 64-bit: 3 CU (expensive on hardware)
        IrOp::Div64 { .. } | IrOp::Mod64 { .. } => 3,

        // ALU 32-bit: 1 CU each
        IrOp::Add32 { .. }
        | IrOp::Sub32 { .. }
        | IrOp::Mul32 { .. }
        | IrOp::And32 { .. }
        | IrOp::Or32  { .. }
        | IrOp::Xor32 { .. }
        | IrOp::Lsh32 { .. }
        | IrOp::Rsh32 { .. }
        | IrOp::Arsh32 { .. }
        | IrOp::Neg32 { .. } => 1,

        // Div/Mod 32-bit: 3 CU
        IrOp::Div32 { .. } | IrOp::Mod32 { .. } => 3,

        // Move: 1 CU each
        IrOp::Mov64 { .. } | IrOp::Mov32 { .. } => 1,

        // Memory: 1 CU each
        IrOp::Load { .. } | IrOp::Store { .. } => 1,

        // Syscall: use the embedded cu_cost field
        IrOp::Syscall { cu_cost, .. } => *cu_cost,

        // Call: 5 CU (function call overhead)
        IrOp::Call { .. } => 5,

        // Safety checks: free (0 CU) — these guard against undefined behaviour
        IrOp::TrapIfZero { .. } | IrOp::BoundsCheck { .. } => 0,

        // SSA Phi: 0 CU (not a real instruction)
        IrOp::Phi { .. } => 0,

        // MeterCU itself: 0 CU (don't recursively meter the meter op)
        IrOp::MeterCU { .. } => 0,

        // Nop: 0 CU
        IrOp::Nop => 0,
    }
}
