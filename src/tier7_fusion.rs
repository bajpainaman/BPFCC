use crate::types::*;

/// A fused kernel produced by combining two or more compiled BPF programs.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub ptx: String,
    pub program_hashes: Vec<[u8; 32]>,
    pub config: KernelConfig,
    pub register_usage: u32,
}

/// Attempt to fuse multiple compiled programs into a single kernel.
///
/// Returns `None` if fusion is not beneficial or not feasible.
pub fn try_fuse(programs: &[(BpfCompileResult, KernelConfig)]) -> Option<FusedKernel> {
    // Need at least two programs to fuse.
    if programs.len() < 2 {
        return None;
    }

    // Only fuse the first two candidates for now; recursive fusion can be
    // triggered by the caller if more programs are eligible.
    let (result_a, config_a) = &programs[0];
    let (result_b, config_b) = &programs[1];

    // Both programs must be GPU-eligible with valid PTX.
    if !result_a.gpu_eligible || !result_b.gpu_eligible {
        return None;
    }
    let ptx_a = result_a.ptx.as_deref()?;
    let ptx_b = result_b.ptx.as_deref()?;

    // Estimate register usage for each program.
    let regs_a = estimate_register_usage(ptx_a, config_a);
    let regs_b = estimate_register_usage(ptx_b, config_b);

    // Check combined register pressure.
    let combined_regs = regs_a + regs_b;
    if combined_regs > 255 {
        return None;
    }

    // Estimate instruction counts from IR.
    let instr_a = count_instructions(result_a);
    let instr_b = count_instructions(result_b);
    if instr_a + instr_b > 65_536 {
        return None;
    }

    // Check shared memory conflicts.
    if has_shared_memory_conflict(config_a, config_b) {
        return None;
    }

    // Apply fusion heuristic.
    if !should_fuse(config_a, config_b) {
        return None;
    }

    // Perform the actual fusion.
    let fused_ptx = fuse_ptx(ptx_a, ptx_b)?;

    let fused_config = merged_config(config_a, config_b);

    Some(FusedKernel {
        ptx: fused_ptx,
        program_hashes: vec![result_a.program_hash, result_b.program_hash],
        config: fused_config,
        register_usage: combined_regs,
    })
}

// ---------------------------------------------------------------------------
// Eligibility checks
// ---------------------------------------------------------------------------

/// Estimate the number of registers used by a program's PTX.
/// Falls back to `config.max_registers` when exact data is unavailable.
fn estimate_register_usage(ptx: &str, config: &KernelConfig) -> u32 {
    // Look for `.reg .uXX` declarations and count them.
    // Each declaration can declare N registers, e.g.:  .reg .u64 %r0, %r1, %r2;
    // Simple heuristic: count occurrences of `%r` that follow a `.reg` keyword.
    let reg_count = ptx
        .lines()
        .filter(|l| l.trim().starts_with(".reg"))
        .map(|l| l.matches('%').count() as u32)
        .sum::<u32>();

    if reg_count > 0 {
        reg_count.min(config.max_registers)
    } else {
        config.max_registers
    }
}

/// Count total IR instructions across all blocks.
fn count_instructions(result: &BpfCompileResult) -> usize {
    result
        .ir_program
        .as_ref()
        .map(|ir| {
            ir.blocks.iter().map(|b| b.ops.len() + 1 /* terminator */).sum()
        })
        .unwrap_or(0)
}

/// Return true if the two kernel configs would conflict on shared memory.
/// Shared memory conflict: both use shared memory AND their combined usage
/// exceeds the typical 48 KiB per SM.
fn has_shared_memory_conflict(a: &KernelConfig, b: &KernelConfig) -> bool {
    const MAX_SHARED_MEM: u32 = 49_152; // 48 KiB
    a.shared_mem_bytes > 0
        && b.shared_mem_bytes > 0
        && a.shared_mem_bytes + b.shared_mem_bytes > MAX_SHARED_MEM
}

/// Fusion heuristic: is the kernel-launch overhead larger than the barrier cost
/// plus additional register pressure cost?
fn should_fuse(a: &KernelConfig, b: &KernelConfig) -> bool {
    let launch_overhead_us: u64 = 5_000; // 5 ms
    let barrier_cost_us: u64 = 10;       // 0.01 ms
    let reg_pressure = estimate_reg_pressure(a, b);
    launch_overhead_us > barrier_cost_us + reg_pressure
}

/// Rough register-pressure penalty in microseconds.
/// Higher register usage → lower occupancy → higher effective cost.
fn estimate_reg_pressure(a: &KernelConfig, b: &KernelConfig) -> u64 {
    let combined = a.max_registers + b.max_registers;
    // Each register above 32 costs ~10 µs in lost occupancy (rough model).
    if combined > 32 {
        ((combined - 32) as u64) * 10
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// PTX Fusion
// ---------------------------------------------------------------------------

/// Merge two PTX strings into a single kernel.
///
/// Strategy:
///   1. Extract the parameter list and body of kernel A.
///   2. Extract the body of kernel B, rename its registers to avoid conflicts.
///   3. Emit a combined kernel that runs A's body, inserts `bar.sync 0;`,
///      then runs B's body.
fn fuse_ptx(ptx_a: &str, ptx_b: &str) -> Option<String> {
    let (params_a, body_a) = extract_kernel_parts(ptx_a)?;
    let (_params_b, body_b) = extract_kernel_parts(ptx_b)?;

    // Rename registers in program B to avoid collisions with program A.
    // Convention: %r0–%r15 belong to A; rename B's %r0–%r10 → %r16–%r26.
    let body_b_renamed = rename_registers_b(&body_b);

    // Extract .reg declarations from both bodies.
    let reg_decls_a = collect_reg_decls(&body_a);
    let reg_decls_b = collect_reg_decls(&body_b_renamed);

    // Strip .reg declarations from the code bodies (they go in the merged header).
    let code_a = strip_reg_decls(&body_a);
    let code_b = strip_reg_decls(&body_b_renamed);

    let fused = format!(
        ".version 7.0\n\
         .target sm_80\n\
         .address_size 64\n\
         \n\
         .visible .entry fused_kernel(\n\
         {params}\n\
         )\n\
         {{\n\
         {reg_a}\
         {reg_b}\
         \n\
         // --- Program A ---\n\
         {code_a}\n\
         \n\
         // Synchronize all threads before running Program B\n\
         bar.sync 0;\n\
         \n\
         // --- Program B ---\n\
         {code_b}\n\
         \n\
         ret;\n\
         }}\n",
        params = params_a,
        reg_a  = reg_decls_a,
        reg_b  = reg_decls_b,
        code_a = code_a,
        code_b = code_b,
    );

    Some(fused)
}

/// Extract the parameter list string and body string from a PTX kernel.
/// Returns `(params, body)` where body is the content between `{` and `}`.
fn extract_kernel_parts(ptx: &str) -> Option<(String, String)> {
    // Find the parameter list: between the first `(` and matching `)`.
    let param_start = ptx.find('(')?;
    let param_end   = ptx[param_start..].find(')')? + param_start;
    let params = ptx[param_start + 1..param_end].to_string();

    // Find the kernel body: between the first `{` and the last `}`.
    let body_start = ptx.find('{')? + 1;
    let body_end   = ptx.rfind('}')?;
    if body_end <= body_start {
        return None;
    }
    let body = ptx[body_start..body_end].to_string();

    Some((params, body))
}

/// Rename registers in program B's PTX body to avoid collisions.
/// Maps %r0–%r10 → %r16–%r26, %p0–%p4 → %p8–%p12.
fn rename_registers_b(body: &str) -> String {
    let mut out = body.to_string();

    // Rename predicate registers in reverse order to avoid partial replacements.
    for i in (0..=4u32).rev() {
        let from = format!("%p{}", i);
        let to   = format!("%p{}", i + 8);
        out = out.replace(&from, &to);
    }

    // Rename general registers in reverse order.
    for i in (0..=15u32).rev() {
        let from = format!("%r{}", i);
        let to   = format!("%r{}", i + 16);
        out = out.replace(&from, &to);
    }

    out
}

/// Collect all `.reg` declaration lines from a body string.
fn collect_reg_decls(body: &str) -> String {
    body.lines()
        .filter(|l| l.trim().starts_with(".reg"))
        .map(|l| format!("    {}\n", l.trim()))
        .collect()
}

/// Strip `.reg` declaration lines from a body string, leaving only code.
fn strip_reg_decls(body: &str) -> String {
    body.lines()
        .filter(|l| !l.trim().starts_with(".reg"))
        .map(|l| format!("    {}\n", l.trim()))
        .collect()
}

/// Build a merged KernelConfig for the fused kernel.
fn merged_config(a: &KernelConfig, b: &KernelConfig) -> KernelConfig {
    KernelConfig {
        // Use the larger block size for better occupancy.
        block_size: a.block_size.max(b.block_size),
        // Grid size must cover all work items for both programs.
        grid_size: a.grid_size.max(b.grid_size),
        // Fused kernel uses shared memory for the synchronization barrier.
        shared_mem_bytes: a.shared_mem_bytes + b.shared_mem_bytes,
        // Combined register budget, capped at hardware limit.
        max_registers: (a.max_registers + b.max_registers).min(255),
    }
}
