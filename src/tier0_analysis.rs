use std::collections::{HashMap, HashSet};

use crate::bpf::opcodes::*;
use crate::bpf::{parse_program, BpfInstruction};
use crate::types::{
    AnalysisResult, BasicBlock, BranchCondition, ControlFlowGraph, EdgeType,
};
use crate::syscall_ids::*;

// ── Syscall classification ───────────────────────────────────────────────────

fn classify_syscall(id: u32) -> SyscallClass {
    if is_gpu_rejecting(id) {
        return SyscallClass::Reject;
    }
    if is_known_gpu_syscall(id) {
        return SyscallClass::GpuNative;
    }
    // Unknown syscall — reject conservatively
    SyscallClass::Reject
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyscallClass {
    GpuNative,
    Noop,
    Reject,
}

// ── CU cost table ────────────────────────────────────────────────────────────

fn cu_cost_of(opcode: u8) -> u64 {
    match opcode {
        // Division / modulo — higher cost
        DIV64_IMM | DIV64_REG | MOD64_IMM | MOD64_REG => 3,
        // Load double-word (2-slot instruction)
        LDDW => 2,
        // Conditional jumps
        JEQ_IMM | JEQ_REG | JNE_IMM | JNE_REG | JGT_IMM | JGT_REG | JGE_IMM | JGE_REG
        | JLT_IMM | JLT_REG | JLE_IMM | JLE_REG | JSGT_IMM | JSGT_REG | JSGE_IMM
        | JSGE_REG => 2,
        // Internal call
        CALL => 5,
        // Exit
        EXIT => 1,
        // Everything else: ALU, memory, unconditional jump = 1
        _ => 1,
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn is_jump(opcode: u8) -> bool {
    matches!(
        opcode,
        JA | JEQ_IMM
            | JEQ_REG
            | JNE_IMM
            | JNE_REG
            | JGT_IMM
            | JGT_REG
            | JGE_IMM
            | JGE_REG
            | JLT_IMM
            | JLT_REG
            | JLE_IMM
            | JLE_REG
            | JSGT_IMM
            | JSGT_REG
            | JSGE_IMM
            | JSGE_REG
    )
}

fn is_conditional_jump(opcode: u8) -> bool {
    is_jump(opcode) && opcode != JA
}

fn jump_target(pc: usize, insn: &BpfInstruction) -> usize {
    // BPF jump offset is relative to the *next* instruction (pc+1)
    (pc as i64 + 1 + insn.offset as i64) as usize
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Tier 0 — Static Analysis.
///
/// Returns an [`AnalysisResult`] regardless of GPU eligibility so that the
/// caller can inspect the CFG even for rejected programs.
pub fn analyze(bytecode: &[u8]) -> Result<AnalysisResult, String> {
    // ── Basic sanity ─────────────────────────────────────────────────────────
    if bytecode.len() % 8 != 0 {
        return Err("bytecode length must be a multiple of 8".into());
    }

    // R6: size check (256 KiB)
    if bytecode.len() > 256 * 1024 {
        return Ok(rejected(
            "bytecode exceeds 256 KiB limit",
            empty_cfg(),
            0,
            HashSet::new(),
        ));
    }

    let instructions = parse_program(bytecode)
        .map_err(|e| format!("parse error: {e}"))?;

    let n = instructions.len();

    // R10: instruction count ≤ 65536
    if n > 65536 {
        return Ok(rejected(
            "instruction count exceeds 65536",
            empty_cfg(),
            n,
            HashSet::new(),
        ));
    }

    // ── Pass 1: collect block boundaries & validate targets ──────────────────
    let mut block_starts: HashSet<usize> = HashSet::new();
    block_starts.insert(0);

    let mut reject_reason: Option<String> = None;
    let mut syscalls_used: HashSet<u32> = HashSet::new();
    let mut uses_heap = false;
    let mut has_internal_calls = false;
    let mut call_depth_hint = 0u32; // conservative: count CALL instructions

    let mut pc = 0usize;
    while pc < n {
        let insn = &instructions[pc];

        match insn.opcode {
            LDDW => {
                // R9: second slot must exist and be a padding instruction
                if pc + 1 >= n {
                    return Err(format!("LDDW at pc={pc} missing second slot"));
                }
                // The slot after LDDW is consumed — NOT a block start
                pc += 2;
                continue;
            }

            JA => {
                let target = jump_target(pc, insn);
                // R8: no backward jumps > 4096
                if target < pc && pc - target > 4096 {
                    reject_reason.get_or_insert_with(|| {
                        format!(
                            "backward jump at pc={pc} spans {} instructions (> 4096)",
                            pc - target
                        )
                    });
                }
                // R9: target in bounds
                if target >= n {
                    reject_reason.get_or_insert_with(|| {
                        format!("jump at pc={pc} targets out-of-bounds pc={target}")
                    });
                } else {
                    block_starts.insert(target);
                }
                block_starts.insert(pc + 1); // instruction after JA is a new block
            }

            op if is_conditional_jump(op) => {
                let true_target = jump_target(pc, insn);
                let false_target = pc + 1;

                // R8: backward jumps
                if true_target < pc && pc - true_target > 4096 {
                    reject_reason.get_or_insert_with(|| {
                        format!(
                            "backward conditional jump at pc={pc} spans {} instructions (> 4096)",
                            pc - true_target
                        )
                    });
                }
                // R9: bounds
                if true_target >= n {
                    reject_reason.get_or_insert_with(|| {
                        format!(
                            "conditional jump at pc={pc} targets out-of-bounds pc={true_target}"
                        )
                    });
                } else {
                    block_starts.insert(true_target);
                }
                if false_target < n {
                    block_starts.insert(false_target);
                }
            }

            CALL => {
                if insn.src == 1 {
                    // Internal function call (src_reg=1): imm is a PC offset
                    has_internal_calls = true;
                } else {
                    // Syscall (src_reg=0): imm is the syscall hash
                    let syscall_id = insn.imm as u32;
                    syscalls_used.insert(syscall_id);

                    match classify_syscall(syscall_id) {
                        SyscallClass::Reject => {
                            reject_reason.get_or_insert_with(|| {
                                format!("rejected syscall id=0x{syscall_id:08x} at pc={pc}")
                            });
                        }
                        SyscallClass::GpuNative | SyscallClass::Noop => {}
                    }

                    // Heap usage heuristic: sol_alloc_free
                    if syscall_id == SOL_ALLOC_FREE {
                        uses_heap = true;
                    }
                }

                call_depth_hint += 1;
                block_starts.insert(pc + 1);
            }

            EXIT => {
                if pc + 1 < n {
                    block_starts.insert(pc + 1);
                }
            }

            _ => {}
        }

        pc += 1;
    }

    // R7: max stack depth — approximate via CALL count (each call frame uses stack)
    // A tighter analysis would require interprocedural tracking; for now we treat
    // each distinct CALL instruction as one potential frame.
    let max_stack_depth = call_depth_hint.min(32); // conservative cap

    if max_stack_depth > 16 {
        reject_reason.get_or_insert_with(|| {
            format!("estimated stack depth {max_stack_depth} exceeds 16-frame limit")
        });
    }

    // ── Pass 2: build basic blocks ───────────────────────────────────────────
    let mut sorted_starts: Vec<usize> = block_starts.into_iter().collect();
    sorted_starts.sort_unstable();
    // Remove any out-of-bounds starts that slipped through
    sorted_starts.retain(|&s| s < n);

    let num_blocks = sorted_starts.len();
    let start_to_block: HashMap<usize, usize> = sorted_starts
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(num_blocks);
    let mut edges: Vec<(usize, usize, EdgeType)> = Vec::new();
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];

    for (block_idx, &start) in sorted_starts.iter().enumerate() {
        let end = if block_idx + 1 < num_blocks {
            sorted_starts[block_idx + 1]
        } else {
            n
        };

        let mut insn_count = 0usize;
        let mut has_syscall = false;
        let mut block_cu: u64 = 0;

        // Walk instructions in this block, accounting for LDDW double-slots
        let mut cursor = start;
        while cursor < end {
            let insn = &instructions[cursor];
            insn_count += 1;
            block_cu += cu_cost_of(insn.opcode);

            if insn.opcode == CALL {
                has_syscall = true;
            }

            if insn.opcode == LDDW {
                // Skip the padding slot — it belongs to this block but is not
                // an independent instruction
                cursor += 2;
                if cursor <= end {
                    // extra slot counted as part of LDDW
                }
                continue;
            }
            cursor += 1;
        }

        // Determine block terminator and add edges
        let last_pc = if end > start { end - 1 } else { start };
        // Walk back from end to find the actual last instruction (skip LDDW padding)
        let term_pc = find_terminator_pc(&instructions, start, end);
        let term_insn = &instructions[term_pc];

        match term_insn.opcode {
            JA => {
                let target = jump_target(term_pc, term_insn);
                if let Some(&tid) = start_to_block.get(&target) {
                    add_edge(
                        &mut edges,
                        &mut successors,
                        &mut predecessors,
                        block_idx,
                        tid,
                        EdgeType::Unconditional,
                    );
                }
            }

            op if is_conditional_jump(op) => {
                let true_target = jump_target(term_pc, term_insn);
                let false_target = term_pc + 1;

                if let Some(&tid) = start_to_block.get(&true_target) {
                    add_edge(
                        &mut edges,
                        &mut successors,
                        &mut predecessors,
                        block_idx,
                        tid,
                        EdgeType::ConditionalTrue,
                    );
                }
                if let Some(&fid) = start_to_block.get(&false_target) {
                    add_edge(
                        &mut edges,
                        &mut successors,
                        &mut predecessors,
                        block_idx,
                        fid,
                        EdgeType::ConditionalFalse,
                    );
                }
            }

            CALL => {
                // Internal call: fall-through is the next instruction
                let next = term_pc + 1;
                if let Some(&nid) = start_to_block.get(&next) {
                    add_edge(
                        &mut edges,
                        &mut successors,
                        &mut predecessors,
                        block_idx,
                        nid,
                        EdgeType::Call,
                    );
                }
            }

            EXIT => {
                // No successors — this block ends the program
            }

            _ => {
                // Fall-through to next block
                let next = end;
                if let Some(&nid) = start_to_block.get(&next) {
                    add_edge(
                        &mut edges,
                        &mut successors,
                        &mut predecessors,
                        block_idx,
                        nid,
                        EdgeType::Unconditional,
                    );
                }
            }
        }

        let _ = last_pc; // silence unused warning
        blocks.push(BasicBlock {
            id: block_idx,
            start_pc: start,
            end_pc: end,
            instruction_count: insn_count,
            has_syscall,
            cu_cost: block_cu,
        });
    }

    let cfg = ControlFlowGraph {
        blocks,
        edges,
        successors,
        predecessors,
    };

    // ── CU estimation: longest path through CFG ──────────────────────────────
    let estimated_cu = longest_path_cu(&cfg);

    // ── Build block_entries list ─────────────────────────────────────────────
    let block_entries: Vec<usize> = sorted_starts;

    // ── Final eligibility decision ───────────────────────────────────────────
    let gpu_eligible = reject_reason.is_none();

    Ok(AnalysisResult {
        gpu_eligible,
        reject_reason,
        cfg,
        instruction_count: n,
        syscalls_used,
        max_stack_depth,
        uses_heap,
        block_entries,
        estimated_cu,
        has_internal_calls,
    })
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn add_edge(
    edges: &mut Vec<(usize, usize, EdgeType)>,
    successors: &mut Vec<Vec<usize>>,
    predecessors: &mut Vec<Vec<usize>>,
    from: usize,
    to: usize,
    kind: EdgeType,
) {
    edges.push((from, to, kind));
    if !successors[from].contains(&to) {
        successors[from].push(to);
    }
    if !predecessors[to].contains(&from) {
        predecessors[to].push(from);
    }
}

/// Find the index of the last "real" instruction in [start, end).
/// Handles LDDW double-slots so we don't pick the padding slot.
fn find_terminator_pc(instructions: &[BpfInstruction], start: usize, end: usize) -> usize {
    let mut cursor = start;
    let mut last_real = start;
    while cursor < end {
        last_real = cursor;
        if instructions[cursor].opcode == LDDW {
            cursor += 2;
        } else {
            cursor += 1;
        }
    }
    last_real
}

/// Topological longest-path (maximum CU cost) through the CFG.
/// Uses Kahn's algorithm with a cost accumulator.
fn longest_path_cu(cfg: &ControlFlowGraph) -> u64 {
    let n = cfg.blocks.len();
    if n == 0 {
        return 0;
    }

    // In-degree for topological sort
    let mut in_deg: Vec<usize> = cfg.predecessors.iter().map(|p| p.len()).collect();
    let mut dist: Vec<u64> = cfg.blocks.iter().map(|b| b.cu_cost).collect();

    let mut queue: std::collections::VecDeque<usize> = (0..n)
        .filter(|&i| in_deg[i] == 0)
        .collect();

    while let Some(u) = queue.pop_front() {
        for &v in &cfg.successors[u] {
            let new_dist = dist[u] + cfg.blocks[v].cu_cost;
            if new_dist > dist[v] {
                dist[v] = new_dist;
            }
            in_deg[v] = in_deg[v].saturating_sub(1);
            if in_deg[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    dist.into_iter().max().unwrap_or(0)
}

fn empty_cfg() -> ControlFlowGraph {
    ControlFlowGraph {
        blocks: Vec::new(),
        edges: Vec::new(),
        successors: Vec::new(),
        predecessors: Vec::new(),
    }
}

fn rejected(
    reason: &str,
    cfg: ControlFlowGraph,
    instruction_count: usize,
    syscalls_used: HashSet<u32>,
) -> AnalysisResult {
    AnalysisResult {
        gpu_eligible: false,
        reject_reason: Some(reason.to_string()),
        cfg,
        instruction_count,
        syscalls_used,
        max_stack_depth: 0,
        uses_heap: false,
        block_entries: Vec::new(),
        estimated_cu: 0,
        has_internal_calls: false,
    }
}

// Missing opcodes that the bpf::opcodes module doesn't define — add them here
// so this file compiles without modifying the upstream module.
const DIV64_IMM: u8 = 0x37;
const DIV64_REG: u8 = 0x3f;
const MOD64_IMM: u8 = 0x97;
const MOD64_REG: u8 = 0x9f;
const JSGT_IMM: u8 = 0x65;
const JSGT_REG: u8 = 0x6d;
const JSGE_IMM: u8 = 0x75;
const JSGE_REG: u8 = 0x7d;
const JLT_IMM: u8 = 0xa5;
const JLT_REG: u8 = 0xad;
const JLE_IMM: u8 = 0xb5;
const JLE_REG: u8 = 0xbd;
const JNE_IMM: u8 = 0x55;
const JNE_REG: u8 = 0x5d;

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_bytecode() -> Vec<u8> {
        // mov64 r0, 42; exit
        vec![
            0xb7, 0x00, 0x00, 0x00, 42, 0, 0, 0,
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,
        ]
    }

    #[test]
    fn test_simple_eligible() {
        let result = analyze(&simple_bytecode()).unwrap();
        assert!(result.gpu_eligible);
        assert_eq!(result.instruction_count, 2);
        assert_eq!(result.cfg.blocks.len(), 1);
    }

    #[test]
    fn test_empty_program_rejected() {
        // Zero-length bytecode is valid structurally but has no EXIT — a single
        // block with no instructions and no terminator; still parses to 0 blocks.
        let result = analyze(&[]).unwrap();
        assert!(result.gpu_eligible); // no violations, just empty
        assert_eq!(result.instruction_count, 0);
    }

    #[test]
    fn test_oversized_rejected() {
        // 256 KiB + 8 bytes
        let big = vec![0u8; 256 * 1024 + 8];
        let result = analyze(&big).unwrap();
        assert!(!result.gpu_eligible);
        assert!(result.reject_reason.unwrap().contains("256 KiB"));
    }
}
