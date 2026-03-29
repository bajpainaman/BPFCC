//! Tier 2: SSA Construction & Optimization
//!
//! Applies dominance-based SSA construction followed by constant folding,
//! dead code elimination, and strength reduction to the IR program.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::types::*;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Apply all Tier 2 optimizations to an IR program.
pub fn optimize(program: &mut IrProgram) {
    let dom_tree = build_dom_tree(program);
    construct_ssa(program, &dom_tree);
    constant_fold(program);
    dead_code_eliminate(program);
    strength_reduce(program);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn log2(n: i64) -> i64 {
    debug_assert!(n > 0);
    let mut v = n;
    let mut result = 0i64;
    while v > 1 {
        v >>= 1;
        result += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// 1a. Dominance Tree (iterative dataflow)
// ---------------------------------------------------------------------------

/// Build a dominance tree using the iterative dataflow algorithm.
///
/// idom[b] = immediate dominator of block b (index into program.blocks).
/// Entry block's idom is itself.
pub fn build_dom_tree(program: &IrProgram) -> DomTree {
    let n = program.blocks.len();
    if n == 0 {
        return DomTree {
            idom: vec![],
            frontier: vec![],
            children: vec![],
        };
    }

    // Compute reverse-postorder (RPO) of the CFG.
    let rpo = reverse_postorder(program);
    // rpo_index[b] = position of block b in RPO (lower = earlier).
    let mut rpo_index = vec![0usize; n];
    for (pos, &b) in rpo.iter().enumerate() {
        rpo_index[b] = pos;
    }

    // Build predecessor lists.
    let preds = build_predecessors(program);

    // idom[b] stores the immediate dominator of b as Option<usize>.
    // None means "not yet computed".
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[rpo[0]] = Some(rpo[0]); // entry dominates itself

    let mut changed = true;
    while changed {
        changed = false;
        // Skip entry (index 0 in RPO).
        for &b in rpo.iter().skip(1) {
            // Find first processed predecessor.
            let processed_preds: Vec<usize> = preds[b]
                .iter()
                .copied()
                .filter(|&p| idom[p].is_some())
                .collect();

            if processed_preds.is_empty() {
                continue;
            }

            // Start with first processed predecessor.
            let mut new_idom = processed_preds[0];
            for &p in processed_preds.iter().skip(1) {
                new_idom = intersect(p, new_idom, &idom, &rpo_index);
            }

            if idom[b] != Some(new_idom) {
                idom[b] = Some(new_idom);
                changed = true;
            }
        }
    }

    // Replace any remaining None with self (unreachable blocks dominate themselves).
    for i in 0..n {
        if idom[i].is_none() {
            idom[i] = Some(i);
        }
    }

    // Build children list.
    let mut children = vec![vec![]; n];
    for b in 0..n {
        let d = idom[b].unwrap();
        if d != b {
            children[d].push(b);
        }
    }

    // Build dominance frontiers.
    // DF[b] = { y | ∃ pred p of y such that b dom p and b does not strictly dom y }
    let mut frontier: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for b in 0..n {
        if preds[b].len() >= 2 {
            for &p in &preds[b] {
                let mut runner = p;
                while Some(runner) != idom[b] {
                    frontier[runner].insert(b);
                    runner = idom[runner].unwrap();
                }
            }
        }
    }

    DomTree {
        idom,
        frontier,
        children,
    }
}

/// Compute reverse-postorder of the CFG starting from block 0.
fn reverse_postorder(program: &IrProgram) -> Vec<usize> {
    let n = program.blocks.len();
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    dfs_postorder(program, 0, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

fn dfs_postorder(
    program: &IrProgram,
    block: usize,
    visited: &mut Vec<bool>,
    postorder: &mut Vec<usize>,
) {
    if visited[block] {
        return;
    }
    visited[block] = true;
    for succ in block_successors(program, block) {
        dfs_postorder(program, succ, visited, postorder);
    }
    postorder.push(block);
}

/// Return successor block indices for a given block.
fn block_successors(program: &IrProgram, block: usize) -> Vec<usize> {
    match &program.blocks[block].terminator {
        IrTerminator::Jump { target } => vec![*target],
        IrTerminator::Branch { true_target, false_target, .. } => {
            vec![*true_target, *false_target]
        }
        IrTerminator::Return { .. } => vec![],
        IrTerminator::Trap { .. } => vec![],
    }
}

/// Build predecessor lists for all blocks.
fn build_predecessors(program: &IrProgram) -> Vec<Vec<usize>> {
    let n = program.blocks.len();
    let mut preds = vec![vec![]; n];
    for b in 0..n {
        for succ in block_successors(program, b) {
            preds[succ].push(b);
        }
    }
    preds
}

/// Intersect dominator sets (Cooper et al. algorithm).
fn intersect(mut b1: usize, mut b2: usize, idom: &[Option<usize>], rpo_index: &[usize]) -> usize {
    while b1 != b2 {
        while rpo_index[b1] > rpo_index[b2] {
            b1 = idom[b1].unwrap();
        }
        while rpo_index[b2] > rpo_index[b1] {
            b2 = idom[b2].unwrap();
        }
    }
    b1
}

// ---------------------------------------------------------------------------
// 1b. SSA Construction
// ---------------------------------------------------------------------------

fn construct_ssa(program: &mut IrProgram, dom_tree: &DomTree) {
    let n = program.blocks.len();
    if n == 0 {
        return;
    }

    // BPF has registers 0-10.
    const BPF_REGS: usize = 11;

    // Step 1: Find def_blocks for each register.
    let mut def_blocks: Vec<HashSet<usize>> = vec![HashSet::new(); BPF_REGS];
    for b in 0..n {
        for op in &program.blocks[b].ops {
            if let Some(dst) = op_dst_reg(op) {
                if dst < BPF_REGS {
                    def_blocks[dst].insert(b);
                }
            }
        }
    }

    // Step 2: Insert Phi nodes using dominance frontiers.
    // phi_inserted[reg][block] = true if we already placed a phi there.
    let mut phi_inserted: Vec<HashSet<usize>> = vec![HashSet::new(); BPF_REGS];
    for reg in 0..BPF_REGS {
        // Only insert phis for registers defined in more than one block.
        if def_blocks[reg].len() < 2 {
            continue;
        }
        let mut worklist: VecDeque<usize> = def_blocks[reg].iter().copied().collect();
        while let Some(b) = worklist.pop_front() {
            for &frontier in &dom_tree.frontier[b] {
                if phi_inserted[reg].insert(frontier) {
                    // Prepend a Phi node.
                    program.blocks[frontier]
                        .ops
                        .insert(0, IrOp::Phi { dst: reg as u8, sources: vec![] });
                    // The frontier itself now defines the register, so propagate.
                    if def_blocks[reg].insert(frontier) {
                        worklist.push_back(frontier);
                    }
                }
            }
        }
    }

    // Step 3: Rename variables using SSA counter starting at 11.
    // We use a simple dominator-tree walk (DFS).
    let mut ssa_counter: u32 = 11;
    // Stack of current name for each original register (0-10).
    let mut name_stack: Vec<Vec<u32>> = (0..BPF_REGS).map(|i| vec![i as u32]).collect();

    rename_block(
        program,
        0,
        dom_tree,
        &mut name_stack,
        &mut ssa_counter,
        BPF_REGS,
    );
}

fn rename_block(
    program: &mut IrProgram,
    block: usize,
    dom_tree: &DomTree,
    name_stack: &mut Vec<Vec<u32>>,
    ssa_counter: &mut u32,
    bpf_regs: usize,
) {
    // Track which names we pushed so we can pop them on exit.
    let mut pushed: Vec<usize> = vec![];

    // Rename ops in this block.
    let ops_len = program.blocks[block].ops.len();
    for i in 0..ops_len {
        // Rename source operands (uses) first, then definition (dst).
        // We need to update sources before the new name for dst is pushed.
        rename_op_uses(&mut program.blocks[block].ops[i], name_stack, bpf_regs);

        // Rename dst (definition).
        if let Some(dst) = op_dst_reg_mut(&mut program.blocks[block].ops[i]) {
            if (*dst as usize) < bpf_regs {
                let orig = *dst as usize;
                let new_name = *ssa_counter;
                *ssa_counter += 1;
                name_stack[orig].push(new_name);
                pushed.push(orig);
                *dst = new_name as u8;
            }
        }
    }

    // Rename terminator sources.
    rename_terminator_uses(&mut program.blocks[block].terminator, name_stack, bpf_regs);

    // Fill in Phi node sources in successors.
    let succs = block_successors(program, block);
    for succ in succs {
        // For each Phi in succ, add the current name of the original register.
        for op in &mut program.blocks[succ].ops {
            if let IrOp::Phi { dst, sources } = op {
                let orig = *dst as usize;
                if orig < bpf_regs && !name_stack[orig].is_empty() {
                    let current = *name_stack[orig].last().unwrap();
                    sources.push((block, current as u8));
                }
            }
        }
    }

    // Recurse into dominated children.
    let children = dom_tree.children[block].clone();
    for child in children {
        rename_block(program, child, dom_tree, name_stack, ssa_counter, bpf_regs);
    }

    // Pop names pushed in this block.
    for orig in pushed {
        name_stack[orig].pop();
    }
}

/// Return the destination register of an op, if any (as u8).
fn op_dst_reg(op: &IrOp) -> Option<usize> {
    match op {
        IrOp::Add64 { dst, .. }
        | IrOp::Sub64 { dst, .. }
        | IrOp::Mul64 { dst, .. }
        | IrOp::Div64 { dst, .. }
        | IrOp::Mod64 { dst, .. }
        | IrOp::And64 { dst, .. }
        | IrOp::Or64 { dst, .. }
        | IrOp::Xor64 { dst, .. }
        | IrOp::Lsh64 { dst, .. }
        | IrOp::Rsh64 { dst, .. }
        | IrOp::Arsh64 { dst, .. }
        | IrOp::Add32 { dst, .. }
        | IrOp::Sub32 { dst, .. }
        | IrOp::Mul32 { dst, .. }
        | IrOp::Div32 { dst, .. }
        | IrOp::Mod32 { dst, .. }
        | IrOp::And32 { dst, .. }
        | IrOp::Or32 { dst, .. }
        | IrOp::Xor32 { dst, .. }
        | IrOp::Lsh32 { dst, .. }
        | IrOp::Rsh32 { dst, .. }
        | IrOp::Arsh32 { dst, .. }
        | IrOp::Mov64 { dst, .. }
        | IrOp::Mov32 { dst, .. }
        | IrOp::Neg64 { dst }
        | IrOp::Neg32 { dst }
        | IrOp::Load { dst, .. }
        | IrOp::Phi { dst, .. } => Some(*dst as usize),
        IrOp::Syscall { .. }
        | IrOp::Store { .. }
        | IrOp::Call { .. }
        | IrOp::TrapIfZero { .. }
        | IrOp::BoundsCheck { .. }
        | IrOp::MeterCU { .. }
        | IrOp::Nop => None,
    }
}

/// Return a mutable reference to the destination register field of an op.
fn op_dst_reg_mut(op: &mut IrOp) -> Option<&mut u8> {
    match op {
        IrOp::Add64 { dst, .. }
        | IrOp::Sub64 { dst, .. }
        | IrOp::Mul64 { dst, .. }
        | IrOp::Div64 { dst, .. }
        | IrOp::Mod64 { dst, .. }
        | IrOp::And64 { dst, .. }
        | IrOp::Or64 { dst, .. }
        | IrOp::Xor64 { dst, .. }
        | IrOp::Lsh64 { dst, .. }
        | IrOp::Rsh64 { dst, .. }
        | IrOp::Add32 { dst, .. }
        | IrOp::Sub32 { dst, .. }
        | IrOp::Mul32 { dst, .. }
        | IrOp::Div32 { dst, .. }
        | IrOp::Mod32 { dst, .. }
        | IrOp::And32 { dst, .. }
        | IrOp::Or32 { dst, .. }
        | IrOp::Xor32 { dst, .. }
        | IrOp::Lsh32 { dst, .. }
        | IrOp::Rsh32 { dst, .. }
        | IrOp::Arsh64 { dst, .. }
        | IrOp::Arsh32 { dst, .. }
        | IrOp::Mov64 { dst, .. }
        | IrOp::Mov32 { dst, .. }
        | IrOp::Neg64 { dst }
        | IrOp::Neg32 { dst }
        | IrOp::Load { dst, .. }
        | IrOp::Phi { dst, .. } => Some(dst),
        IrOp::Syscall { .. }
        | IrOp::Store { .. }
        | IrOp::Call { .. }
        | IrOp::TrapIfZero { .. }
        | IrOp::BoundsCheck { .. }
        | IrOp::MeterCU { .. }
        | IrOp::Nop => None,
    }
}

/// Rename the source operands of an op using the current SSA name stacks.
fn rename_op_uses(op: &mut IrOp, name_stack: &[Vec<u32>], bpf_regs: usize) {
    match op {
        IrOp::Add64 { src, .. }
        | IrOp::Sub64 { src, .. }
        | IrOp::Mul64 { src, .. }
        | IrOp::Div64 { src, .. }
        | IrOp::Mod64 { src, .. }
        | IrOp::And64 { src, .. }
        | IrOp::Or64 { src, .. }
        | IrOp::Xor64 { src, .. }
        | IrOp::Lsh64 { src, .. }
        | IrOp::Rsh64 { src, .. }
        | IrOp::Arsh64 { src, .. }
        | IrOp::Add32 { src, .. }
        | IrOp::Sub32 { src, .. }
        | IrOp::Mul32 { src, .. }
        | IrOp::Div32 { src, .. }
        | IrOp::Mod32 { src, .. }
        | IrOp::And32 { src, .. }
        | IrOp::Or32 { src, .. }
        | IrOp::Xor32 { src, .. }
        | IrOp::Lsh32 { src, .. }
        | IrOp::Rsh32 { src, .. }
        | IrOp::Arsh32 { src, .. } => {
            rename_operand(src, name_stack, bpf_regs);
        }
        IrOp::Mov64 { src, .. } | IrOp::Mov32 { src, .. } => {
            rename_operand(src, name_stack, bpf_regs);
        }
        IrOp::Store { src, base, .. } => {
            rename_reg_field(src, name_stack, bpf_regs);
            rename_reg_field(base, name_stack, bpf_regs);
        }
        IrOp::Load { base, .. } => {
            rename_reg_field(base, name_stack, bpf_regs);
        }
        IrOp::TrapIfZero { src } => {
            rename_reg_field(src, name_stack, bpf_regs);
        }
        IrOp::BoundsCheck { addr, .. } => {
            rename_reg_field(addr, name_stack, bpf_regs);
        }
        IrOp::Syscall { .. } => {}
        IrOp::Neg64 { .. }
        | IrOp::Neg32 { .. }
        | IrOp::Call { .. }
        | IrOp::Phi { .. }
        | IrOp::MeterCU { .. }
        | IrOp::Nop => {}
    }
}

fn rename_operand(operand: &mut Operand, name_stack: &[Vec<u32>], bpf_regs: usize) {
    if let Operand::Reg(r) = operand {
        rename_reg_field(r, name_stack, bpf_regs);
    }
}

fn rename_reg_field(reg: &mut u8, name_stack: &[Vec<u32>], bpf_regs: usize) {
    let r = *reg as usize;
    if r < bpf_regs {
        if let Some(&name) = name_stack[r].last() {
            *reg = name as u8;
        }
    }
}

fn rename_terminator_uses(
    term: &mut IrTerminator,
    name_stack: &[Vec<u32>],
    bpf_regs: usize,
) {
    match term {
        IrTerminator::Branch { dst, src, .. } => {
            rename_reg_field(dst, name_stack, bpf_regs);
            rename_operand(src, name_stack, bpf_regs);
        }
        IrTerminator::Return { value } => {
            rename_reg_field(value, name_stack, bpf_regs);
        }
        IrTerminator::Jump { .. } | IrTerminator::Trap { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// 1c. Constant Folding
// ---------------------------------------------------------------------------

fn constant_fold(program: &mut IrProgram) {
    for block in &mut program.blocks {
        // Map from register index to known constant value.
        let mut constants: HashMap<u8, i64> = HashMap::new();

        for op in &mut block.ops {
            // Try to fold the current op given known constants, then record
            // any new constant definition it produces.
            let folded = try_fold(op, &constants);
            if let Some(new_op) = folded {
                *op = new_op;
            }
            // Update constants map based on what this op defines.
            record_constant(op, &mut constants);
        }
    }
}

/// If all source operands are constants, evaluate the op and return a
/// replacement Mov64/Mov32 carrying the result.  Returns None otherwise.
fn try_fold(op: &IrOp, constants: &HashMap<u8, i64>) -> Option<IrOp> {
    match op {
        // 64-bit binary ALU — operand is Imm or Reg
        IrOp::Add64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a.wrapping_add(b)),
        IrOp::Sub64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a.wrapping_sub(b)),
        IrOp::Mul64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a.wrapping_mul(b)),
        IrOp::Div64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| {
            if b == 0 { 0 } else { ((a as u64) / (b as u64)) as i64 }
        }),
        IrOp::Mod64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| {
            if b == 0 { 0 } else { ((a as u64) % (b as u64)) as i64 }
        }),
        IrOp::And64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a & b),
        IrOp::Or64  { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a | b),
        IrOp::Xor64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| a ^ b),
        IrOp::Lsh64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| {
            let shift = (b as u64) & 63;
            (a as u64).wrapping_shl(shift as u32) as i64
        }),
        IrOp::Rsh64 { dst, src } => fold_bin64(*dst, *src, constants, |a, b| {
            let shift = (b as u64) & 63;
            (a as u64).wrapping_shr(shift as u32) as i64
        }),

        // 32-bit binary ALU (mask result to u32)
        IrOp::Add32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a.wrapping_add(b)),
        IrOp::Sub32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a.wrapping_sub(b)),
        IrOp::Mul32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a.wrapping_mul(b)),
        IrOp::Div32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| {
            if b == 0 { 0 } else { a / b }
        }),
        IrOp::Mod32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| {
            if b == 0 { 0 } else { a % b }
        }),
        IrOp::And32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a & b),
        IrOp::Or32  { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a | b),
        IrOp::Xor32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| a ^ b),
        IrOp::Lsh32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| {
            let shift = b & 31;
            a.wrapping_shl(shift as u32)
        }),
        IrOp::Rsh32 { dst, src } => fold_bin32(*dst, *src, constants, |a, b| {
            let shift = b & 31;
            (a as u32).wrapping_shr(shift as u32) as i32
        }),

        // Neg
        IrOp::Neg64 { dst } => {
            let dst_val = constants.get(dst)?;
            Some(IrOp::Mov64 { dst: *dst, src: Operand::Imm(dst_val.wrapping_neg()) })
        }
        IrOp::Neg32 { dst } => {
            let dst_val = *constants.get(dst)? as i32;
            Some(IrOp::Mov32 {
                dst: *dst,
                src: Operand::Imm(dst_val.wrapping_neg() as i64),
            })
        }

        // Mov64/Mov32 with a constant source register.
        IrOp::Mov64 { dst, src: Operand::Reg(r) } => {
            let val = *constants.get(r)?;
            Some(IrOp::Mov64 { dst: *dst, src: Operand::Imm(val) })
        }
        IrOp::Mov32 { dst, src: Operand::Reg(r) } => {
            let val = (*constants.get(r)? as i32) as i64;
            Some(IrOp::Mov32 { dst: *dst, src: Operand::Imm(val) })
        }

        _ => None,
    }
}

fn resolve_operand(src: Operand, _dst_reg: u8, constants: &HashMap<u8, i64>) -> Option<i64> {
    match src {
        Operand::Imm(v) => Some(v),
        Operand::Imm64(v) => Some(v as i64),
        Operand::Reg(r) => {
            // If src == dst we use the current (pre-update) value.
            constants.get(&r).copied()
        }
    }
}

fn fold_bin64(
    dst: u8,
    src: Operand,
    constants: &HashMap<u8, i64>,
    f: impl Fn(i64, i64) -> i64,
) -> Option<IrOp> {
    let a = *constants.get(&dst)?;
    let b = resolve_operand(src, dst, constants)?;
    Some(IrOp::Mov64 { dst, src: Operand::Imm(f(a, b)) })
}

fn fold_bin32(
    dst: u8,
    src: Operand,
    constants: &HashMap<u8, i64>,
    f: impl Fn(i32, i32) -> i32,
) -> Option<IrOp> {
    let a = *constants.get(&dst)? as i32;
    let b = resolve_operand(src, dst, constants)? as i32;
    Some(IrOp::Mov32 {
        dst,
        src: Operand::Imm(f(a, b) as i64),
    })
}

/// Update the constants map after processing an op.
fn record_constant(op: &IrOp, constants: &mut HashMap<u8, i64>) {
    match op {
        IrOp::Mov64 { dst, src: Operand::Imm(v) } => {
            constants.insert(*dst, *v);
        }
        IrOp::Mov32 { dst, src: Operand::Imm(v) } => {
            // Zero-extend: 32-bit result stored as unsigned in i64.
            constants.insert(*dst, (*v as i32) as i64);
        }
        // Any op that writes dst but isn't a constant Mov kills the constant.
        op => {
            if let Some(dst) = op_dst_reg(op) {
                constants.remove(&(dst as u8));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1d. Dead Code Elimination
// ---------------------------------------------------------------------------

fn dead_code_eliminate(program: &mut IrProgram) {
    // Iterate to fixpoint.
    loop {
        let changed = dce_pass(program);
        if !changed {
            break;
        }
    }
}

fn dce_pass(program: &mut IrProgram) -> bool {
    let n = program.blocks.len();
    if n == 0 {
        return false;
    }

    // Global liveness: sets of registers live at the entry of each block.
    // We iterate backwards over a topological order until fixpoint.
    let preds = build_predecessors(program);

    // Compute successors for all blocks.
    let succs: Vec<Vec<usize>> = (0..n)
        .map(|b| block_successors(program, b))
        .collect();

    // live_in[b] = set of registers live at the entry of block b.
    let mut live_in: Vec<HashSet<u8>> = vec![HashSet::new(); n];

    // Iterate backwards until stable.
    let mut global_changed = true;
    while global_changed {
        global_changed = false;
        // Process blocks in reverse order (approximate reverse postorder).
        for b in (0..n).rev() {
            // live_out[b] = union of live_in[succ] for all successors.
            let mut live_out: HashSet<u8> = HashSet::new();
            for &s in &succs[b] {
                live_out.extend(live_in[s].iter().copied());
            }
            // Add registers used by the terminator.
            terminator_uses(&program.blocks[b].terminator, &mut live_out);

            // Walk ops backwards; for each op, remove dst from live set (it's
            // defined here) and add uses.
            let mut live = live_out.clone();
            for op in program.blocks[b].ops.iter().rev() {
                if let Some(dst) = op_dst_reg(op) {
                    live.remove(&(dst as u8));
                }
                op_uses(op, &mut live);
            }

            // live is now live_in[b].
            if live != live_in[b] {
                live_in[b] = live;
                global_changed = true;
            }
        }
    }

    // Now eliminate dead ops: ops whose dst is never live after the op.
    let mut any_changed = false;
    for b in 0..n {
        let mut live_out: HashSet<u8> = HashSet::new();
        for &s in &succs[b] {
            live_out.extend(live_in[s].iter().copied());
        }
        terminator_uses(&program.blocks[b].terminator, &mut live_out);

        // Walk backwards, tracking live set.
        let ops_len = program.blocks[b].ops.len();
        let mut live = live_out;
        for i in (0..ops_len).rev() {
            let op = program.blocks[b].ops[i].clone();
            let dst = op_dst_reg(&op);
            let is_dead = dst.map_or(false, |d| !live.contains(&(d as u8)));
            let has_side_effects = op_has_side_effects(&op);

            if is_dead && !has_side_effects && !matches!(op, IrOp::Nop) {
                program.blocks[b].ops[i] = IrOp::Nop;
                any_changed = true;
            }

            // Update live set: remove dst, add uses.
            if let Some(d) = dst {
                live.remove(&(d as u8));
            }
            op_uses(&op, &mut live);
        }
    }

    any_changed
}

fn terminator_uses(term: &IrTerminator, live: &mut HashSet<u8>) {
    match term {
        IrTerminator::Branch { dst, src, .. } => {
            live.insert(*dst);
            if let Operand::Reg(r) = src {
                live.insert(*r);
            }
        }
        IrTerminator::Return { value } => {
            live.insert(*value);
        }
        IrTerminator::Jump { .. } | IrTerminator::Trap { .. } => {}
    }
}

fn branch_condition_uses(_cond: &BranchCondition, _live: &mut HashSet<u8>) {
    // BranchCondition variants are unit variants with no fields.
    // The dst/src registers are in IrTerminator::Branch, handled in terminator_uses.
}

fn op_uses(op: &IrOp, live: &mut HashSet<u8>) {
    match op {
        IrOp::Add64 { dst, src }
        | IrOp::Sub64 { dst, src }
        | IrOp::Mul64 { dst, src }
        | IrOp::Div64 { dst, src }
        | IrOp::Mod64 { dst, src }
        | IrOp::And64 { dst, src }
        | IrOp::Or64  { dst, src }
        | IrOp::Xor64 { dst, src }
        | IrOp::Lsh64 { dst, src }
        | IrOp::Rsh64 { dst, src }
        | IrOp::Arsh64 { dst, src }
        | IrOp::Add32 { dst, src }
        | IrOp::Sub32 { dst, src }
        | IrOp::Mul32 { dst, src }
        | IrOp::Div32 { dst, src }
        | IrOp::Mod32 { dst, src }
        | IrOp::And32 { dst, src }
        | IrOp::Or32  { dst, src }
        | IrOp::Xor32 { dst, src }
        | IrOp::Lsh32 { dst, src }
        | IrOp::Rsh32 { dst, src }
        | IrOp::Arsh32 { dst, src } => {
            // dst is both a use and a def for in-place ALU.
            live.insert(*dst);
            if let Operand::Reg(r) = src {
                live.insert(*r);
            }
        }
        IrOp::Mov64 { src, .. } | IrOp::Mov32 { src, .. } => {
            if let Operand::Reg(r) = src {
                live.insert(*r);
            }
        }
        IrOp::Neg64 { dst } | IrOp::Neg32 { dst } => {
            live.insert(*dst);
        }
        IrOp::Store { src, base, .. } => {
            live.insert(*src);
            live.insert(*base);
        }
        IrOp::Load { base, .. } => {
            live.insert(*base);
        }
        IrOp::TrapIfZero { src } => {
            live.insert(*src);
        }
        IrOp::BoundsCheck { addr, .. } => {
            live.insert(*addr);
        }
        IrOp::Syscall { .. } => {}
        IrOp::Phi { sources, .. } => {
            for &(_, s) in sources {
                live.insert(s);
            }
        }
        IrOp::Call { .. } | IrOp::MeterCU { .. } | IrOp::Nop => {}
    }
}

fn op_has_side_effects(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::Store { .. }
            | IrOp::Syscall { .. }
            | IrOp::Call { .. }
            | IrOp::TrapIfZero { .. }
            | IrOp::BoundsCheck { .. }
            | IrOp::MeterCU { .. }
    )
}

// ---------------------------------------------------------------------------
// 1e. Strength Reduction
// ---------------------------------------------------------------------------

fn strength_reduce(program: &mut IrProgram) {
    // Compute liveness after DCE and store it in program.liveness.
    let liveness = compute_liveness(program);
    program.liveness = Some(liveness);

    for block in &mut program.blocks {
        for op in &mut block.ops {
            let replacement = try_strength_reduce(op);
            if let Some(new_op) = replacement {
                *op = new_op;
            }
        }
    }
}

fn try_strength_reduce(op: &IrOp) -> Option<IrOp> {
    match op {
        // --- 64-bit ---

        // mul * 0 → mov dst, 0
        IrOp::Mul64 { dst, src: Operand::Imm(0) } => {
            Some(IrOp::Mov64 { dst: *dst, src: Operand::Imm(0) })
        }
        // mul * 1 → nop
        IrOp::Mul64 { src: Operand::Imm(1), .. } => Some(IrOp::Nop),
        // mul * power-of-two → lsh
        IrOp::Mul64 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::Lsh64 { dst: *dst, src: Operand::Imm(log2(*n)) })
        }
        // div / power-of-two → rsh (unsigned semantics)
        IrOp::Div64 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::Rsh64 { dst: *dst, src: Operand::Imm(log2(*n)) })
        }
        // mod % power-of-two → and (n-1)
        IrOp::Mod64 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::And64 { dst: *dst, src: Operand::Imm(*n - 1) })
        }
        // add + 0 → nop
        IrOp::Add64 { src: Operand::Imm(0), .. } => Some(IrOp::Nop),

        // --- 32-bit ---

        IrOp::Mul32 { dst, src: Operand::Imm(0) } => {
            Some(IrOp::Mov32 { dst: *dst, src: Operand::Imm(0) })
        }
        IrOp::Mul32 { src: Operand::Imm(1), .. } => Some(IrOp::Nop),
        IrOp::Mul32 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::Lsh32 { dst: *dst, src: Operand::Imm(log2(*n)) })
        }
        IrOp::Div32 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::Rsh32 { dst: *dst, src: Operand::Imm(log2(*n)) })
        }
        IrOp::Mod32 { dst, src: Operand::Imm(n) } if is_power_of_two(*n) => {
            Some(IrOp::And32 { dst: *dst, src: Operand::Imm(*n - 1) })
        }
        IrOp::Add32 { src: Operand::Imm(0), .. } => Some(IrOp::Nop),

        _ => None,
    }
}

/// Compute full liveness information across all blocks.
fn compute_liveness(program: &IrProgram) -> LivenessInfo {
    let n = program.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n)
        .map(|b| block_successors(program, b))
        .collect();

    let mut live_in: Vec<HashSet<u8>> = vec![HashSet::new(); n];
    let mut live_out: Vec<HashSet<u8>> = vec![HashSet::new(); n];

    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut out: HashSet<u8> = HashSet::new();
            for &s in &succs[b] {
                out.extend(live_in[s].iter().copied());
            }

            terminator_uses(&program.blocks[b].terminator, &mut out);

            let mut live = out.clone();
            for op in program.blocks[b].ops.iter().rev() {
                if let Some(dst) = op_dst_reg(op) {
                    live.remove(&(dst as u8));
                }
                op_uses(op, &mut live);
            }

            if live != live_in[b] || out != live_out[b] {
                live_in[b] = live;
                live_out[b] = out;
                changed = true;
            }
        }
    }

    LivenessInfo { live_in, live_out }
}
