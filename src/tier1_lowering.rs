use std::collections::HashMap;

use crate::bpf::opcodes::*;
use crate::bpf::{parse_program, BpfInstruction};
use crate::types::{
    AnalysisResult, BranchCondition, IrBlock, IrOp, IrProgram, IrTerminator, MemRegion, Operand,
    Reg, TrapReason,
};

// ── Missing opcode constants (not in bpf::opcodes) ──────────────────────────

const DIV64_IMM: u8 = 0x37;
const DIV64_REG: u8 = 0x3f;
const MOD64_IMM: u8 = 0x97;
const MOD64_REG: u8 = 0x9f;
const SUB32_IMM: u8 = 0x14;
const SUB32_REG: u8 = 0x1c;
const MUL32_IMM: u8 = 0x24;
const MUL32_REG: u8 = 0x2c;
const DIV32_IMM: u8 = 0x34;
const DIV32_REG: u8 = 0x3c;
const OR32_IMM: u8 = 0x44;
const OR32_REG: u8 = 0x4c;
const AND32_IMM: u8 = 0x54;
const AND32_REG: u8 = 0x5c;
const MOD32_IMM: u8 = 0x94;
const MOD32_REG: u8 = 0x9c;
const XOR32_IMM: u8 = 0xa4;
const XOR32_REG: u8 = 0xac;
const ARSH32_IMM: u8 = 0xc4;
const ARSH32_REG: u8 = 0xcc;
const NEG32: u8 = 0x84;
const NEG64: u8 = 0x87;
const LSH32_IMM: u8 = 0x64;
const LSH32_REG: u8 = 0x6c;
const RSH32_IMM: u8 = 0x74;
const RSH32_REG: u8 = 0x7c;
const JSGT_IMM: u8 = 0x65;
const JSGT_REG: u8 = 0x6d;
const JSGE_IMM: u8 = 0x75;
const JSGE_REG: u8 = 0x7d;
const JSLT_IMM: u8 = 0xc5;
const JSLT_REG: u8 = 0xcd;
const JSLE_IMM: u8 = 0xd5;
const JSLE_REG: u8 = 0xdd;
const JNE_IMM: u8 = 0x55;
const JNE_REG: u8 = 0x5d;

// CU costs for syscall stubs (GPU-native helpers)
fn syscall_cu_cost(id: u32) -> u64 {
    match id {
        0x686093bb => 100,  // sol_log_
        0x207559bd => 100,  // sol_log_64_
        0x52ba5096 => 100,  // sol_log_pubkey
        0x7ef088ca => 85,   // sol_sha256 (per 32B)
        0x9c2e990c => 85,   // sol_keccak256
        0x3a6bb866 => 25000, // sol_secp256k1_recover
        0xed255190 => 25_000, // sol_ed25519_verify — same as secp256k1
        0x8e17e9e8 => 85,   // sol_blake3
        0xdf5d20d4 => 1,    // sol_memcpy_
        0x3770fb22 => 1,    // sol_memmove_
        0x05b8de8a => 1,    // sol_memset_
        0x3b32d05b => 1,    // sol_memcmp_
        _ => 10,
    }
}

// ── Memory region inference ──────────────────────────────────────────────────

/// Track which registers hold which memory region at entry to each instruction.
/// r10 = Stack always; r1 at entry = Input; others = Dynamic until proven.
fn infer_region(reg: u8, region_map: &HashMap<u8, MemRegion>) -> MemRegion {
    region_map
        .get(&reg)
        .copied()
        .unwrap_or(MemRegion::Dynamic)
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Tier 1 — BPF → IR Lowering.
///
/// Translates raw BPF bytecode into a structured [`IrProgram`] using the
/// block layout already computed by [`crate::bpfcc::tier0_analysis::analyze`].
pub fn lower(bytecode: &[u8], analysis: &AnalysisResult) -> Result<IrProgram, String> {
    if bytecode.len() % 8 != 0 {
        return Err("bytecode length must be a multiple of 8".into());
    }

    let instructions = parse_program(bytecode)
        .map_err(|e| format!("parse error: {e}"))?;

    let n = instructions.len();
    let num_blocks = analysis.cfg.blocks.len();

    // Build a pc → block_id lookup
    let pc_to_block: HashMap<usize, usize> = analysis
        .cfg
        .blocks
        .iter()
        .map(|b| (b.start_pc, b.id))
        .collect();

    // region_map per block: initialize entry block with r10=Stack, r1=Input
    let mut region_maps: Vec<HashMap<u8, MemRegion>> = vec![HashMap::new(); num_blocks];
    if !region_maps.is_empty() {
        region_maps[0].insert(10, MemRegion::Stack);
        region_maps[0].insert(1, MemRegion::Input);
    }

    let mut ir_blocks: Vec<IrBlock> = Vec::with_capacity(num_blocks);

    for (block_idx, cfg_block) in analysis.cfg.blocks.iter().enumerate() {
        let start = cfg_block.start_pc;
        let end = cfg_block.end_pc;

        let region_map = region_maps[block_idx].clone();
        let mut ops: Vec<IrOp> = Vec::new();

        let mut cursor = start;
        let mut terminator: Option<IrTerminator> = None;

        while cursor < end {
            let insn = &instructions[cursor];

            // Check if this is the last instruction in the block
            let is_last = is_block_terminator(insn.opcode, cursor, end, &instructions);

            if is_last && is_jump_or_exit(insn.opcode) {
                terminator = Some(build_terminator(
                    insn,
                    cursor,
                    end,
                    &pc_to_block,
                    &region_map,
                    &mut ops,
                    block_idx,
                    num_blocks,
                )?);

                break;
            }

            // Translate non-terminator instruction
            let new_ops = translate_insn(insn, cursor, &instructions, &mut region_maps[block_idx], n)?;

            // If LDDW, advance cursor by 2
            let advance = if insn.opcode == LDDW { 2 } else { 1 };
            ops.extend(new_ops);
            cursor += advance;
        }

        // If no explicit terminator found, create a fall-through jump
        let terminator = match terminator {
            Some(t) => t,
            None => {
                // Fall-through to next block
                let next_start = end;
                if let Some(&next_id) = pc_to_block.get(&next_start) {
                    IrTerminator::Jump { target: next_id }
                } else if end >= n {
                    // End of program
                    IrTerminator::Return { value: 0 }
                } else {
                    IrTerminator::Trap {
                        reason: TrapReason::InvalidInstruction,
                    }
                }
            }
        };

        ir_blocks.push(IrBlock {
            id: block_idx,
            entry_pc: start,
            ops,
            terminator,
        });

        // Propagate region map to successor blocks (simple forward propagation)
        let current_regions: Vec<(u8, MemRegion)> = region_maps[block_idx]
            .iter()
            .map(|(&r, &m)| (r, m))
            .collect();
        for &succ in &analysis.cfg.successors[block_idx] {
            for &(reg, region) in &current_regions {
                region_maps[succ].entry(reg).or_insert(region);
            }
        }
    }

    // Build per-block region_map snapshots for the IrProgram
    let region_map_snapshots: Vec<HashMap<u8, MemRegion>> = region_maps;

    Ok(IrProgram {
        blocks: ir_blocks,
        liveness: None,
        region_map: region_map_snapshots,
    })
}

// ── Instruction translation ───────────────────────────────────────────────────

fn translate_insn(
    insn: &BpfInstruction,
    pc: usize,
    _instructions: &[BpfInstruction],
    region_map: &mut HashMap<u8, MemRegion>,
    _n: usize,
) -> Result<Vec<IrOp>, String> {
    let dst = insn.dst;
    let src = insn.src;
    let imm = insn.imm as i64;

    let mut ops = Vec::new();

    match insn.opcode {
        // ── ALU 64-bit ────────────────────────────────────────────────────────
        ADD64_IMM  => ops.push(IrOp::Add64  { dst, src: Operand::Imm(imm) }),
        ADD64_REG  => ops.push(IrOp::Add64  { dst, src: Operand::Reg(src) }),
        SUB64_IMM  => ops.push(IrOp::Sub64  { dst, src: Operand::Imm(imm) }),
        SUB64_REG  => ops.push(IrOp::Sub64  { dst, src: Operand::Reg(src) }),
        MUL64_IMM  => ops.push(IrOp::Mul64  { dst, src: Operand::Imm(imm) }),
        MUL64_REG  => ops.push(IrOp::Mul64  { dst, src: Operand::Reg(src) }),

        DIV64_IMM => {
            if imm == 0 {
                // Division by literal 0 always traps — load 0 into r0 and trap
                ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(0) });
                ops.push(IrOp::TrapIfZero { src: 0 });
            } else {
                ops.push(IrOp::Div64 { dst, src: Operand::Imm(imm) });
            }
        }
        DIV64_REG => {
            ops.push(IrOp::TrapIfZero { src });
            ops.push(IrOp::Div64 { dst, src: Operand::Reg(src) });
        }
        MOD64_IMM => {
            if imm == 0 {
                ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(0) });
                ops.push(IrOp::TrapIfZero { src: 0 });
            } else {
                ops.push(IrOp::Mod64 { dst, src: Operand::Imm(imm) });
            }
        }
        MOD64_REG => {
            ops.push(IrOp::TrapIfZero { src });
            ops.push(IrOp::Mod64 { dst, src: Operand::Reg(src) });
        }

        OR64_IMM   => ops.push(IrOp::Or64   { dst, src: Operand::Imm(imm) }),
        OR64_REG   => ops.push(IrOp::Or64   { dst, src: Operand::Reg(src) }),
        AND64_IMM  => ops.push(IrOp::And64  { dst, src: Operand::Imm(imm) }),
        AND64_REG  => ops.push(IrOp::And64  { dst, src: Operand::Reg(src) }),
        LSH64_IMM  => ops.push(IrOp::Lsh64  { dst, src: Operand::Imm(imm) }),
        LSH64_REG  => ops.push(IrOp::Lsh64  { dst, src: Operand::Reg(src) }),
        RSH64_IMM  => ops.push(IrOp::Rsh64  { dst, src: Operand::Imm(imm) }),
        RSH64_REG  => ops.push(IrOp::Rsh64  { dst, src: Operand::Reg(src) }),
        ARSH64_IMM => ops.push(IrOp::Arsh64 { dst, src: Operand::Imm(imm) }),
        ARSH64_REG => ops.push(IrOp::Arsh64 { dst, src: Operand::Reg(src) }),
        XOR64_IMM  => ops.push(IrOp::Xor64  { dst, src: Operand::Imm(imm) }),
        XOR64_REG  => ops.push(IrOp::Xor64  { dst, src: Operand::Reg(src) }),
        NEG64      => ops.push(IrOp::Neg64  { dst }),
        MOV64_IMM  => {
            ops.push(IrOp::Mov64 { dst, src: Operand::Imm(imm) });
            // Clear region for this reg — it's now a plain value
            region_map.remove(&dst);
        }
        MOV64_REG  => {
            // Propagate region from src to dst
            let r = infer_region(src, region_map);
            if r != MemRegion::Dynamic {
                region_map.insert(dst, r);
            } else {
                region_map.remove(&dst);
            }
            ops.push(IrOp::Mov64 { dst, src: Operand::Reg(src) });
        }

        // ── ALU 32-bit ────────────────────────────────────────────────────────
        ADD32_IMM  => ops.push(IrOp::Add32  { dst, src: Operand::Imm(imm) }),
        ADD32_REG  => ops.push(IrOp::Add32  { dst, src: Operand::Reg(src) }),
        SUB32_IMM  => ops.push(IrOp::Sub32  { dst, src: Operand::Imm(imm) }),
        SUB32_REG  => ops.push(IrOp::Sub32  { dst, src: Operand::Reg(src) }),
        MUL32_IMM  => ops.push(IrOp::Mul32  { dst, src: Operand::Imm(imm) }),
        MUL32_REG  => ops.push(IrOp::Mul32  { dst, src: Operand::Reg(src) }),

        DIV32_IMM => {
            if imm == 0 {
                ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(0) });
                ops.push(IrOp::TrapIfZero { src: 0 });
            } else {
                ops.push(IrOp::Div32 { dst, src: Operand::Imm(imm) });
            }
        }
        DIV32_REG => {
            ops.push(IrOp::TrapIfZero { src });
            ops.push(IrOp::Div32 { dst, src: Operand::Reg(src) });
        }
        MOD32_IMM => {
            if imm == 0 {
                ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(0) });
                ops.push(IrOp::TrapIfZero { src: 0 });
            } else {
                ops.push(IrOp::Mod32 { dst, src: Operand::Imm(imm) });
            }
        }
        MOD32_REG => {
            ops.push(IrOp::TrapIfZero { src });
            ops.push(IrOp::Mod32 { dst, src: Operand::Reg(src) });
        }

        OR32_IMM   => ops.push(IrOp::Or32   { dst, src: Operand::Imm(imm) }),
        OR32_REG   => ops.push(IrOp::Or32   { dst, src: Operand::Reg(src) }),
        AND32_IMM  => ops.push(IrOp::And32  { dst, src: Operand::Imm(imm) }),
        AND32_REG  => ops.push(IrOp::And32  { dst, src: Operand::Reg(src) }),
        LSH32_IMM  => ops.push(IrOp::Lsh32  { dst, src: Operand::Imm(imm) }),
        LSH32_REG  => ops.push(IrOp::Lsh32  { dst, src: Operand::Reg(src) }),
        RSH32_IMM  => ops.push(IrOp::Rsh32  { dst, src: Operand::Imm(imm) }),
        RSH32_REG  => ops.push(IrOp::Rsh32  { dst, src: Operand::Reg(src) }),
        ARSH32_IMM => ops.push(IrOp::Arsh32 { dst, src: Operand::Imm(imm) }),
        ARSH32_REG => ops.push(IrOp::Arsh32 { dst, src: Operand::Reg(src) }),
        XOR32_IMM  => ops.push(IrOp::Xor32  { dst, src: Operand::Imm(imm) }),
        XOR32_REG  => ops.push(IrOp::Xor32  { dst, src: Operand::Reg(src) }),
        NEG32      => ops.push(IrOp::Neg32  { dst }),
        MOV32_IMM  => {
            region_map.remove(&dst);
            ops.push(IrOp::Mov32 { dst, src: Operand::Imm(imm) });
        }
        MOV32_REG  => {
            let r = infer_region(src, region_map);
            if r != MemRegion::Dynamic {
                region_map.insert(dst, r);
            } else {
                region_map.remove(&dst);
            }
            ops.push(IrOp::Mov32 { dst, src: Operand::Reg(src) });
        }

        // ── LDDW — 64-bit immediate (2 instruction slots) ────────────────────
        LDDW => {
            if pc + 1 >= instructions.len() {
                return Err(format!("LDDW at pc={pc} missing second slot"));
            }
            let next = &instructions[pc + 1];
            let low  = insn.imm as u32 as u64;
            let high = (next.imm as u32 as u64) << 32;
            let val  = low | high;
            region_map.remove(&dst);
            ops.push(IrOp::Mov64 { dst, src: Operand::Imm64(val) });
        }

        // ── Memory loads ──────────────────────────────────────────────────────
        LDXB => {
            let region = infer_region(src, region_map);
            ops.push(IrOp::Load { dst, base: src, offset: insn.offset, size: 1, region });
        }
        LDXH => {
            let region = infer_region(src, region_map);
            ops.push(IrOp::Load { dst, base: src, offset: insn.offset, size: 2, region });
        }
        LDXW => {
            let region = infer_region(src, region_map);
            ops.push(IrOp::Load { dst, base: src, offset: insn.offset, size: 4, region });
        }
        LDXDW => {
            let region = infer_region(src, region_map);
            ops.push(IrOp::Load { dst, base: src, offset: insn.offset, size: 8, region });
        }

        // ── Memory stores (from register) ────────────────────────────────────
        STXB => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src, size: 1, region });
        }
        STXH => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src, size: 2, region });
        }
        STXW => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src, size: 4, region });
        }
        STXDW => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src, size: 8, region });
        }

        // ── Memory stores (immediate) — use src=0 as placeholder ─────────────
        STB => {
            let region = infer_region(dst, region_map);
            // Store the immediate via a temp reg write — represent as Nop+Store
            // In practice, PTX emit will inline the immediate.  Use src=0 (r0).
            ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(imm) });
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src: 0, size: 1, region });
        }
        STH => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(imm) });
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src: 0, size: 2, region });
        }
        STW => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(imm) });
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src: 0, size: 4, region });
        }
        STDW => {
            let region = infer_region(dst, region_map);
            ops.push(IrOp::Mov64 { dst: 0, src: Operand::Imm(imm) });
            ops.push(IrOp::Store { base: dst, offset: insn.offset, src: 0, size: 8, region });
        }

        // ── Syscall / internal call ───────────────────────────────────────────
        CALL => {
            let syscall_id = insn.imm as u32;
            let cu_cost = syscall_cu_cost(syscall_id);
            ops.push(IrOp::Syscall { id: syscall_id, cu_cost });
        }

        // ── Unconditional jump / conditional jumps / exit are handled as
        //    terminators elsewhere — if we reach here it means they are in
        //    the middle of a block (should not happen for well-formed CFG).
        JA | EXIT => {
            // Will be handled as terminator; treat as Nop if seen here.
            ops.push(IrOp::Nop);
        }

        op if is_conditional_jump_op(op) => {
            ops.push(IrOp::Nop);
        }

        _ => {
            return Err(format!("unknown opcode 0x{:02x} at pc={pc}", insn.opcode));
        }
    }

    Ok(ops)
}

// ── Terminator construction ──────────────────────────────────────────────────

fn build_terminator(
    insn: &BpfInstruction,
    pc: usize,
    _block_end: usize,
    pc_to_block: &HashMap<usize, usize>,
    _region_map: &HashMap<u8, MemRegion>,
    ops: &mut Vec<IrOp>,
    _block_idx: usize,
    _num_blocks: usize,
) -> Result<IrTerminator, String> {
    let dst = insn.dst;
    let src_reg = insn.src;
    let imm = insn.imm as i64;

    match insn.opcode {
        EXIT => Ok(IrTerminator::Return { value: 0 }),

        JA => {
            let target_pc = jump_target(pc, insn);
            let target = pc_to_block
                .get(&target_pc)
                .copied()
                .ok_or_else(|| format!("JA at pc={pc} targets unmapped pc={target_pc}"))?;
            Ok(IrTerminator::Jump { target })
        }

        CALL => {
            // In this lowering, CALL to an internal function (non-syscall offset)
            // is treated as a fall-through call; the actual target is the
            // return address (pc+1).  Syscalls were already handled in translate_insn.
            let syscall_id = insn.imm as u32;
            let cu_cost = syscall_cu_cost(syscall_id);
            ops.push(IrOp::Syscall { id: syscall_id, cu_cost });
            let next_pc = pc + 1;
            let target = pc_to_block
                .get(&next_pc)
                .copied()
                .ok_or_else(|| format!("CALL at pc={pc}: no successor block at pc={next_pc}"))?;
            Ok(IrTerminator::Jump { target })
        }

        op => {
            // Conditional jump
            let (condition, src_operand) = decode_branch_condition(op, src_reg, imm)?;
            let true_pc = jump_target(pc, insn);
            let false_pc = pc + 1;

            let true_target = pc_to_block
                .get(&true_pc)
                .copied()
                .ok_or_else(|| {
                    format!("branch at pc={pc} true-target pc={true_pc} not in block map")
                })?;
            let false_target = pc_to_block
                .get(&false_pc)
                .copied()
                .unwrap_or_else(|| {
                    // If false_pc is beyond the last block, it's an implicit exit.
                    // Caller should handle; we return 0 (entry block) as sentinel.
                    0
                });

            Ok(IrTerminator::Branch {
                condition,
                dst,
                src: src_operand,
                true_target,
                false_target,
            })
        }
    }
}

fn decode_branch_condition(
    opcode: u8,
    src_reg: Reg,
    imm: i64,
) -> Result<(BranchCondition, Operand), String> {
    let (cond, operand) = match opcode {
        JEQ_IMM  => (BranchCondition::Eq,  Operand::Imm(imm)),
        JEQ_REG  => (BranchCondition::Eq,  Operand::Reg(src_reg)),
        JNE_IMM  => (BranchCondition::Ne,  Operand::Imm(imm)),
        JNE_REG  => (BranchCondition::Ne,  Operand::Reg(src_reg)),
        JGT_IMM  => (BranchCondition::Gtu, Operand::Imm(imm)),
        JGT_REG  => (BranchCondition::Gtu, Operand::Reg(src_reg)),
        JGE_IMM  => (BranchCondition::Geu, Operand::Imm(imm)),
        JGE_REG  => (BranchCondition::Geu, Operand::Reg(src_reg)),
        JLT_IMM  => (BranchCondition::Ltu, Operand::Imm(imm)),
        JLT_REG  => (BranchCondition::Ltu, Operand::Reg(src_reg)),
        JLE_IMM  => (BranchCondition::Leu, Operand::Imm(imm)),
        JLE_REG  => (BranchCondition::Leu, Operand::Reg(src_reg)),
        JSGT_IMM => (BranchCondition::Gts, Operand::Imm(imm)),
        JSGT_REG => (BranchCondition::Gts, Operand::Reg(src_reg)),
        JSGE_IMM => (BranchCondition::Ges, Operand::Imm(imm)),
        JSGE_REG => (BranchCondition::Ges, Operand::Reg(src_reg)),
        JSLT_IMM => (BranchCondition::Lts, Operand::Imm(imm)),
        JSLT_REG => (BranchCondition::Lts, Operand::Reg(src_reg)),
        JSLE_IMM => (BranchCondition::Les, Operand::Imm(imm)),
        JSLE_REG => (BranchCondition::Les, Operand::Reg(src_reg)),
        _ => return Err(format!("unknown conditional jump opcode 0x{opcode:02x}")),
    };
    Ok((cond, operand))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn jump_target(pc: usize, insn: &BpfInstruction) -> usize {
    (pc as i64 + 1 + insn.offset as i64) as usize
}

fn is_jump_or_exit(opcode: u8) -> bool {
    matches!(opcode, JA | EXIT | CALL) || is_conditional_jump_op(opcode)
}

fn is_conditional_jump_op(op: u8) -> bool {
    matches!(
        op,
        JEQ_IMM
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
            | JSLT_IMM
            | JSLT_REG
            | JSLE_IMM
            | JSLE_REG
    )
}

/// Determine whether the instruction at `cursor` is the terminator of [start, end).
/// Accounts for LDDW double-slots.
fn is_block_terminator(
    opcode: u8,
    cursor: usize,
    _block_end: usize,
    _instructions: &[BpfInstruction],
) -> bool {
    if !is_jump_or_exit(opcode) {
        return false;
    }
    // This is only the terminator if nothing comes after it in this block.
    let next = if opcode == LDDW { cursor + 2 } else { cursor + 1 };
    next >= block_end
}

// ── Placeholder stubs for tier-2 through tier-8 and supporting modules ───────
// These are empty shells so that mod.rs can declare them without compile errors
// until the full implementations are added.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier0_analysis::analyze;

    fn simple_bytecode() -> Vec<u8> {
        // mov64 r0, 42; exit
        vec![
            0xb7, 0x00, 0x00, 0x00, 42, 0, 0, 0,
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,
        ]
    }

    #[test]
    fn test_lower_simple() {
        let bc = simple_bytecode();
        let analysis = analyze(&bc).unwrap();
        assert!(analysis.gpu_eligible);

        let prog = lower(&bc, &analysis).unwrap();
        assert_eq!(prog.blocks.len(), 1);

        let block = &prog.blocks[0];
        assert_eq!(block.ops.len(), 1);
        assert!(matches!(block.ops[0], IrOp::Mov64 { dst: 0, src: Operand::Imm(42) }));
        assert!(matches!(block.terminator, IrTerminator::Return { value: 0 }));
    }

    #[test]
    fn test_lower_div_safety() {
        // mov64 r1, 5; div64 r0, r1; exit
        let bc = vec![
            0xb7, 0x01, 0x00, 0x00, 5, 0, 0, 0,   // mov64 r1, 5
            0x3f, 0x10, 0x00, 0x00, 0, 0, 0, 0,   // div64 r0, r1
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,   // exit
        ];
        let analysis = analyze(&bc).unwrap();
        let prog = lower(&bc, &analysis).unwrap();
        let block = &prog.blocks[0];

        // Should have: Mov64, TrapIfZero, Div64
        let has_trap = block.ops.iter().any(|op| matches!(op, IrOp::TrapIfZero { .. }));
        assert!(has_trap, "division safety check not inserted");
    }
}
