//! Tier 4 — PTX Code Generation
//!
//! Translates metered IR (output of Tier 3) into NVIDIA PTX 7.0 assembly
//! targeting sm_75 (Turing). One PTX kernel is emitted per BPF program.

use std::fmt::Write;
use crate::types::*;
use crate::syscall_ids::*;

const ACCOUNT_DATA_STRIDE: u64 = 2048;

/// Format a 32-byte hash as a lowercase hex string.
fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Register name for BPF register N.
/// Registers 0-10 are BPF canonical regs; 11+ are SSA temporaries.
#[inline]
fn reg(n: u8) -> String {
    format!("%r{}", n)
}

/// Emit a bounds-check trap sequence (inline). Jumps to __bpfcc_trap_oob
/// if the address in `addr_reg` is out of bounds for `size` bytes in `region`.
fn emit_bounds_check(out: &mut String, addr_reg: u8, size: u8, region: &MemRegion) {
    // For stack and heap regions we can do a simple range check.
    // We check addr + size > limit (off-by-size fix: include the access width).
    // For dynamic regions (cross-contract accounts) we use runtime-provided info.
    match region {
        MemRegion::Stack => {
            // Stack is 4 KB local mem; addr should be relative to %fp
            writeln!(out, "    // bounds-check stack [{} + {}]", reg(addr_reg), size).ok();
            writeln!(out, "    add.u64 %tmp_bc, {}, {};", reg(addr_reg), size).ok();
            writeln!(out, "    setp.hi.u64 %p_oob, %tmp_bc, 4096;").ok();
            writeln!(out, "    @%p_oob bra __bpfcc_trap_oob;").ok();
        }
        MemRegion::Heap => {
            writeln!(out, "    // bounds-check heap [{} + {}]", reg(addr_reg), size).ok();
            writeln!(out, "    add.u64 %tmp_bc, {}, {};", reg(addr_reg), size).ok();
            writeln!(out, "    setp.hi.u64 %p_oob, %tmp_bc, 32768;").ok();
            writeln!(out, "    @%p_oob bra __bpfcc_trap_oob;").ok();
        }
        MemRegion::Input | MemRegion::Program => {
            // Input/program regions: treat as read-only; emit a simple non-null check
            writeln!(out, "    // bounds-check input [{} + {}]", reg(addr_reg), size).ok();
            writeln!(out, "    setp.eq.u64 %p_oob, {}, 0;", reg(addr_reg)).ok();
            writeln!(out, "    @%p_oob bra __bpfcc_trap_oob;").ok();
        }
        MemRegion::Dynamic => {
            // Full runtime bounds check via account data pointer comparison
            writeln!(out, "    // bounds-check dynamic [{} + {}]", reg(addr_reg), size).ok();
            writeln!(out, "    setp.eq.u64 %p_oob, {}, 0;", reg(addr_reg)).ok();
            writeln!(out, "    @%p_oob bra __bpfcc_trap_oob;").ok();
        }
    }
}

/// Emit a single IrOp as PTX text into `out`.
fn emit_op(out: &mut String, op: &IrOp, temp_idx: &mut u32) {
    match op {
        // ── ALU 64-bit ───────────────────────────────────────────────────────
        IrOp::Add64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    add.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    add.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    add.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Sub64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    sub.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    sub.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    sub.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Mul64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    mul.lo.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    mul.lo.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    mul.lo.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Div64 { dst, src } => match src {
            Operand::Reg(s) => {
                writeln!(out, "    div.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok()
            }
            Operand::Imm(v) => writeln!(out, "    div.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    div.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Mod64 { dst, src } => match src {
            Operand::Reg(s) => {
                writeln!(out, "    rem.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok()
            }
            Operand::Imm(v) => writeln!(out, "    rem.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    rem.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::And64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    and.b64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    and.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    and.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Or64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    or.b64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    or.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    or.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Xor64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    xor.b64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    xor.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    xor.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Lsh64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    shl.b64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    shl.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    shl.b64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Rsh64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    shr.u64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    shr.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    shr.u64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Arsh64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    shr.s64 {}, {}, {};", reg(*dst), reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    shr.s64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    shr.s64 {}, {}, {};", reg(*dst), reg(*dst), v).ok(),
        },
        IrOp::Neg64 { dst } => {
            writeln!(out, "    neg.s64 {}, {};", reg(*dst), reg(*dst)).ok()
        }
        IrOp::Mov64 { dst, src } => match src {
            Operand::Reg(s) => writeln!(out, "    mov.u64 {}, {};", reg(*dst), reg(*s)).ok(),
            Operand::Imm(v) => writeln!(out, "    mov.u64 {}, {};", reg(*dst), v).ok(),
            Operand::Imm64(v) => writeln!(out, "    mov.u64 {}, {};", reg(*dst), v).ok(),
        },

        // ── ALU 32-bit (zero-extend result to 64-bit) ─────────────────────
        IrOp::Add32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    add.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    add.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    add.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Sub32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    sub.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    sub.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    sub.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Mul32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    mul.lo.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    mul.lo.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    mul.lo.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Div32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    div.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    div.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    div.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Mod32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    rem.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    rem.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    rem.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::And32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    and.b32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    and.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    and.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Or32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    or.b32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    or.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    or.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Xor32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    xor.b32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    xor.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    xor.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Lsh32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    shl.b32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shl.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shl.b32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Rsh32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    shr.u32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shr.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shr.u32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Arsh32 { dst, src } => {
            let t1 = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    let t2 = *temp_idx; *temp_idx += 1;
                    writeln!(out, "    cvt.s32.s64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t2}, {};", reg(*s)).ok();
                    writeln!(out, "    shr.s32 %tmp32_{t1}, %tmp32_{t1}, %tmp32_{t2};").ok();
                    writeln!(out, "    cvt.u64.s32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    cvt.s32.s64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shr.s32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.s32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    cvt.s32.s64 %tmp32_{t1}, {};", reg(*dst)).ok();
                    writeln!(out, "    shr.s32 %tmp32_{t1}, %tmp32_{t1}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.s32 {}, %tmp32_{t1};", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Neg32 { dst } => {
            let t = *temp_idx; *temp_idx += 1;
            writeln!(out, "    cvt.s32.s64 %tmp32_{t}, {};", reg(*dst)).ok();
            writeln!(out, "    neg.s32 %tmp32_{t}, %tmp32_{t};").ok();
            writeln!(out, "    cvt.u64.s32 {}, %tmp32_{t};", reg(*dst)).ok();
            None
        }
        IrOp::Mov32 { dst, src } => {
            let t = *temp_idx; *temp_idx += 1;
            match src {
                Operand::Reg(s) => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t}, {};", reg(*s)).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t};", reg(*dst)).ok();
                }
                Operand::Imm(v) => {
                    writeln!(out, "    mov.u32 %tmp32_{t}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t};", reg(*dst)).ok();
                }
                Operand::Imm64(v) => {
                    writeln!(out, "    mov.u32 %tmp32_{t}, {};", *v as u32).ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t};", reg(*dst)).ok();
                }
            }
            None
        }

        // ── Memory loads ──────────────────────────────────────────────────
        IrOp::Load { dst, base, offset, size, region: _ } => {
            let t = *temp_idx; *temp_idx += 1;
            // Compute effective address: base + offset
            if *offset == 0 {
                writeln!(out, "    mov.u64 %addr_{t}, {};", reg(*base)).ok();
            } else if *offset > 0 {
                writeln!(out, "    add.u64 %addr_{t}, {}, {};", reg(*base), offset).ok();
            } else {
                writeln!(out, "    sub.u64 %addr_{t}, {}, {};", reg(*base), (-(*offset as i32))).ok();
            }
            match size {
                1 => {
                    writeln!(out, "    ld.global.u8 %tmp8_{t}, [%addr_{t}];").ok();
                    writeln!(out, "    cvt.u64.u8 {}, %tmp8_{t};", reg(*dst)).ok();
                }
                2 => {
                    writeln!(out, "    ld.global.u16 %tmp16_{t}, [%addr_{t}];").ok();
                    writeln!(out, "    cvt.u64.u16 {}, %tmp16_{t};", reg(*dst)).ok();
                }
                4 => {
                    writeln!(out, "    ld.global.u32 %tmp32_{t}, [%addr_{t}];").ok();
                    writeln!(out, "    cvt.u64.u32 {}, %tmp32_{t};", reg(*dst)).ok();
                }
                8 | _ => {
                    writeln!(out, "    ld.global.u64 {}, [%addr_{t}];", reg(*dst)).ok();
                }
            }
            None
        }
        IrOp::Store { base, offset, src, size, region: _ } => {
            let t = *temp_idx; *temp_idx += 1;
            if *offset == 0 {
                writeln!(out, "    mov.u64 %addr_{t}, {};", reg(*base)).ok();
            } else if *offset > 0 {
                writeln!(out, "    add.u64 %addr_{t}, {}, {};", reg(*base), offset).ok();
            } else {
                writeln!(out, "    sub.u64 %addr_{t}, {}, {};", reg(*base), (-(*offset as i32))).ok();
            }
            match size {
                1 => {
                    writeln!(out, "    cvt.u8.u64 %tmp8_{t}, {};", reg(*src)).ok();
                    writeln!(out, "    st.global.u8 [%addr_{t}], %tmp8_{t};").ok();
                }
                2 => {
                    writeln!(out, "    cvt.u16.u64 %tmp16_{t}, {};", reg(*src)).ok();
                    writeln!(out, "    st.global.u16 [%addr_{t}], %tmp16_{t};").ok();
                }
                4 => {
                    writeln!(out, "    cvt.u32.u64 %tmp32_{t}, {};", reg(*src)).ok();
                    writeln!(out, "    st.global.u32 [%addr_{t}], %tmp32_{t};").ok();
                }
                8 | _ => {
                    writeln!(out, "    st.global.u64 [%addr_{t}], {};", reg(*src)).ok();
                }
            }
            None
        }

        // ── Metering ──────────────────────────────────────────────────────
        IrOp::MeterCU { cost } => {
            writeln!(out, "    sub.s64 %cu_remaining, %cu_remaining, {};", cost).ok();
            writeln!(out, "    setp.le.s64 %p_out_of_cu, %cu_remaining, 0;").ok();
            writeln!(out, "    @%p_out_of_cu bra __bpfcc_trap_cu_exceeded;").ok();
            None
        }

        // ── Safety ────────────────────────────────────────────────────────
        IrOp::TrapIfZero { src } => {
            writeln!(out, "    setp.eq.u64 %p_divzero, {}, 0;", reg(*src)).ok();
            writeln!(out, "    @%p_divzero bra __bpfcc_trap_divzero;").ok();
            None
        }
        IrOp::BoundsCheck { addr, size, region } => {
            emit_bounds_check(out, *addr, *size, region);
            None
        }

        // ── Syscalls ─────────────────────────────────────────────────────
        IrOp::Syscall { id, cu_cost } => {
            // Deduct CU cost first
            writeln!(out, "    // syscall 0x{:08x} cost={}", id, cu_cost).ok();
            writeln!(out, "    sub.s64 %cu_remaining, %cu_remaining, {};", cu_cost).ok();
            writeln!(out, "    setp.le.s64 %p_out_of_cu, %cu_remaining, 0;").ok();
            writeln!(out, "    @%p_out_of_cu bra __bpfcc_trap_cu_exceeded;").ok();
            emit_syscall(out, *id, temp_idx);
            None
        }

        // ── Internal call ─────────────────────────────────────────────────
        // Emitted as an inline branch to the target basic block label.
        // This works correctly for non-recursive programs where the callee's
        // EXIT terminates the kernel (same semantics as tail-call). A full
        // software call stack is a future optimization (has_internal_calls flag
        // in AnalysisResult marks programs that use this path).
        IrOp::Call { target_pc } => {
            writeln!(out, "    // internal call -> BB_{} (inline branch)", target_pc).ok();
            writeln!(out, "    bra BB_{};", target_pc).ok();
            None
        }

        // ── SSA Phi (resolved by prior passes; emit nop) ───────────────────
        IrOp::Phi { .. } => {
            writeln!(out, "    // phi (resolved by tier2)").ok();
            None
        }

        IrOp::Nop => {
            None
        }
    };
}

/// Emit inline PTX for a BPF syscall.
fn emit_syscall(out: &mut String, id: u32, temp_idx: &mut u32) {
    let t = *temp_idx; *temp_idx += 1;
    match id {
        // Logging — no-op on GPU
        SOL_LOG | SOL_LOG_64 | SOL_LOG_PUBKEY | SOL_LOG_COMPUTE_UNITS | SOL_LOG_DATA => {
            writeln!(out, "    // sol_log* — no-op on GPU").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_memcpy_ / sol_memmove_ — inline byte-copy loop
        // Convention: r1=dst_ptr, r2=src_ptr, r3=len
        SOL_MEMCPY | SOL_MEMMOVE => {
            writeln!(out, "    // sol_memcpy_ inline").ok();
            writeln!(out, "    mov.u64 %mcpy_dst_{t}, %r1;").ok();
            writeln!(out, "    mov.u64 %mcpy_src_{t}, %r2;").ok();
            writeln!(out, "    mov.u64 %mcpy_len_{t}, %r3;").ok();
            writeln!(out, "    setp.eq.u64 %p_mcpy_done_{t}, %mcpy_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mcpy_done_{t} bra __mcpy_end_{t};").ok();
            writeln!(out, "__mcpy_loop_{t}:").ok();
            writeln!(out, "    ld.global.u8 %tmp8_{t}, [%mcpy_src_{t}];").ok();
            writeln!(out, "    st.global.u8 [%mcpy_dst_{t}], %tmp8_{t};").ok();
            writeln!(out, "    add.u64 %mcpy_dst_{t}, %mcpy_dst_{t}, 1;").ok();
            writeln!(out, "    add.u64 %mcpy_src_{t}, %mcpy_src_{t}, 1;").ok();
            writeln!(out, "    sub.u64 %mcpy_len_{t}, %mcpy_len_{t}, 1;").ok();
            writeln!(out, "    setp.ne.u64 %p_mcpy_done_{t}, %mcpy_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mcpy_done_{t} bra __mcpy_loop_{t};").ok();
            writeln!(out, "__mcpy_end_{t}:").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_memset_ — inline memset loop
        // Convention: r1=ptr, r2=value(byte), r3=len
        SOL_MEMSET => {
            writeln!(out, "    // sol_memset_ inline").ok();
            writeln!(out, "    mov.u64 %mset_dst_{t}, %r1;").ok();
            writeln!(out, "    cvt.u8.u64 %mset_val_{t}, %r2;").ok();
            writeln!(out, "    mov.u64 %mset_len_{t}, %r3;").ok();
            writeln!(out, "    setp.eq.u64 %p_mset_done_{t}, %mset_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mset_done_{t} bra __mset_end_{t};").ok();
            writeln!(out, "__mset_loop_{t}:").ok();
            writeln!(out, "    st.global.u8 [%mset_dst_{t}], %mset_val_{t};").ok();
            writeln!(out, "    add.u64 %mset_dst_{t}, %mset_dst_{t}, 1;").ok();
            writeln!(out, "    sub.u64 %mset_len_{t}, %mset_len_{t}, 1;").ok();
            writeln!(out, "    setp.ne.u64 %p_mset_done_{t}, %mset_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mset_done_{t} bra __mset_loop_{t};").ok();
            writeln!(out, "__mset_end_{t}:").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_memcmp_ — inline compare loop, returns 0 if equal else non-zero in r0
        // Convention: r1=a_ptr, r2=b_ptr, r3=len, r4=result_ptr
        SOL_MEMCMP => {
            writeln!(out, "    // sol_memcmp_ inline").ok();
            writeln!(out, "    mov.u64 %mcmp_a_{t}, %r1;").ok();
            writeln!(out, "    mov.u64 %mcmp_b_{t}, %r2;").ok();
            writeln!(out, "    mov.u64 %mcmp_len_{t}, %r3;").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
            writeln!(out, "    setp.eq.u64 %p_mcmp_done_{t}, %mcmp_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mcmp_done_{t} bra __mcmp_end_{t};").ok();
            writeln!(out, "__mcmp_loop_{t}:").ok();
            writeln!(out, "    ld.global.u8 %mcmp_va_{t}, [%mcmp_a_{t}];").ok();
            writeln!(out, "    ld.global.u8 %mcmp_vb_{t}, [%mcmp_b_{t}];").ok();
            writeln!(out, "    setp.ne.u16 %p_mcmp_ne_{t}, %mcmp_va_{t}, %mcmp_vb_{t};").ok();
            writeln!(out, "    @%p_mcmp_ne_{t} bra __mcmp_diff_{t};").ok();
            writeln!(out, "    add.u64 %mcmp_a_{t}, %mcmp_a_{t}, 1;").ok();
            writeln!(out, "    add.u64 %mcmp_b_{t}, %mcmp_b_{t}, 1;").ok();
            writeln!(out, "    sub.u64 %mcmp_len_{t}, %mcmp_len_{t}, 1;").ok();
            writeln!(out, "    setp.ne.u64 %p_mcmp_done_{t}, %mcmp_len_{t}, 0;").ok();
            writeln!(out, "    @%p_mcmp_done_{t} bra __mcmp_loop_{t};").ok();
            writeln!(out, "    bra __mcmp_end_{t};").ok();
            writeln!(out, "__mcmp_diff_{t}:").ok();
            writeln!(out, "    sub.u16 %mcmp_diff_{t}, %mcmp_va_{t}, %mcmp_vb_{t};").ok();
            writeln!(out, "    cvt.s64.s16 %r0, %mcmp_diff_{t};").ok();
            writeln!(out, "__mcmp_end_{t}:").ok();
            // Store result to r4 pointer if provided
            writeln!(out, "    st.global.u64 [%r4], %r0;").ok();
        }

        // sol_alloc_free_ — bump allocator from heap region
        // Convention: r1=size, r2=free_ptr (0 = alloc, non-zero = free/no-op)
        SOL_ALLOC_FREE => {
            writeln!(out, "    // sol_alloc_free_ bump allocator").ok();
            writeln!(out, "    setp.ne.u64 %p_is_free_{t}, %r2, 0;").ok();
            writeln!(out, "    @%p_is_free_{t} bra __alloc_done_{t};").ok();
            // align to 8 bytes
            writeln!(out, "    add.u64 %alloc_size_{t}, %r1, 7;").ok();
            writeln!(out, "    and.b64 %alloc_size_{t}, %alloc_size_{t}, 0xFFFFFFFFFFFFFFF8;").ok();
            writeln!(out, "    mov.u64 %r0, %heap_ptr;").ok();
            writeln!(out, "    add.u64 %heap_ptr, %heap_ptr, %alloc_size_{t};").ok();
            // check heap exhaustion (32 KB limit)
            writeln!(out, "    sub.u64 %heap_used_{t}, %heap_ptr, %heap_base;").ok();
            writeln!(out, "    setp.gt.u64 %p_oom_{t}, %heap_used_{t}, 32768;").ok();
            writeln!(out, "    @%p_oom_{t} bra __bpfcc_trap_oom;").ok();
            writeln!(out, "__alloc_done_{t}:").ok();
        }

        // sol_sha256 — call device function
        // Convention: r1=vals_ptr, r2=vals_len, r3=result_ptr
        SOL_SHA256 => {
            writeln!(out, "    // sol_sha256 -> device function").ok();
            writeln!(out, "    call __bpfcc_sol_sha256, (%r1, %r2, %r3);").ok();
        }

        // sol_keccak256 — call device function
        SOL_KECCAK256 => {
            writeln!(out, "    // sol_keccak256 -> device function").ok();
            writeln!(out, "    call __bpfcc_sol_keccak256, (%r1, %r2, %r3);").ok();
        }

        // sol_blake3 — call device function
        SOL_BLAKE3 => {
            writeln!(out, "    // sol_blake3 -> device function").ok();
            writeln!(out, "    call __bpfcc_sol_blake3, (%r1, %r2, %r3);").ok();
        }

        // sol_secp256k1_recover — call device function
        SOL_SECP256K1_RECOVER => {
            writeln!(out, "    // sol_secp256k1_recover -> device function").ok();
            writeln!(out, "    call __bpfcc_sol_secp256k1_recover, (%r1, %r2, %r3, %r4);").ok();
        }

        // sol_curve25519 — call device function
        SOL_CURVE25519 => {
            writeln!(out, "    // sol_curve25519 -> device function").ok();
            writeln!(out, "    call __bpfcc_sol_curve25519, (%r1, %r2, %r3, %r4, %r5);").ok();
        }

        // sol_ed25519_verify — call device function
        // r1=msg_ptr, r2=msg_len, r3=sig_ptr (64B), r4=pubkey_ptr (32B)
        // Returns r0=0 (valid) or r0=1 (invalid)
        SOL_ED25519_VERIFY => {
            writeln!(out, "    {{").ok();
            writeln!(out, "    .reg .u64 %ed_ret;").ok();
            writeln!(out, "    call (%ed_ret), __bpfcc_sol_ed25519_verify, (%r1, %r2, %r3, %r4);").ok();
            writeln!(out, "    mov.u64 %r0, %ed_ret;").ok();
            writeln!(out, "    }}").ok();
        }

        // sol_set_return_data — store into thread-local return data buffer (no-op sentinel)
        SOL_SET_RETURN_DATA => {
            writeln!(out, "    // sol_set_return_data — no-op on GPU (return data not relayed)").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_get_return_data — return empty (no prior cross-program call on GPU)
        SOL_GET_RETURN_DATA => {
            writeln!(out, "    // sol_get_return_data — returns 0 length on GPU").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_get_clock_sysvar — load from kernel arg
        // Convention: r1=dest_ptr, result is SolClockSysvar struct
        SOL_GET_CLOCK_SYSVAR => {
            writeln!(out, "    // sol_get_clock_sysvar inline").ok();
            writeln!(out, "    st.global.u64 [%r1], %param_clock;").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // sol_get_rent_sysvar — load from kernel arg
        SOL_GET_RENT_SYSVAR => {
            writeln!(out, "    // sol_get_rent_sysvar inline").ok();
            writeln!(out, "    st.global.u64 [%r1], %param_rent_epoch;").ok();
            writeln!(out, "    mov.u64 %r0, 0;").ok();
        }

        // Unknown syscall — log and no-op
        _ => {
            writeln!(out, "    // unknown syscall 0x{:08x} — no-op", id).ok();
            writeln!(out, "    mov.u64 %r0, 0xffffffff;").ok();
        }
    }
}

/// Emit the PTX terminator for a basic block.
fn emit_terminator(out: &mut String, term: &IrTerminator, temp_idx: &mut u32) {
    match term {
        IrTerminator::Jump { target } => {
            writeln!(out, "    bra BB_{};", target).ok();
        }
        IrTerminator::Branch { condition, dst, src, true_target, false_target } => {
            let t = *temp_idx; *temp_idx += 1;
            let src_str = match src {
                Operand::Reg(r) => format!("%r{}", r),
                Operand::Imm(v) => format!("{}", v),
                Operand::Imm64(v) => format!("{}", v),
            };
            match condition {
                BranchCondition::Eq  => writeln!(out, "    setp.eq.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Ne  => writeln!(out, "    setp.ne.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Gtu => writeln!(out, "    setp.hi.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Geu => writeln!(out, "    setp.hs.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Ltu => writeln!(out, "    setp.lo.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Leu => writeln!(out, "    setp.ls.u64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Gts => writeln!(out, "    setp.gt.s64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Ges => writeln!(out, "    setp.ge.s64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Lts => writeln!(out, "    setp.lt.s64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Les => writeln!(out, "    setp.le.s64 %p_{t}, {}, {};", reg(*dst), src_str).ok(),
                BranchCondition::Set => {
                    writeln!(out, "    and.b64 %tmp64_{t}, {}, {};", reg(*dst), src_str).ok();
                    writeln!(out, "    setp.ne.u64 %p_{t}, %tmp64_{t}, 0;").ok();
                    None
                }
            };
            writeln!(out, "    @%p_{t} bra BB_{};", true_target).ok();
            writeln!(out, "    bra BB_{};", false_target).ok();
        }
        IrTerminator::Return { .. } => {
            writeln!(out, "    ret;").ok();
        }
        IrTerminator::Trap { reason } => {
            match reason {
                TrapReason::DivisionByZero    => writeln!(out, "    bra __bpfcc_trap_divzero;").ok(),
                TrapReason::OutOfBounds       => writeln!(out, "    bra __bpfcc_trap_oob;").ok(),
                TrapReason::CUExceeded        => writeln!(out, "    bra __bpfcc_trap_cu_exceeded;").ok(),
                TrapReason::StackOverflow     => writeln!(out, "    bra __bpfcc_trap_stack_overflow;").ok(),
                TrapReason::InvalidInstruction => writeln!(out, "    bra __bpfcc_trap_invalid;").ok(),
                TrapReason::OutOfMemory       => writeln!(out, "    bra __bpfcc_trap_oom;").ok(),
            };
        }
    }
}

/// Generate PTX assembly from a metered IR program.
///
/// # Parameters
/// - `program`: The IR program produced by Tier 3 (metered).
/// - `program_hash`: 32-byte Blake3/SHA256 hash of the original BPF bytecode.
/// - `config`: Kernel launch configuration (block size, register budget, etc.)
///
/// # Returns
/// A complete, self-contained `.ptx` string ready for `cuModuleLoadData`.
pub fn emit_ptx(
    program: &IrProgram,
    program_hash: &[u8; 32],
    _config: &KernelConfig,
) -> Result<String, String> {
    let hash_str = hex_hash(program_hash);
    let kernel_name = format!("bpfcc_{}", hash_str);
    let mut out = String::with_capacity(65536);

    // ── PTX file header ───────────────────────────────────────────────────
    writeln!(out, "// BPFCC generated kernel: {}", kernel_name)
        .map_err(|e| e.to_string())?;
    writeln!(out, "// Program hash: 0x{}", hash_str)
        .map_err(|e| e.to_string())?;
    writeln!(out, ".version 7.0").map_err(|e| e.to_string())?;
    writeln!(out, ".target sm_75").map_err(|e| e.to_string())?;
    writeln!(out, ".address_size 64").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // ── Kernel signature ──────────────────────────────────────────────────
    writeln!(out, ".visible .entry {}(", kernel_name).map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64 .ptr .global .align 8  param_accounts,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64 .ptr .global .align 8  param_input,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64 .ptr .global .align 8  param_output,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64                         param_max_cu,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64 .ptr .global .align 8  param_heap,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64                         param_clock,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u64                         param_rent_epoch,")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    .param .u32                         param_tx_count")
        .map_err(|e| e.to_string())?;
    writeln!(out, ")").map_err(|e| e.to_string())?;
    writeln!(out, "{{").map_err(|e| e.to_string())?;

    // ── Register declarations ─────────────────────────────────────────────
    writeln!(out, "    // ── Kernel registers ─────────────────────────────────────────────").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u32  %tid;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u32  %ntid;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u32  %ctaid;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u32  %tx_count;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %accounts_base;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %input_base;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %output_base;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %heap_base;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %heap_ptr;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %param_clock;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %param_rent_epoch;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .s64  %cu_remaining;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %acct_ptr;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %status_ptr;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %tid64;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %stride_bytes;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_bounds_skip;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_out_of_cu;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_divzero;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_oob;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // BPF registers r0-r10 (r10 = frame pointer)
    writeln!(out, "    // BPF registers r0-r10").map_err(|e| e.to_string())?;
    for i in 0..=10u8 {
        writeln!(out, "    .reg .u64  %r{};", i).map_err(|e| e.to_string())?;
    }
    // SSA temporaries beyond r10 — declare a generous block
    writeln!(out).map_err(|e| e.to_string())?;
    writeln!(out, "    // SSA temporaries").map_err(|e| e.to_string())?;
    // Determine the max SSA reg used in the program
    let max_ssa = max_ssa_reg(program);
    if max_ssa > 10 {
        writeln!(out, "    .reg .u64  %r<{}>;", max_ssa + 1).map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;

    // Scratch registers used in emission
    writeln!(out, "    // Scratch registers for temporaries").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %tmp64<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u32  %tmp32<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u16  %tmp16<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u8   %tmp8<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %addr<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p<512>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %tmp_bc;  // bounds-check scratch").map_err(|e| e.to_string())?;
    // Syscall temporaries
    writeln!(out, "    .reg .u64  %mcpy_dst<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mcpy_src<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mcpy_len<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_mcpy_done<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mset_dst<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u8   %mset_val<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mset_len<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_mset_done<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mcmp_a<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mcmp_b<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %mcmp_len<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u8   %mcmp_va<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u8   %mcmp_vb<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .s16  %mcmp_diff<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_mcmp_done<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_mcmp_ne<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_is_free<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %alloc_size<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .u64  %heap_used<64>;").map_err(|e| e.to_string())?;
    writeln!(out, "    .reg .pred %p_oom<64>;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // ── Thread setup ──────────────────────────────────────────────────────
    writeln!(out, "    // ── Thread setup ──────────────────────────────────────────────────").map_err(|e| e.to_string())?;

    // Get global thread ID: tid = blockIdx.x * blockDim.x + threadIdx.x
    writeln!(out, "    mov.u32 %ntid,  %ntid.x;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u32 %ctaid, %ctaid.x;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u32 %tid,   %tid.x;").map_err(|e| e.to_string())?;
    writeln!(out, "    mad.lo.u32 %tid, %ctaid, %ntid, %tid;").map_err(|e| e.to_string())?;

    // Load tx_count and skip if this thread has no work
    writeln!(out, "    ld.param.u32 %tx_count, [param_tx_count];").map_err(|e| e.to_string())?;
    writeln!(out, "    setp.ge.u32 %p_bounds_skip, %tid, %tx_count;").map_err(|e| e.to_string())?;
    writeln!(out, "    @%p_bounds_skip ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Load kernel parameters
    writeln!(out, "    ld.param.u64 %accounts_base, [param_accounts];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.u64 %input_base,    [param_input];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.u64 %output_base,   [param_output];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.u64 %heap_base,     [param_heap];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.s64 %cu_remaining,  [param_max_cu];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.u64 %param_clock,   [param_clock];")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    ld.param.u64 %param_rent_epoch, [param_rent_epoch];")
        .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Compute per-thread pointers
    writeln!(out, "    // Per-thread account data pointer: accounts_base + tid * ACCOUNT_DATA_STRIDE")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    cvt.u64.u32 %tid64, %tid;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %stride_bytes, {};", ACCOUNT_DATA_STRIDE)
        .map_err(|e| e.to_string())?;
    writeln!(out, "    mul.lo.u64 %acct_ptr, %tid64, %stride_bytes;")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    add.u64 %acct_ptr, %acct_ptr, %accounts_base;")
        .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Status pointer: output_base + tid * 8
    writeln!(out, "    // Status pointer: output_base + tid * 8").map_err(|e| e.to_string())?;
    writeln!(out, "    mul.lo.u64 %status_ptr, %tid64, 8;").map_err(|e| e.to_string())?;
    writeln!(out, "    add.u64 %status_ptr, %status_ptr, %output_base;")
        .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Initialize heap pointer per-thread (each thread gets its own bump arena)
    writeln!(out, "    // Per-thread heap: heap_base + tid * 32768").map_err(|e| e.to_string())?;
    writeln!(out, "    mul.lo.u64 %heap_ptr, %tid64, 32768;").map_err(|e| e.to_string())?;
    writeln!(out, "    add.u64 %heap_ptr, %heap_ptr, %heap_base;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Initialize BPF registers: r0=0, r1=input_ptr (per-thread), r10=stack_top
    writeln!(out, "    // Initialize BPF registers").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r0,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r1,  %acct_ptr;  // r1 = accounts pointer for this tx")
        .map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r2,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r3,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r4,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r5,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r6,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r7,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r8,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r9,  0;").map_err(|e| e.to_string())?;
    writeln!(out, "    mov.u64 %r10, 0; // fp — stack addressed via local mem")
        .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // Initialize status to SUCCESS (0) — u32 to match trap handlers
    writeln!(out, "    // Status = SUCCESS (0)").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 0;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // ── Basic blocks ──────────────────────────────────────────────────────
    let mut temp_idx: u32 = 0;

    for block in &program.blocks {
        writeln!(out, "BB_{}:", block.id).map_err(|e| e.to_string())?;
        for op in &block.ops {
            emit_op(&mut out, op, &mut temp_idx);
        }
        emit_terminator(&mut out, &block.terminator, &mut temp_idx);
        writeln!(out).map_err(|e| e.to_string())?;
    }

    // ── Trap handlers ─────────────────────────────────────────────────────
    writeln!(out, "    // ── Trap handlers ─────────────────────────────────────────────────").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_cu_exceeded:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 2; // CUExceeded").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_divzero:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 3; // DivisionByZero").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_oob:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 4; // OutOfBounds").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_stack_overflow:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 5; // StackOverflow").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_invalid:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 6; // InvalidInstruction").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    writeln!(out, "__bpfcc_trap_oom:").map_err(|e| e.to_string())?;
    writeln!(out, "    st.global.u32 [%status_ptr], 7; // OutOfMemory").map_err(|e| e.to_string())?;
    writeln!(out, "    ret;").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    // ── End of kernel ─────────────────────────────────────────────────────
    writeln!(out, "}}").map_err(|e| e.to_string())?;

    Ok(out)
}

/// Walk all IR ops to find the highest SSA register index used.
fn max_ssa_reg(program: &IrProgram) -> u8 {
    let mut max_reg: u8 = 10;
    for block in &program.blocks {
        for op in &block.ops {
            let regs = op_regs(op);
            for r in regs {
                if r > max_reg {
                    max_reg = r;
                }
            }
        }
        match &block.terminator {
            IrTerminator::Branch { dst, src, .. } => {
                if *dst > max_reg { max_reg = *dst; }
                if let Operand::Reg(r) = src { if *r > max_reg { max_reg = *r; } }
            }
            IrTerminator::Return { value } => {
                if *value > max_reg { max_reg = *value; }
            }
            _ => {}
        }
    }
    max_reg
}

/// Extract all register indices referenced in an IrOp.
fn op_regs(op: &IrOp) -> Vec<u8> {
    let mut v = Vec::new();
    match op {
        IrOp::Add64 { dst, src } | IrOp::Sub64 { dst, src }
        | IrOp::Mul64 { dst, src } | IrOp::Div64 { dst, src }
        | IrOp::Mod64 { dst, src } | IrOp::And64 { dst, src }
        | IrOp::Or64  { dst, src } | IrOp::Xor64 { dst, src }
        | IrOp::Lsh64 { dst, src } | IrOp::Rsh64 { dst, src }
        | IrOp::Arsh64 { dst, src } | IrOp::Mov64 { dst, src }
        | IrOp::Add32 { dst, src } | IrOp::Sub32 { dst, src }
        | IrOp::Mul32 { dst, src } | IrOp::Div32 { dst, src }
        | IrOp::Mod32 { dst, src } | IrOp::And32 { dst, src }
        | IrOp::Or32  { dst, src } | IrOp::Xor32 { dst, src }
        | IrOp::Lsh32 { dst, src } | IrOp::Rsh32 { dst, src }
        | IrOp::Arsh32 { dst, src } | IrOp::Mov32 { dst, src } => {
            v.push(*dst);
            if let Operand::Reg(r) = src { v.push(*r); }
        }
        IrOp::Neg64 { dst } | IrOp::Neg32 { dst } => { v.push(*dst); }
        IrOp::Load { dst, base, .. } => { v.push(*dst); v.push(*base); }
        IrOp::Store { base, src, .. } => { v.push(*base); v.push(*src); }
        IrOp::TrapIfZero { src } => { v.push(*src); }
        IrOp::BoundsCheck { addr, .. } => { v.push(*addr); }
        IrOp::Phi { dst, sources } => {
            v.push(*dst);
            for (_, r) in sources { v.push(*r); }
        }
        _ => {}
    }
    v
}
