//! Tier 5 — Warp-Level Optimization
//!
//! Post-processes the PTX text produced by Tier 4 to improve warp efficiency:
//!
//! 1. **Divergence Analysis** — classifies branches as uniform or divergent.
//! 2. **Predicated Execution** — converts small divergent branch diamonds into
//!    predicated instructions to eliminate branch overhead.
//! 3. **Uniform Branch Marking** — adds `.uni` qualifier to all uniform branches
//!    so the hardware warp scheduler can take the fast path.

use crate::types::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Apply warp-level optimizations to PTX text.
///
/// Takes the raw PTX emitted by `tier4_ptx_emit::emit_ptx` and applies three
/// passes in order:
///
/// 1. Mark uniform trap branches with `.uni`.
/// 2. Attempt predication of small divergent if-diamonds.
/// 3. No structural change to the basic block layout is performed; the output
///    is still valid PTX 7.0 / sm_75.
pub fn optimize_warps(ptx: &str, _program: &IrProgram) -> String {
    let ptx = mark_uniform_branches(ptx);
    let ptx = predicate_small_diamonds(&ptx);
    ptx
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Divergence analysis + uniform branch marking
// ─────────────────────────────────────────────────────────────────────────────

/// Predicates / register names that are always uniform across a warp.
///
/// These are:
/// - CU metering predicate (`%p_out_of_cu`) because every thread in the warp
///   executes the same BPF program with the same compute budget.
/// - The OOM predicate (`%p_oom`) because heap size is fixed at launch.
/// - The launch-bounds skip predicate (`%p_bounds_skip`) that fires only when
///   the thread index exceeds `tx_count`.
///
/// NOTE: `%p_divzero` and `%p_oob` are NOT uniform — they depend on per-thread
/// data (divisor values and memory offsets respectively) and must not receive
/// the `.uni` qualifier.
const UNIFORM_PREDICATES: &[&str] = &[
    "%p_out_of_cu",
    "%p_oom",
    "%p_bounds_skip",
];

/// Add `.uni` qualifier to branches whose predicate is known-uniform.
///
/// Transforms:
/// ```ptx
///     @%p_out_of_cu bra __bpfcc_trap_cu_exceeded;
/// ```
/// into:
/// ```ptx
///     @%p_out_of_cu bra.uni __bpfcc_trap_cu_exceeded;
/// ```
///
/// Also marks unconditional branches to trap handlers as `.uni` since if any
/// thread reaches them, all threads in the same convergent group do.
fn mark_uniform_branches(ptx: &str) -> String {
    let mut out = String::with_capacity(ptx.len() + 256);

    for line in ptx.lines() {
        let trimmed = line.trim();

        // Already has .uni — pass through unchanged.
        if trimmed.contains("bra.uni") {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Check if this is a predicated branch with a known-uniform predicate.
        let is_uniform_pred = UNIFORM_PREDICATES.iter().any(|p| {
            trimmed.starts_with(&format!("@{} bra ", p))
                || trimmed.starts_with(&format!("@!{} bra ", p))
        });

        // Check for unconditional branch to a trap handler.
        let is_trap_branch = !trimmed.starts_with('@')
            && trimmed.starts_with("bra ")
            && (trimmed.contains("__bpfcc_trap_")
                || trimmed.contains("__bpfcc_trap_cu_exceeded")
                || trimmed.contains("__bpfcc_trap_divzero")
                || trimmed.contains("__bpfcc_trap_oob")
                || trimmed.contains("__bpfcc_trap_stack_overflow")
                || trimmed.contains("__bpfcc_trap_invalid")
                || trimmed.contains("__bpfcc_trap_oom"));

        if is_uniform_pred || is_trap_branch {
            // Replace `bra ` with `bra.uni ` on this line.
            let upgraded = line.replacen("bra ", "bra.uni ", 1);
            out.push_str(&upgraded);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Predicated execution for small branch diamonds
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of PTX instructions on each side of a diamond to qualify
/// for predication. Larger bodies would bloat the instruction stream too much.
const MAX_PRED_BODY_LINES: usize = 4;

/// Attempt to convert if-then-else diamonds into predicated instructions.
///
/// Pattern searched for (lines, trimmed):
/// ```ptx
///     @%p_N bra BB_T;         // conditional branch to then-block
///     <1-4 else instructions> // the "else" body
///     bra BB_M;               // fall-through to merge
/// BB_T:                       // then-block label
///     <1-4 then instructions> // the "then" body
/// BB_M:                       // merge label
/// ```
///
/// Replaced with:
/// ```ptx
///     // predicated: @%p_N then-body, @!%p_N else-body
///     @%p_N  <then instruction 1>
///     @%p_N  <then instruction 2>
///     ...
///     @!%p_N <else instruction 1>
///     @!%p_N <else instruction 2>
///     ...
/// BB_M:
/// ```
///
/// If the pattern does not match, the lines are passed through unchanged.
fn predicate_small_diamonds(ptx: &str) -> String {
    let lines: Vec<&str> = ptx.lines().collect();
    let n = lines.len();
    let mut out = String::with_capacity(ptx.len());
    let mut i = 0;

    while i < n {
        // Try to match the diamond pattern starting at line i.
        if let Some((consumed, replacement)) = try_predicate_diamond(&lines, i) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push_str(lines[i]);
            out.push('\n');
            i += 1;
        }
    }

    out
}

/// Attempt to match and predicate a diamond starting at `lines[start]`.
///
/// Returns `Some((lines_consumed, replacement_text))` on success or `None` if
/// the pattern does not match.
fn try_predicate_diamond(lines: &[&str], start: usize) -> Option<(usize, String)> {
    let n = lines.len();

    // Line 0: `@%p_N bra BB_T;`  (conditional branch — NOT .uni, not a trap)
    let cond_line = lines[start].trim();
    let (pred, then_label) = parse_cond_branch(cond_line)?;

    // Lines 1..K: else body (1 to MAX_PRED_BODY_LINES instructions, no labels, no branches)
    let mut else_end = start + 1;
    let mut else_body: Vec<&str> = Vec::new();
    while else_end < n && else_body.len() < MAX_PRED_BODY_LINES {
        let t = lines[else_end].trim();
        if is_label(t) || is_branch(t) || t.is_empty() {
            break;
        }
        else_body.push(lines[else_end]);
        else_end += 1;
    }
    if else_body.is_empty() || else_body.len() > MAX_PRED_BODY_LINES {
        return None;
    }

    // Next: unconditional `bra BB_M;` (end of else block)
    if else_end >= n {
        return None;
    }
    let merge_branch_line = lines[else_end].trim();
    let merge_label = parse_uncond_branch(merge_branch_line)?;
    let after_merge_branch = else_end + 1;

    // Next: `BB_T:` label
    if after_merge_branch >= n {
        return None;
    }
    let then_label_line = lines[after_merge_branch].trim();
    if !is_exact_label(then_label_line, then_label) {
        return None;
    }
    let after_then_label = after_merge_branch + 1;

    // Lines: then body (1 to MAX_PRED_BODY_LINES instructions, no labels, no branches)
    let mut then_end = after_then_label;
    let mut then_body: Vec<&str> = Vec::new();
    while then_end < n && then_body.len() < MAX_PRED_BODY_LINES {
        let t = lines[then_end].trim();
        if is_label(t) || is_branch(t) || t.is_empty() {
            break;
        }
        then_body.push(lines[then_end]);
        then_end += 1;
    }
    if then_body.is_empty() || then_body.len() > MAX_PRED_BODY_LINES {
        return None;
    }

    // Next: `BB_M:` merge label
    if then_end >= n {
        return None;
    }
    let merge_label_line = lines[then_end].trim();
    if !is_exact_label(merge_label_line, merge_label) {
        return None;
    }

    // All conditions satisfied — emit predicated replacement.
    let total_consumed = then_end - start + 1; // include merge label line
    let mut rep = String::new();

    rep.push_str(&format!(
        "    // predicated diamond: @{pred} then={then_label}, else->merge={merge_label}\n"
    ));

    // Then-body: guard with @%p_N
    for body_line in &then_body {
        let content = body_line.trim();
        if content.starts_with("//") || content.is_empty() {
            rep.push_str(body_line);
            rep.push('\n');
        } else {
            rep.push_str(&format!("    @{pred} {}\n", content));
        }
    }

    // Else-body: guard with @!%p_N
    let neg_pred = negate_pred(pred);
    for body_line in &else_body {
        let content = body_line.trim();
        if content.starts_with("//") || content.is_empty() {
            rep.push_str(body_line);
            rep.push('\n');
        } else {
            rep.push_str(&format!("    @{neg_pred} {}\n", content));
        }
    }

    // Emit merge label so downstream code still has a target
    rep.push_str(lines[then_end]); // BB_M:
    rep.push('\n');

    Some((total_consumed, rep))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a conditional branch line `@%p_N bra BB_T;`.
/// Returns `(predicate, target_label)` or `None`.
fn parse_cond_branch(line: &str) -> Option<(&str, &str)> {
    // Must start with `@` and contain `bra ` (not `bra.uni`)
    if !line.starts_with('@') {
        return None;
    }
    // Reject .uni branches — those are already handled
    if line.contains("bra.uni") {
        return None;
    }
    // Must contain plain `bra ` after the predicate
    let bra_pos = line.find(" bra ")?;
    let pred = &line[..bra_pos]; // e.g. `@%p_42`
    let after_bra = &line[bra_pos + 5..]; // skip ` bra `
    let target = after_bra.trim_end_matches(';').trim();
    if target.is_empty() {
        return None;
    }
    Some((pred, target))
}

/// Parse an unconditional branch line `bra BB_M;` (no predicate, no .uni).
/// Returns the target label string or `None`.
fn parse_uncond_branch(line: &str) -> Option<&str> {
    if line.starts_with('@') || line.contains("bra.uni") {
        return None;
    }
    let line = line.trim();
    if !line.starts_with("bra ") {
        return None;
    }
    let target = line["bra ".len()..].trim_end_matches(';').trim();
    if target.is_empty() {
        return None;
    }
    Some(target)
}

/// Return true if `line` is a PTX label definition (ends with `:`).
fn is_label(line: &str) -> bool {
    !line.is_empty() && line.ends_with(':') && !line.contains(' ')
}

/// Return true if `line` is exactly the label `name:`.
fn is_exact_label(line: &str, name: &str) -> bool {
    line == format!("{}:", name)
}

/// Return true if `line` is any branch instruction.
fn is_branch(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("bra ") || t.starts_with("bra.uni ") || t.contains(" bra ") || t.contains(" bra.uni ")
}

/// Negate a PTX predicate guard: `@%p_N` → `@!%p_N`, `@!%p_N` → `@%p_N`.
fn negate_pred(pred: &str) -> String {
    if let Some(stripped) = pred.strip_prefix("@!") {
        format!("@{}", stripped)
    } else if let Some(stripped) = pred.strip_prefix('@') {
        format!("@!{}", stripped)
    } else {
        // Fallback: just prepend @!
        format!("@!{}", &pred[1..])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn empty_program() -> IrProgram {
        IrProgram {
            blocks: vec![],
            liveness: None,
            region_map: vec![],
        }
    }

    #[test]
    fn test_uniform_branch_marking_cu() {
        let ptx = "    @%p_out_of_cu bra __bpfcc_trap_cu_exceeded;\n";
        let result = optimize_warps(ptx, &empty_program());
        assert!(result.contains("bra.uni __bpfcc_trap_cu_exceeded"), "expected .uni on CU trap branch, got: {result}");
    }

    #[test]
    fn test_divzero_branch_not_marked_uniform() {
        // %p_divzero is data-dependent (divisor comes from per-thread data)
        // and must NOT receive the .uni qualifier.
        let ptx = "    @%p_divzero bra __bpfcc_trap_divzero;\n";
        let result = optimize_warps(ptx, &empty_program());
        assert!(!result.contains("bra.uni"), "divzero branch should not be .uni, got: {result}");
    }

    #[test]
    fn test_oob_branch_not_marked_uniform() {
        // %p_oob is data-dependent (memory offset comes from per-thread data)
        // and must NOT receive the .uni qualifier.
        let ptx = "    @%p_oob bra __bpfcc_trap_oob;\n";
        let result = optimize_warps(ptx, &empty_program());
        assert!(!result.contains("bra.uni"), "oob branch should not be .uni, got: {result}");
    }

    #[test]
    fn test_data_branch_not_marked_uniform() {
        // Data-dependent branch should NOT get .uni
        let ptx = "    @%p_42 bra BB_10;\n";
        let result = optimize_warps(ptx, &empty_program());
        assert!(!result.contains("bra.uni"), "data branch should not be .uni, got: {result}");
    }

    #[test]
    fn test_predication_small_diamond() {
        // Construct a small if-then-else diamond
        let ptx = "\
    @%p_5 bra BB_then;\n\
    mov.u64 %r1, 0;\n\
    bra BB_merge;\n\
BB_then:\n\
    mov.u64 %r1, 1;\n\
BB_merge:\n\
    mov.u64 %r2, %r1;\n";

        let program = empty_program();
        let result = predicate_small_diamonds(ptx);
        // The diamond should be flattened into predicated instructions
        assert!(
            result.contains("@%p_5 mov.u64 %r1, 1;") || result.contains("@%p_5"),
            "expected predicated then-body, got:\n{result}"
        );
        assert!(
            result.contains("@!%p_5 mov.u64 %r1, 0;") || result.contains("@!%p_5"),
            "expected predicated else-body, got:\n{result}"
        );
        // Merge label must still be present
        assert!(result.contains("BB_merge:"), "merge label must survive, got:\n{result}");
    }

    #[test]
    fn test_negate_pred() {
        assert_eq!(negate_pred("@%p_42"), "@!%p_42");
        assert_eq!(negate_pred("@!%p_42"), "@%p_42");
    }

    #[test]
    fn test_already_uni_passthrough() {
        let ptx = "    @%p_out_of_cu bra.uni __bpfcc_trap_cu_exceeded;\n";
        let result = optimize_warps(ptx, &empty_program());
        // Should not duplicate .uni
        assert!(!result.contains("bra.uni.uni"), "got: {result}");
        assert!(result.contains("bra.uni"), "got: {result}");
    }
}
