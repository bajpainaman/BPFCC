use crate::types::*;

/// Known Solana account serialization field offsets and sizes.
const FIELD_IS_DUPLICATE: (u64, u8, &str) = (0, 1, "is_duplicate");
const FIELD_IS_SIGNER:    (u64, u8, &str) = (1, 1, "is_signer");
const FIELD_IS_WRITABLE:  (u64, u8, &str) = (2, 1, "is_writable");
const FIELD_EXECUTABLE:   (u64, u8, &str) = (3, 1, "executable");
const FIELD_KEY:          (u64, u8, &str) = (8,  32, "key");
const FIELD_OWNER:        (u64, u8, &str) = (40, 32, "owner");
const FIELD_LAMPORTS:     (u64, u8, &str) = (72, 8,  "lamports");
const FIELD_DATA_LEN:     (u64, u8, &str) = (80, 8,  "data_len");

const KNOWN_FIELDS: &[(u64, u8, &str)] = &[
    FIELD_IS_DUPLICATE,
    FIELD_IS_SIGNER,
    FIELD_IS_WRITABLE,
    FIELD_EXECUTABLE,
    FIELD_KEY,
    FIELD_OWNER,
    FIELD_LAMPORTS,
    FIELD_DATA_LEN,
];

/// A field that should be transposed to Structure-of-Arrays layout.
#[derive(Debug, Clone)]
pub struct SoaField {
    /// Offset within the account struct.
    pub offset: u64,
    /// Size of the field in bytes.
    pub size: u8,
    /// Name for PTX label (e.g., "lamports", "data_len").
    pub name: String,
}

/// Plan describing which account fields to transpose for memory coalescing.
#[derive(Debug, Clone, Default)]
pub struct TranspositionPlan {
    /// Fields that should be transposed to SoA layout.
    pub fields: Vec<SoaField>,
    /// Whether to use SoA for this program (false if no benefit).
    pub use_soa: bool,
}

/// Optimize memory access patterns for GPU cache-line coalescing.
/// Returns modified PTX with SoA-optimized loads and the transposition plan.
pub fn optimize_memory(ptx: &str, program: &IrProgram) -> (String, TranspositionPlan) {
    let plan = build_transposition_plan(program);
    if !plan.use_soa {
        return (ptx.to_string(), plan);
    }
    let new_ptx = apply_soa_rewrites(ptx, &plan);
    (new_ptx, plan)
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Scan IR to find which known account fields are accessed by multiple threads
/// (i.e., via loads whose base register originates from an account array pointer).
fn build_transposition_plan(program: &IrProgram) -> TranspositionPlan {
    // Count how many Load operations touch each known field offset.
    let mut offset_access_count: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();

    for block in &program.blocks {
        for op in &block.ops {
            if let IrOp::Load { offset, region, .. } = op {
                // Focus on Input memory (account data) and dynamic accesses.
                if *region == MemRegion::Input || *region == MemRegion::Dynamic {
                    let abs_offset = offset.unsigned_abs() as u64;
                    *offset_access_count.entry(abs_offset).or_insert(0) += 1;
                }
            }
        }
    }

    // Identify fields that are accessed enough times to benefit from SoA.
    // Threshold: accessed in at least 2 distinct IR loads (worth coalescing).
    const SOA_THRESHOLD: usize = 2;

    let mut coalescing_fields: Vec<SoaField> = Vec::new();
    for &(field_offset, field_size, field_name) in KNOWN_FIELDS {
        // Check if any accesses fall within this field's byte range.
        let field_end = field_offset + field_size as u64;
        let total_accesses: usize = offset_access_count
            .iter()
            .filter(|(&off, _)| off >= field_offset && off < field_end)
            .map(|(_, &cnt)| cnt)
            .sum();

        if total_accesses >= SOA_THRESHOLD {
            coalescing_fields.push(SoaField {
                offset: field_offset,
                size: field_size,
                name: field_name.to_string(),
            });
        }
    }

    // Also capture the lamports field unconditionally when any account load
    // is detected — it is the most commonly accessed field in BPF programs.
    let has_any_input_load = program.blocks.iter().any(|b| {
        b.ops.iter().any(|op| {
            matches!(op, IrOp::Load { region: MemRegion::Input, .. }
                      | IrOp::Load { region: MemRegion::Dynamic, .. })
        })
    });

    if has_any_input_load && !coalescing_fields.iter().any(|f| f.offset == FIELD_LAMPORTS.0) {
        coalescing_fields.push(SoaField {
            offset: FIELD_LAMPORTS.0,
            size:   FIELD_LAMPORTS.1,
            name:   FIELD_LAMPORTS.2.to_string(),
        });
    }

    let use_soa = !coalescing_fields.is_empty();
    TranspositionPlan {
        fields: coalescing_fields,
        use_soa,
    }
}

// ---------------------------------------------------------------------------
// PTX Transformation
// ---------------------------------------------------------------------------

/// Rewrite scattered account-field loads to coalesced SoA loads in PTX text.
///
/// Pattern replaced per field (example for lamports at offset 72, size 8):
///
///   Before:
///     ld.global.u64 %rN, [%account_base + 72];
///
///   After:
///     mul.lo.u64    %soa_addr_lamports, %tid64, 8;
///     add.u64       %soa_addr_lamports, %soa_addr_lamports, %lamports_soa_base;
///     ld.global.u64 %rN, [%soa_addr_lamports];
fn apply_soa_rewrites(ptx: &str, plan: &TranspositionPlan) -> String {
    let mut result = ptx.to_string();

    // Inject a %tid64 computation declaration after the first `.reg .u64` block
    // if not already present.
    if !result.contains("%tid64") {
        result = inject_tid64_reg(&result);
    }

    for field in &plan.fields {
        let ld_instr = load_instr_for_size(field.size);
        let soa_addr_reg = format!("%soa_addr_{}", field.name);
        let soa_base_reg = format!("%{}_soa_base", field.name);

        // Match patterns like:  ld.global.uXX %rN, [%account_base + OFFSET];
        // where OFFSET falls within the field range.
        let before_pattern = format!(
            "ld.global.{} %r", ld_instr,
        );

        // Build a line-by-line replacement.
        let mut new_lines: Vec<String> = Vec::new();
        for line in result.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(&format!("ld.global.{}", ld_instr)) {
                // Check if the operand references account_base with this field's offset.
                if let Some(replacement) =
                    try_rewrite_load_line(trimmed, field, &soa_addr_reg, &soa_base_reg)
                {
                    // Preserve the original indentation.
                    let indent = leading_whitespace(line);
                    for repl_line in replacement.lines() {
                        new_lines.push(format!("{}{}", indent, repl_line));
                    }
                    continue;
                }
            }
            // Keep lines that reference the before_pattern name for diagnostics.
            let _ = &before_pattern; // suppress unused warning
            new_lines.push(line.to_string());
        }
        result = new_lines.join("\n");
    }

    // Append SoA address register declarations and init block before first `.entry`.
    result = inject_soa_register_decls(&result, plan);

    result
}

/// Attempt to rewrite a single `ld.global.*` line that accesses a known SoA field.
/// Returns `None` if the line does not match the expected pattern.
fn try_rewrite_load_line(
    line: &str,
    field: &SoaField,
    soa_addr_reg: &str,
    soa_base_reg: &str,
) -> Option<String> {
    // Expected pattern fragments:
    //   ld.global.u64 %r5, [%account_base + 72];
    // We look for `account_base` or `acct_base` and the field offset.
    if !line.contains("account_base") && !line.contains("acct_base") {
        return None;
    }

    let offset_str = field.offset.to_string();
    if !line.contains(&offset_str) {
        return None;
    }

    // Extract the destination register (%rN portion).
    let dst_reg = extract_dst_reg(line)?;
    let ld_type = load_instr_for_size(field.size);
    let field_size = field.size as u64;

    // Emit coalesced SoA load sequence.
    let replacement = format!(
        "mul.lo.u64 {soa_addr}, %tid64, {size};\n\
         add.u64    {soa_addr}, {soa_addr}, {soa_base};\n\
         ld.global.{ld} {dst}, [{soa_addr}];",
        soa_addr = soa_addr_reg,
        size      = field_size,
        soa_base  = soa_base_reg,
        ld        = ld_type,
        dst       = dst_reg,
    );

    Some(replacement)
}

/// Extract the destination register token (e.g., `%r5`) from a PTX load line.
fn extract_dst_reg(line: &str) -> Option<&str> {
    // e.g. "ld.global.u64 %r5, [...];"
    let after_instr = line.splitn(2, ' ').nth(1)?.trim();
    let dst = after_instr.splitn(2, ',').next()?.trim();
    if dst.starts_with('%') {
        Some(dst)
    } else {
        None
    }
}

/// Return the PTX load type suffix for a given field size.
fn load_instr_for_size(size: u8) -> &'static str {
    match size {
        1 => "u8",
        2 => "u16",
        4 => "u32",
        8 => "u64",
        _ => "u64",
    }
}

/// Return the leading whitespace of a line for indentation preservation.
fn leading_whitespace(line: &str) -> &str {
    let trimmed_len = line.trim_start().len();
    &line[..line.len() - trimmed_len]
}

/// Inject a `.reg .u64 %tid64;` declaration and a `mov.u32` / `cvt` init into
/// the PTX .reg block if not already present.
fn inject_tid64_reg(ptx: &str) -> String {
    // Insert after the first `.reg .u64` directive, or before the first `.entry`.
    let tid_decl = ".reg .u64 %tid64;\n.reg .u32 %r_tid32;\n";
    let tid_init = "\n    // SoA coalescing: compute lane index\n    \
                    mov.u32  %r_tid32, %tid.x;\n    \
                    cvt.u64.u32 %tid64, %r_tid32;\n";

    let mut out = String::with_capacity(ptx.len() + tid_decl.len() + tid_init.len() + 64);
    let mut injected_decl = false;
    let mut injected_init = false;

    for line in ptx.lines() {
        out.push_str(line);
        out.push('\n');

        // Inject declaration after the first .reg .u64 block.
        if !injected_decl && line.trim().starts_with(".reg .u64") {
            out.push_str(tid_decl);
            injected_decl = true;
        }

        // Inject init after the opening `{` of a kernel body.
        if !injected_init && line.trim() == "{" {
            out.push_str(tid_init);
            out.push('\n');
            injected_init = true;
        }
    }

    if !injected_decl {
        // Prepend before any .entry if we never found a .reg .u64 block.
        out = format!("{}{}", tid_decl, out);
    }

    out
}

/// Inject `.reg .u64 %soa_addr_FIELD;`, `.reg .u64 %FIELD_soa_base;`, and
/// `.param .u64 %FIELD_soa_base_param;` declarations for every SoA field,
/// plus bridge `ld.param` instructions so the SoA load address is available
/// under the name `%{name}_soa_base` that the rewritten loads reference.
fn inject_soa_register_decls(ptx: &str, plan: &TranspositionPlan) -> String {
    // Build the extra register lines (.reg for soa_addr and soa_base).
    let mut reg_decls = String::new();
    for field in &plan.fields {
        reg_decls.push_str(&format!(
            ".reg .u64 %soa_addr_{};\n",
            field.name
        ));
        reg_decls.push_str(&format!(
            ".reg .u64 %{}_soa_base;\n",
            field.name
        ));
    }
    // Build the extra parameter lines (soa base pointers passed in as .param).
    let mut param_decls = String::new();
    for field in &plan.fields {
        param_decls.push_str(&format!(
            ".param .u64 %{}_soa_base_param,\n",
            field.name
        ));
    }
    // Build bridge load instructions: load param into the .reg.
    let mut bridge_loads = String::new();
    for field in &plan.fields {
        bridge_loads.push_str(&format!(
            "    ld.param.u64 %{name}_soa_base, [%{name}_soa_base_param];\n",
            name = field.name
        ));
    }

    if reg_decls.is_empty() {
        return ptx.to_string();
    }

    // Inject register decls after the first `.reg .u64` line,
    // and bridge loads after the first `{` (kernel body open brace).
    let mut out = String::with_capacity(
        ptx.len() + reg_decls.len() + param_decls.len() + bridge_loads.len(),
    );
    let mut injected_decl = false;
    let mut injected_bridge = false;
    for line in ptx.lines() {
        out.push_str(line);
        out.push('\n');
        if !injected_decl && line.trim().starts_with(".reg .u64") {
            out.push_str(&reg_decls);
            injected_decl = true;
        }
        if !injected_bridge && line.trim() == "{" {
            out.push_str(&bridge_loads);
            injected_bridge = true;
        }
    }
    if !injected_decl {
        out = format!("{}{}", reg_decls, out);
    }
    out
}
