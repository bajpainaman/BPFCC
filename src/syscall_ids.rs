//! Canonical Solana syscall IDs (Murmur3 hashes).
//!
//! Single authoritative table used by both tier0_analysis and tier4_ptx_emit.
//! All IDs are the standard SBF syscall hashes as shipped in the Solana runtime.

// ── Logging (GPU: no-op) ─────────────────────────────────────────────────────
pub const SOL_LOG: u32               = 0x71e3cf81;
pub const SOL_LOG_64: u32            = 0x7ef088ca;
pub const SOL_LOG_PUBKEY: u32        = 0x83f00e8f;
pub const SOL_LOG_COMPUTE_UNITS: u32 = 0xbcf20137;
pub const SOL_LOG_DATA: u32          = 0x1f1a984e;

// ── Crypto (GPU: device function) ────────────────────────────────────────────
pub const SOL_SHA256: u32            = 0xb6fc1a11;
pub const SOL_KECCAK256: u32         = 0x42b9c653;
pub const SOL_BLAKE3: u32            = 0x7317b434;
pub const SOL_SECP256K1_RECOVER: u32 = 0xc7a2563e;
pub const SOL_CURVE25519: u32        = 0x717cc4a3;

// ── Memory (GPU: inline) ─────────────────────────────────────────────────────
pub const SOL_MEMCPY: u32            = 0x3770fb22;
pub const SOL_MEMMOVE: u32           = 0xdfe4dbb3;
pub const SOL_MEMSET: u32            = 0xa22b9c18;
pub const SOL_MEMCMP: u32            = 0x5fdcde31;

// ── Allocation (GPU: bump alloc) ─────────────────────────────────────────────
pub const SOL_ALLOC_FREE: u32        = 0x5be92f1a;

// ── Return data (GPU: buffer) ─────────────────────────────────────────────────
pub const SOL_SET_RETURN_DATA: u32   = 0x48504732;
pub const SOL_GET_RETURN_DATA: u32   = 0x5d2245e7;

// ── Sysvars (GPU: inline from kernel arg) ────────────────────────────────────
pub const SOL_GET_CLOCK_SYSVAR: u32  = 0x70a30946;
pub const SOL_GET_RENT_SYSVAR: u32   = 0xa2694e21;

// ── Ed25519 (GPU: device function — ETO extension) ──────────────────────────
pub const SOL_ED25519_VERIFY: u32    = 0xed255190; // Custom ETO syscall ID

// ── CPI — REJECT (not GPU-eligible) ─────────────────────────────────────────
pub const SOL_INVOKE_SIGNED: u32     = 0xe83f5415;
pub const SOL_INVOKE_SIGNED_C: u32   = 0x5c2a3178;

/// Returns true if this syscall ID makes a program NOT GPU-eligible.
pub fn is_gpu_rejecting(id: u32) -> bool {
    matches!(id, SOL_INVOKE_SIGNED | SOL_INVOKE_SIGNED_C)
}

/// Returns true if this syscall ID is known and GPU-compatible.
pub fn is_known_gpu_syscall(id: u32) -> bool {
    matches!(id,
        SOL_LOG | SOL_LOG_64 | SOL_LOG_PUBKEY | SOL_LOG_COMPUTE_UNITS | SOL_LOG_DATA |
        SOL_SHA256 | SOL_KECCAK256 | SOL_BLAKE3 | SOL_SECP256K1_RECOVER | SOL_CURVE25519 | SOL_ED25519_VERIFY |
        SOL_MEMCPY | SOL_MEMMOVE | SOL_MEMSET | SOL_MEMCMP |
        SOL_ALLOC_FREE |
        SOL_SET_RETURN_DATA | SOL_GET_RETURN_DATA |
        SOL_GET_CLOCK_SYSVAR | SOL_GET_RENT_SYSVAR
    )
}
