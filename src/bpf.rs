//! Standalone BPF instruction types and opcodes.
//!
//! Extracted from the ETO runtime bpf module. Replaces the runtime dependency
//! with a self-contained implementation using `Result<T, String>` for errors.

/// BPF instruction opcodes
#[allow(dead_code)]
pub mod opcodes {
    // ALU operations (64-bit)
    pub const ADD64_IMM: u8 = 0x07;
    pub const ADD64_REG: u8 = 0x0f;
    pub const SUB64_IMM: u8 = 0x17;
    pub const SUB64_REG: u8 = 0x1f;
    pub const MUL64_IMM: u8 = 0x27;
    pub const MUL64_REG: u8 = 0x2f;
    pub const DIV64_IMM: u8 = 0x37;
    pub const DIV64_REG: u8 = 0x3f;
    pub const OR64_IMM: u8 = 0x47;
    pub const OR64_REG: u8 = 0x4f;
    pub const AND64_IMM: u8 = 0x57;
    pub const AND64_REG: u8 = 0x5f;
    pub const LSH64_IMM: u8 = 0x67;
    pub const LSH64_REG: u8 = 0x6f;
    pub const RSH64_IMM: u8 = 0x77;
    pub const RSH64_REG: u8 = 0x7f;
    pub const NEG64: u8 = 0x87;
    pub const MOD64_IMM: u8 = 0x97;
    pub const MOD64_REG: u8 = 0x9f;
    pub const XOR64_IMM: u8 = 0xa7;
    pub const XOR64_REG: u8 = 0xaf;
    pub const MOV64_IMM: u8 = 0xb7;
    pub const MOV64_REG: u8 = 0xbf;
    pub const ARSH64_IMM: u8 = 0xc7;
    pub const ARSH64_REG: u8 = 0xcf;

    // ALU operations (32-bit)
    pub const ADD32_IMM: u8 = 0x04;
    pub const ADD32_REG: u8 = 0x0c;
    pub const MOV32_IMM: u8 = 0xb4;
    pub const MOV32_REG: u8 = 0xbc;

    // Memory operations
    pub const LDXB: u8 = 0x71; // Load byte
    pub const LDXH: u8 = 0x69; // Load half-word
    pub const LDXW: u8 = 0x61; // Load word
    pub const LDXDW: u8 = 0x79; // Load double-word
    pub const STB: u8 = 0x72; // Store byte immediate
    pub const STH: u8 = 0x6a; // Store half-word immediate
    pub const STW: u8 = 0x62; // Store word immediate
    pub const STDW: u8 = 0x7a; // Store double-word immediate
    pub const STXB: u8 = 0x73; // Store byte from register
    pub const STXH: u8 = 0x6b; // Store half-word from register
    pub const STXW: u8 = 0x63; // Store word from register
    pub const STXDW: u8 = 0x7b; // Store double-word from register

    // Load immediate
    pub const LDDW: u8 = 0x18; // Load 64-bit immediate (2 instructions)

    // Jump operations
    pub const JA: u8 = 0x05; // Jump always
    pub const JEQ_IMM: u8 = 0x15; // Jump if equal (immediate)
    pub const JEQ_REG: u8 = 0x1d; // Jump if equal (register)
    pub const JGT_IMM: u8 = 0x25; // Jump if greater than (immediate)
    pub const JGT_REG: u8 = 0x2d; // Jump if greater than (register)
    pub const JGE_IMM: u8 = 0x35; // Jump if greater or equal (immediate)
    pub const JGE_REG: u8 = 0x3d; // Jump if greater or equal (register)
    pub const JNE_IMM: u8 = 0x55; // Jump if not equal (immediate)
    pub const JNE_REG: u8 = 0x5d; // Jump if not equal (register)
    pub const JSGT_IMM: u8 = 0x65; // Jump if signed greater than (immediate)
    pub const JSGT_REG: u8 = 0x6d; // Jump if signed greater than (register)
    pub const JSGE_IMM: u8 = 0x75; // Jump if signed greater or equal (immediate)
    pub const JSGE_REG: u8 = 0x7d; // Jump if signed greater or equal (register)
    pub const JLT_IMM: u8 = 0xa5; // Jump if less than (immediate)
    pub const JLT_REG: u8 = 0xad; // Jump if less than (register)
    pub const JLE_IMM: u8 = 0xb5; // Jump if less or equal (immediate)
    pub const JLE_REG: u8 = 0xbd; // Jump if less or equal (register)

    // Call and exit
    pub const CALL: u8 = 0x85; // Function call
    pub const EXIT: u8 = 0x95; // Exit program
}

/// BPF instruction format
#[derive(Debug, Clone, Copy)]
pub struct BpfInstruction {
    pub opcode: u8,
    pub dst: u8,
    pub src: u8,
    pub offset: i16,
    pub imm: i32,
}

impl BpfInstruction {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("instruction too short".into());
        }

        Ok(BpfInstruction {
            opcode: bytes[0],
            dst: bytes[1] & 0x0f,
            src: (bytes[1] >> 4) & 0x0f,
            offset: i16::from_le_bytes([bytes[2], bytes[3]]),
            imm: i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }
}

/// Parse BPF program bytecode into instructions
pub fn parse_program(bytecode: &[u8]) -> Result<Vec<BpfInstruction>, String> {
    if bytecode.len() % 8 != 0 {
        return Err("bytecode length must be multiple of 8".into());
    }

    let mut instructions = Vec::with_capacity(bytecode.len() / 8);

    for chunk in bytecode.chunks(8) {
        instructions.push(BpfInstruction::from_bytes(chunk)?);
    }

    Ok(instructions)
}
