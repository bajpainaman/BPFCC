use std::collections::{HashMap, HashSet};

/// 32-byte public key (Solana-compatible).
pub type Pubkey = [u8; 32];

/// Minimal account for BPFCC execution results.
#[derive(Debug, Clone)]
pub struct Account {
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner: Pubkey,
    pub executable: bool,
    pub rent_epoch: u64,
}

pub type Reg = u8; // 0-10 for BPF regs, 11+ for SSA temporaries

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemRegion {
    Stack,
    Heap,
    Input,
    Program,
    Dynamic, // Requires runtime bounds check
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    Imm(i64),
    Imm64(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrOp {
    // ALU 64-bit
    Add64 { dst: Reg, src: Operand },
    Sub64 { dst: Reg, src: Operand },
    Mul64 { dst: Reg, src: Operand },
    Div64 { dst: Reg, src: Operand },
    Mod64 { dst: Reg, src: Operand },
    And64 { dst: Reg, src: Operand },
    Or64  { dst: Reg, src: Operand },
    Xor64 { dst: Reg, src: Operand },
    Lsh64 { dst: Reg, src: Operand },
    Rsh64 { dst: Reg, src: Operand },
    Arsh64 { dst: Reg, src: Operand },
    Neg64 { dst: Reg },
    Mov64 { dst: Reg, src: Operand },

    // ALU 32-bit (zero-extends to 64-bit)
    Add32 { dst: Reg, src: Operand },
    Sub32 { dst: Reg, src: Operand },
    Mul32 { dst: Reg, src: Operand },
    Div32 { dst: Reg, src: Operand },
    Mod32 { dst: Reg, src: Operand },
    And32 { dst: Reg, src: Operand },
    Or32  { dst: Reg, src: Operand },
    Xor32 { dst: Reg, src: Operand },
    Lsh32 { dst: Reg, src: Operand },
    Rsh32 { dst: Reg, src: Operand },
    Arsh32 { dst: Reg, src: Operand },
    Neg32 { dst: Reg },
    Mov32 { dst: Reg, src: Operand },

    // Memory
    Load  { dst: Reg, base: Reg, offset: i16, size: u8, region: MemRegion },
    Store { base: Reg, offset: i16, src: Reg, size: u8, region: MemRegion },

    // Syscall
    Syscall { id: u32, cu_cost: u64 },

    // Internal function call
    Call { target_pc: usize },

    // Metering (inserted by Tier 3)
    MeterCU { cost: u64 },

    // Safety
    TrapIfZero { src: Reg },
    BoundsCheck { addr: Reg, size: u8, region: MemRegion },

    // SSA (inserted by Tier 2)
    Phi { dst: Reg, sources: Vec<(usize, Reg)> },

    // No-op
    Nop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchCondition {
    Eq, Ne, Gtu, Geu, Ltu, Leu, Gts, Ges, Lts, Les, Set,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrTerminator {
    Jump { target: usize },
    Branch {
        condition: BranchCondition,
        dst: Reg,
        src: Operand,
        true_target: usize,
        false_target: usize,
    },
    Return { value: Reg },
    Trap { reason: TrapReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrapReason {
    DivisionByZero,
    OutOfBounds,
    CUExceeded,
    StackOverflow,
    InvalidInstruction,
    OutOfMemory,
}

#[derive(Debug, Clone)]
pub struct IrBlock {
    pub id: usize,
    pub entry_pc: usize,
    pub ops: Vec<IrOp>,
    pub terminator: IrTerminator,
}

#[derive(Debug, Clone)]
pub struct IrProgram {
    pub blocks: Vec<IrBlock>,
    pub liveness: Option<LivenessInfo>,
    pub region_map: Vec<HashMap<u8, MemRegion>>,
}

#[derive(Debug, Clone, Default)]
pub struct LivenessInfo {
    pub live_in: Vec<HashSet<Reg>>,
    pub live_out: Vec<HashSet<Reg>>,
}

// CFG types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Unconditional,
    ConditionalTrue,
    ConditionalFalse,
    Call,
    Return,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub start_pc: usize,
    pub end_pc: usize,
    pub instruction_count: usize,
    pub has_syscall: bool,
    pub cu_cost: u64,
}

#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub blocks: Vec<BasicBlock>,
    pub edges: Vec<(usize, usize, EdgeType)>,
    pub successors: Vec<Vec<usize>>,
    pub predecessors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub gpu_eligible: bool,
    pub reject_reason: Option<String>,
    pub cfg: ControlFlowGraph,
    pub instruction_count: usize,
    pub syscalls_used: HashSet<u32>,
    pub max_stack_depth: u32,
    pub uses_heap: bool,
    pub block_entries: Vec<usize>,
    pub estimated_cu: u64,
    /// True if the program contains internal CALL instructions (non-syscall).
    /// Such programs are still compiled but Call ops are emitted as inline
    /// branches, which is correct for non-recursive tail-call patterns only.
    pub has_internal_calls: bool,
}

// Compilation result
#[derive(Debug, Clone)]
pub struct BpfCompileResult {
    pub gpu_eligible: bool,
    pub reject_reason: Option<String>,
    pub ptx: Option<String>,
    pub ir_program: Option<IrProgram>,
    pub analysis: Option<AnalysisResult>,
    pub program_hash: [u8; 32],
}

// Execution types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ExecutionStatus {
    Success = 0,
    Revert = 1,
    CUExceeded = 2,
    DivisionByZero = 3,
    OutOfBounds = 4,
    StackOverflow = 5,
    InvalidInstruction = 6,
    OutOfMemory = 7,
}

#[derive(Debug, Clone)]
pub struct BpfExecutionResult {
    pub status: ExecutionStatus,
    pub cu_consumed: u64,
    pub modified_accounts: Vec<(Pubkey, Account)>,
    pub return_data: Vec<u8>,
}

// DomTree for Tier 2
#[derive(Debug, Clone)]
pub struct DomTree {
    pub idom: Vec<Option<usize>>,
    pub frontier: Vec<HashSet<usize>>,
    pub children: Vec<Vec<usize>>,
}

// Kernel config
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub block_size: u32,       // 128 or 256
    pub grid_size: u32,        // ceil(tx_count / block_size)
    pub shared_mem_bytes: u32, // 0 unless fusion
    pub max_registers: u32,    // 32-64
}

// Program profile for Tier 8
#[derive(Debug, Clone, Default)]
pub struct ProgramProfile {
    pub execution_count: u64,
    pub total_gpu_time_us: u64,
    pub avg_occupancy: f64,
    pub divergence_rate: f64,
    pub memory_bandwidth_util: f64,
    pub cache_hit_rate: f64,
    pub register_spills: u32,
    pub cu_exceeded_count: u64,
}
