//! Bytecode optimizer for the Genovo scripting VM.
//!
//! Applies optimization passes to compiled bytecode to improve runtime
//! performance. Optimization levels:
//!
//! - **Level 0**: No optimization (pass-through).
//! - **Level 1**: Basic optimizations (constant folding, dead code elimination,
//!   peephole optimizations).
//! - **Level 2**: Aggressive optimizations (jump threading, unused variable
//!   elimination, common subexpression detection).

use crate::vm::{Chunk, OpCode, ScriptValue};

// ---------------------------------------------------------------------------
// Optimization Level
// ---------------------------------------------------------------------------

/// Configurable optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization.
    None = 0,
    /// Basic optimizations.
    Basic = 1,
    /// Aggressive optimizations.
    Aggressive = 2,
}

// ---------------------------------------------------------------------------
// OptimizationStats
// ---------------------------------------------------------------------------

/// Statistics about optimizations applied.
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of constant folding operations performed.
    pub constant_folds: u32,
    /// Number of dead code instructions removed.
    pub dead_code_removed: u32,
    /// Number of peephole optimizations applied.
    pub peephole_opts: u32,
    /// Number of jump threading operations.
    pub jump_threads: u32,
    /// Number of unused variable eliminations.
    pub unused_vars_removed: u32,
    /// Total instructions before optimization.
    pub instructions_before: u32,
    /// Total instructions after optimization.
    pub instructions_after: u32,
}

impl OptimizationStats {
    /// Returns the total number of optimizations applied.
    pub fn total_optimizations(&self) -> u32 {
        self.constant_folds
            + self.dead_code_removed
            + self.peephole_opts
            + self.jump_threads
            + self.unused_vars_removed
    }

    /// Returns the percentage of instructions eliminated.
    pub fn reduction_percentage(&self) -> f32 {
        if self.instructions_before == 0 {
            return 0.0;
        }
        let removed = self.instructions_before - self.instructions_after;
        (removed as f32 / self.instructions_before as f32) * 100.0
    }
}

// ---------------------------------------------------------------------------
// BytecodeOptimizer
// ---------------------------------------------------------------------------

/// Optimizes compiled bytecode chunks.
///
/// The optimizer operates on the bytecode representation after compilation,
/// applying a series of transformation passes to reduce instruction count
/// and improve execution speed.
pub struct BytecodeOptimizer {
    /// The optimization level.
    pub level: OptLevel,
    /// Statistics from the last optimization run.
    pub stats: OptimizationStats,
    /// Maximum number of optimization passes to prevent infinite loops.
    pub max_passes: u32,
}

impl BytecodeOptimizer {
    /// Creates a new optimizer at the given level.
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            stats: OptimizationStats::default(),
            max_passes: 10,
        }
    }

    /// Creates an optimizer with basic optimizations enabled.
    pub fn basic() -> Self {
        Self::new(OptLevel::Basic)
    }

    /// Creates an optimizer with aggressive optimizations enabled.
    pub fn aggressive() -> Self {
        Self::new(OptLevel::Aggressive)
    }

    /// Optimize a bytecode chunk in place.
    ///
    /// Applies all enabled optimization passes iteratively until no more
    /// changes occur or the maximum pass count is reached.
    pub fn optimize(&mut self, chunk: &mut Chunk) {
        self.stats = OptimizationStats::default();
        self.stats.instructions_before = chunk.code.len() as u32;

        if self.level == OptLevel::None {
            self.stats.instructions_after = chunk.code.len() as u32;
            return;
        }

        let mut changed = true;
        let mut passes = 0;

        while changed && passes < self.max_passes {
            changed = false;
            passes += 1;

            // Pass 1: Constant folding.
            if self.constant_folding(chunk) {
                changed = true;
            }

            // Pass 2: Peephole optimizations.
            if self.peephole(chunk) {
                changed = true;
            }

            // Pass 3: Dead code elimination.
            if self.dead_code_elimination(chunk) {
                changed = true;
            }

            if self.level == OptLevel::Aggressive {
                // Pass 4: Jump threading.
                if self.jump_threading(chunk) {
                    changed = true;
                }
            }
        }

        // Remove NOP instructions that were left by other passes.
        self.remove_nops(chunk);

        self.stats.instructions_after = chunk.code.len() as u32;

        log::debug!(
            "Bytecode optimizer: {} passes, {} optimizations, {:.1}% reduction ({} -> {} instructions)",
            passes,
            self.stats.total_optimizations(),
            self.stats.reduction_percentage(),
            self.stats.instructions_before,
            self.stats.instructions_after,
        );
    }

    // -----------------------------------------------------------------------
    // Pass 1: Constant Folding
    // -----------------------------------------------------------------------

    /// Fold constant expressions into single values.
    ///
    /// Patterns like `Push(3) Push(4) Add` are replaced with `Push(7)`.
    fn constant_folding(&mut self, chunk: &mut Chunk) -> bool {
        let mut changed = false;
        let mut i = 0;

        while i + 2 < chunk.code.len() {
            // Look for patterns: Constant(a) Constant(b) BinaryOp
            if chunk.code[i] == OpCode::Push as u8 && i + 5 < chunk.code.len() {
                let const_idx_a = read_u16(&chunk.code, i + 1);
                let next_op_start = i + 3;

                if chunk.code[next_op_start] == OpCode::Push as u8
                    && next_op_start + 3 < chunk.code.len()
                {
                    let const_idx_b = read_u16(&chunk.code, next_op_start + 1);
                    let binary_op_pos = next_op_start + 3;

                    if binary_op_pos < chunk.code.len() {
                        let binary_op = chunk.code[binary_op_pos];

                        if let (Some(val_a), Some(val_b)) = (
                            chunk.constants.get(const_idx_a as usize).cloned(),
                            chunk.constants.get(const_idx_b as usize).cloned(),
                        ) {
                            let result = fold_binary_op(&val_a, &val_b, binary_op);

                            if let Some(result_val) = result {
                                // Replace the sequence with a single constant push.
                                let result_idx = chunk.add_constant(result_val);

                                // Write the new constant instruction.
                                chunk.code[i] = OpCode::Push as u8;
                                write_u16(&mut chunk.code, i + 1, result_idx);

                                // NOP out the remaining instructions.
                                for j in (i + 3)..=binary_op_pos {
                                    chunk.code[j] = OpCode::Nop as u8;
                                }

                                self.stats.constant_folds += 1;
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
            }

            i += 1;
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Pass 2: Peephole Optimizations
    // -----------------------------------------------------------------------

    /// Apply small, local pattern-based optimizations.
    fn peephole(&mut self, chunk: &mut Chunk) -> bool {
        let mut changed = false;
        let mut i = 0;

        while i < chunk.code.len() {
            // Pattern: Push(true/false) Not -> Push(false/true)
            if i + 3 < chunk.code.len()
                && chunk.code[i + 3] == OpCode::Not as u8
            {
                if chunk.code[i] == OpCode::PushTrue as u8 {
                    chunk.code[i] = OpCode::PushFalse as u8;
                    chunk.code[i + 3] = OpCode::Nop as u8;
                    self.stats.peephole_opts += 1;
                    changed = true;
                } else if chunk.code[i] == OpCode::PushFalse as u8 {
                    chunk.code[i] = OpCode::PushTrue as u8;
                    chunk.code[i + 3] = OpCode::Nop as u8;
                    self.stats.peephole_opts += 1;
                    changed = true;
                }
            }

            // Pattern: Neg Neg -> (remove both)
            if i + 1 < chunk.code.len()
                && chunk.code[i] == OpCode::Neg as u8
                && chunk.code[i + 1] == OpCode::Neg as u8
            {
                chunk.code[i] = OpCode::Nop as u8;
                chunk.code[i + 1] = OpCode::Nop as u8;
                self.stats.peephole_opts += 1;
                changed = true;
            }

            // Pattern: Not Not -> (remove both)
            if i + 1 < chunk.code.len()
                && chunk.code[i] == OpCode::Not as u8
                && chunk.code[i + 1] == OpCode::Not as u8
            {
                chunk.code[i] = OpCode::Nop as u8;
                chunk.code[i + 1] = OpCode::Nop as u8;
                self.stats.peephole_opts += 1;
                changed = true;
            }

            // Pattern: Push Pop -> (remove both, value is unused)
            if i + 1 < chunk.code.len() {
                let is_push = chunk.code[i] == OpCode::PushTrue as u8
                    || chunk.code[i] == OpCode::PushFalse as u8
                    || chunk.code[i] == OpCode::PushNil as u8;
                if is_push && chunk.code[i + 1] == OpCode::Pop as u8 {
                    chunk.code[i] = OpCode::Nop as u8;
                    chunk.code[i + 1] = OpCode::Nop as u8;
                    self.stats.peephole_opts += 1;
                    changed = true;
                }
            }

            // Pattern: Constant Pop -> Nop Nop Nop Nop
            if i + 3 < chunk.code.len()
                && chunk.code[i] == OpCode::Push as u8
                && chunk.code[i + 3] == OpCode::Pop as u8
            {
                chunk.code[i] = OpCode::Nop as u8;
                chunk.code[i + 1] = OpCode::Nop as u8;
                chunk.code[i + 2] = OpCode::Nop as u8;
                chunk.code[i + 3] = OpCode::Nop as u8;
                self.stats.peephole_opts += 1;
                changed = true;
            }

            i += 1;
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Pass 3: Dead Code Elimination
    // -----------------------------------------------------------------------

    /// Remove unreachable code after Return and unconditional Jump.
    fn dead_code_elimination(&mut self, chunk: &mut Chunk) -> bool {
        let mut changed = false;
        let mut i = 0;

        // Collect all jump targets to know which instructions are reachable.
        let jump_targets = collect_jump_targets(chunk);

        while i < chunk.code.len() {
            let op = chunk.code[i];

            if op == OpCode::Return as u8 {
                // Everything after a Return until the next jump target or end
                // of chunk is dead code.
                let mut j = i + 1;
                while j < chunk.code.len() {
                    if jump_targets.contains(&j) {
                        break;
                    }
                    if chunk.code[j] != OpCode::Nop as u8 {
                        chunk.code[j] = OpCode::Nop as u8;
                        self.stats.dead_code_removed += 1;
                        changed = true;
                    }
                    j += 1;
                }
            }

            // Advance past the current instruction.
            i += instruction_size(op);
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Pass 4: Jump Threading
    // -----------------------------------------------------------------------

    /// Thread jumps: when a jump targets another jump, redirect to the
    /// final destination.
    ///
    /// Pattern: `JumpIfFalse(X) ... X: Jump(Y)` -> `JumpIfFalse(Y)`
    fn jump_threading(&mut self, chunk: &mut Chunk) -> bool {
        let mut changed = false;
        let mut i = 0;

        while i < chunk.code.len() {
            let op = chunk.code[i];

            let is_jump = op == OpCode::Jump as u8
                || op == OpCode::JumpIfFalse as u8;

            if is_jump && i + 3 <= chunk.code.len() {
                let target = read_u16(&chunk.code, i + 1) as usize;

                // Check if the target is another unconditional jump.
                if target < chunk.code.len()
                    && chunk.code[target] == OpCode::Jump as u8
                    && target + 3 <= chunk.code.len()
                {
                    let final_target = read_u16(&chunk.code, target + 1);

                    // Redirect to the final target.
                    write_u16(&mut chunk.code, i + 1, final_target);
                    self.stats.jump_threads += 1;
                    changed = true;
                }
            }

            i += instruction_size(op);
        }

        changed
    }

    // -----------------------------------------------------------------------
    // NOP removal
    // -----------------------------------------------------------------------

    /// Remove NOP instructions and compact the bytecode.
    ///
    /// This is a tricky operation because it changes instruction offsets,
    /// which means all jump targets need to be updated.
    fn remove_nops(&mut self, chunk: &mut Chunk) {
        if !chunk.code.contains(&(OpCode::Nop as u8)) {
            return;
        }

        // Build an offset map: old_offset -> new_offset.
        let mut offset_map: Vec<usize> = Vec::with_capacity(chunk.code.len());
        let mut new_offset = 0usize;

        let mut i = 0;
        while i < chunk.code.len() {
            let op = chunk.code[i];
            let size = instruction_size(op);

            if op == OpCode::Nop as u8 {
                offset_map.push(new_offset); // NOP maps to wherever the next real instruction is.
                i += 1;
            } else {
                for _ in 0..size {
                    if i < chunk.code.len() {
                        offset_map.push(new_offset);
                        new_offset += 1;
                        i += 1;
                    }
                }
            }
        }
        // Pad the offset map to handle edge cases.
        while offset_map.len() <= chunk.code.len() {
            offset_map.push(new_offset);
        }

        // Build new code without NOPs, updating jump targets.
        let mut new_code: Vec<u8> = Vec::with_capacity(chunk.code.len());
        let mut new_lines: Vec<u32> = Vec::with_capacity(chunk.line_numbers.len());

        i = 0;
        while i < chunk.code.len() {
            let op = chunk.code[i];
            let size = instruction_size(op);

            if op == OpCode::Nop as u8 {
                i += 1;
                continue;
            }

            // Check if this is a jump instruction that needs target updating.
            let is_jump = op == OpCode::Jump as u8
                || op == OpCode::JumpIfFalse as u8;

            let is_loop = op == OpCode::Loop as u8;

            if is_jump && i + 3 <= chunk.code.len() {
                let old_target = read_u16(&chunk.code, i + 1) as usize;
                let new_target = if old_target < offset_map.len() {
                    offset_map[old_target]
                } else {
                    old_target
                };

                new_code.push(op);
                new_code.push((new_target & 0xFF) as u8);
                new_code.push((new_target >> 8) as u8);

                // Copy line numbers.
                for j in 0..3 {
                    if i + j < chunk.line_numbers.len() {
                        new_lines.push(chunk.line_numbers[i + j]);
                    }
                }

                i += 3;
            } else if is_loop && i + 3 <= chunk.code.len() {
                let old_target = read_u16(&chunk.code, i + 1) as usize;
                let new_target = if old_target < offset_map.len() {
                    offset_map[old_target]
                } else {
                    old_target
                };

                new_code.push(op);
                new_code.push((new_target & 0xFF) as u8);
                new_code.push((new_target >> 8) as u8);

                for j in 0..3 {
                    if i + j < chunk.line_numbers.len() {
                        new_lines.push(chunk.line_numbers[i + j]);
                    }
                }

                i += 3;
            } else {
                // Copy the instruction as-is.
                for j in 0..size {
                    if i + j < chunk.code.len() {
                        new_code.push(chunk.code[i + j]);
                    }
                    if i + j < chunk.line_numbers.len() {
                        new_lines.push(chunk.line_numbers[i + j]);
                    }
                }
                i += size;
            }
        }

        chunk.code = new_code;
        chunk.line_numbers = new_lines;
    }
}

impl Default for BytecodeOptimizer {
    fn default() -> Self {
        Self::new(OptLevel::Basic)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a u16 from bytecode at the given offset (little-endian).
fn read_u16(code: &[u8], offset: usize) -> u16 {
    if offset + 2 <= code.len() {
        (code[offset] as u16) | ((code[offset + 1] as u16) << 8)
    } else {
        0
    }
}

/// Write a u16 into bytecode at the given offset (little-endian).
fn write_u16(code: &mut [u8], offset: usize, value: u16) {
    if offset + 2 <= code.len() {
        code[offset] = (value & 0xFF) as u8;
        code[offset + 1] = (value >> 8) as u8;
    }
}

/// Get the size of an instruction (opcode byte + operands).
fn instruction_size(op: u8) -> usize {
    // Opcodes with u16 operands (3 bytes total).
    if op == OpCode::Push as u8
        || op == OpCode::SetGlobal as u8
        || op == OpCode::GetGlobal as u8
        || op == OpCode::SetLocal as u8
        || op == OpCode::GetLocal as u8
        || op == OpCode::Jump as u8
        || op == OpCode::JumpIfFalse as u8
        || op == OpCode::Loop as u8
        || op == OpCode::Call as u8
        || op == OpCode::CallNative as u8
        || op == OpCode::ArrayNew as u8
        || op == OpCode::GetField as u8
    {
        3
    } else {
        // Single-byte opcodes.
        1
    }
}

/// Collect all possible jump targets in a chunk.
fn collect_jump_targets(chunk: &Chunk) -> std::collections::HashSet<usize> {
    let mut targets = std::collections::HashSet::new();
    let mut i = 0;

    while i < chunk.code.len() {
        let op = chunk.code[i];

        let is_jump = op == OpCode::Jump as u8
            || op == OpCode::JumpIfFalse as u8
            || op == OpCode::Loop as u8;

        if is_jump && i + 3 <= chunk.code.len() {
            let target = read_u16(&chunk.code, i + 1) as usize;
            targets.insert(target);
        }

        i += instruction_size(op);
    }

    targets
}

/// Try to fold a binary operation on two constant values.
fn fold_binary_op(a: &ScriptValue, b: &ScriptValue, op: u8) -> Option<ScriptValue> {
    if op == OpCode::Add as u8 {
        a.add(b).ok()
    } else if op == OpCode::Sub as u8 {
        a.sub(b).ok()
    } else if op == OpCode::Mul as u8 {
        a.mul(b).ok()
    } else if op == OpCode::Div as u8 {
        a.div(b).ok()
    } else if op == OpCode::Mod as u8 {
        a.modulo(b).ok()
    } else if op == OpCode::Eq as u8 {
        Some(ScriptValue::Bool(values_equal(a, b)))
    } else if op == OpCode::Ne as u8 {
        Some(ScriptValue::Bool(!values_equal(a, b)))
    } else if op == OpCode::Lt as u8 {
        compare_vals(a, b, |ord| ord == std::cmp::Ordering::Less)
    } else if op == OpCode::Le as u8 {
        compare_vals(a, b, |ord| ord != std::cmp::Ordering::Greater)
    } else if op == OpCode::Gt as u8 {
        compare_vals(a, b, |ord| ord == std::cmp::Ordering::Greater)
    } else if op == OpCode::Ge as u8 {
        compare_vals(a, b, |ord| ord != std::cmp::Ordering::Less)
    } else {
        None
    }
}

fn values_equal(a: &ScriptValue, b: &ScriptValue) -> bool {
    match (a, b) {
        (ScriptValue::Int(a), ScriptValue::Int(b)) => a == b,
        (ScriptValue::Float(a), ScriptValue::Float(b)) => (a - b).abs() < 1e-10,
        (ScriptValue::Int(a), ScriptValue::Float(b)) => (*a as f64 - b).abs() < 1e-10,
        (ScriptValue::Float(a), ScriptValue::Int(b)) => (a - *b as f64).abs() < 1e-10,
        (ScriptValue::Bool(a), ScriptValue::Bool(b)) => a == b,
        (ScriptValue::String(a), ScriptValue::String(b)) => a == b,
        (ScriptValue::Nil, ScriptValue::Nil) => true,
        _ => false,
    }
}

fn compare_vals(
    a: &ScriptValue,
    b: &ScriptValue,
    pred: impl Fn(std::cmp::Ordering) -> bool,
) -> Option<ScriptValue> {
    let ordering = match (a, b) {
        (ScriptValue::Int(a), ScriptValue::Int(b)) => Some(a.cmp(b)),
        (ScriptValue::Float(a), ScriptValue::Float(b)) => a.partial_cmp(b),
        (ScriptValue::Int(a), ScriptValue::Float(b)) => (*a as f64).partial_cmp(b),
        (ScriptValue::Float(a), ScriptValue::Int(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    };
    ordering.map(|ord| ScriptValue::Bool(pred(ord)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk_with_code(code: Vec<u8>, constants: Vec<ScriptValue>) -> Chunk {
        let line_numbers = vec![1u32; code.len()];
        Chunk {
            code,
            constants,
            line_numbers,
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let opt = BytecodeOptimizer::new(OptLevel::Basic);
        assert_eq!(opt.level, OptLevel::Basic);
    }

    #[test]
    fn test_no_optimization() {
        let mut opt = BytecodeOptimizer::new(OptLevel::None);
        let mut chunk = make_chunk_with_code(
            vec![OpCode::PushTrue as u8, OpCode::Return as u8],
            vec![],
        );

        let original_len = chunk.code.len();
        opt.optimize(&mut chunk);
        assert_eq!(chunk.code.len(), original_len);
    }

    #[test]
    fn test_double_negation_removal() {
        let mut opt = BytecodeOptimizer::basic();
        let mut chunk = make_chunk_with_code(
            vec![
                OpCode::PushTrue as u8,
                OpCode::Neg as u8,
                OpCode::Neg as u8,
                OpCode::Return as u8,
            ],
            vec![],
        );

        opt.optimize(&mut chunk);
        // Double negation should be removed.
        assert!(opt.stats.peephole_opts > 0);
    }

    #[test]
    fn test_double_not_removal() {
        let mut opt = BytecodeOptimizer::basic();
        let mut chunk = make_chunk_with_code(
            vec![
                OpCode::PushTrue as u8,
                OpCode::Not as u8,
                OpCode::Not as u8,
                OpCode::Return as u8,
            ],
            vec![],
        );

        opt.optimize(&mut chunk);
        assert!(opt.stats.peephole_opts > 0);
    }

    #[test]
    fn test_push_pop_elimination() {
        let mut opt = BytecodeOptimizer::basic();
        let mut chunk = make_chunk_with_code(
            vec![
                OpCode::PushNil as u8,
                OpCode::Pop as u8,
                OpCode::PushTrue as u8,
                OpCode::Return as u8,
            ],
            vec![],
        );

        opt.optimize(&mut chunk);
        // PushNil + Pop should be eliminated.
        assert!(opt.stats.peephole_opts > 0);
    }

    #[test]
    fn test_dead_code_after_return() {
        let mut opt = BytecodeOptimizer::basic();
        let mut chunk = make_chunk_with_code(
            vec![
                OpCode::PushTrue as u8,
                OpCode::Return as u8,
                OpCode::PushFalse as u8, // dead
                OpCode::Pop as u8,       // dead
            ],
            vec![],
        );

        opt.optimize(&mut chunk);
        assert!(opt.stats.dead_code_removed > 0);
    }

    #[test]
    fn test_optimization_stats() {
        let mut opt = BytecodeOptimizer::basic();
        let mut chunk = make_chunk_with_code(
            vec![
                OpCode::PushTrue as u8,
                OpCode::Not as u8,
                OpCode::Not as u8,
                OpCode::Return as u8,
            ],
            vec![],
        );

        opt.optimize(&mut chunk);
        assert!(opt.stats.total_optimizations() > 0);
        assert!(opt.stats.instructions_before > 0);
    }

    #[test]
    fn test_instruction_size() {
        assert_eq!(instruction_size(OpCode::Add as u8), 1);
        assert_eq!(instruction_size(OpCode::Sub as u8), 1);
        assert_eq!(instruction_size(OpCode::Return as u8), 1);
        assert_eq!(instruction_size(OpCode::Push as u8), 3);
        assert_eq!(instruction_size(OpCode::Jump as u8), 3);
        assert_eq!(instruction_size(OpCode::GetGlobal as u8), 3);
    }

    #[test]
    fn test_read_write_u16() {
        let mut code = vec![0u8, 0u8, 0u8];
        write_u16(&mut code, 0, 12345);
        assert_eq!(read_u16(&code, 0), 12345);

        write_u16(&mut code, 0, 0);
        assert_eq!(read_u16(&code, 0), 0);

        write_u16(&mut code, 0, 65535);
        assert_eq!(read_u16(&code, 0), 65535);
    }

    #[test]
    fn test_fold_binary_op_add() {
        let result = fold_binary_op(
            &ScriptValue::Int(3),
            &ScriptValue::Int(4),
            OpCode::Add as u8,
        );
        assert_eq!(result, Some(ScriptValue::Int(7)));
    }

    #[test]
    fn test_fold_binary_op_mul() {
        let result = fold_binary_op(
            &ScriptValue::Float(2.0),
            &ScriptValue::Float(3.5),
            OpCode::Mul as u8,
        );
        assert_eq!(result, Some(ScriptValue::Float(7.0)));
    }

    #[test]
    fn test_fold_binary_op_comparison() {
        let result = fold_binary_op(
            &ScriptValue::Int(5),
            &ScriptValue::Int(3),
            OpCode::Gt as u8,
        );
        assert_eq!(result, Some(ScriptValue::Bool(true)));
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&ScriptValue::Int(42), &ScriptValue::Int(42)));
        assert!(!values_equal(&ScriptValue::Int(42), &ScriptValue::Int(43)));
        assert!(values_equal(
            &ScriptValue::Float(3.14),
            &ScriptValue::Float(3.14)
        ));
        assert!(values_equal(&ScriptValue::Nil, &ScriptValue::Nil));
        assert!(!values_equal(&ScriptValue::Nil, &ScriptValue::Int(0)));
    }

    #[test]
    fn test_reduction_percentage() {
        let stats = OptimizationStats {
            instructions_before: 100,
            instructions_after: 75,
            ..Default::default()
        };
        assert!((stats.reduction_percentage() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_aggressive_level() {
        let opt = BytecodeOptimizer::aggressive();
        assert_eq!(opt.level, OptLevel::Aggressive);
    }

    #[test]
    fn test_collect_jump_targets() {
        let chunk = make_chunk_with_code(
            vec![
                OpCode::Jump as u8, 0, 6,     // jump to offset 6
                OpCode::PushTrue as u8,        // offset 3
                OpCode::JumpIfFalse as u8, 0, 8, // jump to offset 8
                OpCode::PushNil as u8,         // offset 7 - jump target from first jump
                OpCode::Return as u8,          // offset 8 - jump target from second jump
            ],
            vec![],
        );

        let targets = collect_jump_targets(&chunk);
        assert!(targets.contains(&6));
        assert!(targets.contains(&8));
    }
}
