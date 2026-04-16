//! Script debugger for the Genovo scripting VM.
//!
//! Provides debugging capabilities for scripted game code:
//! - **Breakpoints**: set by file/line, with optional conditions
//! - **Stepping**: step into, step over, step out
//! - **Inspection**: call stack, local variables, global variables
//! - **Watch expressions**: evaluate expressions in the current scope
//! - **Debug events**: breakpoint hit, step complete, exception
//!
//! The debugger wraps the VM execution loop and intercepts execution at
//! breakpoints and step boundaries. It maintains a paused state where
//! the user can inspect variables and the call stack.

use std::collections::HashMap;

use crate::vm::{ScriptError, ScriptValue};

// ---------------------------------------------------------------------------
// Breakpoint
// ---------------------------------------------------------------------------

/// Condition for a breakpoint.
#[derive(Debug, Clone)]
pub enum BreakpointCondition {
    /// Always break when hit.
    Always,
    /// Break only when the expression evaluates to a truthy value.
    /// The string is a simple comparison expression like "x > 5".
    When(String),
    /// Break after being hit N times.
    HitCount(u32),
}

/// A breakpoint set in the debugger.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Unique breakpoint ID.
    pub id: u32,
    /// The file or script name.
    pub file: String,
    /// The line number (1-based).
    pub line: u32,
    /// The condition for this breakpoint.
    pub condition: BreakpointCondition,
    /// Whether this breakpoint is enabled.
    pub enabled: bool,
    /// Number of times this breakpoint has been hit.
    pub hit_count: u32,
    /// Optional log message (logpoint) — if set, log instead of breaking.
    pub log_message: Option<String>,
}

impl Breakpoint {
    /// Creates a new unconditional breakpoint.
    pub fn new(id: u32, file: impl Into<String>, line: u32) -> Self {
        Self {
            id,
            file: file.into(),
            line,
            condition: BreakpointCondition::Always,
            enabled: true,
            hit_count: 0,
            log_message: None,
        }
    }

    /// Sets a condition on this breakpoint.
    pub fn with_condition(mut self, condition: BreakpointCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Sets a log message (making this a logpoint).
    pub fn with_log_message(mut self, message: impl Into<String>) -> Self {
        self.log_message = Some(message.into());
        self
    }

    /// Check if this breakpoint should trigger.
    fn should_break(&mut self, locals: &HashMap<String, ScriptValue>) -> bool {
        if !self.enabled {
            return false;
        }

        self.hit_count += 1;

        match &self.condition {
            BreakpointCondition::Always => true,
            BreakpointCondition::When(expr) => {
                evaluate_condition(expr, locals)
            }
            BreakpointCondition::HitCount(n) => {
                self.hit_count >= *n
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StackFrame
// ---------------------------------------------------------------------------

/// A frame on the call stack, representing an active function invocation.
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// The name of the function (or "<script>" for the top level).
    pub function_name: String,
    /// The file or script name.
    pub file: String,
    /// The current line number in this frame.
    pub line: u32,
    /// Local variable bindings in this frame.
    pub locals: HashMap<String, ScriptValue>,
    /// The instruction pointer within the function's bytecode.
    pub instruction_pointer: usize,
    /// Frame index in the call stack (0 = bottom).
    pub frame_index: usize,
}

impl StackFrame {
    /// Creates a new stack frame.
    pub fn new(
        function_name: impl Into<String>,
        file: impl Into<String>,
        line: u32,
    ) -> Self {
        Self {
            function_name: function_name.into(),
            file: file.into(),
            line,
            locals: HashMap::new(),
            instruction_pointer: 0,
            frame_index: 0,
        }
    }

    /// Set a local variable.
    pub fn set_local(&mut self, name: impl Into<String>, value: ScriptValue) {
        self.locals.insert(name.into(), value);
    }

    /// Get a local variable.
    pub fn get_local(&self, name: &str) -> Option<&ScriptValue> {
        self.locals.get(name)
    }
}

// ---------------------------------------------------------------------------
// ExecutionState
// ---------------------------------------------------------------------------

/// The current execution state of the debugger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionState {
    /// The script is running normally.
    Running,
    /// Execution is paused at a breakpoint or step boundary.
    Paused(PauseReason),
    /// Stepping into the next instruction.
    StepInto,
    /// Stepping over the current instruction (skip into function calls).
    StepOver {
        /// The frame depth at which step-over was initiated.
        target_depth: usize,
    },
    /// Stepping out of the current function.
    StepOut {
        /// The frame depth to return to.
        target_depth: usize,
    },
    /// Execution has completed.
    Finished,
}

/// The reason execution was paused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PauseReason {
    /// Hit a breakpoint.
    Breakpoint(u32),
    /// Step completed.
    Step,
    /// An exception or error occurred.
    Exception(String),
    /// Manually paused by the user.
    UserRequest,
    /// A watched variable changed.
    VariableChanged(String),
}

// ---------------------------------------------------------------------------
// DebugEvent
// ---------------------------------------------------------------------------

/// Events emitted by the debugger.
#[derive(Debug, Clone)]
pub enum DebugEvent {
    /// A breakpoint was hit.
    BreakpointHit {
        breakpoint_id: u32,
        file: String,
        line: u32,
    },
    /// A step operation completed.
    StepComplete {
        file: String,
        line: u32,
    },
    /// An exception occurred during execution.
    Exception {
        message: String,
        file: String,
        line: u32,
    },
    /// A watched variable's value changed.
    VariableChanged {
        name: String,
        old_value: ScriptValue,
        new_value: ScriptValue,
    },
    /// A logpoint produced output.
    LogpointMessage {
        breakpoint_id: u32,
        message: String,
    },
    /// Execution finished.
    ExecutionFinished,
}

// ---------------------------------------------------------------------------
// WatchExpression
// ---------------------------------------------------------------------------

/// A watch expression that is evaluated each time execution pauses.
#[derive(Debug, Clone)]
pub struct WatchExpression {
    /// Unique ID.
    pub id: u32,
    /// The expression text (e.g., "player.health", "x + y").
    pub expression: String,
    /// The last evaluated value.
    pub last_value: Option<ScriptValue>,
    /// Whether to break when this expression's value changes.
    pub break_on_change: bool,
}

impl WatchExpression {
    /// Creates a new watch expression.
    pub fn new(id: u32, expression: impl Into<String>) -> Self {
        Self {
            id,
            expression: expression.into(),
            last_value: None,
            break_on_change: false,
        }
    }

    /// Enable break-on-change for this watch.
    pub fn with_break_on_change(mut self, enabled: bool) -> Self {
        self.break_on_change = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// ScriptDebugger
// ---------------------------------------------------------------------------

/// Debug controller for the Genovo scripting VM.
///
/// The debugger maintains breakpoints, watch expressions, and execution
/// state. It intercepts the VM's execution loop to provide step-through
/// debugging capabilities.
pub struct ScriptDebugger {
    /// All breakpoints, indexed by ID.
    breakpoints: HashMap<u32, Breakpoint>,
    /// Next breakpoint ID to assign.
    next_bp_id: u32,
    /// Watch expressions, indexed by ID.
    watches: HashMap<u32, WatchExpression>,
    /// Next watch ID to assign.
    next_watch_id: u32,
    /// Current execution state.
    state: ExecutionState,
    /// The call stack.
    call_stack: Vec<StackFrame>,
    /// Global variables snapshot.
    globals: HashMap<String, ScriptValue>,
    /// Event queue (events generated since last poll).
    events: Vec<DebugEvent>,
    /// Whether debugging is enabled.
    pub enabled: bool,
    /// REPL history for console commands.
    repl_history: Vec<String>,
    /// Previous variable values for change detection.
    previous_values: HashMap<String, ScriptValue>,
    /// Log of all debugger output.
    output_log: Vec<String>,
}

impl ScriptDebugger {
    /// Creates a new debugger.
    pub fn new() -> Self {
        Self {
            breakpoints: HashMap::new(),
            next_bp_id: 1,
            watches: HashMap::new(),
            next_watch_id: 1,
            state: ExecutionState::Running,
            call_stack: Vec::new(),
            globals: HashMap::new(),
            events: Vec::new(),
            enabled: true,
            repl_history: Vec::new(),
            previous_values: HashMap::new(),
            output_log: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Breakpoint management
    // -----------------------------------------------------------------------

    /// Add a breakpoint at the given file and line.
    pub fn add_breakpoint(&mut self, file: impl Into<String>, line: u32) -> u32 {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        let bp = Breakpoint::new(id, file, line);
        self.breakpoints.insert(id, bp);
        id
    }

    /// Add a conditional breakpoint.
    pub fn add_conditional_breakpoint(
        &mut self,
        file: impl Into<String>,
        line: u32,
        condition: BreakpointCondition,
    ) -> u32 {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        let bp = Breakpoint::new(id, file, line).with_condition(condition);
        self.breakpoints.insert(id, bp);
        id
    }

    /// Add a logpoint (logs a message instead of breaking).
    pub fn add_logpoint(
        &mut self,
        file: impl Into<String>,
        line: u32,
        message: impl Into<String>,
    ) -> u32 {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        let bp = Breakpoint::new(id, file, line).with_log_message(message);
        self.breakpoints.insert(id, bp);
        id
    }

    /// Remove a breakpoint by ID.
    pub fn remove_breakpoint(&mut self, id: u32) -> bool {
        self.breakpoints.remove(&id).is_some()
    }

    /// Remove a breakpoint by file and line.
    pub fn remove_breakpoint_at(&mut self, file: &str, line: u32) -> bool {
        let ids: Vec<u32> = self
            .breakpoints
            .iter()
            .filter(|(_, bp)| bp.file == file && bp.line == line)
            .map(|(id, _)| *id)
            .collect();
        for id in &ids {
            self.breakpoints.remove(id);
        }
        !ids.is_empty()
    }

    /// Enable or disable a breakpoint.
    pub fn set_breakpoint_enabled(&mut self, id: u32, enabled: bool) {
        if let Some(bp) = self.breakpoints.get_mut(&id) {
            bp.enabled = enabled;
        }
    }

    /// Get all breakpoints.
    pub fn breakpoints(&self) -> Vec<&Breakpoint> {
        self.breakpoints.values().collect()
    }

    /// Get a specific breakpoint.
    pub fn get_breakpoint(&self, id: u32) -> Option<&Breakpoint> {
        self.breakpoints.get(&id)
    }

    /// Clear all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    // -----------------------------------------------------------------------
    // Watch expressions
    // -----------------------------------------------------------------------

    /// Add a watch expression.
    pub fn add_watch(&mut self, expression: impl Into<String>) -> u32 {
        let id = self.next_watch_id;
        self.next_watch_id += 1;
        self.watches.insert(id, WatchExpression::new(id, expression));
        id
    }

    /// Add a watch expression that breaks when the value changes.
    pub fn add_watch_break_on_change(&mut self, expression: impl Into<String>) -> u32 {
        let id = self.next_watch_id;
        self.next_watch_id += 1;
        let watch = WatchExpression::new(id, expression).with_break_on_change(true);
        self.watches.insert(id, watch);
        id
    }

    /// Remove a watch expression.
    pub fn remove_watch(&mut self, id: u32) -> bool {
        self.watches.remove(&id).is_some()
    }

    /// Get all watch expressions.
    pub fn watches(&self) -> Vec<&WatchExpression> {
        self.watches.values().collect()
    }

    /// Evaluate a watch expression in the current scope.
    pub fn evaluate_watch(&self, id: u32) -> Option<ScriptValue> {
        let watch = self.watches.get(&id)?;
        self.evaluate_expression(&watch.expression)
    }

    // -----------------------------------------------------------------------
    // Step controls
    // -----------------------------------------------------------------------

    /// Resume execution (continue until next breakpoint or end).
    pub fn continue_execution(&mut self) {
        self.state = ExecutionState::Running;
    }

    /// Step into: execute the next instruction, entering function calls.
    pub fn step_into(&mut self) {
        self.state = ExecutionState::StepInto;
    }

    /// Step over: execute the next instruction, skipping into function calls.
    pub fn step_over(&mut self) {
        let depth = self.call_stack.len();
        self.state = ExecutionState::StepOver {
            target_depth: depth,
        };
    }

    /// Step out: continue until the current function returns.
    pub fn step_out(&mut self) {
        let depth = self.call_stack.len().saturating_sub(1);
        self.state = ExecutionState::StepOut {
            target_depth: depth,
        };
    }

    /// Pause execution immediately.
    pub fn pause(&mut self) {
        self.state = ExecutionState::Paused(PauseReason::UserRequest);
    }

    /// Stop execution.
    pub fn stop(&mut self) {
        self.state = ExecutionState::Finished;
    }

    // -----------------------------------------------------------------------
    // State inspection
    // -----------------------------------------------------------------------

    /// Get the current execution state.
    pub fn execution_state(&self) -> &ExecutionState {
        &self.state
    }

    /// Check if execution is paused.
    pub fn is_paused(&self) -> bool {
        matches!(self.state, ExecutionState::Paused(_))
    }

    /// Check if execution is running.
    pub fn is_running(&self) -> bool {
        matches!(
            self.state,
            ExecutionState::Running
                | ExecutionState::StepInto
                | ExecutionState::StepOver { .. }
                | ExecutionState::StepOut { .. }
        )
    }

    /// Check if execution has finished.
    pub fn is_finished(&self) -> bool {
        matches!(self.state, ExecutionState::Finished)
    }

    /// Get the call stack.
    pub fn get_call_stack(&self) -> &[StackFrame] {
        &self.call_stack
    }

    /// Get locals for a specific frame.
    pub fn get_locals(&self, frame_index: usize) -> Option<Vec<(String, ScriptValue)>> {
        self.call_stack.get(frame_index).map(|frame| {
            frame
                .locals
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
    }

    /// Get all global variables.
    pub fn get_globals(&self) -> Vec<(String, ScriptValue)> {
        self.globals
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Set a global variable (for interactive debugging).
    pub fn set_global(&mut self, name: impl Into<String>, value: ScriptValue) {
        self.globals.insert(name.into(), value);
    }

    /// Get the current line number.
    pub fn current_line(&self) -> Option<u32> {
        self.call_stack.last().map(|f| f.line)
    }

    /// Get the current file.
    pub fn current_file(&self) -> Option<&str> {
        self.call_stack.last().map(|f| f.file.as_str())
    }

    /// Get the current function name.
    pub fn current_function(&self) -> Option<&str> {
        self.call_stack.last().map(|f| f.function_name.as_str())
    }

    // -----------------------------------------------------------------------
    // Call stack management
    // -----------------------------------------------------------------------

    /// Push a frame onto the call stack.
    pub fn push_frame(&mut self, frame: StackFrame) {
        let mut frame = frame;
        frame.frame_index = self.call_stack.len();
        self.call_stack.push(frame);
    }

    /// Pop a frame from the call stack.
    pub fn pop_frame(&mut self) -> Option<StackFrame> {
        self.call_stack.pop()
    }

    /// Update the current frame's line number.
    pub fn update_line(&mut self, line: u32) {
        if let Some(frame) = self.call_stack.last_mut() {
            frame.line = line;
        }
    }

    /// Update the current frame's instruction pointer.
    pub fn update_ip(&mut self, ip: usize) {
        if let Some(frame) = self.call_stack.last_mut() {
            frame.instruction_pointer = ip;
        }
    }

    /// Update a local variable in the current frame.
    pub fn update_local(&mut self, name: impl Into<String>, value: ScriptValue) {
        if let Some(frame) = self.call_stack.last_mut() {
            frame.set_local(name, value);
        }
    }

    /// Update global variables snapshot.
    pub fn update_globals(&mut self, globals: HashMap<String, ScriptValue>) {
        self.globals = globals;
    }

    // -----------------------------------------------------------------------
    // Execution interception
    // -----------------------------------------------------------------------

    /// Called by the VM before executing each instruction.
    ///
    /// Returns `true` if execution should pause (breakpoint hit, step
    /// complete, etc.), `false` if execution should continue.
    pub fn should_pause(
        &mut self,
        file: &str,
        line: u32,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        // Check breakpoints.
        let mut triggered_bp: Option<(u32, Option<String>)> = None;
        for bp in self.breakpoints.values_mut() {
            if bp.file == file && bp.line == line {
                let locals = self.call_stack.last()
                    .map(|f| &f.locals)
                    .cloned()
                    .unwrap_or_default();

                if bp.should_break(&locals) {
                    if let Some(ref msg) = bp.log_message {
                        // Logpoint: log but don't break.
                        triggered_bp = Some((bp.id, Some(msg.clone())));
                    } else {
                        triggered_bp = Some((bp.id, None));
                    }
                    break;
                }
            }
        }

        if let Some((bp_id, log_msg)) = triggered_bp {
            if let Some(msg) = log_msg {
                // Logpoint.
                self.events.push(DebugEvent::LogpointMessage {
                    breakpoint_id: bp_id,
                    message: msg.clone(),
                });
                self.output_log.push(msg);
                return false;
            } else {
                // Real breakpoint.
                self.state = ExecutionState::Paused(PauseReason::Breakpoint(bp_id));
                self.events.push(DebugEvent::BreakpointHit {
                    breakpoint_id: bp_id,
                    file: file.to_string(),
                    line,
                });
                return true;
            }
        }

        // Check step state.
        match &self.state {
            ExecutionState::StepInto => {
                self.state = ExecutionState::Paused(PauseReason::Step);
                self.events.push(DebugEvent::StepComplete {
                    file: file.to_string(),
                    line,
                });
                return true;
            }
            ExecutionState::StepOver { target_depth } => {
                if self.call_stack.len() <= *target_depth {
                    self.state = ExecutionState::Paused(PauseReason::Step);
                    self.events.push(DebugEvent::StepComplete {
                        file: file.to_string(),
                        line,
                    });
                    return true;
                }
            }
            ExecutionState::StepOut { target_depth } => {
                if self.call_stack.len() <= *target_depth {
                    self.state = ExecutionState::Paused(PauseReason::Step);
                    self.events.push(DebugEvent::StepComplete {
                        file: file.to_string(),
                        line,
                    });
                    return true;
                }
            }
            _ => {}
        }

        false
    }

    /// Report an exception to the debugger.
    pub fn report_exception(&mut self, error: &ScriptError) {
        let file = self.current_file().unwrap_or("<unknown>").to_string();
        let line = self.current_line().unwrap_or(0);
        let message = format!("{error}");

        self.state = ExecutionState::Paused(PauseReason::Exception(message.clone()));
        self.events.push(DebugEvent::Exception {
            message,
            file,
            line,
        });
    }

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    /// Poll and drain all pending debug events.
    pub fn poll_events(&mut self) -> Vec<DebugEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if there are pending events.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    // -----------------------------------------------------------------------
    // REPL / Console
    // -----------------------------------------------------------------------

    /// Evaluate an expression in the current scope.
    ///
    /// Supports simple variable lookups and basic comparisons.
    pub fn evaluate_expression(&self, expression: &str) -> Option<ScriptValue> {
        let expr = expression.trim();

        // Check if it's a simple variable lookup.
        // First check locals in the topmost frame.
        if let Some(frame) = self.call_stack.last() {
            if let Some(val) = frame.locals.get(expr) {
                return Some(val.clone());
            }
        }

        // Then check globals.
        if let Some(val) = self.globals.get(expr) {
            return Some(val.clone());
        }

        // Try to parse as a literal.
        if let Ok(i) = expr.parse::<i64>() {
            return Some(ScriptValue::Int(i));
        }
        if let Ok(f) = expr.parse::<f64>() {
            return Some(ScriptValue::Float(f));
        }
        if expr == "true" {
            return Some(ScriptValue::Bool(true));
        }
        if expr == "false" {
            return Some(ScriptValue::Bool(false));
        }
        if expr == "nil" {
            return Some(ScriptValue::Nil);
        }
        if expr.starts_with('"') && expr.ends_with('"') && expr.len() >= 2 {
            return Some(ScriptValue::from_string(&expr[1..expr.len() - 1]));
        }

        None
    }

    /// Execute a REPL command while paused.
    ///
    /// Supports commands like:
    /// - `p <expr>` or `print <expr>` — evaluate and print an expression
    /// - `locals` — list all local variables
    /// - `globals` — list all global variables
    /// - `stack` — show the call stack
    /// - `bp list` — list breakpoints
    /// - `c` or `continue` — resume execution
    /// - `n` or `next` — step over
    /// - `s` or `step` — step into
    /// - `out` — step out
    pub fn repl_command(&mut self, command: &str) -> String {
        let cmd = command.trim();
        self.repl_history.push(cmd.to_string());

        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let verb = parts[0];
        let arg = if parts.len() > 1 { parts[1] } else { "" };

        match verb {
            "p" | "print" | "eval" => {
                if arg.is_empty() {
                    return "Usage: print <expression>".to_string();
                }
                match self.evaluate_expression(arg) {
                    Some(val) => format!("{val}"),
                    None => format!("Cannot evaluate: '{arg}'"),
                }
            }
            "locals" | "l" => {
                if let Some(frame) = self.call_stack.last() {
                    let mut output = format!(
                        "Locals in {} (line {}):\n",
                        frame.function_name, frame.line
                    );
                    if frame.locals.is_empty() {
                        output.push_str("  (none)");
                    } else {
                        let mut sorted: Vec<_> = frame.locals.iter().collect();
                        sorted.sort_by_key(|(k, _)| k.as_str());
                        for (name, val) in sorted {
                            output.push_str(&format!("  {name} = {val}\n"));
                        }
                    }
                    output
                } else {
                    "No active frame".to_string()
                }
            }
            "globals" | "g" => {
                let mut output = "Global variables:\n".to_string();
                if self.globals.is_empty() {
                    output.push_str("  (none)");
                } else {
                    let mut sorted: Vec<_> = self.globals.iter().collect();
                    sorted.sort_by_key(|(k, _)| k.as_str());
                    for (name, val) in sorted {
                        output.push_str(&format!("  {name} = {val}\n"));
                    }
                }
                output
            }
            "stack" | "bt" | "backtrace" => {
                let mut output = "Call stack:\n".to_string();
                for (i, frame) in self.call_stack.iter().rev().enumerate() {
                    output.push_str(&format!(
                        "  #{} {} at {}:{}\n",
                        i, frame.function_name, frame.file, frame.line
                    ));
                }
                output
            }
            "bp" => match arg {
                "list" | "" => {
                    let mut output = "Breakpoints:\n".to_string();
                    let mut bps: Vec<_> = self.breakpoints.values().collect();
                    bps.sort_by_key(|bp| bp.id);
                    for bp in bps {
                        let status = if bp.enabled { "enabled" } else { "disabled" };
                        output.push_str(&format!(
                            "  #{}: {}:{} ({}, hits: {})\n",
                            bp.id, bp.file, bp.line, status, bp.hit_count
                        ));
                    }
                    output
                }
                "clear" => {
                    self.clear_breakpoints();
                    "All breakpoints cleared".to_string()
                }
                _ => format!("Unknown bp subcommand: '{arg}'"),
            },
            "c" | "continue" => {
                self.continue_execution();
                "Continuing...".to_string()
            }
            "n" | "next" => {
                self.step_over();
                "Stepping over...".to_string()
            }
            "s" | "step" => {
                self.step_into();
                "Stepping into...".to_string()
            }
            "out" | "finish" => {
                self.step_out();
                "Stepping out...".to_string()
            }
            "help" | "h" | "?" => {
                "Commands:\n  p/print <expr>  - evaluate expression\n  locals/l        - show local variables\n  globals/g       - show global variables\n  stack/bt        - show call stack\n  bp list         - list breakpoints\n  bp clear        - clear all breakpoints\n  c/continue      - resume execution\n  n/next          - step over\n  s/step          - step into\n  out/finish      - step out\n  help/h/?        - show this help".to_string()
            }
            _ => format!("Unknown command: '{verb}'. Type 'help' for available commands."),
        }
    }

    /// Get the REPL history.
    pub fn repl_history(&self) -> &[String] {
        &self.repl_history
    }

    /// Get the debugger output log.
    pub fn output_log(&self) -> &[String] {
        &self.output_log
    }

    /// Reset the debugger state for a new execution.
    pub fn reset(&mut self) {
        self.state = ExecutionState::Running;
        self.call_stack.clear();
        self.globals.clear();
        self.events.clear();
        self.previous_values.clear();
        // Keep breakpoints and watches.
    }

    /// Full reset including breakpoints and watches.
    pub fn reset_all(&mut self) {
        self.reset();
        self.breakpoints.clear();
        self.watches.clear();
        self.repl_history.clear();
        self.output_log.clear();
    }
}

impl Default for ScriptDebugger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: evaluate simple condition expressions
// ---------------------------------------------------------------------------

/// Evaluate a simple condition expression against local variables.
///
/// Supports basic forms:
/// - `variable` — checks truthiness
/// - `variable == value`
/// - `variable != value`
/// - `variable > value`
/// - `variable < value`
/// - `variable >= value`
/// - `variable <= value`
fn evaluate_condition(expr: &str, locals: &HashMap<String, ScriptValue>) -> bool {
    let expr = expr.trim();

    // Try comparison operators.
    for (op, evaluator) in &[
        (">=", eval_ge as fn(&ScriptValue, &str) -> bool),
        ("<=", eval_le as fn(&ScriptValue, &str) -> bool),
        ("==", eval_eq as fn(&ScriptValue, &str) -> bool),
        ("!=", eval_ne as fn(&ScriptValue, &str) -> bool),
        (">", eval_gt as fn(&ScriptValue, &str) -> bool),
        ("<", eval_lt as fn(&ScriptValue, &str) -> bool),
    ] {
        if let Some(idx) = expr.find(op) {
            let var_name = expr[..idx].trim();
            let val_str = expr[idx + op.len()..].trim();

            if let Some(var_val) = locals.get(var_name) {
                return evaluator(var_val, val_str);
            }
            return false;
        }
    }

    // Simple truthiness check.
    if let Some(val) = locals.get(expr) {
        return val.is_truthy();
    }

    false
}

fn parse_comparison_value(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn to_compare_f64(val: &ScriptValue) -> Option<f64> {
    match val {
        ScriptValue::Int(i) => Some(*i as f64),
        ScriptValue::Float(f) => Some(*f),
        _ => None,
    }
}

fn eval_eq(val: &ScriptValue, rhs: &str) -> bool {
    if let (Some(a), Some(b)) = (to_compare_f64(val), parse_comparison_value(rhs)) {
        (a - b).abs() < 1e-10
    } else {
        let rhs_trimmed = rhs.trim().trim_matches('"');
        if let ScriptValue::String(s) = val {
            s.as_ref() == rhs_trimmed
        } else if rhs.trim() == "true" {
            val.is_truthy()
        } else if rhs.trim() == "false" {
            val.is_falsy()
        } else if rhs.trim() == "nil" {
            val.is_nil()
        } else {
            false
        }
    }
}

fn eval_ne(val: &ScriptValue, rhs: &str) -> bool {
    !eval_eq(val, rhs)
}

fn eval_gt(val: &ScriptValue, rhs: &str) -> bool {
    if let (Some(a), Some(b)) = (to_compare_f64(val), parse_comparison_value(rhs)) {
        a > b
    } else {
        false
    }
}

fn eval_lt(val: &ScriptValue, rhs: &str) -> bool {
    if let (Some(a), Some(b)) = (to_compare_f64(val), parse_comparison_value(rhs)) {
        a < b
    } else {
        false
    }
}

fn eval_ge(val: &ScriptValue, rhs: &str) -> bool {
    if let (Some(a), Some(b)) = (to_compare_f64(val), parse_comparison_value(rhs)) {
        a >= b
    } else {
        false
    }
}

fn eval_le(val: &ScriptValue, rhs: &str) -> bool {
    if let (Some(a), Some(b)) = (to_compare_f64(val), parse_comparison_value(rhs)) {
        a <= b
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let dbg = ScriptDebugger::new();
        assert!(dbg.enabled);
        assert!(dbg.is_running());
        assert!(!dbg.is_paused());
    }

    #[test]
    fn test_add_remove_breakpoint() {
        let mut dbg = ScriptDebugger::new();
        let id = dbg.add_breakpoint("test.gs", 10);
        assert_eq!(dbg.breakpoints().len(), 1);
        assert_eq!(dbg.get_breakpoint(id).unwrap().line, 10);

        dbg.remove_breakpoint(id);
        assert_eq!(dbg.breakpoints().len(), 0);
    }

    #[test]
    fn test_breakpoint_at_location() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);
        dbg.add_breakpoint("test.gs", 20);

        assert!(dbg.remove_breakpoint_at("test.gs", 10));
        assert_eq!(dbg.breakpoints().len(), 1);
    }

    #[test]
    fn test_breakpoint_enable_disable() {
        let mut dbg = ScriptDebugger::new();
        let id = dbg.add_breakpoint("test.gs", 10);
        assert!(dbg.get_breakpoint(id).unwrap().enabled);

        dbg.set_breakpoint_enabled(id, false);
        assert!(!dbg.get_breakpoint(id).unwrap().enabled);
    }

    #[test]
    fn test_should_pause_at_breakpoint() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);

        // Push a frame so the debugger has context.
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        assert!(dbg.should_pause("test.gs", 10));
        assert!(dbg.is_paused());
    }

    #[test]
    fn test_should_not_pause_at_wrong_line() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);

        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        assert!(!dbg.should_pause("test.gs", 5));
        assert!(!dbg.is_paused());
    }

    #[test]
    fn test_conditional_breakpoint() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_conditional_breakpoint(
            "test.gs",
            10,
            BreakpointCondition::When("x > 5".into()),
        );

        let mut frame = StackFrame::new("main", "test.gs", 10);
        frame.set_local("x", ScriptValue::Int(3));
        dbg.push_frame(frame);

        // x = 3, condition "x > 5" is false.
        assert!(!dbg.should_pause("test.gs", 10));

        // Update x to 10.
        dbg.update_local("x", ScriptValue::Int(10));
        assert!(dbg.should_pause("test.gs", 10));
    }

    #[test]
    fn test_hit_count_breakpoint() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_conditional_breakpoint(
            "test.gs",
            10,
            BreakpointCondition::HitCount(3),
        );

        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        // First two hits should not pause.
        assert!(!dbg.should_pause("test.gs", 10));
        dbg.state = ExecutionState::Running;
        assert!(!dbg.should_pause("test.gs", 10));
        dbg.state = ExecutionState::Running;
        // Third hit should pause.
        assert!(dbg.should_pause("test.gs", 10));
    }

    #[test]
    fn test_step_into() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        dbg.step_into();
        assert!(dbg.should_pause("test.gs", 2));
        assert!(dbg.is_paused());
    }

    #[test]
    fn test_step_over() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        dbg.step_over();

        // Same depth: should pause.
        assert!(dbg.should_pause("test.gs", 2));
    }

    #[test]
    fn test_step_over_skips_deeper_frame() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.step_over();

        // Enter a function (push deeper frame).
        dbg.push_frame(StackFrame::new("helper", "test.gs", 50));

        // Should NOT pause inside the function.
        assert!(!dbg.should_pause("test.gs", 51));

        // Pop back to the original frame.
        dbg.pop_frame();

        // Should pause now at the original depth.
        assert!(dbg.should_pause("test.gs", 3));
    }

    #[test]
    fn test_step_out() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.push_frame(StackFrame::new("helper", "test.gs", 50));

        dbg.step_out();

        // Should not pause while still in the function.
        assert!(!dbg.should_pause("test.gs", 51));

        // Pop back to main.
        dbg.pop_frame();

        // Should pause after returning.
        assert!(dbg.should_pause("test.gs", 2));
    }

    #[test]
    fn test_call_stack() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.push_frame(StackFrame::new("helper", "test.gs", 50));

        let stack = dbg.get_call_stack();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0].function_name, "main");
        assert_eq!(stack[1].function_name, "helper");

        assert_eq!(dbg.current_function(), Some("helper"));
    }

    #[test]
    fn test_locals_inspection() {
        let mut dbg = ScriptDebugger::new();
        let mut frame = StackFrame::new("main", "test.gs", 1);
        frame.set_local("x", ScriptValue::Int(42));
        frame.set_local("name", ScriptValue::from_string("test"));
        dbg.push_frame(frame);

        let locals = dbg.get_locals(0).unwrap();
        assert_eq!(locals.len(), 2);
    }

    #[test]
    fn test_globals_inspection() {
        let mut dbg = ScriptDebugger::new();
        dbg.set_global("player_health", ScriptValue::Int(100));

        let globals = dbg.get_globals();
        assert_eq!(globals.len(), 1);
        assert_eq!(globals[0].0, "player_health");
    }

    #[test]
    fn test_evaluate_expression() {
        let mut dbg = ScriptDebugger::new();
        let mut frame = StackFrame::new("main", "test.gs", 1);
        frame.set_local("x", ScriptValue::Int(42));
        dbg.push_frame(frame);
        dbg.set_global("g", ScriptValue::Float(3.14));

        assert_eq!(
            dbg.evaluate_expression("x"),
            Some(ScriptValue::Int(42))
        );
        assert_eq!(
            dbg.evaluate_expression("g"),
            Some(ScriptValue::Float(3.14))
        );
        assert_eq!(
            dbg.evaluate_expression("42"),
            Some(ScriptValue::Int(42))
        );
        assert_eq!(
            dbg.evaluate_expression("true"),
            Some(ScriptValue::Bool(true))
        );
        assert_eq!(
            dbg.evaluate_expression("nil"),
            Some(ScriptValue::Nil)
        );
    }

    #[test]
    fn test_watch_expression() {
        let mut dbg = ScriptDebugger::new();
        let id = dbg.add_watch("x + y");
        assert_eq!(dbg.watches().len(), 1);

        dbg.remove_watch(id);
        assert_eq!(dbg.watches().len(), 0);
    }

    #[test]
    fn test_logpoint() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_logpoint("test.gs", 10, "Value at line 10");

        dbg.push_frame(StackFrame::new("main", "test.gs", 1));

        // Should NOT pause (logpoint just logs).
        assert!(!dbg.should_pause("test.gs", 10));

        let events = dbg.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], DebugEvent::LogpointMessage { .. }));
    }

    #[test]
    fn test_events() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);

        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.should_pause("test.gs", 10);

        assert!(dbg.has_events());
        let events = dbg.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], DebugEvent::BreakpointHit { .. }));

        // Events should be drained.
        assert!(!dbg.has_events());
    }

    #[test]
    fn test_repl_command_help() {
        let mut dbg = ScriptDebugger::new();
        let output = dbg.repl_command("help");
        assert!(output.contains("Commands:"));
    }

    #[test]
    fn test_repl_command_print() {
        let mut dbg = ScriptDebugger::new();
        let mut frame = StackFrame::new("main", "test.gs", 1);
        frame.set_local("x", ScriptValue::Int(42));
        dbg.push_frame(frame);

        let output = dbg.repl_command("p x");
        assert!(output.contains("42"));
    }

    #[test]
    fn test_repl_command_locals() {
        let mut dbg = ScriptDebugger::new();
        let mut frame = StackFrame::new("main", "test.gs", 1);
        frame.set_local("x", ScriptValue::Int(42));
        dbg.push_frame(frame);

        let output = dbg.repl_command("locals");
        assert!(output.contains("x = 42"));
    }

    #[test]
    fn test_repl_command_stack() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.push_frame(StackFrame::new("helper", "test.gs", 50));

        let output = dbg.repl_command("stack");
        assert!(output.contains("main"));
        assert!(output.contains("helper"));
    }

    #[test]
    fn test_repl_command_continue() {
        let mut dbg = ScriptDebugger::new();
        dbg.state = ExecutionState::Paused(PauseReason::Step);

        dbg.repl_command("c");
        assert!(dbg.is_running());
    }

    #[test]
    fn test_reset() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);
        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        dbg.set_global("x", ScriptValue::Int(1));

        dbg.reset();

        assert!(dbg.is_running());
        assert!(dbg.get_call_stack().is_empty());
        assert!(dbg.get_globals().is_empty());
        // Breakpoints should be preserved.
        assert_eq!(dbg.breakpoints().len(), 1);
    }

    #[test]
    fn test_reset_all() {
        let mut dbg = ScriptDebugger::new();
        dbg.add_breakpoint("test.gs", 10);
        dbg.add_watch("x");

        dbg.reset_all();

        assert_eq!(dbg.breakpoints().len(), 0);
        assert_eq!(dbg.watches().len(), 0);
    }

    #[test]
    fn test_condition_evaluation() {
        let mut locals = HashMap::new();
        locals.insert("x".to_string(), ScriptValue::Int(10));
        locals.insert("name".to_string(), ScriptValue::from_string("hello"));

        assert!(evaluate_condition("x > 5", &locals));
        assert!(!evaluate_condition("x > 15", &locals));
        assert!(evaluate_condition("x == 10", &locals));
        assert!(evaluate_condition("x != 5", &locals));
        assert!(evaluate_condition("x >= 10", &locals));
        assert!(evaluate_condition("x <= 10", &locals));
        assert!(evaluate_condition("x", &locals)); // truthiness
    }

    #[test]
    fn test_disabled_debugger() {
        let mut dbg = ScriptDebugger::new();
        dbg.enabled = false;
        dbg.add_breakpoint("test.gs", 10);

        dbg.push_frame(StackFrame::new("main", "test.gs", 1));
        assert!(!dbg.should_pause("test.gs", 10));
    }

    #[test]
    fn test_exception_reporting() {
        let mut dbg = ScriptDebugger::new();
        dbg.push_frame(StackFrame::new("main", "test.gs", 10));

        dbg.report_exception(&ScriptError::RuntimeError("division by zero".into()));

        assert!(dbg.is_paused());
        let events = dbg.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], DebugEvent::Exception { .. }));
    }
}
