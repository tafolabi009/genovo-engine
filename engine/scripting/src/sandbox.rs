//! Script sandboxing for the Genovo engine.
//!
//! Provides resource limits and permission controls for executing untrusted
//! scripts safely. Supports instruction count limits, memory limits, blocked
//! operations, execution timeouts, per-script resource quotas, audit logging,
//! and multiple permission levels.
//!
//! # Permission Levels
//!
//! | Level       | Filesystem | Network | Natives | Instruction Limit |
//! |-------------|-----------|---------|---------|-------------------|
//! | Untrusted   | No        | No      | Limited | 100,000           |
//! | Game        | Read-only | No      | Most    | 10,000,000        |
//! | Editor      | Read/Write| Limited | All     | 100,000,000       |
//! | System      | Full      | Full    | All     | Unlimited         |

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PermissionLevel
// ---------------------------------------------------------------------------

/// Permission level for a script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PermissionLevel {
    /// Untrusted: community mods, downloaded scripts.
    Untrusted,
    /// Game: scripts that ship with the game.
    Game,
    /// Editor: editor plugins and tools.
    Editor,
    /// System: engine-internal scripts, full access.
    System,
}

impl PermissionLevel {
    /// Default instruction limit for this level.
    pub fn default_instruction_limit(self) -> u64 {
        match self {
            PermissionLevel::Untrusted => 100_000,
            PermissionLevel::Game => 10_000_000,
            PermissionLevel::Editor => 100_000_000,
            PermissionLevel::System => u64::MAX,
        }
    }

    /// Default memory limit (bytes) for this level.
    pub fn default_memory_limit(self) -> u64 {
        match self {
            PermissionLevel::Untrusted => 1 * 1024 * 1024,      // 1 MB
            PermissionLevel::Game => 64 * 1024 * 1024,           // 64 MB
            PermissionLevel::Editor => 256 * 1024 * 1024,        // 256 MB
            PermissionLevel::System => u64::MAX,
        }
    }

    /// Default execution timeout for this level.
    pub fn default_timeout(self) -> Duration {
        match self {
            PermissionLevel::Untrusted => Duration::from_millis(100),
            PermissionLevel::Game => Duration::from_secs(5),
            PermissionLevel::Editor => Duration::from_secs(30),
            PermissionLevel::System => Duration::from_secs(u64::MAX / 2),
        }
    }

    /// Whether filesystem access is allowed.
    pub fn allows_filesystem(self) -> bool {
        matches!(self, PermissionLevel::Editor | PermissionLevel::System)
    }

    /// Whether filesystem read is allowed.
    pub fn allows_filesystem_read(self) -> bool {
        matches!(
            self,
            PermissionLevel::Game | PermissionLevel::Editor | PermissionLevel::System
        )
    }

    /// Whether network access is allowed.
    pub fn allows_network(self) -> bool {
        matches!(self, PermissionLevel::System)
    }

    /// Whether limited network (e.g., HTTP GET to allowlisted domains) is allowed.
    pub fn allows_limited_network(self) -> bool {
        matches!(self, PermissionLevel::Editor | PermissionLevel::System)
    }
}

impl fmt::Display for PermissionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PermissionLevel::Untrusted => write!(f, "Untrusted"),
            PermissionLevel::Game => write!(f, "Game"),
            PermissionLevel::Editor => write!(f, "Editor"),
            PermissionLevel::System => write!(f, "System"),
        }
    }
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

/// An operation that may be restricted by the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Read a file.
    FileRead(String),
    /// Write a file.
    FileWrite(String),
    /// Delete a file.
    FileDelete(String),
    /// List directory contents.
    DirectoryList(String),
    /// Create a directory.
    DirectoryCreate(String),
    /// Open a network connection.
    NetworkConnect(String),
    /// Send network data.
    NetworkSend,
    /// Execute an OS command.
    OsCommand(String),
    /// Load a native library.
    LoadNative(String),
    /// Call a specific native function.
    CallNative(String),
    /// Allocate memory beyond the limit.
    MemoryAlloc(u64),
    /// Spawn a thread.
    SpawnThread,
    /// Access environment variables.
    EnvAccess(String),
    /// Modify global state.
    GlobalMutation(String),
    /// Arbitrary custom operation.
    Custom(String),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::FileRead(path) => write!(f, "FileRead({})", path),
            Operation::FileWrite(path) => write!(f, "FileWrite({})", path),
            Operation::FileDelete(path) => write!(f, "FileDelete({})", path),
            Operation::DirectoryList(path) => write!(f, "DirectoryList({})", path),
            Operation::DirectoryCreate(path) => write!(f, "DirectoryCreate({})", path),
            Operation::NetworkConnect(addr) => write!(f, "NetworkConnect({})", addr),
            Operation::NetworkSend => write!(f, "NetworkSend"),
            Operation::OsCommand(cmd) => write!(f, "OsCommand({})", cmd),
            Operation::LoadNative(lib) => write!(f, "LoadNative({})", lib),
            Operation::CallNative(func) => write!(f, "CallNative({})", func),
            Operation::MemoryAlloc(bytes) => write!(f, "MemoryAlloc({} bytes)", bytes),
            Operation::SpawnThread => write!(f, "SpawnThread"),
            Operation::EnvAccess(var) => write!(f, "EnvAccess({})", var),
            Operation::GlobalMutation(name) => write!(f, "GlobalMutation({})", name),
            Operation::Custom(op) => write!(f, "Custom({})", op),
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxViolation
// ---------------------------------------------------------------------------

/// A recorded sandbox violation.
#[derive(Debug, Clone)]
pub struct SandboxViolation {
    /// The operation that was blocked.
    pub operation: Operation,
    /// The script that attempted the operation.
    pub script_id: String,
    /// When the violation occurred.
    pub timestamp: Instant,
    /// The permission level of the script.
    pub permission_level: PermissionLevel,
    /// Additional context/message.
    pub message: String,
    /// Whether the violation was fatal (script terminated).
    pub fatal: bool,
}

impl fmt::Display for SandboxViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} tried {} (level: {}) - {}{}",
            if self.fatal { "FATAL" } else { "WARN" },
            self.script_id,
            self.operation,
            self.permission_level,
            self.message,
            if self.fatal { " [TERMINATED]" } else { "" },
        )
    }
}

// ---------------------------------------------------------------------------
// ResourceQuota
// ---------------------------------------------------------------------------

/// Resource quota for a single script.
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    /// Maximum instructions that can be executed.
    pub instruction_limit: u64,
    /// Maximum memory usage in bytes.
    pub memory_limit: u64,
    /// Maximum execution time.
    pub timeout: Duration,
    /// Maximum number of function calls (call stack depth protection).
    pub max_call_depth: u32,
    /// Maximum number of objects/allocations.
    pub max_objects: u64,
    /// Maximum output size (print/log characters).
    pub max_output_chars: u64,
    /// Maximum file I/O operations per execution.
    pub max_file_ops: u32,
    /// Maximum network operations per execution.
    pub max_network_ops: u32,
}

impl ResourceQuota {
    /// Create a quota for a given permission level.
    pub fn for_level(level: PermissionLevel) -> Self {
        Self {
            instruction_limit: level.default_instruction_limit(),
            memory_limit: level.default_memory_limit(),
            timeout: level.default_timeout(),
            max_call_depth: match level {
                PermissionLevel::Untrusted => 32,
                PermissionLevel::Game => 128,
                PermissionLevel::Editor => 256,
                PermissionLevel::System => 1024,
            },
            max_objects: match level {
                PermissionLevel::Untrusted => 10_000,
                PermissionLevel::Game => 1_000_000,
                PermissionLevel::Editor => 10_000_000,
                PermissionLevel::System => u64::MAX,
            },
            max_output_chars: match level {
                PermissionLevel::Untrusted => 10_000,
                PermissionLevel::Game => 1_000_000,
                PermissionLevel::Editor => u64::MAX,
                PermissionLevel::System => u64::MAX,
            },
            max_file_ops: match level {
                PermissionLevel::Untrusted => 0,
                PermissionLevel::Game => 10,
                PermissionLevel::Editor => 1000,
                PermissionLevel::System => u32::MAX,
            },
            max_network_ops: match level {
                PermissionLevel::Untrusted => 0,
                PermissionLevel::Game => 0,
                PermissionLevel::Editor => 100,
                PermissionLevel::System => u32::MAX,
            },
        }
    }

    /// Create an unlimited quota (for system-level scripts).
    pub fn unlimited() -> Self {
        Self::for_level(PermissionLevel::System)
    }
}

impl fmt::Display for ResourceQuota {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resource Quota:")?;
        writeln!(f, "  instructions: {}", self.instruction_limit)?;
        writeln!(f, "  memory:       {} MB", self.memory_limit / (1024 * 1024))?;
        writeln!(f, "  timeout:      {:.1}s", self.timeout.as_secs_f64())?;
        writeln!(f, "  call depth:   {}", self.max_call_depth)?;
        writeln!(f, "  max objects:  {}", self.max_objects)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ResourceUsage
// ---------------------------------------------------------------------------

/// Current resource usage counters for a sandboxed script.
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Instructions executed.
    pub instructions: u64,
    /// Current memory usage in bytes.
    pub memory_bytes: u64,
    /// Peak memory usage.
    pub peak_memory_bytes: u64,
    /// Current call depth.
    pub call_depth: u32,
    /// Peak call depth.
    pub peak_call_depth: u32,
    /// Number of objects allocated.
    pub objects_allocated: u64,
    /// Output characters produced.
    pub output_chars: u64,
    /// File operations performed.
    pub file_ops: u32,
    /// Network operations performed.
    pub network_ops: u32,
    /// Execution start time.
    pub start_time: Option<Instant>,
}

impl ResourceUsage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start tracking execution time.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Elapsed execution time.
    pub fn elapsed(&self) -> Duration {
        self.start_time
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Memory usage as a fraction of peak.
    pub fn memory_utilization(&self) -> f64 {
        if self.peak_memory_bytes == 0 {
            0.0
        } else {
            self.memory_bytes as f64 / self.peak_memory_bytes as f64
        }
    }
}

impl fmt::Display for ResourceUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resource Usage:")?;
        writeln!(f, "  instructions: {}", self.instructions)?;
        writeln!(f, "  memory:       {} KB (peak {} KB)", self.memory_bytes / 1024, self.peak_memory_bytes / 1024)?;
        writeln!(f, "  call depth:   {} (peak {})", self.call_depth, self.peak_call_depth)?;
        writeln!(f, "  objects:      {}", self.objects_allocated)?;
        writeln!(f, "  elapsed:      {:.3}ms", self.elapsed().as_secs_f64() * 1000.0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SandboxPolicy
// ---------------------------------------------------------------------------

/// Policy that controls what operations are allowed.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Permission level.
    pub level: PermissionLevel,
    /// Explicitly blocked operations (overrides level defaults).
    pub blocked_ops: HashSet<String>,
    /// Explicitly allowed operations (overrides level defaults).
    pub allowed_ops: HashSet<String>,
    /// Allowed native functions.
    pub allowed_natives: HashSet<String>,
    /// Blocked native functions.
    pub blocked_natives: HashSet<String>,
    /// Allowed filesystem paths (prefix matching).
    pub allowed_paths: Vec<String>,
    /// Blocked filesystem paths.
    pub blocked_paths: Vec<String>,
    /// Allowed network hosts.
    pub allowed_hosts: Vec<String>,
    /// Whether to terminate on violation or just log.
    pub strict_mode: bool,
}

impl SandboxPolicy {
    /// Create a policy for the given permission level.
    pub fn for_level(level: PermissionLevel) -> Self {
        Self {
            level,
            blocked_ops: HashSet::new(),
            allowed_ops: HashSet::new(),
            allowed_natives: HashSet::new(),
            blocked_natives: HashSet::new(),
            allowed_paths: Vec::new(),
            blocked_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            strict_mode: true,
        }
    }

    /// Block a specific operation category.
    pub fn block_op(&mut self, op: &str) {
        self.blocked_ops.insert(op.to_string());
    }

    /// Allow a specific operation category.
    pub fn allow_op(&mut self, op: &str) {
        self.allowed_ops.insert(op.to_string());
    }

    /// Allow a native function.
    pub fn allow_native(&mut self, name: &str) {
        self.allowed_natives.insert(name.to_string());
    }

    /// Block a native function.
    pub fn block_native(&mut self, name: &str) {
        self.blocked_natives.insert(name.to_string());
    }

    /// Add an allowed filesystem path prefix.
    pub fn allow_path(&mut self, path: &str) {
        self.allowed_paths.push(path.to_string());
    }

    /// Block a filesystem path prefix.
    pub fn block_path(&mut self, path: &str) {
        self.blocked_paths.push(path.to_string());
    }

    /// Add an allowed network host.
    pub fn allow_host(&mut self, host: &str) {
        self.allowed_hosts.push(host.to_string());
    }

    /// Check whether a filesystem path is allowed.
    pub fn is_path_allowed(&self, path: &str) -> bool {
        // Check blocked first.
        for blocked in &self.blocked_paths {
            if path.starts_with(blocked) {
                return false;
            }
        }
        // If there are allowed paths, path must match one.
        if !self.allowed_paths.is_empty() {
            return self.allowed_paths.iter().any(|p| path.starts_with(p));
        }
        // Otherwise, defer to permission level.
        self.level.allows_filesystem()
    }

    /// Check whether a network host is allowed.
    pub fn is_host_allowed(&self, host: &str) -> bool {
        if !self.allowed_hosts.is_empty() {
            return self.allowed_hosts.iter().any(|h| host == h || host.ends_with(h));
        }
        self.level.allows_network()
    }

    /// Check whether a native function is allowed.
    pub fn is_native_allowed(&self, name: &str) -> bool {
        if self.blocked_natives.contains(name) {
            return false;
        }
        if !self.allowed_natives.is_empty() {
            return self.allowed_natives.contains(name);
        }
        // At system level, all natives are allowed.
        self.level >= PermissionLevel::Game
    }
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// A sandbox environment for executing scripts with resource limits
/// and permission controls.
pub struct Sandbox {
    /// Script identifier.
    script_id: String,
    /// Policy controlling permissions.
    policy: SandboxPolicy,
    /// Resource quota.
    quota: ResourceQuota,
    /// Current resource usage.
    usage: ResourceUsage,
    /// Audit log of violations.
    violations: Vec<SandboxViolation>,
    /// Maximum violations before termination (even in non-strict mode).
    max_violations: usize,
    /// Whether the sandbox has been terminated.
    terminated: bool,
    /// Termination reason.
    termination_reason: Option<String>,
    /// Audit log of allowed operations (for logging).
    audit_log: Vec<AuditEntry>,
    /// Whether to log allowed operations.
    audit_enabled: bool,
    /// Maximum audit log entries.
    max_audit_entries: usize,
}

/// An audit log entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// The operation performed.
    pub operation: String,
    /// Whether it was allowed.
    pub allowed: bool,
    /// When it occurred.
    pub timestamp: Instant,
    /// Script that performed it.
    pub script_id: String,
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} - {} - {}",
            if self.allowed { "ALLOW" } else { "DENY" },
            self.script_id,
            self.operation,
            "audit",
        )
    }
}

/// Result of a sandbox check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxCheck {
    /// Operation is allowed.
    Allowed,
    /// Operation is denied (with reason).
    Denied(String),
    /// Script must be terminated.
    Terminate(String),
}

impl Sandbox {
    /// Create a new sandbox for a script.
    pub fn new(script_id: &str, level: PermissionLevel) -> Self {
        Self {
            script_id: script_id.to_string(),
            policy: SandboxPolicy::for_level(level),
            quota: ResourceQuota::for_level(level),
            usage: ResourceUsage::new(),
            violations: Vec::new(),
            max_violations: 100,
            terminated: false,
            termination_reason: None,
            audit_log: Vec::new(),
            audit_enabled: false,
            max_audit_entries: 10000,
        }
    }

    /// Create with a custom policy and quota.
    pub fn with_policy(script_id: &str, policy: SandboxPolicy, quota: ResourceQuota) -> Self {
        Self {
            script_id: script_id.to_string(),
            policy,
            quota,
            usage: ResourceUsage::new(),
            violations: Vec::new(),
            max_violations: 100,
            terminated: false,
            termination_reason: None,
            audit_log: Vec::new(),
            audit_enabled: false,
            max_audit_entries: 10000,
        }
    }

    /// Enable audit logging.
    pub fn enable_audit(&mut self) {
        self.audit_enabled = true;
    }

    /// Start execution tracking.
    pub fn begin_execution(&mut self) {
        self.usage.start();
    }

    /// Check if the script should continue executing (resource limits).
    pub fn check_continue(&mut self) -> SandboxCheck {
        if self.terminated {
            return SandboxCheck::Terminate(
                self.termination_reason
                    .clone()
                    .unwrap_or_else(|| "terminated".to_string()),
            );
        }

        // Instruction limit.
        if self.usage.instructions >= self.quota.instruction_limit {
            return self.terminate("instruction limit exceeded");
        }

        // Memory limit.
        if self.usage.memory_bytes > self.quota.memory_limit {
            return self.terminate("memory limit exceeded");
        }

        // Timeout.
        if self.usage.elapsed() >= self.quota.timeout {
            return self.terminate("execution timeout");
        }

        // Call depth.
        if self.usage.call_depth > self.quota.max_call_depth {
            return self.terminate("maximum call depth exceeded");
        }

        SandboxCheck::Allowed
    }

    /// Record an instruction execution.
    pub fn record_instruction(&mut self) {
        self.usage.instructions += 1;
    }

    /// Record N instruction executions.
    pub fn record_instructions(&mut self, count: u64) {
        self.usage.instructions += count;
    }

    /// Record a memory allocation.
    pub fn record_alloc(&mut self, bytes: u64) -> SandboxCheck {
        let new_total = self.usage.memory_bytes + bytes;
        if new_total > self.quota.memory_limit {
            return self.record_violation(
                Operation::MemoryAlloc(bytes),
                &format!(
                    "allocation of {} bytes would exceed limit ({}/{} bytes)",
                    bytes, new_total, self.quota.memory_limit
                ),
            );
        }
        self.usage.memory_bytes = new_total;
        if self.usage.memory_bytes > self.usage.peak_memory_bytes {
            self.usage.peak_memory_bytes = self.usage.memory_bytes;
        }
        self.usage.objects_allocated += 1;
        SandboxCheck::Allowed
    }

    /// Record a memory deallocation.
    pub fn record_free(&mut self, bytes: u64) {
        self.usage.memory_bytes = self.usage.memory_bytes.saturating_sub(bytes);
    }

    /// Record entering a function call.
    pub fn record_call_enter(&mut self) -> SandboxCheck {
        self.usage.call_depth += 1;
        if self.usage.call_depth > self.usage.peak_call_depth {
            self.usage.peak_call_depth = self.usage.call_depth;
        }
        if self.usage.call_depth > self.quota.max_call_depth {
            return self.terminate("maximum call depth exceeded (stack overflow)");
        }
        SandboxCheck::Allowed
    }

    /// Record exiting a function call.
    pub fn record_call_exit(&mut self) {
        self.usage.call_depth = self.usage.call_depth.saturating_sub(1);
    }

    /// Check whether an operation is allowed.
    pub fn check_operation(&mut self, operation: &Operation) -> SandboxCheck {
        let allowed = match operation {
            Operation::FileRead(path) => {
                self.policy.level.allows_filesystem_read()
                    && self.policy.is_path_allowed(path)
                    && self.usage.file_ops < self.quota.max_file_ops
            }
            Operation::FileWrite(path) | Operation::FileDelete(path) => {
                self.policy.level.allows_filesystem()
                    && self.policy.is_path_allowed(path)
                    && self.usage.file_ops < self.quota.max_file_ops
            }
            Operation::DirectoryList(path) | Operation::DirectoryCreate(path) => {
                self.policy.level.allows_filesystem_read()
                    && self.policy.is_path_allowed(path)
            }
            Operation::NetworkConnect(host) => {
                (self.policy.level.allows_network()
                    || self.policy.level.allows_limited_network())
                    && self.policy.is_host_allowed(host)
                    && self.usage.network_ops < self.quota.max_network_ops
            }
            Operation::NetworkSend => {
                self.policy.level.allows_network()
                    && self.usage.network_ops < self.quota.max_network_ops
            }
            Operation::OsCommand(_) => {
                self.policy.level == PermissionLevel::System
            }
            Operation::LoadNative(_) => {
                self.policy.level >= PermissionLevel::Editor
            }
            Operation::CallNative(name) => {
                self.policy.is_native_allowed(name)
            }
            Operation::SpawnThread => {
                self.policy.level == PermissionLevel::System
            }
            Operation::EnvAccess(_) => {
                self.policy.level >= PermissionLevel::Editor
            }
            Operation::GlobalMutation(_) => {
                self.policy.level >= PermissionLevel::Game
            }
            Operation::MemoryAlloc(bytes) => {
                self.usage.memory_bytes + bytes <= self.quota.memory_limit
            }
            Operation::Custom(op) => {
                if self.policy.blocked_ops.contains(op) {
                    false
                } else if self.policy.allowed_ops.contains(op) {
                    true
                } else {
                    self.policy.level >= PermissionLevel::Game
                }
            }
        };

        if self.audit_enabled {
            self.audit_log.push(AuditEntry {
                operation: format!("{}", operation),
                allowed,
                timestamp: Instant::now(),
                script_id: self.script_id.clone(),
            });
            if self.audit_log.len() > self.max_audit_entries {
                self.audit_log.remove(0);
            }
        }

        if allowed {
            // Track resource usage for file/network ops.
            match operation {
                Operation::FileRead(_) | Operation::FileWrite(_) | Operation::FileDelete(_) => {
                    self.usage.file_ops += 1;
                }
                Operation::NetworkConnect(_) | Operation::NetworkSend => {
                    self.usage.network_ops += 1;
                }
                _ => {}
            }
            SandboxCheck::Allowed
        } else {
            self.record_violation(operation.clone(), &format!("operation not allowed: {}", operation))
        }
    }

    /// Record a violation.
    fn record_violation(&mut self, operation: Operation, message: &str) -> SandboxCheck {
        let fatal = self.policy.strict_mode || self.violations.len() >= self.max_violations;

        let violation = SandboxViolation {
            operation,
            script_id: self.script_id.clone(),
            timestamp: Instant::now(),
            permission_level: self.policy.level,
            message: message.to_string(),
            fatal,
        };

        self.violations.push(violation);

        if fatal {
            self.terminate(message)
        } else {
            SandboxCheck::Denied(message.to_string())
        }
    }

    /// Terminate the sandbox.
    fn terminate(&mut self, reason: &str) -> SandboxCheck {
        self.terminated = true;
        self.termination_reason = Some(reason.to_string());
        SandboxCheck::Terminate(reason.to_string())
    }

    // -- Accessors --

    /// Whether the sandbox has been terminated.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Termination reason.
    pub fn termination_reason(&self) -> Option<&str> {
        self.termination_reason.as_deref()
    }

    /// Current resource usage.
    pub fn usage(&self) -> &ResourceUsage {
        &self.usage
    }

    /// Resource quota.
    pub fn quota(&self) -> &ResourceQuota {
        &self.quota
    }

    /// Policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Mutable policy access.
    pub fn policy_mut(&mut self) -> &mut SandboxPolicy {
        &mut self.policy
    }

    /// All recorded violations.
    pub fn violations(&self) -> &[SandboxViolation] {
        &self.violations
    }

    /// Number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Audit log entries.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Script ID.
    pub fn script_id(&self) -> &str {
        &self.script_id
    }

    /// Permission level.
    pub fn permission_level(&self) -> PermissionLevel {
        self.policy.level
    }

    /// Remaining instruction budget.
    pub fn remaining_instructions(&self) -> u64 {
        self.quota
            .instruction_limit
            .saturating_sub(self.usage.instructions)
    }

    /// Remaining memory budget.
    pub fn remaining_memory(&self) -> u64 {
        self.quota
            .memory_limit
            .saturating_sub(self.usage.memory_bytes)
    }

    /// Remaining time budget.
    pub fn remaining_time(&self) -> Duration {
        self.quota
            .timeout
            .checked_sub(self.usage.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Reset the sandbox for a new execution (preserves policy/quota).
    pub fn reset(&mut self) {
        self.usage.reset();
        self.violations.clear();
        self.terminated = false;
        self.termination_reason = None;
        self.audit_log.clear();
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "Sandbox '{}' ({}): {} instructions, {} KB memory, {} violations, {}",
            self.script_id,
            self.policy.level,
            self.usage.instructions,
            self.usage.memory_bytes / 1024,
            self.violations.len(),
            if self.terminated {
                format!("TERMINATED: {}", self.termination_reason.as_deref().unwrap_or("unknown"))
            } else {
                "active".to_string()
            },
        )
    }
}

impl fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sandbox")
            .field("script_id", &self.script_id)
            .field("level", &self.policy.level)
            .field("terminated", &self.terminated)
            .field("instructions", &self.usage.instructions)
            .field("memory_bytes", &self.usage.memory_bytes)
            .field("violations", &self.violations.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SandboxManager
// ---------------------------------------------------------------------------

/// Manages multiple sandboxes for different scripts.
pub struct SandboxManager {
    sandboxes: HashMap<String, Sandbox>,
    default_level: PermissionLevel,
    global_blocked_ops: HashSet<String>,
}

impl SandboxManager {
    pub fn new() -> Self {
        Self {
            sandboxes: HashMap::new(),
            default_level: PermissionLevel::Game,
            global_blocked_ops: HashSet::new(),
        }
    }

    /// Set the default permission level for new sandboxes.
    pub fn set_default_level(&mut self, level: PermissionLevel) {
        self.default_level = level;
    }

    /// Create a sandbox for a script.
    pub fn create(&mut self, script_id: &str, level: PermissionLevel) -> &mut Sandbox {
        let sandbox = Sandbox::new(script_id, level);
        self.sandboxes.insert(script_id.to_string(), sandbox);
        self.sandboxes.get_mut(script_id).unwrap()
    }

    /// Get a sandbox by script ID.
    pub fn get(&self, script_id: &str) -> Option<&Sandbox> {
        self.sandboxes.get(script_id)
    }

    /// Get a mutable sandbox.
    pub fn get_mut(&mut self, script_id: &str) -> Option<&mut Sandbox> {
        self.sandboxes.get_mut(script_id)
    }

    /// Remove a sandbox.
    pub fn remove(&mut self, script_id: &str) -> Option<Sandbox> {
        self.sandboxes.remove(script_id)
    }

    /// Get all active (non-terminated) sandboxes.
    pub fn active_sandboxes(&self) -> Vec<&str> {
        self.sandboxes
            .iter()
            .filter(|(_, s)| !s.is_terminated())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get all violations across all sandboxes.
    pub fn all_violations(&self) -> Vec<&SandboxViolation> {
        self.sandboxes
            .values()
            .flat_map(|s| s.violations())
            .collect()
    }

    /// Total number of managed sandboxes.
    pub fn sandbox_count(&self) -> usize {
        self.sandboxes.len()
    }

    /// Block an operation globally.
    pub fn global_block(&mut self, op: &str) {
        self.global_blocked_ops.insert(op.to_string());
    }
}

impl Default for SandboxManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_levels() {
        assert!(PermissionLevel::System > PermissionLevel::Editor);
        assert!(PermissionLevel::Editor > PermissionLevel::Game);
        assert!(PermissionLevel::Game > PermissionLevel::Untrusted);
    }

    #[test]
    fn test_instruction_limit() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Untrusted);
        sandbox.begin_execution();

        sandbox.record_instructions(99_999);
        assert_eq!(sandbox.check_continue(), SandboxCheck::Allowed);

        sandbox.record_instruction();
        assert!(matches!(sandbox.check_continue(), SandboxCheck::Terminate(_)));
        assert!(sandbox.is_terminated());
    }

    #[test]
    fn test_memory_limit() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Untrusted);
        sandbox.begin_execution();

        // 1 MB limit for untrusted.
        let result = sandbox.record_alloc(512 * 1024);
        assert_eq!(result, SandboxCheck::Allowed);

        let result = sandbox.record_alloc(512 * 1024);
        assert_eq!(result, SandboxCheck::Allowed);

        // This should exceed the limit.
        let result = sandbox.record_alloc(1024);
        assert!(matches!(result, SandboxCheck::Terminate(_)));
    }

    #[test]
    fn test_filesystem_blocked() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Untrusted);
        let result = sandbox.check_operation(&Operation::FileRead("/etc/passwd".to_string()));
        assert!(matches!(result, SandboxCheck::Terminate(_)));
    }

    #[test]
    fn test_filesystem_allowed() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Editor);
        sandbox.policy_mut().allow_path("/game/data/");

        let result = sandbox.check_operation(&Operation::FileRead("/game/data/config.txt".to_string()));
        assert_eq!(result, SandboxCheck::Allowed);
    }

    #[test]
    fn test_native_function_control() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Game);
        sandbox.policy_mut().block_native("dangerous_func");

        let result = sandbox.check_operation(&Operation::CallNative("safe_func".to_string()));
        assert_eq!(result, SandboxCheck::Allowed);

        let result = sandbox.check_operation(&Operation::CallNative("dangerous_func".to_string()));
        assert!(matches!(result, SandboxCheck::Terminate(_)));
    }

    #[test]
    fn test_call_depth() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Untrusted);
        sandbox.begin_execution();

        for _ in 0..32 {
            assert_eq!(sandbox.record_call_enter(), SandboxCheck::Allowed);
        }
        // 33rd call should exceed the limit.
        assert!(matches!(sandbox.record_call_enter(), SandboxCheck::Terminate(_)));
    }

    #[test]
    fn test_audit_logging() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Game);
        sandbox.enable_audit();

        sandbox.check_operation(&Operation::CallNative("print".to_string()));

        assert_eq!(sandbox.audit_log().len(), 1);
        assert!(sandbox.audit_log()[0].allowed);
    }

    #[test]
    fn test_sandbox_manager() {
        let mut manager = SandboxManager::new();
        manager.create("script_a", PermissionLevel::Game);
        manager.create("script_b", PermissionLevel::Untrusted);

        assert_eq!(manager.sandbox_count(), 2);
        assert_eq!(manager.active_sandboxes().len(), 2);

        // Terminate one.
        if let Some(sb) = manager.get_mut("script_a") {
            sb.begin_execution();
            sb.record_instructions(10_000_001);
            sb.check_continue();
        }

        assert_eq!(manager.active_sandboxes().len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut sandbox = Sandbox::new("test", PermissionLevel::Game);
        sandbox.begin_execution();
        sandbox.record_instructions(1000);
        sandbox.record_alloc(5000);

        sandbox.reset();
        assert_eq!(sandbox.usage().instructions, 0);
        assert_eq!(sandbox.usage().memory_bytes, 0);
        assert!(!sandbox.is_terminated());
    }

    #[test]
    fn test_summary() {
        let sandbox = Sandbox::new("my_script", PermissionLevel::Game);
        let summary = sandbox.summary();
        assert!(summary.contains("my_script"));
        assert!(summary.contains("Game"));
    }
}
