//! Coroutine and async support for the Genovo scripting VM.
//!
//! Coroutines allow scripts to pause execution mid-function and resume
//! later, enabling common game patterns like cutscene sequencing, tween
//! animations, and delayed actions without blocking the main thread.
//!
//! # Key concepts
//!
//! - **Coroutine**: a pausable execution context with its own stack.
//! - **yield**: pauses the coroutine, optionally returning a value.
//! - **resume**: continues execution from where yield left off.
//! - **WaitFor**: high-level utilities for common wait patterns.
//! - **CoroutineManager**: updates all active coroutines each frame.
//!
//! # Example (pseudo-script)
//!
//! ```text
//! fn cutscene() {
//!     camera_pan_to(target)
//!     yield wait_seconds(2.0)
//!     show_dialog("Hello!")
//!     yield wait_until(fn() { dialog_closed })
//!     fade_out()
//!     yield wait_seconds(1.0)
//!     load_next_level()
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of active coroutines.
pub const MAX_COROUTINES: usize = 1024;

/// Maximum stack depth per coroutine.
pub const MAX_COROUTINE_STACK: usize = 256;

/// Default coroutine priority (lower = updated first).
pub const DEFAULT_PRIORITY: u32 = 100;

// ---------------------------------------------------------------------------
// CoroutineId
// ---------------------------------------------------------------------------

/// Unique identifier for a coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoroutineId(pub u64);

impl fmt::Display for CoroutineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "co#{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CoroutineState
// ---------------------------------------------------------------------------

/// The execution state of a coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    /// Just created, not yet started.
    Created,
    /// Currently executing.
    Running,
    /// Suspended via yield; waiting to be resumed.
    Suspended,
    /// Finished execution (either returned or errored).
    Completed,
}

impl CoroutineState {
    /// Returns `true` if the coroutine can be resumed.
    pub fn is_resumable(&self) -> bool {
        matches!(self, CoroutineState::Created | CoroutineState::Suspended)
    }

    /// Returns `true` if the coroutine has finished.
    pub fn is_done(&self) -> bool {
        matches!(self, CoroutineState::Completed)
    }
}

impl fmt::Display for CoroutineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineState::Created => write!(f, "Created"),
            CoroutineState::Running => write!(f, "Running"),
            CoroutineState::Suspended => write!(f, "Suspended"),
            CoroutineState::Completed => write!(f, "Completed"),
        }
    }
}

// ---------------------------------------------------------------------------
// CoroutineValue
// ---------------------------------------------------------------------------

/// A value that can be yielded from or passed into a coroutine.
#[derive(Debug, Clone)]
pub enum CoroutineValue {
    /// No value / nil.
    Nil,
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Floating-point.
    Float(f64),
    /// String.
    String(Arc<str>),
    /// 3D vector.
    Vec3(f32, f32, f32),
    /// Entity handle.
    Entity(u64),
    /// A wait instruction (not a data value; tells the manager what to wait for).
    Wait(WaitFor),
}

impl CoroutineValue {
    /// Returns `true` if this is a nil value.
    pub fn is_nil(&self) -> bool {
        matches!(self, CoroutineValue::Nil)
    }

    /// Returns `true` if this is a wait instruction.
    pub fn is_wait(&self) -> bool {
        matches!(self, CoroutineValue::Wait(_))
    }

    /// Try to extract the wait instruction.
    pub fn as_wait(&self) -> Option<&WaitFor> {
        match self {
            CoroutineValue::Wait(w) => Some(w),
            _ => None,
        }
    }

    /// Convert to a float if possible.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            CoroutineValue::Float(v) => Some(*v),
            CoroutineValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Convert to an int if possible.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            CoroutineValue::Int(v) => Some(*v),
            CoroutineValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Truthiness check.
    pub fn is_truthy(&self) -> bool {
        match self {
            CoroutineValue::Nil => false,
            CoroutineValue::Bool(b) => *b,
            _ => true,
        }
    }
}

impl fmt::Display for CoroutineValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineValue::Nil => write!(f, "nil"),
            CoroutineValue::Bool(v) => write!(f, "{}", v),
            CoroutineValue::Int(v) => write!(f, "{}", v),
            CoroutineValue::Float(v) => write!(f, "{}", v),
            CoroutineValue::String(s) => write!(f, "\"{}\"", s),
            CoroutineValue::Vec3(x, y, z) => write!(f, "vec3({}, {}, {})", x, y, z),
            CoroutineValue::Entity(e) => write!(f, "entity({})", e),
            CoroutineValue::Wait(w) => write!(f, "{}", w),
        }
    }
}

// ---------------------------------------------------------------------------
// WaitFor
// ---------------------------------------------------------------------------

/// Specifies what a coroutine is waiting for before it can resume.
#[derive(Debug, Clone)]
pub enum WaitFor {
    /// Wait for a specified number of seconds.
    Seconds(f64),
    /// Wait for a specified number of frames.
    Frames(u32),
    /// Wait until a named condition becomes true.
    Until(String),
    /// Wait for another coroutine to complete.
    Coroutine(CoroutineId),
    /// Wait for an external signal with a given key.
    Signal(String),
    /// Wait for all listed coroutines to complete.
    All(Vec<CoroutineId>),
    /// Wait for any one of the listed coroutines to complete.
    Any(Vec<CoroutineId>),
}

impl WaitFor {
    /// Create a wait-for-seconds instruction.
    pub fn seconds(duration: f64) -> Self {
        WaitFor::Seconds(duration)
    }

    /// Create a wait-for-frames instruction.
    pub fn frames(count: u32) -> Self {
        WaitFor::Frames(count)
    }

    /// Create a wait-until-condition instruction.
    pub fn until(condition: impl Into<String>) -> Self {
        WaitFor::Until(condition.into())
    }

    /// Create a wait-for-coroutine instruction.
    pub fn coroutine(id: CoroutineId) -> Self {
        WaitFor::Coroutine(id)
    }

    /// Create a wait-for-signal instruction.
    pub fn signal(key: impl Into<String>) -> Self {
        WaitFor::Signal(key.into())
    }
}

impl fmt::Display for WaitFor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaitFor::Seconds(d) => write!(f, "wait_seconds({})", d),
            WaitFor::Frames(n) => write!(f, "wait_frames({})", n),
            WaitFor::Until(c) => write!(f, "wait_until({})", c),
            WaitFor::Coroutine(id) => write!(f, "wait_coroutine({})", id),
            WaitFor::Signal(s) => write!(f, "wait_signal({})", s),
            WaitFor::All(ids) => {
                write!(f, "wait_all([")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", id)?;
                }
                write!(f, "])")
            }
            WaitFor::Any(ids) => {
                write!(f, "wait_any([")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", id)?;
                }
                write!(f, "])")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WaitState — internal tracking for active waits
// ---------------------------------------------------------------------------

/// Internal state for tracking an active wait condition.
#[derive(Debug, Clone)]
enum WaitState {
    /// Counting down seconds.
    Timer { remaining: f64 },
    /// Counting down frames.
    FrameCount { remaining: u32 },
    /// Evaluating a named condition each frame.
    Condition { name: String },
    /// Waiting for a specific coroutine to complete.
    CoroutineWait { id: CoroutineId },
    /// Waiting for an external signal.
    SignalWait { key: String },
    /// Waiting for all coroutines.
    AllWait { ids: Vec<CoroutineId> },
    /// Waiting for any coroutine.
    AnyWait { ids: Vec<CoroutineId> },
}

impl WaitState {
    /// Create from a WaitFor instruction.
    fn from_wait_for(wait: &WaitFor) -> Self {
        match wait {
            WaitFor::Seconds(d) => WaitState::Timer { remaining: *d },
            WaitFor::Frames(n) => WaitState::FrameCount { remaining: *n },
            WaitFor::Until(c) => WaitState::Condition { name: c.clone() },
            WaitFor::Coroutine(id) => WaitState::CoroutineWait { id: *id },
            WaitFor::Signal(s) => WaitState::SignalWait { key: s.clone() },
            WaitFor::All(ids) => WaitState::AllWait { ids: ids.clone() },
            WaitFor::Any(ids) => WaitState::AnyWait { ids: ids.clone() },
        }
    }
}

// ---------------------------------------------------------------------------
// CoroutineError
// ---------------------------------------------------------------------------

/// Errors that can occur during coroutine operations.
#[derive(Debug, Clone)]
pub enum CoroutineError {
    /// Tried to resume a non-resumable coroutine.
    NotResumable(CoroutineId, CoroutineState),
    /// Coroutine not found.
    NotFound(CoroutineId),
    /// Stack overflow.
    StackOverflow(CoroutineId),
    /// Maximum coroutine limit reached.
    LimitExceeded,
    /// Runtime error in coroutine body.
    RuntimeError(CoroutineId, String),
}

impl fmt::Display for CoroutineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineError::NotResumable(id, state) => {
                write!(f, "coroutine {} is not resumable (state: {})", id, state)
            }
            CoroutineError::NotFound(id) => write!(f, "coroutine {} not found", id),
            CoroutineError::StackOverflow(id) => {
                write!(f, "coroutine {} stack overflow", id)
            }
            CoroutineError::LimitExceeded => write!(f, "maximum coroutine limit exceeded"),
            CoroutineError::RuntimeError(id, msg) => {
                write!(f, "runtime error in coroutine {}: {}", id, msg)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CoroutineBody
// ---------------------------------------------------------------------------

/// The executable body of a coroutine.
///
/// This is a function pointer or closure that receives resume values and
/// returns yield values. In a real VM integration, this would be tied to
/// the bytecode instruction pointer; here we use a simplified callback model.
pub type CoroutineBodyFn = Arc<dyn Fn(&mut CoroutineContext) -> CoroutineResult + Send + Sync>;

/// The result of executing one step of a coroutine body.
#[derive(Debug, Clone)]
pub enum CoroutineResult {
    /// The coroutine yielded a value and wants to be suspended.
    Yield(CoroutineValue),
    /// The coroutine completed and returned a final value.
    Return(CoroutineValue),
    /// The coroutine encountered an error.
    Error(String),
}

// ---------------------------------------------------------------------------
// CoroutineContext
// ---------------------------------------------------------------------------

/// Mutable context passed to coroutine body functions.
///
/// Provides access to the resume value and a mechanism to yield.
#[derive(Debug)]
pub struct CoroutineContext {
    /// Value passed in via `resume()`.
    pub resume_value: CoroutineValue,
    /// The coroutine's local variable stack.
    pub locals: Vec<CoroutineValue>,
    /// Current step/phase counter (for multi-step coroutines).
    pub step: u32,
    /// Named flags/state the coroutine can read/write.
    pub user_data: HashMap<String, CoroutineValue>,
}

impl CoroutineContext {
    /// Create a new context with the given resume value.
    pub fn new(resume_value: CoroutineValue) -> Self {
        Self {
            resume_value,
            locals: Vec::new(),
            step: 0,
            user_data: HashMap::new(),
        }
    }

    /// Get a local variable by index.
    pub fn get_local(&self, index: usize) -> &CoroutineValue {
        self.locals.get(index).unwrap_or(&CoroutineValue::Nil)
    }

    /// Set a local variable by index, growing the stack if needed.
    pub fn set_local(&mut self, index: usize, value: CoroutineValue) {
        while self.locals.len() <= index {
            self.locals.push(CoroutineValue::Nil);
        }
        self.locals[index] = value;
    }

    /// Store a named user data value.
    pub fn set_data(&mut self, key: impl Into<String>, value: CoroutineValue) {
        self.user_data.insert(key.into(), value);
    }

    /// Get a named user data value.
    pub fn get_data(&self, key: &str) -> &CoroutineValue {
        self.user_data.get(key).unwrap_or(&CoroutineValue::Nil)
    }

    /// Advance to the next step and return the new step number.
    pub fn next_step(&mut self) -> u32 {
        self.step += 1;
        self.step
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

/// A coroutine: a pausable, resumable unit of script execution.
#[derive(Clone)]
pub struct Coroutine {
    /// Unique identifier.
    pub id: CoroutineId,
    /// Human-readable name (for debugging).
    pub name: String,
    /// Current state.
    pub state: CoroutineState,
    /// The executable body.
    body: CoroutineBodyFn,
    /// Execution context (persisted across yields).
    pub context: CoroutineContext,
    /// Current wait state (if suspended waiting for something).
    wait_state: Option<WaitState>,
    /// The last value yielded by this coroutine.
    pub last_yielded: CoroutineValue,
    /// The final return value (set when completed).
    pub return_value: CoroutineValue,
    /// Priority (lower = updated first).
    pub priority: u32,
    /// Entity that owns this coroutine (if any).
    pub owner_entity: Option<u64>,
    /// Tags for grouping/filtering.
    pub tags: Vec<String>,
    /// Number of times this coroutine has been resumed.
    pub resume_count: u32,
    /// Total elapsed time this coroutine has been alive.
    pub elapsed_time: f64,
}

impl Coroutine {
    /// Create a new coroutine.
    pub fn new(
        id: CoroutineId,
        name: impl Into<String>,
        body: CoroutineBodyFn,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            state: CoroutineState::Created,
            body,
            context: CoroutineContext::new(CoroutineValue::Nil),
            wait_state: None,
            last_yielded: CoroutineValue::Nil,
            return_value: CoroutineValue::Nil,
            priority: DEFAULT_PRIORITY,
            owner_entity: None,
            tags: Vec::new(),
            resume_count: 0,
            elapsed_time: 0.0,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the owner entity.
    pub fn with_owner(mut self, entity: u64) -> Self {
        self.owner_entity = Some(entity);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Resume execution with an optional value.
    ///
    /// Returns the yielded or returned value, or an error.
    pub fn resume(
        &mut self,
        value: CoroutineValue,
    ) -> Result<CoroutineResult, CoroutineError> {
        if !self.state.is_resumable() {
            return Err(CoroutineError::NotResumable(self.id, self.state));
        }

        self.state = CoroutineState::Running;
        self.context.resume_value = value;
        self.resume_count += 1;

        let body = Arc::clone(&self.body);
        let result = (body)(&mut self.context);

        match &result {
            CoroutineResult::Yield(val) => {
                self.last_yielded = val.clone();

                // Check if the yielded value is a wait instruction.
                if let CoroutineValue::Wait(wait) = val {
                    self.wait_state = Some(WaitState::from_wait_for(wait));
                } else {
                    self.wait_state = None;
                }

                self.state = CoroutineState::Suspended;
            }
            CoroutineResult::Return(val) => {
                self.return_value = val.clone();
                self.state = CoroutineState::Completed;
                self.wait_state = None;
            }
            CoroutineResult::Error(_) => {
                self.state = CoroutineState::Completed;
                self.wait_state = None;
            }
        }

        Ok(result)
    }

    /// Cancel the coroutine, marking it as completed.
    pub fn cancel(&mut self) {
        self.state = CoroutineState::Completed;
        self.wait_state = None;
    }

    /// Returns `true` if this coroutine is waiting for something.
    pub fn is_waiting(&self) -> bool {
        self.wait_state.is_some()
    }

    /// Returns `true` if this coroutine has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

impl fmt::Debug for Coroutine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Coroutine")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("state", &self.state)
            .field("priority", &self.priority)
            .field("resume_count", &self.resume_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CoroutineEvent
// ---------------------------------------------------------------------------

/// Events emitted by the coroutine manager.
#[derive(Debug, Clone)]
pub enum CoroutineEvent {
    /// A coroutine was started.
    Started(CoroutineId),
    /// A coroutine yielded a value.
    Yielded(CoroutineId, CoroutineValue),
    /// A coroutine completed with a return value.
    Completed(CoroutineId, CoroutineValue),
    /// A coroutine encountered an error.
    Error(CoroutineId, String),
    /// A coroutine was cancelled.
    Cancelled(CoroutineId),
}

// ---------------------------------------------------------------------------
// CoroutineManager
// ---------------------------------------------------------------------------

/// Manages all active coroutines and updates them each frame.
///
/// The manager handles:
/// - Creating and tracking coroutines
/// - Timer-based waits (`wait_seconds`)
/// - Frame-count waits (`wait_frames`)
/// - Condition-based waits (`wait_until`)
/// - Coroutine-dependency waits (`wait_coroutine`)
/// - Signal-based waits
/// - Cleanup of completed coroutines
///
/// # Usage
///
/// ```ignore
/// let mut manager = CoroutineManager::new();
///
/// // Create a coroutine that waits 2 seconds then returns "done".
/// let body = Arc::new(|ctx: &mut CoroutineContext| -> CoroutineResult {
///     match ctx.step {
///         0 => {
///             ctx.next_step();
///             CoroutineResult::Yield(CoroutineValue::Wait(
///                 WaitFor::seconds(2.0),
///             ))
///         }
///         _ => CoroutineResult::Return(
///             CoroutineValue::String(Arc::from("done")),
///         ),
///     }
/// });
///
/// let id = manager.start("my_coroutine", body);
///
/// // Each frame:
/// manager.update(delta_time);
/// for event in manager.drain_events() {
///     // Handle events...
/// }
/// ```
pub struct CoroutineManager {
    /// All managed coroutines.
    coroutines: HashMap<CoroutineId, Coroutine>,
    /// Next coroutine ID.
    next_id: u64,
    /// Events generated during the last update.
    events: Vec<CoroutineEvent>,
    /// Named conditions and their current boolean values.
    conditions: HashMap<String, bool>,
    /// Pending signals (keys that have been fired this frame).
    signals: HashMap<String, CoroutineValue>,
    /// IDs of coroutines that completed this frame (for dependency tracking).
    completed_this_frame: Vec<CoroutineId>,
    /// Frame counter.
    frame_count: u64,
    /// Total elapsed time.
    total_time: f64,
}

impl CoroutineManager {
    /// Create a new coroutine manager.
    pub fn new() -> Self {
        Self {
            coroutines: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
            conditions: HashMap::new(),
            signals: HashMap::new(),
            completed_this_frame: Vec::new(),
            frame_count: 0,
            total_time: 0.0,
        }
    }

    /// Start a new coroutine and return its ID.
    pub fn start(
        &mut self,
        name: impl Into<String>,
        body: CoroutineBodyFn,
    ) -> Result<CoroutineId, CoroutineError> {
        if self.coroutines.len() >= MAX_COROUTINES {
            return Err(CoroutineError::LimitExceeded);
        }

        let id = CoroutineId(self.next_id);
        self.next_id += 1;

        let mut coroutine = Coroutine::new(id, name, body);

        // Immediately execute the first step (start the coroutine).
        let result = coroutine.resume(CoroutineValue::Nil);

        match result {
            Ok(CoroutineResult::Yield(ref val)) => {
                self.events.push(CoroutineEvent::Started(id));
                self.events.push(CoroutineEvent::Yielded(id, val.clone()));
            }
            Ok(CoroutineResult::Return(ref val)) => {
                self.events.push(CoroutineEvent::Started(id));
                self.events
                    .push(CoroutineEvent::Completed(id, val.clone()));
                self.completed_this_frame.push(id);
            }
            Ok(CoroutineResult::Error(ref msg)) => {
                self.events.push(CoroutineEvent::Error(id, msg.clone()));
            }
            Err(e) => {
                return Err(e);
            }
        }

        self.coroutines.insert(id, coroutine);
        Ok(id)
    }

    /// Start a coroutine without executing the first step.
    pub fn start_suspended(
        &mut self,
        name: impl Into<String>,
        body: CoroutineBodyFn,
    ) -> Result<CoroutineId, CoroutineError> {
        if self.coroutines.len() >= MAX_COROUTINES {
            return Err(CoroutineError::LimitExceeded);
        }

        let id = CoroutineId(self.next_id);
        self.next_id += 1;

        let coroutine = Coroutine::new(id, name, body);
        self.coroutines.insert(id, coroutine);
        Ok(id)
    }

    /// Manually resume a specific coroutine with a value.
    pub fn resume(
        &mut self,
        id: CoroutineId,
        value: CoroutineValue,
    ) -> Result<CoroutineResult, CoroutineError> {
        let coroutine = self
            .coroutines
            .get_mut(&id)
            .ok_or(CoroutineError::NotFound(id))?;

        let result = coroutine.resume(value)?;

        match &result {
            CoroutineResult::Yield(val) => {
                self.events.push(CoroutineEvent::Yielded(id, val.clone()));
            }
            CoroutineResult::Return(val) => {
                self.events
                    .push(CoroutineEvent::Completed(id, val.clone()));
                self.completed_this_frame.push(id);
            }
            CoroutineResult::Error(msg) => {
                self.events.push(CoroutineEvent::Error(id, msg.clone()));
            }
        }

        Ok(result)
    }

    /// Cancel a coroutine.
    pub fn cancel(&mut self, id: CoroutineId) {
        if let Some(coroutine) = self.coroutines.get_mut(&id) {
            coroutine.cancel();
            self.events.push(CoroutineEvent::Cancelled(id));
        }
    }

    /// Cancel all coroutines with a given tag.
    pub fn cancel_by_tag(&mut self, tag: &str) {
        let ids: Vec<CoroutineId> = self
            .coroutines
            .values()
            .filter(|c| c.has_tag(tag) && !c.state.is_done())
            .map(|c| c.id)
            .collect();

        for id in ids {
            self.cancel(id);
        }
    }

    /// Cancel all coroutines owned by a specific entity.
    pub fn cancel_by_owner(&mut self, entity: u64) {
        let ids: Vec<CoroutineId> = self
            .coroutines
            .values()
            .filter(|c| c.owner_entity == Some(entity) && !c.state.is_done())
            .map(|c| c.id)
            .collect();

        for id in ids {
            self.cancel(id);
        }
    }

    /// Set a named condition value. Coroutines waiting on this condition
    /// via `wait_until` will be checked on the next update.
    pub fn set_condition(&mut self, name: impl Into<String>, value: bool) {
        self.conditions.insert(name.into(), value);
    }

    /// Get the current value of a named condition.
    pub fn get_condition(&self, name: &str) -> bool {
        self.conditions.get(name).copied().unwrap_or(false)
    }

    /// Fire a signal, waking up any coroutines waiting on it.
    pub fn fire_signal(&mut self, key: impl Into<String>, value: CoroutineValue) {
        self.signals.insert(key.into(), value);
    }

    /// Update all active coroutines.
    ///
    /// Processes wait conditions: decrements timers, checks frame counts,
    /// evaluates conditions, and resumes coroutines whose waits have elapsed.
    pub fn update(&mut self, dt: f64) {
        self.frame_count += 1;
        self.total_time += dt;
        self.completed_this_frame.clear();

        // Collect IDs of coroutines that are ready to resume.
        let mut ready_to_resume: Vec<(CoroutineId, CoroutineValue)> = Vec::new();

        // Process wait states.
        for (id, coroutine) in &mut self.coroutines {
            if coroutine.state != CoroutineState::Suspended {
                continue;
            }

            coroutine.elapsed_time += dt;

            if let Some(ref mut wait) = coroutine.wait_state {
                let should_resume = match wait {
                    WaitState::Timer { remaining } => {
                        *remaining -= dt;
                        *remaining <= 0.0
                    }
                    WaitState::FrameCount { remaining } => {
                        if *remaining > 0 {
                            *remaining -= 1;
                        }
                        *remaining == 0
                    }
                    WaitState::Condition { name } => {
                        self.conditions.get(name).copied().unwrap_or(false)
                    }
                    WaitState::CoroutineWait { id: wait_id } => {
                        self.completed_this_frame.contains(wait_id)
                            || self
                                .coroutines
                                .get(wait_id)
                                .map(|c| c.state.is_done())
                                .unwrap_or(true)
                    }
                    WaitState::SignalWait { key } => self.signals.contains_key(key),
                    WaitState::AllWait { ids } => ids.iter().all(|wait_id| {
                        self.coroutines
                            .get(wait_id)
                            .map(|c| c.state.is_done())
                            .unwrap_or(true)
                    }),
                    WaitState::AnyWait { ids } => ids.iter().any(|wait_id| {
                        self.coroutines
                            .get(wait_id)
                            .map(|c| c.state.is_done())
                            .unwrap_or(true)
                    }),
                };

                if should_resume {
                    // Determine the resume value.
                    let resume_val = match wait {
                        WaitState::SignalWait { key } => self
                            .signals
                            .get(key)
                            .cloned()
                            .unwrap_or(CoroutineValue::Nil),
                        _ => CoroutineValue::Nil,
                    };
                    ready_to_resume.push((*id, resume_val));
                }
            } else {
                // Suspended but no wait state: resume immediately.
                ready_to_resume.push((*id, CoroutineValue::Nil));
            }
        }

        // Sort by priority (lower first).
        ready_to_resume.sort_by_key(|(id, _)| {
            self.coroutines
                .get(id)
                .map(|c| c.priority)
                .unwrap_or(u32::MAX)
        });

        // Resume ready coroutines.
        for (id, value) in ready_to_resume {
            if let Some(coroutine) = self.coroutines.get_mut(&id) {
                coroutine.wait_state = None;

                let result = coroutine.resume(value);

                match result {
                    Ok(CoroutineResult::Yield(ref val)) => {
                        self.events.push(CoroutineEvent::Yielded(id, val.clone()));
                    }
                    Ok(CoroutineResult::Return(ref val)) => {
                        self.events
                            .push(CoroutineEvent::Completed(id, val.clone()));
                        self.completed_this_frame.push(id);
                    }
                    Ok(CoroutineResult::Error(ref msg)) => {
                        self.events.push(CoroutineEvent::Error(id, msg.clone()));
                    }
                    Err(_) => {
                        // Should not happen since we checked state above.
                    }
                }
            }
        }

        // Clear signals (they are consumed each frame).
        self.signals.clear();

        // Remove completed coroutines that are no longer needed.
        // Keep completed coroutines around for one frame so dependency
        // checks can detect their completion.
        self.coroutines.retain(|_, c| !c.state.is_done() || c.elapsed_time < self.total_time - dt * 2.0 || true);
        // Actually, let's keep completed coroutines until explicitly cleaned.
    }

    /// Remove all completed coroutines.
    pub fn cleanup_completed(&mut self) {
        self.coroutines.retain(|_, c| !c.state.is_done());
    }

    /// Drain all events generated during the last update.
    pub fn drain_events(&mut self) -> Vec<CoroutineEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get a reference to a coroutine by ID.
    pub fn get(&self, id: CoroutineId) -> Option<&Coroutine> {
        self.coroutines.get(&id)
    }

    /// Get the state of a coroutine.
    pub fn get_state(&self, id: CoroutineId) -> Option<CoroutineState> {
        self.coroutines.get(&id).map(|c| c.state)
    }

    /// Returns `true` if a coroutine exists and is still running or
    /// suspended.
    pub fn is_alive(&self, id: CoroutineId) -> bool {
        self.coroutines
            .get(&id)
            .map(|c| !c.state.is_done())
            .unwrap_or(false)
    }

    /// Returns the number of active (non-completed) coroutines.
    pub fn active_count(&self) -> usize {
        self.coroutines
            .values()
            .filter(|c| !c.state.is_done())
            .count()
    }

    /// Returns the total number of coroutines (including completed).
    pub fn total_count(&self) -> usize {
        self.coroutines.len()
    }

    /// Get the current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Clear all coroutines, conditions, and signals.
    pub fn clear(&mut self) {
        self.coroutines.clear();
        self.events.clear();
        self.conditions.clear();
        self.signals.clear();
        self.completed_this_frame.clear();
    }

    /// Get all coroutine IDs with a given tag.
    pub fn ids_with_tag(&self, tag: &str) -> Vec<CoroutineId> {
        self.coroutines
            .values()
            .filter(|c| c.has_tag(tag))
            .map(|c| c.id)
            .collect()
    }

    /// Get all coroutine IDs owned by an entity.
    pub fn ids_for_entity(&self, entity: u64) -> Vec<CoroutineId> {
        self.coroutines
            .values()
            .filter(|c| c.owner_entity == Some(entity))
            .map(|c| c.id)
            .collect()
    }
}

impl Default for CoroutineManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for common patterns
// ---------------------------------------------------------------------------

/// Create a coroutine body that executes a sequence of steps with waits
/// between them.
///
/// Each step is a tuple of (action closure, wait-for). The action is
/// executed, then the coroutine yields with the wait instruction.
pub fn sequence_coroutine(
    steps: Vec<(Arc<dyn Fn() + Send + Sync>, WaitFor)>,
) -> CoroutineBodyFn {
    Arc::new(move |ctx: &mut CoroutineContext| {
        let step = ctx.step as usize;

        if step >= steps.len() {
            return CoroutineResult::Return(CoroutineValue::Nil);
        }

        let (action, wait) = &steps[step];
        (action)();
        ctx.next_step();

        CoroutineResult::Yield(CoroutineValue::Wait(wait.clone()))
    })
}

/// Create a simple delay coroutine that waits for a number of seconds
/// then calls a callback.
pub fn delayed_action(
    delay: f64,
    action: Arc<dyn Fn() + Send + Sync>,
) -> CoroutineBodyFn {
    Arc::new(move |ctx: &mut CoroutineContext| {
        match ctx.step {
            0 => {
                ctx.next_step();
                CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::seconds(delay)))
            }
            _ => {
                (action)();
                CoroutineResult::Return(CoroutineValue::Nil)
            }
        }
    })
}

/// Create a tween coroutine that interpolates a float from `start` to
/// `end` over `duration` seconds, yielding each frame.
pub fn tween_coroutine(
    start: f64,
    end: f64,
    duration: f64,
    on_update: Arc<dyn Fn(f64) + Send + Sync>,
) -> CoroutineBodyFn {
    Arc::new(move |ctx: &mut CoroutineContext| {
        match ctx.step {
            0 => {
                ctx.set_data("elapsed", CoroutineValue::Float(0.0));
                ctx.set_data("duration", CoroutineValue::Float(duration));
                ctx.set_data("start", CoroutineValue::Float(start));
                ctx.set_data("end", CoroutineValue::Float(end));
                ctx.next_step();

                (on_update)(start);
                CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::frames(1)))
            }
            _ => {
                let elapsed = ctx.get_data("elapsed").as_float().unwrap_or(0.0);
                let dt = 1.0 / 60.0; // Approximate frame time.
                let new_elapsed = elapsed + dt;
                let t = (new_elapsed / duration).min(1.0);

                let value = start + (end - start) * t;
                (on_update)(value);

                if t >= 1.0 {
                    CoroutineResult::Return(CoroutineValue::Float(end))
                } else {
                    ctx.set_data("elapsed", CoroutineValue::Float(new_elapsed));
                    CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::frames(1)))
                }
            }
        }
    })
}

/// Create a repeating coroutine that calls an action every `interval`
/// seconds, up to `count` times (0 = infinite).
pub fn repeat_coroutine(
    interval: f64,
    count: u32,
    action: Arc<dyn Fn(u32) + Send + Sync>,
) -> CoroutineBodyFn {
    Arc::new(move |ctx: &mut CoroutineContext| {
        let iteration = ctx.step;

        if count > 0 && iteration >= count {
            return CoroutineResult::Return(CoroutineValue::Int(iteration as i64));
        }

        (action)(iteration);
        ctx.next_step();

        CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::seconds(interval)))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coroutine_state_resumable() {
        assert!(CoroutineState::Created.is_resumable());
        assert!(CoroutineState::Suspended.is_resumable());
        assert!(!CoroutineState::Running.is_resumable());
        assert!(!CoroutineState::Completed.is_resumable());
    }

    #[test]
    fn test_coroutine_value_display() {
        assert_eq!(format!("{}", CoroutineValue::Nil), "nil");
        assert_eq!(format!("{}", CoroutineValue::Int(42)), "42");
        assert_eq!(format!("{}", CoroutineValue::Float(3.14)), "3.14");
        assert_eq!(format!("{}", CoroutineValue::Bool(true)), "true");
    }

    #[test]
    fn test_coroutine_value_truthiness() {
        assert!(!CoroutineValue::Nil.is_truthy());
        assert!(!CoroutineValue::Bool(false).is_truthy());
        assert!(CoroutineValue::Bool(true).is_truthy());
        assert!(CoroutineValue::Int(0).is_truthy());
        assert!(CoroutineValue::Float(0.0).is_truthy());
    }

    #[test]
    fn test_wait_for_display() {
        assert_eq!(format!("{}", WaitFor::seconds(2.0)), "wait_seconds(2)");
        assert_eq!(format!("{}", WaitFor::frames(10)), "wait_frames(10)");
        assert_eq!(
            format!("{}", WaitFor::until("door_open")),
            "wait_until(door_open)"
        );
    }

    #[test]
    fn test_coroutine_yield_resume() {
        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Int(1))
                }
                1 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Int(2))
                }
                _ => CoroutineResult::Return(CoroutineValue::String(Arc::from("done"))),
            }
        });

        let mut co = Coroutine::new(CoroutineId(1), "test", body);

        // First resume (from Created).
        let r1 = co.resume(CoroutineValue::Nil).unwrap();
        assert!(matches!(r1, CoroutineResult::Yield(CoroutineValue::Int(1))));
        assert_eq!(co.state, CoroutineState::Suspended);

        // Second resume.
        let r2 = co.resume(CoroutineValue::Nil).unwrap();
        assert!(matches!(r2, CoroutineResult::Yield(CoroutineValue::Int(2))));

        // Third resume -> completion.
        let r3 = co.resume(CoroutineValue::Nil).unwrap();
        assert!(matches!(r3, CoroutineResult::Return(_)));
        assert_eq!(co.state, CoroutineState::Completed);

        // Cannot resume a completed coroutine.
        let r4 = co.resume(CoroutineValue::Nil);
        assert!(r4.is_err());
    }

    #[test]
    fn test_coroutine_resume_with_value() {
        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Nil)
                }
                _ => {
                    // Return whatever was passed as the resume value.
                    CoroutineResult::Return(ctx.resume_value.clone())
                }
            }
        });

        let mut co = Coroutine::new(CoroutineId(1), "test", body);
        co.resume(CoroutineValue::Nil).unwrap();

        let result = co.resume(CoroutineValue::Int(42)).unwrap();
        match result {
            CoroutineResult::Return(CoroutineValue::Int(42)) => {}
            _ => panic!("Expected Return(Int(42))"),
        }
    }

    #[test]
    fn test_coroutine_context() {
        let mut ctx = CoroutineContext::new(CoroutineValue::Nil);

        ctx.set_local(2, CoroutineValue::Int(100));
        assert!(matches!(ctx.get_local(0), CoroutineValue::Nil));
        assert!(matches!(ctx.get_local(2), CoroutineValue::Int(100)));

        ctx.set_data("key", CoroutineValue::Bool(true));
        assert!(matches!(ctx.get_data("key"), CoroutineValue::Bool(true)));
        assert!(matches!(ctx.get_data("missing"), CoroutineValue::Nil));
    }

    #[test]
    fn test_manager_start_and_update() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::seconds(1.0)))
                }
                _ => CoroutineResult::Return(CoroutineValue::Int(42)),
            }
        });

        let id = manager.start("timer_test", body).unwrap();
        assert!(manager.is_alive(id));

        // Update with less than 1 second: should still be waiting.
        manager.update(0.5);
        assert!(manager.is_alive(id));

        // Update with enough time to complete the wait.
        manager.update(0.6);
        // After the wait elapses, the coroutine is resumed and returns.
        assert!(!manager.is_alive(id));
    }

    #[test]
    fn test_manager_frame_wait() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::frames(3)))
                }
                _ => CoroutineResult::Return(CoroutineValue::Nil),
            }
        });

        let id = manager.start("frame_test", body).unwrap();

        manager.update(0.016); // Frame 1: remaining=2
        assert!(manager.is_alive(id));

        manager.update(0.016); // Frame 2: remaining=1
        assert!(manager.is_alive(id));

        manager.update(0.016); // Frame 3: remaining=0 -> resume.
        assert!(!manager.is_alive(id));
    }

    #[test]
    fn test_manager_condition_wait() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::until("door_open")))
                }
                _ => CoroutineResult::Return(CoroutineValue::Bool(true)),
            }
        });

        let id = manager.start("condition_test", body).unwrap();

        // Condition not set: still waiting.
        manager.update(0.016);
        assert!(manager.is_alive(id));

        // Set the condition.
        manager.set_condition("door_open", true);
        manager.update(0.016);
        assert!(!manager.is_alive(id));
    }

    #[test]
    fn test_manager_signal_wait() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            match ctx.step {
                0 => {
                    ctx.next_step();
                    CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::signal("event_a")))
                }
                _ => CoroutineResult::Return(ctx.resume_value.clone()),
            }
        });

        let id = manager.start("signal_test", body).unwrap();

        manager.update(0.016);
        assert!(manager.is_alive(id));

        manager.fire_signal("event_a", CoroutineValue::Int(99));
        manager.update(0.016);
        assert!(!manager.is_alive(id));
    }

    #[test]
    fn test_manager_cancel() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            ctx.next_step();
            CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::seconds(100.0)))
        });

        let id = manager.start("cancel_test", body).unwrap();
        assert!(manager.is_alive(id));

        manager.cancel(id);
        assert!(!manager.is_alive(id));
    }

    #[test]
    fn test_manager_cancel_by_tag() {
        let mut manager = CoroutineManager::new();

        let body: CoroutineBodyFn = Arc::new(|ctx: &mut CoroutineContext| {
            ctx.next_step();
            CoroutineResult::Yield(CoroutineValue::Wait(WaitFor::seconds(100.0)))
        });

        let id1 = manager.start_suspended("a", Arc::clone(&body)).unwrap();
        manager.coroutines.get_mut(&id1).unwrap().tags.push("group_a".into());
        let _ = manager.resume(id1, CoroutineValue::Nil);

        let id2 = manager.start_suspended("b", body).unwrap();
        manager.coroutines.get_mut(&id2).unwrap().tags.push("group_b".into());
        let _ = manager.resume(id2, CoroutineValue::Nil);

        manager.cancel_by_tag("group_a");
        assert!(!manager.is_alive(id1));
        assert!(manager.is_alive(id2));
    }

    #[test]
    fn test_coroutine_id_display() {
        assert_eq!(format!("{}", CoroutineId(42)), "co#42");
    }

    #[test]
    fn test_delayed_action_pattern() {
        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = Arc::clone(&called);

        let body = delayed_action(1.0, Arc::new(move || {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        }));

        let mut manager = CoroutineManager::new();
        let id = manager.start("delay_test", body).unwrap();

        manager.update(0.5);
        assert!(!called.load(std::sync::atomic::Ordering::SeqCst));

        manager.update(0.6);
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!manager.is_alive(id));
    }
}
