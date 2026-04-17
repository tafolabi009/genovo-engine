//! Pushdown automaton / stack-based state machine for the Genovo engine.
//!
//! Provides a flexible, stack-based state machine that supports push/pop
//! transitions, enter/exit callbacks, state history, parallel state machines,
//! and state serialization. This is particularly useful for game state
//! management (menus, gameplay, pause), AI behavior, and UI flow control.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────┐
//! │        State Machine         │
//! │  ┌────────────────────────┐  │
//! │  │  State Stack           │  │
//! │  │  ┌──────┐              │  │
//! │  │  │ Top  │ ← current   │  │
//! │  │  ├──────┤              │  │
//! │  │  │  S2  │              │  │
//! │  │  ├──────┤              │  │
//! │  │  │  S1  │              │  │
//! │  │  └──────┘              │  │
//! │  └────────────────────────┘  │
//! │  History: [S1→S2, S2→Top]    │
//! └──────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// StateId
// ---------------------------------------------------------------------------

/// Identifier for a state within the machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateId(pub String);

impl StateId {
    /// Create a new state ID from a string.
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }

    /// Returns the name of this state.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for StateId {
    fn from(s: &str) -> Self {
        StateId::new(s)
    }
}

impl From<String> for StateId {
    fn from(s: String) -> Self {
        StateId(s)
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A requested transition between states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transition {
    /// Push a new state on top of the stack.
    Push(StateId),
    /// Pop the current state, returning to the one below.
    Pop,
    /// Replace the current (top) state with a new one.
    Switch(StateId),
    /// Clear the entire stack and push a new root state.
    Reset(StateId),
    /// No transition requested.
    None,
}

impl fmt::Display for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Transition::Push(id) => write!(f, "Push({})", id),
            Transition::Pop => write!(f, "Pop"),
            Transition::Switch(id) => write!(f, "Switch({})", id),
            Transition::Reset(id) => write!(f, "Reset({})", id),
            Transition::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// State Trait
// ---------------------------------------------------------------------------

/// Trait implemented by state objects managed by the state machine.
pub trait State: Send + Sync {
    /// Called when this state becomes the active (top) state.
    fn on_enter(&mut self, _context: &mut StateContext) {}

    /// Called when this state is no longer the active state (popped or covered).
    fn on_exit(&mut self, _context: &mut StateContext) {}

    /// Called when this state becomes the top state again after a higher state
    /// was popped (resume from background).
    fn on_resume(&mut self, _context: &mut StateContext) {}

    /// Called when a new state is pushed on top of this one (going to background).
    fn on_pause(&mut self, _context: &mut StateContext) {}

    /// Called every tick while this state is the current (top) state.
    /// Returns a [`Transition`] to indicate what should happen next.
    fn update(&mut self, context: &mut StateContext, dt: f64) -> Transition;

    /// Returns the state's unique identifier.
    fn id(&self) -> StateId;

    /// Optional: serialize this state's data for save/load.
    fn serialize(&self) -> Option<StateData> {
        None
    }

    /// Optional: restore state from serialized data.
    fn deserialize(&mut self, _data: &StateData) {}
}

// ---------------------------------------------------------------------------
// StateContext
// ---------------------------------------------------------------------------

/// Shared context passed to state callbacks, allowing states to communicate
/// with the rest of the engine or with each other.
pub struct StateContext {
    /// Arbitrary key-value data shared between states.
    pub data: HashMap<String, ContextValue>,
    /// Elapsed time since the state machine was created.
    pub elapsed: f64,
    /// Current frame number.
    pub frame: u64,
}

impl StateContext {
    /// Create a new, empty context.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            elapsed: 0.0,
            frame: 0,
        }
    }

    /// Set a value in the context.
    pub fn set(&mut self, key: &str, value: ContextValue) {
        self.data.insert(key.to_string(), value);
    }

    /// Get a value from the context.
    pub fn get(&self, key: &str) -> Option<&ContextValue> {
        self.data.get(key)
    }

    /// Get a boolean value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(ContextValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Get an integer value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(ContextValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// Get a float value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(ContextValue::Float(f)) => Some(*f),
            _ => None,
        }
    }

    /// Get a string value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(ContextValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Remove a value.
    pub fn remove(&mut self, key: &str) -> Option<ContextValue> {
        self.data.remove(key)
    }

    /// Clear all context data.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for StateContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A value stored in the state context.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl fmt::Display for ContextValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextValue::Bool(b) => write!(f, "{}", b),
            ContextValue::Int(i) => write!(f, "{}", i),
            ContextValue::Float(v) => write!(f, "{}", v),
            ContextValue::String(s) => write!(f, "{}", s),
            ContextValue::Bytes(b) => write!(f, "<{} bytes>", b.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// StateData (serialization)
// ---------------------------------------------------------------------------

/// Serialized state data for save/load.
#[derive(Debug, Clone)]
pub struct StateData {
    /// The state ID.
    pub state_id: StateId,
    /// Key-value properties.
    pub properties: HashMap<String, ContextValue>,
}

impl StateData {
    pub fn new(state_id: StateId) -> Self {
        Self {
            state_id,
            properties: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: ContextValue) {
        self.properties.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&ContextValue> {
        self.properties.get(key)
    }
}

// ---------------------------------------------------------------------------
// HistoryEntry
// ---------------------------------------------------------------------------

/// An entry in the state machine's transition history.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// The transition that occurred.
    pub transition: String,
    /// The state that was active before the transition.
    pub from_state: Option<StateId>,
    /// The state that became active after the transition.
    pub to_state: Option<StateId>,
    /// When the transition happened.
    pub timestamp: Instant,
    /// The frame number at which the transition happened.
    pub frame: u64,
}

impl fmt::Display for HistoryEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[frame {}] {} -> {} ({})",
            self.frame,
            self.from_state.as_ref().map_or("None", |s| s.name()),
            self.to_state.as_ref().map_or("None", |s| s.name()),
            self.transition,
        )
    }
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// A pushdown automaton that manages a stack of states.
pub struct StateMachine {
    /// The state stack. The last element is the current (active) state.
    stack: Vec<Box<dyn State>>,
    /// Registry of state constructors for creating states by ID.
    factory: HashMap<String, Box<dyn Fn() -> Box<dyn State> + Send + Sync>>,
    /// Shared context.
    context: StateContext,
    /// Transition history.
    history: Vec<HistoryEntry>,
    /// Maximum history entries to keep.
    max_history: usize,
    /// Whether the state machine is running.
    running: bool,
    /// Total elapsed time.
    total_time: f64,
    /// Total update count.
    total_frames: u64,
    /// Name of this state machine (for debugging).
    name: String,
}

impl StateMachine {
    /// Create a new, empty state machine.
    pub fn new(name: &str) -> Self {
        Self {
            stack: Vec::new(),
            factory: HashMap::new(),
            context: StateContext::new(),
            history: Vec::new(),
            max_history: 256,
            running: true,
            total_time: 0.0,
            total_frames: 0,
            name: name.to_string(),
        }
    }

    /// Register a state factory function.
    pub fn register_state<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn State> + Send + Sync + 'static,
    {
        self.factory.insert(name.to_string(), Box::new(factory));
    }

    /// Create a state by ID using the factory registry.
    pub fn create_state(&self, id: &StateId) -> Option<Box<dyn State>> {
        self.factory.get(id.name()).map(|f| f())
    }

    /// Push a state onto the stack.
    pub fn push(&mut self, mut state: Box<dyn State>) {
        let from = self.current_state_id();
        let to = state.id();

        // Pause the current top state.
        if let Some(current) = self.stack.last_mut() {
            current.on_pause(&mut self.context);
        }

        // Enter the new state.
        state.on_enter(&mut self.context);

        self.record_history("Push", from.clone(), Some(to.clone()));
        self.stack.push(state);
    }

    /// Pop the current state from the stack.
    pub fn pop(&mut self) -> Option<Box<dyn State>> {
        if let Some(mut state) = self.stack.pop() {
            let from = Some(state.id());
            state.on_exit(&mut self.context);

            // Resume the state below.
            let to = self.current_state_id();
            if let Some(current) = self.stack.last_mut() {
                current.on_resume(&mut self.context);
            }

            self.record_history("Pop", from, to);
            Some(state)
        } else {
            None
        }
    }

    /// Replace the current (top) state with a new one.
    pub fn switch(&mut self, mut state: Box<dyn State>) {
        let from = self.current_state_id();
        let to = state.id();

        // Exit the current top state.
        if let Some(mut current) = self.stack.pop() {
            current.on_exit(&mut self.context);
        }

        // Enter the new state.
        state.on_enter(&mut self.context);

        self.record_history("Switch", from, Some(to));
        self.stack.push(state);
    }

    /// Clear the entire stack and push a new root state.
    pub fn reset(&mut self, mut state: Box<dyn State>) {
        let from = self.current_state_id();
        let to = state.id();

        // Exit all states from top to bottom.
        while let Some(mut s) = self.stack.pop() {
            s.on_exit(&mut self.context);
        }

        // Enter the new root state.
        state.on_enter(&mut self.context);

        self.record_history("Reset", from, Some(to));
        self.stack.push(state);
    }

    /// Update the current state and process any resulting transitions.
    pub fn update(&mut self, dt: f64) {
        if !self.running || self.stack.is_empty() {
            return;
        }

        self.total_time += dt;
        self.total_frames += 1;
        self.context.elapsed = self.total_time;
        self.context.frame = self.total_frames;

        // Update the top state.
        let transition = {
            let top = self.stack.last_mut().unwrap();
            top.update(&mut self.context, dt)
        };

        // Process the transition.
        self.process_transition(transition);
    }

    /// Process a transition request.
    fn process_transition(&mut self, transition: Transition) {
        match transition {
            Transition::Push(id) => {
                if let Some(state) = self.create_state(&id) {
                    self.push(state);
                }
            }
            Transition::Pop => {
                self.pop();
            }
            Transition::Switch(id) => {
                if let Some(state) = self.create_state(&id) {
                    self.switch(state);
                }
            }
            Transition::Reset(id) => {
                if let Some(state) = self.create_state(&id) {
                    self.reset(state);
                }
            }
            Transition::None => {}
        }
    }

    /// Returns the ID of the current (top) state.
    pub fn current_state_id(&self) -> Option<StateId> {
        self.stack.last().map(|s| s.id())
    }

    /// Peek at the current state without removing it.
    pub fn current_state(&self) -> Option<&dyn State> {
        self.stack.last().map(|s| s.as_ref())
    }

    /// Peek at the current state mutably.
    pub fn current_state_mut(&mut self) -> Option<&mut dyn State> {
        self.stack.last_mut().map(|s| s.as_mut())
    }

    /// Returns the depth of the state stack.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns `true` if the state machine is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Stop the state machine.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Start/resume the state machine.
    pub fn start(&mut self) {
        self.running = true;
    }

    /// Returns a reference to the shared context.
    pub fn context(&self) -> &StateContext {
        &self.context
    }

    /// Returns a mutable reference to the shared context.
    pub fn context_mut(&mut self) -> &mut StateContext {
        &mut self.context
    }

    /// Returns the transition history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Clear the transition history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns the state IDs on the stack (bottom to top).
    pub fn stack_ids(&self) -> Vec<StateId> {
        self.stack.iter().map(|s| s.id()).collect()
    }

    /// Check if a specific state is anywhere on the stack.
    pub fn contains_state(&self, id: &StateId) -> bool {
        self.stack.iter().any(|s| s.id() == *id)
    }

    /// Returns the name of this state machine.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total elapsed time.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Total frame count.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Serialize the state stack.
    pub fn serialize_stack(&self) -> Vec<StateData> {
        self.stack
            .iter()
            .filter_map(|s| s.serialize())
            .collect()
    }

    /// Restore state from serialized data (requires factory registration).
    pub fn deserialize_stack(&mut self, data: &[StateData]) {
        // Clear existing stack.
        while let Some(mut s) = self.stack.pop() {
            s.on_exit(&mut self.context);
        }

        // Rebuild from serialized data.
        for state_data in data {
            if let Some(mut state) = self.create_state(&state_data.state_id) {
                state.deserialize(state_data);
                state.on_enter(&mut self.context);
                self.stack.push(state);
            }
        }
    }

    fn record_history(
        &mut self,
        kind: &str,
        from: Option<StateId>,
        to: Option<StateId>,
    ) {
        self.history.push(HistoryEntry {
            transition: kind.to_string(),
            from_state: from,
            to_state: to,
            timestamp: Instant::now(),
            frame: self.total_frames,
        });
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
}

impl fmt::Debug for StateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateMachine")
            .field("name", &self.name)
            .field("depth", &self.stack.len())
            .field("current", &self.current_state_id())
            .field("running", &self.running)
            .field("total_frames", &self.total_frames)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ParallelStateMachine
// ---------------------------------------------------------------------------

/// Runs multiple state machines in parallel, each with its own stack.
/// Useful for managing orthogonal concerns (e.g., game state + audio state
/// + UI state simultaneously).
pub struct ParallelStateMachine {
    /// Named sub-machines.
    machines: HashMap<String, StateMachine>,
    /// Order of execution.
    execution_order: Vec<String>,
}

impl ParallelStateMachine {
    /// Create a new parallel state machine.
    pub fn new() -> Self {
        Self {
            machines: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add a sub-machine.
    pub fn add_machine(&mut self, machine: StateMachine) {
        let name = machine.name().to_string();
        self.execution_order.push(name.clone());
        self.machines.insert(name, machine);
    }

    /// Remove a sub-machine by name.
    pub fn remove_machine(&mut self, name: &str) -> Option<StateMachine> {
        self.execution_order.retain(|n| n != name);
        self.machines.remove(name)
    }

    /// Get a reference to a sub-machine.
    pub fn get(&self, name: &str) -> Option<&StateMachine> {
        self.machines.get(name)
    }

    /// Get a mutable reference to a sub-machine.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut StateMachine> {
        self.machines.get_mut(name)
    }

    /// Update all sub-machines.
    pub fn update(&mut self, dt: f64) {
        for name in &self.execution_order {
            if let Some(machine) = self.machines.get_mut(name) {
                machine.update(dt);
            }
        }
    }

    /// Returns the names of all sub-machines.
    pub fn machine_names(&self) -> &[String] {
        &self.execution_order
    }

    /// Number of sub-machines.
    pub fn machine_count(&self) -> usize {
        self.machines.len()
    }
}

impl Default for ParallelStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ParallelStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelStateMachine")
            .field("machines", &self.execution_order)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Simple State implementations for common patterns
// ---------------------------------------------------------------------------

/// A simple state that runs a closure on update.
pub struct ClosureState {
    id: StateId,
    on_update: Box<dyn FnMut(&mut StateContext, f64) -> Transition + Send + Sync>,
    on_enter_fn: Option<Box<dyn FnMut(&mut StateContext) + Send + Sync>>,
    on_exit_fn: Option<Box<dyn FnMut(&mut StateContext) + Send + Sync>>,
}

impl ClosureState {
    /// Create a closure-based state.
    pub fn new<F>(id: &str, on_update: F) -> Self
    where
        F: FnMut(&mut StateContext, f64) -> Transition + Send + Sync + 'static,
    {
        Self {
            id: StateId::new(id),
            on_update: Box::new(on_update),
            on_enter_fn: None,
            on_exit_fn: None,
        }
    }

    /// Set an on-enter callback.
    pub fn with_enter<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut StateContext) + Send + Sync + 'static,
    {
        self.on_enter_fn = Some(Box::new(f));
        self
    }

    /// Set an on-exit callback.
    pub fn with_exit<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut StateContext) + Send + Sync + 'static,
    {
        self.on_exit_fn = Some(Box::new(f));
        self
    }
}

impl State for ClosureState {
    fn on_enter(&mut self, context: &mut StateContext) {
        if let Some(ref mut f) = self.on_enter_fn {
            f(context);
        }
    }

    fn on_exit(&mut self, context: &mut StateContext) {
        if let Some(ref mut f) = self.on_exit_fn {
            f(context);
        }
    }

    fn update(&mut self, context: &mut StateContext, dt: f64) -> Transition {
        (self.on_update)(context, dt)
    }

    fn id(&self) -> StateId {
        self.id.clone()
    }
}

/// A timed state that automatically transitions after a duration.
pub struct TimedState {
    id: StateId,
    duration: f64,
    elapsed: f64,
    next_transition: Transition,
}

impl TimedState {
    /// Create a state that transitions after `duration` seconds.
    pub fn new(id: &str, duration: f64, next: Transition) -> Self {
        Self {
            id: StateId::new(id),
            duration,
            elapsed: 0.0,
            next_transition: next,
        }
    }

    /// Returns the remaining time.
    pub fn remaining(&self) -> f64 {
        (self.duration - self.elapsed).max(0.0)
    }

    /// Returns the progress ratio (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        (self.elapsed / self.duration).min(1.0)
    }
}

impl State for TimedState {
    fn update(&mut self, _context: &mut StateContext, dt: f64) -> Transition {
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.next_transition.clone()
        } else {
            Transition::None
        }
    }

    fn id(&self) -> StateId {
        self.id.clone()
    }
}

/// A sequence state that runs through a list of transitions in order.
pub struct SequenceState {
    id: StateId,
    transitions: Vec<Transition>,
    current_index: usize,
    interval: f64,
    elapsed: f64,
}

impl SequenceState {
    /// Create a sequence state with the given transitions.
    pub fn new(id: &str, transitions: Vec<Transition>, interval: f64) -> Self {
        Self {
            id: StateId::new(id),
            transitions,
            current_index: 0,
            interval,
            elapsed: 0.0,
        }
    }
}

impl State for SequenceState {
    fn update(&mut self, _context: &mut StateContext, dt: f64) -> Transition {
        self.elapsed += dt;
        if self.elapsed >= self.interval && self.current_index < self.transitions.len() {
            self.elapsed = 0.0;
            let transition = self.transitions[self.current_index].clone();
            self.current_index += 1;
            transition
        } else {
            Transition::None
        }
    }

    fn id(&self) -> StateId {
        self.id.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct TestState {
        name: String,
        enter_count: Arc<AtomicU32>,
        exit_count: Arc<AtomicU32>,
        update_count: Arc<AtomicU32>,
        next_transition: Transition,
    }

    impl TestState {
        fn new(name: &str, transition: Transition) -> (Self, Arc<AtomicU32>, Arc<AtomicU32>, Arc<AtomicU32>) {
            let enter = Arc::new(AtomicU32::new(0));
            let exit = Arc::new(AtomicU32::new(0));
            let update = Arc::new(AtomicU32::new(0));
            (
                Self {
                    name: name.to_string(),
                    enter_count: enter.clone(),
                    exit_count: exit.clone(),
                    update_count: update.clone(),
                    next_transition: transition,
                },
                enter,
                exit,
                update,
            )
        }
    }

    impl State for TestState {
        fn on_enter(&mut self, _context: &mut StateContext) {
            self.enter_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_exit(&mut self, _context: &mut StateContext) {
            self.exit_count.fetch_add(1, Ordering::SeqCst);
        }

        fn update(&mut self, _context: &mut StateContext, _dt: f64) -> Transition {
            self.update_count.fetch_add(1, Ordering::SeqCst);
            self.next_transition.clone()
        }

        fn id(&self) -> StateId {
            StateId::new(&self.name)
        }
    }

    #[test]
    fn test_push_pop() {
        let mut sm = StateMachine::new("test");

        let (state_a, enter_a, exit_a, _) = TestState::new("A", Transition::None);
        let (state_b, enter_b, exit_b, _) = TestState::new("B", Transition::None);

        sm.push(Box::new(state_a));
        assert_eq!(enter_a.load(Ordering::SeqCst), 1);
        assert_eq!(sm.current_state_id().unwrap().name(), "A");

        sm.push(Box::new(state_b));
        assert_eq!(enter_b.load(Ordering::SeqCst), 1);
        assert_eq!(sm.current_state_id().unwrap().name(), "B");
        assert_eq!(sm.depth(), 2);

        sm.pop();
        assert_eq!(exit_b.load(Ordering::SeqCst), 1);
        assert_eq!(sm.current_state_id().unwrap().name(), "A");
        assert_eq!(sm.depth(), 1);
    }

    #[test]
    fn test_switch() {
        let mut sm = StateMachine::new("test");

        let (state_a, _, exit_a, _) = TestState::new("A", Transition::None);
        let (state_b, enter_b, _, _) = TestState::new("B", Transition::None);

        sm.push(Box::new(state_a));
        sm.switch(Box::new(state_b));

        assert_eq!(exit_a.load(Ordering::SeqCst), 1);
        assert_eq!(enter_b.load(Ordering::SeqCst), 1);
        assert_eq!(sm.depth(), 1);
        assert_eq!(sm.current_state_id().unwrap().name(), "B");
    }

    #[test]
    fn test_reset() {
        let mut sm = StateMachine::new("test");

        let (state_a, _, exit_a, _) = TestState::new("A", Transition::None);
        let (state_b, _, exit_b, _) = TestState::new("B", Transition::None);
        let (state_c, enter_c, _, _) = TestState::new("C", Transition::None);

        sm.push(Box::new(state_a));
        sm.push(Box::new(state_b));
        sm.reset(Box::new(state_c));

        assert_eq!(exit_a.load(Ordering::SeqCst), 1);
        assert_eq!(exit_b.load(Ordering::SeqCst), 1);
        assert_eq!(enter_c.load(Ordering::SeqCst), 1);
        assert_eq!(sm.depth(), 1);
    }

    #[test]
    fn test_update() {
        let mut sm = StateMachine::new("test");
        let (state, _, _, update) = TestState::new("A", Transition::None);
        sm.push(Box::new(state));

        sm.update(1.0 / 60.0);
        sm.update(1.0 / 60.0);
        sm.update(1.0 / 60.0);

        assert_eq!(update.load(Ordering::SeqCst), 3);
        assert_eq!(sm.total_frames(), 3);
    }

    #[test]
    fn test_history() {
        let mut sm = StateMachine::new("test");
        let (state_a, _, _, _) = TestState::new("A", Transition::None);
        let (state_b, _, _, _) = TestState::new("B", Transition::None);

        sm.push(Box::new(state_a));
        sm.push(Box::new(state_b));
        sm.pop();

        assert_eq!(sm.history().len(), 3);
    }

    #[test]
    fn test_parallel_machine() {
        let mut parallel = ParallelStateMachine::new();

        let mut sm1 = StateMachine::new("game");
        let (state, _, _, update1) = TestState::new("playing", Transition::None);
        sm1.push(Box::new(state));

        let mut sm2 = StateMachine::new("audio");
        let (state, _, _, update2) = TestState::new("music", Transition::None);
        sm2.push(Box::new(state));

        parallel.add_machine(sm1);
        parallel.add_machine(sm2);

        parallel.update(1.0 / 60.0);

        assert_eq!(update1.load(Ordering::SeqCst), 1);
        assert_eq!(update2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_closure_state() {
        let mut sm = StateMachine::new("test");
        let counter = Arc::new(AtomicU32::new(0));
        let counter2 = counter.clone();

        let state = ClosureState::new("closure", move |_ctx, _dt| {
            counter2.fetch_add(1, Ordering::SeqCst);
            Transition::None
        });
        sm.push(Box::new(state));

        sm.update(0.016);
        sm.update(0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_timed_state() {
        let state = TimedState::new("wait", 2.0, Transition::Pop);
        let mut sm = StateMachine::new("test");
        sm.push(Box::new(state));

        sm.update(1.0);
        assert!(!sm.is_empty());

        sm.update(1.5);
        assert!(sm.is_empty()); // Should have popped after 2.0 seconds.
    }
}
