//! Finite State Machine for game logic.
//!
//! Provides a generic, hierarchical finite state machine (HFSM) suitable for
//! character AI, game flow, UI states, and any other game logic that can be
//! modelled as a set of states with transitions triggered by events.
//!
//! # Features
//!
//! - **Generic over state and event types**: `StateMachine<S, E>` works with
//!   any `Clone + Eq + Hash + Debug` types for states and events.
//! - **State trait**: `on_enter`, `on_exit`, `on_update`, `handle_event`.
//! - **Transitions**: from-state + event -> to-state with optional guard.
//! - **Builder pattern**: fluent API via [`StateMachineBuilder`].
//! - **Hierarchical states**: nested sub-machines with automatic enter/exit
//!   propagation.
//! - **History**: remember the last active sub-state when re-entering a
//!   parent state.
//! - **GameState example**: `MainMenu`, `Loading`, `Playing`, `Paused`,
//!   `GameOver`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

// ---------------------------------------------------------------------------
// State trait
// ---------------------------------------------------------------------------

/// Trait implemented by state handlers.
///
/// Each state in the machine can optionally implement lifecycle hooks and
/// event handling. The default implementations are no-ops, so you only need
/// to override what you care about.
pub trait State<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Called when this state is entered.
    fn on_enter(&mut self, _state: &S) {}

    /// Called when this state is exited.
    fn on_exit(&mut self, _state: &S) {}

    /// Called every frame while this state is active.
    ///
    /// Returns an optional event to trigger a transition.
    fn on_update(&mut self, _state: &S, _dt: f32) -> Option<E> {
        None
    }

    /// Handles an external event while this state is active.
    ///
    /// Returns `Some(target_state)` to request a transition, or `None` to
    /// stay in the current state.
    fn handle_event(&mut self, _state: &S, _event: &E) -> Option<S> {
        None
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A transition rule: when in `from_state` and `event` occurs, move to
/// `to_state` (if the guard permits).
#[derive(Debug, Clone)]
pub struct Transition<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// The state this transition originates from.
    pub from_state: S,
    /// The event that triggers this transition.
    pub event: E,
    /// The target state.
    pub to_state: S,
    /// Optional guard name (evaluated by the guard registry).
    pub guard: Option<String>,
    /// Priority: higher priority transitions are checked first when
    /// multiple transitions match.
    pub priority: i32,
}

impl<S, E> Transition<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Creates a new transition.
    pub fn new(from_state: S, event: E, to_state: S) -> Self {
        Self {
            from_state,
            event,
            to_state,
            guard: None,
            priority: 0,
        }
    }

    /// Adds a guard condition name to this transition.
    pub fn with_guard(mut self, guard: impl Into<String>) -> Self {
        self.guard = Some(guard.into());
        self
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

// ---------------------------------------------------------------------------
// TransitionKey
// ---------------------------------------------------------------------------

/// Key for looking up transitions: (from_state, event discriminant).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TransitionKey<S: Clone + Eq + Hash> {
    from_state: S,
    event_name: String,
}

// ---------------------------------------------------------------------------
// Guard function
// ---------------------------------------------------------------------------

/// A guard function that determines whether a transition is allowed.
pub type GuardFn<S> = Box<dyn Fn(&S) -> bool + Send + Sync>;

// ---------------------------------------------------------------------------
// TransitionResult
// ---------------------------------------------------------------------------

/// Result of attempting a state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionResult<S: Clone + Eq + Hash + Debug> {
    /// Transition succeeded; includes the previous state and new state.
    Transitioned { from: S, to: S },
    /// No matching transition was found.
    NoTransition,
    /// A transition was found but the guard condition blocked it.
    GuardBlocked { from: S, to: S, guard: String },
}

// ---------------------------------------------------------------------------
// StateMachineHistory
// ---------------------------------------------------------------------------

/// History mode for hierarchical state re-entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryMode {
    /// No history: always enter the default initial sub-state.
    None,
    /// Shallow history: remember the immediate sub-state.
    Shallow,
    /// Deep history: remember the full nested sub-state hierarchy.
    Deep,
}

impl Default for HistoryMode {
    fn default() -> Self {
        HistoryMode::None
    }
}

// ---------------------------------------------------------------------------
// SubMachine
// ---------------------------------------------------------------------------

/// A hierarchical sub-machine embedded inside a parent state.
#[derive(Debug)]
pub struct SubMachine<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// The state machine for this sub-level.
    pub machine: StateMachine<S, E>,
    /// History mode for this sub-machine.
    pub history_mode: HistoryMode,
    /// Remembered state for history (last state before exit).
    history_state: Option<S>,
}

impl<S, E> SubMachine<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Creates a new sub-machine.
    pub fn new(machine: StateMachine<S, E>, history_mode: HistoryMode) -> Self {
        Self {
            machine,
            history_mode,
            history_state: None,
        }
    }

    /// Saves the current state as history.
    pub fn save_history(&mut self) {
        self.history_state = Some(self.machine.current_state().clone());
    }

    /// Returns the history state, if any.
    pub fn history_state(&self) -> Option<&S> {
        self.history_state.as_ref()
    }

    /// Enters the sub-machine, potentially restoring history.
    pub fn enter(&mut self) {
        match self.history_mode {
            HistoryMode::None => {
                self.machine.reset();
            }
            HistoryMode::Shallow | HistoryMode::Deep => {
                if let Some(ref history) = self.history_state {
                    self.machine.force_state(history.clone());
                } else {
                    self.machine.reset();
                }
            }
        }
    }

    /// Clears stored history.
    pub fn clear_history(&mut self) {
        self.history_state = None;
    }
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// A generic finite state machine.
///
/// `S` is the state type, `E` is the event type. Both must be cloneable,
/// equatable, hashable, and debuggable.
pub struct StateMachine<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Current active state.
    current: S,
    /// The initial state (used by `reset()`).
    initial: S,
    /// Transition table: maps (from_state, event_name) -> list of transitions
    /// (sorted by priority, highest first).
    transitions: HashMap<S, Vec<Transition<S, E>>>,
    /// Named guard functions.
    guards: HashMap<String, GuardFn<S>>,
    /// Sub-machines keyed by parent state.
    sub_machines: HashMap<S, SubMachine<S, E>>,
    /// Callback invoked on every transition.
    on_transition: Option<Box<dyn Fn(&S, &S) + Send + Sync>>,
    /// How long the machine has been in the current state.
    time_in_state: f32,
    /// History of state transitions (for debugging).
    transition_log: Vec<TransitionLogEntry<S>>,
    /// Maximum log size (0 = unlimited).
    max_log_size: usize,
    /// Whether the machine is running.
    running: bool,
    /// Event name extraction function (maps event to string for lookup).
    event_name_fn: fn(&E) -> String,
}

impl<S, E> Debug for StateMachine<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateMachine")
            .field("current", &self.current)
            .field("initial", &self.initial)
            .field("running", &self.running)
            .field("time_in_state", &self.time_in_state)
            .field("transition_count", &self.transitions.len())
            .finish()
    }
}

/// A log entry recording a state transition.
#[derive(Debug, Clone)]
pub struct TransitionLogEntry<S: Clone + Eq + Hash + Debug> {
    /// The state transitioned from.
    pub from: S,
    /// The state transitioned to.
    pub to: S,
    /// The time spent in the previous state (seconds).
    pub time_in_previous: f32,
}

impl<S, E> StateMachine<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Creates a new state machine with the given initial state and event
    /// name extraction function.
    ///
    /// The `event_name_fn` converts an event to a string key for transition
    /// table lookup. For enums, this is typically `format!("{:?}", event)`.
    pub fn new(initial_state: S, event_name_fn: fn(&E) -> String) -> Self {
        Self {
            current: initial_state.clone(),
            initial: initial_state,
            transitions: HashMap::new(),
            guards: HashMap::new(),
            sub_machines: HashMap::new(),
            on_transition: None,
            time_in_state: 0.0,
            transition_log: Vec::new(),
            max_log_size: 100,
            running: true,
            event_name_fn,
        }
    }

    /// Adds a transition rule.
    pub fn add_transition(&mut self, transition: Transition<S, E>) {
        self.transitions
            .entry(transition.from_state.clone())
            .or_default()
            .push(transition);
    }

    /// Registers a named guard function.
    pub fn add_guard(
        &mut self,
        name: impl Into<String>,
        guard: impl Fn(&S) -> bool + Send + Sync + 'static,
    ) {
        self.guards.insert(name.into(), Box::new(guard));
    }

    /// Attaches a sub-machine to a parent state.
    pub fn add_sub_machine(
        &mut self,
        parent_state: S,
        sub_machine: StateMachine<S, E>,
        history_mode: HistoryMode,
    ) {
        self.sub_machines
            .insert(parent_state, SubMachine::new(sub_machine, history_mode));
    }

    /// Sets a callback invoked on every successful transition.
    pub fn set_on_transition(
        &mut self,
        callback: impl Fn(&S, &S) + Send + Sync + 'static,
    ) {
        self.on_transition = Some(Box::new(callback));
    }

    /// Sets the maximum transition log size.
    pub fn set_max_log_size(&mut self, max: usize) {
        self.max_log_size = max;
    }

    /// Returns the current state.
    pub fn current_state(&self) -> &S {
        &self.current
    }

    /// Returns how long the machine has been in the current state.
    pub fn time_in_state(&self) -> f32 {
        self.time_in_state
    }

    /// Returns whether the machine is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Stops the machine.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Starts the machine.
    pub fn start(&mut self) {
        self.running = true;
    }

    /// Returns the transition log.
    pub fn transition_log(&self) -> &[TransitionLogEntry<S>] {
        &self.transition_log
    }

    /// Returns the active sub-machine for the current state, if any.
    pub fn active_sub_machine(&self) -> Option<&SubMachine<S, E>> {
        self.sub_machines.get(&self.current)
    }

    /// Returns the active sub-machine for the current state, mutably.
    pub fn active_sub_machine_mut(&mut self) -> Option<&mut SubMachine<S, E>> {
        self.sub_machines.get_mut(&self.current)
    }

    /// Returns the full state path including sub-machine states.
    pub fn state_path(&self) -> Vec<S> {
        let mut path = vec![self.current.clone()];
        if let Some(sub) = self.sub_machines.get(&self.current) {
            let sub_path = sub.machine.state_path();
            path.extend(sub_path);
        }
        path
    }

    /// Sends an event to the state machine and processes the resulting
    /// transition, if any.
    pub fn send_event(&mut self, event: &E) -> TransitionResult<S> {
        if !self.running {
            return TransitionResult::NoTransition;
        }

        // First, try to handle the event in the active sub-machine
        if let Some(sub) = self.sub_machines.get_mut(&self.current) {
            let sub_result = sub.machine.send_event(event);
            if matches!(sub_result, TransitionResult::Transitioned { .. }) {
                return sub_result;
            }
        }

        // Look up transitions from the current state
        let event_name = (self.event_name_fn)(event);

        let matching_transitions: Vec<Transition<S, E>> = self
            .transitions
            .get(&self.current)
            .map(|trans_list| {
                let mut matches: Vec<_> = trans_list
                    .iter()
                    .filter(|t| (self.event_name_fn)(&t.event) == event_name)
                    .cloned()
                    .collect();
                matches.sort_by(|a, b| b.priority.cmp(&a.priority));
                matches
            })
            .unwrap_or_default();

        for transition in &matching_transitions {
            // Check guard
            if let Some(ref guard_name) = transition.guard {
                if let Some(guard_fn) = self.guards.get(guard_name) {
                    if !guard_fn(&self.current) {
                        return TransitionResult::GuardBlocked {
                            from: self.current.clone(),
                            to: transition.to_state.clone(),
                            guard: guard_name.clone(),
                        };
                    }
                }
            }

            // Execute transition
            let from = self.current.clone();
            let to = transition.to_state.clone();

            // Exit sub-machine if present
            if let Some(sub) = self.sub_machines.get_mut(&from) {
                sub.save_history();
            }

            // Log the transition
            let entry = TransitionLogEntry {
                from: from.clone(),
                to: to.clone(),
                time_in_previous: self.time_in_state,
            };
            self.transition_log.push(entry);
            if self.max_log_size > 0 && self.transition_log.len() > self.max_log_size {
                self.transition_log.remove(0);
            }

            // Fire callback
            if let Some(ref callback) = self.on_transition {
                callback(&from, &to);
            }

            self.current = to.clone();
            self.time_in_state = 0.0;

            // Enter sub-machine if the new state has one
            if let Some(sub) = self.sub_machines.get_mut(&to) {
                sub.enter();
            }

            return TransitionResult::Transitioned { from, to };
        }

        TransitionResult::NoTransition
    }

    /// Updates the state machine (call once per frame).
    ///
    /// Increments the time-in-state counter. If this state machine has
    /// sub-machines, the active sub-machine is also updated.
    pub fn update(&mut self, dt: f32) {
        if !self.running {
            return;
        }

        self.time_in_state += dt;

        // Update active sub-machine
        if let Some(sub) = self.sub_machines.get_mut(&self.current) {
            sub.machine.update(dt);
        }
    }

    /// Forces the machine into a specific state without triggering
    /// transitions or callbacks. Used internally for history restore.
    pub fn force_state(&mut self, state: S) {
        self.current = state;
        self.time_in_state = 0.0;
    }

    /// Resets the machine to its initial state.
    pub fn reset(&mut self) {
        self.current = self.initial.clone();
        self.time_in_state = 0.0;
        self.transition_log.clear();

        // Reset all sub-machines
        for sub in self.sub_machines.values_mut() {
            sub.clear_history();
            sub.machine.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// StateMachineBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`StateMachine`].
///
/// # Example
///
/// ```rust,ignore
/// let fsm = StateMachineBuilder::new(GameState::MainMenu, |e| format!("{:?}", e))
///     .add_transition(GameState::MainMenu, GameEvent::StartGame, GameState::Loading)
///     .add_transition(GameState::Loading, GameEvent::LoadComplete, GameState::Playing)
///     .add_transition(GameState::Playing, GameEvent::Pause, GameState::Paused)
///     .add_transition(GameState::Paused, GameEvent::Resume, GameState::Playing)
///     .add_transition(GameState::Playing, GameEvent::Die, GameState::GameOver)
///     .add_transition(GameState::GameOver, GameEvent::Restart, GameState::Loading)
///     .add_transition(GameState::GameOver, GameEvent::QuitToMenu, GameState::MainMenu)
///     .build();
/// ```
pub struct StateMachineBuilder<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    machine: StateMachine<S, E>,
}

impl<S, E> StateMachineBuilder<S, E>
where
    S: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    /// Creates a new builder with the given initial state.
    pub fn new(initial_state: S, event_name_fn: fn(&E) -> String) -> Self {
        Self {
            machine: StateMachine::new(initial_state, event_name_fn),
        }
    }

    /// Adds a simple transition.
    pub fn add_transition(mut self, from: S, event: E, to: S) -> Self {
        self.machine
            .add_transition(Transition::new(from, event, to));
        self
    }

    /// Adds a guarded transition.
    pub fn add_guarded_transition(
        mut self,
        from: S,
        event: E,
        to: S,
        guard: impl Into<String>,
    ) -> Self {
        self.machine
            .add_transition(Transition::new(from, event, to).with_guard(guard));
        self
    }

    /// Adds a transition with priority.
    pub fn add_prioritized_transition(
        mut self,
        from: S,
        event: E,
        to: S,
        priority: i32,
    ) -> Self {
        self.machine
            .add_transition(Transition::new(from, event, to).with_priority(priority));
        self
    }

    /// Registers a named guard function.
    pub fn add_guard(
        mut self,
        name: impl Into<String>,
        guard: impl Fn(&S) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.machine.add_guard(name, guard);
        self
    }

    /// Attaches a sub-machine to a parent state.
    pub fn add_sub_machine(
        mut self,
        parent_state: S,
        sub_machine: StateMachine<S, E>,
        history_mode: HistoryMode,
    ) -> Self {
        self.machine
            .add_sub_machine(parent_state, sub_machine, history_mode);
        self
    }

    /// Sets a transition callback.
    pub fn on_transition(
        mut self,
        callback: impl Fn(&S, &S) + Send + Sync + 'static,
    ) -> Self {
        self.machine.set_on_transition(callback);
        self
    }

    /// Sets the maximum transition log size.
    pub fn max_log_size(mut self, max: usize) -> Self {
        self.machine.set_max_log_size(max);
        self
    }

    /// Builds and returns the configured state machine.
    pub fn build(self) -> StateMachine<S, E> {
        self.machine
    }
}

// ---------------------------------------------------------------------------
// GameState example
// ---------------------------------------------------------------------------

/// Example game state enum demonstrating common game flow states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameState {
    /// Main menu / title screen.
    MainMenu,
    /// Loading / streaming assets.
    Loading,
    /// Active gameplay.
    Playing,
    /// Game is paused (overlay).
    Paused,
    /// Player died or game ended.
    GameOver,
    /// Settings / options screen.
    Settings,
    /// Credits screen.
    Credits,
}

impl std::fmt::Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Example game event enum for the [`GameState`] state machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GameEvent {
    /// Start a new game from the main menu.
    StartGame,
    /// Loading has completed.
    LoadComplete,
    /// Player paused the game.
    Pause,
    /// Player resumed from pause.
    Resume,
    /// Player died or a game-over condition was met.
    Die,
    /// Restart after game over.
    Restart,
    /// Return to the main menu.
    QuitToMenu,
    /// Open the settings screen.
    OpenSettings,
    /// Close the settings screen (back to previous).
    CloseSettings,
    /// Open credits.
    OpenCredits,
    /// Close credits.
    CloseCredits,
}

impl std::fmt::Display for GameEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Creates a standard game flow state machine using [`GameState`] and
/// [`GameEvent`].
///
/// State graph:
/// ```text
/// MainMenu --StartGame--> Loading --LoadComplete--> Playing
/// Playing --Pause--> Paused --Resume--> Playing
/// Playing --Die--> GameOver --Restart--> Loading
/// GameOver --QuitToMenu--> MainMenu
/// MainMenu --OpenSettings--> Settings --CloseSettings--> MainMenu
/// MainMenu --OpenCredits--> Credits --CloseCredits--> MainMenu
/// Paused --QuitToMenu--> MainMenu
/// ```
pub fn create_game_state_machine() -> StateMachine<GameState, GameEvent> {
    StateMachineBuilder::new(GameState::MainMenu, |e| format!("{:?}", e))
        .add_transition(GameState::MainMenu, GameEvent::StartGame, GameState::Loading)
        .add_transition(
            GameState::Loading,
            GameEvent::LoadComplete,
            GameState::Playing,
        )
        .add_transition(GameState::Playing, GameEvent::Pause, GameState::Paused)
        .add_transition(GameState::Paused, GameEvent::Resume, GameState::Playing)
        .add_transition(GameState::Playing, GameEvent::Die, GameState::GameOver)
        .add_transition(
            GameState::GameOver,
            GameEvent::Restart,
            GameState::Loading,
        )
        .add_transition(
            GameState::GameOver,
            GameEvent::QuitToMenu,
            GameState::MainMenu,
        )
        .add_transition(
            GameState::Paused,
            GameEvent::QuitToMenu,
            GameState::MainMenu,
        )
        .add_transition(
            GameState::MainMenu,
            GameEvent::OpenSettings,
            GameState::Settings,
        )
        .add_transition(
            GameState::Settings,
            GameEvent::CloseSettings,
            GameState::MainMenu,
        )
        .add_transition(
            GameState::MainMenu,
            GameEvent::OpenCredits,
            GameState::Credits,
        )
        .add_transition(
            GameState::Credits,
            GameEvent::CloseCredits,
            GameState::MainMenu,
        )
        .build()
}

// ---------------------------------------------------------------------------
// PlayingSubState example
// ---------------------------------------------------------------------------

/// Example sub-states for the Playing state, demonstrating hierarchical FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayingSubState {
    /// Player is exploring the world.
    Exploring,
    /// Player is in combat.
    InCombat,
    /// Player is in a dialogue/cutscene.
    InDialogue,
    /// Player is in their inventory/menu.
    InMenu,
}

/// Events for the playing sub-state machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlayingSubEvent {
    /// An enemy spotted the player.
    EnemyEngaged,
    /// All enemies defeated or fled.
    CombatEnded,
    /// Player interacted with an NPC.
    DialogueStarted,
    /// Dialogue completed.
    DialogueEnded,
    /// Player opened in-game menu.
    MenuOpened,
    /// Player closed in-game menu.
    MenuClosed,
}

/// Creates a sub-state machine for the Playing state.
pub fn create_playing_sub_machine() -> StateMachine<PlayingSubState, PlayingSubEvent> {
    StateMachineBuilder::new(PlayingSubState::Exploring, |e| format!("{:?}", e))
        .add_transition(
            PlayingSubState::Exploring,
            PlayingSubEvent::EnemyEngaged,
            PlayingSubState::InCombat,
        )
        .add_transition(
            PlayingSubState::InCombat,
            PlayingSubEvent::CombatEnded,
            PlayingSubState::Exploring,
        )
        .add_transition(
            PlayingSubState::Exploring,
            PlayingSubEvent::DialogueStarted,
            PlayingSubState::InDialogue,
        )
        .add_transition(
            PlayingSubState::InDialogue,
            PlayingSubEvent::DialogueEnded,
            PlayingSubState::Exploring,
        )
        .add_transition(
            PlayingSubState::Exploring,
            PlayingSubEvent::MenuOpened,
            PlayingSubState::InMenu,
        )
        .add_transition(
            PlayingSubState::InMenu,
            PlayingSubEvent::MenuClosed,
            PlayingSubState::Exploring,
        )
        .build()
}

// ---------------------------------------------------------------------------
// AnyStateMachine (type-erased wrapper)
// ---------------------------------------------------------------------------

/// Type-erased trait for state machines, useful for storing heterogeneous
/// machines in a collection.
pub trait AnyStateMachine: Debug + Send + Sync {
    /// Returns the current state as a debug string.
    fn current_state_name(&self) -> String;

    /// Returns the time spent in the current state.
    fn time_in_current_state(&self) -> f32;

    /// Updates the machine.
    fn update_dt(&mut self, dt: f32);

    /// Returns whether the machine is running.
    fn is_active(&self) -> bool;

    /// Resets the machine.
    fn reset_machine(&mut self);
}

impl<S, E> AnyStateMachine for StateMachine<S, E>
where
    S: Clone + Eq + Hash + Debug + Send + Sync + 'static,
    E: Clone + Debug + Send + Sync + 'static,
{
    fn current_state_name(&self) -> String {
        format!("{:?}", self.current)
    }

    fn time_in_current_state(&self) -> f32 {
        self.time_in_state
    }

    fn update_dt(&mut self, dt: f32) {
        self.update(dt);
    }

    fn is_active(&self) -> bool {
        self.running
    }

    fn reset_machine(&mut self) {
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// StateMachineComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component that holds a type-erased state machine.
pub struct StateMachineComponent {
    /// The state machine instance.
    pub machine: Box<dyn AnyStateMachine>,
    /// Label for debugging.
    pub label: String,
}

impl StateMachineComponent {
    /// Creates a new component with the given state machine and label.
    pub fn new(
        machine: impl AnyStateMachine + 'static,
        label: impl Into<String>,
    ) -> Self {
        Self {
            machine: Box::new(machine),
            label: label.into(),
        }
    }
}

impl Debug for StateMachineComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateMachineComponent")
            .field("label", &self.label)
            .field("current_state", &self.machine.current_state_name())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TimedTransition
// ---------------------------------------------------------------------------

/// A utility for transitions that should occur after a time delay.
///
/// Useful for timed states like "Loading" that transition after a fixed
/// duration, or "Stunned" states with a recovery timer.
#[derive(Debug, Clone)]
pub struct TimedTransition<E: Clone + Debug> {
    /// The event to fire when the timer expires.
    pub event: E,
    /// Duration in seconds before the event fires.
    pub duration: f32,
    /// Elapsed time.
    elapsed: f32,
    /// Whether the event has been fired.
    fired: bool,
}

impl<E: Clone + Debug> TimedTransition<E> {
    /// Creates a new timed transition.
    pub fn new(event: E, duration: f32) -> Self {
        Self {
            event,
            duration,
            elapsed: 0.0,
            fired: false,
        }
    }

    /// Updates the timer and returns the event if it should fire.
    pub fn update(&mut self, dt: f32) -> Option<&E> {
        if self.fired {
            return None;
        }
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.fired = true;
            Some(&self.event)
        } else {
            None
        }
    }

    /// Resets the timer.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.fired = false;
    }

    /// Returns the remaining time.
    pub fn remaining(&self) -> f32 {
        (self.duration - self.elapsed).max(0.0)
    }

    /// Returns the progress as a 0..1 fraction.
    pub fn progress(&self) -> f32 {
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }

    /// Returns whether the timer has fired.
    pub fn has_fired(&self) -> bool {
        self.fired
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_transitions() {
        let mut fsm = create_game_state_machine();
        assert_eq!(*fsm.current_state(), GameState::MainMenu);

        fsm.send_event(&GameEvent::StartGame);
        assert_eq!(*fsm.current_state(), GameState::Loading);

        fsm.send_event(&GameEvent::LoadComplete);
        assert_eq!(*fsm.current_state(), GameState::Playing);

        fsm.send_event(&GameEvent::Pause);
        assert_eq!(*fsm.current_state(), GameState::Paused);

        fsm.send_event(&GameEvent::Resume);
        assert_eq!(*fsm.current_state(), GameState::Playing);
    }

    #[test]
    fn test_no_transition() {
        let mut fsm = create_game_state_machine();
        let result = fsm.send_event(&GameEvent::LoadComplete);
        assert_eq!(result, TransitionResult::NoTransition);
        assert_eq!(*fsm.current_state(), GameState::MainMenu);
    }

    #[test]
    fn test_game_over_flow() {
        let mut fsm = create_game_state_machine();
        fsm.send_event(&GameEvent::StartGame);
        fsm.send_event(&GameEvent::LoadComplete);
        fsm.send_event(&GameEvent::Die);
        assert_eq!(*fsm.current_state(), GameState::GameOver);

        fsm.send_event(&GameEvent::Restart);
        assert_eq!(*fsm.current_state(), GameState::Loading);
    }

    #[test]
    fn test_quit_to_menu_from_paused() {
        let mut fsm = create_game_state_machine();
        fsm.send_event(&GameEvent::StartGame);
        fsm.send_event(&GameEvent::LoadComplete);
        fsm.send_event(&GameEvent::Pause);
        fsm.send_event(&GameEvent::QuitToMenu);
        assert_eq!(*fsm.current_state(), GameState::MainMenu);
    }

    #[test]
    fn test_settings_flow() {
        let mut fsm = create_game_state_machine();
        fsm.send_event(&GameEvent::OpenSettings);
        assert_eq!(*fsm.current_state(), GameState::Settings);
        fsm.send_event(&GameEvent::CloseSettings);
        assert_eq!(*fsm.current_state(), GameState::MainMenu);
    }

    #[test]
    fn test_guard_blocks_transition() {
        let mut fsm = StateMachineBuilder::new(GameState::MainMenu, |e| format!("{:?}", e))
            .add_guarded_transition(
                GameState::MainMenu,
                GameEvent::StartGame,
                GameState::Loading,
                "has_save_data",
            )
            .add_guard("has_save_data", |_| false) // Guard always blocks
            .build();

        let result = fsm.send_event(&GameEvent::StartGame);
        assert!(matches!(result, TransitionResult::GuardBlocked { .. }));
        assert_eq!(*fsm.current_state(), GameState::MainMenu);
    }

    #[test]
    fn test_reset() {
        let mut fsm = create_game_state_machine();
        fsm.send_event(&GameEvent::StartGame);
        fsm.send_event(&GameEvent::LoadComplete);
        assert_eq!(*fsm.current_state(), GameState::Playing);

        fsm.reset();
        assert_eq!(*fsm.current_state(), GameState::MainMenu);
    }

    #[test]
    fn test_time_in_state() {
        let mut fsm = create_game_state_machine();
        fsm.update(0.5);
        assert!((fsm.time_in_state() - 0.5).abs() < 1e-6);
        fsm.update(0.3);
        assert!((fsm.time_in_state() - 0.8).abs() < 1e-6);

        fsm.send_event(&GameEvent::StartGame);
        assert!((fsm.time_in_state() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transition_log() {
        let mut fsm = create_game_state_machine();
        fsm.send_event(&GameEvent::StartGame);
        fsm.send_event(&GameEvent::LoadComplete);
        assert_eq!(fsm.transition_log().len(), 2);
        assert_eq!(fsm.transition_log()[0].from, GameState::MainMenu);
        assert_eq!(fsm.transition_log()[0].to, GameState::Loading);
    }

    #[test]
    fn test_stop_and_start() {
        let mut fsm = create_game_state_machine();
        fsm.stop();
        assert!(!fsm.is_running());

        let result = fsm.send_event(&GameEvent::StartGame);
        assert_eq!(result, TransitionResult::NoTransition);

        fsm.start();
        fsm.send_event(&GameEvent::StartGame);
        assert_eq!(*fsm.current_state(), GameState::Loading);
    }

    #[test]
    fn test_state_path() {
        let fsm = create_game_state_machine();
        let path = fsm.state_path();
        assert_eq!(path, vec![GameState::MainMenu]);
    }

    #[test]
    fn test_timed_transition() {
        let mut timer = TimedTransition::new(GameEvent::LoadComplete, 2.0);
        assert!(timer.update(0.5).is_none());
        assert!((timer.remaining() - 1.5).abs() < 1e-6);
        assert!((timer.progress() - 0.25).abs() < 1e-6);

        assert!(timer.update(0.5).is_none());
        assert!(timer.update(1.0).is_some());
        assert!(timer.has_fired());

        // Should not fire again
        assert!(timer.update(1.0).is_none());

        timer.reset();
        assert!(!timer.has_fired());
    }

    #[test]
    fn test_playing_sub_machine() {
        let mut sub = create_playing_sub_machine();
        assert_eq!(*sub.current_state(), PlayingSubState::Exploring);

        sub.send_event(&PlayingSubEvent::EnemyEngaged);
        assert_eq!(*sub.current_state(), PlayingSubState::InCombat);

        sub.send_event(&PlayingSubEvent::CombatEnded);
        assert_eq!(*sub.current_state(), PlayingSubState::Exploring);
    }

    #[test]
    fn test_any_state_machine_trait() {
        let fsm = create_game_state_machine();
        let any: Box<dyn AnyStateMachine> = Box::new(fsm);
        assert_eq!(any.current_state_name(), "MainMenu");
        assert!(any.is_active());
    }

    #[test]
    fn test_state_machine_component() {
        let fsm = create_game_state_machine();
        let comp = StateMachineComponent::new(fsm, "game_flow");
        assert_eq!(comp.label, "game_flow");
        assert_eq!(comp.machine.current_state_name(), "MainMenu");
    }

    #[test]
    fn test_on_transition_callback() {
        use std::sync::{Arc, Mutex};

        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let mut fsm = StateMachineBuilder::new(GameState::MainMenu, |e| format!("{:?}", e))
            .add_transition(GameState::MainMenu, GameEvent::StartGame, GameState::Loading)
            .on_transition(move |from, to| {
                log_clone.lock().unwrap().push(format!("{:?} -> {:?}", from, to));
            })
            .build();

        fsm.send_event(&GameEvent::StartGame);
        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], "MainMenu -> Loading");
    }
}
