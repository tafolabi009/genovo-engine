// engine/gameplay/src/tutorial_system.rs
//
// Tutorial and onboarding system for the Genovo gameplay framework.
//
// Provides tutorial steps with conditions/triggers, tooltip positioning,
// highlight UI elements, forced player actions, tutorial progress tracking,
// and skip tutorial option.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a tutorial step.
pub type TutorialStepId = u32;

/// Unique identifier for a tutorial sequence.
pub type TutorialId = u32;

/// 2D position for UI elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    #[inline]
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

/// Screen rectangle.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self { Self { x, y, width: w, height: h } }
    pub fn center(&self) -> Vec2 { Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5) }
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x <= self.x + self.width && p.y >= self.y && p.y <= self.y + self.height
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum tutorial steps per sequence.
pub const MAX_STEPS_PER_TUTORIAL: usize = 100;
/// Default tooltip display duration (seconds).
pub const DEFAULT_TOOLTIP_DURATION: f32 = 0.0; // 0 = wait for condition.
/// Highlight pulse speed.
pub const HIGHLIGHT_PULSE_SPEED: f32 = 3.0;
/// Default tooltip margin from highlighted element.
pub const DEFAULT_TOOLTIP_MARGIN: f32 = 8.0;

// ---------------------------------------------------------------------------
// Tutorial conditions
// ---------------------------------------------------------------------------

/// Condition that must be met to advance a tutorial step.
#[derive(Debug, Clone)]
pub enum TutorialCondition {
    /// Player pressed a specific key/button.
    InputPressed(String),
    /// Player performed an action (string identifier).
    ActionPerformed(String),
    /// A game variable reached a threshold.
    VariableReached { variable: String, threshold: f32 },
    /// Player moved to a location (within radius).
    ReachedLocation { x: f32, y: f32, z: f32, radius: f32 },
    /// Time elapsed since step started.
    TimeElapsed(f32),
    /// Player clicked on a specific UI element.
    UiClicked(String),
    /// An item was collected/obtained.
    ItemObtained(String),
    /// Enemy was defeated.
    EnemyDefeated(String),
    /// Custom condition (checked via callback ID).
    Custom(u32),
    /// Multiple conditions all met (AND).
    All(Vec<TutorialCondition>),
    /// At least one condition met (OR).
    Any(Vec<TutorialCondition>),
    /// Immediate (auto-advance).
    Immediate,
}

/// Trigger that starts a tutorial or step.
#[derive(Debug, Clone)]
pub enum TutorialTrigger {
    /// Triggered when the player enters an area.
    EnterArea { x: f32, y: f32, z: f32, radius: f32 },
    /// Triggered by a game event.
    GameEvent(String),
    /// Triggered when a level starts.
    LevelStart(String),
    /// Triggered manually by game code.
    Manual,
    /// Triggered on first play (new save).
    FirstPlay,
    /// Triggered when an item is first acquired.
    FirstItemAcquired(String),
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

/// Tooltip anchor position relative to the highlighted element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TooltipAnchor {
    Above,
    Below,
    Left,
    Right,
    Center,
    Auto,
}

/// A tooltip to display during a tutorial step.
#[derive(Debug, Clone)]
pub struct TutorialTooltip {
    /// Text to display.
    pub text: String,
    /// Optional title/header.
    pub title: Option<String>,
    /// Anchor position relative to the highlighted element.
    pub anchor: TooltipAnchor,
    /// Custom screen position (overrides anchor if set).
    pub custom_position: Option<Vec2>,
    /// Margin from the highlighted element.
    pub margin: f32,
    /// Maximum width of the tooltip.
    pub max_width: f32,
    /// Whether to show an arrow pointing to the highlighted element.
    pub show_arrow: bool,
    /// Optional image/icon identifier.
    pub icon: Option<String>,
    /// Optional input prompt (e.g. "Press [E]").
    pub input_prompt: Option<String>,
}

impl Default for TutorialTooltip {
    fn default() -> Self {
        Self {
            text: String::new(),
            title: None,
            anchor: TooltipAnchor::Auto,
            custom_position: None,
            margin: DEFAULT_TOOLTIP_MARGIN,
            max_width: 400.0,
            show_arrow: true,
            icon: None,
            input_prompt: None,
        }
    }
}

impl TutorialTooltip {
    /// Create a simple text tooltip.
    pub fn text(text: &str) -> Self {
        Self { text: text.to_string(), ..Default::default() }
    }

    /// Create a tooltip with a title.
    pub fn with_title(text: &str, title: &str) -> Self {
        Self { text: text.to_string(), title: Some(title.to_string()), ..Default::default() }
    }

    /// Compute the screen position for this tooltip given the highlight rect and viewport.
    pub fn compute_position(&self, highlight: &Rect, viewport_width: f32, viewport_height: f32) -> Vec2 {
        if let Some(pos) = self.custom_position {
            return pos;
        }

        let center = highlight.center();
        let anchor = if self.anchor == TooltipAnchor::Auto {
            // Choose best anchor based on available space.
            if center.y > viewport_height * 0.5 { TooltipAnchor::Above } else { TooltipAnchor::Below }
        } else {
            self.anchor
        };

        match anchor {
            TooltipAnchor::Above => Vec2::new(center.x, highlight.y - self.margin),
            TooltipAnchor::Below => Vec2::new(center.x, highlight.y + highlight.height + self.margin),
            TooltipAnchor::Left => Vec2::new(highlight.x - self.margin, center.y),
            TooltipAnchor::Right => Vec2::new(highlight.x + highlight.width + self.margin, center.y),
            TooltipAnchor::Center | TooltipAnchor::Auto => center,
        }
    }
}

// ---------------------------------------------------------------------------
// Highlight
// ---------------------------------------------------------------------------

/// Type of UI highlight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightStyle {
    /// Pulsing border around the element.
    PulsingBorder,
    /// Spotlight effect (darken everything except the element).
    Spotlight,
    /// Glowing outline.
    GlowOutline,
    /// Arrow pointing to the element.
    Arrow,
    /// No highlight.
    None,
}

/// A UI element highlight during a tutorial step.
#[derive(Debug, Clone)]
pub struct TutorialHighlight {
    /// ID of the UI element to highlight.
    pub element_id: String,
    /// Screen rect of the element (updated each frame by the UI system).
    pub rect: Rect,
    /// Highlight style.
    pub style: HighlightStyle,
    /// Highlight color (RGBA).
    pub color: [f32; 4],
    /// Pulse animation phase.
    pub pulse_phase: f32,
    /// Whether to block input outside the highlighted area.
    pub block_outside_input: bool,
}

impl Default for TutorialHighlight {
    fn default() -> Self {
        Self {
            element_id: String::new(),
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            style: HighlightStyle::PulsingBorder,
            color: [1.0, 0.8, 0.2, 1.0],
            pulse_phase: 0.0,
            block_outside_input: false,
        }
    }
}

impl TutorialHighlight {
    /// Update the pulse animation.
    pub fn update(&mut self, dt: f32) {
        self.pulse_phase += HIGHLIGHT_PULSE_SPEED * dt;
        if self.pulse_phase > std::f32::consts::TAU {
            self.pulse_phase -= std::f32::consts::TAU;
        }
    }

    /// Get the current pulse intensity (0..1).
    pub fn pulse_intensity(&self) -> f32 {
        (self.pulse_phase.sin() * 0.5 + 0.5)
    }
}

// ---------------------------------------------------------------------------
// Tutorial step
// ---------------------------------------------------------------------------

/// A single step in a tutorial sequence.
#[derive(Debug, Clone)]
pub struct TutorialStep {
    /// Unique step identifier.
    pub id: TutorialStepId,
    /// Display name for debugging.
    pub name: String,
    /// Tooltip to show.
    pub tooltip: TutorialTooltip,
    /// UI element to highlight (optional).
    pub highlight: Option<TutorialHighlight>,
    /// Condition to complete this step.
    pub completion_condition: TutorialCondition,
    /// Whether to force the player to complete this step (disable other actions).
    pub force_completion: bool,
    /// Optional timeout (auto-advance after this many seconds).
    pub timeout: Option<f32>,
    /// Whether to pause the game during this step.
    pub pause_game: bool,
    /// Whether to allow skipping this step.
    pub skippable: bool,
    /// Camera target (optional: focus camera on a world position).
    pub camera_focus: Option<[f32; 3]>,
    /// Delay before showing this step (seconds).
    pub delay: f32,
    /// Callback ID to invoke when this step starts.
    pub on_start_callback: Option<u32>,
    /// Callback ID to invoke when this step completes.
    pub on_complete_callback: Option<u32>,
}

impl TutorialStep {
    /// Create a simple tutorial step.
    pub fn new(id: TutorialStepId, name: &str, text: &str, condition: TutorialCondition) -> Self {
        Self {
            id,
            name: name.to_string(),
            tooltip: TutorialTooltip::text(text),
            highlight: None,
            completion_condition: condition,
            force_completion: false,
            timeout: None,
            pause_game: false,
            skippable: true,
            camera_focus: None,
            delay: 0.0,
            on_start_callback: None,
            on_complete_callback: None,
        }
    }

    /// Set the highlight element.
    pub fn with_highlight(mut self, element_id: &str, style: HighlightStyle) -> Self {
        self.highlight = Some(TutorialHighlight {
            element_id: element_id.to_string(),
            style,
            ..Default::default()
        });
        self
    }

    /// Force the player to complete this step.
    pub fn force(mut self) -> Self { self.force_completion = true; self }

    /// Set a timeout.
    pub fn with_timeout(mut self, seconds: f32) -> Self { self.timeout = Some(seconds); self }

    /// Pause the game.
    pub fn paused(mut self) -> Self { self.pause_game = true; self }
}

// ---------------------------------------------------------------------------
// Tutorial sequence
// ---------------------------------------------------------------------------

/// State of a tutorial step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    /// Not yet started.
    Pending,
    /// Waiting for delay.
    Delayed,
    /// Active (showing tooltip, waiting for condition).
    Active,
    /// Completed.
    Completed,
    /// Skipped by the player.
    Skipped,
}

/// Runtime state for a step.
#[derive(Debug, Clone)]
pub struct StepRuntime {
    pub step_id: TutorialStepId,
    pub state: StepState,
    pub elapsed: f32,
    pub delay_remaining: f32,
}

/// A complete tutorial sequence.
#[derive(Debug, Clone)]
pub struct Tutorial {
    pub id: TutorialId,
    pub name: String,
    pub steps: Vec<TutorialStep>,
    pub trigger: TutorialTrigger,
    pub repeatable: bool,
    pub priority: u32,
}

impl Tutorial {
    /// Create a new tutorial.
    pub fn new(id: TutorialId, name: &str, trigger: TutorialTrigger) -> Self {
        Self {
            id,
            name: name.to_string(),
            steps: Vec::new(),
            trigger,
            repeatable: false,
            priority: 0,
        }
    }

    /// Add a step.
    pub fn add_step(&mut self, step: TutorialStep) {
        self.steps.push(step);
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize { self.steps.len() }
}

// ---------------------------------------------------------------------------
// Progress tracking
// ---------------------------------------------------------------------------

/// Progress record for a single tutorial.
#[derive(Debug, Clone)]
pub struct TutorialProgress {
    pub tutorial_id: TutorialId,
    pub completed_steps: HashSet<TutorialStepId>,
    pub current_step_index: usize,
    pub started: bool,
    pub completed: bool,
    pub skipped: bool,
    pub start_time: f32,
    pub completion_time: f32,
    pub times_completed: u32,
}

impl TutorialProgress {
    pub fn new(tutorial_id: TutorialId) -> Self {
        Self {
            tutorial_id,
            completed_steps: HashSet::new(),
            current_step_index: 0,
            started: false,
            completed: false,
            skipped: false,
            start_time: 0.0,
            completion_time: 0.0,
            times_completed: 0,
        }
    }

    /// Completion percentage (0..1).
    pub fn completion_ratio(&self, total_steps: usize) -> f32 {
        if total_steps == 0 { return 1.0; }
        self.completed_steps.len() as f32 / total_steps as f32
    }
}

// ---------------------------------------------------------------------------
// Tutorial system
// ---------------------------------------------------------------------------

/// Events emitted by the tutorial system.
#[derive(Debug, Clone)]
pub enum TutorialEvent {
    TutorialStarted(TutorialId),
    TutorialCompleted(TutorialId),
    TutorialSkipped(TutorialId),
    StepStarted(TutorialId, TutorialStepId),
    StepCompleted(TutorialId, TutorialStepId),
    StepSkipped(TutorialId, TutorialStepId),
}

/// Statistics for the tutorial system.
#[derive(Debug, Clone, Copy, Default)]
pub struct TutorialSystemStats {
    pub tutorials_registered: u32,
    pub tutorials_completed: u32,
    pub tutorials_skipped: u32,
    pub tutorials_active: u32,
    pub steps_completed: u32,
    pub steps_skipped: u32,
}

/// Game variable store for tutorial conditions.
pub type GameVariables = HashMap<String, f32>;

/// Callback type for custom conditions and events.
type TutorialCallback = Box<dyn Fn() + Send + Sync>;

/// Main tutorial system.
pub struct TutorialSystem {
    tutorials: HashMap<TutorialId, Tutorial>,
    progress: HashMap<TutorialId, TutorialProgress>,
    active_tutorial: Option<TutorialId>,
    active_step_runtime: Option<StepRuntime>,
    events: VecDeque<TutorialEvent>,
    callbacks: HashMap<u32, TutorialCallback>,
    game_variables: GameVariables,
    performed_actions: HashSet<String>,
    pressed_inputs: HashSet<String>,
    clicked_ui: HashSet<String>,
    stats: TutorialSystemStats,
    allow_skip: bool,
    game_time: f32,
    next_callback_id: u32,
}

impl TutorialSystem {
    /// Create a new tutorial system.
    pub fn new() -> Self {
        Self {
            tutorials: HashMap::new(),
            progress: HashMap::new(),
            active_tutorial: None,
            active_step_runtime: None,
            events: VecDeque::new(),
            callbacks: HashMap::new(),
            game_variables: HashMap::new(),
            performed_actions: HashSet::new(),
            pressed_inputs: HashSet::new(),
            clicked_ui: HashSet::new(),
            stats: TutorialSystemStats::default(),
            allow_skip: true,
            game_time: 0.0,
            next_callback_id: 0,
        }
    }

    /// Register a tutorial.
    pub fn register_tutorial(&mut self, tutorial: Tutorial) {
        let id = tutorial.id;
        self.tutorials.insert(id, tutorial);
        self.progress.entry(id).or_insert_with(|| TutorialProgress::new(id));
        self.stats.tutorials_registered += 1;
    }

    /// Start a specific tutorial.
    pub fn start_tutorial(&mut self, id: TutorialId) -> bool {
        let tutorial = match self.tutorials.get(&id) {
            Some(t) => t.clone(),
            None => return false,
        };

        let progress = self.progress.get(&id);
        if let Some(p) = progress {
            if p.completed && !tutorial.repeatable { return false; }
        }

        self.active_tutorial = Some(id);
        let progress = self.progress.entry(id).or_insert_with(|| TutorialProgress::new(id));
        progress.started = true;
        progress.start_time = self.game_time;
        progress.current_step_index = 0;

        self.events.push_back(TutorialEvent::TutorialStarted(id));
        self.stats.tutorials_active += 1;

        // Start first step.
        if let Some(step) = tutorial.steps.first() {
            self.start_step(id, step.id, step.delay);
        }

        true
    }

    /// Start a tutorial step.
    fn start_step(&mut self, tutorial_id: TutorialId, step_id: TutorialStepId, delay: f32) {
        let state = if delay > 0.0 { StepState::Delayed } else { StepState::Active };
        self.active_step_runtime = Some(StepRuntime {
            step_id,
            state,
            elapsed: 0.0,
            delay_remaining: delay,
        });

        if state == StepState::Active {
            self.events.push_back(TutorialEvent::StepStarted(tutorial_id, step_id));
        }
    }

    /// Skip the current tutorial.
    pub fn skip_current_tutorial(&mut self) {
        if !self.allow_skip { return; }
        if let Some(id) = self.active_tutorial.take() {
            if let Some(progress) = self.progress.get_mut(&id) {
                progress.skipped = true;
                progress.completed = true;
            }
            self.active_step_runtime = None;
            self.events.push_back(TutorialEvent::TutorialSkipped(id));
            self.stats.tutorials_skipped += 1;
            self.stats.tutorials_active = self.stats.tutorials_active.saturating_sub(1);
        }
    }

    /// Skip the current step.
    pub fn skip_current_step(&mut self) {
        if let (Some(tid), Some(runtime)) = (self.active_tutorial, &self.active_step_runtime) {
            let step_id = runtime.step_id;
            self.events.push_back(TutorialEvent::StepSkipped(tid, step_id));
            self.stats.steps_skipped += 1;
            self.advance_to_next_step();
        }
    }

    /// Report that a player action was performed.
    pub fn report_action(&mut self, action: &str) {
        self.performed_actions.insert(action.to_string());
    }

    /// Report that an input was pressed.
    pub fn report_input(&mut self, input: &str) {
        self.pressed_inputs.insert(input.to_string());
    }

    /// Report that a UI element was clicked.
    pub fn report_ui_click(&mut self, element_id: &str) {
        self.clicked_ui.insert(element_id.to_string());
    }

    /// Set a game variable.
    pub fn set_variable(&mut self, name: &str, value: f32) {
        self.game_variables.insert(name.to_string(), value);
    }

    /// Update the tutorial system.
    pub fn update(&mut self, dt: f32) {
        self.game_time += dt;

        if let Some(tid) = self.active_tutorial {
            let tutorial = self.tutorials.get(&tid).cloned();
            if let Some(tutorial) = tutorial {
                if let Some(ref mut runtime) = self.active_step_runtime {
                    // Handle delay.
                    if runtime.state == StepState::Delayed {
                        runtime.delay_remaining -= dt;
                        if runtime.delay_remaining <= 0.0 {
                            runtime.state = StepState::Active;
                            self.events.push_back(TutorialEvent::StepStarted(tid, runtime.step_id));
                        }
                        return;
                    }

                    runtime.elapsed += dt;

                    // Update highlight animation.
                    let step = tutorial.steps.iter().find(|s| s.id == runtime.step_id);
                    if let Some(_step) = step {
                        // Check timeout.
                        if let Some(step) = tutorial.steps.iter().find(|s| s.id == runtime.step_id) {
                            if let Some(timeout) = step.timeout {
                                if runtime.elapsed >= timeout {
                                    self.complete_current_step();
                                    return;
                                }
                            }

                            // Check completion condition.
                            if self.evaluate_condition(&step.completion_condition) {
                                self.complete_current_step();
                            }
                        }
                    }
                }
            }
        }

        // Clear per-frame inputs.
        self.pressed_inputs.clear();
        self.clicked_ui.clear();
        self.performed_actions.clear();
    }

    /// Evaluate a tutorial condition.
    fn evaluate_condition(&self, condition: &TutorialCondition) -> bool {
        match condition {
            TutorialCondition::InputPressed(input) => self.pressed_inputs.contains(input),
            TutorialCondition::ActionPerformed(action) => self.performed_actions.contains(action),
            TutorialCondition::VariableReached { variable, threshold } => {
                self.game_variables.get(variable).map_or(false, |v| *v >= *threshold)
            }
            TutorialCondition::ReachedLocation { .. } => false, // Requires world position check.
            TutorialCondition::TimeElapsed(t) => {
                self.active_step_runtime.as_ref().map_or(false, |r| r.elapsed >= *t)
            }
            TutorialCondition::UiClicked(element) => self.clicked_ui.contains(element),
            TutorialCondition::ItemObtained(item) => self.performed_actions.contains(item),
            TutorialCondition::EnemyDefeated(enemy) => self.performed_actions.contains(enemy),
            TutorialCondition::Custom(_id) => false,
            TutorialCondition::All(conditions) => conditions.iter().all(|c| self.evaluate_condition(c)),
            TutorialCondition::Any(conditions) => conditions.iter().any(|c| self.evaluate_condition(c)),
            TutorialCondition::Immediate => true,
        }
    }

    /// Complete the current step and advance.
    fn complete_current_step(&mut self) {
        if let Some(tid) = self.active_tutorial {
            if let Some(runtime) = self.active_step_runtime.take() {
                if let Some(progress) = self.progress.get_mut(&tid) {
                    progress.completed_steps.insert(runtime.step_id);
                    progress.current_step_index += 1;
                }
                self.events.push_back(TutorialEvent::StepCompleted(tid, runtime.step_id));
                self.stats.steps_completed += 1;

                self.advance_to_next_step();
            }
        }
    }

    /// Advance to the next step or complete the tutorial.
    fn advance_to_next_step(&mut self) {
        if let Some(tid) = self.active_tutorial {
            let next_index = self.progress.get(&tid).map_or(0, |p| p.current_step_index);
            let tutorial = self.tutorials.get(&tid).cloned();

            if let Some(tutorial) = tutorial {
                if next_index < tutorial.steps.len() {
                    let step = &tutorial.steps[next_index];
                    self.start_step(tid, step.id, step.delay);
                } else {
                    // Tutorial complete.
                    if let Some(progress) = self.progress.get_mut(&tid) {
                        progress.completed = true;
                        progress.completion_time = self.game_time;
                        progress.times_completed += 1;
                    }
                    self.active_tutorial = None;
                    self.active_step_runtime = None;
                    self.events.push_back(TutorialEvent::TutorialCompleted(tid));
                    self.stats.tutorials_completed += 1;
                    self.stats.tutorials_active = self.stats.tutorials_active.saturating_sub(1);
                }
            }
        }
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<TutorialEvent> {
        self.events.drain(..).collect()
    }

    /// Get the active tutorial ID.
    pub fn active_tutorial(&self) -> Option<TutorialId> { self.active_tutorial }

    /// Get the active step runtime.
    pub fn active_step(&self) -> Option<&StepRuntime> { self.active_step_runtime.as_ref() }

    /// Get the active step definition.
    pub fn active_step_definition(&self) -> Option<&TutorialStep> {
        let tid = self.active_tutorial?;
        let runtime = self.active_step_runtime.as_ref()?;
        let tutorial = self.tutorials.get(&tid)?;
        tutorial.steps.iter().find(|s| s.id == runtime.step_id)
    }

    /// Whether a tutorial has been completed.
    pub fn is_completed(&self, id: TutorialId) -> bool {
        self.progress.get(&id).map_or(false, |p| p.completed)
    }

    /// Get progress for a tutorial.
    pub fn get_progress(&self, id: TutorialId) -> Option<&TutorialProgress> {
        self.progress.get(&id)
    }

    /// Get stats.
    pub fn stats(&self) -> &TutorialSystemStats { &self.stats }

    /// Set whether tutorials can be skipped.
    pub fn set_allow_skip(&mut self, allow: bool) { self.allow_skip = allow; }

    /// Whether a tutorial is active.
    pub fn is_tutorial_active(&self) -> bool { self.active_tutorial.is_some() }

    /// Whether the game should be paused.
    pub fn should_pause_game(&self) -> bool {
        self.active_step_definition().map_or(false, |s| s.pause_game)
    }

    /// Whether input should be blocked outside the highlight.
    pub fn should_block_input(&self) -> bool {
        self.active_step_definition()
            .and_then(|s| s.highlight.as_ref())
            .map_or(false, |h| h.block_outside_input)
    }

    /// Reset all tutorial progress (for new game).
    pub fn reset_all_progress(&mut self) {
        self.progress.clear();
        for &id in self.tutorials.keys() {
            self.progress.insert(id, TutorialProgress::new(id));
        }
        self.active_tutorial = None;
        self.active_step_runtime = None;
        self.events.clear();
        self.stats = TutorialSystemStats {
            tutorials_registered: self.tutorials.len() as u32,
            ..Default::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tutorial() -> Tutorial {
        let mut tutorial = Tutorial::new(1, "Test Tutorial", TutorialTrigger::Manual);
        tutorial.add_step(TutorialStep::new(
            1, "Step 1", "Press W to move forward",
            TutorialCondition::InputPressed("W".to_string()),
        ));
        tutorial.add_step(TutorialStep::new(
            2, "Step 2", "Great! Now press Space to jump",
            TutorialCondition::InputPressed("Space".to_string()),
        ));
        tutorial.add_step(TutorialStep::new(
            3, "Step 3", "Tutorial complete!",
            TutorialCondition::TimeElapsed(2.0),
        ));
        tutorial
    }

    #[test]
    fn test_register_and_start() {
        let mut system = TutorialSystem::new();
        system.register_tutorial(make_test_tutorial());
        assert!(system.start_tutorial(1));
        assert!(system.is_tutorial_active());
        assert_eq!(system.active_tutorial(), Some(1));
    }

    #[test]
    fn test_step_completion() {
        let mut system = TutorialSystem::new();
        system.register_tutorial(make_test_tutorial());
        system.start_tutorial(1);

        // Complete step 1 by pressing W.
        system.report_input("W");
        system.update(0.016);

        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, TutorialEvent::StepCompleted(1, 1))));
    }

    #[test]
    fn test_skip_tutorial() {
        let mut system = TutorialSystem::new();
        system.register_tutorial(make_test_tutorial());
        system.start_tutorial(1);
        system.skip_current_tutorial();

        assert!(!system.is_tutorial_active());
        assert!(system.is_completed(1));
    }

    #[test]
    fn test_tooltip_position() {
        let tooltip = TutorialTooltip::text("Test");
        let rect = Rect::new(100.0, 100.0, 50.0, 20.0);
        let pos = tooltip.compute_position(&rect, 800.0, 600.0);
        assert!(pos.y > rect.y); // Auto should pick below since y < half viewport.
    }

    #[test]
    fn test_tutorial_progress() {
        let mut system = TutorialSystem::new();
        system.register_tutorial(make_test_tutorial());
        system.start_tutorial(1);

        system.report_input("W");
        system.update(0.016);
        system.report_input("Space");
        system.update(0.016);
        system.update(3.0); // Wait for time elapsed step.

        assert!(system.is_completed(1));
        assert_eq!(system.stats().tutorials_completed, 1);
    }
}
