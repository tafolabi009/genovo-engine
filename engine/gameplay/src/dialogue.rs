//! Advanced dialogue system inspired by Ink-style narrative engines.
//!
//! Provides:
//! - **Ink-style flow**: knots, stitches, diverts, gathers, and weave
//! - **Variable tracking**: integer, float, string, and boolean story variables
//! - **Conditional branches**: inline logic for branching based on variables
//! - **Inline logic**: set/check variables within dialogue text
//! - **Bark system**: short one-liner ambient dialogues triggered by context
//! - **Dialogue queuing**: multiple conversations can queue and play sequentially
//! - **NPC schedule-based availability**: NPCs are only available at certain times
//! - **Tagging**: lines can be tagged for audio, animation, or camera cues
//! - **Localization support**: all text goes through string keys
//!
//! # Architecture
//!
//! A [`Story`] contains [`Knot`]s, each containing [`Stitch`]es. The
//! [`StoryRunner`] navigates the story graph, evaluating conditions and
//! tracking state. The [`BarkManager`] handles ambient one-liners.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum story variable name length.
pub const MAX_VAR_NAME_LEN: usize = 128;
/// Maximum number of choices per node.
pub const MAX_CHOICES: usize = 16;
/// Maximum conversation queue depth.
pub const MAX_QUEUE_DEPTH: usize = 8;
/// Maximum bark cooldown in seconds.
pub const DEFAULT_BARK_COOLDOWN: f32 = 30.0;
/// Maximum number of active barks.
pub const MAX_ACTIVE_BARKS: usize = 4;
/// Default bark display duration in seconds.
pub const DEFAULT_BARK_DURATION: f32 = 3.0;
/// Maximum knots per story.
pub const MAX_KNOTS: usize = 1024;
/// Maximum stitches per knot.
pub const MAX_STITCHES: usize = 256;
/// Maximum tags per content line.
pub const MAX_TAGS: usize = 8;
/// Default NPC interaction range.
pub const DEFAULT_INTERACTION_RANGE: f32 = 3.0;

// ---------------------------------------------------------------------------
// StoryVariableValue
// ---------------------------------------------------------------------------

/// A value that can be stored in the story's variable table.
#[derive(Debug, Clone, PartialEq)]
pub enum StoryVariableValue {
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// String value.
    String(String),
    /// Boolean value.
    Bool(bool),
    /// List of string values (for tracking visited paths, etc.).
    List(Vec<String>),
}

impl StoryVariableValue {
    /// Try to get as integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            StoryVariableValue::Int(v) => Some(*v),
            StoryVariableValue::Float(v) => Some(*v as i64),
            StoryVariableValue::Bool(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to get as float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            StoryVariableValue::Int(v) => Some(*v as f64),
            StoryVariableValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            StoryVariableValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            StoryVariableValue::Bool(v) => Some(*v),
            StoryVariableValue::Int(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Check if the value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            StoryVariableValue::Int(v) => *v != 0,
            StoryVariableValue::Float(v) => *v != 0.0,
            StoryVariableValue::String(s) => !s.is_empty(),
            StoryVariableValue::Bool(v) => *v,
            StoryVariableValue::List(l) => !l.is_empty(),
        }
    }
}

impl Default for StoryVariableValue {
    fn default() -> Self {
        StoryVariableValue::Int(0)
    }
}

// ---------------------------------------------------------------------------
// StoryVariables
// ---------------------------------------------------------------------------

/// The variable store for a story, tracking all named variables and visit counts.
#[derive(Debug, Clone)]
pub struct StoryVariables {
    /// Named variables.
    variables: HashMap<String, StoryVariableValue>,
    /// Visit counts for knots/stitches (path -> count).
    visit_counts: HashMap<String, u32>,
    /// Turn index (incremented each time a choice is made).
    pub turn_index: u32,
    /// Global tags (set once, never unset).
    global_tags: HashSet<String>,
    /// Variable change listeners (variable_name -> listener_ids).
    listeners: HashMap<String, Vec<u64>>,
    /// Next listener ID.
    next_listener_id: u64,
}

impl StoryVariables {
    /// Create a new empty variable store.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            visit_counts: HashMap::new(),
            turn_index: 0,
            global_tags: HashSet::new(),
            listeners: HashMap::new(),
            next_listener_id: 1,
        }
    }

    /// Set a variable value.
    pub fn set(&mut self, name: impl Into<String>, value: StoryVariableValue) {
        let name = name.into();
        self.variables.insert(name, value);
    }

    /// Set an integer variable.
    pub fn set_int(&mut self, name: impl Into<String>, value: i64) {
        self.set(name, StoryVariableValue::Int(value));
    }

    /// Set a float variable.
    pub fn set_float(&mut self, name: impl Into<String>, value: f64) {
        self.set(name, StoryVariableValue::Float(value));
    }

    /// Set a string variable.
    pub fn set_string(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.set(name, StoryVariableValue::String(value.into()));
    }

    /// Set a boolean variable.
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.set(name, StoryVariableValue::Bool(value));
    }

    /// Get a variable value.
    pub fn get(&self, name: &str) -> Option<&StoryVariableValue> {
        self.variables.get(name)
    }

    /// Get an integer variable (returns 0 if not set or wrong type).
    pub fn get_int(&self, name: &str) -> i64 {
        self.variables.get(name).and_then(|v| v.as_int()).unwrap_or(0)
    }

    /// Get a float variable.
    pub fn get_float(&self, name: &str) -> f64 {
        self.variables.get(name).and_then(|v| v.as_float()).unwrap_or(0.0)
    }

    /// Get a string variable.
    pub fn get_string(&self, name: &str) -> &str {
        self.variables.get(name).and_then(|v| v.as_string()).unwrap_or("")
    }

    /// Get a bool variable.
    pub fn get_bool(&self, name: &str) -> bool {
        self.variables.get(name).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    /// Check if a variable exists.
    pub fn has(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Increment an integer variable.
    pub fn increment(&mut self, name: &str, amount: i64) {
        let current = self.get_int(name);
        self.set_int(name.to_string(), current + amount);
    }

    /// Record a visit to a knot/stitch path.
    pub fn record_visit(&mut self, path: &str) {
        let count = self.visit_counts.entry(path.to_string()).or_insert(0);
        *count += 1;
    }

    /// Get the visit count for a path.
    pub fn visit_count(&self, path: &str) -> u32 {
        self.visit_counts.get(path).copied().unwrap_or(0)
    }

    /// Check if a path has been visited.
    pub fn has_visited(&self, path: &str) -> bool {
        self.visit_count(path) > 0
    }

    /// Set a global tag.
    pub fn set_tag(&mut self, tag: impl Into<String>) {
        self.global_tags.insert(tag.into());
    }

    /// Check if a global tag is set.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.global_tags.contains(tag)
    }

    /// Advance the turn index.
    pub fn advance_turn(&mut self) {
        self.turn_index += 1;
    }

    /// Get all variable names.
    pub fn variable_names(&self) -> Vec<&str> {
        self.variables.keys().map(|k| k.as_str()).collect()
    }

    /// Clear all variables (but keep visit counts).
    pub fn clear_variables(&mut self) {
        self.variables.clear();
    }

    /// Clear everything.
    pub fn clear_all(&mut self) {
        self.variables.clear();
        self.visit_counts.clear();
        self.turn_index = 0;
        self.global_tags.clear();
    }
}

// ---------------------------------------------------------------------------
// InkCondition
// ---------------------------------------------------------------------------

/// A condition that can be evaluated against the story state.
#[derive(Debug, Clone)]
pub enum InkCondition {
    /// Variable equals a value.
    Equals { variable: String, value: StoryVariableValue },
    /// Variable does not equal a value.
    NotEquals { variable: String, value: StoryVariableValue },
    /// Variable is greater than a value.
    GreaterThan { variable: String, value: f64 },
    /// Variable is less than a value.
    LessThan { variable: String, value: f64 },
    /// Variable is greater than or equal to a value.
    GreaterOrEqual { variable: String, value: f64 },
    /// Variable is less than or equal to a value.
    LessOrEqual { variable: String, value: f64 },
    /// A path has been visited.
    Visited(String),
    /// A path has NOT been visited.
    NotVisited(String),
    /// A path has been visited at least N times.
    VisitedAtLeast { path: String, count: u32 },
    /// A global tag is set.
    TagSet(String),
    /// A global tag is NOT set.
    TagNotSet(String),
    /// Boolean AND of sub-conditions.
    And(Vec<InkCondition>),
    /// Boolean OR of sub-conditions.
    Or(Vec<InkCondition>),
    /// Boolean NOT of a sub-condition.
    Not(Box<InkCondition>),
    /// Variable is truthy (non-zero, non-empty).
    Truthy(String),
    /// Custom condition evaluated by external code.
    External { function: String, args: Vec<String> },
    /// Turn index comparison.
    TurnsSince { path: String, min_turns: u32 },
}

impl InkCondition {
    /// Evaluate this condition against the story variables.
    pub fn evaluate(&self, vars: &StoryVariables) -> bool {
        match self {
            InkCondition::Equals { variable, value } => {
                vars.get(variable).map_or(false, |v| v == value)
            }
            InkCondition::NotEquals { variable, value } => {
                vars.get(variable).map_or(true, |v| v != value)
            }
            InkCondition::GreaterThan { variable, value } => {
                vars.get_float(variable) > *value
            }
            InkCondition::LessThan { variable, value } => {
                vars.get_float(variable) < *value
            }
            InkCondition::GreaterOrEqual { variable, value } => {
                vars.get_float(variable) >= *value
            }
            InkCondition::LessOrEqual { variable, value } => {
                vars.get_float(variable) <= *value
            }
            InkCondition::Visited(path) => vars.has_visited(path),
            InkCondition::NotVisited(path) => !vars.has_visited(path),
            InkCondition::VisitedAtLeast { path, count } => {
                vars.visit_count(path) >= *count
            }
            InkCondition::TagSet(tag) => vars.has_tag(tag),
            InkCondition::TagNotSet(tag) => !vars.has_tag(tag),
            InkCondition::And(conditions) => conditions.iter().all(|c| c.evaluate(vars)),
            InkCondition::Or(conditions) => conditions.iter().any(|c| c.evaluate(vars)),
            InkCondition::Not(condition) => !condition.evaluate(vars),
            InkCondition::Truthy(variable) => {
                vars.get(variable).map_or(false, |v| v.is_truthy())
            }
            InkCondition::External { .. } => {
                // External conditions are resolved by game code
                true
            }
            InkCondition::TurnsSince { path, min_turns } => {
                // Simplified: just check if visited
                vars.has_visited(path) && vars.turn_index >= *min_turns
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InkAction
// ---------------------------------------------------------------------------

/// An action to perform when dialogue content is reached.
#[derive(Debug, Clone)]
pub enum InkAction {
    /// Set a variable.
    SetVariable { name: String, value: StoryVariableValue },
    /// Increment a variable.
    Increment { name: String, amount: i64 },
    /// Set a global tag.
    SetTag(String),
    /// Remove a global tag.
    RemoveTag(String),
    /// Trigger an external event (handled by game code).
    ExternalEvent { event: String, args: Vec<String> },
    /// Start a quest.
    StartQuest(String),
    /// Complete a quest objective.
    CompleteObjective { quest: String, objective: String },
    /// Give an item to the player.
    GiveItem { item_id: String, quantity: u32 },
    /// Remove an item from the player.
    TakeItem { item_id: String, quantity: u32 },
    /// Change reputation with a faction.
    ChangeReputation { faction: String, amount: i32 },
    /// Play an animation on the speaker.
    PlayAnimation { animation: String },
    /// Play a sound effect.
    PlaySound { sound: String },
    /// Trigger a camera movement.
    CameraAction { action: String },
    /// Wait for a duration before continuing.
    Wait(f32),
}

impl InkAction {
    /// Execute this action against the story variables.
    /// Returns any external events that need to be dispatched.
    pub fn execute(&self, vars: &mut StoryVariables) -> Option<DialogueEvent> {
        match self {
            InkAction::SetVariable { name, value } => {
                vars.set(name.clone(), value.clone());
                None
            }
            InkAction::Increment { name, amount } => {
                vars.increment(name, *amount);
                None
            }
            InkAction::SetTag(tag) => {
                vars.set_tag(tag.clone());
                None
            }
            InkAction::RemoveTag(_) => None,
            InkAction::ExternalEvent { event, args } => {
                Some(DialogueEvent::External {
                    event: event.clone(),
                    args: args.clone(),
                })
            }
            InkAction::StartQuest(quest_id) => {
                Some(DialogueEvent::QuestStarted(quest_id.clone()))
            }
            InkAction::CompleteObjective { quest, objective } => {
                Some(DialogueEvent::ObjectiveCompleted {
                    quest: quest.clone(),
                    objective: objective.clone(),
                })
            }
            InkAction::GiveItem { item_id, quantity } => {
                Some(DialogueEvent::ItemGiven {
                    item_id: item_id.clone(),
                    quantity: *quantity,
                })
            }
            InkAction::TakeItem { item_id, quantity } => {
                Some(DialogueEvent::ItemTaken {
                    item_id: item_id.clone(),
                    quantity: *quantity,
                })
            }
            InkAction::ChangeReputation { faction, amount } => {
                Some(DialogueEvent::ReputationChanged {
                    faction: faction.clone(),
                    amount: *amount,
                })
            }
            InkAction::PlayAnimation { animation } => {
                Some(DialogueEvent::AnimationTriggered(animation.clone()))
            }
            InkAction::PlaySound { sound } => {
                Some(DialogueEvent::SoundTriggered(sound.clone()))
            }
            InkAction::CameraAction { action } => {
                Some(DialogueEvent::CameraAction(action.clone()))
            }
            InkAction::Wait(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// DialogueEvent
// ---------------------------------------------------------------------------

/// Events emitted by the dialogue system for game code to handle.
#[derive(Debug, Clone)]
pub enum DialogueEvent {
    /// A conversation started.
    ConversationStarted { npc_id: String, story_id: String },
    /// A conversation ended.
    ConversationEnded { npc_id: String, story_id: String },
    /// A dialogue line was displayed.
    LineDisplayed { speaker: String, text: String },
    /// The player made a choice.
    ChoiceMade { choice_index: usize, choice_text: String },
    /// An external event was triggered by dialogue.
    External { event: String, args: Vec<String> },
    /// A quest was started from dialogue.
    QuestStarted(String),
    /// An objective was completed from dialogue.
    ObjectiveCompleted { quest: String, objective: String },
    /// An item was given to the player.
    ItemGiven { item_id: String, quantity: u32 },
    /// An item was taken from the player.
    ItemTaken { item_id: String, quantity: u32 },
    /// Reputation was changed.
    ReputationChanged { faction: String, amount: i32 },
    /// An animation was triggered.
    AnimationTriggered(String),
    /// A sound was triggered.
    SoundTriggered(String),
    /// A camera action was triggered.
    CameraAction(String),
    /// A bark was displayed.
    BarkDisplayed { npc_id: String, text: String },
}

// ---------------------------------------------------------------------------
// ContentLine
// ---------------------------------------------------------------------------

/// A single line of dialogue content.
#[derive(Debug, Clone)]
pub struct ContentLine {
    /// Speaker ID (empty for narration).
    pub speaker: String,
    /// The dialogue text (or localization key).
    pub text: String,
    /// Whether the text is a localization key.
    pub is_localized: bool,
    /// Tags on this line (e.g., #sad, #whisper, #camera_close).
    pub tags: Vec<String>,
    /// Actions to execute when this line is reached.
    pub actions: Vec<InkAction>,
    /// Condition that must be true for this line to appear.
    pub condition: Option<InkCondition>,
    /// Speaker emotion/mood hint.
    pub emotion: Option<String>,
    /// Audio clip reference.
    pub audio_clip: Option<String>,
    /// Duration override for typewriter effect (seconds per character).
    pub typing_speed: Option<f32>,
}

impl ContentLine {
    /// Create a simple content line.
    pub fn new(speaker: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            text: text.into(),
            is_localized: false,
            tags: Vec::new(),
            actions: Vec::new(),
            condition: None,
            emotion: None,
            audio_clip: None,
            typing_speed: None,
        }
    }

    /// Create a narration line (no speaker).
    pub fn narration(text: impl Into<String>) -> Self {
        Self::new("", text)
    }

    /// Builder: add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builder: add an action.
    pub fn with_action(mut self, action: InkAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Builder: set condition.
    pub fn with_condition(mut self, condition: InkCondition) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Builder: set emotion.
    pub fn with_emotion(mut self, emotion: impl Into<String>) -> Self {
        self.emotion = Some(emotion.into());
        self
    }

    /// Check if this line should be displayed.
    pub fn should_display(&self, vars: &StoryVariables) -> bool {
        self.condition.as_ref().map_or(true, |c| c.evaluate(vars))
    }
}

// ---------------------------------------------------------------------------
// Choice
// ---------------------------------------------------------------------------

/// A choice the player can make in a conversation.
#[derive(Debug, Clone)]
pub struct Choice {
    /// Choice text.
    pub text: String,
    /// Whether the text is a localization key.
    pub is_localized: bool,
    /// Target path to divert to if chosen (knot.stitch or knot).
    pub target: String,
    /// Condition for this choice to be available.
    pub condition: Option<InkCondition>,
    /// Actions to execute when this choice is selected.
    pub actions: Vec<InkAction>,
    /// Whether this choice is a "sticky" choice (reappears after selection).
    pub sticky: bool,
    /// Whether this choice has been selected before.
    pub selected: bool,
    /// Whether to show this choice even if it's been selected (greyed out).
    pub show_when_selected: bool,
    /// Tags on this choice.
    pub tags: Vec<String>,
    /// Fallback choice (used if no other choices are available).
    pub is_fallback: bool,
}

impl Choice {
    /// Create a new choice.
    pub fn new(text: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_localized: false,
            target: target.into(),
            condition: None,
            actions: Vec::new(),
            sticky: false,
            selected: false,
            show_when_selected: false,
            tags: Vec::new(),
            is_fallback: false,
        }
    }

    /// Builder: set condition.
    pub fn with_condition(mut self, condition: InkCondition) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Builder: add action.
    pub fn with_action(mut self, action: InkAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Builder: make sticky.
    pub fn sticky(mut self) -> Self {
        self.sticky = true;
        self
    }

    /// Builder: mark as fallback.
    pub fn as_fallback(mut self) -> Self {
        self.is_fallback = true;
        self
    }

    /// Check if this choice is available.
    pub fn is_available(&self, vars: &StoryVariables) -> bool {
        if self.selected && !self.sticky {
            return false;
        }
        self.condition.as_ref().map_or(true, |c| c.evaluate(vars))
    }
}

// ---------------------------------------------------------------------------
// Stitch
// ---------------------------------------------------------------------------

/// A sub-section within a knot (similar to a function or paragraph).
#[derive(Debug, Clone)]
pub struct Stitch {
    /// Name of this stitch.
    pub name: String,
    /// Content lines in this stitch (displayed sequentially).
    pub content: Vec<ContentLine>,
    /// Choices available after content is exhausted.
    pub choices: Vec<Choice>,
    /// Divert to another path after content (if no choices).
    pub divert: Option<String>,
    /// Actions to execute when entering this stitch.
    pub on_enter: Vec<InkAction>,
    /// Actions to execute when leaving this stitch.
    pub on_exit: Vec<InkAction>,
    /// Condition for this stitch to be accessible.
    pub condition: Option<InkCondition>,
}

impl Stitch {
    /// Create a new stitch.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: Vec::new(),
            choices: Vec::new(),
            divert: None,
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            condition: None,
        }
    }

    /// Add a content line.
    pub fn add_line(&mut self, line: ContentLine) {
        self.content.push(line);
    }

    /// Add a choice.
    pub fn add_choice(&mut self, choice: Choice) {
        self.choices.push(choice);
    }

    /// Set the divert target.
    pub fn set_divert(&mut self, target: impl Into<String>) {
        self.divert = Some(target.into());
    }

    /// Get available choices given current state.
    pub fn available_choices(&self, vars: &StoryVariables) -> Vec<(usize, &Choice)> {
        self.choices.iter().enumerate()
            .filter(|(_, c)| c.is_available(vars))
            .collect()
    }

    /// Get displayable content lines given current state.
    pub fn displayable_content(&self, vars: &StoryVariables) -> Vec<&ContentLine> {
        self.content.iter()
            .filter(|line| line.should_display(vars))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Knot
// ---------------------------------------------------------------------------

/// A major section of a story (chapter, scene, conversation topic).
#[derive(Debug, Clone)]
pub struct Knot {
    /// Name of this knot.
    pub name: String,
    /// Stitches within this knot.
    pub stitches: HashMap<String, Stitch>,
    /// Name of the default stitch (first stitch to enter).
    pub default_stitch: String,
    /// Tags on this knot.
    pub tags: Vec<String>,
    /// Actions to execute when entering this knot.
    pub on_enter: Vec<InkAction>,
    /// Condition for this knot to be accessible.
    pub condition: Option<InkCondition>,
}

impl Knot {
    /// Create a new knot.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name,
            stitches: HashMap::new(),
            default_stitch: "default".to_string(),
            tags: Vec::new(),
            on_enter: Vec::new(),
            condition: None,
        }
    }

    /// Add a stitch to this knot.
    pub fn add_stitch(&mut self, stitch: Stitch) {
        if self.stitches.is_empty() {
            self.default_stitch = stitch.name.clone();
        }
        self.stitches.insert(stitch.name.clone(), stitch);
    }

    /// Get a stitch by name.
    pub fn get_stitch(&self, name: &str) -> Option<&Stitch> {
        self.stitches.get(name)
    }

    /// Get the default stitch.
    pub fn get_default_stitch(&self) -> Option<&Stitch> {
        self.stitches.get(&self.default_stitch)
    }
}

// ---------------------------------------------------------------------------
// Story
// ---------------------------------------------------------------------------

/// A complete dialogue story containing knots and global variables.
#[derive(Debug, Clone)]
pub struct Story {
    /// Story identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Knots in this story.
    pub knots: HashMap<String, Knot>,
    /// Name of the starting knot.
    pub start_knot: String,
    /// Global variables with initial values.
    pub initial_variables: HashMap<String, StoryVariableValue>,
    /// Author information.
    pub author: String,
    /// Version string.
    pub version: String,
    /// Tags for this story (genre, theme, etc.).
    pub tags: Vec<String>,
}

impl Story {
    /// Create a new story.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            knots: HashMap::new(),
            start_knot: "start".to_string(),
            initial_variables: HashMap::new(),
            author: String::new(),
            version: "1.0".to_string(),
            tags: Vec::new(),
        }
    }

    /// Add a knot.
    pub fn add_knot(&mut self, knot: Knot) {
        if self.knots.is_empty() {
            self.start_knot = knot.name.clone();
        }
        self.knots.insert(knot.name.clone(), knot);
    }

    /// Get a knot by name.
    pub fn get_knot(&self, name: &str) -> Option<&Knot> {
        self.knots.get(name)
    }

    /// Resolve a path (e.g., "knot_name.stitch_name") to a knot and stitch.
    pub fn resolve_path(&self, path: &str) -> Option<(&Knot, &Stitch)> {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        let knot_name = parts[0];
        let knot = self.knots.get(knot_name)?;
        let stitch_name = if parts.len() > 1 {
            parts[1]
        } else {
            &knot.default_stitch
        };
        let stitch = knot.stitches.get(stitch_name)?;
        Some((knot, stitch))
    }
}

// ---------------------------------------------------------------------------
// StoryRunnerState
// ---------------------------------------------------------------------------

/// State of the story runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoryRunnerState {
    /// The runner is idle (no active conversation).
    Idle,
    /// The runner is displaying content.
    DisplayingContent,
    /// The runner is waiting for the player to make a choice.
    WaitingForChoice,
    /// The runner is processing actions.
    Processing,
    /// The story has ended.
    Ended,
    /// The runner is paused.
    Paused,
}

// ---------------------------------------------------------------------------
// StoryRunner
// ---------------------------------------------------------------------------

/// Runs a story, managing navigation, state, and event emission.
#[derive(Debug)]
pub struct StoryRunner {
    /// The story being run.
    story: Story,
    /// Variable store.
    pub variables: StoryVariables,
    /// Current state.
    pub state: StoryRunnerState,
    /// Current knot name.
    current_knot: String,
    /// Current stitch name.
    current_stitch: String,
    /// Current line index within the stitch.
    current_line_index: usize,
    /// Events generated this step.
    events: Vec<DialogueEvent>,
    /// NPC ID for this conversation.
    npc_id: String,
    /// History of visited paths.
    history: Vec<String>,
    /// Current available choices (after content is exhausted).
    current_choices: Vec<Choice>,
    /// Whether the content for the current stitch has been fully displayed.
    content_exhausted: bool,
    /// Wait timer (for Wait actions).
    wait_timer: f32,
}

impl StoryRunner {
    /// Create a new story runner.
    pub fn new(story: Story, npc_id: impl Into<String>) -> Self {
        let start_knot = story.start_knot.clone();
        let mut variables = StoryVariables::new();

        // Initialize variables
        for (name, value) in &story.initial_variables {
            variables.set(name.clone(), value.clone());
        }

        Self {
            story,
            variables,
            state: StoryRunnerState::Idle,
            current_knot: start_knot,
            current_stitch: String::new(),
            current_line_index: 0,
            events: Vec::new(),
            npc_id: npc_id.into(),
            history: Vec::new(),
            current_choices: Vec::new(),
            content_exhausted: false,
            wait_timer: 0.0,
        }
    }

    /// Start the conversation.
    pub fn start(&mut self) {
        self.state = StoryRunnerState::Processing;
        let start = self.story.start_knot.clone();
        self.enter_knot(&start);

        self.events.push(DialogueEvent::ConversationStarted {
            npc_id: self.npc_id.clone(),
            story_id: self.story.id.clone(),
        });
    }

    /// Enter a knot.
    fn enter_knot(&mut self, knot_name: &str) {
        if let Some(knot) = self.story.knots.get(knot_name).cloned() {
            self.current_knot = knot_name.to_string();
            self.variables.record_visit(knot_name);
            self.history.push(knot_name.to_string());

            // Execute on_enter actions
            for action in &knot.on_enter {
                if let Some(event) = action.execute(&mut self.variables) {
                    self.events.push(event);
                }
            }

            let default_stitch = knot.default_stitch.clone();
            self.enter_stitch(&default_stitch);
        } else {
            self.state = StoryRunnerState::Ended;
        }
    }

    /// Enter a stitch within the current knot.
    fn enter_stitch(&mut self, stitch_name: &str) {
        self.current_stitch = stitch_name.to_string();
        self.current_line_index = 0;
        self.content_exhausted = false;

        let path = format!("{}.{}", self.current_knot, stitch_name);
        self.variables.record_visit(&path);
        self.history.push(path);

        // Execute on_enter actions for the stitch
        let knot_name = self.current_knot.clone();
        if let Some(knot) = self.story.knots.get(&knot_name) {
            if let Some(stitch) = knot.stitches.get(stitch_name) {
                for action in &stitch.on_enter {
                    if let Some(event) = action.execute(&mut self.variables) {
                        self.events.push(event);
                    }
                }
            }
        }

        self.state = StoryRunnerState::DisplayingContent;
    }

    /// Divert to a path (e.g., "knot_name" or "knot_name.stitch_name").
    pub fn divert_to(&mut self, path: &str) {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        let knot_name = parts[0];

        if knot_name != self.current_knot {
            self.enter_knot(knot_name);
            if parts.len() > 1 {
                self.enter_stitch(parts[1]);
            }
        } else if parts.len() > 1 {
            self.enter_stitch(parts[1]);
        }
    }

    /// Get the next content line to display. Returns None if no more content.
    pub fn next_line(&mut self) -> Option<ContentLine> {
        if self.state != StoryRunnerState::DisplayingContent {
            return None;
        }

        let knot_name = self.current_knot.clone();
        let stitch_name = self.current_stitch.clone();

        let line = {
            let knot = self.story.knots.get(&knot_name)?;
            let stitch = knot.stitches.get(&stitch_name)?;
            let displayable = stitch.displayable_content(&self.variables);

            if self.current_line_index < displayable.len() {
                let line = displayable[self.current_line_index].clone();
                self.current_line_index += 1;
                Some(line)
            } else {
                None
            }
        };

        if let Some(ref line) = line {
            // Execute line actions
            for action in &line.actions {
                if let Some(event) = action.execute(&mut self.variables) {
                    self.events.push(event);
                }
            }

            self.events.push(DialogueEvent::LineDisplayed {
                speaker: line.speaker.clone(),
                text: line.text.clone(),
            });
        } else {
            // Content exhausted, check for choices or divert
            self.content_exhausted = true;
            self.check_post_content();
        }

        line
    }

    /// Check what to do after content is exhausted.
    fn check_post_content(&mut self) {
        let knot_name = self.current_knot.clone();
        let stitch_name = self.current_stitch.clone();

        if let Some(knot) = self.story.knots.get(&knot_name) {
            if let Some(stitch) = knot.stitches.get(&stitch_name) {
                let available = stitch.available_choices(&self.variables);
                if !available.is_empty() {
                    self.current_choices = available.iter()
                        .map(|(_, c)| (*c).clone())
                        .collect();
                    self.state = StoryRunnerState::WaitingForChoice;
                } else if let Some(ref divert) = stitch.divert {
                    let divert = divert.clone();
                    if divert == "END" || divert == "DONE" {
                        self.end_conversation();
                    } else {
                        self.divert_to(&divert);
                    }
                } else {
                    self.end_conversation();
                }
            }
        }
    }

    /// Get the current available choices.
    pub fn choices(&self) -> &[Choice] {
        &self.current_choices
    }

    /// Make a choice (by index into the current choices).
    pub fn choose(&mut self, index: usize) {
        if self.state != StoryRunnerState::WaitingForChoice {
            return;
        }
        if index >= self.current_choices.len() {
            return;
        }

        let choice = self.current_choices[index].clone();

        // Execute choice actions
        for action in &choice.actions {
            if let Some(event) = action.execute(&mut self.variables) {
                self.events.push(event);
            }
        }

        self.events.push(DialogueEvent::ChoiceMade {
            choice_index: index,
            choice_text: choice.text.clone(),
        });

        self.variables.advance_turn();
        self.current_choices.clear();

        // Divert to target
        let target = choice.target.clone();
        if target == "END" || target == "DONE" {
            self.end_conversation();
        } else {
            self.divert_to(&target);
        }
    }

    /// End the conversation.
    pub fn end_conversation(&mut self) {
        self.state = StoryRunnerState::Ended;
        self.events.push(DialogueEvent::ConversationEnded {
            npc_id: self.npc_id.clone(),
            story_id: self.story.id.clone(),
        });
    }

    /// Pause the conversation.
    pub fn pause(&mut self) {
        if self.state != StoryRunnerState::Ended {
            self.state = StoryRunnerState::Paused;
        }
    }

    /// Resume the conversation.
    pub fn resume(&mut self) {
        if self.state == StoryRunnerState::Paused {
            if self.content_exhausted {
                self.check_post_content();
            } else {
                self.state = StoryRunnerState::DisplayingContent;
            }
        }
    }

    /// Drain and return all pending events.
    pub fn drain_events(&mut self) -> Vec<DialogueEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if the conversation is active.
    pub fn is_active(&self) -> bool {
        !matches!(self.state, StoryRunnerState::Idle | StoryRunnerState::Ended)
    }

    /// Get the current path.
    pub fn current_path(&self) -> String {
        format!("{}.{}", self.current_knot, self.current_stitch)
    }

    /// Get the conversation history.
    pub fn history(&self) -> &[String] {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// NPC Schedule
// ---------------------------------------------------------------------------

/// Time of day (24-hour format).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeOfDay {
    /// Hour (0..23).
    pub hour: u8,
    /// Minute (0..59).
    pub minute: u8,
}

impl TimeOfDay {
    /// Create a new time of day.
    pub fn new(hour: u8, minute: u8) -> Self {
        Self {
            hour: hour.min(23),
            minute: minute.min(59),
        }
    }

    /// Convert to minutes since midnight.
    pub fn to_minutes(&self) -> u32 {
        self.hour as u32 * 60 + self.minute as u32
    }

    /// Check if this time is within a range (handles wrapping past midnight).
    pub fn is_within(&self, start: TimeOfDay, end: TimeOfDay) -> bool {
        let t = self.to_minutes();
        let s = start.to_minutes();
        let e = end.to_minutes();

        if s <= e {
            t >= s && t <= e
        } else {
            // Wraps past midnight
            t >= s || t <= e
        }
    }
}

/// Day of the week.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

/// A schedule entry for NPC availability.
#[derive(Debug, Clone)]
pub struct ScheduleEntry {
    /// Days this schedule applies to.
    pub days: HashSet<DayOfWeek>,
    /// Start time.
    pub start_time: TimeOfDay,
    /// End time.
    pub end_time: TimeOfDay,
    /// Location where the NPC is during this time.
    pub location: String,
    /// Activity the NPC is performing.
    pub activity: String,
    /// Whether the NPC is available for dialogue during this time.
    pub available_for_dialogue: bool,
    /// Override story to use during this schedule (if any).
    pub story_override: Option<String>,
}

/// NPC dialogue availability configuration.
#[derive(Debug, Clone)]
pub struct NpcDialogueConfig {
    /// NPC identifier.
    pub npc_id: String,
    /// NPC display name.
    pub display_name: String,
    /// Default story ID for this NPC.
    pub default_story: String,
    /// Schedule entries.
    pub schedule: Vec<ScheduleEntry>,
    /// Interaction range (how close the player must be).
    pub interaction_range: f32,
    /// Whether the NPC is globally available (ignoring schedule).
    pub always_available: bool,
    /// Cooldown between conversations in seconds.
    pub conversation_cooldown: f32,
    /// Time since last conversation.
    pub last_conversation_time: f32,
    /// Conditions for the NPC to be available (beyond schedule).
    pub availability_conditions: Vec<InkCondition>,
}

impl NpcDialogueConfig {
    /// Create a new NPC dialogue config.
    pub fn new(npc_id: impl Into<String>, display_name: impl Into<String>, default_story: impl Into<String>) -> Self {
        Self {
            npc_id: npc_id.into(),
            display_name: display_name.into(),
            default_story: default_story.into(),
            schedule: Vec::new(),
            interaction_range: DEFAULT_INTERACTION_RANGE,
            always_available: false,
            conversation_cooldown: 0.0,
            last_conversation_time: f32::NEG_INFINITY,
            availability_conditions: Vec::new(),
        }
    }

    /// Check if the NPC is available at a given time.
    pub fn is_available(&self, time: TimeOfDay, day: DayOfWeek, current_time: f32, vars: &StoryVariables) -> bool {
        // Check cooldown
        if current_time - self.last_conversation_time < self.conversation_cooldown {
            return false;
        }

        // Check additional conditions
        for condition in &self.availability_conditions {
            if !condition.evaluate(vars) {
                return false;
            }
        }

        if self.always_available {
            return true;
        }

        // Check schedule
        for entry in &self.schedule {
            if entry.days.contains(&day) && time.is_within(entry.start_time, entry.end_time) {
                return entry.available_for_dialogue;
            }
        }

        false
    }

    /// Get the story to use at a given time.
    pub fn story_at(&self, time: TimeOfDay, day: DayOfWeek) -> &str {
        for entry in &self.schedule {
            if entry.days.contains(&day) && time.is_within(entry.start_time, entry.end_time) {
                if let Some(ref override_story) = entry.story_override {
                    return override_story;
                }
            }
        }
        &self.default_story
    }
}

// ---------------------------------------------------------------------------
// Bark system
// ---------------------------------------------------------------------------

/// A bark is a short one-liner that an NPC says without entering full dialogue.
#[derive(Debug, Clone)]
pub struct Bark {
    /// Unique identifier.
    pub id: String,
    /// NPC who says this bark.
    pub npc_id: String,
    /// The text of the bark.
    pub text: String,
    /// Whether the text is a localization key.
    pub is_localized: bool,
    /// Condition for this bark to trigger.
    pub condition: Option<InkCondition>,
    /// Priority (higher = more likely to be selected).
    pub priority: i32,
    /// Cooldown in seconds before this bark can trigger again.
    pub cooldown: f32,
    /// Time remaining on cooldown.
    cooldown_remaining: f32,
    /// Trigger context (what triggers this bark).
    pub trigger: BarkTrigger,
    /// Audio clip reference.
    pub audio_clip: Option<String>,
    /// Maximum times this bark can be played (0 = unlimited).
    pub max_plays: u32,
    /// Number of times this bark has been played.
    play_count: u32,
    /// Actions to execute when this bark plays.
    pub actions: Vec<InkAction>,
}

/// What triggers a bark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarkTrigger {
    /// Player enters proximity of NPC.
    ProximityEnter,
    /// Player is in proximity for a duration (idle bark).
    ProximityIdle,
    /// Player performs an action near the NPC.
    PlayerAction(String),
    /// Time-of-day event (e.g., morning greeting).
    TimeOfDay { hour: u8 },
    /// Combat event (e.g., enemy spotted, taking damage).
    CombatEvent(String),
    /// World event (e.g., weather change, quest completion).
    WorldEvent(String),
    /// Random ambient bark (chance-based each interval).
    Ambient { chance: f32 },
    /// Scripted trigger (from game code).
    Scripted,
}

impl Bark {
    /// Create a new bark.
    pub fn new(id: impl Into<String>, npc_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            npc_id: npc_id.into(),
            text: text.into(),
            is_localized: false,
            condition: None,
            priority: 0,
            cooldown: DEFAULT_BARK_COOLDOWN,
            cooldown_remaining: 0.0,
            trigger: BarkTrigger::Ambient { chance: 0.1 },
            audio_clip: None,
            max_plays: 0,
            play_count: 0,
            actions: Vec::new(),
        }
    }

    /// Check if this bark is ready to play.
    pub fn is_ready(&self, vars: &StoryVariables) -> bool {
        if self.cooldown_remaining > 0.0 {
            return false;
        }
        if self.max_plays > 0 && self.play_count >= self.max_plays {
            return false;
        }
        self.condition.as_ref().map_or(true, |c| c.evaluate(vars))
    }

    /// Mark this bark as played.
    pub fn mark_played(&mut self) {
        self.play_count += 1;
        self.cooldown_remaining = self.cooldown;
    }

    /// Update cooldown.
    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
        }
    }
}

/// Manager for the bark system.
#[derive(Debug)]
pub struct BarkManager {
    /// All registered barks.
    barks: Vec<Bark>,
    /// Currently active barks (npc_id -> bark text + remaining duration).
    active_barks: HashMap<String, ActiveBark>,
    /// Events generated this frame.
    events: Vec<DialogueEvent>,
}

/// An active bark being displayed.
#[derive(Debug, Clone)]
pub struct ActiveBark {
    /// NPC displaying the bark.
    pub npc_id: String,
    /// Bark text.
    pub text: String,
    /// Remaining display duration.
    pub remaining_duration: f32,
    /// Audio clip reference.
    pub audio_clip: Option<String>,
}

impl BarkManager {
    /// Create a new bark manager.
    pub fn new() -> Self {
        Self {
            barks: Vec::new(),
            active_barks: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Register a bark.
    pub fn add_bark(&mut self, bark: Bark) {
        self.barks.push(bark);
    }

    /// Update the bark manager.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();

        // Update bark cooldowns
        for bark in &mut self.barks {
            bark.update(dt);
        }

        // Update active bark durations
        let mut expired = Vec::new();
        for (npc_id, bark) in &mut self.active_barks {
            bark.remaining_duration -= dt;
            if bark.remaining_duration <= 0.0 {
                expired.push(npc_id.clone());
            }
        }
        for npc_id in expired {
            self.active_barks.remove(&npc_id);
        }
    }

    /// Try to trigger barks for a given trigger type.
    pub fn trigger(&mut self, trigger: &BarkTrigger, npc_id: &str, vars: &StoryVariables) {
        // Don't overlap barks from the same NPC
        if self.active_barks.contains_key(npc_id) {
            return;
        }
        if self.active_barks.len() >= MAX_ACTIVE_BARKS {
            return;
        }

        // Find the highest-priority ready bark for this NPC and trigger
        let mut best_index: Option<usize> = None;
        let mut best_priority = i32::MIN;

        for (i, bark) in self.barks.iter().enumerate() {
            if bark.npc_id != npc_id {
                continue;
            }
            if bark.trigger != *trigger {
                continue;
            }
            if !bark.is_ready(vars) {
                continue;
            }
            if bark.priority > best_priority {
                best_priority = bark.priority;
                best_index = Some(i);
            }
        }

        if let Some(index) = best_index {
            let bark = &mut self.barks[index];
            bark.mark_played();

            self.active_barks.insert(npc_id.to_string(), ActiveBark {
                npc_id: npc_id.to_string(),
                text: bark.text.clone(),
                remaining_duration: DEFAULT_BARK_DURATION,
                audio_clip: bark.audio_clip.clone(),
            });

            self.events.push(DialogueEvent::BarkDisplayed {
                npc_id: npc_id.to_string(),
                text: bark.text.clone(),
            });
        }
    }

    /// Get active barks.
    pub fn active_barks(&self) -> &HashMap<String, ActiveBark> {
        &self.active_barks
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<DialogueEvent> {
        std::mem::take(&mut self.events)
    }
}

// ---------------------------------------------------------------------------
// DialogueQueue
// ---------------------------------------------------------------------------

/// Queue for managing multiple conversations in sequence.
#[derive(Debug)]
pub struct DialogueQueue {
    /// Queued conversations.
    queue: VecDeque<QueuedConversation>,
    /// Currently active runner.
    active_runner: Option<StoryRunner>,
}

/// A conversation waiting in the queue.
#[derive(Debug)]
pub struct QueuedConversation {
    /// The story to run.
    pub story: Story,
    /// NPC ID.
    pub npc_id: String,
    /// Priority (higher = moved ahead in queue).
    pub priority: i32,
}

impl DialogueQueue {
    /// Create a new dialogue queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            active_runner: None,
        }
    }

    /// Add a conversation to the queue.
    pub fn enqueue(&mut self, story: Story, npc_id: impl Into<String>, priority: i32) {
        if self.queue.len() >= MAX_QUEUE_DEPTH {
            return;
        }

        let conversation = QueuedConversation {
            story,
            npc_id: npc_id.into(),
            priority,
        };

        // Insert sorted by priority (highest first)
        let pos = self.queue.iter()
            .position(|c| c.priority < priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(pos, conversation);
    }

    /// Get the active story runner.
    pub fn active_runner(&self) -> Option<&StoryRunner> {
        self.active_runner.as_ref()
    }

    /// Get the active story runner mutably.
    pub fn active_runner_mut(&mut self) -> Option<&mut StoryRunner> {
        self.active_runner.as_mut()
    }

    /// Start the next conversation in the queue.
    pub fn start_next(&mut self) -> bool {
        if let Some(conv) = self.queue.pop_front() {
            let mut runner = StoryRunner::new(conv.story, conv.npc_id);
            runner.start();
            self.active_runner = Some(runner);
            true
        } else {
            false
        }
    }

    /// Check if there's an active conversation.
    pub fn has_active(&self) -> bool {
        self.active_runner.as_ref().map_or(false, |r| r.is_active())
    }

    /// Update the queue (auto-advance to next conversation when current ends).
    pub fn update(&mut self) {
        if let Some(ref runner) = self.active_runner {
            if !runner.is_active() {
                self.active_runner = None;
                self.start_next();
            }
        } else {
            self.start_next();
        }
    }

    /// Get the number of queued conversations.
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }

    /// Clear the queue (does not affect active conversation).
    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }
}

// ---------------------------------------------------------------------------
// DialogueV2Component (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities that can engage in dialogue.
#[derive(Debug)]
pub struct DialogueV2Component {
    /// NPC dialogue configuration.
    pub config: NpcDialogueConfig,
    /// Current bark manager for this NPC.
    pub bark_manager: BarkManager,
    /// Whether the NPC has an active conversation indicator.
    pub show_indicator: bool,
    /// Indicator type.
    pub indicator: DialogueIndicator,
}

/// Visual indicator type for NPCs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueIndicator {
    /// No indicator.
    None,
    /// Exclamation mark (new conversation available).
    Exclamation,
    /// Question mark (in-progress quest).
    Question,
    /// Ellipsis (generic interaction available).
    Ellipsis,
    /// Chat bubble.
    ChatBubble,
}

impl DialogueV2Component {
    /// Create a new dialogue component.
    pub fn new(config: NpcDialogueConfig) -> Self {
        Self {
            config,
            bark_manager: BarkManager::new(),
            show_indicator: true,
            indicator: DialogueIndicator::Exclamation,
        }
    }
}

// ---------------------------------------------------------------------------
// DialogueV2System
// ---------------------------------------------------------------------------

/// System managing all dialogue interactions.
pub struct DialogueV2System {
    /// Story library (story_id -> Story).
    stories: HashMap<String, Story>,
    /// Global story variables (shared across all conversations).
    pub global_variables: StoryVariables,
    /// Dialogue queue.
    pub queue: DialogueQueue,
    /// Bark managers (npc_id -> BarkManager).
    bark_managers: HashMap<String, BarkManager>,
    /// Events from last update.
    events: Vec<DialogueEvent>,
}

impl DialogueV2System {
    /// Create a new dialogue system.
    pub fn new() -> Self {
        Self {
            stories: HashMap::new(),
            global_variables: StoryVariables::new(),
            queue: DialogueQueue::new(),
            bark_managers: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Register a story.
    pub fn register_story(&mut self, story: Story) {
        self.stories.insert(story.id.clone(), story);
    }

    /// Get a story by ID.
    pub fn get_story(&self, id: &str) -> Option<&Story> {
        self.stories.get(id)
    }

    /// Start a conversation.
    pub fn start_conversation(&mut self, story_id: &str, npc_id: &str) -> bool {
        if let Some(story) = self.stories.get(story_id).cloned() {
            self.queue.enqueue(story, npc_id, 0);
            true
        } else {
            false
        }
    }

    /// Update the dialogue system.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();
        self.queue.update();

        // Collect events from active runner
        if let Some(runner) = self.queue.active_runner_mut() {
            let events = runner.drain_events();
            self.events.extend(events);
        }

        // Update bark managers
        for manager in self.bark_managers.values_mut() {
            manager.update(dt);
            let events = manager.drain_events();
            self.events.extend(events);
        }
    }

    /// Get events from the last update.
    pub fn events(&self) -> &[DialogueEvent] {
        &self.events
    }

    /// Register a bark manager for an NPC.
    pub fn register_bark_manager(&mut self, npc_id: impl Into<String>, manager: BarkManager) {
        self.bark_managers.insert(npc_id.into(), manager);
    }

    /// Trigger barks for a specific NPC.
    pub fn trigger_bark(&mut self, npc_id: &str, trigger: &BarkTrigger) {
        if let Some(manager) = self.bark_managers.get_mut(npc_id) {
            manager.trigger(trigger, npc_id, &self.global_variables);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_story_variables() {
        let mut vars = StoryVariables::new();
        vars.set_int("gold", 100);
        vars.set_bool("quest_accepted", true);
        vars.set_string("player_name", "Hero");

        assert_eq!(vars.get_int("gold"), 100);
        assert!(vars.get_bool("quest_accepted"));
        assert_eq!(vars.get_string("player_name"), "Hero");
        assert_eq!(vars.get_int("nonexistent"), 0);
    }

    #[test]
    fn test_variable_increment() {
        let mut vars = StoryVariables::new();
        vars.set_int("counter", 5);
        vars.increment("counter", 3);
        assert_eq!(vars.get_int("counter"), 8);
    }

    #[test]
    fn test_visit_counts() {
        let mut vars = StoryVariables::new();
        vars.record_visit("intro");
        vars.record_visit("intro");
        assert_eq!(vars.visit_count("intro"), 2);
        assert!(vars.has_visited("intro"));
        assert!(!vars.has_visited("outro"));
    }

    #[test]
    fn test_ink_condition_equals() {
        let vars = {
            let mut v = StoryVariables::new();
            v.set_int("level", 5);
            v
        };
        let cond = InkCondition::GreaterOrEqual {
            variable: "level".to_string(),
            value: 3.0,
        };
        assert!(cond.evaluate(&vars));
    }

    #[test]
    fn test_ink_condition_composite() {
        let vars = {
            let mut v = StoryVariables::new();
            v.set_int("gold", 100);
            v.set_bool("has_key", true);
            v
        };
        let cond = InkCondition::And(vec![
            InkCondition::GreaterOrEqual { variable: "gold".to_string(), value: 50.0 },
            InkCondition::Truthy("has_key".to_string()),
        ]);
        assert!(cond.evaluate(&vars));
    }

    #[test]
    fn test_choice_availability() {
        let vars = StoryVariables::new();
        let mut choice = Choice::new("Test", "target");
        assert!(choice.is_available(&vars));
        choice.selected = true;
        assert!(!choice.is_available(&vars));
    }

    #[test]
    fn test_time_of_day() {
        let morning = TimeOfDay::new(8, 0);
        let noon = TimeOfDay::new(12, 0);
        let evening = TimeOfDay::new(20, 0);
        let start = TimeOfDay::new(9, 0);
        let end = TimeOfDay::new(17, 0);

        assert!(!morning.is_within(start, end));
        assert!(noon.is_within(start, end));
        assert!(!evening.is_within(start, end));
    }

    #[test]
    fn test_story_path_resolution() {
        let mut story = Story::new("test", "Test Story");
        let mut knot = Knot::new("greeting");
        let stitch = Stitch::new("hello");
        knot.add_stitch(stitch);
        story.add_knot(knot);

        assert!(story.resolve_path("greeting.hello").is_some());
        assert!(story.resolve_path("greeting").is_some());
        assert!(story.resolve_path("nonexistent").is_none());
    }
}
