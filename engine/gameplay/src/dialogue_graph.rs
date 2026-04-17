// engine/gameplay/src/dialogue_graph.rs
//
// Visual dialogue graph system for the Genovo engine.
//
// Provides a node-graph based dialogue authoring and runtime system:
//
// - Node types: text, choice, condition, action, random, hub.
// - Edge-based flow between nodes with conditions.
// - Graph evaluation with backtracking and history.
// - Variable substitution in dialogue text.
// - Localization key support for all text content.
// - Dialogue events for game system integration.
// - Graph validation and cycle detection.
// - Serialization-friendly graph structure.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum dialogue choices presented at once.
const MAX_CHOICES: usize = 8;

/// Maximum variable substitution depth (to prevent infinite loops).
const MAX_SUBSTITUTION_DEPTH: u32 = 10;

/// Maximum dialogue graph traversal depth.
const MAX_TRAVERSAL_DEPTH: u32 = 1000;

/// Default typing speed (characters per second for display).
const DEFAULT_TYPING_SPEED: f32 = 30.0;

// ---------------------------------------------------------------------------
// Node ID / Edge ID
// ---------------------------------------------------------------------------

/// Unique identifier for a dialogue graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogueNodeId(pub u32);

impl DialogueNodeId {
    /// Invalid/null node.
    pub const NONE: Self = Self(u32::MAX);

    /// Check if this is a valid node ID.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Unique identifier for a dialogue edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogueEdgeId(pub u32);

// ---------------------------------------------------------------------------
// Variable System
// ---------------------------------------------------------------------------

/// A dialogue variable value.
#[derive(Debug, Clone)]
pub enum DialogueValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
}

impl DialogueValue {
    /// Convert to boolean.
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::String(s) => !s.is_empty(),
        }
    }

    /// Convert to integer.
    pub fn as_int(&self) -> i64 {
        match self {
            Self::Bool(b) => if *b { 1 } else { 0 },
            Self::Int(i) => *i,
            Self::Float(f) => *f as i64,
            Self::String(s) => s.parse().unwrap_or(0),
        }
    }

    /// Convert to float.
    pub fn as_float(&self) -> f64 {
        match self {
            Self::Bool(b) => if *b { 1.0 } else { 0.0 },
            Self::Int(i) => *i as f64,
            Self::Float(f) => *f,
            Self::String(s) => s.parse().unwrap_or(0.0),
        }
    }

    /// Convert to string.
    pub fn as_string(&self) -> String {
        match self {
            Self::Bool(b) => b.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{:.2}", f),
            Self::String(s) => s.clone(),
        }
    }
}

/// Variable store for dialogue evaluation.
#[derive(Debug, Clone)]
pub struct DialogueVariables {
    /// Variable storage.
    pub variables: HashMap<String, DialogueValue>,
}

impl DialogueVariables {
    /// Create an empty variable store.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Set a variable.
    pub fn set(&mut self, name: &str, value: DialogueValue) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable.
    pub fn get(&self, name: &str) -> Option<&DialogueValue> {
        self.variables.get(name)
    }

    /// Get a boolean variable with a default.
    pub fn get_bool(&self, name: &str, default: bool) -> bool {
        self.variables.get(name).map(|v| v.as_bool()).unwrap_or(default)
    }

    /// Get an integer variable with a default.
    pub fn get_int(&self, name: &str, default: i64) -> i64 {
        self.variables.get(name).map(|v| v.as_int()).unwrap_or(default)
    }

    /// Get a string variable with a default.
    pub fn get_string(&self, name: &str, default: &str) -> String {
        self.variables.get(name).map(|v| v.as_string()).unwrap_or_else(|| default.to_string())
    }

    /// Increment an integer variable.
    pub fn increment(&mut self, name: &str, amount: i64) {
        let current = self.get_int(name, 0);
        self.set(name, DialogueValue::Int(current + amount));
    }

    /// Substitute variables in a text string. Variables are denoted by {variable_name}.
    pub fn substitute(&self, text: &str) -> String {
        self.substitute_recursive(text, 0)
    }

    fn substitute_recursive(&self, text: &str, depth: u32) -> String {
        if depth >= MAX_SUBSTITUTION_DEPTH {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                let mut var_name = String::new();
                let mut found_end = false;
                for next_ch in chars.by_ref() {
                    if next_ch == '}' {
                        found_end = true;
                        break;
                    }
                    var_name.push(next_ch);
                }

                if found_end {
                    if let Some(value) = self.get(&var_name) {
                        let substituted = value.as_string();
                        result.push_str(&self.substitute_recursive(&substituted, depth + 1));
                    } else {
                        result.push('{');
                        result.push_str(&var_name);
                        result.push('}');
                    }
                } else {
                    result.push('{');
                    result.push_str(&var_name);
                }
            } else {
                result.push(ch);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

/// A condition that can be evaluated to determine graph flow.
#[derive(Debug, Clone)]
pub enum DialogueCondition {
    /// Variable equals a value.
    Equals { variable: String, value: DialogueValue },
    /// Variable is greater than a value.
    GreaterThan { variable: String, value: DialogueValue },
    /// Variable is less than a value.
    LessThan { variable: String, value: DialogueValue },
    /// Variable is greater than or equal to a value.
    GreaterOrEqual { variable: String, value: DialogueValue },
    /// Variable is less than or equal to a value.
    LessOrEqual { variable: String, value: DialogueValue },
    /// Variable exists (is set).
    Exists { variable: String },
    /// Boolean AND of two conditions.
    And(Box<DialogueCondition>, Box<DialogueCondition>),
    /// Boolean OR of two conditions.
    Or(Box<DialogueCondition>, Box<DialogueCondition>),
    /// Boolean NOT of a condition.
    Not(Box<DialogueCondition>),
    /// Always true.
    Always,
    /// Always false.
    Never,
    /// Custom condition evaluated by game code.
    Custom { name: String, params: Vec<String> },
}

impl DialogueCondition {
    /// Evaluate this condition against the given variables.
    pub fn evaluate(&self, vars: &DialogueVariables) -> bool {
        match self {
            Self::Equals { variable, value } => {
                vars.get(variable).map_or(false, |v| v.as_string() == value.as_string())
            }
            Self::GreaterThan { variable, value } => {
                vars.get(variable).map_or(false, |v| v.as_float() > value.as_float())
            }
            Self::LessThan { variable, value } => {
                vars.get(variable).map_or(false, |v| v.as_float() < value.as_float())
            }
            Self::GreaterOrEqual { variable, value } => {
                vars.get(variable).map_or(false, |v| v.as_float() >= value.as_float())
            }
            Self::LessOrEqual { variable, value } => {
                vars.get(variable).map_or(false, |v| v.as_float() <= value.as_float())
            }
            Self::Exists { variable } => vars.get(variable).is_some(),
            Self::And(a, b) => a.evaluate(vars) && b.evaluate(vars),
            Self::Or(a, b) => a.evaluate(vars) || b.evaluate(vars),
            Self::Not(c) => !c.evaluate(vars),
            Self::Always => true,
            Self::Never => false,
            Self::Custom { .. } => true, // Default: custom conditions pass.
        }
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// An action triggered when entering/exiting a node.
#[derive(Debug, Clone)]
pub enum DialogueAction {
    /// Set a variable.
    SetVariable { variable: String, value: DialogueValue },
    /// Increment a variable.
    IncrementVariable { variable: String, amount: i64 },
    /// Fire a game event.
    FireEvent { event_name: String, params: HashMap<String, String> },
    /// Play an audio clip.
    PlayAudio { clip_id: String },
    /// Start a camera animation.
    CameraAnimation { animation_id: String },
    /// Give item to player.
    GiveItem { item_id: String, count: u32 },
    /// Remove item from player.
    RemoveItem { item_id: String, count: u32 },
    /// Start a quest.
    StartQuest { quest_id: String },
    /// Complete a quest objective.
    CompleteObjective { quest_id: String, objective_id: String },
    /// Modify reputation.
    ModifyReputation { faction: String, amount: i32 },
    /// Custom action.
    Custom { name: String, params: Vec<String> },
}

impl DialogueAction {
    /// Execute this action on the variable store (for variable-related actions).
    pub fn execute(&self, vars: &mut DialogueVariables) {
        match self {
            Self::SetVariable { variable, value } => {
                vars.set(variable, value.clone());
            }
            Self::IncrementVariable { variable, amount } => {
                vars.increment(variable, *amount);
            }
            _ => {} // Other actions require game system integration.
        }
    }
}

// ---------------------------------------------------------------------------
// Node Types
// ---------------------------------------------------------------------------

/// A choice within a choice node.
#[derive(Debug, Clone)]
pub struct DialogueChoice {
    /// Display text for this choice.
    pub text: String,
    /// Localization key for the text.
    pub loc_key: Option<String>,
    /// Condition that must be met for this choice to appear.
    pub condition: Option<DialogueCondition>,
    /// Target node when this choice is selected.
    pub target: DialogueNodeId,
    /// Whether this choice has been previously selected.
    pub visited: bool,
    /// Actions to execute when this choice is selected.
    pub actions: Vec<DialogueAction>,
    /// Whether this choice is only shown once.
    pub one_time: bool,
}

/// Speaker emotion for a text node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionTag {
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Fearful,
    Disgusted,
    Thoughtful,
    Excited,
    Worried,
}

/// A dialogue graph node.
#[derive(Debug, Clone)]
pub struct DialogueNode {
    /// Node identifier.
    pub id: DialogueNodeId,
    /// Node type and data.
    pub node_type: DialogueNodeType,
    /// Actions executed on entering this node.
    pub on_enter_actions: Vec<DialogueAction>,
    /// Actions executed on leaving this node.
    pub on_exit_actions: Vec<DialogueAction>,
    /// Editor position (for visual graph editor).
    pub editor_position: [f32; 2],
    /// Comment/note for the author.
    pub comment: String,
    /// Tags for filtering/searching.
    pub tags: Vec<String>,
}

/// Type-specific data for dialogue nodes.
#[derive(Debug, Clone)]
pub enum DialogueNodeType {
    /// Displays text from a speaker.
    Text {
        speaker: String,
        text: String,
        loc_key: Option<String>,
        emotion: EmotionTag,
        duration: Option<f32>,
        voice_clip: Option<String>,
        next: DialogueNodeId,
    },
    /// Presents choices to the player.
    Choice {
        prompt: String,
        prompt_loc_key: Option<String>,
        choices: Vec<DialogueChoice>,
    },
    /// Conditional branching.
    Condition {
        condition: DialogueCondition,
        true_branch: DialogueNodeId,
        false_branch: DialogueNodeId,
    },
    /// Execute actions and continue.
    Action {
        actions: Vec<DialogueAction>,
        next: DialogueNodeId,
    },
    /// Random branching (picks one of the targets).
    Random {
        targets: Vec<(DialogueNodeId, f32)>, // (target, weight)
    },
    /// Hub node: a central point that multiple paths converge to.
    Hub {
        label: String,
        next: DialogueNodeId,
    },
    /// Start node (entry point).
    Start {
        label: String,
        next: DialogueNodeId,
    },
    /// End node (terminates the dialogue).
    End {
        result: DialogueResult,
    },
}

/// Result of completing a dialogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialogueResult {
    /// Normal completion.
    Complete,
    /// Aborted by the player.
    Aborted,
    /// Failed (e.g., skill check failed).
    Failed,
    /// Success (e.g., persuasion succeeded).
    Success,
    /// Custom result code.
    Custom(u32),
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// An edge connecting two nodes in the dialogue graph.
#[derive(Debug, Clone)]
pub struct DialogueEdge {
    /// Edge identifier.
    pub id: DialogueEdgeId,
    /// Source node.
    pub from: DialogueNodeId,
    /// Target node.
    pub to: DialogueNodeId,
    /// Condition for traversing this edge.
    pub condition: Option<DialogueCondition>,
    /// Priority (higher priority edges are checked first).
    pub priority: i32,
}

// ---------------------------------------------------------------------------
// Dialogue Graph
// ---------------------------------------------------------------------------

/// A complete dialogue graph with nodes and edges.
#[derive(Debug, Clone)]
pub struct DialogueGraph {
    /// Graph name/identifier.
    pub name: String,
    /// Localization table name.
    pub localization_table: String,
    /// All nodes in the graph.
    pub nodes: HashMap<DialogueNodeId, DialogueNode>,
    /// All edges in the graph.
    pub edges: Vec<DialogueEdge>,
    /// Entry node IDs (can have multiple entry points).
    pub entry_nodes: Vec<DialogueNodeId>,
    /// Default entry node.
    pub default_entry: DialogueNodeId,
    /// Next node ID to assign.
    next_node_id: u32,
    /// Next edge ID to assign.
    next_edge_id: u32,
}

impl DialogueGraph {
    /// Create a new empty dialogue graph.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            localization_table: String::new(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_nodes: Vec::new(),
            default_entry: DialogueNodeId::NONE,
            next_node_id: 0,
            next_edge_id: 0,
        }
    }

    /// Add a node to the graph. Returns the node ID.
    pub fn add_node(&mut self, node_type: DialogueNodeType) -> DialogueNodeId {
        let id = DialogueNodeId(self.next_node_id);
        self.next_node_id += 1;

        let node = DialogueNode {
            id,
            node_type,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
            editor_position: [0.0, 0.0],
            comment: String::new(),
            tags: Vec::new(),
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(
        &mut self,
        from: DialogueNodeId,
        to: DialogueNodeId,
        condition: Option<DialogueCondition>,
    ) -> DialogueEdgeId {
        let id = DialogueEdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        self.edges.push(DialogueEdge {
            id,
            from,
            to,
            condition,
            priority: 0,
        });

        id
    }

    /// Set the default entry node.
    pub fn set_entry(&mut self, node_id: DialogueNodeId) {
        self.default_entry = node_id;
        if !self.entry_nodes.contains(&node_id) {
            self.entry_nodes.push(node_id);
        }
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: DialogueNodeId) -> Option<&DialogueNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID.
    pub fn get_node_mut(&mut self, id: DialogueNodeId) -> Option<&mut DialogueNode> {
        self.nodes.get_mut(&id)
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: DialogueNodeId) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        self.entry_nodes.retain(|&e| e != id);
    }

    /// Get outgoing edges from a node.
    pub fn outgoing_edges(&self, node_id: DialogueNodeId) -> Vec<&DialogueEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Validate the graph structure.
    pub fn validate(&self) -> Vec<GraphError> {
        let mut errors = Vec::new();

        if self.default_entry == DialogueNodeId::NONE {
            errors.push(GraphError::NoEntryNode);
        }

        // Check for unreachable nodes.
        let reachable = self.find_reachable_nodes();
        for (&id, _) in &self.nodes {
            if !reachable.contains(&id) && !self.entry_nodes.contains(&id) {
                errors.push(GraphError::UnreachableNode(id));
            }
        }

        // Check for dangling edges.
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                errors.push(GraphError::DanglingEdge(edge.id, edge.from));
            }
            if !self.nodes.contains_key(&edge.to) {
                errors.push(GraphError::DanglingEdge(edge.id, edge.to));
            }
        }

        errors
    }

    /// Find all nodes reachable from entry nodes.
    fn find_reachable_nodes(&self) -> std::collections::HashSet<DialogueNodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut stack: Vec<DialogueNodeId> = self.entry_nodes.clone();

        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }

            if let Some(node) = self.nodes.get(&node_id) {
                match &node.node_type {
                    DialogueNodeType::Text { next, .. } => {
                        if next.is_valid() { stack.push(*next); }
                    }
                    DialogueNodeType::Choice { choices, .. } => {
                        for choice in choices {
                            if choice.target.is_valid() { stack.push(choice.target); }
                        }
                    }
                    DialogueNodeType::Condition { true_branch, false_branch, .. } => {
                        if true_branch.is_valid() { stack.push(*true_branch); }
                        if false_branch.is_valid() { stack.push(*false_branch); }
                    }
                    DialogueNodeType::Action { next, .. } => {
                        if next.is_valid() { stack.push(*next); }
                    }
                    DialogueNodeType::Random { targets } => {
                        for (target, _) in targets {
                            if target.is_valid() { stack.push(*target); }
                        }
                    }
                    DialogueNodeType::Hub { next, .. } => {
                        if next.is_valid() { stack.push(*next); }
                    }
                    DialogueNodeType::Start { next, .. } => {
                        if next.is_valid() { stack.push(*next); }
                    }
                    DialogueNodeType::End { .. } => {}
                }
            }

            for edge in &self.edges {
                if edge.from == node_id && !visited.contains(&edge.to) {
                    stack.push(edge.to);
                }
            }
        }

        visited
    }

    /// Node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// Graph validation errors.
#[derive(Debug, Clone)]
pub enum GraphError {
    /// No entry node defined.
    NoEntryNode,
    /// A node is not reachable from any entry point.
    UnreachableNode(DialogueNodeId),
    /// An edge references a non-existent node.
    DanglingEdge(DialogueEdgeId, DialogueNodeId),
    /// The graph contains a cycle (may or may not be an error).
    Cycle(Vec<DialogueNodeId>),
}

// ---------------------------------------------------------------------------
// Graph Runner
// ---------------------------------------------------------------------------

/// State of the dialogue graph runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRunnerState {
    /// Not started.
    Idle,
    /// Displaying text, waiting to advance.
    ShowingText,
    /// Waiting for player choice.
    WaitingForChoice,
    /// Processing (evaluating conditions, executing actions).
    Processing,
    /// Dialogue complete.
    Finished,
}

/// Dialogue graph runner -- evaluates the graph at runtime.
#[derive(Debug)]
pub struct DialogueGraphRunner {
    /// The graph being run.
    pub graph: DialogueGraph,
    /// Current node.
    pub current_node: DialogueNodeId,
    /// Current state.
    pub state: GraphRunnerState,
    /// Variables.
    pub variables: DialogueVariables,
    /// History of visited nodes.
    pub history: Vec<DialogueNodeId>,
    /// Available choices (when in WaitingForChoice state).
    pub available_choices: Vec<(usize, String)>,
    /// Current display text.
    pub current_text: String,
    /// Current speaker.
    pub current_speaker: String,
    /// Current emotion.
    pub current_emotion: EmotionTag,
    /// Events generated during evaluation.
    pub events: Vec<DialogueGraphEvent>,
    /// Result when finished.
    pub result: Option<DialogueResult>,
    /// Traversal depth counter (for infinite loop prevention).
    traversal_depth: u32,
}

/// Events emitted by the graph runner.
#[derive(Debug, Clone)]
pub enum DialogueGraphEvent {
    /// A node was entered.
    NodeEntered(DialogueNodeId),
    /// Text is being displayed.
    TextDisplayed { speaker: String, text: String, emotion: EmotionTag },
    /// Choices are being presented.
    ChoicesPresented(Vec<String>),
    /// A choice was made.
    ChoiceMade { index: usize, text: String },
    /// An action was triggered.
    ActionTriggered(DialogueAction),
    /// Dialogue ended.
    DialogueEnded(DialogueResult),
}

impl DialogueGraphRunner {
    /// Create a new graph runner.
    pub fn new(graph: DialogueGraph) -> Self {
        let entry = graph.default_entry;
        Self {
            graph,
            current_node: entry,
            state: GraphRunnerState::Idle,
            variables: DialogueVariables::new(),
            history: Vec::new(),
            available_choices: Vec::new(),
            current_text: String::new(),
            current_speaker: String::new(),
            current_emotion: EmotionTag::Neutral,
            events: Vec::new(),
            result: None,
            traversal_depth: 0,
        }
    }

    /// Start the dialogue.
    pub fn start(&mut self) {
        self.state = GraphRunnerState::Processing;
        self.traverse_to(self.graph.default_entry);
    }

    /// Select a choice (when in WaitingForChoice state).
    pub fn select_choice(&mut self, choice_index: usize) {
        if self.state != GraphRunnerState::WaitingForChoice {
            return;
        }

        if let Some(node) = self.graph.nodes.get(&self.current_node) {
            if let DialogueNodeType::Choice { choices, .. } = &node.node_type {
                let visible: Vec<_> = choices.iter().enumerate()
                    .filter(|(_, c)| c.condition.as_ref().map_or(true, |cond| cond.evaluate(&self.variables)))
                    .collect();

                if let Some((_, choice)) = visible.get(choice_index) {
                    let target = choice.target;
                    let text = choice.text.clone();

                    // Execute choice actions.
                    for action in &choice.actions {
                        action.execute(&mut self.variables);
                        self.events.push(DialogueGraphEvent::ActionTriggered(action.clone()));
                    }

                    self.events.push(DialogueGraphEvent::ChoiceMade {
                        index: choice_index,
                        text,
                    });

                    self.traverse_to(target);
                }
            }
        }
    }

    /// Advance past the current text node.
    pub fn advance(&mut self) {
        if self.state != GraphRunnerState::ShowingText {
            return;
        }

        if let Some(node) = self.graph.nodes.get(&self.current_node) {
            if let DialogueNodeType::Text { next, .. } = &node.node_type {
                let next = *next;
                self.traverse_to(next);
            }
        }
    }

    /// Traverse to a specific node, processing automatic nodes.
    fn traverse_to(&mut self, node_id: DialogueNodeId) {
        self.traversal_depth += 1;
        if self.traversal_depth > MAX_TRAVERSAL_DEPTH {
            self.state = GraphRunnerState::Finished;
            self.result = Some(DialogueResult::Failed);
            return;
        }

        if !node_id.is_valid() {
            self.state = GraphRunnerState::Finished;
            self.result = Some(DialogueResult::Complete);
            self.events.push(DialogueGraphEvent::DialogueEnded(DialogueResult::Complete));
            return;
        }

        self.current_node = node_id;
        self.history.push(node_id);
        self.events.push(DialogueGraphEvent::NodeEntered(node_id));

        // Execute on_enter actions.
        if let Some(node) = self.graph.nodes.get(&node_id) {
            for action in &node.on_enter_actions {
                action.execute(&mut self.variables);
            }
        }

        // Process the node based on type.
        let node = match self.graph.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => {
                self.state = GraphRunnerState::Finished;
                self.result = Some(DialogueResult::Failed);
                return;
            }
        };

        match &node.node_type {
            DialogueNodeType::Text { speaker, text, emotion, .. } => {
                let substituted = self.variables.substitute(text);
                self.current_speaker = speaker.clone();
                self.current_text = substituted.clone();
                self.current_emotion = *emotion;
                self.state = GraphRunnerState::ShowingText;
                self.events.push(DialogueGraphEvent::TextDisplayed {
                    speaker: speaker.clone(),
                    text: substituted,
                    emotion: *emotion,
                });
            }
            DialogueNodeType::Choice { prompt, choices, .. } => {
                let substituted_prompt = self.variables.substitute(prompt);
                self.current_text = substituted_prompt;
                self.available_choices.clear();

                let visible: Vec<_> = choices.iter().enumerate()
                    .filter(|(_, c)| {
                        if c.one_time && c.visited {
                            return false;
                        }
                        c.condition.as_ref().map_or(true, |cond| cond.evaluate(&self.variables))
                    })
                    .collect();

                for (i, (_, choice)) in visible.iter().enumerate() {
                    let text = self.variables.substitute(&choice.text);
                    self.available_choices.push((i, text));
                }

                self.state = GraphRunnerState::WaitingForChoice;
                let choice_texts: Vec<String> = self.available_choices.iter().map(|(_, t)| t.clone()).collect();
                self.events.push(DialogueGraphEvent::ChoicesPresented(choice_texts));
            }
            DialogueNodeType::Condition { condition, true_branch, false_branch } => {
                let result = condition.evaluate(&self.variables);
                let target = if result { *true_branch } else { *false_branch };
                self.traverse_to(target);
            }
            DialogueNodeType::Action { actions, next } => {
                for action in actions {
                    action.execute(&mut self.variables);
                    self.events.push(DialogueGraphEvent::ActionTriggered(action.clone()));
                }
                let next = *next;
                self.traverse_to(next);
            }
            DialogueNodeType::Random { targets } => {
                let total_weight: f32 = targets.iter().map(|(_, w)| w).sum();
                if total_weight > 0.0 && !targets.is_empty() {
                    // Simple deterministic "random" based on history length.
                    let pseudo_random = (self.history.len() as f32 * 0.618033988) % 1.0;
                    let threshold = pseudo_random * total_weight;
                    let mut cumulative = 0.0f32;
                    let mut chosen = targets[0].0;
                    for (target, weight) in targets {
                        cumulative += weight;
                        if cumulative >= threshold {
                            chosen = *target;
                            break;
                        }
                    }
                    self.traverse_to(chosen);
                }
            }
            DialogueNodeType::Hub { next, .. } | DialogueNodeType::Start { next, .. } => {
                let next = *next;
                self.traverse_to(next);
            }
            DialogueNodeType::End { result } => {
                self.state = GraphRunnerState::Finished;
                self.result = Some(*result);
                self.events.push(DialogueGraphEvent::DialogueEnded(*result));
            }
        }
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<DialogueGraphEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if the dialogue is finished.
    pub fn is_finished(&self) -> bool {
        self.state == GraphRunnerState::Finished
    }
}
