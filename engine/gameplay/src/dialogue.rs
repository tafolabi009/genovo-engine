//! Branching dialogue system with conditions and consequences.
//!
//! Provides a graph-based dialogue tree suitable for RPGs, adventure games,
//! and interactive fiction. Features include:
//!
//! - Branching conversation graphs with multiple choices
//! - Condition-filtered dialogue options
//! - Consequences that fire when nodes are visited (set flags, give items, etc.)
//! - Speaker portraits and emotion tags
//! - Localization-ready text keys
//! - A [`DialogueRunner`] that manages conversation flow
//!
//! # Data model
//!
//! A [`DialogueTree`] is a directed graph of [`DialogueNode`]s connected by
//! [`DialogueChoice`]s. Each choice can have [`DialogueCondition`]s that
//! determine visibility, and each node can trigger [`DialogueConsequence`]s
//! when visited.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Dialogue condition
// ---------------------------------------------------------------------------

/// A condition that must be met for a dialogue choice to be available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DialogueCondition {
    /// A named flag must be true.
    FlagSet(String),
    /// A named flag must NOT be set.
    FlagNotSet(String),
    /// A quest must be in a specific state.
    QuestState {
        quest_id: String,
        state: String,
    },
    /// The player must have at least N of an item.
    HasItem {
        item_id: String,
        quantity: u32,
    },
    /// A numeric variable must satisfy a comparison.
    Variable {
        name: String,
        op: ComparisonOp,
        value: i32,
    },
    /// The player must have at least this level.
    MinLevel(u32),
    /// The player must have a specific ability or skill.
    HasAbility(String),
    /// Reputation with a faction must be at or above a threshold.
    MinReputation {
        faction: String,
        threshold: i32,
    },
    /// All of the sub-conditions must be true.
    All(Vec<DialogueCondition>),
    /// At least one of the sub-conditions must be true.
    Any(Vec<DialogueCondition>),
    /// Negation of a condition.
    Not(Box<DialogueCondition>),
    /// Custom condition evaluated by game code via a string key.
    Custom {
        key: String,
        parameter: String,
    },
}

/// Comparison operators for variable conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
}

impl ComparisonOp {
    /// Evaluate the comparison.
    pub fn evaluate(&self, lhs: i32, rhs: i32) -> bool {
        match self {
            Self::Equal => lhs == rhs,
            Self::NotEqual => lhs != rhs,
            Self::GreaterThan => lhs > rhs,
            Self::GreaterOrEqual => lhs >= rhs,
            Self::LessThan => lhs < rhs,
            Self::LessOrEqual => lhs <= rhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Dialogue consequence
// ---------------------------------------------------------------------------

/// An effect triggered when a dialogue node is visited or a choice is selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DialogueConsequence {
    /// Set a flag to true.
    SetFlag(String),
    /// Clear (unset) a flag.
    ClearFlag(String),
    /// Set a numeric variable.
    SetVariable {
        name: String,
        value: i32,
    },
    /// Add to a numeric variable.
    AddVariable {
        name: String,
        amount: i32,
    },
    /// Give an item to the player.
    GiveItem {
        item_id: String,
        quantity: u32,
    },
    /// Remove an item from the player.
    TakeItem {
        item_id: String,
        quantity: u32,
    },
    /// Start a quest.
    StartQuest(String),
    /// Complete a quest.
    CompleteQuest(String),
    /// Change reputation with a faction.
    ChangeReputation {
        faction: String,
        amount: i32,
    },
    /// Give XP.
    GiveXp(u32),
    /// Trigger a game event by key.
    TriggerEvent(String),
    /// Play an animation on the speaker.
    PlayAnimation {
        target: String,
        animation: String,
    },
    /// Spawn an entity (e.g., an enemy appears).
    SpawnEntity {
        entity_type: String,
        position: [f32; 3],
    },
    /// Teleport the player.
    Teleport {
        position: [f32; 3],
    },
    /// Custom consequence handled by game code.
    Custom {
        key: String,
        parameter: String,
    },
}

// ---------------------------------------------------------------------------
// Dialogue choice
// ---------------------------------------------------------------------------

/// A player-selectable choice within a dialogue node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueChoice {
    /// Unique id of this choice within the node.
    pub choice_id: String,
    /// Display text for this choice.
    pub text: String,
    /// Optional localization key.
    pub text_key: Option<String>,
    /// The node to transition to when this choice is selected.
    pub next_node: String,
    /// Conditions that must be met for this choice to appear.
    pub conditions: Vec<DialogueCondition>,
    /// Consequences triggered when this choice is selected.
    pub consequences: Vec<DialogueConsequence>,
    /// Whether this choice has been selected before (for display hints).
    #[serde(default)]
    pub visited: bool,
    /// Skill check associated with this choice (if any).
    pub skill_check: Option<SkillCheck>,
    /// Tooltip or additional info shown on hover.
    pub tooltip: Option<String>,
}

impl DialogueChoice {
    /// Create a simple choice.
    pub fn new(
        choice_id: impl Into<String>,
        text: impl Into<String>,
        next_node: impl Into<String>,
    ) -> Self {
        Self {
            choice_id: choice_id.into(),
            text: text.into(),
            text_key: None,
            next_node: next_node.into(),
            conditions: Vec::new(),
            consequences: Vec::new(),
            visited: false,
            skill_check: None,
            tooltip: None,
        }
    }

    /// Add a condition.
    pub fn with_condition(mut self, condition: DialogueCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Add a consequence.
    pub fn with_consequence(mut self, consequence: DialogueConsequence) -> Self {
        self.consequences.push(consequence);
        self
    }

    /// Set a skill check.
    pub fn with_skill_check(mut self, check: SkillCheck) -> Self {
        self.skill_check = Some(check);
        self
    }
}

/// A skill check attached to a dialogue choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCheck {
    /// Skill name.
    pub skill: String,
    /// Required skill level / difficulty.
    pub difficulty: u32,
    /// Text shown if the check succeeds.
    pub success_text: String,
    /// Text shown if the check fails.
    pub failure_text: String,
    /// Node to go to on failure (if different from the choice's next_node).
    pub failure_node: Option<String>,
}

// ---------------------------------------------------------------------------
// Dialogue node
// ---------------------------------------------------------------------------

/// Speaker emotion/expression tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeakerEmotion {
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Afraid,
    Disgusted,
    Thoughtful,
    Smug,
    Pleading,
}

impl Default for SpeakerEmotion {
    fn default() -> Self {
        Self::Neutral
    }
}

/// A single node in a dialogue tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueNode {
    /// Unique id within the tree.
    pub node_id: String,
    /// Speaker name/id.
    pub speaker: String,
    /// Speaker portrait/image id.
    pub portrait: Option<String>,
    /// Speaker emotion.
    pub emotion: SpeakerEmotion,
    /// Main text content.
    pub text: String,
    /// Optional localization key for the text.
    pub text_key: Option<String>,
    /// Choices available at this node.
    pub choices: Vec<DialogueChoice>,
    /// Consequences triggered when this node is displayed.
    pub on_enter: Vec<DialogueConsequence>,
    /// Consequences triggered when leaving this node.
    pub on_exit: Vec<DialogueConsequence>,
    /// Whether this is a terminal node (ends the conversation).
    pub is_terminal: bool,
    /// Auto-advance: if true, automatically proceed after a delay
    /// (for non-interactive cutscene-style dialogue).
    pub auto_advance: Option<f32>,
    /// Audio clip id for voice acting.
    pub voice_clip: Option<String>,
    /// Conditions that must be met for this node to be reachable.
    /// (If not met, the dialogue skips to a fallback or ends.)
    pub conditions: Vec<DialogueCondition>,
    /// Fallback node id if conditions are not met.
    pub fallback_node: Option<String>,
}

impl DialogueNode {
    /// Create a new dialogue node.
    pub fn new(
        node_id: impl Into<String>,
        speaker: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            speaker: speaker.into(),
            portrait: None,
            emotion: SpeakerEmotion::Neutral,
            text: text.into(),
            text_key: None,
            choices: Vec::new(),
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            is_terminal: false,
            auto_advance: None,
            voice_clip: None,
            conditions: Vec::new(),
            fallback_node: None,
        }
    }

    /// Mark this node as terminal (ends the conversation).
    pub fn terminal(mut self) -> Self {
        self.is_terminal = true;
        self
    }

    /// Add a choice.
    pub fn with_choice(mut self, choice: DialogueChoice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Add an on-enter consequence.
    pub fn with_on_enter(mut self, consequence: DialogueConsequence) -> Self {
        self.on_enter.push(consequence);
        self
    }

    /// Add an on-exit consequence.
    pub fn with_on_exit(mut self, consequence: DialogueConsequence) -> Self {
        self.on_exit.push(consequence);
        self
    }

    /// Set the emotion.
    pub fn with_emotion(mut self, emotion: SpeakerEmotion) -> Self {
        self.emotion = emotion;
        self
    }

    /// Set auto-advance delay.
    pub fn with_auto_advance(mut self, delay: f32) -> Self {
        self.auto_advance = Some(delay);
        self
    }

    /// Set the portrait.
    pub fn with_portrait(mut self, portrait: impl Into<String>) -> Self {
        self.portrait = Some(portrait.into());
        self
    }

    /// Whether this node has choices (i.e., is not auto-advance or terminal).
    pub fn has_choices(&self) -> bool {
        !self.choices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Dialogue tree
// ---------------------------------------------------------------------------

/// A complete dialogue conversation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTree {
    /// Unique tree identifier.
    pub tree_id: String,
    /// Display name for this conversation.
    pub name: String,
    /// Id of the starting node.
    pub start_node: String,
    /// All nodes in the tree, keyed by node_id.
    pub nodes: HashMap<String, DialogueNode>,
    /// Tags for categorization (e.g., "main_story", "merchant", "tutorial").
    pub tags: Vec<String>,
    /// Whether this tree can only be played once.
    pub one_shot: bool,
    /// Global conditions for the entire tree (must be met to start).
    pub conditions: Vec<DialogueCondition>,
}

impl DialogueTree {
    /// Create a new dialogue tree.
    pub fn new(tree_id: impl Into<String>, start_node: impl Into<String>) -> Self {
        Self {
            tree_id: tree_id.into(),
            name: String::new(),
            start_node: start_node.into(),
            nodes: HashMap::new(),
            tags: Vec::new(),
            one_shot: false,
            conditions: Vec::new(),
        }
    }

    /// Set the tree name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a node.
    pub fn with_node(mut self, node: DialogueNode) -> Self {
        self.nodes.insert(node.node_id.clone(), node);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Get a node by id.
    pub fn get_node(&self, node_id: &str) -> Option<&DialogueNode> {
        self.nodes.get(node_id)
    }

    /// Get the start node.
    pub fn start(&self) -> Option<&DialogueNode> {
        self.nodes.get(&self.start_node)
    }

    /// Validate the tree: check that all choice targets exist.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if !self.nodes.contains_key(&self.start_node) {
            errors.push(format!(
                "Start node '{}' not found in tree '{}'",
                self.start_node, self.tree_id
            ));
        }

        for (node_id, node) in &self.nodes {
            for choice in &node.choices {
                if !self.nodes.contains_key(&choice.next_node) {
                    errors.push(format!(
                        "Choice '{}' in node '{}' references missing node '{}'",
                        choice.choice_id, node_id, choice.next_node
                    ));
                }
            }
            if let Some(fallback) = &node.fallback_node {
                if !self.nodes.contains_key(fallback) {
                    errors.push(format!(
                        "Node '{}' fallback references missing node '{}'",
                        node_id, fallback
                    ));
                }
            }
        }

        errors
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Dialogue context (world state for condition evaluation)
// ---------------------------------------------------------------------------

/// World-state interface for evaluating dialogue conditions.
///
/// Implement this trait to connect the dialogue system to your game state.
pub trait DialogueContext {
    /// Check if a flag is set.
    fn has_flag(&self, flag: &str) -> bool;

    /// Get a numeric variable value.
    fn get_variable(&self, name: &str) -> i32;

    /// Get the player's level.
    fn player_level(&self) -> u32;

    /// Check if the player has an ability.
    fn has_ability(&self, ability: &str) -> bool;

    /// Check if the player has an item.
    fn has_item(&self, item_id: &str, quantity: u32) -> bool;

    /// Get quest state as a string.
    fn quest_state(&self, quest_id: &str) -> String;

    /// Get reputation with a faction.
    fn reputation(&self, faction: &str) -> i32;

    /// Evaluate a custom condition.
    fn evaluate_custom(&self, key: &str, parameter: &str) -> bool;
}

/// Simple in-memory dialogue context for testing and simple games.
#[derive(Debug, Clone, Default)]
pub struct SimpleDialogueContext {
    /// Boolean flags.
    pub flags: HashSet<String>,
    /// Numeric variables.
    pub variables: HashMap<String, i32>,
    /// Player level.
    pub level: u32,
    /// Player abilities.
    pub abilities: HashSet<String>,
    /// Player items (id -> quantity).
    pub items: HashMap<String, u32>,
    /// Quest states (id -> state string).
    pub quest_states: HashMap<String, String>,
    /// Faction reputations.
    pub reputations: HashMap<String, i32>,
}

impl SimpleDialogueContext {
    /// Create a new context.
    pub fn new() -> Self {
        Self {
            level: 1,
            ..Default::default()
        }
    }

    /// Set a flag.
    pub fn set_flag(&mut self, flag: impl Into<String>) {
        self.flags.insert(flag.into());
    }

    /// Clear a flag.
    pub fn clear_flag(&mut self, flag: &str) {
        self.flags.remove(flag);
    }

    /// Set a variable.
    pub fn set_variable(&mut self, name: impl Into<String>, value: i32) {
        self.variables.insert(name.into(), value);
    }

    /// Add an item.
    pub fn add_item(&mut self, item_id: impl Into<String>, quantity: u32) {
        let entry = self.items.entry(item_id.into()).or_insert(0);
        *entry += quantity;
    }
}

impl DialogueContext for SimpleDialogueContext {
    fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    fn get_variable(&self, name: &str) -> i32 {
        *self.variables.get(name).unwrap_or(&0)
    }

    fn player_level(&self) -> u32 {
        self.level
    }

    fn has_ability(&self, ability: &str) -> bool {
        self.abilities.contains(ability)
    }

    fn has_item(&self, item_id: &str, quantity: u32) -> bool {
        self.items.get(item_id).copied().unwrap_or(0) >= quantity
    }

    fn quest_state(&self, quest_id: &str) -> String {
        self.quest_states
            .get(quest_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn reputation(&self, faction: &str) -> i32 {
        *self.reputations.get(faction).unwrap_or(&0)
    }

    fn evaluate_custom(&self, _key: &str, _parameter: &str) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Condition evaluator
// ---------------------------------------------------------------------------

/// Evaluate a dialogue condition against a context.
pub fn evaluate_condition(condition: &DialogueCondition, context: &dyn DialogueContext) -> bool {
    match condition {
        DialogueCondition::FlagSet(flag) => context.has_flag(flag),
        DialogueCondition::FlagNotSet(flag) => !context.has_flag(flag),
        DialogueCondition::QuestState { quest_id, state } => {
            context.quest_state(quest_id) == *state
        }
        DialogueCondition::HasItem { item_id, quantity } => {
            context.has_item(item_id, *quantity)
        }
        DialogueCondition::Variable { name, op, value } => {
            let actual = context.get_variable(name);
            op.evaluate(actual, *value)
        }
        DialogueCondition::MinLevel(level) => context.player_level() >= *level,
        DialogueCondition::HasAbility(ability) => context.has_ability(ability),
        DialogueCondition::MinReputation {
            faction,
            threshold,
        } => context.reputation(faction) >= *threshold,
        DialogueCondition::All(conditions) => {
            conditions.iter().all(|c| evaluate_condition(c, context))
        }
        DialogueCondition::Any(conditions) => {
            conditions.iter().any(|c| evaluate_condition(c, context))
        }
        DialogueCondition::Not(inner) => !evaluate_condition(inner, context),
        DialogueCondition::Custom { key, parameter } => {
            context.evaluate_custom(key, parameter)
        }
    }
}

// ---------------------------------------------------------------------------
// Dialogue runner
// ---------------------------------------------------------------------------

/// State of the dialogue runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueRunnerState {
    /// Not running any dialogue.
    Idle,
    /// Showing a node and waiting for player input.
    WaitingForChoice,
    /// Auto-advancing (cutscene mode).
    AutoAdvancing,
    /// Conversation has ended.
    Finished,
}

/// Runtime dialogue processor that manages conversation flow.
///
/// Feed it a [`DialogueTree`] and a [`DialogueContext`], then call
/// `advance()` or `select_choice()` to progress through the conversation.
pub struct DialogueRunner {
    /// Current tree being executed.
    tree: Option<DialogueTree>,
    /// Current node id.
    current_node_id: Option<String>,
    /// State.
    state: DialogueRunnerState,
    /// Visited node ids (for tracking).
    visited_nodes: HashSet<String>,
    /// Consequences accumulated during the conversation (to be processed
    /// by game code).
    pending_consequences: Vec<DialogueConsequence>,
    /// Auto-advance timer.
    auto_advance_timer: f32,
    /// History of visited nodes (for back-tracking / review).
    history: Vec<String>,
}

impl DialogueRunner {
    /// Create a new idle dialogue runner.
    pub fn new() -> Self {
        Self {
            tree: None,
            current_node_id: None,
            state: DialogueRunnerState::Idle,
            visited_nodes: HashSet::new(),
            pending_consequences: Vec::new(),
            auto_advance_timer: 0.0,
            history: Vec::new(),
        }
    }

    /// Start a dialogue tree.
    pub fn start(&mut self, tree: DialogueTree, context: &dyn DialogueContext) -> bool {
        // Check global conditions.
        for condition in &tree.conditions {
            if !evaluate_condition(condition, context) {
                log::debug!(
                    "Dialogue tree '{}' conditions not met",
                    tree.tree_id
                );
                return false;
            }
        }

        let start_id = tree.start_node.clone();
        self.tree = Some(tree);
        self.visited_nodes.clear();
        self.pending_consequences.clear();
        self.history.clear();
        self.enter_node(&start_id, context);
        true
    }

    /// Enter a node: process on_enter consequences, determine state.
    fn enter_node(&mut self, node_id: &str, context: &dyn DialogueContext) {
        let tree = match &self.tree {
            Some(t) => t,
            None => return,
        };

        let node = match tree.get_node(node_id) {
            Some(n) => n,
            None => {
                log::warn!("Dialogue node '{}' not found", node_id);
                self.state = DialogueRunnerState::Finished;
                return;
            }
        };

        // Check node conditions.
        if !node.conditions.is_empty() {
            let all_met = node
                .conditions
                .iter()
                .all(|c| evaluate_condition(c, context));
            if !all_met {
                if let Some(fallback) = &node.fallback_node {
                    let fb = fallback.clone();
                    self.enter_node(&fb, context);
                    return;
                }
                self.state = DialogueRunnerState::Finished;
                return;
            }
        }

        // Track visited nodes.
        self.visited_nodes.insert(node_id.to_string());
        self.history.push(node_id.to_string());
        self.current_node_id = Some(node_id.to_string());

        // Process on_enter consequences.
        self.pending_consequences
            .extend(node.on_enter.iter().cloned());

        // Determine state.
        if node.is_terminal {
            self.pending_consequences
                .extend(node.on_exit.iter().cloned());
            self.state = DialogueRunnerState::Finished;
        } else if let Some(delay) = node.auto_advance {
            self.auto_advance_timer = delay;
            self.state = DialogueRunnerState::AutoAdvancing;
        } else {
            self.state = DialogueRunnerState::WaitingForChoice;
        }

        log::trace!(
            "Entered dialogue node '{}' (speaker: {})",
            node_id,
            node.speaker
        );
    }

    /// Get the currently displayed node.
    pub fn current_node(&self) -> Option<&DialogueNode> {
        let tree = self.tree.as_ref()?;
        let node_id = self.current_node_id.as_ref()?;
        tree.get_node(node_id)
    }

    /// Get the available choices for the current node, filtered by conditions.
    pub fn available_choices(&self, context: &dyn DialogueContext) -> Vec<&DialogueChoice> {
        let Some(node) = self.current_node() else {
            return Vec::new();
        };

        node.choices
            .iter()
            .filter(|choice| {
                choice
                    .conditions
                    .iter()
                    .all(|c| evaluate_condition(c, context))
            })
            .collect()
    }

    /// Select a choice by index (among available/filtered choices).
    pub fn select_choice(
        &mut self,
        choice_index: usize,
        context: &dyn DialogueContext,
    ) -> bool {
        if self.state != DialogueRunnerState::WaitingForChoice {
            return false;
        }

        let available = self.available_choices(context);
        let choice = match available.get(choice_index) {
            Some(c) => (*c).clone(),
            None => return false,
        };

        // Process choice consequences.
        self.pending_consequences
            .extend(choice.consequences.iter().cloned());

        // Process on_exit of the current node.
        let on_exit: Vec<_> = self
            .current_node()
            .map(|n| n.on_exit.clone())
            .unwrap_or_default();
        self.pending_consequences.extend(on_exit);

        // Mark choice as visited in the tree.
        if let Some(tree) = &mut self.tree {
            if let Some(node_id) = &self.current_node_id {
                if let Some(node) = tree.nodes.get_mut(node_id) {
                    if let Some(c) = node
                        .choices
                        .iter_mut()
                        .find(|c| c.choice_id == choice.choice_id)
                    {
                        c.visited = true;
                    }
                }
            }
        }

        // Advance to next node.
        let next = choice.next_node.clone();
        self.enter_node(&next, context);
        true
    }

    /// Advance auto-advancing dialogue. Call each frame with dt.
    pub fn update(&mut self, dt: f32, context: &dyn DialogueContext) {
        if self.state != DialogueRunnerState::AutoAdvancing {
            return;
        }

        self.auto_advance_timer -= dt;
        if self.auto_advance_timer <= 0.0 {
            // Find the first choice (auto-advance nodes typically have exactly one).
            let next = {
                let available = self.available_choices(context);
                available.first().map(|c| c.next_node.clone())
            };

            // Collect on_exit before mutating.
            let on_exit: Vec<_> = self
                .current_node()
                .map(|n| n.on_exit.clone())
                .unwrap_or_default();
            self.pending_consequences.extend(on_exit);

            if let Some(next_node) = next {
                self.enter_node(&next_node, context);
            } else {
                // No choices -- end dialogue.
                self.state = DialogueRunnerState::Finished;
            }
        }
    }

    /// Drain pending consequences (game code should process these).
    pub fn drain_consequences(&mut self) -> Vec<DialogueConsequence> {
        std::mem::take(&mut self.pending_consequences)
    }

    /// Current runner state.
    #[inline]
    pub fn state(&self) -> DialogueRunnerState {
        self.state
    }

    /// Whether the dialogue is running.
    #[inline]
    pub fn is_running(&self) -> bool {
        matches!(
            self.state,
            DialogueRunnerState::WaitingForChoice | DialogueRunnerState::AutoAdvancing
        )
    }

    /// Whether the dialogue has finished.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.state == DialogueRunnerState::Finished
    }

    /// Stop the dialogue immediately.
    pub fn stop(&mut self) {
        self.state = DialogueRunnerState::Finished;
    }

    /// Reset to idle state.
    pub fn reset(&mut self) {
        self.tree = None;
        self.current_node_id = None;
        self.state = DialogueRunnerState::Idle;
        self.visited_nodes.clear();
        self.pending_consequences.clear();
        self.auto_advance_timer = 0.0;
        self.history.clear();
    }

    /// Check if a node has been visited.
    pub fn was_visited(&self, node_id: &str) -> bool {
        self.visited_nodes.contains(node_id)
    }

    /// Get the conversation history (ordered list of visited node ids).
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Get the current tree id, if any.
    pub fn tree_id(&self) -> Option<&str> {
        self.tree.as_ref().map(|t| t.tree_id.as_str())
    }
}

impl Default for DialogueRunner {
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

    fn sample_tree() -> DialogueTree {
        DialogueTree::new("test_dialogue", "greeting")
            .with_name("Test Dialogue")
            .with_node(
                DialogueNode::new("greeting", "Elder", "Hello, traveler! What brings you here?")
                    .with_choice(
                        DialogueChoice::new("c1", "I'm looking for adventure.", "quest_offer"),
                    )
                    .with_choice(
                        DialogueChoice::new("c2", "Just passing through.", "farewell"),
                    )
                    .with_choice(
                        DialogueChoice::new("c3", "I have the ancient relic.", "relic_branch")
                            .with_condition(DialogueCondition::HasItem {
                                item_id: "ancient_relic".to_string(),
                                quantity: 1,
                            }),
                    ),
            )
            .with_node(
                DialogueNode::new(
                    "quest_offer",
                    "Elder",
                    "Excellent! I have a task for you. Slay the dragon in the mountains.",
                )
                .with_choice(DialogueChoice::new("c4", "I accept!", "quest_accepted"))
                .with_choice(DialogueChoice::new(
                    "c5",
                    "That sounds dangerous...",
                    "farewell",
                ))
                .with_on_enter(DialogueConsequence::SetFlag("talked_about_quest".into())),
            )
            .with_node(
                DialogueNode::new(
                    "quest_accepted",
                    "Elder",
                    "May the gods protect you. Return when the deed is done.",
                )
                .with_on_enter(DialogueConsequence::StartQuest("slay_dragon".into()))
                .with_on_enter(DialogueConsequence::SetFlag("quest_accepted".into()))
                .terminal(),
            )
            .with_node(
                DialogueNode::new(
                    "relic_branch",
                    "Elder",
                    "The ancient relic! You've found it! This changes everything.",
                )
                .with_on_enter(DialogueConsequence::TakeItem {
                    item_id: "ancient_relic".into(),
                    quantity: 1,
                })
                .with_on_enter(DialogueConsequence::GiveXp(500))
                .terminal(),
            )
            .with_node(
                DialogueNode::new("farewell", "Elder", "Safe travels, friend.")
                    .terminal(),
            )
    }

    #[test]
    fn tree_validation() {
        let tree = sample_tree();
        let errors = tree.validate();
        assert!(
            errors.is_empty(),
            "Tree should be valid: {:?}",
            errors
        );
    }

    #[test]
    fn start_dialogue() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();

        assert!(runner.start(tree, &context));
        assert!(runner.is_running());

        let node = runner.current_node().unwrap();
        assert_eq!(node.speaker, "Elder");
        assert!(node.text.contains("Hello"));
    }

    #[test]
    fn choices_filtered_by_condition() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        // Without the relic, only 2 choices should be available.
        let choices = runner.available_choices(&context);
        assert_eq!(choices.len(), 2);
    }

    #[test]
    fn choices_with_condition_met() {
        let tree = sample_tree();
        let mut context = SimpleDialogueContext::new();
        context.add_item("ancient_relic", 1);

        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        // With the relic, all 3 choices should be available.
        let choices = runner.available_choices(&context);
        assert_eq!(choices.len(), 3);
    }

    #[test]
    fn select_choice_advances() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        // Select "I'm looking for adventure."
        assert!(runner.select_choice(0, &context));

        let node = runner.current_node().unwrap();
        assert_eq!(node.node_id, "quest_offer");
    }

    #[test]
    fn terminal_node_finishes() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        // Go to farewell.
        runner.select_choice(1, &context);
        assert!(runner.is_finished());
    }

    #[test]
    fn consequences_collected() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        // Navigate to quest_accepted.
        runner.select_choice(0, &context); // -> quest_offer
        runner.select_choice(0, &context); // -> quest_accepted

        let consequences = runner.drain_consequences();
        assert!(
            consequences
                .iter()
                .any(|c| matches!(c, DialogueConsequence::StartQuest(q) if q == "slay_dragon")),
            "Should have StartQuest consequence"
        );
        assert!(
            consequences
                .iter()
                .any(|c| matches!(c, DialogueConsequence::SetFlag(f) if f == "quest_accepted")),
            "Should have SetFlag consequence"
        );
    }

    #[test]
    fn visited_tracking() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        assert!(runner.was_visited("greeting"));
        assert!(!runner.was_visited("quest_offer"));

        runner.select_choice(0, &context);
        assert!(runner.was_visited("quest_offer"));

        assert_eq!(runner.history().len(), 2);
    }

    #[test]
    fn evaluate_flag_conditions() {
        let mut ctx = SimpleDialogueContext::new();
        let cond = DialogueCondition::FlagSet("test_flag".to_string());

        assert!(!evaluate_condition(&cond, &ctx));
        ctx.set_flag("test_flag");
        assert!(evaluate_condition(&cond, &ctx));
    }

    #[test]
    fn evaluate_variable_conditions() {
        let mut ctx = SimpleDialogueContext::new();
        ctx.set_variable("strength", 15);

        let cond = DialogueCondition::Variable {
            name: "strength".to_string(),
            op: ComparisonOp::GreaterOrEqual,
            value: 10,
        };
        assert!(evaluate_condition(&cond, &ctx));

        let cond2 = DialogueCondition::Variable {
            name: "strength".to_string(),
            op: ComparisonOp::GreaterOrEqual,
            value: 20,
        };
        assert!(!evaluate_condition(&cond2, &ctx));
    }

    #[test]
    fn evaluate_composite_conditions() {
        let mut ctx = SimpleDialogueContext::new();
        ctx.set_flag("flag_a");
        ctx.level = 10;

        let cond = DialogueCondition::All(vec![
            DialogueCondition::FlagSet("flag_a".to_string()),
            DialogueCondition::MinLevel(5),
        ]);
        assert!(evaluate_condition(&cond, &ctx));

        let cond2 = DialogueCondition::Any(vec![
            DialogueCondition::FlagSet("flag_b".to_string()),
            DialogueCondition::MinLevel(5),
        ]);
        assert!(evaluate_condition(&cond2, &ctx));

        let cond3 = DialogueCondition::Not(Box::new(DialogueCondition::FlagSet(
            "nonexistent".to_string(),
        )));
        assert!(evaluate_condition(&cond3, &ctx));
    }

    #[test]
    fn auto_advance_dialogue() {
        let tree = DialogueTree::new("auto", "n1")
            .with_node(
                DialogueNode::new("n1", "NPC", "First line.")
                    .with_auto_advance(0.5)
                    .with_choice(DialogueChoice::new("c1", "", "n2")),
            )
            .with_node(
                DialogueNode::new("n2", "NPC", "Second line.").terminal(),
            );

        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        assert_eq!(runner.state(), DialogueRunnerState::AutoAdvancing);

        // Not enough time.
        runner.update(0.3, &context);
        assert_eq!(runner.state(), DialogueRunnerState::AutoAdvancing);

        // Enough time.
        runner.update(0.3, &context);
        assert!(runner.is_finished());
    }

    #[test]
    fn comparison_operators() {
        assert!(ComparisonOp::Equal.evaluate(5, 5));
        assert!(!ComparisonOp::Equal.evaluate(5, 3));
        assert!(ComparisonOp::NotEqual.evaluate(5, 3));
        assert!(ComparisonOp::GreaterThan.evaluate(5, 3));
        assert!(!ComparisonOp::GreaterThan.evaluate(3, 5));
        assert!(ComparisonOp::LessThan.evaluate(3, 5));
        assert!(ComparisonOp::GreaterOrEqual.evaluate(5, 5));
        assert!(ComparisonOp::LessOrEqual.evaluate(5, 5));
    }

    #[test]
    fn invalid_tree_detected() {
        let tree = DialogueTree::new("bad", "start")
            .with_node(
                DialogueNode::new("start", "NPC", "Hello")
                    .with_choice(DialogueChoice::new("c1", "Go", "nonexistent")),
            );

        let errors = tree.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn runner_reset() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        runner.reset();
        assert_eq!(runner.state(), DialogueRunnerState::Idle);
        assert!(!runner.is_running());
        assert!(runner.current_node().is_none());
    }

    #[test]
    fn stop_dialogue() {
        let tree = sample_tree();
        let context = SimpleDialogueContext::new();
        let mut runner = DialogueRunner::new();
        runner.start(tree, &context);

        runner.stop();
        assert!(runner.is_finished());
    }
}
