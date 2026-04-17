// engine/ai/src/dialogue_manager.rs
//
// Dialogue management for the Genovo engine.
//
// Manages the lifecycle of in-game dialogues:
//
// - **Dialogue queue** -- Queue dialogues for sequential playback.
// - **Simultaneous conversations** -- Multiple dialogues at once.
// - **Interruption priority** -- Higher-priority dialogues interrupt lower ones.
// - **Dialogue cooldown** -- Prevent the same dialogue from replaying too soon.
// - **Subtitled/voiced/bark modes** -- Different presentation styles.
// - **Dialogue events** -- Trigger gameplay actions from dialogue nodes.
// - **Conversation state machine** -- Enter/exit/update per node.
// - **Branching dialogue** -- Condition evaluation (variables, inventory, reputation).
// - **Variable substitution** -- {player_name}, {item_count} in text.
// - **Dialogue history** -- Remember what was said, avoid repetition.
// - **NPC mood** -- Mood changes affecting available responses.
// - **Skill checks** -- Bartering, persuasion, intimidation.
// - **Typewriter reveal** -- Character-by-character text animation.
// - **Voice timing** -- Synchronize text reveal with voice line duration.
// - **Localization** -- Resolve localization keys to translated strings.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConversationId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogueNodeId(pub u32);

impl fmt::Display for DialogueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dialogue({})", self.0)
    }
}

impl fmt::Display for DialogueNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Dialogue mode
// ---------------------------------------------------------------------------

/// How the dialogue is presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialogueMode {
    /// Full dialogue with subtitle UI and camera focus.
    Cinematic,
    /// Subtitles only (no camera change).
    Subtitled,
    /// Voiced with subtitles.
    VoicedSubtitled,
    /// Short bark (spoken text above NPC head).
    Bark,
    /// Radio/comm channel dialogue.
    Radio,
    /// Internal thought (italicized, no speaker visible).
    Thought,
}

impl Default for DialogueMode {
    fn default() -> Self {
        Self::Subtitled
    }
}

// ---------------------------------------------------------------------------
// Condition system
// ---------------------------------------------------------------------------

/// A condition that can be evaluated against the game state.
#[derive(Debug, Clone)]
pub enum DialogueCondition {
    /// Check if a variable equals a value.
    VariableEquals { name: String, value: DialogueValue },
    /// Check if a variable is greater than a value.
    VariableGreaterThan { name: String, value: f32 },
    /// Check if a variable is less than a value.
    VariableLessThan { name: String, value: f32 },
    /// Check if a flag is set.
    FlagSet { flag: String },
    /// Check if player has an item (and optionally a minimum count).
    HasItem { item_id: String, min_count: u32 },
    /// Check if player's reputation with a faction meets a threshold.
    ReputationAtLeast { faction: String, min_value: f32 },
    /// Check if a dialogue has been completed before.
    DialogueCompleted { dialogue_id: DialogueId },
    /// Check if the NPC mood is in a certain range.
    MoodInRange { mood_type: NpcMoodType, min: f32, max: f32 },
    /// Check player skill level.
    SkillAtLeast { skill: SkillCheckType, min_level: u32 },
    /// Logical AND of multiple conditions.
    And(Vec<DialogueCondition>),
    /// Logical OR of multiple conditions.
    Or(Vec<DialogueCondition>),
    /// Logical NOT of a condition.
    Not(Box<DialogueCondition>),
    /// Always true.
    Always,
    /// Always false.
    Never,
    /// Check if a specific dialogue node was visited.
    NodeVisited { dialogue_id: DialogueId, node_id: DialogueNodeId },
    /// Check elapsed time since a dialogue was last played.
    TimeSince { dialogue_id: DialogueId, min_seconds: f32 },
}

/// A value that can be stored in dialogue variables.
#[derive(Debug, Clone, PartialEq)]
pub enum DialogueValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

impl DialogueValue {
    pub fn as_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Integer(i) => i.to_string(),
            Self::Float(f) => format!("{:.1}", f),
            Self::Bool(b) => b.to_string(),
        }
    }

    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Float(f) => *f as f32,
            Self::Integer(i) => *i as f32,
            Self::Bool(b) => if *b { 1.0 } else { 0.0 },
            Self::String(s) => s.parse::<f32>().unwrap_or(0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Dialogue variables (game state for condition evaluation)
// ---------------------------------------------------------------------------

/// Holds all variables, flags, and inventory data for condition evaluation.
#[derive(Debug, Clone)]
pub struct DialogueContext {
    /// Named variables.
    pub variables: HashMap<String, DialogueValue>,
    /// Boolean flags.
    pub flags: HashSet<String>,
    /// Player inventory: item_id -> count.
    pub inventory: HashMap<String, u32>,
    /// Faction reputations: faction -> value.
    pub reputations: HashMap<String, f32>,
    /// Completed dialogues.
    pub completed_dialogues: HashSet<DialogueId>,
    /// Player skill levels.
    pub skills: HashMap<SkillCheckType, u32>,
    /// Visited nodes: (dialogue_id, node_id).
    pub visited_nodes: HashSet<(DialogueId, DialogueNodeId)>,
    /// Dialogue completion timestamps.
    pub dialogue_timestamps: HashMap<DialogueId, f64>,
    /// Current game time.
    pub game_time: f64,
}

impl DialogueContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            flags: HashSet::new(),
            inventory: HashMap::new(),
            reputations: HashMap::new(),
            completed_dialogues: HashSet::new(),
            skills: HashMap::new(),
            visited_nodes: HashSet::new(),
            dialogue_timestamps: HashMap::new(),
            game_time: 0.0,
        }
    }

    /// Set a variable.
    pub fn set_variable(&mut self, name: impl Into<String>, value: DialogueValue) {
        self.variables.insert(name.into(), value);
    }

    /// Get a variable value as a string for substitution.
    pub fn get_variable_string(&self, name: &str) -> String {
        self.variables.get(name)
            .map(|v| v.as_string())
            .unwrap_or_default()
    }

    /// Set a flag.
    pub fn set_flag(&mut self, flag: impl Into<String>) {
        self.flags.insert(flag.into());
    }

    /// Clear a flag.
    pub fn clear_flag(&mut self, flag: &str) {
        self.flags.remove(flag);
    }

    /// Evaluate a condition against this context.
    pub fn evaluate(&self, condition: &DialogueCondition, npc_mood: &NpcMood) -> bool {
        match condition {
            DialogueCondition::VariableEquals { name, value } => {
                self.variables.get(name).map(|v| v == value).unwrap_or(false)
            }
            DialogueCondition::VariableGreaterThan { name, value } => {
                self.variables.get(name).map(|v| v.as_f32() > *value).unwrap_or(false)
            }
            DialogueCondition::VariableLessThan { name, value } => {
                self.variables.get(name).map(|v| v.as_f32() < *value).unwrap_or(false)
            }
            DialogueCondition::FlagSet { flag } => {
                self.flags.contains(flag)
            }
            DialogueCondition::HasItem { item_id, min_count } => {
                self.inventory.get(item_id).copied().unwrap_or(0) >= *min_count
            }
            DialogueCondition::ReputationAtLeast { faction, min_value } => {
                self.reputations.get(faction).copied().unwrap_or(0.0) >= *min_value
            }
            DialogueCondition::DialogueCompleted { dialogue_id } => {
                self.completed_dialogues.contains(dialogue_id)
            }
            DialogueCondition::MoodInRange { mood_type, min, max } => {
                let value = npc_mood.get(*mood_type);
                value >= *min && value <= *max
            }
            DialogueCondition::SkillAtLeast { skill, min_level } => {
                self.skills.get(skill).copied().unwrap_or(0) >= *min_level
            }
            DialogueCondition::And(conditions) => {
                conditions.iter().all(|c| self.evaluate(c, npc_mood))
            }
            DialogueCondition::Or(conditions) => {
                conditions.iter().any(|c| self.evaluate(c, npc_mood))
            }
            DialogueCondition::Not(condition) => {
                !self.evaluate(condition, npc_mood)
            }
            DialogueCondition::Always => true,
            DialogueCondition::Never => false,
            DialogueCondition::NodeVisited { dialogue_id, node_id } => {
                self.visited_nodes.contains(&(*dialogue_id, *node_id))
            }
            DialogueCondition::TimeSince { dialogue_id, min_seconds } => {
                match self.dialogue_timestamps.get(dialogue_id) {
                    Some(&timestamp) => (self.game_time - timestamp) as f32 >= *min_seconds,
                    None => true, // Never played counts as infinite time since.
                }
            }
        }
    }
}

impl Default for DialogueContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NPC Mood system
// ---------------------------------------------------------------------------

/// Types of NPC mood axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpcMoodType {
    Friendliness,
    Anger,
    Fear,
    Trust,
    Patience,
    Greed,
}

/// NPC mood state during a conversation.
#[derive(Debug, Clone)]
pub struct NpcMood {
    /// Current mood values (0.0 to 1.0).
    values: HashMap<NpcMoodType, f32>,
    /// Baseline mood values (what the NPC returns to over time).
    baselines: HashMap<NpcMoodType, f32>,
    /// Decay rate toward baseline.
    decay_rate: f32,
}

impl NpcMood {
    pub fn new() -> Self {
        let mut values = HashMap::new();
        values.insert(NpcMoodType::Friendliness, 0.5);
        values.insert(NpcMoodType::Anger, 0.0);
        values.insert(NpcMoodType::Fear, 0.0);
        values.insert(NpcMoodType::Trust, 0.5);
        values.insert(NpcMoodType::Patience, 1.0);
        values.insert(NpcMoodType::Greed, 0.3);
        Self {
            baselines: values.clone(),
            values,
            decay_rate: 0.05,
        }
    }

    /// Create with custom baseline values.
    pub fn with_baselines(baselines: HashMap<NpcMoodType, f32>) -> Self {
        Self {
            values: baselines.clone(),
            baselines,
            decay_rate: 0.05,
        }
    }

    /// Get a mood value.
    pub fn get(&self, mood_type: NpcMoodType) -> f32 {
        self.values.get(&mood_type).copied().unwrap_or(0.5)
    }

    /// Modify a mood value by a delta.
    pub fn modify(&mut self, mood_type: NpcMoodType, delta: f32) {
        let current = self.values.get(&mood_type).copied().unwrap_or(0.5);
        self.values.insert(mood_type, (current + delta).clamp(0.0, 1.0));
    }

    /// Set a mood value directly.
    pub fn set(&mut self, mood_type: NpcMoodType, value: f32) {
        self.values.insert(mood_type, value.clamp(0.0, 1.0));
    }

    /// Decay all moods toward their baselines.
    pub fn decay(&mut self, dt: f32) {
        for (&mood_type, &baseline) in &self.baselines {
            let current = self.values.get(&mood_type).copied().unwrap_or(0.5);
            let diff = baseline - current;
            let decay = diff * self.decay_rate * dt;
            self.values.insert(mood_type, current + decay);
        }
    }

    /// Get a composite "disposition" score: how likely the NPC is to help.
    pub fn disposition(&self) -> f32 {
        let friendly = self.get(NpcMoodType::Friendliness);
        let trust = self.get(NpcMoodType::Trust);
        let anger = self.get(NpcMoodType::Anger);
        let fear = self.get(NpcMoodType::Fear);
        (friendly * 0.4 + trust * 0.3 - anger * 0.2 - fear * 0.1).clamp(0.0, 1.0)
    }

    /// Whether the NPC is hostile (anger high, friendliness low).
    pub fn is_hostile(&self) -> bool {
        self.get(NpcMoodType::Anger) > 0.7 && self.get(NpcMoodType::Friendliness) < 0.3
    }

    /// Whether the NPC has lost patience.
    pub fn has_lost_patience(&self) -> bool {
        self.get(NpcMoodType::Patience) < 0.1
    }
}

impl Default for NpcMood {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skill checks
// ---------------------------------------------------------------------------

/// Types of skill checks in dialogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkillCheckType {
    Persuasion,
    Intimidation,
    Bartering,
    Deception,
    Knowledge,
    Charm,
}

/// Result of a skill check in dialogue.
#[derive(Debug, Clone)]
pub struct SkillCheckResult {
    pub skill: SkillCheckType,
    pub player_level: u32,
    pub required_level: u32,
    pub mood_modifier: f32,
    pub success: bool,
    pub margin: i32,
    pub critical: bool,
}

/// Perform a skill check, factoring in NPC mood.
pub fn perform_skill_check(
    skill: SkillCheckType,
    player_level: u32,
    required_level: u32,
    npc_mood: &NpcMood,
    rng_value: f32,
) -> SkillCheckResult {
    // Mood modifiers affect effective difficulty.
    let mood_modifier = match skill {
        SkillCheckType::Persuasion => {
            npc_mood.get(NpcMoodType::Friendliness) * 0.3
                + npc_mood.get(NpcMoodType::Trust) * 0.2
                - npc_mood.get(NpcMoodType::Anger) * 0.1
        }
        SkillCheckType::Intimidation => {
            npc_mood.get(NpcMoodType::Fear) * 0.4
                - npc_mood.get(NpcMoodType::Anger) * 0.2
                + (1.0 - npc_mood.get(NpcMoodType::Trust)) * 0.1
        }
        SkillCheckType::Bartering => {
            npc_mood.get(NpcMoodType::Greed) * -0.3
                + npc_mood.get(NpcMoodType::Friendliness) * 0.2
                + npc_mood.get(NpcMoodType::Trust) * 0.2
        }
        SkillCheckType::Deception => {
            (1.0 - npc_mood.get(NpcMoodType::Trust)) * -0.3
                + npc_mood.get(NpcMoodType::Friendliness) * 0.1
        }
        SkillCheckType::Knowledge => {
            0.0 // Knowledge checks are not affected by mood.
        }
        SkillCheckType::Charm => {
            npc_mood.get(NpcMoodType::Friendliness) * 0.3
                - npc_mood.get(NpcMoodType::Anger) * 0.2
        }
    };

    // Effective player level includes mood bonus (converted to levels).
    let bonus_levels = (mood_modifier * 5.0) as i32;
    let effective_level = (player_level as i32 + bonus_levels).max(0) as u32;

    // Add randomness: rng_value in [0,1] maps to [-2, +2] bonus levels.
    let random_bonus = ((rng_value - 0.5) * 4.0) as i32;
    let roll_level = (effective_level as i32 + random_bonus).max(0) as u32;

    let margin = roll_level as i32 - required_level as i32;
    let success = margin >= 0;
    let critical = margin >= 5 || margin <= -5;

    SkillCheckResult {
        skill,
        player_level,
        required_level,
        mood_modifier,
        success,
        margin,
        critical,
    }
}

// ---------------------------------------------------------------------------
// Dialogue line
// ---------------------------------------------------------------------------

/// A single line of dialogue.
#[derive(Debug, Clone)]
pub struct DialogueLine {
    /// Speaker.
    pub speaker: SpeakerId,
    /// Speaker display name.
    pub speaker_name: String,
    /// Text content (may contain {variable} substitution tokens).
    pub text: String,
    /// Localization key (if set, resolves to localized text).
    pub localization_key: Option<String>,
    /// Audio clip identifier (empty = no audio).
    pub audio_clip: String,
    /// Duration to display (seconds, 0 = auto based on text length).
    pub duration: f32,
    /// Animation to play on the speaker.
    pub animation: String,
    /// Emotion/mood of the speaker.
    pub emotion: String,
    /// Camera angle suggestion.
    pub camera: DialogueCamera,
    /// Events to fire when this line starts.
    pub events: Vec<DialogueGameEvent>,
    /// Mood modifications caused by this line.
    pub mood_effects: Vec<(NpcMoodType, f32)>,
    /// Whether this line has a typewriter reveal effect.
    pub typewriter: bool,
    /// Characters per second for typewriter effect.
    pub typewriter_speed: f32,
}

impl DialogueLine {
    pub fn new(speaker: SpeakerId, name: &str, text: &str) -> Self {
        Self {
            speaker,
            speaker_name: name.to_string(),
            text: text.to_string(),
            localization_key: None,
            audio_clip: String::new(),
            duration: 0.0,
            animation: String::new(),
            emotion: String::new(),
            camera: DialogueCamera::Default,
            events: Vec::new(),
            mood_effects: Vec::new(),
            typewriter: true,
            typewriter_speed: 30.0,
        }
    }

    /// Compute display duration based on text length.
    pub fn auto_duration(&self) -> f32 {
        if self.duration > 0.0 {
            self.duration
        } else {
            let word_count = self.text.split_whitespace().count();
            (word_count as f32 * 0.4).max(2.0).min(10.0)
        }
    }

    /// Perform variable substitution on the text.
    pub fn substitute_variables(&self, context: &DialogueContext) -> String {
        let mut result = self.text.clone();
        // Find all {variable} patterns and replace them.
        loop {
            let start = match result.find('{') {
                Some(i) => i,
                None => break,
            };
            let end = match result[start..].find('}') {
                Some(i) => start + i,
                None => break,
            };
            let var_name = &result[start + 1..end];
            let replacement = context.get_variable_string(var_name);
            result = format!("{}{}{}", &result[..start], replacement, &result[end + 1..]);
        }
        result
    }

    /// Compute the typewriter reveal state: how many characters should be visible.
    pub fn typewriter_visible_chars(&self, elapsed: f32) -> usize {
        if !self.typewriter {
            return self.text.len();
        }
        let chars_count = self.text.chars().count();
        let revealed = (elapsed * self.typewriter_speed) as usize;
        revealed.min(chars_count)
    }

    /// Whether the typewriter animation is complete.
    pub fn typewriter_complete(&self, elapsed: f32) -> bool {
        if !self.typewriter {
            return true;
        }
        let chars_count = self.text.chars().count();
        (elapsed * self.typewriter_speed) as usize >= chars_count
    }

    /// Get the text visible so far with typewriter effect.
    pub fn visible_text(&self, elapsed: f32, context: &DialogueContext) -> String {
        let full = self.substitute_variables(context);
        if !self.typewriter {
            return full;
        }
        let chars_count = full.chars().count();
        let visible = ((elapsed * self.typewriter_speed) as usize).min(chars_count);
        full.chars().take(visible).collect()
    }
}

/// Camera behavior during a dialogue line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueCamera {
    Default,
    CloseUp,
    OverShoulder,
    Wide,
    Reaction,
    Custom(u32),
}

/// A gameplay event triggered by dialogue.
#[derive(Debug, Clone)]
pub struct DialogueGameEvent {
    pub event_name: String,
    pub parameters: HashMap<String, String>,
    pub delay: f32,
}

// ---------------------------------------------------------------------------
// Dialogue node (for branching dialogues)
// ---------------------------------------------------------------------------

/// A node in a branching dialogue tree.
#[derive(Debug, Clone)]
pub struct DialogueNode {
    /// Unique ID within the dialogue.
    pub id: DialogueNodeId,
    /// The line spoken at this node.
    pub line: DialogueLine,
    /// Responses/choices available from this node.
    pub responses: Vec<DialogueResponse>,
    /// Condition for this node to be reachable.
    pub condition: DialogueCondition,
    /// Script to execute when entering this node.
    pub on_enter: Vec<DialogueAction>,
    /// Script to execute when exiting this node.
    pub on_exit: Vec<DialogueAction>,
    /// Whether this is a terminal node (conversation ends here if no responses).
    pub is_terminal: bool,
    /// Auto-advance to next node after duration (no player input needed).
    pub auto_advance: Option<DialogueNodeId>,
    /// Tags for filtering/categorizing.
    pub tags: Vec<String>,
}

impl DialogueNode {
    pub fn new(id: DialogueNodeId, line: DialogueLine) -> Self {
        Self {
            id,
            line,
            responses: Vec::new(),
            condition: DialogueCondition::Always,
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            is_terminal: false,
            auto_advance: None,
            tags: Vec::new(),
        }
    }

    /// Get available responses given the current context and NPC mood.
    pub fn available_responses(&self, context: &DialogueContext, mood: &NpcMood) -> Vec<&DialogueResponse> {
        self.responses.iter()
            .filter(|r| context.evaluate(&r.condition, mood))
            .collect()
    }
}

/// A player response/choice at a dialogue node.
#[derive(Debug, Clone)]
pub struct DialogueResponse {
    /// Display text for this response.
    pub text: String,
    /// Localization key.
    pub localization_key: Option<String>,
    /// Condition for this response to be available.
    pub condition: DialogueCondition,
    /// Target node when this response is chosen.
    pub target_node: DialogueNodeId,
    /// Skill check required (if any).
    pub skill_check: Option<(SkillCheckType, u32)>,
    /// Display tag like "[Persuasion 5]" or "[Lie]".
    pub display_tag: Option<String>,
    /// Mood effects when this response is chosen.
    pub mood_effects: Vec<(NpcMoodType, f32)>,
    /// Actions triggered when chosen.
    pub on_select: Vec<DialogueAction>,
    /// Node to go to if skill check fails.
    pub fail_target: Option<DialogueNodeId>,
    /// Whether to show even when conditions aren't met (grayed out).
    pub show_when_unavailable: bool,
    /// Tooltip to show when hovering (optional).
    pub tooltip: Option<String>,
}

impl DialogueResponse {
    pub fn new(text: &str, target: DialogueNodeId) -> Self {
        Self {
            text: text.to_string(),
            localization_key: None,
            condition: DialogueCondition::Always,
            target_node: target,
            skill_check: None,
            display_tag: None,
            mood_effects: Vec::new(),
            on_select: Vec::new(),
            fail_target: None,
            show_when_unavailable: false,
            tooltip: None,
        }
    }

    pub fn with_condition(mut self, condition: DialogueCondition) -> Self {
        self.condition = condition;
        self
    }

    pub fn with_skill_check(mut self, skill: SkillCheckType, level: u32, fail_target: DialogueNodeId) -> Self {
        self.skill_check = Some((skill, level));
        self.fail_target = Some(fail_target);
        self.display_tag = Some(format!("[{:?} {}]", skill, level));
        self
    }
}

/// An action that modifies game state during dialogue.
#[derive(Debug, Clone)]
pub enum DialogueAction {
    /// Set a variable.
    SetVariable { name: String, value: DialogueValue },
    /// Set a flag.
    SetFlag { flag: String },
    /// Clear a flag.
    ClearFlag { flag: String },
    /// Add item to inventory.
    GiveItem { item_id: String, count: u32 },
    /// Remove item from inventory.
    TakeItem { item_id: String, count: u32 },
    /// Modify reputation.
    ModifyReputation { faction: String, delta: f32 },
    /// Modify NPC mood.
    ModifyMood { mood_type: NpcMoodType, delta: f32 },
    /// Fire a game event.
    FireEvent(DialogueGameEvent),
    /// Start a quest.
    StartQuest { quest_id: String },
    /// Complete a quest objective.
    CompleteObjective { quest_id: String, objective_id: String },
    /// Grant experience points.
    GrantXp { amount: u32 },
    /// Teleport player.
    Teleport { location: String },
    /// Play a sound effect.
    PlaySound { sound_id: String },
    /// Change NPC animation.
    PlayAnimation { speaker: SpeakerId, animation: String },
}

/// Execute a dialogue action on the context.
pub fn execute_action(action: &DialogueAction, context: &mut DialogueContext, mood: &mut NpcMood) {
    match action {
        DialogueAction::SetVariable { name, value } => {
            context.set_variable(name.clone(), value.clone());
        }
        DialogueAction::SetFlag { flag } => {
            context.set_flag(flag.clone());
        }
        DialogueAction::ClearFlag { flag } => {
            context.clear_flag(flag);
        }
        DialogueAction::GiveItem { item_id, count } => {
            *context.inventory.entry(item_id.clone()).or_insert(0) += count;
        }
        DialogueAction::TakeItem { item_id, count } => {
            if let Some(current) = context.inventory.get_mut(item_id) {
                *current = current.saturating_sub(*count);
            }
        }
        DialogueAction::ModifyReputation { faction, delta } => {
            let rep = context.reputations.entry(faction.clone()).or_insert(0.0);
            *rep = (*rep + delta).clamp(-100.0, 100.0);
        }
        DialogueAction::ModifyMood { mood_type, delta } => {
            mood.modify(*mood_type, *delta);
        }
        _ => {
            // FireEvent, StartQuest, etc. are handled by the game event system.
            // They generate events that the game processes externally.
        }
    }
}

// ---------------------------------------------------------------------------
// Dialogue history
// ---------------------------------------------------------------------------

/// Tracks dialogue history to avoid repetition and enable callbacks.
#[derive(Debug, Clone)]
pub struct DialogueHistory {
    /// Lines that have been said, keyed by (dialogue_id, node_id).
    said_lines: HashSet<(DialogueId, DialogueNodeId)>,
    /// Full history with timestamps.
    entries: Vec<DialogueHistoryEntry>,
    /// Maximum entries to keep.
    max_entries: usize,
    /// Responses the player has chosen.
    chosen_responses: Vec<(DialogueId, DialogueNodeId, String)>,
}

#[derive(Debug, Clone)]
pub struct DialogueHistoryEntry {
    pub dialogue_id: DialogueId,
    pub node_id: DialogueNodeId,
    pub speaker: SpeakerId,
    pub text: String,
    pub timestamp: f64,
}

impl DialogueHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            said_lines: HashSet::new(),
            entries: Vec::new(),
            max_entries,
            chosen_responses: Vec::new(),
        }
    }

    /// Record a line being said.
    pub fn record_line(
        &mut self,
        dialogue_id: DialogueId,
        node_id: DialogueNodeId,
        speaker: SpeakerId,
        text: &str,
        timestamp: f64,
    ) {
        self.said_lines.insert((dialogue_id, node_id));
        self.entries.push(DialogueHistoryEntry {
            dialogue_id,
            node_id,
            speaker,
            text: text.to_string(),
            timestamp,
        });
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    /// Record a player response choice.
    pub fn record_response(&mut self, dialogue_id: DialogueId, node_id: DialogueNodeId, text: &str) {
        self.chosen_responses.push((dialogue_id, node_id, text.to_string()));
    }

    /// Check if a line has been said before.
    pub fn has_been_said(&self, dialogue_id: DialogueId, node_id: DialogueNodeId) -> bool {
        self.said_lines.contains(&(dialogue_id, node_id))
    }

    /// Get the last N entries.
    pub fn recent(&self, count: usize) -> &[DialogueHistoryEntry] {
        let start = self.entries.len().saturating_sub(count);
        &self.entries[start..]
    }

    /// Count how many times a specific dialogue has been played.
    pub fn times_played(&self, dialogue_id: DialogueId) -> usize {
        self.entries.iter()
            .filter(|e| e.dialogue_id == dialogue_id && e.node_id == DialogueNodeId(0))
            .count()
    }

    /// Get all entries for a specific dialogue.
    pub fn entries_for(&self, dialogue_id: DialogueId) -> Vec<&DialogueHistoryEntry> {
        self.entries.iter().filter(|e| e.dialogue_id == dialogue_id).collect()
    }

    /// Total entries recorded.
    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }
}

impl Default for DialogueHistory {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ---------------------------------------------------------------------------
// Voice timing synchronization
// ---------------------------------------------------------------------------

/// Timing data for synchronizing text with voice lines.
#[derive(Debug, Clone)]
pub struct VoiceTimingData {
    /// Total voice clip duration in seconds.
    pub clip_duration: f32,
    /// Word timestamps: (word_index, start_time, end_time).
    pub word_timings: Vec<(usize, f32, f32)>,
    /// Phoneme timestamps for lip sync.
    pub phoneme_timings: Vec<PhonemeTiming>,
    /// Emphasis markers (for volume/pitch changes).
    pub emphasis_markers: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct PhonemeTiming {
    pub phoneme: String,
    pub start_time: f32,
    pub duration: f32,
}

impl VoiceTimingData {
    pub fn new(clip_duration: f32) -> Self {
        Self {
            clip_duration,
            word_timings: Vec::new(),
            phoneme_timings: Vec::new(),
            emphasis_markers: Vec::new(),
        }
    }

    /// Get how many words should be visible at a given playback time.
    pub fn visible_words_at(&self, time: f32) -> usize {
        self.word_timings.iter()
            .filter(|(_, start, _)| *start <= time)
            .count()
    }

    /// Get the character position the typewriter should be at, synchronized
    /// with the voice line.
    pub fn synced_char_position(&self, time: f32, total_text: &str) -> usize {
        if self.clip_duration <= 0.0 {
            return total_text.len();
        }
        let fraction = (time / self.clip_duration).clamp(0.0, 1.0);
        let chars_count = total_text.chars().count();
        (fraction * chars_count as f32) as usize
    }

    /// Get the current phoneme at a given time (for lip sync).
    pub fn current_phoneme(&self, time: f32) -> Option<&str> {
        self.phoneme_timings.iter()
            .find(|p| time >= p.start_time && time < p.start_time + p.duration)
            .map(|p| p.phoneme.as_str())
    }
}

// ---------------------------------------------------------------------------
// Localization
// ---------------------------------------------------------------------------

/// Localization table for dialogue text.
#[derive(Debug, Clone)]
pub struct DialogueLocalization {
    /// Current language code (e.g., "en", "fr", "de", "ja").
    pub current_language: String,
    /// Translations: language -> (key -> text).
    translations: HashMap<String, HashMap<String, String>>,
    /// Fallback language.
    pub fallback_language: String,
}

impl DialogueLocalization {
    pub fn new(default_language: &str) -> Self {
        Self {
            current_language: default_language.to_string(),
            translations: HashMap::new(),
            fallback_language: "en".to_string(),
        }
    }

    /// Add a translation.
    pub fn add_translation(&mut self, language: &str, key: &str, text: &str) {
        self.translations
            .entry(language.to_string())
            .or_default()
            .insert(key.to_string(), text.to_string());
    }

    /// Load translations for a language from a key-value map.
    pub fn load_language(&mut self, language: &str, translations: HashMap<String, String>) {
        self.translations.insert(language.to_string(), translations);
    }

    /// Resolve a localization key to text.
    pub fn resolve(&self, key: &str) -> Option<String> {
        // Try current language first.
        if let Some(lang) = self.translations.get(&self.current_language) {
            if let Some(text) = lang.get(key) {
                return Some(text.clone());
            }
        }
        // Try fallback language.
        if self.current_language != self.fallback_language {
            if let Some(lang) = self.translations.get(&self.fallback_language) {
                if let Some(text) = lang.get(key) {
                    return Some(text.clone());
                }
            }
        }
        None
    }

    /// Resolve a key, falling back to the key itself if not found.
    pub fn resolve_or_key(&self, key: &str) -> String {
        self.resolve(key).unwrap_or_else(|| key.to_string())
    }

    /// Set the current language.
    pub fn set_language(&mut self, language: &str) {
        self.current_language = language.to_string();
    }

    /// Get available languages.
    pub fn available_languages(&self) -> Vec<&str> {
        self.translations.keys().map(|s| s.as_str()).collect()
    }

    /// Resolve a dialogue line's text (uses localization key if present, else raw text).
    pub fn resolve_line(&self, line: &DialogueLine) -> String {
        if let Some(ref key) = line.localization_key {
            self.resolve_or_key(key)
        } else {
            line.text.clone()
        }
    }

    /// Get the number of keys in a language.
    pub fn key_count(&self, language: &str) -> usize {
        self.translations.get(language).map(|t| t.len()).unwrap_or(0)
    }

    /// Check coverage: how many keys from the fallback language are present in another.
    pub fn coverage(&self, language: &str) -> f32 {
        let fallback_keys = self.key_count(&self.fallback_language);
        if fallback_keys == 0 {
            return 1.0;
        }
        let lang_keys = self.key_count(language);
        lang_keys as f32 / fallback_keys as f32
    }
}

// ---------------------------------------------------------------------------
// Typewriter state
// ---------------------------------------------------------------------------

/// Animation state for typewriter text reveal.
#[derive(Debug, Clone)]
pub struct TypewriterState {
    /// Characters revealed so far.
    pub revealed_chars: usize,
    /// Total characters in the line.
    pub total_chars: usize,
    /// Time elapsed in the current line.
    pub elapsed: f32,
    /// Characters per second.
    pub speed: f32,
    /// Whether the reveal is complete.
    pub complete: bool,
    /// Accumulated fractional character.
    accumulator: f32,
    /// Pause timer (for punctuation pauses).
    pause_timer: f32,
    /// Full text being revealed.
    full_text: String,
    /// Voice timing data for sync (if available).
    voice_timing: Option<VoiceTimingData>,
}

impl TypewriterState {
    pub fn new(text: &str, speed: f32) -> Self {
        Self {
            revealed_chars: 0,
            total_chars: text.chars().count(),
            elapsed: 0.0,
            speed,
            complete: false,
            accumulator: 0.0,
            pause_timer: 0.0,
            full_text: text.to_string(),
            voice_timing: None,
        }
    }

    /// Create with voice timing sync.
    pub fn with_voice_timing(text: &str, speed: f32, timing: VoiceTimingData) -> Self {
        let mut state = Self::new(text, speed);
        state.voice_timing = Some(timing);
        state
    }

    /// Update the typewriter state.
    pub fn update(&mut self, dt: f32) {
        if self.complete {
            return;
        }

        self.elapsed += dt;

        // If we have voice timing, sync to it.
        if let Some(ref timing) = self.voice_timing {
            self.revealed_chars = timing.synced_char_position(self.elapsed, &self.full_text);
            if self.elapsed >= timing.clip_duration {
                self.revealed_chars = self.total_chars;
                self.complete = true;
            }
            return;
        }

        // Handle punctuation pauses.
        if self.pause_timer > 0.0 {
            self.pause_timer -= dt;
            return;
        }

        self.accumulator += self.speed * dt;
        while self.accumulator >= 1.0 && self.revealed_chars < self.total_chars {
            self.accumulator -= 1.0;
            self.revealed_chars += 1;

            // Check for punctuation pauses.
            let current_char = self.full_text.chars().nth(self.revealed_chars.saturating_sub(1));
            match current_char {
                Some('.') | Some('!') | Some('?') => {
                    self.pause_timer = 0.3; // Longer pause at sentence ends.
                }
                Some(',') | Some(';') | Some(':') => {
                    self.pause_timer = 0.1; // Short pause at commas.
                }
                Some('-') => {
                    self.pause_timer = 0.15;
                }
                _ => {}
            }
        }

        if self.revealed_chars >= self.total_chars {
            self.complete = true;
        }
    }

    /// Skip to the end (reveal all text immediately).
    pub fn skip(&mut self) {
        self.revealed_chars = self.total_chars;
        self.complete = true;
    }

    /// Get the currently visible text.
    pub fn visible_text(&self) -> String {
        self.full_text.chars().take(self.revealed_chars).collect()
    }

    /// Get the completion fraction (0.0 to 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_chars == 0 {
            return 1.0;
        }
        self.revealed_chars as f32 / self.total_chars as f32
    }
}

// ---------------------------------------------------------------------------
// Dialogue definition (now with branching nodes)
// ---------------------------------------------------------------------------

/// A complete dialogue definition, supporting both linear and branching modes.
#[derive(Debug, Clone)]
pub struct DialogueDefinition {
    pub id: DialogueId,
    pub name: String,
    pub mode: DialogueMode,
    pub priority: i32,
    /// Linear dialogue lines (used when nodes is empty).
    pub lines: Vec<DialogueLine>,
    /// Branching dialogue nodes.
    pub nodes: HashMap<DialogueNodeId, DialogueNode>,
    /// Starting node ID for branching dialogues.
    pub start_node: Option<DialogueNodeId>,
    pub cooldown: f32,
    pub can_be_interrupted: bool,
    pub skippable: bool,
    pub conditions: Vec<DialogueCondition>,
    pub on_complete_events: Vec<DialogueGameEvent>,
    /// Max times this dialogue can be played (0 = unlimited).
    pub max_plays: u32,
    /// Condition tags for filtering.
    pub tags: Vec<String>,
}

impl DialogueDefinition {
    pub fn new(id: DialogueId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            mode: DialogueMode::Subtitled,
            priority: 0,
            lines: Vec::new(),
            nodes: HashMap::new(),
            start_node: None,
            cooldown: 0.0,
            can_be_interrupted: true,
            skippable: true,
            conditions: Vec::new(),
            on_complete_events: Vec::new(),
            max_plays: 0,
            tags: Vec::new(),
        }
    }

    pub fn add_line(&mut self, line: DialogueLine) {
        self.lines.push(line);
    }

    /// Add a branching node.
    pub fn add_node(&mut self, node: DialogueNode) {
        if self.start_node.is_none() {
            self.start_node = Some(node.id);
        }
        self.nodes.insert(node.id, node);
    }

    /// Whether this dialogue uses branching nodes.
    pub fn is_branching(&self) -> bool {
        !self.nodes.is_empty()
    }

    pub fn total_duration(&self) -> f32 {
        self.lines.iter().map(|l| l.auto_duration()).sum()
    }
}

// ---------------------------------------------------------------------------
// Active conversation
// ---------------------------------------------------------------------------

/// State of an active conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    Playing,
    Paused,
    WaitingForInput,
    Finished,
    Interrupted,
}

/// An active conversation instance.
#[derive(Debug, Clone)]
pub struct ActiveConversation {
    pub id: ConversationId,
    pub dialogue_id: DialogueId,
    pub state: ConversationState,
    pub current_line: usize,
    pub line_timer: f32,
    pub mode: DialogueMode,
    pub priority: i32,
    pub participants: Vec<SpeakerId>,
    pub elapsed: f32,
    /// Current node ID (for branching dialogues).
    pub current_node: Option<DialogueNodeId>,
    /// NPC mood for this conversation.
    pub npc_mood: NpcMood,
    /// Typewriter state for the current line.
    pub typewriter: Option<TypewriterState>,
    /// Whether we are in branching mode.
    pub branching: bool,
}

impl ActiveConversation {
    pub fn new(id: ConversationId, dialogue: &DialogueDefinition) -> Self {
        let participants: Vec<SpeakerId> = dialogue
            .lines
            .iter()
            .map(|l| l.speaker)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let branching = dialogue.is_branching();
        let current_node = dialogue.start_node;

        Self {
            id,
            dialogue_id: dialogue.id,
            state: if branching && current_node.is_some() {
                ConversationState::Playing
            } else {
                ConversationState::Playing
            },
            current_line: 0,
            line_timer: 0.0,
            mode: dialogue.mode,
            priority: dialogue.priority,
            participants,
            elapsed: 0.0,
            current_node,
            npc_mood: NpcMood::new(),
            typewriter: None,
            branching,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, ConversationState::Playing | ConversationState::Paused | ConversationState::WaitingForInput)
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, ConversationState::Finished | ConversationState::Interrupted)
    }
}

// ---------------------------------------------------------------------------
// Dialogue events
// ---------------------------------------------------------------------------

/// Events emitted by the dialogue manager.
#[derive(Debug, Clone)]
pub enum DialogueMgrEvent {
    ConversationStarted { conversation: ConversationId, dialogue: DialogueId },
    LineStarted { conversation: ConversationId, line_index: usize, speaker: SpeakerId, text: String },
    LineEnded { conversation: ConversationId, line_index: usize },
    ConversationEnded { conversation: ConversationId, dialogue: DialogueId },
    ConversationInterrupted { conversation: ConversationId, by: ConversationId },
    GameEvent { conversation: ConversationId, event: DialogueGameEvent },
    CooldownStarted { dialogue: DialogueId, duration: f32 },
    BarkDisplayed { speaker: SpeakerId, text: String },
    /// Player needs to choose a response.
    ResponsesAvailable {
        conversation: ConversationId,
        responses: Vec<ResponseOption>,
    },
    /// Player chose a response.
    ResponseChosen {
        conversation: ConversationId,
        response_index: usize,
        text: String,
    },
    /// Skill check result.
    SkillCheckPerformed {
        conversation: ConversationId,
        result: SkillCheckResult,
    },
    /// NPC mood changed significantly.
    MoodChanged {
        conversation: ConversationId,
        speaker: SpeakerId,
        mood_type: NpcMoodType,
        new_value: f32,
    },
    /// Typewriter text updated (for UI to re-render).
    TypewriterUpdate {
        conversation: ConversationId,
        visible_text: String,
        fraction: f32,
    },
    /// Node entered in branching dialogue.
    NodeEntered {
        conversation: ConversationId,
        node_id: DialogueNodeId,
    },
    /// Node exited in branching dialogue.
    NodeExited {
        conversation: ConversationId,
        node_id: DialogueNodeId,
    },
}

/// A response option presented to the player.
#[derive(Debug, Clone)]
pub struct ResponseOption {
    pub index: usize,
    pub text: String,
    pub display_tag: Option<String>,
    pub available: bool,
    pub tooltip: Option<String>,
}

// ---------------------------------------------------------------------------
// Dialogue manager
// ---------------------------------------------------------------------------

/// Manages dialogue playback, queueing, branching, and lifecycle.
pub struct DialogueManager {
    /// Dialogue definitions.
    definitions: HashMap<DialogueId, DialogueDefinition>,
    /// Active conversations.
    active: Vec<ActiveConversation>,
    /// Queued dialogues waiting to play.
    queue: VecDeque<DialogueId>,
    /// Cooldowns (dialogue_id -> remaining cooldown).
    cooldowns: HashMap<DialogueId, f32>,
    /// Next conversation ID.
    next_conv_id: u32,
    /// Maximum simultaneous conversations.
    max_simultaneous: usize,
    /// Events this frame.
    events: Vec<DialogueMgrEvent>,
    /// Bark display list.
    active_barks: Vec<ActiveBark>,
    /// Game time.
    time: f64,
    /// Dialogue history.
    history: DialogueHistory,
    /// Dialogue context (game state for conditions).
    context: DialogueContext,
    /// Localization table.
    localization: DialogueLocalization,
    /// Simple RNG for skill checks.
    rng_state: u64,
}

/// An active bark (short text above an NPC's head).
#[derive(Debug, Clone)]
pub struct ActiveBark {
    pub speaker: SpeakerId,
    pub text: String,
    pub remaining: f32,
    pub position: [f32; 3],
}

impl DialogueManager {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            active: Vec::new(),
            queue: VecDeque::new(),
            cooldowns: HashMap::new(),
            next_conv_id: 0,
            max_simultaneous: 3,
            events: Vec::new(),
            active_barks: Vec::new(),
            time: 0.0,
            history: DialogueHistory::default(),
            context: DialogueContext::new(),
            localization: DialogueLocalization::new("en"),
            rng_state: 12345,
        }
    }

    fn next_rng(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Get mutable context for modifying game state.
    pub fn context_mut(&mut self) -> &mut DialogueContext {
        &mut self.context
    }

    /// Get the context.
    pub fn context(&self) -> &DialogueContext {
        &self.context
    }

    /// Get the localization table.
    pub fn localization(&self) -> &DialogueLocalization {
        &self.localization
    }

    /// Get mutable localization table.
    pub fn localization_mut(&mut self) -> &mut DialogueLocalization {
        &mut self.localization
    }

    /// Get the dialogue history.
    pub fn history(&self) -> &DialogueHistory {
        &self.history
    }

    /// Register a dialogue definition.
    pub fn register(&mut self, definition: DialogueDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Get a definition.
    pub fn definition(&self, id: DialogueId) -> Option<&DialogueDefinition> {
        self.definitions.get(&id)
    }

    /// Check if a dialogue's conditions are met.
    pub fn can_start(&self, dialogue_id: DialogueId) -> bool {
        let def = match self.definitions.get(&dialogue_id) {
            Some(d) => d,
            None => return false,
        };

        // Check cooldown.
        if let Some(&cd) = self.cooldowns.get(&dialogue_id) {
            if cd > 0.0 {
                return false;
            }
        }

        // Check max plays.
        if def.max_plays > 0 && self.history.times_played(dialogue_id) >= def.max_plays as usize {
            return false;
        }

        // Check conditions.
        let dummy_mood = NpcMood::new();
        for condition in &def.conditions {
            if !self.context.evaluate(condition, &dummy_mood) {
                return false;
            }
        }

        true
    }

    /// Start a dialogue immediately.
    pub fn start(&mut self, dialogue_id: DialogueId) -> Option<ConversationId> {
        if !self.can_start(dialogue_id) {
            return None;
        }

        let def = self.definitions.get(&dialogue_id)?.clone();

        // Check if we need to interrupt lower-priority conversations.
        if self.active.len() >= self.max_simultaneous {
            let lowest = self.active.iter().min_by_key(|c| c.priority);
            if let Some(low) = lowest {
                if low.priority < def.priority {
                    let low_id = low.id;
                    self.interrupt(low_id);
                } else {
                    self.queue.push_back(dialogue_id);
                    return None;
                }
            }
        }

        let conv_id = ConversationId(self.next_conv_id);
        self.next_conv_id += 1;

        let mut conv = ActiveConversation::new(conv_id, &def);

        self.events.push(DialogueMgrEvent::ConversationStarted {
            conversation: conv_id,
            dialogue: dialogue_id,
        });

        if conv.branching {
            // Start branching dialogue at the start node.
            if let Some(node_id) = conv.current_node {
                self.enter_node(&mut conv, &def, node_id);
            }
        } else {
            // Linear dialogue: fire first line events.
            if !def.lines.is_empty() {
                let line = &def.lines[0];
                let resolved_text = self.localization.resolve_line(line);
                let substituted = line.substitute_variables(&self.context);
                let display_text = if line.localization_key.is_some() {
                    resolved_text
                } else {
                    substituted
                };

                // Set up typewriter.
                if line.typewriter {
                    conv.typewriter = Some(TypewriterState::new(&display_text, line.typewriter_speed));
                }

                self.events.push(DialogueMgrEvent::LineStarted {
                    conversation: conv_id,
                    line_index: 0,
                    speaker: line.speaker,
                    text: display_text,
                });
                for event in &line.events {
                    self.events.push(DialogueMgrEvent::GameEvent {
                        conversation: conv_id,
                        event: event.clone(),
                    });
                }
            }
        }

        self.active.push(conv);
        Some(conv_id)
    }

    /// Enter a branching dialogue node.
    fn enter_node(&mut self, conv: &mut ActiveConversation, def: &DialogueDefinition, node_id: DialogueNodeId) {
        let node = match def.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => {
                conv.state = ConversationState::Finished;
                return;
            }
        };

        // Execute on_enter actions.
        for action in &node.on_enter {
            execute_action(action, &mut self.context, &mut conv.npc_mood);
        }

        // Record in history.
        self.context.visited_nodes.insert((def.id, node_id));
        self.history.record_line(
            def.id,
            node_id,
            node.line.speaker,
            &node.line.text,
            self.time,
        );

        // Apply mood effects from the line.
        for &(mood_type, delta) in &node.line.mood_effects {
            let old_value = conv.npc_mood.get(mood_type);
            conv.npc_mood.modify(mood_type, delta);
            let new_value = conv.npc_mood.get(mood_type);
            if (new_value - old_value).abs() > 0.05 {
                self.events.push(DialogueMgrEvent::MoodChanged {
                    conversation: conv.id,
                    speaker: node.line.speaker,
                    mood_type,
                    new_value,
                });
            }
        }

        let resolved_text = self.localization.resolve_line(&node.line);
        let display_text = if node.line.localization_key.is_some() {
            resolved_text
        } else {
            node.line.substitute_variables(&self.context)
        };

        // Set up typewriter.
        if node.line.typewriter {
            conv.typewriter = Some(TypewriterState::new(&display_text, node.line.typewriter_speed));
        } else {
            conv.typewriter = None;
        }

        conv.current_node = Some(node_id);
        conv.line_timer = 0.0;

        self.events.push(DialogueMgrEvent::NodeEntered {
            conversation: conv.id,
            node_id,
        });

        self.events.push(DialogueMgrEvent::LineStarted {
            conversation: conv.id,
            line_index: node_id.0 as usize,
            speaker: node.line.speaker,
            text: display_text,
        });

        for event in &node.line.events {
            self.events.push(DialogueMgrEvent::GameEvent {
                conversation: conv.id,
                event: event.clone(),
            });
        }
    }

    /// Present responses for a branching node.
    fn present_responses(&mut self, conv: &mut ActiveConversation, def: &DialogueDefinition) {
        let node_id = match conv.current_node {
            Some(id) => id,
            None => return,
        };
        let node = match def.nodes.get(&node_id) {
            Some(n) => n,
            None => return,
        };

        // Check for auto-advance.
        if let Some(next_id) = node.auto_advance {
            // Exit current node.
            for action in &node.on_exit {
                execute_action(action, &mut self.context, &mut conv.npc_mood);
            }
            self.events.push(DialogueMgrEvent::NodeExited {
                conversation: conv.id,
                node_id,
            });
            self.enter_node(conv, def, next_id);
            return;
        }

        let available = node.available_responses(&self.context, &conv.npc_mood);

        if available.is_empty() {
            // No responses available: conversation ends.
            for action in &node.on_exit {
                execute_action(action, &mut self.context, &mut conv.npc_mood);
            }
            self.events.push(DialogueMgrEvent::NodeExited {
                conversation: conv.id,
                node_id,
            });
            conv.state = ConversationState::Finished;
            return;
        }

        // Build response options for the UI.
        let options: Vec<ResponseOption> = node.responses.iter()
            .enumerate()
            .map(|(i, r)| {
                let available = self.context.evaluate(&r.condition, &conv.npc_mood);
                let display_text = if let Some(ref key) = r.localization_key {
                    self.localization.resolve_or_key(key)
                } else {
                    r.text.clone()
                };
                let tag_text = if let Some(ref tag) = r.display_tag {
                    format!("{} {}", tag, display_text)
                } else {
                    display_text
                };
                ResponseOption {
                    index: i,
                    text: tag_text,
                    display_tag: r.display_tag.clone(),
                    available,
                    tooltip: if !available { r.tooltip.clone() } else { None },
                }
            })
            .filter(|o| o.available || node.responses[o.index].show_when_unavailable)
            .collect();

        conv.state = ConversationState::WaitingForInput;
        self.events.push(DialogueMgrEvent::ResponsesAvailable {
            conversation: conv.id,
            responses: options,
        });
    }

    /// Player selects a response in a branching conversation.
    pub fn select_response(&mut self, conv_id: ConversationId, response_index: usize) {
        // Find the conversation.
        let conv_idx = match self.active.iter().position(|c| c.id == conv_id) {
            Some(i) => i,
            None => return,
        };

        let dialogue_id = self.active[conv_idx].dialogue_id;
        let def = match self.definitions.get(&dialogue_id) {
            Some(d) => d.clone(),
            None => return,
        };

        let node_id = match self.active[conv_idx].current_node {
            Some(id) => id,
            None => return,
        };

        let node = match def.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => return,
        };

        if response_index >= node.responses.len() {
            return;
        }

        let response = &node.responses[response_index];

        // Record the response choice.
        self.history.record_response(dialogue_id, node_id, &response.text);
        self.events.push(DialogueMgrEvent::ResponseChosen {
            conversation: conv_id,
            response_index,
            text: response.text.clone(),
        });

        // Apply mood effects from the response.
        for &(mood_type, delta) in &response.mood_effects {
            self.active[conv_idx].npc_mood.modify(mood_type, delta);
        }

        // Execute response actions.
        for action in &response.on_select {
            let mood = &mut self.active[conv_idx].npc_mood;
            execute_action(action, &mut self.context, mood);
        }

        // Handle skill check if present.
        let mut target_node = response.target_node;
        if let Some((skill, required)) = response.skill_check {
            let player_level = self.context.skills.get(&skill).copied().unwrap_or(0);
            let rng = self.next_rng();
            let mood = &self.active[conv_idx].npc_mood;
            let result = perform_skill_check(skill, player_level, required, mood, rng);
            self.events.push(DialogueMgrEvent::SkillCheckPerformed {
                conversation: conv_id,
                result: result.clone(),
            });
            if !result.success {
                if let Some(fail_target) = response.fail_target {
                    target_node = fail_target;
                }
            }
        }

        // Exit current node.
        for action in &node.on_exit {
            let mood = &mut self.active[conv_idx].npc_mood;
            execute_action(action, &mut self.context, mood);
        }
        self.events.push(DialogueMgrEvent::NodeExited {
            conversation: conv_id,
            node_id,
        });

        // Enter the target node.
        self.active[conv_idx].state = ConversationState::Playing;
        self.active[conv_idx].line_timer = 0.0;
        let mut conv = self.active[conv_idx].clone();
        self.enter_node(&mut conv, &def, target_node);
        self.active[conv_idx] = conv;
    }

    /// Queue a dialogue for later playback.
    pub fn enqueue(&mut self, dialogue_id: DialogueId) {
        self.queue.push_back(dialogue_id);
    }

    /// Interrupt a conversation.
    pub fn interrupt(&mut self, conv_id: ConversationId) {
        if let Some(conv) = self.active.iter_mut().find(|c| c.id == conv_id) {
            conv.state = ConversationState::Interrupted;
            self.events.push(DialogueMgrEvent::ConversationEnded {
                conversation: conv_id,
                dialogue: conv.dialogue_id,
            });
        }
    }

    /// Skip the current line of a conversation.
    pub fn skip_line(&mut self, conv_id: ConversationId) {
        if let Some(conv) = self.active.iter_mut().find(|c| c.id == conv_id) {
            if let Some(def) = self.definitions.get(&conv.dialogue_id) {
                if !def.skippable {
                    return;
                }
            }
            // Skip typewriter.
            if let Some(ref mut tw) = conv.typewriter {
                if !tw.complete {
                    tw.skip();
                    return; // First skip finishes the typewriter. Second skip advances.
                }
            }
            conv.line_timer = f32::MAX; // Will advance on next update.
        }
    }

    /// Display a bark (short text over NPC head).
    pub fn bark(&mut self, speaker: SpeakerId, text: &str, position: [f32; 3], duration: f32) {
        // Variable substitution in barks too.
        let substituted = {
            let mut result = text.to_string();
            loop {
                let start = match result.find('{') {
                    Some(i) => i,
                    None => break,
                };
                let end = match result[start..].find('}') {
                    Some(i) => start + i,
                    None => break,
                };
                let var_name = &result[start + 1..end].to_string();
                let replacement = self.context.get_variable_string(var_name);
                result = format!("{}{}{}", &result[..start], replacement, &result[end + 1..]);
            }
            result
        };

        self.active_barks.push(ActiveBark {
            speaker,
            text: substituted.clone(),
            remaining: duration,
            position,
        });
        self.events.push(DialogueMgrEvent::BarkDisplayed {
            speaker,
            text: substituted,
        });
    }

    /// Update all conversations.
    pub fn update(&mut self, dt: f32) {
        self.time += dt as f64;
        self.context.game_time = self.time;

        // Update cooldowns.
        for cd in self.cooldowns.values_mut() {
            *cd = (*cd - dt).max(0.0);
        }

        // Update barks.
        self.active_barks.retain_mut(|b| {
            b.remaining -= dt;
            b.remaining > 0.0
        });

        // We need to collect operations to apply after iteration to avoid
        // double borrow issues.
        let mut finished = Vec::new();
        let mut node_transitions: Vec<(usize, DialogueNodeId)> = Vec::new();
        let mut present_responses_indices: Vec<usize> = Vec::new();

        for (conv_idx, conv) in self.active.iter_mut().enumerate() {
            if !conv.is_active() {
                continue;
            }

            if conv.state == ConversationState::Paused {
                continue;
            }

            if conv.state == ConversationState::WaitingForInput {
                // Decay NPC mood while waiting.
                conv.npc_mood.decay(dt);

                // Check if NPC lost patience.
                if conv.npc_mood.has_lost_patience() {
                    conv.state = ConversationState::Finished;
                    finished.push((conv.id, conv.dialogue_id));
                }
                continue;
            }

            conv.elapsed += dt;

            // Update typewriter.
            if let Some(ref mut tw) = conv.typewriter {
                tw.update(dt);
                self.events.push(DialogueMgrEvent::TypewriterUpdate {
                    conversation: conv.id,
                    visible_text: tw.visible_text(),
                    fraction: tw.fraction(),
                });
            }

            if conv.branching {
                // Branching mode: wait for line duration, then present responses.
                let node_id = match conv.current_node {
                    Some(id) => id,
                    None => {
                        conv.state = ConversationState::Finished;
                        finished.push((conv.id, conv.dialogue_id));
                        continue;
                    }
                };

                let def = match self.definitions.get(&conv.dialogue_id) {
                    Some(d) => d,
                    None => {
                        conv.state = ConversationState::Finished;
                        continue;
                    }
                };

                let node = match def.nodes.get(&node_id) {
                    Some(n) => n,
                    None => {
                        conv.state = ConversationState::Finished;
                        finished.push((conv.id, conv.dialogue_id));
                        continue;
                    }
                };

                let line_duration = node.line.auto_duration();
                conv.line_timer += dt;

                // Wait for typewriter to complete OR line duration, whichever is longer.
                let tw_complete = conv.typewriter.as_ref().map(|tw| tw.complete).unwrap_or(true);
                if conv.line_timer >= line_duration && tw_complete {
                    if node.is_terminal {
                        conv.state = ConversationState::Finished;
                        finished.push((conv.id, conv.dialogue_id));
                    } else {
                        present_responses_indices.push(conv_idx);
                    }
                }
            } else {
                // Linear mode.
                let def = match self.definitions.get(&conv.dialogue_id) {
                    Some(d) => d,
                    None => {
                        conv.state = ConversationState::Finished;
                        continue;
                    }
                };

                if conv.current_line >= def.lines.len() {
                    conv.state = ConversationState::Finished;
                    finished.push((conv.id, conv.dialogue_id));
                    continue;
                }

                let line = &def.lines[conv.current_line];
                let line_duration = line.auto_duration();

                conv.line_timer += dt;
                let tw_complete = conv.typewriter.as_ref().map(|tw| tw.complete).unwrap_or(true);

                if conv.line_timer >= line_duration && tw_complete {
                    self.events.push(DialogueMgrEvent::LineEnded {
                        conversation: conv.id,
                        line_index: conv.current_line,
                    });

                    conv.current_line += 1;
                    conv.line_timer = 0.0;

                    if conv.current_line < def.lines.len() {
                        let next_line = &def.lines[conv.current_line];
                        let display_text = if let Some(ref key) = next_line.localization_key {
                            self.localization.resolve_or_key(key)
                        } else {
                            next_line.substitute_variables(&self.context)
                        };

                        if next_line.typewriter {
                            conv.typewriter = Some(TypewriterState::new(&display_text, next_line.typewriter_speed));
                        } else {
                            conv.typewriter = None;
                        }

                        self.events.push(DialogueMgrEvent::LineStarted {
                            conversation: conv.id,
                            line_index: conv.current_line,
                            speaker: next_line.speaker,
                            text: display_text,
                        });
                        for event in &next_line.events {
                            self.events.push(DialogueMgrEvent::GameEvent {
                                conversation: conv.id,
                                event: event.clone(),
                            });
                        }
                    } else {
                        conv.state = ConversationState::Finished;
                        finished.push((conv.id, conv.dialogue_id));
                    }
                }
            }
        }

        // Present responses for branching conversations that reached end of line.
        for conv_idx in present_responses_indices {
            let def = match self.definitions.get(&self.active[conv_idx].dialogue_id) {
                Some(d) => d.clone(),
                None => continue,
            };
            let mut conv = self.active[conv_idx].clone();
            self.present_responses(&mut conv, &def);
            self.active[conv_idx] = conv;
        }

        // Handle finished conversations.
        for (conv_id, dialogue_id) in &finished {
            self.events.push(DialogueMgrEvent::ConversationEnded {
                conversation: *conv_id,
                dialogue: *dialogue_id,
            });

            self.context.completed_dialogues.insert(*dialogue_id);
            self.context.dialogue_timestamps.insert(*dialogue_id, self.time);

            if let Some(def) = self.definitions.get(dialogue_id) {
                if def.cooldown > 0.0 {
                    self.cooldowns.insert(*dialogue_id, def.cooldown);
                    self.events.push(DialogueMgrEvent::CooldownStarted {
                        dialogue: *dialogue_id,
                        duration: def.cooldown,
                    });
                }
                for event in &def.on_complete_events {
                    self.events.push(DialogueMgrEvent::GameEvent {
                        conversation: *conv_id,
                        event: event.clone(),
                    });
                }
            }
        }

        // Remove finished conversations.
        self.active.retain(|c| c.is_active());

        // Process queue.
        while self.active.len() < self.max_simultaneous {
            if let Some(next) = self.queue.pop_front() {
                self.start(next);
            } else {
                break;
            }
        }
    }

    /// Get active conversations.
    pub fn active_conversations(&self) -> &[ActiveConversation] {
        &self.active
    }

    /// Get active barks.
    pub fn active_barks(&self) -> &[ActiveBark] {
        &self.active_barks
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<DialogueMgrEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if a dialogue is on cooldown.
    pub fn is_on_cooldown(&self, id: DialogueId) -> bool {
        self.cooldowns.get(&id).map(|&cd| cd > 0.0).unwrap_or(false)
    }

    /// Set the maximum simultaneous conversations.
    pub fn set_max_simultaneous(&mut self, max: usize) {
        self.max_simultaneous = max.max(1);
    }

    /// Get the number of active conversations.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get the queue length.
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }

    /// Set the NPC mood for a specific conversation.
    pub fn set_conversation_mood(&mut self, conv_id: ConversationId, mood: NpcMood) {
        if let Some(conv) = self.active.iter_mut().find(|c| c.id == conv_id) {
            conv.npc_mood = mood;
        }
    }
}

impl Default for DialogueManager {
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

    fn make_dialogue(id: u32) -> DialogueDefinition {
        let mut def = DialogueDefinition::new(DialogueId(id), &format!("Dialogue {}", id));
        def.lines.push(DialogueLine::new(SpeakerId(0), "NPC", "Hello there!"));
        def.lines.push(DialogueLine::new(SpeakerId(0), "NPC", "How are you?"));
        def
    }

    #[test]
    fn test_start_dialogue() {
        let mut mgr = DialogueManager::new();
        mgr.register(make_dialogue(0));
        let conv = mgr.start(DialogueId(0));
        assert!(conv.is_some());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_dialogue_progression() {
        let mut mgr = DialogueManager::new();
        let mut def = make_dialogue(0);
        for line in &mut def.lines {
            line.duration = 1.0;
            line.typewriter = false;
        }
        mgr.register(def);
        mgr.start(DialogueId(0));

        mgr.update(1.5);
        let events = mgr.drain_events();
        assert!(events.iter().any(|e| matches!(e, DialogueMgrEvent::LineEnded { .. })));
    }

    #[test]
    fn test_cooldown() {
        let mut mgr = DialogueManager::new();
        let mut def = make_dialogue(0);
        def.cooldown = 5.0;
        for line in &mut def.lines {
            line.duration = 0.1;
            line.typewriter = false;
        }
        mgr.register(def);

        mgr.start(DialogueId(0));
        mgr.update(1.0);

        assert!(mgr.start(DialogueId(0)).is_none());
        assert!(mgr.is_on_cooldown(DialogueId(0)));
    }

    #[test]
    fn test_bark() {
        let mut mgr = DialogueManager::new();
        mgr.bark(SpeakerId(1), "Watch out!", [0.0; 3], 3.0);
        assert_eq!(mgr.active_barks().len(), 1);
        mgr.update(4.0);
        assert_eq!(mgr.active_barks().len(), 0);
    }

    #[test]
    fn test_queue() {
        let mut mgr = DialogueManager::new();
        mgr.set_max_simultaneous(1);
        mgr.register(make_dialogue(0));
        mgr.register(make_dialogue(1));

        mgr.start(DialogueId(0));
        mgr.enqueue(DialogueId(1));

        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.queue_length(), 1);
    }

    #[test]
    fn test_variable_substitution() {
        let mut context = DialogueContext::new();
        context.set_variable("player_name", DialogueValue::String("Hero".to_string()));
        context.set_variable("item_count", DialogueValue::Integer(5));

        let line = DialogueLine::new(SpeakerId(0), "NPC", "Hello, {player_name}! You have {item_count} items.");
        let result = line.substitute_variables(&context);
        assert_eq!(result, "Hello, Hero! You have 5 items.");
    }

    #[test]
    fn test_condition_evaluation() {
        let mut context = DialogueContext::new();
        context.set_flag("quest_started");
        context.inventory.insert("key".to_string(), 1);
        context.reputations.insert("guild".to_string(), 50.0);
        let mood = NpcMood::new();

        assert!(context.evaluate(&DialogueCondition::FlagSet { flag: "quest_started".to_string() }, &mood));
        assert!(!context.evaluate(&DialogueCondition::FlagSet { flag: "quest_done".to_string() }, &mood));
        assert!(context.evaluate(&DialogueCondition::HasItem { item_id: "key".to_string(), min_count: 1 }, &mood));
        assert!(context.evaluate(&DialogueCondition::ReputationAtLeast { faction: "guild".to_string(), min_value: 25.0 }, &mood));
    }

    #[test]
    fn test_npc_mood() {
        let mut mood = NpcMood::new();
        mood.modify(NpcMoodType::Anger, 0.5);
        assert!(mood.get(NpcMoodType::Anger) > 0.4);

        mood.modify(NpcMoodType::Friendliness, -0.4);
        assert!(mood.is_hostile());
    }

    #[test]
    fn test_skill_check() {
        let mood = NpcMood::new();
        let result = perform_skill_check(SkillCheckType::Persuasion, 10, 5, &mood, 0.5);
        assert!(result.success);
        assert!(result.margin > 0);
    }

    #[test]
    fn test_typewriter() {
        let mut tw = TypewriterState::new("Hello, world!", 10.0);
        assert!(!tw.complete);
        assert_eq!(tw.revealed_chars, 0);

        tw.update(0.5); // 5 chars
        assert!(tw.revealed_chars >= 4);

        tw.skip();
        assert!(tw.complete);
        assert_eq!(tw.visible_text(), "Hello, world!");
    }

    #[test]
    fn test_localization() {
        let mut loc = DialogueLocalization::new("en");
        loc.add_translation("en", "greeting_01", "Hello, adventurer!");
        loc.add_translation("fr", "greeting_01", "Bonjour, aventurier!");

        assert_eq!(loc.resolve("greeting_01").unwrap(), "Hello, adventurer!");
        loc.set_language("fr");
        assert_eq!(loc.resolve("greeting_01").unwrap(), "Bonjour, aventurier!");
    }

    #[test]
    fn test_dialogue_history() {
        let mut history = DialogueHistory::new(100);
        history.record_line(DialogueId(0), DialogueNodeId(0), SpeakerId(0), "Hello", 0.0);
        history.record_line(DialogueId(0), DialogueNodeId(1), SpeakerId(0), "World", 1.0);

        assert!(history.has_been_said(DialogueId(0), DialogueNodeId(0)));
        assert!(!history.has_been_said(DialogueId(1), DialogueNodeId(0)));
        assert_eq!(history.total_entries(), 2);
    }

    #[test]
    fn test_branching_dialogue() {
        let mut mgr = DialogueManager::new();

        let mut def = DialogueDefinition::new(DialogueId(100), "Branching Test");
        def.mode = DialogueMode::Subtitled;

        let mut node0 = DialogueNode::new(
            DialogueNodeId(0),
            DialogueLine::new(SpeakerId(0), "NPC", "What do you want?"),
        );
        node0.line.duration = 0.1;
        node0.line.typewriter = false;
        node0.responses.push(DialogueResponse::new("Tell me about the quest.", DialogueNodeId(1)));
        node0.responses.push(DialogueResponse::new("Goodbye.", DialogueNodeId(2)));

        let mut node1 = DialogueNode::new(
            DialogueNodeId(1),
            DialogueLine::new(SpeakerId(0), "NPC", "The quest is to slay the dragon."),
        );
        node1.is_terminal = true;
        node1.line.duration = 0.1;
        node1.line.typewriter = false;

        let mut node2 = DialogueNode::new(
            DialogueNodeId(2),
            DialogueLine::new(SpeakerId(0), "NPC", "Farewell."),
        );
        node2.is_terminal = true;
        node2.line.duration = 0.1;
        node2.line.typewriter = false;

        def.add_node(node0);
        def.add_node(node1);
        def.add_node(node2);

        mgr.register(def);

        let conv = mgr.start(DialogueId(100)).unwrap();
        mgr.update(0.2); // Should present responses after line plays.
        let events = mgr.drain_events();
        assert!(events.iter().any(|e| matches!(e, DialogueMgrEvent::ResponsesAvailable { .. })));

        // Select response 0 (quest info).
        mgr.select_response(conv, 0);
        mgr.update(0.2);
        // Conversation should finish after terminal node.
    }
}
