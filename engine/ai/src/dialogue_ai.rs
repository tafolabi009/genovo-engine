//! NPC dialogue decision-making system.
//!
//! Manages NPC dispositions, faction-based reputation tracking, relationship
//! graphs between NPCs, conversation topic selection based on context and
//! history, and mood modeling. This system sits alongside the gameplay
//! dialogue tree runner and provides the AI-side decision logic for what
//! NPCs choose to say and how they react.
//!
//! # Key concepts
//!
//! - **Disposition**: An NPC's attitude toward another entity, ranging from
//!   hostile through friendly. Computed from base personality, reputation,
//!   relationship, and mood.
//! - **ReputationTracker**: Tracks per-faction reputation for each entity.
//! - **RelationshipGraph**: Directed weighted graph of NPC-to-NPC feelings.
//! - **ConversationContext**: Accumulated history of a conversation, used for
//!   topic selection and avoiding repetition.
//! - **MoodModel**: Dynamic emotional state that affects dialogue choices.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default neutral disposition score.
pub const NEUTRAL_DISPOSITION: f32 = 0.0;

/// Maximum disposition value (very friendly).
pub const MAX_DISPOSITION: f32 = 100.0;

/// Minimum disposition value (extremely hostile).
pub const MIN_DISPOSITION: f32 = -100.0;

/// Threshold above which an NPC is considered friendly.
pub const FRIENDLY_THRESHOLD: f32 = 25.0;

/// Threshold below which an NPC is considered hostile.
pub const HOSTILE_THRESHOLD: f32 = -25.0;

/// Threshold for unfriendly (between hostile and neutral).
pub const UNFRIENDLY_THRESHOLD: f32 = -10.0;

/// Threshold for warm (between neutral and friendly).
pub const WARM_THRESHOLD: f32 = 10.0;

/// Maximum conversation history length.
pub const MAX_CONVERSATION_HISTORY: usize = 64;

/// Maximum number of topics tracked per NPC.
pub const MAX_TOPICS_PER_NPC: usize = 128;

/// Mood decay rate per second (toward neutral).
pub const MOOD_DECAY_RATE: f32 = 0.02;

/// Maximum number of relationships per NPC.
pub const MAX_RELATIONSHIPS: usize = 256;

/// Maximum number of factions.
pub const MAX_FACTIONS: usize = 64;

/// Default mood value (neutral).
pub const DEFAULT_MOOD: f32 = 0.5;

/// Weight of reputation in disposition calculation.
pub const REPUTATION_WEIGHT: f32 = 0.4;

/// Weight of relationship in disposition calculation.
pub const RELATIONSHIP_WEIGHT: f32 = 0.3;

/// Weight of mood in disposition calculation.
pub const MOOD_WEIGHT: f32 = 0.15;

/// Weight of personality in disposition calculation.
pub const PERSONALITY_WEIGHT: f32 = 0.15;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// DispositionLevel
// ---------------------------------------------------------------------------

/// Discrete disposition levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispositionLevel {
    /// Actively hostile — will attack on sight or refuse dialogue.
    Hostile,
    /// Unfriendly — short, curt responses, may refuse help.
    Unfriendly,
    /// Neutral — standard interactions.
    Neutral,
    /// Warm — friendly, offers help, shares information.
    Warm,
    /// Friendly — enthusiastic, grants discounts, reveals secrets.
    Friendly,
}

impl DispositionLevel {
    /// Convert a numeric disposition to a level.
    pub fn from_score(score: f32) -> Self {
        if score <= HOSTILE_THRESHOLD {
            Self::Hostile
        } else if score <= UNFRIENDLY_THRESHOLD {
            Self::Unfriendly
        } else if score >= FRIENDLY_THRESHOLD {
            Self::Friendly
        } else if score >= WARM_THRESHOLD {
            Self::Warm
        } else {
            Self::Neutral
        }
    }

    /// Get a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Hostile => "Hostile",
            Self::Unfriendly => "Unfriendly",
            Self::Neutral => "Neutral",
            Self::Warm => "Warm",
            Self::Friendly => "Friendly",
        }
    }

    /// Whether this level allows dialogue to occur.
    pub fn allows_dialogue(&self) -> bool {
        !matches!(self, Self::Hostile)
    }

    /// Whether this level allows trade.
    pub fn allows_trade(&self) -> bool {
        matches!(self, Self::Neutral | Self::Warm | Self::Friendly)
    }

    /// Price multiplier for shops (friendlier = cheaper).
    pub fn price_multiplier(&self) -> f32 {
        match self {
            Self::Hostile => 2.0,
            Self::Unfriendly => 1.3,
            Self::Neutral => 1.0,
            Self::Warm => 0.9,
            Self::Friendly => 0.8,
        }
    }

    /// Information sharing level (0 = none, 1 = everything).
    pub fn info_sharing(&self) -> f32 {
        match self {
            Self::Hostile => 0.0,
            Self::Unfriendly => 0.1,
            Self::Neutral => 0.4,
            Self::Warm => 0.7,
            Self::Friendly => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Mood
// ---------------------------------------------------------------------------

/// Emotional states that affect dialogue behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoodType {
    /// Happy, cheerful, upbeat.
    Happy,
    /// Sad, melancholy, subdued.
    Sad,
    /// Angry, aggressive, short-tempered.
    Angry,
    /// Fearful, nervous, cautious.
    Afraid,
    /// Calm, relaxed, patient.
    Calm,
    /// Excited, enthusiastic, energetic.
    Excited,
    /// Suspicious, distrustful, guarded.
    Suspicious,
    /// Grateful, thankful, indebted.
    Grateful,
    /// Bored, disinterested, impatient.
    Bored,
    /// Neutral, no strong emotion.
    Neutral,
}

impl MoodType {
    /// How this mood affects disposition (positive = friendlier).
    pub fn disposition_modifier(&self) -> f32 {
        match self {
            Self::Happy => 10.0,
            Self::Sad => -5.0,
            Self::Angry => -15.0,
            Self::Afraid => -10.0,
            Self::Calm => 5.0,
            Self::Excited => 5.0,
            Self::Suspicious => -10.0,
            Self::Grateful => 15.0,
            Self::Bored => -3.0,
            Self::Neutral => 0.0,
        }
    }

    /// Dialogue tone modifiers for text generation hints.
    pub fn tone_hint(&self) -> &'static str {
        match self {
            Self::Happy => "cheerful",
            Self::Sad => "somber",
            Self::Angry => "aggressive",
            Self::Afraid => "nervous",
            Self::Calm => "measured",
            Self::Excited => "enthusiastic",
            Self::Suspicious => "guarded",
            Self::Grateful => "warm",
            Self::Bored => "disinterested",
            Self::Neutral => "neutral",
        }
    }
}

/// Dynamic mood state for an NPC.
#[derive(Debug, Clone)]
pub struct MoodModel {
    /// Current primary mood.
    pub primary_mood: MoodType,
    /// Intensity of the primary mood (0..1).
    pub intensity: f32,
    /// Secondary/mixed mood, if any.
    pub secondary_mood: Option<MoodType>,
    /// Base mood that the NPC trends toward (personality-based).
    pub base_mood: MoodType,
    /// Mood history for tracking patterns.
    mood_history: VecDeque<(MoodType, f64)>,
    /// Maximum history entries.
    max_history: usize,
}

impl MoodModel {
    /// Create a new mood model with a default base mood.
    pub fn new(base_mood: MoodType) -> Self {
        Self {
            primary_mood: base_mood,
            intensity: 0.5,
            secondary_mood: None,
            base_mood,
            mood_history: VecDeque::new(),
            max_history: 32,
        }
    }

    /// Set the primary mood with an intensity.
    pub fn set_mood(&mut self, mood: MoodType, intensity: f32, game_time: f64) {
        // Record old mood
        self.mood_history
            .push_back((self.primary_mood, game_time));
        if self.mood_history.len() > self.max_history {
            self.mood_history.pop_front();
        }

        self.primary_mood = mood;
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Apply a mood influence (blends with current mood).
    pub fn influence(&mut self, mood: MoodType, strength: f32, game_time: f64) {
        if mood == self.primary_mood {
            // Same mood — just increase intensity
            self.intensity = (self.intensity + strength * 0.5).clamp(0.0, 1.0);
        } else if self.intensity < strength {
            // New mood is stronger — switch
            self.secondary_mood = Some(self.primary_mood);
            self.set_mood(mood, strength, game_time);
        } else {
            // Weaker influence — set as secondary
            self.secondary_mood = Some(mood);
        }
    }

    /// Decay mood toward base mood over time.
    pub fn decay(&mut self, dt: f32) {
        self.intensity -= MOOD_DECAY_RATE * dt;
        if self.intensity <= 0.0 {
            self.primary_mood = self.base_mood;
            self.intensity = 0.3;
            self.secondary_mood = None;
        }
    }

    /// Get the effective disposition modifier.
    pub fn disposition_modifier(&self) -> f32 {
        self.primary_mood.disposition_modifier() * self.intensity
    }

    /// Get the most common recent mood.
    pub fn dominant_recent_mood(&self) -> MoodType {
        if self.mood_history.is_empty() {
            return self.base_mood;
        }

        let mut counts: HashMap<MoodType, usize> = HashMap::new();
        for (mood, _) in &self.mood_history {
            *counts.entry(*mood).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(mood, _)| mood)
            .unwrap_or(self.base_mood)
    }

    /// Check if the NPC has been in a negative mood recently.
    pub fn recently_negative(&self, window: f64, current_time: f64) -> bool {
        self.mood_history.iter().any(|(mood, time)| {
            (current_time - time) < window
                && matches!(mood, MoodType::Angry | MoodType::Afraid | MoodType::Sad)
        })
    }
}

// ---------------------------------------------------------------------------
// Personality
// ---------------------------------------------------------------------------

/// Personality traits that affect dialogue behavior.
#[derive(Debug, Clone)]
pub struct Personality {
    /// Openness to conversation (0 = closed, 1 = very open).
    pub openness: f32,
    /// Agreeableness (0 = confrontational, 1 = agreeable).
    pub agreeableness: f32,
    /// Assertiveness (0 = passive, 1 = assertive).
    pub assertiveness: f32,
    /// Humor level (0 = serious, 1 = comedic).
    pub humor: f32,
    /// Honesty (0 = deceitful, 1 = honest).
    pub honesty: f32,
    /// Bravery (0 = cowardly, 1 = courageous).
    pub bravery: f32,
    /// Curiosity (0 = incurious, 1 = highly curious).
    pub curiosity: f32,
    /// Greed (0 = generous, 1 = greedy).
    pub greed: f32,
}

impl Personality {
    /// Create a balanced/neutral personality.
    pub fn neutral() -> Self {
        Self {
            openness: 0.5,
            agreeableness: 0.5,
            assertiveness: 0.5,
            humor: 0.3,
            honesty: 0.7,
            bravery: 0.5,
            curiosity: 0.5,
            greed: 0.3,
        }
    }

    /// Create a friendly merchant personality.
    pub fn friendly_merchant() -> Self {
        Self {
            openness: 0.8,
            agreeableness: 0.7,
            assertiveness: 0.4,
            humor: 0.5,
            honesty: 0.6,
            bravery: 0.3,
            curiosity: 0.4,
            greed: 0.6,
        }
    }

    /// Create a gruff guard personality.
    pub fn guard() -> Self {
        Self {
            openness: 0.3,
            agreeableness: 0.3,
            assertiveness: 0.8,
            humor: 0.1,
            honesty: 0.7,
            bravery: 0.7,
            curiosity: 0.2,
            greed: 0.2,
        }
    }

    /// Create a scholarly sage personality.
    pub fn sage() -> Self {
        Self {
            openness: 0.9,
            agreeableness: 0.6,
            assertiveness: 0.3,
            humor: 0.3,
            honesty: 0.9,
            bravery: 0.4,
            curiosity: 0.9,
            greed: 0.1,
        }
    }

    /// Create a sneaky rogue personality.
    pub fn rogue() -> Self {
        Self {
            openness: 0.6,
            agreeableness: 0.4,
            assertiveness: 0.6,
            humor: 0.7,
            honesty: 0.2,
            bravery: 0.5,
            curiosity: 0.6,
            greed: 0.7,
        }
    }

    /// Calculate a base disposition modifier from personality.
    pub fn disposition_modifier(&self) -> f32 {
        (self.openness + self.agreeableness - 1.0) * 15.0
    }

    /// Whether this personality type would volunteer information.
    pub fn volunteers_info(&self, topic_relevance: f32) -> bool {
        (self.openness * 0.5 + self.curiosity * 0.3 + self.honesty * 0.2)
            * topic_relevance
            > 0.4
    }

    /// Whether this personality would attempt deception.
    pub fn would_deceive(&self, stakes: f32) -> bool {
        (1.0 - self.honesty) * stakes > 0.5
    }

    /// How patient this NPC is in conversation (higher = more patient).
    pub fn patience(&self) -> f32 {
        (self.agreeableness + self.openness) * 0.5
    }
}

impl Default for Personality {
    fn default() -> Self {
        Self::neutral()
    }
}

// ---------------------------------------------------------------------------
// Relationship
// ---------------------------------------------------------------------------

/// A directed relationship between two entities.
#[derive(Debug, Clone)]
pub struct Relationship {
    /// Source entity.
    pub from: u64,
    /// Target entity.
    pub to: u64,
    /// Affection/liking score (-100..100).
    pub affection: f32,
    /// Trust level (0..1).
    pub trust: f32,
    /// Familiarity / how well they know each other (0..1).
    pub familiarity: f32,
    /// Specific feelings or flags.
    pub tags: HashSet<String>,
    /// Number of positive interactions.
    pub positive_interactions: u32,
    /// Number of negative interactions.
    pub negative_interactions: u32,
    /// Last interaction timestamp.
    pub last_interaction: f64,
}

impl Relationship {
    /// Create a new default relationship.
    pub fn new(from: u64, to: u64) -> Self {
        Self {
            from,
            to,
            affection: 0.0,
            trust: 0.5,
            familiarity: 0.0,
            tags: HashSet::new(),
            positive_interactions: 0,
            negative_interactions: 0,
            last_interaction: 0.0,
        }
    }

    /// Record a positive interaction.
    pub fn positive_interaction(&mut self, amount: f32, game_time: f64) {
        self.affection = (self.affection + amount).clamp(MIN_DISPOSITION, MAX_DISPOSITION);
        self.trust = (self.trust + amount * 0.01).clamp(0.0, 1.0);
        self.familiarity = (self.familiarity + 0.05).clamp(0.0, 1.0);
        self.positive_interactions += 1;
        self.last_interaction = game_time;
    }

    /// Record a negative interaction.
    pub fn negative_interaction(&mut self, amount: f32, game_time: f64) {
        self.affection = (self.affection - amount).clamp(MIN_DISPOSITION, MAX_DISPOSITION);
        self.trust = (self.trust - amount * 0.02).clamp(0.0, 1.0);
        self.familiarity = (self.familiarity + 0.03).clamp(0.0, 1.0);
        self.negative_interactions += 1;
        self.last_interaction = game_time;
    }

    /// Get a disposition modifier from this relationship.
    pub fn disposition_modifier(&self) -> f32 {
        self.affection * 0.5 + (self.trust - 0.5) * 20.0
    }

    /// Whether there is a meaningful relationship.
    pub fn is_meaningful(&self) -> bool {
        self.familiarity > 0.1 || self.affection.abs() > 10.0
    }

    /// Add a relationship tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.insert(tag.into());
    }

    /// Check for a relationship tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Check if this is a positive relationship.
    pub fn is_positive(&self) -> bool {
        self.affection > 10.0 && self.trust > 0.3
    }

    /// Check if this is a negative relationship.
    pub fn is_negative(&self) -> bool {
        self.affection < -10.0 || self.trust < 0.2
    }

    /// Total number of interactions.
    pub fn total_interactions(&self) -> u32 {
        self.positive_interactions + self.negative_interactions
    }
}

// ---------------------------------------------------------------------------
// RelationshipGraph
// ---------------------------------------------------------------------------

/// Directed graph of relationships between entities.
pub struct RelationshipGraph {
    /// Edges: (from, to) -> Relationship.
    edges: HashMap<(u64, u64), Relationship>,
}

impl RelationshipGraph {
    /// Create a new empty relationship graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Get or create a relationship between two entities.
    pub fn get_or_create(&mut self, from: u64, to: u64) -> &mut Relationship {
        self.edges
            .entry((from, to))
            .or_insert_with(|| Relationship::new(from, to))
    }

    /// Get a relationship (immutable).
    pub fn get(&self, from: u64, to: u64) -> Option<&Relationship> {
        self.edges.get(&(from, to))
    }

    /// Get a relationship (mutable).
    pub fn get_mut(&mut self, from: u64, to: u64) -> Option<&mut Relationship> {
        self.edges.get_mut(&(from, to))
    }

    /// Get all relationships from a given entity.
    pub fn relationships_from(&self, from: u64) -> Vec<&Relationship> {
        self.edges
            .iter()
            .filter(|((f, _), _)| *f == from)
            .map(|(_, r)| r)
            .collect()
    }

    /// Get all relationships toward a given entity.
    pub fn relationships_toward(&self, to: u64) -> Vec<&Relationship> {
        self.edges
            .iter()
            .filter(|((_, t), _)| *t == to)
            .map(|(_, r)| r)
            .collect()
    }

    /// Get friends of an entity (positive affection).
    pub fn friends(&self, entity: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|((f, _), r)| *f == entity && r.is_positive())
            .map(|((_, t), _)| *t)
            .collect()
    }

    /// Get enemies of an entity (negative affection).
    pub fn enemies(&self, entity: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|((f, _), r)| *f == entity && r.is_negative())
            .map(|((_, t), _)| *t)
            .collect()
    }

    /// Check if two entities are mutual friends.
    pub fn mutual_friends(&self, a: u64, b: u64) -> bool {
        let a_to_b = self.get(a, b).map_or(false, |r| r.is_positive());
        let b_to_a = self.get(b, a).map_or(false, |r| r.is_positive());
        a_to_b && b_to_a
    }

    /// Remove all relationships involving an entity.
    pub fn remove_entity(&mut self, entity: u64) {
        self.edges
            .retain(|&(from, to), _| from != entity && to != entity);
    }

    /// Total number of relationships.
    pub fn relationship_count(&self) -> usize {
        self.edges.len()
    }

    /// Find common friends between two entities.
    pub fn common_friends(&self, a: u64, b: u64) -> Vec<u64> {
        let friends_a: HashSet<u64> = self.friends(a).into_iter().collect();
        let friends_b: HashSet<u64> = self.friends(b).into_iter().collect();
        friends_a.intersection(&friends_b).copied().collect()
    }

    /// Calculate social distance between two entities via mutual connections.
    /// Returns None if no path exists.
    pub fn social_distance(&self, from: u64, to: u64) -> Option<u32> {
        if from == to {
            return Some(0);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((from, 0u32));
        visited.insert(from);

        while let Some((current, depth)) = queue.pop_front() {
            if depth > 10 {
                break; // prevent excessive search
            }
            for friend in self.friends(current) {
                if friend == to {
                    return Some(depth + 1);
                }
                if !visited.contains(&friend) {
                    visited.insert(friend);
                    queue.push_back((friend, depth + 1));
                }
            }
        }

        None
    }
}

impl Default for RelationshipGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConversationTopic
// ---------------------------------------------------------------------------

/// A topic that can be discussed in conversation.
#[derive(Debug, Clone)]
pub struct ConversationTopic {
    /// Unique topic identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Category for grouping.
    pub category: TopicCategory,
    /// Relevance score (0..1) — how relevant this topic is right now.
    pub relevance: f32,
    /// Whether this topic has been discussed in the current conversation.
    pub discussed: bool,
    /// Number of times this topic has been discussed overall.
    pub times_discussed: u32,
    /// Required disposition level to discuss this topic.
    pub required_disposition: DispositionLevel,
    /// Required familiarity to bring up this topic.
    pub required_familiarity: f32,
    /// Tags for contextual filtering.
    pub tags: Vec<String>,
    /// Priority override (higher = more likely to be selected).
    pub priority: i32,
    /// Whether this topic is time-sensitive.
    pub time_sensitive: bool,
    /// Expiration time for time-sensitive topics.
    pub expires_at: Option<f64>,
}

/// Category of conversation topic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopicCategory {
    /// Greetings and small talk.
    SmallTalk,
    /// Quest-related information.
    Quest,
    /// Trading and commerce.
    Trade,
    /// Lore and world-building.
    Lore,
    /// Rumors and gossip.
    Rumor,
    /// Personal stories.
    Personal,
    /// Warnings and threats.
    Warning,
    /// Requests for help.
    Request,
    /// Teaching or instruction.
    Instruction,
    /// Farewell and goodbye.
    Farewell,
}

impl TopicCategory {
    /// Base relevance boost for this category.
    pub fn base_relevance(&self) -> f32 {
        match self {
            Self::SmallTalk => 0.3,
            Self::Quest => 0.8,
            Self::Trade => 0.5,
            Self::Lore => 0.4,
            Self::Rumor => 0.6,
            Self::Personal => 0.3,
            Self::Warning => 0.9,
            Self::Request => 0.7,
            Self::Instruction => 0.5,
            Self::Farewell => 0.2,
        }
    }
}

impl ConversationTopic {
    /// Create a new conversation topic.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: TopicCategory,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            relevance: category.base_relevance(),
            discussed: false,
            times_discussed: 0,
            required_disposition: DispositionLevel::Neutral,
            required_familiarity: 0.0,
            tags: Vec::new(),
            priority: 0,
            time_sensitive: false,
            expires_at: None,
        }
    }

    /// Set the required disposition.
    pub fn require_disposition(mut self, level: DispositionLevel) -> Self {
        self.required_disposition = level;
        self
    }

    /// Set the required familiarity.
    pub fn require_familiarity(mut self, level: f32) -> Self {
        self.required_familiarity = level;
        self
    }

    /// Mark as time-sensitive with an expiration.
    pub fn time_sensitive(mut self, expires_at: f64) -> Self {
        self.time_sensitive = true;
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if the topic is still valid.
    pub fn is_valid(&self, game_time: f64) -> bool {
        if let Some(exp) = self.expires_at {
            game_time < exp
        } else {
            true
        }
    }

    /// Check if this topic can be discussed given current context.
    pub fn can_discuss(
        &self,
        disposition: DispositionLevel,
        familiarity: f32,
        game_time: f64,
    ) -> bool {
        if !self.is_valid(game_time) {
            return false;
        }
        if self.discussed {
            return false;
        }

        // Check disposition requirement
        let disposition_ok = match self.required_disposition {
            DispositionLevel::Hostile => true,
            DispositionLevel::Unfriendly => {
                !matches!(disposition, DispositionLevel::Hostile)
            }
            DispositionLevel::Neutral => matches!(
                disposition,
                DispositionLevel::Neutral | DispositionLevel::Warm | DispositionLevel::Friendly
            ),
            DispositionLevel::Warm => {
                matches!(disposition, DispositionLevel::Warm | DispositionLevel::Friendly)
            }
            DispositionLevel::Friendly => {
                matches!(disposition, DispositionLevel::Friendly)
            }
        };

        disposition_ok && familiarity >= self.required_familiarity
    }

    /// Calculate the final selection score.
    pub fn selection_score(&self, disposition: &DispositionLevel, mood: &MoodType) -> f32 {
        let mut score = self.relevance;

        // Priority boost
        score += self.priority as f32 * 0.1;

        // Time-sensitive boost
        if self.time_sensitive {
            score += 0.3;
        }

        // Penalty for having been discussed many times
        score -= self.times_discussed as f32 * 0.1;

        // Mood-based adjustments
        match mood {
            MoodType::Angry => {
                if self.category == TopicCategory::SmallTalk {
                    score -= 0.3;
                }
                if self.category == TopicCategory::Warning {
                    score += 0.2;
                }
            }
            MoodType::Happy => {
                if self.category == TopicCategory::SmallTalk {
                    score += 0.2;
                }
            }
            MoodType::Suspicious => {
                if self.category == TopicCategory::Personal {
                    score -= 0.4;
                }
            }
            MoodType::Grateful => {
                if self.category == TopicCategory::Request {
                    score += 0.2;
                }
            }
            _ => {}
        }

        // Disposition adjustments
        match disposition {
            DispositionLevel::Friendly => {
                if self.category == TopicCategory::Personal {
                    score += 0.2;
                }
            }
            DispositionLevel::Hostile => {
                if self.category == TopicCategory::Farewell {
                    score += 0.5;
                }
            }
            _ => {}
        }

        score.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// ConversationContext
// ---------------------------------------------------------------------------

/// Tracking state for an ongoing conversation.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// Participants in the conversation.
    pub participants: Vec<u64>,
    /// Topics discussed so far (topic IDs).
    pub discussed_topics: Vec<String>,
    /// Conversation history entries.
    pub history: VecDeque<ConversationEntry>,
    /// Current emotional tone of the conversation.
    pub tone: ConversationTone,
    /// Number of exchanges so far.
    pub exchange_count: u32,
    /// Start time of the conversation.
    pub start_time: f64,
    /// Whether the conversation has ended.
    pub ended: bool,
    /// Why the conversation ended.
    pub end_reason: Option<String>,
    /// Contextual data (key-value pairs for game-specific info).
    pub context_data: HashMap<String, String>,
}

/// A single entry in the conversation history.
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    /// Who spoke.
    pub speaker: u64,
    /// Topic discussed.
    pub topic_id: String,
    /// Sentiment of this exchange (-1..1).
    pub sentiment: f32,
    /// Timestamp.
    pub timestamp: f64,
}

/// Overall tone of the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationTone {
    Friendly,
    Neutral,
    Tense,
    Hostile,
    Formal,
    Casual,
}

impl ConversationContext {
    /// Create a new conversation context.
    pub fn new(participants: Vec<u64>, start_time: f64) -> Self {
        Self {
            participants,
            discussed_topics: Vec::new(),
            history: VecDeque::new(),
            tone: ConversationTone::Neutral,
            exchange_count: 0,
            start_time,
            ended: false,
            end_reason: None,
            context_data: HashMap::new(),
        }
    }

    /// Record a conversation exchange.
    pub fn record_exchange(
        &mut self,
        speaker: u64,
        topic_id: impl Into<String>,
        sentiment: f32,
        timestamp: f64,
    ) {
        let topic_id = topic_id.into();

        if !self.discussed_topics.contains(&topic_id) {
            self.discussed_topics.push(topic_id.clone());
        }

        self.history.push_back(ConversationEntry {
            speaker,
            topic_id,
            sentiment,
            timestamp,
        });

        if self.history.len() > MAX_CONVERSATION_HISTORY {
            self.history.pop_front();
        }

        self.exchange_count += 1;
        self.update_tone();
    }

    /// Update the conversation tone based on recent history.
    fn update_tone(&mut self) {
        let recent: Vec<f32> = self
            .history
            .iter()
            .rev()
            .take(5)
            .map(|e| e.sentiment)
            .collect();

        if recent.is_empty() {
            return;
        }

        let avg_sentiment: f32 = recent.iter().sum::<f32>() / recent.len() as f32;

        self.tone = if avg_sentiment > 0.5 {
            ConversationTone::Friendly
        } else if avg_sentiment < -0.5 {
            ConversationTone::Hostile
        } else if avg_sentiment < -0.2 {
            ConversationTone::Tense
        } else {
            ConversationTone::Neutral
        };
    }

    /// Check if a topic has been discussed.
    pub fn was_discussed(&self, topic_id: &str) -> bool {
        self.discussed_topics.iter().any(|t| t == topic_id)
    }

    /// End the conversation.
    pub fn end(&mut self, reason: impl Into<String>) {
        self.ended = true;
        self.end_reason = Some(reason.into());
    }

    /// Get the average sentiment of the conversation.
    pub fn average_sentiment(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|e| e.sentiment).sum();
        sum / self.history.len() as f32
    }

    /// Duration of the conversation in game time.
    pub fn duration(&self, current_time: f64) -> f64 {
        current_time - self.start_time
    }

    /// Set a context data value.
    pub fn set_data(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.context_data.insert(key.into(), value.into());
    }

    /// Get a context data value.
    pub fn get_data(&self, key: &str) -> Option<&str> {
        self.context_data.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// DialogueDecisionMaker
// ---------------------------------------------------------------------------

/// The main NPC dialogue decision-making component.
pub struct DialogueDecisionMaker {
    /// Entity ID of the NPC.
    pub entity_id: u64,
    /// Personality traits.
    pub personality: Personality,
    /// Current mood.
    pub mood: MoodModel,
    /// Available conversation topics.
    topics: Vec<ConversationTopic>,
    /// Faction ID this NPC belongs to.
    pub faction_id: Option<u32>,
    /// Cached disposition scores toward other entities.
    disposition_cache: HashMap<u64, f32>,
    /// Active conversation, if any.
    active_conversation: Option<ConversationContext>,
}

impl DialogueDecisionMaker {
    /// Create a new dialogue decision maker.
    pub fn new(entity_id: u64, personality: Personality) -> Self {
        let base_mood = MoodType::Neutral;
        Self {
            entity_id,
            personality,
            mood: MoodModel::new(base_mood),
            topics: Vec::new(),
            faction_id: None,
            disposition_cache: HashMap::new(),
            active_conversation: None,
        }
    }

    /// Set the faction.
    pub fn with_faction(mut self, faction_id: u32) -> Self {
        self.faction_id = Some(faction_id);
        self
    }

    /// Add a conversation topic.
    pub fn add_topic(&mut self, topic: ConversationTopic) {
        if self.topics.len() < MAX_TOPICS_PER_NPC {
            self.topics.push(topic);
        }
    }

    /// Remove a topic by ID.
    pub fn remove_topic(&mut self, topic_id: &str) {
        self.topics.retain(|t| t.id != topic_id);
    }

    /// Calculate the disposition toward another entity.
    pub fn calculate_disposition(
        &self,
        target: u64,
        reputation: f32,
        relationship: Option<&Relationship>,
    ) -> f32 {
        let rep_component = reputation * REPUTATION_WEIGHT;
        let rel_component = relationship
            .map_or(0.0, |r| r.disposition_modifier())
            * RELATIONSHIP_WEIGHT;
        let mood_component = self.mood.disposition_modifier() * MOOD_WEIGHT;
        let personality_component =
            self.personality.disposition_modifier() * PERSONALITY_WEIGHT;

        let raw = rep_component + rel_component + mood_component + personality_component;
        raw.clamp(MIN_DISPOSITION, MAX_DISPOSITION)
    }

    /// Get the cached disposition toward a target.
    pub fn get_disposition(&self, target: u64) -> f32 {
        self.disposition_cache
            .get(&target)
            .copied()
            .unwrap_or(NEUTRAL_DISPOSITION)
    }

    /// Update the cached disposition.
    pub fn update_disposition(
        &mut self,
        target: u64,
        reputation: f32,
        relationship: Option<&Relationship>,
    ) {
        let score = self.calculate_disposition(target, reputation, relationship);
        self.disposition_cache.insert(target, score);
    }

    /// Get the disposition level toward a target.
    pub fn disposition_level(&self, target: u64) -> DispositionLevel {
        DispositionLevel::from_score(self.get_disposition(target))
    }

    /// Select the best conversation topic for the current context.
    pub fn select_topic(
        &self,
        target: u64,
        familiarity: f32,
        game_time: f64,
    ) -> Option<&ConversationTopic> {
        let disposition = self.disposition_level(target);

        let mut best: Option<(&ConversationTopic, f32)> = None;

        for topic in &self.topics {
            if !topic.can_discuss(disposition, familiarity, game_time) {
                continue;
            }

            let score = topic.selection_score(&disposition, &self.mood.primary_mood);

            if best.as_ref().map_or(true, |(_, best_score)| score > *best_score) {
                best = Some((topic, score));
            }
        }

        best.map(|(topic, _)| topic)
    }

    /// Select multiple topics ranked by score.
    pub fn select_topics_ranked(
        &self,
        target: u64,
        familiarity: f32,
        game_time: f64,
        max_count: usize,
    ) -> Vec<(&ConversationTopic, f32)> {
        let disposition = self.disposition_level(target);

        let mut scored: Vec<(&ConversationTopic, f32)> = self
            .topics
            .iter()
            .filter(|t| t.can_discuss(disposition, familiarity, game_time))
            .map(|t| {
                let score = t.selection_score(&disposition, &self.mood.primary_mood);
                (t, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_count);
        scored
    }

    /// Start a conversation with a target entity.
    pub fn start_conversation(&mut self, target: u64, game_time: f64) -> bool {
        if self.active_conversation.is_some() {
            return false;
        }

        let disposition = self.disposition_level(target);
        if !disposition.allows_dialogue() {
            return false;
        }

        self.active_conversation = Some(ConversationContext::new(
            vec![self.entity_id, target],
            game_time,
        ));
        true
    }

    /// End the current conversation.
    pub fn end_conversation(&mut self, reason: impl Into<String>) {
        if let Some(ref mut conv) = self.active_conversation {
            conv.end(reason);
        }
        self.active_conversation = None;
    }

    /// Get the active conversation context.
    pub fn conversation(&self) -> Option<&ConversationContext> {
        self.active_conversation.as_ref()
    }

    /// Get mutable conversation context.
    pub fn conversation_mut(&mut self) -> Option<&mut ConversationContext> {
        self.active_conversation.as_mut()
    }

    /// Update mood decay.
    pub fn update(&mut self, dt: f32) {
        self.mood.decay(dt);
    }

    /// Get number of available topics.
    pub fn topic_count(&self) -> usize {
        self.topics.len()
    }

    /// Mark a topic as discussed.
    pub fn mark_discussed(&mut self, topic_id: &str) {
        if let Some(topic) = self.topics.iter_mut().find(|t| t.id == topic_id) {
            topic.discussed = true;
            topic.times_discussed += 1;
        }
    }

    /// Reset all topics to undiscussed (for a new conversation session).
    pub fn reset_topics(&mut self) {
        for topic in &mut self.topics {
            topic.discussed = false;
        }
    }

    /// Should the NPC initiate conversation with a target?
    pub fn should_initiate(&self, target: u64, game_time: f64) -> bool {
        let disposition = self.disposition_level(target);
        if !disposition.allows_dialogue() {
            return false;
        }

        // Check if there are high-priority topics
        let has_urgent = self.topics.iter().any(|t| {
            t.priority > 5
                && !t.discussed
                && t.is_valid(game_time)
        });

        // Personality-driven initiative
        let initiative_chance = self.personality.openness * 0.3
            + self.personality.assertiveness * 0.2
            + if has_urgent { 0.5 } else { 0.0 };

        initiative_chance > 0.4
    }

    /// Get a dialogue response style hint based on current state.
    pub fn response_style(&self, target: u64) -> DialogueStyle {
        let disposition = self.disposition_level(target);
        let mood = &self.mood;

        DialogueStyle {
            tone: mood.primary_mood.tone_hint().to_string(),
            verbosity: self.personality.openness,
            formality: 1.0 - self.personality.humor,
            honesty: self.personality.honesty,
            disposition: disposition,
            patience: self.personality.patience(),
        }
    }
}

/// Hints for dialogue text generation.
#[derive(Debug, Clone)]
pub struct DialogueStyle {
    /// Emotional tone.
    pub tone: String,
    /// How verbose the response should be (0 = curt, 1 = elaborate).
    pub verbosity: f32,
    /// How formal (0 = casual, 1 = formal).
    pub formality: f32,
    /// How honest (0 = evasive/lying, 1 = completely honest).
    pub honesty: f32,
    /// Current disposition.
    pub disposition: DispositionLevel,
    /// Patience level.
    pub patience: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disposition_levels() {
        assert_eq!(
            DispositionLevel::from_score(50.0),
            DispositionLevel::Friendly
        );
        assert_eq!(
            DispositionLevel::from_score(0.0),
            DispositionLevel::Neutral
        );
        assert_eq!(
            DispositionLevel::from_score(-50.0),
            DispositionLevel::Hostile
        );
    }

    #[test]
    fn test_mood_model() {
        let mut mood = MoodModel::new(MoodType::Calm);
        assert_eq!(mood.primary_mood, MoodType::Calm);

        mood.set_mood(MoodType::Angry, 0.8, 10.0);
        assert_eq!(mood.primary_mood, MoodType::Angry);
        assert_eq!(mood.intensity, 0.8);

        // Decay toward base
        for _ in 0..100 {
            mood.decay(1.0);
        }
        assert_eq!(mood.primary_mood, MoodType::Calm);
    }

    #[test]
    fn test_relationship() {
        let mut rel = Relationship::new(1, 2);
        assert_eq!(rel.affection, 0.0);

        rel.positive_interaction(10.0, 1.0);
        assert!(rel.affection > 0.0);
        assert!(rel.familiarity > 0.0);

        rel.negative_interaction(20.0, 2.0);
        assert!(rel.affection < 0.0);
    }

    #[test]
    fn test_topic_selection() {
        let mut maker = DialogueDecisionMaker::new(1, Personality::friendly_merchant());
        maker.disposition_cache.insert(2, 30.0); // friendly

        maker.add_topic(ConversationTopic::new("greeting", "Hello!", TopicCategory::SmallTalk));
        maker.add_topic(
            ConversationTopic::new("quest_info", "The Dragon", TopicCategory::Quest)
                .require_disposition(DispositionLevel::Warm),
        );

        let topic = maker.select_topic(2, 0.5, 0.0);
        assert!(topic.is_some());
    }

    #[test]
    fn test_conversation_context() {
        let mut ctx = ConversationContext::new(vec![1, 2], 0.0);
        ctx.record_exchange(1, "greeting", 0.5, 1.0);
        ctx.record_exchange(2, "quest_info", 0.3, 2.0);

        assert!(ctx.was_discussed("greeting"));
        assert!(!ctx.was_discussed("farewell"));
        assert_eq!(ctx.exchange_count, 2);
        assert!(ctx.average_sentiment() > 0.0);
    }

    #[test]
    fn test_social_distance() {
        let mut graph = RelationshipGraph::new();
        // A -> B (friends)
        graph.get_or_create(1, 2).positive_interaction(50.0, 0.0);
        // B -> C (friends)
        graph.get_or_create(2, 3).positive_interaction(50.0, 0.0);

        assert_eq!(graph.social_distance(1, 2), Some(1));
        assert_eq!(graph.social_distance(1, 3), Some(2));
    }
}
