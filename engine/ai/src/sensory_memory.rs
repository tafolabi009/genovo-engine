//! Extended sensory memory with stimulus aging and attention.
//!
//! Provides:
//! - **Stimulus aging**: decay curves for memory degradation over time
//! - **Stimulus priority ranking**: importance-based ordering
//! - **Attention system**: focus on the most important stimulus
//! - **Multi-modal fusion**: combine sight + sound for confidence
//! - **Memory capacity limits**: forgetting least important memories
//! - **Spatial memory**: remember where things were last seen
//! - **Stimulus categorization**: threats, allies, objects, sounds
//! - **Confidence tracking**: how certain the AI is about each memory
//! - **ECS integration**: `SensoryMemoryComponent`, `SensoryMemorySystem`
//!
//! # Design
//!
//! The [`SensoryMemory`] stores [`StimulusMemory`] entries that decay over
//! time according to configurable [`DecayCurve`]s. The [`AttentionSystem`]
//! selects the most important stimulus to focus on. Multi-modal fusion
//! combines evidence from different senses to increase confidence.

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of memories.
pub const MAX_MEMORIES: usize = 64;
/// Default memory decay rate.
pub const DEFAULT_DECAY_RATE: f32 = 0.1;
/// Minimum confidence before a memory is forgotten.
pub const MIN_CONFIDENCE: f32 = 0.05;
/// Default attention span (seconds before attention shifts).
pub const DEFAULT_ATTENTION_SPAN: f32 = 3.0;
/// Maximum sight range for confidence boost.
pub const MAX_SIGHT_RANGE: f32 = 50.0;
/// Maximum hearing range.
pub const MAX_HEARING_RANGE: f32 = 30.0;
/// Visual confidence weight (for multi-modal fusion).
pub const VISUAL_WEIGHT: f32 = 0.6;
/// Auditory confidence weight.
pub const AUDITORY_WEIGHT: f32 = 0.25;
/// Touch/proximity confidence weight.
pub const TOUCH_WEIGHT: f32 = 0.15;
/// Confidence boost when multiple senses agree.
pub const FUSION_BONUS: f32 = 0.2;
/// Priority boost for threats.
pub const THREAT_PRIORITY_BOOST: f32 = 50.0;
/// Priority boost for recently seen stimuli.
pub const RECENCY_PRIORITY_BOOST: f32 = 20.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// StimulusId
// ---------------------------------------------------------------------------

/// Unique identifier for a stimulus source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StimulusId(pub u64);

// ---------------------------------------------------------------------------
// StimulusCategory
// ---------------------------------------------------------------------------

/// Category of a sensory stimulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StimulusCategory {
    /// An enemy/hostile entity.
    Threat,
    /// A friendly/allied entity.
    Ally,
    /// A neutral entity.
    Neutral,
    /// An environmental sound.
    EnvironmentalSound,
    /// A projectile or attack.
    Projectile,
    /// A static object of interest.
    Object,
    /// A point of interest (e.g., noise source).
    PointOfInterest,
    /// Food or resource.
    Resource,
    /// An unknown stimulus.
    Unknown,
}

impl StimulusCategory {
    /// Base priority for this category.
    pub fn base_priority(&self) -> f32 {
        match self {
            Self::Threat => 80.0,
            Self::Projectile => 90.0,
            Self::Ally => 30.0,
            Self::Neutral => 20.0,
            Self::EnvironmentalSound => 40.0,
            Self::Object => 10.0,
            Self::PointOfInterest => 50.0,
            Self::Resource => 25.0,
            Self::Unknown => 60.0,
        }
    }

    /// Whether this category represents a potential danger.
    pub fn is_dangerous(&self) -> bool {
        matches!(self, Self::Threat | Self::Projectile)
    }
}

// ---------------------------------------------------------------------------
// SenseType
// ---------------------------------------------------------------------------

/// Type of sense that detected the stimulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SenseType {
    /// Visual detection.
    Sight,
    /// Auditory detection.
    Hearing,
    /// Physical contact.
    Touch,
    /// Smell.
    Smell,
    /// Radar/sixth sense.
    Radar,
    /// Direct knowledge (e.g., communicated by an ally).
    Knowledge,
}

impl SenseType {
    /// Base confidence for a detection by this sense.
    pub fn base_confidence(&self) -> f32 {
        match self {
            Self::Sight => 0.8,
            Self::Hearing => 0.5,
            Self::Touch => 0.95,
            Self::Smell => 0.4,
            Self::Radar => 0.7,
            Self::Knowledge => 0.9,
        }
    }

    /// Weight for multi-modal fusion.
    pub fn fusion_weight(&self) -> f32 {
        match self {
            Self::Sight => VISUAL_WEIGHT,
            Self::Hearing => AUDITORY_WEIGHT,
            Self::Touch => TOUCH_WEIGHT,
            Self::Smell => 0.1,
            Self::Radar => 0.3,
            Self::Knowledge => 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// DecayCurve
// ---------------------------------------------------------------------------

/// How a memory decays over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayCurve {
    /// Linear decay: confidence -= rate * dt.
    Linear,
    /// Exponential decay: confidence *= (1 - rate * dt).
    Exponential,
    /// Step function: full confidence until timeout, then zero.
    Step { timeout: u32 }, // in deciseconds
    /// Logarithmic decay (fast initially, then slow).
    Logarithmic,
    /// No decay (perfect memory).
    None,
}

impl DecayCurve {
    /// Apply decay to a confidence value.
    pub fn apply(&self, confidence: f32, rate: f32, dt: f32) -> f32 {
        match self {
            DecayCurve::Linear => (confidence - rate * dt).max(0.0),
            DecayCurve::Exponential => confidence * (1.0 - rate * dt).max(0.0),
            DecayCurve::Step { timeout } => {
                // This is handled externally via age checking
                confidence
            }
            DecayCurve::Logarithmic => {
                let decay = rate * dt / (1.0 + confidence * 5.0);
                (confidence - decay).max(0.0)
            }
            DecayCurve::None => confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// StimulusMemory
// ---------------------------------------------------------------------------

/// A stored memory of a sensory stimulus.
#[derive(Debug, Clone)]
pub struct StimulusMemory {
    /// Source entity/stimulus ID.
    pub source_id: StimulusId,
    /// Category of the stimulus.
    pub category: StimulusCategory,
    /// Last known position.
    pub last_known_position: Vec3,
    /// Last known velocity (if moving).
    pub last_known_velocity: Vec3,
    /// Predicted current position (extrapolated).
    pub predicted_position: Vec3,
    /// Time since last sensory update (seconds).
    pub age: f32,
    /// Total lifetime of this memory (seconds).
    pub lifetime: f32,
    /// Confidence level (0..1).
    pub confidence: f32,
    /// Peak confidence ever recorded.
    pub peak_confidence: f32,
    /// Decay curve for this memory.
    pub decay_curve: DecayCurve,
    /// Decay rate.
    pub decay_rate: f32,
    /// Priority (for attention ranking).
    pub priority: f32,
    /// Senses that have detected this stimulus.
    pub detected_by: HashMap<SenseType, f32>,
    /// Number of times this stimulus has been re-detected.
    pub detection_count: u32,
    /// Whether this memory is currently being attended to.
    pub attended: bool,
    /// Whether the stimulus is currently being perceived (real-time).
    pub currently_perceived: bool,
    /// Threat level (0 = no threat, 1 = maximum threat).
    pub threat_level: f32,
    /// Additional data tags.
    pub tags: Vec<String>,
    /// Distance to the observer at last detection.
    pub last_distance: f32,
    /// Direction from observer at last detection.
    pub last_direction: Vec3,
}

impl StimulusMemory {
    /// Create a new stimulus memory.
    pub fn new(
        source_id: StimulusId,
        category: StimulusCategory,
        position: Vec3,
        sense: SenseType,
    ) -> Self {
        let confidence = sense.base_confidence();
        let mut detected_by = HashMap::new();
        detected_by.insert(sense, confidence);

        Self {
            source_id,
            category,
            last_known_position: position,
            last_known_velocity: Vec3::ZERO,
            predicted_position: position,
            age: 0.0,
            lifetime: 0.0,
            confidence,
            peak_confidence: confidence,
            decay_curve: DecayCurve::Exponential,
            decay_rate: DEFAULT_DECAY_RATE,
            priority: category.base_priority(),
            detected_by,
            detection_count: 1,
            attended: false,
            currently_perceived: true,
            threat_level: if category.is_dangerous() { 0.8 } else { 0.0 },
            tags: Vec::new(),
            last_distance: 0.0,
            last_direction: Vec3::ZERO,
        }
    }

    /// Update with a new detection from a sense.
    pub fn update_detection(
        &mut self,
        position: Vec3,
        sense: SenseType,
        distance: f32,
        direction: Vec3,
    ) {
        let old_pos = self.last_known_position;
        self.last_known_position = position;
        self.last_known_velocity = position - old_pos; // Simplified
        self.predicted_position = position;
        self.last_distance = distance;
        self.last_direction = direction;
        self.detection_count += 1;
        self.currently_perceived = true;

        // Multi-modal fusion
        let sense_confidence = sense.base_confidence() * (1.0 - (distance / MAX_SIGHT_RANGE).min(1.0));
        self.detected_by.insert(sense, sense_confidence);
        self.fuse_confidence();

        self.age = 0.0;
    }

    /// Fuse confidence from multiple senses.
    fn fuse_confidence(&mut self) {
        if self.detected_by.len() <= 1 {
            self.confidence = self.detected_by.values().next().copied().unwrap_or(0.0);
        } else {
            // Weighted average + fusion bonus
            let mut weighted_sum = 0.0_f32;
            let mut total_weight = 0.0_f32;

            for (sense, conf) in &self.detected_by {
                let weight = sense.fusion_weight();
                weighted_sum += conf * weight;
                total_weight += weight;
            }

            let base_confidence = if total_weight > EPSILON {
                weighted_sum / total_weight
            } else {
                0.0
            };

            // Bonus for multiple senses agreeing
            let modal_count = self.detected_by.len() as f32;
            let bonus = FUSION_BONUS * (modal_count - 1.0).min(2.0);

            self.confidence = (base_confidence + bonus).min(1.0);
        }

        self.peak_confidence = self.peak_confidence.max(self.confidence);
    }

    /// Apply time-based decay.
    pub fn decay(&mut self, dt: f32) {
        if self.currently_perceived {
            return; // Don't decay while actively perceived
        }

        self.age += dt;
        self.lifetime += dt;

        // Apply decay curve
        self.confidence = self.decay_curve.apply(self.confidence, self.decay_rate, dt);

        // Extrapolate predicted position
        self.predicted_position = self.last_known_position + self.last_known_velocity * self.age;

        // Decay individual sense confidences
        for conf in self.detected_by.values_mut() {
            *conf *= (1.0 - self.decay_rate * dt * 0.5).max(0.0);
        }
        self.detected_by.retain(|_, conf| *conf > MIN_CONFIDENCE);

        // Update priority based on age
        self.update_priority();
    }

    /// Update the priority of this memory.
    fn update_priority(&mut self) {
        let mut priority = self.category.base_priority();

        // Boost for threats
        if self.category.is_dangerous() {
            priority += THREAT_PRIORITY_BOOST * self.threat_level;
        }

        // Recency boost
        let recency = 1.0 / (1.0 + self.age * 0.5);
        priority += RECENCY_PRIORITY_BOOST * recency;

        // Confidence factor
        priority *= self.confidence;

        // Detection count factor (more detections = more important)
        priority *= 1.0 + (self.detection_count as f32 * 0.1).min(1.0);

        self.priority = priority;
    }

    /// Check if this memory should be forgotten.
    pub fn should_forget(&self) -> bool {
        self.confidence < MIN_CONFIDENCE
    }

    /// Mark as no longer currently perceived.
    pub fn mark_not_perceived(&mut self) {
        self.currently_perceived = false;
    }

    /// Get the estimated reliability of the position.
    pub fn position_reliability(&self) -> f32 {
        if self.currently_perceived {
            self.confidence
        } else {
            // Decrease reliability with age (extrapolated position is less reliable)
            self.confidence * (1.0 / (1.0 + self.age * 0.3))
        }
    }
}

// ---------------------------------------------------------------------------
// AttentionFocus
// ---------------------------------------------------------------------------

/// The AI's current attention focus.
#[derive(Debug, Clone)]
pub struct AttentionFocus {
    /// Currently attended stimulus (if any).
    pub current_focus: Option<StimulusId>,
    /// Time spent on current focus.
    pub focus_time: f32,
    /// Maximum attention span before shifting.
    pub attention_span: f32,
    /// Focus locked (won't shift automatically).
    pub locked: bool,
    /// Previous focus (for tracking shifts).
    pub previous_focus: Option<StimulusId>,
    /// Focus history (last N foci).
    focus_history: Vec<StimulusId>,
    /// Distraction threshold (priority difference needed to shift attention).
    pub distraction_threshold: f32,
}

impl Default for AttentionFocus {
    fn default() -> Self {
        Self {
            current_focus: None,
            focus_time: 0.0,
            attention_span: DEFAULT_ATTENTION_SPAN,
            locked: false,
            previous_focus: None,
            focus_history: Vec::new(),
            distraction_threshold: 20.0,
        }
    }
}

impl AttentionFocus {
    /// Try to shift attention to a new stimulus.
    pub fn try_shift(&mut self, new_focus: StimulusId, new_priority: f32, current_priority: f32) -> bool {
        if self.locked {
            return false;
        }

        let should_shift = match self.current_focus {
            None => true,
            Some(current) => {
                if current == new_focus {
                    return false;
                }
                // Shift if new priority significantly exceeds current
                new_priority > current_priority + self.distraction_threshold
                    || self.focus_time > self.attention_span
            }
        };

        if should_shift {
            self.previous_focus = self.current_focus;
            self.current_focus = Some(new_focus);
            self.focus_time = 0.0;
            self.focus_history.push(new_focus);
            if self.focus_history.len() > 10 {
                self.focus_history.remove(0);
            }
            true
        } else {
            false
        }
    }

    /// Update attention (increment focus time).
    pub fn update(&mut self, dt: f32) {
        self.focus_time += dt;
    }

    /// Lock attention on current focus.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlock attention.
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Clear attention focus.
    pub fn clear(&mut self) {
        self.previous_focus = self.current_focus;
        self.current_focus = None;
        self.focus_time = 0.0;
    }
}

// ---------------------------------------------------------------------------
// SensoryMemory
// ---------------------------------------------------------------------------

/// Main sensory memory store for an AI agent.
#[derive(Debug)]
pub struct SensoryMemory {
    /// All stored memories.
    memories: HashMap<StimulusId, StimulusMemory>,
    /// Attention system.
    pub attention: AttentionFocus,
    /// Maximum memory capacity.
    pub capacity: usize,
    /// Default decay curve for new memories.
    pub default_decay: DecayCurve,
    /// Default decay rate.
    pub default_decay_rate: f32,
    /// Observer position (for distance calculations).
    pub observer_position: Vec3,
    /// Observer forward direction.
    pub observer_forward: Vec3,
    /// Sorted priority list (refreshed each update).
    priority_list: Vec<(StimulusId, f32)>,
    /// Events since last query.
    events: Vec<SensoryEvent>,
}

/// Events from the sensory memory system.
#[derive(Debug, Clone)]
pub enum SensoryEvent {
    /// A new stimulus was detected.
    NewStimulus { id: StimulusId, category: StimulusCategory },
    /// A stimulus was re-detected (updated).
    StimulusUpdated { id: StimulusId },
    /// A stimulus was forgotten.
    StimulusForgotten { id: StimulusId },
    /// Attention shifted to a new stimulus.
    AttentionShifted { from: Option<StimulusId>, to: StimulusId },
    /// A threat was detected.
    ThreatDetected { id: StimulusId, threat_level: f32 },
    /// Memory capacity reached (something was forgotten to make room).
    MemoryOverflow { forgotten: StimulusId },
}

impl SensoryMemory {
    /// Create a new sensory memory.
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            attention: AttentionFocus::default(),
            capacity: MAX_MEMORIES,
            default_decay: DecayCurve::Exponential,
            default_decay_rate: DEFAULT_DECAY_RATE,
            observer_position: Vec3::ZERO,
            observer_forward: Vec3::new(0.0, 0.0, 1.0),
            priority_list: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Record a new stimulus detection.
    pub fn detect(
        &mut self,
        source_id: StimulusId,
        category: StimulusCategory,
        position: Vec3,
        sense: SenseType,
    ) {
        let distance = (position - self.observer_position).length();
        let direction = (position - self.observer_position).normalize_or_zero();

        if let Some(memory) = self.memories.get_mut(&source_id) {
            // Update existing memory
            memory.update_detection(position, sense, distance, direction);
            self.events.push(SensoryEvent::StimulusUpdated { id: source_id });
        } else {
            // Check capacity
            if self.memories.len() >= self.capacity {
                self.forget_least_important();
            }

            // Create new memory
            let mut memory = StimulusMemory::new(source_id, category, position, sense);
            memory.decay_curve = self.default_decay;
            memory.decay_rate = self.default_decay_rate;
            memory.last_distance = distance;
            memory.last_direction = direction;

            self.memories.insert(source_id, memory);
            self.events.push(SensoryEvent::NewStimulus { id: source_id, category });

            if category.is_dangerous() {
                self.events.push(SensoryEvent::ThreatDetected {
                    id: source_id,
                    threat_level: 0.8,
                });
            }
        }
    }

    /// Forget the least important memory.
    fn forget_least_important(&mut self) {
        if let Some((&least_id, _)) = self.memories.iter()
            .filter(|(_, m)| !m.attended && !m.currently_perceived)
            .min_by(|a, b| a.1.priority.partial_cmp(&b.1.priority).unwrap_or(std::cmp::Ordering::Equal))
        {
            self.memories.remove(&least_id);
            self.events.push(SensoryEvent::MemoryOverflow { forgotten: least_id });
        }
    }

    /// Mark a stimulus as no longer perceived.
    pub fn mark_not_perceived(&mut self, source_id: StimulusId) {
        if let Some(memory) = self.memories.get_mut(&source_id) {
            memory.mark_not_perceived();
        }
    }

    /// Mark all stimuli as not currently perceived (call at start of perception update).
    pub fn mark_all_not_perceived(&mut self) {
        for memory in self.memories.values_mut() {
            memory.mark_not_perceived();
        }
    }

    /// Update: decay memories, update attention, clean up forgotten.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();

        // Decay all memories
        for memory in self.memories.values_mut() {
            memory.decay(dt);
        }

        // Forget memories below threshold
        let forgotten: Vec<StimulusId> = self.memories.iter()
            .filter(|(_, m)| m.should_forget())
            .map(|(id, _)| *id)
            .collect();
        for id in forgotten {
            self.memories.remove(&id);
            self.events.push(SensoryEvent::StimulusForgotten { id });
        }

        // Rebuild priority list
        self.priority_list.clear();
        for (id, memory) in &self.memories {
            self.priority_list.push((*id, memory.priority));
        }
        self.priority_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update attention
        self.attention.update(dt);
        if let Some(&(top_id, top_priority)) = self.priority_list.first() {
            let current_priority = self.attention.current_focus
                .and_then(|id| self.memories.get(&id))
                .map(|m| m.priority)
                .unwrap_or(0.0);

            if self.attention.try_shift(top_id, top_priority, current_priority) {
                // Mark attended
                for memory in self.memories.values_mut() {
                    memory.attended = false;
                }
                if let Some(memory) = self.memories.get_mut(&top_id) {
                    memory.attended = true;
                }
                self.events.push(SensoryEvent::AttentionShifted {
                    from: self.attention.previous_focus,
                    to: top_id,
                });
            }
        }
    }

    /// Get a memory by stimulus ID.
    pub fn get_memory(&self, id: StimulusId) -> Option<&StimulusMemory> {
        self.memories.get(&id)
    }

    /// Get all memories.
    pub fn all_memories(&self) -> impl Iterator<Item = &StimulusMemory> {
        self.memories.values()
    }

    /// Get memories sorted by priority (highest first).
    pub fn by_priority(&self) -> &[(StimulusId, f32)] {
        &self.priority_list
    }

    /// Get all threat memories.
    pub fn threats(&self) -> Vec<&StimulusMemory> {
        self.memories.values()
            .filter(|m| m.category.is_dangerous() && m.confidence > MIN_CONFIDENCE)
            .collect()
    }

    /// Get the currently attended memory.
    pub fn attended_memory(&self) -> Option<&StimulusMemory> {
        self.attention.current_focus
            .and_then(|id| self.memories.get(&id))
    }

    /// Get the most threatening memory.
    pub fn highest_threat(&self) -> Option<&StimulusMemory> {
        self.memories.values()
            .filter(|m| m.category.is_dangerous())
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the number of stored memories.
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get the number of currently perceived stimuli.
    pub fn perceived_count(&self) -> usize {
        self.memories.values().filter(|m| m.currently_perceived).count()
    }

    /// Get events since last update.
    pub fn events(&self) -> &[SensoryEvent] {
        &self.events
    }

    /// Clear all memories.
    pub fn clear(&mut self) {
        self.memories.clear();
        self.priority_list.clear();
        self.attention.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> SensoryMemoryStats {
        let threats = self.memories.values().filter(|m| m.category.is_dangerous()).count();
        let perceived = self.perceived_count();
        let avg_confidence = if self.memories.is_empty() {
            0.0
        } else {
            self.memories.values().map(|m| m.confidence).sum::<f32>() / self.memories.len() as f32
        };

        SensoryMemoryStats {
            total_memories: self.memories.len(),
            perceived_count: perceived,
            threat_count: threats,
            average_confidence: avg_confidence,
            capacity_used: self.memories.len() as f32 / self.capacity as f32,
            attended_stimulus: self.attention.current_focus,
        }
    }
}

/// Statistics for the sensory memory.
#[derive(Debug, Clone)]
pub struct SensoryMemoryStats {
    pub total_memories: usize,
    pub perceived_count: usize,
    pub threat_count: usize,
    pub average_confidence: f32,
    pub capacity_used: f32,
    pub attended_stimulus: Option<StimulusId>,
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component for entities with sensory memory.
#[derive(Debug)]
pub struct SensoryMemoryComponent {
    /// The sensory memory store.
    pub memory: SensoryMemory,
    /// Whether sensory processing is enabled.
    pub enabled: bool,
}

impl SensoryMemoryComponent {
    /// Create a new component.
    pub fn new() -> Self {
        Self {
            memory: SensoryMemory::new(),
            enabled: true,
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
    fn test_new_detection() {
        let mut memory = SensoryMemory::new();
        memory.detect(
            StimulusId(1),
            StimulusCategory::Threat,
            Vec3::new(10.0, 0.0, 0.0),
            SenseType::Sight,
        );
        assert_eq!(memory.memory_count(), 1);
        assert_eq!(memory.perceived_count(), 1);
    }

    #[test]
    fn test_multi_modal_fusion() {
        let mut memory = SensoryMemory::new();
        memory.detect(
            StimulusId(1),
            StimulusCategory::Threat,
            Vec3::new(10.0, 0.0, 0.0),
            SenseType::Sight,
        );
        let conf_sight = memory.get_memory(StimulusId(1)).unwrap().confidence;

        memory.detect(
            StimulusId(1),
            StimulusCategory::Threat,
            Vec3::new(10.0, 0.0, 0.0),
            SenseType::Hearing,
        );
        let conf_fused = memory.get_memory(StimulusId(1)).unwrap().confidence;

        // Fusion should increase confidence
        assert!(conf_fused >= conf_sight);
    }

    #[test]
    fn test_memory_decay() {
        let mut memory = SensoryMemory::new();
        memory.detect(
            StimulusId(1),
            StimulusCategory::Object,
            Vec3::new(5.0, 0.0, 0.0),
            SenseType::Sight,
        );
        memory.mark_all_not_perceived();

        let initial_conf = memory.get_memory(StimulusId(1)).unwrap().confidence;

        // Simulate several seconds of decay
        for _ in 0..100 {
            memory.update(0.1);
        }

        let final_conf = memory.get_memory(StimulusId(1)).map(|m| m.confidence).unwrap_or(0.0);
        assert!(final_conf < initial_conf);
    }

    #[test]
    fn test_attention_shift() {
        let mut memory = SensoryMemory::new();
        memory.detect(StimulusId(1), StimulusCategory::Object, Vec3::ZERO, SenseType::Sight);
        memory.update(0.016);
        assert_eq!(memory.attention.current_focus, Some(StimulusId(1)));

        // Higher priority stimulus should steal attention
        memory.detect(StimulusId(2), StimulusCategory::Threat, Vec3::new(5.0, 0.0, 0.0), SenseType::Sight);
        memory.update(0.016);
        assert_eq!(memory.attention.current_focus, Some(StimulusId(2)));
    }

    #[test]
    fn test_capacity_limit() {
        let mut memory = SensoryMemory::new();
        memory.capacity = 3;

        for i in 0..5 {
            memory.detect(
                StimulusId(i),
                StimulusCategory::Object,
                Vec3::new(i as f32, 0.0, 0.0),
                SenseType::Sight,
            );
        }

        assert!(memory.memory_count() <= 3);
    }

    #[test]
    fn test_threat_detection() {
        let mut memory = SensoryMemory::new();
        memory.detect(StimulusId(1), StimulusCategory::Threat, Vec3::ZERO, SenseType::Sight);
        let threats = memory.threats();
        assert_eq!(threats.len(), 1);
        assert!(threats[0].category.is_dangerous());
    }
}
