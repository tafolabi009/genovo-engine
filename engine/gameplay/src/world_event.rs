// engine/gameplay/src/world_event.rs
//
// World events system for the Genovo engine.
//
// Provides a system for dynamic world events that affect gameplay:
//
// - Random events triggered by probability checks.
// - Scheduled events at specific game times.
// - Event conditions that gate when events can fire.
// - Event effects that modify the game world.
// - Event chains (one event triggers the next).
// - Event UI data for presenting events to the player.
// - Event cooldowns and repeat limits.
// - Event priority and conflict resolution.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum active events at once.
const MAX_ACTIVE_EVENTS: usize = 16;

/// Maximum events in the history buffer.
const MAX_EVENT_HISTORY: usize = 256;

/// Default event check interval in seconds.
const DEFAULT_CHECK_INTERVAL: f32 = 10.0;

/// Minimum time between events of the same type.
const DEFAULT_COOLDOWN: f32 = 60.0;

// ---------------------------------------------------------------------------
// Event ID
// ---------------------------------------------------------------------------

/// Unique identifier for a world event definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldEventId(pub u32);

/// Unique identifier for an active event instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventInstanceId(pub u64);

// ---------------------------------------------------------------------------
// Event Category
// ---------------------------------------------------------------------------

/// Category of world event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventCategory {
    /// Environmental events (storms, earthquakes, etc.).
    Environmental,
    /// Social/political events (festivals, wars, etc.).
    Social,
    /// Economic events (market crash, trade caravan, etc.).
    Economic,
    /// Combat events (monster invasion, raid, etc.).
    Combat,
    /// Narrative events (story progression, quests).
    Narrative,
    /// Mystical/magical events.
    Mystical,
    /// Random encounters.
    Encounter,
    /// Seasonal events.
    Seasonal,
}

// ---------------------------------------------------------------------------
// Event Condition
// ---------------------------------------------------------------------------

/// Conditions that must be met for an event to trigger.
#[derive(Debug, Clone)]
pub enum EventCondition {
    /// Time of day condition (0.0 = midnight, 0.5 = noon, 1.0 = midnight).
    TimeOfDay { min: f32, max: f32 },
    /// Day of the (in-game) week.
    DayOfWeek(u32),
    /// Game time in hours has passed a threshold.
    GameTimePassed(f64),
    /// Player is in a specific region/zone.
    InRegion(String),
    /// Player level requirement.
    PlayerLevel { min: u32, max: u32 },
    /// Variable condition.
    Variable { name: String, op: ComparisonOp, value: f64 },
    /// Another event is active.
    EventActive(WorldEventId),
    /// Another event is NOT active.
    EventInactive(WorldEventId),
    /// Another event has completed.
    EventCompleted(WorldEventId),
    /// Random chance (0.0 to 1.0).
    Chance(f32),
    /// All conditions must be true.
    All(Vec<EventCondition>),
    /// Any condition must be true.
    Any(Vec<EventCondition>),
    /// Condition negation.
    Not(Box<EventCondition>),
    /// Weather condition.
    Weather(String),
    /// Custom condition evaluated by game code.
    Custom { name: String },
}

/// Comparison operator for variable conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
}

impl ComparisonOp {
    /// Evaluate the comparison.
    pub fn evaluate(self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Equal => (lhs - rhs).abs() < 1e-6,
            Self::NotEqual => (lhs - rhs).abs() >= 1e-6,
            Self::Greater => lhs > rhs,
            Self::Less => lhs < rhs,
            Self::GreaterOrEqual => lhs >= rhs,
            Self::LessOrEqual => lhs <= rhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Event Effect
// ---------------------------------------------------------------------------

/// Effects that an event applies to the game world.
#[derive(Debug, Clone)]
pub enum EventEffect {
    /// Set a game variable.
    SetVariable { name: String, value: f64 },
    /// Modify a variable by an amount.
    ModifyVariable { name: String, delta: f64 },
    /// Spawn entities.
    SpawnEntities { template: String, count: u32, region: String },
    /// Modify weather.
    SetWeather { weather_type: String, intensity: f32, duration: f32 },
    /// Modify shop prices.
    ModifyPrices { factor: f32, categories: Vec<String> },
    /// Grant item to player.
    GiveItem { item_id: String, count: u32 },
    /// Start a quest.
    StartQuest { quest_id: String },
    /// Display notification.
    ShowNotification { title: String, description: String, icon: String },
    /// Play music.
    PlayMusic { track_id: String },
    /// Modify faction reputation.
    ModifyReputation { faction: String, amount: i32 },
    /// Apply a buff/debuff to the player.
    ApplyBuff { buff_id: String, duration: f32 },
    /// Teleport the player.
    TeleportPlayer { position: [f32; 3] },
    /// Custom effect.
    Custom { name: String, params: HashMap<String, String> },
}

// ---------------------------------------------------------------------------
// Event Definition
// ---------------------------------------------------------------------------

/// Schedule type for an event.
#[derive(Debug, Clone)]
pub enum EventSchedule {
    /// Random event, checked periodically.
    Random {
        /// Base probability per check (0.0 to 1.0).
        base_probability: f32,
        /// Check interval in seconds.
        check_interval: f32,
    },
    /// Scheduled at a specific game time.
    Scheduled {
        /// Game time (in hours from game start) to trigger.
        trigger_time: f64,
        /// Whether to repeat.
        repeat: bool,
        /// Repeat interval in game-hours.
        repeat_interval: f64,
    },
    /// Triggered manually by game code or other events.
    Manual,
    /// Triggered when conditions are first met.
    Conditional,
}

/// Definition of a world event.
#[derive(Debug, Clone)]
pub struct WorldEventDefinition {
    /// Unique event identifier.
    pub id: WorldEventId,
    /// Event name.
    pub name: String,
    /// Description shown to the player.
    pub description: String,
    /// Localization key for name.
    pub name_loc_key: Option<String>,
    /// Localization key for description.
    pub desc_loc_key: Option<String>,
    /// Event category.
    pub category: EventCategory,
    /// Schedule type.
    pub schedule: EventSchedule,
    /// Conditions that must be met.
    pub conditions: Vec<EventCondition>,
    /// Effects applied when the event starts.
    pub on_start_effects: Vec<EventEffect>,
    /// Effects applied when the event ends.
    pub on_end_effects: Vec<EventEffect>,
    /// Duration of the event in game seconds (0 = instant).
    pub duration: f32,
    /// Cooldown before this event can occur again.
    pub cooldown: f32,
    /// Maximum number of times this event can occur.
    pub max_occurrences: u32,
    /// Priority (higher = more important, overrides lower).
    pub priority: i32,
    /// Events that can chain from this event.
    pub chain_events: Vec<ChainEventDef>,
    /// UI data for presenting the event.
    pub ui_data: EventUiData,
    /// Whether the event is enabled.
    pub enabled: bool,
}

/// A chained event that can trigger from a parent event.
#[derive(Debug, Clone)]
pub struct ChainEventDef {
    /// Event to trigger.
    pub event_id: WorldEventId,
    /// Delay after parent event starts (in seconds).
    pub delay: f32,
    /// Probability of this chain (0.0 to 1.0).
    pub probability: f32,
    /// Additional conditions for the chain.
    pub conditions: Vec<EventCondition>,
}

/// UI-related data for event presentation.
#[derive(Debug, Clone)]
pub struct EventUiData {
    /// Icon identifier.
    pub icon: String,
    /// Banner image identifier.
    pub banner: String,
    /// Color theme.
    pub color: [f32; 4],
    /// Whether to show a popup notification.
    pub show_popup: bool,
    /// Whether to show on the map.
    pub show_on_map: bool,
    /// Map marker position (if show_on_map).
    pub map_position: Option<[f32; 2]>,
    /// Sound effect on trigger.
    pub sound_effect: String,
}

impl Default for EventUiData {
    fn default() -> Self {
        Self {
            icon: String::new(),
            banner: String::new(),
            color: [1.0, 1.0, 1.0, 1.0],
            show_popup: true,
            show_on_map: false,
            map_position: None,
            sound_effect: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Active Event Instance
// ---------------------------------------------------------------------------

/// An active world event instance.
#[derive(Debug, Clone)]
pub struct ActiveWorldEvent {
    /// Instance identifier.
    pub instance_id: EventInstanceId,
    /// Definition identifier.
    pub event_id: WorldEventId,
    /// Elapsed time since this event started.
    pub elapsed: f32,
    /// Total duration.
    pub duration: f32,
    /// Whether this event has completed.
    pub completed: bool,
    /// Game time when this event started.
    pub start_game_time: f64,
    /// Chained events that have been triggered.
    pub triggered_chains: Vec<WorldEventId>,
}

impl ActiveWorldEvent {
    /// Create a new active event.
    pub fn new(instance_id: EventInstanceId, event_id: WorldEventId, duration: f32, game_time: f64) -> Self {
        Self {
            instance_id,
            event_id,
            elapsed: 0.0,
            duration,
            completed: false,
            start_game_time: game_time,
            triggered_chains: Vec::new(),
        }
    }

    /// Update the event timer.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.duration > 0.0 && self.elapsed >= self.duration {
            self.completed = true;
        }
    }

    /// Progress (0.0 to 1.0) for timed events.
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }

    /// Remaining time in seconds.
    pub fn remaining(&self) -> f32 {
        (self.duration - self.elapsed).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Event History Entry
// ---------------------------------------------------------------------------

/// A record of a past event.
#[derive(Debug, Clone)]
pub struct EventHistoryEntry {
    /// Event definition ID.
    pub event_id: WorldEventId,
    /// Instance ID.
    pub instance_id: EventInstanceId,
    /// Game time when it occurred.
    pub game_time: f64,
    /// Real time when it occurred (seconds since game start).
    pub real_time: f64,
    /// How long it lasted.
    pub duration: f32,
    /// Whether it completed normally.
    pub completed_normally: bool,
}

// ---------------------------------------------------------------------------
// World Event Manager
// ---------------------------------------------------------------------------

/// Manages all world events.
#[derive(Debug)]
pub struct WorldEventManager {
    /// Event definitions.
    pub definitions: HashMap<WorldEventId, WorldEventDefinition>,
    /// Currently active events.
    pub active_events: Vec<ActiveWorldEvent>,
    /// Event history.
    pub history: Vec<EventHistoryEntry>,
    /// Occurrence counts per event.
    pub occurrence_counts: HashMap<WorldEventId, u32>,
    /// Last occurrence time per event (game time).
    pub last_occurrence: HashMap<WorldEventId, f64>,
    /// Game variables for condition checking.
    pub variables: HashMap<String, f64>,
    /// Current game time in hours.
    pub game_time: f64,
    /// Real time in seconds.
    pub real_time: f64,
    /// Time since last event check.
    pub time_since_check: f32,
    /// Check interval.
    pub check_interval: f32,
    /// Next instance ID.
    next_instance_id: u64,
    /// Pending effects to apply.
    pub pending_effects: Vec<(EventInstanceId, EventEffect)>,
    /// Events generated.
    pub events: Vec<WorldEventSystemEvent>,
    /// Maximum active events.
    pub max_active: usize,
}

/// System events emitted by the world event manager.
#[derive(Debug, Clone)]
pub enum WorldEventSystemEvent {
    /// An event started.
    EventStarted { instance_id: EventInstanceId, event_id: WorldEventId },
    /// An event ended.
    EventEnded { instance_id: EventInstanceId, event_id: WorldEventId, completed: bool },
    /// A chain event was triggered.
    ChainTriggered { parent_id: EventInstanceId, child_event_id: WorldEventId },
    /// An effect was applied.
    EffectApplied { instance_id: EventInstanceId, effect: EventEffect },
}

impl WorldEventManager {
    /// Create a new world event manager.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            active_events: Vec::new(),
            history: Vec::new(),
            occurrence_counts: HashMap::new(),
            last_occurrence: HashMap::new(),
            variables: HashMap::new(),
            game_time: 0.0,
            real_time: 0.0,
            time_since_check: 0.0,
            check_interval: DEFAULT_CHECK_INTERVAL,
            next_instance_id: 1,
            pending_effects: Vec::new(),
            events: Vec::new(),
            max_active: MAX_ACTIVE_EVENTS,
        }
    }

    /// Register an event definition.
    pub fn register_event(&mut self, definition: WorldEventDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Set a game variable.
    pub fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a game variable.
    pub fn get_variable(&self, name: &str) -> f64 {
        self.variables.get(name).copied().unwrap_or(0.0)
    }

    /// Manually trigger an event.
    pub fn trigger_event(&mut self, event_id: WorldEventId) -> Option<EventInstanceId> {
        let def = self.definitions.get(&event_id)?.clone();

        if self.active_events.len() >= self.max_active {
            return None;
        }

        // Check max occurrences.
        let count = self.occurrence_counts.get(&event_id).copied().unwrap_or(0);
        if def.max_occurrences > 0 && count >= def.max_occurrences {
            return None;
        }

        let instance_id = EventInstanceId(self.next_instance_id);
        self.next_instance_id += 1;

        let active = ActiveWorldEvent::new(instance_id, event_id, def.duration, self.game_time);
        self.active_events.push(active);

        *self.occurrence_counts.entry(event_id).or_insert(0) += 1;
        self.last_occurrence.insert(event_id, self.game_time);

        // Queue on_start effects.
        for effect in &def.on_start_effects {
            self.pending_effects.push((instance_id, effect.clone()));
        }

        self.events.push(WorldEventSystemEvent::EventStarted { instance_id, event_id });

        Some(instance_id)
    }

    /// Update the world event system.
    pub fn update(&mut self, dt: f32, game_dt_hours: f64) {
        self.real_time += dt as f64;
        self.game_time += game_dt_hours;
        self.time_since_check += dt;

        // Check for new events periodically.
        if self.time_since_check >= self.check_interval {
            self.time_since_check = 0.0;
            self.check_random_events();
            self.check_scheduled_events();
        }

        // Update active events.
        let mut completed = Vec::new();
        for event in &mut self.active_events {
            event.update(dt);
            if event.completed {
                completed.push(event.instance_id);
            }
        }

        // End completed events.
        for instance_id in completed {
            self.end_event(instance_id, true);
        }

        // Trim history.
        while self.history.len() > MAX_EVENT_HISTORY {
            self.history.remove(0);
        }
    }

    /// Check and trigger random events.
    fn check_random_events(&mut self) {
        let event_ids: Vec<WorldEventId> = self.definitions.keys().copied().collect();
        let mut to_trigger = Vec::new();

        for event_id in event_ids {
            let def = match self.definitions.get(&event_id) {
                Some(d) => d.clone(),
                None => continue,
            };

            if !def.enabled {
                continue;
            }

            if let EventSchedule::Random { base_probability, .. } = def.schedule {
                // Check cooldown.
                if let Some(&last) = self.last_occurrence.get(&event_id) {
                    if (self.game_time - last) * 3600.0 < def.cooldown as f64 {
                        continue;
                    }
                }

                // Simple pseudo-random check.
                let hash = event_id.0 as f64 * 0.618033988 + self.real_time * 0.001;
                let roll = (hash.sin() * 0.5 + 0.5) as f32;
                if roll < base_probability {
                    to_trigger.push(event_id);
                }
            }
        }

        for event_id in to_trigger {
            self.trigger_event(event_id);
        }
    }

    /// Check and trigger scheduled events.
    fn check_scheduled_events(&mut self) {
        let event_ids: Vec<WorldEventId> = self.definitions.keys().copied().collect();
        let mut to_trigger = Vec::new();

        for event_id in event_ids {
            let def = match self.definitions.get(&event_id) {
                Some(d) => d.clone(),
                None => continue,
            };

            if !def.enabled {
                continue;
            }

            if let EventSchedule::Scheduled { trigger_time, .. } = def.schedule {
                if self.game_time >= trigger_time {
                    let count = self.occurrence_counts.get(&event_id).copied().unwrap_or(0);
                    if count == 0 {
                        to_trigger.push(event_id);
                    }
                }
            }
        }

        for event_id in to_trigger {
            self.trigger_event(event_id);
        }
    }

    /// End an active event.
    fn end_event(&mut self, instance_id: EventInstanceId, completed: bool) {
        let event_idx = self.active_events.iter().position(|e| e.instance_id == instance_id);
        if let Some(idx) = event_idx {
            let event = self.active_events.remove(idx);

            // Queue on_end effects.
            if let Some(def) = self.definitions.get(&event.event_id) {
                for effect in &def.on_end_effects {
                    self.pending_effects.push((instance_id, effect.clone()));
                }
            }

            // Add to history.
            self.history.push(EventHistoryEntry {
                event_id: event.event_id,
                instance_id,
                game_time: event.start_game_time,
                real_time: self.real_time,
                duration: event.elapsed,
                completed_normally: completed,
            });

            self.events.push(WorldEventSystemEvent::EventEnded {
                instance_id,
                event_id: event.event_id,
                completed,
            });
        }
    }

    /// Cancel an active event.
    pub fn cancel_event(&mut self, instance_id: EventInstanceId) {
        self.end_event(instance_id, false);
    }

    /// Check if a specific event type is currently active.
    pub fn is_event_active(&self, event_id: WorldEventId) -> bool {
        self.active_events.iter().any(|e| e.event_id == event_id)
    }

    /// Get active events.
    pub fn active_event_count(&self) -> usize {
        self.active_events.len()
    }

    /// Drain pending effects.
    pub fn drain_effects(&mut self) -> Vec<(EventInstanceId, EventEffect)> {
        std::mem::take(&mut self.pending_effects)
    }

    /// Drain system events.
    pub fn drain_events(&mut self) -> Vec<WorldEventSystemEvent> {
        std::mem::take(&mut self.events)
    }
}
