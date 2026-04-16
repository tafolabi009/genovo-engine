//! Quest system with objectives, rewards, and a quest journal.
//!
//! Provides a data-driven quest framework supporting multiple objective types,
//! quest chains, rewards, and a journal that tracks active/completed quests.
//!
//! # Architecture
//!
//! - [`Quest`] -- a quest definition with objectives and rewards.
//! - [`QuestObjective`] -- a specific goal (kill, collect, reach, talk, etc.).
//! - [`QuestState`] -- lifecycle state of a quest instance.
//! - [`QuestReward`] -- items, XP, currency, or reputation given on completion.
//! - [`QuestJournal`] -- per-player collection of active and completed quests.
//! - [`QuestEvent`] -- events emitted when quests change state.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Quest state
// ---------------------------------------------------------------------------

/// Lifecycle state of a quest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestState {
    /// Quest is available but not yet accepted.
    Available,
    /// Quest has been accepted and is in progress.
    Active,
    /// All objectives are complete; waiting for turn-in.
    ReadyToComplete,
    /// Quest has been completed and rewards claimed.
    Completed,
    /// Quest was abandoned by the player.
    Abandoned,
    /// Quest has been failed (timer expired, escort died, etc.).
    Failed,
    /// Quest is locked (prerequisites not met).
    Locked,
}

impl QuestState {
    /// Whether the quest is currently trackable (shown in HUD).
    pub fn is_trackable(&self) -> bool {
        matches!(self, Self::Active | Self::ReadyToComplete)
    }

    /// Whether the quest can be abandoned.
    pub fn can_abandon(&self) -> bool {
        matches!(self, Self::Active | Self::ReadyToComplete)
    }

    /// Whether the quest is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

impl Default for QuestState {
    fn default() -> Self {
        Self::Available
    }
}

// ---------------------------------------------------------------------------
// Quest objective type
// ---------------------------------------------------------------------------

/// The kind of objective the player must fulfill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Kill a certain number of enemies.
    Kill {
        /// Target enemy type/tag.
        target: String,
        /// Required kill count.
        required: u32,
    },
    /// Collect a certain number of items.
    Collect {
        /// Item id to collect.
        item_id: String,
        /// Required quantity.
        required: u32,
    },
    /// Reach a location.
    Reach {
        /// Name/id of the target location.
        location: String,
        /// World-space position (for distance checks).
        position: [f32; 3],
        /// Radius around the position to trigger completion.
        radius: f32,
    },
    /// Talk to an NPC.
    Talk {
        /// NPC entity name/id.
        npc: String,
    },
    /// Escort an NPC to a location.
    Escort {
        /// NPC entity name/id.
        npc: String,
        /// Destination location name.
        destination: String,
        /// Destination position.
        destination_position: [f32; 3],
        /// Radius for arrival check.
        radius: f32,
    },
    /// Complete within a time limit.
    Timer {
        /// Time limit in seconds.
        time_limit: f32,
    },
    /// Custom objective with a string key (game code provides the logic).
    Custom {
        /// Unique key for game code to check.
        key: String,
        /// Description of what to do.
        description: String,
    },
}

// ---------------------------------------------------------------------------
// Quest objective
// ---------------------------------------------------------------------------

/// A single objective within a quest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestObjective {
    /// Unique id within the quest.
    pub objective_id: String,
    /// Display text.
    pub description: String,
    /// Type of objective.
    pub objective_type: ObjectiveType,
    /// Current progress.
    pub current: u32,
    /// Required progress for completion (derived from type).
    pub required: u32,
    /// Whether this objective is complete.
    pub completed: bool,
    /// Whether this objective is optional (not required for quest completion).
    pub optional: bool,
    /// Whether this objective is hidden until revealed.
    pub hidden: bool,
    /// Remaining time for timer objectives.
    pub timer_remaining: Option<f32>,
}

impl QuestObjective {
    /// Create a kill objective.
    pub fn kill(id: impl Into<String>, target: impl Into<String>, count: u32) -> Self {
        let target = target.into();
        Self {
            objective_id: id.into(),
            description: format!("Kill {} {}", count, &target),
            objective_type: ObjectiveType::Kill {
                target,
                required: count,
            },
            current: 0,
            required: count,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Create a collect objective.
    pub fn collect(id: impl Into<String>, item_id: impl Into<String>, count: u32) -> Self {
        let item = item_id.into();
        Self {
            objective_id: id.into(),
            description: format!("Collect {} {}", count, &item),
            objective_type: ObjectiveType::Collect {
                item_id: item,
                required: count,
            },
            current: 0,
            required: count,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Create a reach-location objective.
    pub fn reach(
        id: impl Into<String>,
        location: impl Into<String>,
        position: [f32; 3],
        radius: f32,
    ) -> Self {
        let location = location.into();
        Self {
            objective_id: id.into(),
            description: format!("Reach {}", &location),
            objective_type: ObjectiveType::Reach {
                location,
                position,
                radius,
            },
            current: 0,
            required: 1,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Create a talk-to-NPC objective.
    pub fn talk(id: impl Into<String>, npc: impl Into<String>) -> Self {
        let npc = npc.into();
        Self {
            objective_id: id.into(),
            description: format!("Talk to {}", &npc),
            objective_type: ObjectiveType::Talk { npc },
            current: 0,
            required: 1,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Create an escort objective.
    pub fn escort(
        id: impl Into<String>,
        npc: impl Into<String>,
        destination: impl Into<String>,
        position: [f32; 3],
        radius: f32,
    ) -> Self {
        let npc = npc.into();
        let destination = destination.into();
        Self {
            objective_id: id.into(),
            description: format!("Escort {} to {}", &npc, &destination),
            objective_type: ObjectiveType::Escort {
                npc,
                destination,
                destination_position: position,
                radius,
            },
            current: 0,
            required: 1,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Create a timer objective.
    pub fn timer(id: impl Into<String>, time_limit: f32) -> Self {
        Self {
            objective_id: id.into(),
            description: format!("Complete within {:.0} seconds", time_limit),
            objective_type: ObjectiveType::Timer { time_limit },
            current: 0,
            required: 1,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: Some(time_limit),
        }
    }

    /// Create a custom objective.
    pub fn custom(
        id: impl Into<String>,
        key: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let desc = description.into();
        Self {
            objective_id: id.into(),
            description: desc,
            objective_type: ObjectiveType::Custom {
                key: key.into(),
                description: String::new(),
            },
            current: 0,
            required: 1,
            completed: false,
            optional: false,
            hidden: false,
            timer_remaining: None,
        }
    }

    /// Mark this objective as optional.
    pub fn set_optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Mark this objective as hidden.
    pub fn set_hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Increment progress and check for completion.
    pub fn add_progress(&mut self, amount: u32) -> bool {
        if self.completed {
            return false;
        }
        self.current = (self.current + amount).min(self.required);
        if self.current >= self.required {
            self.completed = true;
            log::debug!(
                "Objective '{}' completed ({}/{})",
                self.objective_id,
                self.current,
                self.required
            );
            true
        } else {
            false
        }
    }

    /// Set progress to a specific value.
    pub fn set_progress(&mut self, value: u32) -> bool {
        if self.completed {
            return false;
        }
        self.current = value.min(self.required);
        if self.current >= self.required {
            self.completed = true;
            true
        } else {
            false
        }
    }

    /// Force-complete this objective.
    pub fn force_complete(&mut self) {
        self.current = self.required;
        self.completed = true;
    }

    /// Progress as a fraction (0..1).
    #[inline]
    pub fn progress_fraction(&self) -> f32 {
        if self.required == 0 {
            return 1.0;
        }
        self.current as f32 / self.required as f32
    }

    /// Update the timer, if applicable. Returns `true` if the timer expired.
    pub fn update_timer(&mut self, dt: f32) -> bool {
        if let Some(remaining) = &mut self.timer_remaining {
            *remaining -= dt;
            if *remaining <= 0.0 {
                *remaining = 0.0;
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Quest reward
// ---------------------------------------------------------------------------

/// A reward granted on quest completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestReward {
    /// Experience points.
    Experience(u32),
    /// Currency amount.
    Currency {
        currency_type: String,
        amount: u32,
    },
    /// Item drop.
    Item {
        item_id: String,
        quantity: u32,
    },
    /// Reputation change.
    Reputation {
        faction: String,
        amount: i32,
    },
    /// Unlock a recipe, skill, area, etc.
    Unlock {
        unlock_type: String,
        unlock_id: String,
    },
    /// Custom reward handled by game code.
    Custom {
        key: String,
        value: String,
    },
}

impl QuestReward {
    /// Create an XP reward.
    pub fn xp(amount: u32) -> Self {
        Self::Experience(amount)
    }

    /// Create an item reward.
    pub fn item(item_id: impl Into<String>, quantity: u32) -> Self {
        Self::Item {
            item_id: item_id.into(),
            quantity,
        }
    }

    /// Create a currency reward.
    pub fn currency(currency_type: impl Into<String>, amount: u32) -> Self {
        Self::Currency {
            currency_type: currency_type.into(),
            amount,
        }
    }

    /// Create a reputation reward.
    pub fn reputation(faction: impl Into<String>, amount: i32) -> Self {
        Self::Reputation {
            faction: faction.into(),
            amount,
        }
    }
}

// ---------------------------------------------------------------------------
// Quest
// ---------------------------------------------------------------------------

/// A quest definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    /// Unique quest identifier.
    pub quest_id: String,
    /// Display name.
    pub name: String,
    /// Description text.
    pub description: String,
    /// Quest category (main, side, daily, etc.).
    pub category: String,
    /// Recommended level.
    pub level: u32,
    /// Objectives to complete.
    pub objectives: Vec<QuestObjective>,
    /// Rewards granted on completion.
    pub rewards: Vec<QuestReward>,
    /// Optional rewards the player can choose from.
    pub choice_rewards: Vec<QuestReward>,
    /// Prerequisite quest ids (all must be completed).
    pub prerequisites: Vec<String>,
    /// Whether this quest is repeatable.
    pub repeatable: bool,
    /// Whether this quest auto-completes when objectives are done
    /// (vs. requiring a turn-in).
    pub auto_complete: bool,
    /// Current state of this quest instance.
    pub state: QuestState,
    /// NPC that gives this quest.
    pub quest_giver: Option<String>,
    /// NPC to turn in this quest to (if different from giver).
    pub turn_in_npc: Option<String>,
    /// Whether this quest is tracked in the HUD.
    pub tracked: bool,
    /// Time limit for the entire quest (0 = no limit).
    pub time_limit: f32,
    /// Remaining time (if time-limited).
    pub time_remaining: Option<f32>,
}

impl Quest {
    /// Create a new quest.
    pub fn new(quest_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            quest_id: quest_id.into(),
            name: name.into(),
            description: String::new(),
            category: "main".into(),
            level: 0,
            objectives: Vec::new(),
            rewards: Vec::new(),
            choice_rewards: Vec::new(),
            prerequisites: Vec::new(),
            repeatable: false,
            auto_complete: false,
            state: QuestState::Available,
            quest_giver: None,
            turn_in_npc: None,
            tracked: false,
            time_limit: 0.0,
            time_remaining: None,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add an objective.
    pub fn with_objective(mut self, objective: QuestObjective) -> Self {
        self.objectives.push(objective);
        self
    }

    /// Add a reward.
    pub fn with_reward(mut self, reward: QuestReward) -> Self {
        self.rewards.push(reward);
        self
    }

    /// Add a prerequisite.
    pub fn with_prerequisite(mut self, quest_id: impl Into<String>) -> Self {
        self.prerequisites.push(quest_id.into());
        self
    }

    /// Set the quest giver NPC.
    pub fn with_quest_giver(mut self, npc: impl Into<String>) -> Self {
        self.quest_giver = Some(npc.into());
        self
    }

    /// Set time limit.
    pub fn with_time_limit(mut self, seconds: f32) -> Self {
        self.time_limit = seconds;
        self.time_remaining = Some(seconds);
        self
    }

    /// Whether all required (non-optional) objectives are complete.
    pub fn all_objectives_complete(&self) -> bool {
        self.objectives
            .iter()
            .filter(|o| !o.optional)
            .all(|o| o.completed)
    }

    /// Number of completed objectives (excluding optional).
    pub fn completed_objective_count(&self) -> usize {
        self.objectives
            .iter()
            .filter(|o| !o.optional && o.completed)
            .count()
    }

    /// Total number of required objectives.
    pub fn required_objective_count(&self) -> usize {
        self.objectives.iter().filter(|o| !o.optional).count()
    }

    /// Overall progress as a fraction (0..1).
    pub fn progress_fraction(&self) -> f32 {
        let total = self.required_objective_count();
        if total == 0 {
            return 1.0;
        }
        self.completed_objective_count() as f32 / total as f32
    }

    /// Get an objective by id.
    pub fn get_objective(&self, objective_id: &str) -> Option<&QuestObjective> {
        self.objectives
            .iter()
            .find(|o| o.objective_id == objective_id)
    }

    /// Get a mutable objective by id.
    pub fn get_objective_mut(&mut self, objective_id: &str) -> Option<&mut QuestObjective> {
        self.objectives
            .iter_mut()
            .find(|o| o.objective_id == objective_id)
    }

    /// Update quest timers. Returns `true` if the quest timed out.
    pub fn update_timers(&mut self, dt: f32) -> bool {
        if self.state != QuestState::Active {
            return false;
        }

        // Quest-level timer.
        if let Some(remaining) = &mut self.time_remaining {
            *remaining -= dt;
            if *remaining <= 0.0 {
                *remaining = 0.0;
                self.state = QuestState::Failed;
                log::info!("Quest '{}' failed: time expired", self.quest_id);
                return true;
            }
        }

        // Objective-level timers.
        for objective in &mut self.objectives {
            if objective.update_timer(dt) && !objective.optional {
                self.state = QuestState::Failed;
                log::info!(
                    "Quest '{}' failed: timer objective '{}' expired",
                    self.quest_id,
                    objective.objective_id
                );
                return true;
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Quest events
// ---------------------------------------------------------------------------

/// Events emitted by the quest system for other systems to react to.
#[derive(Debug, Clone)]
pub enum QuestEvent {
    /// A quest was accepted.
    QuestAccepted {
        quest_id: String,
    },
    /// A quest objective progressed.
    ObjectiveProgressed {
        quest_id: String,
        objective_id: String,
        current: u32,
        required: u32,
    },
    /// A quest objective was completed.
    ObjectiveCompleted {
        quest_id: String,
        objective_id: String,
    },
    /// A quest is ready for turn-in.
    QuestReadyToComplete {
        quest_id: String,
    },
    /// A quest was completed and rewards granted.
    QuestCompleted {
        quest_id: String,
        rewards: Vec<QuestReward>,
    },
    /// A quest was failed.
    QuestFailed {
        quest_id: String,
        reason: String,
    },
    /// A quest was abandoned.
    QuestAbandoned {
        quest_id: String,
    },
}

// ---------------------------------------------------------------------------
// Quest journal
// ---------------------------------------------------------------------------

/// Per-player quest journal that tracks all quest states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuestJournal {
    /// All quests by id.
    quests: HashMap<String, Quest>,
    /// Ids of completed quests (for prerequisite checking).
    completed_quest_ids: Vec<String>,
    /// Events generated during the last update (drained by the caller).
    #[serde(skip)]
    pending_events: Vec<QuestEvent>,
}

impl QuestJournal {
    /// Create an empty journal.
    pub fn new() -> Self {
        Self {
            quests: HashMap::new(),
            completed_quest_ids: Vec::new(),
            pending_events: Vec::new(),
        }
    }

    /// Accept a quest, moving it to the Active state.
    pub fn accept_quest(&mut self, mut quest: Quest) -> Result<(), QuestError> {
        // Check prerequisites.
        for prereq in &quest.prerequisites {
            if !self.completed_quest_ids.contains(prereq) {
                return Err(QuestError::PrerequisiteNotMet(prereq.clone()));
            }
        }

        // Check if already active.
        if self.quests.contains_key(&quest.quest_id) {
            let existing = &self.quests[&quest.quest_id];
            if existing.state == QuestState::Active || existing.state == QuestState::ReadyToComplete
            {
                return Err(QuestError::AlreadyActive(quest.quest_id.clone()));
            }
            if existing.state == QuestState::Completed && !quest.repeatable {
                return Err(QuestError::AlreadyCompleted(quest.quest_id.clone()));
            }
        }

        quest.state = QuestState::Active;
        quest.tracked = true;

        let id = quest.quest_id.clone();
        self.pending_events
            .push(QuestEvent::QuestAccepted { quest_id: id.clone() });

        log::info!("Quest accepted: '{}'", quest.name);
        self.quests.insert(id, quest);

        Ok(())
    }

    /// Update objective progress for a kill event.
    pub fn on_kill(&mut self, target_type: &str) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            self.update_kill_objective(&quest_id, target_type);
        }
    }

    /// Update kill objectives for a specific quest.
    fn update_kill_objective(&mut self, quest_id: &str, target_type: &str) {
        let quest = match self.quests.get_mut(quest_id) {
            Some(q) => q,
            None => return,
        };

        let mut objective_completed = false;
        let mut events = Vec::new();

        for objective in &mut quest.objectives {
            if objective.completed {
                continue;
            }
            if let ObjectiveType::Kill { target, .. } = &objective.objective_type {
                if target == target_type {
                    let just_completed = objective.add_progress(1);
                    events.push(QuestEvent::ObjectiveProgressed {
                        quest_id: quest_id.to_string(),
                        objective_id: objective.objective_id.clone(),
                        current: objective.current,
                        required: objective.required,
                    });
                    if just_completed {
                        events.push(QuestEvent::ObjectiveCompleted {
                            quest_id: quest_id.to_string(),
                            objective_id: objective.objective_id.clone(),
                        });
                        objective_completed = true;
                    }
                }
            }
        }

        self.pending_events.extend(events);

        if objective_completed {
            self.check_quest_completion(quest_id);
        }
    }

    /// Update objective progress for a collect event.
    pub fn on_collect(&mut self, item_id: &str, count: u32) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            self.update_collect_objective(&quest_id, item_id, count);
        }
    }

    /// Update collect objectives for a specific quest.
    fn update_collect_objective(&mut self, quest_id: &str, item_id: &str, count: u32) {
        let quest = match self.quests.get_mut(quest_id) {
            Some(q) => q,
            None => return,
        };

        let mut objective_completed = false;
        let mut events = Vec::new();

        for objective in &mut quest.objectives {
            if objective.completed {
                continue;
            }
            if let ObjectiveType::Collect {
                item_id: target_id, ..
            } = &objective.objective_type
            {
                if target_id == item_id {
                    let just_completed = objective.add_progress(count);
                    events.push(QuestEvent::ObjectiveProgressed {
                        quest_id: quest_id.to_string(),
                        objective_id: objective.objective_id.clone(),
                        current: objective.current,
                        required: objective.required,
                    });
                    if just_completed {
                        events.push(QuestEvent::ObjectiveCompleted {
                            quest_id: quest_id.to_string(),
                            objective_id: objective.objective_id.clone(),
                        });
                        objective_completed = true;
                    }
                }
            }
        }

        self.pending_events.extend(events);

        if objective_completed {
            self.check_quest_completion(quest_id);
        }
    }

    /// Update progress for a location reach event.
    pub fn on_reach_location(&mut self, location: &str) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            let quest = match self.quests.get_mut(&quest_id) {
                Some(q) => q,
                None => continue,
            };

            let mut events = Vec::new();
            let mut any_completed = false;

            for objective in &mut quest.objectives {
                if objective.completed {
                    continue;
                }
                if let ObjectiveType::Reach {
                    location: target, ..
                } = &objective.objective_type
                {
                    if target == location {
                        objective.force_complete();
                        events.push(QuestEvent::ObjectiveCompleted {
                            quest_id: quest_id.clone(),
                            objective_id: objective.objective_id.clone(),
                        });
                        any_completed = true;
                    }
                }
            }

            self.pending_events.extend(events);
            if any_completed {
                self.check_quest_completion(&quest_id);
            }
        }
    }

    /// Update progress for a talk-to-NPC event.
    pub fn on_talk_to(&mut self, npc_id: &str) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            let quest = match self.quests.get_mut(&quest_id) {
                Some(q) => q,
                None => continue,
            };

            let mut events = Vec::new();
            let mut any_completed = false;

            for objective in &mut quest.objectives {
                if objective.completed {
                    continue;
                }
                if let ObjectiveType::Talk { npc } = &objective.objective_type {
                    if npc == npc_id {
                        objective.force_complete();
                        events.push(QuestEvent::ObjectiveCompleted {
                            quest_id: quest_id.clone(),
                            objective_id: objective.objective_id.clone(),
                        });
                        any_completed = true;
                    }
                }
            }

            self.pending_events.extend(events);
            if any_completed {
                self.check_quest_completion(&quest_id);
            }
        }
    }

    /// Update a custom objective by key.
    pub fn on_custom_objective(&mut self, key: &str, progress: u32) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            let quest = match self.quests.get_mut(&quest_id) {
                Some(q) => q,
                None => continue,
            };

            let mut events = Vec::new();
            let mut any_completed = false;

            for objective in &mut quest.objectives {
                if objective.completed {
                    continue;
                }
                if let ObjectiveType::Custom {
                    key: obj_key, ..
                } = &objective.objective_type
                {
                    if obj_key == key {
                        let just_completed = objective.add_progress(progress);
                        events.push(QuestEvent::ObjectiveProgressed {
                            quest_id: quest_id.clone(),
                            objective_id: objective.objective_id.clone(),
                            current: objective.current,
                            required: objective.required,
                        });
                        if just_completed {
                            events.push(QuestEvent::ObjectiveCompleted {
                                quest_id: quest_id.clone(),
                                objective_id: objective.objective_id.clone(),
                            });
                            any_completed = true;
                        }
                    }
                }
            }

            self.pending_events.extend(events);
            if any_completed {
                self.check_quest_completion(&quest_id);
            }
        }
    }

    /// Check if a quest should transition to ReadyToComplete or Completed.
    fn check_quest_completion(&mut self, quest_id: &str) {
        let quest = match self.quests.get_mut(quest_id) {
            Some(q) if q.state == QuestState::Active => q,
            _ => return,
        };

        if quest.all_objectives_complete() {
            if quest.auto_complete {
                quest.state = QuestState::Completed;
                let id = quest.quest_id.clone();
                let rewards = quest.rewards.clone();
                self.completed_quest_ids.push(id.clone());
                self.pending_events.push(QuestEvent::QuestCompleted {
                    quest_id: id.clone(),
                    rewards,
                });
                log::info!("Quest '{}' auto-completed", id);
            } else {
                quest.state = QuestState::ReadyToComplete;
                self.pending_events.push(QuestEvent::QuestReadyToComplete {
                    quest_id: quest.quest_id.clone(),
                });
                log::info!("Quest '{}' ready to complete", quest.quest_id);
            }
        }
    }

    /// Complete a quest that is in the ReadyToComplete state.
    pub fn complete_quest(&mut self, quest_id: &str) -> Result<Vec<QuestReward>, QuestError> {
        let quest = self
            .quests
            .get_mut(quest_id)
            .ok_or_else(|| QuestError::NotFound(quest_id.to_string()))?;

        if quest.state != QuestState::ReadyToComplete {
            return Err(QuestError::NotReady(quest_id.to_string()));
        }

        quest.state = QuestState::Completed;
        let rewards = quest.rewards.clone();

        self.completed_quest_ids.push(quest_id.to_string());
        self.pending_events.push(QuestEvent::QuestCompleted {
            quest_id: quest_id.to_string(),
            rewards: rewards.clone(),
        });

        log::info!("Quest '{}' completed", quest_id);
        Ok(rewards)
    }

    /// Abandon a quest.
    pub fn abandon_quest(&mut self, quest_id: &str) -> Result<(), QuestError> {
        let quest = self
            .quests
            .get_mut(quest_id)
            .ok_or_else(|| QuestError::NotFound(quest_id.to_string()))?;

        if !quest.state.can_abandon() {
            return Err(QuestError::CannotAbandon(quest_id.to_string()));
        }

        quest.state = QuestState::Abandoned;
        quest.tracked = false;
        self.pending_events.push(QuestEvent::QuestAbandoned {
            quest_id: quest_id.to_string(),
        });

        log::info!("Quest '{}' abandoned", quest_id);
        Ok(())
    }

    /// Update all active quest timers.
    pub fn update(&mut self, dt: f32) {
        let quest_ids: Vec<String> = self
            .quests
            .iter()
            .filter(|(_, q)| q.state == QuestState::Active)
            .map(|(id, _)| id.clone())
            .collect();

        for quest_id in quest_ids {
            let failed = if let Some(quest) = self.quests.get_mut(&quest_id) {
                quest.update_timers(dt)
            } else {
                false
            };

            if failed {
                self.pending_events.push(QuestEvent::QuestFailed {
                    quest_id: quest_id.clone(),
                    reason: "Time expired".to_string(),
                });
            }
        }
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<QuestEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Get a quest by id.
    pub fn get_quest(&self, quest_id: &str) -> Option<&Quest> {
        self.quests.get(quest_id)
    }

    /// Get all active quests.
    pub fn active_quests(&self) -> Vec<&Quest> {
        self.quests
            .values()
            .filter(|q| q.state == QuestState::Active || q.state == QuestState::ReadyToComplete)
            .collect()
    }

    /// Get all completed quest ids.
    pub fn completed_quests(&self) -> &[String] {
        &self.completed_quest_ids
    }

    /// Whether a quest has been completed.
    pub fn is_quest_completed(&self, quest_id: &str) -> bool {
        self.completed_quest_ids.iter().any(|id| id == quest_id)
    }

    /// Get tracked quests (for HUD display).
    pub fn tracked_quests(&self) -> Vec<&Quest> {
        self.quests
            .values()
            .filter(|q| q.tracked && q.state.is_trackable())
            .collect()
    }

    /// Number of active quests.
    pub fn active_count(&self) -> usize {
        self.quests
            .values()
            .filter(|q| q.state == QuestState::Active)
            .count()
    }

    /// Total number of quests in the journal.
    pub fn total_count(&self) -> usize {
        self.quests.len()
    }
}

impl genovo_ecs::Component for QuestJournal {}

// ---------------------------------------------------------------------------
// Quest errors
// ---------------------------------------------------------------------------

/// Errors from quest operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum QuestError {
    #[error("quest '{0}' not found")]
    NotFound(String),
    #[error("prerequisite quest '{0}' not completed")]
    PrerequisiteNotMet(String),
    #[error("quest '{0}' is already active")]
    AlreadyActive(String),
    #[error("quest '{0}' has already been completed")]
    AlreadyCompleted(String),
    #[error("quest '{0}' is not ready to complete")]
    NotReady(String),
    #[error("quest '{0}' cannot be abandoned in its current state")]
    CannotAbandon(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_quest() -> Quest {
        Quest::new("q1", "Goblin Slayer")
            .with_description("Kill goblins and collect their ears.")
            .with_objective(QuestObjective::kill("obj1", "goblin", 5))
            .with_objective(QuestObjective::collect("obj2", "goblin_ear", 3))
            .with_reward(QuestReward::xp(100))
            .with_reward(QuestReward::item("gold", 50))
    }

    #[test]
    fn accept_quest() {
        let mut journal = QuestJournal::new();
        let quest = sample_quest();
        journal.accept_quest(quest).unwrap();

        assert_eq!(journal.active_count(), 1);
        let q = journal.get_quest("q1").unwrap();
        assert_eq!(q.state, QuestState::Active);
    }

    #[test]
    fn kill_objective_progress() {
        let mut journal = QuestJournal::new();
        journal.accept_quest(sample_quest()).unwrap();

        for _ in 0..5 {
            journal.on_kill("goblin");
        }

        let q = journal.get_quest("q1").unwrap();
        let obj = q.get_objective("obj1").unwrap();
        assert!(obj.completed);
        assert_eq!(obj.current, 5);
    }

    #[test]
    fn collect_objective_progress() {
        let mut journal = QuestJournal::new();
        journal.accept_quest(sample_quest()).unwrap();

        journal.on_collect("goblin_ear", 3);

        let q = journal.get_quest("q1").unwrap();
        let obj = q.get_objective("obj2").unwrap();
        assert!(obj.completed);
    }

    #[test]
    fn quest_auto_completes() {
        let mut journal = QuestJournal::new();
        let mut quest = sample_quest();
        quest.auto_complete = true;
        journal.accept_quest(quest).unwrap();

        for _ in 0..5 {
            journal.on_kill("goblin");
        }
        journal.on_collect("goblin_ear", 3);

        let q = journal.get_quest("q1").unwrap();
        assert_eq!(q.state, QuestState::Completed);
        assert!(journal.is_quest_completed("q1"));
    }

    #[test]
    fn quest_ready_to_complete() {
        let mut journal = QuestJournal::new();
        journal.accept_quest(sample_quest()).unwrap();

        for _ in 0..5 {
            journal.on_kill("goblin");
        }
        journal.on_collect("goblin_ear", 3);

        let q = journal.get_quest("q1").unwrap();
        assert_eq!(q.state, QuestState::ReadyToComplete);

        let rewards = journal.complete_quest("q1").unwrap();
        assert!(!rewards.is_empty());
        assert!(journal.is_quest_completed("q1"));
    }

    #[test]
    fn abandon_quest() {
        let mut journal = QuestJournal::new();
        journal.accept_quest(sample_quest()).unwrap();

        journal.abandon_quest("q1").unwrap();
        let q = journal.get_quest("q1").unwrap();
        assert_eq!(q.state, QuestState::Abandoned);
    }

    #[test]
    fn prerequisite_enforced() {
        let mut journal = QuestJournal::new();
        let quest = Quest::new("q2", "Sequel Quest").with_prerequisite("q1");

        let result = journal.accept_quest(quest);
        assert!(matches!(result, Err(QuestError::PrerequisiteNotMet(_))));
    }

    #[test]
    fn quest_events_emitted() {
        let mut journal = QuestJournal::new();
        journal.accept_quest(sample_quest()).unwrap();

        let events = journal.drain_events();
        assert!(!events.is_empty());
        assert!(matches!(events[0], QuestEvent::QuestAccepted { .. }));
    }

    #[test]
    fn timer_quest_fails() {
        let mut journal = QuestJournal::new();
        let quest = Quest::new("timed", "Timed Quest")
            .with_objective(QuestObjective::kill("obj1", "wolf", 3))
            .with_time_limit(10.0);

        journal.accept_quest(quest).unwrap();

        // Advance past time limit.
        journal.update(11.0);

        let q = journal.get_quest("timed").unwrap();
        assert_eq!(q.state, QuestState::Failed);
    }

    #[test]
    fn talk_objective() {
        let mut journal = QuestJournal::new();
        let quest = Quest::new("talk", "Talk Quest")
            .with_objective(QuestObjective::talk("obj1", "elder"))
            .with_reward(QuestReward::xp(50));
        quest.clone(); // Ensure Clone works.

        let mut quest = quest;
        quest.auto_complete = true;
        journal.accept_quest(quest).unwrap();

        journal.on_talk_to("elder");

        let q = journal.get_quest("talk").unwrap();
        assert_eq!(q.state, QuestState::Completed);
    }

    #[test]
    fn reach_location_objective() {
        let mut journal = QuestJournal::new();
        let quest = Quest::new("explore", "Explore Quest")
            .with_objective(QuestObjective::reach(
                "obj1",
                "cave",
                [10.0, 0.0, 20.0],
                5.0,
            ));
        let mut quest = quest;
        quest.auto_complete = true;
        journal.accept_quest(quest).unwrap();

        journal.on_reach_location("cave");

        let q = journal.get_quest("explore").unwrap();
        assert_eq!(q.state, QuestState::Completed);
    }

    #[test]
    fn quest_progress_fraction() {
        let mut quest = sample_quest();
        quest.state = QuestState::Active;
        assert!((quest.progress_fraction() - 0.0).abs() < 0.01);

        quest.objectives[0].force_complete();
        assert!((quest.progress_fraction() - 0.5).abs() < 0.01);

        quest.objectives[1].force_complete();
        assert!((quest.progress_fraction() - 1.0).abs() < 0.01);
    }

    #[test]
    fn optional_objective_not_required() {
        let quest = Quest::new("opt", "Optional Test")
            .with_objective(QuestObjective::kill("req", "wolf", 3))
            .with_objective(QuestObjective::kill("opt", "bear", 1).set_optional());

        assert_eq!(quest.required_objective_count(), 1);
    }

    #[test]
    fn custom_objective() {
        let mut journal = QuestJournal::new();
        let mut quest = Quest::new("custom", "Custom Quest")
            .with_objective(QuestObjective::custom("obj1", "puzzle_solved", "Solve the puzzle"));
        quest.auto_complete = true;
        journal.accept_quest(quest).unwrap();

        journal.on_custom_objective("puzzle_solved", 1);

        let q = journal.get_quest("custom").unwrap();
        assert_eq!(q.state, QuestState::Completed);
    }
}
