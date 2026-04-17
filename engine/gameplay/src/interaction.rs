//! World interaction system for interactable objects and entities.
//!
//! Provides a framework for player-world interactions such as opening doors,
//! picking up items, flipping switches, and starting conversations.
//!
//! - [`Interactable`] — component marking an entity as interactable.
//! - [`InteractionSystem`] — finds the nearest interactable, manages
//!   hold-to-interact progress, and dispatches interaction events.
//! - Concrete types: [`Door`], [`Pickup`], [`Switch`], [`DialogueTrigger`].

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ============================================================================
// Interaction types
// ============================================================================

/// How the player interacts with an object.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Instant interaction (press button → done).
    Instant,
    /// Hold the interaction button for a duration.
    HoldDuration(f32),
    /// Toggle on/off.
    Toggle,
}

impl Default for InteractionType {
    fn default() -> Self {
        Self::Instant
    }
}

// ============================================================================
// Interactable component
// ============================================================================

/// Component that marks an entity as interactable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interactable {
    /// Unique entity ID.
    pub entity_id: u32,
    /// Text shown on the interaction prompt (e.g. "Press E to open").
    pub prompt_text: String,
    /// Maximum distance from which the player can interact.
    pub interaction_range: f32,
    /// Type of interaction (instant, hold, or toggle).
    pub interaction_type: InteractionType,
    /// Whether this interactable is currently enabled.
    pub enabled: bool,
    /// Whether this can only be interacted with once.
    pub one_shot: bool,
    /// Whether this has already been used (for one-shot interactables).
    pub used: bool,
    /// Priority for when multiple interactables overlap (higher = preferred).
    pub priority: i32,
    /// Optional required item ID the player must hold.
    pub required_item: Option<String>,
    /// Category tag for filtering.
    pub category: String,
    /// World-space position of this interactable.
    pub position: Vec3,
}

impl Interactable {
    /// Create a simple instant interactable.
    pub fn instant(entity_id: u32, prompt: impl Into<String>, range: f32, position: Vec3) -> Self {
        Self {
            entity_id,
            prompt_text: prompt.into(),
            interaction_range: range,
            interaction_type: InteractionType::Instant,
            enabled: true,
            one_shot: false,
            used: false,
            priority: 0,
            required_item: None,
            category: String::new(),
            position,
        }
    }

    /// Create a hold-to-interact interactable.
    pub fn hold(
        entity_id: u32,
        prompt: impl Into<String>,
        range: f32,
        duration: f32,
        position: Vec3,
    ) -> Self {
        Self {
            entity_id,
            prompt_text: prompt.into(),
            interaction_range: range,
            interaction_type: InteractionType::HoldDuration(duration),
            enabled: true,
            one_shot: false,
            used: false,
            priority: 0,
            required_item: None,
            category: String::new(),
            position,
        }
    }

    /// Create a toggle interactable.
    pub fn toggle(entity_id: u32, prompt: impl Into<String>, range: f32, position: Vec3) -> Self {
        Self {
            entity_id,
            prompt_text: prompt.into(),
            interaction_range: range,
            interaction_type: InteractionType::Toggle,
            enabled: true,
            one_shot: false,
            used: false,
            priority: 0,
            required_item: None,
            category: String::new(),
            position,
        }
    }

    /// Whether the player can currently interact with this.
    pub fn can_interact(&self, player_pos: Vec3, has_required_item: bool) -> bool {
        if !self.enabled {
            return false;
        }
        if self.one_shot && self.used {
            return false;
        }
        if self.required_item.is_some() && !has_required_item {
            return false;
        }
        let distance = (self.position - player_pos).length();
        distance <= self.interaction_range
    }

    /// Squared distance to a point (avoids sqrt).
    pub fn distance_sq(&self, point: Vec3) -> f32 {
        (self.position - point).length_squared()
    }
}

// ============================================================================
// Interaction events
// ============================================================================

/// Events emitted by the interaction system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionEvent {
    /// A prompt should be shown to the player.
    ShowPrompt {
        entity_id: u32,
        prompt_text: String,
    },
    /// The prompt should be hidden.
    HidePrompt,
    /// A hold-interaction has started.
    HoldStarted {
        entity_id: u32,
    },
    /// A hold-interaction is in progress.
    HoldProgress {
        entity_id: u32,
        progress: f32,
    },
    /// A hold-interaction was cancelled.
    HoldCancelled {
        entity_id: u32,
    },
    /// An interaction was completed.
    InteractionComplete {
        entity_id: u32,
        interaction_type: InteractionType,
    },
    /// An interaction failed (missing required item, etc.).
    InteractionFailed {
        entity_id: u32,
        reason: String,
    },
    /// A door state changed.
    DoorStateChanged {
        entity_id: u32,
        is_open: bool,
    },
    /// An item was picked up.
    ItemPickedUp {
        entity_id: u32,
        item_id: String,
        count: u32,
    },
    /// A switch was toggled.
    SwitchToggled {
        entity_id: u32,
        is_active: bool,
        linked_entities: Vec<u32>,
    },
    /// A dialogue was triggered.
    DialogueStarted {
        entity_id: u32,
        dialogue_id: String,
    },
}

// ============================================================================
// Interaction system
// ============================================================================

/// Active hold state when the player is holding the interact button.
#[derive(Debug, Clone)]
struct HoldState {
    entity_id: u32,
    elapsed: f32,
    total: f32,
}

/// The interaction system manages finding nearby interactables, displaying
/// prompts, processing hold-to-interact, and dispatching events.
#[derive(Debug)]
pub struct InteractionSystem {
    interactables: Vec<Interactable>,
    /// The entity ID of the currently focused (nearest) interactable.
    focused: Option<u32>,
    /// Active hold-to-interact state.
    hold_state: Option<HoldState>,
    /// Pending events to be consumed by the game.
    pending_events: Vec<InteractionEvent>,
    /// Maximum number of interactables to consider (performance guard).
    pub max_search_count: usize,
}

impl Default for InteractionSystem {
    fn default() -> Self {
        Self {
            interactables: Vec::new(),
            focused: None,
            hold_state: None,
            pending_events: Vec::new(),
            max_search_count: 256,
        }
    }
}

impl InteractionSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an interactable.
    pub fn add(&mut self, interactable: Interactable) {
        self.interactables.push(interactable);
    }

    /// Remove an interactable by entity ID.
    pub fn remove(&mut self, entity_id: u32) {
        self.interactables.retain(|i| i.entity_id != entity_id);
        if self.focused == Some(entity_id) {
            self.focused = None;
            self.pending_events.push(InteractionEvent::HidePrompt);
        }
    }

    /// Update the interactable's position (e.g. for moving platforms).
    pub fn update_position(&mut self, entity_id: u32, new_pos: Vec3) {
        if let Some(interactable) = self
            .interactables
            .iter_mut()
            .find(|i| i.entity_id == entity_id)
        {
            interactable.position = new_pos;
        }
    }

    /// Enable or disable an interactable.
    pub fn set_enabled(&mut self, entity_id: u32, enabled: bool) {
        if let Some(interactable) = self
            .interactables
            .iter_mut()
            .find(|i| i.entity_id == entity_id)
        {
            interactable.enabled = enabled;
        }
    }

    /// Get a reference to an interactable by ID.
    pub fn get(&self, entity_id: u32) -> Option<&Interactable> {
        self.interactables.iter().find(|i| i.entity_id == entity_id)
    }

    /// Get a mutable reference to an interactable by ID.
    pub fn get_mut(&mut self, entity_id: u32) -> Option<&mut Interactable> {
        self.interactables
            .iter_mut()
            .find(|i| i.entity_id == entity_id)
    }

    /// Find the nearest interactable within range of the player. Updates the
    /// focus and generates prompt show/hide events.
    pub fn update_focus(&mut self, player_pos: Vec3) {
        let mut best: Option<(u32, f32)> = None;

        for interactable in &self.interactables {
            if !interactable.enabled {
                continue;
            }
            if interactable.one_shot && interactable.used {
                continue;
            }

            let dist_sq = interactable.distance_sq(player_pos);
            let range_sq = interactable.interaction_range * interactable.interaction_range;

            if dist_sq <= range_sq {
                let is_better = match best {
                    None => true,
                    Some((_, best_dist)) => {
                        // Prefer higher priority, then closer distance
                        dist_sq < best_dist
                    }
                };
                if is_better {
                    best = Some((interactable.entity_id, dist_sq));
                }
            }
        }

        let new_focus = best.map(|(id, _)| id);

        if new_focus != self.focused {
            // Hide old prompt
            if self.focused.is_some() {
                self.pending_events.push(InteractionEvent::HidePrompt);
                // Cancel any active hold
                if let Some(hold) = self.hold_state.take() {
                    self.pending_events.push(InteractionEvent::HoldCancelled {
                        entity_id: hold.entity_id,
                    });
                }
            }

            // Show new prompt
            if let Some(id) = new_focus {
                if let Some(interactable) = self.get(id) {
                    self.pending_events.push(InteractionEvent::ShowPrompt {
                        entity_id: id,
                        prompt_text: interactable.prompt_text.clone(),
                    });
                }
            }

            self.focused = new_focus;
        }
    }

    /// Attempt to start an interaction with the currently focused interactable.
    /// For instant interactions, completes immediately. For hold interactions,
    /// starts the hold timer. For toggles, toggles state.
    pub fn try_interact(&mut self, player_pos: Vec3, has_required_item: bool) {
        let Some(entity_id) = self.focused else {
            return;
        };

        // Find the interactable
        let interactable = match self
            .interactables
            .iter()
            .find(|i| i.entity_id == entity_id)
        {
            Some(i) => i,
            None => return,
        };

        if !interactable.can_interact(player_pos, has_required_item) {
            if interactable.required_item.is_some() && !has_required_item {
                self.pending_events.push(InteractionEvent::InteractionFailed {
                    entity_id,
                    reason: format!(
                        "Requires item: {}",
                        interactable.required_item.as_deref().unwrap_or("unknown")
                    ),
                });
            }
            return;
        }

        let interaction_type = interactable.interaction_type;

        match interaction_type {
            InteractionType::Instant => {
                self.complete_interaction(entity_id);
            }
            InteractionType::HoldDuration(duration) => {
                if self.hold_state.is_none() {
                    self.hold_state = Some(HoldState {
                        entity_id,
                        elapsed: 0.0,
                        total: duration,
                    });
                    self.pending_events
                        .push(InteractionEvent::HoldStarted { entity_id });
                }
            }
            InteractionType::Toggle => {
                self.complete_interaction(entity_id);
            }
        }
    }

    /// Update hold-to-interact progress. Call each frame while the interact
    /// button is held.
    pub fn update_hold(&mut self, dt: f32, is_button_held: bool) {
        let Some(hold) = &mut self.hold_state else {
            return;
        };

        if !is_button_held {
            let entity_id = hold.entity_id;
            self.hold_state = None;
            self.pending_events
                .push(InteractionEvent::HoldCancelled { entity_id });
            return;
        }

        hold.elapsed += dt;
        let progress = (hold.elapsed / hold.total).clamp(0.0, 1.0);

        self.pending_events.push(InteractionEvent::HoldProgress {
            entity_id: hold.entity_id,
            progress,
        });

        if hold.elapsed >= hold.total {
            let entity_id = hold.entity_id;
            self.hold_state = None;
            self.complete_interaction(entity_id);
        }
    }

    /// Cancel any active hold interaction.
    pub fn cancel_hold(&mut self) {
        if let Some(hold) = self.hold_state.take() {
            self.pending_events.push(InteractionEvent::HoldCancelled {
                entity_id: hold.entity_id,
            });
        }
    }

    /// Mark an interaction as complete and emit the event.
    fn complete_interaction(&mut self, entity_id: u32) {
        let interaction_type = self
            .interactables
            .iter()
            .find(|i| i.entity_id == entity_id)
            .map(|i| i.interaction_type)
            .unwrap_or(InteractionType::Instant);

        // Mark as used for one-shot interactables
        if let Some(interactable) = self
            .interactables
            .iter_mut()
            .find(|i| i.entity_id == entity_id)
        {
            if interactable.one_shot {
                interactable.used = true;
            }
        }

        self.pending_events
            .push(InteractionEvent::InteractionComplete {
                entity_id,
                interaction_type,
            });
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<InteractionEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Get the currently focused entity ID.
    pub fn focused_entity(&self) -> Option<u32> {
        self.focused
    }

    /// Whether a hold interaction is currently in progress.
    pub fn is_holding(&self) -> bool {
        self.hold_state.is_some()
    }

    /// Progress of the current hold interaction [0, 1].
    pub fn hold_progress(&self) -> f32 {
        self.hold_state
            .as_ref()
            .map(|h| (h.elapsed / h.total).clamp(0.0, 1.0))
            .unwrap_or(0.0)
    }

    /// Number of registered interactables.
    pub fn count(&self) -> usize {
        self.interactables.len()
    }
}

// ============================================================================
// Concrete interactable types
// ============================================================================

/// A door that can be opened and closed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Door {
    /// Entity ID of the door.
    pub entity_id: u32,
    /// Whether the door is currently open.
    pub is_open: bool,
    /// Whether the door is locked.
    pub locked: bool,
    /// Item ID required to unlock (if locked).
    pub key_item: Option<String>,
    /// Animation trigger name for opening.
    pub open_animation: String,
    /// Animation trigger name for closing.
    pub close_animation: String,
    /// Sound effect name.
    pub sound: String,
    /// Time to open/close (seconds).
    pub transition_time: f32,
    /// Whether the door auto-closes after a delay.
    pub auto_close: bool,
    /// Auto-close delay in seconds.
    pub auto_close_delay: f32,
    /// Timer for auto-close.
    auto_close_timer: f32,
    /// Whether the door is currently in transition.
    pub in_transition: bool,
    /// Transition progress [0, 1].
    transition_progress: f32,
}

impl Door {
    /// Create a new unlocked door.
    pub fn new(entity_id: u32) -> Self {
        Self {
            entity_id,
            is_open: false,
            locked: false,
            key_item: None,
            open_animation: "door_open".into(),
            close_animation: "door_close".into(),
            sound: "door_creak".into(),
            transition_time: 0.5,
            auto_close: false,
            auto_close_delay: 5.0,
            auto_close_timer: 0.0,
            in_transition: false,
            transition_progress: 0.0,
        }
    }

    /// Create a locked door.
    pub fn locked(entity_id: u32, key_item: impl Into<String>) -> Self {
        Self {
            locked: true,
            key_item: Some(key_item.into()),
            ..Self::new(entity_id)
        }
    }

    /// Try to toggle the door. Returns an event if successful.
    pub fn toggle(&mut self, has_key: bool) -> Option<InteractionEvent> {
        if self.locked && !has_key {
            return Some(InteractionEvent::InteractionFailed {
                entity_id: self.entity_id,
                reason: "Door is locked".into(),
            });
        }

        if self.in_transition {
            return None;
        }

        if self.locked && has_key {
            self.locked = false;
        }

        self.is_open = !self.is_open;
        self.in_transition = true;
        self.transition_progress = 0.0;

        if self.is_open && self.auto_close {
            self.auto_close_timer = 0.0;
        }

        Some(InteractionEvent::DoorStateChanged {
            entity_id: self.entity_id,
            is_open: self.is_open,
        })
    }

    /// Update door animation and auto-close timer.
    pub fn update(&mut self, dt: f32) -> Option<InteractionEvent> {
        if self.in_transition {
            self.transition_progress += dt / self.transition_time;
            if self.transition_progress >= 1.0 {
                self.transition_progress = 1.0;
                self.in_transition = false;
            }
        }

        if self.is_open && self.auto_close && !self.in_transition {
            self.auto_close_timer += dt;
            if self.auto_close_timer >= self.auto_close_delay {
                self.auto_close_timer = 0.0;
                return self.toggle(true);
            }
        }

        None
    }

    /// Current animation progress [0, 1].
    pub fn animation_progress(&self) -> f32 {
        self.transition_progress
    }

    /// The animation name to play.
    pub fn current_animation(&self) -> &str {
        if self.is_open {
            &self.open_animation
        } else {
            &self.close_animation
        }
    }
}

/// An item pickup in the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pickup {
    /// Entity ID of the pickup.
    pub entity_id: u32,
    /// Item ID that will be added to the player's inventory.
    pub item_id: String,
    /// Number of items in this pickup.
    pub count: u32,
    /// Whether the pickup has been collected.
    pub collected: bool,
    /// Optional float/bob animation amplitude.
    pub bob_amplitude: f32,
    /// Optional rotation speed (radians/sec).
    pub rotation_speed: f32,
    /// Sound effect on pickup.
    pub sound: String,
    /// Visual effect on pickup.
    pub vfx: String,
    /// Respawn time (0 = no respawn).
    pub respawn_time: f32,
    /// Timer for respawn.
    respawn_timer: f32,
}

impl Pickup {
    /// Create a new pickup.
    pub fn new(entity_id: u32, item_id: impl Into<String>, count: u32) -> Self {
        Self {
            entity_id,
            item_id: item_id.into(),
            count,
            collected: false,
            bob_amplitude: 0.15,
            rotation_speed: 1.5,
            sound: "pickup_collect".into(),
            vfx: "pickup_sparkle".into(),
            respawn_time: 0.0,
            respawn_timer: 0.0,
        }
    }

    /// Collect the pickup. Returns the event with item info.
    pub fn collect(&mut self) -> Option<InteractionEvent> {
        if self.collected {
            return None;
        }
        self.collected = true;
        Some(InteractionEvent::ItemPickedUp {
            entity_id: self.entity_id,
            item_id: self.item_id.clone(),
            count: self.count,
        })
    }

    /// Update respawn timer. Returns `true` if the pickup respawned.
    pub fn update(&mut self, dt: f32) -> bool {
        if self.collected && self.respawn_time > 0.0 {
            self.respawn_timer += dt;
            if self.respawn_timer >= self.respawn_time {
                self.respawn_timer = 0.0;
                self.collected = false;
                return true;
            }
        }
        false
    }

    /// Whether this pickup is currently visible/active.
    pub fn is_active(&self) -> bool {
        !self.collected
    }
}

/// A switch or lever that toggles state and activates linked entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Switch {
    /// Entity ID of the switch.
    pub entity_id: u32,
    /// Whether the switch is currently active/on.
    pub is_active: bool,
    /// Entity IDs that are activated/deactivated when this switch toggles.
    pub linked_entities: Vec<u32>,
    /// Sound effect name.
    pub sound: String,
    /// Animation name.
    pub animation: String,
    /// Cooldown between toggles (prevents spamming).
    pub cooldown: f32,
    /// Time since last toggle.
    cooldown_timer: f32,
}

impl Switch {
    /// Create a new switch.
    pub fn new(entity_id: u32, linked_entities: Vec<u32>) -> Self {
        Self {
            entity_id,
            is_active: false,
            linked_entities,
            sound: "switch_click".into(),
            animation: "switch_toggle".into(),
            cooldown: 0.5,
            cooldown_timer: 0.0,
        }
    }

    /// Toggle the switch. Returns the event.
    pub fn toggle(&mut self) -> Option<InteractionEvent> {
        if self.cooldown_timer > 0.0 {
            return None;
        }

        self.is_active = !self.is_active;
        self.cooldown_timer = self.cooldown;

        Some(InteractionEvent::SwitchToggled {
            entity_id: self.entity_id,
            is_active: self.is_active,
            linked_entities: self.linked_entities.clone(),
        })
    }

    /// Update cooldown timer.
    pub fn update(&mut self, dt: f32) {
        if self.cooldown_timer > 0.0 {
            self.cooldown_timer = (self.cooldown_timer - dt).max(0.0);
        }
    }

    /// Whether the switch can be toggled right now.
    pub fn can_toggle(&self) -> bool {
        self.cooldown_timer <= 0.0
    }
}

/// A trigger that starts a dialogue conversation when interacted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTrigger {
    /// Entity ID of the NPC / trigger.
    pub entity_id: u32,
    /// Dialogue tree ID to start.
    pub dialogue_id: String,
    /// Whether the dialogue has been started (for one-time triggers).
    pub triggered: bool,
    /// Whether this trigger is one-time only.
    pub one_shot: bool,
    /// Optional quest ID that must be active for this dialogue to appear.
    pub required_quest: Option<String>,
    /// NPC display name.
    pub speaker_name: String,
}

impl DialogueTrigger {
    /// Create a new dialogue trigger.
    pub fn new(
        entity_id: u32,
        dialogue_id: impl Into<String>,
        speaker_name: impl Into<String>,
    ) -> Self {
        Self {
            entity_id,
            dialogue_id: dialogue_id.into(),
            triggered: false,
            one_shot: false,
            required_quest: None,
            speaker_name: speaker_name.into(),
        }
    }

    /// Trigger the dialogue. Returns the event.
    pub fn trigger(&mut self) -> Option<InteractionEvent> {
        if self.one_shot && self.triggered {
            return None;
        }
        self.triggered = true;
        Some(InteractionEvent::DialogueStarted {
            entity_id: self.entity_id,
            dialogue_id: self.dialogue_id.clone(),
        })
    }

    /// Whether this dialogue can be triggered.
    pub fn can_trigger(&self) -> bool {
        !(self.one_shot && self.triggered)
    }

    /// Reset the trigger (for repeatable dialogues or game restart).
    pub fn reset(&mut self) {
        self.triggered = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn player_pos() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }

    #[test]
    fn test_interactable_in_range() {
        let interactable = Interactable::instant(1, "Open", 3.0, Vec3::new(2.0, 0.0, 0.0));
        assert!(interactable.can_interact(player_pos(), true));
    }

    #[test]
    fn test_interactable_out_of_range() {
        let interactable = Interactable::instant(1, "Open", 3.0, Vec3::new(5.0, 0.0, 0.0));
        assert!(!interactable.can_interact(player_pos(), true));
    }

    #[test]
    fn test_interactable_disabled() {
        let mut interactable = Interactable::instant(1, "Open", 3.0, Vec3::new(1.0, 0.0, 0.0));
        interactable.enabled = false;
        assert!(!interactable.can_interact(player_pos(), true));
    }

    #[test]
    fn test_interactable_one_shot() {
        let mut interactable = Interactable::instant(1, "Open", 3.0, Vec3::new(1.0, 0.0, 0.0));
        interactable.one_shot = true;
        assert!(interactable.can_interact(player_pos(), true));
        interactable.used = true;
        assert!(!interactable.can_interact(player_pos(), true));
    }

    #[test]
    fn test_interactable_required_item() {
        let mut interactable = Interactable::instant(1, "Unlock", 3.0, Vec3::new(1.0, 0.0, 0.0));
        interactable.required_item = Some("key_gold".into());
        assert!(!interactable.can_interact(player_pos(), false));
        assert!(interactable.can_interact(player_pos(), true));
    }

    #[test]
    fn test_interaction_system_focus() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::instant(
            1,
            "Chest",
            5.0,
            Vec3::new(3.0, 0.0, 0.0),
        ));
        system.add(Interactable::instant(
            2,
            "Door",
            5.0,
            Vec3::new(4.0, 0.0, 0.0),
        ));

        system.update_focus(player_pos());
        // Entity 1 is closer
        assert_eq!(system.focused_entity(), Some(1));

        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::ShowPrompt { entity_id: 1, .. })));
    }

    #[test]
    fn test_interaction_system_focus_change() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::instant(
            1,
            "A",
            3.0,
            Vec3::new(2.0, 0.0, 0.0),
        ));
        system.add(Interactable::instant(
            2,
            "B",
            3.0,
            Vec3::new(10.0, 0.0, 0.0),
        ));

        system.update_focus(player_pos());
        assert_eq!(system.focused_entity(), Some(1));
        system.drain_events();

        // Move player closer to entity 2
        system.update_focus(Vec3::new(9.0, 0.0, 0.0));
        assert_eq!(system.focused_entity(), Some(2));

        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::HidePrompt)));
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::ShowPrompt { entity_id: 2, .. })));
    }

    #[test]
    fn test_interaction_system_instant_interact() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::instant(
            1,
            "Open",
            5.0,
            Vec3::new(1.0, 0.0, 0.0),
        ));

        system.update_focus(player_pos());
        system.drain_events();

        system.try_interact(player_pos(), true);
        let events = system.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, InteractionEvent::InteractionComplete { entity_id: 1, .. })));
    }

    #[test]
    fn test_interaction_system_hold() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::hold(
            1,
            "Hold to Open",
            5.0,
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
        ));

        system.update_focus(player_pos());
        system.drain_events();

        system.try_interact(player_pos(), true);
        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::HoldStarted { entity_id: 1 })));
        assert!(system.is_holding());

        // Partial hold
        system.update_hold(1.0, true);
        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::HoldProgress { progress, .. } if *progress > 0.4)));

        // Complete hold
        system.update_hold(1.5, true);
        let events = system.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, InteractionEvent::InteractionComplete { entity_id: 1, .. })));
        assert!(!system.is_holding());
    }

    #[test]
    fn test_interaction_system_hold_cancel() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::hold(
            1,
            "Hold",
            5.0,
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
        ));

        system.update_focus(player_pos());
        system.drain_events();

        system.try_interact(player_pos(), true);
        system.drain_events();

        // Release button
        system.update_hold(0.5, false);
        let events = system.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::HoldCancelled { entity_id: 1 })));
        assert!(!system.is_holding());
    }

    #[test]
    fn test_door_toggle() {
        let mut door = Door::new(1);
        assert!(!door.is_open);

        let event = door.toggle(false);
        assert!(event.is_some());
        assert!(door.is_open);

        let event = door.toggle(false);
        assert!(event.is_some());
        assert!(!door.is_open);
    }

    #[test]
    fn test_door_locked() {
        let mut door = Door::locked(1, "golden_key");
        let event = door.toggle(false);
        assert!(matches!(event, Some(InteractionEvent::InteractionFailed { .. })));
        assert!(!door.is_open);

        let event = door.toggle(true);
        assert!(matches!(event, Some(InteractionEvent::DoorStateChanged { is_open: true, .. })));
        assert!(door.is_open);
        assert!(!door.locked); // Unlocked after using key
    }

    #[test]
    fn test_door_auto_close() {
        let mut door = Door::new(1);
        door.auto_close = true;
        door.auto_close_delay = 1.0;
        door.transition_time = 0.1;

        door.toggle(false);
        assert!(door.is_open);

        // Complete the open transition
        door.update(0.2);
        assert!(!door.in_transition);

        // Wait for auto-close
        let event = door.update(1.1);
        assert!(event.is_some());
        assert!(!door.is_open);
    }

    #[test]
    fn test_pickup_collect() {
        let mut pickup = Pickup::new(1, "health_potion", 3);
        assert!(pickup.is_active());

        let event = pickup.collect();
        assert!(matches!(event, Some(InteractionEvent::ItemPickedUp { count: 3, .. })));
        assert!(!pickup.is_active());

        // Can't collect again
        let event = pickup.collect();
        assert!(event.is_none());
    }

    #[test]
    fn test_pickup_respawn() {
        let mut pickup = Pickup::new(1, "ammo", 10);
        pickup.respawn_time = 2.0;

        pickup.collect();
        assert!(!pickup.is_active());

        assert!(!pickup.update(1.0));
        assert!(!pickup.is_active());

        assert!(pickup.update(1.5));
        assert!(pickup.is_active());
    }

    #[test]
    fn test_switch_toggle() {
        let mut switch = Switch::new(1, vec![10, 11, 12]);
        assert!(!switch.is_active);

        let event = switch.toggle();
        assert!(matches!(
            event,
            Some(InteractionEvent::SwitchToggled { is_active: true, .. })
        ));
        assert!(switch.is_active);

        // Cooldown prevents immediate re-toggle
        assert!(switch.toggle().is_none());

        // Wait out cooldown
        switch.update(0.6);
        let event = switch.toggle();
        assert!(matches!(
            event,
            Some(InteractionEvent::SwitchToggled {
                is_active: false,
                ..
            })
        ));
    }

    #[test]
    fn test_dialogue_trigger() {
        let mut trigger = DialogueTrigger::new(1, "intro_dialogue", "Guard");
        assert!(trigger.can_trigger());

        let event = trigger.trigger();
        assert!(matches!(event, Some(InteractionEvent::DialogueStarted { .. })));

        // Can trigger again (not one-shot)
        let event = trigger.trigger();
        assert!(event.is_some());
    }

    #[test]
    fn test_dialogue_trigger_one_shot() {
        let mut trigger = DialogueTrigger::new(1, "boss_intro", "Boss");
        trigger.one_shot = true;

        trigger.trigger();
        assert!(trigger.triggered);
        assert!(!trigger.can_trigger());

        let event = trigger.trigger();
        assert!(event.is_none());

        // Reset allows re-trigger
        trigger.reset();
        assert!(trigger.can_trigger());
    }

    #[test]
    fn test_interaction_system_remove() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::instant(1, "A", 5.0, Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(system.count(), 1);

        system.update_focus(player_pos());
        assert_eq!(system.focused_entity(), Some(1));

        system.remove(1);
        assert_eq!(system.count(), 0);
        assert_eq!(system.focused_entity(), None);
    }

    #[test]
    fn test_interaction_system_update_position() {
        let mut system = InteractionSystem::new();
        system.add(Interactable::instant(1, "Moving", 2.0, Vec3::new(10.0, 0.0, 0.0)));

        system.update_focus(player_pos());
        assert_eq!(system.focused_entity(), None); // Too far

        system.update_position(1, Vec3::new(1.0, 0.0, 0.0));
        system.update_focus(player_pos());
        assert_eq!(system.focused_entity(), Some(1)); // Now in range
    }
}
