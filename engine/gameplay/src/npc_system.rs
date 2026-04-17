// engine/gameplay/src/npc_system.rs
//
// NPC management system for the Genovo engine.
// Provides NPC schedules, spawning/despawning, merchant inventory,
// dialogue triggers, pathfinding integration, animations, barks,
// and relationship tracking.

use std::collections::HashMap;

pub const MAX_NPCS: usize = 1024;
pub const MAX_SCHEDULE_ENTRIES: usize = 24;
pub const MAX_BARK_COOLDOWN: f32 = 30.0;
pub const DESPAWN_DISTANCE: f32 = 200.0;
pub const SPAWN_DISTANCE: f32 = 150.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NpcId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpcState {
    Idle, Walking, Talking, Working, Sleeping, Eating, Shopping,
    Fleeing, Fighting, Dead, Sitting, Patrolling, Guarding, Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpcAnimState {
    Idle, Walk, Run, Talk, Work, Sleep, Sit, Eat, Fight, Die,
    Wave, Bow, Pray, Dance, Custom(u32),
}

#[derive(Debug, Clone)]
pub struct NpcScheduleEntry {
    pub hour_start: u32,
    pub hour_end: u32,
    pub state: NpcState,
    pub location: [f32; 3],
    pub animation: NpcAnimState,
    pub interruptible: bool,
    pub dialogue_available: bool,
    pub priority: u32,
}

impl NpcScheduleEntry {
    pub fn new(start: u32, end: u32, state: NpcState, location: [f32; 3]) -> Self {
        Self {
            hour_start: start, hour_end: end, state, location,
            animation: NpcAnimState::Idle, interruptible: true,
            dialogue_available: true, priority: 0,
        }
    }

    pub fn is_active(&self, hour: u32) -> bool {
        if self.hour_start <= self.hour_end {
            hour >= self.hour_start && hour < self.hour_end
        } else {
            hour >= self.hour_start || hour < self.hour_end
        }
    }
}

#[derive(Debug, Clone)]
pub struct NpcSchedule {
    pub entries: Vec<NpcScheduleEntry>,
    pub default_state: NpcState,
    pub default_location: [f32; 3],
}

impl NpcSchedule {
    pub fn new() -> Self {
        Self { entries: Vec::new(), default_state: NpcState::Idle, default_location: [0.0; 3] }
    }

    pub fn add(&mut self, entry: NpcScheduleEntry) {
        if self.entries.len() < MAX_SCHEDULE_ENTRIES {
            self.entries.push(entry);
        }
    }

    pub fn current_entry(&self, hour: u32) -> Option<&NpcScheduleEntry> {
        self.entries.iter().filter(|e| e.is_active(hour)).max_by_key(|e| e.priority)
    }
}

impl Default for NpcSchedule {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct MerchantItem {
    pub item_type_id: u32,
    pub stock: i32,
    pub base_price: u32,
    pub price_markup: f32,
    pub restock_rate: f32,
    pub max_stock: i32,
    pub level_requirement: u32,
}

#[derive(Debug, Clone)]
pub struct NpcMerchant {
    pub items: Vec<MerchantItem>,
    pub buy_multiplier: f32,
    pub sell_multiplier: f32,
    pub restock_interval: f32,
    pub time_since_restock: f32,
    pub currency: u32,
    pub max_currency: u32,
}

impl NpcMerchant {
    pub fn new() -> Self {
        Self {
            items: Vec::new(), buy_multiplier: 1.0, sell_multiplier: 0.3,
            restock_interval: 3600.0, time_since_restock: 0.0,
            currency: 10000, max_currency: 50000,
        }
    }

    pub fn add_item(&mut self, item: MerchantItem) { self.items.push(item); }

    pub fn restock(&mut self) {
        for item in &mut self.items {
            item.stock = (item.stock + item.restock_rate as i32).min(item.max_stock);
        }
        self.time_since_restock = 0.0;
    }

    pub fn update(&mut self, dt: f32) {
        self.time_since_restock += dt;
        if self.time_since_restock >= self.restock_interval { self.restock(); }
    }

    pub fn buy_price(&self, base: u32) -> u32 { (base as f32 * self.buy_multiplier) as u32 }
    pub fn sell_price(&self, base: u32) -> u32 { (base as f32 * self.sell_multiplier) as u32 }
}

impl Default for NpcMerchant { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueTriggerKind {
    Proximity, Interaction, QuestState, TimeOfDay, Reputation,
    ItemInInventory, FirstMeeting, Combat, Custom(u32),
}

#[derive(Debug, Clone)]
pub struct NpcDialogueTrigger {
    pub kind: DialogueTriggerKind,
    pub dialogue_id: u32,
    pub priority: u32,
    pub one_shot: bool,
    pub triggered: bool,
    pub condition_value: f32,
    pub cooldown: f32,
    pub current_cooldown: f32,
}

impl NpcDialogueTrigger {
    pub fn new(kind: DialogueTriggerKind, dialogue_id: u32) -> Self {
        Self {
            kind, dialogue_id, priority: 0, one_shot: false,
            triggered: false, condition_value: 0.0, cooldown: 0.0, current_cooldown: 0.0,
        }
    }

    pub fn can_trigger(&self) -> bool { !self.triggered && self.current_cooldown <= 0.0 }

    pub fn trigger(&mut self) {
        if self.one_shot { self.triggered = true; }
        self.current_cooldown = self.cooldown;
    }

    pub fn update(&mut self, dt: f32) {
        if self.current_cooldown > 0.0 { self.current_cooldown -= dt; }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkCategory {
    Greeting, Farewell, Idle, Combat, Hurt, Surprised, Warning,
    Thanks, Angry, Happy, Sad, Weather, TimeOfDay, Custom(u32),
}

#[derive(Debug, Clone)]
pub struct NpcBark {
    pub category: BarkCategory,
    pub text: String,
    pub audio_id: Option<u32>,
    pub weight: f32,
    pub cooldown: f32,
    pub current_cooldown: f32,
}

impl NpcBark {
    pub fn new(category: BarkCategory, text: &str) -> Self {
        Self {
            category, text: text.to_string(), audio_id: None,
            weight: 1.0, cooldown: MAX_BARK_COOLDOWN, current_cooldown: 0.0,
        }
    }

    pub fn is_available(&self) -> bool { self.current_cooldown <= 0.0 }
    pub fn play(&mut self) { self.current_cooldown = self.cooldown; }
    pub fn update(&mut self, dt: f32) { if self.current_cooldown > 0.0 { self.current_cooldown -= dt; } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpcRelationshipType {
    Neutral, Friendly, Hostile, Feared, Respected, Romantic, Rival, Ally,
}

#[derive(Debug, Clone)]
pub struct RelationshipMemory {
    pub event_type: String,
    pub disposition_change: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct NpcRelationship {
    pub target_id: u64,
    pub rel_type: NpcRelationshipType,
    pub disposition: f32,
    pub trust: f32,
    pub fear: f32,
    pub memory: Vec<RelationshipMemory>,
}

impl NpcRelationship {
    pub fn new(target_id: u64) -> Self {
        Self {
            target_id, rel_type: NpcRelationshipType::Neutral,
            disposition: 50.0, trust: 50.0, fear: 0.0, memory: Vec::new(),
        }
    }

    pub fn modify_disposition(&mut self, amount: f32) {
        self.disposition = (self.disposition + amount).clamp(0.0, 100.0);
        self.update_type();
    }

    pub fn add_event(&mut self, event: &str, change: f32, time: f64) {
        self.disposition = (self.disposition + change).clamp(0.0, 100.0);
        self.memory.push(RelationshipMemory {
            event_type: event.to_string(), disposition_change: change, timestamp: time,
        });
        self.update_type();
    }

    fn update_type(&mut self) {
        self.rel_type = if self.disposition > 80.0 { NpcRelationshipType::Friendly }
        else if self.disposition < 20.0 { NpcRelationshipType::Hostile }
        else if self.fear > 70.0 { NpcRelationshipType::Feared }
        else { NpcRelationshipType::Neutral };
    }
}

#[derive(Debug, Clone)]
pub struct NpcDefinition {
    pub id: NpcId,
    pub name: String,
    pub title: String,
    pub faction_id: u32,
    pub level: u32,
    pub position: [f32; 3],
    pub home_position: [f32; 3],
    pub state: NpcState,
    pub anim_state: NpcAnimState,
    pub schedule: NpcSchedule,
    pub merchant: Option<NpcMerchant>,
    pub dialogue_triggers: Vec<NpcDialogueTrigger>,
    pub barks: Vec<NpcBark>,
    pub relationships: HashMap<u64, NpcRelationship>,
    pub spawned: bool,
    pub essential: bool,
    pub patrol_waypoints: Vec<[f32; 3]>,
    pub current_waypoint: usize,
    pub move_speed: f32,
    pub interaction_radius: f32,
    pub awareness_radius: f32,
}

impl NpcDefinition {
    pub fn new(id: NpcId, name: &str) -> Self {
        Self {
            id, name: name.to_string(), title: String::new(),
            faction_id: 0, level: 1,
            position: [0.0; 3], home_position: [0.0; 3],
            state: NpcState::Idle, anim_state: NpcAnimState::Idle,
            schedule: NpcSchedule::new(), merchant: None,
            dialogue_triggers: Vec::new(), barks: Vec::new(),
            relationships: HashMap::new(), spawned: false, essential: false,
            patrol_waypoints: Vec::new(), current_waypoint: 0,
            move_speed: 2.0, interaction_radius: 3.0, awareness_radius: 15.0,
        }
    }

    pub fn distance_to(&self, pos: [f32; 3]) -> f32 {
        let dx = self.position[0] - pos[0];
        let dy = self.position[1] - pos[1];
        let dz = self.position[2] - pos[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug)]
pub struct NpcManager {
    pub npcs: HashMap<NpcId, NpcDefinition>,
    pub spawned_ids: Vec<NpcId>,
    pub current_hour: u32,
    pub active: bool,
}

impl NpcManager {
    pub fn new() -> Self {
        Self { npcs: HashMap::new(), spawned_ids: Vec::new(), current_hour: 12, active: true }
    }

    pub fn register(&mut self, npc: NpcDefinition) -> NpcId {
        let id = npc.id;
        self.npcs.insert(id, npc);
        id
    }

    pub fn update(&mut self, dt: f32, player_pos: [f32; 3], hour: u32) {
        self.current_hour = hour;
        let ids: Vec<_> = self.npcs.keys().cloned().collect();
        for id in ids {
            if let Some(npc) = self.npcs.get_mut(&id) {
                let dist = npc.distance_to(player_pos);
                if !npc.spawned && dist < SPAWN_DISTANCE {
                    npc.spawned = true;
                    self.spawned_ids.push(id);
                } else if npc.spawned && dist > DESPAWN_DISTANCE {
                    npc.spawned = false;
                    self.spawned_ids.retain(|i| *i != id);
                }
                if let Some(entry) = npc.schedule.current_entry(hour) {
                    npc.state = entry.state;
                    npc.anim_state = entry.animation;
                }
                if let Some(ref mut merchant) = npc.merchant {
                    merchant.update(dt);
                }
                for bark in &mut npc.barks { bark.update(dt); }
                for trigger in &mut npc.dialogue_triggers { trigger.update(dt); }
            }
        }
    }

    pub fn get(&self, id: NpcId) -> Option<&NpcDefinition> { self.npcs.get(&id) }
    pub fn get_mut(&mut self, id: NpcId) -> Option<&mut NpcDefinition> { self.npcs.get_mut(&id) }
    pub fn spawned_count(&self) -> usize { self.spawned_ids.len() }
    pub fn total_count(&self) -> usize { self.npcs.len() }

    pub fn nearby_npcs(&self, pos: [f32; 3], radius: f32) -> Vec<NpcId> {
        self.npcs.iter().filter_map(|(id, npc)| {
            if npc.spawned && npc.distance_to(pos) <= radius { Some(*id) } else { None }
        }).collect()
    }
}

impl Default for NpcManager { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct NpcComponent {
    pub npc_id: NpcId,
    pub enabled: bool,
}

impl NpcComponent {
    pub fn new(npc_id: NpcId) -> Self { Self { npc_id, enabled: true } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule() {
        let mut schedule = NpcSchedule::new();
        schedule.add(NpcScheduleEntry::new(8, 17, NpcState::Working, [10.0, 0.0, 10.0]));
        assert!(schedule.current_entry(12).is_some());
        assert_eq!(schedule.current_entry(12).unwrap().state, NpcState::Working);
        assert!(schedule.current_entry(20).is_none());
    }

    #[test]
    fn test_merchant() {
        let mut m = NpcMerchant::new();
        m.add_item(MerchantItem {
            item_type_id: 1, stock: 10, base_price: 100,
            price_markup: 1.0, restock_rate: 1.0, max_stock: 20, level_requirement: 1,
        });
        assert_eq!(m.buy_price(100), 100);
        assert_eq!(m.sell_price(100), 30);
    }

    #[test]
    fn test_relationship() {
        let mut rel = NpcRelationship::new(42);
        rel.add_event("helped", 20.0, 100.0);
        assert!(rel.disposition > 50.0);
    }

    #[test]
    fn test_npc_manager() {
        let mut mgr = NpcManager::new();
        mgr.register(NpcDefinition::new(NpcId(1), "Guard"));
        mgr.update(0.1, [0.0, 0.0, 0.0], 12);
        assert_eq!(mgr.total_count(), 1);
        assert_eq!(mgr.spawned_count(), 1);
    }
}
