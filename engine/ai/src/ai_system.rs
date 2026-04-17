// engine/ai/src/ai_system.rs
//
// Central AI system for the Genovo engine.
//
// Manages all AI agents: ticks perception, decision, and action phases.
// Implements LOD AI (reduces update rate for distant agents) and AI pooling.

use std::collections::HashMap;

pub const MAX_AI_AGENTS: usize = 512;
pub const DEFAULT_FULL_UPDATE_RANGE: f32 = 50.0;
pub const DEFAULT_MEDIUM_UPDATE_RANGE: f32 = 100.0;
pub const DEFAULT_LOW_UPDATE_RANGE: f32 = 200.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiLodLevel { Full, Medium, Low, Dormant }

impl AiLodLevel {
    pub fn update_interval(&self) -> f32 {
        match self { Self::Full => 0.0, Self::Medium => 0.1, Self::Low => 0.5, Self::Dormant => 2.0 }
    }
    pub fn from_distance(dist: f32, ranges: &AiLodRanges) -> Self {
        if dist <= ranges.full { Self::Full } else if dist <= ranges.medium { Self::Medium }
        else if dist <= ranges.low { Self::Low } else { Self::Dormant }
    }
}

#[derive(Debug, Clone)]
pub struct AiLodRanges { pub full: f32, pub medium: f32, pub low: f32 }
impl Default for AiLodRanges { fn default() -> Self { Self { full: DEFAULT_FULL_UPDATE_RANGE, medium: DEFAULT_MEDIUM_UPDATE_RANGE, low: DEFAULT_LOW_UPDATE_RANGE } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiPhase { Perception, Decision, Action }

#[derive(Debug, Clone)]
pub struct AiAgent {
    pub id: u64,
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub lod: AiLodLevel,
    pub update_timer: f32,
    pub active: bool,
    pub behavior_tree_id: Option<u32>,
    pub current_action: Option<String>,
    pub blackboard: HashMap<String, AiValue>,
    pub perception_data: PerceptionData,
    pub priority: f32,
    pub group_id: Option<u32>,
    pub health_fraction: f32,
    pub alert_level: AlertLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertLevel { Idle, Suspicious, Alert, Combat }

#[derive(Debug, Clone)]
pub enum AiValue { Float(f32), Int(i32), Bool(bool), Vec3([f32; 3]), Entity(u64), String(String) }

#[derive(Debug, Clone, Default)]
pub struct PerceptionData {
    pub visible_entities: Vec<PerceivedEntity>,
    pub heard_sounds: Vec<PerceivedSound>,
    pub threat_level: f32,
    pub last_known_threat_pos: Option<[f32; 3]>,
}

#[derive(Debug, Clone)]
pub struct PerceivedEntity { pub id: u64, pub position: [f32; 3], pub is_threat: bool, pub distance: f32, pub last_seen: f32 }
#[derive(Debug, Clone)]
pub struct PerceivedSound { pub position: [f32; 3], pub loudness: f32, pub sound_type: SoundCategory, pub time: f32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum SoundCategory { Footstep, Gunshot, Explosion, Voice, Impact, Ambient }

impl AiAgent {
    pub fn new(id: u64, position: [f32; 3]) -> Self {
        Self { id, position, forward: [0.0, 0.0, 1.0], lod: AiLodLevel::Full, update_timer: 0.0, active: true,
            behavior_tree_id: None, current_action: None, blackboard: HashMap::new(), perception_data: PerceptionData::default(),
            priority: 1.0, group_id: None, health_fraction: 1.0, alert_level: AlertLevel::Idle }
    }

    pub fn should_update(&mut self, dt: f32) -> bool {
        self.update_timer -= dt;
        if self.update_timer <= 0.0 { self.update_timer = self.lod.update_interval(); true } else { false }
    }

    pub fn set_blackboard(&mut self, key: &str, value: AiValue) { self.blackboard.insert(key.to_string(), value); }
    pub fn get_blackboard_float(&self, key: &str) -> Option<f32> { match self.blackboard.get(key) { Some(AiValue::Float(v)) => Some(*v), _ => None } }
    pub fn get_blackboard_bool(&self, key: &str) -> Option<bool> { match self.blackboard.get(key) { Some(AiValue::Bool(v)) => Some(*v), _ => None } }
    pub fn nearest_threat(&self) -> Option<&PerceivedEntity> { self.perception_data.visible_entities.iter().filter(|e| e.is_threat).min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)) }
}

#[derive(Debug, Clone, Default)]
pub struct AiSystemStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub full_lod: usize,
    pub medium_lod: usize,
    pub low_lod: usize,
    pub dormant: usize,
    pub agents_updated: usize,
    pub perception_queries: usize,
    pub decisions_made: usize,
}

#[derive(Debug)]
pub struct AiSystemConfig {
    pub lod_ranges: AiLodRanges,
    pub max_full_updates_per_frame: usize,
    pub max_medium_updates_per_frame: usize,
    pub stagger_updates: bool,
    pub perception_budget_ms: f32,
}

impl Default for AiSystemConfig { fn default() -> Self { Self { lod_ranges: AiLodRanges::default(), max_full_updates_per_frame: 32, max_medium_updates_per_frame: 16, stagger_updates: true, perception_budget_ms: 2.0 } } }

/// Pool of reusable AI agent slots.
#[derive(Debug)]
pub struct AiPool {
    free_ids: Vec<u64>,
    next_id: u64,
}

impl AiPool {
    pub fn new() -> Self { Self { free_ids: Vec::new(), next_id: 1 } }
    pub fn acquire(&mut self) -> u64 { self.free_ids.pop().unwrap_or_else(|| { let id = self.next_id; self.next_id += 1; id }) }
    pub fn release(&mut self, id: u64) { self.free_ids.push(id); }
}

#[derive(Debug)]
pub struct AiSystem {
    pub config: AiSystemConfig,
    pub agents: HashMap<u64, AiAgent>,
    pub pool: AiPool,
    pub stats: AiSystemStats,
    pub camera_position: [f32; 3],
    pub groups: HashMap<u32, Vec<u64>>,
    frame: u64,
}

impl AiSystem {
    pub fn new(config: AiSystemConfig) -> Self {
        Self { config, agents: HashMap::new(), pool: AiPool::new(), stats: AiSystemStats::default(), camera_position: [0.0; 3], groups: HashMap::new(), frame: 0 }
    }

    pub fn spawn_agent(&mut self, position: [f32; 3]) -> u64 {
        let id = self.pool.acquire();
        let agent = AiAgent::new(id, position);
        self.agents.insert(id, agent);
        id
    }

    pub fn despawn_agent(&mut self, id: u64) {
        if let Some(agent) = self.agents.remove(&id) {
            if let Some(gid) = agent.group_id {
                if let Some(group) = self.groups.get_mut(&gid) { group.retain(|&i| i != id); }
            }
            self.pool.release(id);
        }
    }

    pub fn get_agent(&self, id: u64) -> Option<&AiAgent> { self.agents.get(&id) }
    pub fn get_agent_mut(&mut self, id: u64) -> Option<&mut AiAgent> { self.agents.get_mut(&id) }

    pub fn assign_group(&mut self, agent_id: u64, group_id: u32) {
        if let Some(agent) = self.agents.get_mut(&agent_id) { agent.group_id = Some(group_id); }
        self.groups.entry(group_id).or_default().push(agent_id);
    }

    pub fn update(&mut self, dt: f32, camera_pos: [f32; 3]) {
        self.frame += 1;
        self.camera_position = camera_pos;
        self.stats = AiSystemStats::default();
        self.stats.total_agents = self.agents.len();

        let mut update_list: Vec<(u64, AiLodLevel)> = Vec::new();

        for (id, agent) in &mut self.agents {
            if !agent.active { continue; }
            self.stats.active_agents += 1;

            let dx = agent.position[0] - camera_pos[0];
            let dy = agent.position[1] - camera_pos[1];
            let dz = agent.position[2] - camera_pos[2];
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            agent.lod = AiLodLevel::from_distance(dist, &self.config.lod_ranges);

            match agent.lod {
                AiLodLevel::Full => self.stats.full_lod += 1,
                AiLodLevel::Medium => self.stats.medium_lod += 1,
                AiLodLevel::Low => self.stats.low_lod += 1,
                AiLodLevel::Dormant => self.stats.dormant += 1,
            }

            if agent.should_update(dt) { update_list.push((*id, agent.lod)); }
        }

        // Budget-limited updates.
        let mut full_count = 0usize;
        let mut medium_count = 0usize;
        for (id, lod) in &update_list {
            match lod {
                AiLodLevel::Full => { if full_count >= self.config.max_full_updates_per_frame { continue; } full_count += 1; }
                AiLodLevel::Medium => { if medium_count >= self.config.max_medium_updates_per_frame { continue; } medium_count += 1; }
                _ => {}
            }
            self.tick_agent(*id, dt, *lod);
            self.stats.agents_updated += 1;
        }
    }

    fn tick_agent(&mut self, id: u64, dt: f32, lod: AiLodLevel) {
        // Phase 1: Perception (simplified).
        if let Some(agent) = self.agents.get_mut(&id) {
            agent.perception_data.visible_entities.retain(|e| e.last_seen < 5.0);
            for e in &mut agent.perception_data.visible_entities { e.last_seen += dt; }
            self.stats.perception_queries += 1;

            // Phase 2: Decision (simplified).
            if agent.perception_data.threat_level > 0.5 { agent.alert_level = AlertLevel::Combat; }
            else if agent.perception_data.threat_level > 0.2 { agent.alert_level = AlertLevel::Alert; }
            else if agent.perception_data.threat_level > 0.0 { agent.alert_level = AlertLevel::Suspicious; }
            else { agent.alert_level = AlertLevel::Idle; }
            self.stats.decisions_made += 1;
        }
    }

    pub fn agent_count(&self) -> usize { self.agents.len() }
    pub fn active_count(&self) -> usize { self.agents.values().filter(|a| a.active).count() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_despawn() {
        let mut sys = AiSystem::new(AiSystemConfig::default());
        let id = sys.spawn_agent([0.0, 0.0, 0.0]);
        assert_eq!(sys.agent_count(), 1);
        sys.despawn_agent(id);
        assert_eq!(sys.agent_count(), 0);
    }

    #[test]
    fn test_lod_levels() {
        let ranges = AiLodRanges::default();
        assert_eq!(AiLodLevel::from_distance(10.0, &ranges), AiLodLevel::Full);
        assert_eq!(AiLodLevel::from_distance(75.0, &ranges), AiLodLevel::Medium);
        assert_eq!(AiLodLevel::from_distance(150.0, &ranges), AiLodLevel::Low);
        assert_eq!(AiLodLevel::from_distance(300.0, &ranges), AiLodLevel::Dormant);
    }

    #[test]
    fn test_agent_pool() {
        let mut pool = AiPool::new();
        let a = pool.acquire();
        let b = pool.acquire();
        assert_ne!(a, b);
        pool.release(a);
        let c = pool.acquire();
        assert_eq!(a, c);
    }

    #[test]
    fn test_blackboard() {
        let mut agent = AiAgent::new(1, [0.0; 3]);
        agent.set_blackboard("health", AiValue::Float(0.5));
        assert!((agent.get_blackboard_float("health").unwrap() - 0.5).abs() < 0.01);
    }
}
