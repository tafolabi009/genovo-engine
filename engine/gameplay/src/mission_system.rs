// engine/gameplay/src/mission_system.rs
//
// Mission/objective system for the Genovo engine.
// Supports main missions, side missions, mission chains, waypoints,
// timers, rating (bronze/silver/gold), replay, and reward scaling.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MissionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectiveId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionType { Main, Side, Daily, Weekly, Challenge, Tutorial, Hidden }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionState { Locked, Available, Active, Completed, Failed, Abandoned }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MissionRating { None, Bronze, Silver, Gold, Platinum }

impl MissionRating {
    pub fn reward_multiplier(&self) -> f32 {
        match self { Self::None => 0.0, Self::Bronze => 1.0, Self::Silver => 1.5, Self::Gold => 2.0, Self::Platinum => 3.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveType {
    Kill, Collect, Escort, Defend, Reach, Interact, Survive, Deliver,
    Destroy, Photograph, Stealth, Race, Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveState { Inactive, Active, Completed, Failed, Optional }

#[derive(Debug, Clone)]
pub struct MissionObjective {
    pub id: ObjectiveId,
    pub description: String,
    pub obj_type: ObjectiveType,
    pub state: ObjectiveState,
    pub current_count: u32,
    pub target_count: u32,
    pub optional: bool,
    pub hidden: bool,
    pub waypoint: Option<[f32; 3]>,
    pub waypoint_radius: f32,
    pub time_limit: Option<f32>,
    pub elapsed_time: f32,
    pub order: u32,
}

impl MissionObjective {
    pub fn new(id: ObjectiveId, desc: &str, obj_type: ObjectiveType, target: u32) -> Self {
        Self {
            id, description: desc.to_string(), obj_type, state: ObjectiveState::Inactive,
            current_count: 0, target_count: target, optional: false, hidden: false,
            waypoint: None, waypoint_radius: 5.0, time_limit: None, elapsed_time: 0.0, order: 0,
        }
    }
    pub fn with_waypoint(mut self, pos: [f32; 3], radius: f32) -> Self { self.waypoint = Some(pos); self.waypoint_radius = radius; self }
    pub fn with_time_limit(mut self, seconds: f32) -> Self { self.time_limit = Some(seconds); self }
    pub fn as_optional(mut self) -> Self { self.optional = true; self }
    pub fn progress_fraction(&self) -> f32 { if self.target_count == 0 { 1.0 } else { self.current_count as f32 / self.target_count as f32 } }
    pub fn is_complete(&self) -> bool { self.current_count >= self.target_count }
    pub fn is_timed_out(&self) -> bool { self.time_limit.map_or(false, |limit| self.elapsed_time >= limit) }
    pub fn advance(&mut self, amount: u32) {
        self.current_count = (self.current_count + amount).min(self.target_count);
        if self.is_complete() { self.state = ObjectiveState::Completed; }
    }
    pub fn update_timer(&mut self, dt: f32) {
        if self.state == ObjectiveState::Active { self.elapsed_time += dt; }
        if self.is_timed_out() { self.state = ObjectiveState::Failed; }
    }
}

#[derive(Debug, Clone)]
pub struct MissionReward {
    pub experience: u32,
    pub currency: u32,
    pub item_ids: Vec<u32>,
    pub reputation_faction: Option<u32>,
    pub reputation_amount: i32,
    pub unlock_mission: Option<MissionId>,
}

impl MissionReward {
    pub fn new() -> Self { Self { experience: 0, currency: 0, item_ids: Vec::new(), reputation_faction: None, reputation_amount: 0, unlock_mission: None } }
    pub fn with_xp(mut self, xp: u32) -> Self { self.experience = xp; self }
    pub fn with_currency(mut self, c: u32) -> Self { self.currency = c; self }
    pub fn with_item(mut self, id: u32) -> Self { self.item_ids.push(id); self }
    pub fn scaled(&self, rating: MissionRating) -> MissionReward {
        let mult = rating.reward_multiplier();
        MissionReward {
            experience: (self.experience as f32 * mult) as u32,
            currency: (self.currency as f32 * mult) as u32,
            item_ids: self.item_ids.clone(),
            reputation_faction: self.reputation_faction,
            reputation_amount: (self.reputation_amount as f32 * mult) as i32,
            unlock_mission: self.unlock_mission,
        }
    }
}

impl Default for MissionReward { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct RatingThresholds {
    pub bronze_time: f32,
    pub silver_time: f32,
    pub gold_time: f32,
    pub platinum_time: f32,
    pub bronze_score: u32,
    pub silver_score: u32,
    pub gold_score: u32,
    pub platinum_score: u32,
}

impl Default for RatingThresholds {
    fn default() -> Self {
        Self {
            bronze_time: 600.0, silver_time: 300.0, gold_time: 180.0, platinum_time: 120.0,
            bronze_score: 100, silver_score: 500, gold_score: 1000, platinum_score: 2000,
        }
    }
}

impl RatingThresholds {
    pub fn rate_by_time(&self, time: f32) -> MissionRating {
        if time <= self.platinum_time { MissionRating::Platinum }
        else if time <= self.gold_time { MissionRating::Gold }
        else if time <= self.silver_time { MissionRating::Silver }
        else if time <= self.bronze_time { MissionRating::Bronze }
        else { MissionRating::None }
    }
    pub fn rate_by_score(&self, score: u32) -> MissionRating {
        if score >= self.platinum_score { MissionRating::Platinum }
        else if score >= self.gold_score { MissionRating::Gold }
        else if score >= self.silver_score { MissionRating::Silver }
        else if score >= self.bronze_score { MissionRating::Bronze }
        else { MissionRating::None }
    }
}

#[derive(Debug, Clone)]
pub struct Mission {
    pub id: MissionId,
    pub name: String,
    pub description: String,
    pub mission_type: MissionType,
    pub state: MissionState,
    pub objectives: Vec<MissionObjective>,
    pub reward: MissionReward,
    pub rating_thresholds: RatingThresholds,
    pub best_rating: MissionRating,
    pub completion_count: u32,
    pub elapsed_time: f32,
    pub score: u32,
    pub chain_next: Option<MissionId>,
    pub chain_prev: Option<MissionId>,
    pub prerequisites: Vec<MissionId>,
    pub level_requirement: u32,
    pub replayable: bool,
    pub auto_track: bool,
}

impl Mission {
    pub fn new(id: MissionId, name: &str, mission_type: MissionType) -> Self {
        Self {
            id, name: name.to_string(), description: String::new(),
            mission_type, state: MissionState::Locked,
            objectives: Vec::new(), reward: MissionReward::new(),
            rating_thresholds: RatingThresholds::default(),
            best_rating: MissionRating::None, completion_count: 0,
            elapsed_time: 0.0, score: 0,
            chain_next: None, chain_prev: None,
            prerequisites: Vec::new(), level_requirement: 1,
            replayable: false, auto_track: true,
        }
    }
    pub fn add_objective(&mut self, obj: MissionObjective) { self.objectives.push(obj); }
    pub fn with_reward(mut self, reward: MissionReward) -> Self { self.reward = reward; self }
    pub fn with_chain(mut self, next: MissionId) -> Self { self.chain_next = Some(next); self }
    pub fn all_required_complete(&self) -> bool {
        self.objectives.iter().all(|o| o.optional || o.state == ObjectiveState::Completed)
    }
    pub fn any_failed(&self) -> bool { self.objectives.iter().any(|o| !o.optional && o.state == ObjectiveState::Failed) }
    pub fn progress_fraction(&self) -> f32 {
        let required: Vec<_> = self.objectives.iter().filter(|o| !o.optional).collect();
        if required.is_empty() { return 1.0; }
        let total: f32 = required.iter().map(|o| o.progress_fraction()).sum();
        total / required.len() as f32
    }
    pub fn start(&mut self) {
        self.state = MissionState::Active;
        self.elapsed_time = 0.0;
        self.score = 0;
        for obj in &mut self.objectives { if obj.order == 0 { obj.state = ObjectiveState::Active; } }
    }
    pub fn complete(&mut self) -> MissionRating {
        self.state = MissionState::Completed;
        self.completion_count += 1;
        let time_rating = self.rating_thresholds.rate_by_time(self.elapsed_time);
        let score_rating = self.rating_thresholds.rate_by_score(self.score);
        let rating = time_rating.max(score_rating);
        if rating > self.best_rating { self.best_rating = rating; }
        rating
    }
    pub fn fail(&mut self) { self.state = MissionState::Failed; }
    pub fn abandon(&mut self) {
        self.state = MissionState::Abandoned;
        for obj in &mut self.objectives { obj.state = ObjectiveState::Inactive; obj.current_count = 0; }
    }
    pub fn update(&mut self, dt: f32) {
        if self.state != MissionState::Active { return; }
        self.elapsed_time += dt;
        for obj in &mut self.objectives { obj.update_timer(dt); }
        if self.all_required_complete() { self.complete(); }
        else if self.any_failed() { self.fail(); }
    }
    pub fn active_waypoints(&self) -> Vec<[f32; 3]> {
        self.objectives.iter().filter_map(|o| {
            if o.state == ObjectiveState::Active { o.waypoint } else { None }
        }).collect()
    }
}

#[derive(Debug)]
pub struct MissionSystem {
    pub missions: HashMap<MissionId, Mission>,
    pub active_missions: Vec<MissionId>,
    pub tracked_mission: Option<MissionId>,
    pub completed_missions: Vec<MissionId>,
    pub failed_missions: Vec<MissionId>,
    pub events: Vec<MissionEvent>,
}

#[derive(Debug, Clone)]
pub enum MissionEvent {
    Started(MissionId),
    Completed(MissionId, MissionRating),
    Failed(MissionId),
    ObjectiveCompleted(MissionId, ObjectiveId),
    ObjectiveFailed(MissionId, ObjectiveId),
    Abandoned(MissionId),
}

impl MissionSystem {
    pub fn new() -> Self {
        Self {
            missions: HashMap::new(), active_missions: Vec::new(),
            tracked_mission: None, completed_missions: Vec::new(),
            failed_missions: Vec::new(), events: Vec::new(),
        }
    }
    pub fn register(&mut self, mission: Mission) { self.missions.insert(mission.id, mission); }
    pub fn start_mission(&mut self, id: MissionId) -> bool {
        if let Some(m) = self.missions.get_mut(&id) {
            if m.state != MissionState::Available && !(m.replayable && m.state == MissionState::Completed) { return false; }
            m.start();
            self.active_missions.push(id);
            if self.tracked_mission.is_none() && m.auto_track { self.tracked_mission = Some(id); }
            self.events.push(MissionEvent::Started(id));
            true
        } else { false }
    }
    pub fn update(&mut self, dt: f32) {
        let ids: Vec<_> = self.active_missions.clone();
        for id in ids {
            if let Some(m) = self.missions.get_mut(&id) {
                let prev_state = m.state;
                m.update(dt);
                if m.state == MissionState::Completed && prev_state != MissionState::Completed {
                    self.completed_missions.push(id);
                    self.active_missions.retain(|i| *i != id);
                    self.events.push(MissionEvent::Completed(id, m.best_rating));
                    if let Some(next) = m.chain_next {
                        if let Some(next_m) = self.missions.get_mut(&next) { next_m.state = MissionState::Available; }
                    }
                } else if m.state == MissionState::Failed && prev_state != MissionState::Failed {
                    self.failed_missions.push(id);
                    self.active_missions.retain(|i| *i != id);
                    self.events.push(MissionEvent::Failed(id));
                }
            }
        }
    }
    pub fn advance_objective(&mut self, mission_id: MissionId, obj_id: ObjectiveId, amount: u32) {
        if let Some(m) = self.missions.get_mut(&mission_id) {
            if let Some(obj) = m.objectives.iter_mut().find(|o| o.id == obj_id) {
                obj.advance(amount);
                if obj.is_complete() { self.events.push(MissionEvent::ObjectiveCompleted(mission_id, obj_id)); }
            }
        }
    }
    pub fn track(&mut self, id: MissionId) { self.tracked_mission = Some(id); }
    pub fn drain_events(&mut self) -> Vec<MissionEvent> { std::mem::take(&mut self.events) }
    pub fn active_count(&self) -> usize { self.active_missions.len() }
    pub fn completed_count(&self) -> usize { self.completed_missions.len() }
}

impl Default for MissionSystem { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mission_lifecycle() {
        let mut sys = MissionSystem::new();
        let mut m = Mission::new(MissionId(1), "Test Mission", MissionType::Main);
        m.state = MissionState::Available;
        m.add_objective(MissionObjective::new(ObjectiveId(1), "Kill 3 enemies", ObjectiveType::Kill, 3));
        sys.register(m);
        assert!(sys.start_mission(MissionId(1)));
        sys.advance_objective(MissionId(1), ObjectiveId(1), 3);
        sys.update(0.1);
        assert_eq!(sys.completed_count(), 1);
    }
    #[test]
    fn test_rating() {
        let t = RatingThresholds::default();
        assert_eq!(t.rate_by_time(100.0), MissionRating::Platinum);
        assert_eq!(t.rate_by_time(500.0), MissionRating::Silver);
        assert_eq!(t.rate_by_time(1000.0), MissionRating::None);
    }
    #[test]
    fn test_reward_scaling() {
        let reward = MissionReward::new().with_xp(100).with_currency(50);
        let scaled = reward.scaled(MissionRating::Gold);
        assert_eq!(scaled.experience, 200);
        assert_eq!(scaled.currency, 100);
    }
}
