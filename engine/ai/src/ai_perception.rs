// engine/ai/src/ai_perception_v2.rs
// Enhanced perception: team knowledge sharing, threat assessment, target prioritization, visibility prediction.
use std::collections::{HashMap, VecDeque};
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

pub type EntityId = u32;
pub type TeamId = u32;

#[derive(Debug, Clone)]
pub struct PerceptionTarget { pub entity: EntityId, pub position: Vec3, pub velocity: Vec3, pub last_seen: f64, pub confidence: f32, pub threat_level: f32, pub is_visible: bool, pub team: TeamId }

#[derive(Debug, Clone)]
pub struct ThreatAssessment { pub target: EntityId, pub score: f32, pub distance: f32, pub is_facing_us: bool, pub weapon_type: u32, pub health_percent: f32 }

impl ThreatAssessment {
    pub fn compute(target: &PerceptionTarget, our_pos: Vec3, our_team: TeamId) -> Self {
        let dist = our_pos.distance(target.position);
        let base_threat = 1.0 / (dist.max(1.0) * 0.1);
        let facing_bonus = if target.velocity.normalize().dot(our_pos.sub(target.position).normalize()) > 0.5 { 2.0 } else { 1.0 };
        let recency = 1.0 / (1.0 + (0.0 - target.last_seen) as f32 * 0.1); // simplified
        let score = base_threat * facing_bonus * recency * target.confidence;
        Self { target: target.entity, score, distance: dist, is_facing_us: facing_bonus > 1.5, weapon_type: 0, health_percent: 1.0 }
    }
}

pub struct TeamKnowledge { pub team: TeamId, pub known_enemies: HashMap<EntityId, PerceptionTarget>, pub shared_positions: Vec<(EntityId, Vec3, f64)> }
impl TeamKnowledge {
    pub fn new(team: TeamId) -> Self { Self { team, known_enemies: HashMap::new(), shared_positions: Vec::new() } }
    pub fn share(&mut self, target: PerceptionTarget) { self.known_enemies.insert(target.entity, target.clone()); self.shared_positions.push((target.entity, target.position, target.last_seen)); }
    pub fn get_threats(&self, our_pos: Vec3) -> Vec<ThreatAssessment> {
        let mut threats: Vec<_> = self.known_enemies.values().map(|t| ThreatAssessment::compute(t, our_pos, self.team)).collect();
        threats.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        threats
    }
    pub fn predict_position(&self, entity: EntityId, time_ahead: f32) -> Option<Vec3> {
        self.known_enemies.get(&entity).map(|t| t.position.add(t.velocity.scale(time_ahead)))
    }
    pub fn forget_old(&mut self, current_time: f64, max_age: f64) {
        self.known_enemies.retain(|_, t| current_time - t.last_seen < max_age);
        self.shared_positions.retain(|&(_, _, t)| current_time - t < max_age);
    }
}

pub struct PerceptionSystemV2 { pub teams: HashMap<TeamId, TeamKnowledge>, pub view_distance: f32, pub fov_degrees: f32, pub hearing_range: f32, pub memory_duration: f64 }
impl PerceptionSystemV2 {
    pub fn new() -> Self { Self { teams: HashMap::new(), view_distance: 50.0, fov_degrees: 120.0, hearing_range: 30.0, memory_duration: 30.0 } }
    pub fn register_team(&mut self, team: TeamId) { self.teams.insert(team, TeamKnowledge::new(team)); }
    pub fn can_see(&self, observer_pos: Vec3, observer_fwd: Vec3, target_pos: Vec3) -> bool {
        let to_target = target_pos.sub(observer_pos);
        let dist = to_target.length();
        if dist > self.view_distance { return false; }
        let cos_angle = observer_fwd.dot(to_target.normalize());
        let fov_cos = (self.fov_degrees * 0.5 * std::f32::consts::PI / 180.0).cos();
        cos_angle >= fov_cos
    }
    pub fn report_sighting(&mut self, team: TeamId, target: PerceptionTarget) {
        if let Some(tk) = self.teams.get_mut(&team) { tk.share(target); }
    }
    pub fn get_priority_target(&self, team: TeamId, pos: Vec3) -> Option<EntityId> {
        self.teams.get(&team).and_then(|tk| tk.get_threats(pos).first().map(|t| t.target))
    }
    pub fn update(&mut self, current_time: f64) {
        for tk in self.teams.values_mut() { tk.forget_old(current_time, self.memory_duration); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_perception() {
        let mut sys = PerceptionSystemV2::new();
        sys.register_team(0);
        assert!(sys.can_see(Vec3::ZERO, Vec3::new(1.0,0.0,0.0), Vec3::new(10.0,0.0,0.0)));
        assert!(!sys.can_see(Vec3::ZERO, Vec3::new(1.0,0.0,0.0), Vec3::new(-10.0,0.0,0.0)));
    }
    #[test]
    fn test_threat_assessment() {
        let target = PerceptionTarget { entity: 1, position: Vec3::new(10.0,0.0,0.0), velocity: Vec3::ZERO, last_seen: 0.0, confidence: 1.0, threat_level: 0.5, is_visible: true, team: 1 };
        let threat = ThreatAssessment::compute(&target, Vec3::ZERO, 0);
        assert!(threat.score > 0.0);
    }
}
