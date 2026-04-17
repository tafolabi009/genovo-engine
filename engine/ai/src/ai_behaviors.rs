// engine/ai/src/ai_behaviors.rs
// Common AI behaviors: patrol, guard, investigate, chase, flee, hide, search, wander, follow, escort, ambush.
use std::collections::VecDeque;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStatus { Running, Success, Failure }

#[derive(Debug, Clone)]
pub struct BehaviorContext {
    pub entity: EntityId, pub position: Vec3, pub forward: Vec3, pub target_position: Option<Vec3>,
    pub target_entity: Option<EntityId>, pub dt: f32, pub time: f64, pub alert_level: f32,
    pub health_percent: f32, pub ammo_percent: f32, pub has_line_of_sight: bool,
}

pub trait AiBehavior: std::fmt::Debug {
    fn name(&self) -> &str;
    fn start(&mut self, ctx: &BehaviorContext) {}
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus;
    fn stop(&mut self, ctx: &BehaviorContext) {}
}

#[derive(Debug)]
pub struct PatrolBehavior { pub waypoints: Vec<Vec3>, pub current: usize, pub wait_time: f32, pub wait_timer: f32, pub loop_mode: bool, pub speed: f32, pub reach_dist: f32 }
impl PatrolBehavior {
    pub fn new(waypoints: Vec<Vec3>) -> Self { Self { waypoints, current: 0, wait_time: 2.0, wait_timer: 0.0, loop_mode: true, speed: 3.0, reach_dist: 0.5 } }
}
impl AiBehavior for PatrolBehavior {
    fn name(&self) -> &str { "Patrol" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        if self.waypoints.is_empty() { return BehaviorStatus::Failure; }
        if self.wait_timer > 0.0 { self.wait_timer -= ctx.dt; return BehaviorStatus::Running; }
        let target = self.waypoints[self.current];
        if ctx.position.distance(target) < self.reach_dist {
            self.wait_timer = self.wait_time;
            self.current = if self.loop_mode { (self.current + 1) % self.waypoints.len() } else { (self.current + 1).min(self.waypoints.len() - 1) };
            if !self.loop_mode && self.current >= self.waypoints.len() - 1 { return BehaviorStatus::Success; }
        }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct GuardBehavior { pub guard_pos: Vec3, pub guard_radius: f32, pub alert_radius: f32, pub return_speed: f32 }
impl AiBehavior for GuardBehavior {
    fn name(&self) -> &str { "Guard" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        let dist = ctx.position.distance(self.guard_pos);
        if dist > self.guard_radius { return BehaviorStatus::Running; } // need to return
        if let Some(target) = ctx.target_position {
            if target.distance(self.guard_pos) < self.alert_radius { return BehaviorStatus::Failure; } // alert!
        }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct ChaseBehavior { pub speed: f32, pub give_up_dist: f32, pub give_up_time: f32, pub chase_timer: f32, pub catch_dist: f32 }
impl ChaseBehavior { pub fn new(speed: f32) -> Self { Self { speed, give_up_dist: 50.0, give_up_time: 10.0, chase_timer: 0.0, catch_dist: 1.5 } } }
impl AiBehavior for ChaseBehavior {
    fn name(&self) -> &str { "Chase" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        let target = match ctx.target_position { Some(p) => p, None => return BehaviorStatus::Failure };
        let dist = ctx.position.distance(target);
        if dist < self.catch_dist { return BehaviorStatus::Success; }
        if dist > self.give_up_dist { return BehaviorStatus::Failure; }
        self.chase_timer += ctx.dt;
        if self.chase_timer > self.give_up_time { return BehaviorStatus::Failure; }
        BehaviorStatus::Running
    }
    fn start(&mut self, _: &BehaviorContext) { self.chase_timer = 0.0; }
}

#[derive(Debug)]
pub struct FleeBehavior { pub speed: f32, pub safe_dist: f32 }
impl AiBehavior for FleeBehavior {
    fn name(&self) -> &str { "Flee" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        let target = match ctx.target_position { Some(p) => p, None => return BehaviorStatus::Success };
        if ctx.position.distance(target) > self.safe_dist { return BehaviorStatus::Success; }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct WanderBehavior { pub radius: f32, pub center: Vec3, pub change_interval: f32, pub timer: f32, pub current_target: Vec3 }
impl WanderBehavior {
    pub fn new(center: Vec3, radius: f32) -> Self { Self { radius, center, change_interval: 3.0, timer: 0.0, current_target: center } }
}
impl AiBehavior for WanderBehavior {
    fn name(&self) -> &str { "Wander" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        self.timer += ctx.dt;
        if self.timer >= self.change_interval || ctx.position.distance(self.current_target) < 1.0 {
            self.timer = 0.0;
            let angle = (ctx.time as f32 * 2.71828).fract() * std::f32::consts::TAU;
            let r = self.radius * (ctx.time as f32 * 1.618).fract();
            self.current_target = self.center.add(Vec3::new(angle.cos() * r, 0.0, angle.sin() * r));
        }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct InvestigateBehavior { pub target: Vec3, pub investigate_time: f32, pub timer: f32, pub look_around_time: f32 }
impl InvestigateBehavior {
    pub fn new(target: Vec3) -> Self { Self { target, investigate_time: 5.0, timer: 0.0, look_around_time: 3.0 } }
}
impl AiBehavior for InvestigateBehavior {
    fn name(&self) -> &str { "Investigate" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        if ctx.position.distance(self.target) < 1.0 {
            self.timer += ctx.dt;
            if self.timer >= self.investigate_time { return BehaviorStatus::Success; }
        }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct FollowBehavior { pub follow_dist: f32, pub max_dist: f32, pub speed: f32 }
impl AiBehavior for FollowBehavior {
    fn name(&self) -> &str { "Follow" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        let target = match ctx.target_position { Some(p) => p, None => return BehaviorStatus::Failure };
        let dist = ctx.position.distance(target);
        if dist > self.max_dist { return BehaviorStatus::Failure; }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct SearchBehavior { pub search_points: Vec<Vec3>, pub current: usize, pub time_per_point: f32, pub timer: f32 }
impl AiBehavior for SearchBehavior {
    fn name(&self) -> &str { "Search" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        if self.current >= self.search_points.len() { return BehaviorStatus::Success; }
        let target = self.search_points[self.current];
        if ctx.position.distance(target) < 1.5 {
            self.timer += ctx.dt;
            if self.timer >= self.time_per_point { self.current += 1; self.timer = 0.0; }
        }
        BehaviorStatus::Running
    }
}

#[derive(Debug)]
pub struct AmbushBehavior { pub ambush_pos: Vec3, pub trigger_dist: f32, pub is_waiting: bool }
impl AiBehavior for AmbushBehavior {
    fn name(&self) -> &str { "Ambush" }
    fn tick(&mut self, ctx: &BehaviorContext) -> BehaviorStatus {
        if !self.is_waiting {
            if ctx.position.distance(self.ambush_pos) < 1.0 { self.is_waiting = true; }
            return BehaviorStatus::Running;
        }
        if let Some(target) = ctx.target_position {
            if target.distance(self.ambush_pos) < self.trigger_dist { return BehaviorStatus::Success; }
        }
        BehaviorStatus::Running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx() -> BehaviorContext { BehaviorContext { entity: 0, position: Vec3::ZERO, forward: Vec3::new(1.0,0.0,0.0), target_position: None, target_entity: None, dt: 0.016, time: 0.0, alert_level: 0.0, health_percent: 1.0, ammo_percent: 1.0, has_line_of_sight: false } }
    #[test]
    fn test_patrol() {
        let mut b = PatrolBehavior::new(vec![Vec3::new(5.0,0.0,0.0), Vec3::new(0.0,0.0,5.0)]);
        assert_eq!(b.tick(&ctx()), BehaviorStatus::Running);
    }
    #[test]
    fn test_wander() {
        let mut b = WanderBehavior::new(Vec3::ZERO, 10.0);
        assert_eq!(b.tick(&ctx()), BehaviorStatus::Running);
    }
    #[test]
    fn test_chase_no_target() {
        let mut b = ChaseBehavior::new(5.0);
        assert_eq!(b.tick(&ctx()), BehaviorStatus::Failure);
    }
}
