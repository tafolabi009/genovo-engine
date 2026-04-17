// engine/physics/src/body_manager.rs
//
// Body lifecycle management for rigid body physics. Provides:
//   - Create/destroy bodies with generational handles
//   - Body pool with free-list reuse
//   - Activation/deactivation (sleeping)
//   - Body queries and iteration
//   - Bulk operations and body groups

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const GRAVITY: Self = Self { x: 0.0, y: -9.81, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn length_sq(self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z }

    #[inline]
    pub fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline]
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z) }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z) }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self::new(self.x*s, self.y*s, self.z*s) }
}
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, r: Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self { min: center - half, max: center + half }
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

// ---------------------------------------------------------------------------
// Body types and handle
// ---------------------------------------------------------------------------

/// Type of physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyType {
    /// Fully simulated, affected by forces and collisions.
    Dynamic,
    /// Infinite mass, zero velocity. Never moves from physics.
    Static,
    /// Moved by user code, has infinite mass but finite velocity for collision.
    Kinematic,
}

/// A generational handle to a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle {
    pub index: u32,
    pub generation: u32,
}

impl BodyHandle {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };

    pub fn is_valid(&self) -> bool { self.index != u32::MAX }
}

/// Descriptor for creating a new body.
#[derive(Debug, Clone)]
pub struct BodyDescriptor {
    pub body_type: BodyType,
    pub position: Vec3,
    pub rotation: [f32; 4], // quaternion (x,y,z,w)
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inertia: Vec3, // diagonal inertia tensor
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub user_data: u64,
    pub name: String,
}

impl Default for BodyDescriptor {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            position: Vec3::ZERO,
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            inertia: Vec3::new(1.0, 1.0, 1.0),
            linear_damping: 0.01,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            can_sleep: true,
            collision_group: 1,
            collision_mask: 0xFFFF_FFFF,
            user_data: 0,
            name: String::new(),
        }
    }
}

/// The activation/sleep state of a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationState {
    Active,
    WantsSleep,
    Sleeping,
}

// ---------------------------------------------------------------------------
// Body data
// ---------------------------------------------------------------------------

/// The internal representation of a physics body.
#[derive(Debug, Clone)]
pub struct Body {
    pub handle: BodyHandle,
    pub body_type: BodyType,
    pub position: Vec3,
    pub rotation: [f32; 4],
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub force: Vec3,
    pub torque: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: Vec3,
    pub inv_inertia: Vec3,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
    pub activation: ActivationState,
    pub sleep_timer: f32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub user_data: u64,
    pub name: String,
    pub aabb: AABB,
    pub island_id: i32,
    alive: bool,
}

impl Body {
    fn from_descriptor(desc: &BodyDescriptor, handle: BodyHandle) -> Self {
        let (inv_mass, inv_inertia) = match desc.body_type {
            BodyType::Dynamic => {
                let im = if desc.mass > 0.0 { 1.0 / desc.mass } else { 0.0 };
                let ii = Vec3::new(
                    if desc.inertia.x > 0.0 { 1.0 / desc.inertia.x } else { 0.0 },
                    if desc.inertia.y > 0.0 { 1.0 / desc.inertia.y } else { 0.0 },
                    if desc.inertia.z > 0.0 { 1.0 / desc.inertia.z } else { 0.0 },
                );
                (im, ii)
            }
            _ => (0.0, Vec3::ZERO),
        };

        Self {
            handle,
            body_type: desc.body_type,
            position: desc.position,
            rotation: desc.rotation,
            linear_velocity: desc.linear_velocity,
            angular_velocity: desc.angular_velocity,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            mass: desc.mass,
            inv_mass,
            inertia: desc.inertia,
            inv_inertia,
            linear_damping: desc.linear_damping,
            angular_damping: desc.angular_damping,
            gravity_scale: desc.gravity_scale,
            can_sleep: desc.can_sleep,
            activation: ActivationState::Active,
            sleep_timer: 0.0,
            collision_group: desc.collision_group,
            collision_mask: desc.collision_mask,
            user_data: desc.user_data,
            name: desc.name.clone(),
            aabb: AABB::from_center_half(desc.position, Vec3::new(0.5, 0.5, 0.5)),
            island_id: -1,
            alive: true,
        }
    }

    pub fn is_dynamic(&self) -> bool { self.body_type == BodyType::Dynamic }
    pub fn is_static(&self) -> bool { self.body_type == BodyType::Static }
    pub fn is_kinematic(&self) -> bool { self.body_type == BodyType::Kinematic }
    pub fn is_sleeping(&self) -> bool { self.activation == ActivationState::Sleeping }
    pub fn is_active(&self) -> bool { self.activation == ActivationState::Active }

    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.linear_velocity.length_sq()
            + 0.5 * (self.inertia.x * self.angular_velocity.x * self.angular_velocity.x
                + self.inertia.y * self.angular_velocity.y * self.angular_velocity.y
                + self.inertia.z * self.angular_velocity.z * self.angular_velocity.z)
    }

    pub fn apply_force(&mut self, force: Vec3) {
        self.force += force;
        self.wake();
    }

    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque += torque;
        self.wake();
    }

    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.body_type != BodyType::Dynamic { return; }
        self.linear_velocity += impulse * self.inv_mass;
        self.wake();
    }

    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        self.force += force;
        let r = point - self.position;
        self.torque += Vec3::new(
            r.y * force.z - r.z * force.y,
            r.z * force.x - r.x * force.z,
            r.x * force.y - r.y * force.x,
        );
        self.wake();
    }

    pub fn wake(&mut self) {
        if self.activation == ActivationState::Sleeping {
            self.activation = ActivationState::Active;
            self.sleep_timer = 0.0;
        }
    }

    pub fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Body manager configuration
// ---------------------------------------------------------------------------

/// Configuration for the body manager.
#[derive(Debug, Clone)]
pub struct BodyManagerConfig {
    /// Velocity threshold below which a body starts the sleep timer.
    pub sleep_velocity_threshold: f32,
    /// Angular velocity threshold for sleeping.
    pub sleep_angular_threshold: f32,
    /// Time (seconds) a body must be below threshold before sleeping.
    pub sleep_time_threshold: f32,
    /// Maximum number of bodies.
    pub max_bodies: usize,
    /// Initial pool capacity.
    pub initial_capacity: usize,
}

impl Default for BodyManagerConfig {
    fn default() -> Self {
        Self {
            sleep_velocity_threshold: 0.05,
            sleep_angular_threshold: 0.05,
            sleep_time_threshold: 0.5,
            max_bodies: 65536,
            initial_capacity: 1024,
        }
    }
}

/// Statistics for the body manager.
#[derive(Debug, Clone, Default)]
pub struct BodyManagerStats {
    pub total_bodies: u32,
    pub dynamic_bodies: u32,
    pub static_bodies: u32,
    pub kinematic_bodies: u32,
    pub active_bodies: u32,
    pub sleeping_bodies: u32,
    pub pool_capacity: u32,
    pub free_slots: u32,
}

// ---------------------------------------------------------------------------
// Body manager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of all physics bodies.
pub struct BodyManager {
    bodies: Vec<Body>,
    free_indices: Vec<u32>,
    generations: Vec<u32>,
    name_to_handle: HashMap<String, BodyHandle>,
    config: BodyManagerConfig,
    stats: BodyManagerStats,
    body_count: u32,
}

impl BodyManager {
    pub fn new() -> Self {
        Self::with_config(BodyManagerConfig::default())
    }

    pub fn with_config(config: BodyManagerConfig) -> Self {
        let cap = config.initial_capacity;
        Self {
            bodies: Vec::with_capacity(cap),
            free_indices: Vec::new(),
            generations: Vec::with_capacity(cap),
            name_to_handle: HashMap::new(),
            config,
            stats: BodyManagerStats::default(),
            body_count: 0,
        }
    }

    pub fn config(&self) -> &BodyManagerConfig { &self.config }
    pub fn stats(&self) -> &BodyManagerStats { &self.stats }

    /// Create a new body from a descriptor.
    pub fn create_body(&mut self, desc: &BodyDescriptor) -> Result<BodyHandle, BodyError> {
        if self.body_count as usize >= self.config.max_bodies {
            return Err(BodyError::MaxBodiesReached);
        }

        let (index, generation) = if let Some(free_idx) = self.free_indices.pop() {
            let gen = self.generations[free_idx as usize] + 1;
            self.generations[free_idx as usize] = gen;
            (free_idx, gen)
        } else {
            let idx = self.bodies.len() as u32;
            self.generations.push(0);
            self.bodies.push(Body::from_descriptor(desc, BodyHandle { index: idx, generation: 0 }));
            return self.finish_create(idx, 0, desc);
        };

        let handle = BodyHandle { index, generation };
        let body = Body::from_descriptor(desc, handle);
        self.bodies[index as usize] = body;

        self.finish_create(index, generation, desc)
    }

    fn finish_create(&mut self, index: u32, generation: u32, desc: &BodyDescriptor) -> Result<BodyHandle, BodyError> {
        let handle = BodyHandle { index, generation };
        self.bodies[index as usize].handle = handle;

        if !desc.name.is_empty() {
            self.name_to_handle.insert(desc.name.clone(), handle);
        }

        self.body_count += 1;
        self.update_stats();
        Ok(handle)
    }

    /// Destroy a body by handle.
    pub fn destroy_body(&mut self, handle: BodyHandle) -> Result<(), BodyError> {
        let body = self.get_body_mut(handle)?;
        let name = body.name.clone();
        body.alive = false;
        body.activation = ActivationState::Sleeping;
        body.linear_velocity = Vec3::ZERO;
        body.angular_velocity = Vec3::ZERO;

        if !name.is_empty() {
            self.name_to_handle.remove(&name);
        }

        self.free_indices.push(handle.index);
        self.body_count -= 1;
        self.update_stats();
        Ok(())
    }

    /// Get a reference to a body.
    pub fn get_body(&self, handle: BodyHandle) -> Result<&Body, BodyError> {
        let idx = handle.index as usize;
        if idx >= self.bodies.len() {
            return Err(BodyError::InvalidHandle);
        }
        let body = &self.bodies[idx];
        if !body.alive || self.generations[idx] != handle.generation {
            return Err(BodyError::InvalidHandle);
        }
        Ok(body)
    }

    /// Get a mutable reference to a body.
    pub fn get_body_mut(&mut self, handle: BodyHandle) -> Result<&mut Body, BodyError> {
        let idx = handle.index as usize;
        if idx >= self.bodies.len() {
            return Err(BodyError::InvalidHandle);
        }
        if !self.bodies[idx].alive || self.generations[idx] != handle.generation {
            return Err(BodyError::InvalidHandle);
        }
        Ok(&mut self.bodies[idx])
    }

    /// Find a body by name.
    pub fn find_by_name(&self, name: &str) -> Option<BodyHandle> {
        self.name_to_handle.get(name).copied()
    }

    /// Iterate all alive bodies.
    pub fn iter_bodies(&self) -> impl Iterator<Item = &Body> {
        self.bodies.iter().filter(|b| b.alive)
    }

    /// Iterate all alive bodies (mutable).
    pub fn iter_bodies_mut(&mut self) -> impl Iterator<Item = &mut Body> {
        self.bodies.iter_mut().filter(|b| b.alive)
    }

    /// Iterate only active (non-sleeping) dynamic bodies.
    pub fn iter_active_dynamic(&self) -> impl Iterator<Item = &Body> {
        self.bodies.iter().filter(|b| {
            b.alive && b.body_type == BodyType::Dynamic && b.activation == ActivationState::Active
        })
    }

    /// Get all body handles.
    pub fn all_handles(&self) -> Vec<BodyHandle> {
        self.bodies.iter()
            .filter(|b| b.alive)
            .map(|b| b.handle)
            .collect()
    }

    /// Get bodies within an AABB.
    pub fn query_aabb(&self, aabb: &AABB) -> Vec<BodyHandle> {
        self.bodies.iter()
            .filter(|b| b.alive && b.aabb.intersects(aabb))
            .map(|b| b.handle)
            .collect()
    }

    /// Get bodies within a sphere.
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> Vec<BodyHandle> {
        let r2 = radius * radius;
        self.bodies.iter()
            .filter(|b| b.alive && (b.position - center).length_sq() <= r2)
            .map(|b| b.handle)
            .collect()
    }

    /// Get the closest body to a point.
    pub fn closest_body(&self, point: Vec3) -> Option<BodyHandle> {
        let mut best: Option<(BodyHandle, f32)> = None;
        for body in self.bodies.iter().filter(|b| b.alive) {
            let dist = (body.position - point).length_sq();
            if best.map_or(true, |(_, d)| dist < d) {
                best = Some((body.handle, dist));
            }
        }
        best.map(|(h, _)| h)
    }

    /// Wake a body.
    pub fn wake_body(&mut self, handle: BodyHandle) -> Result<(), BodyError> {
        let body = self.get_body_mut(handle)?;
        body.wake();
        Ok(())
    }

    /// Put a body to sleep.
    pub fn sleep_body(&mut self, handle: BodyHandle) -> Result<(), BodyError> {
        let body = self.get_body_mut(handle)?;
        if body.can_sleep {
            body.activation = ActivationState::Sleeping;
            body.linear_velocity = Vec3::ZERO;
            body.angular_velocity = Vec3::ZERO;
        }
        Ok(())
    }

    /// Update activation states based on velocity thresholds.
    pub fn update_activation(&mut self, dt: f32) {
        let lin_threshold = self.config.sleep_velocity_threshold;
        let ang_threshold = self.config.sleep_angular_threshold;
        let time_threshold = self.config.sleep_time_threshold;

        for body in self.bodies.iter_mut().filter(|b| b.alive && b.body_type == BodyType::Dynamic) {
            if !body.can_sleep {
                body.activation = ActivationState::Active;
                continue;
            }

            let lin_speed = body.linear_velocity.length();
            let ang_speed = body.angular_velocity.length();

            if lin_speed < lin_threshold && ang_speed < ang_threshold {
                body.sleep_timer += dt;
                if body.sleep_timer >= time_threshold {
                    body.activation = ActivationState::Sleeping;
                    body.linear_velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                } else {
                    body.activation = ActivationState::WantsSleep;
                }
            } else {
                body.sleep_timer = 0.0;
                body.activation = ActivationState::Active;
            }
        }

        self.update_stats();
    }

    /// Integrate positions for all active dynamic bodies.
    pub fn integrate(&mut self, dt: f32, gravity: Vec3) {
        for body in self.bodies.iter_mut() {
            if !body.alive || body.body_type != BodyType::Dynamic {
                continue;
            }
            if body.activation == ActivationState::Sleeping {
                continue;
            }

            // Apply gravity.
            let g = gravity * body.gravity_scale;
            body.linear_velocity += (body.force * body.inv_mass + g) * dt;

            // Apply angular acceleration.
            body.angular_velocity += Vec3::new(
                body.torque.x * body.inv_inertia.x,
                body.torque.y * body.inv_inertia.y,
                body.torque.z * body.inv_inertia.z,
            ) * dt;

            // Apply damping.
            body.linear_velocity = body.linear_velocity * (1.0 - body.linear_damping * dt).max(0.0);
            body.angular_velocity = body.angular_velocity * (1.0 - body.angular_damping * dt).max(0.0);

            // Integrate position.
            body.position += body.linear_velocity * dt;

            // Simple quaternion integration for rotation.
            let w = body.angular_velocity;
            let q = body.rotation;
            let dq = [
                0.5 * dt * (w.x * q[3] + w.y * q[2] - w.z * q[1]),
                0.5 * dt * (-w.x * q[2] + w.y * q[3] + w.z * q[0]),
                0.5 * dt * (w.x * q[1] - w.y * q[0] + w.z * q[3]),
                0.5 * dt * (-w.x * q[0] - w.y * q[1] - w.z * q[2]),
            ];
            body.rotation[0] += dq[0];
            body.rotation[1] += dq[1];
            body.rotation[2] += dq[2];
            body.rotation[3] += dq[3];

            // Normalize quaternion.
            let len = (body.rotation[0]*body.rotation[0] + body.rotation[1]*body.rotation[1]
                + body.rotation[2]*body.rotation[2] + body.rotation[3]*body.rotation[3]).sqrt();
            if len > 1e-8 {
                let inv = 1.0 / len;
                body.rotation[0] *= inv;
                body.rotation[1] *= inv;
                body.rotation[2] *= inv;
                body.rotation[3] *= inv;
            }

            // Clear forces.
            body.clear_forces();

            // Update AABB (rough estimate).
            body.aabb = AABB::from_center_half(body.position, Vec3::new(0.5, 0.5, 0.5));
        }
    }

    /// Set body type (e.g. dynamic -> kinematic).
    pub fn set_body_type(&mut self, handle: BodyHandle, body_type: BodyType) -> Result<(), BodyError> {
        let body = self.get_body_mut(handle)?;
        body.body_type = body_type;
        match body_type {
            BodyType::Static | BodyType::Kinematic => {
                body.inv_mass = 0.0;
                body.inv_inertia = Vec3::ZERO;
                body.linear_velocity = Vec3::ZERO;
                body.angular_velocity = Vec3::ZERO;
            }
            BodyType::Dynamic => {
                if body.mass > 0.0 { body.inv_mass = 1.0 / body.mass; }
                body.inv_inertia = Vec3::new(
                    if body.inertia.x > 0.0 { 1.0 / body.inertia.x } else { 0.0 },
                    if body.inertia.y > 0.0 { 1.0 / body.inertia.y } else { 0.0 },
                    if body.inertia.z > 0.0 { 1.0 / body.inertia.z } else { 0.0 },
                );
            }
        }
        Ok(())
    }

    /// Total number of alive bodies.
    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    fn update_stats(&mut self) {
        let mut stats = BodyManagerStats::default();
        for body in &self.bodies {
            if !body.alive { continue; }
            stats.total_bodies += 1;
            match body.body_type {
                BodyType::Dynamic => stats.dynamic_bodies += 1,
                BodyType::Static => stats.static_bodies += 1,
                BodyType::Kinematic => stats.kinematic_bodies += 1,
            }
            match body.activation {
                ActivationState::Active | ActivationState::WantsSleep => stats.active_bodies += 1,
                ActivationState::Sleeping => stats.sleeping_bodies += 1,
            }
        }
        stats.pool_capacity = self.bodies.len() as u32;
        stats.free_slots = self.free_indices.len() as u32;
        self.stats = stats;
    }
}

impl Default for BodyManager {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum BodyError {
    InvalidHandle,
    MaxBodiesReached,
}

impl std::fmt::Display for BodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHandle => write!(f, "Invalid body handle"),
            Self::MaxBodiesReached => write!(f, "Maximum number of bodies reached"),
        }
    }
}

impl std::error::Error for BodyError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let mut mgr = BodyManager::new();
        let h = mgr.create_body(&BodyDescriptor::default()).unwrap();
        assert_eq!(mgr.body_count(), 1);
        assert!(mgr.get_body(h).is_ok());
        mgr.destroy_body(h).unwrap();
        assert_eq!(mgr.body_count(), 0);
        assert!(mgr.get_body(h).is_err());
    }

    #[test]
    fn test_generational_handle() {
        let mut mgr = BodyManager::new();
        let h1 = mgr.create_body(&BodyDescriptor::default()).unwrap();
        mgr.destroy_body(h1).unwrap();
        let h2 = mgr.create_body(&BodyDescriptor::default()).unwrap();
        assert_eq!(h1.index, h2.index);
        assert_ne!(h1.generation, h2.generation);
        assert!(mgr.get_body(h1).is_err());
        assert!(mgr.get_body(h2).is_ok());
    }

    #[test]
    fn test_find_by_name() {
        let mut mgr = BodyManager::new();
        let desc = BodyDescriptor { name: "player".into(), ..Default::default() };
        let h = mgr.create_body(&desc).unwrap();
        assert_eq!(mgr.find_by_name("player"), Some(h));
    }

    #[test]
    fn test_integrate() {
        let mut mgr = BodyManager::new();
        let mut desc = BodyDescriptor::default();
        desc.position = Vec3::new(0.0, 10.0, 0.0);
        let h = mgr.create_body(&desc).unwrap();
        mgr.integrate(1.0 / 60.0, Vec3::GRAVITY);
        let body = mgr.get_body(h).unwrap();
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn test_activation() {
        let mut mgr = BodyManager::new();
        let h = mgr.create_body(&BodyDescriptor::default()).unwrap();
        // Body starts active with zero velocity, should go to sleep.
        for _ in 0..60 {
            mgr.update_activation(1.0 / 60.0);
        }
        let body = mgr.get_body(h).unwrap();
        assert!(body.is_sleeping());
    }

    #[test]
    fn test_query_sphere() {
        let mut mgr = BodyManager::new();
        let d1 = BodyDescriptor { position: Vec3::new(0.0, 0.0, 0.0), ..Default::default() };
        let d2 = BodyDescriptor { position: Vec3::new(100.0, 0.0, 0.0), ..Default::default() };
        let h1 = mgr.create_body(&d1).unwrap();
        let _h2 = mgr.create_body(&d2).unwrap();
        let results = mgr.query_sphere(Vec3::ZERO, 5.0);
        assert!(results.contains(&h1));
        assert_eq!(results.len(), 1);
    }
}
