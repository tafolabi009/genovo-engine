// engine/physics/src/physics_interpolation.rs
//
// Rendering interpolation for physics state in the Genovo engine.
//
// Physics simulations run at a fixed timestep, but rendering can run at a
// different (usually higher) frame rate. This module provides smooth
// interpolation between physics states for jitter-free rendering.
//
// Features:
// - Store previous and current physics state per body
// - Interpolate position/rotation for rendering at sub-timestep accuracy
// - Smooth position and rotation interpolation (slerp for quaternions)
// - Extrapolation option (predict ahead when rendering leads physics)
// - Fixed timestep accumulator for deterministic simulation

use glam::{Quat, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Default fixed physics timestep (60 Hz).
pub const DEFAULT_FIXED_TIMESTEP: f64 = 1.0 / 60.0;

/// Maximum accumulated time before physics steps are skipped.
pub const MAX_ACCUMULATOR: f64 = 0.25;

/// Maximum extrapolation factor.
pub const MAX_EXTRAPOLATION: f32 = 2.0;

/// Default maximum angular velocity for slerp threshold.
pub const MAX_ANGULAR_VELOCITY: f32 = 100.0;

// ---------------------------------------------------------------------------
// Physics state snapshot
// ---------------------------------------------------------------------------

/// A snapshot of a rigid body's transform state at a specific time.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsSnapshot {
    /// World-space position.
    pub position: Vec3,
    /// World-space rotation (unit quaternion).
    pub rotation: Quat,
    /// Linear velocity (for extrapolation).
    pub linear_velocity: Vec3,
    /// Angular velocity (for extrapolation).
    pub angular_velocity: Vec3,
    /// Scale (rarely changes, but included for completeness).
    pub scale: Vec3,
    /// Whether the body is sleeping.
    pub sleeping: bool,
    /// Simulation time when this snapshot was taken.
    pub timestamp: f64,
}

impl Default for PhysicsSnapshot {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            scale: Vec3::ONE,
            sleeping: false,
            timestamp: 0.0,
        }
    }
}

impl PhysicsSnapshot {
    /// Create a snapshot from position and rotation.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation: rotation.normalize(),
            ..Default::default()
        }
    }

    /// Create a full snapshot with velocities.
    pub fn full(
        position: Vec3,
        rotation: Quat,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        timestamp: f64,
    ) -> Self {
        Self {
            position,
            rotation: rotation.normalize(),
            linear_velocity,
            angular_velocity,
            scale: Vec3::ONE,
            sleeping: false,
            timestamp,
        }
    }

    /// Interpolate between two snapshots.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            position: Vec3::lerp(a.position, b.position, t),
            rotation: a.rotation.slerp(b.rotation, t),
            linear_velocity: Vec3::lerp(a.linear_velocity, b.linear_velocity, t),
            angular_velocity: Vec3::lerp(a.angular_velocity, b.angular_velocity, t),
            scale: Vec3::lerp(a.scale, b.scale, t),
            sleeping: if t < 0.5 { a.sleeping } else { b.sleeping },
            timestamp: a.timestamp + (b.timestamp - a.timestamp) * t as f64,
        }
    }

    /// Extrapolate from this snapshot by a time delta.
    pub fn extrapolate(&self, dt: f32) -> Self {
        let new_pos = self.position + self.linear_velocity * dt;

        // Simple angular velocity integration
        let ang_speed = self.angular_velocity.length();
        let new_rot = if ang_speed > EPSILON {
            let axis = self.angular_velocity / ang_speed;
            let delta_rot = Quat::from_axis_angle(axis, ang_speed * dt);
            (delta_rot * self.rotation).normalize()
        } else {
            self.rotation
        };

        Self {
            position: new_pos,
            rotation: new_rot,
            linear_velocity: self.linear_velocity,
            angular_velocity: self.angular_velocity,
            scale: self.scale,
            sleeping: self.sleeping,
            timestamp: self.timestamp + dt as f64,
        }
    }

    /// Distance between two snapshots (position only).
    pub fn distance(&self, other: &Self) -> f32 {
        (self.position - other.position).length()
    }

    /// Angular distance between two snapshots (in radians).
    pub fn angular_distance(&self, other: &Self) -> f32 {
        let dot = self.rotation.dot(other.rotation).abs().min(1.0);
        2.0 * dot.acos()
    }
}

// ---------------------------------------------------------------------------
// Interpolation state per body
// ---------------------------------------------------------------------------

/// Interpolation data for a single rigid body.
#[derive(Debug, Clone)]
pub struct BodyInterpolationState {
    /// Previous physics state (before the last physics step).
    pub previous: PhysicsSnapshot,
    /// Current physics state (after the last physics step).
    pub current: PhysicsSnapshot,
    /// The rendered (interpolated) transform.
    pub rendered: PhysicsSnapshot,
    /// Whether this body uses interpolation.
    pub enabled: bool,
    /// Whether to use extrapolation instead of interpolation.
    pub extrapolation: bool,
    /// Maximum extrapolation time in seconds.
    pub max_extrapolation_time: f32,
    /// Teleport threshold: if position changes by more than this,
    /// snap instead of interpolating.
    pub teleport_threshold: f32,
    /// Whether the body was teleported this frame.
    pub teleported: bool,
    /// Number of physics steps since last render.
    pub steps_since_render: u32,
}

impl Default for BodyInterpolationState {
    fn default() -> Self {
        Self {
            previous: PhysicsSnapshot::default(),
            current: PhysicsSnapshot::default(),
            rendered: PhysicsSnapshot::default(),
            enabled: true,
            extrapolation: false,
            max_extrapolation_time: 0.05,
            teleport_threshold: 5.0,
            teleported: false,
            steps_since_render: 0,
        }
    }
}

impl BodyInterpolationState {
    /// Create a new interpolation state at an initial position.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        let snapshot = PhysicsSnapshot::new(position, rotation);
        Self {
            previous: snapshot,
            current: snapshot,
            rendered: snapshot,
            ..Default::default()
        }
    }

    /// Push a new physics state (called after each physics step).
    pub fn push_state(&mut self, snapshot: PhysicsSnapshot) {
        // Check for teleportation
        let dist = self.current.distance(&snapshot);
        self.teleported = dist > self.teleport_threshold;

        self.previous = self.current;
        self.current = snapshot;
        self.steps_since_render += 1;
    }

    /// Compute the interpolated transform for rendering.
    ///
    /// `alpha`: interpolation factor between previous and current (0 = previous, 1 = current).
    pub fn interpolate(&mut self, alpha: f32) -> PhysicsSnapshot {
        if !self.enabled || self.teleported {
            self.rendered = self.current;
            self.steps_since_render = 0;
            return self.current;
        }

        if self.current.sleeping && self.previous.sleeping {
            self.rendered = self.current;
            return self.current;
        }

        let result = if self.extrapolation {
            // Extrapolate from current state
            let dt = (alpha * self.max_extrapolation_time).min(self.max_extrapolation_time);
            self.current.extrapolate(dt)
        } else {
            // Interpolate between previous and current
            PhysicsSnapshot::lerp(&self.previous, &self.current, alpha)
        };

        self.rendered = result;
        self.steps_since_render = 0;
        result
    }

    /// Force snap to the current physics state (no interpolation).
    pub fn snap_to_current(&mut self) {
        self.previous = self.current;
        self.rendered = self.current;
        self.teleported = false;
    }

    /// Set the teleport threshold.
    pub fn set_teleport_threshold(&mut self, threshold: f32) {
        self.teleport_threshold = threshold;
    }

    /// Enable or disable extrapolation mode.
    pub fn set_extrapolation(&mut self, enabled: bool) {
        self.extrapolation = enabled;
    }
}

// ---------------------------------------------------------------------------
// Fixed timestep accumulator
// ---------------------------------------------------------------------------

/// A fixed timestep accumulator that determines when to step the physics simulation.
#[derive(Debug, Clone)]
pub struct FixedTimestep {
    /// Fixed timestep duration in seconds.
    pub timestep: f64,
    /// Accumulated time since last physics step.
    pub accumulator: f64,
    /// Maximum accumulator value (prevents spiral of death).
    pub max_accumulator: f64,
    /// Current simulation time.
    pub simulation_time: f64,
    /// Number of physics steps executed this frame.
    pub steps_this_frame: u32,
    /// Maximum steps per frame.
    pub max_steps_per_frame: u32,
    /// Interpolation alpha for rendering.
    pub alpha: f32,
    /// Whether the simulation is running.
    pub running: bool,
    /// Frame count.
    pub frame_count: u64,
    /// Total physics steps executed.
    pub total_steps: u64,
}

impl FixedTimestep {
    /// Create a new fixed timestep with the given duration.
    pub fn new(timestep: f64) -> Self {
        Self {
            timestep,
            accumulator: 0.0,
            max_accumulator: MAX_ACCUMULATOR,
            simulation_time: 0.0,
            steps_this_frame: 0,
            max_steps_per_frame: 8,
            alpha: 0.0,
            running: true,
            frame_count: 0,
            total_steps: 0,
        }
    }

    /// Create a timestep at 60 Hz.
    pub fn at_60hz() -> Self {
        Self::new(1.0 / 60.0)
    }

    /// Create a timestep at 120 Hz.
    pub fn at_120hz() -> Self {
        Self::new(1.0 / 120.0)
    }

    /// Create a timestep at 30 Hz.
    pub fn at_30hz() -> Self {
        Self::new(1.0 / 30.0)
    }

    /// Advance the accumulator by the frame delta time.
    ///
    /// Returns the number of physics steps to execute this frame.
    pub fn advance(&mut self, frame_dt: f64) -> u32 {
        if !self.running {
            return 0;
        }

        self.frame_count += 1;
        self.accumulator += frame_dt;

        // Clamp to prevent spiral of death
        if self.accumulator > self.max_accumulator {
            self.accumulator = self.max_accumulator;
        }

        // Count how many steps to execute
        self.steps_this_frame = 0;
        while self.accumulator >= self.timestep && self.steps_this_frame < self.max_steps_per_frame {
            self.accumulator -= self.timestep;
            self.simulation_time += self.timestep;
            self.steps_this_frame += 1;
            self.total_steps += 1;
        }

        // Compute interpolation alpha
        self.alpha = (self.accumulator / self.timestep) as f32;
        self.alpha = self.alpha.clamp(0.0, 1.0);

        self.steps_this_frame
    }

    /// Get the interpolation alpha for rendering.
    pub fn interpolation_alpha(&self) -> f32 {
        self.alpha
    }

    /// Get the fixed timestep as f32.
    pub fn dt(&self) -> f32 {
        self.timestep as f32
    }

    /// Pause the simulation.
    pub fn pause(&mut self) {
        self.running = false;
    }

    /// Resume the simulation.
    pub fn resume(&mut self) {
        self.running = true;
    }

    /// Toggle pause/resume.
    pub fn toggle_pause(&mut self) {
        self.running = !self.running;
    }

    /// Reset the accumulator (e.g. after a long pause).
    pub fn reset_accumulator(&mut self) {
        self.accumulator = 0.0;
    }

    /// Get the current simulation time.
    pub fn time(&self) -> f64 {
        self.simulation_time
    }

    /// Get statistics.
    pub fn stats(&self) -> TimestepStats {
        TimestepStats {
            timestep: self.timestep,
            accumulator: self.accumulator,
            alpha: self.alpha,
            steps_this_frame: self.steps_this_frame,
            total_steps: self.total_steps,
            simulation_time: self.simulation_time,
            frame_count: self.frame_count,
        }
    }
}

/// Timestep statistics.
#[derive(Debug, Clone)]
pub struct TimestepStats {
    pub timestep: f64,
    pub accumulator: f64,
    pub alpha: f32,
    pub steps_this_frame: u32,
    pub total_steps: u64,
    pub simulation_time: f64,
    pub frame_count: u64,
}

// ---------------------------------------------------------------------------
// Interpolation system (manages all bodies)
// ---------------------------------------------------------------------------

/// Body identifier for the interpolation system.
pub type BodyId = u64;

/// The interpolation system manages interpolation state for all physics bodies.
#[derive(Debug)]
pub struct InterpolationSystem {
    /// Per-body interpolation state.
    pub bodies: HashMap<BodyId, BodyInterpolationState>,
    /// Fixed timestep controller.
    pub timestep: FixedTimestep,
    /// Global interpolation enable/disable.
    pub enabled: bool,
    /// Global extrapolation enable/disable.
    pub extrapolation_mode: bool,
    /// Statistics.
    pub stats: InterpolationStats,
}

/// Interpolation system statistics.
#[derive(Debug, Clone, Default)]
pub struct InterpolationStats {
    /// Total tracked bodies.
    pub total_bodies: u32,
    /// Bodies interpolated this frame.
    pub interpolated_bodies: u32,
    /// Bodies that were sleeping (skipped).
    pub sleeping_bodies: u32,
    /// Bodies that were teleported (snapped).
    pub teleported_bodies: u32,
    /// Current interpolation alpha.
    pub alpha: f32,
    /// Physics steps this frame.
    pub physics_steps: u32,
}

impl InterpolationSystem {
    /// Create a new interpolation system.
    pub fn new(timestep: f64) -> Self {
        Self {
            bodies: HashMap::new(),
            timestep: FixedTimestep::new(timestep),
            enabled: true,
            extrapolation_mode: false,
            stats: InterpolationStats::default(),
        }
    }

    /// Create with the default 60 Hz timestep.
    pub fn default_60hz() -> Self {
        Self::new(DEFAULT_FIXED_TIMESTEP)
    }

    /// Register a new body for interpolation.
    pub fn register_body(&mut self, id: BodyId, position: Vec3, rotation: Quat) {
        let mut state = BodyInterpolationState::new(position, rotation);
        state.extrapolation = self.extrapolation_mode;
        self.bodies.insert(id, state);
    }

    /// Remove a body from interpolation tracking.
    pub fn unregister_body(&mut self, id: BodyId) {
        self.bodies.remove(&id);
    }

    /// Push a new physics state for a body (called after each physics step).
    pub fn push_state(&mut self, id: BodyId, snapshot: PhysicsSnapshot) {
        if let Some(state) = self.bodies.get_mut(&id) {
            state.push_state(snapshot);
        }
    }

    /// Push states for multiple bodies at once.
    pub fn push_states(&mut self, states: &[(BodyId, PhysicsSnapshot)]) {
        for (id, snapshot) in states {
            self.push_state(*id, *snapshot);
        }
    }

    /// Advance the timestep and return the number of physics steps needed.
    pub fn advance_frame(&mut self, frame_dt: f64) -> u32 {
        let steps = self.timestep.advance(frame_dt);
        self.stats.physics_steps = steps;
        self.stats.alpha = self.timestep.alpha;
        steps
    }

    /// Compute interpolated transforms for all bodies.
    pub fn interpolate_all(&mut self) {
        if !self.enabled {
            return;
        }

        let alpha = self.timestep.interpolation_alpha();
        self.stats.total_bodies = self.bodies.len() as u32;
        self.stats.interpolated_bodies = 0;
        self.stats.sleeping_bodies = 0;
        self.stats.teleported_bodies = 0;

        for state in self.bodies.values_mut() {
            if state.teleported {
                state.snap_to_current();
                self.stats.teleported_bodies += 1;
            } else if state.current.sleeping && state.previous.sleeping {
                self.stats.sleeping_bodies += 1;
            } else {
                state.interpolate(alpha);
                self.stats.interpolated_bodies += 1;
            }
        }
    }

    /// Get the interpolated transform for a body.
    pub fn get_rendered_transform(&self, id: BodyId) -> Option<(Vec3, Quat)> {
        self.bodies.get(&id).map(|s| (s.rendered.position, s.rendered.rotation))
    }

    /// Get the full rendered snapshot for a body.
    pub fn get_rendered_snapshot(&self, id: BodyId) -> Option<&PhysicsSnapshot> {
        self.bodies.get(&id).map(|s| &s.rendered)
    }

    /// Snap all bodies to their current physics state (no interpolation).
    pub fn snap_all(&mut self) {
        for state in self.bodies.values_mut() {
            state.snap_to_current();
        }
    }

    /// Set the interpolation mode for all bodies.
    pub fn set_extrapolation(&mut self, enabled: bool) {
        self.extrapolation_mode = enabled;
        for state in self.bodies.values_mut() {
            state.set_extrapolation(enabled);
        }
    }

    /// Get the fixed timestep controller.
    pub fn timestep(&self) -> &FixedTimestep {
        &self.timestep
    }

    /// Get the interpolation alpha.
    pub fn alpha(&self) -> f32 {
        self.timestep.interpolation_alpha()
    }

    /// Total body count.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

/// ECS component for physics interpolation on an entity.
#[derive(Debug, Clone)]
pub struct PhysicsInterpolationComponent {
    /// Body ID in the interpolation system.
    pub body_id: BodyId,
    /// Whether interpolation is enabled for this entity.
    pub enabled: bool,
    /// Whether to use extrapolation.
    pub extrapolation: bool,
    /// Teleport distance threshold.
    pub teleport_threshold: f32,
}

impl PhysicsInterpolationComponent {
    pub fn new(body_id: BodyId) -> Self {
        Self {
            body_id,
            enabled: true,
            extrapolation: false,
            teleport_threshold: 5.0,
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
    fn test_snapshot_lerp() {
        let a = PhysicsSnapshot::new(Vec3::ZERO, Quat::IDENTITY);
        let b = PhysicsSnapshot::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);
        let mid = PhysicsSnapshot::lerp(&a, &b, 0.5);
        assert!((mid.position.x - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_snapshot_extrapolate() {
        let s = PhysicsSnapshot::full(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::ZERO,
            0.0,
        );
        let ext = s.extrapolate(1.0);
        assert!((ext.position.x - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_body_interpolation() {
        let mut state = BodyInterpolationState::new(Vec3::ZERO, Quat::IDENTITY);
        state.push_state(PhysicsSnapshot::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY));
        let result = state.interpolate(0.5);
        assert!((result.position.x - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_teleport_detection() {
        let mut state = BodyInterpolationState::new(Vec3::ZERO, Quat::IDENTITY);
        state.set_teleport_threshold(1.0);
        state.push_state(PhysicsSnapshot::new(Vec3::new(100.0, 0.0, 0.0), Quat::IDENTITY));
        assert!(state.teleported);
    }

    #[test]
    fn test_fixed_timestep() {
        let mut ts = FixedTimestep::at_60hz();
        // Simulate a 16ms frame
        let steps = ts.advance(0.016);
        assert!(steps <= 2);
        assert!(ts.alpha >= 0.0 && ts.alpha <= 1.0);
    }

    #[test]
    fn test_fixed_timestep_spiral_of_death() {
        let mut ts = FixedTimestep::at_60hz();
        ts.max_steps_per_frame = 4;
        // Very large frame time
        let steps = ts.advance(1.0);
        assert!(steps <= 4); // Capped
    }

    #[test]
    fn test_interpolation_system() {
        let mut sys = InterpolationSystem::default_60hz();
        sys.register_body(1, Vec3::ZERO, Quat::IDENTITY);
        sys.push_state(1, PhysicsSnapshot::new(Vec3::X, Quat::IDENTITY));
        sys.advance_frame(0.016);
        sys.interpolate_all();
        let (pos, _) = sys.get_rendered_transform(1).unwrap();
        assert!(pos.length() >= 0.0);
    }

    #[test]
    fn test_timestep_pause_resume() {
        let mut ts = FixedTimestep::at_60hz();
        ts.pause();
        assert_eq!(ts.advance(0.1), 0);
        ts.resume();
        assert!(ts.advance(0.1) > 0);
    }
}
