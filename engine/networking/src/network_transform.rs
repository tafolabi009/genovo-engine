// engine/networking/src/network_transform.rs
//
// Network transform: interpolated position/rotation, extrapolation,
// ownership, authority, smooth corrections, and dead reckoning.
// Handles the visual representation of networked entities.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Vec3 / Quat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32, pub y: f32, pub z: f32,
}
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self::new(a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t)
    }
    pub fn distance_sq(self, o: Self) -> f32 {
        let d = Vec3::new(self.x-o.x, self.y-o.y, self.z-o.z);
        d.x*d.x + d.y*d.y + d.z*d.z
    }
    pub fn length(self) -> f32 { (self.x*self.x+self.y*self.y+self.z*self.z).sqrt() }
}
impl std::ops::Add for Vec3 { type Output=Self; fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)}}
impl std::ops::Sub for Vec3 { type Output=Self; fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)}}
impl std::ops::Mul<f32> for Vec3 { type Output=Self; fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)}}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub fn slerp(a: Self, b: Self, t: f32) -> Self {
        let mut dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        let (mut bx, mut by, mut bz, mut bw) = (b.x, b.y, b.z, b.w);
        if dot < 0.0 { dot = -dot; bx = -bx; by = -by; bz = -bz; bw = -bw; }
        if dot > 0.9995 {
            let r = Self { x: a.x+(bx-a.x)*t, y: a.y+(by-a.y)*t, z: a.z+(bz-a.z)*t, w: a.w+(bw-a.w)*t };
            let l = (r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w).sqrt();
            if l > 0.0 { return Self { x: r.x/l, y: r.y/l, z: r.z/l, w: r.w/l }; }
            return a;
        }
        let theta = dot.acos(); let st = theta.sin();
        let wa = ((1.0-t)*theta).sin()/st; let wb = (t*theta).sin()/st;
        Self { x: a.x*wa+bx*wb, y: a.y*wa+by*wb, z: a.z*wa+bz*wb, w: a.w*wa+bw*wb }
    }
    pub fn angle_between(a: Self, b: Self) -> f32 {
        let dot = (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w).abs().min(1.0);
        2.0 * dot.acos()
    }
}

// ---------------------------------------------------------------------------
// Network authority
// ---------------------------------------------------------------------------

/// Who has authority over this entity's transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authority {
    /// Server is authoritative (default for most entities).
    Server,
    /// A specific client owns this entity.
    Client(u32),
    /// Shared authority (e.g., physics objects).
    Shared,
}

/// Ownership state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// This client owns the entity (we are authoritative).
    Owned,
    /// Another client or the server owns the entity.
    Remote,
}

// ---------------------------------------------------------------------------
// Interpolation mode
// ---------------------------------------------------------------------------

/// How to interpolate between network updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Snap to latest position (no smoothing).
    None,
    /// Linear interpolation between snapshots.
    Interpolation,
    /// Extrapolate forward using velocity.
    Extrapolation,
    /// Hermite interpolation using position and velocity.
    Hermite,
}

// ---------------------------------------------------------------------------
// Transform snapshot
// ---------------------------------------------------------------------------

/// A timestamped transform received from the network.
#[derive(Debug, Clone, Copy)]
pub struct TransformSnapshot {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub server_time: f64,
    pub server_tick: u64,
}

// ---------------------------------------------------------------------------
// Correction smoothing
// ---------------------------------------------------------------------------

/// Smoothly corrects a position error over time.
#[derive(Debug, Clone)]
struct CorrectionSmoother {
    correction: Vec3,
    rotation_correction: Quat,
    remaining: f32,
    duration: f32,
}

impl CorrectionSmoother {
    fn new(duration: f32) -> Self {
        Self {
            correction: Vec3::ZERO,
            rotation_correction: Quat::IDENTITY,
            remaining: 0.0,
            duration,
        }
    }

    fn apply_correction(&mut self, position_error: Vec3, rotation_error: Quat) {
        self.correction = position_error;
        self.rotation_correction = rotation_error;
        self.remaining = self.duration;
    }

    fn update(&mut self, dt: f32) -> (Vec3, Quat) {
        if self.remaining <= 0.0 {
            return (Vec3::ZERO, Quat::IDENTITY);
        }
        let t = (dt / self.remaining).min(1.0);
        let pos_correction = self.correction * t;
        let rot_correction = Quat::slerp(Quat::IDENTITY, self.rotation_correction, t);
        self.correction = self.correction - pos_correction;
        self.rotation_correction = Quat::slerp(
            self.rotation_correction, Quat::IDENTITY, t);
        self.remaining -= dt;
        (pos_correction, rot_correction)
    }
}

// ---------------------------------------------------------------------------
// NetworkTransform component
// ---------------------------------------------------------------------------

/// Configuration for a network transform.
#[derive(Debug, Clone)]
pub struct NetworkTransformConfig {
    pub interpolation_mode: InterpolationMode,
    pub max_extrapolation_time: f32,
    pub teleport_distance: f32,
    pub teleport_angle: f32,
    pub correction_duration: f32,
    pub max_snapshots: usize,
    pub send_rate_hz: f32,
    pub position_threshold: f32,
    pub rotation_threshold: f32,
}

impl Default for NetworkTransformConfig {
    fn default() -> Self {
        Self {
            interpolation_mode: InterpolationMode::Interpolation,
            max_extrapolation_time: 0.25,
            teleport_distance: 10.0,
            teleport_angle: std::f32::consts::PI,
            correction_duration: 0.1,
            max_snapshots: 32,
            send_rate_hz: 20.0,
            position_threshold: 0.001,
            rotation_threshold: 0.01,
        }
    }
}

/// A networked transform that handles interpolation, extrapolation,
/// and smooth corrections.
pub struct NetworkTransform {
    config: NetworkTransformConfig,
    snapshots: VecDeque<TransformSnapshot>,
    current_position: Vec3,
    current_rotation: Quat,
    target_position: Vec3,
    target_rotation: Quat,
    velocity: Vec3,
    angular_velocity: Vec3,
    authority: Authority,
    ownership: Ownership,
    smoother: CorrectionSmoother,
    last_received_time: f64,
    last_sent_time: f64,
    extrapolating: bool,
    extrapolation_time: f32,
    entity_id: u64,
    dirty: bool,
}

impl NetworkTransform {
    pub fn new(entity_id: u64, config: NetworkTransformConfig) -> Self {
        let correction_dur = config.correction_duration;
        Self {
            config,
            snapshots: VecDeque::with_capacity(32),
            current_position: Vec3::ZERO,
            current_rotation: Quat::IDENTITY,
            target_position: Vec3::ZERO,
            target_rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            authority: Authority::Server,
            ownership: Ownership::Remote,
            smoother: CorrectionSmoother::new(correction_dur),
            last_received_time: 0.0,
            last_sent_time: 0.0,
            extrapolating: false,
            extrapolation_time: 0.0,
            entity_id,
            dirty: false,
        }
    }

    pub fn entity_id(&self) -> u64 { self.entity_id }
    pub fn position(&self) -> Vec3 { self.current_position }
    pub fn rotation(&self) -> Quat { self.current_rotation }
    pub fn velocity(&self) -> Vec3 { self.velocity }
    pub fn authority(&self) -> Authority { self.authority }
    pub fn ownership(&self) -> Ownership { self.ownership }
    pub fn is_extrapolating(&self) -> bool { self.extrapolating }
    pub fn is_dirty(&self) -> bool { self.dirty }

    pub fn set_authority(&mut self, auth: Authority) { self.authority = auth; }
    pub fn set_ownership(&mut self, own: Ownership) { self.ownership = own; }

    /// Receive a new transform snapshot from the network.
    pub fn receive_snapshot(&mut self, snapshot: TransformSnapshot) {
        self.last_received_time = snapshot.server_time;

        // Check for teleport.
        let dist_sq = self.current_position.distance_sq(snapshot.position);
        let teleport_dist = self.config.teleport_distance;
        if dist_sq > teleport_dist * teleport_dist {
            self.current_position = snapshot.position;
            self.current_rotation = snapshot.rotation;
            self.velocity = snapshot.velocity;
            self.angular_velocity = snapshot.angular_velocity;
            self.snapshots.clear();
            self.snapshots.push_back(snapshot);
            self.extrapolating = false;
            return;
        }

        // Check for rotation teleport.
        let angle = Quat::angle_between(self.current_rotation, snapshot.rotation);
        if angle > self.config.teleport_angle {
            self.current_rotation = snapshot.rotation;
        }

        self.snapshots.push_back(snapshot);
        while self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.pop_front();
        }

        self.target_position = snapshot.position;
        self.target_rotation = snapshot.rotation;
        self.velocity = snapshot.velocity;
        self.angular_velocity = snapshot.angular_velocity;
        self.extrapolating = false;
        self.extrapolation_time = 0.0;
    }

    /// Set the local transform (for owned entities).
    pub fn set_local_transform(&mut self, position: Vec3, rotation: Quat, velocity: Vec3) {
        self.current_position = position;
        self.current_rotation = rotation;
        self.velocity = velocity;

        // Check if we need to send.
        let pos_changed = (position - self.target_position).length() > self.config.position_threshold;
        let rot_changed = Quat::angle_between(rotation, self.target_rotation) > self.config.rotation_threshold;
        self.dirty = pos_changed || rot_changed;

        self.target_position = position;
        self.target_rotation = rotation;
    }

    /// Should we send an update now?
    pub fn should_send(&self, current_time: f64) -> bool {
        if self.ownership != Ownership::Owned { return false; }
        let interval = 1.0 / self.config.send_rate_hz as f64;
        self.dirty && (current_time - self.last_sent_time) >= interval
    }

    /// Mark that we sent an update.
    pub fn mark_sent(&mut self, time: f64) {
        self.last_sent_time = time;
        self.dirty = false;
    }

    /// Build a snapshot for sending.
    pub fn build_snapshot(&self, server_time: f64, server_tick: u64) -> TransformSnapshot {
        TransformSnapshot {
            position: self.current_position,
            rotation: self.current_rotation,
            velocity: self.velocity,
            angular_velocity: self.angular_velocity,
            server_time,
            server_tick,
        }
    }

    /// Update the visual transform. Call every frame.
    pub fn update(&mut self, dt: f32, current_server_time: f64) {
        if self.ownership == Ownership::Owned {
            return; // Local entity, no interpolation needed.
        }

        match self.config.interpolation_mode {
            InterpolationMode::None => {
                self.current_position = self.target_position;
                self.current_rotation = self.target_rotation;
            }
            InterpolationMode::Interpolation => {
                self.update_interpolation(dt, current_server_time);
            }
            InterpolationMode::Extrapolation => {
                self.update_extrapolation(dt, current_server_time);
            }
            InterpolationMode::Hermite => {
                self.update_hermite(dt, current_server_time);
            }
        }

        // Apply correction smoothing.
        let (pos_corr, _rot_corr) = self.smoother.update(dt);
        self.current_position = self.current_position + pos_corr;
    }

    fn update_interpolation(&mut self, dt: f32, _current_time: f64) {
        // Simple exponential interpolation toward target.
        let speed = 15.0;
        let t = (speed * dt).min(1.0);
        self.current_position = Vec3::lerp(self.current_position, self.target_position, t);
        self.current_rotation = Quat::slerp(self.current_rotation, self.target_rotation, t);
    }

    fn update_extrapolation(&mut self, dt: f32, current_time: f64) {
        let time_since_update = (current_time - self.last_received_time) as f32;

        if time_since_update > self.config.max_extrapolation_time {
            // Stop extrapolating after max time.
            self.extrapolating = true;
            return;
        }

        // Extrapolate using velocity.
        let extrapolated_pos = self.target_position + self.velocity * time_since_update;

        // Smoothly move toward extrapolated position.
        let t = (10.0 * dt).min(1.0);
        self.current_position = Vec3::lerp(self.current_position, extrapolated_pos, t);
        self.current_rotation = Quat::slerp(self.current_rotation, self.target_rotation, t);
    }

    fn update_hermite(&mut self, dt: f32, _current_time: f64) {
        if self.snapshots.len() < 2 {
            self.update_interpolation(dt, _current_time);
            return;
        }

        let a = &self.snapshots[self.snapshots.len() - 2];
        let b = &self.snapshots[self.snapshots.len() - 1];

        let total_time = (b.server_time - a.server_time) as f32;
        if total_time <= 0.0 {
            self.update_interpolation(dt, _current_time);
            return;
        }

        // Use Hermite interpolation.
        let speed = 10.0;
        let t = (speed * dt).min(1.0);

        let p0 = a.position;
        let p1 = b.position;
        let v0 = a.velocity * total_time;
        let v1 = b.velocity * total_time;

        // Hermite basis functions.
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0*t3 - 3.0*t2 + 1.0;
        let h10 = t3 - 2.0*t2 + t;
        let h01 = -2.0*t3 + 3.0*t2;
        let h11 = t3 - t2;

        let interp_pos = Vec3::new(
            h00*p0.x + h10*v0.x + h01*p1.x + h11*v1.x,
            h00*p0.y + h10*v0.y + h01*p1.y + h11*v1.y,
            h00*p0.z + h10*v0.z + h01*p1.z + h11*v1.z,
        );

        self.current_position = interp_pos;
        self.current_rotation = Quat::slerp(self.current_rotation, self.target_rotation, t);
    }

    /// Apply a server correction to this entity's position.
    pub fn apply_correction(&mut self, correct_position: Vec3, correct_rotation: Quat) {
        let error = correct_position - self.current_position;
        self.smoother.apply_correction(error, correct_rotation);
        self.target_position = correct_position;
        self.target_rotation = correct_rotation;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation() {
        let mut nt = NetworkTransform::new(1, NetworkTransformConfig::default());
        nt.set_ownership(Ownership::Remote);

        nt.receive_snapshot(TransformSnapshot {
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            server_time: 1.0,
            server_tick: 1,
        });

        for _ in 0..60 {
            nt.update(1.0 / 60.0, 1.0);
        }

        assert!((nt.position().x - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_teleport() {
        let mut nt = NetworkTransform::new(1, NetworkTransformConfig::default());
        nt.set_ownership(Ownership::Remote);

        nt.receive_snapshot(TransformSnapshot {
            position: Vec3::new(1000.0, 0.0, 0.0), // far away
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            server_time: 1.0,
            server_tick: 1,
        });

        // Should teleport immediately.
        assert!((nt.position().x - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_owned_entity() {
        let mut nt = NetworkTransform::new(1, NetworkTransformConfig::default());
        nt.set_ownership(Ownership::Owned);

        nt.set_local_transform(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        assert!(nt.is_dirty());
        assert!(nt.should_send(1.0));

        nt.mark_sent(1.0);
        assert!(!nt.is_dirty());
    }

    #[test]
    fn test_correction() {
        let mut nt = NetworkTransform::new(1, NetworkTransformConfig::default());
        nt.set_ownership(Ownership::Remote);
        nt.current_position = Vec3::new(5.0, 0.0, 0.0);

        nt.apply_correction(Vec3::new(6.0, 0.0, 0.0), Quat::IDENTITY);

        for _ in 0..30 {
            nt.update(1.0 / 60.0, 1.0);
        }
        // Should move toward corrected position.
        assert!(nt.position().x > 5.0);
    }
}
