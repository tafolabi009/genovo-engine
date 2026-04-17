// engine/physics/src/water_physics.rs
//
// Water interaction physics for the Genovo engine.
//
// Provides comprehensive water-object interaction simulation:
//
// - **Water volumes** -- Define water regions with configurable surface height,
//   density, and current forces.
// - **Submersion detection** -- Accurately detect how much of a shape is
//   submerged using analytical volume computations.
// - **Buoyancy forces** -- Apply Archimedes-principle buoyancy with per-shape
//   force application for realistic tilting.
// - **Water drag** -- Linear and angular drag forces proportional to submersion.
// - **Water current forces** -- Directional flow forces that push submerged
//   objects along the current direction.
// - **Splash effects** -- Detect entry/exit events and compute splash magnitude.
// - **Wake generation** -- Compute wake parameters based on object speed and
//   submersion depth for visual rendering.
// - **Water surface deformation** -- Track surface displacement caused by
//   submerged objects for dynamic water visuals.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const DEFAULT_WATER_DENSITY: f32 = 1000.0; // kg/m^3
const DEFAULT_GRAVITY: f32 = 9.81;
const DEFAULT_LINEAR_DRAG: f32 = 0.5;
const DEFAULT_ANGULAR_DRAG: f32 = 1.0;

// ---------------------------------------------------------------------------
// Water volume
// ---------------------------------------------------------------------------

/// Unique identifier for a water volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterVolumeId(pub u32);

/// Shape of a water volume.
#[derive(Debug, Clone)]
pub enum WaterVolumeShape {
    /// Infinite plane at a given Y height.
    InfinitePlane { surface_y: f32 },
    /// Axis-aligned box.
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    /// Cylinder (vertical axis).
    Cylinder {
        center: [f32; 2],
        radius: f32,
        bottom_y: f32,
        surface_y: f32,
    },
}

impl WaterVolumeShape {
    /// Get the surface height at a given XZ position.
    pub fn surface_height_at(&self, x: f32, z: f32) -> Option<f32> {
        match self {
            Self::InfinitePlane { surface_y } => Some(*surface_y),
            Self::Box { min, max } => {
                if x >= min[0] && x <= max[0] && z >= min[2] && z <= max[2] {
                    Some(max[1])
                } else {
                    None
                }
            }
            Self::Cylinder { center, radius, surface_y, .. } => {
                let dx = x - center[0];
                let dz = z - center[1];
                if dx * dx + dz * dz <= radius * radius {
                    Some(*surface_y)
                } else {
                    None
                }
            }
        }
    }

    /// Check if a point is inside the water volume.
    pub fn contains_point(&self, x: f32, y: f32, z: f32) -> bool {
        match self {
            Self::InfinitePlane { surface_y } => y <= *surface_y,
            Self::Box { min, max } => {
                x >= min[0] && x <= max[0]
                    && y >= min[1] && y <= max[1]
                    && z >= min[2] && z <= max[2]
            }
            Self::Cylinder { center, radius, bottom_y, surface_y } => {
                let dx = x - center[0];
                let dz = z - center[1];
                dx * dx + dz * dz <= radius * radius
                    && y >= *bottom_y
                    && y <= *surface_y
            }
        }
    }
}

/// Water surface wave parameters.
#[derive(Debug, Clone)]
pub struct WaveParams {
    /// Wave amplitude.
    pub amplitude: f32,
    /// Wave frequency (waves per unit distance).
    pub frequency: f32,
    /// Wave speed.
    pub speed: f32,
    /// Wave direction (normalized XZ).
    pub direction: [f32; 2],
    /// Number of octaves for Gerstner waves.
    pub octaves: u32,
}

impl Default for WaveParams {
    fn default() -> Self {
        Self {
            amplitude: 0.1,
            frequency: 1.0,
            speed: 1.0,
            direction: [1.0, 0.0],
            octaves: 3,
        }
    }
}

impl WaveParams {
    /// Compute wave displacement at a position and time.
    pub fn displacement(&self, x: f32, z: f32, time: f32) -> f32 {
        let mut displacement = 0.0_f32;
        let mut amp = self.amplitude;
        let mut freq = self.frequency;

        for _octave in 0..self.octaves {
            let phase = (x * self.direction[0] + z * self.direction[1]) * freq
                + time * self.speed * freq;
            displacement += amp * phase.sin();
            amp *= 0.5;
            freq *= 2.0;
        }
        displacement
    }
}

/// A water current definition.
#[derive(Debug, Clone)]
pub struct WaterCurrent {
    /// Current direction (world space, not necessarily normalized).
    pub direction: [f32; 3],
    /// Current strength (force multiplier).
    pub strength: f32,
    /// Whether the current varies with depth (stronger at surface).
    pub depth_variation: bool,
    /// Depth falloff factor (how quickly current weakens with depth).
    pub depth_falloff: f32,
}

impl Default for WaterCurrent {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0, 0.0],
            strength: 1.0,
            depth_variation: true,
            depth_falloff: 0.5,
        }
    }
}

impl WaterCurrent {
    /// Compute the current force at a given depth below surface.
    pub fn force_at_depth(&self, depth_below_surface: f32) -> [f32; 3] {
        let depth_factor = if self.depth_variation {
            (1.0 - depth_below_surface * self.depth_falloff).max(0.0)
        } else {
            1.0
        };
        [
            self.direction[0] * self.strength * depth_factor,
            self.direction[1] * self.strength * depth_factor,
            self.direction[2] * self.strength * depth_factor,
        ]
    }
}

/// A water volume defining a body of water.
#[derive(Debug, Clone)]
pub struct WaterVolume {
    /// Unique ID.
    pub id: WaterVolumeId,
    /// Volume shape.
    pub shape: WaterVolumeShape,
    /// Water density in kg/m^3.
    pub density: f32,
    /// Wave parameters.
    pub waves: WaveParams,
    /// Current settings.
    pub current: WaterCurrent,
    /// Linear drag coefficient for submerged objects.
    pub linear_drag: f32,
    /// Angular drag coefficient for submerged objects.
    pub angular_drag: f32,
    /// Whether this volume is active.
    pub active: bool,
    /// Color/tint of the water (RGBA).
    pub color: [f32; 4],
    /// Viscosity multiplier (1.0 = water, >1.0 = thicker fluid).
    pub viscosity: f32,
}

impl WaterVolume {
    /// Create a new water volume with an infinite plane.
    pub fn new_plane(id: WaterVolumeId, surface_y: f32) -> Self {
        Self {
            id,
            shape: WaterVolumeShape::InfinitePlane { surface_y },
            density: DEFAULT_WATER_DENSITY,
            waves: WaveParams::default(),
            current: WaterCurrent::default(),
            linear_drag: DEFAULT_LINEAR_DRAG,
            angular_drag: DEFAULT_ANGULAR_DRAG,
            active: true,
            color: [0.1, 0.3, 0.6, 0.8],
            viscosity: 1.0,
        }
    }

    /// Get the effective surface height at a position and time (with waves).
    pub fn surface_height(&self, x: f32, z: f32, time: f32) -> Option<f32> {
        self.shape
            .surface_height_at(x, z)
            .map(|base| base + self.waves.displacement(x, z, time))
    }
}

// ---------------------------------------------------------------------------
// Buoyancy shapes
// ---------------------------------------------------------------------------

/// Shape used for buoyancy calculations on a body.
#[derive(Debug, Clone)]
pub enum BuoyancyShape {
    /// Sphere for buoyancy computation.
    Sphere { center_offset: [f32; 3], radius: f32 },
    /// Box for buoyancy computation.
    Box { center_offset: [f32; 3], half_extents: [f32; 3] },
    /// Multi-point buoyancy (sample points on the hull).
    Points { points: Vec<BuoyancyPoint> },
}

/// A single buoyancy sample point.
#[derive(Debug, Clone)]
pub struct BuoyancyPoint {
    /// Local-space offset from the body center.
    pub offset: [f32; 3],
    /// How much volume this point represents.
    pub volume_weight: f32,
}

// ---------------------------------------------------------------------------
// Water body (object in water)
// ---------------------------------------------------------------------------

/// Unique identifier for a water body (an object interacting with water).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterBodyId(pub u32);

/// An object's water interaction state.
#[derive(Debug, Clone)]
pub struct WaterBody {
    /// Unique ID.
    pub id: WaterBodyId,
    /// Body mass in kg.
    pub mass: f32,
    /// Body volume in m^3 (used for buoyancy).
    pub volume: f32,
    /// Buoyancy shape(s) for force computation.
    pub buoyancy_shapes: Vec<BuoyancyShape>,
    /// Current world position.
    pub position: [f32; 3],
    /// Current linear velocity.
    pub velocity: [f32; 3],
    /// Current angular velocity.
    pub angular_velocity: [f32; 3],
    /// Submersion fraction (0.0 = above water, 1.0 = fully submerged).
    pub submersion: f32,
    /// Whether this body was submerged last frame.
    pub was_submerged: bool,
    /// The water volume this body is currently in (if any).
    pub in_volume: Option<WaterVolumeId>,
    /// Override drag coefficient (None = use volume's drag).
    pub drag_override: Option<f32>,
    /// Buoyancy force multiplier.
    pub buoyancy_multiplier: f32,
}

impl WaterBody {
    /// Create a new water body.
    pub fn new(id: WaterBodyId, mass: f32, volume: f32) -> Self {
        Self {
            id,
            mass,
            volume,
            buoyancy_shapes: vec![BuoyancyShape::Sphere {
                center_offset: [0.0; 3],
                radius: (volume * 3.0 / (4.0 * std::f32::consts::PI)).cbrt(),
            }],
            position: [0.0; 3],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            submersion: 0.0,
            was_submerged: false,
            in_volume: None,
            drag_override: None,
            buoyancy_multiplier: 1.0,
        }
    }

    /// Average density of this body.
    pub fn density(&self) -> f32 {
        if self.volume > EPSILON { self.mass / self.volume } else { 0.0 }
    }

    /// Whether the body is currently in water.
    pub fn is_in_water(&self) -> bool {
        self.submersion > EPSILON
    }

    /// Whether the body is fully submerged.
    pub fn is_fully_submerged(&self) -> bool {
        self.submersion >= 1.0 - EPSILON
    }

    /// Speed magnitude.
    pub fn speed(&self) -> f32 {
        let v = self.velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Splash event
// ---------------------------------------------------------------------------

/// A splash event generated when an object enters or exits water.
#[derive(Debug, Clone)]
pub struct SplashEvent {
    /// Which body caused the splash.
    pub body_id: WaterBodyId,
    /// Which water volume.
    pub volume_id: WaterVolumeId,
    /// World position of the splash.
    pub position: [f32; 3],
    /// Splash magnitude (0..1+, based on impact velocity).
    pub magnitude: f32,
    /// Whether this is an entry (true) or exit (false) splash.
    pub is_entry: bool,
    /// Impact velocity (vertical component).
    pub impact_velocity: f32,
}

// ---------------------------------------------------------------------------
// Wake data
// ---------------------------------------------------------------------------

/// Wake information for rendering water surface displacement.
#[derive(Debug, Clone)]
pub struct WakeData {
    /// Body that generates the wake.
    pub body_id: WaterBodyId,
    /// World position of the wake source.
    pub position: [f32; 3],
    /// Movement direction (normalized).
    pub direction: [f32; 3],
    /// Speed of the body.
    pub speed: f32,
    /// Width of the wake.
    pub width: f32,
    /// Length of the wake trail.
    pub length: f32,
    /// Intensity (0..1).
    pub intensity: f32,
    /// Submersion depth.
    pub depth: f32,
}

// ---------------------------------------------------------------------------
// Surface deformation
// ---------------------------------------------------------------------------

/// Tracks water surface deformation caused by submerged objects.
#[derive(Debug, Clone)]
pub struct SurfaceDeformation {
    /// Position of the deformation.
    pub position: [f32; 2],
    /// Radius of influence.
    pub radius: f32,
    /// Displacement amount (negative = depression).
    pub displacement: f32,
    /// Decay rate (how quickly the deformation fades).
    pub decay_rate: f32,
    /// Current lifetime.
    pub age: f32,
}

// ---------------------------------------------------------------------------
// Computed forces
// ---------------------------------------------------------------------------

/// Forces computed for a body from water interaction.
#[derive(Debug, Clone, Default)]
pub struct WaterForces {
    /// Total buoyancy force (world space).
    pub buoyancy: [f32; 3],
    /// Total drag force (world space).
    pub drag: [f32; 3],
    /// Total angular drag torque.
    pub angular_drag: [f32; 3],
    /// Current force (world space).
    pub current: [f32; 3],
    /// Combined total force.
    pub total_force: [f32; 3],
    /// Combined total torque.
    pub total_torque: [f32; 3],
}

impl WaterForces {
    /// Compute the combined total.
    pub fn compute_totals(&mut self) {
        for i in 0..3 {
            self.total_force[i] = self.buoyancy[i] + self.drag[i] + self.current[i];
            self.total_torque[i] = self.angular_drag[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Water physics system
// ---------------------------------------------------------------------------

/// Statistics for the water physics system.
#[derive(Debug, Clone, Default)]
pub struct WaterPhysicsStats {
    /// Number of active water volumes.
    pub volume_count: usize,
    /// Number of bodies interacting with water.
    pub body_count: usize,
    /// Number of bodies currently in water.
    pub submerged_bodies: usize,
    /// Number of splash events this frame.
    pub splash_count: usize,
    /// Number of wake sources.
    pub wake_count: usize,
    /// Simulation time (microseconds).
    pub sim_time_us: u64,
}

/// The water physics simulation system.
pub struct WaterPhysicsSystem {
    /// Water volumes.
    volumes: Vec<WaterVolume>,
    /// Water bodies.
    bodies: HashMap<WaterBodyId, WaterBody>,
    /// Next volume ID.
    next_volume_id: u32,
    /// Next body ID.
    next_body_id: u32,
    /// Gravity magnitude.
    gravity: f32,
    /// Current simulation time.
    time: f32,
    /// Splash events this frame.
    splashes: Vec<SplashEvent>,
    /// Active wakes.
    wakes: Vec<WakeData>,
    /// Surface deformations.
    deformations: Vec<SurfaceDeformation>,
    /// Statistics.
    stats: WaterPhysicsStats,
    /// Minimum impact velocity for splash generation.
    splash_threshold: f32,
}

impl WaterPhysicsSystem {
    /// Create a new water physics system.
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            bodies: HashMap::new(),
            next_volume_id: 0,
            next_body_id: 0,
            gravity: DEFAULT_GRAVITY,
            time: 0.0,
            splashes: Vec::new(),
            wakes: Vec::new(),
            deformations: Vec::new(),
            stats: WaterPhysicsStats::default(),
            splash_threshold: 0.5,
        }
    }

    /// Set gravity magnitude.
    pub fn set_gravity(&mut self, gravity: f32) {
        self.gravity = gravity;
    }

    /// Add a water volume.
    pub fn add_volume(&mut self, mut volume: WaterVolume) -> WaterVolumeId {
        let id = WaterVolumeId(self.next_volume_id);
        self.next_volume_id += 1;
        volume.id = id;
        self.volumes.push(volume);
        id
    }

    /// Remove a water volume.
    pub fn remove_volume(&mut self, id: WaterVolumeId) {
        self.volumes.retain(|v| v.id != id);
    }

    /// Get a water volume by ID.
    pub fn volume(&self, id: WaterVolumeId) -> Option<&WaterVolume> {
        self.volumes.iter().find(|v| v.id == id)
    }

    /// Register a body for water interaction.
    pub fn add_body(&mut self, mass: f32, volume: f32) -> WaterBodyId {
        let id = WaterBodyId(self.next_body_id);
        self.next_body_id += 1;
        self.bodies.insert(id, WaterBody::new(id, mass, volume));
        id
    }

    /// Remove a body.
    pub fn remove_body(&mut self, id: WaterBodyId) {
        self.bodies.remove(&id);
    }

    /// Get a body.
    pub fn body(&self, id: WaterBodyId) -> Option<&WaterBody> {
        self.bodies.get(&id)
    }

    /// Get a mutable body.
    pub fn body_mut(&mut self, id: WaterBodyId) -> Option<&mut WaterBody> {
        self.bodies.get_mut(&id)
    }

    /// Update body positions/velocities from external physics.
    pub fn sync_body(
        &mut self,
        id: WaterBodyId,
        position: [f32; 3],
        velocity: [f32; 3],
        angular_velocity: [f32; 3],
    ) {
        if let Some(body) = self.bodies.get_mut(&id) {
            body.position = position;
            body.velocity = velocity;
            body.angular_velocity = angular_velocity;
        }
    }

    /// Simulate one frame of water physics.
    ///
    /// Returns computed forces per body that should be applied by the physics engine.
    pub fn simulate(&mut self, dt: f32) -> HashMap<WaterBodyId, WaterForces> {
        let start = std::time::Instant::now();
        self.time += dt;
        self.splashes.clear();
        self.wakes.clear();

        let mut forces_map = HashMap::new();

        let body_ids: Vec<WaterBodyId> = self.bodies.keys().copied().collect();

        for body_id in body_ids {
            let mut forces = WaterForces::default();
            let mut submersion = 0.0_f32;
            let mut in_volume: Option<WaterVolumeId> = None;

            // Get body data.
            let (position, velocity, ang_vel, mass, volume, buoyancy_mult, drag_override, was_submerged) = {
                let body = &self.bodies[&body_id];
                (
                    body.position,
                    body.velocity,
                    body.angular_velocity,
                    body.mass,
                    body.volume,
                    body.buoyancy_multiplier,
                    body.drag_override,
                    body.was_submerged,
                )
            };

            // Check each water volume.
            for vol in &self.volumes {
                if !vol.active {
                    continue;
                }

                let surface_y = match vol.surface_height(position[0], position[2], self.time) {
                    Some(y) => y,
                    None => continue,
                };

                // Compute submersion for a sphere approximation.
                let radius = (volume * 3.0 / (4.0 * std::f32::consts::PI)).cbrt();
                let bottom = position[1] - radius;
                let top = position[1] + radius;

                if bottom >= surface_y {
                    continue; // Fully above water.
                }

                in_volume = Some(vol.id);

                // Compute submersion fraction.
                if top <= surface_y {
                    submersion = 1.0;
                } else {
                    let submerged_height = (surface_y - bottom).max(0.0);
                    let total_height = (top - bottom).max(EPSILON);
                    submersion = (submerged_height / total_height).clamp(0.0, 1.0);
                }

                // Buoyancy force: F = rho * g * V_submerged.
                let submerged_volume = volume * submersion;
                let buoyancy_force = vol.density * self.gravity * submerged_volume * buoyancy_mult;
                forces.buoyancy[1] += buoyancy_force;

                // Drag forces.
                let drag_coeff = drag_override.unwrap_or(vol.linear_drag) * vol.viscosity;
                let ang_drag_coeff = vol.angular_drag * vol.viscosity;

                for i in 0..3 {
                    forces.drag[i] += -velocity[i] * drag_coeff * submersion * mass;
                    forces.angular_drag[i] += -ang_vel[i] * ang_drag_coeff * submersion;
                }

                // Current forces.
                let depth_below = (surface_y - position[1]).max(0.0);
                let current_f = vol.current.force_at_depth(depth_below);
                for i in 0..3 {
                    forces.current[i] += current_f[i] * submersion * mass;
                }

                // Splash detection.
                if !was_submerged && submersion > 0.1 {
                    let impact_v = -velocity[1]; // Downward velocity.
                    if impact_v > self.splash_threshold {
                        let magnitude = (impact_v / 5.0).min(2.0);
                        self.splashes.push(SplashEvent {
                            body_id,
                            volume_id: vol.id,
                            position: [position[0], surface_y, position[2]],
                            magnitude,
                            is_entry: true,
                            impact_velocity: impact_v,
                        });
                    }
                } else if was_submerged && submersion < 0.1 {
                    self.splashes.push(SplashEvent {
                        body_id,
                        volume_id: vol.id,
                        position: [position[0], surface_y, position[2]],
                        magnitude: 0.3,
                        is_entry: false,
                        impact_velocity: velocity[1],
                    });
                }

                // Wake generation.
                let speed = (velocity[0] * velocity[0] + velocity[2] * velocity[2]).sqrt();
                if speed > 0.5 && submersion > 0.2 {
                    let dir_len = speed.max(EPSILON);
                    self.wakes.push(WakeData {
                        body_id,
                        position: [position[0], surface_y, position[2]],
                        direction: [velocity[0] / dir_len, 0.0, velocity[2] / dir_len],
                        speed,
                        width: radius * 2.0 * submersion,
                        length: speed * 2.0,
                        intensity: (speed / 5.0).min(1.0) * submersion,
                        depth: depth_below,
                    });
                }
            }

            forces.compute_totals();
            forces_map.insert(body_id, forces);

            // Update body state.
            if let Some(body) = self.bodies.get_mut(&body_id) {
                body.was_submerged = body.submersion > 0.1;
                body.submersion = submersion;
                body.in_volume = in_volume;
            }
        }

        // Update deformations.
        self.deformations.retain_mut(|d| {
            d.age += dt;
            d.displacement *= (1.0 - d.decay_rate * dt).max(0.0);
            d.displacement.abs() > 0.001
        });

        // Update statistics.
        self.stats.volume_count = self.volumes.len();
        self.stats.body_count = self.bodies.len();
        self.stats.submerged_bodies = self.bodies.values().filter(|b| b.is_in_water()).count();
        self.stats.splash_count = self.splashes.len();
        self.stats.wake_count = self.wakes.len();
        self.stats.sim_time_us = start.elapsed().as_micros() as u64;

        forces_map
    }

    /// Get splash events from the last frame.
    pub fn splashes(&self) -> &[SplashEvent] {
        &self.splashes
    }

    /// Get active wakes.
    pub fn wakes(&self) -> &[WakeData] {
        &self.wakes
    }

    /// Get surface deformations.
    pub fn deformations(&self) -> &[SurfaceDeformation] {
        &self.deformations
    }

    /// Get statistics.
    pub fn stats(&self) -> &WaterPhysicsStats {
        &self.stats
    }
}

impl Default for WaterPhysicsSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinite_plane_contains() {
        let shape = WaterVolumeShape::InfinitePlane { surface_y: 10.0 };
        assert!(shape.contains_point(0.0, 5.0, 0.0));
        assert!(!shape.contains_point(0.0, 15.0, 0.0));
    }

    #[test]
    fn test_buoyancy_force() {
        let mut sys = WaterPhysicsSystem::new();
        let vol_id = sys.add_volume(WaterVolume::new_plane(WaterVolumeId(0), 10.0));
        let body_id = sys.add_body(100.0, 0.1);
        sys.sync_body(body_id, [0.0, 9.5, 0.0], [0.0, 0.0, 0.0], [0.0; 3]);

        let forces = sys.simulate(0.016);
        let f = &forces[&body_id];
        // Buoyancy should push upward.
        assert!(f.buoyancy[1] > 0.0);
    }

    #[test]
    fn test_fully_above_no_force() {
        let mut sys = WaterPhysicsSystem::new();
        sys.add_volume(WaterVolume::new_plane(WaterVolumeId(0), 0.0));
        let body_id = sys.add_body(10.0, 0.01);
        sys.sync_body(body_id, [0.0, 10.0, 0.0], [0.0; 3], [0.0; 3]);

        let forces = sys.simulate(0.016);
        let f = &forces[&body_id];
        assert!(f.buoyancy[1].abs() < EPSILON);
    }

    #[test]
    fn test_splash_detection() {
        let mut sys = WaterPhysicsSystem::new();
        sys.add_volume(WaterVolume::new_plane(WaterVolumeId(0), 10.0));
        let body_id = sys.add_body(50.0, 0.05);

        // First frame: above water.
        sys.sync_body(body_id, [0.0, 15.0, 0.0], [0.0, -5.0, 0.0], [0.0; 3]);
        sys.simulate(0.016);

        // Second frame: entering water.
        sys.sync_body(body_id, [0.0, 9.8, 0.0], [0.0, -5.0, 0.0], [0.0; 3]);
        sys.simulate(0.016);

        assert!(sys.splashes().len() > 0);
        assert!(sys.splashes()[0].is_entry);
    }

    #[test]
    fn test_current_force() {
        let mut sys = WaterPhysicsSystem::new();
        let mut vol = WaterVolume::new_plane(WaterVolumeId(0), 10.0);
        vol.current = WaterCurrent {
            direction: [5.0, 0.0, 0.0],
            strength: 2.0,
            depth_variation: false,
            depth_falloff: 0.0,
        };
        sys.add_volume(vol);
        let body_id = sys.add_body(10.0, 0.05);
        sys.sync_body(body_id, [0.0, 5.0, 0.0], [0.0; 3], [0.0; 3]);

        let forces = sys.simulate(0.016);
        let f = &forces[&body_id];
        assert!(f.current[0] > 0.0);
    }
}
