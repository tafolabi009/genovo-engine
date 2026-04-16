//! Buoyancy and water physics simulation.
//!
//! Provides:
//! - `BuoyancyBody`: Archimedes-principle buoyancy with partial submersion,
//!   drag, angular drag, and surface wave perturbation
//! - `WaterVolume`: AABB-defined water region with current (directional flow)
//!   and splash detection
//! - Analytical displaced-volume computation for box and sphere shapes
//! - `BuoyancyComponent`, `BuoyancySystem` for ECS integration

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default water density in kg/m^3.
const DEFAULT_WATER_DENSITY: f32 = 1000.0;
/// Default gravity magnitude.
const DEFAULT_GRAVITY: f32 = 9.81;
/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// AABB (local to this module)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Create a new AABB.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Check if a point is inside this AABB.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Get the center of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get the half-extents.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Get the volume.
    pub fn volume(&self) -> f32 {
        let size = self.max - self.min;
        size.x * size.y * size.z
    }
}

// ---------------------------------------------------------------------------
// Body shape for buoyancy
// ---------------------------------------------------------------------------

/// Shape of a buoyant body for displaced-volume computation.
#[derive(Debug, Clone)]
pub enum BuoyancyShape {
    /// Box shape defined by half-extents.
    Box { half_extents: Vec3 },
    /// Sphere shape defined by radius.
    Sphere { radius: f32 },
    /// Generic shape with a precomputed total volume and approximate AABB.
    Generic { total_volume: f32, half_extents: Vec3 },
}

impl BuoyancyShape {
    /// Compute the total volume of this shape.
    pub fn total_volume(&self) -> f32 {
        match self {
            BuoyancyShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            BuoyancyShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius
            }
            BuoyancyShape::Generic { total_volume, .. } => *total_volume,
        }
    }
}

// ---------------------------------------------------------------------------
// Surface wave parameters
// ---------------------------------------------------------------------------

/// Parameters for surface wave perturbation of the effective water height.
#[derive(Debug, Clone)]
pub struct WaveParams {
    /// Wave amplitude in metres.
    pub amplitude: f32,
    /// Wave frequency in Hz.
    pub frequency: f32,
    /// Wave speed (wavelength * frequency).
    pub speed: f32,
    /// Wave direction (XZ plane, normalised).
    pub direction: Vec3,
    /// Secondary wave amplitude.
    pub secondary_amplitude: f32,
    /// Secondary wave frequency.
    pub secondary_frequency: f32,
}

impl Default for WaveParams {
    fn default() -> Self {
        Self {
            amplitude: 0.1,
            frequency: 0.5,
            speed: 2.0,
            direction: Vec3::X,
            secondary_amplitude: 0.05,
            secondary_frequency: 1.2,
        }
    }
}

impl WaveParams {
    /// Compute the effective water surface height at a given XZ position and time.
    pub fn sample_height(&self, base_height: f32, position: Vec3, time: f32) -> f32 {
        let phase = position.dot(self.direction) * self.frequency - time * self.speed;
        let primary = self.amplitude * phase.sin();
        let secondary = self.secondary_amplitude
            * (position.dot(Vec3::new(-self.direction.z, 0.0, self.direction.x))
                * self.secondary_frequency
                + time * self.speed * 0.7)
                .sin();
        base_height + primary + secondary
    }
}

// ---------------------------------------------------------------------------
// Displaced volume computation
// ---------------------------------------------------------------------------

/// Compute the displaced volume of a box partially submerged in water.
///
/// The box is defined by its center position and half-extents. The water
/// surface is at `water_height` along the Y axis.
///
/// Returns the volume of the box below the water surface.
pub fn compute_box_displaced_volume(
    center: Vec3,
    half_extents: Vec3,
    water_height: f32,
) -> f32 {
    let box_bottom = center.y - half_extents.y;
    let box_top = center.y + half_extents.y;

    if box_top <= water_height {
        // Fully submerged
        return 8.0 * half_extents.x * half_extents.y * half_extents.z;
    }
    if box_bottom >= water_height {
        // Fully above water
        return 0.0;
    }

    // Partially submerged: compute the height of the submerged portion
    let submerged_height = water_height - box_bottom;
    let total_height = 2.0 * half_extents.y;
    let fraction = (submerged_height / total_height).clamp(0.0, 1.0);

    // Volume = width * depth * submerged_height
    let width = 2.0 * half_extents.x;
    let depth = 2.0 * half_extents.z;
    width * depth * submerged_height
}

/// Compute the displaced volume of a sphere partially submerged in water.
///
/// Uses the spherical cap volume formula:
///   V_cap = (pi * h^2 / 3) * (3r - h)
/// where h is the submerged depth and r is the sphere radius.
pub fn compute_sphere_displaced_volume(
    center: Vec3,
    radius: f32,
    water_height: f32,
) -> f32 {
    let sphere_bottom = center.y - radius;
    let sphere_top = center.y + radius;

    if sphere_top <= water_height {
        // Fully submerged
        return (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
    }
    if sphere_bottom >= water_height {
        // Fully above water
        return 0.0;
    }

    // Partially submerged: spherical cap
    let h = water_height - sphere_bottom; // depth of submersion
    let h = h.clamp(0.0, 2.0 * radius);

    // Volume of a spherical cap: V = (pi * h^2 / 3) * (3r - h)
    std::f32::consts::PI * h * h / 3.0 * (3.0 * radius - h)
}

/// Compute the displaced volume for a generic AABB-approximated shape.
pub fn compute_generic_displaced_volume(
    center: Vec3,
    half_extents: Vec3,
    total_volume: f32,
    water_height: f32,
) -> f32 {
    let box_bottom = center.y - half_extents.y;
    let box_top = center.y + half_extents.y;

    if box_top <= water_height {
        return total_volume;
    }
    if box_bottom >= water_height {
        return 0.0;
    }

    let submerged_height = water_height - box_bottom;
    let total_height = 2.0 * half_extents.y;
    let fraction = (submerged_height / total_height).clamp(0.0, 1.0);
    total_volume * fraction
}

/// Compute the displaced volume for any `BuoyancyShape`.
pub fn compute_displaced_volume(
    shape: &BuoyancyShape,
    center: Vec3,
    water_height: f32,
) -> f32 {
    match shape {
        BuoyancyShape::Box { half_extents } => {
            compute_box_displaced_volume(center, *half_extents, water_height)
        }
        BuoyancyShape::Sphere { radius } => {
            compute_sphere_displaced_volume(center, *radius, water_height)
        }
        BuoyancyShape::Generic {
            total_volume,
            half_extents,
        } => compute_generic_displaced_volume(center, *half_extents, *total_volume, water_height),
    }
}

// ---------------------------------------------------------------------------
// BuoyancyBody
// ---------------------------------------------------------------------------

/// A body subject to buoyancy forces.
///
/// Implements Archimedes' principle: the buoyant force equals the weight
/// of the fluid displaced by the submerged portion of the body.
///
///   F_buoyancy = fluid_density * gravity * displaced_volume
#[derive(Debug, Clone)]
pub struct BuoyancyBody {
    /// The body's shape for volume computation.
    pub shape: BuoyancyShape,
    /// Current world-space position (center of mass).
    pub position: Vec3,
    /// Current linear velocity.
    pub velocity: Vec3,
    /// Current angular velocity.
    pub angular_velocity: Vec3,
    /// Body mass in kg.
    pub mass: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// Linear drag coefficient in water.
    pub linear_drag: f32,
    /// Quadratic drag coefficient in water (proportional to v^2).
    pub quadratic_drag: f32,
    /// Angular drag coefficient in water.
    pub angular_drag: f32,
    /// Submersion fraction [0, 1]: how much of the body is underwater.
    pub submersion_fraction: f32,
    /// Whether this body is currently in water.
    pub in_water: bool,
    /// Entry/exit velocity for splash detection.
    pub entry_velocity: f32,
    /// Previous submersion state (for detecting enter/exit).
    prev_in_water: bool,
    /// Accumulated buoyancy force (for external queries / debug).
    pub buoyancy_force: Vec3,
    /// Accumulated drag force.
    pub drag_force: Vec3,
}

impl BuoyancyBody {
    /// Create a new buoyancy body.
    pub fn new(shape: BuoyancyShape, mass: f32) -> Self {
        let inv_mass = if mass > EPSILON { 1.0 / mass } else { 0.0 };
        Self {
            shape,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            linear_drag: 0.5,
            quadratic_drag: 0.2,
            angular_drag: 1.0,
            submersion_fraction: 0.0,
            in_water: false,
            entry_velocity: 0.0,
            prev_in_water: false,
            buoyancy_force: Vec3::ZERO,
            drag_force: Vec3::ZERO,
        }
    }

    /// Compute buoyancy and drag forces for this body.
    ///
    /// # Arguments
    /// * `water_height` - The Y coordinate of the water surface.
    /// * `fluid_density` - Density of the fluid in kg/m^3.
    /// * `gravity` - Gravitational acceleration (magnitude).
    /// * `current` - Water current velocity (directional flow).
    ///
    /// # Returns
    /// `(linear_force, angular_damping_torque)` to be applied to the body.
    pub fn compute_forces(
        &mut self,
        water_height: f32,
        fluid_density: f32,
        gravity: f32,
        current: Vec3,
    ) -> (Vec3, Vec3) {
        // Compute displaced volume
        let displaced_volume = compute_displaced_volume(&self.shape, self.position, water_height);
        let total_volume = self.shape.total_volume();

        self.submersion_fraction = if total_volume > EPSILON {
            (displaced_volume / total_volume).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.prev_in_water = self.in_water;
        self.in_water = self.submersion_fraction > EPSILON;

        // Detect entry/exit for splash
        if self.in_water && !self.prev_in_water {
            self.entry_velocity = self.velocity.length();
        }

        if !self.in_water {
            self.buoyancy_force = Vec3::ZERO;
            self.drag_force = Vec3::ZERO;
            return (Vec3::ZERO, Vec3::ZERO);
        }

        // Buoyancy force: F = rho * g * V_displaced (upward)
        let buoyancy = Vec3::new(0.0, fluid_density * gravity * displaced_volume, 0.0);
        self.buoyancy_force = buoyancy;

        // Drag force: F_drag = -c_linear * v - c_quadratic * |v| * v
        // Only apply drag proportional to submersion
        let relative_velocity = self.velocity - current * self.submersion_fraction;
        let speed = relative_velocity.length();

        let drag = if speed > EPSILON {
            let vel_dir = relative_velocity / speed;
            let linear_drag = -vel_dir * self.linear_drag * speed * self.submersion_fraction;
            let quad_drag =
                -vel_dir * self.quadratic_drag * speed * speed * self.submersion_fraction;
            linear_drag + quad_drag
        } else {
            Vec3::ZERO
        };
        self.drag_force = drag;

        // Angular drag: resist rotation when submerged
        let angular_damping = -self.angular_velocity
            * self.angular_drag
            * self.submersion_fraction;

        (buoyancy + drag, angular_damping)
    }

    /// Simple self-contained step (for bodies not integrated with a full physics system).
    pub fn step(
        &mut self,
        dt: f32,
        water_height: f32,
        fluid_density: f32,
        gravity: f32,
        current: Vec3,
    ) {
        let (force, angular_torque) =
            self.compute_forces(water_height, fluid_density, gravity, current);

        // Apply gravity
        let gravity_force = Vec3::new(0.0, -gravity * self.mass, 0.0);
        let total_force = gravity_force + force;

        // Integrate
        let acceleration = total_force * self.inv_mass;
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;

        // Angular damping
        self.angular_velocity += angular_torque * dt;
        self.angular_velocity *= (1.0 - self.angular_drag * self.submersion_fraction * dt).max(0.0);
    }

    /// Whether the body just entered water this frame.
    pub fn just_entered_water(&self) -> bool {
        self.in_water && !self.prev_in_water
    }

    /// Whether the body just exited water this frame.
    pub fn just_exited_water(&self) -> bool {
        !self.in_water && self.prev_in_water
    }

    /// Get the splash intensity (based on entry velocity). Returns 0 if not splashing.
    pub fn splash_intensity(&self) -> f32 {
        if self.just_entered_water() {
            self.entry_velocity
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// WaterVolume
// ---------------------------------------------------------------------------

/// A region of water defined by an AABB.
///
/// Bodies inside this volume experience buoyancy, drag, and optional
/// current (directional flow).
#[derive(Debug, Clone)]
pub struct WaterVolume {
    /// The bounding box of the water region.
    pub bounds: AABB,
    /// Fluid density in kg/m^3.
    pub fluid_density: f32,
    /// Water surface height (Y coordinate, typically bounds.max.y).
    pub surface_height: f32,
    /// Current (directional flow force applied to submerged bodies).
    pub current: Vec3,
    /// Wave parameters for surface perturbation.
    pub waves: Option<WaveParams>,
    /// Whether this volume is active.
    pub active: bool,
    /// Priority for overlapping volumes (higher wins).
    pub priority: i32,
}

impl WaterVolume {
    /// Create a new water volume.
    pub fn new(bounds: AABB) -> Self {
        Self {
            surface_height: bounds.max.y,
            bounds,
            fluid_density: DEFAULT_WATER_DENSITY,
            current: Vec3::ZERO,
            waves: None,
            active: true,
            priority: 0,
        }
    }

    /// Create a water volume from center and half-extents.
    pub fn from_center(center: Vec3, half_extents: Vec3) -> Self {
        Self::new(AABB::new(center - half_extents, center + half_extents))
    }

    /// Check if a point is inside this water volume (ignoring Y for surface check).
    pub fn contains_xz(&self, position: Vec3) -> bool {
        position.x >= self.bounds.min.x
            && position.x <= self.bounds.max.x
            && position.z >= self.bounds.min.z
            && position.z <= self.bounds.max.z
    }

    /// Check if a point is inside the full 3D volume.
    pub fn contains(&self, position: Vec3) -> bool {
        self.bounds.contains_point(position)
    }

    /// Get the effective water surface height at a given position and time,
    /// accounting for waves.
    pub fn effective_surface_height(&self, position: Vec3, time: f32) -> f32 {
        match &self.waves {
            Some(waves) => waves.sample_height(self.surface_height, position, time),
            None => self.surface_height,
        }
    }

    /// Compute buoyancy forces for a body within this volume.
    pub fn apply_to_body(
        &self,
        body: &mut BuoyancyBody,
        gravity: f32,
        time: f32,
    ) -> (Vec3, Vec3) {
        if !self.active {
            return (Vec3::ZERO, Vec3::ZERO);
        }

        let effective_height = self.effective_surface_height(body.position, time);
        body.compute_forces(effective_height, self.fluid_density, gravity, self.current)
    }
}

// ---------------------------------------------------------------------------
// SplashEvent
// ---------------------------------------------------------------------------

/// An event generated when a body enters or exits water.
#[derive(Debug, Clone)]
pub struct SplashEvent {
    /// World position of the splash.
    pub position: Vec3,
    /// Velocity of the body at impact.
    pub velocity: Vec3,
    /// Intensity (speed at impact).
    pub intensity: f32,
    /// Whether this is an entry (true) or exit (false) event.
    pub is_entry: bool,
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching buoyancy behaviour to an entity.
#[derive(Debug, Clone)]
pub struct BuoyancyComponent {
    /// The buoyancy body.
    pub body: BuoyancyBody,
    /// Whether the buoyancy simulation is active.
    pub active: bool,
    /// Whether to use the position from the entity's transform (set by system).
    pub sync_transform: bool,
}

impl BuoyancyComponent {
    /// Create a new buoyancy component.
    pub fn new(shape: BuoyancyShape, mass: f32) -> Self {
        Self {
            body: BuoyancyBody::new(shape, mass),
            active: true,
            sync_transform: true,
        }
    }

    /// Create a box buoyancy component.
    pub fn box_shape(half_extents: Vec3, mass: f32) -> Self {
        Self::new(BuoyancyShape::Box { half_extents }, mass)
    }

    /// Create a sphere buoyancy component.
    pub fn sphere(radius: f32, mass: f32) -> Self {
        Self::new(BuoyancyShape::Sphere { radius }, mass)
    }
}

/// System that updates all buoyancy bodies each frame.
pub struct BuoyancySystem {
    /// Water volumes in the scene.
    pub water_volumes: Vec<WaterVolume>,
    /// Gravity magnitude.
    pub gravity: f32,
    /// Running simulation time (for wave animation).
    sim_time: f32,
    /// Splash events from the most recent update.
    pub splash_events: Vec<SplashEvent>,
}

impl Default for BuoyancySystem {
    fn default() -> Self {
        Self {
            water_volumes: Vec::new(),
            gravity: DEFAULT_GRAVITY,
            sim_time: 0.0,
            splash_events: Vec::new(),
        }
    }
}

impl BuoyancySystem {
    /// Create a new buoyancy system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a water volume.
    pub fn add_water_volume(&mut self, volume: WaterVolume) {
        self.water_volumes.push(volume);
    }

    /// Update all buoyancy bodies.
    ///
    /// For each body, finds the highest-priority water volume that contains it
    /// and applies buoyancy forces.
    pub fn update(&mut self, dt: f32, bodies: &mut [BuoyancyComponent]) {
        self.sim_time += dt;
        self.splash_events.clear();

        for component in bodies.iter_mut() {
            if !component.active {
                continue;
            }

            let body = &mut component.body;

            // Find the best matching water volume
            let mut best_volume: Option<&WaterVolume> = None;
            let mut best_priority = i32::MIN;

            for volume in &self.water_volumes {
                if !volume.active {
                    continue;
                }
                if volume.contains_xz(body.position) && volume.priority > best_priority {
                    best_volume = Some(volume);
                    best_priority = volume.priority;
                }
            }

            if let Some(volume) = best_volume {
                let effective_height =
                    volume.effective_surface_height(body.position, self.sim_time);
                let prev_in_water = body.in_water;

                body.compute_forces(
                    effective_height,
                    volume.fluid_density,
                    self.gravity,
                    volume.current,
                );

                // Generate splash events
                if body.just_entered_water() {
                    self.splash_events.push(SplashEvent {
                        position: body.position,
                        velocity: body.velocity,
                        intensity: body.entry_velocity,
                        is_entry: true,
                    });
                } else if body.just_exited_water() {
                    self.splash_events.push(SplashEvent {
                        position: body.position,
                        velocity: body.velocity,
                        intensity: body.velocity.length(),
                        is_entry: false,
                    });
                }
            } else {
                // Not in any water volume
                body.in_water = false;
                body.submersion_fraction = 0.0;
                body.buoyancy_force = Vec3::ZERO;
                body.drag_force = Vec3::ZERO;
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_fully_submerged() {
        let center = Vec3::new(0.0, -2.0, 0.0);
        let half_extents = Vec3::new(0.5, 0.5, 0.5);
        let water_height = 0.0;

        let volume = compute_box_displaced_volume(center, half_extents, water_height);
        let expected = 8.0 * 0.5 * 0.5 * 0.5; // 1.0
        assert!((volume - expected).abs() < 1e-4, "Got {}", volume);
    }

    #[test]
    fn test_box_above_water() {
        let center = Vec3::new(0.0, 2.0, 0.0);
        let half_extents = Vec3::new(0.5, 0.5, 0.5);
        let water_height = 0.0;

        let volume = compute_box_displaced_volume(center, half_extents, water_height);
        assert!(volume.abs() < 1e-6, "Above water should have 0 volume");
    }

    #[test]
    fn test_box_half_submerged() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let half_extents = Vec3::new(1.0, 1.0, 1.0);
        let water_height = 0.0; // Water at center, box from -1 to +1

        let volume = compute_box_displaced_volume(center, half_extents, water_height);
        let total = 8.0;
        assert!(
            (volume - total * 0.5).abs() < 1e-4,
            "Half submerged: got {}",
            volume
        );
    }

    #[test]
    fn test_sphere_fully_submerged() {
        let center = Vec3::new(0.0, -5.0, 0.0);
        let radius = 1.0;
        let water_height = 0.0;

        let volume = compute_sphere_displaced_volume(center, radius, water_height);
        let expected = (4.0 / 3.0) * std::f32::consts::PI;
        assert!((volume - expected).abs() < 1e-3, "Got {}", volume);
    }

    #[test]
    fn test_sphere_above_water() {
        let center = Vec3::new(0.0, 5.0, 0.0);
        let radius = 1.0;
        let water_height = 0.0;

        let volume = compute_sphere_displaced_volume(center, radius, water_height);
        assert!(volume.abs() < 1e-6);
    }

    #[test]
    fn test_sphere_half_submerged() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let radius = 1.0;
        let water_height = 0.0; // Water at center

        let volume = compute_sphere_displaced_volume(center, radius, water_height);
        let total = (4.0 / 3.0) * std::f32::consts::PI;
        assert!(
            (volume - total * 0.5).abs() < 1e-3,
            "Half sphere should be half volume: got {} expected {}",
            volume,
            total * 0.5
        );
    }

    #[test]
    fn test_buoyancy_body_floats() {
        // A light box should float (buoyancy > gravity)
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            0.5, // Very light
        );
        body.position = Vec3::new(0.0, -0.3, 0.0); // Partially submerged

        let (force, _) = body.compute_forces(0.0, 1000.0, 9.81, Vec3::ZERO);

        // Buoyancy should push upward
        assert!(
            force.y > 0.0,
            "Buoyancy force should be positive: {}",
            force.y
        );
    }

    #[test]
    fn test_buoyancy_body_sinks_when_heavy() {
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Sphere { radius: 0.1 },
            1000.0, // Very heavy, small sphere
        );
        body.position = Vec3::new(0.0, -0.05, 0.0); // Partially submerged

        let (force, _) = body.compute_forces(0.0, 1000.0, 9.81, Vec3::ZERO);
        let gravity_force = 9.81 * 1000.0;

        // Net force should be downward (gravity > buoyancy)
        assert!(
            force.y < gravity_force,
            "Heavy body buoyancy {} should be less than gravity {}",
            force.y,
            gravity_force
        );
    }

    #[test]
    fn test_drag_opposes_motion() {
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            10.0,
        );
        body.position = Vec3::new(0.0, -1.0, 0.0); // Fully submerged
        body.velocity = Vec3::new(5.0, 0.0, 0.0); // Moving in +X

        let (force, _) = body.compute_forces(0.0, 1000.0, 9.81, Vec3::ZERO);

        // Drag should oppose motion (negative X component)
        assert!(
            force.x < 0.0,
            "Drag should oppose +X motion: force.x = {}",
            force.x
        );
    }

    #[test]
    fn test_water_volume() {
        let volume = WaterVolume::new(AABB::new(
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
        ));

        assert!(volume.contains_xz(Vec3::ZERO));
        assert!(!volume.contains_xz(Vec3::new(15.0, 0.0, 0.0)));
        assert_eq!(volume.surface_height, 0.0);
    }

    #[test]
    fn test_water_volume_with_waves() {
        let mut volume = WaterVolume::new(AABB::new(
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
        ));
        volume.waves = Some(WaveParams {
            amplitude: 0.5,
            ..Default::default()
        });

        let h1 = volume.effective_surface_height(Vec3::ZERO, 0.0);
        let h2 = volume.effective_surface_height(Vec3::ZERO, 1.0);

        // Heights should differ due to wave animation
        assert!(
            (h1 - h2).abs() > 0.001,
            "Wave should vary over time: h1={}, h2={}",
            h1,
            h2
        );
    }

    #[test]
    fn test_buoyancy_body_simple_step() {
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            5.0,
        );
        body.position = Vec3::new(0.0, 0.0, 0.0);

        // Step for a few frames
        for _ in 0..60 {
            body.step(1.0 / 60.0, 0.5, 1000.0, 9.81, Vec3::ZERO);
        }

        // Body should have settled near the water surface
        // (oscillating around equilibrium)
        assert!(
            body.position.y > -5.0 && body.position.y < 5.0,
            "Body should be near surface: y = {}",
            body.position.y
        );
    }

    #[test]
    fn test_splash_detection() {
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Sphere { radius: 0.3 },
            2.0,
        );
        body.position = Vec3::new(0.0, 1.0, 0.0); // Above water
        body.velocity = Vec3::new(0.0, -5.0, 0.0); // Falling

        // First compute: not in water
        body.compute_forces(0.0, 1000.0, 9.81, Vec3::ZERO);
        assert!(!body.in_water);

        // Move below water surface
        body.position = Vec3::new(0.0, -0.1, 0.0);
        body.compute_forces(0.0, 1000.0, 9.81, Vec3::ZERO);

        assert!(body.in_water);
        assert!(body.just_entered_water());
        assert!(body.splash_intensity() > 0.0);
    }

    #[test]
    fn test_water_current() {
        let mut body = BuoyancyBody::new(
            BuoyancyShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            10.0,
        );
        body.position = Vec3::new(0.0, -1.0, 0.0); // Fully submerged
        body.velocity = Vec3::ZERO;

        // Apply current in +X direction
        let current = Vec3::new(3.0, 0.0, 0.0);
        let (force, _) = body.compute_forces(0.0, 1000.0, 9.81, current);

        // The drag from the current should push the body in +X
        assert!(
            force.x > 0.0,
            "Current should push in +X: force.x = {}",
            force.x
        );
    }

    #[test]
    fn test_buoyancy_component() {
        let component = BuoyancyComponent::box_shape(Vec3::splat(0.5), 5.0);
        assert!(component.active);
    }

    #[test]
    fn test_buoyancy_system() {
        let mut system = BuoyancySystem::new();
        system.add_water_volume(WaterVolume::new(AABB::new(
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
        )));

        let mut bodies = vec![BuoyancyComponent::sphere(0.5, 5.0)];
        bodies[0].body.position = Vec3::new(0.0, -0.3, 0.0);

        system.update(1.0 / 60.0, &mut bodies);

        assert!(bodies[0].body.in_water);
        assert!(bodies[0].body.submersion_fraction > 0.0);
    }
}
