// engine/physics/src/conveyor.rs
//
// Conveyor belt and moving surface physics for the Genovo engine.
//
// Features:
// - Surface velocity applied to contacting bodies
// - Configurable direction and speed
// - Belt sections with independent speeds
// - Acceleration zones (speed up/slow down bodies)
// - Curved conveyors (circular arc paths)
// - ECS integration via components

use glam::{Quat, Vec2, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Maximum number of conveyors in the system.
pub const MAX_CONVEYORS: usize = 256;

/// Maximum number of sections per conveyor.
pub const MAX_SECTIONS_PER_CONVEYOR: usize = 32;

/// Default friction coefficient between belt and objects.
pub const DEFAULT_BELT_FRICTION: f32 = 0.8;

/// Default acceleration for acceleration zones.
pub const DEFAULT_ACCELERATION: f32 = 5.0;

// ---------------------------------------------------------------------------
// Conveyor direction
// ---------------------------------------------------------------------------

/// How the conveyor direction is defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConveyorDirection {
    /// Fixed world-space direction.
    WorldSpace(Vec3),
    /// Local-space direction (relative to conveyor transform).
    LocalSpace(Vec3),
    /// Direction along a spline path.
    AlongPath,
    /// Direction toward a target point.
    TowardPoint(Vec3),
}

impl ConveyorDirection {
    /// Resolve the world-space direction.
    pub fn resolve(&self, local_to_world: &Quat, position: Vec3) -> Vec3 {
        match self {
            Self::WorldSpace(dir) => dir.normalize_or_zero(),
            Self::LocalSpace(dir) => (*local_to_world * *dir).normalize_or_zero(),
            Self::AlongPath => Vec3::X, // Placeholder; real impl uses spline tangent
            Self::TowardPoint(target) => (*target - position).normalize_or_zero(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conveyor belt
// ---------------------------------------------------------------------------

/// A single conveyor belt that applies surface velocity to contacting bodies.
#[derive(Debug, Clone)]
pub struct ConveyorBelt {
    /// Unique identifier.
    pub id: ConveyorId,
    /// Whether the conveyor is active.
    pub active: bool,
    /// World-space position of the conveyor.
    pub position: Vec3,
    /// Orientation of the conveyor.
    pub rotation: Quat,
    /// Conveyor surface dimensions (length x width).
    pub dimensions: Vec2,
    /// Direction of the belt movement.
    pub direction: ConveyorDirection,
    /// Belt speed in world units per second.
    pub speed: f32,
    /// Target speed (for smooth speed changes).
    pub target_speed: f32,
    /// Speed change rate.
    pub speed_lerp: f32,
    /// Friction coefficient between belt and contacting objects.
    pub friction: f32,
    /// Whether the belt reverses direction.
    pub reversible: bool,
    /// Current reverse state.
    pub reversed: bool,
    /// Maximum force the belt can exert on an object.
    pub max_force: f32,
    /// Sections of this conveyor (for variable speed along the belt).
    pub sections: Vec<ConveyorSection>,
    /// Bodies currently in contact with this conveyor.
    pub contacting_bodies: Vec<ContactInfo>,
    /// Whether this is a roller conveyor (discrete rollers vs continuous belt).
    pub roller: bool,
    /// Roller spacing (if roller conveyor).
    pub roller_spacing: f32,
    /// Visual belt scroll offset (for UV scrolling).
    pub scroll_offset: f32,
}

/// Unique identifier for a conveyor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConveyorId(pub u32);

/// A section of a conveyor with its own speed and behaviour.
#[derive(Debug, Clone)]
pub struct ConveyorSection {
    /// Start position along the conveyor (0 = beginning, 1 = end).
    pub start_t: f32,
    /// End position along the conveyor.
    pub end_t: f32,
    /// Speed multiplier for this section (relative to belt speed).
    pub speed_multiplier: f32,
    /// Direction override (if different from main belt direction).
    pub direction_override: Option<ConveyorDirection>,
    /// Section type.
    pub section_type: SectionType,
}

/// Type of conveyor section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    /// Normal belt section.
    Normal,
    /// Acceleration zone (speeds up objects).
    Accelerate,
    /// Deceleration zone (slows down objects).
    Decelerate,
    /// Diverter (redirects objects to the side).
    Diverter,
    /// Stopper (holds objects until released).
    Stopper,
    /// Sensor (detects objects passing through).
    Sensor,
}

/// Information about a body contacting the conveyor.
#[derive(Debug, Clone)]
pub struct ContactInfo {
    /// ID of the contacting body.
    pub body_id: u64,
    /// Contact point on the conveyor surface.
    pub contact_point: Vec3,
    /// Contact normal.
    pub contact_normal: Vec3,
    /// How long the body has been in contact (seconds).
    pub contact_duration: f32,
    /// Which section the body is on (index into sections, or -1 for no section).
    pub section_index: i32,
    /// Current velocity being applied to this body by the conveyor.
    pub applied_velocity: Vec3,
}

impl ConveyorBelt {
    /// Create a new conveyor belt.
    pub fn new(id: ConveyorId, position: Vec3, dimensions: Vec2, speed: f32) -> Self {
        Self {
            id,
            active: true,
            position,
            rotation: Quat::IDENTITY,
            dimensions,
            direction: ConveyorDirection::LocalSpace(Vec3::X),
            speed,
            target_speed: speed,
            speed_lerp: 5.0,
            friction: DEFAULT_BELT_FRICTION,
            reversible: false,
            reversed: false,
            max_force: 1000.0,
            sections: Vec::new(),
            contacting_bodies: Vec::new(),
            roller: false,
            roller_spacing: 0.1,
            scroll_offset: 0.0,
        }
    }

    /// Set the belt direction.
    pub fn with_direction(mut self, direction: ConveyorDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set friction.
    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Add a section to the conveyor.
    pub fn add_section(&mut self, section: ConveyorSection) {
        if self.sections.len() < MAX_SECTIONS_PER_CONVEYOR {
            self.sections.push(section);
        }
    }

    /// Toggle the belt direction.
    pub fn reverse(&mut self) {
        if self.reversible {
            self.reversed = !self.reversed;
            self.target_speed = -self.target_speed;
        }
    }

    /// Set the target speed (smooth transition).
    pub fn set_speed(&mut self, speed: f32) {
        self.target_speed = if self.reversed { -speed } else { speed };
    }

    /// Stop the conveyor (smooth deceleration).
    pub fn stop(&mut self) {
        self.target_speed = 0.0;
    }

    /// Start the conveyor at its default speed.
    pub fn start(&mut self, speed: f32) {
        self.active = true;
        self.target_speed = if self.reversed { -speed } else { speed };
    }

    /// Update the conveyor belt.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        // Smooth speed transition
        let speed_diff = self.target_speed - self.speed;
        if speed_diff.abs() > EPSILON {
            self.speed += speed_diff * (self.speed_lerp * dt).min(1.0);
        } else {
            self.speed = self.target_speed;
        }

        // Update scroll offset for visual animation
        self.scroll_offset += self.speed * dt;
        if self.scroll_offset.abs() > 100.0 {
            self.scroll_offset %= 100.0;
        }

        // Update contact durations
        for contact in &mut self.contacting_bodies {
            contact.contact_duration += dt;
        }
    }

    /// Compute the surface velocity at a given point on the conveyor.
    pub fn surface_velocity_at(&self, point: Vec3) -> Vec3 {
        if !self.active || self.speed.abs() < EPSILON {
            return Vec3::ZERO;
        }

        let base_dir = self.direction.resolve(&self.rotation, self.position);
        let mut speed = self.speed;

        // Check if point is in a section
        let local_pos = self.world_to_local(point);
        let t = (local_pos.x / self.dimensions.x + 0.5).clamp(0.0, 1.0);

        for section in &self.sections {
            if t >= section.start_t && t <= section.end_t {
                speed *= section.speed_multiplier;
                if let Some(ref dir_override) = section.direction_override {
                    let override_dir = dir_override.resolve(&self.rotation, self.position);
                    return override_dir * speed;
                }
                break;
            }
        }

        base_dir * speed
    }

    /// Compute the force to apply to a contacting body.
    pub fn compute_force(
        &self,
        body_velocity: Vec3,
        body_mass: f32,
        contact_point: Vec3,
        contact_normal: Vec3,
        dt: f32,
    ) -> Vec3 {
        if !self.active || self.speed.abs() < EPSILON {
            return Vec3::ZERO;
        }

        let belt_velocity = self.surface_velocity_at(contact_point);

        // Project belt velocity onto the conveyor plane (remove normal component)
        let belt_tangent = belt_velocity - contact_normal * belt_velocity.dot(contact_normal);
        let body_tangent = body_velocity - contact_normal * body_velocity.dot(contact_normal);

        // Relative velocity of body vs belt
        let relative_velocity = body_tangent - belt_tangent;

        // Apply friction force to match body velocity to belt velocity
        let force_magnitude = relative_velocity.length() * body_mass * self.friction / (dt + EPSILON);
        let force_magnitude = force_magnitude.min(self.max_force);

        if relative_velocity.length_squared() > EPSILON {
            -relative_velocity.normalize() * force_magnitude
        } else {
            Vec3::ZERO
        }
    }

    /// Transform a world-space point to local conveyor space.
    fn world_to_local(&self, point: Vec3) -> Vec3 {
        let inv_rot = self.rotation.conjugate();
        inv_rot * (point - self.position)
    }

    /// Check if a point is on the conveyor surface.
    pub fn is_on_surface(&self, point: Vec3) -> bool {
        let local = self.world_to_local(point);
        local.x.abs() <= self.dimensions.x * 0.5
            && local.z.abs() <= self.dimensions.y * 0.5
            && local.y.abs() < 0.1 // Small tolerance above the surface
    }

    /// Register a body contact.
    pub fn register_contact(
        &mut self,
        body_id: u64,
        contact_point: Vec3,
        contact_normal: Vec3,
    ) {
        // Check if already tracking this body
        for contact in &mut self.contacting_bodies {
            if contact.body_id == body_id {
                contact.contact_point = contact_point;
                contact.contact_normal = contact_normal;
                return;
            }
        }

        // Find which section
        let local_pos = self.world_to_local(contact_point);
        let t = (local_pos.x / self.dimensions.x + 0.5).clamp(0.0, 1.0);
        let section_index = self
            .sections
            .iter()
            .position(|s| t >= s.start_t && t <= s.end_t)
            .map(|i| i as i32)
            .unwrap_or(-1);

        self.contacting_bodies.push(ContactInfo {
            body_id,
            contact_point,
            contact_normal,
            contact_duration: 0.0,
            section_index,
            applied_velocity: Vec3::ZERO,
        });
    }

    /// Remove a body contact.
    pub fn remove_contact(&mut self, body_id: u64) {
        self.contacting_bodies.retain(|c| c.body_id != body_id);
    }

    /// Get the number of bodies currently on the conveyor.
    pub fn body_count(&self) -> usize {
        self.contacting_bodies.len()
    }
}

// ---------------------------------------------------------------------------
// Curved conveyor
// ---------------------------------------------------------------------------

/// A curved conveyor that follows a circular arc.
#[derive(Debug, Clone)]
pub struct CurvedConveyor {
    /// Base conveyor properties.
    pub base: ConveyorBelt,
    /// Center of the arc.
    pub arc_center: Vec3,
    /// Inner radius of the curve.
    pub inner_radius: f32,
    /// Outer radius of the curve.
    pub outer_radius: f32,
    /// Start angle of the arc (radians).
    pub start_angle: f32,
    /// End angle of the arc (radians).
    pub end_angle: f32,
    /// Whether the belt curves clockwise.
    pub clockwise: bool,
    /// Number of segments for collision approximation.
    pub segment_count: u32,
}

impl CurvedConveyor {
    /// Create a new curved conveyor.
    pub fn new(
        id: ConveyorId,
        center: Vec3,
        inner_radius: f32,
        outer_radius: f32,
        start_angle: f32,
        end_angle: f32,
        speed: f32,
    ) -> Self {
        let mid_radius = (inner_radius + outer_radius) * 0.5;
        let width = outer_radius - inner_radius;
        let arc_length = (end_angle - start_angle).abs() * mid_radius;

        Self {
            base: ConveyorBelt::new(
                id,
                center,
                Vec2::new(arc_length, width),
                speed,
            ),
            arc_center: center,
            inner_radius,
            outer_radius,
            start_angle,
            end_angle,
            clockwise: true,
            segment_count: 16,
        }
    }

    /// Compute the tangent direction at a point on the curve.
    pub fn tangent_at(&self, angle: f32) -> Vec3 {
        let dir = if self.clockwise {
            Vec3::new(-angle.sin(), 0.0, angle.cos())
        } else {
            Vec3::new(angle.sin(), 0.0, -angle.cos())
        };
        dir
    }

    /// Get the surface velocity at a world position.
    pub fn surface_velocity_at(&self, point: Vec3) -> Vec3 {
        if !self.base.active {
            return Vec3::ZERO;
        }

        let relative = point - self.arc_center;
        let angle = relative.z.atan2(relative.x);
        let tangent = self.tangent_at(angle);

        let radius = (relative.x * relative.x + relative.z * relative.z).sqrt();
        let mid_radius = (self.inner_radius + self.outer_radius) * 0.5;

        // Adjust speed based on radius (inner track moves slower than outer)
        let speed_factor = radius / (mid_radius + EPSILON);
        tangent * self.base.speed * speed_factor
    }

    /// Check if a point is within the curved conveyor area.
    pub fn contains_point(&self, point: Vec3) -> bool {
        let relative = point - self.arc_center;
        let radius = (relative.x * relative.x + relative.z * relative.z).sqrt();
        let angle = relative.z.atan2(relative.x);

        let (min_angle, max_angle) = if self.start_angle < self.end_angle {
            (self.start_angle, self.end_angle)
        } else {
            (self.end_angle, self.start_angle)
        };

        radius >= self.inner_radius
            && radius <= self.outer_radius
            && angle >= min_angle
            && angle <= max_angle
    }
}

// ---------------------------------------------------------------------------
// Acceleration zone
// ---------------------------------------------------------------------------

/// An acceleration zone that gradually changes body velocity.
#[derive(Debug, Clone)]
pub struct AccelerationZone {
    /// Unique ID.
    pub id: u32,
    /// World-space centre.
    pub position: Vec3,
    /// Half-extents of the zone.
    pub half_extents: Vec3,
    /// Acceleration direction.
    pub direction: Vec3,
    /// Acceleration magnitude (units/s^2).
    pub acceleration: f32,
    /// Maximum speed the zone can accelerate a body to.
    pub max_speed: f32,
    /// Whether this zone decelerates instead of accelerates.
    pub decelerate: bool,
    /// Whether the zone is active.
    pub active: bool,
    /// Whether to only affect the velocity component along the zone direction.
    pub directional_only: bool,
}

impl AccelerationZone {
    /// Create a new acceleration zone.
    pub fn new(id: u32, position: Vec3, half_extents: Vec3, direction: Vec3, acceleration: f32) -> Self {
        Self {
            id,
            position,
            half_extents,
            direction: direction.normalize_or_zero(),
            acceleration,
            max_speed: 20.0,
            decelerate: false,
            active: true,
            directional_only: true,
        }
    }

    /// Create a deceleration zone.
    pub fn deceleration(id: u32, position: Vec3, half_extents: Vec3, deceleration: f32) -> Self {
        Self {
            id,
            position,
            half_extents,
            direction: Vec3::ZERO,
            acceleration: deceleration,
            max_speed: 0.0,
            decelerate: true,
            active: true,
            directional_only: false,
        }
    }

    /// Check if a point is inside the zone.
    pub fn contains(&self, point: Vec3) -> bool {
        let local = point - self.position;
        local.x.abs() <= self.half_extents.x
            && local.y.abs() <= self.half_extents.y
            && local.z.abs() <= self.half_extents.z
    }

    /// Compute the velocity change for a body inside the zone.
    pub fn compute_velocity_change(&self, body_velocity: Vec3, body_mass: f32, dt: f32) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }

        if self.decelerate {
            // Decelerate: reduce velocity magnitude
            let speed = body_velocity.length();
            if speed < EPSILON {
                return Vec3::ZERO;
            }
            let decel = (self.acceleration * dt).min(speed);
            return -body_velocity.normalize() * decel;
        }

        if self.directional_only {
            // Accelerate along the zone direction
            let current_speed = body_velocity.dot(self.direction);
            if current_speed >= self.max_speed {
                return Vec3::ZERO;
            }
            let accel = (self.acceleration * dt).min(self.max_speed - current_speed);
            self.direction * accel
        } else {
            // Accelerate in all directions toward max speed
            let speed = body_velocity.length();
            if speed >= self.max_speed {
                return Vec3::ZERO;
            }
            let accel = self.acceleration * dt;
            self.direction * accel
        }
    }
}

// ---------------------------------------------------------------------------
// Conveyor system (ECS)
// ---------------------------------------------------------------------------

/// The conveyor system manages all conveyors and applies forces to bodies.
#[derive(Debug)]
pub struct ConveyorSystem {
    /// All straight conveyors.
    pub conveyors: Vec<ConveyorBelt>,
    /// All curved conveyors.
    pub curved_conveyors: Vec<CurvedConveyor>,
    /// All acceleration zones.
    pub acceleration_zones: Vec<AccelerationZone>,
    /// Next conveyor ID.
    next_id: u32,
    /// Whether the system is active.
    pub active: bool,
    /// Statistics.
    pub stats: ConveyorStats,
}

/// Conveyor system statistics.
#[derive(Debug, Clone, Default)]
pub struct ConveyorStats {
    /// Total conveyor count.
    pub conveyor_count: u32,
    /// Active conveyor count.
    pub active_conveyors: u32,
    /// Total bodies on conveyors.
    pub total_bodies_on_conveyors: u32,
    /// Total acceleration zones.
    pub acceleration_zone_count: u32,
}

impl ConveyorSystem {
    /// Create a new conveyor system.
    pub fn new() -> Self {
        Self {
            conveyors: Vec::new(),
            curved_conveyors: Vec::new(),
            acceleration_zones: Vec::new(),
            next_id: 0,
            active: true,
            stats: ConveyorStats::default(),
        }
    }

    /// Create and add a straight conveyor.
    pub fn add_conveyor(&mut self, position: Vec3, dimensions: Vec2, speed: f32) -> ConveyorId {
        let id = ConveyorId(self.next_id);
        self.next_id += 1;
        self.conveyors
            .push(ConveyorBelt::new(id, position, dimensions, speed));
        id
    }

    /// Add an acceleration zone.
    pub fn add_acceleration_zone(
        &mut self,
        position: Vec3,
        half_extents: Vec3,
        direction: Vec3,
        acceleration: f32,
    ) -> u32 {
        let id = self.acceleration_zones.len() as u32;
        self.acceleration_zones.push(AccelerationZone::new(
            id, position, half_extents, direction, acceleration,
        ));
        id
    }

    /// Update all conveyors.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        for conveyor in &mut self.conveyors {
            conveyor.update(dt);
        }
        for curved in &mut self.curved_conveyors {
            curved.base.update(dt);
        }

        // Update stats
        self.stats.conveyor_count = (self.conveyors.len() + self.curved_conveyors.len()) as u32;
        self.stats.active_conveyors = self
            .conveyors
            .iter()
            .filter(|c| c.active)
            .count() as u32
            + self
                .curved_conveyors
                .iter()
                .filter(|c| c.base.active)
                .count() as u32;
        self.stats.total_bodies_on_conveyors = self
            .conveyors
            .iter()
            .map(|c| c.body_count() as u32)
            .sum::<u32>()
            + self
                .curved_conveyors
                .iter()
                .map(|c| c.base.body_count() as u32)
                .sum::<u32>();
        self.stats.acceleration_zone_count = self.acceleration_zones.len() as u32;
    }

    /// Get a conveyor by ID.
    pub fn get(&self, id: ConveyorId) -> Option<&ConveyorBelt> {
        self.conveyors.iter().find(|c| c.id == id)
    }

    /// Get a mutable conveyor by ID.
    pub fn get_mut(&mut self, id: ConveyorId) -> Option<&mut ConveyorBelt> {
        self.conveyors.iter_mut().find(|c| c.id == id)
    }

    /// Query the surface velocity at a point from any conveyor.
    pub fn query_surface_velocity(&self, point: Vec3) -> Vec3 {
        for conveyor in &self.conveyors {
            if conveyor.active && conveyor.is_on_surface(point) {
                return conveyor.surface_velocity_at(point);
            }
        }
        for curved in &self.curved_conveyors {
            if curved.base.active && curved.contains_point(point) {
                return curved.surface_velocity_at(point);
            }
        }
        Vec3::ZERO
    }

    /// Query the acceleration at a point from any acceleration zone.
    pub fn query_acceleration(&self, point: Vec3, body_velocity: Vec3, body_mass: f32, dt: f32) -> Vec3 {
        let mut total_accel = Vec3::ZERO;
        for zone in &self.acceleration_zones {
            if zone.active && zone.contains(point) {
                total_accel += zone.compute_velocity_change(body_velocity, body_mass, dt);
            }
        }
        total_accel
    }
}

impl Default for ConveyorSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// ECS component for marking an entity as a conveyor belt.
#[derive(Debug, Clone)]
pub struct ConveyorComponent {
    /// Conveyor ID in the system.
    pub conveyor_id: ConveyorId,
    /// Whether this component is enabled.
    pub enabled: bool,
}

impl ConveyorComponent {
    pub fn new(id: ConveyorId) -> Self {
        Self {
            conveyor_id: id,
            enabled: true,
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
    fn test_conveyor_belt_basic() {
        let mut belt = ConveyorBelt::new(
            ConveyorId(0),
            Vec3::ZERO,
            Vec2::new(10.0, 2.0),
            5.0,
        );
        belt.update(0.1);
        let vel = belt.surface_velocity_at(Vec3::ZERO);
        assert!(vel.length() > 0.0);
    }

    #[test]
    fn test_conveyor_on_surface() {
        let belt = ConveyorBelt::new(
            ConveyorId(0),
            Vec3::ZERO,
            Vec2::new(10.0, 2.0),
            5.0,
        );
        assert!(belt.is_on_surface(Vec3::new(1.0, 0.0, 0.5)));
        assert!(!belt.is_on_surface(Vec3::new(100.0, 0.0, 0.0)));
    }

    #[test]
    fn test_conveyor_stop() {
        let mut belt = ConveyorBelt::new(
            ConveyorId(0),
            Vec3::ZERO,
            Vec2::new(10.0, 2.0),
            5.0,
        );
        belt.stop();
        // After many updates, speed should approach 0
        for _ in 0..100 {
            belt.update(0.1);
        }
        assert!(belt.speed.abs() < 0.1);
    }

    #[test]
    fn test_acceleration_zone() {
        let zone = AccelerationZone::new(
            0,
            Vec3::ZERO,
            Vec3::splat(5.0),
            Vec3::X,
            10.0,
        );
        assert!(zone.contains(Vec3::ZERO));
        assert!(!zone.contains(Vec3::new(10.0, 0.0, 0.0)));

        let dv = zone.compute_velocity_change(Vec3::ZERO, 1.0, 0.1);
        assert!(dv.x > 0.0);
    }

    #[test]
    fn test_deceleration_zone() {
        let zone = AccelerationZone::deceleration(0, Vec3::ZERO, Vec3::splat(5.0), 10.0);
        let velocity = Vec3::new(5.0, 0.0, 0.0);
        let dv = zone.compute_velocity_change(velocity, 1.0, 0.1);
        assert!(dv.x < 0.0); // Should decelerate
    }

    #[test]
    fn test_conveyor_system() {
        let mut sys = ConveyorSystem::new();
        let id = sys.add_conveyor(Vec3::ZERO, Vec2::new(10.0, 2.0), 5.0);
        sys.update(0.1);
        assert_eq!(sys.stats.conveyor_count, 1);
        assert_eq!(sys.stats.active_conveyors, 1);
        assert!(sys.get(id).is_some());
    }

    #[test]
    fn test_curved_conveyor() {
        let curved = CurvedConveyor::new(
            ConveyorId(0),
            Vec3::ZERO,
            5.0,
            10.0,
            0.0,
            std::f32::consts::FRAC_PI_2,
            3.0,
        );
        assert!(curved.contains_point(Vec3::new(7.0, 0.0, 0.5)));
        assert!(!curved.contains_point(Vec3::new(100.0, 0.0, 0.0)));
    }
}
