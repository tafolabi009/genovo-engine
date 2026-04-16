// engine/render/src/particles/emitter.rs
//
// Particle emitter configuration and spawn logic. An emitter defines *where*,
// *when*, and *how* particles are born. The actual particle storage lives in
// [`ParticlePool`]; the emitter only produces spawn descriptors that the pool
// consumes.

use glam::{Mat4, Quat, Vec3};
use super::{ColorGradient, Curve, Rng};

// ---------------------------------------------------------------------------
// SimulationSpace
// ---------------------------------------------------------------------------

/// Determines whether particles are simulated in world space or relative to
/// the emitter's local transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulationSpace {
    /// Particles are simulated in world space. Moving the emitter after
    /// emission does not affect already-spawned particles.
    World,
    /// Particles are simulated relative to the emitter. Moving the emitter
    /// moves all its particles.
    Local,
}

impl Default for SimulationSpace {
    fn default() -> Self {
        Self::World
    }
}

// ---------------------------------------------------------------------------
// EmitterShape
// ---------------------------------------------------------------------------

/// The geometric shape from which particles are emitted.
///
/// Each variant carries its own configuration and implements a `sample`
/// method that returns a position and direction relative to the emitter's
/// local coordinate system.
#[derive(Debug, Clone)]
pub enum EmitterShape {
    /// Emit from the surface or volume of a sphere.
    Sphere {
        /// Radius of the sphere.
        radius: f32,
        /// If `true`, emit from the entire volume; otherwise surface only.
        emit_from_volume: bool,
    },
    /// Emit from a hemisphere pointing along the emitter's forward (+Y).
    Hemisphere {
        radius: f32,
        emit_from_volume: bool,
    },
    /// Emit from a cone.
    Cone {
        /// Half-angle of the cone in radians.
        angle: f32,
        /// Radius at the base of the cone.
        radius: f32,
        /// Length of the cone along the forward axis.
        length: f32,
        /// If `true`, emit from the volume; otherwise from the base circle.
        emit_from_volume: bool,
    },
    /// Emit from an axis-aligned box.
    Box {
        /// Half-extents of the box.
        half_extents: Vec3,
    },
    /// Emit from a circle (disc) in the XZ plane.
    Circle {
        /// Radius of the circle.
        radius: f32,
        /// If `true`, emit from the entire disc area; otherwise circumference.
        emit_from_area: bool,
    },
    /// Emit along a line segment.
    Edge {
        /// Start point of the segment (local space).
        start: Vec3,
        /// End point of the segment (local space).
        end: Vec3,
    },
    /// Emit from the surface of a mesh.
    Mesh {
        /// Vertex positions (local space).
        vertices: Vec<Vec3>,
        /// Per-vertex normals.
        normals: Vec<Vec3>,
        /// Triangle indices (triplets).
        indices: Vec<u32>,
    },
    /// Emit from a single point (the emitter origin).
    Point,
}

impl EmitterShape {
    /// Samples a local-space spawn position and direction from the shape.
    ///
    /// The direction is a unit vector indicating the initial velocity
    /// direction of the particle. For most shapes this is the outward
    /// normal at the sample point.
    pub fn sample(&self, rng: &mut Rng) -> (Vec3, Vec3) {
        match self {
            EmitterShape::Sphere {
                radius,
                emit_from_volume,
            } => {
                if *emit_from_volume {
                    let p = rng.inside_unit_sphere() * *radius;
                    let dir = if p.length_squared() > 1e-9 {
                        p.normalize()
                    } else {
                        Vec3::Y
                    };
                    (p, dir)
                } else {
                    let dir = rng.unit_sphere();
                    (dir * *radius, dir)
                }
            }

            EmitterShape::Hemisphere {
                radius,
                emit_from_volume,
            } => {
                if *emit_from_volume {
                    let mut p = rng.inside_unit_sphere() * *radius;
                    // Force positive Y (hemisphere pointing up).
                    p.y = p.y.abs();
                    let dir = if p.length_squared() > 1e-9 {
                        p.normalize()
                    } else {
                        Vec3::Y
                    };
                    (p, dir)
                } else {
                    let mut dir = rng.unit_sphere();
                    dir.y = dir.y.abs();
                    let dir = dir.normalize();
                    (dir * *radius, dir)
                }
            }

            EmitterShape::Cone {
                angle,
                radius,
                length,
                emit_from_volume,
            } => {
                // The cone axis is +Y. Generate a random direction within
                // the half-angle cone.
                let cos_angle = angle.cos();
                // Uniform distribution in a cone: z in [cos(angle), 1]
                let z = rng.range_f32(cos_angle, 1.0);
                let phi = rng.range_f32(0.0, std::f32::consts::TAU);
                let sin_theta = (1.0 - z * z).sqrt();
                let dir = Vec3::new(sin_theta * phi.cos(), z, sin_theta * phi.sin());

                if *emit_from_volume {
                    let t = rng.next_f32();
                    let r = rng.next_f32().sqrt() * *radius * t;
                    let base_phi = rng.range_f32(0.0, std::f32::consts::TAU);
                    let pos = Vec3::new(
                        r * base_phi.cos(),
                        t * *length,
                        r * base_phi.sin(),
                    );
                    (pos, dir.normalize())
                } else {
                    let r = rng.next_f32().sqrt() * *radius;
                    let base_phi = rng.range_f32(0.0, std::f32::consts::TAU);
                    let pos = Vec3::new(r * base_phi.cos(), 0.0, r * base_phi.sin());
                    (pos, dir.normalize())
                }
            }

            EmitterShape::Box { half_extents } => {
                let pos = Vec3::new(
                    rng.next_f32_signed() * half_extents.x,
                    rng.next_f32_signed() * half_extents.y,
                    rng.next_f32_signed() * half_extents.z,
                );
                (pos, Vec3::Y)
            }

            EmitterShape::Circle {
                radius,
                emit_from_area,
            } => {
                if *emit_from_area {
                    let p = rng.inside_unit_circle() * *radius;
                    (Vec3::new(p.x, 0.0, p.y), Vec3::Y)
                } else {
                    let angle = rng.range_f32(0.0, std::f32::consts::TAU);
                    let pos = Vec3::new(angle.cos() * *radius, 0.0, angle.sin() * *radius);
                    let dir = Vec3::new(pos.x, 0.0, pos.z).normalize();
                    (pos, dir)
                }
            }

            EmitterShape::Edge { start, end } => {
                let t = rng.next_f32();
                let pos = *start + (*end - *start) * t;
                let edge_dir = (*end - *start).normalize();
                // Perpendicular direction: cross with Y or Z.
                let up = if edge_dir.y.abs() < 0.99 {
                    Vec3::Y
                } else {
                    Vec3::Z
                };
                let normal = edge_dir.cross(up).normalize();
                (pos, normal)
            }

            EmitterShape::Mesh {
                vertices,
                normals,
                indices,
            } => {
                if indices.len() < 3 || vertices.is_empty() {
                    return (Vec3::ZERO, Vec3::Y);
                }
                // Pick a random triangle (uniform by index, not area-weighted
                // for simplicity; area-weighted would be better for production).
                let tri_count = indices.len() / 3;
                let tri_idx = (rng.next_u64() as usize) % tri_count;
                let i0 = indices[tri_idx * 3] as usize;
                let i1 = indices[tri_idx * 3 + 1] as usize;
                let i2 = indices[tri_idx * 3 + 2] as usize;

                // Clamp to valid range.
                let v0 = vertices[i0.min(vertices.len() - 1)];
                let v1 = vertices[i1.min(vertices.len() - 1)];
                let v2 = vertices[i2.min(vertices.len() - 1)];

                // Random barycentric coordinates.
                let mut u = rng.next_f32();
                let mut v = rng.next_f32();
                if u + v > 1.0 {
                    u = 1.0 - u;
                    v = 1.0 - v;
                }
                let w = 1.0 - u - v;

                let pos = v0 * w + v1 * u + v2 * v;

                let normal = if !normals.is_empty() {
                    let n0 = normals[i0.min(normals.len() - 1)];
                    let n1 = normals[i1.min(normals.len() - 1)];
                    let n2 = normals[i2.min(normals.len() - 1)];
                    (n0 * w + n1 * u + n2 * v).normalize()
                } else {
                    let e1 = v1 - v0;
                    let e2 = v2 - v0;
                    e1.cross(e2).normalize()
                };

                (pos, normal)
            }

            EmitterShape::Point => (Vec3::ZERO, Vec3::Y),
        }
    }
}

impl Default for EmitterShape {
    fn default() -> Self {
        EmitterShape::Cone {
            angle: std::f32::consts::FRAC_PI_4,
            radius: 0.0,
            length: 1.0,
            emit_from_volume: false,
        }
    }
}

// ---------------------------------------------------------------------------
// EmissionMode
// ---------------------------------------------------------------------------

/// Controls how particles are spawned over time.
#[derive(Debug, Clone)]
pub enum EmissionMode {
    /// Spawn particles at a constant rate (particles per second).
    Continuous {
        /// Particles per second.
        rate: f32,
    },
    /// Spawn bursts of particles at regular intervals.
    Burst {
        /// Number of particles per burst.
        count: u32,
        /// Time between bursts in seconds.
        interval: f32,
        /// Number of burst cycles. 0 means infinite.
        cycles: u32,
    },
    /// Spawn particles based on distance travelled by the emitter.
    Distance {
        /// Particles per unit of distance.
        per_unit: f32,
    },
}

impl Default for EmissionMode {
    fn default() -> Self {
        EmissionMode::Continuous { rate: 10.0 }
    }
}

// ---------------------------------------------------------------------------
// SpawnParams
// ---------------------------------------------------------------------------

/// Parameters describing a single newly spawned particle.
///
/// The emitter produces these; the particle pool consumes them.
#[derive(Debug, Clone, Copy)]
pub struct SpawnParams {
    /// World-space (or local-space) position.
    pub position: Vec3,
    /// Initial velocity.
    pub velocity: Vec3,
    /// Initial color (RGBA).
    pub color: [f32; 4],
    /// Initial size.
    pub size: f32,
    /// Total lifetime in seconds.
    pub lifetime: f32,
    /// Initial rotation in radians.
    pub rotation: f32,
    /// Angular velocity in radians/second.
    pub angular_velocity: f32,
    /// Custom data channel (user-defined).
    pub custom_data: [f32; 4],
}

// ---------------------------------------------------------------------------
// ParticleEmitter
// ---------------------------------------------------------------------------

/// A particle emitter that spawns new particles according to configurable
/// shape, rate, velocity, and appearance parameters.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    // -- Spatial --
    /// World-space position of the emitter.
    pub position: Vec3,
    /// Emitter rotation (orients the emission shape).
    pub rotation: Quat,
    /// Scale applied to the emission shape.
    pub scale: Vec3,

    // -- Emission --
    /// Shape from which particles are spawned.
    pub shape: EmitterShape,
    /// How particles are spawned over time.
    pub emission_mode: EmissionMode,
    /// Maximum number of particles alive at once from this emitter.
    pub max_particles: u32,

    // -- Initial particle properties --
    /// Minimum initial speed (along the shape's outward direction).
    pub speed_min: f32,
    /// Maximum initial speed.
    pub speed_max: f32,
    /// Minimum lifetime in seconds.
    pub lifetime_min: f32,
    /// Maximum lifetime in seconds.
    pub lifetime_max: f32,
    /// Initial size at birth.
    pub start_size_min: f32,
    /// Maximum initial size at birth.
    pub start_size_max: f32,
    /// Size at death (for size-over-lifetime interpolation).
    pub end_size: f32,
    /// Start color.
    pub start_color: [f32; 4],
    /// End color (for color-over-lifetime interpolation).
    pub end_color: [f32; 4],
    /// Gravity multiplier (1.0 = full gravity).
    pub gravity_modifier: f32,
    /// Simulation space.
    pub simulation_space: SimulationSpace,
    /// Size over lifetime curve.
    pub size_over_lifetime: Curve,
    /// Color over lifetime gradient.
    pub color_over_lifetime: ColorGradient,
    /// Speed over lifetime multiplier curve.
    pub speed_over_lifetime: Curve,
    /// Minimum initial rotation in radians.
    pub rotation_min: f32,
    /// Maximum initial rotation in radians.
    pub rotation_max: f32,
    /// Minimum angular velocity in radians/second.
    pub angular_velocity_min: f32,
    /// Maximum angular velocity in radians/second.
    pub angular_velocity_max: f32,
    /// Additional initial velocity applied in world space.
    pub initial_velocity_offset: Vec3,
    /// Inherit velocity from emitter movement (0..1).
    pub inherit_velocity: f32,

    // -- Internal state --
    /// Accumulated fractional particles from continuous emission.
    spawn_accumulator: f32,
    /// Time since last burst.
    burst_timer: f32,
    /// Number of bursts completed.
    burst_count: u32,
    /// Previous emitter position (for distance-based emission).
    prev_position: Vec3,
    /// Previous emitter velocity (for velocity inheritance).
    emitter_velocity: Vec3,
    /// RNG for deterministic spawning.
    rng: Rng,
    /// Total elapsed time since emitter started.
    elapsed: f32,
}

impl ParticleEmitter {
    /// Creates a new emitter with default settings at the origin.
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            shape: EmitterShape::default(),
            emission_mode: EmissionMode::default(),
            max_particles: 1000,
            speed_min: 1.0,
            speed_max: 3.0,
            lifetime_min: 1.0,
            lifetime_max: 3.0,
            start_size_min: 0.1,
            start_size_max: 0.2,
            end_size: 0.0,
            start_color: [1.0, 1.0, 1.0, 1.0],
            end_color: [1.0, 1.0, 1.0, 0.0],
            gravity_modifier: 0.0,
            simulation_space: SimulationSpace::World,
            size_over_lifetime: Curve::constant(1.0),
            color_over_lifetime: ColorGradient::default(),
            speed_over_lifetime: Curve::constant(1.0),
            rotation_min: 0.0,
            rotation_max: 0.0,
            angular_velocity_min: 0.0,
            angular_velocity_max: 0.0,
            initial_velocity_offset: Vec3::ZERO,
            inherit_velocity: 0.0,
            spawn_accumulator: 0.0,
            burst_timer: 0.0,
            burst_count: 0,
            prev_position: Vec3::ZERO,
            emitter_velocity: Vec3::ZERO,
            rng: Rng::default(),
            elapsed: 0.0,
        }
    }

    /// Creates a sphere emitter.
    pub fn sphere(radius: f32) -> Self {
        let mut e = Self::new();
        e.shape = EmitterShape::Sphere {
            radius,
            emit_from_volume: false,
        };
        e
    }

    /// Creates a cone emitter.
    pub fn cone(angle: f32, radius: f32) -> Self {
        let mut e = Self::new();
        e.shape = EmitterShape::Cone {
            angle,
            radius,
            length: 1.0,
            emit_from_volume: false,
        };
        e
    }

    /// Creates a box emitter.
    pub fn box_shape(half_extents: Vec3) -> Self {
        let mut e = Self::new();
        e.shape = EmitterShape::Box { half_extents };
        e
    }

    /// Sets the emission rate (particles per second) for continuous mode.
    pub fn with_rate(mut self, rate: f32) -> Self {
        self.emission_mode = EmissionMode::Continuous { rate };
        self
    }

    /// Sets burst emission mode.
    pub fn with_burst(mut self, count: u32, interval: f32, cycles: u32) -> Self {
        self.emission_mode = EmissionMode::Burst {
            count,
            interval,
            cycles,
        };
        self
    }

    /// Sets the speed range.
    pub fn with_speed(mut self, min: f32, max: f32) -> Self {
        self.speed_min = min;
        self.speed_max = max;
        self
    }

    /// Sets the lifetime range.
    pub fn with_lifetime(mut self, min: f32, max: f32) -> Self {
        self.lifetime_min = min;
        self.lifetime_max = max;
        self
    }

    /// Sets the start size range.
    pub fn with_size(mut self, min: f32, max: f32) -> Self {
        self.start_size_min = min;
        self.start_size_max = max;
        self
    }

    /// Sets the start and end colors.
    pub fn with_color(mut self, start: [f32; 4], end: [f32; 4]) -> Self {
        self.start_color = start;
        self.end_color = end;
        self.color_over_lifetime = ColorGradient::new(start, end);
        self
    }

    /// Sets the gravity modifier.
    pub fn with_gravity(mut self, modifier: f32) -> Self {
        self.gravity_modifier = modifier;
        self
    }

    /// Sets the simulation space.
    pub fn with_simulation_space(mut self, space: SimulationSpace) -> Self {
        self.simulation_space = space;
        self
    }

    /// Sets the maximum particle count.
    pub fn with_max_particles(mut self, max: u32) -> Self {
        self.max_particles = max;
        self
    }

    /// Seeds the emitter's RNG for deterministic replay.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Updates the emitter position and computes velocity for inheritance.
    pub fn set_position(&mut self, position: Vec3) {
        self.emitter_velocity = position - self.prev_position;
        self.prev_position = self.position;
        self.position = position;
    }

    /// Sets the emitter rotation.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
    }

    /// Returns the total elapsed time.
    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    /// Resets the emitter state.
    pub fn reset(&mut self) {
        self.spawn_accumulator = 0.0;
        self.burst_timer = 0.0;
        self.burst_count = 0;
        self.elapsed = 0.0;
        self.prev_position = self.position;
        self.emitter_velocity = Vec3::ZERO;
    }

    /// Determines how many particles to spawn this frame and returns their
    /// spawn parameters.
    ///
    /// # Arguments
    ///
    /// * `dt` - Frame delta time in seconds.
    /// * `alive_count` - Current number of alive particles.
    ///
    /// # Returns
    ///
    /// A vector of `SpawnParams`, one per particle to spawn.
    pub fn emit(&mut self, dt: f32, alive_count: usize) -> Vec<SpawnParams> {
        self.elapsed += dt;

        let available = (self.max_particles as usize).saturating_sub(alive_count);
        if available == 0 {
            // Advance internal timers even if we can't spawn.
            self.advance_timers(dt);
            return Vec::new();
        }

        let count = self.compute_spawn_count(dt);
        let count = count.min(available);

        let mut spawns = Vec::with_capacity(count);
        for _ in 0..count {
            spawns.push(self.spawn_one());
        }
        spawns
    }

    /// Advances internal emission timers without spawning.
    fn advance_timers(&mut self, dt: f32) {
        match &self.emission_mode {
            EmissionMode::Continuous { rate } => {
                self.spawn_accumulator += dt * rate;
                // Don't let accumulator grow unbounded when capped.
                self.spawn_accumulator = self.spawn_accumulator.min(*rate * 2.0);
            }
            EmissionMode::Burst { interval: _, .. } => {
                self.burst_timer += dt;
            }
            EmissionMode::Distance { .. } => {}
        }
    }

    /// Computes how many particles should be spawned this frame.
    fn compute_spawn_count(&mut self, dt: f32) -> usize {
        match &self.emission_mode {
            EmissionMode::Continuous { rate } => {
                self.spawn_accumulator += dt * rate;
                let count = self.spawn_accumulator as usize;
                self.spawn_accumulator -= count as f32;
                count
            }
            EmissionMode::Burst {
                count,
                interval,
                cycles,
            } => {
                let count_val = *count;
                let interval_val = *interval;
                let cycles_val = *cycles;

                self.burst_timer += dt;
                if self.burst_timer >= interval_val {
                    self.burst_timer -= interval_val;
                    if cycles_val == 0 || self.burst_count < cycles_val {
                        self.burst_count += 1;
                        count_val as usize
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
            EmissionMode::Distance { per_unit } => {
                let distance = (self.position - self.prev_position).length();
                let count = (distance * per_unit) as usize;
                count
            }
        }
    }

    /// Spawns a single particle with randomized properties.
    fn spawn_one(&mut self) -> SpawnParams {
        // Sample shape for position and direction.
        let (local_pos, local_dir) = self.shape.sample(&mut self.rng);

        // Transform to world space via emitter rotation.
        let world_offset = self.rotation * (local_pos * self.scale);
        let world_dir = self.rotation * local_dir;

        let position = match self.simulation_space {
            SimulationSpace::World => self.position + world_offset,
            SimulationSpace::Local => local_pos * self.scale,
        };

        // Randomize speed.
        let speed = self.rng.range_f32(self.speed_min, self.speed_max);
        let mut velocity = world_dir * speed + self.initial_velocity_offset;

        // Inherit emitter velocity.
        if self.inherit_velocity > 0.0 {
            velocity += self.emitter_velocity * self.inherit_velocity;
        }

        // Randomize lifetime.
        let lifetime = self.rng.range_f32(self.lifetime_min, self.lifetime_max);

        // Randomize size.
        let size = self.rng.range_f32(self.start_size_min, self.start_size_max);

        // Randomize rotation.
        let rotation = self.rng.range_f32(self.rotation_min, self.rotation_max);
        let angular_velocity = self.rng.range_f32(
            self.angular_velocity_min,
            self.angular_velocity_max,
        );

        SpawnParams {
            position,
            velocity,
            color: self.start_color,
            size,
            lifetime,
            rotation,
            angular_velocity,
            custom_data: [0.0; 4],
        }
    }

    /// Returns the emitter's transformation matrix.
    pub fn transform_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Returns the emitter's forward direction.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Returns the emitter's right direction.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Returns the emitter's up direction.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    /// Computes the bounding radius of the emitter shape.
    pub fn shape_radius(&self) -> f32 {
        match &self.shape {
            EmitterShape::Sphere { radius, .. } => *radius,
            EmitterShape::Hemisphere { radius, .. } => *radius,
            EmitterShape::Cone {
                radius, length, ..
            } => (*radius * *radius + *length * *length).sqrt(),
            EmitterShape::Box { half_extents } => half_extents.length(),
            EmitterShape::Circle { radius, .. } => *radius,
            EmitterShape::Edge { start, end } => {
                start.length().max(end.length())
            }
            EmitterShape::Mesh { vertices, .. } => {
                vertices
                    .iter()
                    .map(|v| v.length())
                    .fold(0.0f32, f32::max)
            }
            EmitterShape::Point => 0.0,
        }
    }

    /// Estimates the maximum bounding radius of the particle system
    /// (emitter shape + max particle travel distance).
    pub fn estimated_bounds_radius(&self) -> f32 {
        let max_travel = self.speed_max * self.lifetime_max;
        self.shape_radius() + max_travel
    }

    /// Creates a mesh emitter from triangle data.
    pub fn from_mesh(
        vertices: Vec<Vec3>,
        normals: Vec<Vec3>,
        indices: Vec<u32>,
    ) -> Self {
        let mut e = Self::new();
        e.shape = EmitterShape::Mesh {
            vertices,
            normals,
            indices,
        };
        e
    }

    /// Sets the angular velocity range.
    pub fn with_angular_velocity(mut self, min: f32, max: f32) -> Self {
        self.angular_velocity_min = min;
        self.angular_velocity_max = max;
        self
    }

    /// Sets the rotation range.
    pub fn with_rotation_range(mut self, min: f32, max: f32) -> Self {
        self.rotation_min = min;
        self.rotation_max = max;
        self
    }

    /// Sets velocity inheritance factor.
    pub fn with_inherit_velocity(mut self, factor: f32) -> Self {
        self.inherit_velocity = factor;
        self
    }

    /// Sets an additional initial velocity offset applied in world space.
    pub fn with_velocity_offset(mut self, offset: Vec3) -> Self {
        self.initial_velocity_offset = offset;
        self
    }

    /// Sets the size-over-lifetime curve.
    pub fn with_size_over_lifetime(mut self, curve: Curve) -> Self {
        self.size_over_lifetime = curve;
        self
    }

    /// Sets the color-over-lifetime gradient.
    pub fn with_color_over_lifetime(mut self, gradient: ColorGradient) -> Self {
        self.color_over_lifetime = gradient;
        self
    }

    /// Sets the speed-over-lifetime curve.
    pub fn with_speed_over_lifetime(mut self, curve: Curve) -> Self {
        self.speed_over_lifetime = curve;
        self
    }
}

impl Default for ParticleEmitter {
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
    fn sphere_emitter_samples_on_surface() {
        let shape = EmitterShape::Sphere {
            radius: 5.0,
            emit_from_volume: false,
        };
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let (pos, dir) = shape.sample(&mut rng);
            let r = pos.length();
            assert!(
                (r - 5.0).abs() < 0.01,
                "Surface sample should be at radius 5.0, got {r}"
            );
            assert!(
                (dir.length() - 1.0).abs() < 0.01,
                "Direction should be unit length"
            );
        }
    }

    #[test]
    fn continuous_emission_accumulates() {
        let mut emitter = ParticleEmitter::new().with_rate(100.0);
        // 1/60 s at 100 pps = ~1.67 particles per frame.
        let dt = 1.0 / 60.0;
        let mut total = 0;
        for _ in 0..60 {
            let spawns = emitter.emit(dt, total);
            total += spawns.len();
        }
        // Should spawn approximately 100 particles over 1 second.
        assert!(
            total >= 95 && total <= 105,
            "Expected ~100 particles, got {total}"
        );
    }

    #[test]
    fn burst_emission_respects_cycles() {
        let mut emitter = ParticleEmitter::new().with_burst(10, 0.5, 3);
        let mut total = 0;
        // Simulate 5 seconds at 60 fps.
        for _ in 0..300 {
            let spawns = emitter.emit(1.0 / 60.0, total);
            total += spawns.len();
        }
        // 3 bursts of 10 = 30 particles.
        assert_eq!(total, 30, "Expected 30 particles from 3 bursts, got {total}");
    }

    #[test]
    fn box_emitter_within_bounds() {
        let half = Vec3::new(1.0, 2.0, 3.0);
        let shape = EmitterShape::Box {
            half_extents: half,
        };
        let mut rng = Rng::new(123);
        for _ in 0..200 {
            let (pos, _) = shape.sample(&mut rng);
            assert!(pos.x.abs() <= half.x + 0.001);
            assert!(pos.y.abs() <= half.y + 0.001);
            assert!(pos.z.abs() <= half.z + 0.001);
        }
    }

    #[test]
    fn max_particles_cap() {
        let mut emitter = ParticleEmitter::new()
            .with_rate(10000.0)
            .with_max_particles(50);
        let spawns = emitter.emit(1.0, 40);
        // Only 10 slots available.
        assert!(
            spawns.len() <= 10,
            "Should not exceed available slots, got {}",
            spawns.len()
        );
    }
}
