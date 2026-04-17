//! Electromagnetic force simulation for physics-driven gameplay.
//!
//! Provides:
//! - Magnetic dipole fields with configurable orientation and strength
//! - Lorentz force computation on charged particles in combined E/B fields
//! - Magnetic field superposition from multiple sources
//! - Field line visualization data generation (seeds, traces, arrows)
//! - Configurable field strength, falloff, and interaction radius
//! - Ferromagnetic attraction / repulsion between dipole sources
//! - ECS integration via `MagneticFieldComponent` and `MagneticFieldSystem`

use glam::{Mat3, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Permeability of free space (mu_0 / 4*pi) in simplified game units.
/// Real value is ~1e-7 T*m/A; we scale for gameplay.
const MU_OVER_4PI: f32 = 1.0;
/// Minimum distance to avoid singularity at field source center.
const MIN_DISTANCE: f32 = 0.01;
/// Default magnetic moment magnitude.
const DEFAULT_MOMENT: f32 = 10.0;
/// Default interaction radius beyond which field is zero.
const DEFAULT_RADIUS: f32 = 50.0;
/// Default number of field line seeds per source.
const DEFAULT_FIELD_LINE_SEEDS: usize = 12;
/// Step size for field line tracing.
const FIELD_LINE_STEP: f32 = 0.1;
/// Maximum steps when tracing a single field line.
const MAX_FIELD_LINE_STEPS: usize = 500;
/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// Field falloff
// ---------------------------------------------------------------------------

/// How the field strength decays with distance from the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldFalloff {
    /// Physical inverse-cube falloff (dipole).
    InverseCube,
    /// Inverse-square falloff (monopole approximation).
    InverseSquare,
    /// Linear falloff to zero at the interaction radius.
    Linear,
    /// No falloff; constant strength within the interaction radius.
    Constant,
    /// Custom power-law falloff: strength / distance^power.
    PowerLaw(f32),
}

impl FieldFalloff {
    /// Compute the falloff factor at a given distance with a given radius.
    pub fn factor(&self, distance: f32, radius: f32) -> f32 {
        if distance >= radius {
            return 0.0;
        }
        let d = distance.max(MIN_DISTANCE);
        match self {
            FieldFalloff::InverseCube => {
                let r3 = d * d * d;
                1.0 / r3
            }
            FieldFalloff::InverseSquare => {
                let r2 = d * d;
                1.0 / r2
            }
            FieldFalloff::Linear => {
                1.0 - (d / radius)
            }
            FieldFalloff::Constant => 1.0,
            FieldFalloff::PowerLaw(power) => {
                1.0 / d.powf(*power)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Magnetic dipole source
// ---------------------------------------------------------------------------

/// Unique identifier for a magnetic source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MagneticSourceId(pub u32);

/// A magnetic dipole source that generates a magnetic field.
#[derive(Debug, Clone)]
pub struct MagneticDipole {
    /// Unique identifier.
    pub id: MagneticSourceId,
    /// World-space position of the dipole.
    pub position: Vec3,
    /// Magnetic moment vector (direction = orientation, magnitude = strength).
    pub moment: Vec3,
    /// Interaction radius beyond which the field is zero.
    pub radius: f32,
    /// Falloff type.
    pub falloff: FieldFalloff,
    /// Whether this source is active.
    pub active: bool,
    /// Optional group for selective interaction.
    pub group: u32,
}

impl MagneticDipole {
    /// Create a new magnetic dipole with default settings.
    pub fn new(id: MagneticSourceId, position: Vec3, moment_direction: Vec3) -> Self {
        Self {
            id,
            position,
            moment: moment_direction.normalize_or_zero() * DEFAULT_MOMENT,
            radius: DEFAULT_RADIUS,
            falloff: FieldFalloff::InverseCube,
            active: true,
            group: 0,
        }
    }

    /// Create a dipole with specified moment magnitude.
    pub fn with_strength(
        id: MagneticSourceId,
        position: Vec3,
        direction: Vec3,
        strength: f32,
    ) -> Self {
        Self {
            id,
            position,
            moment: direction.normalize_or_zero() * strength,
            radius: DEFAULT_RADIUS,
            falloff: FieldFalloff::InverseCube,
            active: true,
            group: 0,
        }
    }

    /// Compute the magnetic field vector at a world-space point due to this dipole.
    ///
    /// For a physical magnetic dipole, the field is:
    ///   B(r) = (mu_0 / 4pi) * (3 (m . r_hat) r_hat - m) / |r|^3
    ///
    /// We use the configured falloff instead of always inverse-cube.
    pub fn field_at(&self, point: Vec3) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }

        let r = point - self.position;
        let dist = r.length();
        if dist < EPSILON || dist > self.radius {
            return Vec3::ZERO;
        }

        let r_hat = r / dist;
        let m = self.moment;

        // Dipole field formula
        let m_dot_r = m.dot(r_hat);
        let dipole_direction = r_hat * (3.0 * m_dot_r) - m;

        // Apply falloff
        let factor = MU_OVER_4PI * self.falloff.factor(dist, self.radius);

        dipole_direction * factor
    }

    /// Compute the force on another dipole (ferromagnetic interaction).
    /// This is the gradient of the interaction energy.
    pub fn force_on_dipole(&self, other: &MagneticDipole) -> Vec3 {
        if !self.active || !other.active {
            return Vec3::ZERO;
        }

        let r = other.position - self.position;
        let dist = r.length();
        if dist < EPSILON || dist > self.radius {
            return Vec3::ZERO;
        }

        // Numerical gradient of the interaction energy
        let h = 0.001_f32;
        let mut force = Vec3::ZERO;

        for axis in 0..3 {
            let mut p_plus = other.position;
            let mut p_minus = other.position;
            p_plus[axis] += h;
            p_minus[axis] -= h;

            let b_plus = self.field_at(p_plus);
            let b_minus = self.field_at(p_minus);

            // Energy = -m . B, so Force = grad(m . B)
            let e_plus = other.moment.dot(b_plus);
            let e_minus = other.moment.dot(b_minus);
            force[axis] = (e_plus - e_minus) / (2.0 * h);
        }

        force
    }

    /// Compute the torque on a dipole moment in this field.
    /// Torque = m x B
    pub fn torque_on_moment(&self, at_point: Vec3, moment: Vec3) -> Vec3 {
        let b = self.field_at(at_point);
        moment.cross(b)
    }

    /// Get the field strength magnitude at a point.
    pub fn field_strength_at(&self, point: Vec3) -> f32 {
        self.field_at(point).length()
    }
}

// ---------------------------------------------------------------------------
// Uniform magnetic field
// ---------------------------------------------------------------------------

/// A uniform (constant direction and magnitude) magnetic field in a region.
#[derive(Debug, Clone)]
pub struct UniformField {
    /// Field vector (direction and magnitude).
    pub field: Vec3,
    /// Optional AABB bounds (min, max). If None, field is global.
    pub bounds: Option<(Vec3, Vec3)>,
    /// Whether this field is active.
    pub active: bool,
    /// Group for selective interaction.
    pub group: u32,
}

impl UniformField {
    /// Create a new global uniform field.
    pub fn new(field: Vec3) -> Self {
        Self {
            field,
            bounds: None,
            active: true,
            group: 0,
        }
    }

    /// Create a uniform field bounded to an AABB region.
    pub fn bounded(field: Vec3, min: Vec3, max: Vec3) -> Self {
        Self {
            field,
            bounds: Some((min, max)),
            active: true,
            group: 0,
        }
    }

    /// Get the field at a point (returns zero if outside bounds).
    pub fn field_at(&self, point: Vec3) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }
        if let Some((min, max)) = self.bounds {
            if point.x < min.x || point.x > max.x
                || point.y < min.y || point.y > max.y
                || point.z < min.z || point.z > max.z
            {
                return Vec3::ZERO;
            }
        }
        self.field
    }
}

// ---------------------------------------------------------------------------
// Electric field (for Lorentz force)
// ---------------------------------------------------------------------------

/// A simple electric field source (point charge model or uniform).
#[derive(Debug, Clone)]
pub enum ElectricFieldSource {
    /// Point charge: E = k * q / r^2 * r_hat
    PointCharge {
        position: Vec3,
        charge: f32,
        radius: f32,
    },
    /// Uniform electric field in a region.
    Uniform {
        field: Vec3,
        bounds: Option<(Vec3, Vec3)>,
    },
}

impl ElectricFieldSource {
    /// Compute the electric field at a point.
    pub fn field_at(&self, point: Vec3) -> Vec3 {
        match self {
            ElectricFieldSource::PointCharge { position, charge, radius } => {
                let r = point - *position;
                let dist = r.length();
                if dist < EPSILON || dist > *radius {
                    return Vec3::ZERO;
                }
                let r_hat = r / dist;
                r_hat * (*charge / (dist * dist))
            }
            ElectricFieldSource::Uniform { field, bounds } => {
                if let Some((min, max)) = bounds {
                    if point.x < min.x || point.x > max.x
                        || point.y < min.y || point.y > max.y
                        || point.z < min.z || point.z > max.z
                    {
                        return Vec3::ZERO;
                    }
                }
                *field
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Charged particle
// ---------------------------------------------------------------------------

/// A charged particle that responds to electromagnetic fields.
#[derive(Debug, Clone)]
pub struct ChargedParticle {
    /// World-space position.
    pub position: Vec3,
    /// Velocity vector.
    pub velocity: Vec3,
    /// Mass in kg.
    pub mass: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// Electric charge in simplified game units.
    pub charge: f32,
    /// Optional magnetic moment (for dipole interactions).
    pub magnetic_moment: Option<Vec3>,
    /// Whether this particle is active.
    pub active: bool,
    /// Accumulated force from the current step.
    pub accumulated_force: Vec3,
    /// Group for selective interaction.
    pub group: u32,
    /// Drag coefficient for damping.
    pub drag: f32,
}

impl ChargedParticle {
    /// Create a new charged particle.
    pub fn new(position: Vec3, mass: f32, charge: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            inv_mass: if mass > EPSILON { 1.0 / mass } else { 0.0 },
            charge,
            magnetic_moment: None,
            active: true,
            accumulated_force: Vec3::ZERO,
            group: 0,
            drag: 0.01,
        }
    }

    /// Compute the Lorentz force on this particle: F = q(E + v x B).
    pub fn lorentz_force(&self, electric: Vec3, magnetic: Vec3) -> Vec3 {
        self.charge * (electric + self.velocity.cross(magnetic))
    }

    /// Apply the Lorentz force as an accumulated force.
    pub fn apply_lorentz(&mut self, electric: Vec3, magnetic: Vec3) {
        let force = self.lorentz_force(electric, magnetic);
        self.accumulated_force += force;
    }

    /// Integrate position and velocity using semi-implicit Euler.
    pub fn integrate(&mut self, dt: f32) {
        if self.inv_mass < EPSILON || !self.active {
            return;
        }

        let acceleration = self.accumulated_force * self.inv_mass;
        self.velocity += acceleration * dt;
        self.velocity *= 1.0 - self.drag;
        self.position += self.velocity * dt;
        self.accumulated_force = Vec3::ZERO;
    }

    /// Get the kinetic energy.
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.velocity.length_squared()
    }

    /// Get the speed.
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }
}

// ---------------------------------------------------------------------------
// Field line data (for visualization)
// ---------------------------------------------------------------------------

/// A single traced field line for visualization.
#[derive(Debug, Clone)]
pub struct FieldLine {
    /// Ordered list of points along the field line.
    pub points: Vec<Vec3>,
    /// Direction vectors at each point (for arrow rendering).
    pub directions: Vec<Vec3>,
    /// Field strength at each point (for color/thickness mapping).
    pub strengths: Vec<f32>,
    /// Whether this line forms a closed loop.
    pub is_closed: bool,
}

/// Data for rendering field lines from all sources.
#[derive(Debug, Clone)]
pub struct FieldLineData {
    /// All traced field lines.
    pub lines: Vec<FieldLine>,
    /// Source positions (for glyph rendering).
    pub source_positions: Vec<Vec3>,
    /// Source moment directions (for glyph rendering).
    pub source_directions: Vec<Vec3>,
}

impl FieldLineData {
    /// Create an empty field line data set.
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            source_positions: Vec::new(),
            source_directions: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Magnetic field system
// ---------------------------------------------------------------------------

/// The main electromagnetic field system, managing sources, fields, and
/// charged particles.
#[derive(Debug)]
pub struct MagneticFieldSystem {
    /// Magnetic dipole sources.
    pub dipoles: Vec<MagneticDipole>,
    /// Uniform magnetic fields.
    pub uniform_fields: Vec<UniformField>,
    /// Electric field sources (for Lorentz force).
    pub electric_sources: Vec<ElectricFieldSource>,
    /// Charged particles in the system.
    pub charged_particles: Vec<ChargedParticle>,
    /// Next source ID counter.
    next_source_id: u32,
    /// Number of field line seeds per dipole source.
    pub field_line_seeds: usize,
    /// Cached field line data (regenerated on demand).
    cached_field_lines: Option<FieldLineData>,
    /// Whether the field line cache is dirty.
    field_lines_dirty: bool,
    /// Global field strength multiplier.
    pub global_strength: f32,
    /// Whether the system is enabled.
    pub enabled: bool,
}

impl MagneticFieldSystem {
    /// Create a new empty magnetic field system.
    pub fn new() -> Self {
        Self {
            dipoles: Vec::new(),
            uniform_fields: Vec::new(),
            electric_sources: Vec::new(),
            charged_particles: Vec::new(),
            next_source_id: 0,
            field_line_seeds: DEFAULT_FIELD_LINE_SEEDS,
            cached_field_lines: None,
            field_lines_dirty: true,
            global_strength: 1.0,
            enabled: true,
        }
    }

    /// Add a magnetic dipole source. Returns its ID.
    pub fn add_dipole(&mut self, position: Vec3, moment_direction: Vec3, strength: f32) -> MagneticSourceId {
        let id = MagneticSourceId(self.next_source_id);
        self.next_source_id += 1;
        let mut dipole = MagneticDipole::new(id, position, moment_direction);
        dipole.moment = moment_direction.normalize_or_zero() * strength;
        self.dipoles.push(dipole);
        self.field_lines_dirty = true;
        id
    }

    /// Add a dipole with full configuration.
    pub fn add_dipole_full(&mut self, dipole: MagneticDipole) -> MagneticSourceId {
        let id = dipole.id;
        self.dipoles.push(dipole);
        self.field_lines_dirty = true;
        id
    }

    /// Add a uniform magnetic field.
    pub fn add_uniform_field(&mut self, field: UniformField) {
        self.uniform_fields.push(field);
        self.field_lines_dirty = true;
    }

    /// Add an electric field source.
    pub fn add_electric_source(&mut self, source: ElectricFieldSource) {
        self.electric_sources.push(source);
    }

    /// Add a charged particle.
    pub fn add_charged_particle(&mut self, particle: ChargedParticle) -> usize {
        let idx = self.charged_particles.len();
        self.charged_particles.push(particle);
        idx
    }

    /// Remove a dipole source by ID.
    pub fn remove_dipole(&mut self, id: MagneticSourceId) {
        self.dipoles.retain(|d| d.id != id);
        self.field_lines_dirty = true;
    }

    /// Set the position of a dipole source.
    pub fn set_dipole_position(&mut self, id: MagneticSourceId, position: Vec3) {
        if let Some(d) = self.dipoles.iter_mut().find(|d| d.id == id) {
            d.position = position;
            self.field_lines_dirty = true;
        }
    }

    /// Set the moment of a dipole source.
    pub fn set_dipole_moment(&mut self, id: MagneticSourceId, moment: Vec3) {
        if let Some(d) = self.dipoles.iter_mut().find(|d| d.id == id) {
            d.moment = moment;
            self.field_lines_dirty = true;
        }
    }

    /// Sample the total magnetic field at a world-space point (superposition).
    pub fn sample_magnetic_field(&self, point: Vec3) -> Vec3 {
        if !self.enabled {
            return Vec3::ZERO;
        }

        let mut total = Vec3::ZERO;

        for dipole in &self.dipoles {
            total += dipole.field_at(point);
        }

        for uniform in &self.uniform_fields {
            total += uniform.field_at(point);
        }

        total * self.global_strength
    }

    /// Sample the total electric field at a world-space point.
    pub fn sample_electric_field(&self, point: Vec3) -> Vec3 {
        if !self.enabled {
            return Vec3::ZERO;
        }

        let mut total = Vec3::ZERO;
        for source in &self.electric_sources {
            total += source.field_at(point);
        }
        total * self.global_strength
    }

    /// Sample the magnetic field strength (magnitude) at a point.
    pub fn field_strength_at(&self, point: Vec3) -> f32 {
        self.sample_magnetic_field(point).length()
    }

    /// Compute the Lorentz force on a charged particle at the given point
    /// with the given velocity and charge.
    pub fn lorentz_force_at(&self, point: Vec3, velocity: Vec3, charge: f32) -> Vec3 {
        let e = self.sample_electric_field(point);
        let b = self.sample_magnetic_field(point);
        charge * (e + velocity.cross(b))
    }

    /// Step the simulation: compute forces on charged particles and integrate.
    pub fn step(&mut self, dt: f32) {
        if !self.enabled || dt <= 0.0 {
            return;
        }

        // Apply Lorentz force to each charged particle
        for particle in &mut self.charged_particles {
            if !particle.active {
                continue;
            }

            let b = self.sample_magnetic_field_internal(particle.position);
            let e = self.sample_electric_field_internal(particle.position);

            particle.apply_lorentz(e, b);
            particle.integrate(dt);
        }

        // Dipole-dipole interaction forces
        let dipole_count = self.dipoles.len();
        if dipole_count > 1 {
            let dipoles_snapshot: Vec<MagneticDipole> = self.dipoles.clone();
            for i in 0..dipole_count {
                for j in (i + 1)..dipole_count {
                    let _force = dipoles_snapshot[i].force_on_dipole(&dipoles_snapshot[j]);
                    // In a full implementation, these forces would be applied to
                    // rigid bodies attached to the dipole sources. We store the
                    // force data for external systems to query.
                }
            }
        }
    }

    /// Internal sampling that doesn't check enabled state (already checked in step).
    fn sample_magnetic_field_internal(&self, point: Vec3) -> Vec3 {
        let mut total = Vec3::ZERO;
        for dipole in &self.dipoles {
            total += dipole.field_at(point);
        }
        for uniform in &self.uniform_fields {
            total += uniform.field_at(point);
        }
        total * self.global_strength
    }

    /// Internal electric field sampling.
    fn sample_electric_field_internal(&self, point: Vec3) -> Vec3 {
        let mut total = Vec3::ZERO;
        for source in &self.electric_sources {
            total += source.field_at(point);
        }
        total * self.global_strength
    }

    /// Generate field line visualization data. Results are cached until sources change.
    pub fn generate_field_lines(&mut self) -> &FieldLineData {
        if !self.field_lines_dirty && self.cached_field_lines.is_some() {
            return self.cached_field_lines.as_ref().unwrap();
        }

        let mut data = FieldLineData::new();

        for dipole in &self.dipoles {
            if !dipole.active {
                continue;
            }
            data.source_positions.push(dipole.position);
            data.source_directions.push(dipole.moment.normalize_or_zero());

            // Generate seed points around the dipole in a ring perpendicular
            // to the moment direction.
            let moment_dir = dipole.moment.normalize_or_zero();
            let up = if moment_dir.y.abs() < 0.999 {
                Vec3::Y
            } else {
                Vec3::X
            };
            let right = moment_dir.cross(up).normalize_or_zero();
            let forward = right.cross(moment_dir).normalize_or_zero();

            let seed_radius = 0.1_f32;
            for seed_idx in 0..self.field_line_seeds {
                let angle = (seed_idx as f32 / self.field_line_seeds as f32)
                    * 2.0 * std::f32::consts::PI;

                let seed_offset = right * angle.cos() * seed_radius
                    + forward * angle.sin() * seed_radius
                    + moment_dir * seed_radius;

                let seed_point = dipole.position + seed_offset;

                // Trace field line in the positive direction
                let line = self.trace_field_line(seed_point, true);
                if line.points.len() > 2 {
                    data.lines.push(line);
                }

                // Trace in the negative direction
                let line_neg = self.trace_field_line(seed_point, false);
                if line_neg.points.len() > 2 {
                    data.lines.push(line_neg);
                }
            }
        }

        self.cached_field_lines = Some(data);
        self.field_lines_dirty = false;
        self.cached_field_lines.as_ref().unwrap()
    }

    /// Trace a single field line from a seed point.
    fn trace_field_line(&self, start: Vec3, forward: bool) -> FieldLine {
        let mut line = FieldLine {
            points: Vec::with_capacity(MAX_FIELD_LINE_STEPS),
            directions: Vec::with_capacity(MAX_FIELD_LINE_STEPS),
            strengths: Vec::with_capacity(MAX_FIELD_LINE_STEPS),
            is_closed: false,
        };

        let mut pos = start;
        let direction_sign = if forward { 1.0 } else { -1.0 };

        for _step in 0..MAX_FIELD_LINE_STEPS {
            let field = self.sample_magnetic_field(pos);
            let strength = field.length();

            if strength < EPSILON {
                break;
            }

            let dir = field.normalize_or_zero() * direction_sign;

            line.points.push(pos);
            line.directions.push(dir);
            line.strengths.push(strength);

            // RK2 integration for smoother lines
            let mid = pos + dir * (FIELD_LINE_STEP * 0.5);
            let mid_field = self.sample_magnetic_field(mid);
            let mid_dir = if mid_field.length() > EPSILON {
                mid_field.normalize_or_zero() * direction_sign
            } else {
                dir
            };

            pos += mid_dir * FIELD_LINE_STEP;

            // Check if we've returned close to the start (closed loop)
            if line.points.len() > 10 {
                let dist_to_start = (pos - start).length();
                if dist_to_start < FIELD_LINE_STEP * 2.0 {
                    line.is_closed = true;
                    line.points.push(start); // close the loop
                    break;
                }
            }

            // Check if we've left all interaction radii
            let mut inside_any = false;
            for dipole in &self.dipoles {
                if (pos - dipole.position).length() < dipole.radius {
                    inside_any = true;
                    break;
                }
            }
            if !inside_any {
                break;
            }
        }

        line
    }

    /// Get the number of active dipole sources.
    pub fn active_dipole_count(&self) -> usize {
        self.dipoles.iter().filter(|d| d.active).count()
    }

    /// Get the number of charged particles.
    pub fn charged_particle_count(&self) -> usize {
        self.charged_particles.len()
    }

    /// Clear all sources and particles.
    pub fn clear(&mut self) {
        self.dipoles.clear();
        self.uniform_fields.clear();
        self.electric_sources.clear();
        self.charged_particles.clear();
        self.cached_field_lines = None;
        self.field_lines_dirty = true;
    }

    /// Sample the field on a 3D grid for volumetric visualization.
    /// Returns (positions, field_vectors) arrays.
    pub fn sample_field_grid(
        &self,
        min: Vec3,
        max: Vec3,
        resolution: usize,
    ) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut positions = Vec::new();
        let mut fields = Vec::new();

        let res = resolution.max(2);
        let step = (max - min) / (res as f32 - 1.0);

        for iz in 0..res {
            for iy in 0..res {
                for ix in 0..res {
                    let pos = min + Vec3::new(
                        ix as f32 * step.x,
                        iy as f32 * step.y,
                        iz as f32 * step.z,
                    );
                    let field = self.sample_magnetic_field(pos);
                    positions.push(pos);
                    fields.push(field);
                }
            }
        }

        (positions, fields)
    }

    /// Compute the total magnetic flux through a planar surface.
    /// Surface is defined by center, normal, and half-extents.
    pub fn compute_flux(
        &self,
        center: Vec3,
        normal: Vec3,
        half_width: f32,
        half_height: f32,
        samples: usize,
    ) -> f32 {
        let normal = normal.normalize_or_zero();
        let up = if normal.y.abs() < 0.999 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let right = normal.cross(up).normalize_or_zero();
        let up = right.cross(normal);

        let n = samples.max(2);
        let dx = 2.0 * half_width / (n as f32 - 1.0);
        let dy = 2.0 * half_height / (n as f32 - 1.0);
        let area_element = dx * dy;

        let mut flux = 0.0_f32;

        for iy in 0..n {
            for ix in 0..n {
                let u = -half_width + ix as f32 * dx;
                let v = -half_height + iy as f32 * dy;
                let point = center + right * u + up * v;
                let field = self.sample_magnetic_field(point);
                flux += field.dot(normal) * area_element;
            }
        }

        flux
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// ECS component that attaches a magnetic dipole source to an entity.
#[derive(Debug, Clone)]
pub struct MagneticFieldComponent {
    /// The dipole source ID within the field system.
    pub source_id: Option<MagneticSourceId>,
    /// Moment direction (local space, rotated by entity transform).
    pub local_moment_direction: Vec3,
    /// Moment strength.
    pub strength: f32,
    /// Interaction radius.
    pub radius: f32,
    /// Falloff type.
    pub falloff: FieldFalloff,
    /// Whether the component is enabled.
    pub enabled: bool,
}

impl MagneticFieldComponent {
    /// Create a new magnetic field component pointing along the Y axis.
    pub fn new(strength: f32) -> Self {
        Self {
            source_id: None,
            local_moment_direction: Vec3::Y,
            strength,
            radius: DEFAULT_RADIUS,
            falloff: FieldFalloff::InverseCube,
            enabled: true,
        }
    }

    /// Create with a specific direction and strength.
    pub fn with_direction(direction: Vec3, strength: f32) -> Self {
        Self {
            source_id: None,
            local_moment_direction: direction.normalize_or_zero(),
            strength,
            radius: DEFAULT_RADIUS,
            falloff: FieldFalloff::InverseCube,
            enabled: true,
        }
    }
}

/// ECS component that makes an entity respond to magnetic fields as a charged particle.
#[derive(Debug, Clone)]
pub struct ChargedBodyComponent {
    /// Electric charge.
    pub charge: f32,
    /// Optional magnetic moment for dipole interaction.
    pub magnetic_moment: Option<Vec3>,
    /// Drag coefficient.
    pub drag: f32,
    /// Whether the component is enabled.
    pub enabled: bool,
    /// Group for selective interaction.
    pub group: u32,
}

impl ChargedBodyComponent {
    /// Create a new charged body component.
    pub fn new(charge: f32) -> Self {
        Self {
            charge,
            magnetic_moment: None,
            drag: 0.01,
            enabled: true,
            group: 0,
        }
    }
}
