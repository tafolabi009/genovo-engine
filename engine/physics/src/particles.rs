//! Position-based fluid dynamics (PBD) with viscosity and surface reconstruction.
//!
//! Provides:
//! - **XSPH viscosity**: smoothed velocity field for coherent fluid motion
//! - **Incompressibility constraint**: density-based position correction
//! - **Surface reconstruction**: screen-space curvature for rendering hints
//! - **Boundary particle handling**: solid boundary enforcement
//! - **Vorticity confinement**: re-inject lost rotational energy
//! - **Neighbor search**: spatial hash grid for efficient pair finding
//! - **Kernel functions**: Poly6, Spiky gradient, Viscosity Laplacian
//! - **ECS integration**: `FluidParticleComponent`, `FluidParticleSystem`
//!
//! # References
//!
//! - Macklin & Mueller, "Position Based Fluids" (2013)
//! - Muller et al., "Particle-based Fluid Simulation for Interactive Applications" (2003)

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default particle rest density (kg/m^3).
pub const REST_DENSITY: f32 = 1000.0;
/// Default smoothing radius (h).
pub const DEFAULT_SMOOTHING_RADIUS: f32 = 0.1;
/// Default particle mass.
pub const DEFAULT_PARTICLE_MASS: f32 = 0.02;
/// Gravity.
pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
/// XSPH viscosity coefficient.
pub const DEFAULT_XSPH_VISCOSITY: f32 = 0.01;
/// Relaxation parameter for incompressibility.
pub const DEFAULT_RELAXATION: f32 = 600.0;
/// Vorticity confinement coefficient.
pub const DEFAULT_VORTICITY_EPSILON: f32 = 0.01;
/// Maximum solver iterations per step.
pub const MAX_SOLVER_ITERATIONS: usize = 4;
/// Tensile instability correction coefficient.
pub const TENSILE_K: f32 = 0.1;
/// Tensile instability reference distance fraction.
pub const TENSILE_DELTA_Q: f32 = 0.2;
/// Tensile instability exponent.
pub const TENSILE_N: u32 = 4;
/// Maximum particles.
pub const MAX_PARTICLES_V2: usize = 65536;
/// Spatial hash cell size multiplier (relative to smoothing radius).
pub const CELL_SIZE_MULTIPLIER: f32 = 2.0;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;
/// Small epsilon.
const EPSILON: f32 = 1e-7;
/// Boundary particle repulsion coefficient.
pub const BOUNDARY_REPULSION: f32 = 5000.0;
/// Default time step for fluid simulation.
pub const DEFAULT_DT: f32 = 0.008;
/// Surface detection threshold (ratio of neighbor count to expected).
pub const SURFACE_THRESHOLD: f32 = 0.7;

// ---------------------------------------------------------------------------
// Kernel functions
// ---------------------------------------------------------------------------

/// Poly6 kernel value.
pub fn poly6(r_sq: f32, h: f32) -> f32 {
    let h_sq = h * h;
    if r_sq > h_sq {
        return 0.0;
    }
    let coeff = 315.0 / (64.0 * PI * h.powi(9));
    let diff = h_sq - r_sq;
    coeff * diff * diff * diff
}

/// Poly6 kernel gradient.
pub fn poly6_gradient(r: Vec3, r_len_sq: f32, h: f32) -> Vec3 {
    let h_sq = h * h;
    if r_len_sq > h_sq || r_len_sq < EPSILON {
        return Vec3::ZERO;
    }
    let coeff = -945.0 / (32.0 * PI * h.powi(9));
    let diff = h_sq - r_len_sq;
    r * coeff * diff * diff
}

/// Spiky kernel gradient (used for pressure).
pub fn spiky_gradient(r: Vec3, r_len: f32, h: f32) -> Vec3 {
    if r_len > h || r_len < EPSILON {
        return Vec3::ZERO;
    }
    let coeff = -45.0 / (PI * h.powi(6));
    let diff = h - r_len;
    (r / r_len) * coeff * diff * diff
}

/// Viscosity kernel Laplacian.
pub fn viscosity_laplacian(r_len: f32, h: f32) -> f32 {
    if r_len > h || r_len < EPSILON {
        return 0.0;
    }
    let coeff = 45.0 / (PI * h.powi(6));
    coeff * (h - r_len)
}

/// Cubic spline kernel (more stable than Poly6 for some applications).
pub fn cubic_spline(r_len: f32, h: f32) -> f32 {
    let q = r_len / h;
    let sigma = 8.0 / (PI * h * h * h);
    if q > 1.0 {
        0.0
    } else if q > 0.5 {
        let t = 1.0 - q;
        sigma * 2.0 * t * t * t
    } else {
        sigma * (6.0 * q * q * q - 6.0 * q * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// FluidParticle
// ---------------------------------------------------------------------------

/// A single fluid particle.
#[derive(Debug, Clone)]
pub struct FluidParticle {
    /// Position.
    pub position: Vec3,
    /// Predicted position (used during solver iterations).
    pub predicted_position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Mass.
    pub mass: f32,
    /// Density (computed each step).
    pub density: f32,
    /// Lambda (Lagrange multiplier for density constraint).
    pub lambda: f32,
    /// Position correction (delta_p).
    pub delta_p: Vec3,
    /// Vorticity (curl of velocity field).
    pub vorticity: Vec3,
    /// Whether this particle is on the fluid surface.
    pub is_surface: bool,
    /// Color / fluid type index (for multi-fluid).
    pub fluid_type: u32,
    /// Particle age (for effects).
    pub age: f32,
    /// Whether this particle is active.
    pub active: bool,
    /// Neighbor indices (filled each step).
    pub neighbors: Vec<u32>,
    /// Neighbor count.
    pub neighbor_count: u32,
}

impl FluidParticle {
    /// Create a new fluid particle.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            predicted_position: position,
            velocity: Vec3::ZERO,
            mass: DEFAULT_PARTICLE_MASS,
            density: REST_DENSITY,
            lambda: 0.0,
            delta_p: Vec3::ZERO,
            vorticity: Vec3::ZERO,
            is_surface: false,
            fluid_type: 0,
            age: 0.0,
            active: true,
            neighbors: Vec::new(),
            neighbor_count: 0,
        }
    }

    /// Create with velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }
}

// ---------------------------------------------------------------------------
// BoundaryParticle
// ---------------------------------------------------------------------------

/// A boundary (solid wall) particle.
#[derive(Debug, Clone)]
pub struct BoundaryParticle {
    /// Position (fixed).
    pub position: Vec3,
    /// Normal pointing into the fluid.
    pub normal: Vec3,
    /// Psi (volume contribution for boundary handling).
    pub psi: f32,
}

impl BoundaryParticle {
    /// Create a new boundary particle.
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self {
            position,
            normal: normal.normalize_or_zero(),
            psi: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// SpatialHashGrid
// ---------------------------------------------------------------------------

/// Spatial hash grid for efficient neighbor searches.
#[derive(Debug)]
pub struct SpatialHashGrid {
    /// Cell size.
    pub cell_size: f32,
    /// Inverse cell size.
    inv_cell_size: f32,
    /// Hash table: cell_hash -> list of particle indices.
    cells: HashMap<u64, Vec<u32>>,
    /// Table size for hashing.
    table_size: u64,
}

impl SpatialHashGrid {
    /// Create a new spatial hash grid.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            table_size: 65536,
        }
    }

    /// Clear the grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Hash a cell coordinate to a table index.
    fn hash_cell(&self, cx: i32, cy: i32, cz: i32) -> u64 {
        let h = (cx as u64).wrapping_mul(73856093)
            ^ (cy as u64).wrapping_mul(19349663)
            ^ (cz as u64).wrapping_mul(83492791);
        h % self.table_size
    }

    /// Get the cell coordinates for a position.
    fn cell_coords(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x * self.inv_cell_size).floor() as i32,
            (pos.y * self.inv_cell_size).floor() as i32,
            (pos.z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Insert a particle into the grid.
    pub fn insert(&mut self, index: u32, position: Vec3) {
        let (cx, cy, cz) = self.cell_coords(position);
        let hash = self.hash_cell(cx, cy, cz);
        self.cells.entry(hash).or_default().push(index);
    }

    /// Build the grid from a set of positions.
    pub fn build(&mut self, positions: &[Vec3]) {
        self.clear();
        for (i, pos) in positions.iter().enumerate() {
            self.insert(i as u32, *pos);
        }
    }

    /// Query neighbors within radius.
    pub fn query_neighbors(&self, position: Vec3, radius: f32) -> Vec<u32> {
        let mut result = Vec::new();
        let cells_to_check = (radius * self.inv_cell_size).ceil() as i32;
        let (cx, cy, cz) = self.cell_coords(position);

        for dx in -cells_to_check..=cells_to_check {
            for dy in -cells_to_check..=cells_to_check {
                for dz in -cells_to_check..=cells_to_check {
                    let hash = self.hash_cell(cx + dx, cy + dy, cz + dz);
                    if let Some(cell) = self.cells.get(&hash) {
                        result.extend_from_slice(cell);
                    }
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// FluidSettings
// ---------------------------------------------------------------------------

/// Configuration for the PBD fluid simulation.
#[derive(Debug, Clone)]
pub struct FluidSettingsV2 {
    /// Smoothing radius (h).
    pub smoothing_radius: f32,
    /// Rest density.
    pub rest_density: f32,
    /// Particle mass.
    pub particle_mass: f32,
    /// XSPH viscosity coefficient.
    pub xsph_viscosity: f32,
    /// Relaxation parameter (epsilon in lambda denominator).
    pub relaxation: f32,
    /// Vorticity confinement coefficient.
    pub vorticity_epsilon: f32,
    /// Number of solver iterations.
    pub solver_iterations: usize,
    /// Tensile instability correction.
    pub tensile_correction: bool,
    /// Gravity.
    pub gravity: Vec3,
    /// Time step.
    pub dt: f32,
    /// Whether to apply vorticity confinement.
    pub apply_vorticity: bool,
    /// Whether to compute surface particles.
    pub compute_surface: bool,
    /// Boundary handling.
    pub boundary_handling: bool,
    /// World bounds (min, max).
    pub world_bounds: Option<(Vec3, Vec3)>,
    /// Coefficient of restitution for boundary collisions.
    pub boundary_restitution: f32,
}

impl Default for FluidSettingsV2 {
    fn default() -> Self {
        Self {
            smoothing_radius: DEFAULT_SMOOTHING_RADIUS,
            rest_density: REST_DENSITY,
            particle_mass: DEFAULT_PARTICLE_MASS,
            xsph_viscosity: DEFAULT_XSPH_VISCOSITY,
            relaxation: DEFAULT_RELAXATION,
            vorticity_epsilon: DEFAULT_VORTICITY_EPSILON,
            solver_iterations: MAX_SOLVER_ITERATIONS,
            tensile_correction: true,
            gravity: GRAVITY,
            dt: DEFAULT_DT,
            apply_vorticity: true,
            compute_surface: true,
            boundary_handling: true,
            world_bounds: Some((
                Vec3::new(-10.0, -1.0, -10.0),
                Vec3::new(10.0, 10.0, 10.0),
            )),
            boundary_restitution: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// SurfaceInfo
// ---------------------------------------------------------------------------

/// Surface reconstruction data for a particle.
#[derive(Debug, Clone, Default)]
pub struct SurfaceInfo {
    /// Normal at the surface (pointing outward).
    pub normal: Vec3,
    /// Curvature estimate.
    pub curvature: f32,
    /// Color field value (for surface detection).
    pub color_field: f32,
    /// Whether this is a confirmed surface particle.
    pub is_surface: bool,
}

// ---------------------------------------------------------------------------
// FluidSimulationV2
// ---------------------------------------------------------------------------

/// Position-based fluid simulation.
pub struct FluidSimulationV2 {
    /// Fluid particles.
    pub particles: Vec<FluidParticle>,
    /// Boundary particles.
    pub boundary_particles: Vec<BoundaryParticle>,
    /// Settings.
    pub settings: FluidSettingsV2,
    /// Spatial hash grid.
    grid: SpatialHashGrid,
    /// Surface info per particle.
    surface_info: Vec<SurfaceInfo>,
    /// Simulation time.
    pub time: f64,
    /// Frame counter.
    pub frame: u64,
    /// Statistics.
    stats: FluidSimStats,
}

/// Statistics for the fluid simulation.
#[derive(Debug, Clone, Default)]
pub struct FluidSimStats {
    /// Total particle count.
    pub particle_count: usize,
    /// Active particle count.
    pub active_count: usize,
    /// Surface particle count.
    pub surface_count: usize,
    /// Average density.
    pub avg_density: f32,
    /// Maximum density.
    pub max_density: f32,
    /// Average neighbor count.
    pub avg_neighbors: f32,
    /// Solver iterations used.
    pub iterations: usize,
}

impl FluidSimulationV2 {
    /// Create a new fluid simulation.
    pub fn new(settings: FluidSettingsV2) -> Self {
        let cell_size = settings.smoothing_radius * CELL_SIZE_MULTIPLIER;
        Self {
            particles: Vec::new(),
            boundary_particles: Vec::new(),
            settings,
            grid: SpatialHashGrid::new(cell_size),
            surface_info: Vec::new(),
            time: 0.0,
            frame: 0,
            stats: FluidSimStats::default(),
        }
    }

    /// Add a fluid particle.
    pub fn add_particle(&mut self, particle: FluidParticle) -> u32 {
        let idx = self.particles.len() as u32;
        self.particles.push(particle);
        self.surface_info.push(SurfaceInfo::default());
        idx
    }

    /// Add multiple particles in a block.
    pub fn add_block(
        &mut self,
        min: Vec3,
        max: Vec3,
        spacing: f32,
    ) -> Vec<u32> {
        let mut indices = Vec::new();
        let mut x = min.x;
        while x <= max.x {
            let mut y = min.y;
            while y <= max.y {
                let mut z = min.z;
                while z <= max.z {
                    let idx = self.add_particle(FluidParticle::new(Vec3::new(x, y, z)));
                    indices.push(idx);
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }
        indices
    }

    /// Add a boundary particle.
    pub fn add_boundary(&mut self, particle: BoundaryParticle) {
        self.boundary_particles.push(particle);
    }

    /// Add a boundary plane (generates boundary particles).
    pub fn add_boundary_plane(
        &mut self,
        origin: Vec3,
        normal: Vec3,
        extent1: Vec3,
        extent2: Vec3,
        spacing: f32,
    ) {
        let n1 = (extent1.length() / spacing) as i32;
        let n2 = (extent2.length() / spacing) as i32;
        let d1 = extent1.normalize_or_zero() * spacing;
        let d2 = extent2.normalize_or_zero() * spacing;

        for i in 0..=n1 {
            for j in 0..=n2 {
                let pos = origin + d1 * i as f32 + d2 * j as f32;
                self.add_boundary(BoundaryParticle::new(pos, normal));
            }
        }
    }

    /// Step the simulation.
    pub fn step(&mut self) {
        let dt = self.settings.dt;
        let h = self.settings.smoothing_radius;
        let h_sq = h * h;
        let n = self.particles.len();

        if n == 0 {
            return;
        }

        // 1. Apply external forces and predict positions
        for p in &mut self.particles {
            if !p.active {
                continue;
            }
            p.velocity += self.settings.gravity * dt;
            p.predicted_position = p.position + p.velocity * dt;
        }

        // 2. Build spatial hash from predicted positions
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.predicted_position).collect();
        self.grid.build(&positions);

        // 3. Find neighbors
        for i in 0..n {
            let pos = self.particles[i].predicted_position;
            let neighbor_indices = self.grid.query_neighbors(pos, h);
            self.particles[i].neighbors = neighbor_indices.into_iter()
                .filter(|&j| j != i as u32)
                .collect();
            self.particles[i].neighbor_count = self.particles[i].neighbors.len() as u32;
        }

        // 4. Solver iterations
        for _iter in 0..self.settings.solver_iterations {
            // Compute density and lambda for each particle
            for i in 0..n {
                if !self.particles[i].active {
                    continue;
                }

                let pi = self.particles[i].predicted_position;
                let mut density = 0.0_f32;

                // Self-contribution
                density += self.settings.particle_mass * poly6(0.0, h);

                // Neighbor contributions
                let neighbors = self.particles[i].neighbors.clone();
                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let r = pi - pj;
                    let r_sq = r.length_squared();
                    if r_sq < h_sq {
                        density += self.settings.particle_mass * poly6(r_sq, h);
                    }
                }

                // Boundary contributions
                if self.settings.boundary_handling {
                    for bp in &self.boundary_particles {
                        let r = pi - bp.position;
                        let r_sq = r.length_squared();
                        if r_sq < h_sq {
                            density += bp.psi * self.settings.particle_mass * poly6(r_sq, h);
                        }
                    }
                }

                self.particles[i].density = density;

                // Compute constraint: C_i = density / rest_density - 1
                let constraint = density / self.settings.rest_density - 1.0;

                // Compute gradient sum for lambda denominator
                let mut grad_sum_sq = 0.0_f32;
                let mut grad_ci = Vec3::ZERO;

                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let r = pi - pj;
                    let r_len = r.length();
                    let grad = spiky_gradient(r, r_len, h) / self.settings.rest_density;
                    grad_sum_sq += grad.length_squared();
                    grad_ci += grad;
                }

                grad_sum_sq += grad_ci.length_squared();

                // Lambda = -C / (sum(|grad_C|^2) + epsilon)
                self.particles[i].lambda = -constraint / (grad_sum_sq + self.settings.relaxation);
            }

            // Compute position correction delta_p
            for i in 0..n {
                if !self.particles[i].active {
                    continue;
                }

                let pi = self.particles[i].predicted_position;
                let lambda_i = self.particles[i].lambda;
                let mut delta_p = Vec3::ZERO;

                let neighbors = self.particles[i].neighbors.clone();
                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let lambda_j = self.particles[j as usize].lambda;

                    let r = pi - pj;
                    let r_len = r.length();

                    // Tensile instability correction
                    let s_corr = if self.settings.tensile_correction {
                        let dq = TENSILE_DELTA_Q * h;
                        let w_dq = poly6(dq * dq, h);
                        let w_r = poly6(r.length_squared(), h);
                        if w_dq > EPSILON {
                            -TENSILE_K * (w_r / w_dq).powi(TENSILE_N as i32)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    let grad = spiky_gradient(r, r_len, h);
                    delta_p += (lambda_i + lambda_j + s_corr) * grad / self.settings.rest_density;
                }

                self.particles[i].delta_p = delta_p;
            }

            // Apply position corrections
            for p in &mut self.particles {
                if !p.active {
                    continue;
                }
                p.predicted_position += p.delta_p;
            }
        }

        // 5. Update velocities
        let inv_dt = 1.0 / dt;
        for p in &mut self.particles {
            if !p.active {
                continue;
            }
            p.velocity = (p.predicted_position - p.position) * inv_dt;
        }

        // 6. Vorticity confinement
        if self.settings.apply_vorticity {
            // Compute vorticity (curl of velocity field)
            for i in 0..n {
                if !self.particles[i].active {
                    continue;
                }
                let pi = self.particles[i].predicted_position;
                let vi = self.particles[i].velocity;
                let mut curl = Vec3::ZERO;

                let neighbors = self.particles[i].neighbors.clone();
                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let vj = self.particles[j as usize].velocity;
                    let r = pi - pj;
                    let r_len = r.length();
                    let grad = spiky_gradient(r, r_len, h);
                    let v_diff = vj - vi;
                    curl += v_diff.cross(grad);
                }

                self.particles[i].vorticity = curl;
            }

            // Apply vorticity confinement force
            for i in 0..n {
                if !self.particles[i].active {
                    continue;
                }
                let vort = self.particles[i].vorticity;
                let vort_mag = vort.length();
                if vort_mag < EPSILON {
                    continue;
                }

                // Compute eta (gradient of vorticity magnitude)
                let pi = self.particles[i].predicted_position;
                let mut eta = Vec3::ZERO;
                let neighbors = self.particles[i].neighbors.clone();
                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let vort_j_mag = self.particles[j as usize].vorticity.length();
                    let r = pi - pj;
                    let r_len = r.length();
                    let grad = spiky_gradient(r, r_len, h);
                    eta += grad * vort_j_mag;
                }

                let eta_len = eta.length();
                if eta_len < EPSILON {
                    continue;
                }
                let n_vec = eta / eta_len;
                let f_vorticity = n_vec.cross(vort) * self.settings.vorticity_epsilon;
                self.particles[i].velocity += f_vorticity * dt;
            }
        }

        // 7. XSPH viscosity
        if self.settings.xsph_viscosity > EPSILON {
            let mut velocity_corrections = vec![Vec3::ZERO; n];
            for i in 0..n {
                if !self.particles[i].active {
                    continue;
                }
                let pi = self.particles[i].predicted_position;
                let vi = self.particles[i].velocity;
                let mut v_xsph = Vec3::ZERO;

                let neighbors = self.particles[i].neighbors.clone();
                for &j in &neighbors {
                    let pj = self.particles[j as usize].predicted_position;
                    let vj = self.particles[j as usize].velocity;
                    let r_sq = (pi - pj).length_squared();
                    let w = poly6(r_sq, h);
                    let density_j = self.particles[j as usize].density.max(EPSILON);
                    v_xsph += (vj - vi) * (self.settings.particle_mass / density_j) * w;
                }

                velocity_corrections[i] = v_xsph * self.settings.xsph_viscosity;
            }
            for (i, correction) in velocity_corrections.into_iter().enumerate() {
                self.particles[i].velocity += correction;
            }
        }

        // 8. Finalize positions
        for p in &mut self.particles {
            if !p.active {
                continue;
            }
            p.position = p.predicted_position;
            p.age += dt;
        }

        // 9. Enforce world bounds
        if let Some((bmin, bmax)) = self.settings.world_bounds {
            let restitution = self.settings.boundary_restitution;
            for p in &mut self.particles {
                if !p.active {
                    continue;
                }
                for axis in 0..3 {
                    if p.position[axis] < bmin[axis] {
                        p.position[axis] = bmin[axis];
                        p.velocity[axis] *= -restitution;
                    }
                    if p.position[axis] > bmax[axis] {
                        p.position[axis] = bmax[axis];
                        p.velocity[axis] *= -restitution;
                    }
                }
            }
        }

        // 10. Surface detection
        if self.settings.compute_surface {
            for i in 0..n {
                let expected_neighbors = (4.0 / 3.0 * PI * h * h * h * self.settings.rest_density
                    / self.settings.particle_mass) as u32;
                let ratio = self.particles[i].neighbor_count as f32 / expected_neighbors.max(1) as f32;
                self.particles[i].is_surface = ratio < SURFACE_THRESHOLD;

                if i < self.surface_info.len() {
                    self.surface_info[i].is_surface = self.particles[i].is_surface;

                    // Estimate surface normal from density gradient
                    if self.particles[i].is_surface {
                        let pi = self.particles[i].position;
                        let mut normal = Vec3::ZERO;
                        let neighbors = &self.particles[i].neighbors;
                        for &j in neighbors {
                            let pj = self.particles[j as usize].position;
                            let r = pi - pj;
                            let r_sq = r.length_squared();
                            normal += poly6_gradient(r, r_sq, h) * self.settings.particle_mass;
                        }
                        let normal_len = normal.length();
                        if normal_len > EPSILON {
                            self.surface_info[i].normal = normal / normal_len;
                        }
                    }
                }
            }
        }

        // Update stats
        self.update_stats();

        self.time += dt as f64;
        self.frame += 1;
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        let active: Vec<&FluidParticle> = self.particles.iter().filter(|p| p.active).collect();
        let n = active.len();

        self.stats.particle_count = self.particles.len();
        self.stats.active_count = n;
        self.stats.surface_count = active.iter().filter(|p| p.is_surface).count();
        self.stats.iterations = self.settings.solver_iterations;

        if n > 0 {
            self.stats.avg_density = active.iter().map(|p| p.density).sum::<f32>() / n as f32;
            self.stats.max_density = active.iter().map(|p| p.density).fold(0.0_f32, f32::max);
            self.stats.avg_neighbors = active.iter().map(|p| p.neighbor_count as f32).sum::<f32>() / n as f32;
        }
    }

    /// Get particle positions for rendering.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter()
            .filter(|p| p.active)
            .map(|p| p.position)
            .collect()
    }

    /// Get surface info.
    pub fn surface_info(&self) -> &[SurfaceInfo] {
        &self.surface_info
    }

    /// Get statistics.
    pub fn stats(&self) -> &FluidSimStats {
        &self.stats
    }

    /// Get the number of active particles.
    pub fn active_count(&self) -> usize {
        self.stats.active_count
    }

    /// Remove particles below a Y threshold (drain).
    pub fn drain_below(&mut self, y_threshold: f32) {
        for p in &mut self.particles {
            if p.position.y < y_threshold {
                p.active = false;
            }
        }
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        self.particles.clear();
        self.boundary_particles.clear();
        self.surface_info.clear();
        self.time = 0.0;
        self.frame = 0;
        self.grid.clear();
    }
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component for fluid particle systems.
#[derive(Debug)]
pub struct FluidParticleComponent {
    /// Index into the simulation's particle array.
    pub sim_id: u64,
    /// Whether to render this fluid.
    pub visible: bool,
    /// Color tint for this fluid.
    pub color: [f32; 4],
}

impl FluidParticleComponent {
    /// Create a new component.
    pub fn new(sim_id: u64) -> Self {
        Self {
            sim_id,
            visible: true,
            color: [0.2, 0.5, 1.0, 0.8],
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
    fn test_poly6_kernel() {
        let h = 0.1;
        // At r=0, kernel should be maximum
        let w0 = poly6(0.0, h);
        assert!(w0 > 0.0);

        // At r=h, kernel should be 0
        let wh = poly6(h * h, h);
        assert!(wh.abs() < EPSILON);

        // Beyond r=h, kernel should be 0
        let beyond = poly6(h * h * 2.0, h);
        assert!(beyond.abs() < EPSILON);
    }

    #[test]
    fn test_spiky_gradient() {
        let h = 0.1;
        let r = Vec3::new(0.05, 0.0, 0.0);
        let grad = spiky_gradient(r, r.length(), h);
        // Gradient should point in the direction of r
        assert!(grad.x < 0.0); // Spiky gradient is negative
    }

    #[test]
    fn test_spatial_hash() {
        let mut grid = SpatialHashGrid::new(0.2);
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        grid.build(&positions);

        let neighbors = grid.query_neighbors(Vec3::ZERO, 0.1);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn test_fluid_simulation_basic() {
        let mut sim = FluidSimulationV2::new(FluidSettingsV2::default());
        sim.add_block(
            Vec3::new(-0.1, 0.0, -0.1),
            Vec3::new(0.1, 0.2, 0.1),
            DEFAULT_SMOOTHING_RADIUS * 0.8,
        );
        assert!(sim.active_count() > 0);

        // Step the simulation
        sim.step();
        assert!(sim.frame == 1);
    }

    #[test]
    fn test_boundary_particle() {
        let bp = BoundaryParticle::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!((bp.normal.length() - 1.0).abs() < EPSILON);
    }
}
