//! Smoothed Particle Hydrodynamics (SPH) fluid simulation.
//!
//! Implements a complete SPH fluid pipeline:
//! - SPH kernel functions (Poly6, Spiky, Viscosity)
//! - Density and pressure computation
//! - Pressure force (Navier-Stokes)
//! - Viscosity force
//! - Surface tension (color field gradient)
//! - Spatial hash grid neighbor search
//! - Boundary handling (penalty forces)
//! - Marching cubes mesh extraction for rendering
//! - ECS integration

use std::collections::HashMap;

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Mathematical constant pi.
const PI: f32 = std::f32::consts::PI;
/// Default smoothing length for SPH kernels.
const DEFAULT_SMOOTHING_LENGTH: f32 = 0.1;
/// Default rest density of water in kg/m^3.
const DEFAULT_REST_DENSITY: f32 = 1000.0;
/// Default gas constant (stiffness of equation of state).
const DEFAULT_GAS_CONSTANT: f32 = 2000.0;
/// Default viscosity coefficient.
const DEFAULT_VISCOSITY: f32 = 1.0;
/// Default surface tension coefficient.
const DEFAULT_SURFACE_TENSION: f32 = 0.0728;
/// Epsilon to avoid division by zero.
const EPSILON: f32 = 1e-8;

// ---------------------------------------------------------------------------
// SPH Kernel functions
// ---------------------------------------------------------------------------

/// Poly6 kernel: W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
///
/// Used for density estimation. Smooth and non-negative.
#[derive(Debug, Clone, Copy)]
pub struct Poly6Kernel {
    /// Smoothing length.
    pub h: f32,
    /// Precomputed coefficient: 315 / (64 * pi * h^9).
    coeff: f32,
    /// h^2 precomputed.
    h2: f32,
}

impl Poly6Kernel {
    /// Create a new Poly6 kernel with the given smoothing length.
    pub fn new(h: f32) -> Self {
        let h2 = h * h;
        let h9 = h2 * h2 * h2 * h2 * h;
        let coeff = 315.0 / (64.0 * PI * h9);
        Self { h, coeff, h2 }
    }

    /// Evaluate the kernel: W(r, h).
    /// `r_sq` is the squared distance between particles.
    #[inline]
    pub fn evaluate(&self, r_sq: f32) -> f32 {
        if r_sq >= self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r_sq;
        self.coeff * diff * diff * diff
    }

    /// Gradient of the Poly6 kernel.
    /// Returns the gradient vector given the displacement `r` (from j to i).
    #[inline]
    pub fn gradient(&self, r: Vec3, r_sq: f32) -> Vec3 {
        if r_sq >= self.h2 || r_sq < EPSILON {
            return Vec3::ZERO;
        }
        let diff = self.h2 - r_sq;
        // grad W = -945 / (32 * pi * h^9) * (h^2 - r^2)^2 * r
        let grad_coeff = -6.0 * self.coeff * diff * diff;
        r * grad_coeff
    }

    /// Laplacian of the Poly6 kernel.
    #[inline]
    pub fn laplacian(&self, r_sq: f32) -> f32 {
        if r_sq >= self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r_sq;
        // lap W = -945 / (32 * pi * h^9) * (h^2 - r^2) * (3h^2 - 7r^2)
        let lap_coeff = -6.0 * self.coeff;
        lap_coeff * diff * (3.0 * self.h2 - 7.0 * r_sq)
    }
}

/// Spiky kernel gradient: grad_W = -45 / (pi * h^6) * (h - r)^2 * r_hat
///
/// Used for pressure force computation. Non-zero at the center, which prevents
/// particle clustering.
#[derive(Debug, Clone, Copy)]
pub struct SpikyKernel {
    /// Smoothing length.
    pub h: f32,
    /// Precomputed coefficient: -45 / (pi * h^6).
    coeff: f32,
}

impl SpikyKernel {
    /// Create a new Spiky kernel.
    pub fn new(h: f32) -> Self {
        let h6 = h * h * h * h * h * h;
        let coeff = -45.0 / (PI * h6);
        Self { h, coeff }
    }

    /// Gradient of the Spiky kernel.
    /// `r` is the displacement vector (from j to i), `r_len` is |r|.
    #[inline]
    pub fn gradient(&self, r: Vec3, r_len: f32) -> Vec3 {
        if r_len >= self.h || r_len < EPSILON {
            return Vec3::ZERO;
        }
        let diff = self.h - r_len;
        let r_hat = r / r_len;
        r_hat * (self.coeff * diff * diff)
    }
}

/// Viscosity kernel Laplacian: lap_W = 45 / (pi * h^6) * (h - r)
///
/// Used for viscosity force computation.
#[derive(Debug, Clone, Copy)]
pub struct ViscosityKernel {
    /// Smoothing length.
    pub h: f32,
    /// Precomputed coefficient: 45 / (pi * h^6).
    coeff: f32,
}

impl ViscosityKernel {
    /// Create a new Viscosity kernel.
    pub fn new(h: f32) -> Self {
        let h6 = h * h * h * h * h * h;
        let coeff = 45.0 / (PI * h6);
        Self { h, coeff }
    }

    /// Laplacian of the Viscosity kernel.
    #[inline]
    pub fn laplacian(&self, r_len: f32) -> f32 {
        if r_len >= self.h {
            return 0.0;
        }
        self.coeff * (self.h - r_len)
    }
}

// ---------------------------------------------------------------------------
// Fluid particle
// ---------------------------------------------------------------------------

/// A single SPH fluid particle.
#[derive(Debug, Clone)]
pub struct FluidParticle {
    /// World-space position.
    pub position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Computed density at this particle's location.
    pub density: f32,
    /// Computed pressure at this particle's location.
    pub pressure: f32,
    /// Accumulated force for the current step.
    pub force: Vec3,
    /// Particle mass.
    pub mass: f32,
    /// Color field value (for surface tension).
    pub color_field: f32,
    /// Color field gradient (for surface normal estimation).
    pub color_gradient: Vec3,
    /// Color field Laplacian (for surface tension force).
    pub color_laplacian: f32,
}

impl FluidParticle {
    /// Create a new fluid particle.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            force: Vec3::ZERO,
            mass,
            color_field: 0.0,
            color_gradient: Vec3::ZERO,
            color_laplacian: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial hash grid for neighbor search
// ---------------------------------------------------------------------------

/// Spatial hash grid for efficient neighbor lookup in SPH.
///
/// Divides space into uniform cells of size `h` (smoothing length).
/// For each particle, we only need to check the 27 neighboring cells.
#[derive(Debug)]
pub struct FluidSpatialHash {
    /// Cell size (should be >= smoothing length).
    cell_size: f32,
    /// Inverse cell size (for fast coordinate computation).
    inv_cell_size: f32,
    /// Maps cell key -> list of particle indices.
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl FluidSpatialHash {
    /// Create a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    /// Compute the cell coordinate for a given position component.
    #[inline]
    fn cell_coord(&self, v: f32) -> i32 {
        (v * self.inv_cell_size).floor() as i32
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert all particles into the grid.
    pub fn build(&mut self, particles: &[FluidParticle]) {
        self.cells.clear();
        for (i, p) in particles.iter().enumerate() {
            let cx = self.cell_coord(p.position.x);
            let cy = self.cell_coord(p.position.y);
            let cz = self.cell_coord(p.position.z);
            self.cells.entry((cx, cy, cz)).or_default().push(i);
        }
    }

    /// Query all neighbor particle indices within the smoothing radius of a position.
    pub fn query_neighbors(&self, position: Vec3) -> Vec<usize> {
        let cx = self.cell_coord(position.x);
        let cy = self.cell_coord(position.y);
        let cz = self.cell_coord(position.z);

        let mut neighbors = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cell) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        neighbors.extend_from_slice(cell);
                    }
                }
            }
        }
        neighbors
    }
}

// ---------------------------------------------------------------------------
// Boundary collider
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box boundary for containing fluid.
#[derive(Debug, Clone)]
pub struct FluidBoundary {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
    /// Penalty force stiffness for boundary enforcement.
    pub stiffness: f32,
    /// Damping factor for boundary collision.
    pub damping: f32,
}

impl Default for FluidBoundary {
    fn default() -> Self {
        Self {
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
            stiffness: 10000.0,
            damping: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Fluid settings
// ---------------------------------------------------------------------------

/// Configuration for the SPH fluid simulation.
#[derive(Debug, Clone)]
pub struct FluidSettings {
    /// Particle radius for rendering.
    pub particle_radius: f32,
    /// Smoothing length (h) for SPH kernels.
    pub smoothing_length: f32,
    /// Rest density of the fluid in kg/m^3.
    pub rest_density: f32,
    /// Gas constant for the equation of state.
    pub gas_constant: f32,
    /// Viscosity coefficient.
    pub viscosity: f32,
    /// Surface tension coefficient.
    pub surface_tension: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Simulation time step.
    pub time_step: f32,
    /// Whether surface tension computation is enabled.
    pub enable_surface_tension: bool,
    /// Particle mass (derived from rest_density and particle spacing).
    pub particle_mass: f32,
    /// Boundary for containing the fluid.
    pub boundary: FluidBoundary,
}

impl Default for FluidSettings {
    fn default() -> Self {
        let h = DEFAULT_SMOOTHING_LENGTH;
        // Mass = density * volume_per_particle
        // For a cubic arrangement: volume = (0.5 * h)^3
        let spacing = h * 0.5;
        let particle_mass = DEFAULT_REST_DENSITY * spacing * spacing * spacing;

        Self {
            particle_radius: spacing * 0.5,
            smoothing_length: h,
            rest_density: DEFAULT_REST_DENSITY,
            gas_constant: DEFAULT_GAS_CONSTANT,
            viscosity: DEFAULT_VISCOSITY,
            surface_tension: DEFAULT_SURFACE_TENSION,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            time_step: 1.0 / 120.0,
            enable_surface_tension: true,
            particle_mass,
            boundary: FluidBoundary::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// SPH Fluid simulation
// ---------------------------------------------------------------------------

/// Complete SPH fluid simulation.
///
/// Implements the Muller et al. 2003 SPH method with:
/// - Poly6 kernel for density
/// - Spiky gradient for pressure force
/// - Viscosity Laplacian for viscosity force
/// - Color field method for surface tension
pub struct SPHFluid {
    /// All fluid particles.
    pub particles: Vec<FluidParticle>,
    /// Simulation settings.
    pub settings: FluidSettings,
    /// Spatial hash grid for neighbor search.
    spatial_hash: FluidSpatialHash,
    /// Poly6 kernel instance.
    poly6: Poly6Kernel,
    /// Spiky kernel instance.
    spiky: SpikyKernel,
    /// Viscosity kernel instance.
    viscosity_kernel: ViscosityKernel,
    /// Cached neighbor lists per particle (rebuilt each step).
    neighbor_cache: Vec<Vec<usize>>,
    /// Running simulation time.
    sim_time: f32,
}

impl Clone for SPHFluid {
    fn clone(&self) -> Self {
        let h = self.settings.smoothing_length;
        Self {
            particles: self.particles.clone(),
            settings: self.settings.clone(),
            spatial_hash: FluidSpatialHash::new(h),
            poly6: Poly6Kernel::new(h),
            spiky: SpikyKernel::new(h),
            viscosity_kernel: ViscosityKernel::new(h),
            neighbor_cache: self.neighbor_cache.clone(),
            sim_time: self.sim_time,
        }
    }
}

impl SPHFluid {
    /// Create a new SPH fluid simulation with the given settings.
    pub fn new(settings: FluidSettings) -> Self {
        let h = settings.smoothing_length;
        Self {
            particles: Vec::new(),
            settings: settings.clone(),
            spatial_hash: FluidSpatialHash::new(h),
            poly6: Poly6Kernel::new(h),
            spiky: SpikyKernel::new(h),
            viscosity_kernel: ViscosityKernel::new(h),
            neighbor_cache: Vec::new(),
            sim_time: 0.0,
        }
    }

    /// Add a single particle at the given position.
    pub fn add_particle(&mut self, position: Vec3) {
        let mass = self.settings.particle_mass;
        self.particles.push(FluidParticle::new(position, mass));
    }

    /// Add a block of particles filling a box region.
    ///
    /// Particles are placed in a regular grid with spacing = 0.5 * smoothing_length.
    pub fn add_block(&mut self, min: Vec3, max: Vec3) {
        let spacing = self.settings.smoothing_length * 0.5;
        let mass = self.settings.particle_mass;

        let mut x = min.x;
        while x <= max.x {
            let mut y = min.y;
            while y <= max.y {
                let mut z = min.z;
                while z <= max.z {
                    self.particles
                        .push(FluidParticle::new(Vec3::new(x, y, z), mass));
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }
    }

    /// Get the number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Step the fluid simulation forward by one time step.
    ///
    /// Pipeline:
    /// 1. Build spatial hash and find neighbors
    /// 2. Compute density for each particle
    /// 3. Compute pressure from density
    /// 4. Compute pressure force
    /// 5. Compute viscosity force
    /// 6. Compute surface tension (optional)
    /// 7. Apply gravity
    /// 8. Apply boundary forces
    /// 9. Integrate (symplectic Euler)
    pub fn step(&mut self, dt: f32) {
        let dt = if dt > 0.0 { dt } else { self.settings.time_step };
        self.sim_time += dt;

        if self.particles.is_empty() {
            return;
        }

        // 1. Build spatial hash and find neighbors
        self.build_neighbors();

        // 2. Compute density
        self.compute_density();

        // 3. Compute pressure
        self.compute_pressure();

        // 4. Compute pressure force
        self.compute_pressure_force();

        // 5. Compute viscosity force
        self.compute_viscosity_force();

        // 6. Surface tension
        if self.settings.enable_surface_tension {
            self.compute_surface_tension();
        }

        // 7. Apply gravity
        self.apply_gravity();

        // 8. Boundary forces
        self.apply_boundary_forces();

        // 9. Integrate
        self.integrate(dt);
    }

    // -----------------------------------------------------------------------
    // Neighbor search
    // -----------------------------------------------------------------------

    fn build_neighbors(&mut self) {
        self.spatial_hash.build(&self.particles);

        let n = self.particles.len();
        self.neighbor_cache.resize(n, Vec::new());

        let h2 = self.settings.smoothing_length * self.settings.smoothing_length;

        for i in 0..n {
            let pos = self.particles[i].position;
            let raw_neighbors = self.spatial_hash.query_neighbors(pos);

            self.neighbor_cache[i].clear();
            for &j in &raw_neighbors {
                if i == j {
                    continue;
                }
                let r_sq = (self.particles[j].position - pos).length_squared();
                if r_sq < h2 {
                    self.neighbor_cache[i].push(j);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Density computation
    // -----------------------------------------------------------------------

    /// Compute density for each particle:
    ///   rho_i = sum_j(m_j * W(|r_i - r_j|, h))
    fn compute_density(&mut self) {
        let n = self.particles.len();
        for i in 0..n {
            let pos_i = self.particles[i].position;

            // Self-contribution
            let self_density = self.particles[i].mass * self.poly6.evaluate(0.0);
            let mut density = self_density;

            let neighbors = self.neighbor_cache[i].clone();
            for &j in &neighbors {
                let r = pos_i - self.particles[j].position;
                let r_sq = r.length_squared();
                density += self.particles[j].mass * self.poly6.evaluate(r_sq);
            }

            self.particles[i].density = density.max(EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // Pressure computation
    // -----------------------------------------------------------------------

    /// Compute pressure from density using the equation of state:
    ///   P = k * (rho - rho_0)
    fn compute_pressure(&mut self) {
        let k = self.settings.gas_constant;
        let rho_0 = self.settings.rest_density;

        for p in &mut self.particles {
            p.pressure = k * (p.density - rho_0);
        }
    }

    // -----------------------------------------------------------------------
    // Pressure force
    // -----------------------------------------------------------------------

    /// Compute pressure force:
    ///   f_pressure_i = -sum_j(m_j * (P_i + P_j) / (2 * rho_j) * grad_W_spiky(r_i - r_j))
    fn compute_pressure_force(&mut self) {
        let n = self.particles.len();
        let mut forces = vec![Vec3::ZERO; n];

        for i in 0..n {
            let pos_i = self.particles[i].position;
            let p_i = self.particles[i].pressure;
            let rho_i = self.particles[i].density;

            let neighbors = self.neighbor_cache[i].clone();
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let p_j = self.particles[j].pressure;
                let rho_j = self.particles[j].density;
                let mass_j = self.particles[j].mass;

                let r = pos_i - pos_j;
                let r_len = r.length();

                if r_len < EPSILON || rho_j < EPSILON {
                    continue;
                }

                // Pressure force: symmetric formulation
                let grad = self.spiky.gradient(r, r_len);
                let pressure_avg = (p_i + p_j) / (2.0 * rho_j);
                forces[i] -= grad * (mass_j * pressure_avg);
            }

            // Scale by 1/rho_i for proper SPH formulation
            if rho_i > EPSILON {
                forces[i] *= rho_i;
            }
        }

        for i in 0..n {
            self.particles[i].force += forces[i];
        }
    }

    // -----------------------------------------------------------------------
    // Viscosity force
    // -----------------------------------------------------------------------

    /// Compute viscosity force:
    ///   f_viscosity_i = mu * sum_j(m_j * (v_j - v_i) / rho_j * lap_W_viscosity(r_ij))
    fn compute_viscosity_force(&mut self) {
        let mu = self.settings.viscosity;
        let n = self.particles.len();
        let mut forces = vec![Vec3::ZERO; n];

        for i in 0..n {
            let pos_i = self.particles[i].position;
            let vel_i = self.particles[i].velocity;

            let neighbors = self.neighbor_cache[i].clone();
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let vel_j = self.particles[j].velocity;
                let rho_j = self.particles[j].density;
                let mass_j = self.particles[j].mass;

                let r = pos_i - pos_j;
                let r_len = r.length();

                if rho_j < EPSILON {
                    continue;
                }

                let lap = self.viscosity_kernel.laplacian(r_len);
                forces[i] += (vel_j - vel_i) * (mass_j / rho_j * lap);
            }

            forces[i] *= mu;
        }

        for i in 0..n {
            self.particles[i].force += forces[i];
        }
    }

    // -----------------------------------------------------------------------
    // Surface tension
    // -----------------------------------------------------------------------

    /// Compute surface tension using the color field method.
    ///
    /// 1. Compute color field: c_i = sum_j(m_j / rho_j * W(r_ij))
    /// 2. Compute gradient of color field: n_i = sum_j(m_j / rho_j * grad_W(r_ij))
    /// 3. Compute Laplacian of color field: lap_c_i = sum_j(m_j / rho_j * lap_W(r_ij))
    /// 4. Surface tension force: f_st = -sigma * lap_c * n / |n| (only where |n| > threshold)
    fn compute_surface_tension(&mut self) {
        let n = self.particles.len();
        let sigma = self.settings.surface_tension;

        // Compute color field, gradient, and Laplacian
        let mut color_gradients = vec![Vec3::ZERO; n];
        let mut color_laplacians = vec![0.0f32; n];

        for i in 0..n {
            let pos_i = self.particles[i].position;
            let mut gradient = Vec3::ZERO;
            let mut laplacian = 0.0f32;

            let neighbors = self.neighbor_cache[i].clone();
            for &j in &neighbors {
                let pos_j = self.particles[j].position;
                let rho_j = self.particles[j].density;
                let mass_j = self.particles[j].mass;

                if rho_j < EPSILON {
                    continue;
                }

                let r = pos_i - pos_j;
                let r_sq = r.length_squared();
                let scale = mass_j / rho_j;

                gradient += self.poly6.gradient(r, r_sq) * scale;
                laplacian += self.poly6.laplacian(r_sq) * scale;
            }

            color_gradients[i] = gradient;
            color_laplacians[i] = laplacian;
        }

        // Apply surface tension force
        let threshold = 6.0 / self.settings.smoothing_length;
        for i in 0..n {
            let grad_len = color_gradients[i].length();
            if grad_len > threshold {
                let normal = color_gradients[i] / grad_len;
                self.particles[i].force -= normal * (sigma * color_laplacians[i]);
            }
            self.particles[i].color_gradient = color_gradients[i];
            self.particles[i].color_laplacian = color_laplacians[i];
        }
    }

    // -----------------------------------------------------------------------
    // Gravity and boundary
    // -----------------------------------------------------------------------

    fn apply_gravity(&mut self) {
        let gravity = self.settings.gravity;
        for p in &mut self.particles {
            p.force += gravity * p.density;
        }
    }

    /// Apply penalty forces at the boundary walls.
    fn apply_boundary_forces(&mut self) {
        let boundary = &self.settings.boundary;
        let stiffness = boundary.stiffness;
        let damping = boundary.damping;

        for p in &mut self.particles {
            // X min
            if p.position.x < boundary.min.x {
                let penetration = boundary.min.x - p.position.x;
                p.force.x += stiffness * penetration - damping * p.velocity.x;
            }
            // X max
            if p.position.x > boundary.max.x {
                let penetration = p.position.x - boundary.max.x;
                p.force.x -= stiffness * penetration + damping * p.velocity.x;
            }
            // Y min
            if p.position.y < boundary.min.y {
                let penetration = boundary.min.y - p.position.y;
                p.force.y += stiffness * penetration - damping * p.velocity.y;
            }
            // Y max
            if p.position.y > boundary.max.y {
                let penetration = p.position.y - boundary.max.y;
                p.force.y -= stiffness * penetration + damping * p.velocity.y;
            }
            // Z min
            if p.position.z < boundary.min.z {
                let penetration = boundary.min.z - p.position.z;
                p.force.z += stiffness * penetration - damping * p.velocity.z;
            }
            // Z max
            if p.position.z > boundary.max.z {
                let penetration = p.position.z - boundary.max.z;
                p.force.z -= stiffness * penetration + damping * p.velocity.z;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Integration
    // -----------------------------------------------------------------------

    /// Symplectic Euler integration:
    ///   v_{n+1} = v_n + (f / rho) * dt
    ///   x_{n+1} = x_n + v_{n+1} * dt
    fn integrate(&mut self, dt: f32) {
        let boundary = self.settings.boundary.clone();
        let max_speed = self.settings.smoothing_length / dt * 0.5; // CFL condition

        for p in &mut self.particles {
            let acceleration = if p.density > EPSILON {
                p.force / p.density
            } else {
                Vec3::ZERO
            };

            p.velocity += acceleration * dt;

            // Clamp velocity (CFL condition for stability)
            let speed = p.velocity.length();
            if speed > max_speed {
                p.velocity *= max_speed / speed;
            }

            p.position += p.velocity * dt;

            // Hard clamp to boundary (safety net)
            let margin = self.settings.smoothing_length;
            let clamped_min = boundary.min - Vec3::splat(margin);
            let clamped_max = boundary.max + Vec3::splat(margin);
            if p.position.x < clamped_min.x {
                p.position.x = clamped_min.x;
                p.velocity.x = p.velocity.x.abs() * 0.1;
            }
            if p.position.x > clamped_max.x {
                p.position.x = clamped_max.x;
                p.velocity.x = -p.velocity.x.abs() * 0.1;
            }
            if p.position.y < clamped_min.y {
                p.position.y = clamped_min.y;
                p.velocity.y = p.velocity.y.abs() * 0.1;
            }
            if p.position.y > clamped_max.y {
                p.position.y = clamped_max.y;
                p.velocity.y = -p.velocity.y.abs() * 0.1;
            }
            if p.position.z < clamped_min.z {
                p.position.z = clamped_min.z;
                p.velocity.z = p.velocity.z.abs() * 0.1;
            }
            if p.position.z > clamped_max.z {
                p.position.z = clamped_max.z;
                p.velocity.z = -p.velocity.z.abs() * 0.1;
            }

            p.force = Vec3::ZERO;
        }
    }

    /// Get all particle positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Get all particle velocities.
    pub fn velocities(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.velocity).collect()
    }

    /// Get average density of all particles.
    pub fn average_density(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.particles.iter().map(|p| p.density).sum();
        sum / self.particles.len() as f32
    }

    /// Get maximum velocity magnitude.
    pub fn max_velocity(&self) -> f32 {
        self.particles
            .iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max)
    }

    /// Compute the total kinetic energy of the fluid.
    pub fn kinetic_energy(&self) -> f32 {
        self.particles
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.length_squared())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Marching cubes mesh extraction
// ---------------------------------------------------------------------------

/// A vertex in the extracted fluid surface mesh.
#[derive(Debug, Clone)]
pub struct FluidSurfaceVertex {
    /// World-space position.
    pub position: Vec3,
    /// Surface normal.
    pub normal: Vec3,
}

/// Extracted fluid surface mesh for rendering.
#[derive(Debug, Clone)]
pub struct FluidSurfaceMesh {
    /// Vertices of the mesh.
    pub vertices: Vec<FluidSurfaceVertex>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

/// Scalar field value at a grid point for marching cubes.
struct ScalarField {
    /// Grid dimensions.
    nx: usize,
    ny: usize,
    nz: usize,
    /// Cell size.
    cell_size: f32,
    /// Origin (minimum corner).
    origin: Vec3,
    /// Scalar values at grid points.
    values: Vec<f32>,
}

impl ScalarField {
    fn new(min: Vec3, max: Vec3, cell_size: f32) -> Self {
        let extent = max - min;
        let nx = (extent.x / cell_size).ceil() as usize + 1;
        let ny = (extent.y / cell_size).ceil() as usize + 1;
        let nz = (extent.z / cell_size).ceil() as usize + 1;
        let total = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            cell_size,
            origin: min,
            values: vec![0.0; total],
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.ny * self.nx + y * self.nx + x
    }

    fn position(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                x as f32 * self.cell_size,
                y as f32 * self.cell_size,
                z as f32 * self.cell_size,
            )
    }

    fn gradient(&self, x: usize, y: usize, z: usize) -> Vec3 {
        let gx = if x > 0 && x < self.nx - 1 {
            (self.values[self.index(x + 1, y, z)] - self.values[self.index(x - 1, y, z)])
                / (2.0 * self.cell_size)
        } else {
            0.0
        };
        let gy = if y > 0 && y < self.ny - 1 {
            (self.values[self.index(x, y + 1, z)] - self.values[self.index(x, y - 1, z)])
                / (2.0 * self.cell_size)
        } else {
            0.0
        };
        let gz = if z > 0 && z < self.nz - 1 {
            (self.values[self.index(x, y, z + 1)] - self.values[self.index(x, y, z - 1)])
                / (2.0 * self.cell_size)
        } else {
            0.0
        };
        Vec3::new(gx, gy, gz)
    }
}

/// Extract a fluid surface mesh using marching cubes.
///
/// 1. Build a scalar field by accumulating particle contributions using the Poly6 kernel.
/// 2. Apply marching cubes to extract the isosurface at the given threshold.
pub fn extract_surface_mesh(
    fluid: &SPHFluid,
    cell_size: f32,
    iso_threshold: f32,
) -> FluidSurfaceMesh {
    if fluid.particles.is_empty() {
        return FluidSurfaceMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Compute bounding box
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let margin = fluid.settings.smoothing_length * 2.0;

    for p in &fluid.particles {
        min = min.min(p.position);
        max = max.max(p.position);
    }
    min -= Vec3::splat(margin);
    max += Vec3::splat(margin);

    // Build scalar field
    let mut field = ScalarField::new(min, max, cell_size);
    let poly6 = Poly6Kernel::new(fluid.settings.smoothing_length);
    let h2 = fluid.settings.smoothing_length * fluid.settings.smoothing_length;

    for p in &fluid.particles {
        // Only update grid points within the smoothing radius
        let p_min = p.position - Vec3::splat(fluid.settings.smoothing_length);
        let p_max = p.position + Vec3::splat(fluid.settings.smoothing_length);

        let xi_min = ((p_min.x - field.origin.x) / field.cell_size).floor().max(0.0) as usize;
        let yi_min = ((p_min.y - field.origin.y) / field.cell_size).floor().max(0.0) as usize;
        let zi_min = ((p_min.z - field.origin.z) / field.cell_size).floor().max(0.0) as usize;
        let xi_max = ((p_max.x - field.origin.x) / field.cell_size).ceil() as usize;
        let yi_max = ((p_max.y - field.origin.y) / field.cell_size).ceil() as usize;
        let zi_max = ((p_max.z - field.origin.z) / field.cell_size).ceil() as usize;

        let xi_max = xi_max.min(field.nx - 1);
        let yi_max = yi_max.min(field.ny - 1);
        let zi_max = zi_max.min(field.nz - 1);

        for zi in zi_min..=zi_max {
            for yi in yi_min..=yi_max {
                for xi in xi_min..=xi_max {
                    let grid_pos = field.position(xi, yi, zi);
                    let r_sq = (grid_pos - p.position).length_squared();
                    if r_sq < h2 {
                        let idx = field.index(xi, yi, zi);
                        let density_contribution = if p.density > EPSILON {
                            p.mass / p.density
                        } else {
                            p.mass / DEFAULT_REST_DENSITY
                        };
                        field.values[idx] += density_contribution * poly6.evaluate(r_sq);
                    }
                }
            }
        }
    }

    // Marching cubes (simplified: generate vertices at edge crossings)
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut edge_vertex_map: HashMap<(usize, usize), u32> = HashMap::new();

    for z in 0..field.nz.saturating_sub(1) {
        for y in 0..field.ny.saturating_sub(1) {
            for x in 0..field.nx.saturating_sub(1) {
                // 8 corners of the cube
                let corners = [
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y + 1, z),
                    (x, y + 1, z),
                    (x, y, z + 1),
                    (x + 1, y, z + 1),
                    (x + 1, y + 1, z + 1),
                    (x, y + 1, z + 1),
                ];

                let mut cube_index = 0u8;
                for (i, &(cx, cy, cz)) in corners.iter().enumerate() {
                    let idx = field.index(cx, cy, cz);
                    if field.values[idx] > iso_threshold {
                        cube_index |= 1 << i;
                    }
                }

                if cube_index == 0 || cube_index == 255 {
                    continue;
                }

                // Process edges using the marching cubes edge table
                let edges = MC_TRI_TABLE[cube_index as usize];
                let mut tri_idx = 0;
                while tri_idx < edges.len() && edges[tri_idx] != -1 {
                    let mut tri_verts = [0u32; 3];
                    for k in 0..3 {
                        let edge = edges[tri_idx + k] as usize;
                        let (v0, v1) = MC_EDGE_VERTICES[edge];
                        let c0 = corners[v0];
                        let c1 = corners[v1];
                        let idx0 = field.index(c0.0, c0.1, c0.2);
                        let idx1 = field.index(c1.0, c1.1, c1.2);

                        let key = if idx0 < idx1 {
                            (idx0, idx1)
                        } else {
                            (idx1, idx0)
                        };

                        let vi = edge_vertex_map.entry(key).or_insert_with(|| {
                            let val0 = field.values[idx0];
                            let val1 = field.values[idx1];
                            let t = if (val1 - val0).abs() > EPSILON {
                                (iso_threshold - val0) / (val1 - val0)
                            } else {
                                0.5
                            };
                            let t = t.clamp(0.0, 1.0);
                            let pos0 = field.position(c0.0, c0.1, c0.2);
                            let pos1 = field.position(c1.0, c1.1, c1.2);
                            let pos = pos0 + (pos1 - pos0) * t;

                            // Interpolate normal from gradient
                            let n0 = field.gradient(c0.0, c0.1, c0.2);
                            let n1 = field.gradient(c1.0, c1.1, c1.2);
                            let normal = (n0 + (n1 - n0) * t).normalize_or_zero();

                            let idx = vertices.len() as u32;
                            vertices.push(FluidSurfaceVertex {
                                position: pos,
                                normal: -normal, // Outward normal
                            });
                            idx
                        });

                        tri_verts[k] = *vi;
                    }
                    indices.push(tri_verts[0]);
                    indices.push(tri_verts[1]);
                    indices.push(tri_verts[2]);
                    tri_idx += 3;
                }
            }
        }
    }

    FluidSurfaceMesh { vertices, indices }
}

// ---------------------------------------------------------------------------
// Marching cubes lookup tables (subset for the 256 cases)
// ---------------------------------------------------------------------------

/// Edge vertex indices for each of the 12 edges of a cube.
const MC_EDGE_VERTICES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Simplified triangle table for marching cubes.
/// Each entry is a list of edge indices forming triangles, terminated by -1.
/// This is a subset covering the most common configurations.
const MC_TRI_TABLE: [[i8; 16]; 256] = {
    let mut table = [[-1i8; 16]; 256];

    // Case 1: single corner inside (8 rotations)
    table[1] = [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[2] = [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[4] = [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[8] = [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[16] = [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[32] = [4, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[64] = [5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[128] = [6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

    // Case 2: two adjacent corners (12 rotations, showing a few)
    table[3] = [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[5] = [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[6] = [9, 2, 10, 9, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[9] = [0, 8, 11, 0, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[10] = [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[12] = [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

    // Additional cases for basic rendering
    table[15] = [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[17] = [0, 7, 3, 0, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[48] = [8, 9, 5, 8, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[51] = [3, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[68] = [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
    table[85] = [9, 5, 4, 0, 8, 3, 10, 6, 1, -1, -1, -1, -1, -1, -1, -1];
    table[170] = [1, 9, 0, 2, 3, 11, 5, 10, 6, 4, 7, 8, -1, -1, -1, -1];
    table[240] = [8, 11, 10, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

    table
};

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a fluid simulation to an entity.
#[derive(Clone)]
pub struct FluidComponent {
    /// The fluid simulation.
    pub fluid: SPHFluid,
    /// Whether the simulation is active.
    pub active: bool,
    /// The extracted surface mesh (updated after each step).
    pub surface_mesh: Option<FluidSurfaceMesh>,
    /// Whether to extract surface mesh each frame.
    pub extract_surface: bool,
    /// Cell size for marching cubes extraction.
    pub surface_cell_size: f32,
    /// Iso-surface threshold for marching cubes.
    pub iso_threshold: f32,
}

impl FluidComponent {
    /// Create a new fluid component.
    pub fn new(fluid: SPHFluid) -> Self {
        Self {
            fluid,
            active: true,
            surface_mesh: None,
            extract_surface: false,
            surface_cell_size: 0.05,
            iso_threshold: 0.5,
        }
    }
}

/// System that steps all fluid simulations each frame.
pub struct FluidSystem {
    /// Fixed time step.
    pub fixed_timestep: f32,
    /// Accumulated time.
    time_accumulator: f32,
}

impl Default for FluidSystem {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 120.0,
            time_accumulator: 0.0,
        }
    }
}

impl FluidSystem {
    /// Create a new fluid system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all fluid simulations.
    pub fn update(&mut self, dt: f32, fluids: &mut [FluidComponent]) {
        self.time_accumulator += dt;
        let mut steps = 0u32;

        while self.time_accumulator >= self.fixed_timestep && steps < 4 {
            for fluid in fluids.iter_mut() {
                if !fluid.active {
                    continue;
                }
                fluid.fluid.step(self.fixed_timestep);
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        if self.time_accumulator > self.fixed_timestep {
            self.time_accumulator = 0.0;
        }

        // Extract surface meshes
        for fluid in fluids.iter_mut() {
            if fluid.active && fluid.extract_surface {
                fluid.surface_mesh = Some(extract_surface_mesh(
                    &fluid.fluid,
                    fluid.surface_cell_size,
                    fluid.iso_threshold,
                ));
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
    fn test_poly6_kernel() {
        let kernel = Poly6Kernel::new(0.1);
        // At r=0, kernel should be maximum
        let w0 = kernel.evaluate(0.0);
        assert!(w0 > 0.0);

        // At r=h, kernel should be 0
        let wh = kernel.evaluate(0.01); // h^2 = 0.01
        assert!(wh.abs() < 1e-6);

        // At r > h, kernel should be 0
        let wo = kernel.evaluate(0.02);
        assert_eq!(wo, 0.0);
    }

    #[test]
    fn test_spiky_gradient() {
        let kernel = SpikyKernel::new(0.1);
        // At r=0, gradient should be zero (degenerate)
        let g0 = kernel.gradient(Vec3::ZERO, 0.0);
        assert_eq!(g0, Vec3::ZERO);

        // At r = h/2, gradient should be non-zero
        let r = Vec3::new(0.05, 0.0, 0.0);
        let g = kernel.gradient(r, 0.05);
        assert!(g.length() > 0.0);
    }

    #[test]
    fn test_viscosity_laplacian() {
        let kernel = ViscosityKernel::new(0.1);
        let lap = kernel.laplacian(0.05);
        assert!(lap > 0.0);

        let lap_out = kernel.laplacian(0.11);
        assert_eq!(lap_out, 0.0);
    }

    #[test]
    fn test_spatial_hash() {
        let mut hash = FluidSpatialHash::new(0.1);
        let particles = vec![
            FluidParticle::new(Vec3::ZERO, 1.0),
            FluidParticle::new(Vec3::new(0.05, 0.0, 0.0), 1.0),
            FluidParticle::new(Vec3::new(10.0, 0.0, 0.0), 1.0),
        ];
        hash.build(&particles);

        let neighbors = hash.query_neighbors(Vec3::ZERO);
        // Should find particles 0 and 1 (both in same or adjacent cell)
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        // Particle 2 is far away
        assert!(!neighbors.contains(&2));
    }

    #[test]
    fn test_add_block() {
        let settings = FluidSettings::default();
        let mut fluid = SPHFluid::new(settings);
        fluid.add_block(Vec3::ZERO, Vec3::splat(0.1));

        assert!(fluid.particle_count() > 0);
    }

    #[test]
    fn test_density_computation() {
        let settings = FluidSettings {
            smoothing_length: 0.2,
            ..Default::default()
        };
        let mut fluid = SPHFluid::new(settings);
        fluid.add_block(Vec3::ZERO, Vec3::splat(0.15));

        // Build neighbors and compute density
        fluid.build_neighbors();
        fluid.compute_density();

        // All particles should have positive density
        for p in &fluid.particles {
            assert!(p.density > 0.0, "Density = {}", p.density);
        }
    }

    #[test]
    fn test_fluid_step() {
        let settings = FluidSettings {
            smoothing_length: 0.2,
            boundary: FluidBoundary {
                min: Vec3::new(-1.0, -1.0, -1.0),
                max: Vec3::new(1.0, 1.0, 1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut fluid = SPHFluid::new(settings);
        fluid.add_block(Vec3::new(-0.1, 0.0, -0.1), Vec3::new(0.1, 0.2, 0.1));

        let initial_y: Vec<f32> = fluid.particles.iter().map(|p| p.position.y).collect();

        fluid.step(1.0 / 120.0);

        // Particles should have moved (under gravity)
        let any_moved = fluid
            .particles
            .iter()
            .enumerate()
            .any(|(i, p)| (p.position.y - initial_y[i]).abs() > 1e-8);
        assert!(any_moved, "Particles should move under gravity");
    }

    #[test]
    fn test_boundary_containment() {
        let settings = FluidSettings {
            smoothing_length: 0.1,
            boundary: FluidBoundary {
                min: Vec3::new(-1.0, -1.0, -1.0),
                max: Vec3::new(1.0, 1.0, 1.0),
                stiffness: 100000.0,
                damping: 5.0,
            },
            ..Default::default()
        };
        let mut fluid = SPHFluid::new(settings);
        fluid.add_block(Vec3::new(-0.1, 0.0, -0.1), Vec3::new(0.1, 0.2, 0.1));

        // Simulate for a while
        for _ in 0..100 {
            fluid.step(1.0 / 120.0);
        }

        // Particles should stay roughly within bounds (generous tolerance for SPH)
        let margin = 0.5;
        for p in &fluid.particles {
            assert!(
                p.position.x > -1.0 - margin && p.position.x < 1.0 + margin,
                "x out of bounds: {}",
                p.position.x
            );
            assert!(
                p.position.y > -1.0 - margin && p.position.y < 1.0 + margin,
                "y out of bounds: {}",
                p.position.y
            );
        }
    }

    #[test]
    fn test_kinetic_energy() {
        let settings = FluidSettings::default();
        let mut fluid = SPHFluid::new(settings);
        fluid.add_particle(Vec3::ZERO);
        fluid.particles[0].velocity = Vec3::new(1.0, 0.0, 0.0);

        let ke = fluid.kinetic_energy();
        let expected = 0.5 * fluid.particles[0].mass * 1.0;
        assert!((ke - expected).abs() < 1e-6);
    }

    #[test]
    fn test_surface_mesh_extraction() {
        let settings = FluidSettings {
            smoothing_length: 0.2,
            ..Default::default()
        };
        let mut fluid = SPHFluid::new(settings);
        fluid.add_block(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1));

        // Compute density first
        fluid.build_neighbors();
        fluid.compute_density();

        let mesh = extract_surface_mesh(&fluid, 0.05, 0.3);
        // Should produce some geometry (exact count depends on configuration)
        // Just verify it doesn't crash and produces valid output
        assert!(mesh.indices.len() % 3 == 0);
    }

    #[test]
    fn test_fluid_component() {
        let settings = FluidSettings::default();
        let fluid = SPHFluid::new(settings);
        let component = FluidComponent::new(fluid);
        assert!(component.active);
    }

    #[test]
    fn test_fluid_system() {
        let settings = FluidSettings::default();
        let mut fluid = SPHFluid::new(settings);
        fluid.add_particle(Vec3::new(0.0, 1.0, 0.0));

        let mut fluids = vec![FluidComponent::new(fluid)];
        let mut system = FluidSystem::new();

        system.update(1.0 / 60.0, &mut fluids);
        // Should not panic
    }
}
