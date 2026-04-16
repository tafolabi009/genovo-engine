//! Soft body physics simulation using Finite Element Method (FEM) and shape matching.
//!
//! This module implements deformable mesh simulation with two approaches:
//! - **FEM (co-rotational linear elasticity)**: physically accurate deformation using
//!   tetrahedral meshes, deformation gradients, polar decomposition, and Hooke's law.
//! - **Shape matching (Mueller et al.)**: faster, more stable alternative that matches
//!   the deformed shape to the original rest shape.
//!
//! Features:
//! - Tetrahedral mesh representation (nodes + tetrahedra)
//! - Volume preservation pressure constraint
//! - Collision with rigid bodies
//! - Configurable material properties (Young's modulus, Poisson's ratio)

use std::collections::HashMap;

use glam::{Mat3, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon to avoid division by zero.
const EPSILON: f32 = 1e-8;
/// Maximum number of solver iterations for FEM.
const DEFAULT_FEM_ITERATIONS: usize = 10;

// ---------------------------------------------------------------------------
// Soft body settings
// ---------------------------------------------------------------------------

/// Material properties and simulation settings for a soft body.
#[derive(Debug, Clone)]
pub struct SoftBodySettings {
    /// Young's modulus (stiffness). Higher = stiffer. Units: Pa.
    /// Typical: rubber ~1e6, soft tissue ~1e4, jelly ~1e3.
    pub young_modulus: f32,
    /// Poisson's ratio [0, 0.5). Ratio of transverse to axial strain.
    /// 0.0 = no lateral expansion, 0.499 = nearly incompressible.
    pub poisson_ratio: f32,
    /// Velocity damping factor [0, 1].
    pub damping: f32,
    /// Material density in kg/m^3.
    pub density: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Number of solver iterations per step.
    pub solver_iterations: usize,
    /// Whether to use volume preservation (pressure constraint).
    pub volume_preservation: bool,
    /// Pressure coefficient for volume preservation.
    pub pressure_coefficient: f32,
    /// Whether to use shape matching instead of FEM.
    pub use_shape_matching: bool,
    /// Shape matching stiffness [0, 1].
    pub shape_matching_stiffness: f32,
    /// Time step.
    pub time_step: f32,
}

impl Default for SoftBodySettings {
    fn default() -> Self {
        Self {
            young_modulus: 10_000.0,
            poisson_ratio: 0.3,
            damping: 0.01,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_iterations: DEFAULT_FEM_ITERATIONS,
            volume_preservation: true,
            pressure_coefficient: 1.0,
            use_shape_matching: false,
            shape_matching_stiffness: 0.5,
            time_step: 1.0 / 60.0,
        }
    }
}

impl SoftBodySettings {
    /// Compute the first Lame parameter (lambda) from Young's modulus and Poisson's ratio.
    ///   lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    pub fn lame_lambda(&self) -> f32 {
        let e = self.young_modulus;
        let nu = self.poisson_ratio.min(0.499);
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Compute the second Lame parameter (mu, shear modulus) from Young's modulus and Poisson's ratio.
    ///   mu = E / (2 * (1 + nu))
    pub fn lame_mu(&self) -> f32 {
        let e = self.young_modulus;
        let nu = self.poisson_ratio.min(0.499);
        e / (2.0 * (1.0 + nu))
    }
}

// ---------------------------------------------------------------------------
// Node (vertex in the soft body)
// ---------------------------------------------------------------------------

/// A single node (vertex) in the soft body mesh.
#[derive(Debug, Clone)]
pub struct SoftBodyNode {
    /// Current world-space position.
    pub position: Vec3,
    /// Previous position for Verlet integration.
    pub prev_position: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
    /// Accumulated force for the current step.
    pub force: Vec3,
    /// Rest (undeformed) position.
    pub rest_position: Vec3,
    /// Node mass in kg.
    pub mass: f32,
    /// Inverse mass (0 for fixed nodes).
    pub inv_mass: f32,
    /// Whether this node is fixed (immovable).
    pub fixed: bool,
    /// Surface normal (for rendering and collision).
    pub normal: Vec3,
}

impl SoftBodyNode {
    /// Create a new node at the given position.
    pub fn new(position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > EPSILON { 1.0 / mass } else { 0.0 };
        Self {
            position,
            prev_position: position,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            rest_position: position,
            mass,
            inv_mass,
            fixed: false,
            normal: Vec3::Y,
        }
    }

    /// Fix this node in place.
    pub fn fix(&mut self) {
        self.fixed = true;
        self.inv_mass = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tetrahedron
// ---------------------------------------------------------------------------

/// A single tetrahedron in the FEM mesh, connecting four nodes.
#[derive(Debug, Clone)]
pub struct Tetrahedron {
    /// Indices of the four nodes forming this tetrahedron.
    pub nodes: [usize; 4],
    /// Rest volume of this tetrahedron.
    pub rest_volume: f32,
    /// Inverse of the rest shape matrix (Dm^-1), precomputed.
    pub dm_inverse: Mat3,
    /// Precomputed shape function gradients (for force calculation).
    /// grad_phi[i] for i=0..3, where grad_phi[3] = -sum(grad_phi[0..3])
    pub shape_gradients: [Vec3; 4],
}

impl Tetrahedron {
    /// Compute and store the rest-state shape matrix inverse and volume.
    ///
    /// The shape matrix Dm = [x1-x0, x2-x0, x3-x0] where x0..x3 are rest positions.
    pub fn precompute(&mut self, nodes: &[SoftBodyNode]) {
        let x0 = nodes[self.nodes[0]].rest_position;
        let x1 = nodes[self.nodes[1]].rest_position;
        let x2 = nodes[self.nodes[2]].rest_position;
        let x3 = nodes[self.nodes[3]].rest_position;

        // Shape matrix columns
        let d1 = x1 - x0;
        let d2 = x2 - x0;
        let d3 = x3 - x0;

        let dm = Mat3::from_cols(d1, d2, d3);
        let det = dm.determinant();

        // Volume = |det(Dm)| / 6
        self.rest_volume = det.abs() / 6.0;

        // Inverse of the rest shape matrix
        if det.abs() > EPSILON {
            self.dm_inverse = dm.inverse();
        } else {
            self.dm_inverse = Mat3::ZERO;
        }

        // Shape function gradients (for co-rotational FEM)
        // grad_phi_i = row i of Dm^-1 (for i=1,2,3, node 0 gradient is -sum)
        let inv_t = self.dm_inverse.transpose();
        self.shape_gradients[1] = inv_t.x_axis;
        self.shape_gradients[2] = inv_t.y_axis;
        self.shape_gradients[3] = inv_t.z_axis;
        self.shape_gradients[0] =
            -(self.shape_gradients[1] + self.shape_gradients[2] + self.shape_gradients[3]);
    }

    /// Compute the current volume of this tetrahedron.
    pub fn current_volume(&self, nodes: &[SoftBodyNode]) -> f32 {
        let x0 = nodes[self.nodes[0]].position;
        let x1 = nodes[self.nodes[1]].position;
        let x2 = nodes[self.nodes[2]].position;
        let x3 = nodes[self.nodes[3]].position;

        let d1 = x1 - x0;
        let d2 = x2 - x0;
        let d3 = x3 - x0;

        let dm = Mat3::from_cols(d1, d2, d3);
        dm.determinant().abs() / 6.0
    }

    /// Compute the deformation gradient F = Ds * Dm^-1
    /// where Ds = [x1-x0, x2-x0, x3-x0] is the deformed shape matrix.
    pub fn deformation_gradient(&self, nodes: &[SoftBodyNode]) -> Mat3 {
        let x0 = nodes[self.nodes[0]].position;
        let x1 = nodes[self.nodes[1]].position;
        let x2 = nodes[self.nodes[2]].position;
        let x3 = nodes[self.nodes[3]].position;

        let ds = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
        ds * self.dm_inverse
    }
}

// ---------------------------------------------------------------------------
// Surface triangle (for rendering)
// ---------------------------------------------------------------------------

/// A surface triangle referencing three node indices.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceTriangle {
    pub indices: [usize; 3],
}

// ---------------------------------------------------------------------------
// Polar decomposition
// ---------------------------------------------------------------------------

/// Extract the rotation component R from deformation gradient F using polar decomposition.
///
/// F = R * S where R is a rotation and S is symmetric.
/// We use iterative polar decomposition: R_{n+1} = 0.5 * (F_n + F_n^{-T})
/// converging to the rotation part.
fn polar_decomposition_rotation(f: Mat3) -> Mat3 {
    let mut r = f;

    for _ in 0..10 {
        let det = r.determinant();
        if det.abs() < EPSILON {
            return Mat3::IDENTITY;
        }

        let r_inv_t = r.inverse().transpose();
        let r_new = (r + r_inv_t) * 0.5;

        // Check convergence
        let diff = r_new.x_axis - r.x_axis;
        if diff.length_squared() < 1e-10 {
            r = r_new;
            break;
        }
        r = r_new;
    }

    // Ensure it's a proper rotation (det = 1)
    let det = r.determinant();
    if det < 0.0 {
        r = Mat3::from_cols(-r.x_axis, r.y_axis, r.z_axis);
    }

    r
}

// ---------------------------------------------------------------------------
// SoftBody — main simulation structure
// ---------------------------------------------------------------------------

/// A deformable soft body simulated with FEM or shape matching.
#[derive(Debug, Clone)]
pub struct SoftBody {
    /// All nodes in the mesh.
    pub nodes: Vec<SoftBodyNode>,
    /// All tetrahedra in the mesh.
    pub tetrahedra: Vec<Tetrahedron>,
    /// Surface triangles for rendering.
    pub surface_triangles: Vec<SurfaceTriangle>,
    /// Material and simulation settings.
    pub settings: SoftBodySettings,
    /// Total rest volume (sum of all tetrahedra rest volumes).
    pub total_rest_volume: f32,
    /// Center of mass (rest state).
    rest_center_of_mass: Vec3,
    /// Running simulation time.
    sim_time: f32,
}

impl SoftBody {
    /// Create a new soft body from nodes, tetrahedra, and surface triangles.
    pub fn new(
        nodes: Vec<SoftBodyNode>,
        tetrahedra: Vec<Tetrahedron>,
        surface_triangles: Vec<SurfaceTriangle>,
        settings: SoftBodySettings,
    ) -> Self {
        let mut body = Self {
            nodes,
            tetrahedra,
            surface_triangles,
            settings,
            total_rest_volume: 0.0,
            rest_center_of_mass: Vec3::ZERO,
            sim_time: 0.0,
        };
        body.precompute();
        body
    }

    /// Precompute all per-tetrahedron data (inverse shape matrices, volumes).
    fn precompute(&mut self) {
        for tet in &mut self.tetrahedra {
            tet.precompute(&self.nodes);
        }
        self.total_rest_volume = self.tetrahedra.iter().map(|t| t.rest_volume).sum();

        // Compute rest center of mass
        let total_mass: f32 = self.nodes.iter().map(|n| n.mass).sum();
        if total_mass > EPSILON {
            let weighted_sum: Vec3 = self
                .nodes
                .iter()
                .map(|n| n.rest_position * n.mass)
                .fold(Vec3::ZERO, |acc, v| acc + v);
            self.rest_center_of_mass = weighted_sum / total_mass;
        }
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of tetrahedra.
    pub fn tet_count(&self) -> usize {
        self.tetrahedra.len()
    }

    /// Compute the current total volume.
    pub fn current_volume(&self) -> f32 {
        self.tetrahedra
            .iter()
            .map(|t| t.current_volume(&self.nodes))
            .sum()
    }

    /// Compute the current center of mass.
    pub fn center_of_mass(&self) -> Vec3 {
        let total_mass: f32 = self.nodes.iter().map(|n| n.mass).sum();
        if total_mass < EPSILON {
            return Vec3::ZERO;
        }
        let weighted_sum: Vec3 = self
            .nodes
            .iter()
            .map(|n| n.position * n.mass)
            .fold(Vec3::ZERO, |acc, v| acc + v);
        weighted_sum / total_mass
    }

    /// Step the soft body simulation forward.
    pub fn step(&mut self, dt: f32) {
        let dt = if dt > 0.0 { dt } else { self.settings.time_step };
        self.sim_time += dt;

        // Apply gravity
        self.apply_gravity();

        if self.settings.use_shape_matching {
            // Shape matching path
            self.integrate_velocities(dt);
            self.integrate_positions(dt);
            self.shape_matching();
        } else {
            // FEM path
            self.compute_fem_forces();

            if self.settings.volume_preservation {
                self.apply_pressure_forces();
            }

            self.integrate_velocities(dt);
            self.integrate_positions(dt);
        }

        // Apply damping
        self.apply_damping();

        // Compute surface normals
        self.compute_surface_normals();

        // Clear forces
        for node in &mut self.nodes {
            node.force = Vec3::ZERO;
        }
    }

    // -----------------------------------------------------------------------
    // Force computation
    // -----------------------------------------------------------------------

    fn apply_gravity(&mut self) {
        let gravity = self.settings.gravity;
        for node in &mut self.nodes {
            if !node.fixed {
                node.force += gravity * node.mass;
            }
        }
    }

    /// Compute FEM forces using co-rotational linear elasticity.
    ///
    /// For each tetrahedron:
    /// 1. Compute deformation gradient F = Ds * Dm^-1
    /// 2. Polar decomposition: extract rotation R from F
    /// 3. Compute strain: epsilon = R^T * F - I
    /// 4. Compute stress: sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon (Hooke's law)
    /// 5. Compute force per node: -volume * P * grad_phi
    ///    where P = R * sigma (first Piola-Kirchhoff stress)
    fn compute_fem_forces(&mut self) {
        let lambda = self.settings.lame_lambda();
        let mu = self.settings.lame_mu();

        for tet_idx in 0..self.tetrahedra.len() {
            let tet = &self.tetrahedra[tet_idx];
            if tet.rest_volume < EPSILON {
                continue;
            }

            // 1. Deformation gradient
            let f = tet.deformation_gradient(&self.nodes);

            // 2. Polar decomposition: F = R * S
            let r = polar_decomposition_rotation(f);

            // 3. Strain: epsilon = R^T * F - I
            let strain = r.transpose() * f - Mat3::IDENTITY;

            // 4. Stress (Hooke's law): sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
            let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
            let stress =
                Mat3::from_diagonal(Vec3::splat(lambda * trace)) + strain * (2.0 * mu);

            // 5. First Piola-Kirchhoff stress: P = R * sigma
            let piola = r * stress;

            // 6. Force per node: f_i = -V_0 * P * grad_phi_i
            let vol = tet.rest_volume;
            for i in 0..4 {
                let grad = tet.shape_gradients[i];
                let force = -(piola * grad) * vol;
                let node_idx = tet.nodes[i];
                self.nodes[node_idx].force += force;
            }
        }
    }

    /// Apply pressure forces for volume preservation.
    ///
    /// Computes the ratio of current volume to rest volume and applies
    /// outward forces proportional to the deficit.
    fn apply_pressure_forces(&mut self) {
        let current_vol = self.current_volume();
        let rest_vol = self.total_rest_volume;
        if rest_vol < EPSILON {
            return;
        }

        let volume_ratio = current_vol / rest_vol;
        let pressure = self.settings.pressure_coefficient * (1.0 - volume_ratio);

        // Apply pressure force along surface normals
        for tet_idx in 0..self.tetrahedra.len() {
            let tet = &self.tetrahedra[tet_idx];

            // Compute face normals for each face of the tetrahedron
            let faces: [[usize; 3]; 4] = [
                [tet.nodes[0], tet.nodes[1], tet.nodes[2]],
                [tet.nodes[0], tet.nodes[1], tet.nodes[3]],
                [tet.nodes[0], tet.nodes[2], tet.nodes[3]],
                [tet.nodes[1], tet.nodes[2], tet.nodes[3]],
            ];

            for face in &faces {
                let p0 = self.nodes[face[0]].position;
                let p1 = self.nodes[face[1]].position;
                let p2 = self.nodes[face[2]].position;

                let e1 = p1 - p0;
                let e2 = p2 - p0;
                let face_normal = e1.cross(e2);
                let area = face_normal.length() * 0.5;

                if area < EPSILON {
                    continue;
                }

                let normal = face_normal / (area * 2.0);
                let force = normal * pressure * area / 3.0;

                for &idx in face {
                    self.nodes[idx].force += force;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Shape matching (Mueller et al. 2005)
    // -----------------------------------------------------------------------

    /// Shape matching: compute the optimal rotation and translation that matches
    /// the current positions to the rest shape, then blend toward that configuration.
    ///
    /// Algorithm:
    /// 1. Compute current center of mass
    /// 2. Compute relative positions in rest and current frames
    /// 3. Compute the optimal rotation via polar decomposition of A = sum(m_i * q_i * p_i^T)
    /// 4. Compute goal positions: g_i = R * (rest_i - rest_com) + current_com
    /// 5. Blend: position = lerp(position, goal, stiffness)
    fn shape_matching(&mut self) {
        let stiffness = self.settings.shape_matching_stiffness;

        // 1. Current center of mass
        let current_com = self.center_of_mass();

        // 2. Compute A = sum(m_i * q_i * p_i^T) where
        //    q_i = current_pos - current_com
        //    p_i = rest_pos - rest_com
        let mut a = Mat3::ZERO;
        for node in &self.nodes {
            let q = node.position - current_com;
            let p = node.rest_position - self.rest_center_of_mass;
            // Outer product: q * p^T, weighted by mass
            let m = node.mass;
            a += Mat3::from_cols(q * p.x * m, q * p.y * m, q * p.z * m);
        }

        // 3. Polar decomposition to extract rotation
        let r = polar_decomposition_rotation(a);

        // 4 & 5. Compute goal positions and blend
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }
            let goal = r * (node.rest_position - self.rest_center_of_mass) + current_com;
            node.position = node.position + (goal - node.position) * stiffness;
        }
    }

    // -----------------------------------------------------------------------
    // Integration
    // -----------------------------------------------------------------------

    fn integrate_velocities(&mut self, dt: f32) {
        for node in &mut self.nodes {
            if node.fixed {
                node.velocity = Vec3::ZERO;
                continue;
            }
            let acceleration = node.force * node.inv_mass;
            node.velocity += acceleration * dt;
        }
    }

    fn integrate_positions(&mut self, dt: f32) {
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }
            node.prev_position = node.position;
            node.position += node.velocity * dt;
        }
    }

    fn apply_damping(&mut self) {
        let damping = 1.0 - self.settings.damping;
        for node in &mut self.nodes {
            node.velocity *= damping;
        }
    }

    // -----------------------------------------------------------------------
    // Surface normals
    // -----------------------------------------------------------------------

    fn compute_surface_normals(&mut self) {
        for node in &mut self.nodes {
            node.normal = Vec3::ZERO;
        }

        for tri in &self.surface_triangles {
            let p0 = self.nodes[tri.indices[0]].position;
            let p1 = self.nodes[tri.indices[1]].position;
            let p2 = self.nodes[tri.indices[2]].position;

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let normal = e1.cross(e2);

            for &idx in &tri.indices {
                self.nodes[idx].normal += normal;
            }
        }

        for node in &mut self.nodes {
            let len = node.normal.length();
            if len > EPSILON {
                node.normal /= len;
            } else {
                node.normal = Vec3::Y;
            }
        }
    }

    /// Apply an external impulse at a world position, affecting nearby nodes.
    pub fn apply_impulse(&mut self, world_pos: Vec3, impulse: Vec3, radius: f32) {
        let radius_sq = radius * radius;
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }
            let diff = node.position - world_pos;
            let dist_sq = diff.length_squared();
            if dist_sq < radius_sq {
                let falloff = 1.0 - (dist_sq / radius_sq).sqrt();
                node.velocity += impulse * falloff * node.inv_mass;
            }
        }
    }

    /// Get all node positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.nodes.iter().map(|n| n.position).collect()
    }

    /// Get all node normals.
    pub fn normals(&self) -> Vec<Vec3> {
        self.nodes.iter().map(|n| n.normal).collect()
    }

    /// Collide the soft body with a plane.
    pub fn collide_plane(&mut self, point: Vec3, normal: Vec3, friction: f32) {
        let n = normal.normalize();
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }
            let dist = (node.position - point).dot(n);
            if dist < 0.0 {
                node.position -= n * dist;
                let vn = node.velocity.dot(n);
                if vn < 0.0 {
                    let vel_normal = n * vn;
                    let vel_tangent = node.velocity - vel_normal;
                    node.velocity = vel_tangent * (1.0 - friction) - vel_normal * 0.1;
                }
            }
        }
    }

    /// Collide the soft body with a sphere.
    pub fn collide_sphere(&mut self, center: Vec3, radius: f32, friction: f32) {
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }
            let diff = node.position - center;
            let dist = diff.length();
            if dist < radius && dist > EPSILON {
                let normal = diff / dist;
                node.position = center + normal * radius;
                let vn = node.velocity.dot(normal);
                if vn < 0.0 {
                    let vel_normal = normal * vn;
                    let vel_tangent = node.velocity - vel_normal;
                    node.velocity = vel_tangent * (1.0 - friction);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tetrahedral mesh generation
// ---------------------------------------------------------------------------

/// A tetrahedral mesh suitable for soft body simulation.
#[derive(Debug, Clone)]
pub struct TetMesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Tetrahedra as groups of 4 vertex indices.
    pub tetrahedra: Vec<[usize; 4]>,
    /// Surface triangles for rendering.
    pub surface_triangles: Vec<[usize; 3]>,
}

/// Generate a simple tetrahedral mesh from a bounding box.
///
/// Creates a regular grid of tetrahedra filling the box, suitable for
/// testing and simple soft body shapes.
///
/// Each cube cell is divided into 5 tetrahedra (BCC lattice subdivision).
pub fn generate_box_tet_mesh(
    half_extents: Vec3,
    resolution: usize,
) -> TetMesh {
    let res = resolution.max(1);
    let cell_size = half_extents * 2.0 / res as f32;
    let origin = -half_extents;

    // Generate grid vertices
    let verts_per_axis = res + 1;
    let mut vertices = Vec::new();
    let mut vert_map: HashMap<(usize, usize, usize), usize> = HashMap::new();

    for z in 0..verts_per_axis {
        for y in 0..verts_per_axis {
            for x in 0..verts_per_axis {
                let idx = vertices.len();
                let pos = origin + Vec3::new(
                    x as f32 * cell_size.x,
                    y as f32 * cell_size.y,
                    z as f32 * cell_size.z,
                );
                vertices.push(pos);
                vert_map.insert((x, y, z), idx);
            }
        }
    }

    // Generate tetrahedra (5 per cube cell)
    let mut tetrahedra = Vec::new();

    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                // 8 corners of the cube
                let v000 = vert_map[&(x, y, z)];
                let v100 = vert_map[&(x + 1, y, z)];
                let v010 = vert_map[&(x, y + 1, z)];
                let v110 = vert_map[&(x + 1, y + 1, z)];
                let v001 = vert_map[&(x, y, z + 1)];
                let v101 = vert_map[&(x + 1, y, z + 1)];
                let v011 = vert_map[&(x, y + 1, z + 1)];
                let v111 = vert_map[&(x + 1, y + 1, z + 1)];

                // 5-tet decomposition of a cube (alternating to avoid bias)
                let parity = (x + y + z) % 2;
                if parity == 0 {
                    tetrahedra.push([v000, v100, v010, v001]);
                    tetrahedra.push([v100, v110, v010, v111]);
                    tetrahedra.push([v001, v101, v100, v111]);
                    tetrahedra.push([v001, v011, v010, v111]);
                    tetrahedra.push([v000, v100, v001, v010]); // center bridge
                } else {
                    tetrahedra.push([v100, v000, v110, v101]);
                    tetrahedra.push([v000, v010, v110, v011]);
                    tetrahedra.push([v101, v111, v110, v011]);
                    tetrahedra.push([v000, v101, v011, v001]);
                    tetrahedra.push([v000, v110, v101, v011]); // center bridge
                }
            }
        }
    }

    // Identify surface triangles (faces shared by only one tetrahedron)
    let mut face_count: HashMap<[usize; 3], usize> = HashMap::new();
    for tet in &tetrahedra {
        let faces: [[usize; 3]; 4] = [
            sorted_face(tet[0], tet[1], tet[2]),
            sorted_face(tet[0], tet[1], tet[3]),
            sorted_face(tet[0], tet[2], tet[3]),
            sorted_face(tet[1], tet[2], tet[3]),
        ];
        for face in &faces {
            *face_count.entry(*face).or_default() += 1;
        }
    }

    let surface_triangles: Vec<[usize; 3]> = face_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(face, _)| face)
        .collect();

    TetMesh {
        vertices,
        tetrahedra,
        surface_triangles,
    }
}

/// Sort three indices for face comparison.
fn sorted_face(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut f = [a, b, c];
    f.sort();
    f
}

/// Create a `SoftBody` from a `TetMesh` with given settings.
pub fn create_soft_body(tet_mesh: &TetMesh, settings: SoftBodySettings) -> SoftBody {
    let density = settings.density;

    // Create nodes with mass = 0 initially
    let mut nodes: Vec<SoftBodyNode> = tet_mesh
        .vertices
        .iter()
        .map(|&pos| SoftBodyNode::new(pos, 0.0))
        .collect();

    // Create tetrahedra and distribute mass
    let mut tetrahedra = Vec::new();
    for tet_indices in &tet_mesh.tetrahedra {
        let mut tet = Tetrahedron {
            nodes: *tet_indices,
            rest_volume: 0.0,
            dm_inverse: Mat3::ZERO,
            shape_gradients: [Vec3::ZERO; 4],
        };
        tet.precompute(&nodes);

        // Distribute mass to nodes: each node gets 1/4 of the tet mass
        let tet_mass = tet.rest_volume * density;
        let node_mass = tet_mass / 4.0;
        for &ni in &tet.nodes {
            nodes[ni].mass += node_mass;
        }

        tetrahedra.push(tet);
    }

    // Update inverse masses
    for node in &mut nodes {
        node.inv_mass = if node.mass > EPSILON {
            1.0 / node.mass
        } else {
            0.0
        };
    }

    let surface_triangles: Vec<SurfaceTriangle> = tet_mesh
        .surface_triangles
        .iter()
        .map(|&indices| SurfaceTriangle { indices })
        .collect();

    SoftBody::new(nodes, tetrahedra, surface_triangles, settings)
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a soft body to an entity.
#[derive(Clone)]
pub struct SoftBodyComponent {
    /// The soft body simulation.
    pub body: SoftBody,
    /// Whether the simulation is active.
    pub active: bool,
}

impl SoftBodyComponent {
    /// Create a new soft body component.
    pub fn new(body: SoftBody) -> Self {
        Self { body, active: true }
    }

    /// Create a simple box-shaped soft body.
    pub fn box_shape(half_extents: Vec3, resolution: usize, settings: SoftBodySettings) -> Self {
        let tet_mesh = generate_box_tet_mesh(half_extents, resolution);
        let body = create_soft_body(&tet_mesh, settings);
        Self::new(body)
    }
}

/// System that steps all soft body simulations each frame.
pub struct SoftBodySystem {
    /// Fixed time step.
    pub fixed_timestep: f32,
    /// Accumulated time.
    time_accumulator: f32,
}

impl Default for SoftBodySystem {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 60.0,
            time_accumulator: 0.0,
        }
    }
}

impl SoftBodySystem {
    /// Create a new soft body system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all soft bodies.
    pub fn update(&mut self, dt: f32, bodies: &mut [SoftBodyComponent]) {
        self.time_accumulator += dt;
        let mut steps = 0u32;

        while self.time_accumulator >= self.fixed_timestep && steps < 4 {
            for body in bodies.iter_mut() {
                if body.active {
                    body.body.step(self.fixed_timestep);
                }
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        if self.time_accumulator > self.fixed_timestep {
            self.time_accumulator = 0.0;
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tet() -> TetMesh {
        // A single tetrahedron
        TetMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            tetrahedra: vec![[0, 1, 2, 3]],
            surface_triangles: vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        }
    }

    #[test]
    fn test_lame_parameters() {
        let settings = SoftBodySettings {
            young_modulus: 10_000.0,
            poisson_ratio: 0.3,
            ..Default::default()
        };
        let lambda = settings.lame_lambda();
        let mu = settings.lame_mu();

        // lambda = 10000 * 0.3 / (1.3 * 0.4) = 3000 / 0.52 ~ 5769
        assert!((lambda - 5769.23).abs() < 1.0);
        // mu = 10000 / (2 * 1.3) ~ 3846
        assert!((mu - 3846.15).abs() < 1.0);
    }

    #[test]
    fn test_tetrahedron_volume() {
        let tet_mesh = make_simple_tet();
        let nodes: Vec<SoftBodyNode> = tet_mesh
            .vertices
            .iter()
            .map(|&p| SoftBodyNode::new(p, 1.0))
            .collect();

        let mut tet = Tetrahedron {
            nodes: [0, 1, 2, 3],
            rest_volume: 0.0,
            dm_inverse: Mat3::ZERO,
            shape_gradients: [Vec3::ZERO; 4],
        };
        tet.precompute(&nodes);

        // Volume of this tetrahedron should be 1/6
        assert!((tet.rest_volume - 1.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_deformation_gradient_identity() {
        let tet_mesh = make_simple_tet();
        let nodes: Vec<SoftBodyNode> = tet_mesh
            .vertices
            .iter()
            .map(|&p| SoftBodyNode::new(p, 1.0))
            .collect();

        let mut tet = Tetrahedron {
            nodes: [0, 1, 2, 3],
            rest_volume: 0.0,
            dm_inverse: Mat3::ZERO,
            shape_gradients: [Vec3::ZERO; 4],
        };
        tet.precompute(&nodes);

        let f = tet.deformation_gradient(&nodes);
        let diff = f - Mat3::IDENTITY;
        assert!(diff.x_axis.length() < 1e-4);
        assert!(diff.y_axis.length() < 1e-4);
        assert!(diff.z_axis.length() < 1e-4);
    }

    #[test]
    fn test_polar_decomposition_identity() {
        let r = polar_decomposition_rotation(Mat3::IDENTITY);
        let diff = r - Mat3::IDENTITY;
        assert!(diff.x_axis.length() < 1e-4);
    }

    #[test]
    fn test_polar_decomposition_rotation() {
        let angle = 0.5;
        let rot_mat = Mat3::from_rotation_z(angle);
        let r = polar_decomposition_rotation(rot_mat);
        let diff = r - rot_mat;
        assert!(diff.x_axis.length() < 1e-3);
    }

    #[test]
    fn test_soft_body_creation() {
        let tet_mesh = make_simple_tet();
        let settings = SoftBodySettings::default();
        let body = create_soft_body(&tet_mesh, settings);

        assert_eq!(body.node_count(), 4);
        assert_eq!(body.tet_count(), 1);
        assert!(body.total_rest_volume > 0.0);
    }

    #[test]
    fn test_soft_body_falls_under_gravity() {
        let tet_mesh = make_simple_tet();
        let settings = SoftBodySettings::default();
        let mut body = create_soft_body(&tet_mesh, settings);

        let initial_y = body.center_of_mass().y;

        for _ in 0..60 {
            body.step(1.0 / 60.0);
        }

        let final_y = body.center_of_mass().y;
        assert!(final_y < initial_y, "Soft body should fall under gravity");
    }

    #[test]
    fn test_soft_body_plane_collision() {
        let tet_mesh = make_simple_tet();
        let settings = SoftBodySettings::default();
        let mut body = create_soft_body(&tet_mesh, settings);

        // Simulate and collide with ground
        for _ in 0..120 {
            body.step(1.0 / 60.0);
            body.collide_plane(Vec3::new(0.0, -2.0, 0.0), Vec3::Y, 0.5);
        }

        // All nodes should be above the plane
        for node in &body.nodes {
            assert!(
                node.position.y >= -2.0 - 0.01,
                "Node below plane: y = {}",
                node.position.y
            );
        }
    }

    #[test]
    fn test_shape_matching() {
        let tet_mesh = make_simple_tet();
        let settings = SoftBodySettings {
            use_shape_matching: true,
            shape_matching_stiffness: 0.8,
            gravity: Vec3::ZERO,
            damping: 0.5,
            ..Default::default()
        };
        let mut body = create_soft_body(&tet_mesh, settings);

        // Deform
        body.nodes[0].position += Vec3::new(1.0, 0.0, 0.0);

        // Multiple steps to allow convergence
        for _ in 0..10 {
            body.step(1.0 / 60.0);
        }

        // The node should have moved back toward its rest position
        let dist = (body.nodes[0].position - body.nodes[0].rest_position).length();
        assert!(dist < 1.0, "Shape matching should restore shape, dist = {}", dist);
    }

    #[test]
    fn test_generate_box_tet_mesh() {
        let mesh = generate_box_tet_mesh(Vec3::ONE, 2);

        assert!(mesh.vertices.len() > 0);
        assert!(mesh.tetrahedra.len() > 0);
        assert!(mesh.surface_triangles.len() > 0);

        // Verify all tet indices are valid
        for tet in &mesh.tetrahedra {
            for &idx in tet {
                assert!(idx < mesh.vertices.len());
            }
        }
    }

    #[test]
    fn test_soft_body_component() {
        let component =
            SoftBodyComponent::box_shape(Vec3::splat(0.5), 2, SoftBodySettings::default());
        assert!(component.active);
        assert!(component.body.node_count() > 0);
    }

    #[test]
    fn test_soft_body_system() {
        let mut system = SoftBodySystem::new();
        let mut bodies = vec![SoftBodyComponent::box_shape(
            Vec3::splat(0.5),
            1,
            SoftBodySettings::default(),
        )];

        let initial_com = bodies[0].body.center_of_mass();
        system.update(1.0 / 60.0, &mut bodies);
        let final_com = bodies[0].body.center_of_mass();

        assert!(final_com.y < initial_com.y);
    }

    #[test]
    fn test_volume_preservation() {
        let tet_mesh = make_simple_tet();
        let settings = SoftBodySettings {
            volume_preservation: true,
            pressure_coefficient: 10.0,
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let mut body = create_soft_body(&tet_mesh, settings);

        let initial_vol = body.current_volume();

        // Compress slightly
        for node in &mut body.nodes {
            node.position *= 0.8;
        }

        // Pressure should push it back
        for _ in 0..60 {
            body.step(1.0 / 60.0);
        }

        let final_vol = body.current_volume();
        // Volume should increase from the compressed state
        let compressed_vol = initial_vol * 0.8_f32.powi(3);
        assert!(
            final_vol > compressed_vol,
            "Volume preservation should increase volume: {} > {}",
            final_vol,
            compressed_vol
        );
    }
}
