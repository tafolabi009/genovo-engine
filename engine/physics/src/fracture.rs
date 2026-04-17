//! Advanced runtime fracture system with stress propagation.
//!
//! Unlike the pre-computed Voronoi system in `destruction`, this module provides:
//! - Runtime fracture generation (no pre-computation needed)
//! - Stress propagation through mesh connectivity
//! - Crack initiation from impact points
//! - Crack propagation along stress lines
//! - Fragment mass and inertia tensor computation
//! - Debris cleanup with lifetime and distance culling
//! - Multi-material fracture with per-material toughness
//! - Fracture pattern generation: radial, branching, shatter

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{Mat3, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for geometric comparisons.
const EPSILON: f32 = 1e-6;
/// Default fracture toughness in Joules/m^2 (energy to create unit crack area).
const DEFAULT_TOUGHNESS: f32 = 500.0;
/// Default maximum number of fragments from a single fracture event.
const DEFAULT_MAX_FRAGMENTS: usize = 32;
/// Default debris lifetime in seconds.
const DEFAULT_DEBRIS_LIFETIME: f32 = 15.0;
/// Default debris distance cull radius from camera/player.
const DEFAULT_DEBRIS_CULL_DISTANCE: f32 = 100.0;
/// Minimum fragment volume threshold; below this the fragment becomes debris.
const MIN_FRAGMENT_VOLUME: f32 = 0.0005;
/// Default stress propagation speed in m/s.
const DEFAULT_STRESS_SPEED: f32 = 1000.0;
/// Maximum stress level before automatic fracture.
const MAX_STRESS: f32 = 1.0e6;
/// Default number of crack propagation iterations per frame.
const DEFAULT_PROPAGATION_STEPS: usize = 8;
/// Minimum crack segment length.
const MIN_CRACK_LENGTH: f32 = 0.01;

// ---------------------------------------------------------------------------
// Fracture material
// ---------------------------------------------------------------------------

/// Material properties that influence fracture behavior.
#[derive(Debug, Clone)]
pub struct FractureMaterial {
    /// Material name.
    pub name: String,
    /// Fracture toughness (energy per unit area to create a crack).
    pub toughness: f32,
    /// Young's modulus (stiffness) in Pa. Determines stress from strain.
    pub youngs_modulus: f32,
    /// Poisson's ratio (0-0.5). Determines lateral contraction.
    pub poisson_ratio: f32,
    /// Density in kg/m^3.
    pub density: f32,
    /// Preferred crack pattern.
    pub crack_pattern: CrackPattern,
    /// Minimum fragment size as a fraction of the original object.
    pub min_fragment_fraction: f32,
    /// Sound hint for fracture audio.
    pub fracture_sound: FractureSound,
    /// Whether the material shatters (many small pieces) or cracks (few large pieces).
    pub shatters: bool,
}

/// Preferred crack propagation pattern for a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrackPattern {
    /// Cracks radiate from impact point (glass-like).
    Radial,
    /// Cracks branch and fork (wood-like).
    Branching,
    /// Complete shatter into many pieces (ceramic-like).
    Shatter,
    /// Regular grid-like fracture (brick-like).
    GridLike,
    /// Random fracture (generic).
    Random,
}

/// Sound hint for fracture audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractureSound {
    /// Glass breaking.
    Glass,
    /// Wood cracking.
    Wood,
    /// Metal tearing.
    Metal,
    /// Stone crumbling.
    Stone,
    /// Ceramic shattering.
    Ceramic,
    /// Ice cracking.
    Ice,
    /// Generic cracking sound.
    Generic,
}

impl FractureMaterial {
    /// Create a glass-like material.
    pub fn glass() -> Self {
        Self {
            name: "glass".to_string(),
            toughness: 50.0,
            youngs_modulus: 70.0e9,
            poisson_ratio: 0.22,
            density: 2500.0,
            crack_pattern: CrackPattern::Radial,
            min_fragment_fraction: 0.01,
            fracture_sound: FractureSound::Glass,
            shatters: true,
        }
    }

    /// Create a wood-like material.
    pub fn wood() -> Self {
        Self {
            name: "wood".to_string(),
            toughness: 2000.0,
            youngs_modulus: 12.0e9,
            poisson_ratio: 0.35,
            density: 600.0,
            crack_pattern: CrackPattern::Branching,
            min_fragment_fraction: 0.05,
            fracture_sound: FractureSound::Wood,
            shatters: false,
        }
    }

    /// Create a stone-like material.
    pub fn stone() -> Self {
        Self {
            name: "stone".to_string(),
            toughness: 300.0,
            youngs_modulus: 50.0e9,
            poisson_ratio: 0.25,
            density: 2700.0,
            crack_pattern: CrackPattern::Random,
            min_fragment_fraction: 0.03,
            fracture_sound: FractureSound::Stone,
            shatters: false,
        }
    }

    /// Create a metal material (hard to fracture).
    pub fn metal() -> Self {
        Self {
            name: "metal".to_string(),
            toughness: 10000.0,
            youngs_modulus: 200.0e9,
            poisson_ratio: 0.30,
            density: 7800.0,
            crack_pattern: CrackPattern::Branching,
            min_fragment_fraction: 0.1,
            fracture_sound: FractureSound::Metal,
            shatters: false,
        }
    }

    /// Create a ceramic material (shatters easily).
    pub fn ceramic() -> Self {
        Self {
            name: "ceramic".to_string(),
            toughness: 100.0,
            youngs_modulus: 300.0e9,
            poisson_ratio: 0.22,
            density: 3900.0,
            crack_pattern: CrackPattern::Shatter,
            min_fragment_fraction: 0.005,
            fracture_sound: FractureSound::Ceramic,
            shatters: true,
        }
    }

    /// Create an ice material.
    pub fn ice() -> Self {
        Self {
            name: "ice".to_string(),
            toughness: 80.0,
            youngs_modulus: 9.0e9,
            poisson_ratio: 0.33,
            density: 917.0,
            crack_pattern: CrackPattern::Radial,
            min_fragment_fraction: 0.02,
            fracture_sound: FractureSound::Ice,
            shatters: true,
        }
    }

    /// Create a concrete material.
    pub fn concrete() -> Self {
        Self {
            name: "concrete".to_string(),
            toughness: 400.0,
            youngs_modulus: 30.0e9,
            poisson_ratio: 0.20,
            density: 2400.0,
            crack_pattern: CrackPattern::Random,
            min_fragment_fraction: 0.03,
            fracture_sound: FractureSound::Stone,
            shatters: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh data structures
// ---------------------------------------------------------------------------

/// A triangle mesh used for fracture operations.
#[derive(Debug, Clone)]
pub struct FractureMeshV2 {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

impl FractureMeshV2 {
    /// Create an empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Get the number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Compute the AABB of the mesh.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            min = min.min(*v);
            max = max.max(*v);
        }
        (min, max)
    }

    /// Compute the centroid of the mesh.
    pub fn centroid(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.vertices.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
        sum / self.vertices.len() as f32
    }

    /// Compute the volume of a closed mesh using signed tetrahedra.
    pub fn volume(&self) -> f32 {
        let mut vol = 0.0_f32;
        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];
            vol += v0.dot(v1.cross(v2)) / 6.0;
        }
        vol.abs()
    }

    /// Compute smooth vertex normals from triangle face normals.
    pub fn compute_normals(&mut self) {
        self.normals = vec![Vec3::ZERO; self.vertices.len()];
        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let e1 = self.vertices[i1] - self.vertices[i0];
            let e2 = self.vertices[i2] - self.vertices[i0];
            let face_normal = e1.cross(e2);
            self.normals[i0] += face_normal;
            self.normals[i1] += face_normal;
            self.normals[i2] += face_normal;
        }
        for n in &mut self.normals {
            let len = n.length();
            if len > EPSILON {
                *n /= len;
            } else {
                *n = Vec3::Y;
            }
        }
    }

    /// Split this mesh by a plane into two halves.
    /// Returns (positive_side, negative_side) meshes.
    pub fn split_by_plane(&self, plane_point: Vec3, plane_normal: Vec3) -> (Self, Self) {
        let normal = plane_normal.normalize_or_zero();
        let mut pos_mesh = FractureMeshV2::new();
        let mut neg_mesh = FractureMeshV2::new();

        // Classify vertices
        let signs: Vec<f32> = self
            .vertices
            .iter()
            .map(|v| (*v - plane_point).dot(normal))
            .collect();

        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;

            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }

            let s0 = signs[i0] >= 0.0;
            let s1 = signs[i1] >= 0.0;
            let s2 = signs[i2] >= 0.0;

            if s0 && s1 && s2 {
                // All positive
                Self::add_triangle(&mut pos_mesh, self.vertices[i0], self.vertices[i1], self.vertices[i2]);
            } else if !s0 && !s1 && !s2 {
                // All negative
                Self::add_triangle(&mut neg_mesh, self.vertices[i0], self.vertices[i1], self.vertices[i2]);
            } else {
                // Triangle spans the plane - clip it
                let verts = [self.vertices[i0], self.vertices[i1], self.vertices[i2]];
                let dists = [signs[i0], signs[i1], signs[i2]];

                let mut pos_verts = Vec::new();
                let mut neg_verts = Vec::new();

                for edge in 0..3 {
                    let a = edge;
                    let b = (edge + 1) % 3;

                    if dists[a] >= 0.0 {
                        pos_verts.push(verts[a]);
                    } else {
                        neg_verts.push(verts[a]);
                    }

                    // Check if edge crosses the plane
                    if (dists[a] >= 0.0) != (dists[b] >= 0.0) {
                        let t_param = dists[a] / (dists[a] - dists[b]);
                        let intersection = verts[a] + (verts[b] - verts[a]) * t_param;
                        pos_verts.push(intersection);
                        neg_verts.push(intersection);
                    }
                }

                // Triangulate the positive side
                if pos_verts.len() >= 3 {
                    for i in 1..(pos_verts.len() - 1) {
                        Self::add_triangle(&mut pos_mesh, pos_verts[0], pos_verts[i], pos_verts[i + 1]);
                    }
                }

                // Triangulate the negative side
                if neg_verts.len() >= 3 {
                    for i in 1..(neg_verts.len() - 1) {
                        Self::add_triangle(&mut neg_mesh, neg_verts[0], neg_verts[i], neg_verts[i + 1]);
                    }
                }
            }
        }

        pos_mesh.compute_normals();
        neg_mesh.compute_normals();

        (pos_mesh, neg_mesh)
    }

    /// Add a triangle to the mesh (with deduplication threshold).
    fn add_triangle(mesh: &mut FractureMeshV2, v0: Vec3, v1: Vec3, v2: Vec3) {
        // Check for degenerate triangles
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let area = e1.cross(e2).length();
        if area < EPSILON * EPSILON {
            return;
        }

        let base = mesh.vertices.len() as u32;
        mesh.vertices.push(v0);
        mesh.vertices.push(v1);
        mesh.vertices.push(v2);
        mesh.indices.push(base);
        mesh.indices.push(base + 1);
        mesh.indices.push(base + 2);
    }
}

// ---------------------------------------------------------------------------
// Stress field
// ---------------------------------------------------------------------------

/// Per-vertex stress information.
#[derive(Debug, Clone)]
pub struct StressField {
    /// Stress tensor at each vertex (stored as principal stress magnitude).
    pub stress: Vec<f32>,
    /// Direction of maximum principal stress at each vertex.
    pub stress_direction: Vec<Vec3>,
    /// Whether each vertex has reached the fracture threshold.
    pub fractured: Vec<bool>,
    /// Material toughness threshold.
    pub threshold: f32,
}

impl StressField {
    /// Create a new stress field for a mesh with the given vertex count.
    pub fn new(vertex_count: usize, threshold: f32) -> Self {
        Self {
            stress: vec![0.0; vertex_count],
            stress_direction: vec![Vec3::ZERO; vertex_count],
            fractured: vec![false; vertex_count],
            threshold,
        }
    }

    /// Apply an impact at a vertex, setting initial stress.
    pub fn apply_impact(&mut self, vertex_index: usize, energy: f32, direction: Vec3) {
        if vertex_index < self.stress.len() {
            self.stress[vertex_index] += energy;
            self.stress_direction[vertex_index] = direction.normalize_or_zero();
            if self.stress[vertex_index] >= self.threshold {
                self.fractured[vertex_index] = true;
            }
        }
    }

    /// Propagate stress through mesh connectivity.
    /// `adjacency[i]` contains the indices of vertices adjacent to vertex i.
    pub fn propagate(
        &mut self,
        adjacency: &[Vec<usize>],
        propagation_factor: f32,
        attenuation: f32,
    ) {
        let count = self.stress.len();
        let mut new_stress = self.stress.clone();
        let mut new_directions = self.stress_direction.clone();

        for i in 0..count {
            if self.stress[i] < EPSILON {
                continue;
            }

            for &neighbor in &adjacency[i] {
                if neighbor >= count {
                    continue;
                }
                let transfer = self.stress[i] * propagation_factor * attenuation;
                if transfer > EPSILON {
                    new_stress[neighbor] += transfer;
                    // Blend stress direction
                    let weight = transfer / (new_stress[neighbor].max(EPSILON));
                    new_directions[neighbor] = new_directions[neighbor] * (1.0 - weight)
                        + self.stress_direction[i] * weight;
                }
            }

            // Attenuate the source
            new_stress[i] *= 1.0 - propagation_factor;
        }

        // Update fractured state
        for i in 0..count {
            if new_stress[i] >= self.threshold {
                self.fractured[i] = true;
            }
        }

        self.stress = new_stress;
        self.stress_direction = new_directions;
    }

    /// Get the maximum stress in the field.
    pub fn max_stress(&self) -> f32 {
        self.stress.iter().copied().fold(0.0_f32, f32::max)
    }

    /// Get the number of fractured vertices.
    pub fn fractured_count(&self) -> usize {
        self.fractured.iter().filter(|&&f| f).count()
    }

    /// Reset all stress values.
    pub fn reset(&mut self) {
        self.stress.fill(0.0);
        self.stress_direction.fill(Vec3::ZERO);
        self.fractured.fill(false);
    }
}

// ---------------------------------------------------------------------------
// Crack system
// ---------------------------------------------------------------------------

/// A single crack segment.
#[derive(Debug, Clone)]
pub struct CrackSegment {
    /// Start position.
    pub start: Vec3,
    /// End position.
    pub end: Vec3,
    /// Crack plane normal (defines the cutting plane at this segment).
    pub normal: Vec3,
    /// Remaining energy in this crack front.
    pub energy: f32,
    /// Whether this segment has finished propagating.
    pub terminated: bool,
    /// Depth of branching (0 = root).
    pub branch_depth: u32,
}

/// Tracks crack propagation through a mesh.
#[derive(Debug, Clone)]
pub struct CrackNetwork {
    /// All crack segments.
    pub segments: Vec<CrackSegment>,
    /// Active crack fronts (indices into segments).
    pub active_fronts: Vec<usize>,
    /// Maximum branch depth.
    pub max_branch_depth: u32,
    /// Branching probability (0-1).
    pub branch_probability: f32,
    /// Minimum energy to continue propagation.
    pub min_energy: f32,
    /// Energy attenuation per unit distance.
    pub energy_attenuation: f32,
    /// Pseudo-random state for branching decisions.
    rng_state: u32,
}

impl CrackNetwork {
    /// Create a new empty crack network.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            active_fronts: Vec::new(),
            max_branch_depth: 4,
            branch_probability: 0.3,
            min_energy: 10.0,
            energy_attenuation: 0.1,
            rng_state: 54321,
        }
    }

    /// Initiate a crack from an impact point.
    pub fn initiate_crack(
        &mut self,
        impact_point: Vec3,
        impact_direction: Vec3,
        impact_energy: f32,
        pattern: CrackPattern,
    ) {
        let impact_dir = impact_direction.normalize_or_zero();

        match pattern {
            CrackPattern::Radial => {
                // Create radial cracks from impact point
                let num_cracks = 6 + (impact_energy / 100.0) as usize;
                let num_cracks = num_cracks.min(16);

                // Build basis perpendicular to impact direction
                let up = if impact_dir.y.abs() < 0.999 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let right = impact_dir.cross(up).normalize_or_zero();
                let forward = right.cross(impact_dir).normalize_or_zero();

                for i in 0..num_cracks {
                    let angle = (i as f32 / num_cracks as f32) * 2.0 * std::f32::consts::PI;
                    let crack_dir = right * angle.cos() + forward * angle.sin();
                    let crack_normal = crack_dir.cross(impact_dir).normalize_or_zero();

                    let seg_idx = self.segments.len();
                    self.segments.push(CrackSegment {
                        start: impact_point,
                        end: impact_point + crack_dir * MIN_CRACK_LENGTH,
                        normal: crack_normal,
                        energy: impact_energy / num_cracks as f32,
                        terminated: false,
                        branch_depth: 0,
                    });
                    self.active_fronts.push(seg_idx);
                }
            }
            CrackPattern::Branching => {
                // Start with fewer cracks that branch
                let num_cracks = 2 + (impact_energy / 200.0) as usize;
                let num_cracks = num_cracks.min(6);

                let up = if impact_dir.y.abs() < 0.999 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let right = impact_dir.cross(up).normalize_or_zero();
                let forward = right.cross(impact_dir).normalize_or_zero();

                for i in 0..num_cracks {
                    let angle = (i as f32 / num_cracks as f32) * 2.0 * std::f32::consts::PI;
                    let crack_dir = right * angle.cos() + forward * angle.sin();
                    let crack_normal = crack_dir.cross(impact_dir).normalize_or_zero();

                    let seg_idx = self.segments.len();
                    self.segments.push(CrackSegment {
                        start: impact_point,
                        end: impact_point + crack_dir * MIN_CRACK_LENGTH,
                        normal: crack_normal,
                        energy: impact_energy / num_cracks as f32,
                        terminated: false,
                        branch_depth: 0,
                    });
                    self.active_fronts.push(seg_idx);
                }
            }
            CrackPattern::Shatter => {
                // Many radial cracks plus random ones
                let num_cracks = 12 + (impact_energy / 50.0) as usize;
                let num_cracks = num_cracks.min(32);

                let up = if impact_dir.y.abs() < 0.999 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let right = impact_dir.cross(up).normalize_or_zero();
                let forward = right.cross(impact_dir).normalize_or_zero();

                for i in 0..num_cracks {
                    let angle = (i as f32 / num_cracks as f32) * 2.0 * std::f32::consts::PI;
                    let jitter = self.next_random() * 0.3;
                    let crack_dir = right * (angle + jitter).cos()
                        + forward * (angle + jitter).sin();
                    let crack_normal = crack_dir.cross(impact_dir).normalize_or_zero();

                    let seg_idx = self.segments.len();
                    self.segments.push(CrackSegment {
                        start: impact_point,
                        end: impact_point + crack_dir * MIN_CRACK_LENGTH,
                        normal: crack_normal,
                        energy: impact_energy / num_cracks as f32,
                        terminated: false,
                        branch_depth: 0,
                    });
                    self.active_fronts.push(seg_idx);
                }
            }
            CrackPattern::GridLike => {
                // Four orthogonal cracks
                let up = if impact_dir.y.abs() < 0.999 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let right = impact_dir.cross(up).normalize_or_zero();
                let forward = right.cross(impact_dir).normalize_or_zero();

                let dirs = [right, -right, forward, -forward];
                for crack_dir in &dirs {
                    let crack_normal = crack_dir.cross(impact_dir).normalize_or_zero();
                    let seg_idx = self.segments.len();
                    self.segments.push(CrackSegment {
                        start: impact_point,
                        end: impact_point + *crack_dir * MIN_CRACK_LENGTH,
                        normal: crack_normal,
                        energy: impact_energy / 4.0,
                        terminated: false,
                        branch_depth: 0,
                    });
                    self.active_fronts.push(seg_idx);
                }
            }
            CrackPattern::Random => {
                // Random number and direction of cracks
                let num_cracks = 3 + (self.next_random() * 5.0) as usize;
                let num_cracks = num_cracks.min(12);

                for _ in 0..num_cracks {
                    let theta = self.next_random() * 2.0 * std::f32::consts::PI;
                    let phi = (self.next_random() * 2.0 - 1.0).acos();
                    let crack_dir = Vec3::new(
                        phi.sin() * theta.cos(),
                        phi.sin() * theta.sin(),
                        phi.cos(),
                    );
                    let crack_normal = crack_dir
                        .cross(if crack_dir.y.abs() < 0.999 { Vec3::Y } else { Vec3::X })
                        .normalize_or_zero();

                    let seg_idx = self.segments.len();
                    self.segments.push(CrackSegment {
                        start: impact_point,
                        end: impact_point + crack_dir * MIN_CRACK_LENGTH,
                        normal: crack_normal,
                        energy: impact_energy / num_cracks as f32,
                        terminated: false,
                        branch_depth: 0,
                    });
                    self.active_fronts.push(seg_idx);
                }
            }
        }
    }

    /// Propagate active crack fronts by one step.
    /// Returns the number of still-active fronts.
    pub fn propagate_step(&mut self, step_length: f32) -> usize {
        let mut new_fronts = Vec::new();
        let mut fronts_to_remove = Vec::new();

        for (fi, &seg_idx) in self.active_fronts.iter().enumerate() {
            let seg = &self.segments[seg_idx];
            if seg.terminated || seg.energy < self.min_energy {
                fronts_to_remove.push(fi);
                continue;
            }

            let direction = (seg.end - seg.start).normalize_or_zero();
            let new_start = seg.end;
            let attenuation = (-self.energy_attenuation * step_length).exp();
            let new_energy = seg.energy * attenuation;

            // Slight random deviation in crack direction
            let deviation = Vec3::new(
                (self.next_random() - 0.5) * 0.2,
                (self.next_random() - 0.5) * 0.2,
                (self.next_random() - 0.5) * 0.2,
            );
            let new_dir = (direction + deviation).normalize_or_zero();
            let new_end = new_start + new_dir * step_length;

            // Update the current segment as terminated
            self.segments[seg_idx].terminated = true;

            // Create new segment for the continuation
            let new_seg_idx = self.segments.len();
            self.segments.push(CrackSegment {
                start: new_start,
                end: new_end,
                normal: seg.normal,
                energy: new_energy,
                terminated: false,
                branch_depth: seg.branch_depth,
            });
            new_fronts.push(new_seg_idx);

            // Branching
            if seg.branch_depth < self.max_branch_depth
                && self.next_random() < self.branch_probability
            {
                // Create a branch at an angle
                let branch_angle = 0.5 + self.next_random() * 0.5; // 0.5 to 1.0 rad
                let perp = seg.normal.cross(direction).normalize_or_zero();
                let branch_dir = (direction * branch_angle.cos() + perp * branch_angle.sin())
                    .normalize_or_zero();

                let branch_idx = self.segments.len();
                self.segments.push(CrackSegment {
                    start: new_start,
                    end: new_start + branch_dir * step_length,
                    normal: seg.normal,
                    energy: new_energy * 0.5,
                    terminated: false,
                    branch_depth: seg.branch_depth + 1,
                });
                new_fronts.push(branch_idx);
            }
        }

        // Remove terminated fronts (reverse order)
        fronts_to_remove.sort_unstable();
        for &fi in fronts_to_remove.iter().rev() {
            self.active_fronts.swap_remove(fi);
        }

        self.active_fronts.extend(new_fronts);
        self.active_fronts.len()
    }

    /// Simple xorshift32.
    fn next_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f32) / (u32::MAX as f32)
    }

    /// Get the total number of crack segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get the number of active fronts.
    pub fn active_front_count(&self) -> usize {
        self.active_fronts.len()
    }

    /// Get all cutting planes from the crack network for mesh splitting.
    pub fn cutting_planes(&self) -> Vec<(Vec3, Vec3)> {
        self.segments
            .iter()
            .filter(|s| !s.terminated || s.energy > self.min_energy * 0.5)
            .map(|s| {
                let mid = (s.start + s.end) * 0.5;
                (mid, s.normal)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

/// A mesh fragment resulting from fracture.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Fragment mesh.
    pub mesh: FractureMeshV2,
    /// World-space position (center of mass).
    pub position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Mass.
    pub mass: f32,
    /// Inertia tensor.
    pub inertia: Mat3,
    /// Volume.
    pub volume: f32,
    /// Whether this fragment counts as debris (small piece).
    pub is_debris: bool,
    /// Remaining lifetime (-1 = infinite).
    pub lifetime: f32,
    /// Material of this fragment.
    pub material_name: String,
    /// Unique fragment ID.
    pub id: u32,
}

impl Fragment {
    /// Compute mass and inertia from mesh geometry and density.
    pub fn compute_mass_properties(&mut self, density: f32) {
        self.volume = self.mesh.volume();
        self.mass = self.volume * density;

        if self.mass < EPSILON {
            self.mass = EPSILON;
        }

        // Compute inertia tensor from mesh vertices around centroid
        let centroid = self.mesh.centroid();
        let mut ixx = 0.0_f32;
        let mut iyy = 0.0_f32;
        let mut izz = 0.0_f32;
        let mut ixy = 0.0_f32;
        let mut ixz = 0.0_f32;
        let mut iyz = 0.0_f32;

        let mass_per_vertex = self.mass / self.mesh.vertices.len().max(1) as f32;

        for v in &self.mesh.vertices {
            let r = *v - centroid;
            ixx += mass_per_vertex * (r.y * r.y + r.z * r.z);
            iyy += mass_per_vertex * (r.x * r.x + r.z * r.z);
            izz += mass_per_vertex * (r.x * r.x + r.y * r.y);
            ixy -= mass_per_vertex * r.x * r.y;
            ixz -= mass_per_vertex * r.x * r.z;
            iyz -= mass_per_vertex * r.y * r.z;
        }

        self.inertia = Mat3::from_cols(
            Vec3::new(ixx, ixy, ixz),
            Vec3::new(ixy, iyy, iyz),
            Vec3::new(ixz, iyz, izz),
        );
    }

    /// Apply an impulse at a point on the fragment.
    pub fn apply_impulse(&mut self, impulse: Vec3, point: Vec3) {
        if self.mass < EPSILON {
            return;
        }
        let inv_mass = 1.0 / self.mass;
        self.velocity += impulse * inv_mass;

        let r = point - self.position;
        let torque_impulse = r.cross(impulse);
        // Simplified angular impulse (using diagonal inertia only)
        self.angular_velocity += Vec3::new(
            torque_impulse.x / self.inertia.x_axis.x.max(EPSILON),
            torque_impulse.y / self.inertia.y_axis.y.max(EPSILON),
            torque_impulse.z / self.inertia.z_axis.z.max(EPSILON),
        );
    }

    /// Check if this fragment should be classified as debris.
    pub fn classify_debris(&mut self, min_volume: f32) {
        self.is_debris = self.volume < min_volume;
    }
}

// ---------------------------------------------------------------------------
// Fracture manager
// ---------------------------------------------------------------------------

/// Configuration for the fracture manager.
#[derive(Debug, Clone)]
pub struct FractureConfig {
    /// Maximum fragments per fracture event.
    pub max_fragments: usize,
    /// Default debris lifetime.
    pub debris_lifetime: f32,
    /// Debris cull distance from a reference point.
    pub debris_cull_distance: f32,
    /// Minimum fragment volume before becoming debris.
    pub min_fragment_volume: f32,
    /// Crack propagation step length.
    pub crack_step_length: f32,
    /// Number of crack propagation iterations per fracture event.
    pub propagation_steps: usize,
    /// Whether to compute interior faces on fragments.
    pub generate_interior_faces: bool,
    /// Maximum total fragments in the scene.
    pub max_total_fragments: usize,
}

impl Default for FractureConfig {
    fn default() -> Self {
        Self {
            max_fragments: DEFAULT_MAX_FRAGMENTS,
            debris_lifetime: DEFAULT_DEBRIS_LIFETIME,
            debris_cull_distance: DEFAULT_DEBRIS_CULL_DISTANCE,
            min_fragment_volume: MIN_FRAGMENT_VOLUME,
            crack_step_length: 0.05,
            propagation_steps: DEFAULT_PROPAGATION_STEPS,
            generate_interior_faces: true,
            max_total_fragments: 256,
        }
    }
}

/// Events emitted by the fracture system.
#[derive(Debug, Clone)]
pub enum FractureEvent {
    /// A fracture occurred, producing fragments.
    Fractured {
        /// Source object ID.
        source_id: u64,
        /// Impact point.
        impact_point: Vec3,
        /// Impact energy.
        impact_energy: f32,
        /// Number of fragments produced.
        fragment_count: usize,
        /// Fragment IDs.
        fragment_ids: Vec<u32>,
        /// Sound hint.
        sound: FractureSound,
    },
    /// A fragment was cleaned up.
    DebrisRemoved {
        /// Fragment ID.
        fragment_id: u32,
    },
}

/// The main fracture manager that handles runtime fracture events.
#[derive(Debug)]
pub struct FractureManager {
    /// Configuration.
    pub config: FractureConfig,
    /// Active fragments in the scene.
    pub fragments: Vec<Fragment>,
    /// Pending fracture events.
    pub events: Vec<FractureEvent>,
    /// Next fragment ID.
    next_fragment_id: u32,
    /// Reference point for distance culling (e.g., camera position).
    pub cull_reference: Vec3,
    /// Whether the system is enabled.
    pub enabled: bool,
}

impl FractureManager {
    /// Create a new fracture manager with default config.
    pub fn new() -> Self {
        Self {
            config: FractureConfig::default(),
            fragments: Vec::new(),
            events: Vec::new(),
            next_fragment_id: 0,
            cull_reference: Vec3::ZERO,
            enabled: true,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: FractureConfig) -> Self {
        Self {
            config,
            fragments: Vec::new(),
            events: Vec::new(),
            next_fragment_id: 0,
            cull_reference: Vec3::ZERO,
            enabled: true,
        }
    }

    /// Perform a runtime fracture on a mesh at the given impact point.
    /// Returns the indices of the created fragments.
    pub fn fracture(
        &mut self,
        mesh: &FractureMeshV2,
        material: &FractureMaterial,
        impact_point: Vec3,
        impact_direction: Vec3,
        impact_energy: f32,
        source_id: u64,
    ) -> Vec<usize> {
        if !self.enabled || impact_energy < material.toughness * 0.1 {
            return Vec::new();
        }

        if self.fragments.len() >= self.config.max_total_fragments {
            // Clean up oldest debris first
            self.cleanup_oldest_debris(self.config.max_total_fragments / 4);
        }

        // Generate crack network
        let mut cracks = CrackNetwork::new();
        if material.shatters {
            cracks.branch_probability = 0.5;
            cracks.max_branch_depth = 5;
        } else {
            cracks.branch_probability = 0.2;
            cracks.max_branch_depth = 3;
        }

        cracks.initiate_crack(
            impact_point,
            impact_direction,
            impact_energy,
            material.crack_pattern,
        );

        // Propagate cracks
        for _ in 0..self.config.propagation_steps {
            let active = cracks.propagate_step(self.config.crack_step_length);
            if active == 0 {
                break;
            }
        }

        // Get cutting planes from cracks
        let planes = cracks.cutting_planes();
        if planes.is_empty() {
            return Vec::new();
        }

        // Split mesh by cutting planes
        let mut pieces = vec![mesh.clone()];
        for (plane_point, plane_normal) in &planes {
            if pieces.len() >= self.config.max_fragments {
                break;
            }

            let mut new_pieces = Vec::new();
            for piece in &pieces {
                if piece.vertices.is_empty() {
                    continue;
                }
                let (pos, neg) = piece.split_by_plane(*plane_point, *plane_normal);
                if !pos.vertices.is_empty() {
                    new_pieces.push(pos);
                }
                if !neg.vertices.is_empty() {
                    new_pieces.push(neg);
                }
            }
            pieces = new_pieces;
        }

        // Create fragments from pieces
        let mut fragment_indices = Vec::new();
        let mut fragment_ids = Vec::new();

        for piece in pieces {
            if piece.vertices.is_empty() {
                continue;
            }

            let centroid = piece.centroid();
            let frag_id = self.next_fragment_id;
            self.next_fragment_id += 1;

            let mut fragment = Fragment {
                mesh: piece,
                position: centroid,
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                mass: 0.0,
                inertia: Mat3::IDENTITY,
                volume: 0.0,
                is_debris: false,
                lifetime: -1.0,
                material_name: material.name.clone(),
                id: frag_id,
            };

            fragment.compute_mass_properties(material.density);
            fragment.classify_debris(self.config.min_fragment_volume);

            if fragment.is_debris {
                fragment.lifetime = self.config.debris_lifetime;
            }

            // Apply impulse from impact
            let to_fragment = centroid - impact_point;
            let dist = to_fragment.length().max(0.01);
            let impulse_strength = impact_energy * 0.1 / (dist * dist + 1.0);
            let impulse_dir = to_fragment.normalize_or_zero();
            fragment.velocity = impulse_dir * impulse_strength / fragment.mass.max(EPSILON);

            // Small random angular velocity
            fragment.angular_velocity = Vec3::new(
                (frag_id as f32 * 1.37).sin() * 3.0,
                (frag_id as f32 * 2.41).cos() * 3.0,
                (frag_id as f32 * 0.93).sin() * 3.0,
            );

            let idx = self.fragments.len();
            fragment_ids.push(frag_id);
            self.fragments.push(fragment);
            fragment_indices.push(idx);
        }

        // Emit event
        self.events.push(FractureEvent::Fractured {
            source_id,
            impact_point,
            impact_energy,
            fragment_count: fragment_indices.len(),
            fragment_ids: fragment_ids.clone(),
            sound: material.fracture_sound,
        });

        fragment_indices
    }

    /// Update fragments: decrease lifetimes, cull distant/expired debris.
    pub fn update(&mut self, dt: f32) {
        // Update lifetimes
        for frag in &mut self.fragments {
            if frag.lifetime > 0.0 {
                frag.lifetime -= dt;
            }
        }

        // Remove expired and distant debris
        let cull_dist_sq = self.config.debris_cull_distance * self.config.debris_cull_distance;
        let cull_ref = self.cull_reference;

        self.fragments.retain(|f| {
            if f.lifetime >= 0.0 && f.lifetime <= 0.0 {
                return false; // expired
            }
            if f.is_debris {
                let dist_sq = (f.position - cull_ref).length_squared();
                if dist_sq > cull_dist_sq {
                    return false; // too far
                }
            }
            true
        });
    }

    /// Clean up the oldest debris fragments.
    fn cleanup_oldest_debris(&mut self, count: usize) {
        let mut debris_indices: Vec<usize> = self
            .fragments
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_debris)
            .map(|(i, _)| i)
            .collect();

        // Sort by remaining lifetime (ascending)
        debris_indices.sort_by(|&a, &b| {
            self.fragments[a]
                .lifetime
                .partial_cmp(&self.fragments[b].lifetime)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove the first `count`
        let to_remove: HashSet<usize> = debris_indices.iter().take(count).copied().collect();

        let mut idx = 0;
        self.fragments.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }

    /// Get the total number of fragments.
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Get the number of debris fragments.
    pub fn debris_count(&self) -> usize {
        self.fragments.iter().filter(|f| f.is_debris).count()
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<FractureEvent> {
        std::mem::take(&mut self.events)
    }

    /// Clear all fragments and events.
    pub fn clear(&mut self) {
        self.fragments.clear();
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component marking an entity as fracturable.
#[derive(Debug, Clone)]
pub struct FracturableComponent {
    /// Mesh data (reference or owned).
    pub mesh: FractureMeshV2,
    /// Material properties.
    pub material: FractureMaterial,
    /// Minimum impact energy to trigger fracture.
    pub threshold_energy: f32,
    /// Whether this entity can be fractured.
    pub enabled: bool,
    /// Whether this entity has already been fractured.
    pub fractured: bool,
    /// Source entity ID.
    pub entity_id: u64,
}

impl FracturableComponent {
    /// Create a new fracturable component.
    pub fn new(mesh: FractureMeshV2, material: FractureMaterial, entity_id: u64) -> Self {
        let threshold = material.toughness;
        Self {
            mesh,
            material,
            threshold_energy: threshold,
            enabled: true,
            fractured: false,
            entity_id,
        }
    }
}

/// ECS system that manages fracture events.
pub struct FractureSystem {
    /// The fracture manager.
    pub manager: FractureManager,
}

impl FractureSystem {
    /// Create a new fracture system.
    pub fn new() -> Self {
        Self {
            manager: FractureManager::new(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: FractureConfig) -> Self {
        Self {
            manager: FractureManager::with_config(config),
        }
    }

    /// Process a potential fracture on a component.
    pub fn try_fracture(
        &mut self,
        component: &mut FracturableComponent,
        impact_point: Vec3,
        impact_direction: Vec3,
        impact_energy: f32,
    ) -> bool {
        if !component.enabled || component.fractured {
            return false;
        }

        if impact_energy < component.threshold_energy {
            return false;
        }

        let fragments = self.manager.fracture(
            &component.mesh,
            &component.material,
            impact_point,
            impact_direction,
            impact_energy,
            component.entity_id,
        );

        if !fragments.is_empty() {
            component.fractured = true;
            true
        } else {
            false
        }
    }

    /// Update the system (debris cleanup, etc.).
    pub fn update(&mut self, dt: f32) {
        self.manager.update(dt);
    }
}
