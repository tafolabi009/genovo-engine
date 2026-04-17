//! Procedural vegetation generation for the Genovo engine.
//!
//! Generates realistic trees and vegetation using:
//!
//! - **Space colonization algorithm** — data-driven tree branching that
//!   produces natural-looking canopy shapes.
//! - **Branch structure** — hierarchical branches with radius tapering,
//!   curvature, and gravity response.
//! - **Leaf placement** — distributes leaves on terminal branches with
//!   density and clustering controls.
//! - **LOD generation** — automatic simplification for distance rendering.
//! - **Biome-based species selection** — choose tree species based on
//!   temperature, moisture, and altitude.
//! - **Growth simulation** — age-based trunk/canopy size progression.
//! - **Forest density** — Poisson disk distribution for natural spacing.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// Simple deterministic RNG for vegetation generation.
#[derive(Debug, Clone)]
pub struct VegRng {
    state: u64,
}

impl VegRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    pub fn range_u32(&mut self, max: u32) -> u32 {
        (self.next_u64() % max as u64) as u32
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// Minimal 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 { Self::ZERO } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }

    pub fn add(&self, o: &Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn sub(&self, o: &Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn scale(&self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    pub fn dot(&self, o: &Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    pub fn distance(&self, o: &Self) -> f32 { self.sub(o).length() }

    pub fn lerp(&self, o: &Self, t: f32) -> Self {
        Self::new(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )
    }

    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

// ---------------------------------------------------------------------------
// Biome / species
// ---------------------------------------------------------------------------

/// Climate biome classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Biome {
    Tropical,
    Temperate,
    Boreal,
    Desert,
    Tundra,
    Wetland,
    Mediterranean,
    Savanna,
}

/// Tree species with growth parameters.
#[derive(Debug, Clone)]
pub struct TreeSpecies {
    /// Species name.
    pub name: String,
    /// Biomes this species is found in.
    pub biomes: Vec<Biome>,
    /// Maximum trunk height at maturity (meters).
    pub max_trunk_height: f32,
    /// Maximum canopy radius at maturity (meters).
    pub max_canopy_radius: f32,
    /// Trunk radius at base at maturity (meters).
    pub max_trunk_radius: f32,
    /// Number of major branch levels.
    pub branch_levels: u32,
    /// Branching angle from parent (degrees).
    pub branch_angle: f32,
    /// Branch length ratio (child / parent).
    pub branch_length_ratio: f32,
    /// Branch radius ratio (child / parent).
    pub branch_radius_ratio: f32,
    /// Leaf density (leaves per unit volume).
    pub leaf_density: f32,
    /// Leaf size (meters).
    pub leaf_size: f32,
    /// Whether the tree is deciduous (loses leaves).
    pub deciduous: bool,
    /// Growth rate (0..1 scale, higher = faster).
    pub growth_rate: f32,
    /// Minimum altitude where this species grows.
    pub min_altitude: f32,
    /// Maximum altitude where this species grows.
    pub max_altitude: f32,
    /// Minimum moisture requirement (0..1).
    pub min_moisture: f32,
    /// Maximum temperature (Celsius).
    pub max_temperature: f32,
}

impl TreeSpecies {
    /// Creates a generic oak tree.
    pub fn oak() -> Self {
        Self {
            name: "Oak".to_string(),
            biomes: vec![Biome::Temperate, Biome::Mediterranean],
            max_trunk_height: 15.0,
            max_canopy_radius: 8.0,
            max_trunk_radius: 0.4,
            branch_levels: 4,
            branch_angle: 35.0,
            branch_length_ratio: 0.7,
            branch_radius_ratio: 0.6,
            leaf_density: 1.5,
            leaf_size: 0.08,
            deciduous: true,
            growth_rate: 0.3,
            min_altitude: 0.0,
            max_altitude: 1500.0,
            min_moisture: 0.3,
            max_temperature: 35.0,
        }
    }

    /// Creates a pine tree.
    pub fn pine() -> Self {
        Self {
            name: "Pine".to_string(),
            biomes: vec![Biome::Boreal, Biome::Temperate],
            max_trunk_height: 20.0,
            max_canopy_radius: 4.0,
            max_trunk_radius: 0.3,
            branch_levels: 5,
            branch_angle: 25.0,
            branch_length_ratio: 0.65,
            branch_radius_ratio: 0.5,
            leaf_density: 2.0,
            leaf_size: 0.04,
            deciduous: false,
            growth_rate: 0.4,
            min_altitude: 200.0,
            max_altitude: 2500.0,
            min_moisture: 0.2,
            max_temperature: 30.0,
        }
    }

    /// Creates a birch tree.
    pub fn birch() -> Self {
        Self {
            name: "Birch".to_string(),
            biomes: vec![Biome::Boreal, Biome::Temperate],
            max_trunk_height: 18.0,
            max_canopy_radius: 5.0,
            max_trunk_radius: 0.2,
            branch_levels: 4,
            branch_angle: 40.0,
            branch_length_ratio: 0.6,
            branch_radius_ratio: 0.45,
            leaf_density: 1.8,
            leaf_size: 0.05,
            deciduous: true,
            growth_rate: 0.5,
            min_altitude: 0.0,
            max_altitude: 2000.0,
            min_moisture: 0.4,
            max_temperature: 28.0,
        }
    }

    /// Creates a palm tree.
    pub fn palm() -> Self {
        Self {
            name: "Palm".to_string(),
            biomes: vec![Biome::Tropical, Biome::Desert],
            max_trunk_height: 12.0,
            max_canopy_radius: 4.0,
            max_trunk_radius: 0.25,
            branch_levels: 1,
            branch_angle: 45.0,
            branch_length_ratio: 0.9,
            branch_radius_ratio: 0.3,
            leaf_density: 0.5,
            leaf_size: 0.6,
            deciduous: false,
            growth_rate: 0.6,
            min_altitude: 0.0,
            max_altitude: 500.0,
            min_moisture: 0.1,
            max_temperature: 45.0,
        }
    }

    /// Creates a cactus.
    pub fn cactus() -> Self {
        Self {
            name: "Cactus".to_string(),
            biomes: vec![Biome::Desert],
            max_trunk_height: 5.0,
            max_canopy_radius: 1.5,
            max_trunk_radius: 0.15,
            branch_levels: 2,
            branch_angle: 90.0,
            branch_length_ratio: 0.5,
            branch_radius_ratio: 0.7,
            leaf_density: 0.0,
            leaf_size: 0.0,
            deciduous: false,
            growth_rate: 0.1,
            min_altitude: 0.0,
            max_altitude: 1000.0,
            min_moisture: 0.0,
            max_temperature: 50.0,
        }
    }

    /// Whether this species can grow in the given conditions.
    pub fn can_grow_in(&self, biome: Biome, altitude: f32, moisture: f32, temperature: f32) -> bool {
        self.biomes.contains(&biome)
            && altitude >= self.min_altitude
            && altitude <= self.max_altitude
            && moisture >= self.min_moisture
            && temperature <= self.max_temperature
    }
}

// ---------------------------------------------------------------------------
// Branch structure
// ---------------------------------------------------------------------------

/// A single branch in the tree skeleton.
#[derive(Debug, Clone)]
pub struct Branch {
    /// Branch id.
    pub id: u32,
    /// Parent branch id (0 for trunk root).
    pub parent_id: Option<u32>,
    /// Start position (local space, relative to tree root).
    pub start: Vec3,
    /// End position (local space).
    pub end: Vec3,
    /// Radius at the start.
    pub start_radius: f32,
    /// Radius at the end.
    pub end_radius: f32,
    /// Growth direction (normalized).
    pub direction: Vec3,
    /// Branch depth (0 = trunk, 1 = primary branches, etc.).
    pub depth: u32,
    /// Length of this branch segment.
    pub length: f32,
    /// Child branch ids.
    pub children: Vec<u32>,
    /// Whether this branch has leaves.
    pub has_leaves: bool,
}

impl Branch {
    /// Creates a new branch.
    pub fn new(
        id: u32,
        parent_id: Option<u32>,
        start: Vec3,
        end: Vec3,
        start_radius: f32,
        end_radius: f32,
        depth: u32,
    ) -> Self {
        let diff = end.sub(&start);
        let length = diff.length();
        let direction = diff.normalized();

        Self {
            id,
            parent_id,
            start,
            end,
            start_radius,
            end_radius,
            direction,
            depth,
            length,
            children: Vec::new(),
            has_leaves: false,
        }
    }

    /// Returns the midpoint of the branch.
    pub fn midpoint(&self) -> Vec3 {
        self.start.lerp(&self.end, 0.5)
    }

    /// Returns the position at parameter t (0 = start, 1 = end).
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.start.lerp(&self.end, t)
    }

    /// Returns the radius at parameter t.
    pub fn radius_at(&self, t: f32) -> f32 {
        self.start_radius + (self.end_radius - self.start_radius) * t
    }
}

// ---------------------------------------------------------------------------
// Leaf
// ---------------------------------------------------------------------------

/// A leaf instance on the tree.
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    /// Position in local tree space.
    pub position: Vec3,
    /// Normal direction (face direction).
    pub normal: Vec3,
    /// Size scale factor.
    pub size: f32,
    /// Rotation around the normal axis (radians).
    pub rotation: f32,
    /// Color variation factor (0..1).
    pub color_variation: f32,
    /// Branch id this leaf is attached to.
    pub branch_id: u32,
}

// ---------------------------------------------------------------------------
// Attraction point (space colonization)
// ---------------------------------------------------------------------------

/// An attraction point for the space colonization algorithm.
#[derive(Debug, Clone)]
struct AttractionPoint {
    position: Vec3,
    active: bool,
}

// ---------------------------------------------------------------------------
// Generated tree
// ---------------------------------------------------------------------------

/// The complete output of the tree generator.
#[derive(Debug, Clone)]
pub struct GeneratedTree {
    /// The species of this tree.
    pub species_name: String,
    /// All branches (index 0 is the trunk base).
    pub branches: Vec<Branch>,
    /// All leaves.
    pub leaves: Vec<Leaf>,
    /// Tree age (0..1 maturity factor).
    pub age: f32,
    /// Total height of the tree.
    pub height: f32,
    /// Maximum canopy radius.
    pub canopy_radius: f32,
    /// Trunk base radius.
    pub trunk_radius: f32,
    /// LOD simplified branch counts.
    pub lod_branch_counts: Vec<usize>,
    /// LOD simplified leaf counts.
    pub lod_leaf_counts: Vec<usize>,
    /// Bounding sphere radius.
    pub bounding_radius: f32,
}

impl GeneratedTree {
    /// Returns the branch count.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Returns the leaf count.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Returns branches at a specific depth.
    pub fn branches_at_depth(&self, depth: u32) -> Vec<&Branch> {
        self.branches.iter().filter(|b| b.depth == depth).collect()
    }

    /// Returns the number of LOD levels.
    pub fn lod_count(&self) -> usize {
        self.lod_branch_counts.len()
    }
}

// ---------------------------------------------------------------------------
// Tree generation configuration
// ---------------------------------------------------------------------------

/// Configuration for tree generation.
#[derive(Debug, Clone)]
pub struct TreeGenConfig {
    /// Number of attraction points for space colonization.
    pub attraction_points: u32,
    /// Kill distance -- attraction points closer than this are removed.
    pub kill_distance: f32,
    /// Influence distance -- attraction points further than this are ignored.
    pub influence_distance: f32,
    /// Growth step size per iteration.
    pub growth_step: f32,
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Gravity influence (0 = none, 1 = strong droop).
    pub gravity_influence: f32,
    /// Tropism (tendency to grow upward, 0..1).
    pub tropism: f32,
    /// Branch curvature randomness.
    pub curvature_noise: f32,
    /// Minimum branch radius (below this, branches become leaves).
    pub min_branch_radius: f32,
    /// Number of LOD levels to generate.
    pub lod_levels: u32,
    /// Random seed.
    pub seed: u64,
}

impl TreeGenConfig {
    /// Default configuration for a medium tree.
    pub fn default_medium() -> Self {
        Self {
            attraction_points: 200,
            kill_distance: 0.5,
            influence_distance: 5.0,
            growth_step: 0.3,
            max_iterations: 100,
            gravity_influence: 0.05,
            tropism: 0.6,
            curvature_noise: 0.1,
            min_branch_radius: 0.01,
            lod_levels: 4,
            seed: 42,
        }
    }

    /// Configuration for a small/young tree.
    pub fn small() -> Self {
        Self {
            attraction_points: 80,
            kill_distance: 0.3,
            influence_distance: 3.0,
            growth_step: 0.2,
            max_iterations: 60,
            gravity_influence: 0.03,
            tropism: 0.7,
            curvature_noise: 0.05,
            min_branch_radius: 0.005,
            lod_levels: 3,
            seed: 42,
        }
    }

    /// Configuration for a large/old tree.
    pub fn large() -> Self {
        Self {
            attraction_points: 500,
            kill_distance: 0.8,
            influence_distance: 8.0,
            growth_step: 0.4,
            max_iterations: 150,
            gravity_influence: 0.08,
            tropism: 0.5,
            curvature_noise: 0.15,
            min_branch_radius: 0.015,
            lod_levels: 5,
            seed: 42,
        }
    }
}

impl Default for TreeGenConfig {
    fn default() -> Self {
        Self::default_medium()
    }
}

// ---------------------------------------------------------------------------
// Tree generator
// ---------------------------------------------------------------------------

/// Procedural tree generator using space colonization.
pub struct TreeGenerator {
    config: TreeGenConfig,
    rng: VegRng,
    next_branch_id: u32,
}

impl TreeGenerator {
    /// Creates a new tree generator.
    pub fn new(config: TreeGenConfig) -> Self {
        let rng = VegRng::new(config.seed);
        Self {
            config,
            rng,
            next_branch_id: 0,
        }
    }

    /// Generates a tree of the given species at the given maturity.
    pub fn generate(&mut self, species: &TreeSpecies, age: f32) -> GeneratedTree {
        let age = age.clamp(0.0, 1.0);

        // Scale parameters by age.
        let trunk_height = species.max_trunk_height * age;
        let canopy_radius = species.max_canopy_radius * age;
        let trunk_radius = species.max_trunk_radius * age;

        // 1. Generate attraction point cloud.
        let attraction_points = self.generate_attraction_points(
            trunk_height,
            canopy_radius,
        );

        // 2. Run space colonization to build branch skeleton.
        let mut branches = self.space_colonization(
            trunk_height,
            trunk_radius,
            species,
            &attraction_points,
        );

        // 3. Compute branch radii using pipe model.
        self.compute_radii(&mut branches, trunk_radius, species.branch_radius_ratio);

        // 4. Mark terminal branches for leaf placement.
        self.mark_leaf_branches(&mut branches, species.branch_levels);

        // 5. Generate leaves.
        let leaves = self.generate_leaves(&branches, species);

        // 6. Generate LOD data.
        let (lod_branch_counts, lod_leaf_counts) =
            self.generate_lods(&branches, &leaves);

        // 7. Compute bounding info.
        let height = branches
            .iter()
            .map(|b| b.end.y.max(b.start.y))
            .fold(0.0f32, f32::max);

        let bounding_radius = branches
            .iter()
            .map(|b| b.end.length().max(b.start.length()))
            .fold(0.0f32, f32::max);

        GeneratedTree {
            species_name: species.name.clone(),
            branches,
            leaves,
            age,
            height,
            canopy_radius,
            trunk_radius,
            lod_branch_counts,
            lod_leaf_counts,
            bounding_radius,
        }
    }

    fn generate_attraction_points(
        &mut self,
        trunk_height: f32,
        canopy_radius: f32,
    ) -> Vec<AttractionPoint> {
        let mut points = Vec::new();
        let canopy_center_y = trunk_height * 0.7;

        for _ in 0..self.config.attraction_points {
            // Generate points in an ellipsoid (canopy shape).
            let theta = self.rng.range_f32(0.0, std::f32::consts::TAU);
            let phi = self.rng.range_f32(0.0, std::f32::consts::PI);
            let r = self.rng.range_f32(0.2, 1.0);

            let x = r * canopy_radius * phi.sin() * theta.cos();
            let y = canopy_center_y + r * trunk_height * 0.4 * phi.cos();
            let z = r * canopy_radius * phi.sin() * theta.sin();

            if y > 0.0 {
                points.push(AttractionPoint {
                    position: Vec3::new(x, y, z),
                    active: true,
                });
            }
        }

        points
    }

    fn space_colonization(
        &mut self,
        trunk_height: f32,
        trunk_radius: f32,
        species: &TreeSpecies,
        attraction_points: &[AttractionPoint],
    ) -> Vec<Branch> {
        let mut branches = Vec::new();
        let mut active_points: Vec<AttractionPoint> = attraction_points.to_vec();

        // Create initial trunk.
        let trunk_segments = (trunk_height / self.config.growth_step).ceil() as u32;
        let mut prev_end = Vec3::ZERO;

        for i in 0..trunk_segments.max(1) {
            let start = prev_end;
            let end = Vec3::new(
                start.x + self.rng.range_f32(-0.02, 0.02),
                start.y + self.config.growth_step,
                start.z + self.rng.range_f32(-0.02, 0.02),
            );
            let id = self.next_branch_id;
            self.next_branch_id += 1;

            let parent = if i > 0 { Some(id - 1) } else { None };
            let branch = Branch::new(id, parent, start, end, trunk_radius, trunk_radius * 0.95, 0);
            branches.push(branch);

            if i > 0 {
                let parent_idx = (id - 1) as usize;
                if parent_idx < branches.len() {
                    branches[parent_idx].children.push(id);
                }
            }

            prev_end = end;
        }

        // Grow branches toward attraction points.
        let tip_indices: Vec<usize> = vec![branches.len() - 1];
        let mut tips: Vec<usize> = tip_indices;

        for _iteration in 0..self.config.max_iterations {
            if active_points.iter().all(|p| !p.active) {
                break;
            }

            let mut new_tips = Vec::new();

            for &tip_idx in &tips {
                if tip_idx >= branches.len() {
                    continue;
                }
                let tip_pos = branches[tip_idx].end;
                let tip_depth = branches[tip_idx].depth;

                // Find attraction points influencing this tip.
                let mut avg_dir = Vec3::ZERO;
                let mut influence_count = 0;

                for point in &active_points {
                    if !point.active {
                        continue;
                    }
                    let dist = tip_pos.distance(&point.position);
                    if dist < self.config.influence_distance {
                        let dir = point.position.sub(&tip_pos).normalized();
                        avg_dir = avg_dir.add(&dir);
                        influence_count += 1;
                    }
                }

                if influence_count > 0 {
                    avg_dir = avg_dir.scale(1.0 / influence_count as f32).normalized();

                    // Add tropism (upward tendency).
                    avg_dir = avg_dir.add(&Vec3::UP.scale(self.config.tropism)).normalized();

                    // Add gravity.
                    let gravity = Vec3::new(0.0, -self.config.gravity_influence, 0.0);
                    avg_dir = avg_dir.add(&gravity).normalized();

                    // Add noise.
                    let noise = Vec3::new(
                        self.rng.range_f32(-1.0, 1.0) * self.config.curvature_noise,
                        self.rng.range_f32(-1.0, 1.0) * self.config.curvature_noise,
                        self.rng.range_f32(-1.0, 1.0) * self.config.curvature_noise,
                    );
                    avg_dir = avg_dir.add(&noise).normalized();

                    // Create new branch.
                    let new_end = tip_pos.add(&avg_dir.scale(self.config.growth_step));
                    let new_id = self.next_branch_id;
                    self.next_branch_id += 1;

                    let new_depth = tip_depth + 1;
                    let radius = trunk_radius * species.branch_radius_ratio.powi(new_depth as i32);

                    let branch = Branch::new(
                        new_id,
                        Some(branches[tip_idx].id),
                        tip_pos,
                        new_end,
                        radius,
                        radius * 0.8,
                        new_depth.min(species.branch_levels),
                    );
                    branches.push(branch);
                    branches[tip_idx].children.push(new_id);
                    new_tips.push(branches.len() - 1);
                }
            }

            // Remove attraction points that are too close to branches.
            for point in &mut active_points {
                if !point.active {
                    continue;
                }
                for branch in &branches {
                    if branch.end.distance(&point.position) < self.config.kill_distance {
                        point.active = false;
                        break;
                    }
                }
            }

            if new_tips.is_empty() {
                break;
            }
            tips = new_tips;
        }

        branches
    }

    fn compute_radii(&self, branches: &mut [Branch], base_radius: f32, ratio: f32) {
        // Compute radii from depth.
        for branch in branches.iter_mut() {
            let depth_factor = ratio.powi(branch.depth as i32);
            branch.start_radius = base_radius * depth_factor;
            branch.end_radius = branch.start_radius * 0.85;
        }
    }

    fn mark_leaf_branches(&self, branches: &mut [Branch], max_depth: u32) {
        for branch in branches.iter_mut() {
            branch.has_leaves = branch.children.is_empty() || branch.depth >= max_depth - 1;
        }
    }

    fn generate_leaves(&mut self, branches: &[Branch], species: &TreeSpecies) -> Vec<Leaf> {
        let mut leaves = Vec::new();

        if species.leaf_density <= 0.0 {
            return leaves;
        }

        for branch in branches {
            if !branch.has_leaves {
                continue;
            }

            let leaf_count = (branch.length * species.leaf_density * 10.0) as u32;

            for _ in 0..leaf_count {
                let t = self.rng.next_f32();
                let pos = branch.point_at(t);

                // Offset from branch axis.
                let offset = Vec3::new(
                    self.rng.range_f32(-0.3, 0.3),
                    self.rng.range_f32(-0.1, 0.2),
                    self.rng.range_f32(-0.3, 0.3),
                );
                let leaf_pos = pos.add(&offset);

                let normal = Vec3::new(
                    self.rng.range_f32(-1.0, 1.0),
                    self.rng.range_f32(0.3, 1.0),
                    self.rng.range_f32(-1.0, 1.0),
                )
                .normalized();

                leaves.push(Leaf {
                    position: leaf_pos,
                    normal,
                    size: species.leaf_size * self.rng.range_f32(0.7, 1.3),
                    rotation: self.rng.range_f32(0.0, std::f32::consts::TAU),
                    color_variation: self.rng.next_f32(),
                    branch_id: branch.id,
                });
            }
        }

        leaves
    }

    fn generate_lods(
        &self,
        branches: &[Branch],
        leaves: &[Leaf],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut branch_counts = Vec::new();
        let mut leaf_counts = Vec::new();

        let total_branches = branches.len();
        let total_leaves = leaves.len();

        for lod in 0..self.config.lod_levels {
            let factor = 1.0 / (1 << lod) as f32;
            branch_counts.push((total_branches as f32 * factor) as usize);
            leaf_counts.push((total_leaves as f32 * factor) as usize);
        }

        (branch_counts, leaf_counts)
    }
}

// ---------------------------------------------------------------------------
// Forest generator
// ---------------------------------------------------------------------------

/// Configuration for forest generation.
#[derive(Debug, Clone)]
pub struct ForestConfig {
    /// Area width.
    pub width: f32,
    /// Area depth.
    pub depth: f32,
    /// Minimum distance between trees.
    pub min_spacing: f32,
    /// Maximum distance between trees.
    pub max_spacing: f32,
    /// Biome for species selection.
    pub biome: Biome,
    /// Altitude (meters above sea level).
    pub altitude: f32,
    /// Moisture level (0..1).
    pub moisture: f32,
    /// Temperature (Celsius).
    pub temperature: f32,
    /// Random seed.
    pub seed: u64,
    /// Maximum number of trees.
    pub max_trees: u32,
    /// Age variation range (0..1).
    pub age_min: f32,
    /// Maximum age.
    pub age_max: f32,
}

impl ForestConfig {
    /// Default temperate forest.
    pub fn temperate() -> Self {
        Self {
            width: 100.0,
            depth: 100.0,
            min_spacing: 3.0,
            max_spacing: 8.0,
            biome: Biome::Temperate,
            altitude: 300.0,
            moisture: 0.5,
            temperature: 15.0,
            seed: 42,
            max_trees: 200,
            age_min: 0.3,
            age_max: 1.0,
        }
    }

    /// Dense boreal forest.
    pub fn boreal() -> Self {
        Self {
            width: 100.0,
            depth: 100.0,
            min_spacing: 2.0,
            max_spacing: 5.0,
            biome: Biome::Boreal,
            altitude: 800.0,
            moisture: 0.6,
            temperature: 5.0,
            seed: 42,
            max_trees: 400,
            age_min: 0.2,
            age_max: 0.9,
        }
    }
}

impl Default for ForestConfig {
    fn default() -> Self {
        Self::temperate()
    }
}

/// A tree placement in a forest.
#[derive(Debug, Clone)]
pub struct TreePlacement {
    /// Position (x, z) in forest-local space.
    pub position: (f32, f32),
    /// Species index.
    pub species_index: usize,
    /// Tree age (maturity factor 0..1).
    pub age: f32,
    /// Rotation around Y axis (radians).
    pub rotation: f32,
    /// Scale variation.
    pub scale: f32,
}

/// Generates tree placements for a forest area using Poisson disk sampling.
pub fn generate_forest_placements(
    config: &ForestConfig,
    species: &[TreeSpecies],
) -> Vec<TreePlacement> {
    let mut rng = VegRng::new(config.seed);
    let mut placements = Vec::new();

    // Filter to valid species for this biome.
    let valid_indices: Vec<usize> = species
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.can_grow_in(config.biome, config.altitude, config.moisture, config.temperature)
        })
        .map(|(i, _)| i)
        .collect();

    if valid_indices.is_empty() {
        return placements;
    }

    // Simple Poisson disk sampling.
    let cell_size = config.min_spacing / std::f32::consts::SQRT_2;
    let grid_w = (config.width / cell_size).ceil() as usize;
    let grid_h = (config.depth / cell_size).ceil() as usize;
    let mut grid = vec![None::<usize>; grid_w * grid_h];
    let mut active_list: Vec<usize> = Vec::new();

    // Place first tree.
    let first_x = rng.range_f32(0.0, config.width);
    let first_z = rng.range_f32(0.0, config.depth);
    let species_idx = valid_indices[rng.range_u32(valid_indices.len() as u32) as usize];
    let age = rng.range_f32(config.age_min, config.age_max);

    placements.push(TreePlacement {
        position: (first_x, first_z),
        species_index: species_idx,
        age,
        rotation: rng.range_f32(0.0, std::f32::consts::TAU),
        scale: rng.range_f32(0.8, 1.2),
    });

    let gx = (first_x / cell_size) as usize;
    let gz = (first_z / cell_size) as usize;
    if gx < grid_w && gz < grid_h {
        grid[gz * grid_w + gx] = Some(0);
    }
    active_list.push(0);

    let max_attempts = 30;

    while !active_list.is_empty() && placements.len() < config.max_trees as usize {
        let active_idx = rng.range_u32(active_list.len() as u32) as usize;
        let parent_idx = active_list[active_idx];
        let parent_pos = placements[parent_idx].position;

        let mut found = false;
        for _ in 0..max_attempts {
            let angle = rng.range_f32(0.0, std::f32::consts::TAU);
            let dist = rng.range_f32(config.min_spacing, config.max_spacing);
            let x = parent_pos.0 + angle.cos() * dist;
            let z = parent_pos.1 + angle.sin() * dist;

            if x < 0.0 || z < 0.0 || x >= config.width || z >= config.depth {
                continue;
            }

            let gx = (x / cell_size) as usize;
            let gz = (z / cell_size) as usize;

            if gx >= grid_w || gz >= grid_h {
                continue;
            }

            // Check neighbours.
            let mut too_close = false;
            let check_range = 2i32;
            for dz in -check_range..=check_range {
                for dx in -check_range..=check_range {
                    let nx = gx as i32 + dx;
                    let nz = gz as i32 + dz;
                    if nx >= 0 && nx < grid_w as i32 && nz >= 0 && nz < grid_h as i32 {
                        if let Some(idx) = grid[nz as usize * grid_w + nx as usize] {
                            let other = &placements[idx];
                            let dx = x - other.position.0;
                            let dz = z - other.position.1;
                            if dx * dx + dz * dz < config.min_spacing * config.min_spacing {
                                too_close = true;
                                break;
                            }
                        }
                    }
                }
                if too_close {
                    break;
                }
            }

            if !too_close {
                let new_idx = placements.len();
                let sp_idx = valid_indices[rng.range_u32(valid_indices.len() as u32) as usize];
                let tree_age = rng.range_f32(config.age_min, config.age_max);

                placements.push(TreePlacement {
                    position: (x, z),
                    species_index: sp_idx,
                    age: tree_age,
                    rotation: rng.range_f32(0.0, std::f32::consts::TAU),
                    scale: rng.range_f32(0.8, 1.2),
                });

                grid[gz * grid_w + gx] = Some(new_idx);
                active_list.push(new_idx);
                found = true;
                break;
            }
        }

        if !found {
            active_list.swap_remove(active_idx);
        }
    }

    placements
}

/// Selects suitable species for a given biome and conditions.
pub fn select_species(
    all_species: &[TreeSpecies],
    biome: Biome,
    altitude: f32,
    moisture: f32,
    temperature: f32,
) -> Vec<&TreeSpecies> {
    all_species
        .iter()
        .filter(|s| s.can_grow_in(biome, altitude, moisture, temperature))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn species_creation() {
        let oak = TreeSpecies::oak();
        assert_eq!(oak.name, "Oak");
        assert!(oak.deciduous);
        assert!(oak.biomes.contains(&Biome::Temperate));
    }

    #[test]
    fn species_can_grow() {
        let oak = TreeSpecies::oak();
        assert!(oak.can_grow_in(Biome::Temperate, 500.0, 0.5, 20.0));
        assert!(!oak.can_grow_in(Biome::Desert, 500.0, 0.5, 20.0));
    }

    #[test]
    fn species_altitude_constraint() {
        let pine = TreeSpecies::pine();
        assert!(!pine.can_grow_in(Biome::Boreal, 50.0, 0.5, 10.0)); // Too low
        assert!(pine.can_grow_in(Biome::Boreal, 500.0, 0.5, 10.0));
    }

    #[test]
    fn branch_creation() {
        let branch = Branch::new(0, None, Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0), 0.3, 0.2, 0);
        assert!((branch.length - 5.0).abs() < 0.01);
        assert!((branch.midpoint().y - 2.5).abs() < 0.01);
    }

    #[test]
    fn branch_point_at() {
        let branch = Branch::new(0, None, Vec3::ZERO, Vec3::new(0.0, 10.0, 0.0), 0.3, 0.2, 0);
        let mid = branch.point_at(0.5);
        assert!((mid.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn branch_radius_at() {
        let branch = Branch::new(0, None, Vec3::ZERO, Vec3::UP, 0.4, 0.2, 0);
        let r = branch.radius_at(0.5);
        assert!((r - 0.3).abs() < 0.01);
    }

    #[test]
    fn generate_tree() {
        let config = TreeGenConfig {
            attraction_points: 50,
            max_iterations: 30,
            seed: 123,
            ..TreeGenConfig::default_medium()
        };
        let species = TreeSpecies::oak();
        let mut tree_gen = TreeGenerator::new(config);
        let tree = tree_gen.generate(&species, 0.8);

        assert!(!tree.branches.is_empty());
        assert!(tree.height > 0.0);
        assert!(!tree.species_name.is_empty());
    }

    #[test]
    fn generate_tree_with_leaves() {
        let config = TreeGenConfig {
            attraction_points: 30,
            max_iterations: 20,
            seed: 456,
            ..TreeGenConfig::default_medium()
        };
        let species = TreeSpecies::birch();
        let mut tree_gen = TreeGenerator::new(config);
        let tree = tree_gen.generate(&species, 1.0);

        assert!(!tree.leaves.is_empty());
    }

    #[test]
    fn generate_cactus_no_leaves() {
        let config = TreeGenConfig {
            attraction_points: 20,
            max_iterations: 15,
            seed: 789,
            ..TreeGenConfig::small()
        };
        let species = TreeSpecies::cactus();
        let mut tree_gen = TreeGenerator::new(config);
        let tree = tree_gen.generate(&species, 0.5);

        assert!(tree.leaves.is_empty());
    }

    #[test]
    fn tree_lod_generation() {
        let config = TreeGenConfig {
            attraction_points: 30,
            max_iterations: 20,
            lod_levels: 4,
            seed: 101,
            ..TreeGenConfig::default_medium()
        };
        let species = TreeSpecies::pine();
        let mut tree_gen = TreeGenerator::new(config);
        let tree = tree_gen.generate(&species, 0.7);

        assert_eq!(tree.lod_count(), 4);
        // Each LOD should have fewer branches than the previous.
        for i in 1..tree.lod_branch_counts.len() {
            assert!(tree.lod_branch_counts[i] <= tree.lod_branch_counts[i - 1]);
        }
    }

    #[test]
    fn forest_placement() {
        let species = vec![TreeSpecies::oak(), TreeSpecies::pine(), TreeSpecies::birch()];
        let config = ForestConfig {
            width: 50.0,
            depth: 50.0,
            min_spacing: 3.0,
            max_spacing: 6.0,
            max_trees: 50,
            ..ForestConfig::temperate()
        };

        let placements = generate_forest_placements(&config, &species);

        assert!(!placements.is_empty());
        assert!(placements.len() <= 50);

        // All positions should be within bounds.
        for p in &placements {
            assert!(p.position.0 >= 0.0 && p.position.0 < config.width);
            assert!(p.position.1 >= 0.0 && p.position.1 < config.depth);
        }
    }

    #[test]
    fn forest_minimum_spacing() {
        let species = vec![TreeSpecies::oak()];
        let config = ForestConfig {
            width: 30.0,
            depth: 30.0,
            min_spacing: 5.0,
            max_spacing: 8.0,
            max_trees: 100,
            ..ForestConfig::temperate()
        };

        let placements = generate_forest_placements(&config, &species);

        // Check that no two trees are closer than min_spacing.
        for i in 0..placements.len() {
            for j in (i + 1)..placements.len() {
                let dx = placements[i].position.0 - placements[j].position.0;
                let dz = placements[i].position.1 - placements[j].position.1;
                let dist = (dx * dx + dz * dz).sqrt();
                assert!(
                    dist >= config.min_spacing * 0.99,
                    "Trees too close: {} < {}",
                    dist,
                    config.min_spacing,
                );
            }
        }
    }

    #[test]
    fn species_selection() {
        let all = vec![
            TreeSpecies::oak(),
            TreeSpecies::pine(),
            TreeSpecies::palm(),
            TreeSpecies::cactus(),
        ];

        let temperate = select_species(&all, Biome::Temperate, 500.0, 0.5, 20.0);
        assert!(!temperate.is_empty());
        assert!(temperate.iter().any(|s| s.name == "Oak"));

        let desert = select_species(&all, Biome::Desert, 200.0, 0.1, 35.0);
        assert!(desert.iter().any(|s| s.name == "Cactus"));
    }

    #[test]
    fn deterministic_generation() {
        let config = TreeGenConfig { seed: 999, attraction_points: 30, max_iterations: 20, ..TreeGenConfig::small() };
        let species = TreeSpecies::oak();

        let mut gen1 = TreeGenerator::new(config.clone());
        let mut gen2 = TreeGenerator::new(config);

        let tree1 = gen1.generate(&species, 0.5);
        let tree2 = gen2.generate(&species, 0.5);

        assert_eq!(tree1.branch_count(), tree2.branch_count());
        assert_eq!(tree1.leaf_count(), tree2.leaf_count());
    }
}
