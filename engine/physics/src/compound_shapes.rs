// engine/physics/src/compound_shapes.rs
//
// Compound collision shapes for the Genovo engine.
//
// A compound shape combines multiple primitive shapes (spheres, boxes, capsules)
// into a single collision body. Each child shape can have its own:
// - Local transform (offset + rotation from parent centre of mass)
// - Material properties (friction, restitution)
// - Collision layer
//
// The compound shape maintains an efficient bounding volume hierarchy for
// fast broad-phase culling, and supports adding/removing shapes at runtime.

use glam::{Quat, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Maximum child shapes per compound.
pub const MAX_CHILD_SHAPES: usize = 64;

/// AABB padding for broad-phase.
pub const AABB_PADDING: f32 = 0.01;

// ---------------------------------------------------------------------------
// Primitive shape types
// ---------------------------------------------------------------------------

/// A primitive collision shape.
#[derive(Debug, Clone)]
pub enum PrimitiveShape {
    /// Sphere with radius.
    Sphere { radius: f32 },
    /// Axis-aligned box with half-extents.
    Box { half_extents: Vec3 },
    /// Capsule with radius and half-height along the Y axis.
    Capsule { radius: f32, half_height: f32 },
    /// Cylinder with radius and half-height along the Y axis.
    Cylinder { radius: f32, half_height: f32 },
    /// Cone with radius and height along the Y axis.
    Cone { radius: f32, height: f32 },
    /// Convex hull defined by a set of points.
    ConvexHull { points: Vec<Vec3> },
    /// Plane (infinite, defined by normal and offset).
    Plane { normal: Vec3, offset: f32 },
}

impl PrimitiveShape {
    /// Create a sphere shape.
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a box shape.
    pub fn cuboid(half_x: f32, half_y: f32, half_z: f32) -> Self {
        Self::Box {
            half_extents: Vec3::new(half_x, half_y, half_z),
        }
    }

    /// Create a capsule shape.
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::Capsule { radius, half_height }
    }

    /// Create a cylinder shape.
    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        Self::Cylinder { radius, half_height }
    }

    /// Compute the local AABB of this primitive.
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere { radius } => Aabb {
                min: Vec3::splat(-*radius),
                max: Vec3::splat(*radius),
            },
            Self::Box { half_extents } => Aabb {
                min: -*half_extents,
                max: *half_extents,
            },
            Self::Capsule { radius, half_height } => {
                let h = *half_height + *radius;
                Aabb {
                    min: Vec3::new(-*radius, -h, -*radius),
                    max: Vec3::new(*radius, h, *radius),
                }
            }
            Self::Cylinder { radius, half_height } => Aabb {
                min: Vec3::new(-*radius, -*half_height, -*radius),
                max: Vec3::new(*radius, *half_height, *radius),
            },
            Self::Cone { radius, height } => Aabb {
                min: Vec3::new(-*radius, 0.0, -*radius),
                max: Vec3::new(*radius, *height, *radius),
            },
            Self::ConvexHull { points } => {
                if points.is_empty() {
                    return Aabb::empty();
                }
                let mut min = points[0];
                let mut max = points[0];
                for p in points.iter().skip(1) {
                    min = min.min(*p);
                    max = max.max(*p);
                }
                Aabb { min, max }
            }
            Self::Plane { .. } => Aabb {
                min: Vec3::splat(-1e6),
                max: Vec3::splat(1e6),
            },
        }
    }

    /// Compute the volume of this primitive.
    pub fn volume(&self) -> f32 {
        match self {
            Self::Sphere { radius } => (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius,
            Self::Box { half_extents } => 8.0 * half_extents.x * half_extents.y * half_extents.z,
            Self::Capsule { radius, half_height } => {
                let sphere_vol = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
                let cylinder_vol = std::f32::consts::PI * radius * radius * half_height * 2.0;
                sphere_vol + cylinder_vol
            }
            Self::Cylinder { radius, half_height } => {
                std::f32::consts::PI * radius * radius * half_height * 2.0
            }
            Self::Cone { radius, height } => {
                (1.0 / 3.0) * std::f32::consts::PI * radius * radius * height
            }
            Self::ConvexHull { points } => {
                // Approximate volume from AABB
                let aabb = self.local_aabb();
                let size = aabb.max - aabb.min;
                size.x * size.y * size.z * 0.5 // Rough approximation
            }
            Self::Plane { .. } => f32::MAX,
        }
    }

    /// Compute the inertia tensor (diagonal) for this shape with given mass.
    pub fn inertia_tensor(&self, mass: f32) -> Vec3 {
        match self {
            Self::Sphere { radius } => {
                let i = 0.4 * mass * radius * radius;
                Vec3::splat(i)
            }
            Self::Box { half_extents } => {
                let w2 = (half_extents.x * 2.0).powi(2);
                let h2 = (half_extents.y * 2.0).powi(2);
                let d2 = (half_extents.z * 2.0).powi(2);
                Vec3::new(
                    mass / 12.0 * (h2 + d2),
                    mass / 12.0 * (w2 + d2),
                    mass / 12.0 * (w2 + h2),
                )
            }
            Self::Capsule { radius, half_height } => {
                // Approximate as cylinder + sphere ends
                let r2 = radius * radius;
                let h2 = (half_height * 2.0).powi(2);
                let iy = 0.5 * mass * r2;
                let ixz = mass * (3.0 * r2 + h2) / 12.0;
                Vec3::new(ixz, iy, ixz)
            }
            Self::Cylinder { radius, half_height } => {
                let r2 = radius * radius;
                let h2 = (half_height * 2.0).powi(2);
                let iy = 0.5 * mass * r2;
                let ixz = mass * (3.0 * r2 + h2) / 12.0;
                Vec3::new(ixz, iy, ixz)
            }
            _ => {
                // Fallback: use AABB-based estimation
                let aabb = self.local_aabb();
                let size = aabb.max - aabb.min;
                Vec3::new(
                    mass / 12.0 * (size.y * size.y + size.z * size.z),
                    mass / 12.0 * (size.x * size.x + size.z * size.z),
                    mass / 12.0 * (size.x * size.x + size.y * size.y),
                )
            }
        }
    }

    /// Test if a point is inside this primitive (in local space).
    pub fn contains_point(&self, point: Vec3) -> bool {
        match self {
            Self::Sphere { radius } => point.length_squared() <= radius * radius,
            Self::Box { half_extents } => {
                point.x.abs() <= half_extents.x
                    && point.y.abs() <= half_extents.y
                    && point.z.abs() <= half_extents.z
            }
            Self::Capsule { radius, half_height } => {
                let clamped_y = point.y.clamp(-*half_height, *half_height);
                let closest = Vec3::new(0.0, clamped_y, 0.0);
                (point - closest).length_squared() <= radius * radius
            }
            Self::Cylinder { radius, half_height } => {
                if point.y.abs() > *half_height {
                    return false;
                }
                let dist_xz = (point.x * point.x + point.z * point.z).sqrt();
                dist_xz <= *radius
            }
            _ => false,
        }
    }

    /// Compute the support point in a given direction (for GJK).
    pub fn support(&self, direction: Vec3) -> Vec3 {
        match self {
            Self::Sphere { radius } => {
                let d = direction.normalize_or_zero();
                d * *radius
            }
            Self::Box { half_extents } => Vec3::new(
                if direction.x >= 0.0 { half_extents.x } else { -half_extents.x },
                if direction.y >= 0.0 { half_extents.y } else { -half_extents.y },
                if direction.z >= 0.0 { half_extents.z } else { -half_extents.z },
            ),
            Self::Capsule { radius, half_height } => {
                let d = direction.normalize_or_zero();
                let center = if d.y >= 0.0 {
                    Vec3::new(0.0, *half_height, 0.0)
                } else {
                    Vec3::new(0.0, -*half_height, 0.0)
                };
                center + d * *radius
            }
            Self::ConvexHull { points } => {
                let mut best = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for p in points {
                    let d = p.dot(direction);
                    if d > best_dot {
                        best_dot = d;
                        best = *p;
                    }
                }
                best
            }
            _ => Vec3::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create an empty (inverted) AABB.
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    /// Create from min and max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Expand this AABB to include a point.
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand this AABB to include another AABB.
    pub fn expand_aabb(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Add padding to the AABB.
    pub fn padded(&self, padding: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(padding),
            max: self.max + Vec3::splat(padding),
        }
    }

    /// Test overlap with another AABB.
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Test if a point is inside.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Get the centre of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get the half-extents.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Get the surface area (for BVH heuristics).
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Transform an AABB by a rotation, producing a new AABB.
    pub fn transformed(&self, position: Vec3, rotation: Quat) -> Self {
        // Transform all 8 corners and recompute AABB
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];

        let mut result = Aabb::empty();
        for corner in &corners {
            let transformed = position + rotation * *corner;
            result.expand_point(transformed);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Child shape
// ---------------------------------------------------------------------------

/// Unique identifier for a child shape within a compound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChildShapeId(pub u32);

/// A child shape within a compound, with its own transform and properties.
#[derive(Debug, Clone)]
pub struct ChildShape {
    /// Unique ID within the compound.
    pub id: ChildShapeId,
    /// The primitive shape.
    pub shape: PrimitiveShape,
    /// Local position offset from the compound's centre of mass.
    pub local_position: Vec3,
    /// Local rotation offset from the compound.
    pub local_rotation: Quat,
    /// Per-shape material properties.
    pub material: ShapeMaterial,
    /// Collision layer for this child shape.
    pub collision_layer: u32,
    /// Collision mask for this child shape.
    pub collision_mask: u32,
    /// Whether this shape is active.
    pub enabled: bool,
    /// Local AABB (precomputed).
    pub local_aabb: Aabb,
    /// User data (for game-specific tagging).
    pub user_data: u64,
    /// Name (for debugging).
    pub name: String,
}

/// Per-shape material properties.
#[derive(Debug, Clone, Copy)]
pub struct ShapeMaterial {
    /// Static friction coefficient.
    pub static_friction: f32,
    /// Dynamic friction coefficient.
    pub dynamic_friction: f32,
    /// Restitution (bounciness).
    pub restitution: f32,
    /// Density (for mass computation).
    pub density: f32,
}

impl Default for ShapeMaterial {
    fn default() -> Self {
        Self {
            static_friction: 0.5,
            dynamic_friction: 0.4,
            restitution: 0.3,
            density: 1.0,
        }
    }
}

impl ShapeMaterial {
    /// Create a material with custom friction and restitution.
    pub fn new(static_friction: f32, dynamic_friction: f32, restitution: f32) -> Self {
        Self {
            static_friction,
            dynamic_friction,
            restitution,
            density: 1.0,
        }
    }

    /// A bouncy material.
    pub fn bouncy() -> Self {
        Self {
            static_friction: 0.3,
            dynamic_friction: 0.2,
            restitution: 0.9,
            density: 1.0,
        }
    }

    /// An icy material.
    pub fn ice() -> Self {
        Self {
            static_friction: 0.05,
            dynamic_friction: 0.02,
            restitution: 0.1,
            density: 0.917,
        }
    }

    /// A rubber material.
    pub fn rubber() -> Self {
        Self {
            static_friction: 1.0,
            dynamic_friction: 0.8,
            restitution: 0.8,
            density: 1.1,
        }
    }
}

impl ChildShape {
    /// Create a new child shape.
    pub fn new(
        id: ChildShapeId,
        shape: PrimitiveShape,
        local_position: Vec3,
        local_rotation: Quat,
    ) -> Self {
        let local_aabb = shape.local_aabb();
        Self {
            id,
            shape,
            local_position,
            local_rotation,
            material: ShapeMaterial::default(),
            collision_layer: 1,
            collision_mask: 0xFFFFFFFF,
            enabled: true,
            local_aabb,
            user_data: 0,
            name: String::new(),
        }
    }

    /// Set the material.
    pub fn with_material(mut self, material: ShapeMaterial) -> Self {
        self.material = material;
        self
    }

    /// Set the collision layer.
    pub fn with_layer(mut self, layer: u32) -> Self {
        self.collision_layer = layer;
        self
    }

    /// Set the name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Get the world-space AABB given the parent's transform.
    pub fn world_aabb(&self, parent_position: Vec3, parent_rotation: Quat) -> Aabb {
        let world_pos = parent_position + parent_rotation * self.local_position;
        let world_rot = parent_rotation * self.local_rotation;
        self.local_aabb.transformed(world_pos, world_rot)
    }

    /// Get the mass based on volume and density.
    pub fn mass(&self) -> f32 {
        self.shape.volume() * self.material.density
    }

    /// Transform a world point into this child's local space.
    pub fn world_to_local(&self, point: Vec3, parent_position: Vec3, parent_rotation: Quat) -> Vec3 {
        let world_pos = parent_position + parent_rotation * self.local_position;
        let world_rot = parent_rotation * self.local_rotation;
        let inv_rot = world_rot.conjugate();
        inv_rot * (point - world_pos)
    }

    /// Test if a world-space point is inside this child shape.
    pub fn contains_point_world(
        &self,
        point: Vec3,
        parent_position: Vec3,
        parent_rotation: Quat,
    ) -> bool {
        if !self.enabled {
            return false;
        }
        let local = self.world_to_local(point, parent_position, parent_rotation);
        self.shape.contains_point(local)
    }
}

// ---------------------------------------------------------------------------
// Compound shape
// ---------------------------------------------------------------------------

/// A compound collision shape built from multiple primitives.
#[derive(Debug)]
pub struct CompoundShape {
    /// All child shapes.
    pub children: Vec<ChildShape>,
    /// Next child shape ID.
    next_child_id: u32,
    /// Combined AABB of all children (in local space).
    pub combined_aabb: Aabb,
    /// Computed centre of mass (in local space).
    pub center_of_mass: Vec3,
    /// Total mass.
    pub total_mass: f32,
    /// Combined inertia tensor (diagonal approximation).
    pub inertia_tensor: Vec3,
    /// Whether the compound needs recalculation.
    pub dirty: bool,
}

impl CompoundShape {
    /// Create a new empty compound shape.
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            next_child_id: 0,
            combined_aabb: Aabb::empty(),
            center_of_mass: Vec3::ZERO,
            total_mass: 0.0,
            inertia_tensor: Vec3::ZERO,
            dirty: true,
        }
    }

    /// Create a compound shape from a list of primitives.
    pub fn from_shapes(shapes: Vec<(PrimitiveShape, Vec3, Quat)>) -> Self {
        let mut compound = Self::new();
        for (shape, pos, rot) in shapes {
            compound.add_shape(shape, pos, rot);
        }
        compound.recalculate();
        compound
    }

    /// Add a primitive shape at a local offset.
    pub fn add_shape(
        &mut self,
        shape: PrimitiveShape,
        local_position: Vec3,
        local_rotation: Quat,
    ) -> ChildShapeId {
        let id = ChildShapeId(self.next_child_id);
        self.next_child_id += 1;
        let child = ChildShape::new(id, shape, local_position, local_rotation);
        self.children.push(child);
        self.dirty = true;
        id
    }

    /// Add a shape with full configuration.
    pub fn add_child(&mut self, mut child: ChildShape) -> ChildShapeId {
        let id = ChildShapeId(self.next_child_id);
        self.next_child_id += 1;
        child.id = id;
        self.children.push(child);
        self.dirty = true;
        id
    }

    /// Remove a child shape by ID.
    pub fn remove_shape(&mut self, id: ChildShapeId) -> bool {
        let before = self.children.len();
        self.children.retain(|c| c.id != id);
        let removed = self.children.len() < before;
        if removed {
            self.dirty = true;
        }
        removed
    }

    /// Enable or disable a child shape.
    pub fn set_shape_enabled(&mut self, id: ChildShapeId, enabled: bool) {
        if let Some(child) = self.children.iter_mut().find(|c| c.id == id) {
            child.enabled = enabled;
            self.dirty = true;
        }
    }

    /// Set the material for a child shape.
    pub fn set_shape_material(&mut self, id: ChildShapeId, material: ShapeMaterial) {
        if let Some(child) = self.children.iter_mut().find(|c| c.id == id) {
            child.material = material;
            self.dirty = true;
        }
    }

    /// Set the collision layer for a child shape.
    pub fn set_shape_layer(&mut self, id: ChildShapeId, layer: u32) {
        if let Some(child) = self.children.iter_mut().find(|c| c.id == id) {
            child.collision_layer = layer;
        }
    }

    /// Get a child shape by ID.
    pub fn get_child(&self, id: ChildShapeId) -> Option<&ChildShape> {
        self.children.iter().find(|c| c.id == id)
    }

    /// Get a mutable reference to a child shape.
    pub fn get_child_mut(&mut self, id: ChildShapeId) -> Option<&mut ChildShape> {
        self.children.iter_mut().find(|c| c.id == id)
    }

    /// Recalculate the combined AABB, centre of mass, and inertia tensor.
    pub fn recalculate(&mut self) {
        self.combined_aabb = Aabb::empty();
        self.total_mass = 0.0;
        self.center_of_mass = Vec3::ZERO;
        self.inertia_tensor = Vec3::ZERO;

        let mut weighted_pos = Vec3::ZERO;

        for child in &self.children {
            if !child.enabled {
                continue;
            }

            // Expand combined AABB
            let child_aabb = child.local_aabb.transformed(child.local_position, child.local_rotation);
            self.combined_aabb.expand_aabb(&child_aabb);

            // Mass and centre of mass
            let mass = child.mass();
            weighted_pos += child.local_position * mass;
            self.total_mass += mass;
        }

        if self.total_mass > EPSILON {
            self.center_of_mass = weighted_pos / self.total_mass;
        }

        // Compute combined inertia tensor using parallel axis theorem
        for child in &self.children {
            if !child.enabled {
                continue;
            }

            let mass = child.mass();
            let local_inertia = child.shape.inertia_tensor(mass);

            // Parallel axis theorem: I_total = I_local + m * d^2
            let offset = child.local_position - self.center_of_mass;
            let d2 = offset.length_squared();
            let parallel = Vec3::splat(mass * d2);

            self.inertia_tensor += local_inertia + parallel;
        }

        self.dirty = false;
    }

    /// Get the combined AABB in world space.
    pub fn world_aabb(&self, position: Vec3, rotation: Quat) -> Aabb {
        if self.dirty {
            // Fallback: compute from children directly
            let mut aabb = Aabb::empty();
            for child in &self.children {
                if child.enabled {
                    aabb.expand_aabb(&child.world_aabb(position, rotation));
                }
            }
            return aabb;
        }
        self.combined_aabb.transformed(position, rotation)
    }

    /// Test if a world-space point is inside any child shape.
    pub fn contains_point(&self, point: Vec3, position: Vec3, rotation: Quat) -> bool {
        for child in &self.children {
            if child.contains_point_world(point, position, rotation) {
                return true;
            }
        }
        false
    }

    /// Find which child shape a world point is inside (if any).
    pub fn find_child_at_point(
        &self,
        point: Vec3,
        position: Vec3,
        rotation: Quat,
    ) -> Option<ChildShapeId> {
        for child in &self.children {
            if child.contains_point_world(point, position, rotation) {
                return Some(child.id);
            }
        }
        None
    }

    /// Number of child shapes.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Number of enabled child shapes.
    pub fn enabled_count(&self) -> usize {
        self.children.iter().filter(|c| c.enabled).count()
    }

    /// Compute the support point for the compound shape (for GJK).
    pub fn support(&self, direction: Vec3, rotation: Quat) -> Vec3 {
        let local_dir = rotation.conjugate() * direction;
        let mut best = Vec3::ZERO;
        let mut best_dot = f32::NEG_INFINITY;

        for child in &self.children {
            if !child.enabled {
                continue;
            }
            let child_dir = child.local_rotation.conjugate() * local_dir;
            let child_support = child.shape.support(child_dir);
            let world_support = child.local_position + child.local_rotation * child_support;
            let d = world_support.dot(local_dir);
            if d > best_dot {
                best_dot = d;
                best = world_support;
            }
        }

        rotation * best
    }
}

impl Default for CompoundShape {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compound shape builder
// ---------------------------------------------------------------------------

/// Builder for constructing compound shapes.
#[derive(Debug)]
pub struct CompoundShapeBuilder {
    shapes: Vec<(PrimitiveShape, Vec3, Quat, ShapeMaterial, u32, String)>,
}

impl CompoundShapeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { shapes: Vec::new() }
    }

    /// Add a shape with default material.
    pub fn add(mut self, shape: PrimitiveShape, position: Vec3, rotation: Quat) -> Self {
        self.shapes.push((shape, position, rotation, ShapeMaterial::default(), 1, String::new()));
        self
    }

    /// Add a shape with a custom material.
    pub fn add_with_material(
        mut self,
        shape: PrimitiveShape,
        position: Vec3,
        rotation: Quat,
        material: ShapeMaterial,
    ) -> Self {
        self.shapes.push((shape, position, rotation, material, 1, String::new()));
        self
    }

    /// Add a named shape.
    pub fn add_named(
        mut self,
        name: &str,
        shape: PrimitiveShape,
        position: Vec3,
        rotation: Quat,
    ) -> Self {
        self.shapes.push((shape, position, rotation, ShapeMaterial::default(), 1, name.to_string()));
        self
    }

    /// Build the compound shape.
    pub fn build(self) -> CompoundShape {
        let mut compound = CompoundShape::new();
        for (shape, pos, rot, mat, layer, name) in self.shapes {
            let child = ChildShape::new(
                ChildShapeId(compound.next_child_id),
                shape,
                pos,
                rot,
            )
            .with_material(mat)
            .with_layer(layer)
            .with_name(&name);
            compound.next_child_id += 1;
            compound.children.push(child);
        }
        compound.recalculate();
        compound
    }
}

impl Default for CompoundShapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Preset compound shapes
// ---------------------------------------------------------------------------

/// Create a humanoid capsule compound (body, head, arms, legs).
pub fn humanoid_compound() -> CompoundShape {
    CompoundShapeBuilder::new()
        .add_named("body", PrimitiveShape::capsule(0.3, 0.4), Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY)
        .add_named("head", PrimitiveShape::sphere(0.15), Vec3::new(0.0, 1.7, 0.0), Quat::IDENTITY)
        .add_named("left_arm", PrimitiveShape::capsule(0.08, 0.25), Vec3::new(-0.4, 1.2, 0.0), Quat::IDENTITY)
        .add_named("right_arm", PrimitiveShape::capsule(0.08, 0.25), Vec3::new(0.4, 1.2, 0.0), Quat::IDENTITY)
        .add_named("left_leg", PrimitiveShape::capsule(0.1, 0.35), Vec3::new(-0.15, 0.35, 0.0), Quat::IDENTITY)
        .add_named("right_leg", PrimitiveShape::capsule(0.1, 0.35), Vec3::new(0.15, 0.35, 0.0), Quat::IDENTITY)
        .build()
}

/// Create a simple vehicle compound (body + 4 wheels).
pub fn vehicle_compound() -> CompoundShape {
    CompoundShapeBuilder::new()
        .add_named("body", PrimitiveShape::cuboid(1.0, 0.5, 2.0), Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY)
        .add_named("front_left", PrimitiveShape::sphere(0.3), Vec3::new(-0.9, 0.0, 1.5), Quat::IDENTITY)
        .add_named("front_right", PrimitiveShape::sphere(0.3), Vec3::new(0.9, 0.0, 1.5), Quat::IDENTITY)
        .add_named("rear_left", PrimitiveShape::sphere(0.3), Vec3::new(-0.9, 0.0, -1.5), Quat::IDENTITY)
        .add_named("rear_right", PrimitiveShape::sphere(0.3), Vec3::new(0.9, 0.0, -1.5), Quat::IDENTITY)
        .build()
}

/// ECS component for entities with compound shapes.
#[derive(Debug)]
pub struct CompoundShapeComponent {
    /// The compound shape.
    pub shape: CompoundShape,
    /// Whether the compound has changed and needs BVH rebuild.
    pub dirty: bool,
}

impl CompoundShapeComponent {
    pub fn new(shape: CompoundShape) -> Self {
        Self { shape, dirty: false }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_aabb() {
        let s = PrimitiveShape::sphere(1.0);
        let aabb = s.local_aabb();
        assert!((aabb.min.x - (-1.0)).abs() < EPSILON);
        assert!((aabb.max.x - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_sphere_volume() {
        let s = PrimitiveShape::sphere(1.0);
        let v = s.volume();
        assert!((v - 4.0 / 3.0 * std::f32::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_sphere_contains() {
        let s = PrimitiveShape::sphere(1.0);
        assert!(s.contains_point(Vec3::ZERO));
        assert!(s.contains_point(Vec3::new(0.9, 0.0, 0.0)));
        assert!(!s.contains_point(Vec3::new(1.1, 0.0, 0.0)));
    }

    #[test]
    fn test_compound_basic() {
        let mut c = CompoundShape::new();
        let id1 = c.add_shape(PrimitiveShape::sphere(1.0), Vec3::ZERO, Quat::IDENTITY);
        let id2 = c.add_shape(PrimitiveShape::cuboid(0.5, 0.5, 0.5), Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        c.recalculate();
        assert_eq!(c.child_count(), 2);
        assert!(c.total_mass > 0.0);
        assert!(c.contains_point(Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY));
    }

    #[test]
    fn test_compound_remove() {
        let mut c = CompoundShape::new();
        let id = c.add_shape(PrimitiveShape::sphere(1.0), Vec3::ZERO, Quat::IDENTITY);
        assert_eq!(c.child_count(), 1);
        assert!(c.remove_shape(id));
        assert_eq!(c.child_count(), 0);
    }

    #[test]
    fn test_compound_builder() {
        let c = CompoundShapeBuilder::new()
            .add(PrimitiveShape::sphere(1.0), Vec3::ZERO, Quat::IDENTITY)
            .add(PrimitiveShape::cuboid(1.0, 1.0, 1.0), Vec3::X * 3.0, Quat::IDENTITY)
            .build();
        assert_eq!(c.child_count(), 2);
        assert!(!c.dirty);
    }

    #[test]
    fn test_humanoid_compound() {
        let h = humanoid_compound();
        assert_eq!(h.child_count(), 6);
        assert!(h.total_mass > 0.0);
    }

    #[test]
    fn test_aabb_overlap() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
        assert!(a.overlaps(&b));
        let c = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_support_sphere() {
        let s = PrimitiveShape::sphere(2.0);
        let sup = s.support(Vec3::X);
        assert!((sup.x - 2.0).abs() < 0.01);
    }
}
