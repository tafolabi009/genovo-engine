//! Enhanced scene graph with transformation cache, dirty bit propagation,
//! spatial index integration, scene queries, comparison, and merging.
//!
//! This module extends the base scene graph with:
//!
//! - **Transformation cache** — pre-computed world-space transforms avoid
//!   redundant matrix multiplications during rendering.
//! - **Dirty bit propagation** — when a node's local transform changes, its
//!   subtree is marked dirty and world transforms are lazily recomputed.
//! - **Spatial index integration** — nodes are indexed in an octree/BVH for
//!   fast spatial queries (raycast, frustum cull, radius search).
//! - **Scene queries** — find nodes by name, tag, component type, or spatial
//!   criteria.
//! - **Scene comparison** — diff two scene graphs to detect structural and
//!   data changes.
//! - **Scene merging** — combine two scenes, resolving node ID conflicts.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Math types (simplified)
// ---------------------------------------------------------------------------

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return *self;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

/// Quaternion for rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let c = half.cos();
        let n = axis.normalize();
        Self {
            x: n.x * s,
            y: n.y * s,
            z: n.z * s,
            w: c,
        }
    }

    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) -> Self {
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();
        Self {
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
            w: cr * cp * cy + sr * sp * sy,
        }
    }

    pub fn rotate_vec3(&self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * (2.0 * u.dot(&v)) + v * (s * s - u.dot(&u)) + u.cross(&v) * (2.0 * s)
    }

    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let mut dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let mut target = *other;
        if dot < 0.0 {
            target.x = -target.x;
            target.y = -target.y;
            target.z = -target.z;
            target.w = -target.w;
            dot = -dot;
        }

        if dot > 0.9995 {
            return Self {
                x: self.x + (target.x - self.x) * t,
                y: self.y + (target.y - self.y) * t,
                z: self.z + (target.z - self.z) * t,
                w: self.w + (target.w - self.w) * t,
            };
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Self {
            x: self.x * a + target.x * b,
            y: self.y * a + target.y * b,
            z: self.z * a + target.z * b,
            w: self.w * a + target.w * b,
        }
    }
}

/// 4x4 transformation matrix (column-major).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub fn from_translation(v: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3][0] = v.x;
        m.cols[3][1] = v.y;
        m.cols[3][2] = v.z;
        m
    }

    pub fn from_scale(s: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[0][0] = s.x;
        m.cols[1][1] = s.y;
        m.cols[2][2] = s.z;
        m
    }

    pub fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        let x2 = rotation.x + rotation.x;
        let y2 = rotation.y + rotation.y;
        let z2 = rotation.z + rotation.z;
        let xx = rotation.x * x2;
        let xy = rotation.x * y2;
        let xz = rotation.x * z2;
        let yy = rotation.y * y2;
        let yz = rotation.y * z2;
        let zz = rotation.z * z2;
        let wx = rotation.w * x2;
        let wy = rotation.w * y2;
        let wz = rotation.w * z2;

        Self {
            cols: [
                [(1.0 - (yy + zz)) * scale.x, (xy + wz) * scale.x, (xz - wy) * scale.x, 0.0],
                [(xy - wz) * scale.y, (1.0 - (xx + zz)) * scale.y, (yz + wx) * scale.y, 0.0],
                [(xz + wy) * scale.z, (yz - wx) * scale.z, (1.0 - (xx + yy)) * scale.z, 0.0],
                [translation.x, translation.y, translation.z, 1.0],
            ],
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.cols[0][j] * other.cols[i][0]
                    + self.cols[1][j] * other.cols[i][1]
                    + self.cols[2][j] * other.cols[i][2]
                    + self.cols[3][j] * other.cols[i][3];
            }
        }
        Self { cols: result }
    }

    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        Vec3 {
            x: self.cols[0][0] * p.x + self.cols[1][0] * p.y + self.cols[2][0] * p.z + self.cols[3][0],
            y: self.cols[0][1] * p.x + self.cols[1][1] * p.y + self.cols[2][1] * p.z + self.cols[3][1],
            z: self.cols[0][2] * p.x + self.cols[1][2] * p.y + self.cols[2][2] * p.z + self.cols[3][2],
        }
    }

    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.cols[3][0], self.cols[3][1], self.cols[3][2])
    }
}

// ---------------------------------------------------------------------------
// Scene node ID
// ---------------------------------------------------------------------------

/// Unique identifier for a scene node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneNodeIdV2(pub u64);

impl SceneNodeIdV2 {
    pub const INVALID: Self = Self(u64::MAX);
    pub const ROOT: Self = Self(0);

    pub fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Display for SceneNodeIdV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Local transform
// ---------------------------------------------------------------------------

/// Local transform of a node relative to its parent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl LocalTransform {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Self::IDENTITY
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_trs(self.position, self.rotation, self.scale)
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            rotation: self.rotation.slerp(&other.rotation, t),
            scale: self.scale.lerp(&other.scale, t),
        }
    }
}

impl Default for LocalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ---------------------------------------------------------------------------
// Scene node
// ---------------------------------------------------------------------------

/// A node in the scene graph.
pub struct SceneNodeV2 {
    /// Unique node identifier.
    pub id: SceneNodeIdV2,
    /// Human-readable name.
    pub name: String,
    /// Local transform relative to parent.
    pub local_transform: LocalTransform,
    /// Cached world transform.
    pub world_transform: Mat4,
    /// Parent node (None for root).
    pub parent: Option<SceneNodeIdV2>,
    /// Child node IDs.
    pub children: Vec<SceneNodeIdV2>,
    /// Whether the world transform is dirty and needs recomputation.
    pub dirty: bool,
    /// Whether this node is visible.
    pub visible: bool,
    /// Tags for fast lookup.
    pub tags: Vec<String>,
    /// User data (arbitrary key-value pairs).
    pub user_data: HashMap<String, String>,
    /// Layer mask for rendering/physics.
    pub layer_mask: u32,
    /// Static flag — if true, transform never changes at runtime.
    pub is_static: bool,
}

impl SceneNodeV2 {
    /// Create a new node.
    pub fn new(id: SceneNodeIdV2, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            local_transform: LocalTransform::IDENTITY,
            world_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            dirty: true,
            visible: true,
            tags: Vec::new(),
            user_data: HashMap::new(),
            layer_mask: u32::MAX,
            is_static: false,
        }
    }

    /// Set the local transform and mark as dirty.
    pub fn set_local_transform(&mut self, transform: LocalTransform) {
        self.local_transform = transform;
        self.dirty = true;
    }

    /// Set position only.
    pub fn set_position(&mut self, position: Vec3) {
        self.local_transform.position = position;
        self.dirty = true;
    }

    /// Set rotation only.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.local_transform.rotation = rotation;
        self.dirty = true;
    }

    /// Set scale only.
    pub fn set_scale(&mut self, scale: Vec3) {
        self.local_transform.scale = scale;
        self.dirty = true;
    }

    /// Get the world-space position (from cached transform).
    pub fn world_position(&self) -> Vec3 {
        self.world_transform.translation()
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if the node has a tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

// ---------------------------------------------------------------------------
// Axis-aligned bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box for spatial queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_center_extents(center: Vec3, extents: Vec3) -> Self {
        Self {
            min: center - extents,
            max: center + extents,
        }
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn extents(&self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn merge(&self, other: &AABB) -> AABB {
        AABB {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn expand(&mut self, point: Vec3) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }

    pub fn volume(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial index entry
// ---------------------------------------------------------------------------

/// An entry in the spatial index.
#[derive(Debug, Clone)]
pub struct SpatialEntry {
    pub node_id: SceneNodeIdV2,
    pub bounds: AABB,
    pub world_position: Vec3,
}

/// Simple flat spatial index (for small/medium scenes).
/// For large scenes, this would be replaced with an octree or BVH.
pub struct SpatialIndex {
    entries: Vec<SpatialEntry>,
    /// Dirty flag — set when entries need rebuilding.
    dirty: bool,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            dirty: false,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, entry: SpatialEntry) {
        self.entries.push(entry);
    }

    pub fn remove(&mut self, node_id: SceneNodeIdV2) {
        self.entries.retain(|e| e.node_id != node_id);
    }

    pub fn update_position(&mut self, node_id: SceneNodeIdV2, position: Vec3) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.node_id == node_id) {
            entry.world_position = position;
            let extents = entry.bounds.extents();
            entry.bounds = AABB::from_center_extents(position, extents);
        }
    }

    /// Find all nodes within a sphere.
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> Vec<SceneNodeIdV2> {
        let r_sq = radius * radius;
        self.entries
            .iter()
            .filter(|e| e.world_position.distance(&center).powi(2) <= r_sq + e.bounds.extents().length_squared())
            .map(|e| e.node_id)
            .collect()
    }

    /// Find all nodes within an AABB.
    pub fn query_aabb(&self, aabb: &AABB) -> Vec<SceneNodeIdV2> {
        self.entries
            .iter()
            .filter(|e| e.bounds.intersects(aabb))
            .map(|e| e.node_id)
            .collect()
    }

    /// Find the nearest node to a point.
    pub fn query_nearest(&self, point: Vec3) -> Option<SceneNodeIdV2> {
        self.entries
            .iter()
            .min_by(|a, b| {
                let da = a.world_position.distance(&point);
                let db = b.world_position.distance(&point);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.node_id)
    }

    /// Find the K nearest nodes.
    pub fn query_k_nearest(&self, point: Vec3, k: usize) -> Vec<SceneNodeIdV2> {
        let mut sorted: Vec<_> = self
            .entries
            .iter()
            .map(|e| (e.node_id, e.world_position.distance(&point)))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Ray intersection test. Returns nodes whose AABB intersects the ray.
    pub fn query_ray(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Vec<(SceneNodeIdV2, f32)> {
        let dir_inv = Vec3::new(
            if direction.x.abs() > 1e-10 { 1.0 / direction.x } else { f32::MAX },
            if direction.y.abs() > 1e-10 { 1.0 / direction.y } else { f32::MAX },
            if direction.z.abs() > 1e-10 { 1.0 / direction.z } else { f32::MAX },
        );

        let mut results = Vec::new();

        for entry in &self.entries {
            // Slab method for AABB-ray intersection.
            let t1x = (entry.bounds.min.x - origin.x) * dir_inv.x;
            let t2x = (entry.bounds.max.x - origin.x) * dir_inv.x;
            let t1y = (entry.bounds.min.y - origin.y) * dir_inv.y;
            let t2y = (entry.bounds.max.y - origin.y) * dir_inv.y;
            let t1z = (entry.bounds.min.z - origin.z) * dir_inv.z;
            let t2z = (entry.bounds.max.z - origin.z) * dir_inv.z;

            let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
            let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

            if tmax >= tmin.max(0.0) && tmin < max_distance {
                results.push((entry.node_id, tmin.max(0.0)));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scene graph V2
// ---------------------------------------------------------------------------

/// Enhanced scene graph with transformation caching and spatial indexing.
pub struct SceneGraphV2 {
    /// All nodes indexed by ID.
    nodes: HashMap<SceneNodeIdV2, SceneNodeV2>,
    /// Root node ID.
    root: SceneNodeIdV2,
    /// Next node ID to allocate.
    next_id: u64,
    /// Spatial index for fast queries.
    spatial_index: SpatialIndex,
    /// Nodes that need their world transform recomputed.
    dirty_nodes: HashSet<SceneNodeIdV2>,
    /// Total node count.
    node_count: usize,
}

impl SceneGraphV2 {
    /// Create a new scene graph with a root node.
    pub fn new() -> Self {
        let root_id = SceneNodeIdV2::ROOT;
        let root_node = SceneNodeV2::new(root_id, "Root");

        let mut nodes = HashMap::new();
        nodes.insert(root_id, root_node);

        Self {
            nodes,
            root: root_id,
            next_id: 1,
            spatial_index: SpatialIndex::new(),
            dirty_nodes: HashSet::new(),
            node_count: 1,
        }
    }

    /// Get the root node ID.
    pub fn root(&self) -> SceneNodeIdV2 {
        self.root
    }

    /// Allocate a new node ID.
    fn alloc_id(&mut self) -> SceneNodeIdV2 {
        let id = SceneNodeIdV2(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a new node as a child of `parent`.
    pub fn add_node(
        &mut self,
        parent: SceneNodeIdV2,
        name: impl Into<String>,
    ) -> SceneNodeIdV2 {
        let id = self.alloc_id();
        let mut node = SceneNodeV2::new(id, name);
        node.parent = Some(parent);
        node.dirty = true;

        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.push(id);
        }

        self.dirty_nodes.insert(id);
        self.nodes.insert(id, node);
        self.node_count += 1;
        id
    }

    /// Remove a node and all its descendants.
    pub fn remove_node(&mut self, node_id: SceneNodeIdV2) -> bool {
        if node_id == self.root {
            return false; // Can't remove root.
        }

        // Collect all descendants (BFS).
        let mut to_remove = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_id);

        while let Some(current) = queue.pop_front() {
            to_remove.push(current);
            if let Some(node) = self.nodes.get(&current) {
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }

        // Detach from parent.
        if let Some(node) = self.nodes.get(&node_id) {
            if let Some(parent_id) = node.parent {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.children.retain(|&c| c != node_id);
                }
            }
        }

        // Remove all collected nodes.
        for id in &to_remove {
            self.nodes.remove(id);
            self.spatial_index.remove(*id);
            self.dirty_nodes.remove(id);
            self.node_count -= 1;
        }

        true
    }

    /// Reparent a node under a new parent.
    pub fn reparent(&mut self, node_id: SceneNodeIdV2, new_parent: SceneNodeIdV2) -> bool {
        if node_id == self.root || node_id == new_parent {
            return false;
        }

        // Check that new_parent is not a descendant of node_id.
        if self.is_ancestor(node_id, new_parent) {
            return false;
        }

        // Detach from old parent.
        if let Some(node) = self.nodes.get(&node_id) {
            if let Some(old_parent_id) = node.parent {
                if let Some(old_parent) = self.nodes.get_mut(&old_parent_id) {
                    old_parent.children.retain(|&c| c != node_id);
                }
            }
        }

        // Attach to new parent.
        if let Some(new_parent_node) = self.nodes.get_mut(&new_parent) {
            new_parent_node.children.push(node_id);
        }

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.parent = Some(new_parent);
            node.dirty = true;
        }

        // Mark subtree dirty.
        self.mark_subtree_dirty(node_id);

        true
    }

    /// Check if `ancestor` is an ancestor of `descendant`.
    pub fn is_ancestor(&self, ancestor: SceneNodeIdV2, descendant: SceneNodeIdV2) -> bool {
        let mut current = descendant;
        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent) = node.parent {
                if parent == ancestor {
                    return true;
                }
                current = parent;
            } else {
                break;
            }
        }
        false
    }

    /// Mark a node and its entire subtree as dirty.
    pub fn mark_subtree_dirty(&mut self, node_id: SceneNodeIdV2) {
        let mut queue = VecDeque::new();
        queue.push_back(node_id);

        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.nodes.get_mut(&current) {
                if !node.dirty {
                    node.dirty = true;
                    self.dirty_nodes.insert(current);
                    for &child in &node.children {
                        queue.push_back(child);
                    }
                }
            }
        }
    }

    /// Update world transforms for all dirty nodes.
    ///
    /// Traverses from root, propagating parent transforms to children.
    pub fn update_transforms(&mut self) {
        if self.dirty_nodes.is_empty() {
            return;
        }

        // BFS from root, updating dirty nodes.
        let mut queue = VecDeque::new();
        queue.push_back(self.root);

        while let Some(current_id) = queue.pop_front() {
            let (parent_world, children) = {
                let node = match self.nodes.get(&current_id) {
                    Some(n) => n,
                    None => continue,
                };

                let parent_world = if let Some(parent_id) = node.parent {
                    self.nodes
                        .get(&parent_id)
                        .map(|p| p.world_transform)
                        .unwrap_or(Mat4::IDENTITY)
                } else {
                    Mat4::IDENTITY
                };

                (parent_world, node.children.clone())
            };

            // Update this node's world transform if dirty.
            if let Some(node) = self.nodes.get_mut(&current_id) {
                if node.dirty {
                    let local_matrix = node.local_transform.to_matrix();
                    node.world_transform = parent_world.multiply(&local_matrix);
                    node.dirty = false;

                    // Update spatial index.
                    self.spatial_index
                        .update_position(current_id, node.world_transform.translation());
                }
            }

            for child in children {
                queue.push_back(child);
            }
        }

        self.dirty_nodes.clear();
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: SceneNodeIdV2) -> Option<&SceneNodeV2> {
        self.nodes.get(&id)
    }

    /// Get a mutable node.
    pub fn get_node_mut(&mut self, id: SceneNodeIdV2) -> Option<&mut SceneNodeV2> {
        let node = self.nodes.get_mut(&id)?;
        node.dirty = true;
        self.dirty_nodes.insert(id);
        Some(node)
    }

    /// Set a node's local transform.
    pub fn set_transform(&mut self, id: SceneNodeIdV2, transform: LocalTransform) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.local_transform = transform;
            node.dirty = true;
        }
        self.mark_subtree_dirty(id);
    }

    /// Find nodes by name.
    pub fn find_by_name(&self, name: &str) -> Vec<SceneNodeIdV2> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.name == name)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Find nodes by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<SceneNodeIdV2> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.has_tag(tag))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Find nodes within a radius of a point.
    pub fn find_in_radius(&self, center: Vec3, radius: f32) -> Vec<SceneNodeIdV2> {
        self.spatial_index.query_sphere(center, radius)
    }

    /// Find nodes within an AABB.
    pub fn find_in_aabb(&self, aabb: &AABB) -> Vec<SceneNodeIdV2> {
        self.spatial_index.query_aabb(aabb)
    }

    /// Raycast into the scene.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Vec<(SceneNodeIdV2, f32)> {
        self.spatial_index.query_ray(origin, direction, max_distance)
    }

    /// Get the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Get all node IDs.
    pub fn all_node_ids(&self) -> Vec<SceneNodeIdV2> {
        self.nodes.keys().copied().collect()
    }

    /// Get the depth of a node (root = 0).
    pub fn depth(&self, node_id: SceneNodeIdV2) -> usize {
        let mut depth = 0;
        let mut current = node_id;
        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent) = node.parent {
                depth += 1;
                current = parent;
            } else {
                break;
            }
        }
        depth
    }

    /// Get the path from root to a node.
    pub fn path_to_node(&self, node_id: SceneNodeIdV2) -> Vec<SceneNodeIdV2> {
        let mut path = Vec::new();
        let mut current = node_id;
        while let Some(node) = self.nodes.get(&current) {
            path.push(current);
            if let Some(parent) = node.parent {
                current = parent;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    // -----------------------------------------------------------------------
    // Scene comparison
    // -----------------------------------------------------------------------

    /// Compare this scene graph with another, producing a diff.
    pub fn diff(&self, other: &SceneGraphV2) -> SceneDiffV2 {
        let self_ids: HashSet<_> = self.nodes.keys().copied().collect();
        let other_ids: HashSet<_> = other.nodes.keys().copied().collect();

        let added: Vec<_> = other_ids.difference(&self_ids).copied().collect();
        let removed: Vec<_> = self_ids.difference(&other_ids).copied().collect();

        let mut modified = Vec::new();
        for id in self_ids.intersection(&other_ids) {
            let a = &self.nodes[id];
            let b = &other.nodes[id];

            if a.local_transform != b.local_transform
                || a.name != b.name
                || a.visible != b.visible
                || a.parent != b.parent
            {
                modified.push(NodeDiff {
                    node_id: *id,
                    name_changed: a.name != b.name,
                    transform_changed: a.local_transform != b.local_transform,
                    visibility_changed: a.visible != b.visible,
                    parent_changed: a.parent != b.parent,
                });
            }
        }

        SceneDiffV2 {
            added,
            removed,
            modified,
        }
    }

    // -----------------------------------------------------------------------
    // Scene merging
    // -----------------------------------------------------------------------

    /// Merge another scene graph into this one under a specified parent node.
    ///
    /// Returns a mapping from old node IDs to new node IDs.
    pub fn merge(
        &mut self,
        other: &SceneGraphV2,
        parent: SceneNodeIdV2,
    ) -> HashMap<SceneNodeIdV2, SceneNodeIdV2> {
        let mut id_mapping = HashMap::new();

        // BFS over the other scene, starting from its root.
        let mut queue: VecDeque<(SceneNodeIdV2, SceneNodeIdV2)> = VecDeque::new();
        // Map other's root to our parent.
        queue.push_back((other.root, parent));
        id_mapping.insert(other.root, parent);

        // Process children of other's root.
        if let Some(other_root) = other.nodes.get(&other.root) {
            for &child in &other_root.children {
                let new_id = self.add_node(parent, "");
                id_mapping.insert(child, new_id);
                queue.push_back((child, new_id));
            }
        }

        // Process remaining nodes.
        while let Some((other_id, my_id)) = queue.pop_front() {
            if let Some(other_node) = other.nodes.get(&other_id) {
                // Copy properties.
                if let Some(my_node) = self.nodes.get_mut(&my_id) {
                    my_node.name = other_node.name.clone();
                    my_node.local_transform = other_node.local_transform;
                    my_node.visible = other_node.visible;
                    my_node.tags = other_node.tags.clone();
                    my_node.user_data = other_node.user_data.clone();
                    my_node.layer_mask = other_node.layer_mask;
                    my_node.is_static = other_node.is_static;
                    my_node.dirty = true;
                }

                // Add children.
                for &child in &other_node.children {
                    if !id_mapping.contains_key(&child) {
                        let new_child = self.add_node(my_id, "");
                        id_mapping.insert(child, new_child);
                        queue.push_back((child, new_child));
                    }
                }
            }
        }

        id_mapping
    }

    /// Iterate over all nodes in depth-first order.
    pub fn iter_depth_first(&self) -> Vec<SceneNodeIdV2> {
        let mut result = Vec::new();
        let mut stack = vec![self.root];

        while let Some(current) = stack.pop() {
            result.push(current);
            if let Some(node) = self.nodes.get(&current) {
                for &child in node.children.iter().rev() {
                    stack.push(child);
                }
            }
        }

        result
    }

    /// Iterate over all nodes in breadth-first order.
    pub fn iter_breadth_first(&self) -> Vec<SceneNodeIdV2> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.root);

        while let Some(current) = queue.pop_front() {
            result.push(current);
            if let Some(node) = self.nodes.get(&current) {
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }

        result
    }
}

impl Default for SceneGraphV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scene diff
// ---------------------------------------------------------------------------

/// Diff between two scene graphs.
#[derive(Debug, Clone)]
pub struct SceneDiffV2 {
    pub added: Vec<SceneNodeIdV2>,
    pub removed: Vec<SceneNodeIdV2>,
    pub modified: Vec<NodeDiff>,
}

impl SceneDiffV2 {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }
}

/// Diff for a single node.
#[derive(Debug, Clone)]
pub struct NodeDiff {
    pub node_id: SceneNodeIdV2,
    pub name_changed: bool,
    pub transform_changed: bool,
    pub visibility_changed: bool,
    pub parent_changed: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_graph_basic() {
        let mut graph = SceneGraphV2::new();
        let child = graph.add_node(SceneNodeIdV2::ROOT, "child1");
        assert_eq!(graph.node_count(), 2);

        let grandchild = graph.add_node(child, "grandchild");
        assert_eq!(graph.node_count(), 3);
        assert!(graph.is_ancestor(SceneNodeIdV2::ROOT, grandchild));
    }

    #[test]
    fn transform_propagation() {
        let mut graph = SceneGraphV2::new();
        let child = graph.add_node(SceneNodeIdV2::ROOT, "child");
        graph.set_transform(
            SceneNodeIdV2::ROOT,
            LocalTransform::from_position(Vec3::new(10.0, 0.0, 0.0)),
        );
        graph.set_transform(
            child,
            LocalTransform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        );
        graph.update_transforms();

        let node = graph.get_node(child).unwrap();
        let world_pos = node.world_position();
        assert!((world_pos.x - 15.0).abs() < 0.01);
    }

    #[test]
    fn remove_subtree() {
        let mut graph = SceneGraphV2::new();
        let c1 = graph.add_node(SceneNodeIdV2::ROOT, "c1");
        let _c2 = graph.add_node(c1, "c2");
        let _c3 = graph.add_node(c1, "c3");
        assert_eq!(graph.node_count(), 4);

        graph.remove_node(c1);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn find_by_name_and_tag() {
        let mut graph = SceneGraphV2::new();
        let n = graph.add_node(SceneNodeIdV2::ROOT, "player");
        graph.get_node_mut(n).unwrap().add_tag("movable");

        assert_eq!(graph.find_by_name("player").len(), 1);
        assert_eq!(graph.find_by_tag("movable").len(), 1);
    }

    #[test]
    fn scene_diff() {
        let graph_a = SceneGraphV2::new();
        let mut graph_b = SceneGraphV2::new();
        graph_b.add_node(SceneNodeIdV2::ROOT, "new_node");

        let diff = graph_a.diff(&graph_b);
        assert_eq!(diff.added.len(), 1);
    }
}
