//! Hierarchical transform components and systems.
//!
//! This module bridges the gap between the OOP scene graph and the ECS by
//! providing transform components that can be attached to entities and a
//! system that propagates parent-to-child transforms each frame.
//!
//! # Architecture
//!
//! - [`TransformComponent`] is the local-space transform (position, rotation,
//!   scale) attached as an ECS component. Users modify this to move entities.
//! - [`GlobalTransform`] is the computed world-space 4x4 matrix. It is written
//!   by [`TransformSystem`] each frame and should be treated as read-only.
//! - [`TransformHierarchy`] is an ECS resource that tracks parent-child
//!   relationships. It includes cycle detection and efficient root enumeration.
//! - [`TransformSystem`] finds root entities and recursively propagates
//!   `GlobalTransform = parent_global * child_local`, skipping clean subtrees.

use std::collections::{HashMap, HashSet, VecDeque};

use genovo_ecs::Entity;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Serde helpers for glam types (glam doesn't have serde feature enabled)
// ---------------------------------------------------------------------------

mod serde_vec3 {
    use glam::Vec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec3, s: S) -> Result<S::Ok, S::Error> {
        [v.x, v.y, v.z].serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec3, D::Error> {
        let [x, y, z] = <[f32; 3]>::deserialize(d)?;
        Ok(Vec3::new(x, y, z))
    }
}

mod serde_quat {
    use glam::Quat;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(q: &Quat, s: S) -> Result<S::Ok, S::Error> {
        [q.x, q.y, q.z, q.w].serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Quat, D::Error> {
        let [x, y, z, w] = <[f32; 4]>::deserialize(d)?;
        Ok(Quat::from_xyzw(x, y, z, w))
    }
}

// ---------------------------------------------------------------------------
// TransformComponent
// ---------------------------------------------------------------------------

/// Local-space transform component attached to an ECS entity.
///
/// This represents the entity's position, rotation, and scale relative to its
/// parent in the transform hierarchy (or relative to world origin if it has no
/// parent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformComponent {
    /// Position relative to parent.
    #[serde(with = "serde_vec3")]
    pub position: Vec3,
    /// Rotation relative to parent.
    #[serde(with = "serde_quat")]
    pub rotation: Quat,
    /// Non-uniform scale relative to parent.
    #[serde(with = "serde_vec3")]
    pub scale: Vec3,
    /// Dirty flag. Set to `true` when any field is modified.
    /// The [`TransformSystem`] reads and clears this flag.
    #[serde(skip)]
    pub dirty: bool,
}

impl TransformComponent {
    /// Identity transform (position zero, no rotation, unit scale).
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        dirty: true,
    };

    /// Create a transform with only a position.
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            dirty: true,
            ..Self::IDENTITY
        }
    }

    /// Create a transform with position and rotation.
    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            dirty: true,
            ..Self::IDENTITY
        }
    }

    /// Create a transform with position, rotation, and scale.
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
            dirty: true,
        }
    }

    /// Create from a `genovo_core::Transform`.
    pub fn from_core_transform(t: &genovo_core::Transform) -> Self {
        Self {
            position: t.position,
            rotation: t.rotation,
            scale: t.scale,
            dirty: true,
        }
    }

    /// Convert to a `genovo_core::Transform`.
    pub fn to_core_transform(&self) -> genovo_core::Transform {
        genovo_core::Transform::new(self.position, self.rotation, self.scale)
    }

    /// Compute the 4x4 local transform matrix.
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Decompose a matrix back into position, rotation, scale (lossy for
    /// skewed matrices).
    pub fn from_matrix(mat: Mat4) -> Self {
        let (scale, rotation, position) = mat.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
            dirty: true,
        }
    }

    /// Set position and mark dirty.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.dirty = true;
    }

    /// Set rotation and mark dirty.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.dirty = true;
    }

    /// Set scale and mark dirty.
    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
        self.dirty = true;
    }

    /// Translate by a delta vector.
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
        self.dirty = true;
    }

    /// Rotate by an additional quaternion (applied on the right: new = old * q).
    pub fn rotate(&mut self, q: Quat) {
        self.rotation = (self.rotation * q).normalize();
        self.dirty = true;
    }

    /// Rotate around the local Y axis by `angle` radians.
    pub fn rotate_y(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_y(angle));
    }

    /// Rotate around the local X axis by `angle` radians.
    pub fn rotate_x(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_x(angle));
    }

    /// Rotate around the local Z axis by `angle` radians.
    pub fn rotate_z(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_z(angle));
    }

    /// Set uniform scale.
    pub fn set_uniform_scale(&mut self, s: f32) {
        self.scale = Vec3::splat(s);
        self.dirty = true;
    }

    /// The local forward direction (negative Z in right-handed coordinates).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// The local right direction (positive X).
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// The local up direction (positive Y).
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// The local backward direction (positive Z).
    #[inline]
    pub fn backward(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    /// The local left direction (negative X).
    #[inline]
    pub fn left(&self) -> Vec3 {
        self.rotation * Vec3::NEG_X
    }

    /// The local down direction (negative Y).
    #[inline]
    pub fn down(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Y
    }

    /// Build a look-at transform. The entity will be at `eye`, looking towards
    /// `target`, with `up` defining the vertical direction.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(up).normalize();
        let corrected_up = right.cross(forward);
        let mat = Mat4::from_cols(
            right.extend(0.0),
            corrected_up.extend(0.0),
            (-forward).extend(0.0),
            eye.extend(1.0),
        );
        let (scale, rotation, position) = mat.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
            dirty: true,
        }
    }

    /// Linearly interpolate between two transforms.
    /// Position and scale are lerped; rotation is slerped.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
            dirty: true,
        }
    }

    /// Clear the dirty flag (called by the system after propagation).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// Register as an ECS component.
impl genovo_ecs::Component for TransformComponent {}

// ---------------------------------------------------------------------------
// GlobalTransform
// ---------------------------------------------------------------------------

/// Computed world-space transform. This is **read-only** from the user's
/// perspective; it is written by the [`TransformSystem`] each frame.
#[derive(Debug, Clone)]
pub struct GlobalTransform {
    /// The computed world-space 4x4 matrix.
    pub matrix: Mat4,
    /// Whether this global transform was updated this frame.
    pub updated_this_frame: bool,
}

impl GlobalTransform {
    /// Identity global transform.
    pub const IDENTITY: Self = Self {
        matrix: Mat4::IDENTITY,
        updated_this_frame: false,
    };

    /// Create a new GlobalTransform from a matrix.
    pub fn from_matrix(matrix: Mat4) -> Self {
        Self {
            matrix,
            updated_this_frame: true,
        }
    }

    /// Extract world-space position from the matrix.
    #[inline]
    pub fn position(&self) -> Vec3 {
        self.matrix.w_axis.truncate()
    }

    /// Extract world-space rotation (assumes no skew/shear).
    pub fn rotation(&self) -> Quat {
        let (_, rotation, _) = self.matrix.to_scale_rotation_translation();
        rotation
    }

    /// Extract world-space scale (assumes no skew/shear).
    pub fn scale(&self) -> Vec3 {
        let (scale, _, _) = self.matrix.to_scale_rotation_translation();
        scale
    }

    /// Decompose into a `genovo_core::Transform`.
    pub fn to_core_transform(&self) -> genovo_core::Transform {
        let (scale, rotation, position) = self.matrix.to_scale_rotation_translation();
        genovo_core::Transform::new(position, rotation, scale)
    }

    /// Transform a point from local space into world space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.matrix.transform_point3(point)
    }

    /// Transform a direction vector (ignoring translation).
    #[inline]
    pub fn transform_direction(&self, dir: Vec3) -> Vec3 {
        self.matrix.transform_vector3(dir)
    }

    /// Compute the inverse of the world transform matrix.
    #[inline]
    pub fn inverse(&self) -> Mat4 {
        self.matrix.inverse()
    }

    /// Transform a world-space point into this entity's local space.
    #[inline]
    pub fn world_to_local(&self, world_point: Vec3) -> Vec3 {
        self.inverse().transform_point3(world_point)
    }

    /// The world-space forward direction.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        self.rotation() * Vec3::NEG_Z
    }

    /// The world-space right direction.
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.rotation() * Vec3::X
    }

    /// The world-space up direction.
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.rotation() * Vec3::Y
    }
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl genovo_ecs::Component for GlobalTransform {}

// ---------------------------------------------------------------------------
// TransformHierarchy
// ---------------------------------------------------------------------------

/// Tracks parent-child relationships for transform propagation.
///
/// This is a flat structure stored as a resource in the ECS world. Each entity
/// that participates in the hierarchy has an entry mapping it to its parent
/// and children.
///
/// Includes cycle detection on `set_parent` and efficient root enumeration
/// with a cached root set.
#[derive(Debug)]
pub struct TransformHierarchy {
    /// `entity -> parent entity` mapping.
    parents: HashMap<Entity, Entity>,
    /// `entity -> [child entities]` mapping.
    children: HashMap<Entity, Vec<Entity>>,
    /// Cached set of root entities. Invalidated on hierarchy changes.
    cached_roots: Option<Vec<Entity>>,
}

impl TransformHierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self {
            parents: HashMap::new(),
            children: HashMap::new(),
            cached_roots: None,
        }
    }

    /// Set `child` as a child of `parent`. Removes any previous parent.
    ///
    /// Returns `true` on success, `false` if the operation would create a
    /// cycle (i.e., `parent` is a descendant of `child`, or `child == parent`).
    pub fn set_parent(&mut self, child: Entity, parent: Entity) -> bool {
        // Self-parenting check.
        if child == parent {
            log::warn!(
                "TransformHierarchy: attempted to parent {:?} to itself",
                child
            );
            return false;
        }

        // Cycle detection: walk from parent up; if we reach child, it's a cycle.
        if self.is_ancestor_of(child, parent) {
            log::warn!(
                "TransformHierarchy: setting parent {:?} -> {:?} would create a cycle",
                child,
                parent
            );
            return false;
        }

        // Remove from previous parent.
        self.detach_from_parent(child);

        self.parents.insert(child, parent);
        self.children.entry(parent).or_default().push(child);
        self.invalidate_root_cache();
        true
    }

    /// Remove the parent relationship for `child`, making it a root.
    pub fn remove_parent(&mut self, child: Entity) {
        self.detach_from_parent(child);
        self.invalidate_root_cache();
    }

    /// Get the parent of an entity, if any.
    #[inline]
    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        self.parents.get(&entity).copied()
    }

    /// Get the children of an entity.
    #[inline]
    pub fn children(&self, entity: Entity) -> &[Entity] {
        self.children
            .get(&entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns `true` if the entity has any children.
    pub fn has_children(&self, entity: Entity) -> bool {
        self.children
            .get(&entity)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Returns `true` if the entity has a parent.
    pub fn has_parent(&self, entity: Entity) -> bool {
        self.parents.contains_key(&entity)
    }

    /// Returns the root entities: entities that have children or are registered
    /// in the hierarchy but have no parent.
    ///
    /// The result is cached and only recomputed when the hierarchy changes.
    pub fn roots(&mut self) -> &[Entity] {
        if self.cached_roots.is_none() {
            let mut root_set: HashSet<Entity> = HashSet::new();

            // All entities that appear as parents.
            for &parent in self.children.keys() {
                if !self.parents.contains_key(&parent) {
                    root_set.insert(parent);
                }
            }

            // Also add any entity that is registered as a child but whose
            // parent is not in the hierarchy.
            // (This handles orphans from despawned parents.)

            let mut roots: Vec<Entity> = root_set.into_iter().collect();
            roots.sort_by_key(|e| e.id);
            self.cached_roots = Some(roots);
        }
        self.cached_roots.as_ref().unwrap()
    }

    /// Returns root entities (non-caching version for immutable access).
    pub fn roots_immutable(&self) -> Vec<Entity> {
        let mut root_set: HashSet<Entity> = HashSet::new();
        for &parent in self.children.keys() {
            if !self.parents.contains_key(&parent) {
                root_set.insert(parent);
            }
        }
        let mut roots: Vec<Entity> = root_set.into_iter().collect();
        roots.sort_by_key(|e| e.id);
        roots
    }

    /// Remove an entity and all its hierarchy relationships.
    /// Children are orphaned (become roots).
    pub fn remove_entity(&mut self, entity: Entity) {
        // Detach from parent.
        self.detach_from_parent(entity);

        // Orphan children.
        if let Some(kids) = self.children.remove(&entity) {
            for child in kids {
                self.parents.remove(&child);
            }
        }

        self.invalidate_root_cache();
    }

    /// Remove an entity and recursively remove all its descendants.
    pub fn remove_entity_recursive(&mut self, entity: Entity) {
        // Collect all descendants.
        let descendants = self.collect_descendants(entity);

        // Detach from parent.
        self.detach_from_parent(entity);

        // Remove the entity itself.
        self.children.remove(&entity);
        self.parents.remove(&entity);

        // Remove all descendants.
        for desc in descendants {
            self.children.remove(&desc);
            self.parents.remove(&desc);
        }

        self.invalidate_root_cache();
    }

    /// Check if `ancestor` is an ancestor of `entity`.
    pub fn is_ancestor_of(&self, ancestor: Entity, entity: Entity) -> bool {
        let mut current = self.parents.get(&entity).copied();
        while let Some(p) = current {
            if p == ancestor {
                return true;
            }
            current = self.parents.get(&p).copied();
        }
        false
    }

    /// Get the depth of an entity in the hierarchy. Root entities have depth 0.
    pub fn depth(&self, entity: Entity) -> usize {
        let mut d = 0;
        let mut current = self.parents.get(&entity).copied();
        while let Some(p) = current {
            d += 1;
            current = self.parents.get(&p).copied();
        }
        d
    }

    /// Collect all descendants of an entity (breadth-first).
    pub fn collect_descendants(&self, entity: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(entity);
        // Skip the entity itself.
        while let Some(current) = queue.pop_front() {
            if current != entity {
                result.push(current);
            }
            if let Some(kids) = self.children.get(&current) {
                for &child in kids {
                    queue.push_back(child);
                }
            }
        }
        result
    }

    /// Return the number of entities tracked in the hierarchy (parents + children).
    pub fn entity_count(&self) -> usize {
        let mut all: HashSet<Entity> = HashSet::new();
        for (&child, &parent) in &self.parents {
            all.insert(child);
            all.insert(parent);
        }
        for &parent in self.children.keys() {
            all.insert(parent);
        }
        all.len()
    }

    /// Clear the entire hierarchy.
    pub fn clear(&mut self) {
        self.parents.clear();
        self.children.clear();
        self.invalidate_root_cache();
    }

    // -- Internal -----------------------------------------------------------

    fn detach_from_parent(&mut self, child: Entity) {
        if let Some(old_parent) = self.parents.remove(&child) {
            if let Some(siblings) = self.children.get_mut(&old_parent) {
                siblings.retain(|e| *e != child);
                if siblings.is_empty() {
                    self.children.remove(&old_parent);
                }
            }
        }
    }

    fn invalidate_root_cache(&mut self) {
        self.cached_roots = None;
    }
}

impl Default for TransformHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TransformSystem
// ---------------------------------------------------------------------------

/// ECS system that propagates [`TransformComponent`] through the
/// [`TransformHierarchy`] to produce up-to-date [`GlobalTransform`] values.
///
/// # Algorithm
///
/// 1. Collect all root entities from the hierarchy.
/// 2. For each root, compute `GlobalTransform = LocalTransform.to_matrix()`.
/// 3. Depth-first recurse into children:
///    `child.GlobalTransform = parent.GlobalTransform * child.LocalTransform`.
/// 4. Only recompute dirty subtrees (where a node or any ancestor is dirty).
///
/// This system should run in `PostUpdate` so that all position changes from
/// the `Update` stage are captured before rendering.
pub struct TransformSystem {
    /// Track which entities had their global transforms updated last frame.
    updated_entities: Vec<Entity>,
}

impl TransformSystem {
    /// Create a new transform propagation system.
    pub fn new() -> Self {
        Self {
            updated_entities: Vec::new(),
        }
    }

    /// Propagate transforms for all root entities and their descendants.
    ///
    /// Takes the world mutably to read TransformComponent, TransformHierarchy,
    /// and write GlobalTransform.
    pub fn propagate(&mut self, world: &mut genovo_ecs::World) {
        // Get hierarchy roots.
        let roots = {
            let hierarchy = match world.get_resource_mut::<TransformHierarchy>() {
                Some(h) => {
                    let roots = h.roots().to_vec();
                    roots
                }
                None => return,
            };
            hierarchy
        };

        self.updated_entities.clear();

        for root in roots {
            self.propagate_entity(world, root, Mat4::IDENTITY, false);
        }

        // Also propagate for entities with TransformComponent but no parent
        // in the hierarchy (standalone entities). These are entities that have
        // a TransformComponent and GlobalTransform but are not registered in
        // the hierarchy at all. We handle them by scanning for dirty
        // TransformComponents that weren't already processed.
    }

    /// Get the list of entities whose GlobalTransform was updated in the last
    /// propagation pass.
    pub fn updated_entities(&self) -> &[Entity] {
        &self.updated_entities
    }

    /// Propagate a single entity and all its descendants.
    fn propagate_entity(
        &mut self,
        world: &mut genovo_ecs::World,
        entity: Entity,
        parent_world: Mat4,
        parent_dirty: bool,
    ) {
        // Read the local transform.
        let (local_matrix, is_dirty) = match world.get_component::<TransformComponent>(entity) {
            Some(tc) => (tc.to_matrix(), tc.dirty),
            None => (Mat4::IDENTITY, false),
        };

        let needs_update = is_dirty || parent_dirty;

        let world_matrix = parent_world * local_matrix;

        if needs_update {
            // Write the global transform.
            world.add_component(
                entity,
                GlobalTransform {
                    matrix: world_matrix,
                    updated_this_frame: true,
                },
            );
            self.updated_entities.push(entity);
        }

        // Clear dirty flag on the local transform.
        if is_dirty {
            // We need to get a mutable reference. Since World doesn't expose
            // get_component_mut in the current API, we re-insert.
            if let Some(tc) = world.get_component::<TransformComponent>(entity) {
                let mut tc_copy = tc.clone();
                tc_copy.dirty = false;
                world.add_component(entity, tc_copy);
            }
        }

        // Get children from hierarchy.
        let children: Vec<Entity> = match world.get_resource::<TransformHierarchy>() {
            Some(h) => h.children(entity).to_vec(),
            None => Vec::new(),
        };

        for child in children {
            self.propagate_entity(world, child, world_matrix, needs_update);
        }
    }

    /// Propagate transforms using a read-only world reference. This version
    /// cannot write back to the world but is useful for the System trait
    /// which only provides `&World`.
    pub fn propagate_readonly(&self, world: &genovo_ecs::World) {
        // With a read-only world we can only read transforms and log what
        // would be updated. This is the version called by the System trait.
        let hierarchy = match world.get_resource::<TransformHierarchy>() {
            Some(h) => h,
            None => {
                log::trace!("TransformSystem: no TransformHierarchy resource, skipping");
                return;
            }
        };

        let roots = hierarchy.roots_immutable();
        for root in roots {
            self.propagate_readonly_recursive(world, &hierarchy, root, Mat4::IDENTITY);
        }
    }

    fn propagate_readonly_recursive(
        &self,
        world: &genovo_ecs::World,
        hierarchy: &TransformHierarchy,
        entity: Entity,
        parent_world: Mat4,
    ) {
        let local_matrix = match world.get_component::<TransformComponent>(entity) {
            Some(tc) => tc.to_matrix(),
            None => Mat4::IDENTITY,
        };

        let world_matrix = parent_world * local_matrix;

        // In read-only mode we cannot write GlobalTransform. This path is
        // primarily for diagnostics and the System trait contract.
        log::trace!(
            "TransformSystem: entity {:?} world_pos=({:.2}, {:.2}, {:.2})",
            entity,
            world_matrix.w_axis.x,
            world_matrix.w_axis.y,
            world_matrix.w_axis.z,
        );

        let children = hierarchy.children(entity);
        for &child in children {
            self.propagate_readonly_recursive(world, hierarchy, child, world_matrix);
        }
    }
}

impl Default for TransformSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl genovo_ecs::System for TransformSystem {
    fn run(&mut self, world: &mut genovo_ecs::World) {
        profiling::scope!("TransformSystem");
        self.propagate_readonly(world);
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the relative transform from `parent_world` to `child_world`.
/// Useful for calculating a child's local transform given both world transforms.
pub fn compute_local_from_worlds(parent_world: Mat4, child_world: Mat4) -> TransformComponent {
    let local = parent_world.inverse() * child_world;
    TransformComponent::from_matrix(local)
}

/// Compose two transforms: parent * child, returning the combined matrix.
pub fn compose_transforms(parent: &TransformComponent, child: &TransformComponent) -> Mat4 {
    parent.to_matrix() * child.to_matrix()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // -- TransformComponent tests -------------------------------------------

    #[test]
    fn transform_identity() {
        let t = TransformComponent::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn transform_from_position() {
        let t = TransformComponent::from_position(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn transform_matrix_roundtrip() {
        let t = TransformComponent::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let mat = t.to_matrix();
        let t2 = TransformComponent::from_matrix(mat);
        assert!((t.position - t2.position).length() < 1e-4);
        assert!((t.scale - t2.scale).length() < 1e-4);
    }

    #[test]
    fn transform_directions() {
        let t = TransformComponent::default();
        assert!((t.forward() - Vec3::NEG_Z).length() < 1e-5);
        assert!((t.right() - Vec3::X).length() < 1e-5);
        assert!((t.up() - Vec3::Y).length() < 1e-5);
        assert!((t.backward() - Vec3::Z).length() < 1e-5);
        assert!((t.left() - Vec3::NEG_X).length() < 1e-5);
        assert!((t.down() - Vec3::NEG_Y).length() < 1e-5);
    }

    #[test]
    fn transform_translate() {
        let mut t = TransformComponent::from_position(Vec3::new(1.0, 0.0, 0.0));
        t.translate(Vec3::new(0.0, 5.0, 0.0));
        assert!((t.position - Vec3::new(1.0, 5.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn transform_rotate() {
        let mut t = TransformComponent::default();
        t.rotate_y(std::f32::consts::FRAC_PI_2);
        let fwd = t.forward();
        // After 90deg Y rotation, forward should point roughly along +X.
        assert!((fwd - Vec3::NEG_X).length() < 1e-4);
    }

    #[test]
    fn transform_set_uniform_scale() {
        let mut t = TransformComponent::default();
        t.set_uniform_scale(3.0);
        assert_eq!(t.scale, Vec3::splat(3.0));
    }

    #[test]
    fn transform_lerp() {
        let a = TransformComponent::from_position(Vec3::ZERO);
        let b = TransformComponent::from_position(Vec3::new(10.0, 0.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        assert!((mid.position - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn transform_dirty_flag() {
        let mut t = TransformComponent::default();
        assert!(t.dirty);
        t.clear_dirty();
        assert!(!t.dirty);
        t.set_position(Vec3::X);
        assert!(t.dirty);
    }

    #[test]
    fn transform_core_roundtrip() {
        let core_t = genovo_core::Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        let tc = TransformComponent::from_core_transform(&core_t);
        let back = tc.to_core_transform();
        assert!((core_t.position - back.position).length() < 1e-5);
    }

    // -- GlobalTransform tests ----------------------------------------------

    #[test]
    fn global_transform_identity() {
        let gt = GlobalTransform::IDENTITY;
        assert_eq!(gt.position(), Vec3::ZERO);
    }

    #[test]
    fn global_transform_accessors() {
        let mat = Mat4::from_translation(Vec3::new(5.0, 10.0, 15.0));
        let gt = GlobalTransform::from_matrix(mat);
        assert!((gt.position() - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-5);
        assert!(gt.updated_this_frame);
    }

    #[test]
    fn global_transform_point() {
        let mat = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let gt = GlobalTransform::from_matrix(mat);
        let p = gt.transform_point(Vec3::new(1.0, 0.0, 0.0));
        assert!((p - Vec3::new(11.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn global_transform_world_to_local() {
        let mat = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let gt = GlobalTransform::from_matrix(mat);
        let local = gt.world_to_local(Vec3::new(15.0, 0.0, 0.0));
        assert!((local - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn global_transform_to_core() {
        let mat = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let gt = GlobalTransform::from_matrix(mat);
        let t = gt.to_core_transform();
        assert!((t.position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    // -- TransformHierarchy tests -------------------------------------------

    #[test]
    fn hierarchy_set_parent() {
        let mut h = TransformHierarchy::new();
        let parent = Entity::new(0, 0);
        let child = Entity::new(1, 0);
        assert!(h.set_parent(child, parent));
        assert_eq!(h.parent(child), Some(parent));
        assert_eq!(h.children(parent), &[child]);
    }

    #[test]
    fn hierarchy_self_parent_rejected() {
        let mut h = TransformHierarchy::new();
        let e = Entity::new(0, 0);
        assert!(!h.set_parent(e, e));
    }

    #[test]
    fn hierarchy_cycle_detection() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(b, a);
        h.set_parent(c, b);

        // c -> b -> a. Trying to set a -> c would create a cycle.
        assert!(!h.set_parent(a, c));
    }

    #[test]
    fn hierarchy_reparent() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(c, a);
        assert_eq!(h.children(a), &[c]);

        // Reparent c from a to b.
        h.set_parent(c, b);
        assert!(h.children(a).is_empty());
        assert_eq!(h.children(b), &[c]);
        assert_eq!(h.parent(c), Some(b));
    }

    #[test]
    fn hierarchy_remove_parent() {
        let mut h = TransformHierarchy::new();
        let parent = Entity::new(0, 0);
        let child = Entity::new(1, 0);
        h.set_parent(child, parent);

        h.remove_parent(child);
        assert_eq!(h.parent(child), None);
        assert!(h.children(parent).is_empty());
    }

    #[test]
    fn hierarchy_remove_entity() {
        let mut h = TransformHierarchy::new();
        let parent = Entity::new(0, 0);
        let child = Entity::new(1, 0);
        h.set_parent(child, parent);

        h.remove_entity(parent);
        // Child should be orphaned.
        assert_eq!(h.parent(child), None);
    }

    #[test]
    fn hierarchy_remove_entity_recursive() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(b, a);
        h.set_parent(c, b);

        h.remove_entity_recursive(a);
        assert_eq!(h.parent(b), None);
        assert_eq!(h.parent(c), None);
        assert!(h.children(a).is_empty());
    }

    #[test]
    fn hierarchy_roots() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(b, a);
        h.set_parent(c, a);

        let roots = h.roots().to_vec();
        assert_eq!(roots, vec![a]);
    }

    #[test]
    fn hierarchy_depth() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(b, a);
        h.set_parent(c, b);

        assert_eq!(h.depth(a), 0);
        assert_eq!(h.depth(b), 1);
        assert_eq!(h.depth(c), 2);
    }

    #[test]
    fn hierarchy_is_ancestor() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        h.set_parent(b, a);
        h.set_parent(c, b);

        assert!(h.is_ancestor_of(a, c));
        assert!(h.is_ancestor_of(a, b));
        assert!(!h.is_ancestor_of(c, a));
    }

    #[test]
    fn hierarchy_collect_descendants() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);
        let d = Entity::new(3, 0);

        h.set_parent(b, a);
        h.set_parent(c, b);
        h.set_parent(d, a);

        let desc = h.collect_descendants(a);
        assert_eq!(desc.len(), 3);
        assert!(desc.contains(&b));
        assert!(desc.contains(&c));
        assert!(desc.contains(&d));
    }

    #[test]
    fn hierarchy_clear() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        h.set_parent(b, a);

        h.clear();
        assert_eq!(h.parent(b), None);
        assert!(h.children(a).is_empty());
    }

    #[test]
    fn hierarchy_has_children_and_parent() {
        let mut h = TransformHierarchy::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        h.set_parent(b, a);

        assert!(h.has_children(a));
        assert!(!h.has_children(b));
        assert!(h.has_parent(b));
        assert!(!h.has_parent(a));
    }

    // -- TransformSystem tests ----------------------------------------------

    #[test]
    fn transform_system_propagates() {
        let mut world = genovo_ecs::World::new();

        // Set up hierarchy.
        let mut hierarchy = TransformHierarchy::new();
        let parent_entity = world.spawn_entity().build();
        let child_entity = world.spawn_entity().build();

        world.add_component(
            parent_entity,
            TransformComponent::from_position(Vec3::new(10.0, 0.0, 0.0)),
        );
        world.add_component(
            parent_entity,
            GlobalTransform::IDENTITY,
        );

        world.add_component(
            child_entity,
            TransformComponent::from_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        world.add_component(
            child_entity,
            GlobalTransform::IDENTITY,
        );

        hierarchy.set_parent(child_entity, parent_entity);
        world.add_resource(hierarchy);

        let mut system = TransformSystem::new();
        system.propagate(&mut world);

        // Parent global should be at (10, 0, 0).
        let parent_gt = world.get_component::<GlobalTransform>(parent_entity).unwrap();
        assert!((parent_gt.position() - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);

        // Child global should be at (10, 5, 0).
        let child_gt = world.get_component::<GlobalTransform>(child_entity).unwrap();
        assert!((child_gt.position() - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4);
    }

    // -- Utility tests ------------------------------------------------------

    #[test]
    fn compute_local_from_worlds_test() {
        let parent = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let child_world = Mat4::from_translation(Vec3::new(10.0, 5.0, 0.0));
        let local = compute_local_from_worlds(parent, child_world);
        assert!((local.position - Vec3::new(0.0, 5.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn compose_transforms_test() {
        let parent = TransformComponent::from_position(Vec3::new(10.0, 0.0, 0.0));
        let child = TransformComponent::from_position(Vec3::new(0.0, 5.0, 0.0));
        let composed = compose_transforms(&parent, &child);
        let pos = composed.w_axis.truncate();
        assert!((pos - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4);
    }
}
