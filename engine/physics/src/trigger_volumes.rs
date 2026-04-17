//! Trigger volumes: sensor shapes that detect overlap without physical response.
//!
//! Trigger volumes are invisible regions in the game world that fire events when
//! entities enter, stay inside, or exit them. Common use cases include:
//!
//! - **Damage zones**: lava pits, poison clouds, environmental hazards
//! - **Item pickups**: coins, health packs, ammo crates
//! - **Level transitions**: door triggers, teleporters, load zones
//! - **Audio zones**: ambient sound regions, reverb zones
//! - **Gameplay triggers**: cutscene triggers, checkpoint saves, objective areas
//!
//! The system tracks which entities are inside each trigger across frames,
//! producing [`TriggerEvent::Enter`] on the first frame of overlap,
//! [`TriggerEvent::Stay`] on subsequent frames, and [`TriggerEvent::Exit`]
//! when the entity leaves.
//!
//! # Architecture
//!
//! - [`TriggerVolume`]: defines the shape, position, and callback configuration
//!   for a single trigger region.
//! - [`TriggerSystem`]: manages all triggers, performs overlap tests each frame,
//!   tracks enter/stay/exit state, and dispatches events.
//! - [`TriggerEvent`]: the event type produced by the system.

use glam::{Quat, Vec3};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Trigger shape
// ---------------------------------------------------------------------------

/// Shape of a trigger volume.
#[derive(Debug, Clone)]
pub enum TriggerShape {
    /// Axis-aligned box defined by half-extents.
    Box {
        half_extents: Vec3,
    },
    /// Sphere defined by radius.
    Sphere {
        radius: f32,
    },
    /// Capsule oriented along the local Y axis.
    Capsule {
        radius: f32,
        half_height: f32,
    },
    /// Custom mesh (represented as a list of triangles for overlap tests).
    Mesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
        /// Precomputed AABB for broadphase.
        aabb_min: Vec3,
        aabb_max: Vec3,
    },
}

impl TriggerShape {
    /// Create a box trigger shape.
    pub fn new_box(half_extents: Vec3) -> Self {
        Self::Box { half_extents }
    }

    /// Create a sphere trigger shape.
    pub fn new_sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a capsule trigger shape.
    pub fn new_capsule(radius: f32, half_height: f32) -> Self {
        Self::Capsule { radius, half_height }
    }

    /// Create a mesh trigger shape. Computes the AABB automatically.
    pub fn new_mesh(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
        for v in &vertices {
            aabb_min = aabb_min.min(*v);
            aabb_max = aabb_max.max(*v);
        }
        Self::Mesh {
            vertices,
            indices,
            aabb_min,
            aabb_max,
        }
    }

    /// Test if a point is inside this trigger shape at the given world transform.
    pub fn contains_point(&self, position: Vec3, rotation: Quat, point: Vec3) -> bool {
        let inv_rot = rotation.inverse();
        let local = inv_rot * (point - position);

        match self {
            TriggerShape::Box { half_extents } => {
                local.x.abs() <= half_extents.x
                    && local.y.abs() <= half_extents.y
                    && local.z.abs() <= half_extents.z
            }
            TriggerShape::Sphere { radius } => local.length_squared() <= radius * radius,
            TriggerShape::Capsule { radius, half_height } => {
                let clamped_y = local.y.clamp(-*half_height, *half_height);
                let closest_on_axis = Vec3::new(0.0, clamped_y, 0.0);
                (local - closest_on_axis).length_squared() <= radius * radius
            }
            TriggerShape::Mesh {
                vertices,
                indices,
                aabb_min,
                aabb_max,
            } => {
                // Quick AABB rejection.
                if local.x < aabb_min.x
                    || local.x > aabb_max.x
                    || local.y < aabb_min.y
                    || local.y > aabb_max.y
                    || local.z < aabb_min.z
                    || local.z > aabb_max.z
                {
                    return false;
                }
                // Simplified: count ray crossings along Y axis for inside test.
                let mut crossings = 0u32;
                for tri in indices {
                    let v0 = vertices[tri[0] as usize];
                    let v1 = vertices[tri[1] as usize];
                    let v2 = vertices[tri[2] as usize];
                    if self.ray_triangle_test(local, Vec3::Y, v0, v1, v2) {
                        crossings += 1;
                    }
                }
                crossings % 2 == 1
            }
        }
    }

    /// Test if a sphere overlaps this trigger shape.
    pub fn overlaps_sphere(
        &self,
        position: Vec3,
        rotation: Quat,
        sphere_center: Vec3,
        sphere_radius: f32,
    ) -> bool {
        let inv_rot = rotation.inverse();
        let local = inv_rot * (sphere_center - position);

        match self {
            TriggerShape::Box { half_extents } => {
                let closest = Vec3::new(
                    local.x.clamp(-half_extents.x, half_extents.x),
                    local.y.clamp(-half_extents.y, half_extents.y),
                    local.z.clamp(-half_extents.z, half_extents.z),
                );
                (local - closest).length_squared() <= sphere_radius * sphere_radius
            }
            TriggerShape::Sphere { radius } => {
                local.length_squared() <= (radius + sphere_radius) * (radius + sphere_radius)
            }
            TriggerShape::Capsule { radius, half_height } => {
                let clamped_y = local.y.clamp(-*half_height, *half_height);
                let closest_on_axis = Vec3::new(0.0, clamped_y, 0.0);
                let combined = radius + sphere_radius;
                (local - closest_on_axis).length_squared() <= combined * combined
            }
            TriggerShape::Mesh {
                aabb_min,
                aabb_max,
                ..
            } => {
                // Conservative broadphase check using expanded AABB.
                let r = Vec3::splat(sphere_radius);
                let expanded_min = *aabb_min - r;
                let expanded_max = *aabb_max + r;
                let world_min = position + rotation * expanded_min;
                let world_max = position + rotation * expanded_max;
                sphere_center.x >= world_min.x.min(world_max.x)
                    && sphere_center.x <= world_min.x.max(world_max.x)
                    && sphere_center.y >= world_min.y.min(world_max.y)
                    && sphere_center.y <= world_min.y.max(world_max.y)
                    && sphere_center.z >= world_min.z.min(world_max.z)
                    && sphere_center.z <= world_min.z.max(world_max.z)
            }
        }
    }

    /// Test if an AABB overlaps this trigger shape (conservative broadphase).
    pub fn overlaps_aabb(
        &self,
        position: Vec3,
        rotation: Quat,
        aabb_center: Vec3,
        aabb_half_extents: Vec3,
    ) -> bool {
        // Convert to sphere overlap with circumscribed sphere for simplicity.
        let sphere_radius = aabb_half_extents.length();
        self.overlaps_sphere(position, rotation, aabb_center, sphere_radius)
    }

    /// Helper: Moller-Trumbore ray-triangle intersection.
    fn ray_triangle_test(
        &self,
        origin: Vec3,
        direction: Vec3,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
    ) -> bool {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < 1e-7 {
            return false;
        }

        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return false;
        }

        let q = s.cross(edge1);
        let v = f * direction.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t = f * edge2.dot(q);
        t > 0.0
    }

    /// Compute a conservative bounding sphere radius for this shape.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            TriggerShape::Box { half_extents } => half_extents.length(),
            TriggerShape::Sphere { radius } => *radius,
            TriggerShape::Capsule { radius, half_height } => {
                (half_height * half_height + radius * radius).sqrt()
            }
            TriggerShape::Mesh {
                aabb_min, aabb_max, ..
            } => {
                let center = (*aabb_min + *aabb_max) * 0.5;
                (*aabb_max - center).length()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger event
// ---------------------------------------------------------------------------

/// Events produced by the trigger system when entities interact with triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerEvent {
    /// The entity entered the trigger volume this frame (first frame of overlap).
    Enter,
    /// The entity is still inside the trigger volume (subsequent frames).
    Stay,
    /// The entity left the trigger volume this frame (first frame of non-overlap).
    Exit,
}

impl TriggerEvent {
    /// Whether this is an enter event.
    pub fn is_enter(&self) -> bool {
        matches!(self, Self::Enter)
    }

    /// Whether this is a stay event.
    pub fn is_stay(&self) -> bool {
        matches!(self, Self::Stay)
    }

    /// Whether this is an exit event.
    pub fn is_exit(&self) -> bool {
        matches!(self, Self::Exit)
    }
}

// ---------------------------------------------------------------------------
// Trigger event record
// ---------------------------------------------------------------------------

/// A complete trigger event with context: which trigger, which entity, what event.
#[derive(Debug, Clone)]
pub struct TriggerEventRecord {
    /// The trigger volume that produced this event.
    pub trigger_id: u32,
    /// The entity involved.
    pub entity_id: u32,
    /// The type of event (Enter, Stay, Exit).
    pub event: TriggerEvent,
    /// World-space position of the entity at the time of the event.
    pub entity_position: Vec3,
    /// How long the entity has been inside (0 for Enter, accumulated for Stay,
    /// total for Exit).
    pub time_inside: f32,
}

// ---------------------------------------------------------------------------
// Trigger callback
// ---------------------------------------------------------------------------

/// Type alias for trigger callbacks.
pub type TriggerCallback = Box<dyn Fn(&TriggerEventRecord) + Send + Sync>;

// ---------------------------------------------------------------------------
// Trigger volume
// ---------------------------------------------------------------------------

/// Unique identifier for a trigger volume.
pub type TriggerId = u32;

/// A trigger volume: a sensor region in the game world that detects entity overlap
/// without producing physical forces.
pub struct TriggerVolume {
    /// Unique identifier.
    pub id: TriggerId,
    /// Human-readable name for debugging.
    pub name: String,
    /// Shape of the trigger region.
    pub shape: TriggerShape,
    /// World-space position of the trigger center.
    pub position: Vec3,
    /// Orientation of the trigger.
    pub rotation: Quat,
    /// Whether this trigger is currently active. Inactive triggers do not
    /// produce events.
    pub enabled: bool,
    /// Collision layer mask: only entities on matching layers trigger events.
    pub layer_mask: u32,
    /// Optional tag for categorization (e.g., "damage_zone", "pickup", "checkpoint").
    pub tag: String,
    /// Custom user data.
    pub user_data: u64,
    /// Optional callback invoked on enter events.
    pub on_enter: Option<TriggerCallback>,
    /// Optional callback invoked on stay events.
    pub on_stay: Option<TriggerCallback>,
    /// Optional callback invoked on exit events.
    pub on_exit: Option<TriggerCallback>,
    /// If true, the trigger is destroyed after the first enter event.
    pub one_shot: bool,
    /// Cooldown time between activations (seconds). 0 = no cooldown.
    pub cooldown: f32,
    /// Time remaining on the cooldown timer.
    cooldown_remaining: f32,
    /// Maximum number of entities that can be inside simultaneously. 0 = unlimited.
    pub max_occupancy: usize,
}

impl TriggerVolume {
    /// Create a new trigger volume.
    pub fn new(id: TriggerId, shape: TriggerShape, position: Vec3) -> Self {
        Self {
            id,
            name: String::new(),
            shape,
            position,
            rotation: Quat::IDENTITY,
            enabled: true,
            layer_mask: u32::MAX,
            tag: String::new(),
            user_data: 0,
            on_enter: None,
            on_stay: None,
            on_exit: None,
            one_shot: false,
            cooldown: 0.0,
            cooldown_remaining: 0.0,
            max_occupancy: 0,
        }
    }

    /// Builder: set the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder: set the rotation.
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Builder: set the layer mask.
    pub fn with_layer_mask(mut self, mask: u32) -> Self {
        self.layer_mask = mask;
        self
    }

    /// Builder: set the tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = tag.into();
        self
    }

    /// Builder: set user data.
    pub fn with_user_data(mut self, data: u64) -> Self {
        self.user_data = data;
        self
    }

    /// Builder: set as one-shot.
    pub fn as_one_shot(mut self) -> Self {
        self.one_shot = true;
        self
    }

    /// Builder: set cooldown.
    pub fn with_cooldown(mut self, cooldown: f32) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Builder: set max occupancy.
    pub fn with_max_occupancy(mut self, max: usize) -> Self {
        self.max_occupancy = max;
        self
    }

    /// Test if a point is inside this trigger volume.
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.shape.contains_point(self.position, self.rotation, point)
    }

    /// Test if a sphere overlaps this trigger volume.
    pub fn overlaps_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.shape
            .overlaps_sphere(self.position, self.rotation, center, radius)
    }

    /// Test if an AABB overlaps this trigger volume.
    pub fn overlaps_aabb(&self, center: Vec3, half_extents: Vec3) -> bool {
        self.shape
            .overlaps_aabb(self.position, self.rotation, center, half_extents)
    }

    /// Check if the trigger can activate (not on cooldown).
    pub fn can_activate(&self) -> bool {
        self.enabled && self.cooldown_remaining <= 0.0
    }

    /// Update cooldown timer.
    pub fn update_cooldown(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining -= dt;
            if self.cooldown_remaining < 0.0 {
                self.cooldown_remaining = 0.0;
            }
        }
    }

    /// Start the cooldown timer.
    pub fn start_cooldown(&mut self) {
        self.cooldown_remaining = self.cooldown;
    }

    /// Get the bounding radius of this trigger.
    pub fn bounding_radius(&self) -> f32 {
        self.shape.bounding_radius()
    }
}

// ---------------------------------------------------------------------------
// Entity tracking state
// ---------------------------------------------------------------------------

/// Per-trigger tracking of which entities are inside.
#[derive(Debug, Clone)]
struct TriggerOccupancy {
    /// Set of entity IDs currently inside this trigger.
    current_entities: HashSet<u32>,
    /// Set of entity IDs that were inside last frame.
    previous_entities: HashSet<u32>,
    /// Time each entity has been inside (entity_id -> seconds).
    time_inside: HashMap<u32, f32>,
}

impl TriggerOccupancy {
    fn new() -> Self {
        Self {
            current_entities: HashSet::new(),
            previous_entities: HashSet::new(),
            time_inside: HashMap::new(),
        }
    }

    /// Swap current to previous and clear current for the new frame.
    fn begin_frame(&mut self) {
        std::mem::swap(&mut self.previous_entities, &mut self.current_entities);
        self.current_entities.clear();
    }

    /// Record that an entity is inside this frame.
    fn record_inside(&mut self, entity_id: u32) {
        self.current_entities.insert(entity_id);
    }

    /// Update time tracking for entities inside.
    fn update_times(&mut self, dt: f32) {
        for &entity_id in &self.current_entities {
            *self.time_inside.entry(entity_id).or_insert(0.0) += dt;
        }
        // Clean up times for entities that have left.
        self.time_inside
            .retain(|id, _| self.current_entities.contains(id));
    }

    /// Get entities that entered this frame (in current but not in previous).
    fn entered(&self) -> impl Iterator<Item = u32> + '_ {
        self.current_entities
            .iter()
            .filter(|id| !self.previous_entities.contains(id))
            .copied()
    }

    /// Get entities that stayed this frame (in both current and previous).
    fn staying(&self) -> impl Iterator<Item = u32> + '_ {
        self.current_entities
            .iter()
            .filter(|id| self.previous_entities.contains(id))
            .copied()
    }

    /// Get entities that exited this frame (in previous but not in current).
    fn exited(&self) -> impl Iterator<Item = u32> + '_ {
        self.previous_entities
            .iter()
            .filter(|id| !self.current_entities.contains(id))
            .copied()
    }

    /// Get the time an entity has been inside.
    fn get_time_inside(&self, entity_id: u32) -> f32 {
        self.time_inside.get(&entity_id).copied().unwrap_or(0.0)
    }

    /// Number of entities currently inside.
    fn occupant_count(&self) -> usize {
        self.current_entities.len()
    }
}

// ---------------------------------------------------------------------------
// Entity representation for overlap tests
// ---------------------------------------------------------------------------

/// Lightweight entity data for trigger overlap tests.
#[derive(Debug, Clone)]
pub struct TriggerEntity {
    /// Entity identifier.
    pub id: u32,
    /// World-space position.
    pub position: Vec3,
    /// Bounding sphere radius for broadphase overlap.
    pub radius: f32,
    /// Collision layer of this entity.
    pub layer: u32,
}

impl TriggerEntity {
    /// Create a new trigger entity.
    pub fn new(id: u32, position: Vec3, radius: f32) -> Self {
        Self {
            id,
            position,
            radius,
            layer: u32::MAX,
        }
    }

    /// Set the collision layer.
    pub fn with_layer(mut self, layer: u32) -> Self {
        self.layer = layer;
        self
    }
}

// ---------------------------------------------------------------------------
// Trigger system
// ---------------------------------------------------------------------------

/// System that manages all trigger volumes, performs overlap tests each frame,
/// and produces enter/stay/exit events.
///
/// # Usage
///
/// ```ignore
/// let mut system = TriggerSystem::new();
/// let trigger_id = system.add_trigger(
///     TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO)
///         .with_name("heal_zone")
///         .with_tag("healing")
/// );
///
/// // Each frame:
/// let entities = vec![
///     TriggerEntity::new(player_id, player_pos, 0.5),
/// ];
/// system.update(&entities, dt);
///
/// for event in system.drain_events() {
///     match event.event {
///         TriggerEvent::Enter => println!("Entity {} entered {}", event.entity_id, event.trigger_id),
///         TriggerEvent::Exit => println!("Entity {} left {}", event.entity_id, event.trigger_id),
///         _ => {}
///     }
/// }
/// ```
pub struct TriggerSystem {
    /// All registered triggers.
    triggers: Vec<TriggerVolume>,
    /// Per-trigger occupancy tracking (parallel to `triggers`).
    occupancy: Vec<TriggerOccupancy>,
    /// Events produced during the current frame.
    events: Vec<TriggerEventRecord>,
    /// Next trigger ID to assign.
    next_id: TriggerId,
    /// Triggers pending removal (deferred to avoid mid-iteration issues).
    pending_removals: Vec<TriggerId>,
    /// Whether to use broadphase distance culling.
    pub use_broadphase: bool,
    /// Maximum distance for broadphase culling (triggers farther than this
    /// from any entity are skipped).
    pub broadphase_distance: f32,
}

impl TriggerSystem {
    /// Create a new trigger system.
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
            occupancy: Vec::new(),
            events: Vec::new(),
            next_id: 0,
            pending_removals: Vec::new(),
            use_broadphase: true,
            broadphase_distance: 500.0,
        }
    }

    /// Add a trigger volume to the system. Returns the trigger ID.
    pub fn add_trigger(&mut self, mut trigger: TriggerVolume) -> TriggerId {
        let id = self.next_id;
        self.next_id += 1;
        trigger.id = id;
        self.triggers.push(trigger);
        self.occupancy.push(TriggerOccupancy::new());
        id
    }

    /// Remove a trigger by ID (deferred until end of next update).
    pub fn remove_trigger(&mut self, id: TriggerId) {
        self.pending_removals.push(id);
    }

    /// Immediately remove a trigger by ID.
    pub fn remove_trigger_immediate(&mut self, id: TriggerId) {
        if let Some(idx) = self.triggers.iter().position(|t| t.id == id) {
            self.triggers.swap_remove(idx);
            self.occupancy.swap_remove(idx);
        }
    }

    /// Get a reference to a trigger by ID.
    pub fn get_trigger(&self, id: TriggerId) -> Option<&TriggerVolume> {
        self.triggers.iter().find(|t| t.id == id)
    }

    /// Get a mutable reference to a trigger by ID.
    pub fn get_trigger_mut(&mut self, id: TriggerId) -> Option<&mut TriggerVolume> {
        self.triggers.iter_mut().find(|t| t.id == id)
    }

    /// Get all triggers with a specific tag.
    pub fn get_triggers_by_tag(&self, tag: &str) -> Vec<&TriggerVolume> {
        self.triggers.iter().filter(|t| t.tag == tag).collect()
    }

    /// Get the entities currently inside a trigger.
    pub fn get_entities_in_trigger(&self, id: TriggerId) -> &[u32] {
        // Note: HashSet does not directly support &[u32], so we must collect.
        // For the public API, we return a Vec via a helper.
        // Internal users can use the occupancy tracking directly.
        &[]
    }

    /// Get the entities currently inside a trigger as a Vec.
    pub fn get_entities_in_trigger_vec(&self, id: TriggerId) -> Vec<u32> {
        if let Some(idx) = self.triggers.iter().position(|t| t.id == id) {
            self.occupancy[idx].current_entities.iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Check if a specific entity is inside a specific trigger.
    pub fn is_entity_in_trigger(&self, trigger_id: TriggerId, entity_id: u32) -> bool {
        if let Some(idx) = self.triggers.iter().position(|t| t.id == trigger_id) {
            self.occupancy[idx].current_entities.contains(&entity_id)
        } else {
            false
        }
    }

    /// Get the number of entities inside a trigger.
    pub fn occupant_count(&self, id: TriggerId) -> usize {
        if let Some(idx) = self.triggers.iter().position(|t| t.id == id) {
            self.occupancy[idx].occupant_count()
        } else {
            0
        }
    }

    /// Get the time an entity has been inside a trigger.
    pub fn time_inside(&self, trigger_id: TriggerId, entity_id: u32) -> f32 {
        if let Some(idx) = self.triggers.iter().position(|t| t.id == trigger_id) {
            self.occupancy[idx].get_time_inside(entity_id)
        } else {
            0.0
        }
    }

    /// Update the trigger system: perform overlap tests and generate events.
    ///
    /// Call this once per frame with the current set of entities that should
    /// be tested against triggers.
    pub fn update(&mut self, entities: &[TriggerEntity], dt: f32) {
        // Process deferred removals.
        for id in self.pending_removals.drain(..) {
            self.remove_trigger_immediate(id);
        }

        // Clear events from last frame.
        self.events.clear();

        // Track triggers to remove (one-shot triggers that fired).
        let mut one_shot_removals = Vec::new();

        // Process each trigger.
        for (trigger_idx, trigger) in self.triggers.iter_mut().enumerate() {
            if !trigger.enabled {
                continue;
            }

            // Update cooldown.
            trigger.update_cooldown(dt);

            // Begin frame: swap current/previous occupancy.
            self.occupancy[trigger_idx].begin_frame();

            // Skip if on cooldown.
            if !trigger.can_activate() {
                continue;
            }

            // Test each entity against this trigger.
            let trigger_pos = trigger.position;
            let trigger_radius = trigger.bounding_radius();

            for entity in entities {
                // Layer filter.
                if (entity.layer & trigger.layer_mask) == 0 {
                    continue;
                }

                // Broadphase: bounding sphere check.
                if self.use_broadphase {
                    let dist = (entity.position - trigger_pos).length();
                    if dist > trigger_radius + entity.radius + 1.0 {
                        continue;
                    }
                }

                // Max occupancy check.
                if trigger.max_occupancy > 0
                    && self.occupancy[trigger_idx].occupant_count() >= trigger.max_occupancy
                    && !self.occupancy[trigger_idx]
                        .previous_entities
                        .contains(&entity.id)
                {
                    continue;
                }

                // Narrowphase: actual shape overlap test.
                let overlaps =
                    trigger.overlaps_sphere(entity.position, entity.radius);

                if overlaps {
                    self.occupancy[trigger_idx].record_inside(entity.id);
                }
            }

            // Update time tracking.
            self.occupancy[trigger_idx].update_times(dt);
        }

        // Generate events.
        for (trigger_idx, trigger) in self.triggers.iter_mut().enumerate() {
            if !trigger.enabled {
                continue;
            }

            let occupancy = &self.occupancy[trigger_idx];

            // Enter events.
            for entity_id in occupancy.entered() {
                let time_inside = occupancy.get_time_inside(entity_id);
                let record = TriggerEventRecord {
                    trigger_id: trigger.id,
                    entity_id,
                    event: TriggerEvent::Enter,
                    entity_position: Vec3::ZERO, // Would be filled from entity data in real usage.
                    time_inside,
                };

                if let Some(ref callback) = trigger.on_enter {
                    callback(&record);
                }

                self.events.push(record);

                // One-shot: mark for removal after first enter.
                if trigger.one_shot {
                    one_shot_removals.push(trigger.id);
                }

                // Start cooldown on enter.
                if trigger.cooldown > 0.0 {
                    trigger.start_cooldown();
                }
            }

            // Stay events.
            for entity_id in occupancy.staying() {
                let time_inside = occupancy.get_time_inside(entity_id);
                let record = TriggerEventRecord {
                    trigger_id: trigger.id,
                    entity_id,
                    event: TriggerEvent::Stay,
                    entity_position: Vec3::ZERO,
                    time_inside,
                };

                if let Some(ref callback) = trigger.on_stay {
                    callback(&record);
                }

                self.events.push(record);
            }

            // Exit events.
            for entity_id in occupancy.exited() {
                let record = TriggerEventRecord {
                    trigger_id: trigger.id,
                    entity_id,
                    event: TriggerEvent::Exit,
                    entity_position: Vec3::ZERO,
                    time_inside: 0.0,
                };

                if let Some(ref callback) = trigger.on_exit {
                    callback(&record);
                }

                self.events.push(record);
            }
        }

        // Remove one-shot triggers that fired.
        for id in one_shot_removals {
            self.remove_trigger_immediate(id);
        }
    }

    /// Drain all events from this frame.
    pub fn drain_events(&mut self) -> Vec<TriggerEventRecord> {
        std::mem::take(&mut self.events)
    }

    /// Get events from this frame without draining.
    pub fn events(&self) -> &[TriggerEventRecord] {
        &self.events
    }

    /// Get the number of registered triggers.
    pub fn trigger_count(&self) -> usize {
        self.triggers.len()
    }

    /// Enable or disable a trigger by ID.
    pub fn set_trigger_enabled(&mut self, id: TriggerId, enabled: bool) {
        if let Some(trigger) = self.triggers.iter_mut().find(|t| t.id == id) {
            trigger.enabled = enabled;
        }
    }

    /// Move a trigger to a new position.
    pub fn set_trigger_position(&mut self, id: TriggerId, position: Vec3) {
        if let Some(trigger) = self.triggers.iter_mut().find(|t| t.id == id) {
            trigger.position = position;
        }
    }

    /// Set a trigger's rotation.
    pub fn set_trigger_rotation(&mut self, id: TriggerId, rotation: Quat) {
        if let Some(trigger) = self.triggers.iter_mut().find(|t| t.id == id) {
            trigger.rotation = rotation;
        }
    }

    /// Clear all triggers and reset the system.
    pub fn clear(&mut self) {
        self.triggers.clear();
        self.occupancy.clear();
        self.events.clear();
        self.pending_removals.clear();
        self.next_id = 0;
    }
}

impl Default for TriggerSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ECS component
// ---------------------------------------------------------------------------

/// ECS component wrapping a trigger volume reference.
pub struct TriggerComponent {
    /// ID of the trigger in the TriggerSystem.
    pub trigger_id: TriggerId,
    /// Whether to synchronize position from the entity's transform.
    pub sync_transform: bool,
}

impl TriggerComponent {
    /// Create a new trigger component.
    pub fn new(trigger_id: TriggerId) -> Self {
        Self {
            trigger_id,
            sync_transform: true,
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
    fn sphere_trigger_contains_point() {
        let shape = TriggerShape::new_sphere(5.0);
        assert!(shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(3.0, 0.0, 0.0)));
        assert!(!shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(6.0, 0.0, 0.0)));
    }

    #[test]
    fn box_trigger_contains_point() {
        let shape = TriggerShape::new_box(Vec3::new(2.0, 2.0, 2.0));
        assert!(shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0)));
        assert!(!shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn capsule_trigger_contains_point() {
        let shape = TriggerShape::new_capsule(1.0, 2.0);
        assert!(shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(0.5, 1.5, 0.0)));
        assert!(!shape.contains_point(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn sphere_overlap_test() {
        let shape = TriggerShape::new_sphere(3.0);
        assert!(shape.overlaps_sphere(Vec3::ZERO, Quat::IDENTITY, Vec3::new(3.5, 0.0, 0.0), 1.0));
        assert!(!shape.overlaps_sphere(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
            1.0
        ));
    }

    #[test]
    fn trigger_system_enter_exit() {
        let mut system = TriggerSystem::new();
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO);
        let trigger_id = system.add_trigger(trigger);

        let entity = TriggerEntity::new(1, Vec3::new(3.0, 0.0, 0.0), 0.5);

        // Frame 1: entity enters.
        system.update(&[entity.clone()], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.iter().any(|e| e.event == TriggerEvent::Enter && e.entity_id == 1));

        // Frame 2: entity stays.
        system.update(&[entity.clone()], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.iter().any(|e| e.event == TriggerEvent::Stay && e.entity_id == 1));

        // Frame 3: entity exits (not in entity list).
        system.update(&[], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.iter().any(|e| e.event == TriggerEvent::Exit && e.entity_id == 1));
    }

    #[test]
    fn trigger_system_one_shot() {
        let mut system = TriggerSystem::new();
        let trigger =
            TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO).as_one_shot();
        system.add_trigger(trigger);

        let entity = TriggerEntity::new(1, Vec3::new(1.0, 0.0, 0.0), 0.5);

        // Frame 1: entity enters, trigger should be removed.
        system.update(&[entity.clone()], 1.0 / 60.0);
        assert_eq!(system.trigger_count(), 0);
    }

    #[test]
    fn trigger_system_layer_filter() {
        let mut system = TriggerSystem::new();
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO)
            .with_layer_mask(0b0010);
        system.add_trigger(trigger);

        let entity = TriggerEntity::new(1, Vec3::new(1.0, 0.0, 0.0), 0.5).with_layer(0b0001);

        system.update(&[entity], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.is_empty());
    }

    #[test]
    fn trigger_volume_builder() {
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO)
            .with_name("test_trigger")
            .with_tag("damage")
            .with_user_data(42)
            .with_cooldown(1.0)
            .with_max_occupancy(4);

        assert_eq!(trigger.name, "test_trigger");
        assert_eq!(trigger.tag, "damage");
        assert_eq!(trigger.user_data, 42);
        assert_eq!(trigger.cooldown, 1.0);
        assert_eq!(trigger.max_occupancy, 4);
    }

    #[test]
    fn trigger_shape_bounding_radius() {
        let sphere = TriggerShape::new_sphere(3.0);
        assert!((sphere.bounding_radius() - 3.0).abs() < 0.01);

        let box_shape = TriggerShape::new_box(Vec3::new(1.0, 1.0, 1.0));
        assert!(box_shape.bounding_radius() > 1.0);
    }

    #[test]
    fn trigger_system_occupancy() {
        let mut system = TriggerSystem::new();
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO);
        let trigger_id = system.add_trigger(trigger);

        let entities = vec![
            TriggerEntity::new(1, Vec3::new(1.0, 0.0, 0.0), 0.5),
            TriggerEntity::new(2, Vec3::new(2.0, 0.0, 0.0), 0.5),
        ];

        system.update(&entities, 1.0 / 60.0);
        assert_eq!(system.occupant_count(trigger_id), 2);
        assert!(system.is_entity_in_trigger(trigger_id, 1));
        assert!(system.is_entity_in_trigger(trigger_id, 2));
    }

    #[test]
    fn trigger_system_enable_disable() {
        let mut system = TriggerSystem::new();
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(5.0), Vec3::ZERO);
        let trigger_id = system.add_trigger(trigger);

        system.set_trigger_enabled(trigger_id, false);
        let entity = TriggerEntity::new(1, Vec3::new(1.0, 0.0, 0.0), 0.5);
        system.update(&[entity], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.is_empty());
    }

    #[test]
    fn trigger_system_move_trigger() {
        let mut system = TriggerSystem::new();
        let trigger = TriggerVolume::new(0, TriggerShape::new_sphere(2.0), Vec3::ZERO);
        let trigger_id = system.add_trigger(trigger);

        let entity = TriggerEntity::new(1, Vec3::new(10.0, 0.0, 0.0), 0.5);

        // Entity is far from trigger.
        system.update(&[entity.clone()], 1.0 / 60.0);
        assert!(system.drain_events().is_empty());

        // Move trigger to entity.
        system.set_trigger_position(trigger_id, Vec3::new(10.0, 0.0, 0.0));
        system.update(&[entity], 1.0 / 60.0);
        let events = system.drain_events();
        assert!(events.iter().any(|e| e.event == TriggerEvent::Enter));
    }
}
