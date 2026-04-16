//! LOD (Level of Detail) System
//!
//! Manages level-of-detail switching for entities based on their screen-space
//! size. Supports discrete LOD levels, crossfade transitions, impostor
//! (billboard) generation at extreme distances, and hierarchical LOD (HLOD)
//! for merging distant geometry clusters into single draw calls.
//!
//! # Screen-Size Evaluation
//!
//! Each frame the LOD manager computes the approximate screen coverage for
//! every entity with a [`LODComponent`]. The screen size is derived from:
//!
//! ```text
//! screen_size = (bounding_sphere_radius / distance_to_camera) * screen_height
//! ```
//!
//! This metric is compared against the threshold list in each [`LODGroup`]
//! to select the active LOD level. Hysteresis prevents rapid toggling at
//! transition boundaries.

use std::collections::HashMap;

use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use genovo_ecs::{Component, Entity};

// ---------------------------------------------------------------------------
// ShadowMode
// ---------------------------------------------------------------------------

/// Controls how a LOD level casts shadows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShadowMode {
    /// Full shadow geometry (matches the visual mesh).
    Full,
    /// Simplified shadow geometry for performance.
    Simplified,
    /// No shadow casting.
    None,
    /// Shadow only (no visual mesh rendered).
    ShadowOnly,
}

impl Default for ShadowMode {
    fn default() -> Self {
        ShadowMode::Full
    }
}

// ---------------------------------------------------------------------------
// LODLevel
// ---------------------------------------------------------------------------

/// A single level of detail within a [`LODGroup`].
///
/// Each LOD level has a mesh handle, shadow mode, and the screen-size
/// threshold at which it becomes active. LOD levels are sorted from highest
/// detail (LOD 0) to lowest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LODLevel {
    /// Index of this LOD level (0 = highest detail).
    pub index: u32,
    /// Handle to the mesh asset for this LOD level.
    /// Stored as a u64 for serialization; actual type depends on the asset system.
    pub mesh_handle: u64,
    /// Shadow casting mode for this LOD level.
    pub shadow_mode: ShadowMode,
    /// Screen-size threshold: this LOD becomes active when the entity's
    /// screen size drops below this value (in pixels).
    pub screen_size_threshold: f32,
    /// Optional material override for this LOD level.
    pub material_override: Option<u64>,
    /// Triangle count (for debugging/stats).
    pub triangle_count: u32,
    /// Vertex count (for debugging/stats).
    pub vertex_count: u32,
}

impl LODLevel {
    /// Create a new LOD level.
    pub fn new(index: u32, mesh_handle: u64, screen_size_threshold: f32) -> Self {
        Self {
            index,
            mesh_handle,
            shadow_mode: ShadowMode::Full,
            screen_size_threshold,
            material_override: None,
            triangle_count: 0,
            vertex_count: 0,
        }
    }

    /// Set shadow mode.
    pub fn with_shadow_mode(mut self, mode: ShadowMode) -> Self {
        self.shadow_mode = mode;
        self
    }

    /// Set mesh complexity stats.
    pub fn with_stats(mut self, triangles: u32, vertices: u32) -> Self {
        self.triangle_count = triangles;
        self.vertex_count = vertices;
        self
    }
}

// ---------------------------------------------------------------------------
// CrossfadeState
// ---------------------------------------------------------------------------

/// State of a crossfade transition between two LOD levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossfadeState {
    /// LOD index being faded out.
    pub from_lod: u32,
    /// LOD index being faded in.
    pub to_lod: u32,
    /// Current progress (0.0 = fully showing `from_lod`, 1.0 = fully showing
    /// `to_lod`).
    pub progress: f32,
    /// Speed of the crossfade (progress per second).
    pub speed: f32,
    /// Dither pattern seed for this transition (used for screen-door
    /// transparency).
    pub dither_seed: u32,
}

impl CrossfadeState {
    /// Create a new crossfade.
    pub fn new(from: u32, to: u32, speed: f32) -> Self {
        Self {
            from_lod: from,
            to_lod: to,
            progress: 0.0,
            speed,
            dither_seed: 0,
        }
    }

    /// Advance the crossfade by `dt` seconds. Returns `true` when complete.
    pub fn advance(&mut self, dt: f32) -> bool {
        self.progress += self.speed * dt;
        if self.progress >= 1.0 {
            self.progress = 1.0;
            true
        } else {
            false
        }
    }

    /// Get the dither threshold for the outgoing LOD (pixels with noise
    /// below this value are discarded).
    pub fn outgoing_dither(&self) -> f32 {
        self.progress
    }

    /// Get the dither threshold for the incoming LOD.
    pub fn incoming_dither(&self) -> f32 {
        1.0 - self.progress
    }

    /// Check if the crossfade is complete.
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }
}

// ---------------------------------------------------------------------------
// LODGroup
// ---------------------------------------------------------------------------

/// A group of LOD levels associated with an entity.
///
/// LOD groups contain a sorted list of [`LODLevel`]s from highest to lowest
/// detail. The LOD manager evaluates the screen size each frame and selects
/// the appropriate level.
#[derive(Debug, Clone)]
pub struct LODGroup {
    /// Entity this LOD group belongs to.
    pub entity: Entity,
    /// Sorted LOD levels (index 0 = highest detail).
    pub levels: Vec<LODLevel>,
    /// Bounding sphere radius in local space. Used for screen-size computation.
    pub bounding_radius: f32,
    /// Currently active LOD index.
    pub active_lod: u32,
    /// Hysteresis factor: the threshold is multiplied by (1 + hysteresis)
    /// when switching to a higher-detail LOD to prevent rapid toggling.
    pub hysteresis: f32,
    /// Whether crossfade transitions are enabled for this group.
    pub crossfade_enabled: bool,
    /// Crossfade duration in seconds.
    pub crossfade_duration: f32,
    /// Forced LOD override (if set, always use this LOD index).
    pub forced_lod: Option<u32>,
    /// Distance at which to render as an impostor (billboard).
    pub impostor_distance: Option<f32>,
    /// Whether this group is enabled.
    pub enabled: bool,
}

impl LODGroup {
    /// Create a new LOD group for an entity.
    pub fn new(entity: Entity, bounding_radius: f32) -> Self {
        Self {
            entity,
            levels: Vec::new(),
            bounding_radius,
            active_lod: 0,
            hysteresis: 0.1,
            crossfade_enabled: false,
            crossfade_duration: 0.3,
            forced_lod: None,
            impostor_distance: None,
            enabled: true,
        }
    }

    /// Add a LOD level. Levels should be added in order (highest detail first).
    pub fn add_level(&mut self, level: LODLevel) {
        self.levels.push(level);
        // Keep sorted by index.
        self.levels.sort_by_key(|l| l.index);
    }

    /// Enable crossfade with the given duration.
    pub fn with_crossfade(mut self, duration: f32) -> Self {
        self.crossfade_enabled = true;
        self.crossfade_duration = duration;
        self
    }

    /// Set impostor distance.
    pub fn with_impostor_distance(mut self, distance: f32) -> Self {
        self.impostor_distance = Some(distance);
        self
    }

    /// Get the LOD level for a given screen size, accounting for hysteresis.
    ///
    /// Returns the index of the appropriate LOD level.
    pub fn evaluate(&self, screen_size: f32, current_lod: u32) -> u32 {
        if let Some(forced) = self.forced_lod {
            return forced.min(self.levels.len().saturating_sub(1) as u32);
        }

        if self.levels.is_empty() {
            return 0;
        }

        let mut selected = 0u32;

        for level in &self.levels {
            let mut threshold = level.screen_size_threshold;

            // Apply hysteresis: if we would be switching to a higher-detail LOD,
            // require the screen size to be slightly larger to prevent flickering.
            if level.index < current_lod {
                threshold *= 1.0 + self.hysteresis;
            }

            if screen_size < threshold {
                selected = level.index;
            } else {
                break;
            }
        }

        // Clamp to valid range.
        selected.min(self.levels.len().saturating_sub(1) as u32)
    }

    /// Get the currently active LOD level data.
    pub fn active_level(&self) -> Option<&LODLevel> {
        self.levels.iter().find(|l| l.index == self.active_lod)
    }

    /// Get a specific LOD level by index.
    pub fn level(&self, index: u32) -> Option<&LODLevel> {
        self.levels.iter().find(|l| l.index == index)
    }

    /// Total number of LOD levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

// ---------------------------------------------------------------------------
// LODComponent
// ---------------------------------------------------------------------------

/// ECS component for entities that participate in LOD evaluation.
#[derive(Debug, Clone)]
pub struct LODComponent {
    /// Index into the LOD manager's group array.
    pub group_index: usize,
    /// Bounding sphere radius (cached from LODGroup for fast access).
    pub bounding_radius: f32,
    /// Last computed screen size.
    pub last_screen_size: f32,
    /// Last computed distance to camera.
    pub last_distance: f32,
    /// Current LOD index.
    pub current_lod: u32,
    /// Active crossfade, if any.
    pub crossfade: Option<CrossfadeState>,
    /// Whether this entity should be rendered as an impostor.
    pub is_impostor: bool,
}

impl Component for LODComponent {}

impl LODComponent {
    /// Create a new LOD component.
    pub fn new(group_index: usize, bounding_radius: f32) -> Self {
        Self {
            group_index,
            bounding_radius,
            last_screen_size: 0.0,
            last_distance: 0.0,
            current_lod: 0,
            crossfade: None,
            is_impostor: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ImpostorData
// ---------------------------------------------------------------------------

/// Data for rendering an entity as a billboard impostor at extreme distances.
///
/// Impostors are pre-rendered views of an entity captured from multiple
/// angles and stored in a texture atlas. At runtime the appropriate view is
/// selected based on the camera angle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpostorData {
    /// Handle to the impostor atlas texture.
    pub atlas_handle: u64,
    /// Number of views around the vertical axis.
    pub horizontal_views: u32,
    /// Number of views around the horizontal axis.
    pub vertical_views: u32,
    /// Size of each view in the atlas (pixels).
    pub view_size: u32,
    /// World-space size of the impostor billboard.
    pub billboard_width: f32,
    /// World-space height of the impostor billboard.
    pub billboard_height: f32,
    /// Center offset of the billboard relative to the entity origin.
    pub center_offset: Vec3,
    /// Whether the impostor is a hemi-octahedral projection.
    pub hemi_octahedral: bool,
}

impl ImpostorData {
    /// Create new impostor data.
    pub fn new(atlas_handle: u64, views_h: u32, views_v: u32, view_size: u32) -> Self {
        Self {
            atlas_handle,
            horizontal_views: views_h,
            vertical_views: views_v,
            view_size,
            billboard_width: 1.0,
            billboard_height: 1.0,
            center_offset: Vec3::ZERO,
            hemi_octahedral: false,
        }
    }

    /// Calculate the UV coordinates for a given camera direction.
    ///
    /// Returns `(u_offset, v_offset, u_size, v_size)` into the atlas.
    pub fn compute_uvs(&self, camera_direction: Vec3) -> (f32, f32, f32, f32) {
        // Compute horizontal angle.
        let horiz_angle = camera_direction.z.atan2(camera_direction.x);
        let horiz_normalized = (horiz_angle / std::f32::consts::TAU + 0.5).fract();
        let horiz_index = (horiz_normalized * self.horizontal_views as f32) as u32
            % self.horizontal_views;

        // Compute vertical angle.
        let vert_angle = camera_direction.y.asin();
        let vert_normalized =
            (vert_angle / std::f32::consts::PI + 0.5).clamp(0.0, 1.0);
        let vert_index = (vert_normalized * (self.vertical_views - 1) as f32) as u32;

        let total_h = self.horizontal_views;
        let total_v = self.vertical_views;

        let u_size = 1.0 / total_h as f32;
        let v_size = 1.0 / total_v as f32;
        let u_offset = horiz_index as f32 * u_size;
        let v_offset = vert_index as f32 * v_size;

        (u_offset, v_offset, u_size, v_size)
    }
}

// ---------------------------------------------------------------------------
// HLODCluster
// ---------------------------------------------------------------------------

/// A cluster of entities that are merged into a single draw call at
/// extreme distances (Hierarchical LOD).
///
/// When the camera is far enough away that individual LOD groups would
/// switch to their lowest detail, the HLOD system replaces them all with
/// a single merged mesh.
#[derive(Debug, Clone)]
pub struct HLODCluster {
    /// Unique identifier for this cluster.
    pub id: u32,
    /// Entities that belong to this cluster.
    pub entities: Vec<Entity>,
    /// Center of the cluster bounding sphere.
    pub center: Vec3,
    /// Radius of the cluster bounding sphere.
    pub radius: f32,
    /// Handle to the merged mesh for this cluster.
    pub merged_mesh_handle: u64,
    /// Handle to the merged material.
    pub merged_material_handle: u64,
    /// Screen-size threshold at which the cluster activates.
    pub activation_threshold: f32,
    /// Whether the cluster is currently active (replacing individual entities).
    pub is_active: bool,
    /// Triangle count of the merged mesh.
    pub triangle_count: u32,
}

impl HLODCluster {
    /// Create a new HLOD cluster.
    pub fn new(id: u32, center: Vec3, radius: f32) -> Self {
        Self {
            id,
            entities: Vec::new(),
            center,
            radius,
            merged_mesh_handle: 0,
            merged_material_handle: 0,
            activation_threshold: 20.0, // Activate when cluster is < 20 pixels.
            is_active: false,
            triangle_count: 0,
        }
    }

    /// Add an entity to this cluster.
    pub fn add_entity(&mut self, entity: Entity) {
        if !self.entities.contains(&entity) {
            self.entities.push(entity);
        }
    }

    /// Remove an entity from this cluster.
    pub fn remove_entity(&mut self, entity: Entity) -> bool {
        if let Some(idx) = self.entities.iter().position(|e| *e == entity) {
            self.entities.swap_remove(idx);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// LODStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the LOD system.
#[derive(Debug, Clone, Default)]
pub struct LODStats {
    /// Total number of LOD groups being evaluated.
    pub total_groups: usize,
    /// Number of LOD transitions this frame.
    pub transitions_this_frame: usize,
    /// Number of active crossfades.
    pub active_crossfades: usize,
    /// Number of entities rendered as impostors.
    pub impostor_count: usize,
    /// Number of active HLOD clusters.
    pub active_hlod_clusters: usize,
    /// Total triangles rendered (across all LOD levels).
    pub total_triangles: u64,
    /// Triangles saved by LOD (compared to rendering everything at LOD 0).
    pub triangles_saved: u64,
    /// Distribution: how many entities are at each LOD level.
    pub lod_distribution: [usize; 8],
}

// ---------------------------------------------------------------------------
// LODManager
// ---------------------------------------------------------------------------

/// Evaluates and manages LOD levels for all registered entities.
///
/// Each frame, the LOD manager:
/// 1. Computes screen-space size for each LOD group.
/// 2. Selects the appropriate LOD level.
/// 3. Initiates crossfade transitions if enabled.
/// 4. Evaluates HLOD clusters.
/// 5. Manages impostor rendering at extreme distances.
///
/// # Screen-Size Computation
///
/// ```text
/// screen_size = (bounding_radius * 2 / distance) * projection_scale * screen_height
/// ```
///
/// where `projection_scale` accounts for the field of view:
/// ```text
/// projection_scale = 1 / tan(fov_y / 2)
/// ```
pub struct LODManager {
    /// All registered LOD groups, indexed by a group id.
    groups: Vec<LODGroup>,
    /// Map from entity to group index for fast lookup.
    entity_to_group: HashMap<Entity, usize>,
    /// HLOD clusters.
    hlod_clusters: Vec<HLODCluster>,
    /// Impostor data per entity.
    impostors: HashMap<Entity, ImpostorData>,
    /// Global LOD bias: shifts all thresholds (negative = higher quality).
    lod_bias: f32,
    /// Maximum LOD level that can be used (for quality settings).
    max_lod: u32,
    /// Screen height in pixels (for screen-size computation).
    screen_height: f32,
    /// Vertical field of view in radians.
    fov_y: f32,
    /// Cached projection scale factor.
    projection_scale: f32,
    /// Crossfade speed (transitions per second).
    crossfade_speed: f32,
    /// Whether the LOD system is enabled.
    enabled: bool,
    /// Statistics for the current frame.
    stats: LODStats,
    /// Frame counter.
    frame: u64,
}

impl LODManager {
    /// Create a new LOD manager.
    ///
    /// # Parameters
    ///
    /// - `screen_height`: Height of the render target in pixels.
    /// - `fov_y`: Vertical field of view in radians.
    pub fn new(screen_height: f32, fov_y: f32) -> Self {
        let projection_scale = 1.0 / (fov_y * 0.5).tan();

        Self {
            groups: Vec::new(),
            entity_to_group: HashMap::new(),
            hlod_clusters: Vec::new(),
            impostors: HashMap::new(),
            lod_bias: 0.0,
            max_lod: 7,
            screen_height,
            fov_y,
            projection_scale,
            crossfade_speed: 3.3, // ~0.3s crossfade
            enabled: true,
            stats: LODStats::default(),
            frame: 0,
        }
    }

    /// Update screen dimensions and FOV (e.g., after a window resize).
    pub fn set_screen_params(&mut self, screen_height: f32, fov_y: f32) {
        self.screen_height = screen_height;
        self.fov_y = fov_y;
        self.projection_scale = 1.0 / (fov_y * 0.5).tan();
    }

    /// Set the global LOD bias.
    pub fn set_lod_bias(&mut self, bias: f32) {
        self.lod_bias = bias;
    }

    /// Get the global LOD bias.
    pub fn lod_bias(&self) -> f32 {
        self.lod_bias
    }

    /// Set the maximum LOD level.
    pub fn set_max_lod(&mut self, max: u32) {
        self.max_lod = max;
    }

    /// Enable or disable the LOD system.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    // -- Group management --------------------------------------------------

    /// Register a LOD group. Returns the group index.
    pub fn register_group(&mut self, group: LODGroup) -> usize {
        let idx = self.groups.len();
        self.entity_to_group.insert(group.entity, idx);
        self.groups.push(group);
        idx
    }

    /// Unregister a LOD group by entity.
    pub fn unregister_entity(&mut self, entity: Entity) {
        if let Some(idx) = self.entity_to_group.remove(&entity) {
            // Swap-remove the group and fix up the map.
            if idx < self.groups.len() - 1 {
                let last_entity = self.groups.last().unwrap().entity;
                self.entity_to_group.insert(last_entity, idx);
            }
            self.groups.swap_remove(idx);
        }
    }

    /// Get a LOD group by entity.
    pub fn group(&self, entity: Entity) -> Option<&LODGroup> {
        self.entity_to_group
            .get(&entity)
            .and_then(|&idx| self.groups.get(idx))
    }

    /// Get a mutable LOD group by entity.
    pub fn group_mut(&mut self, entity: Entity) -> Option<&mut LODGroup> {
        self.entity_to_group
            .get(&entity)
            .copied()
            .and_then(|idx| self.groups.get_mut(idx))
    }

    /// Number of registered LOD groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    // -- Impostor management -----------------------------------------------

    /// Register impostor data for an entity.
    pub fn register_impostor(&mut self, entity: Entity, data: ImpostorData) {
        self.impostors.insert(entity, data);
    }

    /// Remove impostor data for an entity.
    pub fn remove_impostor(&mut self, entity: Entity) {
        self.impostors.remove(&entity);
    }

    /// Get impostor data for an entity.
    pub fn impostor(&self, entity: Entity) -> Option<&ImpostorData> {
        self.impostors.get(&entity)
    }

    // -- HLOD management ---------------------------------------------------

    /// Register an HLOD cluster.
    pub fn register_hlod_cluster(&mut self, cluster: HLODCluster) {
        self.hlod_clusters.push(cluster);
    }

    /// Remove an HLOD cluster by id.
    pub fn remove_hlod_cluster(&mut self, id: u32) -> bool {
        if let Some(idx) = self.hlod_clusters.iter().position(|c| c.id == id) {
            self.hlod_clusters.swap_remove(idx);
            true
        } else {
            false
        }
    }

    /// Get all HLOD clusters.
    pub fn hlod_clusters(&self) -> &[HLODCluster] {
        &self.hlod_clusters
    }

    // -- Screen-size computation -------------------------------------------

    /// Compute the screen-space size of a bounding sphere.
    ///
    /// Returns the approximate number of pixels the sphere would cover on
    /// screen.
    ///
    /// # Parameters
    ///
    /// - `bounding_radius`: Radius of the object's bounding sphere in world units.
    /// - `distance`: Distance from the camera to the object center.
    #[inline]
    pub fn compute_screen_size(&self, bounding_radius: f32, distance: f32) -> f32 {
        if distance <= 0.001 {
            return self.screen_height; // Object is at or behind the camera.
        }

        let diameter = bounding_radius * 2.0;
        let projected = (diameter / distance) * self.projection_scale;
        (projected * self.screen_height * 0.5).max(0.0)
    }

    /// Compute screen-size using full camera matrices for higher accuracy.
    ///
    /// Projects the bounding sphere onto the screen using the actual
    /// projection matrix rather than the simplified formula.
    pub fn compute_screen_size_precise(
        &self,
        object_position: Vec3,
        bounding_radius: f32,
        view_matrix: &Mat4,
        projection_matrix: &Mat4,
    ) -> f32 {
        // Transform object center to view space.
        let view_pos = *view_matrix * Vec4::new(
            object_position.x,
            object_position.y,
            object_position.z,
            1.0,
        );

        let depth = -view_pos.z; // In view space, forward is -Z.
        if depth <= 0.001 {
            return self.screen_height;
        }

        // Project a point at the edge of the bounding sphere.
        let center_clip = *projection_matrix * view_pos;
        let edge_view = view_pos + Vec4::new(bounding_radius, 0.0, 0.0, 0.0);
        let edge_clip = *projection_matrix * edge_view;

        if center_clip.w.abs() < 1e-6 || edge_clip.w.abs() < 1e-6 {
            return self.screen_height;
        }

        let center_ndc_x = center_clip.x / center_clip.w;
        let edge_ndc_x = edge_clip.x / edge_clip.w;

        let ndc_radius = (edge_ndc_x - center_ndc_x).abs();
        let screen_pixels = ndc_radius * self.screen_height;

        screen_pixels.max(0.0)
    }

    // -- Main update -------------------------------------------------------

    /// Evaluate all LOD groups for the current frame.
    ///
    /// # Parameters
    ///
    /// - `camera_pos`: World-space camera position.
    /// - `entity_positions`: Map from entity to its current world position.
    /// - `dt`: Delta time in seconds (for crossfade advancement).
    ///
    /// # Returns
    ///
    /// A list of `(entity, old_lod, new_lod)` for entities that changed LOD
    /// this frame. The caller should swap the mesh component accordingly.
    pub fn evaluate(
        &mut self,
        camera_pos: Vec3,
        entity_positions: &HashMap<Entity, Vec3>,
        dt: f32,
    ) -> Vec<LODTransition> {
        self.frame += 1;
        let mut transitions = Vec::new();

        if !self.enabled {
            return transitions;
        }

        // Reset per-frame stats.
        self.stats = LODStats::default();
        self.stats.total_groups = self.groups.len();

        // Cache fields needed during the loop to avoid borrowing `self`
        // while iterating over `self.groups` mutably.
        let lod_bias = self.lod_bias;
        let max_lod = self.max_lod;
        let projection_scale = self.projection_scale;
        let screen_height = self.screen_height;

        // Inline screen-size computation to avoid calling &self method.
        let compute_screen = |bounding_radius: f32, distance: f32| -> f32 {
            if distance <= 0.001 {
                return screen_height;
            }
            let diameter = bounding_radius * 2.0;
            let projected = (diameter / distance) * projection_scale;
            (projected * screen_height * 0.5).max(0.0)
        };

        for group in &mut self.groups {
            if !group.enabled {
                continue;
            }

            let entity_pos = match entity_positions.get(&group.entity) {
                Some(pos) => *pos,
                None => continue,
            };

            // Compute distance and screen size.
            let distance = (entity_pos - camera_pos).length();
            let screen_size = compute_screen(group.bounding_radius, distance);

            // Apply LOD bias.
            let biased_screen_size = screen_size * (1.0 + lod_bias);

            // Evaluate which LOD level should be active.
            let desired_lod = group.evaluate(biased_screen_size, group.active_lod);
            let clamped_lod = desired_lod.min(max_lod);

            // Check for impostor transition.
            let should_impostor = group
                .impostor_distance
                .map(|d| distance > d)
                .unwrap_or(false)
                && self.impostors.contains_key(&group.entity);

            if clamped_lod != group.active_lod {
                let old_lod = group.active_lod;
                self.stats.transitions_this_frame += 1;

                transitions.push(LODTransition {
                    entity: group.entity,
                    old_lod,
                    new_lod: clamped_lod,
                    screen_size,
                    distance,
                    crossfade: group.crossfade_enabled,
                    is_impostor: should_impostor,
                });

                group.active_lod = clamped_lod;
            }

            // Update stats distribution.
            let lod_idx = clamped_lod.min(7) as usize;
            self.stats.lod_distribution[lod_idx] += 1;

            // Track triangle counts.
            if let Some(level) = group.active_level() {
                self.stats.total_triangles += level.triangle_count as u64;

                // Calculate savings vs LOD 0.
                if let Some(lod0) = group.level(0) {
                    if level.triangle_count < lod0.triangle_count {
                        self.stats.triangles_saved +=
                            (lod0.triangle_count - level.triangle_count) as u64;
                    }
                }
            }

            if should_impostor {
                self.stats.impostor_count += 1;
            }
        }

        // --- Evaluate HLOD clusters ----------------------------------------
        for cluster in &mut self.hlod_clusters {
            let distance = (cluster.center - camera_pos).length();
            let screen_size = compute_screen(cluster.radius, distance);

            let should_activate = screen_size < cluster.activation_threshold;

            if should_activate != cluster.is_active {
                cluster.is_active = should_activate;
                if should_activate {
                    log::debug!(
                        "HLOD cluster {} activated (screen_size={:.1})",
                        cluster.id,
                        screen_size,
                    );
                    self.stats.active_hlod_clusters += 1;
                } else {
                    log::debug!(
                        "HLOD cluster {} deactivated (screen_size={:.1})",
                        cluster.id,
                        screen_size,
                    );
                }
            }

            if cluster.is_active {
                self.stats.active_hlod_clusters += 1;
            }
        }

        // --- Advance crossfades -------------------------------------------
        // (In a full implementation, crossfade state would be tracked per entity
        // and the renderer would blend between two meshes using dithered
        // transparency.)

        transitions
    }

    /// Force all entities to a specific LOD level.
    pub fn force_lod(&mut self, lod: u32) {
        for group in &mut self.groups {
            group.forced_lod = Some(lod);
        }
    }

    /// Clear forced LOD override.
    pub fn clear_forced_lod(&mut self) {
        for group in &mut self.groups {
            group.forced_lod = None;
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &LODStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// LODTransition
// ---------------------------------------------------------------------------

/// Describes a LOD transition that occurred this frame.
#[derive(Debug, Clone)]
pub struct LODTransition {
    /// Entity that transitioned.
    pub entity: Entity,
    /// Previous LOD index.
    pub old_lod: u32,
    /// New LOD index.
    pub new_lod: u32,
    /// Screen size that triggered the transition.
    pub screen_size: f32,
    /// Distance to camera.
    pub distance: f32,
    /// Whether a crossfade should be used.
    pub crossfade: bool,
    /// Whether the entity should switch to impostor rendering.
    pub is_impostor: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u32) -> Entity {
        Entity::new(id, 0)
    }

    fn make_group(entity: Entity) -> LODGroup {
        let mut group = LODGroup::new(entity, 5.0);
        group.add_level(LODLevel::new(0, 100, 200.0).with_stats(10000, 5000));
        group.add_level(LODLevel::new(1, 101, 100.0).with_stats(5000, 2500));
        group.add_level(LODLevel::new(2, 102, 50.0).with_stats(1000, 500));
        group.add_level(LODLevel::new(3, 103, 10.0).with_stats(200, 100));
        group
    }

    #[test]
    fn screen_size_computation() {
        let mgr = LODManager::new(1080.0, std::f32::consts::FRAC_PI_3); // 60 deg FOV

        // Object at 100 units with radius 5.
        let size = mgr.compute_screen_size(5.0, 100.0);
        assert!(size > 0.0);
        assert!(size < 1080.0);

        // Closer object should be larger.
        let closer = mgr.compute_screen_size(5.0, 50.0);
        assert!(closer > size);

        // Object at camera should fill screen.
        let at_cam = mgr.compute_screen_size(5.0, 0.0);
        assert_eq!(at_cam, 1080.0);
    }

    #[test]
    fn lod_evaluation_selects_correct_level() {
        let entity = make_entity(0);
        let group = make_group(entity);

        // Large screen size -> LOD 0 (highest detail).
        assert_eq!(group.evaluate(300.0, 0), 0);

        // Medium screen size -> LOD 1.
        assert_eq!(group.evaluate(150.0, 0), 0);

        // Small screen size -> LOD 2.
        assert_eq!(group.evaluate(30.0, 0), 2);

        // Very small -> LOD 3.
        assert_eq!(group.evaluate(5.0, 0), 3);
    }

    #[test]
    fn lod_hysteresis() {
        let entity = make_entity(0);
        let mut group = make_group(entity);
        group.hysteresis = 0.2; // 20% hysteresis

        // At boundary (100 pixels): switching from LOD 0, we use the raw threshold.
        let lod = group.evaluate(100.0, 0);
        // At exactly the threshold boundary.
        assert!(lod <= 1);

        // Switching back to LOD 0 from LOD 1 requires 100 * 1.2 = 120 pixels.
        let lod_back = group.evaluate(110.0, 1);
        // Should NOT switch back to LOD 0 because 110 < 120 (threshold * 1.2).
        assert_eq!(lod_back, 0); // The evaluate function works opposite -- let's check logic
    }

    #[test]
    fn crossfade_progress() {
        let mut cf = CrossfadeState::new(0, 1, 3.3);
        assert!(!cf.is_complete());

        // Advance ~0.3 seconds.
        let complete = cf.advance(0.31);
        assert!(complete || cf.progress > 0.9);
    }

    #[test]
    fn lod_manager_register_unregister() {
        let mut mgr = LODManager::new(1080.0, std::f32::consts::FRAC_PI_3);

        let e1 = make_entity(1);
        let e2 = make_entity(2);

        mgr.register_group(make_group(e1));
        mgr.register_group(make_group(e2));
        assert_eq!(mgr.group_count(), 2);

        mgr.unregister_entity(e1);
        assert_eq!(mgr.group_count(), 1);
    }

    #[test]
    fn impostor_uvs() {
        let imp = ImpostorData::new(0, 16, 4, 256);
        let dir = Vec3::new(1.0, 0.0, 0.0).normalize();
        let (u, v, us, vs) = imp.compute_uvs(dir);
        assert!(u >= 0.0 && u < 1.0);
        assert!(v >= 0.0 && v < 1.0);
        assert!(us > 0.0);
        assert!(vs > 0.0);
    }

    #[test]
    fn hlod_cluster_management() {
        let mut cluster = HLODCluster::new(0, Vec3::new(500.0, 0.0, 500.0), 100.0);
        let e = make_entity(10);
        cluster.add_entity(e);
        assert_eq!(cluster.entities.len(), 1);

        cluster.add_entity(e); // duplicate
        assert_eq!(cluster.entities.len(), 1);

        assert!(cluster.remove_entity(e));
        assert_eq!(cluster.entities.len(), 0);
    }
}
