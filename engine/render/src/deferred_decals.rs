// engine/render/src/deferred_decals.rs
//
// Deferred decal rendering system for the Genovo engine.
//
// Projects decals into the G-buffer during the deferred rendering pass,
// modifying albedo, normal, and roughness/metallic without requiring
// additional geometry. This approach avoids the mesh-projection issues
// of forward-rendered decals.
//
// Features:
// - Box-projected decals that write directly into G-buffer targets.
// - Normal blending modes: replace, reorient, overlay.
// - Angle fade to prevent stretching on surfaces nearly parallel to projection.
// - Lifetime management with fade-in and fade-out.
// - Sorting and priority for overlapping decals.
// - Material channels: albedo, normal, roughness, metallic, emissive.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum active decals in the scene.
const MAX_ACTIVE_DECALS: usize = 4096;

/// Default fade-in duration in seconds.
const DEFAULT_FADE_IN: f32 = 0.1;

/// Default fade-out duration in seconds.
const DEFAULT_FADE_OUT: f32 = 1.0;

/// Minimum angle between decal projection and surface normal (in radians)
/// below which the decal starts fading.
const MIN_ANGLE_THRESHOLD: f32 = 0.15;

/// Maximum angle between decal projection and surface normal (in radians)
/// above which the decal is fully visible.
const MAX_ANGLE_THRESHOLD: f32 = 0.5;

// ---------------------------------------------------------------------------
// Normal Blend Mode
// ---------------------------------------------------------------------------

/// How decal normals are blended with the G-buffer normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalBlendMode {
    /// Replace the G-buffer normal entirely.
    Replace,
    /// Reoriented normal mapping (preserves surface curvature).
    Reorient,
    /// Linear blend between surface normal and decal normal.
    LinearBlend,
    /// Overlay blend (detail normal map style).
    Overlay,
    /// Do not modify the normal channel.
    None,
}

// ---------------------------------------------------------------------------
// Decal Channel Mask
// ---------------------------------------------------------------------------

/// Which G-buffer channels a decal writes to.
#[derive(Debug, Clone, Copy)]
pub struct DecalChannelMask {
    /// Write to albedo channel.
    pub albedo: bool,
    /// Write to normal channel.
    pub normal: bool,
    /// Write to roughness channel.
    pub roughness: bool,
    /// Write to metallic channel.
    pub metallic: bool,
    /// Write to emissive channel.
    pub emissive: bool,
    /// Write to ambient occlusion channel.
    pub ao: bool,
}

impl Default for DecalChannelMask {
    fn default() -> Self {
        Self {
            albedo: true,
            normal: true,
            roughness: true,
            metallic: false,
            emissive: false,
            ao: false,
        }
    }
}

impl DecalChannelMask {
    /// All channels enabled.
    pub fn all() -> Self {
        Self {
            albedo: true,
            normal: true,
            roughness: true,
            metallic: true,
            emissive: true,
            ao: true,
        }
    }

    /// Only albedo channel.
    pub fn albedo_only() -> Self {
        Self {
            albedo: true,
            ..Self::none()
        }
    }

    /// No channels enabled.
    pub fn none() -> Self {
        Self {
            albedo: false,
            normal: false,
            roughness: false,
            metallic: false,
            emissive: false,
            ao: false,
        }
    }

    /// Returns the number of active channels.
    pub fn active_count(&self) -> u32 {
        self.albedo as u32
            + self.normal as u32
            + self.roughness as u32
            + self.metallic as u32
            + self.emissive as u32
            + self.ao as u32
    }
}

// ---------------------------------------------------------------------------
// Decal Material
// ---------------------------------------------------------------------------

/// Material properties for a deferred decal.
#[derive(Debug, Clone)]
pub struct DecalMaterial {
    /// Unique material identifier.
    pub id: u64,
    /// Albedo texture handle (0 = no texture).
    pub albedo_texture: u64,
    /// Normal map texture handle (0 = no normal map).
    pub normal_texture: u64,
    /// Roughness/metallic packed texture handle.
    pub orm_texture: u64,
    /// Emissive texture handle.
    pub emissive_texture: u64,
    /// Base albedo color tint.
    pub albedo_tint: [f32; 4],
    /// Base roughness value.
    pub roughness: f32,
    /// Base metallic value.
    pub metallic: f32,
    /// Emissive intensity.
    pub emissive_intensity: f32,
    /// Normal intensity (strength of normal mapping).
    pub normal_intensity: f32,
    /// Normal blend mode.
    pub normal_blend: NormalBlendMode,
    /// Channel mask.
    pub channels: DecalChannelMask,
}

impl Default for DecalMaterial {
    fn default() -> Self {
        Self {
            id: 0,
            albedo_texture: 0,
            normal_texture: 0,
            orm_texture: 0,
            emissive_texture: 0,
            albedo_tint: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.5,
            metallic: 0.0,
            emissive_intensity: 0.0,
            normal_intensity: 1.0,
            normal_blend: NormalBlendMode::Reorient,
            channels: DecalChannelMask::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Decal Instance
// ---------------------------------------------------------------------------

/// Unique identifier for a decal instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecalId(pub u64);

/// A single deferred decal instance in the scene.
#[derive(Debug, Clone)]
pub struct DeferredDecal {
    /// Unique decal identifier.
    pub id: DecalId,
    /// World-space position (center of the decal box).
    pub position: [f32; 3],
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Half-extents of the decal box (width/2, height/2, depth/2).
    pub half_extents: [f32; 3],
    /// Material index.
    pub material_id: u64,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Sort priority (higher = rendered on top).
    pub priority: i32,
    /// Rendering layer mask.
    pub layer_mask: u32,
    /// Angle fade configuration.
    pub angle_fade_start: f32,
    /// Angle fade end threshold.
    pub angle_fade_end: f32,
    /// Whether this decal is currently active.
    pub active: bool,
    /// Lifetime management.
    pub lifetime: DecalLifetime,
    /// Current age in seconds.
    pub age: f32,
    /// Current fade value (0.0 to 1.0) based on lifetime.
    pub fade_value: f32,
}

impl DeferredDecal {
    /// Create a new decal at the given position.
    pub fn new(id: DecalId, position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            id,
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            half_extents,
            material_id: 0,
            opacity: 1.0,
            priority: 0,
            layer_mask: 0xFFFFFFFF,
            angle_fade_start: MIN_ANGLE_THRESHOLD,
            angle_fade_end: MAX_ANGLE_THRESHOLD,
            active: true,
            lifetime: DecalLifetime::Permanent,
            age: 0.0,
            fade_value: 1.0,
        }
    }

    /// Set the decal material.
    pub fn with_material(mut self, material_id: u64) -> Self {
        self.material_id = material_id;
        self
    }

    /// Set the decal rotation.
    pub fn with_rotation(mut self, rotation: [f32; 4]) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the decal priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set a timed lifetime.
    pub fn with_lifetime(mut self, duration: f32, fade_in: f32, fade_out: f32) -> Self {
        self.lifetime = DecalLifetime::Timed {
            duration,
            fade_in,
            fade_out,
        };
        self
    }

    /// Compute the model matrix for this decal (4x4 column-major).
    pub fn model_matrix(&self) -> [[f32; 4]; 4] {
        let qx = self.rotation[0];
        let qy = self.rotation[1];
        let qz = self.rotation[2];
        let qw = self.rotation[3];

        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        let sx = self.half_extents[0];
        let sy = self.half_extents[1];
        let sz = self.half_extents[2];

        [
            [(1.0 - 2.0 * (yy + zz)) * sx, 2.0 * (xy + wz) * sx, 2.0 * (xz - wy) * sx, 0.0],
            [2.0 * (xy - wz) * sy, (1.0 - 2.0 * (xx + zz)) * sy, 2.0 * (yz + wx) * sy, 0.0],
            [2.0 * (xz + wy) * sz, 2.0 * (yz - wx) * sz, (1.0 - 2.0 * (xx + yy)) * sz, 0.0],
            [self.position[0], self.position[1], self.position[2], 1.0],
        ]
    }

    /// Update the decal's lifetime and fade.
    pub fn update(&mut self, dt: f32) {
        self.age += dt;

        match &self.lifetime {
            DecalLifetime::Permanent => {
                self.fade_value = self.opacity;
            }
            DecalLifetime::Timed {
                duration,
                fade_in,
                fade_out,
            } => {
                if self.age >= *duration {
                    self.active = false;
                    self.fade_value = 0.0;
                } else if self.age < *fade_in {
                    self.fade_value = self.opacity * (self.age / fade_in);
                } else if self.age > duration - fade_out {
                    let remaining = duration - self.age;
                    self.fade_value = self.opacity * (remaining / fade_out);
                } else {
                    self.fade_value = self.opacity;
                }
            }
            DecalLifetime::UntilRemoved { fade_out } => {
                if !self.active {
                    self.fade_value -= dt / fade_out;
                    if self.fade_value <= 0.0 {
                        self.fade_value = 0.0;
                    }
                } else {
                    self.fade_value = self.opacity;
                }
            }
        }
    }

    /// Compute angle fade factor for a given surface normal.
    pub fn angle_fade(&self, surface_normal: [f32; 3]) -> f32 {
        // Decal projection direction is the local Y axis.
        let qx = self.rotation[0];
        let qy = self.rotation[1];
        let qz = self.rotation[2];
        let qw = self.rotation[3];

        let proj_dir = [
            2.0 * (qx * qy - qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz + qw * qx),
        ];

        let dot = (proj_dir[0] * surface_normal[0]
            + proj_dir[1] * surface_normal[1]
            + proj_dir[2] * surface_normal[2])
            .abs();

        let angle = dot.acos();

        if angle <= self.angle_fade_start {
            1.0
        } else if angle >= self.angle_fade_end {
            0.0
        } else {
            let t = (angle - self.angle_fade_start) / (self.angle_fade_end - self.angle_fade_start);
            1.0 - t
        }
    }

    /// Check if the decal should be removed.
    pub fn should_remove(&self) -> bool {
        match &self.lifetime {
            DecalLifetime::Permanent => false,
            DecalLifetime::Timed { duration, .. } => self.age >= *duration,
            DecalLifetime::UntilRemoved { .. } => !self.active && self.fade_value <= 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Decal Lifetime
// ---------------------------------------------------------------------------

/// Lifetime mode for a decal.
#[derive(Debug, Clone)]
pub enum DecalLifetime {
    /// Permanent decal, never expires.
    Permanent,
    /// Timed decal with fade-in and fade-out.
    Timed {
        /// Total lifetime in seconds.
        duration: f32,
        /// Fade-in duration in seconds.
        fade_in: f32,
        /// Fade-out duration in seconds.
        fade_out: f32,
    },
    /// Remains until explicitly removed, then fades out.
    UntilRemoved {
        /// Fade-out duration when removed.
        fade_out: f32,
    },
}

// ---------------------------------------------------------------------------
// GPU Decal Data
// ---------------------------------------------------------------------------

/// Per-decal data sent to the GPU for rendering.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuDecalData {
    /// Inverse model matrix for projecting world-space to decal-space.
    pub inv_model_matrix: [[f32; 4]; 4],
    /// Albedo tint and opacity (RGBA).
    pub color: [f32; 4],
    /// Normal blend mode and angle fade parameters.
    pub params: [f32; 4],
    /// Channel mask packed into a u32.
    pub channel_mask: u32,
    /// Texture indices packed.
    pub texture_indices: [u32; 3],
}

impl GpuDecalData {
    /// Size of this struct in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Decal Manager
// ---------------------------------------------------------------------------

/// Manages all deferred decals in the scene.
#[derive(Debug)]
pub struct DeferredDecalManager {
    /// All active decals.
    pub decals: Vec<DeferredDecal>,
    /// Materials database.
    pub materials: HashMap<u64, DecalMaterial>,
    /// Next decal ID.
    next_id: u64,
    /// Maximum number of active decals.
    pub max_decals: usize,
    /// GPU data buffer for upload.
    pub gpu_data: Vec<GpuDecalData>,
    /// Whether the GPU data needs updating.
    pub dirty: bool,
    /// Statistics.
    pub stats: DecalStats,
}

impl DeferredDecalManager {
    /// Create a new decal manager.
    pub fn new() -> Self {
        Self {
            decals: Vec::new(),
            materials: HashMap::new(),
            next_id: 1,
            max_decals: MAX_ACTIVE_DECALS,
            gpu_data: Vec::new(),
            dirty: true,
            stats: DecalStats::default(),
        }
    }

    /// Register a decal material.
    pub fn register_material(&mut self, material: DecalMaterial) {
        self.materials.insert(material.id, material);
    }

    /// Spawn a new decal. Returns the decal ID.
    pub fn spawn(&mut self, mut decal: DeferredDecal) -> DecalId {
        let id = DecalId(self.next_id);
        self.next_id += 1;
        decal.id = id;

        // If at capacity, remove the oldest timed decal.
        if self.decals.len() >= self.max_decals {
            self.remove_oldest();
        }

        self.decals.push(decal);
        self.dirty = true;
        id
    }

    /// Remove a decal by ID (immediately or trigger fade-out).
    pub fn remove(&mut self, id: DecalId, immediate: bool) {
        if let Some(decal) = self.decals.iter_mut().find(|d| d.id == id) {
            if immediate {
                decal.active = false;
                decal.fade_value = 0.0;
            } else {
                decal.active = false; // Will fade out based on lifetime.
            }
        }
        self.dirty = true;
    }

    /// Remove the oldest non-permanent decal.
    fn remove_oldest(&mut self) {
        let mut oldest_idx = None;
        let mut oldest_age = -1.0f32;

        for (i, decal) in self.decals.iter().enumerate() {
            if !matches!(decal.lifetime, DecalLifetime::Permanent) && decal.age > oldest_age {
                oldest_age = decal.age;
                oldest_idx = Some(i);
            }
        }

        if let Some(idx) = oldest_idx {
            self.decals.swap_remove(idx);
        }
    }

    /// Update all decals (lifetime, fade, cleanup).
    pub fn update(&mut self, dt: f32) {
        for decal in &mut self.decals {
            decal.update(dt);
        }

        // Remove dead decals.
        let before = self.decals.len();
        self.decals.retain(|d| !d.should_remove());
        if self.decals.len() != before {
            self.dirty = true;
        }

        // Sort by priority for rendering order.
        self.decals.sort_by(|a, b| a.priority.cmp(&b.priority));

        self.update_stats();
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.active_decals = self.decals.len() as u32;
        self.stats.permanent_decals = self.decals.iter()
            .filter(|d| matches!(d.lifetime, DecalLifetime::Permanent))
            .count() as u32;
        self.stats.timed_decals = self.decals.iter()
            .filter(|d| matches!(d.lifetime, DecalLifetime::Timed { .. }))
            .count() as u32;
        self.stats.material_count = self.materials.len() as u32;
    }

    /// Get a decal by ID.
    pub fn get(&self, id: DecalId) -> Option<&DeferredDecal> {
        self.decals.iter().find(|d| d.id == id)
    }

    /// Get a mutable reference to a decal by ID.
    pub fn get_mut(&mut self, id: DecalId) -> Option<&mut DeferredDecal> {
        self.dirty = true;
        self.decals.iter_mut().find(|d| d.id == id)
    }

    /// Remove all decals.
    pub fn clear(&mut self) {
        self.decals.clear();
        self.gpu_data.clear();
        self.dirty = true;
    }

    /// Returns the number of active decals.
    pub fn count(&self) -> usize {
        self.decals.len()
    }
}

/// Statistics for the decal system.
#[derive(Debug, Clone, Default)]
pub struct DecalStats {
    /// Number of active decals.
    pub active_decals: u32,
    /// Number of permanent decals.
    pub permanent_decals: u32,
    /// Number of timed decals.
    pub timed_decals: u32,
    /// Number of registered materials.
    pub material_count: u32,
    /// GPU buffer size in bytes.
    pub gpu_buffer_bytes: usize,
    /// Decals spawned this frame.
    pub spawned_this_frame: u32,
    /// Decals removed this frame.
    pub removed_this_frame: u32,
}
