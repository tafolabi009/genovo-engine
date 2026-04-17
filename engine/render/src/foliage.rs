// engine/render/src/foliage.rs
//
// GPU-instanced foliage rendering system for the Genovo engine.
//
// Renders thousands of vegetation objects (grass, bushes, trees, flowers) using
// hardware instancing. Objects of the same type share a single mesh and material,
// with per-instance data (transform, color variation, LOD alpha) packed into a
// GPU buffer for a single `draw_indexed_instanced` call per type per LOD level.
//
// # Architecture
//
// ```text
//  FoliageManager
//    |-- scatter foliage from density maps
//    |-- stream chunks based on camera distance
//    |-- per-frame: select visible chunks, update LODs
//    |
//    +-> FoliageRenderer
//          |-- batch instances by type and LOD
//          |-- upload per-chunk instance buffers
//          |-- draw_indexed_instanced per batch
//          |-- wind animation via uniform params
// ```
//
// The foliage is spatially divided into `FoliageChunk`s (axis-aligned regions,
// typically matching terrain tiles). Each chunk contains instances of a single
// foliage type and maintains its own instance buffer.

use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default chunk size in world units along X and Z.
pub const DEFAULT_CHUNK_SIZE: f32 = 32.0;

/// Maximum number of instances per chunk.
pub const MAX_INSTANCES_PER_CHUNK: usize = 4096;

/// Maximum number of foliage types.
pub const MAX_FOLIAGE_TYPES: usize = 256;

/// Number of LOD levels supported.
pub const LOD_LEVEL_COUNT: usize = 4;

/// Epsilon for floating point comparisons.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// AABB (local definition)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Creates an AABB from min/max corners.
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// An invalid (inside-out) AABB.
    pub const INVALID: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Center of the box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents.
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Expand to include a point.
    #[inline]
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Returns true if two AABBs overlap.
    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

// ---------------------------------------------------------------------------
// Frustum (minimal for culling)
// ---------------------------------------------------------------------------

/// A plane for frustum culling.
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance: f32,
}

impl Plane {
    #[inline]
    fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }
}

/// A view frustum with six clipping planes.
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    planes: [Plane; 6],
}

impl ViewFrustum {
    /// Extracts frustum planes from a view-projection matrix.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let cols = vp.to_cols_array_2d();
        let row = |r: usize| -> Vec4 {
            Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r])
        };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let extract = |v: Vec4| -> Plane {
            let len = Vec3::new(v.x, v.y, v.z).length();
            if len < EPSILON {
                return Plane {
                    normal: Vec3::Y,
                    distance: 0.0,
                };
            }
            Plane {
                normal: Vec3::new(v.x, v.y, v.z) / len,
                distance: -v.w / len,
            }
        };

        Self {
            planes: [
                extract(r3 + r0), // left
                extract(r3 - r0), // right
                extract(r3 + r1), // bottom
                extract(r3 - r1), // top
                extract(r3 + r2), // near
                extract(r3 - r2), // far
            ],
        }
    }

    /// Returns true if the AABB is at least partially inside the frustum.
    pub fn contains_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            if plane.signed_distance(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

// ===========================================================================
// WindParams
// ===========================================================================

/// Wind animation parameters passed as uniforms to the vertex shader.
///
/// The vertex shader uses these to displace foliage vertices, simulating
/// wind-driven motion. The displacement is a combination of a global sway
/// (bending the trunk/stem) and a local flutter (high-frequency leaf motion).
#[derive(Debug, Clone, Copy)]
pub struct WindParams {
    /// Primary wind direction (normalized XZ).
    pub direction: Vec2,
    /// Wind speed (affects animation frequency).
    pub speed: f32,
    /// Wind strength (affects displacement amplitude).
    pub strength: f32,
    /// Turbulence intensity (random gusts).
    pub turbulence: f32,
    /// Gust frequency (how often gusts occur).
    pub gust_frequency: f32,
    /// Gust strength multiplier.
    pub gust_strength: f32,
    /// Phase offset for spatial variation (world-space frequency).
    pub phase_frequency: f32,
    /// Current time (for animation).
    pub time: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            direction: Vec2::new(1.0, 0.0),
            speed: 1.0,
            strength: 0.5,
            turbulence: 0.3,
            gust_frequency: 0.5,
            gust_strength: 2.0,
            phase_frequency: 0.1,
            time: 0.0,
        }
    }
}

impl WindParams {
    /// Packs wind parameters into a GPU-friendly format (4 Vec4s).
    pub fn to_gpu_data(&self) -> [Vec4; 4] {
        [
            Vec4::new(self.direction.x, self.direction.y, self.speed, self.strength),
            Vec4::new(self.turbulence, self.gust_frequency, self.gust_strength, self.phase_frequency),
            Vec4::new(self.time, 0.0, 0.0, 0.0),
            Vec4::ZERO, // reserved
        ]
    }

    /// Computes the wind displacement at a given world position (CPU-side
    /// preview for LOD/culling adjustments).
    pub fn displacement_at(&self, world_pos: Vec3) -> Vec3 {
        let phase = world_pos.x * self.phase_frequency + world_pos.z * self.phase_frequency * 0.7;
        let t = self.time * self.speed + phase;

        // Main sway (low frequency).
        let sway_x = (t * 1.1).sin() * self.strength * self.direction.x;
        let sway_z = (t * 0.9).sin() * self.strength * self.direction.y;

        // Gust (periodic pulse).
        let gust_phase = (t * self.gust_frequency).sin();
        let gust = if gust_phase > 0.7 {
            (gust_phase - 0.7) / 0.3 * self.gust_strength
        } else {
            0.0
        };

        let gust_x = gust * self.direction.x;
        let gust_z = gust * self.direction.y;

        Vec3::new(sway_x + gust_x, 0.0, sway_z + gust_z)
    }
}

// ===========================================================================
// FoliageType
// ===========================================================================

/// Describes a type of foliage (e.g., "oak_tree", "grass_tuft", "fern").
///
/// All instances of the same type share the same mesh and material. LOD
/// distances define when to switch between detail levels. The highest LOD
/// index is the billboard (camera-facing quad).
#[derive(Debug, Clone)]
pub struct FoliageType {
    /// Unique ID for this foliage type.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Handle to the mesh asset (opaque ID into the asset system).
    pub mesh_handle: u64,
    /// Handle to the material asset.
    pub material_handle: u64,
    /// LOD switch distances. `lod_distances[i]` is the camera distance at
    /// which we switch from LOD `i` to LOD `i+1`.
    /// The last entry is the cull distance (beyond which the instance is hidden).
    pub lod_distances: [f32; LOD_LEVEL_COUNT],
    /// Mesh handles for each LOD level (index 0 = highest detail).
    /// If a slot is 0, the base mesh_handle is used.
    pub lod_mesh_handles: [u64; LOD_LEVEL_COUNT],
    /// Whether instances of this type cast shadows.
    pub cast_shadows: bool,
    /// Whether instances of this type receive shadows.
    pub receive_shadows: bool,
    /// Wind response parameters specific to this foliage type.
    pub wind_response: WindResponse,
    /// Billboard mesh handle (camera-facing quad for the lowest LOD).
    pub billboard_mesh: u64,
    /// Billboard material handle.
    pub billboard_material: u64,
    /// Cross-fade dithering distance (world units over which LOD transitions
    /// are blended using a dithering pattern).
    pub crossfade_distance: f32,
    /// Base scale range for scatter (min, max). Instances are randomly scaled
    /// within this range.
    pub scale_range: (f32, f32),
    /// Minimum slope angle (radians) where this foliage can be placed.
    pub min_slope_angle: f32,
    /// Maximum slope angle (radians) where this foliage can be placed.
    pub max_slope_angle: f32,
    /// Altitude range (min, max) in world units.
    pub altitude_range: (f32, f32),
}

/// Per-type wind response multipliers.
#[derive(Debug, Clone, Copy)]
pub struct WindResponse {
    /// How much the trunk/stem bends (0 = rigid, 1 = full sway).
    pub trunk_sway: f32,
    /// How much the leaves/tips flutter.
    pub leaf_flutter: f32,
    /// Frequency multiplier for this type's animation.
    pub frequency_scale: f32,
}

impl Default for WindResponse {
    fn default() -> Self {
        Self {
            trunk_sway: 0.5,
            leaf_flutter: 1.0,
            frequency_scale: 1.0,
        }
    }
}

impl FoliageType {
    /// Creates a new foliage type with sensible defaults.
    pub fn new(id: u32, name: impl Into<String>, mesh_handle: u64, material_handle: u64) -> Self {
        Self {
            id,
            name: name.into(),
            mesh_handle,
            material_handle,
            lod_distances: [30.0, 60.0, 120.0, 250.0],
            lod_mesh_handles: [0; LOD_LEVEL_COUNT],
            cast_shadows: false,
            receive_shadows: true,
            wind_response: WindResponse::default(),
            billboard_mesh: 0,
            billboard_material: 0,
            crossfade_distance: 5.0,
            scale_range: (0.8, 1.2),
            min_slope_angle: 0.0,
            max_slope_angle: std::f32::consts::FRAC_PI_4, // 45 degrees
            altitude_range: (f32::NEG_INFINITY, f32::INFINITY),
        }
    }

    /// Sets LOD switch distances.
    pub fn with_lod_distances(mut self, distances: [f32; LOD_LEVEL_COUNT]) -> Self {
        self.lod_distances = distances;
        self
    }

    /// Enables shadow casting.
    pub fn with_shadows(mut self, cast: bool, receive: bool) -> Self {
        self.cast_shadows = cast;
        self.receive_shadows = receive;
        self
    }

    /// Sets the wind response.
    pub fn with_wind(mut self, trunk_sway: f32, leaf_flutter: f32) -> Self {
        self.wind_response.trunk_sway = trunk_sway;
        self.wind_response.leaf_flutter = leaf_flutter;
        self
    }

    /// Computes the LOD level for a given distance from the camera.
    /// Returns `None` if the distance exceeds the cull distance.
    pub fn compute_lod(&self, distance: f32) -> Option<u32> {
        for (i, &threshold) in self.lod_distances.iter().enumerate() {
            if distance < threshold {
                return Some(i as u32);
            }
        }
        None // beyond cull distance
    }

    /// Computes the LOD crossfade alpha for smooth transitions.
    /// Returns (lod_level, alpha) where alpha in [0, 1] indicates how much
    /// the next LOD is blended in.
    pub fn compute_lod_alpha(&self, distance: f32) -> Option<(u32, f32)> {
        for (i, &threshold) in self.lod_distances.iter().enumerate() {
            if distance < threshold {
                let lod = i as u32;
                // Compute crossfade alpha within the transition zone.
                let fade_start = threshold - self.crossfade_distance;
                let alpha = if distance > fade_start && self.crossfade_distance > EPSILON {
                    (distance - fade_start) / self.crossfade_distance
                } else {
                    0.0
                };
                return Some((lod, alpha.clamp(0.0, 1.0)));
            }
        }
        None
    }

    /// Returns the mesh handle for the given LOD level.
    pub fn mesh_for_lod(&self, lod: u32) -> u64 {
        let idx = lod as usize;
        if idx < LOD_LEVEL_COUNT && self.lod_mesh_handles[idx] != 0 {
            self.lod_mesh_handles[idx]
        } else if idx == LOD_LEVEL_COUNT - 1 && self.billboard_mesh != 0 {
            self.billboard_mesh
        } else {
            self.mesh_handle
        }
    }
}

// ===========================================================================
// FoliageInstance
// ===========================================================================

/// A single foliage instance placed in the world.
#[derive(Debug, Clone, Copy)]
pub struct FoliageInstance {
    /// World-space position.
    pub position: Vec3,
    /// Rotation around the up axis (radians). Full Quat would waste memory
    /// for foliage that only rotates around Y.
    pub rotation_y: f32,
    /// Uniform scale factor.
    pub scale: f32,
    /// Color variation (hue shift, saturation, brightness).
    pub color_variation: Vec3,
    /// Index into the foliage type registry.
    pub type_index: u32,
    /// Health/density factor [0, 1] (affects scale and color).
    pub health: f32,
}

impl FoliageInstance {
    /// Creates a new foliage instance.
    pub fn new(position: Vec3, type_index: u32) -> Self {
        Self {
            position,
            rotation_y: 0.0,
            scale: 1.0,
            color_variation: Vec3::ONE,
            type_index,
            health: 1.0,
        }
    }

    /// Sets the Y-axis rotation.
    pub fn with_rotation(mut self, rotation_y: f32) -> Self {
        self.rotation_y = rotation_y;
        self
    }

    /// Sets the uniform scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the color variation.
    pub fn with_color(mut self, color: Vec3) -> Self {
        self.color_variation = color;
        self
    }

    /// Computes the world-space transform matrix for this instance.
    pub fn world_matrix(&self) -> Mat4 {
        let rotation = Quat::from_rotation_y(self.rotation_y);
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            rotation,
            self.position,
        )
    }

    /// Computes the 3x4 instance matrix (row-major) for GPU upload.
    /// This is a compact representation: 3 rows of Vec4, each containing
    /// a row of the affine transform.
    pub fn instance_matrix_3x4(&self) -> [[f32; 4]; 3] {
        let m = self.world_matrix();
        let cols = m.to_cols_array_2d();
        // Transpose to row-major 3x4 (dropping the last row which is [0,0,0,1]).
        [
            [cols[0][0], cols[1][0], cols[2][0], cols[3][0]],
            [cols[0][1], cols[1][1], cols[2][1], cols[3][1]],
            [cols[0][2], cols[1][2], cols[2][2], cols[3][2]],
        ]
    }
}

// ===========================================================================
// InstanceData -- GPU-uploadable per-instance data
// ===========================================================================

/// Per-instance data laid out for GPU consumption.
///
/// This struct is designed for a single instanced draw call. All instances
/// of the same foliage type at the same LOD level share one buffer of these.
///
/// Memory layout: 64 bytes per instance (cache-line aligned).
///
/// ```text
/// Offset   Size   Field
///   0      48     world_matrix (3x4 float, row-major)
///  48       4     color_r (f32)
///  52       4     color_g (f32)
///  56       4     color_b (f32)
///  60       4     lod_alpha (f32, crossfade opacity)
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceData {
    /// 3x4 world matrix (row-major). Encodes translation, rotation, and scale.
    /// Row 0: right axis (x) + translation.x
    /// Row 1: up axis (y) + translation.y
    /// Row 2: forward axis (z) + translation.z
    pub world_matrix: [[f32; 4]; 3],
    /// Per-instance color tint (RGB). Used for color variation.
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    /// LOD crossfade alpha (0 = fully this LOD, 1 = fully next LOD).
    pub lod_alpha: f32,
}

impl InstanceData {
    /// Creates instance data from a foliage instance with a given LOD alpha.
    pub fn from_instance(instance: &FoliageInstance, lod_alpha: f32) -> Self {
        Self {
            world_matrix: instance.instance_matrix_3x4(),
            color_r: instance.color_variation.x,
            color_g: instance.color_variation.y,
            color_b: instance.color_variation.z,
            lod_alpha,
        }
    }

    /// Returns the byte size of one instance.
    pub const fn byte_size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Returns instance data as a byte slice (for GPU upload).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    /// Converts a slice of instance data to bytes (for GPU upload).
    pub fn slice_as_bytes(data: &[Self]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<Self>(),
            )
        }
    }
}

impl Default for InstanceData {
    fn default() -> Self {
        Self {
            world_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
            lod_alpha: 0.0,
        }
    }
}

// ===========================================================================
// FoliageChunk
// ===========================================================================

/// Chunk coordinate in the XZ grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Computes the chunk coordinate for a world-space position.
    pub fn from_world_pos(pos: Vec3, chunk_size: f32) -> Self {
        Self {
            x: (pos.x / chunk_size).floor() as i32,
            z: (pos.z / chunk_size).floor() as i32,
        }
    }

    /// Returns the world-space center of this chunk.
    pub fn world_center(&self, chunk_size: f32) -> Vec3 {
        Vec3::new(
            (self.x as f32 + 0.5) * chunk_size,
            0.0,
            (self.z as f32 + 0.5) * chunk_size,
        )
    }

    /// Manhattan distance to another chunk coordinate.
    pub fn manhattan_distance(&self, other: &ChunkCoord) -> i32 {
        (self.x - other.x).abs() + (self.z - other.z).abs()
    }
}

/// The state of a foliage chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    /// Chunk is not loaded.
    Unloaded,
    /// Chunk data is being loaded asynchronously.
    Loading,
    /// Chunk is loaded and its instance buffer is ready.
    Ready,
    /// Chunk is scheduled for unloading.
    PendingUnload,
}

/// A spatial region containing foliage instances of a single type.
///
/// Each chunk has its own instance buffer that can be uploaded to the GPU.
/// Chunks are loaded and unloaded based on camera proximity to support
/// streaming in large worlds.
#[derive(Debug, Clone)]
pub struct FoliageChunk {
    /// Chunk grid coordinate.
    pub coord: ChunkCoord,
    /// Index of the foliage type this chunk contains.
    pub type_index: u32,
    /// Axis-aligned bounding box of this chunk (world space).
    pub bounds: AABB,
    /// All instances in this chunk.
    pub instances: Vec<FoliageInstance>,
    /// Prepared GPU instance data (rebuilt when instances change).
    instance_data: Vec<Vec<InstanceData>>, // one vec per LOD level
    /// Current chunk state.
    pub state: ChunkState,
    /// Current LOD level for the entire chunk (-1 = hidden).
    pub chunk_lod: i32,
    /// Distance from camera center to chunk center (updated per frame).
    pub camera_distance: f32,
    /// Whether the instance data needs to be re-uploaded to the GPU.
    pub dirty: bool,
    /// GPU buffer handle (opaque; 0 = not yet allocated).
    pub gpu_buffer_handle: u64,
    /// Number of instances currently in the GPU buffer per LOD.
    pub gpu_instance_counts: [u32; LOD_LEVEL_COUNT],
}

impl FoliageChunk {
    /// Creates a new empty foliage chunk.
    pub fn new(coord: ChunkCoord, type_index: u32, chunk_size: f32) -> Self {
        let min = Vec3::new(
            coord.x as f32 * chunk_size,
            f32::NEG_INFINITY,
            coord.z as f32 * chunk_size,
        );
        let max = Vec3::new(
            (coord.x + 1) as f32 * chunk_size,
            f32::INFINITY,
            (coord.z + 1) as f32 * chunk_size,
        );

        Self {
            coord,
            type_index,
            bounds: AABB::new(min, max),
            instances: Vec::new(),
            instance_data: (0..LOD_LEVEL_COUNT).map(|_| Vec::new()).collect(),
            state: ChunkState::Unloaded,
            chunk_lod: -1,
            camera_distance: f32::INFINITY,
            dirty: true,
            gpu_buffer_handle: 0,
            gpu_instance_counts: [0; LOD_LEVEL_COUNT],
        }
    }

    /// Adds an instance to this chunk.
    pub fn add_instance(&mut self, instance: FoliageInstance) {
        // Update the AABB to include this instance's position.
        self.bounds.expand(instance.position);
        self.instances.push(instance);
        self.dirty = true;
    }

    /// Returns the number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns true if the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Rebuilds the per-LOD instance data from the raw instances.
    ///
    /// Requires the foliage type to compute LOD levels and crossfade alpha
    /// for each instance based on its distance to the camera.
    pub fn rebuild_instance_data(
        &mut self,
        foliage_type: &FoliageType,
        camera_pos: Vec3,
    ) {
        for lod_data in &mut self.instance_data {
            lod_data.clear();
        }

        for instance in &self.instances {
            let distance = (instance.position - camera_pos).length();
            if let Some((lod, alpha)) = foliage_type.compute_lod_alpha(distance) {
                let lod_idx = lod as usize;
                if lod_idx < LOD_LEVEL_COUNT {
                    self.instance_data[lod_idx].push(
                        InstanceData::from_instance(instance, alpha),
                    );
                }
            }
        }

        for lod in 0..LOD_LEVEL_COUNT {
            self.gpu_instance_counts[lod] = self.instance_data[lod].len() as u32;
        }

        self.dirty = true;
    }

    /// Returns the prepared instance data for a specific LOD level.
    pub fn instance_data_for_lod(&self, lod: usize) -> &[InstanceData] {
        if lod < self.instance_data.len() {
            &self.instance_data[lod]
        } else {
            &[]
        }
    }

    /// Returns the total byte size of instance data across all LODs.
    pub fn total_instance_data_bytes(&self) -> usize {
        self.instance_data
            .iter()
            .map(|v| v.len() * InstanceData::byte_size())
            .sum()
    }

    /// Computes the AABB tightly around all instances (recomputes from scratch).
    pub fn recompute_bounds(&mut self) {
        self.bounds = AABB::INVALID;
        for inst in &self.instances {
            self.bounds.expand(inst.position);
        }
        // Expand vertically to account for foliage height.
        self.bounds.min.y -= 0.5;
        self.bounds.max.y += 5.0;
    }
}

// ===========================================================================
// DrawBatch -- a batched instanced draw call
// ===========================================================================

/// Represents a single GPU instanced draw call.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    /// Foliage type index.
    pub type_index: u32,
    /// LOD level.
    pub lod_level: u32,
    /// Mesh handle for this LOD.
    pub mesh_handle: u64,
    /// Material handle for this LOD.
    pub material_handle: u64,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Offset (in instances) into the combined instance buffer.
    pub instance_offset: u32,
    /// Whether these instances cast shadows.
    pub cast_shadows: bool,
}

// ===========================================================================
// FoliageRenderer
// ===========================================================================

/// GPU-instanced foliage renderer.
///
/// Batches foliage instances by type and LOD level, uploads instance buffers,
/// and issues `draw_indexed_instanced` calls. Wind animation parameters are
/// passed as per-frame uniforms.
#[derive(Debug)]
pub struct FoliageRenderer {
    /// Current wind parameters.
    pub wind: WindParams,
    /// Combined instance buffer for all visible instances this frame.
    combined_buffer: Vec<InstanceData>,
    /// Draw batches for this frame.
    batches: Vec<DrawBatch>,
    /// GPU buffer handle for the combined instance buffer.
    gpu_instance_buffer: u64,
    /// Whether the GPU buffer needs updating.
    buffer_dirty: bool,
    /// Statistics from the last frame.
    last_stats: FoliageRenderStats,
}

/// Rendering statistics for one frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoliageRenderStats {
    /// Total number of foliage instances rendered (across all batches).
    pub total_instances: u32,
    /// Number of draw batches issued.
    pub draw_batches: u32,
    /// Number of chunks that were frustum-culled.
    pub chunks_culled: u32,
    /// Number of chunks that were rendered.
    pub chunks_rendered: u32,
    /// Total bytes uploaded to the GPU for instance data.
    pub instance_bytes: u64,
}

impl FoliageRenderer {
    /// Creates a new foliage renderer.
    pub fn new() -> Self {
        Self {
            wind: WindParams::default(),
            combined_buffer: Vec::new(),
            batches: Vec::new(),
            gpu_instance_buffer: 0,
            buffer_dirty: false,
            last_stats: FoliageRenderStats::default(),
        }
    }

    /// Returns the current draw batches.
    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// Returns the combined instance buffer data.
    pub fn instance_buffer_data(&self) -> &[InstanceData] {
        &self.combined_buffer
    }

    /// Returns statistics from the last frame.
    pub fn last_stats(&self) -> &FoliageRenderStats {
        &self.last_stats
    }

    /// Prepares draw batches from visible foliage chunks.
    ///
    /// This is the main per-frame entry point. It:
    /// 1. Frustum-culls chunks.
    /// 2. Collects per-LOD instance data from visible chunks.
    /// 3. Builds batched draw calls grouped by (type, LOD).
    pub fn prepare_batches(
        &mut self,
        chunks: &[FoliageChunk],
        foliage_types: &[FoliageType],
        frustum: &ViewFrustum,
        camera_pos: Vec3,
    ) {
        self.combined_buffer.clear();
        self.batches.clear();

        let mut stats = FoliageRenderStats::default();

        // Group chunks by (type_index, lod_level) and collect instance data.
        // Key: (type_index, lod_level) -> (start_offset, count).
        let mut batch_map: HashMap<(u32, u32), (u32, u32)> = HashMap::new();

        for chunk in chunks {
            if chunk.state != ChunkState::Ready {
                continue;
            }

            // Frustum cull the chunk.
            if !frustum.contains_aabb(&chunk.bounds) {
                stats.chunks_culled += 1;
                continue;
            }

            stats.chunks_rendered += 1;

            // Determine chunk LOD from camera distance.
            let type_idx = chunk.type_index;
            let foliage_type = match foliage_types.get(type_idx as usize) {
                Some(t) => t,
                None => continue,
            };

            // Add instance data for each LOD level that has instances.
            for lod in 0..LOD_LEVEL_COUNT {
                let data = chunk.instance_data_for_lod(lod);
                if data.is_empty() {
                    continue;
                }

                let offset = self.combined_buffer.len() as u32;
                self.combined_buffer.extend_from_slice(data);
                let count = data.len() as u32;

                let key = (type_idx, lod as u32);
                let entry = batch_map.entry(key).or_insert((offset, 0));
                entry.1 += count;

                stats.total_instances += count;
            }
        }

        // Build draw batches from the map.
        for ((type_idx, lod), (offset, count)) in &batch_map {
            if *count == 0 {
                continue;
            }

            let foliage_type = match foliage_types.get(*type_idx as usize) {
                Some(t) => t,
                None => continue,
            };

            let mesh_handle = foliage_type.mesh_for_lod(*lod);
            let material_handle = if *lod == LOD_LEVEL_COUNT as u32 - 1
                && foliage_type.billboard_material != 0
            {
                foliage_type.billboard_material
            } else {
                foliage_type.material_handle
            };

            self.batches.push(DrawBatch {
                type_index: *type_idx,
                lod_level: *lod,
                mesh_handle,
                material_handle,
                instance_count: *count,
                instance_offset: *offset,
                cast_shadows: foliage_type.cast_shadows,
            });

            stats.draw_batches += 1;
        }

        // Sort batches by type then LOD for optimal state changes.
        self.batches
            .sort_by(|a, b| a.type_index.cmp(&b.type_index).then(a.lod_level.cmp(&b.lod_level)));

        stats.instance_bytes =
            (self.combined_buffer.len() * InstanceData::byte_size()) as u64;
        self.buffer_dirty = true;
        self.last_stats = stats;
    }

    /// Returns the combined instance buffer as bytes (for GPU upload).
    pub fn instance_buffer_bytes(&self) -> &[u8] {
        InstanceData::slice_as_bytes(&self.combined_buffer)
    }

    /// Updates the wind time.
    pub fn update_wind(&mut self, delta_time: f32) {
        self.wind.time += delta_time;
    }

    /// Returns the wind uniform data for shader binding.
    pub fn wind_uniform_data(&self) -> [Vec4; 4] {
        self.wind.to_gpu_data()
    }
}

// ===========================================================================
// DensityMap
// ===========================================================================

/// A 2D density map used to control foliage scatter density.
///
/// Values are in [0, 1] where 0 = no foliage and 1 = maximum density.
/// The map covers a rectangular region in the XZ plane.
#[derive(Debug, Clone)]
pub struct DensityMap {
    /// Density values in row-major order.
    pub data: Vec<f32>,
    /// Width of the density map in texels.
    pub width: usize,
    /// Height (Z direction) of the density map in texels.
    pub height: usize,
    /// World-space origin (bottom-left corner of the map).
    pub origin: Vec2,
    /// World-space size of the area covered by the map.
    pub world_size: Vec2,
}

impl DensityMap {
    /// Creates a new density map filled with a uniform density.
    pub fn uniform(
        width: usize,
        height: usize,
        origin: Vec2,
        world_size: Vec2,
        density: f32,
    ) -> Self {
        Self {
            data: vec![density.clamp(0.0, 1.0); width * height],
            width,
            height,
            origin,
            world_size,
        }
    }

    /// Creates a density map from raw data.
    pub fn from_data(
        data: Vec<f32>,
        width: usize,
        height: usize,
        origin: Vec2,
        world_size: Vec2,
    ) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
            origin,
            world_size,
        }
    }

    /// Samples the density at a world-space XZ position using bilinear
    /// interpolation.
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let u = (world_x - self.origin.x) / self.world_size.x;
        let v = (world_z - self.origin.y) / self.world_size.y;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return 0.0;
        }

        let fx = u * (self.width as f32 - 1.0);
        let fy = v * (self.height as f32 - 1.0);

        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = fx - fx.floor();
        let ty = fy - fy.floor();

        let d00 = self.data[y0 * self.width + x0];
        let d10 = self.data[y0 * self.width + x1];
        let d01 = self.data[y1 * self.width + x0];
        let d11 = self.data[y1 * self.width + x1];

        let top = d00 + (d10 - d00) * tx;
        let bottom = d01 + (d11 - d01) * tx;

        top + (bottom - top) * ty
    }

    /// Sets the density at a specific texel.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value.clamp(0.0, 1.0);
        }
    }
}

// ===========================================================================
// FoliageManager
// ===========================================================================

/// High-level foliage management system.
///
/// Handles:
/// - Registering foliage types.
/// - Scattering foliage instances from density maps.
/// - Streaming chunks based on camera distance.
/// - Per-frame LOD updates and instance data preparation.
#[derive(Debug)]
pub struct FoliageManager {
    /// Registered foliage types.
    foliage_types: Vec<FoliageType>,
    /// All foliage chunks, keyed by (coord, type_index).
    chunks: HashMap<(ChunkCoord, u32), FoliageChunk>,
    /// Chunk size in world units.
    chunk_size: f32,
    /// Maximum streaming distance (chunks beyond this are unloaded).
    stream_distance: f32,
    /// The renderer.
    renderer: FoliageRenderer,
    /// Seed for deterministic scatter.
    scatter_seed: u64,
    /// Stats from the last update.
    last_stats: FoliageManagerStats,
}

/// Statistics from the foliage manager.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoliageManagerStats {
    /// Total number of registered foliage types.
    pub type_count: u32,
    /// Total number of chunks.
    pub total_chunks: u32,
    /// Number of loaded (ready) chunks.
    pub loaded_chunks: u32,
    /// Number of chunks streamed in this frame.
    pub chunks_loaded_this_frame: u32,
    /// Number of chunks unloaded this frame.
    pub chunks_unloaded_this_frame: u32,
    /// Total number of instances across all chunks.
    pub total_instances: u64,
    /// Render stats.
    pub render_stats: FoliageRenderStats,
}

impl FoliageManager {
    /// Creates a new foliage manager.
    pub fn new(chunk_size: f32, stream_distance: f32) -> Self {
        Self {
            foliage_types: Vec::new(),
            chunks: HashMap::new(),
            chunk_size,
            stream_distance,
            renderer: FoliageRenderer::new(),
            scatter_seed: 42,
            last_stats: FoliageManagerStats::default(),
        }
    }

    /// Creates a manager with default settings.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_CHUNK_SIZE, 200.0)
    }

    /// Sets the random seed for scatter.
    pub fn set_scatter_seed(&mut self, seed: u64) {
        self.scatter_seed = seed;
    }

    /// Returns a reference to the renderer.
    pub fn renderer(&self) -> &FoliageRenderer {
        &self.renderer
    }

    /// Returns a mutable reference to the renderer.
    pub fn renderer_mut(&mut self) -> &mut FoliageRenderer {
        &mut self.renderer
    }

    /// Returns statistics from the last update.
    pub fn last_stats(&self) -> &FoliageManagerStats {
        &self.last_stats
    }

    /// Registers a foliage type.
    pub fn register_type(&mut self, foliage_type: FoliageType) -> u32 {
        let index = self.foliage_types.len() as u32;
        self.foliage_types.push(foliage_type);
        index
    }

    /// Returns the registered foliage types.
    pub fn foliage_types(&self) -> &[FoliageType] {
        &self.foliage_types
    }

    /// Returns the chunk size.
    pub fn chunk_size(&self) -> f32 {
        self.chunk_size
    }

    /// Manually adds an instance to the appropriate chunk, creating the chunk
    /// if it doesn't exist.
    pub fn add_instance(&mut self, instance: FoliageInstance) {
        let coord = ChunkCoord::from_world_pos(instance.position, self.chunk_size);
        let key = (coord, instance.type_index);

        let chunk = self.chunks.entry(key).or_insert_with(|| {
            let mut c = FoliageChunk::new(coord, instance.type_index, self.chunk_size);
            c.state = ChunkState::Ready;
            c
        });

        chunk.add_instance(instance);
    }

    /// Scatters foliage instances for a given type using a density map.
    ///
    /// Uses a Poisson-disk-like distribution by placing candidate points on a
    /// regular grid jittered by a hash function, then accepting them based on
    /// the density map value.
    pub fn scatter_from_density_map(
        &mut self,
        type_index: u32,
        density_map: &DensityMap,
        max_density_per_sqm: f32,
        height_fn: &dyn Fn(f32, f32) -> f32,
    ) -> u32 {
        let foliage_type = match self.foliage_types.get(type_index as usize) {
            Some(t) => t.clone(),
            None => return 0,
        };

        // Grid spacing at max density.
        let spacing = 1.0 / max_density_per_sqm.sqrt();
        let mut placed = 0u32;

        let x_start = density_map.origin.x;
        let z_start = density_map.origin.y;
        let x_end = x_start + density_map.world_size.x;
        let z_end = z_start + density_map.world_size.y;

        let mut gx = x_start;
        while gx < x_end {
            let mut gz = z_start;
            while gz < z_end {
                // Hash-based jitter for this grid cell.
                let cell_x = (gx / spacing) as i32;
                let cell_z = (gz / spacing) as i32;
                let hash = Self::hash_cell(cell_x, cell_z, self.scatter_seed);

                // Jitter position within the cell.
                let jx = gx + (((hash >> 0) & 0xFFFF) as f32 / 65535.0 - 0.5) * spacing;
                let jz = gz + (((hash >> 16) & 0xFFFF) as f32 / 65535.0 - 0.5) * spacing;

                // Sample density.
                let density = density_map.sample(jx, jz);

                // Acceptance test: use another hash as a random [0,1] value.
                let acceptance = ((hash >> 32) & 0xFFFF) as f32 / 65535.0;
                if acceptance < density {
                    // Get height from the terrain.
                    let y = height_fn(jx, jz);

                    // Check altitude range.
                    if y < foliage_type.altitude_range.0 || y > foliage_type.altitude_range.1 {
                        gz += spacing;
                        continue;
                    }

                    // Random rotation and scale.
                    let rot = ((hash >> 48) & 0xFFFF) as f32 / 65535.0
                        * std::f32::consts::TAU;
                    let scale_t = ((hash >> 8) & 0xFF) as f32 / 255.0;
                    let scale = foliage_type.scale_range.0
                        + (foliage_type.scale_range.1 - foliage_type.scale_range.0) * scale_t;

                    // Color variation.
                    let color_var = Vec3::new(
                        0.9 + ((hash >> 24) & 0xFF) as f32 / 255.0 * 0.2,
                        0.85 + ((hash >> 40) & 0xFF) as f32 / 255.0 * 0.3,
                        0.9 + ((hash >> 56) & 0xFF) as f32 / 255.0 * 0.2,
                    );

                    let instance = FoliageInstance::new(Vec3::new(jx, y, jz), type_index)
                        .with_rotation(rot)
                        .with_scale(scale)
                        .with_color(color_var);

                    self.add_instance(instance);
                    placed += 1;
                }

                gz += spacing;
            }
            gx += spacing;
        }

        placed
    }

    /// Simple deterministic hash for scatter.
    fn hash_cell(x: i32, z: i32, seed: u64) -> u64 {
        let mut h = seed;
        h = h.wrapping_mul(6364136223846793005).wrapping_add(x as u64);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(z as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Per-frame update: stream chunks, update LODs, rebuild instance data,
    /// and prepare draw batches.
    pub fn update(&mut self, camera_pos: Vec3, view_projection: &Mat4, delta_time: f32) {
        let mut stats = FoliageManagerStats {
            type_count: self.foliage_types.len() as u32,
            total_chunks: self.chunks.len() as u32,
            ..Default::default()
        };

        let camera_chunk = ChunkCoord::from_world_pos(camera_pos, self.chunk_size);
        let stream_chunks = (self.stream_distance / self.chunk_size).ceil() as i32;

        // Stream management: load nearby chunks, unload distant ones.
        let chunk_keys: Vec<(ChunkCoord, u32)> = self.chunks.keys().cloned().collect();
        for key in &chunk_keys {
            let (coord, _type_idx) = key;
            let dist = coord.manhattan_distance(&camera_chunk);

            if dist > stream_chunks + 2 {
                // Unload distant chunk.
                if let Some(chunk) = self.chunks.get_mut(key) {
                    if chunk.state == ChunkState::Ready {
                        chunk.state = ChunkState::PendingUnload;
                        stats.chunks_unloaded_this_frame += 1;
                    }
                }
            }
        }

        // Remove chunks that are pending unload.
        self.chunks
            .retain(|_, chunk| chunk.state != ChunkState::PendingUnload);

        // Update loaded chunks.
        let frustum = ViewFrustum::from_view_projection(view_projection);
        let foliage_types = self.foliage_types.clone();

        let mut ready_chunks: Vec<&FoliageChunk> = Vec::new();

        for chunk in self.chunks.values_mut() {
            if chunk.state != ChunkState::Ready {
                continue;
            }

            stats.loaded_chunks += 1;
            stats.total_instances += chunk.instance_count() as u64;

            // Update camera distance.
            chunk.camera_distance = (chunk.bounds.center() - camera_pos).length();

            // Rebuild instance data if dirty or distance has changed significantly.
            if chunk.dirty {
                if let Some(ft) = foliage_types.get(chunk.type_index as usize) {
                    chunk.rebuild_instance_data(ft, camera_pos);
                }
            }
        }

        // Collect ready chunks for rendering.
        let ready_chunk_refs: Vec<FoliageChunk> = self
            .chunks
            .values()
            .filter(|c| c.state == ChunkState::Ready)
            .cloned()
            .collect();

        // Update wind and prepare batches.
        self.renderer.update_wind(delta_time);
        self.renderer
            .prepare_batches(&ready_chunk_refs, &foliage_types, &frustum, camera_pos);

        stats.render_stats = *self.renderer.last_stats();
        stats.total_chunks = self.chunks.len() as u32;
        self.last_stats = stats;
    }

    /// Returns the total number of instances across all chunks.
    pub fn total_instance_count(&self) -> usize {
        self.chunks.values().map(|c| c.instance_count()).sum()
    }

    /// Removes all foliage instances and chunks.
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// Removes all instances of a specific type.
    pub fn clear_type(&mut self, type_index: u32) {
        self.chunks.retain(|_, chunk| chunk.type_index != type_index);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_data_size() {
        assert_eq!(InstanceData::byte_size(), 64, "InstanceData should be 64 bytes");
    }

    #[test]
    fn test_instance_data_from_instance() {
        let inst = FoliageInstance::new(Vec3::new(10.0, 5.0, -3.0), 0)
            .with_rotation(std::f32::consts::FRAC_PI_2)
            .with_scale(2.0)
            .with_color(Vec3::new(0.9, 1.1, 0.95));

        let data = InstanceData::from_instance(&inst, 0.3);
        assert!((data.color_r - 0.9).abs() < EPSILON);
        assert!((data.color_g - 1.1).abs() < EPSILON);
        assert!((data.color_b - 0.95).abs() < EPSILON);
        assert!((data.lod_alpha - 0.3).abs() < EPSILON);

        // Check that the translation is embedded in the matrix.
        let tx = data.world_matrix[0][3];
        let ty = data.world_matrix[1][3];
        let tz = data.world_matrix[2][3];
        assert!((tx - 10.0).abs() < 0.01);
        assert!((ty - 5.0).abs() < 0.01);
        assert!((tz - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn test_foliage_type_lod() {
        let ft = FoliageType::new(0, "grass", 1, 1)
            .with_lod_distances([20.0, 40.0, 80.0, 150.0]);

        assert_eq!(ft.compute_lod(10.0), Some(0));
        assert_eq!(ft.compute_lod(25.0), Some(1));
        assert_eq!(ft.compute_lod(50.0), Some(2));
        assert_eq!(ft.compute_lod(100.0), Some(3));
        assert_eq!(ft.compute_lod(200.0), None);
    }

    #[test]
    fn test_foliage_type_lod_alpha() {
        let ft = FoliageType::new(0, "grass", 1, 1)
            .with_lod_distances([20.0, 40.0, 80.0, 150.0]);

        // Well within LOD 0 -- alpha should be 0.
        let (lod, alpha) = ft.compute_lod_alpha(5.0).unwrap();
        assert_eq!(lod, 0);
        assert!((alpha - 0.0).abs() < EPSILON);

        // At the crossfade boundary (20.0 - 5.0 = 15.0), alpha should start rising.
        let (lod, alpha) = ft.compute_lod_alpha(17.5).unwrap();
        assert_eq!(lod, 0);
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_chunk_coord_from_world() {
        let coord = ChunkCoord::from_world_pos(Vec3::new(50.0, 0.0, -20.0), 32.0);
        assert_eq!(coord.x, 1);
        assert_eq!(coord.z, -1);
    }

    #[test]
    fn test_chunk_coord_center() {
        let coord = ChunkCoord::new(2, 3);
        let center = coord.world_center(32.0);
        assert!((center.x - 80.0).abs() < EPSILON);
        assert!((center.z - 112.0).abs() < EPSILON);
    }

    #[test]
    fn test_density_map_sample() {
        let map = DensityMap::uniform(4, 4, Vec2::ZERO, Vec2::new(100.0, 100.0), 0.5);
        let d = map.sample(50.0, 50.0);
        assert!((d - 0.5).abs() < EPSILON);

        // Outside the map should return 0.
        let d = map.sample(-10.0, 50.0);
        assert!((d - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_density_map_bilinear() {
        let data = vec![
            0.0, 1.0,
            1.0, 0.0,
        ];
        let map = DensityMap::from_data(data, 2, 2, Vec2::ZERO, Vec2::new(10.0, 10.0));

        // Center should be ~0.5 (bilinear interpolation of corners).
        let d = map.sample(5.0, 5.0);
        assert!((d - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_wind_displacement() {
        let wind = WindParams::default();
        let disp = wind.displacement_at(Vec3::new(10.0, 0.0, 20.0));
        // Just verify it returns a finite value.
        assert!(disp.x.is_finite());
        assert!(disp.y.is_finite());
        assert!(disp.z.is_finite());
    }

    #[test]
    fn test_foliage_manager_add_instance() {
        let mut manager = FoliageManager::new(32.0, 100.0);
        manager.register_type(FoliageType::new(0, "grass", 1, 1));

        manager.add_instance(FoliageInstance::new(Vec3::new(5.0, 0.0, 5.0), 0));
        manager.add_instance(FoliageInstance::new(Vec3::new(6.0, 0.0, 5.0), 0));
        manager.add_instance(FoliageInstance::new(Vec3::new(50.0, 0.0, 50.0), 0));

        assert_eq!(manager.total_instance_count(), 3);
        // First two should be in the same chunk; third in a different one.
        let coord1 = ChunkCoord::from_world_pos(Vec3::new(5.0, 0.0, 5.0), 32.0);
        let coord2 = ChunkCoord::from_world_pos(Vec3::new(50.0, 0.0, 50.0), 32.0);
        assert_ne!(coord1, coord2);
    }

    #[test]
    fn test_foliage_scatter() {
        let mut manager = FoliageManager::new(32.0, 100.0);
        manager.register_type(FoliageType::new(0, "grass", 1, 1));

        let density_map =
            DensityMap::uniform(8, 8, Vec2::ZERO, Vec2::new(64.0, 64.0), 1.0);

        let placed = manager.scatter_from_density_map(0, &density_map, 0.25, &|_x, _z| 0.0);
        assert!(placed > 0, "Should have placed some instances");
        assert_eq!(manager.total_instance_count(), placed as usize);
    }

    #[test]
    fn test_frustum_cull_aabb() {
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = proj * view;
        let frustum = ViewFrustum::from_view_projection(&vp);

        // AABB in front of the camera should be visible.
        let aabb_front = AABB::new(
            Vec3::new(-1.0, -1.0, -10.0),
            Vec3::new(1.0, 1.0, -5.0),
        );
        assert!(frustum.contains_aabb(&aabb_front));

        // AABB behind the camera should not be visible.
        let aabb_behind = AABB::new(
            Vec3::new(-1.0, -1.0, 5.0),
            Vec3::new(1.0, 1.0, 10.0),
        );
        assert!(!frustum.contains_aabb(&aabb_behind));
    }

    #[test]
    fn test_draw_batch_generation() {
        let mut renderer = FoliageRenderer::new();
        let types = vec![
            FoliageType::new(0, "grass", 100, 200)
                .with_lod_distances([50.0, 100.0, 200.0, 400.0]),
        ];

        let mut chunk = FoliageChunk::new(ChunkCoord::new(0, 0), 0, 32.0);
        chunk.state = ChunkState::Ready;
        for i in 0..10 {
            chunk.add_instance(
                FoliageInstance::new(Vec3::new(i as f32, 0.0, 0.0), 0),
            );
        }
        chunk.rebuild_instance_data(&types[0], Vec3::ZERO);

        let vp = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 1000.0)
            * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let frustum = ViewFrustum::from_view_projection(&vp);

        renderer.prepare_batches(&[chunk], &types, &frustum, Vec3::ZERO);

        // Should have produced at least one batch.
        let stats = renderer.last_stats();
        assert!(stats.total_instances > 0);
        assert!(stats.draw_batches > 0);
    }
}
