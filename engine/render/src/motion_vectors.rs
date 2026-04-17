// engine/render/src/motion_vectors.rs
//
// Per-pixel motion vector computation for the Genovo engine.
//
// Motion vectors encode the screen-space displacement of each pixel between
// the current and previous frames. They are consumed by:
//
// - **Temporal Anti-Aliasing (TAA)** — to reproject history samples.
// - **Temporal upscaling** (FSR 2, DLSS, XeSS) — for temporal super-resolution.
// - **Motion blur** — to smear pixels along their motion direction.
// - **Temporal reprojection** in volumetric effects, SSGI, and reflections.
//
// # Types of motion vectors
//
// 1. **Camera motion vectors** — Derived from the delta between the current and
//    previous view-projection matrices. Correct for all static geometry.
// 2. **Per-object motion vectors** — Derived from the current and previous
//    world transforms of moving objects. Required for skeletal animation,
//    physics objects, etc.
// 3. **Depth-based estimation** — Fallback for static objects when no explicit
//    per-object data is available.
// 4. **Jitter removal** — Subtracts the sub-pixel TAA jitter from the motion
//    vectors so that the TAA resolve sees only actual motion.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Multiplies a 4x4 column-major matrix by a 4D vector.
#[inline]
fn mat4_mul_vec4(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    ]
}

/// Multiplies a 4x4 column-major matrix by a 3D point (w=1) and returns
/// the clip-space result (before perspective divide).
#[inline]
fn mat4_mul_point(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
    mat4_mul_vec4(m, [p[0], p[1], p[2], 1.0])
}

/// Clip-to-NDC: perspective divide.
#[inline]
fn clip_to_ndc(clip: [f32; 4]) -> [f32; 3] {
    if clip[3].abs() < 1e-7 {
        return [0.0, 0.0, 0.0];
    }
    let inv_w = 1.0 / clip[3];
    [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w]
}

/// NDC to screen UV: [-1,1] -> [0,1], with Y flip.
#[inline]
fn ndc_to_uv(ndc: [f32; 3]) -> [f32; 2] {
    [ndc[0] * 0.5 + 0.5, 0.5 - ndc[1] * 0.5]
}

/// Screen UV to NDC.
#[inline]
fn uv_to_ndc(uv: [f32; 2]) -> [f32; 3] {
    [uv[0] * 2.0 - 1.0, 1.0 - uv[1] * 2.0, 0.0]
}

/// Multiply two 4x4 column-major matrices.
fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut result = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    result
}

/// 4x4 identity matrix.
fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

// ---------------------------------------------------------------------------
// Motion vector settings
// ---------------------------------------------------------------------------

/// Configuration for motion vector generation.
#[derive(Debug, Clone)]
pub struct MotionVectorSettings {
    /// Whether motion vector generation is enabled.
    pub enabled: bool,
    /// Whether to include camera-motion-only vectors.
    pub camera_motion: bool,
    /// Whether to include per-object motion vectors.
    pub object_motion: bool,
    /// Whether to subtract TAA jitter from the output.
    pub remove_jitter: bool,
    /// Current frame's TAA jitter offset in NDC [-1, 1].
    pub jitter_current: [f32; 2],
    /// Previous frame's TAA jitter offset in NDC.
    pub jitter_previous: [f32; 2],
    /// Maximum motion vector length (screen UV units).
    /// Clamped to prevent artifacts from extremely fast motion.
    pub max_length: f32,
    /// Depth threshold for motion vector validity.
    /// Pixels with depth beyond this are treated as sky.
    pub sky_depth_threshold: f32,
    /// Whether to output velocities in pixels or UV units.
    pub output_in_pixels: bool,
    /// Screen dimensions (for pixel-space conversion).
    pub screen_width: f32,
    pub screen_height: f32,
    /// Whether to apply neighbour velocity clamping for TAA.
    pub velocity_clamping: bool,
    /// Maximum velocity for clamping (UV units/frame).
    pub clamp_max_velocity: f32,
}

impl MotionVectorSettings {
    /// Default settings.
    pub fn new() -> Self {
        Self {
            enabled: true,
            camera_motion: true,
            object_motion: true,
            remove_jitter: true,
            jitter_current: [0.0, 0.0],
            jitter_previous: [0.0, 0.0],
            max_length: 0.1,
            sky_depth_threshold: 0.9999,
            output_in_pixels: false,
            screen_width: 1920.0,
            screen_height: 1080.0,
            velocity_clamping: true,
            clamp_max_velocity: 0.05,
        }
    }

    /// Sets the TAA jitter for the current and previous frames.
    pub fn set_jitter(&mut self, current: [f32; 2], previous: [f32; 2]) {
        self.jitter_current = current;
        self.jitter_previous = previous;
    }

    /// Sets the screen dimensions.
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Returns the jitter delta in UV space.
    pub fn jitter_delta_uv(&self) -> [f32; 2] {
        [
            (self.jitter_current[0] - self.jitter_previous[0]) * 0.5,
            (self.jitter_current[1] - self.jitter_previous[1]) * -0.5,
        ]
    }
}

impl Default for MotionVectorSettings {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Camera motion vectors
// ---------------------------------------------------------------------------

/// Computes camera motion vectors from the view-projection matrix delta.
///
/// For each pixel, reconstructs the world position from depth, projects it
/// using the previous frame's view-projection matrix, and computes the
/// screen-space displacement.
///
/// # Arguments
/// * `depth_buffer` — Linearised depth buffer.
/// * `width`, `height` — Buffer dimensions.
/// * `curr_view_proj` — Current frame's view-projection matrix (col-major 4x4).
/// * `prev_view_proj` — Previous frame's view-projection matrix.
/// * `curr_inv_view_proj` — Inverse of current view-projection matrix.
/// * `settings` — Motion vector settings.
///
/// # Returns
/// Motion vector buffer: 2 floats per pixel (dU, dV) in UV space.
pub fn compute_camera_motion_vectors(
    depth_buffer: &[f32],
    width: u32,
    height: u32,
    curr_view_proj: &[f32; 16],
    prev_view_proj: &[f32; 16],
    curr_inv_view_proj: &[f32; 16],
    settings: &MotionVectorSettings,
) -> Vec<[f32; 2]> {
    let total = (width * height) as usize;
    let mut motion_vectors = vec![[0.0f32; 2]; total];

    if !settings.enabled || !settings.camera_motion {
        return motion_vectors;
    }

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let depth = depth_buffer[idx];

            // Skip sky pixels.
            if depth >= settings.sky_depth_threshold {
                motion_vectors[idx] = [0.0, 0.0];
                continue;
            }

            // Current screen UV.
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];

            // Reconstruct world position from depth.
            let ndc_current = [
                uv[0] * 2.0 - 1.0,
                1.0 - uv[1] * 2.0,
                depth,
                1.0,
            ];

            // Apply current inverse VP to get world position.
            let world_clip = mat4_mul_vec4(curr_inv_view_proj, ndc_current);
            if world_clip[3].abs() < 1e-7 {
                continue;
            }
            let world_pos = [
                world_clip[0] / world_clip[3],
                world_clip[1] / world_clip[3],
                world_clip[2] / world_clip[3],
            ];

            // Project world position using previous frame's VP.
            let prev_clip = mat4_mul_point(prev_view_proj, world_pos);
            let prev_ndc = clip_to_ndc(prev_clip);
            let prev_uv = ndc_to_uv(prev_ndc);

            // Motion vector = current UV - previous UV.
            let mut mv = [uv[0] - prev_uv[0], uv[1] - prev_uv[1]];

            // Remove TAA jitter if requested.
            if settings.remove_jitter {
                let jd = settings.jitter_delta_uv();
                mv[0] -= jd[0];
                mv[1] -= jd[1];
            }

            // Clamp length.
            let len = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt();
            if len > settings.max_length {
                let scale = settings.max_length / len;
                mv[0] *= scale;
                mv[1] *= scale;
            }

            // Convert to pixels if requested.
            if settings.output_in_pixels {
                mv[0] *= settings.screen_width;
                mv[1] *= settings.screen_height;
            }

            motion_vectors[idx] = mv;
        }
    }

    motion_vectors
}

// ---------------------------------------------------------------------------
// Per-object motion vectors
// ---------------------------------------------------------------------------

/// Per-object motion data: stores current and previous world transforms.
#[derive(Debug, Clone)]
pub struct ObjectMotionData {
    /// Object identifier.
    pub object_id: u64,
    /// Current frame's world transform (column-major 4x4).
    pub current_world: [f32; 16],
    /// Previous frame's world transform.
    pub previous_world: [f32; 16],
    /// Whether this object has valid previous frame data.
    pub has_previous: bool,
    /// Whether this object moved since the last frame.
    pub has_moved: bool,
}

impl ObjectMotionData {
    /// Creates motion data for a static object.
    pub fn new_static(object_id: u64, world: [f32; 16]) -> Self {
        Self {
            object_id,
            current_world: world,
            previous_world: world,
            has_previous: true,
            has_moved: false,
        }
    }

    /// Creates motion data for a moving object.
    pub fn new_moving(
        object_id: u64,
        current_world: [f32; 16],
        previous_world: [f32; 16],
    ) -> Self {
        let has_moved = current_world != previous_world;
        Self {
            object_id,
            current_world,
            previous_world,
            has_previous: true,
            has_moved,
        }
    }

    /// Creates motion data for a newly spawned object (no previous frame).
    pub fn new_spawned(object_id: u64, world: [f32; 16]) -> Self {
        Self {
            object_id,
            current_world: world,
            previous_world: world,
            has_previous: false,
            has_moved: false,
        }
    }

    /// Advances to the next frame: current becomes previous.
    pub fn advance(&mut self, new_world: [f32; 16]) {
        self.previous_world = self.current_world;
        self.current_world = new_world;
        self.has_previous = true;
        self.has_moved = self.current_world != self.previous_world;
    }

    /// Computes the motion vector for a vertex in local space.
    ///
    /// # Arguments
    /// * `local_pos` — Vertex position in object space.
    /// * `view_proj` — Current frame's view-projection matrix.
    /// * `prev_view_proj` — Previous frame's view-projection matrix.
    ///
    /// # Returns
    /// Screen-space motion vector in UV units.
    pub fn compute_vertex_motion(
        &self,
        local_pos: [f32; 3],
        view_proj: &[f32; 16],
        prev_view_proj: &[f32; 16],
    ) -> [f32; 2] {
        if !self.has_moved && self.has_previous {
            // No object motion — camera motion is handled separately.
            return [0.0, 0.0];
        }

        // Current frame: local -> world -> clip.
        let curr_world_pos = mat4_mul_point(&self.current_world, local_pos);
        let curr_world = [
            curr_world_pos[0] / curr_world_pos[3],
            curr_world_pos[1] / curr_world_pos[3],
            curr_world_pos[2] / curr_world_pos[3],
        ];
        let curr_clip = mat4_mul_point(view_proj, curr_world);
        let curr_ndc = clip_to_ndc(curr_clip);
        let curr_uv = ndc_to_uv(curr_ndc);

        // Previous frame: local -> prev_world -> prev_clip.
        let prev_world_pos = mat4_mul_point(&self.previous_world, local_pos);
        let prev_world = [
            prev_world_pos[0] / prev_world_pos[3],
            prev_world_pos[1] / prev_world_pos[3],
            prev_world_pos[2] / prev_world_pos[3],
        ];
        let prev_clip = mat4_mul_point(prev_view_proj, prev_world);
        let prev_ndc = clip_to_ndc(prev_clip);
        let prev_uv = ndc_to_uv(prev_ndc);

        [curr_uv[0] - prev_uv[0], curr_uv[1] - prev_uv[1]]
    }
}

// ---------------------------------------------------------------------------
// Object motion manager
// ---------------------------------------------------------------------------

/// Manages per-object motion data for all visible objects.
#[derive(Debug)]
pub struct ObjectMotionManager {
    /// Per-object motion data, indexed by object ID.
    objects: Vec<ObjectMotionData>,
    /// Map from object ID to index.
    id_map: std::collections::HashMap<u64, usize>,
}

impl ObjectMotionManager {
    /// Creates an empty manager.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            id_map: std::collections::HashMap::new(),
        }
    }

    /// Registers or updates an object's transform.
    pub fn update_object(&mut self, object_id: u64, world_transform: [f32; 16]) {
        if let Some(&idx) = self.id_map.get(&object_id) {
            self.objects[idx].advance(world_transform);
        } else {
            let idx = self.objects.len();
            self.objects
                .push(ObjectMotionData::new_spawned(object_id, world_transform));
            self.id_map.insert(object_id, idx);
        }
    }

    /// Gets the motion data for an object.
    pub fn get(&self, object_id: u64) -> Option<&ObjectMotionData> {
        self.id_map
            .get(&object_id)
            .and_then(|&idx| self.objects.get(idx))
    }

    /// Returns the number of tracked objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the manager is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Removes an object from tracking.
    pub fn remove(&mut self, object_id: u64) {
        if let Some(idx) = self.id_map.remove(&object_id) {
            self.objects.swap_remove(idx);
            // Update the swapped object's index.
            if idx < self.objects.len() {
                let swapped_id = self.objects[idx].object_id;
                self.id_map.insert(swapped_id, idx);
            }
        }
    }

    /// Clears all tracked objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.id_map.clear();
    }

    /// Returns all objects that moved this frame.
    pub fn moved_objects(&self) -> Vec<u64> {
        self.objects
            .iter()
            .filter(|o| o.has_moved)
            .map(|o| o.object_id)
            .collect()
    }
}

impl Default for ObjectMotionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Depth-based motion estimation
// ---------------------------------------------------------------------------

/// Estimates motion vectors for static objects using only depth data.
///
/// This is a fallback when per-object transforms are not available.
/// It reprojects each pixel from the current depth buffer to the previous
/// frame using the view-projection delta.
///
/// This is essentially the same as camera motion vectors but can also handle
/// objects that appear/disappear between frames through depth discontinuity
/// detection.
///
/// # Arguments
/// * `curr_depth` — Current frame's depth buffer.
/// * `prev_depth` — Previous frame's depth buffer.
/// * `width`, `height` — Buffer dimensions.
/// * `curr_inv_vp` — Current inverse view-projection.
/// * `prev_vp` — Previous view-projection.
/// * `depth_threshold` — Maximum depth ratio for accepting a match.
///
/// # Returns
/// Motion vector buffer with confidence: `[dU, dV, confidence]` per pixel.
pub fn estimate_motion_from_depth(
    curr_depth: &[f32],
    prev_depth: &[f32],
    width: u32,
    height: u32,
    curr_inv_vp: &[f32; 16],
    prev_vp: &[f32; 16],
    depth_threshold: f32,
) -> Vec<[f32; 3]> {
    let total = (width * height) as usize;
    let mut result = vec![[0.0f32; 3]; total];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let depth = curr_depth[idx];

            if depth >= 0.9999 {
                result[idx] = [0.0, 0.0, 0.0];
                continue;
            }

            // Reconstruct world position.
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            let ndc = [uv[0] * 2.0 - 1.0, 1.0 - uv[1] * 2.0, depth, 1.0];
            let world_clip = mat4_mul_vec4(curr_inv_vp, ndc);

            if world_clip[3].abs() < 1e-7 {
                continue;
            }

            let world_pos = [
                world_clip[0] / world_clip[3],
                world_clip[1] / world_clip[3],
                world_clip[2] / world_clip[3],
            ];

            // Project to previous frame.
            let prev_clip = mat4_mul_point(prev_vp, world_pos);
            let prev_ndc = clip_to_ndc(prev_clip);
            let prev_uv = ndc_to_uv(prev_ndc);

            // Check if the previous UV is in bounds.
            if prev_uv[0] < 0.0 || prev_uv[0] > 1.0 || prev_uv[1] < 0.0 || prev_uv[1] > 1.0 {
                result[idx] = [uv[0] - prev_uv[0], uv[1] - prev_uv[1], 0.0];
                continue;
            }

            // Check depth consistency.
            let prev_px = ((prev_uv[0] * width as f32) as u32).min(width - 1);
            let prev_py = ((prev_uv[1] * height as f32) as u32).min(height - 1);
            let prev_idx = (prev_py * width + prev_px) as usize;

            let prev_d = prev_depth[prev_idx];
            let depth_ratio = if prev_d > 1e-6 {
                (depth / prev_d).max(prev_d / depth)
            } else {
                f32::MAX
            };

            let confidence = if depth_ratio < 1.0 + depth_threshold {
                1.0
            } else {
                ((1.0 + depth_threshold * 2.0 - depth_ratio) / depth_threshold).clamp(0.0, 1.0)
            };

            result[idx] = [uv[0] - prev_uv[0], uv[1] - prev_uv[1], confidence];
        }
    }

    result
}

// ---------------------------------------------------------------------------
// TAA jitter sequences
// ---------------------------------------------------------------------------

/// Halton sequence generator for TAA sub-pixel jitter.
///
/// Produces a low-discrepancy quasi-random sequence that distributes
/// sub-pixel sample positions evenly over time.
pub struct HaltonSequence {
    /// Base for the first dimension.
    pub base_x: u32,
    /// Base for the second dimension.
    pub base_y: u32,
    /// Number of samples in the sequence before it wraps.
    pub length: u32,
}

impl HaltonSequence {
    /// Creates a standard Halton(2,3) sequence with 16 samples.
    pub fn new() -> Self {
        Self {
            base_x: 2,
            base_y: 3,
            length: 16,
        }
    }

    /// Creates a Halton sequence with custom bases and length.
    pub fn custom(base_x: u32, base_y: u32, length: u32) -> Self {
        Self {
            base_x,
            base_y,
            length,
        }
    }

    /// Computes the Halton value for a given index and base.
    fn halton(mut index: u32, base: u32) -> f32 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f32;
        let mut i = index;

        while i > 0 {
            result += f * (i % base) as f32;
            i /= base;
            f /= base as f32;
        }

        result
    }

    /// Returns the jitter offset for a given frame index.
    ///
    /// Returns NDC-space offset [-1/width, 1/width] suitable for adding
    /// to the projection matrix.
    ///
    /// # Arguments
    /// * `frame` — Frame index (wraps automatically).
    /// * `width` — Screen width in pixels.
    /// * `height` — Screen height in pixels.
    ///
    /// # Returns
    /// NDC-space jitter offset `[x, y]`.
    pub fn jitter(&self, frame: u32, width: f32, height: f32) -> [f32; 2] {
        let idx = frame % self.length;
        let x = Self::halton(idx + 1, self.base_x) - 0.5;
        let y = Self::halton(idx + 1, self.base_y) - 0.5;
        [x * 2.0 / width, y * 2.0 / height]
    }

    /// Returns raw Halton values [0, 1] for a given frame.
    pub fn raw(&self, frame: u32) -> [f32; 2] {
        let idx = frame % self.length;
        [
            Self::halton(idx + 1, self.base_x),
            Self::halton(idx + 1, self.base_y),
        ]
    }
}

impl Default for HaltonSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies a TAA jitter offset to a projection matrix.
///
/// Adds a sub-pixel translation to the projection matrix to shift the
/// image by a fraction of a pixel each frame.
///
/// # Arguments
/// * `proj` — Projection matrix (column-major 4x4). Modified in-place.
/// * `jitter_x` — NDC-space horizontal jitter.
/// * `jitter_y` — NDC-space vertical jitter.
pub fn apply_jitter_to_projection(proj: &mut [f32; 16], jitter_x: f32, jitter_y: f32) {
    // Add jitter to the translation components of the projection matrix.
    // For a perspective projection, these affect the off-axis shift.
    proj[8] += jitter_x;  // m[2][0] = col 2, row 0
    proj[9] += jitter_y;  // m[2][1] = col 2, row 1
}

/// Removes jitter from a projection matrix.
pub fn remove_jitter_from_projection(proj: &mut [f32; 16], jitter_x: f32, jitter_y: f32) {
    proj[8] -= jitter_x;
    proj[9] -= jitter_y;
}

// ---------------------------------------------------------------------------
// Motion vector buffer
// ---------------------------------------------------------------------------

/// Complete motion vector buffer with metadata.
#[derive(Debug)]
pub struct MotionVectorBuffer {
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
    /// Motion vector data (dU, dV per pixel).
    data: Vec<[f32; 2]>,
    /// Maximum motion vector length in this frame.
    pub max_velocity: f32,
    /// Average motion vector length.
    pub avg_velocity: f32,
    /// Percentage of pixels with non-zero motion.
    pub motion_coverage: f32,
}

impl MotionVectorBuffer {
    /// Creates an empty buffer.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![[0.0; 2]; total],
            max_velocity: 0.0,
            avg_velocity: 0.0,
            motion_coverage: 0.0,
        }
    }

    /// Creates from pre-computed data.
    pub fn from_data(width: u32, height: u32, data: Vec<[f32; 2]>) -> Self {
        let mut buf = Self {
            width,
            height,
            data,
            max_velocity: 0.0,
            avg_velocity: 0.0,
            motion_coverage: 0.0,
        };
        buf.compute_stats();
        buf
    }

    /// Computes statistics about the motion vectors.
    pub fn compute_stats(&mut self) {
        let mut max_vel = 0.0f32;
        let mut sum_vel = 0.0f32;
        let mut motion_count = 0u32;

        for mv in &self.data {
            let len = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt();
            if len > 1e-6 {
                motion_count += 1;
            }
            max_vel = max_vel.max(len);
            sum_vel += len;
        }

        self.max_velocity = max_vel;
        self.avg_velocity = if !self.data.is_empty() {
            sum_vel / self.data.len() as f32
        } else {
            0.0
        };
        self.motion_coverage = if !self.data.is_empty() {
            motion_count as f32 / self.data.len() as f32
        } else {
            0.0
        };
    }

    /// Samples a motion vector at a pixel.
    pub fn sample(&self, x: u32, y: u32) -> [f32; 2] {
        let idx = (y * self.width + x) as usize;
        self.data.get(idx).copied().unwrap_or([0.0; 2])
    }

    /// Samples with bilinear interpolation.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 2] {
        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;
        let x0 = (fx as u32).min(self.width - 2);
        let y0 = (fy as u32).min(self.height - 2);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let s00 = self.sample(x0, y0);
        let s10 = self.sample(x0 + 1, y0);
        let s01 = self.sample(x0, y0 + 1);
        let s11 = self.sample(x0 + 1, y0 + 1);

        let x0_lerp = [
            s00[0] + (s10[0] - s00[0]) * tx,
            s00[1] + (s10[1] - s00[1]) * tx,
        ];
        let x1_lerp = [
            s01[0] + (s11[0] - s01[0]) * tx,
            s01[1] + (s11[1] - s01[1]) * tx,
        ];

        [
            x0_lerp[0] + (x1_lerp[0] - x0_lerp[0]) * ty,
            x0_lerp[1] + (x1_lerp[1] - x0_lerp[1]) * ty,
        ]
    }

    /// Returns the largest motion vector in a neighbourhood.
    ///
    /// Used by motion blur to find the dominant motion direction.
    pub fn dilated_sample(&self, x: u32, y: u32, radius: u32) -> [f32; 2] {
        let mut best = [0.0f32; 2];
        let mut best_len = 0.0f32;

        let x_min = x.saturating_sub(radius);
        let y_min = y.saturating_sub(radius);
        let x_max = (x + radius).min(self.width - 1);
        let y_max = (y + radius).min(self.height - 1);

        for sy in y_min..=y_max {
            for sx in x_min..=x_max {
                let mv = self.sample(sx, sy);
                let len = mv[0] * mv[0] + mv[1] * mv[1];
                if len > best_len {
                    best_len = len;
                    best = mv;
                }
            }
        }

        best
    }

    /// Returns the raw data for GPU upload.
    pub fn data(&self) -> &[[f32; 2]] {
        &self.data
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<[f32; 2]>()
    }
}

// ---------------------------------------------------------------------------
// WGSL motion vector shader
// ---------------------------------------------------------------------------

/// WGSL fragment shader for camera motion vector generation.
pub const MOTION_VECTORS_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Motion vector generation shader (Genovo Engine)
// -----------------------------------------------------------------------

struct MotionUniforms {
    curr_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    curr_inv_view_proj: mat4x4<f32>,
    jitter_current: vec2<f32>,
    jitter_previous: vec2<f32>,
    screen_size: vec2<f32>,
    max_velocity: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> motion: MotionUniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec2<f32> {
    let depth = textureSample(depth_tex, depth_sampler, input.uv).r;

    if depth >= 0.9999 {
        return vec2<f32>(0.0);
    }

    // Reconstruct NDC position.
    let ndc = vec4<f32>(
        input.uv.x * 2.0 - 1.0,
        (1.0 - input.uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );

    // Reconstruct world position.
    let world_pos = motion.curr_inv_view_proj * ndc;
    let world = world_pos.xyz / world_pos.w;

    // Project to previous frame.
    let prev_clip = motion.prev_view_proj * vec4<f32>(world, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    var velocity = input.uv - prev_uv;

    // Remove jitter.
    let jitter_delta = (motion.jitter_current - motion.jitter_previous) * vec2<f32>(0.5, -0.5);
    velocity -= jitter_delta;

    // Clamp velocity.
    let vel_len = length(velocity);
    if vel_len > motion.max_velocity {
        velocity = velocity * (motion.max_velocity / vel_len);
    }

    return velocity;
}

// Per-object motion vector vertex shader.
struct ObjectMotionUniforms {
    curr_mvp: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> obj_motion: ObjectMotionUniforms;

struct ObjectVertexInput {
    @location(0) position: vec3<f32>,
};

struct ObjectVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) curr_pos: vec4<f32>,
    @location(1) prev_pos: vec4<f32>,
};

@vertex
fn vs_object(@location(0) pos: vec3<f32>) -> ObjectVertexOutput {
    var out: ObjectVertexOutput;
    out.curr_pos = obj_motion.curr_mvp * vec4<f32>(pos, 1.0);
    out.prev_pos = obj_motion.prev_mvp * vec4<f32>(pos, 1.0);
    out.clip_pos = out.curr_pos;
    return out;
}

@fragment
fn fs_object(input: ObjectVertexOutput) -> @location(0) vec2<f32> {
    let curr_ndc = input.curr_pos.xy / input.curr_pos.w;
    let prev_ndc = input.prev_pos.xy / input.prev_pos.w;
    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    return curr_uv - prev_uv;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndc_uv_roundtrip() {
        let uv = [0.3, 0.7];
        let ndc = uv_to_ndc(uv);
        let recovered = ndc_to_uv(ndc);
        assert!((recovered[0] - uv[0]).abs() < 0.01);
        assert!((recovered[1] - uv[1]).abs() < 0.01);
    }

    #[test]
    fn test_identity_no_motion() {
        let vp = identity_matrix();
        let inv_vp = identity_matrix();
        let depth = vec![0.5; 4 * 4];
        let settings = MotionVectorSettings::new();

        let mvs = compute_camera_motion_vectors(&depth, 4, 4, &vp, &vp, &inv_vp, &settings);

        // With identical VP matrices, motion should be zero.
        for mv in &mvs {
            let len = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt();
            assert!(len < 0.01, "Expected zero motion, got {mv:?}");
        }
    }

    #[test]
    fn test_halton_sequence() {
        let halton = HaltonSequence::new();
        let mut samples = Vec::new();

        for i in 0..16 {
            let raw = halton.raw(i);
            assert!(raw[0] >= 0.0 && raw[0] <= 1.0);
            assert!(raw[1] >= 0.0 && raw[1] <= 1.0);
            samples.push(raw);
        }

        // Samples should be unique.
        for (i, a) in samples.iter().enumerate() {
            for (j, b) in samples.iter().enumerate() {
                if i != j {
                    assert!(
                        (a[0] - b[0]).abs() > 0.001 || (a[1] - b[1]).abs() > 0.001,
                        "Halton samples should be unique: {i} vs {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_halton_jitter() {
        let halton = HaltonSequence::new();
        let jitter = halton.jitter(0, 1920.0, 1080.0);
        // Should be small sub-pixel offsets.
        assert!(jitter[0].abs() < 0.002);
        assert!(jitter[1].abs() < 0.002);
    }

    #[test]
    fn test_object_motion_data_static() {
        let motion = ObjectMotionData::new_static(1, identity_matrix());
        assert!(!motion.has_moved);

        let mv = motion.compute_vertex_motion([0.0, 0.0, 0.0], &identity_matrix(), &identity_matrix());
        assert!((mv[0]).abs() < 0.01);
        assert!((mv[1]).abs() < 0.01);
    }

    #[test]
    fn test_object_motion_manager() {
        let mut mgr = ObjectMotionManager::new();
        mgr.update_object(1, identity_matrix());
        assert_eq!(mgr.len(), 1);

        // Update with same transform: no motion.
        mgr.update_object(1, identity_matrix());
        let data = mgr.get(1).unwrap();
        assert!(!data.has_moved);

        // Update with different transform.
        let mut moved = identity_matrix();
        moved[12] = 10.0; // Translate X by 10.
        mgr.update_object(1, moved);
        let data = mgr.get(1).unwrap();
        assert!(data.has_moved);
    }

    #[test]
    fn test_object_motion_manager_remove() {
        let mut mgr = ObjectMotionManager::new();
        mgr.update_object(1, identity_matrix());
        mgr.update_object(2, identity_matrix());
        assert_eq!(mgr.len(), 2);

        mgr.remove(1);
        assert_eq!(mgr.len(), 1);
        assert!(mgr.get(1).is_none());
        assert!(mgr.get(2).is_some());
    }

    #[test]
    fn test_motion_vector_buffer() {
        let data = vec![[0.01, 0.02], [0.0, 0.0], [-0.005, 0.01], [0.0, 0.0]];
        let buf = MotionVectorBuffer::from_data(2, 2, data);

        assert!(buf.max_velocity > 0.0);
        assert!(buf.motion_coverage > 0.0);
    }

    #[test]
    fn test_motion_vector_bilinear() {
        let data = vec![
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
        ];
        let buf = MotionVectorBuffer::from_data(2, 2, data);
        let sampled = buf.sample_bilinear(0.5, 0.5);
        assert!((sampled[0] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_motion_vector_dilated() {
        let mut data = vec![[0.0f32; 2]; 9];
        data[4] = [0.1, 0.05]; // Centre pixel has motion.
        let buf = MotionVectorBuffer::from_data(3, 3, data);

        let dilated = buf.dilated_sample(0, 0, 2);
        // Should find the centre pixel's motion.
        assert!((dilated[0] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_jitter_projection() {
        let mut proj = identity_matrix();
        apply_jitter_to_projection(&mut proj, 0.001, -0.002);
        assert!((proj[8] - 0.001).abs() < 1e-6);
        assert!((proj[9] - -0.002).abs() < 1e-6);

        remove_jitter_from_projection(&mut proj, 0.001, -0.002);
        assert!(proj[8].abs() < 1e-6);
        assert!(proj[9].abs() < 1e-6);
    }

    #[test]
    fn test_settings_jitter_delta() {
        let mut settings = MotionVectorSettings::new();
        settings.set_jitter([0.002, -0.001], [0.001, 0.001]);
        let delta = settings.jitter_delta_uv();
        assert!((delta[0] - 0.0005).abs() < 0.001);
    }

    #[test]
    fn test_depth_motion_estimation() {
        let depth = vec![0.5; 4];
        let prev_depth = vec![0.5; 4];
        let inv_vp = identity_matrix();
        let prev_vp = identity_matrix();

        let result = estimate_motion_from_depth(
            &depth,
            &prev_depth,
            2,
            2,
            &inv_vp,
            &prev_vp,
            0.1,
        );

        assert_eq!(result.len(), 4);
    }
}
