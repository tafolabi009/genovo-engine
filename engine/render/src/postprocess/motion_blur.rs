// engine/render/src/postprocess/motion_blur.rs
//
// Per-pixel velocity-based motion blur.
//
// Uses the per-pixel velocity buffer (written during the geometry pass from
// current-frame vs. previous-frame transforms) to blur pixels along their
// motion direction. A tile-based optimization groups pixels into tiles,
// computes the maximum velocity per tile, and uses that to limit the work
// in the gather pass.
//
// Also supports camera-only motion blur (computed from the view-projection
// delta between frames) for scenes that don't have per-object velocity.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Motion blur configuration.
#[derive(Debug, Clone)]
pub struct MotionBlurSettings {
    /// Overall intensity multiplier for the blur effect.
    pub intensity: f32,
    /// Number of samples along the velocity direction.
    pub sample_count: u32,
    /// Maximum blur radius in pixels. Limits extreme velocities.
    pub max_blur_radius: f32,
    /// Tile size for the tile-based velocity max pass.
    pub tile_size: u32,
    /// Minimum velocity magnitude (pixels) below which blur is skipped.
    pub velocity_threshold: f32,
    /// Whether to use camera motion blur (from view-projection delta)
    /// instead of per-pixel velocity.
    pub camera_motion_blur: bool,
    /// Shutter angle in degrees (180 = half-frame exposure, 360 = full).
    /// Controls how much motion is captured per frame.
    pub shutter_angle: f32,
    /// Depth-aware blur: reduces blur at depth discontinuities.
    pub depth_aware: bool,
    /// Depth threshold for edge detection (view-space units).
    pub depth_threshold: f32,
    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            sample_count: 16,
            max_blur_radius: 20.0,
            tile_size: 16,
            velocity_threshold: 0.5,
            camera_motion_blur: false,
            shutter_angle: 180.0,
            depth_aware: true,
            depth_threshold: 0.5,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera motion blur helpers
// ---------------------------------------------------------------------------

/// Compute per-pixel velocity from camera motion (view-projection delta).
///
/// For each pixel, reprojects the current position using the previous
/// frame's view-projection matrix and computes the screen-space difference.
///
/// Returns velocity in pixels.
pub fn compute_camera_velocity(
    uv: [f32; 2],
    depth: f32,
    inv_view_proj: &[[f32; 4]; 4],
    prev_view_proj: &[[f32; 4]; 4],
    viewport_width: f32,
    viewport_height: f32,
) -> [f32; 2] {
    // Current NDC.
    let ndc = [uv[0] * 2.0 - 1.0, (1.0 - uv[1]) * 2.0 - 1.0, depth, 1.0];

    // Reconstruct world position.
    let world = mat4_mul_vec4(inv_view_proj, &ndc);
    if world[3].abs() < 1e-10 {
        return [0.0; 2];
    }
    let world_pos = [world[0] / world[3], world[1] / world[3], world[2] / world[3], 1.0];

    // Reproject using previous frame's view-projection.
    let prev_clip = mat4_mul_vec4(prev_view_proj, &world_pos);
    if prev_clip[3].abs() < 1e-10 {
        return [0.0; 2];
    }
    let prev_ndc = [prev_clip[0] / prev_clip[3], prev_clip[1] / prev_clip[3]];
    let prev_uv = [prev_ndc[0] * 0.5 + 0.5, 1.0 - (prev_ndc[1] * 0.5 + 0.5)];

    // Velocity in pixels.
    [
        (uv[0] - prev_uv[0]) * viewport_width,
        (uv[1] - prev_uv[1]) * viewport_height,
    ]
}

fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

// ---------------------------------------------------------------------------
// Tile-based velocity
// ---------------------------------------------------------------------------

/// Compute the maximum velocity in a tile of pixels.
///
/// Returns `(max_vx, max_vy)` in pixels.
pub fn compute_tile_max_velocity(
    velocities: &[[f32; 2]],
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    image_width: u32,
) -> [f32; 2] {
    let mut max_mag_sq = 0.0f32;
    let mut max_vel = [0.0f32; 2];

    let start_x = tile_x * tile_size;
    let start_y = tile_y * tile_size;

    for dy in 0..tile_size {
        for dx in 0..tile_size {
            let px = start_x + dx;
            let py = start_y + dy;
            let idx = (py * image_width + px) as usize;
            if idx >= velocities.len() {
                continue;
            }

            let v = velocities[idx];
            let mag_sq = v[0] * v[0] + v[1] * v[1];
            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
                max_vel = v;
            }
        }
    }

    max_vel
}

/// Compute the neighbor-max of tiles (3x3 neighborhood).
///
/// For each tile, take the maximum velocity among itself and its 8
/// neighbors. This prevents visible seams at tile boundaries.
pub fn compute_neighbor_max(
    tile_velocities: &[[f32; 2]],
    tile_x: u32,
    tile_y: u32,
    tiles_wide: u32,
    tiles_high: u32,
) -> [f32; 2] {
    let mut max_mag_sq = 0.0f32;
    let mut max_vel = [0.0f32; 2];

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let nx = tile_x as i32 + dx;
            let ny = tile_y as i32 + dy;

            if nx < 0 || ny < 0 || nx >= tiles_wide as i32 || ny >= tiles_high as i32 {
                continue;
            }

            let idx = (ny as u32 * tiles_wide + nx as u32) as usize;
            if idx >= tile_velocities.len() {
                continue;
            }

            let v = tile_velocities[idx];
            let mag_sq = v[0] * v[0] + v[1] * v[1];
            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
                max_vel = v;
            }
        }
    }

    max_vel
}

// ---------------------------------------------------------------------------
// Motion blur sampling
// ---------------------------------------------------------------------------

/// Apply motion blur to a single pixel by gathering samples along the
/// velocity direction.
///
/// `center_color` is the color at the pixel.
/// `velocity` is the velocity in pixels.
/// `sample_fn` returns the color and depth at a given UV.
/// Returns the blurred color.
pub fn gather_motion_blur(
    uv: [f32; 2],
    center_color: [f32; 3],
    center_depth: f32,
    velocity: [f32; 2],
    settings: &MotionBlurSettings,
    sample_fn: &dyn Fn(f32, f32) -> ([f32; 3], f32),
    viewport_width: f32,
    viewport_height: f32,
) -> [f32; 3] {
    let vel_mag = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
    if vel_mag < settings.velocity_threshold {
        return center_color;
    }

    // Clamp velocity magnitude to max_blur_radius.
    let clamped_mag = vel_mag.min(settings.max_blur_radius);
    let scale = clamped_mag / vel_mag * settings.intensity;
    let shutter_scale = settings.shutter_angle / 360.0;

    let vel_uv = [
        velocity[0] / viewport_width * scale * shutter_scale,
        velocity[1] / viewport_height * scale * shutter_scale,
    ];

    let mut total_color = center_color;
    let mut total_weight = 1.0f32;

    let half_samples = settings.sample_count / 2;

    for i in 1..=half_samples {
        let t = i as f32 / half_samples as f32;

        for sign in [-1.0f32, 1.0] {
            let sample_uv = [
                uv[0] + vel_uv[0] * t * sign,
                uv[1] + vel_uv[1] * t * sign,
            ];

            // Skip out-of-bounds samples.
            if sample_uv[0] < 0.0
                || sample_uv[0] > 1.0
                || sample_uv[1] < 0.0
                || sample_uv[1] > 1.0
            {
                continue;
            }

            let (sample_color, sample_depth) = sample_fn(sample_uv[0], sample_uv[1]);

            // Depth-aware weighting: reduce contribution at depth edges.
            let mut w = 1.0f32;
            if settings.depth_aware {
                let depth_diff = (center_depth - sample_depth).abs();
                if depth_diff > settings.depth_threshold {
                    w *= (settings.depth_threshold / depth_diff).min(1.0);
                }
            }

            total_color[0] += sample_color[0] * w;
            total_color[1] += sample_color[1] * w;
            total_color[2] += sample_color[2] * w;
            total_weight += w;
        }
    }

    [
        total_color[0] / total_weight,
        total_color[1] / total_weight,
        total_color[2] / total_weight,
    ]
}

// ---------------------------------------------------------------------------
// MotionBlurEffect
// ---------------------------------------------------------------------------

/// Motion blur post-process effect.
pub struct MotionBlurEffect {
    pub settings: MotionBlurSettings,
    /// Tile max velocity texture.
    tile_max_texture: TextureId,
    /// Neighbor max velocity texture.
    neighbor_max_texture: TextureId,
    /// Motion blur result texture.
    result_texture: TextureId,
}

impl MotionBlurEffect {
    pub fn new(settings: MotionBlurSettings) -> Self {
        Self {
            settings,
            tile_max_texture: TextureId(700),
            neighbor_max_texture: TextureId(701),
            result_texture: TextureId(702),
        }
    }

    /// Execute the tile max velocity pass.
    fn execute_tile_max(&self, _input: &PostProcessInput) {
        // Dispatch compute shader that processes each tile to find the
        // maximum velocity magnitude.
    }

    /// Execute the neighbor max pass (3x3 tile neighborhood).
    fn execute_neighbor_max(&self) {
        // Dispatch compute shader that takes the max velocity in a 3x3
        // neighborhood of tiles.
    }

    /// Execute the gather blur pass.
    fn execute_gather(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        // For each pixel:
        // 1. Look up the tile max velocity to decide if blur is needed.
        // 2. If velocity is below threshold, skip.
        // 3. Gather samples along the velocity direction.
        // 4. Apply depth-aware weighting.
    }
}

impl PostProcessEffect for MotionBlurEffect {
    fn name(&self) -> &str {
        "MotionBlur"
    }

    fn execute(&self, input: &PostProcessInput, output: &mut PostProcessOutput) {
        if !self.settings.enabled {
            return;
        }

        self.execute_tile_max(input);
        self.execute_neighbor_max();
        self.execute_gather(input, output);
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        400
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // Reallocate tile textures at new tile dimensions.
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

/// Tile max velocity compute shader.
pub const MOTION_BLUR_TILE_MAX_WGSL: &str = r#"
// Motion Blur — tile max velocity computation

struct TileParams {
    tile_size:       u32,
    image_width:     u32,
    image_height:    u32,
    max_blur_radius: f32,
};

@group(0) @binding(0) var velocity_texture: texture_2d<f32>;
@group(0) @binding(1) var tile_max_output:  texture_storage_2d<rg16float, write>;
@group(0) @binding(2) var<uniform> params:  TileParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_dims = textureDimensions(tile_max_output);
    if gid.x >= tile_dims.x || gid.y >= tile_dims.y {
        return;
    }

    let start = gid.xy * params.tile_size;
    var max_vel = vec2<f32>(0.0);
    var max_mag_sq = 0.0;

    for (var dy = 0u; dy < params.tile_size; dy++) {
        for (var dx = 0u; dx < params.tile_size; dx++) {
            let px = start + vec2<u32>(dx, dy);
            if px.x >= params.image_width || px.y >= params.image_height {
                continue;
            }

            let vel = textureLoad(velocity_texture, px, 0).xy;
            let mag_sq = dot(vel, vel);

            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
                max_vel = vel;
            }
        }
    }

    // Clamp to max blur radius
    let mag = sqrt(max_mag_sq);
    if mag > params.max_blur_radius {
        max_vel *= params.max_blur_radius / mag;
    }

    textureStore(tile_max_output, gid.xy, vec4<f32>(max_vel, 0.0, 0.0));
}
"#;

/// Neighbor max velocity compute shader.
pub const MOTION_BLUR_NEIGHBOR_MAX_WGSL: &str = r#"
// Motion Blur — neighbor max (3x3 tile neighborhood)

@group(0) @binding(0) var tile_max_texture:      texture_2d<f32>;
@group(0) @binding(1) var neighbor_max_output:    texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(neighbor_max_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    var max_vel = vec2<f32>(0.0);
    var max_mag_sq = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let coord = vec2<i32>(gid.xy) + vec2<i32>(dx, dy);
            if coord.x < 0 || coord.y < 0 ||
               coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
                continue;
            }

            let vel = textureLoad(tile_max_texture, vec2<u32>(coord), 0).xy;
            let mag_sq = dot(vel, vel);

            if mag_sq > max_mag_sq {
                max_mag_sq = mag_sq;
                max_vel = vel;
            }
        }
    }

    textureStore(neighbor_max_output, gid.xy, vec4<f32>(max_vel, 0.0, 0.0));
}
"#;

/// Motion blur gather compute shader.
pub const MOTION_BLUR_GATHER_WGSL: &str = r#"
// Motion Blur — per-pixel gather

struct GatherParams {
    sample_count:      u32,
    max_blur_radius:   f32,
    velocity_threshold: f32,
    intensity:         f32,
    shutter_scale:     f32,     // shutter_angle / 360
    depth_aware:       u32,
    depth_threshold:   f32,
    inv_width:         f32,
    inv_height:        f32,
    tile_size:         u32,
    _pad0:             f32,
    _pad1:             f32,
};

@group(0) @binding(0)  var color_texture:    texture_2d<f32>;
@group(0) @binding(1)  var depth_texture:    texture_2d<f32>;
@group(0) @binding(2)  var velocity_texture: texture_2d<f32>;
@group(0) @binding(3)  var neighbor_max:     texture_2d<f32>;
@group(0) @binding(4)  var dst_texture:      texture_storage_2d<rgba16float, write>;
@group(0) @binding(5)  var tex_sampler:      sampler;
@group(0) @binding(6)  var<uniform> params:  GatherParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // Check tile max — early out if no motion in this tile
    let tile_coord = gid.xy / params.tile_size;
    let tile_vel = textureLoad(neighbor_max, tile_coord, 0).xy;
    let tile_mag = length(tile_vel);

    if tile_mag < params.velocity_threshold {
        let color = textureSampleLevel(color_texture, tex_sampler, uv, 0.0);
        textureStore(dst_texture, gid.xy, color);
        return;
    }

    // Per-pixel velocity
    let velocity = textureSampleLevel(velocity_texture, tex_sampler, uv, 0.0).xy;
    let vel_mag = length(velocity);

    if vel_mag < params.velocity_threshold {
        let color = textureSampleLevel(color_texture, tex_sampler, uv, 0.0);
        textureStore(dst_texture, gid.xy, color);
        return;
    }

    // Clamp and scale velocity
    let clamped_mag = min(vel_mag, params.max_blur_radius);
    let scale = clamped_mag / vel_mag * params.intensity * params.shutter_scale;
    let vel_uv = velocity * vec2<f32>(params.inv_width, params.inv_height) * scale;

    let center_color = textureSampleLevel(color_texture, tex_sampler, uv, 0.0).rgb;
    let center_depth = textureSampleLevel(depth_texture, tex_sampler, uv, 0.0).r;

    var total_color = center_color;
    var total_weight = 1.0;

    let half_samples = params.sample_count / 2u;

    for (var i = 1u; i <= half_samples; i++) {
        let t = f32(i) / f32(half_samples);

        for (var sign = -1.0; sign <= 1.0; sign += 2.0) {
            let sample_uv = uv + vel_uv * t * sign;

            if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
                continue;
            }

            let sample_color = textureSampleLevel(color_texture, tex_sampler, sample_uv, 0.0).rgb;

            var w = 1.0;
            if params.depth_aware != 0u {
                let sample_depth = textureSampleLevel(depth_texture, tex_sampler, sample_uv, 0.0).r;
                let depth_diff = abs(center_depth - sample_depth);
                if depth_diff > params.depth_threshold {
                    w *= min(params.depth_threshold / depth_diff, 1.0);
                }
            }

            total_color += sample_color * w;
            total_weight += w;
        }
    }

    let result = total_color / total_weight;
    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

/// Camera motion blur velocity computation shader.
pub const CAMERA_VELOCITY_WGSL: &str = r#"
// Camera Motion Blur — velocity from view-projection delta

struct CameraVelocityParams {
    inv_view_proj:   mat4x4<f32>,
    prev_view_proj:  mat4x4<f32>,
    viewport_width:  f32,
    viewport_height: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var depth_texture:    texture_2d<f32>;
@group(0) @binding(1) var velocity_output:  texture_storage_2d<rg16float, write>;
@group(0) @binding(2) var depth_sampler:    sampler;
@group(0) @binding(3) var<uniform> params:  CameraVelocityParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(velocity_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let depth = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0).r;

    // Current clip space position
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);

    // Reconstruct world position
    let world = params.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;

    // Reproject using previous view-projection
    let prev_clip = params.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));

    // Velocity in pixels
    let velocity = (uv - prev_uv) * vec2<f32>(params.viewport_width, params.viewport_height);

    textureStore(velocity_output, gid.xy, vec4<f32>(velocity, 0.0, 0.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_velocity_static() {
        // When current and previous VP are the same, velocity should be ~0.
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let vel = compute_camera_velocity([0.5, 0.5], 0.5, &identity, &identity, 1920.0, 1080.0);
        assert!(vel[0].abs() < 1e-3 && vel[1].abs() < 1e-3);
    }

    #[test]
    fn test_tile_max_velocity() {
        let velocities = vec![
            [1.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.0],
            [3.0, 0.0],
        ];

        let max_vel = compute_tile_max_velocity(&velocities, 0, 0, 2, 2);
        assert!((max_vel[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_neighbor_max() {
        let tiles = vec![
            [1.0, 0.0],
            [5.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.0],
        ];

        let max_vel = compute_neighbor_max(&tiles, 0, 0, 2, 2);
        assert!((max_vel[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_below_threshold() {
        let settings = MotionBlurSettings {
            velocity_threshold: 1.0,
            ..Default::default()
        };

        let color = [0.5, 0.3, 0.1];
        // Velocity below threshold -> should return center color unchanged.
        let result = gather_motion_blur(
            [0.5, 0.5],
            color,
            0.5,
            [0.1, 0.0], // below threshold
            &settings,
            &|_, _| ([1.0, 1.0, 1.0], 0.5),
            1920.0,
            1080.0,
        );

        assert!((result[0] - color[0]).abs() < 1e-5);
    }

    #[test]
    fn test_motion_blur_effect_interface() {
        let effect = MotionBlurEffect::new(MotionBlurSettings::default());
        assert_eq!(effect.name(), "MotionBlur");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 400);
    }

    #[test]
    fn test_shutter_angle_scaling() {
        let full_shutter = MotionBlurSettings {
            shutter_angle: 360.0,
            ..Default::default()
        };
        let half_shutter = MotionBlurSettings {
            shutter_angle: 180.0,
            ..Default::default()
        };

        // Full shutter should produce twice the effective blur.
        let full_scale = full_shutter.shutter_angle / 360.0;
        let half_scale = half_shutter.shutter_angle / 360.0;
        assert!((full_scale / half_scale - 2.0).abs() < 1e-5);
    }
}
