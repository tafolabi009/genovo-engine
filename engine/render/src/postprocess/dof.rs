// engine/render/src/postprocess/dof.rs
//
// Depth of Field post-process effect.
//
// Simulates the optical phenomenon where objects outside the focal plane
// appear blurred. The blur amount is determined by the Circle of Confusion
// (CoC), which depends on the depth of each pixel relative to the focus
// distance, the focal length, and the aperture (f-stop).
//
// The implementation supports:
//   - Physically-based CoC computation from lens parameters.
//   - Separate near-field and far-field processing (different blur
//     behaviour for objects in front of / behind the focal plane).
//   - Gather-based bokeh: samples are collected in a disk pattern whose
//     shape is derived from the aperture blade count.
//   - Smooth alpha-based transitions at the in-focus / out-of-focus
//     boundary.
//   - A tilt-shift mode for fake miniature effects.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Depth of Field configuration.
#[derive(Debug, Clone)]
pub struct DOFSettings {
    /// Distance to the focal plane (world units).
    pub focus_distance: f32,
    /// Focal length of the virtual lens (mm). Typical values: 24–200.
    pub focal_length: f32,
    /// Aperture f-stop (e.g., 1.4, 2.8, 5.6, 11). Smaller = shallower DOF.
    pub aperture: f32,
    /// Number of aperture blades (affects bokeh shape). 0 = perfect circle.
    pub blade_count: u32,
    /// Rotation of the aperture blades (radians).
    pub blade_rotation: f32,
    /// Maximum blur radius in pixels.
    pub max_blur_radius: f32,
    /// Number of samples for the gather pass.
    pub sample_count: u32,
    /// Bokeh brightness boost (increases intensity of bright bokeh disks).
    pub bokeh_brightness: f32,
    /// Enable near-field blur (objects closer than focus distance).
    pub near_field: bool,
    /// Enable far-field blur (objects farther than focus distance).
    pub far_field: bool,
    /// Transition smoothness between focused and unfocused (larger = softer).
    pub focus_transition: f32,
    /// Whether tilt-shift mode is enabled.
    pub tilt_shift: bool,
    /// Tilt-shift center (normalized Y coordinate, 0.0 = top, 1.0 = bottom).
    pub tilt_shift_center: f32,
    /// Tilt-shift blur width (normalized, fraction of screen height).
    pub tilt_shift_width: f32,
    /// Tilt-shift angle (radians, 0 = horizontal band).
    pub tilt_shift_angle: f32,
    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for DOFSettings {
    fn default() -> Self {
        Self {
            focus_distance: 10.0,
            focal_length: 50.0,
            aperture: 2.8,
            blade_count: 6,
            blade_rotation: 0.0,
            max_blur_radius: 15.0,
            sample_count: 49, // 7x7 ring pattern
            bokeh_brightness: 1.0,
            near_field: true,
            far_field: true,
            focus_transition: 1.0,
            tilt_shift: false,
            tilt_shift_center: 0.5,
            tilt_shift_width: 0.15,
            tilt_shift_angle: 0.0,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Circle of Confusion
// ---------------------------------------------------------------------------

/// Compute the Circle of Confusion (CoC) diameter for a pixel at a given
/// depth.
///
/// Based on the thin-lens equation:
///   CoC = |focal_length^2 * (depth - focus_distance)| /
///         (aperture * depth * (focus_distance - focal_length))
///
/// Returns the CoC in millimeters (sensor space). To convert to pixels,
/// divide by sensor width and multiply by image width.
///
/// A positive CoC indicates far-field blur, negative indicates near-field.
pub fn compute_coc(depth: f32, focus_distance: f32, focal_length: f32, aperture: f32) -> f32 {
    // Convert focal length from mm to the same units as depth (assume
    // depth is in meters, focal_length in mm).
    let f = focal_length * 0.001; // mm -> m

    let numerator = f * f * (depth - focus_distance);
    let denominator = aperture * depth * (focus_distance - f);

    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    numerator / denominator
}

/// Compute CoC and convert to pixel radius.
///
/// `sensor_width` is the sensor width in meters (e.g., 0.036 for 36mm
/// full-frame). `image_width` is the render target width in pixels.
pub fn compute_coc_pixels(
    depth: f32,
    settings: &DOFSettings,
    sensor_width: f32,
    image_width: f32,
) -> f32 {
    let coc_m = compute_coc(depth, settings.focus_distance, settings.focal_length, settings.aperture);
    let coc_pixels = (coc_m / sensor_width) * image_width;
    coc_pixels.clamp(-settings.max_blur_radius, settings.max_blur_radius)
}

/// Compute a tilt-shift CoC based on screen position rather than depth.
pub fn compute_tilt_shift_coc(
    uv_y: f32,
    center: f32,
    width: f32,
    max_radius: f32,
) -> f32 {
    let dist = (uv_y - center).abs();
    let t = ((dist - width * 0.5) / (width * 0.5)).clamp(0.0, 1.0);
    // Smooth ramp from 0 (in focus band) to max_radius (outside band).
    let smooth = t * t * (3.0 - 2.0 * t);
    smooth * max_radius
}

// ---------------------------------------------------------------------------
// Bokeh disk sampling
// ---------------------------------------------------------------------------

/// Generate sample offsets for a disk-shaped bokeh kernel.
///
/// If `blade_count` is 0, generates a perfect circle. Otherwise, generates
/// a polygon with the given number of sides (aperture blades).
///
/// Returns sample offsets as `(x, y)` pairs normalized to unit radius.
pub fn generate_bokeh_kernel(sample_count: u32, blade_count: u32, rotation: f32) -> Vec<[f32; 2]> {
    let mut samples = Vec::with_capacity(sample_count as usize);

    if sample_count == 0 {
        return samples;
    }

    // Use a Fibonacci spiral for even disk coverage.
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());

    for i in 0..sample_count {
        let r = ((i as f32 + 0.5) / sample_count as f32).sqrt();
        let theta = i as f32 * golden_angle + rotation;

        let mut x = r * theta.cos();
        let mut y = r * theta.sin();

        // If blade_count > 0, reshape the disk into a polygon.
        if blade_count >= 3 {
            let polygon_radius = polygon_shape(theta - rotation, blade_count);
            x *= polygon_radius;
            y *= polygon_radius;
        }

        samples.push([x, y]);
    }

    samples
}

/// Compute a radial scale factor that reshapes a circle into a regular
/// polygon with the given number of sides.
///
/// `angle` is the angle of the sample in radians.
fn polygon_shape(angle: f32, sides: u32) -> f32 {
    let sector_angle = std::f32::consts::TAU / sides as f32;
    let half_sector = sector_angle * 0.5;

    // Angle within the current sector.
    let sector_local = ((angle % sector_angle) + sector_angle) % sector_angle;
    let centered = (sector_local - half_sector).abs();

    // cos(half_sector) / cos(centered) maps the circle to polygon edges.
    half_sector.cos() / centered.cos()
}

// ---------------------------------------------------------------------------
// Separate near/far fields
// ---------------------------------------------------------------------------

/// Classify a CoC value into near, in-focus, or far field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusField {
    Near,
    InFocus,
    Far,
}

/// Classify a pixel's focus field based on its CoC.
pub fn classify_focus_field(coc: f32, threshold: f32) -> FocusField {
    if coc < -threshold {
        FocusField::Near
    } else if coc > threshold {
        FocusField::Far
    } else {
        FocusField::InFocus
    }
}

/// Compute alpha for near/far field blending.
///
/// Provides a smooth transition at the boundary between in-focus and
/// out-of-focus regions.
pub fn focus_blend_alpha(coc: f32, transition_width: f32) -> f32 {
    let abs_coc = coc.abs();
    if abs_coc <= 0.5 {
        return 0.0; // Fully in focus.
    }
    let t = ((abs_coc - 0.5) / transition_width.max(0.01)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t) // smoothstep
}

// ---------------------------------------------------------------------------
// DOFEffect
// ---------------------------------------------------------------------------

/// Depth of Field post-process effect.
pub struct DOFEffect {
    pub settings: DOFSettings,
    /// Pre-generated bokeh kernel.
    bokeh_kernel: Vec<[f32; 2]>,
    /// CoC texture (R channel = signed CoC in pixels).
    coc_texture: TextureId,
    /// Near-field blur result.
    near_field_texture: TextureId,
    /// Far-field blur result.
    far_field_texture: TextureId,
    /// Composite result.
    composite_texture: TextureId,
}

impl DOFEffect {
    pub fn new(settings: DOFSettings) -> Self {
        let bokeh_kernel = generate_bokeh_kernel(
            settings.sample_count,
            settings.blade_count,
            settings.blade_rotation,
        );
        Self {
            settings,
            bokeh_kernel,
            coc_texture: TextureId(600),
            near_field_texture: TextureId(601),
            far_field_texture: TextureId(602),
            composite_texture: TextureId(603),
        }
    }

    /// Regenerate the bokeh kernel (e.g., when blade_count or sample_count
    /// changes).
    pub fn regenerate_kernel(&mut self) {
        self.bokeh_kernel = generate_bokeh_kernel(
            self.settings.sample_count,
            self.settings.blade_count,
            self.settings.blade_rotation,
        );
    }

    /// Execute the CoC computation pass.
    fn execute_coc_pass(&self, _input: &PostProcessInput) {
        // Dispatch compute shader that reads the depth buffer and writes
        // the signed CoC for each pixel.
    }

    /// Execute the gather blur for the near field.
    fn execute_near_field(&self) {
        if !self.settings.near_field {
            return;
        }
        // Gather pass using bokeh kernel with only negative CoC pixels.
        // Near-field is trickier because objects in front can bleed into
        // the focused region — we dilate the near CoC beforehand.
    }

    /// Execute the gather blur for the far field.
    fn execute_far_field(&self) {
        if !self.settings.far_field {
            return;
        }
        // Gather pass using bokeh kernel with only positive CoC pixels.
    }

    /// Composite near, far, and in-focus regions together.
    fn execute_composite(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        // Alpha-blend near-field over the scene, then far-field behind.
        // Use the CoC magnitude to determine the blend weight.
    }
}

impl PostProcessEffect for DOFEffect {
    fn name(&self) -> &str {
        "DepthOfField"
    }

    fn execute(&self, input: &PostProcessInput, output: &mut PostProcessOutput) {
        if !self.settings.enabled {
            return;
        }

        self.execute_coc_pass(input);
        self.execute_near_field();
        self.execute_far_field();
        self.execute_composite(input, output);
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        300
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // Reallocate CoC and blur textures.
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

/// CoC computation compute shader.
pub const DOF_COC_WGSL: &str = r#"
// Depth of Field — Circle of Confusion computation

struct DOFParams {
    focus_distance: f32,
    focal_length:   f32,
    aperture:       f32,
    sensor_width:   f32,
    image_width:    f32,
    max_blur_radius: f32,
    near_plane:     f32,
    far_plane:      f32,
    tilt_shift:     u32,
    tilt_center:    f32,
    tilt_width:     f32,
    _pad:           f32,
};

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var coc_output:    texture_storage_2d<r16float, write>;
@group(0) @binding(2) var depth_sampler: sampler;
@group(0) @binding(3) var<uniform> params: DOFParams;

fn linearize_depth(d: f32) -> f32 {
    return params.near_plane * params.far_plane /
           (params.far_plane - d * (params.far_plane - params.near_plane));
}

fn compute_coc(depth: f32) -> f32 {
    let f = params.focal_length * 0.001;  // mm -> m
    let s = params.focus_distance;
    let a = params.aperture;

    let numerator = f * f * (depth - s);
    let denominator = a * depth * (s - f);

    if abs(denominator) < 1e-10 {
        return 0.0;
    }

    let coc_m = numerator / denominator;
    let coc_px = (coc_m / params.sensor_width) * params.image_width;
    return clamp(coc_px, -params.max_blur_radius, params.max_blur_radius);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(coc_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    var coc: f32;
    if params.tilt_shift != 0u {
        // Tilt-shift: CoC based on screen position
        let dist = abs(uv.y - params.tilt_center);
        let half_width = params.tilt_width * 0.5;
        let t = clamp((dist - half_width) / half_width, 0.0, 1.0);
        coc = smoothstep(0.0, 1.0, t) * params.max_blur_radius;
    } else {
        // Physical CoC from depth
        let raw_depth = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0).r;
        let linear_depth = linearize_depth(raw_depth);
        coc = compute_coc(linear_depth);
    }

    textureStore(coc_output, gid.xy, vec4<f32>(coc, 0.0, 0.0, 0.0));
}
"#;

/// Gather-based bokeh blur compute shader.
pub const DOF_GATHER_WGSL: &str = r#"
// Depth of Field — gather-based bokeh blur

struct GatherParams {
    sample_count:    u32,
    max_blur_radius: f32,
    bokeh_brightness: f32,
    is_near_field:   u32,     // 0 = far field, 1 = near field
    inv_width:       f32,
    inv_height:      f32,
    _pad0:           f32,
    _pad1:           f32,
};

struct BokehKernel {
    offsets: array<vec4<f32>, 64>,  // xy = offset, zw = unused
};

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var coc_texture:   texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var tex_sampler:   sampler;
@group(0) @binding(4) var<uniform> params: GatherParams;
@group(0) @binding(5) var<uniform> kernel: BokehKernel;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(params.inv_width, params.inv_height);

    let center_coc = textureSampleLevel(coc_texture, tex_sampler, uv, 0.0).r;

    // Determine the blur radius for this pixel.
    var blur_radius: f32;
    if params.is_near_field != 0u {
        // Near field: use absolute negative CoC
        blur_radius = max(-center_coc, 0.0);
    } else {
        // Far field: use positive CoC
        blur_radius = max(center_coc, 0.0);
    }

    if blur_radius < 0.5 {
        // Pixel is in focus — just copy the center color.
        let color = textureSampleLevel(color_texture, tex_sampler, uv, 0.0);
        textureStore(dst_texture, gid.xy, color);
        return;
    }

    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < params.sample_count; i++) {
        let offset = kernel.offsets[i].xy * blur_radius * texel;
        let sample_uv = uv + offset;

        let sample_color = textureSampleLevel(color_texture, tex_sampler, sample_uv, 0.0).rgb;
        let sample_coc = textureSampleLevel(coc_texture, tex_sampler, sample_uv, 0.0).r;

        // Weight: only include samples that are also blurred.
        var w: f32;
        if params.is_near_field != 0u {
            // Near field: use max(abs(sample_coc), abs(center_coc)) to
            // allow near-field to bleed into focused regions.
            let sample_blur = max(-sample_coc, 0.0);
            w = max(sample_blur, blur_radius) / blur_radius;
        } else {
            // Far field: only blur if sample is also far-blurred.
            let sample_blur = max(sample_coc, 0.0);
            w = min(sample_blur, blur_radius) / blur_radius;
        }

        // Bokeh brightness: weight bright samples more.
        let brightness = 1.0 + luminance(sample_color) * params.bokeh_brightness;
        w *= brightness;

        total_color += sample_color * w;
        total_weight += w;
    }

    if total_weight > 0.0 {
        total_color /= total_weight;
    }

    // Alpha stores the blur amount for compositing.
    let alpha = smoothstep(0.0, 2.0, blur_radius);
    textureStore(dst_texture, gid.xy, vec4<f32>(total_color, alpha));
}
"#;

/// DOF composite shader — blends near, far, and in-focus.
pub const DOF_COMPOSITE_WGSL: &str = r#"
// Depth of Field — final composite

@group(0) @binding(0) var scene_texture:     texture_2d<f32>;
@group(0) @binding(1) var near_field_texture: texture_2d<f32>;
@group(0) @binding(2) var far_field_texture:  texture_2d<f32>;
@group(0) @binding(3) var coc_texture:        texture_2d<f32>;
@group(0) @binding(4) var dst_texture:        texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var tex_sampler:        sampler;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let scene = textureSampleLevel(scene_texture, tex_sampler, uv, 0.0);
    let near = textureSampleLevel(near_field_texture, tex_sampler, uv, 0.0);
    let far = textureSampleLevel(far_field_texture, tex_sampler, uv, 0.0);
    let coc = textureSampleLevel(coc_texture, tex_sampler, uv, 0.0).r;

    // Start with scene color
    var result = scene.rgb;

    // Blend far field (behind focus)
    if coc > 0.5 {
        let far_alpha = far.a;
        result = mix(result, far.rgb, far_alpha);
    }

    // Blend near field (in front of focus) — painted on top
    if coc < -0.5 {
        let near_alpha = near.a;
        result = mix(result, near.rgb, near_alpha);
    }

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coc_at_focus() {
        // At the focus distance, CoC should be ~0.
        let coc = compute_coc(10.0, 10.0, 50.0, 2.8);
        assert!(
            coc.abs() < 1e-4,
            "CoC at focus distance should be ~0, got {coc}"
        );
    }

    #[test]
    fn test_coc_sign() {
        // Far field (depth > focus) should be positive.
        let far_coc = compute_coc(20.0, 10.0, 50.0, 2.8);
        assert!(far_coc > 0.0, "Far field CoC should be positive");

        // Near field (depth < focus) should be negative.
        let near_coc = compute_coc(5.0, 10.0, 50.0, 2.8);
        assert!(near_coc < 0.0, "Near field CoC should be negative");
    }

    #[test]
    fn test_coc_aperture_effect() {
        // Wider aperture (lower f-stop) -> larger CoC.
        let coc_wide = compute_coc(20.0, 10.0, 50.0, 1.4).abs();
        let coc_narrow = compute_coc(20.0, 10.0, 50.0, 11.0).abs();
        assert!(
            coc_wide > coc_narrow,
            "Wider aperture should produce larger CoC"
        );
    }

    #[test]
    fn test_bokeh_kernel_size() {
        let kernel = generate_bokeh_kernel(49, 6, 0.0);
        assert_eq!(kernel.len(), 49);
    }

    #[test]
    fn test_bokeh_kernel_within_unit_disk() {
        let kernel = generate_bokeh_kernel(64, 0, 0.0);
        for sample in &kernel {
            let r = (sample[0] * sample[0] + sample[1] * sample[1]).sqrt();
            assert!(r <= 1.01, "Sample outside unit disk: r={r}");
        }
    }

    #[test]
    fn test_polygon_shape() {
        // At a vertex of a hexagon, the radius should be 1 / cos(pi/6) ~= 1.1547.
        let r = polygon_shape(0.0, 6);
        let expected = (std::f32::consts::PI / 6.0).cos() / (0.0f32).cos();
        assert!(
            (r - expected).abs() < 1e-4,
            "polygon_shape(0, 6) = {r}, expected {expected}"
        );
    }

    #[test]
    fn test_tilt_shift_coc() {
        // At center -> 0.
        let coc_center = compute_tilt_shift_coc(0.5, 0.5, 0.2, 10.0);
        assert!(coc_center.abs() < 1e-3);

        // Far from center -> max.
        let coc_edge = compute_tilt_shift_coc(0.0, 0.5, 0.2, 10.0);
        assert!(coc_edge > 5.0);
    }

    #[test]
    fn test_focus_blend_alpha() {
        // Small CoC -> fully in focus.
        assert!(focus_blend_alpha(0.2, 1.0).abs() < 1e-3);

        // Large CoC -> fully blurred.
        let alpha = focus_blend_alpha(5.0, 1.0);
        assert!(alpha > 0.9);
    }

    #[test]
    fn test_focus_field_classification() {
        assert_eq!(classify_focus_field(-2.0, 0.5), FocusField::Near);
        assert_eq!(classify_focus_field(0.0, 0.5), FocusField::InFocus);
        assert_eq!(classify_focus_field(2.0, 0.5), FocusField::Far);
    }

    #[test]
    fn test_dof_effect_interface() {
        let effect = DOFEffect::new(DOFSettings::default());
        assert_eq!(effect.name(), "DepthOfField");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 300);
    }
}
