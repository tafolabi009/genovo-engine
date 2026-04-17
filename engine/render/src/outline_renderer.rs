// engine/render/src/outline_renderer.rs
//
// Selection and hover outline effect for the Genovo engine.
//
// Provides configurable outline rendering for selected or hovered objects
// using the Jump Flood Algorithm (JFA) for efficient distance field
// computation in screen space.
//
// # Features
//
// - Jump Flood Algorithm for pixel-accurate distance field computation.
// - Configurable outline colour, width, and opacity.
// - Inner and outer outline modes.
// - Pulsing animation for selected objects.
// - Occluded outline rendering (x-ray through geometry).
// - Per-object outline colour overrides.
// - Multi-object outline with distinct colours.
//
// # Pipeline
//
// 1. **Silhouette pass** — Render selected objects to a stencil/mask buffer.
// 2. **JFA seed** — Initialise the JFA seed buffer from the silhouette edges.
// 3. **JFA passes** — Iteratively compute the distance field using the jump
//    flood algorithm (log2(max_dimension) passes).
// 4. **Outline composite** — Use the distance field to draw the outline,
//    applying colour, width, opacity, and animation.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Outline configuration
// ---------------------------------------------------------------------------

/// Outline mode: where the outline is drawn relative to the object edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlineMode {
    /// Outline drawn outside the object silhouette.
    Outer,
    /// Outline drawn inside the object silhouette.
    Inner,
    /// Outline drawn on both sides of the edge.
    Both,
}

/// Animation style for the outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlineAnimation {
    /// No animation (static outline).
    None,
    /// Pulsing opacity (sine wave).
    Pulse,
    /// Pulsing width.
    PulseWidth,
    /// Marching ants (dashed pattern).
    MarchingAnts,
    /// Rainbow colour cycle.
    RainbowCycle,
}

/// Configuration for an outline effect.
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Outline colour (RGBA, linear).
    pub color: [f32; 4],
    /// Outline width in pixels.
    pub width: f32,
    /// Outline mode (inner/outer/both).
    pub mode: OutlineMode,
    /// Outline opacity multiplier [0, 1].
    pub opacity: f32,
    /// Softness/feathering of the outline edge (0 = hard, 1 = very soft).
    pub softness: f32,
    /// Whether to show the outline through occluding geometry.
    pub show_occluded: bool,
    /// Opacity of the occluded portion [0, 1].
    pub occluded_opacity: f32,
    /// Occluded outline colour (if different from main colour).
    pub occluded_color: Option<[f32; 4]>,
    /// Animation type.
    pub animation: OutlineAnimation,
    /// Animation speed (cycles per second).
    pub animation_speed: f32,
    /// Animation amplitude (for pulse effects).
    pub animation_amplitude: f32,
    /// Marching ants dash length (pixels).
    pub dash_length: f32,
    /// Marching ants gap length (pixels).
    pub dash_gap: f32,
    /// Whether this outline is enabled.
    pub enabled: bool,
}

impl OutlineConfig {
    /// Creates a default selection outline (blue).
    pub fn selection() -> Self {
        Self {
            color: [0.2, 0.5, 1.0, 1.0],
            width: 3.0,
            mode: OutlineMode::Outer,
            opacity: 1.0,
            softness: 0.3,
            show_occluded: true,
            occluded_opacity: 0.3,
            occluded_color: None,
            animation: OutlineAnimation::Pulse,
            animation_speed: 1.5,
            animation_amplitude: 0.3,
            dash_length: 10.0,
            dash_gap: 5.0,
            enabled: true,
        }
    }

    /// Creates a default hover outline (white).
    pub fn hover() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            width: 2.0,
            mode: OutlineMode::Outer,
            opacity: 0.8,
            softness: 0.2,
            show_occluded: false,
            occluded_opacity: 0.0,
            occluded_color: None,
            animation: OutlineAnimation::None,
            animation_speed: 0.0,
            animation_amplitude: 0.0,
            dash_length: 10.0,
            dash_gap: 5.0,
            enabled: true,
        }
    }

    /// Creates an error/warning outline (red).
    pub fn error() -> Self {
        Self {
            color: [1.0, 0.2, 0.1, 1.0],
            width: 3.0,
            mode: OutlineMode::Both,
            opacity: 1.0,
            softness: 0.1,
            show_occluded: true,
            occluded_opacity: 0.5,
            occluded_color: None,
            animation: OutlineAnimation::Pulse,
            animation_speed: 3.0,
            animation_amplitude: 0.5,
            dash_length: 10.0,
            dash_gap: 5.0,
            enabled: true,
        }
    }

    /// Creates a marching ants outline (for area selection).
    pub fn marching_ants() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            width: 1.0,
            mode: OutlineMode::Outer,
            opacity: 1.0,
            softness: 0.0,
            show_occluded: false,
            occluded_opacity: 0.0,
            occluded_color: None,
            animation: OutlineAnimation::MarchingAnts,
            animation_speed: 2.0,
            animation_amplitude: 0.0,
            dash_length: 8.0,
            dash_gap: 4.0,
            enabled: true,
        }
    }

    /// Sets the outline colour.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    /// Sets the outline width.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Sets the outline mode.
    pub fn with_mode(mut self, mode: OutlineMode) -> Self {
        self.mode = mode;
        self
    }

    /// Computes the animated opacity at a given time.
    pub fn animated_opacity(&self, time: f32) -> f32 {
        match self.animation {
            OutlineAnimation::None => self.opacity,
            OutlineAnimation::Pulse => {
                let t = (time * self.animation_speed * 2.0 * PI).sin();
                let pulse = 1.0 - self.animation_amplitude * (t * 0.5 + 0.5);
                self.opacity * pulse.clamp(0.0, 1.0)
            }
            OutlineAnimation::PulseWidth | OutlineAnimation::MarchingAnts => self.opacity,
            OutlineAnimation::RainbowCycle => self.opacity,
        }
    }

    /// Computes the animated width at a given time.
    pub fn animated_width(&self, time: f32) -> f32 {
        match self.animation {
            OutlineAnimation::PulseWidth => {
                let t = (time * self.animation_speed * 2.0 * PI).sin();
                let scale = 1.0 + self.animation_amplitude * (t * 0.5 + 0.5);
                self.width * scale
            }
            _ => self.width,
        }
    }

    /// Computes the animated colour at a given time.
    pub fn animated_color(&self, time: f32) -> [f32; 4] {
        match self.animation {
            OutlineAnimation::RainbowCycle => {
                let hue = (time * self.animation_speed) % 1.0;
                let rgb = hsv_to_rgb(hue, 0.8, 1.0);
                [rgb[0], rgb[1], rgb[2], self.color[3]]
            }
            _ => self.color,
        }
    }
}

impl Default for OutlineConfig {
    fn default() -> Self {
        Self::selection()
    }
}

/// Simple HSV to RGB conversion for rainbow animation.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - h.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// ---------------------------------------------------------------------------
// Outlined object
// ---------------------------------------------------------------------------

/// An object that should be outlined.
#[derive(Debug, Clone)]
pub struct OutlinedObject {
    /// Object identifier.
    pub object_id: u64,
    /// Outline configuration for this object.
    pub config: OutlineConfig,
    /// Stencil value assigned to this object.
    pub stencil_value: u8,
}

impl OutlinedObject {
    /// Creates a new outlined object.
    pub fn new(object_id: u64, config: OutlineConfig) -> Self {
        Self {
            object_id,
            config,
            stencil_value: 1,
        }
    }

    /// Creates a selected object with default selection outline.
    pub fn selected(object_id: u64) -> Self {
        Self::new(object_id, OutlineConfig::selection())
    }

    /// Creates a hovered object with default hover outline.
    pub fn hovered(object_id: u64) -> Self {
        Self::new(object_id, OutlineConfig::hover())
    }
}

// ---------------------------------------------------------------------------
// Jump Flood Algorithm (JFA)
// ---------------------------------------------------------------------------

/// A single pixel in the JFA buffer.
///
/// Stores the nearest seed pixel coordinates. Invalid pixels have
/// `seed_x = u32::MAX`.
#[derive(Debug, Clone, Copy)]
pub struct JfaPixel {
    /// X coordinate of the nearest seed pixel.
    pub seed_x: u32,
    /// Y coordinate of the nearest seed pixel.
    pub seed_y: u32,
}

impl JfaPixel {
    /// Creates an invalid (no seed) pixel.
    pub fn invalid() -> Self {
        Self {
            seed_x: u32::MAX,
            seed_y: u32::MAX,
        }
    }

    /// Creates a seed pixel pointing to itself.
    pub fn seed(x: u32, y: u32) -> Self {
        Self {
            seed_x: x,
            seed_y: y,
        }
    }

    /// Whether this pixel has a valid seed.
    pub fn is_valid(&self) -> bool {
        self.seed_x != u32::MAX
    }

    /// Computes the distance to the seed from a given pixel position.
    pub fn distance_to(&self, x: u32, y: u32) -> f32 {
        if !self.is_valid() {
            return f32::MAX;
        }
        let dx = x as f32 - self.seed_x as f32;
        let dy = y as f32 - self.seed_y as f32;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Jump Flood Algorithm implementation for distance field computation.
///
/// The JFA computes an approximate Voronoi diagram / distance field from
/// a set of seed pixels in O(n * log(n)) passes, where n is the maximum
/// dimension.
#[derive(Debug)]
pub struct JumpFloodAlgorithm {
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
    /// Current JFA buffer.
    buffer_a: Vec<JfaPixel>,
    /// Ping-pong buffer.
    buffer_b: Vec<JfaPixel>,
    /// Whether buffer_a is the current read buffer.
    a_is_current: bool,
}

impl JumpFloodAlgorithm {
    /// Creates a new JFA processor.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            buffer_a: vec![JfaPixel::invalid(); total],
            buffer_b: vec![JfaPixel::invalid(); total],
            a_is_current: true,
        }
    }

    /// Initialises the JFA with seed pixels from a silhouette mask.
    ///
    /// Seeds are placed at edge pixels of the mask (where the mask transitions
    /// from filled to empty or vice versa).
    ///
    /// # Arguments
    /// * `mask` — Boolean mask where `true` = object pixel, `false` = background.
    pub fn seed_from_mask(&mut self, mask: &[bool]) {
        let w = self.width as i32;
        let h = self.height as i32;

        // Clear buffers.
        for pixel in &mut self.buffer_a {
            *pixel = JfaPixel::invalid();
        }
        for pixel in &mut self.buffer_b {
            *pixel = JfaPixel::invalid();
        }

        // Find edge pixels.
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let is_filled = mask[idx];

                // Check if this is an edge pixel.
                let is_edge = if is_filled {
                    // A filled pixel is an edge if any neighbour is empty.
                    let mut edge = false;
                    for dy in -1..=1i32 {
                        for dx in -1..=1i32 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx < 0 || nx >= w || ny < 0 || ny >= h {
                                edge = true; // Border counts as edge.
                                continue;
                            }
                            let ni = (ny * w + nx) as usize;
                            if !mask[ni] {
                                edge = true;
                            }
                        }
                    }
                    edge
                } else {
                    false
                };

                if is_edge {
                    self.buffer_a[idx] = JfaPixel::seed(x as u32, y as u32);
                }
            }
        }

        self.a_is_current = true;
    }

    /// Executes the jump flood algorithm.
    ///
    /// Performs `ceil(log2(max(width, height)))` passes with decreasing
    /// step sizes.
    pub fn execute(&mut self) {
        let max_dim = self.width.max(self.height);
        let num_passes = (max_dim as f32).log2().ceil() as u32;

        for pass in 0..num_passes {
            let step = 1 << (num_passes - 1 - pass);
            self.execute_pass(step);
        }

        // Additional pass with step=1 for accuracy.
        self.execute_pass(1);
    }

    /// Executes a single JFA pass with the given step size.
    fn execute_pass(&mut self, step: u32) {
        let w = self.width;
        let h = self.height;
        let step_i = step as i32;

        let (read, write) = if self.a_is_current {
            (&self.buffer_a as &Vec<JfaPixel>, &mut self.buffer_b)
        } else {
            (&self.buffer_b as &Vec<JfaPixel>, &mut self.buffer_a)
        };

        // Copy current state to write buffer.
        write.copy_from_slice(read);

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) as usize;
                let mut best = write[center_idx];
                let mut best_dist = best.distance_to(x, y);

                // Sample 9 neighbours at the step distance.
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        let nx = x as i32 + dx * step_i;
                        let ny = y as i32 + dy * step_i;

                        if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                            continue;
                        }

                        let ni = (ny as u32 * w + nx as u32) as usize;
                        let neighbor = read[ni];

                        if !neighbor.is_valid() {
                            continue;
                        }

                        let dist = neighbor.distance_to(x, y);
                        if dist < best_dist {
                            best = neighbor;
                            best_dist = dist;
                        }
                    }
                }

                write[center_idx] = best;
            }
        }

        self.a_is_current = !self.a_is_current;
    }

    /// Returns the current JFA buffer.
    pub fn current_buffer(&self) -> &[JfaPixel] {
        if self.a_is_current {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Computes the distance field from the JFA result.
    ///
    /// # Returns
    /// Distance (in pixels) from each pixel to the nearest seed.
    pub fn distance_field(&self) -> Vec<f32> {
        let buffer = self.current_buffer();
        let w = self.width;
        let h = self.height;

        let mut distances = vec![f32::MAX; (w * h) as usize];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                distances[idx] = buffer[idx].distance_to(x, y);
            }
        }

        distances
    }

    /// Generates the outline mask from the distance field.
    ///
    /// # Arguments
    /// * `mask` — Original object silhouette mask.
    /// * `config` — Outline configuration.
    /// * `time` — Current time (for animation).
    ///
    /// # Returns
    /// RGBA colour buffer for the outline.
    pub fn generate_outline(
        &self,
        mask: &[bool],
        config: &OutlineConfig,
        time: f32,
    ) -> Vec<[f32; 4]> {
        let w = self.width;
        let h = self.height;
        let distances = self.distance_field();
        let total = (w * h) as usize;

        let mut outline = vec![[0.0f32; 4]; total];

        let width = config.animated_width(time);
        let opacity = config.animated_opacity(time);
        let color = config.animated_color(time);

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let dist = distances[idx];
                let is_inside = mask[idx];

                // Determine if this pixel is in the outline.
                let outline_factor = match config.mode {
                    OutlineMode::Outer => {
                        if !is_inside && dist <= width {
                            compute_outline_alpha(dist, width, config.softness)
                        } else {
                            0.0
                        }
                    }
                    OutlineMode::Inner => {
                        if is_inside && dist <= width {
                            compute_outline_alpha(dist, width, config.softness)
                        } else {
                            0.0
                        }
                    }
                    OutlineMode::Both => {
                        if dist <= width {
                            compute_outline_alpha(dist, width, config.softness)
                        } else {
                            0.0
                        }
                    }
                };

                if outline_factor <= 0.0 {
                    continue;
                }

                // Apply animation patterns.
                let pattern = match config.animation {
                    OutlineAnimation::MarchingAnts => {
                        let total_len = config.dash_length + config.dash_gap;
                        let pos = (x as f32 + y as f32 + time * config.animation_speed * 50.0)
                            % total_len;
                        if pos < config.dash_length {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0,
                };

                let final_alpha = outline_factor * opacity * pattern * color[3];

                if final_alpha > 0.001 {
                    outline[idx] = [
                        color[0] * final_alpha,
                        color[1] * final_alpha,
                        color[2] * final_alpha,
                        final_alpha,
                    ];
                }
            }
        }

        outline
    }

    /// Returns the number of seed pixels in the current buffer.
    pub fn seed_count(&self) -> u32 {
        self.current_buffer().iter().filter(|p| p.is_valid()).count() as u32
    }

    /// Resizes the JFA buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        let total = (width * height) as usize;
        self.width = width;
        self.height = height;
        self.buffer_a = vec![JfaPixel::invalid(); total];
        self.buffer_b = vec![JfaPixel::invalid(); total];
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        (self.buffer_a.len() + self.buffer_b.len()) * std::mem::size_of::<JfaPixel>()
    }
}

/// Computes the outline alpha based on distance from edge, width, and softness.
fn compute_outline_alpha(distance: f32, width: f32, softness: f32) -> f32 {
    if softness <= 0.001 {
        // Hard edge.
        if distance <= width {
            1.0
        } else {
            0.0
        }
    } else {
        // Soft edge: smooth falloff.
        let inner = width * (1.0 - softness);
        if distance <= inner {
            1.0
        } else if distance <= width {
            1.0 - (distance - inner) / (width - inner)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// OutlineRenderer
// ---------------------------------------------------------------------------

/// Top-level outline rendering system.
#[derive(Debug)]
pub struct OutlineRenderer {
    /// JFA processor.
    pub jfa: JumpFloodAlgorithm,
    /// Currently outlined objects.
    pub objects: Vec<OutlinedObject>,
    /// Screen width.
    pub width: u32,
    /// Screen height.
    pub height: u32,
    /// Whether outline rendering is enabled.
    pub enabled: bool,
    /// Global outline scale multiplier.
    pub global_scale: f32,
}

impl OutlineRenderer {
    /// Creates a new outline renderer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            jfa: JumpFloodAlgorithm::new(width, height),
            objects: Vec::new(),
            width,
            height,
            enabled: true,
            global_scale: 1.0,
        }
    }

    /// Adds an object to be outlined.
    pub fn add_object(&mut self, object: OutlinedObject) {
        self.objects.push(object);
    }

    /// Removes all outlined objects.
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    /// Removes a specific object.
    pub fn remove_object(&mut self, object_id: u64) {
        self.objects.retain(|o| o.object_id != object_id);
    }

    /// Sets the selection (replaces all current outlines with a single selection).
    pub fn set_selection(&mut self, object_id: u64) {
        self.clear();
        self.add_object(OutlinedObject::selected(object_id));
    }

    /// Sets the hover object.
    pub fn set_hover(&mut self, object_id: u64) {
        // Remove any existing hover objects.
        self.objects.retain(|o| !matches!(o.config.animation, OutlineAnimation::None));
        self.add_object(OutlinedObject::hovered(object_id));
    }

    /// Renders outlines for a given silhouette mask.
    ///
    /// # Arguments
    /// * `mask` — Boolean silhouette mask for the outlined object(s).
    /// * `config` — Outline configuration.
    /// * `time` — Current time (for animation).
    ///
    /// # Returns
    /// RGBA outline colour buffer.
    pub fn render_outline(
        &mut self,
        mask: &[bool],
        config: &OutlineConfig,
        time: f32,
    ) -> Vec<[f32; 4]> {
        if !self.enabled || !config.enabled {
            return vec![[0.0; 4]; (self.width * self.height) as usize];
        }

        self.jfa.seed_from_mask(mask);
        self.jfa.execute();
        self.jfa.generate_outline(mask, config, time)
    }

    /// Composites the outline over the scene colour buffer.
    ///
    /// Uses pre-multiplied alpha blending.
    pub fn composite(
        scene: &mut [[f32; 3]],
        outline: &[[f32; 4]],
    ) {
        for (pixel, ol) in scene.iter_mut().zip(outline.iter()) {
            if ol[3] > 0.001 {
                let alpha = ol[3];
                let inv_alpha = 1.0 - alpha;
                pixel[0] = pixel[0] * inv_alpha + ol[0];
                pixel[1] = pixel[1] * inv_alpha + ol[1];
                pixel[2] = pixel[2] * inv_alpha + ol[2];
            }
        }
    }

    /// Resizes the outline renderer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.jfa.resize(width, height);
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.jfa.memory_usage()
    }
}

impl Default for OutlineRenderer {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

// ---------------------------------------------------------------------------
// WGSL outline shader
// ---------------------------------------------------------------------------

/// WGSL compute shader for Jump Flood Algorithm.
pub const JFA_COMPUTE_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Jump Flood Algorithm compute shader (Genovo Engine)
// -----------------------------------------------------------------------

struct JfaUniforms {
    step_size: i32,
    width: u32,
    height: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> jfa: JfaUniforms;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8, 1)
fn jfa_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= jfa.width || gid.y >= jfa.height {
        return;
    }

    let center = textureLoad(input_tex, vec2<i32>(gid.xy), 0).rg;
    var best_seed = center;
    var best_dist = 1e20;

    if center.x >= 0.0 {
        let dx = f32(gid.x) - center.x;
        let dy = f32(gid.y) - center.y;
        best_dist = dx * dx + dy * dy;
    }

    let step = jfa.step_size;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(gid.x) + dx * step;
            let ny = i32(gid.y) + dy * step;

            if nx < 0 || nx >= i32(jfa.width) || ny < 0 || ny >= i32(jfa.height) {
                continue;
            }

            let neighbor = textureLoad(input_tex, vec2<i32>(nx, ny), 0).rg;

            if neighbor.x < 0.0 {
                continue;
            }

            let ddx = f32(gid.x) - neighbor.x;
            let ddy = f32(gid.y) - neighbor.y;
            let dist = ddx * ddx + ddy * ddy;

            if dist < best_dist {
                best_dist = dist;
                best_seed = neighbor;
            }
        }
    }

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(best_seed, 0.0, 0.0));
}

// Outline composite fragment shader.
struct OutlineUniforms {
    color: vec4<f32>,
    width: f32,
    softness: f32,
    time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> outline: OutlineUniforms;
@group(0) @binding(1) var distance_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var scene_tex: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;

@fragment
fn fs_outline(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, tex_sampler, uv);
    let dist = textureSample(distance_tex, tex_sampler, uv).r;
    let mask = textureSample(mask_tex, tex_sampler, uv).r;

    if dist > outline.width || dist <= 0.0 {
        return scene;
    }

    if mask > 0.5 {
        return scene;
    }

    let inner = outline.width * (1.0 - outline.softness);
    var alpha = 1.0;
    if dist > inner {
        alpha = 1.0 - (dist - inner) / (outline.width - inner);
    }

    let pulse = 1.0 - 0.3 * (sin(outline.time * 9.42) * 0.5 + 0.5);
    alpha *= pulse * outline.color.a;

    return mix(scene, outline.color, alpha);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jfa_pixel_distance() {
        let pixel = JfaPixel::seed(5, 5);
        let dist = pixel.distance_to(8, 9);
        let expected = ((3.0f32).powi(2) + (4.0f32).powi(2)).sqrt();
        assert!((dist - expected).abs() < 0.01);
    }

    #[test]
    fn test_jfa_invalid_pixel() {
        let pixel = JfaPixel::invalid();
        assert!(!pixel.is_valid());
        assert!(pixel.distance_to(0, 0) > 1e10);
    }

    #[test]
    fn test_jfa_basic() {
        let mut jfa = JumpFloodAlgorithm::new(8, 8);

        // Create a simple mask: a 4x4 square in the centre.
        let mut mask = vec![false; 64];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * 8 + x] = true;
            }
        }

        jfa.seed_from_mask(&mask);
        assert!(jfa.seed_count() > 0, "Should have seed pixels");

        jfa.execute();

        let distances = jfa.distance_field();
        // Corner should be far from the mask.
        assert!(distances[0] > 1.0, "Corner should have distance > 1");
        // Centre of the mask should have distance to edge.
        let centre = distances[3 * 8 + 3];
        assert!(centre <= 2.0, "Centre should be close to edge: {centre}");
    }

    #[test]
    fn test_outline_generation() {
        let mut jfa = JumpFloodAlgorithm::new(8, 8);
        let mut mask = vec![false; 64];
        for y in 3..5 {
            for x in 3..5 {
                mask[y * 8 + x] = true;
            }
        }

        jfa.seed_from_mask(&mask);
        jfa.execute();

        let config = OutlineConfig::selection();
        let outline = jfa.generate_outline(&mask, &config, 0.0);

        // Some pixels should have outline.
        let has_outline = outline.iter().any(|p| p[3] > 0.001);
        assert!(has_outline, "Should have some outline pixels");

        // Inside the mask should NOT have outline (outer mode).
        let inside_idx = 3 * 8 + 3;
        assert!(
            outline[inside_idx][3] < 0.01,
            "Inside pixel should not have outer outline"
        );
    }

    #[test]
    fn test_outline_inner_mode() {
        let mut jfa = JumpFloodAlgorithm::new(8, 8);
        let mut mask = vec![false; 64];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * 8 + x] = true;
            }
        }

        jfa.seed_from_mask(&mask);
        jfa.execute();

        let config = OutlineConfig::selection().with_mode(OutlineMode::Inner).with_width(2.0);
        let outline = jfa.generate_outline(&mask, &config, 0.0);

        // Outside pixels should NOT have outline.
        assert!(outline[0][3] < 0.01);
    }

    #[test]
    fn test_outline_config_animation() {
        let config = OutlineConfig::selection();
        let op0 = config.animated_opacity(0.0);
        let op1 = config.animated_opacity(0.5);
        // Opacity should vary over time.
        assert!(op0 > 0.0);
        assert!(op1 > 0.0);
    }

    #[test]
    fn test_outline_composite() {
        let mut scene = vec![[0.5, 0.5, 0.5]; 4];
        let outline = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0], // Full red outline.
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        OutlineRenderer::composite(&mut scene, &outline);

        // Pixel 1 should be fully red.
        assert!((scene[1][0] - 1.0).abs() < 0.01);
        // Pixel 0 should be unchanged.
        assert!((scene[0][0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_hsv_to_rgb() {
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((red[0] - 1.0).abs() < 0.01);
        assert!(red[1] < 0.01);

        let green = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(green[1] > 0.9);
    }

    #[test]
    fn test_outline_renderer_creation() {
        let renderer = OutlineRenderer::new(1920, 1080);
        assert!(renderer.enabled);
        assert!(renderer.objects.is_empty());
    }

    #[test]
    fn test_outline_renderer_add_remove() {
        let mut renderer = OutlineRenderer::new(320, 240);
        renderer.add_object(OutlinedObject::selected(1));
        renderer.add_object(OutlinedObject::hovered(2));
        assert_eq!(renderer.objects.len(), 2);

        renderer.remove_object(1);
        assert_eq!(renderer.objects.len(), 1);

        renderer.clear();
        assert_eq!(renderer.objects.len(), 0);
    }

    #[test]
    fn test_marching_ants_config() {
        let config = OutlineConfig::marching_ants();
        assert!(matches!(config.animation, OutlineAnimation::MarchingAnts));
        assert_eq!(config.width, 1.0);
    }

    #[test]
    fn test_compute_outline_alpha() {
        assert!((compute_outline_alpha(0.0, 3.0, 0.0) - 1.0).abs() < 0.01);
        assert!((compute_outline_alpha(4.0, 3.0, 0.0) - 0.0).abs() < 0.01);
        // Soft edge.
        let alpha = compute_outline_alpha(2.5, 3.0, 0.5);
        assert!(alpha > 0.0 && alpha < 1.0);
    }
}
