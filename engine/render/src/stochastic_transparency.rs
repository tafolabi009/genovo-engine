// engine/render/src/stochastic_transparency.rs
//
// Order-independent transparency (OIT) techniques for the Genovo engine.
//
// Transparent rendering is one of the hardest problems in real-time
// graphics because correct blending requires processing fragments in
// back-to-front order. This module implements several OIT approaches:
//
// - **Stochastic transparency** — Probabilistic alpha testing: each fragment
//   passes or fails a random threshold based on its alpha, producing a noisy
//   but order-independent result that converges with enough samples/frames.
//
// - **Weighted Blended OIT (WBOIT)** — McGuire/Bavoil 2013: accumulate
//   pre-multiplied colour and a weight function into two render targets,
//   then resolve in a full-screen pass. Fast and simple, but approximate.
//
// - **Depth peeling** — Renders the scene multiple times, each time
//   discarding fragments closer than the previous peel. Exact but slow.
//   Dual depth peeling halves the number of passes.
//
// - **Per-pixel linked list** — Conceptual implementation using a linked list
//   of fragments stored in a GPU buffer. Each pixel can store N fragments
//   and sort them for exact compositing.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// OIT Method selection
// ---------------------------------------------------------------------------

/// Order-independent transparency method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OitMethod {
    /// Stochastic transparency.
    Stochastic,
    /// Weighted Blended OIT (McGuire/Bavoil).
    WeightedBlended,
    /// Depth peeling (multi-pass).
    DepthPeeling,
    /// Dual depth peeling.
    DualDepthPeeling,
    /// Per-pixel linked list.
    LinkedList,
    /// Simple sorted (CPU sort by centre distance, no OIT).
    Sorted,
}

/// Configuration for OIT rendering.
#[derive(Debug, Clone)]
pub struct OitSettings {
    /// OIT method to use.
    pub method: OitMethod,
    /// Maximum number of depth-peeling passes.
    pub max_peel_passes: u32,
    /// Maximum fragments per pixel for linked list.
    pub max_fragments_per_pixel: u32,
    /// Stochastic: number of samples for convergence.
    pub stochastic_samples: u32,
    /// Stochastic: whether to use temporal accumulation.
    pub stochastic_temporal: bool,
    /// Stochastic: temporal blend factor.
    pub stochastic_blend: f32,
    /// WBOIT: weight function exponent.
    pub weight_exponent: f32,
    /// WBOIT: near plane distance (for depth weight).
    pub near_plane: f32,
    /// WBOIT: far plane distance.
    pub far_plane: f32,
    /// Whether to render back-faces for transparent objects.
    pub render_back_faces: bool,
    /// Whether to use alpha-to-coverage.
    pub alpha_to_coverage: bool,
}

impl OitSettings {
    /// Creates default settings using WBOIT.
    pub fn weighted_blended() -> Self {
        Self {
            method: OitMethod::WeightedBlended,
            max_peel_passes: 4,
            max_fragments_per_pixel: 8,
            stochastic_samples: 8,
            stochastic_temporal: true,
            stochastic_blend: 0.9,
            weight_exponent: 3.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            render_back_faces: true,
            alpha_to_coverage: false,
        }
    }

    /// Creates settings for stochastic transparency.
    pub fn stochastic() -> Self {
        Self {
            method: OitMethod::Stochastic,
            max_peel_passes: 1,
            max_fragments_per_pixel: 1,
            stochastic_samples: 16,
            stochastic_temporal: true,
            stochastic_blend: 0.92,
            weight_exponent: 3.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            render_back_faces: false,
            alpha_to_coverage: true,
        }
    }

    /// Creates settings for depth peeling.
    pub fn depth_peeling() -> Self {
        Self {
            method: OitMethod::DepthPeeling,
            max_peel_passes: 4,
            max_fragments_per_pixel: 1,
            stochastic_samples: 1,
            stochastic_temporal: false,
            stochastic_blend: 0.0,
            weight_exponent: 3.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            render_back_faces: true,
            alpha_to_coverage: false,
        }
    }
}

impl Default for OitSettings {
    fn default() -> Self {
        Self::weighted_blended()
    }
}

// ---------------------------------------------------------------------------
// Transparent fragment
// ---------------------------------------------------------------------------

/// A single transparent fragment.
#[derive(Debug, Clone, Copy)]
pub struct TransparentFragment {
    /// Pre-multiplied colour (RGBA).
    pub color: [f32; 4],
    /// Linear depth from the camera.
    pub depth: f32,
    /// Object/material ID (for sorting and grouping).
    pub object_id: u32,
}

impl TransparentFragment {
    /// Creates a new fragment.
    pub fn new(color: [f32; 4], depth: f32) -> Self {
        Self {
            color,
            depth,
            object_id: 0,
        }
    }

    /// Alpha value.
    pub fn alpha(&self) -> f32 {
        self.color[3]
    }

    /// Pre-multiplied colour (RGB).
    pub fn premultiplied_rgb(&self) -> [f32; 3] {
        [
            self.color[0] * self.color[3],
            self.color[1] * self.color[3],
            self.color[2] * self.color[3],
        ]
    }
}

// ---------------------------------------------------------------------------
// Stochastic transparency
// ---------------------------------------------------------------------------

/// Stochastic transparency renderer.
///
/// Implements probabilistic alpha testing: for each fragment, generate a
/// random threshold and compare it to the fragment's alpha. If alpha >=
/// threshold, the fragment is rendered; otherwise it is discarded.
///
/// Over many samples (or frames with temporal accumulation), the result
/// converges to the correct alpha-blended output.
#[derive(Debug)]
pub struct StochasticTransparency {
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
    /// Number of stochastic samples per pixel.
    pub num_samples: u32,
    /// Accumulated colour (for temporal convergence).
    accumulation: Vec<[f32; 4]>,
    /// Frame counter.
    frame_count: u32,
    /// Temporal blend factor.
    pub temporal_blend: f32,
}

impl StochasticTransparency {
    /// Creates a new stochastic transparency renderer.
    pub fn new(width: u32, height: u32, num_samples: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            num_samples,
            accumulation: vec![[0.0; 4]; total],
            frame_count: 0,
            temporal_blend: 0.92,
        }
    }

    /// Tests whether a fragment should be rendered (stochastic alpha test).
    ///
    /// # Arguments
    /// * `alpha` — Fragment alpha [0, 1].
    /// * `x`, `y` — Pixel coordinates.
    /// * `sample_index` — Sample index within the current frame.
    ///
    /// # Returns
    /// `true` if the fragment passes the stochastic test.
    pub fn alpha_test(&self, alpha: f32, x: u32, y: u32, sample_index: u32) -> bool {
        let threshold = pseudo_random(x, y, sample_index, self.frame_count);
        alpha >= threshold
    }

    /// Renders a set of transparent fragments using stochastic transparency.
    ///
    /// # Arguments
    /// * `fragments` — Transparent fragments for all pixels (sorted by pixel).
    /// * `opaque_color` — Opaque scene colour buffer.
    /// * `opaque_depth` — Opaque scene depth buffer.
    ///
    /// # Returns
    /// Composited colour buffer.
    pub fn render(
        &mut self,
        pixel_fragments: &[Vec<TransparentFragment>],
        opaque_color: &[[f32; 3]],
        opaque_depth: &[f32],
    ) -> Vec<[f32; 3]> {
        let total = (self.width * self.height) as usize;
        let mut result = vec![[0.0f32; 3]; total];

        for idx in 0..total {
            let x = (idx % self.width as usize) as u32;
            let y = (idx / self.width as usize) as u32;

            if idx >= pixel_fragments.len() || pixel_fragments[idx].is_empty() {
                result[idx] = opaque_color[idx];
                continue;
            }

            let fragments = &pixel_fragments[idx];
            let mut total_color = [0.0f32; 3];
            let mut total_count = 0u32;

            for sample in 0..self.num_samples {
                let mut sample_color = opaque_color[idx];
                let mut sample_depth = opaque_depth[idx];

                // Process fragments back-to-front for this sample.
                for frag in fragments.iter().rev() {
                    if frag.depth > sample_depth {
                        continue;
                    }

                    if self.alpha_test(frag.alpha(), x, y, sample) {
                        // Fragment passes: blend over background.
                        let alpha = frag.alpha();
                        sample_color = [
                            sample_color[0] * (1.0 - alpha) + frag.color[0] * alpha,
                            sample_color[1] * (1.0 - alpha) + frag.color[1] * alpha,
                            sample_color[2] * (1.0 - alpha) + frag.color[2] * alpha,
                        ];
                    }
                }

                total_color[0] += sample_color[0];
                total_color[1] += sample_color[1];
                total_color[2] += sample_color[2];
                total_count += 1;
            }

            if total_count > 0 {
                let inv = 1.0 / total_count as f32;
                let current = [
                    total_color[0] * inv,
                    total_color[1] * inv,
                    total_color[2] * inv,
                ];

                // Temporal accumulation.
                let prev = self.accumulation[idx];
                let blend = self.temporal_blend;
                let blended = [
                    lerp(current[0], prev[0], blend),
                    lerp(current[1], prev[1], blend),
                    lerp(current[2], prev[2], blend),
                ];

                result[idx] = blended;
                self.accumulation[idx] = [blended[0], blended[1], blended[2], 1.0];
            } else {
                result[idx] = opaque_color[idx];
            }
        }

        self.frame_count += 1;
        result
    }

    /// Resets temporal accumulation.
    pub fn reset(&mut self) {
        for v in &mut self.accumulation {
            *v = [0.0; 4];
        }
        self.frame_count = 0;
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.accumulation.len() * std::mem::size_of::<[f32; 4]>()
    }
}

// ---------------------------------------------------------------------------
// Weighted Blended OIT (WBOIT)
// ---------------------------------------------------------------------------

/// Weighted Blended Order-Independent Transparency.
///
/// Accumulates all transparent fragments into two render targets:
/// 1. Accumulation buffer: sum of (colour * alpha * weight)
/// 2. Revealage buffer: product of (1 - alpha)
///
/// The final colour is computed in a full-screen resolve pass.
#[derive(Debug)]
pub struct WeightedBlendedOit {
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
    /// Accumulation buffer (premultiplied RGBA weighted sum).
    accum: Vec<[f32; 4]>,
    /// Revealage buffer (product of 1-alpha).
    revealage: Vec<f32>,
    /// Weight function exponent.
    pub weight_exponent: f32,
    /// Near/far planes for depth weighting.
    pub near: f32,
    pub far: f32,
}

impl WeightedBlendedOit {
    /// Creates a new WBOIT renderer.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            accum: vec![[0.0; 4]; total],
            revealage: vec![1.0; total],
            weight_exponent: 3.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Clears the buffers for a new frame.
    pub fn clear(&mut self) {
        for v in &mut self.accum {
            *v = [0.0; 4];
        }
        for v in &mut self.revealage {
            *v = 1.0;
        }
    }

    /// Computes the depth-based weight for a fragment.
    ///
    /// This weight function gives more importance to closer fragments,
    /// reducing artefacts from distant transparent surfaces.
    pub fn compute_weight(&self, depth: f32, alpha: f32) -> f32 {
        // McGuire/Bavoil weight function.
        let z = (depth - self.near) / (self.far - self.near);
        let z_clamped = z.clamp(0.001, 1.0);

        // w = alpha * max(0.01, min(3000, 10 / (0.00001 + z^3)))
        let depth_weight = 10.0 / (1e-5 + z_clamped.powf(self.weight_exponent));
        let weight = alpha * depth_weight.clamp(0.01, 3000.0);

        weight
    }

    /// Adds a transparent fragment to the accumulation.
    pub fn add_fragment(&mut self, x: u32, y: u32, fragment: &TransparentFragment) {
        let idx = (y * self.width + x) as usize;
        if idx >= self.accum.len() {
            return;
        }

        let alpha = fragment.alpha();
        let weight = self.compute_weight(fragment.depth, alpha);

        // Accumulate weighted colour.
        self.accum[idx][0] += fragment.color[0] * alpha * weight;
        self.accum[idx][1] += fragment.color[1] * alpha * weight;
        self.accum[idx][2] += fragment.color[2] * alpha * weight;
        self.accum[idx][3] += alpha * weight;

        // Revealage: multiply by (1 - alpha).
        self.revealage[idx] *= 1.0 - alpha;
    }

    /// Resolves the WBOIT into a final colour buffer.
    ///
    /// Composites the accumulated transparent colour over the opaque background.
    pub fn resolve(&self, opaque_color: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let total = (self.width * self.height) as usize;
        let mut result = vec![[0.0f32; 3]; total];

        for idx in 0..total {
            let accum = self.accum[idx];
            let reveal = self.revealage[idx];

            if accum[3] < 1e-5 {
                // No transparent fragments.
                result[idx] = opaque_color[idx];
                continue;
            }

            // Resolve colour: accum.rgb / accum.a
            let inv_weight = 1.0 / accum[3].max(1e-5);
            let transparent_color = [
                accum[0] * inv_weight,
                accum[1] * inv_weight,
                accum[2] * inv_weight,
            ];

            // Composite over opaque.
            let total_alpha = 1.0 - reveal;
            result[idx] = [
                opaque_color[idx][0] * reveal + transparent_color[0] * total_alpha,
                opaque_color[idx][1] * reveal + transparent_color[1] * total_alpha,
                opaque_color[idx][2] * reveal + transparent_color[2] * total_alpha,
            ];
        }

        result
    }

    /// Returns the accumulation buffer for GPU access.
    pub fn accum_data(&self) -> &[[f32; 4]] {
        &self.accum
    }

    /// Returns the revealage buffer for GPU access.
    pub fn revealage_data(&self) -> &[f32] {
        &self.revealage
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.accum.len() * std::mem::size_of::<[f32; 4]>()
            + self.revealage.len() * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Depth peeling
// ---------------------------------------------------------------------------

/// Depth peeling OIT implementation.
///
/// Renders the scene multiple times. Each pass peels away the nearest
/// transparent layer, revealing the one behind it.
#[derive(Debug)]
pub struct DepthPeeling {
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
    /// Maximum number of peeling passes.
    pub max_passes: u32,
    /// Peeled layers (colour + depth per pass).
    layers: Vec<DepthPeelLayer>,
}

/// A single peeled depth layer.
#[derive(Debug, Clone)]
pub struct DepthPeelLayer {
    /// Colour buffer (RGBA, pre-multiplied alpha).
    pub color: Vec<[f32; 4]>,
    /// Depth buffer.
    pub depth: Vec<f32>,
    /// Number of fragments rasterised in this layer.
    pub fragment_count: u32,
}

impl DepthPeeling {
    /// Creates a new depth peeling renderer.
    pub fn new(width: u32, height: u32, max_passes: u32) -> Self {
        Self {
            width,
            height,
            max_passes,
            layers: Vec::new(),
        }
    }

    /// Performs depth peeling on a set of transparent fragments.
    ///
    /// # Arguments
    /// * `pixel_fragments` — Fragments per pixel, sorted by depth (front-to-back).
    pub fn peel(&mut self, pixel_fragments: &[Vec<TransparentFragment>]) {
        self.layers.clear();
        let total = (self.width * self.height) as usize;

        // Track the current peel depth per pixel.
        let mut peel_depth = vec![-f32::MAX; total];

        for pass in 0..self.max_passes {
            let mut layer = DepthPeelLayer {
                color: vec![[0.0; 4]; total],
                depth: vec![f32::MAX; total],
                fragment_count: 0,
            };

            let mut any_fragment = false;

            for idx in 0..total {
                if idx >= pixel_fragments.len() {
                    continue;
                }

                // Find the nearest fragment that is farther than the current peel depth.
                let mut best_frag: Option<&TransparentFragment> = None;
                let mut best_depth = f32::MAX;

                for frag in &pixel_fragments[idx] {
                    if frag.depth > peel_depth[idx] && frag.depth < best_depth {
                        best_frag = Some(frag);
                        best_depth = frag.depth;
                    }
                }

                if let Some(frag) = best_frag {
                    layer.color[idx] = frag.color;
                    layer.depth[idx] = frag.depth;
                    layer.fragment_count += 1;
                    peel_depth[idx] = frag.depth;
                    any_fragment = true;
                }
            }

            if !any_fragment {
                break; // No more layers to peel.
            }

            self.layers.push(layer);
        }
    }

    /// Composites the peeled layers over the opaque background.
    ///
    /// Blends layers back-to-front (last peel first).
    pub fn composite(&self, opaque_color: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let total = (self.width * self.height) as usize;
        let mut result: Vec<[f32; 3]> = opaque_color.to_vec();

        // Composite back-to-front.
        for layer in self.layers.iter().rev() {
            for idx in 0..total {
                let frag = layer.color[idx];
                let alpha = frag[3];

                if alpha > 0.001 {
                    result[idx] = [
                        result[idx][0] * (1.0 - alpha) + frag[0] * alpha,
                        result[idx][1] * (1.0 - alpha) + frag[1] * alpha,
                        result[idx][2] * (1.0 - alpha) + frag[2] * alpha,
                    ];
                }
            }
        }

        result
    }

    /// Returns the number of peeled layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns memory usage.
    pub fn memory_usage(&self) -> usize {
        self.layers.iter().map(|l| {
            l.color.len() * std::mem::size_of::<[f32; 4]>()
                + l.depth.len() * std::mem::size_of::<f32>()
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Per-pixel linked list (conceptual)
// ---------------------------------------------------------------------------

/// Node in a per-pixel fragment linked list.
#[derive(Debug, Clone, Copy)]
pub struct FragmentListNode {
    /// Fragment colour (RGBA).
    pub color: [f32; 4],
    /// Fragment depth.
    pub depth: f32,
    /// Index of the next node (u32::MAX = end of list).
    pub next: u32,
}

/// Per-pixel linked list storage for OIT.
///
/// This is a conceptual implementation. On the GPU, this uses atomics
/// for the head pointer and an append buffer for nodes.
#[derive(Debug)]
pub struct FragmentLinkedList {
    /// Head pointers per pixel (index into the node pool, u32::MAX = empty).
    pub heads: Vec<u32>,
    /// Node storage pool.
    pub nodes: Vec<FragmentListNode>,
    /// Next free node index.
    pub next_free: u32,
    /// Maximum total nodes.
    pub max_nodes: u32,
    /// Buffer width.
    pub width: u32,
    /// Buffer height.
    pub height: u32,
}

impl FragmentLinkedList {
    /// Creates a new linked list storage.
    pub fn new(width: u32, height: u32, max_fragments_per_pixel: u32) -> Self {
        let total_pixels = (width * height) as usize;
        let max_nodes = (width * height * max_fragments_per_pixel).min(16 * 1024 * 1024);

        Self {
            heads: vec![u32::MAX; total_pixels],
            nodes: Vec::with_capacity(max_nodes as usize),
            next_free: 0,
            max_nodes,
            width,
            height,
        }
    }

    /// Clears the linked list for a new frame.
    pub fn clear(&mut self) {
        for h in &mut self.heads {
            *h = u32::MAX;
        }
        self.nodes.clear();
        self.next_free = 0;
    }

    /// Inserts a fragment into the linked list for a pixel.
    pub fn insert(&mut self, x: u32, y: u32, color: [f32; 4], depth: f32) -> bool {
        if self.next_free >= self.max_nodes {
            return false; // Pool exhausted.
        }

        let pixel_idx = (y * self.width + x) as usize;
        let node_idx = self.next_free;
        self.next_free += 1;

        let node = FragmentListNode {
            color,
            depth,
            next: self.heads[pixel_idx],
        };

        self.nodes.push(node);
        self.heads[pixel_idx] = node_idx;

        true
    }

    /// Resolves a single pixel by sorting its fragments and compositing.
    pub fn resolve_pixel(&self, x: u32, y: u32, opaque_color: [f32; 3]) -> [f32; 3] {
        let pixel_idx = (y * self.width + x) as usize;
        let mut head = self.heads[pixel_idx];

        // Collect fragments.
        let mut fragments = Vec::new();
        while head != u32::MAX {
            if (head as usize) < self.nodes.len() {
                let node = &self.nodes[head as usize];
                fragments.push((node.color, node.depth));
                head = node.next;
            } else {
                break;
            }
        }

        if fragments.is_empty() {
            return opaque_color;
        }

        // Sort by depth (back-to-front).
        fragments.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Composite back-to-front over opaque.
        let mut result = opaque_color;
        for (color, _depth) in &fragments {
            let alpha = color[3];
            result = [
                result[0] * (1.0 - alpha) + color[0] * alpha,
                result[1] * (1.0 - alpha) + color[1] * alpha,
                result[2] * (1.0 - alpha) + color[2] * alpha,
            ];
        }

        result
    }

    /// Resolves all pixels.
    pub fn resolve(&self, opaque_color: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut result = Vec::with_capacity(opaque_color.len());
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let bg = opaque_color[idx];
                result.push(self.resolve_pixel(x, y, bg));
            }
        }
        result
    }

    /// Returns the number of stored fragments.
    pub fn fragment_count(&self) -> u32 {
        self.next_free
    }

    /// Returns memory usage.
    pub fn memory_usage(&self) -> usize {
        self.heads.len() * std::mem::size_of::<u32>()
            + self.nodes.capacity() * std::mem::size_of::<FragmentListNode>()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Pseudo-random number for stochastic testing.
fn pseudo_random(x: u32, y: u32, sample: u32, frame: u32) -> f32 {
    let mut n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(sample.wrapping_mul(1274126177))
        .wrapping_add(frame.wrapping_mul(48271));
    n = n ^ (n >> 13);
    n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303)).wrapping_add(1376312589));
    (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

/// WGSL fragment shader for Weighted Blended OIT accumulation pass.
pub const WBOIT_ACCUM_WGSL: &str = r#"
// -----------------------------------------------------------------------
// WBOIT accumulation pass (Genovo Engine)
// -----------------------------------------------------------------------

struct WboitUniforms {
    near: f32,
    far: f32,
    weight_exp: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> wboit: WboitUniforms;

struct FragmentOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
};

@fragment
fn fs_accum(
    @location(0) color: vec4<f32>,
    @builtin(position) frag_pos: vec4<f32>,
) -> FragmentOutput {
    var out: FragmentOutput;

    let alpha = color.a;
    if alpha < 0.001 {
        discard;
    }

    let z = (frag_pos.z - wboit.near) / (wboit.far - wboit.near);
    let z_clamped = clamp(z, 0.001, 1.0);
    let depth_weight = 10.0 / (1e-5 + pow(z_clamped, wboit.weight_exp));
    let weight = alpha * clamp(depth_weight, 0.01, 3000.0);

    out.accum = vec4<f32>(color.rgb * alpha * weight, alpha * weight);
    out.revealage = alpha;

    return out;
}
"#;

/// WGSL fragment shader for WBOIT resolve pass.
pub const WBOIT_RESOLVE_WGSL: &str = r#"
// -----------------------------------------------------------------------
// WBOIT resolve pass (Genovo Engine)
// -----------------------------------------------------------------------

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;
@group(0) @binding(2) var opaque_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;

@fragment
fn fs_resolve(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let accum = textureSample(accum_tex, tex_sampler, uv);
    let revealage = textureSample(revealage_tex, tex_sampler, uv).r;
    let opaque = textureSample(opaque_tex, tex_sampler, uv);

    if accum.a < 1e-5 {
        return opaque;
    }

    let avg_color = accum.rgb / max(accum.a, 1e-5);
    let alpha = 1.0 - revealage;

    return vec4<f32>(
        opaque.rgb * revealage + avg_color * alpha,
        1.0
    );
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wboit_basic() {
        let mut wboit = WeightedBlendedOit::new(2, 2);

        let frag = TransparentFragment::new([1.0, 0.0, 0.0, 0.5], 10.0);
        wboit.add_fragment(0, 0, &frag);

        let opaque = vec![[0.0, 0.0, 1.0]; 4]; // Blue background.
        let result = wboit.resolve(&opaque);

        // Pixel (0,0) should be blended between red and blue.
        assert!(result[0][0] > 0.0, "Should have some red: {:?}", result[0]);
        assert!(result[0][2] > 0.0, "Should have some blue: {:?}", result[0]);

        // Other pixels should be pure blue.
        assert!((result[1][2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wboit_multiple_fragments() {
        let mut wboit = WeightedBlendedOit::new(2, 2);

        wboit.add_fragment(0, 0, &TransparentFragment::new([1.0, 0.0, 0.0, 0.3], 5.0));
        wboit.add_fragment(0, 0, &TransparentFragment::new([0.0, 1.0, 0.0, 0.3], 10.0));

        let opaque = vec![[0.0, 0.0, 0.0]; 4];
        let result = wboit.resolve(&opaque);

        // Should have contributions from both fragments.
        assert!(result[0][0] > 0.0, "Should have red");
        assert!(result[0][1] > 0.0, "Should have green");
    }

    #[test]
    fn test_depth_peeling() {
        let mut peeler = DepthPeeling::new(2, 2, 4);

        let pixel_fragments = vec![
            vec![
                TransparentFragment::new([1.0, 0.0, 0.0, 0.5], 5.0),
                TransparentFragment::new([0.0, 1.0, 0.0, 0.5], 10.0),
            ],
            vec![],
            vec![],
            vec![],
        ];

        peeler.peel(&pixel_fragments);
        assert!(peeler.layer_count() >= 1);

        let opaque = vec![[0.0, 0.0, 1.0]; 4];
        let result = peeler.composite(&opaque);
        assert!(result[0][0] > 0.0, "Should have some red");
    }

    #[test]
    fn test_stochastic_alpha_test() {
        let st = StochasticTransparency::new(4, 4, 16);

        // Alpha = 1.0 should always pass.
        let mut pass_count = 0;
        for s in 0..100 {
            if st.alpha_test(1.0, 0, 0, s) {
                pass_count += 1;
            }
        }
        assert_eq!(pass_count, 100, "Alpha 1.0 should always pass");

        // Alpha = 0.0 should never pass.
        pass_count = 0;
        for s in 0..100 {
            if st.alpha_test(0.0, 0, 0, s) {
                pass_count += 1;
            }
        }
        assert_eq!(pass_count, 0, "Alpha 0.0 should never pass");
    }

    #[test]
    fn test_stochastic_convergence() {
        let st = StochasticTransparency::new(1, 1, 1000);

        // Alpha = 0.5 should pass about 50% of the time.
        let mut pass_count = 0;
        for s in 0..1000 {
            if st.alpha_test(0.5, 0, 0, s) {
                pass_count += 1;
            }
        }
        let pass_rate = pass_count as f32 / 1000.0;
        assert!(
            (pass_rate - 0.5).abs() < 0.1,
            "Alpha 0.5 should pass ~50%: got {pass_rate}"
        );
    }

    #[test]
    fn test_linked_list_basic() {
        let mut ll = FragmentLinkedList::new(2, 2, 4);

        ll.insert(0, 0, [1.0, 0.0, 0.0, 0.5], 5.0);
        ll.insert(0, 0, [0.0, 1.0, 0.0, 0.5], 10.0);
        assert_eq!(ll.fragment_count(), 2);

        let opaque = vec![[0.0, 0.0, 1.0]; 4];
        let result = ll.resolve(&opaque);

        assert!(result[0][0] > 0.0, "Should have red");
        assert!(result[0][1] > 0.0, "Should have green");
    }

    #[test]
    fn test_linked_list_clear() {
        let mut ll = FragmentLinkedList::new(2, 2, 4);
        ll.insert(0, 0, [1.0, 0.0, 0.0, 0.5], 5.0);
        ll.clear();
        assert_eq!(ll.fragment_count(), 0);
    }

    #[test]
    fn test_wboit_weight_function() {
        let wboit = WeightedBlendedOit::new(1, 1);
        let near_weight = wboit.compute_weight(1.0, 0.5);
        let far_weight = wboit.compute_weight(100.0, 0.5);
        assert!(near_weight > far_weight, "Near should have higher weight");
    }

    #[test]
    fn test_oit_settings() {
        let wb = OitSettings::weighted_blended();
        assert!(matches!(wb.method, OitMethod::WeightedBlended));

        let st = OitSettings::stochastic();
        assert!(matches!(st.method, OitMethod::Stochastic));
    }

    #[test]
    fn test_fragment_premultiplied() {
        let frag = TransparentFragment::new([1.0, 0.5, 0.0, 0.5], 10.0);
        let pm = frag.premultiplied_rgb();
        assert!((pm[0] - 0.5).abs() < 0.01);
        assert!((pm[1] - 0.25).abs() < 0.01);
    }
}
