//! GPU-accelerated UI renderer using wgpu.
//!
//! This module provides the core GPU rendering backend for the Genovo UI
//! framework. All UI primitives (rectangles, rounded rects, circles, lines,
//! text, images) are rendered via wgpu with SDF-based anti-aliasing in the
//! fragment shader.
//!
//! # Architecture
//!
//! ```text
//!  DrawCommand  -->  CommandBuffer  -->  BatchSorter  -->  VertexBuilder  -->  GPU
//!                                           |                  |
//!                                      TextureAtlas       IndexBuffer
//! ```
//!
//! All geometry is batched by texture to minimise draw calls and state changes.
//! Clipping is handled via the hardware scissor rect stack. Double-buffered
//! vertex/index buffers prevent stalls.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;
use genovo_core::Rect;

use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// WGSL Shaders (embedded)
// ---------------------------------------------------------------------------

/// Vertex shader: transforms screen-space pixel positions into NDC clip space.
/// Also passes through UV, color, corner_radius, border_width, rect_size,
/// rect_center, and draw_flags to the fragment shader.
const VERTEX_SHADER_SOURCE: &str = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) corner_radius: f32,
    @location(4) border_width: f32,
    @location(5) rect_size: vec2<f32>,
    @location(6) rect_center: vec2<f32>,
    @location(7) draw_flags: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) corner_radius: f32,
    @location(3) border_width: f32,
    @location(4) rect_size: vec2<f32>,
    @location(5) rect_center: vec2<f32>,
    @location(6) @interpolate(flat) draw_flags: u32,
    @location(7) world_pos: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Convert screen-space pixel coordinates to NDC [-1, 1].
    // Screen space: (0,0) top-left, (W,H) bottom-right.
    let ndc_x = (input.position.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (input.position.y / uniforms.screen_size.y) * 2.0;

    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    out.corner_radius = input.corner_radius;
    out.border_width = input.border_width;
    out.rect_size = input.rect_size;
    out.rect_center = input.rect_center;
    out.draw_flags = input.draw_flags;
    out.world_pos = input.position;

    return out;
}
"#;

/// Fragment shader with SDF-based rounded rectangle, circle, line, text, and
/// image rendering. The `draw_flags` field selects the rendering mode:
///   0 = solid color (rect/circle with SDF rounding)
///   1 = textured (image sampling with tint)
///   2 = SDF text (alpha from texture, color from vertex)
///   3 = gradient (per-vertex color interpolation, no texture)
///   4 = shadow (expanded rect with gaussian-ish alpha falloff)
///   5 = line (anti-aliased line segment via SDF)
///   6 = circle (SDF circle)
///   7 = border-only (SDF annulus)
const FRAGMENT_SHADER_SOURCE: &str = r#"
@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) corner_radius: f32,
    @location(3) border_width: f32,
    @location(4) rect_size: vec2<f32>,
    @location(5) rect_center: vec2<f32>,
    @location(6) @interpolate(flat) draw_flags: u32,
    @location(7) world_pos: vec2<f32>,
};

// Signed distance function for a rounded rectangle.
// `p` is relative to center, `half_size` is half-extents, `r` is corner radius.
fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let clamped_r = min(r, min(half_size.x, half_size.y));
    let q = abs(p) - half_size + vec2<f32>(clamped_r, clamped_r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - clamped_r;
}

// Signed distance function for a circle.
fn sdf_circle(p: vec2<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

// Approximate Gaussian for shadow blur.
fn gaussian(x: f32, sigma: f32) -> f32 {
    let s2 = sigma * sigma;
    return exp(-(x * x) / (2.0 * s2));
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let flags = input.draw_flags;

    // ---------- SOLID / ROUNDED RECT (flag 0) ----------
    if flags == 0u {
        let half_size = input.rect_size * 0.5;
        let local_pos = input.world_pos - input.rect_center;
        let dist = sdf_rounded_rect(local_pos, half_size, input.corner_radius);

        // Anti-alias: smooth transition over ~1.5 pixels.
        let aa = 1.0 - smoothstep(-1.0, 1.0, dist);
        return vec4<f32>(input.color.rgb, input.color.a * aa);
    }

    // ---------- TEXTURED / IMAGE (flag 1) ----------
    if flags == 1u {
        let tex_color = textureSample(t_diffuse, s_diffuse, input.uv);
        return tex_color * input.color;
    }

    // ---------- BITMAP TEXT (flag 2) ----------
    if flags == 2u {
        let tex_sample = textureSample(t_diffuse, s_diffuse, input.uv);
        // Bitmap font: alpha channel directly encodes glyph coverage.
        let alpha = tex_sample.a;
        return vec4<f32>(input.color.rgb, input.color.a * alpha);
    }

    // ---------- GRADIENT (flag 3) ----------
    if flags == 3u {
        // Per-vertex color interpolation already handled; just apply SDF rounding.
        let half_size = input.rect_size * 0.5;
        let local_pos = input.world_pos - input.rect_center;
        let dist = sdf_rounded_rect(local_pos, half_size, input.corner_radius);
        let aa = 1.0 - smoothstep(-1.0, 1.0, dist);
        return vec4<f32>(input.color.rgb, input.color.a * aa);
    }

    // ---------- SHADOW (flag 4) ----------
    if flags == 4u {
        let half_size = input.rect_size * 0.5;
        let local_pos = input.world_pos - input.rect_center;
        let dist = sdf_rounded_rect(local_pos, half_size, input.corner_radius);

        // Use border_width as blur_radius for shadows.
        let blur = max(input.border_width, 1.0);
        let shadow_alpha = 1.0 - smoothstep(-blur, blur * 1.5, dist);

        return vec4<f32>(input.color.rgb, input.color.a * shadow_alpha);
    }

    // ---------- LINE (flag 5) ----------
    if flags == 5u {
        // For lines, rect_size.x = length, rect_size.y = thickness.
        // rect_center = midpoint of line.
        // The quad is oriented such that the line runs along the local X axis.
        // local_pos is already in screen space, so we use UV-based distance.
        let dist_from_center = abs(input.uv.y - 0.5) * input.rect_size.y;
        let half_thickness = input.rect_size.y * 0.5;
        let aa = 1.0 - smoothstep(half_thickness - 1.0, half_thickness + 1.0, dist_from_center);
        return vec4<f32>(input.color.rgb, input.color.a * aa);
    }

    // ---------- CIRCLE (flag 6) ----------
    if flags == 6u {
        let local_pos = input.world_pos - input.rect_center;
        let radius = min(input.rect_size.x, input.rect_size.y) * 0.5;
        let dist = sdf_circle(local_pos, radius);
        let aa = 1.0 - smoothstep(-1.0, 1.0, dist);
        return vec4<f32>(input.color.rgb, input.color.a * aa);
    }

    // ---------- BORDER ONLY (flag 7) ----------
    if flags == 7u {
        let half_size = input.rect_size * 0.5;
        let local_pos = input.world_pos - input.rect_center;
        let outer_dist = sdf_rounded_rect(local_pos, half_size, input.corner_radius);
        let inner_half = half_size - vec2<f32>(input.border_width, input.border_width);
        let inner_radius = max(input.corner_radius - input.border_width, 0.0);
        let inner_dist = sdf_rounded_rect(local_pos, inner_half, inner_radius);

        let outer_aa = 1.0 - smoothstep(-1.0, 1.0, outer_dist);
        let inner_aa = smoothstep(-1.0, 1.0, inner_dist);
        let border_alpha = outer_aa * inner_aa;

        return vec4<f32>(input.color.rgb, input.color.a * border_alpha);
    }

    // Fallback: solid color.
    return input.color;
}
"#;

// ---------------------------------------------------------------------------
// UIVertex
// ---------------------------------------------------------------------------

/// GPU vertex format for UI rendering. Matches the shader `VertexInput`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UIVertex {
    /// Screen-space position in pixels.
    position: [f32; 2],
    /// Texture UV coordinates.
    uv: [f32; 2],
    /// RGBA color (linear float).
    color: [f32; 4],
    /// Corner radius for SDF rounded rect.
    corner_radius: f32,
    /// Border width (also used as blur_radius for shadows).
    border_width: f32,
    /// Size of the rect (width, height) for SDF computation.
    rect_size: [f32; 2],
    /// Center of the rect in screen space for SDF computation.
    rect_center: [f32; 2],
    /// Draw mode flags (see fragment shader constants).
    draw_flags: u32,
    /// Padding to align to 16 bytes.
    _pad: u32,
}

impl UIVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
        0 => Float32x2,  // position
        1 => Float32x2,  // uv
        2 => Float32x4,  // color
        3 => Float32,    // corner_radius
        4 => Float32,    // border_width
        5 => Float32x2,  // rect_size
        6 => Float32x2,  // rect_center
        7 => Uint32,     // draw_flags
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UIVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw flag constants
// ---------------------------------------------------------------------------

const DRAW_FLAG_SOLID: u32 = 0;
const DRAW_FLAG_TEXTURED: u32 = 1;
const DRAW_FLAG_TEXT: u32 = 2;
const DRAW_FLAG_GRADIENT: u32 = 3;
const DRAW_FLAG_SHADOW: u32 = 4;
const DRAW_FLAG_LINE: u32 = 5;
const DRAW_FLAG_CIRCLE: u32 = 6;
const DRAW_FLAG_BORDER: u32 = 7;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UIUniforms {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

// ---------------------------------------------------------------------------
// InternalDrawCommand
// ---------------------------------------------------------------------------

/// Intermediate draw command used within the renderer for batching.
#[derive(Debug, Clone)]
struct InternalDrawCommand {
    /// Index into the `textures` map (0 = white texture).
    texture_index: usize,
    /// Starting vertex index in the global vertex buffer.
    vertex_offset: u32,
    /// Starting index in the global index buffer.
    index_offset: u32,
    /// Number of indices to draw.
    index_count: u32,
    /// Optional scissor rect.
    scissor: Option<ScissorRect>,
}

#[derive(Debug, Clone, Copy)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------
// TextureEntry
// ---------------------------------------------------------------------------

/// A texture managed by the UI renderer.
struct TextureEntry {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// GlyphInfo (for built-in font atlas)
// ---------------------------------------------------------------------------

/// Pre-computed glyph metrics for the built-in bitmap font.
#[derive(Debug, Clone, Copy)]
struct BuiltinGlyphInfo {
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    advance: f32,
    width: f32,
    height: f32,
    offset_y: f32,
}

// ---------------------------------------------------------------------------
// UIRenderer
// ---------------------------------------------------------------------------

/// The GPU-accelerated UI renderer. Owns wgpu pipelines, buffers, and textures
/// needed to draw the entire UI each frame.
///
/// Usage:
/// 1. Call `begin_frame(screen_size)` at the start of each frame.
/// 2. Issue draw commands (`draw_rect`, `draw_text`, etc.).
/// 3. Call `end_frame(encoder, target_view)` to submit all draw calls.
pub struct UIGpuRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Pipeline
    render_pipeline: wgpu::RenderPipeline,

    // Buffers (double-buffered)
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffers: [wgpu::Buffer; 2],
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    current_buffer: usize,

    // Buffer capacities
    max_vertices: usize,
    max_indices: usize,

    // CPU-side geometry staging
    vertices: Vec<UIVertex>,
    indices: Vec<u32>,
    draw_commands: Vec<InternalDrawCommand>,

    // Textures
    textures: Vec<TextureEntry>,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    next_texture_id: u64,
    texture_id_map: HashMap<u64, usize>,

    // White texture (index 0) for solid color draws
    white_texture_index: usize,

    // Font atlas (index 1)
    font_atlas_index: usize,
    builtin_glyphs: HashMap<char, BuiltinGlyphInfo>,
    font_atlas_width: u32,
    font_atlas_height: u32,

    // Clip/scissor stack
    clip_stack: Vec<Rect>,

    // Screen size
    screen_size: Vec2,

    // Statistics
    frame_draw_calls: u32,
    frame_triangles: u32,
    frame_vertices: u32,

    // Surface format
    surface_format: wgpu::TextureFormat,
}

impl UIGpuRenderer {
    /// Maximum number of vertices per frame (across both buffers).
    const DEFAULT_MAX_VERTICES: usize = 65536;
    /// Maximum number of indices per frame.
    const DEFAULT_MAX_INDICES: usize = 131072;
    /// Glyph cell size in the built-in bitmap font atlas (8x16 VGA font).
    const GLYPH_CELL_W: u32 = 8;
    const GLYPH_CELL_H: u32 = 16;

    /// Creates a new UI GPU renderer.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // --- Shader module ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui_shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE).into(),
            ),
        });

        // --- Uniform buffer + bind group layout ---
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_uniform_buffer"),
            size: std::mem::size_of::<UIUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui_uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_uniform_bg"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Texture bind group layout ---
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui_texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // --- Pipeline layout ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // --- Render pipeline ---
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[UIVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // UI quads may be wound either way
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Vertex / Index buffers (double-buffered) ---
        let vb_size = (Self::DEFAULT_MAX_VERTICES * std::mem::size_of::<UIVertex>()) as u64;
        let ib_size = (Self::DEFAULT_MAX_INDICES * std::mem::size_of::<u32>()) as u64;

        let create_vb = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: vb_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let create_ib = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ib_size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let vertex_buffers = [create_vb("ui_vb_0"), create_vb("ui_vb_1")];
        let index_buffers = [create_ib("ui_ib_0"), create_ib("ui_ib_1")];

        // --- Create white 1x1 texture ---
        let mut renderer = Self {
            device: device.clone(),
            queue: queue.clone(),
            render_pipeline,
            vertex_buffers,
            index_buffers,
            uniform_buffer,
            uniform_bind_group,
            current_buffer: 0,
            max_vertices: Self::DEFAULT_MAX_VERTICES,
            max_indices: Self::DEFAULT_MAX_INDICES,
            vertices: Vec::with_capacity(4096),
            indices: Vec::with_capacity(8192),
            draw_commands: Vec::with_capacity(256),
            textures: Vec::new(),
            texture_bind_group_layout,
            next_texture_id: 1,
            texture_id_map: HashMap::new(),
            white_texture_index: 0,
            font_atlas_index: 0,
            builtin_glyphs: HashMap::new(),
            font_atlas_width: 128,
            font_atlas_height: 96,
            clip_stack: Vec::with_capacity(8),
            screen_size: Vec2::new(1920.0, 1080.0),
            frame_draw_calls: 0,
            frame_triangles: 0,
            frame_vertices: 0,
            surface_format,
        };

        // Create the 1x1 white texture (index 0).
        let white_data = [255u8, 255, 255, 255];
        renderer.white_texture_index = renderer.create_texture_internal(1, 1, &white_data);

        // Create the built-in bitmap font atlas (index 1).
        renderer.build_builtin_font_atlas();

        renderer
    }

    // -----------------------------------------------------------------------
    // Internal texture management
    // -----------------------------------------------------------------------

    fn create_texture_internal(&mut self, width: u32, height: u32, rgba_data: &[u8]) -> usize {
        self.create_texture_internal_filtered(width, height, rgba_data, wgpu::FilterMode::Linear)
    }

    fn create_texture_internal_filtered(&mut self, width: u32, height: u32, rgba_data: &[u8], filter: wgpu::FilterMode) -> usize {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_texture_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let idx = self.textures.len();
        self.textures.push(TextureEntry {
            _texture: texture,
            view,
            bind_group,
        });

        idx
    }

    /// Register a user texture. Returns an opaque ID for later use with
    /// `draw_image`.
    pub fn register_texture(&mut self, width: u32, height: u32, rgba_data: &[u8]) -> u64 {
        let idx = self.create_texture_internal(width, height, rgba_data);
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        self.texture_id_map.insert(id, idx);
        id
    }

    /// Destroy a previously registered texture.
    pub fn destroy_texture(&mut self, id: u64) {
        self.texture_id_map.remove(&id);
        // The wgpu texture is dropped when the entry is removed. We leave the
        // slot in the Vec to avoid invalidating other indices; it will be
        // garbage collected on the next atlas rebuild.
    }

    // -----------------------------------------------------------------------
    // Built-in font atlas
    // -----------------------------------------------------------------------

    /// Builds a bitmap font atlas for ASCII printable chars (32..=126)
    /// using the classic 8x16 VGA/BIOS bitmap font (public domain).
    /// Each glyph is 8 pixels wide and 16 pixels tall.
    fn build_builtin_font_atlas(&mut self) {
        use crate::bitmap_font::get_char_bitmap;

        const CHAR_W: u32 = 8;
        const CHAR_H: u32 = 16;
        const CHARS_PER_ROW: u32 = 16;
        const NUM_CHARS: u32 = 95; // ASCII 32..=126
        const ROWS: u32 = (NUM_CHARS + CHARS_PER_ROW - 1) / CHARS_PER_ROW; // 6

        let atlas_w = CHAR_W * CHARS_PER_ROW; // 128
        let atlas_h = CHAR_H * ROWS;          // 96

        // Store actual atlas dimensions for UV computation.
        self.font_atlas_width = atlas_w;
        self.font_atlas_height = atlas_h;

        let mut pixels = vec![0u8; (atlas_w * atlas_h * 4) as usize];

        for ch in 32u8..=126u8 {
            let idx = (ch - 32) as u32;
            let col = idx % CHARS_PER_ROW;
            let row = idx / CHARS_PER_ROW;
            let base_x = col * CHAR_W;
            let base_y = row * CHAR_H;

            let bitmap = get_char_bitmap(ch);
            for y in 0..CHAR_H {
                let row_bits = bitmap[y as usize];
                for x in 0..CHAR_W {
                    if (row_bits >> (7 - x)) & 1 != 0 {
                        let px = base_x + x;
                        let py = base_y + y;
                        let offset = ((py * atlas_w + px) * 4) as usize;
                        pixels[offset] = 255;     // R
                        pixels[offset + 1] = 255; // G
                        pixels[offset + 2] = 255; // B
                        pixels[offset + 3] = 255; // A
                    }
                }
            }

            // Store glyph UV info.
            let u_min = base_x as f32 / atlas_w as f32;
            let v_min = base_y as f32 / atlas_h as f32;
            let u_max = (base_x + CHAR_W) as f32 / atlas_w as f32;
            let v_max = (base_y + CHAR_H) as f32 / atlas_h as f32;

            // Monospace font: all characters advance the same width.
            // The advance is expressed as a fraction of font_size so that
            // cursor_x += advance * font_size gives the correct spacing.
            // With an 8-wide glyph in a 16-tall cell, advance = 8/16 = 0.5.
            let char_advance = 0.5;

            self.builtin_glyphs.insert(
                ch as char,
                BuiltinGlyphInfo {
                    uv_min: [u_min, v_min],
                    uv_max: [u_max, v_max],
                    advance: char_advance,
                    width: CHAR_W as f32,
                    height: CHAR_H as f32,
                    offset_y: 0.0,
                },
            );
        }

        self.font_atlas_index = self.create_texture_internal(atlas_w, atlas_h, &pixels);
    }

    // -----------------------------------------------------------------------
    // Frame lifecycle
    // -----------------------------------------------------------------------

    /// Begin a new frame. Clears all CPU-side buffers and updates the screen
    /// size uniform.
    pub fn begin_frame(&mut self, screen_size: Vec2) {
        self.screen_size = screen_size;
        self.vertices.clear();
        self.indices.clear();
        self.draw_commands.clear();
        self.clip_stack.clear();
        self.frame_draw_calls = 0;
        self.frame_triangles = 0;
        self.frame_vertices = 0;

        // Update uniform buffer.
        let uniforms = UIUniforms {
            screen_size: [screen_size.x, screen_size.y],
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    /// Finalize the frame: upload geometry to GPU and record render commands
    /// into the provided encoder, drawing into `target_view`.
    pub fn end_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            return;
        }

        // Swap buffer index.
        self.current_buffer = 1 - self.current_buffer;
        let buf_idx = self.current_buffer;

        // Grow buffers if needed.
        if self.vertices.len() > self.max_vertices {
            self.max_vertices = self.vertices.len().next_power_of_two();
            let size = (self.max_vertices * std::mem::size_of::<UIVertex>()) as u64;
            self.vertex_buffers[buf_idx] = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_vb_grown"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if self.indices.len() > self.max_indices {
            self.max_indices = self.indices.len().next_power_of_two();
            let size = (self.max_indices * std::mem::size_of::<u32>()) as u64;
            self.index_buffers[buf_idx] = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_ib_grown"),
                size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload vertex and index data.
        self.queue.write_buffer(
            &self.vertex_buffers[buf_idx],
            0,
            bytemuck::cast_slice(&self.vertices),
        );
        self.queue.write_buffer(
            &self.index_buffers[buf_idx],
            0,
            bytemuck::cast_slice(&self.indices),
        );

        // Batch draw commands by texture and scissor.
        let batches = self.build_batches();

        // Record render pass.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffers[buf_idx].slice(..));
            render_pass.set_index_buffer(
                self.index_buffers[buf_idx].slice(..),
                wgpu::IndexFormat::Uint32,
            );

            let sw = self.screen_size.x as u32;
            let sh = self.screen_size.y as u32;

            for batch in &batches {
                // Set scissor rect.
                if let Some(sc) = batch.scissor {
                    let x = sc.x.min(sw);
                    let y = sc.y.min(sh);
                    let w = sc.width.min(sw.saturating_sub(x)).max(1);
                    let h = sc.height.min(sh.saturating_sub(y)).max(1);
                    render_pass.set_scissor_rect(x, y, w, h);
                } else {
                    render_pass.set_scissor_rect(0, 0, sw.max(1), sh.max(1));
                }

                // Set texture bind group.
                if batch.texture_index < self.textures.len() {
                    render_pass.set_bind_group(
                        1,
                        &self.textures[batch.texture_index].bind_group,
                        &[],
                    );
                }

                render_pass.draw_indexed(
                    batch.index_offset..batch.index_offset + batch.index_count,
                    batch.vertex_offset as i32,
                    0..1,
                );

                self.frame_draw_calls += 1;
                self.frame_triangles += batch.index_count / 3;
            }
        }

        self.frame_vertices = self.vertices.len() as u32;
    }

    /// Build merged draw batches from the list of internal draw commands.
    /// Adjacent commands with the same texture and scissor state are merged
    /// into a single draw call.
    fn build_batches(&self) -> Vec<InternalDrawCommand> {
        if self.draw_commands.is_empty() {
            return Vec::new();
        }

        let mut batches: Vec<InternalDrawCommand> = Vec::with_capacity(self.draw_commands.len());

        for cmd in &self.draw_commands {
            let can_merge = if let Some(last) = batches.last() {
                last.texture_index == cmd.texture_index
                    && scissor_eq(&last.scissor, &cmd.scissor)
                    && last.vertex_offset + (last.index_count / 6 * 4) == cmd.vertex_offset
            } else {
                false
            };

            if can_merge {
                if let Some(last) = batches.last_mut() {
                    last.index_count += cmd.index_count;
                }
            } else {
                batches.push(cmd.clone());
            }
        }

        batches
    }

    // -----------------------------------------------------------------------
    // Clipping
    // -----------------------------------------------------------------------

    /// Push a clipping rectangle. All subsequent draws are scissored to this
    /// rect (intersected with any existing clip).
    pub fn push_clip(&mut self, rect: Rect) {
        let clipped = if let Some(current) = self.clip_stack.last() {
            Rect::new(
                Vec2::new(rect.min.x.max(current.min.x), rect.min.y.max(current.min.y)),
                Vec2::new(rect.max.x.min(current.max.x), rect.max.y.min(current.max.y)),
            )
        } else {
            rect
        };
        self.clip_stack.push(clipped);
    }

    /// Pop the most recent clipping rectangle.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    fn current_scissor(&self) -> Option<ScissorRect> {
        self.clip_stack.last().map(|r| ScissorRect {
            x: r.min.x.max(0.0) as u32,
            y: r.min.y.max(0.0) as u32,
            width: r.width().max(0.0) as u32,
            height: r.height().max(0.0) as u32,
        })
    }

    // -----------------------------------------------------------------------
    // Quad helper
    // -----------------------------------------------------------------------

    /// Emit a quad (two triangles) with the given vertices.
    fn push_quad(
        &mut self,
        v0: UIVertex,
        v1: UIVertex,
        v2: UIVertex,
        v3: UIVertex,
        texture_index: usize,
    ) {
        let base = self.vertices.len() as u32;
        self.vertices.push(v0);
        self.vertices.push(v1);
        self.vertices.push(v2);
        self.vertices.push(v3);

        let idx_offset = self.indices.len() as u32;
        self.indices.push(base);
        self.indices.push(base + 1);
        self.indices.push(base + 2);
        self.indices.push(base);
        self.indices.push(base + 2);
        self.indices.push(base + 3);

        self.draw_commands.push(InternalDrawCommand {
            texture_index,
            vertex_offset: 0, // We use absolute indices
            index_offset: idx_offset,
            index_count: 6,
            scissor: self.current_scissor(),
        });
    }

    fn make_vertex(
        &self,
        pos: [f32; 2],
        uv: [f32; 2],
        color: [f32; 4],
        corner_radius: f32,
        border_width: f32,
        rect_size: [f32; 2],
        rect_center: [f32; 2],
        draw_flags: u32,
    ) -> UIVertex {
        UIVertex {
            position: pos,
            uv,
            color,
            corner_radius,
            border_width,
            rect_size,
            rect_center,
            draw_flags,
            _pad: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Drawing primitives
    // -----------------------------------------------------------------------

    /// Draw a filled rectangle with optional corner radius for rounding.
    pub fn draw_rect(&mut self, rect: Rect, color: Color, corner_radius: f32) {
        let c = [color.r, color.g, color.b, color.a];
        let w = rect.width();
        let h = rect.height();
        let cx = rect.min.x + w * 0.5;
        let cy = rect.min.y + h * 0.5;
        let size = [w, h];
        let center = [cx, cy];

        let v0 = self.make_vertex(
            [rect.min.x, rect.min.y], [0.0, 0.0], c, corner_radius, 0.0, size, center, DRAW_FLAG_SOLID,
        );
        let v1 = self.make_vertex(
            [rect.max.x, rect.min.y], [1.0, 0.0], c, corner_radius, 0.0, size, center, DRAW_FLAG_SOLID,
        );
        let v2 = self.make_vertex(
            [rect.max.x, rect.max.y], [1.0, 1.0], c, corner_radius, 0.0, size, center, DRAW_FLAG_SOLID,
        );
        let v3 = self.make_vertex(
            [rect.min.x, rect.max.y], [0.0, 1.0], c, corner_radius, 0.0, size, center, DRAW_FLAG_SOLID,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw a rectangle with per-corner gradient colors.
    pub fn draw_rect_gradient(
        &mut self,
        rect: Rect,
        colors: [Color; 4],
        corner_radius: f32,
    ) {
        let w = rect.width();
        let h = rect.height();
        let cx = rect.min.x + w * 0.5;
        let cy = rect.min.y + h * 0.5;
        let size = [w, h];
        let center = [cx, cy];

        let c = |col: &Color| [col.r, col.g, col.b, col.a];

        let v0 = self.make_vertex(
            [rect.min.x, rect.min.y], [0.0, 0.0], c(&colors[0]), corner_radius, 0.0, size, center, DRAW_FLAG_GRADIENT,
        );
        let v1 = self.make_vertex(
            [rect.max.x, rect.min.y], [1.0, 0.0], c(&colors[1]), corner_radius, 0.0, size, center, DRAW_FLAG_GRADIENT,
        );
        let v2 = self.make_vertex(
            [rect.max.x, rect.max.y], [1.0, 1.0], c(&colors[2]), corner_radius, 0.0, size, center, DRAW_FLAG_GRADIENT,
        );
        let v3 = self.make_vertex(
            [rect.min.x, rect.max.y], [0.0, 1.0], c(&colors[3]), corner_radius, 0.0, size, center, DRAW_FLAG_GRADIENT,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw a rectangle outline (border only, no fill).
    pub fn draw_rect_outline(
        &mut self,
        rect: Rect,
        color: Color,
        thickness: f32,
        corner_radius: f32,
    ) {
        let c = [color.r, color.g, color.b, color.a];
        let w = rect.width();
        let h = rect.height();
        let cx = rect.min.x + w * 0.5;
        let cy = rect.min.y + h * 0.5;
        let size = [w, h];
        let center = [cx, cy];

        let v0 = self.make_vertex(
            [rect.min.x, rect.min.y], [0.0, 0.0], c, corner_radius, thickness, size, center, DRAW_FLAG_BORDER,
        );
        let v1 = self.make_vertex(
            [rect.max.x, rect.min.y], [1.0, 0.0], c, corner_radius, thickness, size, center, DRAW_FLAG_BORDER,
        );
        let v2 = self.make_vertex(
            [rect.max.x, rect.max.y], [1.0, 1.0], c, corner_radius, thickness, size, center, DRAW_FLAG_BORDER,
        );
        let v3 = self.make_vertex(
            [rect.min.x, rect.max.y], [0.0, 1.0], c, corner_radius, thickness, size, center, DRAW_FLAG_BORDER,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw a drop shadow behind a rectangle.
    pub fn draw_rect_shadow(
        &mut self,
        rect: Rect,
        shadow_color: Color,
        blur_radius: f32,
        offset: Vec2,
    ) {
        let expand = blur_radius * 2.0;
        let shadow_rect = Rect::new(
            Vec2::new(rect.min.x - expand + offset.x, rect.min.y - expand + offset.y),
            Vec2::new(rect.max.x + expand + offset.x, rect.max.y + expand + offset.y),
        );

        let c = [shadow_color.r, shadow_color.g, shadow_color.b, shadow_color.a];
        let w = rect.width();
        let h = rect.height();
        let cx = shadow_rect.min.x + shadow_rect.width() * 0.5;
        let cy = shadow_rect.min.y + shadow_rect.height() * 0.5;
        let size = [w, h]; // Original rect size for SDF
        let center = [cx, cy];

        let v0 = self.make_vertex(
            [shadow_rect.min.x, shadow_rect.min.y], [0.0, 0.0], c, 0.0, blur_radius, size, center, DRAW_FLAG_SHADOW,
        );
        let v1 = self.make_vertex(
            [shadow_rect.max.x, shadow_rect.min.y], [1.0, 0.0], c, 0.0, blur_radius, size, center, DRAW_FLAG_SHADOW,
        );
        let v2 = self.make_vertex(
            [shadow_rect.max.x, shadow_rect.max.y], [1.0, 1.0], c, 0.0, blur_radius, size, center, DRAW_FLAG_SHADOW,
        );
        let v3 = self.make_vertex(
            [shadow_rect.min.x, shadow_rect.max.y], [0.0, 1.0], c, 0.0, blur_radius, size, center, DRAW_FLAG_SHADOW,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw a filled circle.
    pub fn draw_circle(&mut self, center_pos: Vec2, radius: f32, color: Color) {
        let c = [color.r, color.g, color.b, color.a];
        let size = [radius * 2.0, radius * 2.0];
        let center = [center_pos.x, center_pos.y];

        let min_x = center_pos.x - radius;
        let min_y = center_pos.y - radius;
        let max_x = center_pos.x + radius;
        let max_y = center_pos.y + radius;

        let v0 = self.make_vertex(
            [min_x, min_y], [0.0, 0.0], c, 0.0, 0.0, size, center, DRAW_FLAG_CIRCLE,
        );
        let v1 = self.make_vertex(
            [max_x, min_y], [1.0, 0.0], c, 0.0, 0.0, size, center, DRAW_FLAG_CIRCLE,
        );
        let v2 = self.make_vertex(
            [max_x, max_y], [1.0, 1.0], c, 0.0, 0.0, size, center, DRAW_FLAG_CIRCLE,
        );
        let v3 = self.make_vertex(
            [min_x, max_y], [0.0, 1.0], c, 0.0, 0.0, size, center, DRAW_FLAG_CIRCLE,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw an anti-aliased line between two points.
    pub fn draw_line(&mut self, start: Vec2, end: Vec2, color: Color, thickness: f32) {
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length = (dx * dx + dy * dy).sqrt();
        if length < 0.001 {
            return;
        }

        let nx = -dy / length;
        let ny = dx / length;
        let half_t = (thickness + 2.0) * 0.5; // +2 for AA padding

        let c = [color.r, color.g, color.b, color.a];
        let size = [length, thickness];
        let cx = (start.x + end.x) * 0.5;
        let cy = (start.y + end.y) * 0.5;
        let center = [cx, cy];

        let v0 = self.make_vertex(
            [start.x + nx * half_t, start.y + ny * half_t],
            [0.0, 0.0], c, 0.0, 0.0, size, center, DRAW_FLAG_LINE,
        );
        let v1 = self.make_vertex(
            [end.x + nx * half_t, end.y + ny * half_t],
            [1.0, 0.0], c, 0.0, 0.0, size, center, DRAW_FLAG_LINE,
        );
        let v2 = self.make_vertex(
            [end.x - nx * half_t, end.y - ny * half_t],
            [1.0, 1.0], c, 0.0, 0.0, size, center, DRAW_FLAG_LINE,
        );
        let v3 = self.make_vertex(
            [start.x - nx * half_t, start.y - ny * half_t],
            [0.0, 1.0], c, 0.0, 0.0, size, center, DRAW_FLAG_LINE,
        );

        self.push_quad(v0, v1, v2, v3, self.white_texture_index);
    }

    /// Draw text using the built-in bitmap font atlas.
    pub fn draw_text(
        &mut self,
        text: &str,
        position: Vec2,
        font_size: f32,
        color: Color,
    ) {
        let c = [color.r, color.g, color.b, color.a];
        let scale = font_size / Self::GLYPH_CELL_H as f32;
        let mut cursor_x = position.x;
        let cursor_y = position.y;

        for ch in text.chars() {
            let glyph = match self.builtin_glyphs.get(&ch) {
                Some(g) => *g,
                None => {
                    // Skip unprintable characters; advance for space.
                    cursor_x += font_size * 0.4;
                    continue;
                }
            };

            let gw = glyph.width * scale;
            let gh = glyph.height * scale;
            let x0 = cursor_x;
            let y0 = cursor_y + glyph.offset_y * scale;
            let x1 = x0 + gw;
            let y1 = y0 + gh;

            let size = [gw, gh];
            let center = [(x0 + x1) * 0.5, (y0 + y1) * 0.5];

            let v0 = self.make_vertex(
                [x0, y0], glyph.uv_min, c, 0.0, 0.0, size, center, DRAW_FLAG_TEXT,
            );
            let v1 = self.make_vertex(
                [x1, y0], [glyph.uv_max[0], glyph.uv_min[1]], c, 0.0, 0.0, size, center, DRAW_FLAG_TEXT,
            );
            let v2 = self.make_vertex(
                [x1, y1], glyph.uv_max, c, 0.0, 0.0, size, center, DRAW_FLAG_TEXT,
            );
            let v3 = self.make_vertex(
                [x0, y1], [glyph.uv_min[0], glyph.uv_max[1]], c, 0.0, 0.0, size, center, DRAW_FLAG_TEXT,
            );

            self.push_quad(v0, v1, v2, v3, self.font_atlas_index);
            cursor_x += glyph.advance * font_size;
        }
    }

    /// Draw a textured image quad with optional tint.
    pub fn draw_image(
        &mut self,
        texture_id: u64,
        rect: Rect,
        uv_rect: Rect,
        tint: Color,
    ) {
        let tex_idx = self.texture_id_map.get(&texture_id).copied()
            .unwrap_or(self.white_texture_index);

        let c = [tint.r, tint.g, tint.b, tint.a];
        let w = rect.width();
        let h = rect.height();
        let size = [w, h];
        let center = [(rect.min.x + rect.max.x) * 0.5, (rect.min.y + rect.max.y) * 0.5];

        let v0 = self.make_vertex(
            [rect.min.x, rect.min.y],
            [uv_rect.min.x, uv_rect.min.y],
            c, 0.0, 0.0, size, center, DRAW_FLAG_TEXTURED,
        );
        let v1 = self.make_vertex(
            [rect.max.x, rect.min.y],
            [uv_rect.max.x, uv_rect.min.y],
            c, 0.0, 0.0, size, center, DRAW_FLAG_TEXTURED,
        );
        let v2 = self.make_vertex(
            [rect.max.x, rect.max.y],
            [uv_rect.max.x, uv_rect.max.y],
            c, 0.0, 0.0, size, center, DRAW_FLAG_TEXTURED,
        );
        let v3 = self.make_vertex(
            [rect.min.x, rect.max.y],
            [uv_rect.min.x, uv_rect.max.y],
            c, 0.0, 0.0, size, center, DRAW_FLAG_TEXTURED,
        );

        self.push_quad(v0, v1, v2, v3, tex_idx);
    }

    // -----------------------------------------------------------------------
    // Text measurement
    // -----------------------------------------------------------------------

    /// Measure the width and height of a text string at the given font size
    /// using the built-in font atlas.
    pub fn measure_text(&self, text: &str, font_size: f32) -> (f32, f32) {
        let mut width = 0.0f32;
        let height = font_size;

        for ch in text.chars() {
            if let Some(glyph) = self.builtin_glyphs.get(&ch) {
                width += glyph.advance * font_size;
            } else {
                width += font_size * 0.4;
            }
        }

        (width, height)
    }

    /// Measure text width only.
    pub fn text_width(&self, text: &str, font_size: f32) -> f32 {
        self.measure_text(text, font_size).0
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Number of draw calls in the last submitted frame.
    pub fn draw_call_count(&self) -> u32 {
        self.frame_draw_calls
    }

    /// Number of triangles in the last submitted frame.
    pub fn triangle_count(&self) -> u32 {
        self.frame_triangles
    }

    /// Number of vertices in the last submitted frame.
    pub fn vertex_count(&self) -> u32 {
        self.frame_vertices
    }

    /// Current screen size.
    pub fn screen_size(&self) -> Vec2 {
        self.screen_size
    }

    /// The wgpu surface format this renderer was created for.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scissor_eq(a: &Option<ScissorRect>, b: &Option<ScissorRect>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(sa), Some(sb)) => {
            sa.x == sb.x && sa.y == sb.y && sa.width == sb.width && sa.height == sb.height
        }
        _ => false,
    }
}

