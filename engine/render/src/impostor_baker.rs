// engine/render/src/impostor_baker.rs
//
// Impostor (billboard) baking system for the Genovo engine.
//
// Generates impostor atlases for distant LOD rendering. An impostor captures
// an object's appearance from multiple viewing angles and stores them in a
// texture atlas. At runtime, the appropriate view is selected and rendered
// as a simple quad, dramatically reducing draw calls and vertex count for
// distant objects.
//
// # Supported impostor types
//
// - **Billboard** — Simple camera-facing quad with a single captured view.
// - **Cross-billboard** — Two or three intersecting quads.
// - **Octahedral** — Full sphere of views encoded in an octahedral map,
//   supporting viewing from any direction.
// - **Hemispherical** — Like octahedral but only the upper hemisphere (for
//   objects always viewed from above, e.g. trees).
//
// # Captured channels
//
// For each view direction, we capture:
// - Albedo (RGB + alpha for cutout)
// - World-space normal (for relighting)
// - Depth (for parallax and intersection)
//
// # Pipeline
//
// 1. Place a virtual camera at each required viewing angle.
// 2. Render the object to off-screen targets (albedo + normal + depth).
// 3. Pack all views into an atlas texture.
// 4. Generate the impostor mesh (billboard quad or octahedral sphere proxy).
// 5. At runtime, select the closest view(s) and blend.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Impostor type
// ---------------------------------------------------------------------------

/// Type of impostor rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpostorType {
    /// Simple camera-facing billboard with a single view.
    Billboard,
    /// Two intersecting billboard planes (90-degree cross).
    CrossBillboard,
    /// Three intersecting billboard planes (60-degree tri-cross).
    TriCross,
    /// Full octahedral mapping (views from all directions).
    Octahedral,
    /// Upper hemisphere only (views from above).
    Hemispherical,
}

impl ImpostorType {
    /// Returns the number of unique views captured for this type.
    pub fn view_count(&self, grid_size: u32) -> u32 {
        match self {
            Self::Billboard => 1,
            Self::CrossBillboard => 2,
            Self::TriCross => 3,
            Self::Octahedral => grid_size * grid_size,
            Self::Hemispherical => grid_size * (grid_size / 2 + 1),
        }
    }
}

// ---------------------------------------------------------------------------
// Impostor settings
// ---------------------------------------------------------------------------

/// Configuration for impostor baking.
#[derive(Debug, Clone)]
pub struct ImpostorSettings {
    /// Impostor type.
    pub impostor_type: ImpostorType,
    /// Resolution of each view capture (square).
    pub view_resolution: u32,
    /// Grid size for octahedral/hemispherical mapping.
    /// For octahedral: grid_size x grid_size views.
    pub grid_size: u32,
    /// Padding between views in the atlas (in texels).
    pub padding: u32,
    /// Whether to capture alpha (for cutout/transparency).
    pub capture_alpha: bool,
    /// Whether to capture normals.
    pub capture_normals: bool,
    /// Whether to capture depth.
    pub capture_depth: bool,
    /// Background alpha threshold for trimming.
    pub alpha_cutoff: f32,
    /// Whether to generate mipmaps for the atlas.
    pub generate_mipmaps: bool,
    /// Camera distance multiplier (relative to bounding sphere radius).
    pub camera_distance_factor: f32,
    /// Field of view for the impostor camera (radians).
    pub camera_fov: f32,
    /// Whether to use orthographic projection.
    pub orthographic: bool,
    /// LOD transition distance (world units).
    pub transition_start: f32,
    /// LOD transition end distance.
    pub transition_end: f32,
    /// Cross-fade duration (normalised).
    pub crossfade_width: f32,
}

impl ImpostorSettings {
    /// Creates default settings for octahedral impostors.
    pub fn octahedral() -> Self {
        Self {
            impostor_type: ImpostorType::Octahedral,
            view_resolution: 256,
            grid_size: 8,
            padding: 2,
            capture_alpha: true,
            capture_normals: true,
            capture_depth: true,
            alpha_cutoff: 0.1,
            generate_mipmaps: true,
            camera_distance_factor: 2.5,
            camera_fov: PI / 6.0, // 30 degrees.
            orthographic: true,
            transition_start: 100.0,
            transition_end: 120.0,
            crossfade_width: 0.1,
        }
    }

    /// Creates settings for simple billboard impostors.
    pub fn billboard() -> Self {
        Self {
            impostor_type: ImpostorType::Billboard,
            view_resolution: 512,
            grid_size: 1,
            padding: 0,
            capture_alpha: true,
            capture_normals: false,
            capture_depth: false,
            alpha_cutoff: 0.1,
            generate_mipmaps: true,
            camera_distance_factor: 2.5,
            camera_fov: PI / 6.0,
            orthographic: true,
            transition_start: 200.0,
            transition_end: 220.0,
            crossfade_width: 0.1,
        }
    }

    /// Creates settings for hemispherical impostors.
    pub fn hemispherical() -> Self {
        Self {
            impostor_type: ImpostorType::Hemispherical,
            view_resolution: 256,
            grid_size: 8,
            padding: 2,
            capture_alpha: true,
            capture_normals: true,
            capture_depth: true,
            alpha_cutoff: 0.1,
            generate_mipmaps: true,
            camera_distance_factor: 2.5,
            camera_fov: PI / 6.0,
            orthographic: true,
            transition_start: 80.0,
            transition_end: 100.0,
            crossfade_width: 0.1,
        }
    }

    /// Returns the atlas texture dimensions.
    pub fn atlas_dimensions(&self) -> (u32, u32) {
        let views_per_row = self.grid_size;
        let view_with_pad = self.view_resolution + self.padding * 2;

        match self.impostor_type {
            ImpostorType::Billboard => (self.view_resolution, self.view_resolution),
            ImpostorType::CrossBillboard => (view_with_pad * 2, view_with_pad),
            ImpostorType::TriCross => (view_with_pad * 3, view_with_pad),
            ImpostorType::Octahedral | ImpostorType::Hemispherical => {
                let w = views_per_row * view_with_pad;
                let rows = match self.impostor_type {
                    ImpostorType::Octahedral => views_per_row,
                    ImpostorType::Hemispherical => views_per_row / 2 + 1,
                    _ => views_per_row,
                };
                let h = rows * view_with_pad;
                (w, h)
            }
        }
    }

    /// Returns the total number of views.
    pub fn total_views(&self) -> u32 {
        self.impostor_type.view_count(self.grid_size)
    }
}

impl Default for ImpostorSettings {
    fn default() -> Self {
        Self::octahedral()
    }
}

// ---------------------------------------------------------------------------
// Octahedral mapping
// ---------------------------------------------------------------------------

/// Converts a 3D unit direction to 2D octahedral coordinates.
///
/// Maps the unit sphere to a [0, 1]^2 square via octahedral projection.
///
/// # Arguments
/// * `dir` — Normalised 3D direction.
///
/// # Returns
/// 2D octahedral coordinates in [0, 1].
pub fn direction_to_octahedral(dir: [f32; 3]) -> [f32; 2] {
    let abs_sum = dir[0].abs() + dir[1].abs() + dir[2].abs();
    let mut oct_x = dir[0] / abs_sum;
    let mut oct_z = dir[2] / abs_sum;

    // Unfold the bottom hemisphere.
    if dir[1] < 0.0 {
        let tmp_x = (1.0 - oct_z.abs()) * if oct_x >= 0.0 { 1.0 } else { -1.0 };
        let tmp_z = (1.0 - oct_x.abs()) * if oct_z >= 0.0 { 1.0 } else { -1.0 };
        oct_x = tmp_x;
        oct_z = tmp_z;
    }

    // Map from [-1, 1] to [0, 1].
    [oct_x * 0.5 + 0.5, oct_z * 0.5 + 0.5]
}

/// Converts 2D octahedral coordinates back to a 3D direction.
pub fn octahedral_to_direction(oct: [f32; 2]) -> [f32; 3] {
    // Map from [0, 1] to [-1, 1].
    let x = oct[0] * 2.0 - 1.0;
    let z = oct[1] * 2.0 - 1.0;

    let y = 1.0 - x.abs() - z.abs();

    let (x, z) = if y < 0.0 {
        let new_x = (1.0 - z.abs()) * if x >= 0.0 { 1.0 } else { -1.0 };
        let new_z = (1.0 - x.abs()) * if z >= 0.0 { 1.0 } else { -1.0 };
        (new_x, new_z)
    } else {
        (x, z)
    };

    let len = (x * x + y * y + z * z).sqrt();
    if len > 1e-6 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Computes the grid cell (row, column) for a given octahedral coordinate.
pub fn octahedral_to_grid(oct: [f32; 2], grid_size: u32) -> (u32, u32) {
    let col = (oct[0] * grid_size as f32).clamp(0.0, (grid_size - 1) as f32) as u32;
    let row = (oct[1] * grid_size as f32).clamp(0.0, (grid_size - 1) as f32) as u32;
    (col, row)
}

/// Returns the view direction for a given grid cell.
pub fn grid_to_direction(col: u32, row: u32, grid_size: u32) -> [f32; 3] {
    let oct = [
        (col as f32 + 0.5) / grid_size as f32,
        (row as f32 + 0.5) / grid_size as f32,
    ];
    octahedral_to_direction(oct)
}

// ---------------------------------------------------------------------------
// View capture
// ---------------------------------------------------------------------------

/// A single captured view of an object.
#[derive(Debug, Clone)]
pub struct ImpostorView {
    /// View direction (from camera to object centre).
    pub direction: [f32; 3],
    /// Grid position (column, row).
    pub grid_pos: (u32, u32),
    /// Atlas UV offset (top-left corner in atlas UV space).
    pub atlas_offset: [f32; 2],
    /// Atlas UV size (width, height in atlas UV space).
    pub atlas_size: [f32; 2],
    /// Captured albedo data (RGBA, row-major).
    pub albedo: Vec<[f32; 4]>,
    /// Captured normal data (RGB, row-major).
    pub normals: Vec<[f32; 3]>,
    /// Captured depth data (single channel, row-major).
    pub depth: Vec<f32>,
    /// View resolution.
    pub resolution: u32,
}

impl ImpostorView {
    /// Creates an empty view.
    pub fn new(direction: [f32; 3], grid_pos: (u32, u32), resolution: u32) -> Self {
        let total = (resolution * resolution) as usize;
        Self {
            direction,
            grid_pos,
            atlas_offset: [0.0, 0.0],
            atlas_size: [0.0, 0.0],
            albedo: vec![[0.0; 4]; total],
            normals: vec![[0.0, 0.0, 1.0]; total],
            depth: vec![1.0; total],
            resolution,
        }
    }

    /// Returns the memory usage of this view.
    pub fn memory_usage(&self) -> usize {
        self.albedo.len() * std::mem::size_of::<[f32; 4]>()
            + self.normals.len() * std::mem::size_of::<[f32; 3]>()
            + self.depth.len() * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Impostor atlas
// ---------------------------------------------------------------------------

/// A complete impostor atlas containing all captured views.
#[derive(Debug)]
pub struct ImpostorAtlas {
    /// Atlas width in pixels.
    pub width: u32,
    /// Atlas height in pixels.
    pub height: u32,
    /// Combined albedo atlas (RGBA).
    pub albedo: Vec<[f32; 4]>,
    /// Combined normal atlas (RGB).
    pub normals: Vec<[f32; 3]>,
    /// Combined depth atlas.
    pub depth: Vec<f32>,
    /// Grid size used for octahedral mapping.
    pub grid_size: u32,
    /// Individual view data.
    pub views: Vec<ImpostorView>,
    /// Bounding sphere radius of the source object.
    pub bounding_radius: f32,
    /// Centre of the source object.
    pub bounding_centre: [f32; 3],
    /// Settings used for baking.
    pub settings: ImpostorSettings,
}

impl ImpostorAtlas {
    /// Creates a new atlas from settings.
    pub fn new(settings: ImpostorSettings, bounding_radius: f32, bounding_centre: [f32; 3]) -> Self {
        let (width, height) = settings.atlas_dimensions();
        let total = (width * height) as usize;

        Self {
            width,
            height,
            albedo: vec![[0.0; 4]; total],
            normals: vec![[0.0, 0.0, 1.0]; total],
            depth: vec![1.0; total],
            grid_size: settings.grid_size,
            views: Vec::new(),
            bounding_radius,
            bounding_centre,
            settings,
        }
    }

    /// Generates view directions based on the impostor type and grid size.
    pub fn generate_view_directions(&self) -> Vec<([f32; 3], u32, u32)> {
        let mut views = Vec::new();

        match self.settings.impostor_type {
            ImpostorType::Billboard => {
                views.push(([0.0, 0.0, 1.0], 0, 0));
            }
            ImpostorType::CrossBillboard => {
                views.push(([0.0, 0.0, 1.0], 0, 0));
                views.push(([1.0, 0.0, 0.0], 1, 0));
            }
            ImpostorType::TriCross => {
                views.push(([0.0, 0.0, 1.0], 0, 0));
                let angle = PI * 2.0 / 3.0;
                views.push(([angle.cos(), 0.0, angle.sin()], 1, 0));
                let angle2 = angle * 2.0;
                views.push(([angle2.cos(), 0.0, angle2.sin()], 2, 0));
            }
            ImpostorType::Octahedral => {
                let gs = self.settings.grid_size;
                for row in 0..gs {
                    for col in 0..gs {
                        let dir = grid_to_direction(col, row, gs);
                        views.push((dir, col, row));
                    }
                }
            }
            ImpostorType::Hemispherical => {
                let gs = self.settings.grid_size;
                let half_gs = gs / 2 + 1;
                for row in 0..half_gs {
                    for col in 0..gs {
                        let oct = [
                            (col as f32 + 0.5) / gs as f32,
                            (row as f32 + 0.5) / gs as f32 * 0.5 + 0.5,
                        ];
                        let dir = octahedral_to_direction(oct);
                        views.push((dir, col, row));
                    }
                }
            }
        }

        views
    }

    /// Computes the atlas UV offset for a given grid position.
    pub fn view_atlas_offset(&self, col: u32, row: u32) -> [f32; 2] {
        let view_with_pad = self.settings.view_resolution + self.settings.padding * 2;
        let px = col * view_with_pad + self.settings.padding;
        let py = row * view_with_pad + self.settings.padding;
        [px as f32 / self.width as f32, py as f32 / self.height as f32]
    }

    /// Computes the atlas UV size for a single view.
    pub fn view_atlas_size(&self) -> [f32; 2] {
        [
            self.settings.view_resolution as f32 / self.width as f32,
            self.settings.view_resolution as f32 / self.height as f32,
        ]
    }

    /// Copies a captured view into the atlas.
    pub fn copy_view_to_atlas(&mut self, view: &ImpostorView) {
        let view_with_pad = self.settings.view_resolution + self.settings.padding * 2;
        let atlas_px = view.grid_pos.0 * view_with_pad + self.settings.padding;
        let atlas_py = view.grid_pos.1 * view_with_pad + self.settings.padding;
        let vr = view.resolution;

        for vy in 0..vr {
            for vx in 0..vr {
                let view_idx = (vy * vr + vx) as usize;
                let atlas_x = atlas_px + vx;
                let atlas_y = atlas_py + vy;

                if atlas_x >= self.width || atlas_y >= self.height {
                    continue;
                }

                let atlas_idx = (atlas_y * self.width + atlas_x) as usize;

                if view_idx < view.albedo.len() {
                    self.albedo[atlas_idx] = view.albedo[view_idx];
                }
                if view_idx < view.normals.len() {
                    self.normals[atlas_idx] = view.normals[view_idx];
                }
                if view_idx < view.depth.len() {
                    self.depth[atlas_idx] = view.depth[view_idx];
                }
            }
        }
    }

    /// Samples the atlas at a given view direction.
    ///
    /// For octahedral impostors, this maps the direction to octahedral
    /// coordinates and samples the corresponding region.
    pub fn sample_direction(&self, dir: [f32; 3]) -> [f32; 4] {
        let oct = direction_to_octahedral(dir);
        let (col, row) = octahedral_to_grid(oct, self.grid_size);
        let offset = self.view_atlas_offset(col, row);
        let size = self.view_atlas_size();

        // Sample the centre of the view.
        let px = ((offset[0] + size[0] * 0.5) * self.width as f32) as u32;
        let py = ((offset[1] + size[1] * 0.5) * self.height as f32) as u32;
        let px = px.min(self.width - 1);
        let py = py.min(self.height - 1);
        let idx = (py * self.width + px) as usize;

        self.albedo.get(idx).copied().unwrap_or([0.0; 4])
    }

    /// Returns the memory usage of the atlas in bytes.
    pub fn memory_usage(&self) -> usize {
        let total = (self.width * self.height) as usize;
        total * std::mem::size_of::<[f32; 4]>() // albedo
            + total * std::mem::size_of::<[f32; 3]>() // normals
            + total * std::mem::size_of::<f32>() // depth
    }

    /// Returns the total view count.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
}

// ---------------------------------------------------------------------------
// Impostor mesh generation
// ---------------------------------------------------------------------------

/// Vertex for an impostor mesh.
#[derive(Debug, Clone, Copy)]
pub struct ImpostorVertex {
    /// Position.
    pub position: [f32; 3],
    /// UV coordinate into the atlas.
    pub uv: [f32; 2],
    /// Frame data (for runtime view interpolation).
    pub frame_data: [f32; 4],
}

/// Generates a billboard quad mesh for an impostor.
///
/// # Arguments
/// * `centre` — World-space centre of the impostor.
/// * `radius` — Half-size of the billboard quad.
///
/// # Returns
/// Vertices and indices for the quad.
pub fn generate_billboard_mesh(
    centre: [f32; 3],
    radius: f32,
) -> (Vec<ImpostorVertex>, Vec<u32>) {
    let vertices = vec![
        ImpostorVertex {
            position: [centre[0] - radius, centre[1] - radius, centre[2]],
            uv: [0.0, 1.0],
            frame_data: [0.0; 4],
        },
        ImpostorVertex {
            position: [centre[0] + radius, centre[1] - radius, centre[2]],
            uv: [1.0, 1.0],
            frame_data: [0.0; 4],
        },
        ImpostorVertex {
            position: [centre[0] + radius, centre[1] + radius, centre[2]],
            uv: [1.0, 0.0],
            frame_data: [0.0; 4],
        },
        ImpostorVertex {
            position: [centre[0] - radius, centre[1] + radius, centre[2]],
            uv: [0.0, 0.0],
            frame_data: [0.0; 4],
        },
    ];

    let indices = vec![0, 1, 2, 0, 2, 3];

    (vertices, indices)
}

/// Generates a sphere proxy mesh for octahedral impostors.
///
/// # Arguments
/// * `centre` — World-space centre.
/// * `radius` — Sphere radius.
/// * `segments` — Number of segments (latitude and longitude).
///
/// # Returns
/// Vertices and indices for the sphere.
pub fn generate_impostor_sphere(
    centre: [f32; 3],
    radius: f32,
    segments: u32,
) -> (Vec<ImpostorVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let rings = segments;
    let sectors = segments;

    for r in 0..=rings {
        let theta = PI * r as f32 / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for s in 0..=sectors {
            let phi = 2.0 * PI * s as f32 / sectors as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = cos_phi * sin_theta;
            let y = cos_theta;
            let z = sin_phi * sin_theta;

            let oct = direction_to_octahedral([x, y, z]);

            vertices.push(ImpostorVertex {
                position: [
                    centre[0] + x * radius,
                    centre[1] + y * radius,
                    centre[2] + z * radius,
                ],
                uv: oct,
                frame_data: [x, y, z, 0.0],
            });
        }
    }

    for r in 0..rings {
        for s in 0..sectors {
            let first = r * (sectors + 1) + s;
            let second = first + sectors + 1;

            indices.push(first);
            indices.push(second);
            indices.push(first + 1);

            indices.push(second);
            indices.push(second + 1);
            indices.push(first + 1);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// LOD transition
// ---------------------------------------------------------------------------

/// Computes the LOD transition blend factor for impostor rendering.
///
/// # Arguments
/// * `distance` — Distance from camera to object.
/// * `transition_start` — Distance at which transition begins.
/// * `transition_end` — Distance at which transition completes.
///
/// # Returns
/// Blend factor: 0.0 = full mesh, 1.0 = full impostor.
pub fn impostor_transition_factor(
    distance: f32,
    transition_start: f32,
    transition_end: f32,
) -> f32 {
    if distance <= transition_start {
        0.0
    } else if distance >= transition_end {
        1.0
    } else {
        (distance - transition_start) / (transition_end - transition_start)
    }
}

// ---------------------------------------------------------------------------
// WGSL impostor shader
// ---------------------------------------------------------------------------

/// WGSL shader for rendering octahedral impostors.
pub const IMPOSTOR_RENDER_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Impostor rendering shader (Genovo Engine)
// -----------------------------------------------------------------------

struct ImpostorUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    grid_size: f32,
    impostor_center: vec3<f32>,
    impostor_radius: f32,
    sun_direction: vec3<f32>,
    alpha_cutoff: f32,
};

@group(0) @binding(0) var<uniform> imp: ImpostorUniforms;
@group(0) @binding(1) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(2) var normal_atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) view_dir: vec3<f32>,
};

fn dir_to_octahedral(dir: vec3<f32>) -> vec2<f32> {
    let abs_sum = abs(dir.x) + abs(dir.y) + abs(dir.z);
    var oct = dir.xz / abs_sum;
    if dir.y < 0.0 {
        let signs = vec2<f32>(select(-1.0, 1.0, oct.x >= 0.0), select(-1.0, 1.0, oct.y >= 0.0));
        oct = (1.0 - abs(oct.yx)) * signs;
    }
    return oct * 0.5 + 0.5;
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) frame_data: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = imp.view_proj * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.world_pos = position;
    out.view_dir = normalize(imp.camera_pos - position);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(input.view_dir);
    let oct = dir_to_octahedral(view_dir);

    let grid = imp.grid_size;
    let cell = floor(oct * grid) / grid;
    let atlas_uv = cell + fract(input.uv * grid) / grid;

    let albedo = textureSample(albedo_atlas, atlas_sampler, atlas_uv);

    if albedo.a < imp.alpha_cutoff {
        discard;
    }

    let normal_sample = textureSample(normal_atlas, atlas_sampler, atlas_uv).rgb;
    let normal = normalize(normal_sample * 2.0 - 1.0);

    let n_dot_l = max(dot(normal, imp.sun_direction), 0.0);
    let lit_color = albedo.rgb * (n_dot_l * 0.8 + 0.2);

    return vec4<f32>(lit_color, albedo.a);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octahedral_roundtrip() {
        let directions = [
            [0.0, 1.0, 0.0],   // Up.
            [0.0, -1.0, 0.0],  // Down.
            [1.0, 0.0, 0.0],   // Right.
            [0.0, 0.0, 1.0],   // Forward.
            [0.577, 0.577, 0.577], // Diagonal.
        ];

        for dir in &directions {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let norm = [dir[0] / len, dir[1] / len, dir[2] / len];

            let oct = direction_to_octahedral(norm);
            assert!(oct[0] >= 0.0 && oct[0] <= 1.0, "Oct X out of range: {}", oct[0]);
            assert!(oct[1] >= 0.0 && oct[1] <= 1.0, "Oct Y out of range: {}", oct[1]);

            let recovered = octahedral_to_direction(oct);
            let dot = norm[0] * recovered[0] + norm[1] * recovered[1] + norm[2] * recovered[2];
            assert!(
                dot > 0.95,
                "Roundtrip failed for {:?}: recovered {:?}, dot = {}",
                norm,
                recovered,
                dot
            );
        }
    }

    #[test]
    fn test_atlas_dimensions() {
        let settings = ImpostorSettings::octahedral();
        let (w, h) = settings.atlas_dimensions();
        assert!(w > 0);
        assert!(h > 0);
        assert_eq!(w, h); // Octahedral is square.
    }

    #[test]
    fn test_billboard_dimensions() {
        let settings = ImpostorSettings::billboard();
        let (w, h) = settings.atlas_dimensions();
        assert_eq!(w, 512);
        assert_eq!(h, 512);
    }

    #[test]
    fn test_view_count() {
        assert_eq!(ImpostorType::Billboard.view_count(8), 1);
        assert_eq!(ImpostorType::CrossBillboard.view_count(8), 2);
        assert_eq!(ImpostorType::Octahedral.view_count(8), 64);
    }

    #[test]
    fn test_generate_view_directions() {
        let atlas = ImpostorAtlas::new(
            ImpostorSettings::octahedral(),
            5.0,
            [0.0; 3],
        );
        let views = atlas.generate_view_directions();
        assert_eq!(views.len(), 64);

        // All directions should be unit vectors.
        for (dir, _, _) in &views {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 0.01, "Direction not normalised: len = {len}");
        }
    }

    #[test]
    fn test_billboard_mesh() {
        let (verts, indices) = generate_billboard_mesh([0.0, 5.0, 0.0], 2.0);
        assert_eq!(verts.len(), 4);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_impostor_sphere() {
        let (verts, indices) = generate_impostor_sphere([0.0, 0.0, 0.0], 1.0, 4);
        assert!(verts.len() > 0);
        assert!(indices.len() > 0);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_transition_factor() {
        assert!((impostor_transition_factor(50.0, 100.0, 120.0) - 0.0).abs() < 0.01);
        assert!((impostor_transition_factor(150.0, 100.0, 120.0) - 1.0).abs() < 0.01);
        assert!((impostor_transition_factor(110.0, 100.0, 120.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_atlas_memory() {
        let atlas = ImpostorAtlas::new(ImpostorSettings::octahedral(), 5.0, [0.0; 3]);
        assert!(atlas.memory_usage() > 0);
    }

    #[test]
    fn test_copy_view_to_atlas() {
        let settings = ImpostorSettings {
            view_resolution: 4,
            grid_size: 2,
            padding: 0,
            ..ImpostorSettings::octahedral()
        };
        let mut atlas = ImpostorAtlas::new(settings, 1.0, [0.0; 3]);

        let mut view = ImpostorView::new([0.0, 0.0, 1.0], (0, 0), 4);
        view.albedo = vec![[1.0, 0.0, 0.0, 1.0]; 16];

        atlas.copy_view_to_atlas(&view);

        // Check that the top-left 4x4 block has red.
        let idx = 0;
        assert!((atlas.albedo[idx][0] - 1.0).abs() < 0.01);
    }
}
