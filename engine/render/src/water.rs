// engine/render/src/water.rs
//
// Real-time water rendering for the Genovo engine.
//
// Provides inland water surfaces (lakes, rivers, ponds) with:
//
// - Sine-wave vertex displacement (4 overlapping waves)
// - Planar reflection via reflected camera matrix
// - Refraction: scene behind the water plane
// - Fresnel blend between reflection and refraction
// - Animated caustic texture projection onto underwater surfaces
// - Shore foam at terrain intersection based on depth
// - ECS component for attaching water planes to entities
//
// For deep ocean rendering with FFT-based waves, see the `ocean` module.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// WaveParameters
// ---------------------------------------------------------------------------

/// Parameters for a single sine wave used in vertex displacement.
#[derive(Debug, Clone, Copy)]
pub struct WaveParameters {
    /// Direction of wave travel (normalised XZ vector).
    pub direction: Vec2,
    /// Wavelength in world units.
    pub wavelength: f32,
    /// Amplitude (peak displacement from rest).
    pub amplitude: f32,
    /// Speed in world units per second.
    pub speed: f32,
    /// Steepness for Gerstner-style displacement (0..1).
    /// 0 = pure sine, 1 = sharp crest.
    pub steepness: f32,
}

impl WaveParameters {
    /// Creates a new wave with the given properties.
    pub fn new(direction: Vec2, wavelength: f32, amplitude: f32, speed: f32) -> Self {
        Self {
            direction: direction.normalize_or_zero(),
            wavelength,
            amplitude,
            speed,
            steepness: 0.0,
        }
    }

    /// Sets the Gerstner steepness.
    pub fn with_steepness(mut self, steepness: f32) -> Self {
        self.steepness = steepness.clamp(0.0, 1.0);
        self
    }

    /// Angular frequency: `2 * PI / wavelength`.
    #[inline]
    pub fn frequency(&self) -> f32 {
        2.0 * PI / self.wavelength
    }

    /// Phase speed: `speed * frequency`.
    #[inline]
    pub fn phase(&self) -> f32 {
        self.speed * self.frequency()
    }

    /// Evaluates the vertical displacement at a world-space XZ position
    /// and time, returning `(displacement_y, dx, dz)`.
    pub fn evaluate(&self, world_xz: Vec2, time: f32) -> WaveDisplacement {
        let w = self.frequency();
        let phase = self.phase();
        let dot = self.direction.dot(world_xz);
        let theta = w * dot - phase * time;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        let y = self.amplitude * sin_t;

        // Gerstner horizontal displacement
        let q = self.steepness / (w * self.amplitude * 4.0).max(0.001);
        let dx = q * self.amplitude * self.direction.x * cos_t;
        let dz = q * self.amplitude * self.direction.y * cos_t;

        // Partial derivatives for normal calculation
        let dy_dx = w * self.direction.x * self.amplitude * cos_t;
        let dy_dz = w * self.direction.y * self.amplitude * cos_t;

        WaveDisplacement {
            position_offset: Vec3::new(dx, y, dz),
            normal_dx: dy_dx,
            normal_dz: dy_dz,
        }
    }
}

impl Default for WaveParameters {
    fn default() -> Self {
        Self {
            direction: Vec2::new(1.0, 0.0),
            wavelength: 10.0,
            amplitude: 0.2,
            speed: 1.0,
            steepness: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// WaveDisplacement
// ---------------------------------------------------------------------------

/// Result of evaluating a single wave at a point.
#[derive(Debug, Clone, Copy)]
pub struct WaveDisplacement {
    /// Position offset (Gerstner x, sine y, Gerstner z).
    pub position_offset: Vec3,
    /// Partial derivative dY/dX for normal reconstruction.
    pub normal_dx: f32,
    /// Partial derivative dY/dZ for normal reconstruction.
    pub normal_dz: f32,
}

impl WaveDisplacement {
    /// Zero displacement.
    pub const ZERO: Self = Self {
        position_offset: Vec3::ZERO,
        normal_dx: 0.0,
        normal_dz: 0.0,
    };
}

// ---------------------------------------------------------------------------
// WaveSet
// ---------------------------------------------------------------------------

/// A set of 4 overlapping waves for multi-frequency water displacement.
///
/// Using 4 waves with different frequencies, directions, and amplitudes
/// creates visually convincing water motion without the cost of FFT.
#[derive(Debug, Clone)]
pub struct WaveSet {
    pub waves: [WaveParameters; 4],
}

impl WaveSet {
    /// Creates a wave set with 4 configured waves.
    pub fn new(waves: [WaveParameters; 4]) -> Self {
        Self { waves }
    }

    /// Creates a calm lake wave set.
    pub fn calm_lake() -> Self {
        Self {
            waves: [
                WaveParameters::new(Vec2::new(1.0, 0.0), 15.0, 0.05, 0.8),
                WaveParameters::new(Vec2::new(0.7, 0.7), 8.0, 0.03, 1.2),
                WaveParameters::new(Vec2::new(-0.3, 1.0), 5.0, 0.02, 0.6),
                WaveParameters::new(Vec2::new(0.5, -0.5), 3.0, 0.01, 1.5),
            ],
        }
    }

    /// Creates a flowing river wave set.
    pub fn river() -> Self {
        Self {
            waves: [
                WaveParameters::new(Vec2::new(1.0, 0.0), 8.0, 0.15, 2.0)
                    .with_steepness(0.3),
                WaveParameters::new(Vec2::new(0.9, 0.3), 4.0, 0.08, 2.5)
                    .with_steepness(0.2),
                WaveParameters::new(Vec2::new(1.0, -0.2), 2.5, 0.04, 3.0)
                    .with_steepness(0.1),
                WaveParameters::new(Vec2::new(0.8, 0.5), 1.5, 0.02, 1.8),
            ],
        }
    }

    /// Creates a choppy pond wave set (e.g. windy conditions).
    pub fn choppy() -> Self {
        Self {
            waves: [
                WaveParameters::new(Vec2::new(1.0, 0.3), 6.0, 0.25, 1.5)
                    .with_steepness(0.5),
                WaveParameters::new(Vec2::new(-0.5, 1.0), 4.0, 0.15, 2.0)
                    .with_steepness(0.4),
                WaveParameters::new(Vec2::new(0.3, -0.8), 2.0, 0.08, 2.5)
                    .with_steepness(0.3),
                WaveParameters::new(Vec2::new(-0.7, -0.5), 1.2, 0.04, 3.0)
                    .with_steepness(0.2),
            ],
        }
    }

    /// Evaluates all 4 waves at the given world-space XZ position and time.
    /// Returns the combined displacement and the reconstructed surface normal.
    pub fn evaluate(&self, world_xz: Vec2, time: f32) -> WaveResult {
        let mut total_offset = Vec3::ZERO;
        let mut total_dx = 0.0;
        let mut total_dz = 0.0;

        for wave in &self.waves {
            let d = wave.evaluate(world_xz, time);
            total_offset += d.position_offset;
            total_dx += d.normal_dx;
            total_dz += d.normal_dz;
        }

        // Reconstruct normal from accumulated partial derivatives
        let normal = Vec3::new(-total_dx, 1.0, -total_dz).normalize();

        WaveResult {
            displacement: total_offset,
            normal,
        }
    }

    /// Returns the maximum possible displacement amplitude (sum of all wave
    /// amplitudes). Useful for culling and bounding-box calculations.
    pub fn max_amplitude(&self) -> f32 {
        self.waves.iter().map(|w| w.amplitude).sum()
    }
}

impl Default for WaveSet {
    fn default() -> Self {
        Self::calm_lake()
    }
}

// ---------------------------------------------------------------------------
// WaveResult
// ---------------------------------------------------------------------------

/// Combined result of evaluating all waves at a point.
#[derive(Debug, Clone, Copy)]
pub struct WaveResult {
    /// Total position offset from the rest plane.
    pub displacement: Vec3,
    /// Reconstructed surface normal.
    pub normal: Vec3,
}

// ---------------------------------------------------------------------------
// ReflectionCamera
// ---------------------------------------------------------------------------

/// Computes a reflected camera for planar reflection rendering.
///
/// The reflected camera matrix mirrors the main camera across the water
/// plane, and generates an oblique near-clip plane to avoid rendering
/// geometry below the water surface.
#[derive(Debug, Clone)]
pub struct ReflectionCamera;

impl ReflectionCamera {
    /// Computes the reflection matrix for a horizontal water plane at the
    /// given Y height.
    ///
    /// The reflection matrix mirrors geometry across `y = water_y`.
    pub fn reflection_matrix(water_y: f32) -> Mat4 {
        // Reflection across y = water_y:
        // [ 1  0  0  0         ]
        // [ 0 -1  0  2*water_y ]
        // [ 0  0  1  0         ]
        // [ 0  0  0  1         ]
        Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 2.0 * water_y, 0.0, 1.0),
        )
    }

    /// Computes the reflected view matrix from the main camera's view matrix.
    ///
    /// `view` is the main camera's view matrix (world-to-camera).
    /// `water_y` is the world-space Y coordinate of the water plane.
    pub fn reflected_view_matrix(view: Mat4, water_y: f32) -> Mat4 {
        let reflection = Self::reflection_matrix(water_y);
        view * reflection
    }

    /// Computes an oblique near-clip projection matrix.
    ///
    /// This modifies the projection matrix so that the near plane coincides
    /// with the water plane, preventing underwater geometry from appearing
    /// in the reflection.
    pub fn oblique_projection(
        projection: Mat4,
        clip_plane: Vec4,
    ) -> Mat4 {
        let mut proj = projection;

        // Transform clip plane into clip space
        let inv_proj = proj.inverse();
        let q = inv_proj.transpose() * clip_plane;

        // Scale the clip plane so it replaces the near plane
        let c = clip_plane * (2.0 / q.dot(Vec4::new(0.0, 0.0, 1.0, 1.0)));

        // Replace the third row of the projection matrix
        proj.col_mut(0).z = c.x;
        proj.col_mut(1).z = c.y;
        proj.col_mut(2).z = c.z + 1.0;
        proj.col_mut(3).z = c.w;

        proj
    }

    /// Returns the clip plane in world space for a horizontal water plane.
    ///
    /// `water_y`: Y coordinate of the water surface.
    /// `above`: if true, clip below water (for reflection); if false, clip above.
    pub fn clip_plane(water_y: f32, above: bool) -> Vec4 {
        let sign = if above { 1.0 } else { -1.0 };
        Vec4::new(0.0, sign, 0.0, -sign * water_y)
    }
}

// ---------------------------------------------------------------------------
// FresnelSettings
// ---------------------------------------------------------------------------

/// Fresnel effect parameters for blending reflection and refraction.
#[derive(Debug, Clone, Copy)]
pub struct FresnelSettings {
    /// Base reflectivity at normal incidence (typically 0.02 for water).
    pub f0: f32,
    /// Power exponent for the Fresnel curve (higher = sharper falloff).
    pub power: f32,
    /// Bias added to the Fresnel term.
    pub bias: f32,
}

impl Default for FresnelSettings {
    fn default() -> Self {
        Self {
            f0: 0.02,
            power: 5.0,
            bias: 0.0,
        }
    }
}

impl FresnelSettings {
    /// Evaluates the Fresnel reflectance for a given view-normal dot product.
    ///
    /// `cos_theta`: dot product of the view direction and surface normal,
    /// clamped to [0, 1].
    pub fn evaluate(&self, cos_theta: f32) -> f32 {
        let t = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
        let fresnel = self.f0 + (1.0 - self.f0) * t.powf(self.power);
        (fresnel + self.bias).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// CausticSettings
// ---------------------------------------------------------------------------

/// Animated caustic effect projected onto underwater surfaces.
#[derive(Debug, Clone)]
pub struct CausticSettings {
    /// Whether caustics are enabled.
    pub enabled: bool,
    /// Intensity multiplier for the caustic brightness.
    pub intensity: f32,
    /// World-space scale of the caustic texture tiling.
    pub scale: f32,
    /// Speed of caustic animation (texture scroll per second).
    pub speed: f32,
    /// Maximum depth below water at which caustics are visible.
    pub max_depth: f32,
    /// Fade distance: caustics fade out between `max_depth - fade` and `max_depth`.
    pub fade_distance: f32,
    /// Color tint for caustics.
    pub tint: Vec3,
    /// Second caustic layer offset for blending two patterns.
    pub dual_layer_offset: Vec2,
    /// Second layer speed multiplier (relative to primary speed).
    pub dual_layer_speed_ratio: f32,
}

impl Default for CausticSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 0.5,
            scale: 5.0,
            speed: 0.3,
            max_depth: 10.0,
            fade_distance: 3.0,
            tint: Vec3::new(0.8, 0.9, 1.0),
            dual_layer_offset: Vec2::new(0.37, 0.21),
            dual_layer_speed_ratio: -0.7,
        }
    }
}

impl CausticSettings {
    /// Computes the caustic UV offset for a given time.
    pub fn uv_offset(&self, time: f32) -> (Vec2, Vec2) {
        let primary = Vec2::new(time * self.speed, time * self.speed * 0.7);
        let secondary = self.dual_layer_offset
            + Vec2::new(
                time * self.speed * self.dual_layer_speed_ratio,
                time * self.speed * self.dual_layer_speed_ratio * 0.8,
            );
        (primary, secondary)
    }

    /// Computes the depth-based attenuation factor.
    ///
    /// `depth`: distance below the water surface (positive downward).
    pub fn depth_attenuation(&self, depth: f32) -> f32 {
        if depth < 0.0 || depth > self.max_depth {
            return 0.0;
        }
        let fade_start = self.max_depth - self.fade_distance;
        if depth < fade_start {
            1.0
        } else {
            1.0 - (depth - fade_start) / self.fade_distance
        }
    }
}

// ---------------------------------------------------------------------------
// ShoreFoamSettings
// ---------------------------------------------------------------------------

/// Shore foam rendering where water meets terrain.
#[derive(Debug, Clone, Copy)]
pub struct ShoreFoamSettings {
    /// Whether foam is enabled.
    pub enabled: bool,
    /// Depth threshold: foam appears where scene depth is within this
    /// distance of the water surface.
    pub depth_threshold: f32,
    /// Foam texture tiling scale.
    pub scale: f32,
    /// Foam scroll speed.
    pub speed: f32,
    /// Foam intensity.
    pub intensity: f32,
    /// Edge softness (higher = softer foam edge falloff).
    pub softness: f32,
    /// Color of the foam.
    pub color: Vec3,
    /// Distortion amount for foam edge noise.
    pub edge_distortion: f32,
}

impl Default for ShoreFoamSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            depth_threshold: 1.0,
            scale: 8.0,
            speed: 0.2,
            intensity: 0.8,
            softness: 0.5,
            color: Vec3::new(1.0, 1.0, 1.0),
            edge_distortion: 0.1,
        }
    }
}

impl ShoreFoamSettings {
    /// Computes the foam factor based on the depth difference between
    /// the water surface and the scene behind it.
    ///
    /// `depth_diff`: distance from water surface to scene geometry (positive =
    /// geometry is below water). Returns a 0..1 foam factor.
    pub fn compute_foam(&self, depth_diff: f32, time: f32) -> f32 {
        if !self.enabled || depth_diff > self.depth_threshold || depth_diff < 0.0 {
            return 0.0;
        }

        let t = depth_diff / self.depth_threshold;
        let edge = (1.0 - t).powf(self.softness);

        // Animated wave-like pulsing
        let pulse = ((time * self.speed * PI * 2.0).sin() * 0.5 + 0.5) * 0.3 + 0.7;

        (edge * self.intensity * pulse).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// WaterPlane
// ---------------------------------------------------------------------------

/// A finite water plane in the scene with full rendering parameters.
#[derive(Debug, Clone)]
pub struct WaterPlane {
    /// World-space center position of the water plane.
    pub position: Vec3,
    /// Size of the water plane (width, depth) in world units.
    pub size: Vec2,
    /// Rotation of the water plane (around Y axis for river direction, etc.).
    pub rotation: f32,
    /// Base water color (deep water tint).
    pub deep_color: Vec3,
    /// Shallow water color (near shoreline).
    pub shallow_color: Vec3,
    /// Depth at which color transitions from shallow to deep.
    pub color_depth: f32,
    /// Water transparency / opacity (0 = invisible, 1 = opaque).
    pub opacity: f32,
    /// Index of refraction (typically ~1.33 for water).
    pub ior: f32,
    /// Specular highlight intensity.
    pub specular_intensity: f32,
    /// Specular roughness (lower = sharper highlights).
    pub specular_roughness: f32,
    /// Wave configuration.
    pub waves: WaveSet,
    /// Fresnel effect settings.
    pub fresnel: FresnelSettings,
    /// Caustic projection settings.
    pub caustics: CausticSettings,
    /// Shore foam settings.
    pub foam: ShoreFoamSettings,
    /// Normal map texture tiling scale.
    pub normal_map_scale: f32,
    /// Normal map strength (0 = flat, 1 = full detail).
    pub normal_map_strength: f32,
    /// Normal map scroll speed.
    pub normal_map_speed: Vec2,
    /// Whether to render reflection for this water plane.
    pub reflection_enabled: bool,
    /// Whether to render refraction for this water plane.
    pub refraction_enabled: bool,
    /// Reflection texture resolution scale relative to the main viewport.
    pub reflection_resolution_scale: f32,
    /// Distance-based fade: water becomes invisible beyond this distance.
    pub fade_distance: f32,
    /// Number of tessellation subdivisions for the water mesh.
    pub subdivisions: u32,
}

impl Default for WaterPlane {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            size: Vec2::new(100.0, 100.0),
            rotation: 0.0,
            deep_color: Vec3::new(0.02, 0.08, 0.15),
            shallow_color: Vec3::new(0.1, 0.4, 0.5),
            color_depth: 5.0,
            opacity: 0.85,
            ior: 1.33,
            specular_intensity: 1.0,
            specular_roughness: 0.1,
            waves: WaveSet::default(),
            fresnel: FresnelSettings::default(),
            caustics: CausticSettings::default(),
            foam: ShoreFoamSettings::default(),
            normal_map_scale: 10.0,
            normal_map_strength: 0.5,
            normal_map_speed: Vec2::new(0.02, 0.01),
            reflection_enabled: true,
            refraction_enabled: true,
            reflection_resolution_scale: 0.5,
            fade_distance: 500.0,
            subdivisions: 64,
        }
    }
}

impl WaterPlane {
    /// Creates a new water plane at the given position with the given size.
    pub fn new(position: Vec3, size: Vec2) -> Self {
        Self {
            position,
            size,
            ..Default::default()
        }
    }

    /// Sets the water colors.
    pub fn with_colors(mut self, deep: Vec3, shallow: Vec3) -> Self {
        self.deep_color = deep;
        self.shallow_color = shallow;
        self
    }

    /// Sets the wave configuration.
    pub fn with_waves(mut self, waves: WaveSet) -> Self {
        self.waves = waves;
        self
    }

    /// Sets the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Sets the tessellation subdivisions.
    pub fn with_subdivisions(mut self, subdivisions: u32) -> Self {
        self.subdivisions = subdivisions.max(1);
        self
    }

    /// Returns the water Y level (world-space height of the surface at rest).
    pub fn water_y(&self) -> f32 {
        self.position.y
    }

    /// Returns the world-space AABB of the water plane including maximum
    /// wave displacement.
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let max_amp = self.waves.max_amplitude();
        let half = Vec3::new(self.size.x * 0.5, max_amp, self.size.y * 0.5);
        (self.position - half, self.position + half)
    }

    /// Evaluates the water surface at a world-space XZ position.
    pub fn sample_surface(&self, world_xz: Vec2, time: f32) -> WaveResult {
        self.waves.evaluate(world_xz, time)
    }

    /// Evaluates the water height at a world-space XZ position.
    pub fn height_at(&self, world_xz: Vec2, time: f32) -> f32 {
        self.position.y + self.waves.evaluate(world_xz, time).displacement.y
    }

    /// Computes the depth-based color at a point.
    ///
    /// `depth`: how deep the scene geometry is below the water surface.
    pub fn depth_color(&self, depth: f32) -> Vec3 {
        let t = (depth / self.color_depth).clamp(0.0, 1.0);
        Vec3::new(
            self.shallow_color.x + (self.deep_color.x - self.shallow_color.x) * t,
            self.shallow_color.y + (self.deep_color.y - self.shallow_color.y) * t,
            self.shallow_color.z + (self.deep_color.z - self.shallow_color.z) * t,
        )
    }

    /// Returns the reflection view matrix for this water plane.
    pub fn reflection_view_matrix(&self, camera_view: Mat4) -> Mat4 {
        ReflectionCamera::reflected_view_matrix(camera_view, self.water_y())
    }

    /// Returns the clip plane for reflection rendering.
    pub fn reflection_clip_plane(&self) -> Vec4 {
        ReflectionCamera::clip_plane(self.water_y(), true)
    }

    /// Returns the clip plane for refraction rendering.
    pub fn refraction_clip_plane(&self) -> Vec4 {
        ReflectionCamera::clip_plane(self.water_y(), false)
    }

    /// Generates the water mesh vertex grid.
    ///
    /// Returns `(positions, uvs, indices)` for a subdivided quad centered
    /// at the water plane's position.
    pub fn generate_mesh(&self) -> WaterMeshData {
        let n = self.subdivisions as usize + 1;
        let total_verts = n * n;
        let total_indices = (n - 1) * (n - 1) * 6;

        let mut positions = Vec::with_capacity(total_verts);
        let mut uvs = Vec::with_capacity(total_verts);
        let mut indices = Vec::with_capacity(total_indices);

        let half_w = self.size.x * 0.5;
        let half_d = self.size.y * 0.5;

        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        for z in 0..n {
            for x in 0..n {
                let fx = x as f32 / (n - 1) as f32;
                let fz = z as f32 / (n - 1) as f32;

                let local_x = (fx - 0.5) * self.size.x;
                let local_z = (fz - 0.5) * self.size.y;

                // Apply rotation around Y
                let rotated_x = local_x * cos_r - local_z * sin_r;
                let rotated_z = local_x * sin_r + local_z * cos_r;

                positions.push(Vec3::new(
                    self.position.x + rotated_x,
                    self.position.y,
                    self.position.z + rotated_z,
                ));
                uvs.push(Vec2::new(fx, fz));
            }
        }

        for z in 0..(n - 1) {
            for x in 0..(n - 1) {
                let tl = z * n + x;
                let tr = tl + 1;
                let bl = (z + 1) * n + x;
                let br = bl + 1;

                indices.push(tl as u32);
                indices.push(bl as u32);
                indices.push(tr as u32);

                indices.push(tr as u32);
                indices.push(bl as u32);
                indices.push(br as u32);
            }
        }

        WaterMeshData {
            positions,
            uvs,
            indices,
        }
    }

    /// Checks if a world-space XZ point is within the water plane bounds.
    pub fn contains_xz(&self, point: Vec2) -> bool {
        let local_x = point.x - self.position.x;
        let local_z = point.y - self.position.z;

        // Undo rotation
        let cos_r = (-self.rotation).cos();
        let sin_r = (-self.rotation).sin();
        let rx = local_x * cos_r - local_z * sin_r;
        let rz = local_x * sin_r + local_z * cos_r;

        rx.abs() <= self.size.x * 0.5 && rz.abs() <= self.size.y * 0.5
    }

    /// Computes the distance from a world-space point to the water surface.
    /// Positive values mean the point is above water.
    pub fn signed_distance(&self, point: Vec3, time: f32) -> f32 {
        let surface_y = self.height_at(Vec2::new(point.x, point.z), time);
        point.y - surface_y
    }
}

// ---------------------------------------------------------------------------
// WaterMeshData
// ---------------------------------------------------------------------------

/// Generated water mesh vertex data.
#[derive(Debug, Clone)]
pub struct WaterMeshData {
    /// Vertex positions (rest-state, before wave displacement).
    pub positions: Vec<Vec3>,
    /// Texture coordinates for each vertex.
    pub uvs: Vec<Vec2>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

impl WaterMeshData {
    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns the number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------------------
// WaterUniformData
// ---------------------------------------------------------------------------

/// Per-frame uniform data for the water shader, packed for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WaterUniformData {
    pub deep_color: Vec3,
    pub opacity: f32,
    pub shallow_color: Vec3,
    pub color_depth: f32,
    pub normal_map_speed: Vec2,
    pub normal_map_scale: f32,
    pub normal_map_strength: f32,
    pub specular_intensity: f32,
    pub specular_roughness: f32,
    pub fresnel_f0: f32,
    pub fresnel_power: f32,
    pub time: f32,
    pub water_y: f32,
    pub foam_depth_threshold: f32,
    pub foam_intensity: f32,
    pub caustic_intensity: f32,
    pub caustic_scale: f32,
    pub caustic_max_depth: f32,
    pub ior: f32,
    // Wave parameters packed: direction.xy, wavelength, amplitude, speed, steepness
    pub wave0: [f32; 6],
    pub wave1: [f32; 6],
    pub wave2: [f32; 6],
    pub wave3: [f32; 6],
}

impl WaterUniformData {
    /// Packs a `WaterPlane`'s settings into GPU-ready uniform data.
    pub fn from_water_plane(plane: &WaterPlane, time: f32) -> Self {
        let pack_wave = |w: &WaveParameters| -> [f32; 6] {
            [
                w.direction.x,
                w.direction.y,
                w.wavelength,
                w.amplitude,
                w.speed,
                w.steepness,
            ]
        };

        Self {
            deep_color: plane.deep_color,
            opacity: plane.opacity,
            shallow_color: plane.shallow_color,
            color_depth: plane.color_depth,
            normal_map_speed: plane.normal_map_speed,
            normal_map_scale: plane.normal_map_scale,
            normal_map_strength: plane.normal_map_strength,
            specular_intensity: plane.specular_intensity,
            specular_roughness: plane.specular_roughness,
            fresnel_f0: plane.fresnel.f0,
            fresnel_power: plane.fresnel.power,
            time,
            water_y: plane.water_y(),
            foam_depth_threshold: plane.foam.depth_threshold,
            foam_intensity: plane.foam.intensity,
            caustic_intensity: plane.caustics.intensity,
            caustic_scale: plane.caustics.scale,
            caustic_max_depth: plane.caustics.max_depth,
            ior: plane.ior,
            wave0: pack_wave(&plane.waves.waves[0]),
            wave1: pack_wave(&plane.waves.waves[1]),
            wave2: pack_wave(&plane.waves.waves[2]),
            wave3: pack_wave(&plane.waves.waves[3]),
        }
    }
}

// ---------------------------------------------------------------------------
// WaterComponent
// ---------------------------------------------------------------------------

/// ECS component for attaching a water plane to an entity.
///
/// The water plane's position is typically updated from the entity's
/// transform each frame.
#[derive(Debug, Clone)]
pub struct WaterComponent {
    /// The water plane configuration.
    pub plane: WaterPlane,
    /// Whether this water component is currently visible / active.
    pub visible: bool,
    /// Layer mask for reflection rendering (which layers appear in reflections).
    pub reflection_layers: u32,
    /// Priority for rendering order (lower = rendered first).
    pub priority: i32,
}

impl WaterComponent {
    /// Creates a new water component from a water plane.
    pub fn new(plane: WaterPlane) -> Self {
        Self {
            plane,
            visible: true,
            reflection_layers: 0xFFFFFFFF,
            priority: 0,
        }
    }

    /// Sets the reflection layer mask.
    pub fn with_reflection_layers(mut self, layers: u32) -> Self {
        self.reflection_layers = layers;
        self
    }

    /// Sets the rendering priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Updates the water plane's position from the entity transform.
    pub fn update_position(&mut self, entity_position: Vec3) {
        self.plane.position = entity_position;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_frequency() {
        let wave = WaveParameters::new(Vec2::new(1.0, 0.0), 10.0, 0.5, 2.0);
        let expected = 2.0 * PI / 10.0;
        assert!((wave.frequency() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_wave_evaluate_zero() {
        let wave = WaveParameters::new(Vec2::new(1.0, 0.0), 10.0, 0.5, 0.0);
        let result = wave.evaluate(Vec2::ZERO, 0.0);
        // At origin, time=0: sin(0) = 0
        assert!((result.position_offset.y).abs() < 1e-6);
    }

    #[test]
    fn test_wave_set_max_amplitude() {
        let set = WaveSet::calm_lake();
        let expected: f32 = set.waves.iter().map(|w| w.amplitude).sum();
        assert!((set.max_amplitude() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_fresnel_normal_incidence() {
        let fresnel = FresnelSettings { f0: 0.02, power: 5.0, bias: 0.0 };
        let value = fresnel.evaluate(1.0); // looking straight at surface
        assert!((value - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_fresnel_grazing_incidence() {
        let fresnel = FresnelSettings { f0: 0.02, power: 5.0, bias: 0.0 };
        let value = fresnel.evaluate(0.0); // grazing angle
        assert!((value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_caustic_depth_attenuation() {
        let caustics = CausticSettings::default();
        assert!((caustics.depth_attenuation(0.0) - 1.0).abs() < 1e-6);
        assert!(caustics.depth_attenuation(-1.0) == 0.0);
        assert!(caustics.depth_attenuation(caustics.max_depth + 1.0) == 0.0);
    }

    #[test]
    fn test_shore_foam_outside_threshold() {
        let foam = ShoreFoamSettings::default();
        assert_eq!(foam.compute_foam(foam.depth_threshold + 1.0, 0.0), 0.0);
        assert_eq!(foam.compute_foam(-0.1, 0.0), 0.0);
    }

    #[test]
    fn test_water_plane_contains() {
        let plane = WaterPlane::new(Vec3::ZERO, Vec2::new(10.0, 10.0));
        assert!(plane.contains_xz(Vec2::ZERO));
        assert!(plane.contains_xz(Vec2::new(4.9, 4.9)));
        assert!(!plane.contains_xz(Vec2::new(6.0, 0.0)));
    }

    #[test]
    fn test_water_plane_height_at() {
        let mut plane = WaterPlane::new(Vec3::new(0.0, 5.0, 0.0), Vec2::new(10.0, 10.0));
        plane.waves = WaveSet::new([
            WaveParameters::new(Vec2::X, 10.0, 0.0, 0.0), // zero amplitude
            WaveParameters::new(Vec2::X, 10.0, 0.0, 0.0),
            WaveParameters::new(Vec2::X, 10.0, 0.0, 0.0),
            WaveParameters::new(Vec2::X, 10.0, 0.0, 0.0),
        ]);
        let h = plane.height_at(Vec2::ZERO, 0.0);
        assert!((h - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_mesh() {
        let plane = WaterPlane::new(Vec3::ZERO, Vec2::new(10.0, 10.0));
        let mesh = plane.generate_mesh();
        let n = plane.subdivisions as usize + 1;
        assert_eq!(mesh.vertex_count(), n * n);
        assert_eq!(mesh.triangle_count(), (n - 1) * (n - 1) * 2);
    }

    #[test]
    fn test_reflection_matrix_reflects_y() {
        let water_y = 5.0;
        let refl = ReflectionCamera::reflection_matrix(water_y);
        let point = Vec4::new(1.0, 8.0, 2.0, 1.0);
        let reflected = refl * point;
        assert!((reflected.x - 1.0).abs() < 1e-5);
        assert!((reflected.y - 2.0).abs() < 1e-5); // 2*5 - 8 = 2
        assert!((reflected.z - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_water_uniform_data() {
        let plane = WaterPlane::default();
        let data = WaterUniformData::from_water_plane(&plane, 1.0);
        assert_eq!(data.time, 1.0);
        assert_eq!(data.water_y, plane.position.y);
    }

    #[test]
    fn test_depth_color() {
        let plane = WaterPlane::default();
        let shallow = plane.depth_color(0.0);
        let deep = plane.depth_color(plane.color_depth * 10.0);
        // At depth=0, color should be shallow
        assert!((shallow.x - plane.shallow_color.x).abs() < 1e-5);
        // At very deep, color should be deep
        assert!((deep.x - plane.deep_color.x).abs() < 1e-5);
    }

    #[test]
    fn test_wave_presets() {
        let lake = WaveSet::calm_lake();
        let river = WaveSet::river();
        let choppy = WaveSet::choppy();

        assert!(lake.max_amplitude() < river.max_amplitude());
        assert!(river.max_amplitude() < choppy.max_amplitude());
    }

    #[test]
    fn test_water_component() {
        let plane = WaterPlane::new(Vec3::ZERO, Vec2::new(50.0, 50.0));
        let comp = WaterComponent::new(plane)
            .with_reflection_layers(0xFF)
            .with_priority(1);
        assert_eq!(comp.reflection_layers, 0xFF);
        assert_eq!(comp.priority, 1);
        assert!(comp.visible);
    }
}
