// engine/render/src/lighting/light_types.rs
//
// Typed light source definitions for the Genovo renderer. Includes
// directional, point, spot, and area lights, along with attenuation
// functions and a per-frame light manager that collects, sorts, and
// uploads lights to the GPU.

use glam::{Mat4, Vec3, Vec4};
use std::cmp::Ordering;

/// Maximum number of lights supported in a single frame.
///
/// This value determines the size of the GPU light buffer. Lights beyond
/// this limit are culled by importance.
pub const MAX_LIGHTS: usize = 256;

/// Maximum number of shadow-casting lights per frame.
pub const MAX_SHADOW_LIGHTS: usize = 16;

// ---------------------------------------------------------------------------
// LightType
// ---------------------------------------------------------------------------

/// Discriminant for the type of light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LightType {
    /// Infinitely far away, parallel rays (e.g. sun).
    Directional = 0,
    /// Omnidirectional point emitter.
    Point = 1,
    /// Cone-shaped emitter with inner/outer angles.
    Spot = 2,
    /// Rectangular area emitter.
    AreaRect = 3,
    /// Disc-shaped area emitter.
    AreaDisc = 4,
}

// ---------------------------------------------------------------------------
// DirectionalLight
// ---------------------------------------------------------------------------

/// A directional light representing infinitely far-away parallel rays.
///
/// Commonly used for sunlight and moonlight. Direction points *toward* the
/// light source (i.e. the opposite of the ray direction).
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Direction toward the light source (normalised).
    pub direction: Vec3,
    /// Linear-space colour.
    pub color: Vec3,
    /// Illuminance in lux. For a sun at noon this is typically ~100 000 lx,
    /// but for games a value around 1..10 is common after artistic scaling.
    pub intensity: f32,
    /// Whether this light casts shadows.
    pub shadow_casting: bool,
    /// Number of cascaded shadow map splits (1..4).
    pub cascade_count: u32,
    /// Cascade split distances override. If empty, automatic PSSM splits
    /// are computed.
    pub cascade_splits: Vec<f32>,
    /// Angular diameter of the light source (for soft shadows / PCSS
    /// blocker estimation). Sun ≈ 0.53 degrees.
    pub angular_diameter: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, 1.0, 0.0),
            color: Vec3::ONE,
            intensity: 1.0,
            shadow_casting: true,
            cascade_count: 4,
            cascade_splits: Vec::new(),
            angular_diameter: 0.53_f32.to_radians(),
        }
    }
}

impl DirectionalLight {
    /// Create a new directional light pointing in the given direction.
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            direction: direction.normalize_or_zero(),
            color,
            intensity,
            ..Default::default()
        }
    }

    /// Typical noon sunlight.
    pub fn sun() -> Self {
        Self::new(
            Vec3::new(0.2, 1.0, 0.3).normalize(),
            Vec3::new(1.0, 0.98, 0.95),
            3.0,
        )
    }

    /// Moonlight.
    pub fn moon() -> Self {
        let mut light = Self::new(
            Vec3::new(-0.3, 0.8, 0.5).normalize(),
            Vec3::new(0.7, 0.75, 0.9),
            0.05,
        );
        light.cascade_count = 2;
        light
    }

    /// Convert to the generic `Light` enum.
    pub fn to_light(&self) -> Light {
        Light::Directional(self.clone())
    }
}

// ---------------------------------------------------------------------------
// PointLight
// ---------------------------------------------------------------------------

/// An omnidirectional point light.
#[derive(Debug, Clone)]
pub struct PointLight {
    /// World-space position.
    pub position: Vec3,
    /// Linear-space colour.
    pub color: Vec3,
    /// Luminous power in lumens. For games a value of 1..100 is typical.
    pub intensity: f32,
    /// Maximum influence radius. Beyond this distance the light contributes
    /// zero illumination. Used for culling and attenuation windowing.
    pub radius: f32,
    /// Attenuation falloff mode.
    pub falloff: AttenuationFalloff,
    /// Whether this light casts shadows.
    pub shadow_casting: bool,
    /// Physical source radius (for area-light approximation / soft shadows).
    pub source_radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
            falloff: AttenuationFalloff::InverseSquare,
            shadow_casting: false,
            source_radius: 0.0,
        }
    }
}

impl PointLight {
    /// Create a new point light.
    pub fn new(position: Vec3, color: Vec3, intensity: f32, radius: f32) -> Self {
        Self {
            position,
            color,
            intensity,
            radius,
            ..Default::default()
        }
    }

    /// Compute attenuation at a given distance.
    pub fn attenuation(&self, distance: f32) -> f32 {
        compute_attenuation(distance, self.radius, self.falloff)
    }

    /// Convert to the generic `Light` enum.
    pub fn to_light(&self) -> Light {
        Light::Point(self.clone())
    }
}

// ---------------------------------------------------------------------------
// SpotLight
// ---------------------------------------------------------------------------

/// A cone-shaped spot light.
#[derive(Debug, Clone)]
pub struct SpotLight {
    /// World-space position of the light.
    pub position: Vec3,
    /// Direction the spotlight is pointing (normalised).
    pub direction: Vec3,
    /// Linear-space colour.
    pub color: Vec3,
    /// Luminous power.
    pub intensity: f32,
    /// Inner cone angle (full intensity zone) in radians.
    pub inner_angle: f32,
    /// Outer cone angle (falloff zone boundary) in radians.
    pub outer_angle: f32,
    /// Maximum range.
    pub range: f32,
    /// Attenuation falloff mode.
    pub falloff: AttenuationFalloff,
    /// Whether this light casts shadows.
    pub shadow_casting: bool,
    /// Physical source radius.
    pub source_radius: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: -Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
            inner_angle: 15.0_f32.to_radians(),
            outer_angle: 30.0_f32.to_radians(),
            range: 20.0,
            falloff: AttenuationFalloff::InverseSquare,
            shadow_casting: false,
            source_radius: 0.0,
        }
    }
}

impl SpotLight {
    /// Create a new spot light.
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        inner_angle: f32,
        outer_angle: f32,
        range: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize_or_zero(),
            color,
            intensity,
            inner_angle,
            outer_angle,
            range,
            ..Default::default()
        }
    }

    /// Compute distance attenuation at a given distance.
    pub fn distance_attenuation(&self, distance: f32) -> f32 {
        compute_attenuation(distance, self.range, self.falloff)
    }

    /// Compute the angular falloff for a given angle between the spot
    /// direction and the direction to the lit point.
    ///
    /// Returns 1.0 inside the inner cone, 0.0 outside the outer cone,
    /// and a smooth interpolation in between.
    pub fn angle_falloff(&self, cos_angle: f32) -> f32 {
        let inner_cos = self.inner_angle.cos();
        let outer_cos = self.outer_angle.cos();
        spot_angle_falloff(cos_angle, inner_cos, outer_cos)
    }

    /// Combined attenuation (distance * angle).
    pub fn total_attenuation(&self, position: Vec3) -> f32 {
        let to_point = position - self.position;
        let distance = to_point.length();
        if distance < 1e-6 {
            return 1.0;
        }
        let dir_to_point = to_point / distance;
        let cos_angle = dir_to_point.dot(self.direction);
        self.distance_attenuation(distance) * self.angle_falloff(cos_angle)
    }

    /// Build a perspective projection matrix for shadow mapping.
    pub fn shadow_projection(&self) -> Mat4 {
        let fov = self.outer_angle * 2.0;
        Mat4::perspective_rh(fov, 1.0, 0.1, self.range)
    }

    /// Build a view matrix for shadow mapping.
    pub fn shadow_view(&self) -> Mat4 {
        let target = self.position + self.direction;
        let up = if self.direction.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        Mat4::look_at_rh(self.position, target, up)
    }

    /// Convert to the generic `Light` enum.
    pub fn to_light(&self) -> Light {
        Light::Spot(self.clone())
    }
}

// ---------------------------------------------------------------------------
// AreaLight
// ---------------------------------------------------------------------------

/// Shape of an area light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AreaLightShape {
    /// A rectangular area light with width and height.
    Rectangle { width: f32, height: f32 },
    /// A disc-shaped area light with a radius.
    Disc { radius: f32 },
}

/// An area light emitter.
#[derive(Debug, Clone)]
pub struct AreaLight {
    /// World-space position (center of the emitting surface).
    pub position: Vec3,
    /// Surface normal of the emitting surface.
    pub normal: Vec3,
    /// "Right" direction for rectangle lights (tangent).
    pub right: Vec3,
    /// Shape and dimensions.
    pub shape: AreaLightShape,
    /// Linear-space colour.
    pub color: Vec3,
    /// Luminous power.
    pub intensity: f32,
    /// Maximum influence range.
    pub range: f32,
    /// Whether this light casts shadows.
    pub shadow_casting: bool,
    /// Whether the light emits from both sides.
    pub two_sided: bool,
}

impl Default for AreaLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: -Vec3::Z,
            right: Vec3::X,
            shape: AreaLightShape::Rectangle {
                width: 1.0,
                height: 1.0,
            },
            color: Vec3::ONE,
            intensity: 1.0,
            range: 15.0,
            shadow_casting: false,
            two_sided: false,
        }
    }
}

impl AreaLight {
    /// Create a rectangular area light.
    pub fn rectangle(
        position: Vec3,
        normal: Vec3,
        right: Vec3,
        width: f32,
        height: f32,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        Self {
            position,
            normal: normal.normalize_or_zero(),
            right: right.normalize_or_zero(),
            shape: AreaLightShape::Rectangle { width, height },
            color,
            intensity,
            ..Default::default()
        }
    }

    /// Create a disc-shaped area light.
    pub fn disc(
        position: Vec3,
        normal: Vec3,
        radius: f32,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        Self {
            position,
            normal: normal.normalize_or_zero(),
            shape: AreaLightShape::Disc { radius },
            color,
            intensity,
            ..Default::default()
        }
    }

    /// Compute the "up" direction (bitangent) from normal and right.
    pub fn up(&self) -> Vec3 {
        self.normal.cross(self.right).normalize_or_zero()
    }

    /// Get the four corners of a rectangular area light in world space.
    pub fn corners(&self) -> Option<[Vec3; 4]> {
        match self.shape {
            AreaLightShape::Rectangle { width, height } => {
                let half_w = width * 0.5;
                let half_h = height * 0.5;
                let up = self.up();
                Some([
                    self.position - self.right * half_w - up * half_h,
                    self.position + self.right * half_w - up * half_h,
                    self.position + self.right * half_w + up * half_h,
                    self.position - self.right * half_w + up * half_h,
                ])
            }
            AreaLightShape::Disc { .. } => None,
        }
    }

    /// Convert to the generic `Light` enum.
    pub fn to_light(&self) -> Light {
        Light::Area(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Light enum
// ---------------------------------------------------------------------------

/// A tagged union of all supported light types.
#[derive(Debug, Clone)]
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
    Area(AreaLight),
}

impl Light {
    /// Returns the light type discriminant.
    pub fn light_type(&self) -> LightType {
        match self {
            Self::Directional(_) => LightType::Directional,
            Self::Point(_) => LightType::Point,
            Self::Spot(_) => LightType::Spot,
            Self::Area(a) => match a.shape {
                AreaLightShape::Rectangle { .. } => LightType::AreaRect,
                AreaLightShape::Disc { .. } => LightType::AreaDisc,
            },
        }
    }

    /// Returns the colour of the light.
    pub fn color(&self) -> Vec3 {
        match self {
            Self::Directional(d) => d.color,
            Self::Point(p) => p.color,
            Self::Spot(s) => s.color,
            Self::Area(a) => a.color,
        }
    }

    /// Returns the intensity.
    pub fn intensity(&self) -> f32 {
        match self {
            Self::Directional(d) => d.intensity,
            Self::Point(p) => p.intensity,
            Self::Spot(s) => s.intensity,
            Self::Area(a) => a.intensity,
        }
    }

    /// Returns `true` if this light casts shadows.
    pub fn shadow_casting(&self) -> bool {
        match self {
            Self::Directional(d) => d.shadow_casting,
            Self::Point(p) => p.shadow_casting,
            Self::Spot(s) => s.shadow_casting,
            Self::Area(a) => a.shadow_casting,
        }
    }

    /// Returns the world-space position. For directional lights, returns
    /// `None`.
    pub fn position(&self) -> Option<Vec3> {
        match self {
            Self::Directional(_) => None,
            Self::Point(p) => Some(p.position),
            Self::Spot(s) => Some(s.position),
            Self::Area(a) => Some(a.position),
        }
    }

    /// Returns the maximum influence radius. For directional lights, returns
    /// infinity.
    pub fn radius(&self) -> f32 {
        match self {
            Self::Directional(_) => f32::INFINITY,
            Self::Point(p) => p.radius,
            Self::Spot(s) => s.range,
            Self::Area(a) => a.range,
        }
    }

    /// Compute an importance score for this light from a given viewpoint.
    /// Higher values mean more important (should be kept when culling by
    /// light count).
    pub fn importance(&self, view_position: Vec3) -> f32 {
        match self {
            Self::Directional(d) => {
                // Directional lights are always high importance.
                d.intensity * 1000.0
            }
            Self::Point(p) => {
                let dist = (p.position - view_position).length();
                p.intensity / (dist * dist + 1.0)
            }
            Self::Spot(s) => {
                let dist = (s.position - view_position).length();
                s.intensity / (dist * dist + 1.0)
            }
            Self::Area(a) => {
                let dist = (a.position - view_position).length();
                a.intensity / (dist * dist + 1.0)
            }
        }
    }

    /// Convert to `LightData` for GPU upload.
    pub fn to_gpu_data(&self) -> LightData {
        match self {
            Self::Directional(d) => LightData {
                position_type: Vec4::new(d.direction.x, d.direction.y, d.direction.z, 0.0),
                color_intensity: Vec4::new(d.color.x, d.color.y, d.color.z, d.intensity),
                direction_range: Vec4::ZERO,
                spot_params: Vec4::ZERO,
            },
            Self::Point(p) => LightData {
                position_type: Vec4::new(p.position.x, p.position.y, p.position.z, 1.0),
                color_intensity: Vec4::new(p.color.x, p.color.y, p.color.z, p.intensity),
                direction_range: Vec4::new(0.0, 0.0, 0.0, p.radius),
                spot_params: Vec4::ZERO,
            },
            Self::Spot(sp) => LightData {
                position_type: Vec4::new(sp.position.x, sp.position.y, sp.position.z, 2.0),
                color_intensity: Vec4::new(sp.color.x, sp.color.y, sp.color.z, sp.intensity),
                direction_range: Vec4::new(
                    sp.direction.x,
                    sp.direction.y,
                    sp.direction.z,
                    sp.range,
                ),
                spot_params: Vec4::new(sp.inner_angle.cos(), sp.outer_angle.cos(), -1.0, 0.0),
            },
            Self::Area(a) => LightData {
                position_type: Vec4::new(
                    a.position.x,
                    a.position.y,
                    a.position.z,
                    match a.shape {
                        AreaLightShape::Rectangle { .. } => 3.0,
                        AreaLightShape::Disc { .. } => 4.0,
                    },
                ),
                color_intensity: Vec4::new(a.color.x, a.color.y, a.color.z, a.intensity),
                direction_range: Vec4::new(a.normal.x, a.normal.y, a.normal.z, a.range),
                spot_params: match a.shape {
                    AreaLightShape::Rectangle { width, height } => {
                        Vec4::new(width, height, 0.0, 0.0)
                    }
                    AreaLightShape::Disc { radius } => Vec4::new(radius, 0.0, 0.0, 0.0),
                },
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LightData (GPU struct)
// ---------------------------------------------------------------------------

/// GPU-compatible light data structure. Must match the WGSL `LightData` struct.
///
/// Layout: 64 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightData {
    /// xyz = position (point/spot) or direction (directional), w = light type.
    pub position_type: Vec4,
    /// rgb = colour, a = intensity.
    pub color_intensity: Vec4,
    /// xyz = direction (spot/area normal), w = range.
    pub direction_range: Vec4,
    /// x = inner_angle_cos (spot), y = outer_angle_cos (spot),
    /// z = shadow map index, w = padding.
    pub spot_params: Vec4,
}

/// GPU light buffer header.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightBufferHeader {
    /// x = number of lights, yzw = padding.
    pub count: [u32; 4],
}

// ---------------------------------------------------------------------------
// Attenuation
// ---------------------------------------------------------------------------

/// Attenuation falloff mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttenuationFalloff {
    /// Physically correct inverse-square law with smooth windowing.
    InverseSquare,
    /// Linear falloff (for artistic control).
    Linear,
    /// Quadratic falloff without windowing.
    Quadratic,
}

/// Compute distance attenuation with smooth windowing at the radius boundary.
///
/// Uses the UE4-style inverse-square-law with a smooth window function:
///   atten = saturate(1 - (d/r)^4)^2 / (d^2 + epsilon)
///
/// This ensures zero contribution at `distance >= radius` while maintaining
/// physically-plausible inverse-square falloff at close range.
pub fn compute_attenuation(distance: f32, radius: f32, falloff: AttenuationFalloff) -> f32 {
    match falloff {
        AttenuationFalloff::InverseSquare => {
            inverse_square_windowed_attenuation(distance, radius)
        }
        AttenuationFalloff::Linear => {
            (1.0 - (distance / radius).min(1.0)).max(0.0)
        }
        AttenuationFalloff::Quadratic => {
            let d2 = distance * distance;
            let r2 = radius * radius;
            (1.0 - d2 / r2).max(0.0)
        }
    }
}

/// Inverse-square attenuation with smooth windowing (UE4/Filament style).
///
/// atten = saturate(1 - (d/r)^4)^2 / max(d^2, epsilon)
#[inline]
pub fn inverse_square_windowed_attenuation(distance: f32, radius: f32) -> f32 {
    let d2 = distance * distance;
    let r2 = radius * radius;
    let ratio = d2 / r2;
    let factor = (1.0 - ratio * ratio).max(0.0);
    let window = factor * factor;
    window / d2.max(0.0001)
}

/// Spot light angle falloff with smooth transition from inner to outer cone.
///
/// Returns 1.0 inside the inner cone, 0.0 outside the outer cone, and a
/// smooth Hermite interpolation in between.
///
/// # Arguments
/// - `cos_angle` — cosine of the angle between the spot direction and the
///   direction to the lit point.
/// - `inner_cos` — cosine of the inner cone angle.
/// - `outer_cos` — cosine of the outer cone angle.
#[inline]
pub fn spot_angle_falloff(cos_angle: f32, inner_cos: f32, outer_cos: f32) -> f32 {
    let t = ((cos_angle - outer_cos) / (inner_cos - outer_cos).max(1e-6)).clamp(0.0, 1.0);
    // Hermite smoothstep for a soft transition.
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// LightManager
// ---------------------------------------------------------------------------

/// Per-frame light manager. Collects all lights, sorts them by importance
/// relative to the camera, and prepares the GPU light buffer data.
pub struct LightManager {
    /// All lights submitted this frame.
    lights: Vec<Light>,
    /// Indices into `lights` sorted by importance (descending).
    sorted_indices: Vec<usize>,
    /// The packed GPU light data.
    gpu_data: Vec<LightData>,
    /// Number of lights actually uploaded to the GPU buffer.
    gpu_light_count: usize,
    /// Indices of shadow-casting lights.
    shadow_light_indices: Vec<usize>,
}

impl LightManager {
    /// Create a new light manager.
    pub fn new() -> Self {
        Self {
            lights: Vec::with_capacity(64),
            sorted_indices: Vec::with_capacity(64),
            gpu_data: Vec::with_capacity(MAX_LIGHTS),
            gpu_light_count: 0,
            shadow_light_indices: Vec::with_capacity(MAX_SHADOW_LIGHTS),
        }
    }

    /// Clear all lights for a new frame.
    pub fn begin_frame(&mut self) {
        self.lights.clear();
        self.sorted_indices.clear();
        self.gpu_data.clear();
        self.gpu_light_count = 0;
        self.shadow_light_indices.clear();
    }

    /// Add a light.
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    /// Add a directional light.
    pub fn add_directional(&mut self, light: DirectionalLight) {
        self.lights.push(Light::Directional(light));
    }

    /// Add a point light.
    pub fn add_point(&mut self, light: PointLight) {
        self.lights.push(Light::Point(light));
    }

    /// Add a spot light.
    pub fn add_spot(&mut self, light: SpotLight) {
        self.lights.push(Light::Spot(light));
    }

    /// Add an area light.
    pub fn add_area(&mut self, light: AreaLight) {
        self.lights.push(Light::Area(light));
    }

    /// Sort lights by importance and prepare GPU data.
    ///
    /// # Arguments
    /// - `camera_position` — the world-space position of the camera, used
    ///   to compute distance-based importance.
    pub fn prepare(&mut self, camera_position: Vec3) {
        // Build importance-sorted indices.
        self.sorted_indices.clear();
        self.sorted_indices
            .extend(0..self.lights.len());

        self.sorted_indices.sort_by(|&a, &b| {
            let ia = self.lights[a].importance(camera_position);
            let ib = self.lights[b].importance(camera_position);
            ib.partial_cmp(&ia).unwrap_or(Ordering::Equal)
        });

        // Pack GPU data (limited to MAX_LIGHTS).
        self.gpu_data.clear();
        self.shadow_light_indices.clear();
        let count = self.sorted_indices.len().min(MAX_LIGHTS);

        for i in 0..count {
            let idx = self.sorted_indices[i];
            let light = &self.lights[idx];
            let mut data = light.to_gpu_data();

            if light.shadow_casting() && self.shadow_light_indices.len() < MAX_SHADOW_LIGHTS {
                data.spot_params.z = self.shadow_light_indices.len() as f32;
                self.shadow_light_indices.push(idx);
            }

            self.gpu_data.push(data);
        }

        self.gpu_light_count = count;
    }

    /// Returns the GPU light data as a byte slice suitable for buffer upload.
    pub fn gpu_buffer_data(&self) -> &[LightData] {
        &self.gpu_data
    }

    /// Returns the number of lights in the GPU buffer.
    pub fn gpu_light_count(&self) -> usize {
        self.gpu_light_count
    }

    /// Returns the header struct for the GPU light buffer.
    pub fn gpu_header(&self) -> LightBufferHeader {
        LightBufferHeader {
            count: [self.gpu_light_count as u32, 0, 0, 0],
        }
    }

    /// Returns the total number of submitted lights (before culling).
    pub fn total_light_count(&self) -> usize {
        self.lights.len()
    }

    /// Returns the lights sorted by importance.
    pub fn sorted_lights(&self) -> Vec<&Light> {
        self.sorted_indices
            .iter()
            .map(|&i| &self.lights[i])
            .collect()
    }

    /// Returns indices of shadow-casting lights (in the original lights array).
    pub fn shadow_light_indices(&self) -> &[usize] {
        &self.shadow_light_indices
    }

    /// Get a light by its original index.
    pub fn get_light(&self, index: usize) -> Option<&Light> {
        self.lights.get(index)
    }

    /// Get all directional lights.
    pub fn directional_lights(&self) -> Vec<&DirectionalLight> {
        self.lights
            .iter()
            .filter_map(|l| match l {
                Light::Directional(d) => Some(d),
                _ => None,
            })
            .collect()
    }

    /// Get all point lights.
    pub fn point_lights(&self) -> Vec<&PointLight> {
        self.lights
            .iter()
            .filter_map(|l| match l {
                Light::Point(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    /// Get all spot lights.
    pub fn spot_lights(&self) -> Vec<&SpotLight> {
        self.lights
            .iter()
            .filter_map(|l| match l {
                Light::Spot(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    /// Build a combined byte buffer with header + light data for GPU upload.
    pub fn build_gpu_buffer(&self) -> Vec<u8> {
        let header = self.gpu_header();
        let header_bytes = bytemuck::bytes_of(&header);

        let light_bytes: &[u8] = if self.gpu_data.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&self.gpu_data)
        };

        let mut buffer = Vec::with_capacity(header_bytes.len() + light_bytes.len());
        buffer.extend_from_slice(header_bytes);
        buffer.extend_from_slice(light_bytes);
        buffer
    }
}

impl Default for LightManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_square_attenuation_at_zero() {
        let a = inverse_square_windowed_attenuation(0.001, 10.0);
        assert!(a > 0.0);
    }

    #[test]
    fn inverse_square_attenuation_at_radius() {
        let a = inverse_square_windowed_attenuation(10.0, 10.0);
        assert!(a.abs() < 0.01);
    }

    #[test]
    fn spot_falloff_inside_inner() {
        let f = spot_angle_falloff(0.95, 0.9, 0.7);
        assert!((f - 1.0).abs() < 0.01);
    }

    #[test]
    fn spot_falloff_outside_outer() {
        let f = spot_angle_falloff(0.5, 0.9, 0.7);
        assert!(f.abs() < 0.01);
    }

    #[test]
    fn spot_falloff_midpoint() {
        let f = spot_angle_falloff(0.8, 0.9, 0.7);
        assert!(f > 0.0 && f < 1.0);
    }

    #[test]
    fn light_manager_basics() {
        let mut mgr = LightManager::new();
        mgr.begin_frame();
        mgr.add_directional(DirectionalLight::sun());
        mgr.add_point(PointLight::new(Vec3::ZERO, Vec3::ONE, 5.0, 10.0));
        mgr.prepare(Vec3::new(0.0, 5.0, 10.0));
        assert_eq!(mgr.gpu_light_count(), 2);
    }

    #[test]
    fn importance_sorts_directional_first() {
        let mut mgr = LightManager::new();
        mgr.begin_frame();
        mgr.add_point(PointLight::new(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::ONE,
            1.0,
            5.0,
        ));
        mgr.add_directional(DirectionalLight::sun());
        mgr.prepare(Vec3::ZERO);

        let sorted = mgr.sorted_lights();
        assert!(matches!(sorted[0], Light::Directional(_)));
    }

    #[test]
    fn light_data_layout() {
        assert_eq!(std::mem::size_of::<LightData>(), 64);
        assert!(std::mem::align_of::<LightData>() >= 4);
    }
}
