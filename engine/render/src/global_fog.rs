// engine/render/src/global_fog.rs
//
// Comprehensive fog system for the Genovo engine.
//
// Provides multiple fog models that can be combined:
//
// - **Linear fog** — Ramps from zero to full fog between a start and end
//   distance.
// - **Exponential fog** — Density grows exponentially with distance.
// - **Exponential squared fog** — Faster-growing variant of exponential fog.
// - **Height fog** — Density attenuated by an exponential height falloff,
//   simulating fog that settles near the ground.
// - **Animated fog** — Time-driven procedural noise offsets the density field
//   to create billowing, organic fog movement.
// - **Per-object fog override** — Individual objects can override the global
//   fog parameters to appear through or behind fog differently.
// - **Fog volumes** — Axis-aligned boxes that inject additional local fog
//   density into the scene.
// - **Fog colour from sky gradient** — The inscatter colour can be sampled
//   from a sky gradient based on the view direction's elevation angle.
//
// # Pipeline integration
//
// Fog is applied as a post-process pass that reads the depth buffer and
// composites inscattered light and extinction into the scene colour.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Fog mode enumeration
// ---------------------------------------------------------------------------

/// Selects the global distance-fog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FogMode {
    /// No distance fog.
    None,
    /// Linear ramp between `start` and `end` distances.
    Linear,
    /// Exponential: `1 - exp(-density * distance)`.
    Exponential,
    /// Exponential squared: `1 - exp(-(density * distance)^2)`.
    ExponentialSquared,
}

// ---------------------------------------------------------------------------
// Fog colour source
// ---------------------------------------------------------------------------

/// Determines how the fog inscatter colour is computed.
#[derive(Debug, Clone)]
pub enum FogColorSource {
    /// Single constant colour.
    Constant { color: [f32; 3] },
    /// Blend between two colours based on the view direction's angle to the
    /// horizon. `horizon_color` at elevation 0, `zenith_color` at elevation 90.
    SkyGradient {
        horizon_color: [f32; 3],
        zenith_color: [f32; 3],
        /// Exponent controlling the gradient rolloff.
        power: f32,
    },
    /// Sample a directional light colour (simulates sun-tinted fog).
    DirectionalLight {
        base_color: [f32; 3],
        light_direction: [f32; 3],
        light_color: [f32; 3],
        /// How much the sun influences the fog colour.
        sun_intensity: f32,
        /// Angular falloff exponent for the sun contribution.
        sun_falloff: f32,
    },
}

impl Default for FogColorSource {
    fn default() -> Self {
        Self::Constant { color: [0.7, 0.75, 0.8] }
    }
}

impl FogColorSource {
    /// Evaluate the fog colour for a given view direction.
    ///
    /// # Arguments
    /// * `view_dir` — Normalised view direction (world space).
    pub fn evaluate(&self, view_dir: [f32; 3]) -> [f32; 3] {
        match self {
            Self::Constant { color } => *color,
            Self::SkyGradient { horizon_color, zenith_color, power } => {
                // Elevation: how much the view direction points up.
                let elevation = view_dir[1].abs().clamp(0.0, 1.0).powf(*power);
                [
                    lerp(horizon_color[0], zenith_color[0], elevation),
                    lerp(horizon_color[1], zenith_color[1], elevation),
                    lerp(horizon_color[2], zenith_color[2], elevation),
                ]
            }
            Self::DirectionalLight {
                base_color,
                light_direction,
                light_color,
                sun_intensity,
                sun_falloff,
            } => {
                let dot = view_dir[0] * light_direction[0]
                    + view_dir[1] * light_direction[1]
                    + view_dir[2] * light_direction[2];
                let sun_factor = dot.max(0.0).powf(*sun_falloff) * sun_intensity;
                [
                    base_color[0] + light_color[0] * sun_factor,
                    base_color[1] + light_color[1] * sun_factor,
                    base_color[2] + light_color[2] * sun_factor,
                ]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Global fog parameters
// ---------------------------------------------------------------------------

/// Global fog configuration for the entire scene.
#[derive(Debug, Clone)]
pub struct GlobalFogParams {
    /// Distance fog model.
    pub mode: FogMode,
    /// Fog colour source.
    pub color_source: FogColorSource,
    /// For `Linear`: start distance.
    pub linear_start: f32,
    /// For `Linear`: end distance (fully fogged).
    pub linear_end: f32,
    /// For `Exponential` / `ExponentialSquared`: density coefficient.
    pub density: f32,
    /// Maximum fog factor [0, 1]. Prevents fog from completely hiding geometry.
    pub max_opacity: f32,
    /// Height fog enabled.
    pub height_fog_enabled: bool,
    /// Height fog base altitude (world Y where fog is densest).
    pub height_fog_base: f32,
    /// Height fog falloff (how quickly fog thins with altitude). Higher = faster.
    pub height_fog_falloff: f32,
    /// Height fog density multiplier.
    pub height_fog_density: f32,
    /// Height fog maximum distance.
    pub height_fog_max_distance: f32,
    /// Animation enabled.
    pub animated: bool,
    /// Animation speed (world units per second of noise scrolling).
    pub animation_speed: [f32; 3],
    /// Noise frequency for animated fog.
    pub noise_frequency: f32,
    /// Noise amplitude (scales density perturbation).
    pub noise_amplitude: f32,
    /// Noise octaves for fBm.
    pub noise_octaves: u32,
}

impl Default for GlobalFogParams {
    fn default() -> Self {
        Self {
            mode: FogMode::ExponentialSquared,
            color_source: FogColorSource::default(),
            linear_start: 10.0,
            linear_end: 300.0,
            density: 0.01,
            max_opacity: 1.0,
            height_fog_enabled: false,
            height_fog_base: 0.0,
            height_fog_falloff: 0.1,
            height_fog_density: 0.02,
            height_fog_max_distance: 500.0,
            animated: false,
            animation_speed: [0.5, 0.0, 0.3],
            noise_frequency: 0.05,
            noise_amplitude: 0.3,
            noise_octaves: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Distance fog computation
// ---------------------------------------------------------------------------

/// Compute the distance-based fog factor.
///
/// Returns a value in [0, 1] where 0 = no fog and 1 = fully fogged.
pub fn compute_distance_fog(mode: FogMode, distance: f32, density: f32, start: f32, end: f32) -> f32 {
    match mode {
        FogMode::None => 0.0,
        FogMode::Linear => {
            if end <= start {
                return 0.0;
            }
            ((distance - start) / (end - start)).clamp(0.0, 1.0)
        }
        FogMode::Exponential => {
            1.0 - (-density * distance).exp()
        }
        FogMode::ExponentialSquared => {
            let f = density * distance;
            1.0 - (-(f * f)).exp()
        }
    }
}

/// Compute height fog density integrated along a ray from `eye` to `point`.
///
/// Uses the analytical integral of exponential height fog along a line segment.
///
/// # Arguments
/// * `eye_y` — Camera Y position (world space).
/// * `point_y` — Fragment Y position (world space).
/// * `distance` — Linear distance from eye to fragment.
/// * `base_y` — Fog base altitude.
/// * `falloff` — Height falloff rate.
/// * `density` — Base density at `base_y`.
pub fn compute_height_fog(
    eye_y: f32,
    point_y: f32,
    distance: f32,
    base_y: f32,
    falloff: f32,
    density: f32,
) -> f32 {
    if falloff <= 0.0 || density <= 0.0 {
        return 0.0;
    }

    // Height above base for start and end.
    let h_eye = (eye_y - base_y) * falloff;
    let h_point = (point_y - base_y) * falloff;

    // Analytical line integral of exp(-falloff * h).
    let delta_h = h_point - h_eye;
    let fog_amount = if delta_h.abs() > 1e-6 {
        // Integral: density * distance * (exp(-h_eye) - exp(-h_point)) / delta_h
        let e_eye = (-h_eye).exp();
        let e_point = (-h_point).exp();
        density * distance * (e_eye - e_point) / delta_h
    } else {
        // Degenerate case: camera and fragment at same height.
        density * distance * (-h_eye).exp()
    };

    // Clamp to [0, 1].
    (1.0 - (-fog_amount.max(0.0)).exp()).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Noise for animated fog
// ---------------------------------------------------------------------------

/// Simple hash for procedural noise.
#[inline]
fn hash_1d(n: f32) -> f32 {
    let n = (n * 127.1).sin() * 43758.5453123;
    n - n.floor()
}

/// 3D value noise for fog animation.
fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();

    // Quintic interpolation.
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);

    let hash = |i: i32, j: i32, k: i32| -> f32 {
        let n = i.wrapping_mul(374761393)
            .wrapping_add(j.wrapping_mul(668265263))
            .wrapping_add(k.wrapping_mul(1274126177));
        let n = n ^ (n >> 13);
        let n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303)).wrapping_add(1376312589));
        (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32
    };

    let c000 = hash(ix, iy, iz);
    let c100 = hash(ix + 1, iy, iz);
    let c010 = hash(ix, iy + 1, iz);
    let c110 = hash(ix + 1, iy + 1, iz);
    let c001 = hash(ix, iy, iz + 1);
    let c101 = hash(ix + 1, iy, iz + 1);
    let c011 = hash(ix, iy + 1, iz + 1);
    let c111 = hash(ix + 1, iy + 1, iz + 1);

    let x00 = lerp(c000, c100, ux);
    let x10 = lerp(c010, c110, ux);
    let x01 = lerp(c001, c101, ux);
    let x11 = lerp(c011, c111, ux);

    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);

    lerp(y0, y1, uz)
}

/// Fractal Brownian Motion (fBm) noise.
fn fbm_3d(x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0_f32;
    let mut amplitude = 1.0_f32;
    let mut frequency = 1.0_f32;
    let mut max_amp = 0.0_f32;

    for _ in 0..octaves {
        value += value_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if max_amp > 0.0 { value / max_amp } else { 0.0 }
}

/// Compute animated fog noise at a world position and time.
///
/// Returns a density multiplier in [0, 1].
pub fn animated_fog_noise(
    world_pos: [f32; 3],
    time: f32,
    params: &GlobalFogParams,
) -> f32 {
    if !params.animated {
        return 1.0;
    }

    let px = world_pos[0] * params.noise_frequency + params.animation_speed[0] * time;
    let py = world_pos[1] * params.noise_frequency + params.animation_speed[1] * time;
    let pz = world_pos[2] * params.noise_frequency + params.animation_speed[2] * time;

    let noise = fbm_3d(px, py, pz, params.noise_octaves, 2.0, 0.5);
    let density_mod = 1.0 + (noise - 0.5) * 2.0 * params.noise_amplitude;

    density_mod.clamp(0.0, 2.0)
}

// ---------------------------------------------------------------------------
// Per-object fog override
// ---------------------------------------------------------------------------

/// Per-object fog override. Attach to individual renderable objects.
#[derive(Debug, Clone)]
pub struct FogOverride {
    /// Whether this override is active.
    pub enabled: bool,
    /// Override fog mode (if `Some`, replaces the global mode for this object).
    pub mode: Option<FogMode>,
    /// Override density (if `Some`).
    pub density: Option<f32>,
    /// Override colour (if `Some`).
    pub color: Option<[f32; 3]>,
    /// Fog factor multiplier applied after computation.
    pub factor_multiplier: f32,
    /// Fog factor offset added after computation.
    pub factor_offset: f32,
    /// Exclude from height fog.
    pub ignore_height_fog: bool,
    /// Custom start distance (if `Some`, overrides linear start).
    pub custom_start: Option<f32>,
    /// Custom end distance (if `Some`, overrides linear end).
    pub custom_end: Option<f32>,
}

impl Default for FogOverride {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: None,
            density: None,
            color: None,
            factor_multiplier: 1.0,
            factor_offset: 0.0,
            ignore_height_fog: false,
            custom_start: None,
            custom_end: None,
        }
    }
}

impl FogOverride {
    /// Create an override that makes an object ignore fog entirely.
    pub fn no_fog() -> Self {
        Self {
            enabled: true,
            factor_multiplier: 0.0,
            ..Self::default()
        }
    }

    /// Create an override that doubles the fog effect on this object.
    pub fn double_fog() -> Self {
        Self {
            enabled: true,
            factor_multiplier: 2.0,
            ..Self::default()
        }
    }

    /// Apply this override to modify fog parameters.
    pub fn apply_to_params(&self, global: &GlobalFogParams) -> GlobalFogParams {
        if !self.enabled {
            return global.clone();
        }

        let mut params = global.clone();
        if let Some(mode) = self.mode {
            params.mode = mode;
        }
        if let Some(density) = self.density {
            params.density = density;
        }
        if let Some(start) = self.custom_start {
            params.linear_start = start;
        }
        if let Some(end) = self.custom_end {
            params.linear_end = end;
        }
        if self.ignore_height_fog {
            params.height_fog_enabled = false;
        }
        params
    }

    /// Apply this override to modify a computed fog factor.
    pub fn apply_to_factor(&self, factor: f32) -> f32 {
        if !self.enabled {
            return factor;
        }
        (factor * self.factor_multiplier + self.factor_offset).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Fog volumes
// ---------------------------------------------------------------------------

/// Fog volume shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FogVolumeShape {
    /// Axis-aligned box.
    Box,
    /// Sphere.
    Sphere,
    /// Cylinder aligned along Y.
    Cylinder,
}

/// A local fog volume that adds fog density in a bounded region.
#[derive(Debug, Clone)]
pub struct FogVolume {
    /// Shape of the volume.
    pub shape: FogVolumeShape,
    /// Centre of the volume (world space).
    pub center: [f32; 3],
    /// Half-extents (for Box), or [radius, half_height, radius] (Cylinder),
    /// or [radius, radius, radius] (Sphere — only first component used).
    pub half_extents: [f32; 3],
    /// Fog density inside the volume.
    pub density: f32,
    /// Fog colour inside the volume.
    pub color: [f32; 3],
    /// Soft edge falloff distance. 0 = hard edge.
    pub edge_falloff: f32,
    /// Whether this volume overrides or adds to the global fog.
    pub blend_mode: FogVolumeBlend,
    /// Priority for overlapping volumes (higher = takes precedence).
    pub priority: i32,
    /// Height falloff within the volume (0 = uniform).
    pub height_falloff: f32,
    /// Animation wind direction for this volume.
    pub wind: [f32; 3],
    /// Noise frequency for density variation.
    pub noise_freq: f32,
    /// Noise amplitude for density variation.
    pub noise_amp: f32,
}

/// How a fog volume blends with the global fog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FogVolumeBlend {
    /// Add to existing fog.
    Additive,
    /// Replace the fog within the volume.
    Replace,
    /// Multiply the existing fog density.
    Multiply,
    /// Take the maximum of volume and global fog.
    Max,
}

impl Default for FogVolume {
    fn default() -> Self {
        Self {
            shape: FogVolumeShape::Box,
            center: [0.0, 0.0, 0.0],
            half_extents: [5.0, 5.0, 5.0],
            density: 0.05,
            color: [0.8, 0.85, 0.9],
            edge_falloff: 1.0,
            blend_mode: FogVolumeBlend::Additive,
            priority: 0,
            height_falloff: 0.0,
            wind: [0.0, 0.0, 0.0],
            noise_freq: 0.1,
            noise_amp: 0.0,
        }
    }
}

impl FogVolume {
    /// Create a box-shaped fog volume.
    pub fn new_box(center: [f32; 3], half_extents: [f32; 3], density: f32) -> Self {
        Self {
            shape: FogVolumeShape::Box,
            center,
            half_extents,
            density,
            ..Self::default()
        }
    }

    /// Create a sphere-shaped fog volume.
    pub fn new_sphere(center: [f32; 3], radius: f32, density: f32) -> Self {
        Self {
            shape: FogVolumeShape::Sphere,
            center,
            half_extents: [radius, radius, radius],
            density,
            ..Self::default()
        }
    }

    /// Create a cylinder-shaped fog volume.
    pub fn new_cylinder(center: [f32; 3], radius: f32, half_height: f32, density: f32) -> Self {
        Self {
            shape: FogVolumeShape::Cylinder,
            center,
            half_extents: [radius, half_height, radius],
            density,
            ..Self::default()
        }
    }

    /// Set the fog colour.
    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }

    /// Set the edge falloff distance.
    pub fn with_falloff(mut self, falloff: f32) -> Self {
        self.edge_falloff = falloff;
        self
    }

    /// Set the blend mode.
    pub fn with_blend(mut self, mode: FogVolumeBlend) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Compute the signed distance from a world-space point to the volume surface.
    ///
    /// Returns a negative value if the point is inside the volume.
    pub fn signed_distance(&self, world_pos: [f32; 3]) -> f32 {
        let local = [
            world_pos[0] - self.center[0],
            world_pos[1] - self.center[1],
            world_pos[2] - self.center[2],
        ];

        match self.shape {
            FogVolumeShape::Box => {
                let q = [
                    local[0].abs() - self.half_extents[0],
                    local[1].abs() - self.half_extents[1],
                    local[2].abs() - self.half_extents[2],
                ];
                let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2)).sqrt();
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                outside + inside
            }
            FogVolumeShape::Sphere => {
                let dist = (local[0] * local[0] + local[1] * local[1] + local[2] * local[2]).sqrt();
                dist - self.half_extents[0]
            }
            FogVolumeShape::Cylinder => {
                let r = (local[0] * local[0] + local[2] * local[2]).sqrt();
                let d_r = r - self.half_extents[0];
                let d_y = local[1].abs() - self.half_extents[1];
                let outside = (d_r.max(0.0).powi(2) + d_y.max(0.0).powi(2)).sqrt();
                let inside = d_r.max(d_y).min(0.0);
                outside + inside
            }
        }
    }

    /// Evaluate the fog density contribution at a world-space point.
    ///
    /// # Arguments
    /// * `world_pos` — World-space query point.
    /// * `time` — Current time for animation.
    pub fn evaluate(&self, world_pos: [f32; 3], time: f32) -> FogVolumeResult {
        let sd = self.signed_distance(world_pos);

        // Outside the volume.
        if sd > self.edge_falloff {
            return FogVolumeResult::none();
        }

        // Compute falloff.
        let falloff_factor = if self.edge_falloff > 0.0 && sd > 0.0 {
            1.0 - (sd / self.edge_falloff).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Smooth falloff.
        let smooth = falloff_factor * falloff_factor * (3.0 - 2.0 * falloff_factor);

        // Height falloff within the volume.
        let height_factor = if self.height_falloff > 0.0 {
            let rel_y = world_pos[1] - (self.center[1] - self.half_extents[1]);
            let h = rel_y / (self.half_extents[1] * 2.0);
            (-h * self.height_falloff).exp()
        } else {
            1.0
        };

        // Noise.
        let noise_factor = if self.noise_amp > 0.0 {
            let nx = world_pos[0] * self.noise_freq + self.wind[0] * time;
            let ny = world_pos[1] * self.noise_freq + self.wind[1] * time;
            let nz = world_pos[2] * self.noise_freq + self.wind[2] * time;
            let n = value_noise_3d(nx, ny, nz);
            1.0 + (n - 0.5) * 2.0 * self.noise_amp
        } else {
            1.0
        };

        let density = self.density * smooth * height_factor * noise_factor.max(0.0);

        FogVolumeResult {
            density,
            color: self.color,
            blend_mode: self.blend_mode,
            priority: self.priority,
        }
    }
}

/// Result of evaluating a fog volume at a point.
#[derive(Debug, Clone)]
pub struct FogVolumeResult {
    pub density: f32,
    pub color: [f32; 3],
    pub blend_mode: FogVolumeBlend,
    pub priority: i32,
}

impl FogVolumeResult {
    /// No fog contribution.
    pub fn none() -> Self {
        Self {
            density: 0.0,
            color: [0.0, 0.0, 0.0],
            blend_mode: FogVolumeBlend::Additive,
            priority: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Fog evaluation pipeline
// ---------------------------------------------------------------------------

/// Complete fog evaluation for a single pixel/fragment.
///
/// # Arguments
/// * `camera_pos` — Camera world position.
/// * `world_pos` — Fragment world position.
/// * `view_dir` — Normalised view direction from camera to fragment.
/// * `distance` — Linear distance from camera to fragment.
/// * `params` — Global fog parameters.
/// * `volumes` — Active fog volumes.
/// * `fog_override` — Optional per-object fog override.
/// * `time` — Current time (for animation).
///
/// # Returns
/// `(fog_factor, fog_color)` — Factor in [0, 1] and the inscatter colour.
pub fn evaluate_fog(
    camera_pos: [f32; 3],
    world_pos: [f32; 3],
    view_dir: [f32; 3],
    distance: f32,
    params: &GlobalFogParams,
    volumes: &[FogVolume],
    fog_override: Option<&FogOverride>,
    time: f32,
) -> (f32, [f32; 3]) {
    // Apply per-object override to parameters if present.
    let effective_params = if let Some(ovr) = fog_override {
        ovr.apply_to_params(params)
    } else {
        params.clone()
    };

    // Distance fog.
    let mut fog_factor = compute_distance_fog(
        effective_params.mode,
        distance,
        effective_params.density,
        effective_params.linear_start,
        effective_params.linear_end,
    );

    // Animated density modulation.
    if effective_params.animated {
        let noise_mod = animated_fog_noise(world_pos, time, &effective_params);
        fog_factor *= noise_mod;
    }

    // Height fog (additive).
    if effective_params.height_fog_enabled {
        let height_fog = compute_height_fog(
            camera_pos[1],
            world_pos[1],
            distance.min(effective_params.height_fog_max_distance),
            effective_params.height_fog_base,
            effective_params.height_fog_falloff,
            effective_params.height_fog_density,
        );
        fog_factor = 1.0 - (1.0 - fog_factor) * (1.0 - height_fog);
    }

    // Fog volumes.
    let mut volume_density = 0.0_f32;
    let mut volume_color = [0.0_f32; 3];
    let mut best_priority = i32::MIN;
    let mut has_replace = false;

    for vol in volumes {
        let result = vol.evaluate(world_pos, time);
        if result.density <= 0.0 {
            continue;
        }

        match result.blend_mode {
            FogVolumeBlend::Additive => {
                volume_density += result.density;
                volume_color[0] += result.color[0] * result.density;
                volume_color[1] += result.color[1] * result.density;
                volume_color[2] += result.color[2] * result.density;
            }
            FogVolumeBlend::Replace => {
                if result.priority >= best_priority {
                    best_priority = result.priority;
                    volume_density = result.density;
                    volume_color = result.color;
                    has_replace = true;
                }
            }
            FogVolumeBlend::Multiply => {
                fog_factor *= 1.0 + result.density;
            }
            FogVolumeBlend::Max => {
                volume_density = volume_density.max(result.density);
                if result.density > volume_density - 0.001 {
                    volume_color = result.color;
                }
            }
        }
    }

    // Combine global fog with volume fog.
    let mut final_color = effective_params.color_source.evaluate(view_dir);

    if has_replace {
        let vol_factor = (1.0 - (-volume_density * distance * 0.01).exp()).clamp(0.0, 1.0);
        fog_factor = vol_factor;
        final_color = volume_color;
    } else if volume_density > 0.0 {
        let vol_factor = (1.0 - (-volume_density * distance * 0.01).exp()).clamp(0.0, 1.0);
        fog_factor = 1.0 - (1.0 - fog_factor) * (1.0 - vol_factor);

        // Weighted blend of colours.
        let w_global = fog_factor - vol_factor;
        let w_vol = vol_factor;
        let total_w = w_global + w_vol;
        if total_w > 0.0 {
            if volume_density > 0.0 {
                let inv_d = 1.0 / volume_density;
                volume_color[0] *= inv_d;
                volume_color[1] *= inv_d;
                volume_color[2] *= inv_d;
            }
            final_color[0] = (final_color[0] * w_global + volume_color[0] * w_vol) / total_w;
            final_color[1] = (final_color[1] * w_global + volume_color[1] * w_vol) / total_w;
            final_color[2] = (final_color[2] * w_global + volume_color[2] * w_vol) / total_w;
        }
    }

    // Clamp to max opacity.
    fog_factor = fog_factor.min(effective_params.max_opacity);

    // Apply per-object override to factor.
    if let Some(ovr) = fog_override {
        fog_factor = ovr.apply_to_factor(fog_factor);
    }

    (fog_factor, final_color)
}

/// Apply fog to a fragment colour.
///
/// # Arguments
/// * `scene_color` — Original scene colour (linear RGB).
/// * `fog_factor` — Fog factor [0, 1].
/// * `fog_color` — Fog inscatter colour (linear RGB).
///
/// # Returns
/// Fogged colour.
#[inline]
pub fn apply_fog(scene_color: [f32; 3], fog_factor: f32, fog_color: [f32; 3]) -> [f32; 3] {
    [
        lerp(scene_color[0], fog_color[0], fog_factor),
        lerp(scene_color[1], fog_color[1], fog_factor),
        lerp(scene_color[2], fog_color[2], fog_factor),
    ]
}

// ---------------------------------------------------------------------------
// GPU uniform data
// ---------------------------------------------------------------------------

/// Packed fog uniform data for GPU upload.
///
/// Layout designed for a single `uniform` block in WGSL/GLSL.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FogUniforms {
    /// Fog colour (xyz) + mode (w: 0=none, 1=linear, 2=exp, 3=exp2).
    pub color_and_mode: [f32; 4],
    /// density, linear_start, linear_end, max_opacity.
    pub params: [f32; 4],
    /// height_fog_base, height_fog_falloff, height_fog_density, height_fog_enabled.
    pub height_params: [f32; 4],
    /// animation_speed(xyz), noise_frequency.
    pub animation: [f32; 4],
    /// noise_amplitude, noise_octaves, time, animated(0 or 1).
    pub animation2: [f32; 4],
}

impl FogUniforms {
    /// Build GPU uniform data from fog params.
    pub fn from_params(params: &GlobalFogParams, time: f32) -> Self {
        let color = params.color_source.evaluate([0.0, 0.0, 1.0]);
        let mode_f = match params.mode {
            FogMode::None => 0.0,
            FogMode::Linear => 1.0,
            FogMode::Exponential => 2.0,
            FogMode::ExponentialSquared => 3.0,
        };

        Self {
            color_and_mode: [color[0], color[1], color[2], mode_f],
            params: [params.density, params.linear_start, params.linear_end, params.max_opacity],
            height_params: [
                params.height_fog_base,
                params.height_fog_falloff,
                params.height_fog_density,
                if params.height_fog_enabled { 1.0 } else { 0.0 },
            ],
            animation: [
                params.animation_speed[0],
                params.animation_speed[1],
                params.animation_speed[2],
                params.noise_frequency,
            ],
            animation2: [
                params.noise_amplitude,
                params.noise_octaves as f32,
                time,
                if params.animated { 1.0 } else { 0.0 },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fog() {
        assert!((compute_distance_fog(FogMode::Linear, 0.0, 0.0, 10.0, 100.0) - 0.0).abs() < 1e-6);
        assert!((compute_distance_fog(FogMode::Linear, 55.0, 0.0, 10.0, 100.0) - 0.5).abs() < 1e-6);
        assert!((compute_distance_fog(FogMode::Linear, 100.0, 0.0, 10.0, 100.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_fog() {
        let f = compute_distance_fog(FogMode::Exponential, 100.0, 0.02, 0.0, 0.0);
        let expected = 1.0 - (-0.02 * 100.0_f32).exp();
        assert!((f - expected).abs() < 1e-6);
    }

    #[test]
    fn test_height_fog() {
        // At the base height, fog should be near maximum.
        let f = compute_height_fog(0.0, 0.0, 100.0, 0.0, 0.1, 0.05);
        assert!(f > 0.0);

        // High above, fog should be near zero.
        let f = compute_height_fog(100.0, 100.0, 100.0, 0.0, 1.0, 0.05);
        assert!(f < 0.01);
    }

    #[test]
    fn test_fog_volume_box() {
        let vol = FogVolume::new_box([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 0.1);

        // Inside the box.
        let result = vol.evaluate([0.0, 0.0, 0.0], 0.0);
        assert!(result.density > 0.0);

        // Outside the box (beyond falloff).
        let result = vol.evaluate([20.0, 0.0, 0.0], 0.0);
        assert!(result.density <= 0.001);
    }

    #[test]
    fn test_fog_override_no_fog() {
        let ovr = FogOverride::no_fog();
        let factor = ovr.apply_to_factor(0.8);
        assert!((factor - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_fog() {
        let scene = [1.0, 0.0, 0.0];
        let fog = [0.5, 0.5, 0.5];
        let result = apply_fog(scene, 0.5, fog);
        assert!((result[0] - 0.75).abs() < 1e-6);
        assert!((result[1] - 0.25).abs() < 1e-6);
    }
}
