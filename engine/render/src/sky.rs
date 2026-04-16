// engine/render/src/sky.rs
//
// Procedural sky components for the Genovo engine's ECS.
//
// Provides high-level components that wire the atmospheric scattering model
// (from [`crate::atmosphere`]) and HDR environment maps into the rendering
// pipeline.  These are the types an end user attaches to entities or the
// scene to control the sky.
//
// # Components
//
// - [`ProceduralSkybox`] — renders the sky procedurally from the atmosphere
//   model, including sun disc, stars, moon, and clouds.
// - [`HDRISkybox`] — renders from a pre-baked HDR environment map (cubemap
//   or equirectangular).
// - [`SkyComponent`] — ECS component wrapper for either sky type.
// - [`DayNightCycleManager`] — drives time-of-day and blends sky parameters
//   smoothly through the diurnal cycle.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

use crate::atmosphere::{
    AtmosphereParams, AtmosphereRenderState, CloudLayer, Moon, SkyBox, SkyViewLut, StarField,
    TimeOfDay, TransmittanceLut,
};

// ---------------------------------------------------------------------------
// ProceduralSkybox
// ---------------------------------------------------------------------------

/// Renders the sky procedurally from the atmosphere scattering model.
///
/// Attach this to a scene entity to enable physically-based sky rendering
/// without requiring a pre-baked cubemap.
#[derive(Clone)]
pub struct ProceduralSkybox {
    /// Complete atmosphere state (params + time-of-day + stars + moon + clouds).
    pub atmosphere: AtmosphereRenderState,
    /// Whether to render stars at night.
    pub enable_stars: bool,
    /// Whether to render the moon.
    pub enable_moon: bool,
    /// Whether to render clouds.
    pub enable_clouds: bool,
    /// Whether to render the sun disc.
    pub enable_sun_disc: bool,
    /// Resolution of the cubemap generated from the procedural sky
    /// (per face, in pixels).
    pub cubemap_resolution: u32,
    /// Whether to update the sky cubemap every frame (expensive) or only
    /// when marked dirty.
    pub update_every_frame: bool,
    /// Ground-level Y position (used to place the camera in atmosphere space).
    pub ground_y: f32,
    /// Camera altitude above ground (km) for atmosphere rendering.
    pub camera_altitude_km: f32,
    /// Tone-mapping exposure override.
    pub exposure: f32,
    /// Gamma correction (applied after tone mapping; 1.0 = linear).
    pub gamma: f32,
    /// Night sky brightness boost.
    pub night_brightness: f32,
    /// Whether the sky LUTs need regenerating.
    pub dirty: bool,
}

impl ProceduralSkybox {
    pub fn new() -> Self {
        Self {
            atmosphere: AtmosphereRenderState::new(),
            enable_stars: true,
            enable_moon: true,
            enable_clouds: true,
            enable_sun_disc: true,
            cubemap_resolution: 256,
            update_every_frame: false,
            ground_y: 0.0,
            camera_altitude_km: 0.001,
            exposure: 1.0,
            gamma: 2.2,
            night_brightness: 1.5,
            dirty: true,
        }
    }

    /// Creates with a specific time of day.
    pub fn with_time(mut self, hours: f32) -> Self {
        self.atmosphere.time_of_day.set_time(hours);
        self.dirty = true;
        self
    }

    /// Creates with a specific atmosphere preset.
    pub fn with_atmosphere(mut self, params: AtmosphereParams) -> Self {
        self.atmosphere.params = params;
        self.dirty = true;
        self
    }

    /// Sets the cubemap resolution (per face).
    pub fn with_resolution(mut self, resolution: u32) -> Self {
        self.cubemap_resolution = resolution;
        self
    }

    /// Returns the camera position in atmosphere space (km, planet-centred).
    pub fn camera_pos_atmosphere(&self) -> Vec3 {
        Vec3::new(
            0.0,
            self.atmosphere.params.planet_radius + self.camera_altitude_km,
            0.0,
        )
    }

    /// Updates the sky for the current frame.
    ///
    /// `dt` is the frame delta time in seconds.
    pub fn update(&mut self, dt: f32) {
        self.atmosphere.update(dt);
        self.dirty = true;
    }

    /// Rebuilds LUTs if dirty.
    pub fn rebuild_luts_if_needed(&mut self) {
        if !self.dirty && !self.update_every_frame {
            return;
        }
        let cam = self.camera_pos_atmosphere();
        self.atmosphere.rebuild_luts(cam);
        self.dirty = false;
    }

    /// Computes the sky colour for a given view direction.
    ///
    /// This is the "final answer" after compositing atmosphere, sun, moon,
    /// stars, and clouds.
    pub fn sample_sky(&self, view_dir: Vec3) -> Vec3 {
        let cam = self.camera_pos_atmosphere();
        let mut color = self.atmosphere.compute_pixel(cam, view_dir);

        // Apply night brightness boost.
        let star_vis = self.atmosphere.time_of_day.star_visibility();
        if star_vis > 0.0 {
            color *= 1.0 + (self.night_brightness - 1.0) * star_vis;
        }

        // Tone map (simple Reinhard).
        color = tone_map_reinhard(color * self.exposure);

        // Gamma.
        if (self.gamma - 1.0).abs() > 0.001 {
            let inv_gamma = 1.0 / self.gamma;
            color = Vec3::new(
                color.x.powf(inv_gamma),
                color.y.powf(inv_gamma),
                color.z.powf(inv_gamma),
            );
        }

        color
    }

    /// Generates a cubemap (6 faces) of the current sky.
    ///
    /// Returns an array of 6 buffers, each `resolution × resolution` RGB f32
    /// values.  Face order: +X, −X, +Y, −Y, +Z, −Z.
    pub fn generate_cubemap(&self) -> [Vec<Vec3>; 6] {
        let res = self.cubemap_resolution;
        let face_dirs = cubemap_face_directions();
        let mut faces: [Vec<Vec3>; 6] = Default::default();

        for (face_idx, (right, up, forward)) in face_dirs.iter().enumerate() {
            let mut pixels = Vec::with_capacity((res * res) as usize);

            for y in 0..res {
                for x in 0..res {
                    let u = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;

                    let dir = (*forward + *right * u + *up * (-v)).normalize();
                    let color = self.sample_sky(dir);
                    pixels.push(color);
                }
            }

            faces[face_idx] = pixels;
        }

        faces
    }

    /// Generates the sky as an equirectangular map.
    ///
    /// Returns `width × height` RGB f32 values.
    pub fn generate_equirectangular(&self, width: u32, height: u32) -> Vec<Vec3> {
        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            let v = y as f32 / (height - 1).max(1) as f32;
            let theta = v * PI; // 0 (top) to pi (bottom)

            for x in 0..width {
                let u = x as f32 / (width - 1).max(1) as f32;
                let phi = u * 2.0 * PI; // 0 to 2pi

                let sin_theta = theta.sin();
                let dir = Vec3::new(
                    sin_theta * phi.sin(),
                    theta.cos(),
                    sin_theta * phi.cos(),
                )
                .normalize();

                let color = self.sample_sky(dir);
                pixels.push(color);
            }
        }

        pixels
    }

    /// Converts a cubemap face to RGBA u8 for display/debug.
    pub fn face_to_rgba8(pixels: &[Vec3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(pixels.len() * 4);
        for p in pixels {
            out.push((p.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((p.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((p.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push(255);
        }
        out
    }
}

impl Default for ProceduralSkybox {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HDRISkybox
// ---------------------------------------------------------------------------

/// Tone-mapping mode for HDR sky rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapMode {
    /// No tone mapping (linear pass-through).
    None,
    /// Simple Reinhard: color / (1 + color).
    Reinhard,
    /// ACES filmic (approximate).
    AcesFilmic,
    /// Uncharted 2 filmic.
    Uncharted2,
    /// AgX (Blender 4.0+).
    AgX,
}

/// Renders the sky from a pre-baked HDR environment map.
#[derive(Debug, Clone)]
pub struct HDRISkybox {
    /// Path to the HDR image (equirectangular or cubemap).
    pub path: String,
    /// Whether the source is equirectangular (true) or cubemap (false).
    pub is_equirectangular: bool,
    /// Rotation applied to the environment map (Euler YXZ, radians).
    pub rotation: Vec3,
    /// Brightness/exposure multiplier.
    pub exposure: f32,
    /// Tone-mapping mode.
    pub tone_map: ToneMapMode,
    /// Gamma correction value.
    pub gamma: f32,
    /// Tint colour multiplied into the sky.
    pub tint: Vec3,
    /// Whether this HDRI contributes to scene ambient/IBL lighting.
    pub affects_lighting: bool,
    /// LOD bias for cubemap sampling.
    pub lod_bias: f32,
    /// Background blur amount (0 = sharp, 1 = fully blurred).
    pub blur: f32,
    /// Whether the HDRI has been loaded and processed.
    pub loaded: bool,
    /// Cubemap face resolution (after processing).
    pub cubemap_resolution: u32,
}

impl HDRISkybox {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            is_equirectangular: true,
            rotation: Vec3::ZERO,
            exposure: 1.0,
            tone_map: ToneMapMode::AcesFilmic,
            gamma: 2.2,
            tint: Vec3::ONE,
            affects_lighting: true,
            lod_bias: 0.0,
            blur: 0.0,
            loaded: false,
            cubemap_resolution: 512,
        }
    }

    pub fn with_rotation(mut self, y_degrees: f32) -> Self {
        self.rotation.y = y_degrees.to_radians();
        self
    }

    pub fn with_exposure(mut self, exposure: f32) -> Self {
        self.exposure = exposure;
        self
    }

    pub fn with_tone_map(mut self, mode: ToneMapMode) -> Self {
        self.tone_map = mode;
        self
    }

    pub fn with_tint(mut self, tint: Vec3) -> Self {
        self.tint = tint;
        self
    }

    /// Returns the rotation matrix for the environment map.
    pub fn rotation_matrix(&self) -> Mat4 {
        Mat4::from_rotation_y(self.rotation.y)
            * Mat4::from_rotation_x(self.rotation.x)
            * Mat4::from_rotation_z(self.rotation.z)
    }
}

impl Default for HDRISkybox {
    fn default() -> Self {
        Self::new("default_sky.hdr")
    }
}

// ---------------------------------------------------------------------------
// SkyComponent (ECS component)
// ---------------------------------------------------------------------------

/// The type of sky rendering.
#[derive(Clone)]
pub enum SkyType {
    /// Procedurally generated atmosphere.
    Procedural(ProceduralSkybox),
    /// Pre-baked HDR environment map.
    HDRI(HDRISkybox),
    /// Solid colour (for debugging or stylised rendering).
    SolidColor(Vec3),
    /// Gradient sky (top colour, horizon colour, bottom colour).
    Gradient {
        top: Vec3,
        horizon: Vec3,
        bottom: Vec3,
        exponent: f32,
    },
}

/// ECS component that controls which sky is active in the scene.
///
/// Attach to an entity to set the sky.  Only one `SkyComponent` should be
/// active at a time; if multiple exist, the renderer uses the one with the
/// highest `priority`.
#[derive(Clone)]
pub struct SkyComponent {
    /// The sky type / data.
    pub sky: SkyType,
    /// Priority for conflict resolution (higher wins).
    pub priority: i32,
    /// Whether this sky is currently active.
    pub active: bool,
    /// Ambient lighting contribution from this sky.
    pub ambient_intensity: f32,
    /// Environment map contribution for reflections.
    pub reflection_intensity: f32,
}

impl SkyComponent {
    /// Creates a procedural sky component.
    pub fn procedural() -> Self {
        Self {
            sky: SkyType::Procedural(ProceduralSkybox::new()),
            priority: 0,
            active: true,
            ambient_intensity: 1.0,
            reflection_intensity: 1.0,
        }
    }

    /// Creates an HDRI sky component.
    pub fn hdri(path: impl Into<String>) -> Self {
        Self {
            sky: SkyType::HDRI(HDRISkybox::new(path)),
            priority: 0,
            active: true,
            ambient_intensity: 1.0,
            reflection_intensity: 1.0,
        }
    }

    /// Creates a solid-colour sky component.
    pub fn solid(color: Vec3) -> Self {
        Self {
            sky: SkyType::SolidColor(color),
            priority: -1,
            active: true,
            ambient_intensity: 0.5,
            reflection_intensity: 0.5,
        }
    }

    /// Creates a gradient sky component.
    pub fn gradient(top: Vec3, horizon: Vec3, bottom: Vec3) -> Self {
        Self {
            sky: SkyType::Gradient {
                top,
                horizon,
                bottom,
                exponent: 2.0,
            },
            priority: -1,
            active: true,
            ambient_intensity: 0.7,
            reflection_intensity: 0.7,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Samples the sky colour for a given view direction.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        match &self.sky {
            SkyType::Procedural(sky) => sky.sample_sky(view_dir),
            SkyType::HDRI(_hdri) => {
                // In a full implementation this would sample the loaded cubemap.
                // Here we return a placeholder sky gradient.
                let t = view_dir.y * 0.5 + 0.5;
                Vec3::new(0.4, 0.6, 0.9) * t + Vec3::new(0.8, 0.85, 0.9) * (1.0 - t)
            }
            SkyType::SolidColor(c) => *c,
            SkyType::Gradient {
                top,
                horizon,
                bottom,
                exponent,
            } => {
                let y = view_dir.y;
                if y >= 0.0 {
                    let t = y.powf(1.0 / exponent);
                    *horizon * (1.0 - t) + *top * t
                } else {
                    let t = (-y).powf(1.0 / exponent);
                    *horizon * (1.0 - t) + *bottom * t
                }
            }
        }
    }

    /// Updates the sky for a frame (only relevant for procedural skies).
    pub fn update(&mut self, dt: f32) {
        if let SkyType::Procedural(sky) = &mut self.sky {
            sky.update(dt);
        }
    }
}

impl Default for SkyComponent {
    fn default() -> Self {
        Self::procedural()
    }
}

// ---------------------------------------------------------------------------
// DayNightCycleManager
// ---------------------------------------------------------------------------

/// Manages the full day/night cycle, providing smooth transitions between
/// key lighting states (dawn, day, dusk, night).
#[derive(Clone)]
pub struct DayNightCycleManager {
    /// The underlying time-of-day controller.
    pub time_of_day: TimeOfDay,
    /// Key-frame lighting presets at specific times.
    pub key_frames: Vec<SkyKeyFrame>,
    /// Whether the cycle is running.
    pub running: bool,
    /// Cached interpolated values from the latest update.
    pub current_state: SkyKeyFrameState,
}

/// A key-frame for sky/lighting parameters at a specific time of day.
#[derive(Debug, Clone)]
pub struct SkyKeyFrame {
    /// Time in hours [0, 24).
    pub time_hours: f32,
    /// Sun colour / intensity.
    pub sun_color: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Ambient light colour.
    pub ambient_color: Vec3,
    /// Ambient intensity.
    pub ambient_intensity: f32,
    /// Fog colour.
    pub fog_color: Vec3,
    /// Fog density.
    pub fog_density: f32,
    /// Sky tint (multiplied into the atmosphere result).
    pub sky_tint: Vec3,
    /// Shadow intensity [0, 1].
    pub shadow_intensity: f32,
    /// Cloud coverage [0, 1].
    pub cloud_coverage: f32,
    /// Exposure override.
    pub exposure: f32,
}

impl SkyKeyFrame {
    /// Creates a key-frame for bright daylight.
    pub fn daylight(time: f32) -> Self {
        Self {
            time_hours: time,
            sun_color: Vec3::new(1.0, 0.98, 0.92),
            sun_intensity: 1.0,
            ambient_color: Vec3::new(0.15, 0.18, 0.25),
            ambient_intensity: 0.3,
            fog_color: Vec3::new(0.7, 0.75, 0.8),
            fog_density: 0.01,
            sky_tint: Vec3::ONE,
            shadow_intensity: 0.8,
            cloud_coverage: 0.4,
            exposure: 1.0,
        }
    }

    /// Creates a key-frame for dawn / golden hour.
    pub fn dawn(time: f32) -> Self {
        Self {
            time_hours: time,
            sun_color: Vec3::new(1.0, 0.6, 0.3),
            sun_intensity: 0.6,
            ambient_color: Vec3::new(0.12, 0.08, 0.06),
            ambient_intensity: 0.15,
            fog_color: Vec3::new(0.8, 0.5, 0.3),
            fog_density: 0.03,
            sky_tint: Vec3::new(1.0, 0.85, 0.7),
            shadow_intensity: 0.5,
            cloud_coverage: 0.3,
            exposure: 0.8,
        }
    }

    /// Creates a key-frame for dusk / sunset.
    pub fn dusk(time: f32) -> Self {
        Self {
            time_hours: time,
            sun_color: Vec3::new(1.0, 0.45, 0.2),
            sun_intensity: 0.5,
            ambient_color: Vec3::new(0.1, 0.06, 0.08),
            ambient_intensity: 0.12,
            fog_color: Vec3::new(0.6, 0.35, 0.25),
            fog_density: 0.025,
            sky_tint: Vec3::new(1.0, 0.7, 0.5),
            shadow_intensity: 0.4,
            cloud_coverage: 0.35,
            exposure: 0.7,
        }
    }

    /// Creates a key-frame for night.
    pub fn night(time: f32) -> Self {
        Self {
            time_hours: time,
            sun_color: Vec3::new(0.3, 0.4, 0.6),
            sun_intensity: 0.05,
            ambient_color: Vec3::new(0.02, 0.03, 0.06),
            ambient_intensity: 0.05,
            fog_color: Vec3::new(0.05, 0.05, 0.1),
            fog_density: 0.005,
            sky_tint: Vec3::new(0.6, 0.65, 0.9),
            shadow_intensity: 0.1,
            cloud_coverage: 0.2,
            exposure: 0.4,
        }
    }

    /// Linearly interpolates between two key-frames.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> SkyKeyFrameState {
        let t = t.clamp(0.0, 1.0);
        SkyKeyFrameState {
            sun_color: a.sun_color * (1.0 - t) + b.sun_color * t,
            sun_intensity: a.sun_intensity * (1.0 - t) + b.sun_intensity * t,
            ambient_color: a.ambient_color * (1.0 - t) + b.ambient_color * t,
            ambient_intensity: a.ambient_intensity * (1.0 - t) + b.ambient_intensity * t,
            fog_color: a.fog_color * (1.0 - t) + b.fog_color * t,
            fog_density: a.fog_density * (1.0 - t) + b.fog_density * t,
            sky_tint: a.sky_tint * (1.0 - t) + b.sky_tint * t,
            shadow_intensity: a.shadow_intensity * (1.0 - t) + b.shadow_intensity * t,
            cloud_coverage: a.cloud_coverage * (1.0 - t) + b.cloud_coverage * t,
            exposure: a.exposure * (1.0 - t) + b.exposure * t,
        }
    }
}

/// Interpolated state derived from key-frames.
#[derive(Debug, Clone)]
pub struct SkyKeyFrameState {
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub fog_color: Vec3,
    pub fog_density: f32,
    pub sky_tint: Vec3,
    pub shadow_intensity: f32,
    pub cloud_coverage: f32,
    pub exposure: f32,
}

impl Default for SkyKeyFrameState {
    fn default() -> Self {
        Self {
            sun_color: Vec3::ONE,
            sun_intensity: 1.0,
            ambient_color: Vec3::splat(0.15),
            ambient_intensity: 0.3,
            fog_color: Vec3::splat(0.7),
            fog_density: 0.01,
            sky_tint: Vec3::ONE,
            shadow_intensity: 0.8,
            cloud_coverage: 0.4,
            exposure: 1.0,
        }
    }
}

impl DayNightCycleManager {
    /// Creates a new cycle manager with default Earth key-frames.
    pub fn new() -> Self {
        let key_frames = vec![
            SkyKeyFrame::night(0.0),
            SkyKeyFrame::dawn(6.0),
            SkyKeyFrame::daylight(10.0),
            SkyKeyFrame::daylight(14.0),
            SkyKeyFrame::dusk(18.5),
            SkyKeyFrame::night(21.0),
            SkyKeyFrame::night(24.0),
        ];

        Self {
            time_of_day: TimeOfDay::new(),
            key_frames,
            running: true,
            current_state: SkyKeyFrameState::default(),
        }
    }

    /// Advances the cycle and interpolates key-frames.
    pub fn update(&mut self, dt: f32) {
        if self.running {
            self.time_of_day.update(dt);
        }
        self.interpolate();
    }

    /// Sets the time directly and re-interpolates.
    pub fn set_time(&mut self, hours: f32) {
        self.time_of_day.set_time(hours);
        self.interpolate();
    }

    /// Finds the two surrounding key-frames and interpolates.
    fn interpolate(&mut self) {
        let t = self.time_of_day.time_hours;

        if self.key_frames.len() < 2 {
            return;
        }

        // Find the bracketing key-frames.
        let mut prev_idx = 0;
        let mut next_idx = 1;
        for (i, kf) in self.key_frames.iter().enumerate() {
            if kf.time_hours <= t {
                prev_idx = i;
                next_idx = (i + 1).min(self.key_frames.len() - 1);
            }
        }

        let prev = &self.key_frames[prev_idx];
        let next = &self.key_frames[next_idx];

        let range = next.time_hours - prev.time_hours;
        let frac = if range.abs() > 0.001 {
            ((t - prev.time_hours) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.current_state = SkyKeyFrame::lerp(prev, next, frac);
    }

    /// Applies the current interpolated state to atmosphere parameters.
    pub fn apply_to_atmosphere(&self, params: &mut AtmosphereParams) {
        params.sun_direction = self.time_of_day.sun_direction();
        params.sun_intensity = Vec3::splat(20.0) * self.current_state.sun_color
            * self.current_state.sun_intensity;
        params.exposure = self.current_state.exposure;
    }

    /// Applies the current state to a procedural skybox.
    pub fn apply_to_skybox(&self, skybox: &mut ProceduralSkybox) {
        skybox.atmosphere.time_of_day = self.time_of_day.clone();
        self.apply_to_atmosphere(&mut skybox.atmosphere.params);
        skybox.atmosphere.cloud_layer.coverage = self.current_state.cloud_coverage;
        skybox.exposure = self.current_state.exposure;
        skybox.dirty = true;
    }

    /// Returns the sun direction from the current time.
    pub fn sun_direction(&self) -> Vec3 {
        self.time_of_day.sun_direction()
    }

    /// Returns the current sun colour (interpolated).
    pub fn sun_color(&self) -> Vec3 {
        self.current_state.sun_color * self.current_state.sun_intensity
    }

    /// Returns the current ambient colour.
    pub fn ambient_color(&self) -> Vec3 {
        self.current_state.ambient_color * self.current_state.ambient_intensity
    }

    /// Returns the current fog parameters.
    pub fn fog_params(&self) -> (Vec3, f32) {
        (self.current_state.fog_color, self.current_state.fog_density)
    }
}

impl Default for DayNightCycleManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tone mapping
// ---------------------------------------------------------------------------

/// Reinhard tone mapping: `c / (1 + c)`.
fn tone_map_reinhard(color: Vec3) -> Vec3 {
    Vec3::new(
        color.x / (1.0 + color.x),
        color.y / (1.0 + color.y),
        color.z / (1.0 + color.z),
    )
}

/// ACES filmic approximation (Krzysztof Narkowicz).
#[allow(dead_code)]
fn tone_map_aces(color: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let apply = |x: f32| -> f32 {
        ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    };

    Vec3::new(apply(color.x), apply(color.y), apply(color.z))
}

/// Uncharted 2 filmic tone mapping.
#[allow(dead_code)]
fn tone_map_uncharted2(color: Vec3) -> Vec3 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;

    let uncharted = |x: f32| -> f32 {
        ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
    };

    let white = 11.2;
    let scale = 1.0 / uncharted(white);

    Vec3::new(
        uncharted(color.x) * scale,
        uncharted(color.y) * scale,
        uncharted(color.z) * scale,
    )
}

// ---------------------------------------------------------------------------
// Cubemap helpers
// ---------------------------------------------------------------------------

/// Returns the (right, up, forward) vectors for each cubemap face.
///
/// Face order: +X, −X, +Y, −Y, +Z, −Z.
fn cubemap_face_directions() -> [(Vec3, Vec3, Vec3); 6] {
    [
        // +X
        (Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        // -X
        (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        // +Y
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0)),
        // -Y
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0)),
        // +Z
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        // -Z
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
    ]
}

// ---------------------------------------------------------------------------
// WGSL sky rendering shader
// ---------------------------------------------------------------------------

/// WGSL vertex + fragment shader for rendering the sky as a full-screen
/// triangle or skybox cube.
pub const SKY_RENDER_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Sky rendering shader (Genovo Engine)
// -----------------------------------------------------------------------
// Renders the sky from a precomputed sky-view LUT or cubemap.
// Uses a full-screen triangle approach.

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    exposure: f32,
    sun_direction: vec3<f32>,
    sun_angular_radius: f32,
    sky_tint: vec3<f32>,
    gamma: f32,
};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;
@group(0) @binding(1) var sky_lut: texture_2d<f32>;
@group(0) @binding(2) var sky_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

const PI: f32 = 3.141592653589793;

fn reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3<f32>(1.0) + c);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space view direction from screen UV.
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = sky.inv_view_proj * ndc;
    let view_dir = normalize(world_pos.xyz / world_pos.w - sky.camera_pos);

    // Compute LUT coordinates from view direction.
    let elevation = asin(view_dir.y);
    let azimuth = atan2(view_dir.z, view_dir.x);

    let u = azimuth / (2.0 * PI) + 0.5;
    let v = elevation / PI + 0.5;

    var color = textureSample(sky_lut, sky_sampler, vec2<f32>(u, v)).rgb;

    // Sun disc.
    let cos_angle = dot(view_dir, normalize(sky.sun_direction));
    if cos_angle > cos(sky.sun_angular_radius) {
        let angle = acos(clamp(cos_angle, -1.0, 1.0));
        let r = angle / sky.sun_angular_radius;
        let mu = sqrt(max(0.0, 1.0 - r * r));
        let limb = 0.6 + 0.4 * mu;
        color += sky.sun_direction * 50.0 * limb;
    }

    // Apply tint.
    color *= sky.sky_tint;

    // Tone map.
    color = reinhard(color * sky.exposure);

    // Gamma.
    let inv_gamma = 1.0 / sky.gamma;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(inv_gamma));

    return vec4<f32>(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_sky_sample() {
        let sky = ProceduralSkybox::new().with_time(12.0);
        let color = sky.sample_sky(Vec3::new(0.0, 1.0, 0.0));
        // Should not be black at noon looking up.
        assert!(color.x > 0.0 || color.y > 0.0 || color.z > 0.0);
    }

    #[test]
    fn sky_component_gradient() {
        let comp = SkyComponent::gradient(
            Vec3::new(0.2, 0.4, 0.8),
            Vec3::new(0.8, 0.8, 0.9),
            Vec3::new(0.1, 0.1, 0.1),
        );
        let up = comp.sample(Vec3::Y);
        let down = comp.sample(-Vec3::Y);
        // Looking up should be closer to top colour.
        assert!(up.z > down.z, "Up sky should be bluer than down");
    }

    #[test]
    fn day_night_cycle_interpolation() {
        let mut mgr = DayNightCycleManager::new();
        mgr.set_time(12.0);
        assert!(
            mgr.current_state.sun_intensity > 0.5,
            "Noon should have high sun intensity"
        );

        mgr.set_time(0.0);
        assert!(
            mgr.current_state.sun_intensity < 0.2,
            "Midnight should have low sun intensity"
        );
    }

    #[test]
    fn day_night_sun_direction() {
        let mut mgr = DayNightCycleManager::new();
        mgr.set_time(12.0);
        let dir = mgr.sun_direction();
        assert!(dir.y > 0.0, "Sun should be above horizon at noon");
    }

    #[test]
    fn tone_map_reinhard_bounded() {
        let result = tone_map_reinhard(Vec3::new(100.0, 100.0, 100.0));
        assert!(result.x < 1.0);
        assert!(result.x > 0.9);
    }

    #[test]
    fn tone_map_aces_bounded() {
        let result = tone_map_aces(Vec3::new(100.0, 100.0, 100.0));
        assert!(result.x <= 1.0);
    }

    #[test]
    fn hdri_skybox_creation() {
        let hdri = HDRISkybox::new("test_env.hdr")
            .with_rotation(45.0)
            .with_exposure(1.5);
        assert!((hdri.rotation.y - 45.0_f32.to_radians()).abs() < 0.01);
        assert!((hdri.exposure - 1.5).abs() < 0.01);
    }

    #[test]
    fn cubemap_face_directions_orthogonal() {
        let faces = cubemap_face_directions();
        for (right, up, forward) in &faces {
            let dot_ru = right.dot(*up).abs();
            let dot_rf = right.dot(*forward).abs();
            let dot_uf = up.dot(*forward).abs();
            assert!(dot_ru < 0.001, "right·up should be ~0");
            assert!(dot_rf < 0.001, "right·forward should be ~0");
            assert!(dot_uf < 0.001, "up·forward should be ~0");
        }
    }

    #[test]
    fn sky_key_frame_lerp() {
        let dawn = SkyKeyFrame::dawn(6.0);
        let day = SkyKeyFrame::daylight(12.0);
        let mid = SkyKeyFrame::lerp(&dawn, &day, 0.5);
        // Interpolated intensity should be between dawn and day.
        assert!(mid.sun_intensity > dawn.sun_intensity);
        assert!(mid.sun_intensity < day.sun_intensity);
    }
}
