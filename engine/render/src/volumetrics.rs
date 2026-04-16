// engine/render/src/volumetrics.rs
//
// Volumetric fog and lighting for the Genovo engine. Implements:
//
// - Global exponential height fog.
// - Volumetric fog with 3D noise-driven density variation.
// - Froxel-based volumetric fog (subdividing the view frustum into a 3D grid
//   for efficient per-froxel lighting accumulation).
// - God rays / light shafts via radial blur from a light source.
// - Volumetric light raymarching through light volumes (spotlights, etc.).

use glam::{Mat4, Vec2, Vec3, Vec4};

// ---------------------------------------------------------------------------
// FogSettings
// ---------------------------------------------------------------------------

/// Global fog configuration for the scene.
#[derive(Debug, Clone)]
pub struct FogSettings {
    /// If `true`, fog is enabled.
    pub enabled: bool,
    /// Global fog color.
    pub color: Vec3,
    /// Global fog density (exponential fog coefficient).
    pub global_density: f32,
    /// Height fog density. Applied on top of global density, increasing
    /// density below a certain height.
    pub height_fog_density: f32,
    /// Height fog falloff rate. Higher = denser fog closer to the ground.
    pub height_fog_falloff: f32,
    /// Height at which height fog density is at `height_fog_density`.
    pub height_fog_base: f32,
    /// Distance from the camera at which fog begins.
    pub start_distance: f32,
    /// Maximum fog density (clamped). 1.0 = fully opaque.
    pub max_density: f32,
    /// Inscattering color multiplier (tints the fog from light sources).
    pub inscattering_color: Vec3,
    /// Inscattering intensity.
    pub inscattering_intensity: f32,
    /// Directional inscattering (how much the fog scatters light towards
    /// the camera from directional lights). Uses Henyey-Greenstein phase.
    pub directional_inscattering_exponent: f32,
    /// Noise-based density variation.
    pub noise: Option<FogNoiseSettings>,
}

impl FogSettings {
    /// Creates default fog settings.
    pub fn new() -> Self {
        Self {
            enabled: true,
            color: Vec3::new(0.7, 0.75, 0.8),
            global_density: 0.02,
            height_fog_density: 0.1,
            height_fog_falloff: 2.0,
            height_fog_base: 0.0,
            start_distance: 10.0,
            max_density: 1.0,
            inscattering_color: Vec3::new(1.0, 0.95, 0.85),
            inscattering_intensity: 1.0,
            directional_inscattering_exponent: 8.0,
            noise: None,
        }
    }

    /// Sets the fog color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Vec3::new(r, g, b);
        self
    }

    /// Sets the global density.
    pub fn with_density(mut self, density: f32) -> Self {
        self.global_density = density;
        self
    }

    /// Sets height fog parameters.
    pub fn with_height_fog(
        mut self,
        density: f32,
        falloff: f32,
        base_height: f32,
    ) -> Self {
        self.height_fog_density = density;
        self.height_fog_falloff = falloff;
        self.height_fog_base = base_height;
        self
    }

    /// Sets the start distance.
    pub fn with_start_distance(mut self, dist: f32) -> Self {
        self.start_distance = dist;
        self
    }

    /// Enables noise-based density variation.
    pub fn with_noise(mut self, noise: FogNoiseSettings) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Computes the fog density at a given world-space position.
    ///
    /// This combines global density and height fog.
    pub fn density_at(&self, position: Vec3, time: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }

        // Base global density.
        let mut density = self.global_density;

        // Height fog: exponential falloff below base height.
        let height_diff = self.height_fog_base - position.y;
        if height_diff > 0.0 {
            let height_factor =
                1.0 - (-height_diff * self.height_fog_falloff).exp();
            density += self.height_fog_density * height_factor;
        }

        // Noise modulation.
        if let Some(noise) = &self.noise {
            let noise_value = noise.sample(position, time);
            // noise_value in [-1, 1], map to [0, 1] then multiply.
            let noise_mult = (noise_value * 0.5 + 0.5)
                .clamp(0.0, 1.0)
                .powf(noise.contrast);
            density *= noise.intensity * noise_mult + (1.0 - noise.intensity);
        }

        density.clamp(0.0, self.max_density)
    }

    /// Computes the transmittance (how much light reaches the camera) along
    /// a ray using raymarching.
    ///
    /// # Arguments
    /// * `ray_origin` - Camera position.
    /// * `ray_dir` - Normalized direction from camera to sample point.
    /// * `distance` - Distance from camera to sample point.
    /// * `time` - Current time (for animated noise).
    /// * `num_steps` - Number of raymarching steps.
    ///
    /// # Returns
    /// `(transmittance, inscattered_light)` where transmittance is in [0, 1]
    /// and inscattered_light is the accumulated fog color.
    pub fn raymarch(
        &self,
        ray_origin: Vec3,
        ray_dir: Vec3,
        distance: f32,
        time: f32,
        num_steps: u32,
    ) -> (f32, Vec3) {
        if !self.enabled || num_steps == 0 {
            return (1.0, Vec3::ZERO);
        }

        let effective_distance = (distance - self.start_distance).max(0.0);
        if effective_distance <= 0.0 {
            return (1.0, Vec3::ZERO);
        }

        let step_size = effective_distance / num_steps as f32;
        let mut transmittance = 1.0;
        let mut inscattered = Vec3::ZERO;
        let start_offset = self.start_distance;

        for i in 0..num_steps {
            let t = start_offset + (i as f32 + 0.5) * step_size;
            let sample_pos = ray_origin + ray_dir * t;

            let local_density = self.density_at(sample_pos, time);
            let sample_transmittance = (-local_density * step_size).exp();

            // Accumulate inscattering.
            let sample_inscattered = self.color * local_density * step_size;
            inscattered += sample_inscattered * transmittance;

            transmittance *= sample_transmittance;

            // Early termination if fully opaque.
            if transmittance < 0.001 {
                transmittance = 0.0;
                break;
            }
        }

        (transmittance, inscattered)
    }

    /// Computes simple analytical exponential fog (no raymarching).
    ///
    /// # Arguments
    /// * `distance` - Distance from camera to fragment.
    ///
    /// # Returns
    /// Fog factor in [0, 1] where 0 = no fog, 1 = fully fogged.
    pub fn exponential_fog(&self, distance: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let effective_dist = (distance - self.start_distance).max(0.0);
        let fog = 1.0 - (-self.global_density * effective_dist).exp();
        fog.clamp(0.0, self.max_density)
    }

    /// Computes exponential-squared fog (denser fog).
    pub fn exponential_squared_fog(&self, distance: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let effective_dist = (distance - self.start_distance).max(0.0);
        let d = self.global_density * effective_dist;
        let fog = 1.0 - (-d * d).exp();
        fog.clamp(0.0, self.max_density)
    }

    /// Applies fog to a fragment color.
    ///
    /// # Arguments
    /// * `frag_color` - The original fragment color.
    /// * `distance` - Distance from camera to fragment.
    ///
    /// # Returns
    /// The fogged color.
    pub fn apply_fog(&self, frag_color: Vec3, distance: f32) -> Vec3 {
        let fog_factor = self.exponential_fog(distance);
        frag_color * (1.0 - fog_factor) + self.color * fog_factor
    }
}

impl Default for FogSettings {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FogNoiseSettings
// ---------------------------------------------------------------------------

/// Configuration for noise-based fog density variation.
#[derive(Debug, Clone)]
pub struct FogNoiseSettings {
    /// Base noise frequency.
    pub frequency: f32,
    /// Number of octaves for fractal noise.
    pub octaves: u32,
    /// Lacunarity (frequency multiplier per octave).
    pub lacunarity: f32,
    /// Persistence (amplitude multiplier per octave).
    pub persistence: f32,
    /// How much the noise affects density (0 = none, 1 = full).
    pub intensity: f32,
    /// Contrast: higher values make noise more binary (thick/thin).
    pub contrast: f32,
    /// Wind direction that scrolls the noise field.
    pub wind_direction: Vec3,
    /// Wind speed.
    pub wind_speed: f32,
    /// Vertical offset for noise (to break symmetry).
    pub vertical_offset: f32,
}

impl FogNoiseSettings {
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.5,
            intensity: 0.8,
            contrast: 1.0,
            wind_direction: Vec3::new(1.0, 0.0, 0.3).normalize(),
            wind_speed: 0.5,
            vertical_offset: 0.0,
        }
    }

    /// Samples the noise at a given position and time.
    pub fn sample(&self, position: Vec3, time: f32) -> f32 {
        let wind_offset = self.wind_direction * self.wind_speed * time;
        let p = (position + wind_offset) * self.frequency;
        let p = Vec3::new(p.x, p.y + self.vertical_offset, p.z);

        // Use the noise functions from the particles::forces module.
        // We inline a simplified version here for independence.
        noise_fbm_3d(p.x, p.y, p.z, self.octaves, self.lacunarity, self.persistence)
    }
}

impl Default for FogNoiseSettings {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ---------------------------------------------------------------------------
// VolumetricFog (froxel-based)
// ---------------------------------------------------------------------------

/// Froxel-based volumetric fog system.
///
/// Subdivides the view frustum into a 3D grid of froxels (frustum voxels).
/// Each froxel accumulates density and lighting independently, enabling
/// efficient per-voxel light scattering computation.
///
/// The froxel grid uses exponential depth distribution (more resolution
/// near the camera, less at distance).
#[derive(Debug)]
pub struct VolumetricFog {
    /// Grid resolution (X, Y, Z).
    pub resolution: [u32; 3],
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance for the volumetric grid.
    pub far: f32,
    /// Fog settings.
    pub settings: FogSettings,
    /// The 3D density grid (linearised: [z][y][x]).
    density_grid: Vec<f32>,
    /// The 3D inscattering grid (RGBA: scattered light + transmittance).
    inscattering_grid: Vec<[f32; 4]>,
    /// Accumulated results per froxel.
    integrated_grid: Vec<[f32; 4]>,
    /// Temporal reprojection blend factor (reduces flickering).
    pub temporal_blend: f32,
    /// Previous frame's integrated grid (for temporal reprojection).
    prev_integrated_grid: Vec<[f32; 4]>,
}

impl VolumetricFog {
    /// Creates a new froxel-based volumetric fog system.
    ///
    /// # Arguments
    /// * `resolution` - Grid resolution [X, Y, Z]. Common values: [160, 90, 64].
    /// * `near` - Near plane.
    /// * `far` - Far plane for volumetric effects.
    pub fn new(resolution: [u32; 3], near: f32, far: f32) -> Self {
        let total = (resolution[0] * resolution[1] * resolution[2]) as usize;
        Self {
            resolution,
            near,
            far,
            settings: FogSettings::default(),
            density_grid: vec![0.0; total],
            inscattering_grid: vec![[0.0; 4]; total],
            integrated_grid: vec![[0.0; 4]; total],
            temporal_blend: 0.05,
            prev_integrated_grid: vec![[0.0; 4]; total],
        }
    }

    /// Returns the total number of froxels.
    pub fn froxel_count(&self) -> usize {
        (self.resolution[0] * self.resolution[1] * self.resolution[2]) as usize
    }

    /// Converts a 3D grid index to a linear index.
    #[inline]
    fn linear_index(&self, x: u32, y: u32, z: u32) -> usize {
        ((z * self.resolution[1] + y) * self.resolution[0] + x) as usize
    }

    /// Converts a linear depth slice index to the world-space depth.
    ///
    /// Uses exponential distribution: `depth = near * (far/near)^(z/z_count)`.
    pub fn slice_depth(&self, z: u32) -> f32 {
        let t = z as f32 / self.resolution[2] as f32;
        self.near * (self.far / self.near).powf(t)
    }

    /// Converts a depth value to the nearest slice index.
    pub fn depth_to_slice(&self, depth: f32) -> u32 {
        if depth <= self.near {
            return 0;
        }
        if depth >= self.far {
            return self.resolution[2] - 1;
        }
        let t = (depth / self.near).ln() / (self.far / self.near).ln();
        (t * self.resolution[2] as f32)
            .clamp(0.0, (self.resolution[2] - 1) as f32) as u32
    }

    /// Computes the world-space position of a froxel center given screen UV,
    /// depth slice, camera inverse view-projection matrix, and camera position.
    pub fn froxel_world_pos(
        &self,
        x: u32,
        y: u32,
        z: u32,
        inv_view_proj: &Mat4,
    ) -> Vec3 {
        let u = (x as f32 + 0.5) / self.resolution[0] as f32;
        let v = (y as f32 + 0.5) / self.resolution[1] as f32;
        let depth = self.slice_depth(z);

        // Convert UV + depth to NDC.
        let ndc_x = u * 2.0 - 1.0;
        let ndc_y = 1.0 - v * 2.0; // Flip Y.

        // Linearize depth to NDC z (assuming reverse-Z or standard projection).
        // For a simple linear mapping:
        let ndc_z = (depth - self.near) / (self.far - self.near);

        let clip = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
        let world = *inv_view_proj * clip;
        Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
    }

    /// Fills the density grid using the fog settings.
    ///
    /// This would typically be done on the GPU, but the CPU version is
    /// provided for testing and fallback.
    pub fn compute_density(
        &mut self,
        inv_view_proj: &Mat4,
        time: f32,
    ) {
        for z in 0..self.resolution[2] {
            for y in 0..self.resolution[1] {
                for x in 0..self.resolution[0] {
                    let world_pos = self.froxel_world_pos(x, y, z, inv_view_proj);
                    let density = self.settings.density_at(world_pos, time);
                    let idx = self.linear_index(x, y, z);
                    self.density_grid[idx] = density;
                }
            }
        }
    }

    /// Computes inscattering from a single directional light.
    ///
    /// # Arguments
    /// * `light_dir` - Direction TO the light (normalized).
    /// * `light_color` - Light color and intensity.
    /// * `camera_dir` - Camera forward direction (for phase function).
    pub fn compute_inscattering_directional(
        &mut self,
        light_dir: Vec3,
        light_color: Vec3,
        camera_dir: Vec3,
    ) {
        // Henyey-Greenstein phase function.
        let g = 0.7; // Asymmetry parameter (forward scattering).
        let cos_theta = camera_dir.dot(light_dir).clamp(-1.0, 1.0);
        let phase = henyey_greenstein(cos_theta, g);

        let fog_color = self.settings.color;
        let inscatter_color = self.settings.inscattering_color;
        let inscatter_intensity = self.settings.inscattering_intensity;

        let total = self.froxel_count();
        for i in 0..total {
            let density = self.density_grid[i];
            if density < 1e-6 {
                self.inscattering_grid[i] = [0.0, 0.0, 0.0, 1.0];
                continue;
            }

            let scattered = (fog_color + inscatter_color * light_color * phase)
                * density
                * inscatter_intensity;

            self.inscattering_grid[i] = [scattered.x, scattered.y, scattered.z, density];
        }
    }

    /// Integrates the inscattering grid along the depth axis (front-to-back).
    ///
    /// This produces the final accumulated fog for each screen pixel by
    /// marching through depth slices.
    pub fn integrate(&mut self) {
        // Save previous frame for temporal reprojection.
        std::mem::swap(&mut self.prev_integrated_grid, &mut self.integrated_grid);

        for y in 0..self.resolution[1] {
            for x in 0..self.resolution[0] {
                let mut accumulated_color = Vec3::ZERO;
                let mut accumulated_transmittance = 1.0f32;

                for z in 0..self.resolution[2] {
                    let idx = self.linear_index(x, y, z);
                    let inscattered = &self.inscattering_grid[idx];
                    let density = inscattered[3];

                    if density < 1e-6 {
                        self.integrated_grid[idx] = [
                            accumulated_color.x,
                            accumulated_color.y,
                            accumulated_color.z,
                            accumulated_transmittance,
                        ];
                        continue;
                    }

                    // Slice thickness (approximate).
                    let depth_front = self.slice_depth(z);
                    let depth_back = if z + 1 < self.resolution[2] {
                        self.slice_depth(z + 1)
                    } else {
                        self.far
                    };
                    let thickness = depth_back - depth_front;

                    let slice_transmittance = (-density * thickness).exp();
                    let slice_color = Vec3::new(inscattered[0], inscattered[1], inscattered[2]);

                    // Accumulate.
                    accumulated_color +=
                        slice_color * accumulated_transmittance * (1.0 - slice_transmittance);
                    accumulated_transmittance *= slice_transmittance;

                    // Temporal reprojection blend.
                    let prev = &self.prev_integrated_grid[idx];
                    let blended = [
                        lerp(accumulated_color.x, prev[0], self.temporal_blend),
                        lerp(accumulated_color.y, prev[1], self.temporal_blend),
                        lerp(accumulated_color.z, prev[2], self.temporal_blend),
                        lerp(accumulated_transmittance, prev[3], self.temporal_blend),
                    ];

                    self.integrated_grid[idx] = blended;
                }
            }
        }
    }

    /// Samples the integrated fog result at a screen UV coordinate and depth.
    ///
    /// Returns `(inscattered_color, transmittance)`.
    pub fn sample(&self, screen_uv: Vec2, depth: f32) -> (Vec3, f32) {
        let x = ((screen_uv.x * self.resolution[0] as f32) as u32)
            .min(self.resolution[0] - 1);
        let y = ((screen_uv.y * self.resolution[1] as f32) as u32)
            .min(self.resolution[1] - 1);
        let z = self.depth_to_slice(depth);

        let idx = self.linear_index(x, y, z);
        let data = &self.integrated_grid[idx];
        (Vec3::new(data[0], data[1], data[2]), data[3])
    }

    /// Returns the density grid (for GPU upload).
    pub fn density_grid(&self) -> &[f32] {
        &self.density_grid
    }

    /// Returns the integrated grid (for GPU upload).
    pub fn integrated_grid(&self) -> &[[f32; 4]] {
        &self.integrated_grid
    }

    /// Returns the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let count = self.froxel_count();
        count * std::mem::size_of::<f32>()           // density
            + count * std::mem::size_of::<[f32; 4]>() // inscattering
            + count * std::mem::size_of::<[f32; 4]>() // integrated
            + count * std::mem::size_of::<[f32; 4]>() // prev integrated
    }
}

// ---------------------------------------------------------------------------
// GodRays
// ---------------------------------------------------------------------------

/// God rays / light shaft effect using screen-space radial blur.
///
/// The effect works by:
/// 1. Rendering an occlusion mask (bright where light is visible).
/// 2. Applying a radial blur from the light's screen-space position.
/// 3. Compositing the blurred result over the scene.
#[derive(Debug, Clone)]
pub struct GodRays {
    /// If `true`, god rays are enabled.
    pub enabled: bool,
    /// World-space light position (for directional lights, use a very far point).
    pub light_position: Vec3,
    /// Color of the god rays.
    pub color: Vec3,
    /// Overall intensity.
    pub intensity: f32,
    /// Number of radial blur samples.
    pub num_samples: u32,
    /// Density: how quickly rays fade with distance from light.
    pub density: f32,
    /// Weight: intensity per sample.
    pub weight: f32,
    /// Decay: exponential falloff per sample.
    pub decay: f32,
    /// Exposure: post-process brightness multiplier.
    pub exposure: f32,
    /// Maximum ray length (in screen space, 0..1).
    pub max_ray_length: f32,
}

impl GodRays {
    pub fn new(light_position: Vec3) -> Self {
        Self {
            enabled: true,
            light_position,
            color: Vec3::ONE,
            intensity: 1.0,
            num_samples: 64,
            density: 1.0,
            weight: 0.01,
            decay: 0.96,
            exposure: 1.0,
            max_ray_length: 1.0,
        }
    }

    /// Sets the intensity.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Sets the number of blur samples.
    pub fn with_samples(mut self, samples: u32) -> Self {
        self.num_samples = samples;
        self
    }

    /// Sets the decay factor.
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay;
        self
    }

    /// Computes the screen-space position of the light source.
    ///
    /// # Arguments
    /// * `view_proj` - The view-projection matrix.
    ///
    /// # Returns
    /// Screen-space UV coordinates [0, 1], or `None` if the light is behind
    /// the camera.
    pub fn light_screen_pos(&self, view_proj: &Mat4) -> Option<Vec2> {
        let clip = *view_proj * Vec4::new(
            self.light_position.x,
            self.light_position.y,
            self.light_position.z,
            1.0,
        );

        if clip.w <= 0.0 {
            return None; // Behind camera.
        }

        let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
        // Convert from NDC [-1, 1] to UV [0, 1].
        let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

        Some(uv)
    }

    /// Performs the radial blur in screen space (CPU reference implementation).
    ///
    /// In production, this runs as a post-process shader. This CPU version
    /// is for testing and preview.
    ///
    /// # Arguments
    /// * `occlusion_buffer` - Brightness values from the occlusion pass.
    /// * `width` - Buffer width in pixels.
    /// * `height` - Buffer height in pixels.
    /// * `light_uv` - Screen-space light position [0, 1].
    /// * `output` - Output buffer (same dimensions as input).
    pub fn radial_blur(
        &self,
        occlusion_buffer: &[f32],
        width: u32,
        height: u32,
        light_uv: Vec2,
        output: &mut [f32],
    ) {
        let w = width as usize;
        let h = height as usize;

        for py in 0..h {
            for px in 0..w {
                let pixel_uv = Vec2::new(
                    (px as f32 + 0.5) / width as f32,
                    (py as f32 + 0.5) / height as f32,
                );

                let delta = pixel_uv - light_uv;
                let delta = delta * (1.0 / self.num_samples as f32) * self.density;

                let mut sample_uv = pixel_uv;
                let mut illumination = 0.0f32;
                let mut decay_factor = 1.0f32;

                for _ in 0..self.num_samples {
                    // Clamp sample UV to valid range.
                    let sx = (sample_uv.x * width as f32) as i32;
                    let sy = (sample_uv.y * height as f32) as i32;

                    if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
                        let idx = sy as usize * w + sx as usize;
                        illumination +=
                            occlusion_buffer[idx] * decay_factor * self.weight;
                    }

                    decay_factor *= self.decay;
                    sample_uv -= delta;
                }

                let idx = py * w + px;
                output[idx] = (illumination * self.exposure * self.intensity)
                    .clamp(0.0, 1.0);
            }
        }
    }
}

impl Default for GodRays {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 100.0, 0.0))
    }
}

// ---------------------------------------------------------------------------
// VolumetricLight
// ---------------------------------------------------------------------------

/// Volumetric light volume for spotlights and point lights.
///
/// This represents a single light that scatters through participating media
/// (fog/dust). The effect is computed by raymarching through the light's
/// volume of influence.
#[derive(Debug, Clone)]
pub struct VolumetricLight {
    /// Light position.
    pub position: Vec3,
    /// Light direction (for spot lights).
    pub direction: Vec3,
    /// Light color.
    pub color: Vec3,
    /// Light intensity.
    pub intensity: f32,
    /// Light range (maximum influence distance).
    pub range: f32,
    /// Spotlight inner cone angle (radians). Set to PI for point lights.
    pub inner_cone_angle: f32,
    /// Spotlight outer cone angle (radians).
    pub outer_cone_angle: f32,
    /// Number of raymarching steps.
    pub num_steps: u32,
    /// Scattering coefficient.
    pub scattering: f32,
    /// Phase function asymmetry (Henyey-Greenstein g parameter).
    pub phase_g: f32,
    /// Fog density within the light volume.
    pub fog_density: f32,
}

impl VolumetricLight {
    /// Creates a volumetric spotlight.
    pub fn spot(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        cone_angle: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range,
            inner_cone_angle: cone_angle * 0.8,
            outer_cone_angle: cone_angle,
            num_steps: 32,
            scattering: 0.1,
            phase_g: 0.5,
            fog_density: 0.05,
        }
    }

    /// Creates a volumetric point light.
    pub fn point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            position,
            direction: Vec3::NEG_Z,
            color,
            intensity,
            range,
            inner_cone_angle: std::f32::consts::PI,
            outer_cone_angle: std::f32::consts::PI,
            num_steps: 32,
            scattering: 0.1,
            phase_g: 0.0,
            fog_density: 0.05,
        }
    }

    /// Computes the light attenuation at a given world-space position.
    pub fn attenuation(&self, world_pos: Vec3) -> f32 {
        let to_light = self.position - world_pos;
        let dist = to_light.length();

        if dist > self.range || dist < 1e-6 {
            return 0.0;
        }

        // Distance attenuation (inverse-square with smooth cutoff).
        let dist_atten = ((1.0 - (dist / self.range).powi(4)).max(0.0)).powi(2)
            / (dist * dist + 1.0);

        // Spotlight cone attenuation.
        if self.outer_cone_angle < std::f32::consts::PI - 0.01 {
            let light_dir = to_light / dist;
            let cos_angle = (-self.direction).dot(light_dir);
            let cos_inner = self.inner_cone_angle.cos();
            let cos_outer = self.outer_cone_angle.cos();

            if cos_angle < cos_outer {
                return 0.0;
            }

            let spot_atten = if cos_angle > cos_inner {
                1.0
            } else {
                let t = (cos_angle - cos_outer) / (cos_inner - cos_outer);
                t * t
            };

            dist_atten * spot_atten
        } else {
            dist_atten
        }
    }

    /// Raymarches through the light volume from a camera ray.
    ///
    /// # Arguments
    /// * `ray_origin` - Camera position.
    /// * `ray_dir` - Normalized ray direction.
    /// * `max_dist` - Maximum ray distance (scene depth).
    ///
    /// # Returns
    /// Accumulated scattered light color.
    pub fn raymarch(
        &self,
        ray_origin: Vec3,
        ray_dir: Vec3,
        max_dist: f32,
    ) -> Vec3 {
        // Clip ray to the light's bounding sphere.
        let to_center = self.position - ray_origin;
        let t_closest = to_center.dot(ray_dir);
        let dist_sq = to_center.length_squared() - t_closest * t_closest;

        if dist_sq > self.range * self.range {
            return Vec3::ZERO; // Ray misses the light volume.
        }

        let half_chord = (self.range * self.range - dist_sq).sqrt();
        let t_enter = (t_closest - half_chord).max(0.0);
        let t_exit = (t_closest + half_chord).min(max_dist);

        if t_enter >= t_exit {
            return Vec3::ZERO;
        }

        let step_size = (t_exit - t_enter) / self.num_steps as f32;
        let mut accumulated = Vec3::ZERO;
        let mut transmittance = 1.0f32;

        for i in 0..self.num_steps {
            let t = t_enter + (i as f32 + 0.5) * step_size;
            let sample_pos = ray_origin + ray_dir * t;

            let atten = self.attenuation(sample_pos);
            if atten < 1e-6 {
                continue;
            }

            // Phase function.
            let to_light_dir = (self.position - sample_pos).normalize();
            let cos_theta = ray_dir.dot(to_light_dir);
            let phase = henyey_greenstein(cos_theta, self.phase_g);

            // Inscattering contribution.
            let inscatter = self.color
                * self.intensity
                * atten
                * phase
                * self.scattering
                * step_size;

            accumulated += inscatter * transmittance;

            // Extinction.
            transmittance *= (-self.fog_density * step_size).exp();

            if transmittance < 0.001 {
                break;
            }
        }

        accumulated
    }
}

// ---------------------------------------------------------------------------
// Henyey-Greenstein phase function
// ---------------------------------------------------------------------------

/// Henyey-Greenstein phase function for volumetric scattering.
///
/// Models the angular distribution of scattered light in participating media.
///
/// # Arguments
/// * `cos_theta` - Cosine of the angle between light and view directions.
/// * `g` - Asymmetry parameter: 0 = isotropic, >0 = forward scattering,
///   <0 = backward scattering. Typical atmospheric values: 0.5..0.9.
///
/// # Returns
/// Phase function value (not normalized to 4*PI).
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    if denom < 1e-6 {
        return 1.0;
    }
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom * denom.sqrt())
}

/// Schlick's approximation of the Henyey-Greenstein phase function.
/// Faster to evaluate, nearly identical results.
pub fn schlick_phase(cos_theta: f32, g: f32) -> f32 {
    let k = 1.55 * g - 0.55 * g * g * g;
    let denom = 1.0 - k * cos_theta;
    (1.0 - k * k) / (4.0 * std::f32::consts::PI * denom * denom)
}

// ---------------------------------------------------------------------------
// Noise function (self-contained for this module)
// ---------------------------------------------------------------------------

/// Simplified 3D fBm noise for fog (avoids circular dependency with particles).
fn noise_fbm_3d(
    x: f32, y: f32, z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        value += noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if max_amp > 0.0 {
        value / max_amp
    } else {
        0.0
    }
}

/// Simple 3D value noise using a hash function.
fn noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    // Quintic smoothstep.
    let u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0);
    let v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0);
    let w = zf * zf * zf * (zf * (zf * 6.0 - 15.0) + 10.0);

    let hash = |x: i32, y: i32, z: i32| -> f32 {
        // Simple hash function.
        let n = (x.wrapping_mul(374761393))
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(z.wrapping_mul(1274126177));
        let n = n ^ (n >> 13);
        let n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303)).wrapping_add(1376312589));
        (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32 * 2.0 - 1.0
    };

    let c000 = hash(xi, yi, zi);
    let c100 = hash(xi + 1, yi, zi);
    let c010 = hash(xi, yi + 1, zi);
    let c110 = hash(xi + 1, yi + 1, zi);
    let c001 = hash(xi, yi, zi + 1);
    let c101 = hash(xi + 1, yi, zi + 1);
    let c011 = hash(xi, yi + 1, zi + 1);
    let c111 = hash(xi + 1, yi + 1, zi + 1);

    let x00 = c000 + (c100 - c000) * u;
    let x10 = c010 + (c110 - c010) * u;
    let x01 = c001 + (c101 - c001) * u;
    let x11 = c011 + (c111 - c011) * u;

    let y0 = x00 + (x10 - x00) * v;
    let y1 = x01 + (x11 - x01) * v;

    y0 + (y1 - y0) * w
}

/// Linear interpolation.
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
    fn exponential_fog_basics() {
        let fog = FogSettings::new().with_density(0.1).with_start_distance(0.0);

        let f0 = fog.exponential_fog(0.0);
        assert!(f0 < 0.01, "No fog at zero distance");

        let f100 = fog.exponential_fog(100.0);
        assert!(f100 > 0.9, "Nearly full fog at 100 units");
    }

    #[test]
    fn height_fog_increases_below_base() {
        let fog = FogSettings::new()
            .with_density(0.01)
            .with_height_fog(0.5, 1.0, 10.0);

        let above = fog.density_at(Vec3::new(0.0, 20.0, 0.0), 0.0);
        let below = fog.density_at(Vec3::new(0.0, 0.0, 0.0), 0.0);

        assert!(
            below > above,
            "Fog should be denser below base: above={above}, below={below}"
        );
    }

    #[test]
    fn raymarch_produces_fog() {
        let fog = FogSettings::new().with_density(0.1).with_start_distance(0.0);

        let (transmittance, inscattered) =
            fog.raymarch(Vec3::ZERO, Vec3::Z, 50.0, 0.0, 32);

        assert!(transmittance < 1.0, "Should have some extinction");
        assert!(
            inscattered.length() > 0.0,
            "Should have some inscattered light"
        );
    }

    #[test]
    fn froxel_depth_distribution() {
        let vol = VolumetricFog::new([16, 9, 64], 0.1, 100.0);

        let d0 = vol.slice_depth(0);
        let d_mid = vol.slice_depth(32);
        let d_end = vol.slice_depth(64);

        assert!((d0 - 0.1).abs() < 0.01, "First slice at near plane");
        assert!(d_mid > d0 && d_mid < d_end, "Mid is between near and far");
        assert!((d_end - 100.0).abs() < 0.1, "Last slice at far plane");

        // Exponential: mid should be closer to near than to far.
        assert!(
            d_mid < 50.0,
            "Exponential distribution: mid should be < 50, got {d_mid}"
        );
    }

    #[test]
    fn depth_to_slice_roundtrip() {
        let vol = VolumetricFog::new([16, 9, 64], 0.1, 100.0);

        for z in 0..64 {
            let depth = vol.slice_depth(z);
            let recovered = vol.depth_to_slice(depth);
            assert!(
                (recovered as i32 - z as i32).unsigned_abs() <= 1,
                "Roundtrip failed: z={z}, depth={depth}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn henyey_greenstein_isotropic() {
        // When g=0, the phase function should be isotropic (constant).
        let p1 = henyey_greenstein(1.0, 0.0);
        let p2 = henyey_greenstein(0.0, 0.0);
        let p3 = henyey_greenstein(-1.0, 0.0);

        assert!(
            (p1 - p2).abs() < 0.01 && (p2 - p3).abs() < 0.01,
            "Isotropic phase should be constant: {p1}, {p2}, {p3}"
        );
    }

    #[test]
    fn god_rays_screen_pos() {
        let rays = GodRays::new(Vec3::new(0.0, 100.0, -50.0));
        // Simple perspective view-proj looking down -Z.
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2,
            16.0 / 9.0,
            0.1,
            1000.0,
        );
        let vp = proj * view;

        let uv = rays.light_screen_pos(&vp);
        assert!(uv.is_some(), "Light should be in front of camera");
    }

    #[test]
    fn volumetric_light_attenuation() {
        let light = VolumetricLight::point(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::ONE,
            10.0,
            20.0,
        );

        let near = light.attenuation(Vec3::new(0.0, 4.0, 0.0));
        let far = light.attenuation(Vec3::new(0.0, -10.0, 0.0));
        let outside = light.attenuation(Vec3::new(0.0, -30.0, 0.0));

        assert!(near > far, "Near should be brighter than far");
        assert!(
            outside < 0.001,
            "Outside range should be zero, got {outside}"
        );
    }

    #[test]
    fn spotlight_cone() {
        let light = VolumetricLight::spot(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Vec3::ONE,
            10.0,
            50.0,
            std::f32::consts::FRAC_PI_4,
        );

        let in_cone = light.attenuation(Vec3::new(0.0, 0.0, -5.0));
        let out_cone = light.attenuation(Vec3::new(10.0, 0.0, -1.0));

        assert!(in_cone > out_cone, "In-cone should be brighter");
    }
}
