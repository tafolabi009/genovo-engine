// engine/render/src/lumen_gi/mod.rs
//
// Lumen-style Global Illumination integration for the Genovo engine.
// Ties together screen probes, surface cache, BVH ray tracing, reflections,
// and virtual shadow maps into a unified per-frame GI pipeline.
//
// The LumenGI system follows the Unreal Engine 5 Lumen approach:
//   1. Update surface cache (direct lighting on surfels)
//   2. Place screen probes from depth/normal buffers
//   3. Trace rays from probes through the BVH
//   4. Accumulate SH from ray hits (read surface cache)
//   5. Filter and interpolate probe radiance
//   6. Apply probe lighting to screen pixels
//   7. Trace reflection rays for specular
//   8. Denoise reflections
//   9. Composite final image

use crate::gi::SphericalHarmonics;
use crate::raytracing::bvh_tracer::{
    BVHAccelerationStructure, HitInfo, MeshInstance, Ray, Triangle, AABB,
};
use crate::raytracing::gi_tracer::{
    GIMode, GIResultBuffer, GISettings, GITracer, IrradianceField, MaterialResponse,
};
use crate::raytracing::reflections::{
    ReflectionBuffer, ReflectionSettings, ReflectionTracer,
};
use crate::raytracing::screen_probes::{
    ScreenProbeGrid, ScreenProbeSettings, SurfaceCache,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// LumenQuality
// ---------------------------------------------------------------------------

/// Quality presets for the Lumen GI system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LumenQuality {
    /// Probes only, single bounce. Fastest.
    Low,
    /// Probes + reflections. Good balance.
    Medium,
    /// Probes + reflections + multi-bounce. High quality.
    High,
    /// Full path tracing mode. Reference quality.
    Epic,
}

impl LumenQuality {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Epic => "Epic",
        }
    }
}

// ---------------------------------------------------------------------------
// LumenSettings
// ---------------------------------------------------------------------------

/// Top-level settings for the Lumen GI system.
#[derive(Debug, Clone)]
pub struct LumenSettings {
    /// Overall quality level.
    pub quality: LumenQuality,
    /// Maximum ray trace distance.
    pub max_trace_distance: f32,
    /// Screen probe settings.
    pub probe_settings: ScreenProbeSettings,
    /// Reflection settings.
    pub reflection_settings: ReflectionSettings,
    /// GI tracer settings (multi-bounce / path trace).
    pub gi_settings: GISettings,
    /// Surface cache: maximum surfel count.
    pub max_surfels: usize,
    /// GI intensity multiplier.
    pub gi_intensity: f32,
    /// Reflection intensity multiplier.
    pub reflection_intensity: f32,
    /// Temporal blend factor for screen probes.
    pub temporal_blend: f32,
    /// Whether to enable the irradiance field (off-screen GI).
    pub enable_irradiance_field: bool,
    /// Irradiance field resolution.
    pub irradiance_field_resolution: [u32; 3],
    /// Irradiance field extent (world units).
    pub irradiance_field_extent: Vec3,
    /// Whether reflections are enabled.
    pub enable_reflections: bool,
    /// Whether multi-bounce GI is enabled.
    pub enable_multi_bounce: bool,
    /// Debug visualization mode.
    pub debug_view: LumenDebugView,
}

impl Default for LumenSettings {
    fn default() -> Self {
        Self::from_quality(LumenQuality::Medium)
    }
}

impl LumenSettings {
    /// Create settings from a quality preset.
    pub fn from_quality(quality: LumenQuality) -> Self {
        match quality {
            LumenQuality::Low => Self {
                quality,
                max_trace_distance: 100.0,
                probe_settings: ScreenProbeSettings::low(),
                reflection_settings: ReflectionSettings::low(),
                gi_settings: GISettings::low(),
                max_surfels: 100_000,
                gi_intensity: 1.0,
                reflection_intensity: 1.0,
                temporal_blend: 0.95,
                enable_irradiance_field: false,
                irradiance_field_resolution: [8, 4, 8],
                irradiance_field_extent: Vec3::splat(100.0),
                enable_reflections: false,
                enable_multi_bounce: false,
                debug_view: LumenDebugView::None,
            },
            LumenQuality::Medium => Self {
                quality,
                max_trace_distance: 200.0,
                probe_settings: ScreenProbeSettings::medium(),
                reflection_settings: ReflectionSettings::medium(),
                gi_settings: GISettings::medium(),
                max_surfels: 500_000,
                gi_intensity: 1.0,
                reflection_intensity: 1.0,
                temporal_blend: 0.9,
                enable_irradiance_field: true,
                irradiance_field_resolution: [16, 8, 16],
                irradiance_field_extent: Vec3::splat(200.0),
                enable_reflections: true,
                enable_multi_bounce: false,
                debug_view: LumenDebugView::None,
            },
            LumenQuality::High => Self {
                quality,
                max_trace_distance: 400.0,
                probe_settings: ScreenProbeSettings::high(),
                reflection_settings: ReflectionSettings::high(),
                gi_settings: GISettings::high(),
                max_surfels: 1_000_000,
                gi_intensity: 1.0,
                reflection_intensity: 1.0,
                temporal_blend: 0.85,
                enable_irradiance_field: true,
                irradiance_field_resolution: [32, 16, 32],
                irradiance_field_extent: Vec3::splat(400.0),
                enable_reflections: true,
                enable_multi_bounce: true,
                debug_view: LumenDebugView::None,
            },
            LumenQuality::Epic => Self {
                quality,
                max_trace_distance: 1000.0,
                probe_settings: ScreenProbeSettings::ultra(),
                reflection_settings: ReflectionSettings::ultra(),
                gi_settings: GISettings::epic(),
                max_surfels: 2_000_000,
                gi_intensity: 1.0,
                reflection_intensity: 1.0,
                temporal_blend: 0.8,
                enable_irradiance_field: true,
                irradiance_field_resolution: [64, 32, 64],
                irradiance_field_extent: Vec3::splat(1000.0),
                enable_reflections: true,
                enable_multi_bounce: true,
                debug_view: LumenDebugView::None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LumenDebugView
// ---------------------------------------------------------------------------

/// Debug visualization modes for the Lumen system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LumenDebugView {
    /// Normal rendering (no debug overlay).
    None,
    /// Show screen probe positions.
    Probes,
    /// Show traced rays from probes.
    Rays,
    /// Show surface cache coverage.
    SurfaceCache,
    /// Show SH radiance per probe.
    Radiance,
    /// Show reflection results only.
    Reflections,
    /// Show GI contribution only (no direct lighting).
    IndirectOnly,
    /// Show irradiance field probes.
    IrradianceField,
    /// Show ray trace heatmap (rays per pixel).
    RayHeatmap,
}

// ---------------------------------------------------------------------------
// LumenGI -- the main orchestrator
// ---------------------------------------------------------------------------

/// The Lumen-style Global Illumination system.
///
/// Orchestrates all subsystems (screen probes, surface cache, BVH tracing,
/// reflections, irradiance field) into a coherent per-frame pipeline.
pub struct LumenGI {
    /// Settings.
    pub settings: LumenSettings,

    /// Screen probe grid for indirect diffuse.
    pub probe_grid: ScreenProbeGrid,

    /// Surface cache storing pre-computed direct lighting on surfels.
    pub surface_cache: SurfaceCache,

    /// Reflection tracer.
    pub reflection_tracer: ReflectionTracer,

    /// GI tracer (multi-bounce / path trace).
    pub gi_tracer: GITracer,

    /// Per-pixel GI result buffer.
    pub gi_buffer: GIResultBuffer,

    /// World-space irradiance field for off-screen GI.
    pub irradiance_field: Option<IrradianceField>,

    /// Current frame index.
    pub frame_index: u64,

    /// Screen dimensions.
    pub width: u32,
    pub height: u32,

    /// Whether the system has been initialized.
    pub initialized: bool,

    /// Per-frame statistics.
    pub stats: LumenFrameStats,
}

/// Per-frame statistics for the Lumen system.
#[derive(Debug, Default, Clone)]
pub struct LumenFrameStats {
    /// Number of valid screen probes.
    pub valid_probe_count: u32,
    /// Total rays traced from screen probes.
    pub probe_rays_traced: u32,
    /// Total reflection rays traced.
    pub reflection_rays_traced: u32,
    /// Surfel count in the surface cache.
    pub surfel_count: u32,
    /// Irradiance field dirty probe count.
    pub irradiance_dirty_probes: u32,
    /// Total frame time for GI (milliseconds, estimated).
    pub gi_time_ms: f32,
    /// Total frame time for reflections (milliseconds, estimated).
    pub reflection_time_ms: f32,
}

impl LumenGI {
    /// Create a new Lumen GI system.
    pub fn new(width: u32, height: u32, settings: LumenSettings) -> Self {
        let probe_grid = ScreenProbeGrid::with_tile_size(
            width,
            height,
            settings.probe_settings.tile_size,
        );
        let surface_cache = SurfaceCache::new(settings.max_surfels);
        let reflection_tracer = ReflectionTracer::new(
            width,
            height,
            settings.reflection_settings.clone(),
        );
        let gi_tracer = GITracer::new(width, height, settings.gi_settings.clone());
        let gi_buffer = GIResultBuffer::new(width, height);

        let irradiance_field = if settings.enable_irradiance_field {
            Some(IrradianceField::new(
                -settings.irradiance_field_extent * 0.5,
                settings.irradiance_field_extent,
                settings.irradiance_field_resolution,
            ))
        } else {
            None
        };

        Self {
            settings,
            probe_grid,
            surface_cache,
            reflection_tracer,
            gi_tracer,
            gi_buffer,
            irradiance_field,
            frame_index: 0,
            width,
            height,
            initialized: false,
            stats: LumenFrameStats::default(),
        }
    }

    /// Resize the system for a new screen resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.probe_grid.resize(width, height);
        self.reflection_tracer.resize(width, height);
        self.gi_tracer.resize(width, height);
        self.gi_buffer.resize(width, height);
    }

    /// Change quality settings. Rebuilds subsystems as needed.
    pub fn set_quality(&mut self, quality: LumenQuality) {
        let new_settings = LumenSettings::from_quality(quality);
        self.settings = new_settings;

        self.probe_grid = ScreenProbeGrid::with_tile_size(
            self.width,
            self.height,
            self.settings.probe_settings.tile_size,
        );
        self.probe_grid.rays_per_probe = self.settings.probe_settings.rays_per_probe;
        self.probe_grid.temporal_blend = self.settings.temporal_blend;
        self.probe_grid.max_trace_distance = self.settings.max_trace_distance;

        self.reflection_tracer.settings = self.settings.reflection_settings.clone();
        self.gi_tracer.settings = self.settings.gi_settings.clone();

        if self.settings.enable_irradiance_field && self.irradiance_field.is_none() {
            self.irradiance_field = Some(IrradianceField::new(
                -self.settings.irradiance_field_extent * 0.5,
                self.settings.irradiance_field_extent,
                self.settings.irradiance_field_resolution,
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Per-frame pipeline
    // -----------------------------------------------------------------------

    /// Execute the full Lumen GI pipeline for a frame.
    ///
    /// This is the main entry point called once per frame after the
    /// G-buffer has been rendered.
    pub fn render_frame(
        &mut self,
        accel: &BVHAccelerationStructure,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        roughness_buffer: &[f32],
        position_buffer: &[Vec3],
        albedo_buffer: &[Vec3],
        unproject: &dyn Fn(u32, u32, f32) -> Vec3,
        view_dir_fn: &dyn Fn(u32, u32) -> Vec3,
        compute_direct: &dyn Fn(Vec3, Vec3) -> Vec3,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
    ) {
        self.stats = LumenFrameStats::default();

        // Step 1: Update surface cache with direct lighting.
        self.update_surface_cache(accel, compute_direct);

        // Step 2: Place screen probes from the G-buffer.
        self.place_probes(depth_buffer, normal_buffer, unproject);

        // Step 3: Trace rays from probes.
        self.trace_probe_rays(accel, sky_radiance);

        // Step 4 & 5: Finalize and filter probes.
        self.finalize_probes();

        // Step 6: Apply probe lighting to pixels.
        self.apply_probe_lighting(
            depth_buffer,
            normal_buffer,
            position_buffer,
            albedo_buffer,
        );

        // Step 7 & 8: Trace and denoise reflections.
        if self.settings.enable_reflections {
            self.trace_reflections(
                accel,
                depth_buffer,
                normal_buffer,
                roughness_buffer,
                position_buffer,
                view_dir_fn,
                sky_radiance,
            );
        }

        // Update irradiance field (background, partial update per frame).
        if let Some(ref mut field) = self.irradiance_field {
            self.stats.irradiance_dirty_probes = field.dirty_count() as u32;
            // Update a subset of dirty probes each frame.
            let probes_per_frame = 16u32;
            let total = field.probe_count();
            let start = (self.frame_index as usize * probes_per_frame as usize) % total;

            for i in 0..probes_per_frame as usize {
                let probe_idx = (start + i) % total;
                let rx = field.resolution[0];
                let ry = field.resolution[1];
                let x = (probe_idx as u32) % rx;
                let y = ((probe_idx as u32) / rx) % ry;
                let z = (probe_idx as u32) / (rx * ry);

                if field.dirty[probe_idx] {
                    field.update_probe(
                        x,
                        y,
                        z,
                        accel,
                        &self.surface_cache,
                        sky_radiance,
                        16,
                        self.frame_index as u32 + probe_idx as u32,
                    );
                }
            }
        }

        self.frame_index += 1;
        self.initialized = true;
    }

    /// Step 1: Update surface cache with current direct lighting.
    fn update_surface_cache(
        &mut self,
        accel: &BVHAccelerationStructure,
        compute_direct: &dyn Fn(Vec3, Vec3) -> Vec3,
    ) {
        if self.surface_cache.dirty {
            // Full rebuild on first frame or when marked dirty.
            self.surface_cache.build_from_scene(accel, &|_, _| {
                Vec3::splat(0.8) // Default albedo.
            });
        }

        self.surface_cache.update_direct_lighting(compute_direct);

        // Update combined radiance (direct + indirect from previous frame).
        for surfel in &mut self.surface_cache.surfels {
            if surfel.valid {
                surfel.update_radiance(Vec3::ZERO, self.settings.temporal_blend as f32);
            }
        }

        self.stats.surfel_count = self.surface_cache.surfel_count() as u32;
    }

    /// Step 2: Place screen probes.
    fn place_probes(
        &mut self,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        unproject: &dyn Fn(u32, u32, f32) -> Vec3,
    ) {
        self.probe_grid
            .place_probes(depth_buffer, normal_buffer, unproject);
    }

    /// Step 3: Trace rays from screen probes.
    fn trace_probe_rays(
        &mut self,
        accel: &BVHAccelerationStructure,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
    ) {
        let surface_cache = &self.surface_cache;

        self.probe_grid.trace_probes(
            accel,
            &|hit: &HitInfo| surface_cache.lookup_radiance(hit),
            sky_radiance,
        );

        self.stats.probe_rays_traced = self.probe_grid.total_rays_per_frame();
        self.stats.valid_probe_count = self
            .probe_grid
            .probes
            .iter()
            .filter(|p| p.valid)
            .count() as u32;
    }

    /// Steps 4 & 5: Finalize probes (normalize, temporal, spatial filter).
    fn finalize_probes(&mut self) {
        self.probe_grid.finalize_probes();

        if self.settings.probe_settings.spatial_filter_enabled {
            self.probe_grid.spatial_filter();
        }

        self.probe_grid.next_frame();
    }

    /// Step 6: Apply probe-based indirect lighting to pixels.
    fn apply_probe_lighting(
        &mut self,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        position_buffer: &[Vec3],
        albedo_buffer: &[Vec3],
    ) {
        self.gi_buffer.swap_history();

        for py in 0..self.height {
            for px in 0..self.width {
                let idx = (py * self.width + px) as usize;
                let depth = depth_buffer.get(idx).copied().unwrap_or(0.0);

                if depth <= 0.0 || depth >= 1.0 {
                    continue;
                }

                let normal = normal_buffer.get(idx).copied().unwrap_or(Vec3::Y);
                let albedo = albedo_buffer.get(idx).copied().unwrap_or(Vec3::splat(0.8));

                // Sample irradiance from screen probes.
                let mut irradiance = self.probe_grid.sample_irradiance(
                    px,
                    py,
                    depth,
                    normal,
                    normal,
                );

                // Supplement with irradiance field if available.
                if let Some(ref field) = self.irradiance_field {
                    let world_pos =
                        position_buffer.get(idx).copied().unwrap_or(Vec3::ZERO);
                    let field_irradiance = field.sample(world_pos, normal);
                    // Blend: prefer screen probes but fill gaps with field.
                    let probe_confidence = if irradiance.length_squared() > 1e-6 {
                        1.0
                    } else {
                        0.0
                    };
                    irradiance = irradiance * probe_confidence
                        + field_irradiance * (1.0 - probe_confidence);
                }

                // Apply albedo and intensity.
                let gi = albedo * irradiance * self.settings.gi_intensity * (1.0 / PI);

                self.gi_buffer.write(px, py, gi, 1.0);
            }
        }

        // Temporal accumulation.
        self.gi_buffer
            .temporal_accumulate(self.settings.temporal_blend);
    }

    /// Steps 7 & 8: Trace and denoise reflections.
    fn trace_reflections(
        &mut self,
        accel: &BVHAccelerationStructure,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        roughness_buffer: &[f32],
        position_buffer: &[Vec3],
        view_dir_fn: &dyn Fn(u32, u32) -> Vec3,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
    ) {
        self.reflection_tracer.trace_frame(
            accel,
            &self.surface_cache,
            depth_buffer,
            normal_buffer,
            roughness_buffer,
            position_buffer,
            view_dir_fn,
            sky_radiance,
        );

        // Count reflection rays.
        let mut ray_count = 0u32;
        for py in 0..self.height {
            for px in 0..self.width {
                let idx = (py * self.width + px) as usize;
                if roughness_buffer.get(idx).copied().unwrap_or(1.0)
                    < self.reflection_tracer.settings.roughness_threshold
                {
                    ray_count += self.reflection_tracer.settings.rays_for_roughness(
                        roughness_buffer[idx],
                    );
                }
            }
        }
        self.stats.reflection_rays_traced = ray_count;
    }

    // -----------------------------------------------------------------------
    // Output accessors
    // -----------------------------------------------------------------------

    /// Get the final diffuse GI radiance for a pixel.
    pub fn get_gi(&self, px: u32, py: u32) -> Vec3 {
        self.gi_buffer.read(px, py)
    }

    /// Get the reflection radiance for a pixel.
    pub fn get_reflection(&self, px: u32, py: u32) -> Vec3 {
        self.reflection_tracer.buffer.read(px, py)
            * self.settings.reflection_intensity
    }

    /// Composite the final pixel colour: direct + GI + reflections.
    pub fn composite_pixel(
        &self,
        px: u32,
        py: u32,
        direct_lighting: Vec3,
        roughness: f32,
    ) -> Vec3 {
        let gi = self.get_gi(px, py);

        let reflection = if self.settings.enable_reflections {
            // Blend reflection based on roughness.
            let reflection_weight = (1.0 - roughness).clamp(0.0, 1.0);
            self.get_reflection(px, py) * reflection_weight
        } else {
            Vec3::ZERO
        };

        match self.settings.debug_view {
            LumenDebugView::None => direct_lighting + gi + reflection,
            LumenDebugView::IndirectOnly => gi + reflection,
            LumenDebugView::Reflections => reflection,
            LumenDebugView::Radiance => gi,
            _ => direct_lighting + gi + reflection,
        }
    }

    // -----------------------------------------------------------------------
    // Debug visualisation
    // -----------------------------------------------------------------------

    /// Generate a debug visualization for the current debug view mode.
    pub fn debug_visualization(
        &self,
        px: u32,
        py: u32,
    ) -> Vec3 {
        match self.settings.debug_view {
            LumenDebugView::None => Vec3::ZERO,

            LumenDebugView::Probes => {
                // Show probe positions as dots.
                let tx = px / self.probe_grid.tile_size;
                let ty = py / self.probe_grid.tile_size;
                let center_x = tx * self.probe_grid.tile_size
                    + self.probe_grid.tile_size / 2;
                let center_y = ty * self.probe_grid.tile_size
                    + self.probe_grid.tile_size / 2;

                let dx = (px as f32 - center_x as f32).abs();
                let dy = (py as f32 - center_y as f32).abs();
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 2.0 {
                    let probe = self.probe_grid.probe_at_pixel(px, py);
                    if probe.valid {
                        Vec3::new(0.0, 1.0, 0.0)
                    } else {
                        Vec3::new(1.0, 0.0, 0.0)
                    }
                } else {
                    Vec3::ZERO
                }
            }

            LumenDebugView::Radiance => {
                self.get_gi(px, py)
            }

            LumenDebugView::Reflections => {
                self.get_reflection(px, py)
            }

            LumenDebugView::SurfaceCache => {
                // Visualize surfel density (green = more surfels).
                let density = self.surface_cache.surfel_count() as f32 / 100_000.0;
                Vec3::new(0.0, density.min(1.0), 0.0)
            }

            LumenDebugView::IndirectOnly => {
                self.get_gi(px, py) + self.get_reflection(px, py)
            }

            LumenDebugView::IrradianceField => {
                // Irradiance field coverage (blue = has data).
                if self.irradiance_field.is_some() {
                    Vec3::new(0.0, 0.0, 0.5)
                } else {
                    Vec3::ZERO
                }
            }

            LumenDebugView::Rays => {
                // Ray count heatmap.
                let probe = self.probe_grid.probe_at_pixel(px, py);
                let heat = probe.sample_count as f32 / 32.0;
                Vec3::new(heat.min(1.0), (1.0 - heat).max(0.0), 0.0)
            }

            LumenDebugView::RayHeatmap => {
                let probe = self.probe_grid.probe_at_pixel(px, py);
                let heat = probe.sample_count as f32 / 64.0;
                heatmap_color(heat)
            }
        }
    }
}

/// Convert a scalar [0, 1] to a heatmap color (blue -> green -> yellow -> red).
fn heatmap_color(t: f32) -> Vec3 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.25 {
        let f = t / 0.25;
        Vec3::new(0.0, f, 1.0 - f)
    } else if t < 0.5 {
        let f = (t - 0.25) / 0.25;
        Vec3::new(f, 1.0, 0.0)
    } else if t < 0.75 {
        let f = (t - 0.5) / 0.25;
        Vec3::new(1.0, 1.0 - f, 0.0)
    } else {
        let f = (t - 0.75) / 0.25;
        Vec3::new(1.0, 0.0, f)
    }
}

// ---------------------------------------------------------------------------
// LumenComponent -- ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching Lumen GI to an entity (typically the camera
/// or a render target).
#[derive(Debug, Clone)]
pub struct LumenComponent {
    /// Quality level.
    pub quality: LumenQuality,
    /// GI intensity override (default 1.0).
    pub gi_intensity: f32,
    /// Reflection intensity override (default 1.0).
    pub reflection_intensity: f32,
    /// Whether this component is enabled.
    pub enabled: bool,
    /// Debug view mode.
    pub debug_view: LumenDebugView,
    /// Maximum trace distance override.
    pub max_trace_distance: Option<f32>,
}

impl Default for LumenComponent {
    fn default() -> Self {
        Self {
            quality: LumenQuality::Medium,
            gi_intensity: 1.0,
            reflection_intensity: 1.0,
            enabled: true,
            debug_view: LumenDebugView::None,
            max_trace_distance: None,
        }
    }
}

impl LumenComponent {
    /// Create a new Lumen component with the given quality.
    pub fn new(quality: LumenQuality) -> Self {
        Self {
            quality,
            ..Default::default()
        }
    }

    /// Apply this component's settings to a LumenGI instance.
    pub fn apply_to(&self, lumen: &mut LumenGI) {
        if self.quality != lumen.settings.quality {
            lumen.set_quality(self.quality);
        }

        lumen.settings.gi_intensity = self.gi_intensity;
        lumen.settings.reflection_intensity = self.reflection_intensity;
        lumen.settings.debug_view = self.debug_view;

        if let Some(dist) = self.max_trace_distance {
            lumen.settings.max_trace_distance = dist;
            lumen.probe_grid.max_trace_distance = dist;
        }
    }
}

// ---------------------------------------------------------------------------
// Lumen scene setup helpers
// ---------------------------------------------------------------------------

/// Convenience function to build a BVH acceleration structure from a list
/// of triangle meshes with transforms.
pub fn build_scene_accel(
    meshes: &[Vec<Triangle>],
    transforms: &[Mat4],
) -> BVHAccelerationStructure {
    let mut accel = BVHAccelerationStructure::new();

    for (mesh_idx, triangles) in meshes.iter().enumerate() {
        let blas_idx = accel.add_mesh(triangles);

        let transform = transforms.get(mesh_idx).copied().unwrap_or(Mat4::IDENTITY);
        let instance = MeshInstance::new(blas_idx, mesh_idx as u32, transform);
        accel.add_instance(instance);
    }

    accel.rebuild_tlas();
    accel
}

/// Create a simple sky radiance function from a color and sun direction.
pub fn simple_sky(
    sky_color: Vec3,
    sun_color: Vec3,
    sun_direction: Vec3,
    sun_size: f32,
) -> impl Fn(Vec3) -> Vec3 {
    let sun_dir = sun_direction.normalize_or_zero();
    move |direction: Vec3| {
        let sky = sky_color;

        // Sun disc.
        let sun_dot = direction.dot(sun_dir).max(0.0);
        let sun_factor = if sun_dot > (1.0 - sun_size) {
            ((sun_dot - (1.0 - sun_size)) / sun_size).min(1.0)
        } else {
            0.0
        };

        // Atmospheric gradient.
        let y = direction.y.max(0.0);
        let atmosphere = Vec3::new(
            sky.x * (0.5 + 0.5 * y),
            sky.y * (0.6 + 0.4 * y),
            sky.z * (0.7 + 0.3 * y),
        );

        atmosphere + sun_color * sun_factor
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytracing::bvh_tracer::Triangle;

    fn make_test_scene() -> BVHAccelerationStructure {
        let floor = vec![
            Triangle::new(
                Vec3::new(-10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, 10.0),
            ),
            Triangle::new(
                Vec3::new(-10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, 10.0),
                Vec3::new(-10.0, 0.0, 10.0),
            ),
        ];

        let meshes = vec![floor];
        let transforms = vec![Mat4::IDENTITY];
        build_scene_accel(&meshes, &transforms)
    }

    #[test]
    fn lumen_quality_presets() {
        let low = LumenSettings::from_quality(LumenQuality::Low);
        let high = LumenSettings::from_quality(LumenQuality::High);
        assert!(
            low.probe_settings.rays_per_probe < high.probe_settings.rays_per_probe
        );
        assert!(low.max_trace_distance < high.max_trace_distance);
    }

    #[test]
    fn lumen_gi_creation() {
        let lumen = LumenGI::new(64, 64, LumenSettings::default());
        assert_eq!(lumen.width, 64);
        assert_eq!(lumen.height, 64);
        assert!(!lumen.initialized);
    }

    #[test]
    fn lumen_resize() {
        let mut lumen = LumenGI::new(64, 64, LumenSettings::default());
        lumen.resize(128, 128);
        assert_eq!(lumen.width, 128);
        assert_eq!(lumen.height, 128);
    }

    #[test]
    fn lumen_component_defaults() {
        let comp = LumenComponent::default();
        assert!(comp.enabled);
        assert_eq!(comp.quality, LumenQuality::Medium);
    }

    #[test]
    fn build_scene_accel_works() {
        let accel = make_test_scene();
        assert_eq!(accel.total_triangles(), 2);
        assert_eq!(accel.total_instances(), 1);
    }

    #[test]
    fn simple_sky_function() {
        let sky_fn = simple_sky(
            Vec3::new(0.4, 0.6, 1.0),
            Vec3::new(10.0, 9.0, 7.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.01,
        );

        let up_color = sky_fn(Vec3::Y);
        let down_color = sky_fn(-Vec3::Y);

        // Looking up at the sun should be bright.
        assert!(up_color.x > down_color.x);
    }

    #[test]
    fn heatmap_color_range() {
        let c0 = heatmap_color(0.0);
        let c1 = heatmap_color(1.0);
        // Should be different colors.
        assert!(c0 != c1);
        // Both should be valid.
        assert!(c0.x >= 0.0 && c0.y >= 0.0 && c0.z >= 0.0);
        assert!(c1.x >= 0.0 && c1.y >= 0.0 && c1.z >= 0.0);
    }

    #[test]
    fn debug_views() {
        let lumen = LumenGI::new(32, 32, LumenSettings::default());
        let viz = lumen.debug_visualization(16, 16);
        // Default debug view is None, should return zero.
        assert_eq!(viz, Vec3::ZERO);
    }

    #[test]
    fn lumen_set_quality() {
        let mut lumen = LumenGI::new(32, 32, LumenSettings::default());
        lumen.set_quality(LumenQuality::High);
        assert_eq!(lumen.settings.quality, LumenQuality::High);
        assert!(lumen.settings.enable_multi_bounce);
    }

    #[test]
    fn lumen_component_apply() {
        let mut lumen = LumenGI::new(32, 32, LumenSettings::default());
        let comp = LumenComponent {
            quality: LumenQuality::Low,
            gi_intensity: 0.5,
            ..Default::default()
        };
        comp.apply_to(&mut lumen);
        assert_eq!(lumen.settings.quality, LumenQuality::Low);
        assert!((lumen.settings.gi_intensity - 0.5).abs() < 1e-6);
    }
}
