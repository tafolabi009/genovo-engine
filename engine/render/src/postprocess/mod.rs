// engine/render/src/postprocess/mod.rs
//
// Post-processing pipeline framework for the Genovo engine.
// Manages an ordered chain of screen-space effects applied after the main
// scene render pass, transforming the raw HDR scene color into the final
// display-ready image.

pub mod bloom;
pub mod color_grading;
pub mod dof;
pub mod fxaa;
pub mod motion_blur;
pub mod ssao;
pub mod ssr;
pub mod tonemapping;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use bloom::{BloomEffect, BloomSettings};
pub use color_grading::{
    ChromaticAberrationSettings, ColorGradingEffect, ColorGradingSettings, FilmGrainSettings,
    SplitToningSettings, VignetteSettings,
};
pub use dof::{DOFEffect, DOFSettings};
pub use fxaa::{FXAAEffect, FXAAQualityPreset, FXAASettings, TAASettings};
pub use motion_blur::{MotionBlurEffect, MotionBlurSettings};
pub use ssao::{SSAOEffect, SSAOMode, SSAOSettings};
pub use ssr::{SSREffect, SSRSettings};
pub use tonemapping::{
    AutoExposureState, ExposureSettings, ToneMapOperator, ToneMappingEffect, ToneMappingSettings,
};

use std::any::Any;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Camera data shared across all post-process effects
// ---------------------------------------------------------------------------

/// Camera information required by most post-process effects.
#[derive(Debug, Clone)]
pub struct CameraData {
    /// View matrix (world -> view space).
    pub view: [[f32; 4]; 4],
    /// Projection matrix (view -> clip space).
    pub projection: [[f32; 4]; 4],
    /// Inverse view matrix.
    pub inv_view: [[f32; 4]; 4],
    /// Inverse projection matrix.
    pub inv_projection: [[f32; 4]; 4],
    /// Previous frame view-projection (for temporal effects).
    pub prev_view_projection: [[f32; 4]; 4],
    /// Current frame view-projection.
    pub view_projection: [[f32; 4]; 4],
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Near clip plane distance.
    pub near_plane: f32,
    /// Far clip plane distance.
    pub far_plane: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Viewport width in pixels.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
    /// Jitter offset for TAA (in pixels).
    pub jitter: [f32; 2],
    /// Frame index (monotonically increasing).
    pub frame_index: u64,
    /// Delta time since last frame in seconds.
    pub delta_time: f32,
}

impl Default for CameraData {
    fn default() -> Self {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            view: identity,
            projection: identity,
            inv_view: identity,
            inv_projection: identity,
            prev_view_projection: identity,
            view_projection: identity,
            position: [0.0; 3],
            near_plane: 0.1,
            far_plane: 1000.0,
            fov_y: std::f32::consts::FRAC_PI_4,
            viewport_width: 1920,
            viewport_height: 1080,
            jitter: [0.0; 2],
            frame_index: 0,
            delta_time: 1.0 / 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture handle (lightweight reference to a GPU texture)
// ---------------------------------------------------------------------------

/// Lightweight handle referencing a GPU texture resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u64);

impl TextureId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

/// Describes a texture used in the post-process chain.
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub id: TextureId,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub mip_levels: u32,
}

/// Supported texture formats for post-process intermediates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba16Float,
    Rgba32Float,
    R16Float,
    R32Float,
    Rg16Float,
    Rg32Float,
    R8Unorm,
    Rg8Unorm,
    Depth32Float,
    Depth24PlusStencil8,
}

// ---------------------------------------------------------------------------
// Post-process I/O
// ---------------------------------------------------------------------------

/// Input data provided to each post-process effect.
pub struct PostProcessInput {
    /// HDR scene color texture (linear, pre-tonemap).
    pub color_texture: TextureId,
    /// Scene depth buffer.
    pub depth_texture: TextureId,
    /// G-Buffer normal texture (view-space or world-space normals).
    pub normal_texture: TextureId,
    /// Per-pixel velocity / motion vector texture.
    pub velocity_texture: TextureId,
    /// SSAO result (may be INVALID if SSAO hasn't run yet).
    pub ao_texture: TextureId,
    /// SSR result (may be INVALID if SSR hasn't run yet).
    pub ssr_texture: TextureId,
    /// Camera data for the current frame.
    pub camera: CameraData,
    /// Viewport dimensions (may differ from camera if rendering at lower res).
    pub viewport_width: u32,
    pub viewport_height: u32,
    /// History color from the previous frame (for temporal effects).
    pub history_color: TextureId,
    /// History depth from the previous frame.
    pub history_depth: TextureId,
}

/// Output target for a post-process effect.
pub struct PostProcessOutput {
    /// Destination texture that the effect writes to.
    pub target_texture: TextureId,
    /// Width of the output target.
    pub width: u32,
    /// Height of the output target.
    pub height: u32,
}

// ---------------------------------------------------------------------------
// PostProcessEffect trait
// ---------------------------------------------------------------------------

/// Trait that all post-processing effects must implement.
///
/// Effects are executed in order by the `PostProcessPipeline`. Each effect
/// reads from `PostProcessInput` (which includes the previous effect's output
/// as the color texture) and writes to `PostProcessOutput`.
pub trait PostProcessEffect: Send + Sync {
    /// Human-readable name for debug UI and profiling.
    fn name(&self) -> &str;

    /// Execute the effect, reading from `input` and writing to `output`.
    fn execute(&self, input: &PostProcessInput, output: &mut PostProcessOutput);

    /// Whether this effect is currently enabled.
    fn is_enabled(&self) -> bool;

    /// Enable or disable the effect at runtime.
    fn set_enabled(&mut self, enabled: bool);

    /// Priority used for default ordering (lower = earlier in chain).
    /// Standard values:
    ///   SSAO = 100, SSR = 200, DOF = 300, Motion Blur = 400,
    ///   Bloom = 500, Tonemapping = 600, Color Grading = 700,
    ///   FXAA/TAA = 800
    fn priority(&self) -> u32;

    /// Allows effects to allocate or resize internal textures when the
    /// viewport dimensions change.
    fn on_resize(&mut self, width: u32, height: u32);

    /// Downcast support for accessing concrete effect settings.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ---------------------------------------------------------------------------
// PostProcessPipeline
// ---------------------------------------------------------------------------

/// Entry in the pipeline storing an effect with its metadata.
struct EffectEntry {
    effect: Box<dyn PostProcessEffect>,
    /// User-assigned order override. If `None`, uses the effect's priority().
    order_override: Option<u32>,
}

impl EffectEntry {
    fn effective_order(&self) -> u32 {
        self.order_override.unwrap_or_else(|| self.effect.priority())
    }
}

/// Manages an ordered chain of post-process effects.
///
/// The pipeline owns all effects and executes them in priority order.
/// Intermediate ping-pong textures are managed automatically so each
/// effect reads the previous effect's output as its input color.
pub struct PostProcessPipeline {
    effects: Vec<EffectEntry>,
    /// Ping-pong intermediate textures.
    ping_texture: TextureId,
    pong_texture: TextureId,
    /// Current viewport dimensions.
    viewport_width: u32,
    viewport_height: u32,
    /// Whether the chain needs re-sorting after an add/remove/reorder.
    dirty: bool,
    /// Named texture slots that effects can write to / read from.
    named_textures: HashMap<String, TextureId>,
    /// Global enable/disable for the entire post-process chain.
    enabled: bool,
}

impl PostProcessPipeline {
    /// Create a new empty pipeline.
    pub fn new(viewport_width: u32, viewport_height: u32) -> Self {
        Self {
            effects: Vec::new(),
            ping_texture: TextureId(0),
            pong_texture: TextureId(1),
            viewport_width,
            viewport_height,
            dirty: false,
            named_textures: HashMap::new(),
            enabled: true,
        }
    }

    /// Add an effect to the pipeline. It will be inserted according to its
    /// priority.
    pub fn add_effect(&mut self, effect: Box<dyn PostProcessEffect>) {
        self.effects.push(EffectEntry {
            effect,
            order_override: None,
        });
        self.dirty = true;
    }

    /// Add an effect with a custom ordering value, overriding its default
    /// priority.
    pub fn add_effect_with_order(&mut self, effect: Box<dyn PostProcessEffect>, order: u32) {
        self.effects.push(EffectEntry {
            effect,
            order_override: Some(order),
        });
        self.dirty = true;
    }

    /// Remove an effect by name. Returns the removed effect if found.
    pub fn remove_effect(&mut self, name: &str) -> Option<Box<dyn PostProcessEffect>> {
        if let Some(idx) = self.effects.iter().position(|e| e.effect.name() == name) {
            Some(self.effects.remove(idx).effect)
        } else {
            None
        }
    }

    /// Enable or disable an effect by name.
    pub fn set_effect_enabled(&mut self, name: &str, enabled: bool) {
        for entry in &mut self.effects {
            if entry.effect.name() == name {
                entry.effect.set_enabled(enabled);
                break;
            }
        }
    }

    /// Override the ordering of an existing effect.
    pub fn set_effect_order(&mut self, name: &str, order: u32) {
        for entry in &mut self.effects {
            if entry.effect.name() == name {
                entry.order_override = Some(order);
                self.dirty = true;
                break;
            }
        }
    }

    /// Get a reference to an effect by name.
    pub fn get_effect(&self, name: &str) -> Option<&dyn PostProcessEffect> {
        self.effects
            .iter()
            .find(|e| e.effect.name() == name)
            .map(|e| e.effect.as_ref())
    }

    /// Get a mutable reference to an effect by name.
    pub fn get_effect_mut(&mut self, name: &str) -> Option<&mut (dyn PostProcessEffect + 'static)> {
        for entry in &mut self.effects {
            if entry.effect.name() == name {
                return Some(entry.effect.as_mut());
            }
        }
        None
    }

    /// Get a typed reference to a specific effect.
    pub fn get_effect_as<T: PostProcessEffect + 'static>(&self, name: &str) -> Option<&T> {
        self.get_effect(name)
            .and_then(|e| e.as_any().downcast_ref::<T>())
    }

    /// Get a typed mutable reference to a specific effect.
    pub fn get_effect_as_mut<T: PostProcessEffect + 'static>(
        &mut self,
        name: &str,
    ) -> Option<&mut T> {
        self.get_effect_mut(name)
            .and_then(|e| e.as_any_mut().downcast_mut::<T>())
    }

    /// Register a named texture slot (e.g., "ssao_result", "ssr_result").
    pub fn register_texture(&mut self, name: &str, texture: TextureId) {
        self.named_textures.insert(name.to_string(), texture);
    }

    /// Look up a named texture.
    pub fn get_named_texture(&self, name: &str) -> Option<TextureId> {
        self.named_textures.get(name).copied()
    }

    /// Enable or disable the entire pipeline.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether the pipeline is globally enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Notify all effects that the viewport has been resized.
    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.viewport_width = width;
        self.viewport_height = height;
        for entry in &mut self.effects {
            entry.effect.on_resize(width, height);
        }
    }

    /// Return a list of effect names in execution order.
    pub fn effect_names(&mut self) -> Vec<String> {
        self.ensure_sorted();
        self.effects
            .iter()
            .map(|e| e.effect.name().to_string())
            .collect()
    }

    /// Return the number of effects in the pipeline.
    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }

    /// Sort effects by their effective order if dirty.
    fn ensure_sorted(&mut self) {
        if self.dirty {
            self.effects.sort_by_key(|e| e.effective_order());
            self.dirty = false;
        }
    }

    /// Execute the entire post-processing chain.
    ///
    /// `scene_color` is the HDR render target from the main pass.
    /// `scene_depth` is the depth buffer.
    /// `camera` provides view/projection data.
    ///
    /// Returns the `TextureId` of the final output.
    pub fn execute_chain(
        &mut self,
        scene_color: TextureId,
        scene_depth: TextureId,
        scene_normals: TextureId,
        scene_velocity: TextureId,
        camera: &CameraData,
    ) -> TextureId {
        if !self.enabled {
            return scene_color;
        }

        self.ensure_sorted();

        // Count enabled effects to determine if we need to do anything.
        let enabled_count = self
            .effects
            .iter()
            .filter(|e| e.effect.is_enabled())
            .count();

        if enabled_count == 0 {
            return scene_color;
        }

        // Build the base input. The color_texture is updated after each
        // effect to point to the previous effect's output.
        let mut current_color = scene_color;
        let mut use_ping = true;

        let ao_texture = self
            .named_textures
            .get("ssao_result")
            .copied()
            .unwrap_or(TextureId::INVALID);
        let ssr_texture = self
            .named_textures
            .get("ssr_result")
            .copied()
            .unwrap_or(TextureId::INVALID);
        let history_color = self
            .named_textures
            .get("history_color")
            .copied()
            .unwrap_or(TextureId::INVALID);
        let history_depth = self
            .named_textures
            .get("history_depth")
            .copied()
            .unwrap_or(TextureId::INVALID);

        // We need indices because we borrow self.effects immutably in the
        // loop body but the pipeline owns the ping/pong state.
        let effect_count = self.effects.len();

        for i in 0..effect_count {
            if !self.effects[i].effect.is_enabled() {
                continue;
            }

            let target = if use_ping {
                self.ping_texture
            } else {
                self.pong_texture
            };

            let input = PostProcessInput {
                color_texture: current_color,
                depth_texture: scene_depth,
                normal_texture: scene_normals,
                velocity_texture: scene_velocity,
                ao_texture,
                ssr_texture,
                camera: camera.clone(),
                viewport_width: self.viewport_width,
                viewport_height: self.viewport_height,
                history_color,
                history_depth,
            };

            let mut output = PostProcessOutput {
                target_texture: target,
                width: self.viewport_width,
                height: self.viewport_height,
            };

            self.effects[i].effect.execute(&input, &mut output);

            current_color = output.target_texture;
            use_ping = !use_ping;
        }

        current_color
    }

    /// Build a default pipeline with common effects pre-configured.
    pub fn default_pipeline(width: u32, height: u32) -> Self {
        let mut pipeline = Self::new(width, height);

        pipeline.add_effect(Box::new(SSAOEffect::new(SSAOSettings::default())));
        pipeline.add_effect(Box::new(SSREffect::new(SSRSettings::default())));
        pipeline.add_effect(Box::new(DOFEffect::new(DOFSettings::default())));
        pipeline.add_effect(Box::new(MotionBlurEffect::new(
            MotionBlurSettings::default(),
        )));
        pipeline.add_effect(Box::new(BloomEffect::new(BloomSettings::default())));
        pipeline.add_effect(Box::new(ToneMappingEffect::new(
            ToneMappingSettings::default(),
        )));
        pipeline.add_effect(Box::new(ColorGradingEffect::new(
            ColorGradingSettings::default(),
        )));
        pipeline.add_effect(Box::new(FXAAEffect::new(FXAASettings::default())));

        pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal effect for testing the pipeline.
    struct DummyEffect {
        name: String,
        enabled: bool,
        priority: u32,
        execute_count: std::sync::atomic::AtomicU32,
    }

    impl DummyEffect {
        fn new(name: &str, priority: u32) -> Self {
            Self {
                name: name.to_string(),
                enabled: true,
                priority,
                execute_count: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl PostProcessEffect for DummyEffect {
        fn name(&self) -> &str {
            &self.name
        }
        fn execute(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
            self.execute_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn is_enabled(&self) -> bool {
            self.enabled
        }
        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }
        fn priority(&self) -> u32 {
            self.priority
        }
        fn on_resize(&mut self, _w: u32, _h: u32) {}
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_pipeline_ordering() {
        let mut pipeline = PostProcessPipeline::new(1920, 1080);
        pipeline.add_effect(Box::new(DummyEffect::new("C", 300)));
        pipeline.add_effect(Box::new(DummyEffect::new("A", 100)));
        pipeline.add_effect(Box::new(DummyEffect::new("B", 200)));

        let names = pipeline.effect_names();
        assert_eq!(names, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_enable_disable() {
        let mut pipeline = PostProcessPipeline::new(1920, 1080);
        pipeline.add_effect(Box::new(DummyEffect::new("test", 100)));
        assert!(pipeline.get_effect("test").unwrap().is_enabled());

        pipeline.set_effect_enabled("test", false);
        assert!(!pipeline.get_effect("test").unwrap().is_enabled());
    }

    #[test]
    fn test_remove_effect() {
        let mut pipeline = PostProcessPipeline::new(1920, 1080);
        pipeline.add_effect(Box::new(DummyEffect::new("remove_me", 100)));
        assert_eq!(pipeline.effect_count(), 1);

        let removed = pipeline.remove_effect("remove_me");
        assert!(removed.is_some());
        assert_eq!(pipeline.effect_count(), 0);
    }

    #[test]
    fn test_order_override() {
        let mut pipeline = PostProcessPipeline::new(1920, 1080);
        pipeline.add_effect(Box::new(DummyEffect::new("first", 100)));
        pipeline.add_effect(Box::new(DummyEffect::new("second", 200)));

        // Override so "second" runs before "first".
        pipeline.set_effect_order("second", 50);

        let names = pipeline.effect_names();
        assert_eq!(names, vec!["second", "first"]);
    }
}
