// engine/render/src/post_process_stack.rs
//
// Configurable post-processing stack for the Genovo engine.
//
// Provides an ordered chain of post-processing effects with runtime
// configuration:
//
// - **Ordered effect chain** -- Effects are applied in a user-defined order,
//   each reading from the previous output and writing to the next input.
// - **Per-effect enable/disable/weight** -- Each effect can be toggled and
//   has a blend weight (0.0 = bypass, 1.0 = full effect).
// - **Volume-based overrides** -- Post-process volumes define spatial regions
//   that override effect parameters when the camera enters them.
// - **Transition blending** -- Smooth interpolation between volume parameter
//   sets when transitioning between zones.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of effects in the stack.
pub const MAX_EFFECTS: usize = 32;

/// Maximum number of active volumes.
pub const MAX_VOLUMES: usize = 64;

/// Default transition duration in seconds.
const DEFAULT_TRANSITION_DURATION: f32 = 1.0;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Effect parameter value
// ---------------------------------------------------------------------------

/// A typed parameter value for post-process effects.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Enum(u32),
}

impl ParamValue {
    /// Linearly interpolate between two parameter values.
    pub fn lerp(&self, other: &ParamValue, t: f32) -> ParamValue {
        let t = t.clamp(0.0, 1.0);
        match (self, other) {
            (ParamValue::Float(a), ParamValue::Float(b)) => {
                ParamValue::Float(a + (b - a) * t)
            }
            (ParamValue::Int(a), ParamValue::Int(b)) => {
                ParamValue::Int((*a as f32 + (*b - *a) as f32 * t) as i32)
            }
            (ParamValue::Bool(a), ParamValue::Bool(_b)) => {
                if t < 0.5 { ParamValue::Bool(*a) } else { other.clone() }
            }
            (ParamValue::Vec2(a), ParamValue::Vec2(b)) => {
                ParamValue::Vec2([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                ])
            }
            (ParamValue::Vec3(a), ParamValue::Vec3(b)) => {
                ParamValue::Vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ])
            }
            (ParamValue::Vec4(a), ParamValue::Vec4(b)) | (ParamValue::Color(a), ParamValue::Color(b)) => {
                let v = [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ];
                if matches!(self, ParamValue::Color(_)) {
                    ParamValue::Color(v)
                } else {
                    ParamValue::Vec4(v)
                }
            }
            (ParamValue::Enum(_), _) => {
                if t < 0.5 { self.clone() } else { other.clone() }
            }
            _ => self.clone(),
        }
    }

    /// Get as f32 if applicable.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as bool if applicable.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Effect parameter set
// ---------------------------------------------------------------------------

/// A set of named parameters for a post-process effect.
#[derive(Debug, Clone, Default)]
pub struct ParamSet {
    pub params: HashMap<String, ParamValue>,
}

impl ParamSet {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Set a parameter value.
    pub fn set(&mut self, name: &str, value: ParamValue) {
        self.params.insert(name.to_string(), value);
    }

    /// Get a parameter value.
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.params.get(name)
    }

    /// Get a float parameter with a default value.
    pub fn get_float(&self, name: &str, default: f32) -> f32 {
        self.params
            .get(name)
            .and_then(|v| v.as_float())
            .unwrap_or(default)
    }

    /// Get a bool parameter with a default value.
    pub fn get_bool(&self, name: &str, default: bool) -> bool {
        self.params
            .get(name)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Lerp between this parameter set and another.
    pub fn lerp(&self, other: &ParamSet, t: f32) -> ParamSet {
        let mut result = self.clone();
        for (key, other_val) in &other.params {
            if let Some(self_val) = self.params.get(key) {
                result.params.insert(key.clone(), self_val.lerp(other_val, t));
            } else {
                result.params.insert(key.clone(), other_val.clone());
            }
        }
        result
    }

    /// Returns true if this set has no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }
}

// ---------------------------------------------------------------------------
// Effect type identifier
// ---------------------------------------------------------------------------

/// Identifies a type of post-process effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectType {
    Bloom,
    ToneMapping,
    ColorGrading,
    Vignette,
    ChromaticAberration,
    FilmGrain,
    DepthOfField,
    MotionBlur,
    ScreenSpaceReflections,
    AmbientOcclusion,
    Fog,
    LensFlare,
    Sharpen,
    Fxaa,
    Taa,
    Custom(u32),
}

impl fmt::Display for EffectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bloom => write!(f, "Bloom"),
            Self::ToneMapping => write!(f, "ToneMapping"),
            Self::ColorGrading => write!(f, "ColorGrading"),
            Self::Vignette => write!(f, "Vignette"),
            Self::ChromaticAberration => write!(f, "ChromaticAberration"),
            Self::FilmGrain => write!(f, "FilmGrain"),
            Self::DepthOfField => write!(f, "DepthOfField"),
            Self::MotionBlur => write!(f, "MotionBlur"),
            Self::ScreenSpaceReflections => write!(f, "SSR"),
            Self::AmbientOcclusion => write!(f, "AO"),
            Self::Fog => write!(f, "Fog"),
            Self::LensFlare => write!(f, "LensFlare"),
            Self::Sharpen => write!(f, "Sharpen"),
            Self::Fxaa => write!(f, "FXAA"),
            Self::Taa => write!(f, "TAA"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Post-process effect
// ---------------------------------------------------------------------------

/// A single post-process effect in the stack.
#[derive(Debug, Clone)]
pub struct PostProcessEffect {
    /// Effect type.
    pub effect_type: EffectType,
    /// Display name.
    pub name: String,
    /// Whether this effect is enabled.
    pub enabled: bool,
    /// Blend weight (0.0 = bypass, 1.0 = full).
    pub weight: f32,
    /// Default parameter values.
    pub default_params: ParamSet,
    /// Current (possibly volume-overridden) parameter values.
    pub current_params: ParamSet,
    /// Render order (lower = earlier in the chain).
    pub order: i32,
    /// Whether this effect requires depth buffer input.
    pub needs_depth: bool,
    /// Whether this effect requires motion vectors.
    pub needs_motion_vectors: bool,
    /// Whether this effect requires normal buffer.
    pub needs_normals: bool,
}

impl PostProcessEffect {
    /// Create a new effect with default settings.
    pub fn new(effect_type: EffectType, name: &str, order: i32) -> Self {
        Self {
            effect_type,
            name: name.to_string(),
            enabled: true,
            weight: 1.0,
            default_params: ParamSet::new(),
            current_params: ParamSet::new(),
            order,
            needs_depth: false,
            needs_motion_vectors: false,
            needs_normals: false,
        }
    }

    /// Set a default parameter.
    pub fn with_param(mut self, name: &str, value: ParamValue) -> Self {
        self.default_params.set(name, value.clone());
        self.current_params.set(name, value);
        self
    }

    /// Set whether depth is required.
    pub fn with_depth(mut self) -> Self {
        self.needs_depth = true;
        self
    }

    /// Set whether motion vectors are required.
    pub fn with_motion_vectors(mut self) -> Self {
        self.needs_motion_vectors = true;
        self
    }

    /// Reset current params to defaults.
    pub fn reset_to_defaults(&mut self) {
        self.current_params = self.default_params.clone();
    }

    /// Get the effective weight (0 if disabled).
    pub fn effective_weight(&self) -> f32 {
        if self.enabled { self.weight } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Volume shape
// ---------------------------------------------------------------------------

/// Shape of a post-process volume trigger region.
#[derive(Debug, Clone)]
pub enum VolumeShape {
    /// Axis-aligned box.
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    /// Sphere.
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Global (always active, used for default settings).
    Global,
}

impl VolumeShape {
    /// Check if a point is inside this volume.
    pub fn contains(&self, point: [f32; 3]) -> bool {
        match self {
            Self::Box { min, max } => {
                point[0] >= min[0]
                    && point[0] <= max[0]
                    && point[1] >= min[1]
                    && point[1] <= max[1]
                    && point[2] >= min[2]
                    && point[2] <= max[2]
            }
            Self::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }
            Self::Global => true,
        }
    }

    /// Compute the blend factor based on distance to the volume boundary.
    pub fn blend_factor(&self, point: [f32; 3], blend_distance: f32) -> f32 {
        if blend_distance <= EPSILON {
            return if self.contains(point) { 1.0 } else { 0.0 };
        }
        match self {
            Self::Global => 1.0,
            Self::Box { min, max } => {
                // Distance to nearest box face (negative if inside).
                let dx = (min[0] - point[0]).max(point[0] - max[0]).max(0.0);
                let dy = (min[1] - point[1]).max(point[1] - max[1]).max(0.0);
                let dz = (min[2] - point[2]).max(point[2] - max[2]).max(0.0);
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= 0.0 {
                    1.0
                } else {
                    (1.0 - dist / blend_distance).max(0.0)
                }
            }
            Self::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= *radius {
                    1.0
                } else {
                    (1.0 - (dist - radius) / blend_distance).max(0.0)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Post-process volume
// ---------------------------------------------------------------------------

/// Volume ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VolumeId(pub u32);

/// A post-process volume that overrides effect parameters in a spatial region.
#[derive(Debug, Clone)]
pub struct PostProcessVolume {
    /// Unique ID.
    pub id: VolumeId,
    /// Display name.
    pub name: String,
    /// Shape of the volume.
    pub shape: VolumeShape,
    /// Priority (higher overrides lower).
    pub priority: i32,
    /// Blend distance for smooth transitions at volume boundaries.
    pub blend_distance: f32,
    /// Whether this volume is active.
    pub active: bool,
    /// Per-effect parameter overrides.
    pub overrides: HashMap<EffectType, VolumeOverride>,
}

/// Parameter overrides for a specific effect within a volume.
#[derive(Debug, Clone)]
pub struct VolumeOverride {
    /// Whether to override the effect's enabled state.
    pub override_enabled: Option<bool>,
    /// Whether to override the weight.
    pub override_weight: Option<f32>,
    /// Parameter value overrides.
    pub params: ParamSet,
}

impl VolumeOverride {
    /// Create an empty override.
    pub fn new() -> Self {
        Self {
            override_enabled: None,
            override_weight: None,
            params: ParamSet::new(),
        }
    }

    /// Set an enabled override.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.override_enabled = Some(enabled);
        self
    }

    /// Set a weight override.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.override_weight = Some(weight);
        self
    }

    /// Set a parameter override.
    pub fn with_param(mut self, name: &str, value: ParamValue) -> Self {
        self.params.set(name, value);
        self
    }
}

impl Default for VolumeOverride {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Transition state
// ---------------------------------------------------------------------------

/// Tracks transition blending between volume configurations.
#[derive(Debug, Clone)]
struct TransitionState {
    /// Previous parameter set (what we are blending from).
    from_params: HashMap<EffectType, ParamSet>,
    /// Target parameter set (what we are blending to).
    to_params: HashMap<EffectType, ParamSet>,
    /// Current blend progress (0.0 = from, 1.0 = to).
    progress: f32,
    /// Duration of the transition in seconds.
    duration: f32,
    /// Whether a transition is active.
    active: bool,
}

impl TransitionState {
    fn new() -> Self {
        Self {
            from_params: HashMap::new(),
            to_params: HashMap::new(),
            progress: 1.0,
            duration: DEFAULT_TRANSITION_DURATION,
            active: false,
        }
    }

    fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        if self.duration <= EPSILON {
            self.progress = 1.0;
            self.active = false;
            return;
        }
        self.progress += dt / self.duration;
        if self.progress >= 1.0 {
            self.progress = 1.0;
            self.active = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Post-process stack
// ---------------------------------------------------------------------------

/// Statistics for the post-process stack.
#[derive(Debug, Clone, Default)]
pub struct PostProcessStats {
    /// Number of effects in the stack.
    pub total_effects: usize,
    /// Number of enabled effects.
    pub enabled_effects: usize,
    /// Number of active volumes.
    pub active_volumes: usize,
    /// Number of volumes currently influencing the camera.
    pub influencing_volumes: usize,
    /// Whether a transition is in progress.
    pub transitioning: bool,
    /// Transition progress (0..1).
    pub transition_progress: f32,
}

/// Manages an ordered stack of post-process effects with volume overrides.
pub struct PostProcessStack {
    /// Ordered list of effects.
    effects: Vec<PostProcessEffect>,
    /// Post-process volumes.
    volumes: Vec<PostProcessVolume>,
    /// Next volume ID.
    next_volume_id: u32,
    /// Transition state.
    transition: TransitionState,
    /// Default transition duration.
    transition_duration: f32,
    /// Current camera position (for volume evaluation).
    camera_position: [f32; 3],
    /// Statistics.
    stats: PostProcessStats,
}

impl PostProcessStack {
    /// Create a new empty post-process stack.
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            volumes: Vec::new(),
            next_volume_id: 0,
            transition: TransitionState::new(),
            transition_duration: DEFAULT_TRANSITION_DURATION,
            camera_position: [0.0; 3],
            stats: PostProcessStats::default(),
        }
    }

    /// Add an effect to the stack.
    pub fn add_effect(&mut self, effect: PostProcessEffect) {
        self.effects.push(effect);
        self.sort_effects();
    }

    /// Remove an effect by type.
    pub fn remove_effect(&mut self, effect_type: EffectType) -> bool {
        let len_before = self.effects.len();
        self.effects.retain(|e| e.effect_type != effect_type);
        self.effects.len() != len_before
    }

    /// Get an effect by type.
    pub fn get_effect(&self, effect_type: EffectType) -> Option<&PostProcessEffect> {
        self.effects.iter().find(|e| e.effect_type == effect_type)
    }

    /// Get a mutable effect by type.
    pub fn get_effect_mut(&mut self, effect_type: EffectType) -> Option<&mut PostProcessEffect> {
        self.effects.iter_mut().find(|e| e.effect_type == effect_type)
    }

    /// Enable or disable an effect.
    pub fn set_effect_enabled(&mut self, effect_type: EffectType, enabled: bool) {
        if let Some(effect) = self.get_effect_mut(effect_type) {
            effect.enabled = enabled;
        }
    }

    /// Set the weight of an effect.
    pub fn set_effect_weight(&mut self, effect_type: EffectType, weight: f32) {
        if let Some(effect) = self.get_effect_mut(effect_type) {
            effect.weight = weight.clamp(0.0, 1.0);
        }
    }

    /// Get all effects in order.
    pub fn effects(&self) -> &[PostProcessEffect] {
        &self.effects
    }

    /// Get enabled effects in order.
    pub fn enabled_effects(&self) -> Vec<&PostProcessEffect> {
        self.effects.iter().filter(|e| e.enabled && e.weight > EPSILON).collect()
    }

    /// Sort effects by their order field.
    fn sort_effects(&mut self) {
        self.effects.sort_by_key(|e| e.order);
    }

    /// Add a post-process volume.
    pub fn add_volume(&mut self, name: &str, shape: VolumeShape, priority: i32) -> VolumeId {
        let id = VolumeId(self.next_volume_id);
        self.next_volume_id += 1;
        self.volumes.push(PostProcessVolume {
            id,
            name: name.to_string(),
            shape,
            priority,
            blend_distance: 2.0,
            active: true,
            overrides: HashMap::new(),
        });
        id
    }

    /// Remove a volume by ID.
    pub fn remove_volume(&mut self, id: VolumeId) -> bool {
        let len_before = self.volumes.len();
        self.volumes.retain(|v| v.id != id);
        self.volumes.len() != len_before
    }

    /// Get a mutable reference to a volume.
    pub fn volume_mut(&mut self, id: VolumeId) -> Option<&mut PostProcessVolume> {
        self.volumes.iter_mut().find(|v| v.id == id)
    }

    /// Set an effect override in a volume.
    pub fn set_volume_override(
        &mut self,
        volume_id: VolumeId,
        effect_type: EffectType,
        override_data: VolumeOverride,
    ) {
        if let Some(volume) = self.volume_mut(volume_id) {
            volume.overrides.insert(effect_type, override_data);
        }
    }

    /// Set the transition duration.
    pub fn set_transition_duration(&mut self, seconds: f32) {
        self.transition_duration = seconds.max(0.0);
    }

    /// Update the post-process stack.
    ///
    /// This evaluates volumes, applies overrides, and updates transitions.
    pub fn update(&mut self, camera_position: [f32; 3], dt: f32) {
        self.camera_position = camera_position;

        // Update transition.
        self.transition.update(dt);

        // Reset all effects to defaults first.
        for effect in &mut self.effects {
            effect.current_params = effect.default_params.clone();
        }

        // Gather active volumes sorted by priority.
        let mut active_volumes: Vec<(usize, f32)> = Vec::new();
        for (i, volume) in self.volumes.iter().enumerate() {
            if !volume.active {
                continue;
            }
            let blend = volume.shape.blend_factor(camera_position, volume.blend_distance);
            if blend > EPSILON {
                active_volumes.push((i, blend));
            }
        }
        active_volumes.sort_by_key(|&(i, _)| self.volumes[i].priority);

        // Apply volume overrides in priority order.
        for &(vol_idx, blend_factor) in &active_volumes {
            let volume = &self.volumes[vol_idx];
            for (effect_type, override_data) in &volume.overrides {
                if let Some(effect) = self.effects.iter_mut().find(|e| e.effect_type == *effect_type) {
                    // Apply enabled override.
                    if let Some(enabled) = override_data.override_enabled {
                        if blend_factor > 0.5 {
                            effect.enabled = enabled;
                        }
                    }
                    // Apply weight override.
                    if let Some(weight) = override_data.override_weight {
                        let current = effect.weight;
                        effect.weight = current + (weight - current) * blend_factor;
                    }
                    // Apply parameter overrides.
                    for (param_name, param_value) in &override_data.params.params {
                        if let Some(current_value) = effect.current_params.params.get(param_name) {
                            let blended = current_value.lerp(param_value, blend_factor);
                            effect.current_params.params.insert(param_name.clone(), blended);
                        } else {
                            effect.current_params.params.insert(param_name.clone(), param_value.clone());
                        }
                    }
                }
            }
        }

        // Update statistics.
        self.stats.total_effects = self.effects.len();
        self.stats.enabled_effects = self.effects.iter().filter(|e| e.enabled).count();
        self.stats.active_volumes = self.volumes.iter().filter(|v| v.active).count();
        self.stats.influencing_volumes = active_volumes.len();
        self.stats.transitioning = self.transition.active;
        self.stats.transition_progress = self.transition.progress;
    }

    /// Get current statistics.
    pub fn stats(&self) -> &PostProcessStats {
        &self.stats
    }

    /// Create a default stack with common effects.
    pub fn default_stack() -> Self {
        let mut stack = Self::new();

        stack.add_effect(
            PostProcessEffect::new(EffectType::AmbientOcclusion, "Ambient Occlusion", 100)
                .with_depth()
                .with_param("radius", ParamValue::Float(0.5))
                .with_param("intensity", ParamValue::Float(1.0))
                .with_param("bias", ParamValue::Float(0.025)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::ScreenSpaceReflections, "SSR", 200)
                .with_depth()
                .with_param("max_steps", ParamValue::Int(64))
                .with_param("thickness", ParamValue::Float(0.1)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::Bloom, "Bloom", 300)
                .with_param("threshold", ParamValue::Float(1.0))
                .with_param("intensity", ParamValue::Float(0.5))
                .with_param("radius", ParamValue::Float(4.0)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::DepthOfField, "Depth of Field", 400)
                .with_depth()
                .with_param("focus_distance", ParamValue::Float(10.0))
                .with_param("aperture", ParamValue::Float(5.6))
                .with_param("focal_length", ParamValue::Float(50.0)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::MotionBlur, "Motion Blur", 500)
                .with_depth()
                .with_motion_vectors()
                .with_param("intensity", ParamValue::Float(1.0))
                .with_param("max_samples", ParamValue::Int(16)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::ColorGrading, "Color Grading", 600)
                .with_param("temperature", ParamValue::Float(0.0))
                .with_param("tint", ParamValue::Float(0.0))
                .with_param("saturation", ParamValue::Float(1.0))
                .with_param("contrast", ParamValue::Float(1.0))
                .with_param("brightness", ParamValue::Float(0.0)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::ToneMapping, "Tone Mapping", 700)
                .with_param("mode", ParamValue::Enum(0))
                .with_param("exposure", ParamValue::Float(1.0)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::Vignette, "Vignette", 800)
                .with_param("intensity", ParamValue::Float(0.3))
                .with_param("smoothness", ParamValue::Float(0.3))
                .with_param("roundness", ParamValue::Float(1.0))
                .with_param("color", ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::ChromaticAberration, "Chromatic Aberration", 850)
                .with_param("intensity", ParamValue::Float(0.0)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::FilmGrain, "Film Grain", 900)
                .with_param("intensity", ParamValue::Float(0.0))
                .with_param("response", ParamValue::Float(0.8)),
        );

        stack.add_effect(
            PostProcessEffect::new(EffectType::Fxaa, "FXAA", 1000)
                .with_param("quality", ParamValue::Enum(1)),
        );

        stack
    }
}

impl Default for PostProcessStack {
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
    fn test_param_lerp() {
        let a = ParamValue::Float(0.0);
        let b = ParamValue::Float(1.0);
        if let ParamValue::Float(v) = a.lerp(&b, 0.5) {
            assert!((v - 0.5).abs() < EPSILON);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_volume_contains() {
        let shape = VolumeShape::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        assert!(shape.contains([1.0, 1.0, 1.0]));
        assert!(!shape.contains([10.0, 0.0, 0.0]));
    }

    #[test]
    fn test_effect_ordering() {
        let mut stack = PostProcessStack::new();
        stack.add_effect(PostProcessEffect::new(EffectType::Bloom, "Bloom", 200));
        stack.add_effect(PostProcessEffect::new(EffectType::ToneMapping, "TM", 100));
        assert_eq!(stack.effects()[0].effect_type, EffectType::ToneMapping);
        assert_eq!(stack.effects()[1].effect_type, EffectType::Bloom);
    }

    #[test]
    fn test_volume_override() {
        let mut stack = PostProcessStack::new();
        stack.add_effect(
            PostProcessEffect::new(EffectType::Bloom, "Bloom", 100)
                .with_param("intensity", ParamValue::Float(0.5)),
        );
        let vol_id = stack.add_volume(
            "indoor",
            VolumeShape::Box {
                min: [-10.0, -10.0, -10.0],
                max: [10.0, 10.0, 10.0],
            },
            1,
        );
        stack.set_volume_override(
            vol_id,
            EffectType::Bloom,
            VolumeOverride::new().with_param("intensity", ParamValue::Float(2.0)),
        );

        // Camera inside volume.
        stack.update([0.0, 0.0, 0.0], 0.016);
        let bloom = stack.get_effect(EffectType::Bloom).unwrap();
        let intensity = bloom.current_params.get_float("intensity", 0.0);
        assert!((intensity - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_default_stack() {
        let stack = PostProcessStack::default_stack();
        assert!(stack.effects().len() >= 8);
    }
}
