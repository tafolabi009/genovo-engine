// engine/render/src/material_instance_v2.rs
//
// Material instances with GPU uniform buffers: per-material uniform buffer,
// dirty tracking, batch update, material parameter animation, material LOD
// (simplified shader at distance).
//
// A material instance is a lightweight runtime clone of a material template
// that can override parameters (colors, floats, textures) without duplicating
// the shader or pipeline state. Each instance owns a small GPU uniform buffer
// containing the overridden parameters, and dirty tracking ensures only changed
// parameters are uploaded each frame.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

/// Unique identifier for a material template.
pub type MaterialTemplateId = u64;

/// Unique identifier for a material instance.
pub type MaterialInstanceId = u64;

/// Unique identifier for a texture slot.
pub type TextureSlotId = u32;

/// Supported material parameter types.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    UInt(u32),
    Bool(bool),
    Mat3([[f32; 3]; 3]),
    Mat4([[f32; 4]; 4]),
    Texture(TextureSlotId),
    Color([f32; 4]),
}

impl MaterialParamValue {
    /// Size of this parameter in bytes when packed into a uniform buffer.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float(_) => 4,
            Self::Vec2(_) => 8,
            Self::Vec3(_) => 12,
            Self::Vec4(_) | Self::Color(_) => 16,
            Self::Int(_) | Self::UInt(_) => 4,
            Self::Bool(_) => 4,
            Self::Mat3(_) => 48, // 3x vec4 with padding.
            Self::Mat4(_) => 64,
            Self::Texture(_) => 4,
        }
    }

    /// Write this parameter into a byte buffer at the given offset.
    pub fn write_to_buffer(&self, buffer: &mut [u8], offset: usize) {
        match self {
            Self::Float(v) => {
                let bytes = v.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::Vec2(v) => {
                for (i, val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Vec3(v) => {
                for (i, val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Vec4(v) | Self::Color(v) => {
                for (i, val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Int(v) => {
                let bytes = v.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::UInt(v) => {
                let bytes = v.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::Bool(v) => {
                let val: u32 = if *v { 1 } else { 0 };
                let bytes = val.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::Mat3(m) => {
                // GPU mat3 is typically stored as 3 vec4 rows (with padding).
                for row in 0..3 {
                    for col in 0..3 {
                        let bytes = m[row][col].to_le_bytes();
                        let pos = offset + row * 16 + col * 4;
                        buffer[pos..pos + 4].copy_from_slice(&bytes);
                    }
                    // Pad the 4th component to 0.
                    let pad_pos = offset + row * 16 + 12;
                    buffer[pad_pos..pad_pos + 4].copy_from_slice(&0.0f32.to_le_bytes());
                }
            }
            Self::Mat4(m) => {
                for row in 0..4 {
                    for col in 0..4 {
                        let bytes = m[row][col].to_le_bytes();
                        let pos = offset + row * 16 + col * 4;
                        buffer[pos..pos + 4].copy_from_slice(&bytes);
                    }
                }
            }
            Self::Texture(id) => {
                let bytes = id.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
        }
    }

    /// Linearly interpolate between two parameter values.
    pub fn lerp(&self, other: &Self, t: f32) -> Option<Self> {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => Some(Self::Float(a + (b - a) * t)),
            (Self::Vec2(a), Self::Vec2(b)) => {
                Some(Self::Vec2([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                ]))
            }
            (Self::Vec3(a), Self::Vec3(b)) => {
                Some(Self::Vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ]))
            }
            (Self::Vec4(a), Self::Vec4(b)) | (Self::Color(a), Self::Color(b)) => {
                Some(Self::Vec4([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ]))
            }
            (Self::Int(a), Self::Int(b)) => {
                Some(Self::Int((*a as f32 + (*b - *a) as f32 * t) as i32))
            }
            _ => None, // Cannot interpolate textures, bools, matrices.
        }
    }
}

/// A named material parameter descriptor.
#[derive(Debug, Clone)]
pub struct MaterialParamDescriptor {
    /// Parameter name (e.g. "baseColor", "roughness").
    pub name: String,
    /// Byte offset in the uniform buffer.
    pub offset: usize,
    /// Default value.
    pub default_value: MaterialParamValue,
    /// Whether this parameter can be animated.
    pub animatable: bool,
    /// Optional min/max range for float/vec parameters.
    pub range: Option<(f32, f32)>,
    /// Display name for editor UI.
    pub display_name: Option<String>,
    /// Group/category for editor UI.
    pub group: Option<String>,
}

// ---------------------------------------------------------------------------
// Material template
// ---------------------------------------------------------------------------

/// A material template defines the shader, pipeline state, and available
/// parameters. Instances are created from templates.
#[derive(Debug, Clone)]
pub struct MaterialTemplate {
    pub id: MaterialTemplateId,
    pub name: String,
    /// Shader identifier for the vertex stage.
    pub vertex_shader: String,
    /// Shader identifier for the fragment stage.
    pub fragment_shader: String,
    /// Parameter descriptors in uniform buffer order.
    pub parameters: Vec<MaterialParamDescriptor>,
    /// Total size of the uniform buffer in bytes.
    pub uniform_buffer_size: usize,
    /// Render state.
    pub render_state: MaterialRenderState,
    /// LOD variants (distance thresholds and simplified shader names).
    pub lod_variants: Vec<MaterialLodVariant>,
    /// Whether this material supports instancing.
    pub supports_instancing: bool,
    /// Tags for material system queries.
    pub tags: Vec<String>,
    /// Parameter name -> index lookup.
    param_index: HashMap<String, usize>,
}

/// Render state for a material.
#[derive(Debug, Clone)]
pub struct MaterialRenderState {
    pub blend_enabled: bool,
    pub blend_src: BlendFactor,
    pub blend_dst: BlendFactor,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: CullMode,
    pub alpha_test: bool,
    pub alpha_threshold: f32,
    pub wireframe: bool,
    pub two_sided: bool,
}

impl Default for MaterialRenderState {
    fn default() -> Self {
        Self {
            blend_enabled: false,
            blend_src: BlendFactor::One,
            blend_dst: BlendFactor::Zero,
            depth_write: true,
            depth_test: true,
            cull_mode: CullMode::Back,
            alpha_test: false,
            alpha_threshold: 0.5,
            wireframe: false,
            two_sided: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// A LOD variant of a material (simplified shader at distance).
#[derive(Debug, Clone)]
pub struct MaterialLodVariant {
    /// Distance threshold beyond which this LOD is used.
    pub distance_threshold: f32,
    /// Simplified vertex shader (or None to use base).
    pub vertex_shader: Option<String>,
    /// Simplified fragment shader.
    pub fragment_shader: Option<String>,
    /// Parameters to drop at this LOD (set to default).
    pub dropped_parameters: Vec<String>,
    /// LOD level index (0 = highest quality).
    pub lod_level: u8,
}

impl MaterialTemplate {
    pub fn new(id: MaterialTemplateId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            vertex_shader: String::new(),
            fragment_shader: String::new(),
            parameters: Vec::new(),
            uniform_buffer_size: 0,
            render_state: MaterialRenderState::default(),
            lod_variants: Vec::new(),
            supports_instancing: false,
            tags: Vec::new(),
            param_index: HashMap::new(),
        }
    }

    /// Add a parameter to the template.
    pub fn add_parameter(&mut self, desc: MaterialParamDescriptor) {
        let idx = self.parameters.len();
        self.param_index.insert(desc.name.clone(), idx);
        self.parameters.push(desc);
        self.recalculate_buffer_size();
    }

    /// Get a parameter descriptor by name.
    pub fn get_parameter(&self, name: &str) -> Option<&MaterialParamDescriptor> {
        self.param_index.get(name).map(|&i| &self.parameters[i])
    }

    /// Create an instance from this template.
    pub fn create_instance(&self, instance_id: MaterialInstanceId) -> MaterialInstance {
        // Populate default buffer.
        let mut buffer = vec![0u8; self.uniform_buffer_size];
        for param in &self.parameters {
            param.default_value.write_to_buffer(&mut buffer, param.offset);
        }

        MaterialInstance {
            id: instance_id,
            template_id: self.id,
            overrides: HashMap::new(),
            uniform_buffer: buffer,
            dirty: true,
            dirty_params: Vec::new(),
            animations: Vec::new(),
            active_lod: 0,
            render_state_override: None,
            enabled: true,
            name: format!("{}_{}", self.name, instance_id),
        }
    }

    /// Recalculate the uniform buffer size from parameter offsets and sizes.
    fn recalculate_buffer_size(&mut self) {
        let mut max_end = 0usize;
        for param in &self.parameters {
            let end = param.offset + param.default_value.byte_size();
            max_end = max_end.max(end);
        }
        // Align to 256 bytes (typical GPU UBO alignment).
        self.uniform_buffer_size = (max_end + 255) & !255;
    }

    /// Select the appropriate LOD level for a given distance.
    pub fn select_lod(&self, distance: f32) -> u8 {
        let mut best_lod = 0u8;
        for variant in &self.lod_variants {
            if distance >= variant.distance_threshold {
                best_lod = best_lod.max(variant.lod_level);
            }
        }
        best_lod
    }
}

// ---------------------------------------------------------------------------
// Material instance
// ---------------------------------------------------------------------------

/// A runtime material instance with per-instance parameter overrides and a
/// GPU uniform buffer.
#[derive(Debug, Clone)]
pub struct MaterialInstance {
    /// Unique instance ID.
    pub id: MaterialInstanceId,
    /// Template this instance was created from.
    pub template_id: MaterialTemplateId,
    /// Parameter overrides (name -> value).
    pub overrides: HashMap<String, MaterialParamValue>,
    /// The raw uniform buffer data.
    pub uniform_buffer: Vec<u8>,
    /// Whether the uniform buffer needs uploading.
    pub dirty: bool,
    /// List of parameter names that changed since last upload.
    pub dirty_params: Vec<String>,
    /// Active animations on this instance.
    pub animations: Vec<MaterialAnimation>,
    /// Current LOD level.
    pub active_lod: u8,
    /// Optional render state override.
    pub render_state_override: Option<MaterialRenderState>,
    /// Whether this instance is enabled (visible).
    pub enabled: bool,
    /// Debug name.
    pub name: String,
}

impl MaterialInstance {
    /// Set a parameter value, marking it dirty.
    pub fn set_param(&mut self, name: &str, value: MaterialParamValue) {
        self.overrides.insert(name.to_string(), value);
        if !self.dirty_params.contains(&name.to_string()) {
            self.dirty_params.push(name.to_string());
        }
        self.dirty = true;
    }

    /// Get a parameter value (override or default from template).
    pub fn get_param(&self, name: &str) -> Option<&MaterialParamValue> {
        self.overrides.get(name)
    }

    /// Set a float parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.set_param(name, MaterialParamValue::Float(value));
    }

    /// Set a color parameter.
    pub fn set_color(&mut self, name: &str, r: f32, g: f32, b: f32, a: f32) {
        self.set_param(name, MaterialParamValue::Color([r, g, b, a]));
    }

    /// Set a vec3 parameter.
    pub fn set_vec3(&mut self, name: &str, x: f32, y: f32, z: f32) {
        self.set_param(name, MaterialParamValue::Vec3([x, y, z]));
    }

    /// Set a vec4 parameter.
    pub fn set_vec4(&mut self, name: &str, x: f32, y: f32, z: f32, w: f32) {
        self.set_param(name, MaterialParamValue::Vec4([x, y, z, w]));
    }

    /// Set a texture parameter.
    pub fn set_texture(&mut self, name: &str, texture_id: TextureSlotId) {
        self.set_param(name, MaterialParamValue::Texture(texture_id));
    }

    /// Set a boolean parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.set_param(name, MaterialParamValue::Bool(value));
    }

    /// Reset a parameter to its template default.
    pub fn reset_param(&mut self, name: &str) {
        self.overrides.remove(name);
        if !self.dirty_params.contains(&name.to_string()) {
            self.dirty_params.push(name.to_string());
        }
        self.dirty = true;
    }

    /// Reset all parameters to template defaults.
    pub fn reset_all(&mut self) {
        let names: Vec<String> = self.overrides.keys().cloned().collect();
        self.overrides.clear();
        self.dirty_params = names;
        self.dirty = true;
    }

    /// Update the uniform buffer with current parameter values.
    pub fn update_buffer(&mut self, template: &MaterialTemplate) {
        if !self.dirty { return; }

        for param_name in &self.dirty_params {
            if let Some(desc) = template.get_parameter(param_name) {
                let value = self.overrides.get(param_name).unwrap_or(&desc.default_value);
                value.write_to_buffer(&mut self.uniform_buffer, desc.offset);
            }
        }
        self.dirty_params.clear();
        self.dirty = false;
    }

    /// Rebuild the entire uniform buffer from scratch.
    pub fn rebuild_buffer(&mut self, template: &MaterialTemplate) {
        self.uniform_buffer = vec![0u8; template.uniform_buffer_size];
        for param in &template.parameters {
            let value = self.overrides.get(&param.name).unwrap_or(&param.default_value);
            value.write_to_buffer(&mut self.uniform_buffer, param.offset);
        }
        self.dirty_params.clear();
        self.dirty = false;
    }

    /// Add an animation to this instance.
    pub fn add_animation(&mut self, anim: MaterialAnimation) {
        self.animations.push(anim);
    }

    /// Remove all animations for a specific parameter.
    pub fn remove_animation(&mut self, param_name: &str) {
        self.animations.retain(|a| a.parameter_name != param_name);
    }

    /// Update all active animations.
    pub fn update_animations(&mut self, dt: f32) {
        let mut completed = Vec::new();
        let animations = std::mem::take(&mut self.animations);

        for (i, anim) in animations.iter().enumerate() {
            let mut anim = anim.clone();
            anim.elapsed += dt;

            let t = if anim.duration > 0.0 {
                (anim.elapsed / anim.duration).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let eased_t = anim.easing.apply(t);

            if let Some(interpolated) = anim.from.lerp(&anim.to, eased_t) {
                self.set_param(&anim.parameter_name, interpolated);
            }

            if anim.elapsed >= anim.duration {
                if anim.looping {
                    // Reset for looping.
                    let mut new_anim = anim.clone();
                    new_anim.elapsed = 0.0;
                    self.animations.push(new_anim);
                } else {
                    completed.push(i);
                    // Set final value.
                    self.set_param(&anim.parameter_name, anim.to.clone());
                }
            } else {
                self.animations.push(anim);
            }
        }
    }

    /// Update LOD level based on distance from camera.
    pub fn update_lod(&mut self, template: &MaterialTemplate, distance: f32) {
        let new_lod = template.select_lod(distance);
        if new_lod != self.active_lod {
            self.active_lod = new_lod;
        }
    }
}

// ---------------------------------------------------------------------------
// Material animation
// ---------------------------------------------------------------------------

/// Easing function for material parameter animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    Spring,
    Bounce,
}

impl EasingFunction {
    pub fn apply(&self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => {
                if t < 0.5 { 2.0 * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(2) / 2.0 }
            }
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => t * (2.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 { 2.0 * t * t } else { -1.0 + (4.0 - 2.0 * t) * t }
            }
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            Self::EaseInOutCubic => {
                if t < 0.5 { 4.0 * t * t * t } else {
                    let t1 = -2.0 * t + 2.0;
                    1.0 - t1 * t1 * t1 / 2.0
                }
            }
            Self::Spring => {
                let freq = 4.5;
                let decay = 5.0;
                1.0 - (1.0 - t) * ((-decay * t).exp() * (freq * std::f32::consts::TAU * t).cos())
            }
            Self::Bounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                let mut t = t;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    t -= 1.5 / d1;
                    n1 * t * t + 0.75
                } else if t < 2.5 / d1 {
                    t -= 2.25 / d1;
                    n1 * t * t + 0.9375
                } else {
                    t -= 2.625 / d1;
                    n1 * t * t + 0.984375
                }
            }
        }
    }
}

/// An animation that transitions a material parameter over time.
#[derive(Debug, Clone)]
pub struct MaterialAnimation {
    /// Name of the parameter to animate.
    pub parameter_name: String,
    /// Starting value.
    pub from: MaterialParamValue,
    /// Target value.
    pub to: MaterialParamValue,
    /// Duration in seconds.
    pub duration: f32,
    /// Time elapsed so far.
    pub elapsed: f32,
    /// Easing function.
    pub easing: EasingFunction,
    /// Whether to loop the animation.
    pub looping: bool,
    /// Delay before starting.
    pub delay: f32,
}

impl MaterialAnimation {
    pub fn new(name: &str, from: MaterialParamValue, to: MaterialParamValue, duration: f32) -> Self {
        Self {
            parameter_name: name.to_string(),
            from,
            to,
            duration,
            elapsed: 0.0,
            easing: EasingFunction::Linear,
            looping: false,
            delay: 0.0,
        }
    }

    pub fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    pub fn with_loop(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    pub fn with_delay(mut self, delay: f32) -> Self {
        self.delay = delay;
        self
    }

    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration && !self.looping
    }

    pub fn progress(&self) -> f32 {
        if self.duration > 0.0 { (self.elapsed / self.duration).clamp(0.0, 1.0) } else { 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Material instance manager
// ---------------------------------------------------------------------------

/// Manages all material instances and their templates.
pub struct MaterialInstanceManager {
    /// Templates by ID.
    templates: HashMap<MaterialTemplateId, MaterialTemplate>,
    /// Instances by ID.
    instances: HashMap<MaterialInstanceId, MaterialInstance>,
    /// Next instance ID counter.
    next_instance_id: MaterialInstanceId,
    /// Batch update list: instances that need buffer uploads.
    dirty_instances: Vec<MaterialInstanceId>,
    /// Statistics.
    pub stats: MaterialInstanceStats,
}

/// Statistics for the material instance system.
#[derive(Debug, Clone, Default)]
pub struct MaterialInstanceStats {
    pub total_templates: u32,
    pub total_instances: u32,
    pub dirty_instances: u32,
    pub total_uniform_buffer_bytes: u64,
    pub buffer_uploads_this_frame: u32,
    pub active_animations: u32,
}

impl MaterialInstanceManager {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            instances: HashMap::new(),
            next_instance_id: 1,
            dirty_instances: Vec::new(),
            stats: MaterialInstanceStats::default(),
        }
    }

    /// Register a material template.
    pub fn register_template(&mut self, template: MaterialTemplate) {
        self.templates.insert(template.id, template);
    }

    /// Get a template by ID.
    pub fn get_template(&self, id: MaterialTemplateId) -> Option<&MaterialTemplate> {
        self.templates.get(&id)
    }

    /// Create an instance from a template.
    pub fn create_instance(&mut self, template_id: MaterialTemplateId) -> Option<MaterialInstanceId> {
        let template = self.templates.get(&template_id)?;
        let instance_id = self.next_instance_id;
        self.next_instance_id += 1;
        let instance = template.create_instance(instance_id);
        self.instances.insert(instance_id, instance);
        Some(instance_id)
    }

    /// Get an instance by ID.
    pub fn get_instance(&self, id: MaterialInstanceId) -> Option<&MaterialInstance> {
        self.instances.get(&id)
    }

    /// Get a mutable instance by ID.
    pub fn get_instance_mut(&mut self, id: MaterialInstanceId) -> Option<&mut MaterialInstance> {
        self.instances.get_mut(&id)
    }

    /// Destroy an instance.
    pub fn destroy_instance(&mut self, id: MaterialInstanceId) -> bool {
        self.instances.remove(&id).is_some()
    }

    /// Update all dirty instances (rebuild their uniform buffers).
    pub fn update_dirty(&mut self) {
        self.dirty_instances.clear();
        let template_refs: HashMap<MaterialTemplateId, &MaterialTemplate> =
            self.templates.iter().map(|(k, v)| (*k, v)).collect();

        let mut uploads = 0u32;
        for (id, instance) in &mut self.instances {
            if instance.dirty {
                if let Some(template) = template_refs.get(&instance.template_id) {
                    instance.update_buffer(template);
                    self.dirty_instances.push(*id);
                    uploads += 1;
                }
            }
        }
        self.stats.buffer_uploads_this_frame = uploads;
    }

    /// Update all material animations.
    pub fn update_animations(&mut self, dt: f32) {
        let mut total_animations = 0u32;
        for instance in self.instances.values_mut() {
            if !instance.animations.is_empty() {
                instance.update_animations(dt);
                total_animations += instance.animations.len() as u32;
            }
        }
        self.stats.active_animations = total_animations;
    }

    /// Update LOD levels for all instances.
    pub fn update_lod(&mut self, camera_position: [f32; 3]) {
        let template_refs: HashMap<MaterialTemplateId, &MaterialTemplate> =
            self.templates.iter().map(|(k, v)| (*k, v)).collect();

        for instance in self.instances.values_mut() {
            if let Some(template) = template_refs.get(&instance.template_id) {
                // Distance approximation using the instance name (in practice,
                // the position would come from the entity's transform).
                // For now, just use the current LOD.
                let _ = template; // Template available for LOD queries.
            }
        }
    }

    /// Full frame update: animations, dirty buffers, statistics.
    pub fn frame_update(&mut self, dt: f32) {
        self.update_animations(dt);
        self.update_dirty();

        // Update statistics.
        let mut total_buffer_bytes = 0u64;
        for instance in self.instances.values() {
            total_buffer_bytes += instance.uniform_buffer.len() as u64;
        }
        self.stats.total_templates = self.templates.len() as u32;
        self.stats.total_instances = self.instances.len() as u32;
        self.stats.dirty_instances = self.dirty_instances.len() as u32;
        self.stats.total_uniform_buffer_bytes = total_buffer_bytes;
    }

    /// Get all instances for a specific template.
    pub fn instances_for_template(&self, template_id: MaterialTemplateId) -> Vec<MaterialInstanceId> {
        self.instances.iter()
            .filter(|(_, inst)| inst.template_id == template_id)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Batch set a parameter on all instances of a template.
    pub fn batch_set_param(
        &mut self,
        template_id: MaterialTemplateId,
        param_name: &str,
        value: MaterialParamValue,
    ) {
        let ids: Vec<MaterialInstanceId> = self.instances.iter()
            .filter(|(_, inst)| inst.template_id == template_id)
            .map(|(id, _)| *id)
            .collect();

        for id in ids {
            if let Some(inst) = self.instances.get_mut(&id) {
                inst.set_param(param_name, value.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Common material templates
// ---------------------------------------------------------------------------

/// Create a standard PBR material template.
pub fn create_pbr_template(id: MaterialTemplateId) -> MaterialTemplate {
    let mut template = MaterialTemplate::new(id, "PBR_Standard");
    template.vertex_shader = "shaders/pbr_vert".to_string();
    template.fragment_shader = "shaders/pbr_frag".to_string();

    template.add_parameter(MaterialParamDescriptor {
        name: "baseColor".to_string(),
        offset: 0,
        default_value: MaterialParamValue::Color([1.0, 1.0, 1.0, 1.0]),
        animatable: true,
        range: Some((0.0, 1.0)),
        display_name: Some("Base Color".to_string()),
        group: Some("Surface".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "metallic".to_string(),
        offset: 16,
        default_value: MaterialParamValue::Float(0.0),
        animatable: true,
        range: Some((0.0, 1.0)),
        display_name: Some("Metallic".to_string()),
        group: Some("Surface".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "roughness".to_string(),
        offset: 20,
        default_value: MaterialParamValue::Float(0.5),
        animatable: true,
        range: Some((0.0, 1.0)),
        display_name: Some("Roughness".to_string()),
        group: Some("Surface".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "ao".to_string(),
        offset: 24,
        default_value: MaterialParamValue::Float(1.0),
        animatable: false,
        range: Some((0.0, 1.0)),
        display_name: Some("Ambient Occlusion".to_string()),
        group: Some("Surface".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "emissiveColor".to_string(),
        offset: 32,
        default_value: MaterialParamValue::Vec3([0.0, 0.0, 0.0]),
        animatable: true,
        range: None,
        display_name: Some("Emissive Color".to_string()),
        group: Some("Emission".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "emissiveIntensity".to_string(),
        offset: 44,
        default_value: MaterialParamValue::Float(0.0),
        animatable: true,
        range: Some((0.0, 100.0)),
        display_name: Some("Emissive Intensity".to_string()),
        group: Some("Emission".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "normalStrength".to_string(),
        offset: 48,
        default_value: MaterialParamValue::Float(1.0),
        animatable: true,
        range: Some((0.0, 2.0)),
        display_name: Some("Normal Strength".to_string()),
        group: Some("Detail".to_string()),
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "uvScale".to_string(),
        offset: 52,
        default_value: MaterialParamValue::Vec2([1.0, 1.0]),
        animatable: true,
        range: None,
        display_name: Some("UV Scale".to_string()),
        group: Some("Detail".to_string()),
    });

    // Add LOD variants.
    template.lod_variants.push(MaterialLodVariant {
        distance_threshold: 50.0,
        vertex_shader: None,
        fragment_shader: Some("shaders/pbr_frag_lod1".to_string()),
        dropped_parameters: vec!["normalStrength".to_string()],
        lod_level: 1,
    });
    template.lod_variants.push(MaterialLodVariant {
        distance_threshold: 100.0,
        vertex_shader: None,
        fragment_shader: Some("shaders/pbr_frag_lod2".to_string()),
        dropped_parameters: vec!["normalStrength".to_string(), "emissiveColor".to_string()],
        lod_level: 2,
    });

    template
}

/// Create a simple unlit material template.
pub fn create_unlit_template(id: MaterialTemplateId) -> MaterialTemplate {
    let mut template = MaterialTemplate::new(id, "Unlit");
    template.vertex_shader = "shaders/unlit_vert".to_string();
    template.fragment_shader = "shaders/unlit_frag".to_string();

    template.add_parameter(MaterialParamDescriptor {
        name: "color".to_string(),
        offset: 0,
        default_value: MaterialParamValue::Color([1.0, 1.0, 1.0, 1.0]),
        animatable: true,
        range: Some((0.0, 1.0)),
        display_name: Some("Color".to_string()),
        group: None,
    });
    template.add_parameter(MaterialParamDescriptor {
        name: "opacity".to_string(),
        offset: 16,
        default_value: MaterialParamValue::Float(1.0),
        animatable: true,
        range: Some((0.0, 1.0)),
        display_name: Some("Opacity".to_string()),
        group: None,
    });

    template
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_write_read_float() {
        let mut buffer = vec![0u8; 64];
        let val = MaterialParamValue::Float(3.14);
        val.write_to_buffer(&mut buffer, 0);
        let read_val = f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        assert!((read_val - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_param_lerp() {
        let a = MaterialParamValue::Float(0.0);
        let b = MaterialParamValue::Float(10.0);
        let result = a.lerp(&b, 0.5).unwrap();
        match result {
            MaterialParamValue::Float(v) => assert!((v - 5.0).abs() < 1e-6),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_create_instance() {
        let template = create_pbr_template(1);
        let instance = template.create_instance(42);
        assert_eq!(instance.id, 42);
        assert_eq!(instance.template_id, 1);
        assert!(instance.dirty);
    }

    #[test]
    fn test_instance_set_param() {
        let template = create_pbr_template(1);
        let mut instance = template.create_instance(1);
        instance.set_float("roughness", 0.8);
        assert!(instance.dirty);
        instance.update_buffer(&template);
        assert!(!instance.dirty);
    }

    #[test]
    fn test_lod_selection() {
        let template = create_pbr_template(1);
        assert_eq!(template.select_lod(10.0), 0); // Close = LOD 0.
        assert_eq!(template.select_lod(60.0), 1); // > 50 = LOD 1.
        assert_eq!(template.select_lod(150.0), 2); // > 100 = LOD 2.
    }

    #[test]
    fn test_easing_functions() {
        assert!((EasingFunction::Linear.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((EasingFunction::EaseIn.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((EasingFunction::EaseIn.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((EasingFunction::EaseOut.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((EasingFunction::EaseOut.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manager_create_destroy() {
        let mut mgr = MaterialInstanceManager::new();
        mgr.register_template(create_pbr_template(1));
        let id = mgr.create_instance(1).unwrap();
        assert!(mgr.get_instance(id).is_some());
        mgr.destroy_instance(id);
        assert!(mgr.get_instance(id).is_none());
    }
}
