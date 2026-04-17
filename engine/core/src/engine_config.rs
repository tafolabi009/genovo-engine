// engine/core/src/engine_config.rs
// Engine configuration: rendering, physics, audio, network settings, quality presets, per-platform defaults.
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub rendering: RenderConfig,
    pub physics: PhysicsConfig,
    pub audio: AudioConfig,
    pub network: NetworkConfig,
    pub platform: PlatformConfig,
    pub debug: DebugConfig,
    pub custom: HashMap<String, ConfigValue>,
}

#[derive(Debug, Clone)]
pub enum ConfigValue { Bool(bool), Int(i64), Float(f64), String(String) }

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub max_fps: u32, pub vsync: bool, pub render_scale: f32,
    pub shadow_map_size: u32, pub max_shadow_cascades: u32,
    pub max_point_lights: u32, pub max_spot_lights: u32,
    pub texture_budget_mb: u32, pub mesh_budget_mb: u32,
    pub enable_hdr: bool, pub tonemap_operator: String,
    pub max_draw_calls: u32, pub enable_instancing: bool,
    pub enable_occlusion_culling: bool, pub frustum_culling: bool,
    pub lod_bias: f32, pub anisotropic_level: u32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            max_fps: 0, vsync: true, render_scale: 1.0,
            shadow_map_size: 2048, max_shadow_cascades: 4,
            max_point_lights: 128, max_spot_lights: 64,
            texture_budget_mb: 512, mesh_budget_mb: 256,
            enable_hdr: true, tonemap_operator: "ACES".into(),
            max_draw_calls: 10000, enable_instancing: true,
            enable_occlusion_culling: true, frustum_culling: true,
            lod_bias: 0.0, anisotropic_level: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    pub fixed_timestep: f32, pub max_substeps: u32,
    pub gravity: [f32; 3], pub solver_iterations: u32,
    pub position_iterations: u32, pub enable_ccd: bool,
    pub ccd_threshold: f32, pub max_bodies: u32,
    pub broadphase_type: String, pub sleep_threshold: f32,
    pub sleep_time: f32, pub max_contacts_per_pair: u32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0/60.0, max_substeps: 4,
            gravity: [0.0, -9.81, 0.0], solver_iterations: 8,
            position_iterations: 3, enable_ccd: true,
            ccd_threshold: 1.0, max_bodies: 10000,
            broadphase_type: "SAP".into(), sleep_threshold: 0.05,
            sleep_time: 2.0, max_contacts_per_pair: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32, pub buffer_size: u32,
    pub max_voices: u32, pub max_virtual_voices: u32,
    pub doppler_scale: f32, pub distance_model: String,
    pub rolloff_factor: f32, pub max_distance: f32,
    pub enable_reverb: bool, pub enable_occlusion: bool,
    pub hrtf_enabled: bool, pub stream_buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000, buffer_size: 1024,
            max_voices: 64, max_virtual_voices: 256,
            doppler_scale: 1.0, distance_model: "InverseSquare".into(),
            rolloff_factor: 1.0, max_distance: 100.0,
            enable_reverb: true, enable_occlusion: true,
            hrtf_enabled: false, stream_buffer_size: 65536,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub tick_rate: u32, pub max_clients: u32,
    pub timeout_seconds: f32, pub max_packet_size: u32,
    pub enable_compression: bool, pub enable_encryption: bool,
    pub interpolation_delay: f32, pub snapshot_rate: u32,
    pub bandwidth_limit: u32, pub reliable_window_size: u32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            tick_rate: 60, max_clients: 64, timeout_seconds: 10.0,
            max_packet_size: 1400, enable_compression: true,
            enable_encryption: true, interpolation_delay: 0.1,
            snapshot_rate: 20, bandwidth_limit: 0, reliable_window_size: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlatformConfig { pub target: PlatformTarget, pub thread_count: u32, pub memory_budget_mb: u32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformTarget { Windows, Linux, MacOS, iOS, Android, WebAssembly, PlayStation, Xbox, Switch }
impl Default for PlatformConfig { fn default() -> Self { Self { target: PlatformTarget::Windows, thread_count: 0, memory_budget_mb: 2048 } } }

#[derive(Debug, Clone)]
pub struct DebugConfig { pub enable_profiler: bool, pub enable_debug_draw: bool, pub enable_console: bool, pub log_level: String, pub show_fps: bool, pub show_stats: bool }
impl Default for DebugConfig { fn default() -> Self { Self { enable_profiler: false, enable_debug_draw: false, enable_console: true, log_level: "Info".into(), show_fps: false, show_stats: false } } }

impl Default for EngineConfig {
    fn default() -> Self {
        Self { rendering: RenderConfig::default(), physics: PhysicsConfig::default(), audio: AudioConfig::default(), network: NetworkConfig::default(), platform: PlatformConfig::default(), debug: DebugConfig::default(), custom: HashMap::new() }
    }
}

impl EngineConfig {
    pub fn for_platform(platform: PlatformTarget) -> Self {
        let mut config = Self::default();
        config.platform.target = platform;
        match platform {
            PlatformTarget::Android | PlatformTarget::iOS => {
                config.rendering.shadow_map_size = 1024; config.rendering.max_point_lights = 32;
                config.rendering.texture_budget_mb = 128; config.physics.max_substeps = 2;
                config.audio.max_voices = 32; config.platform.memory_budget_mb = 512;
            }
            PlatformTarget::WebAssembly => {
                config.rendering.shadow_map_size = 512; config.rendering.max_point_lights = 16;
                config.rendering.enable_occlusion_culling = false; config.physics.solver_iterations = 4;
                config.audio.max_voices = 16; config.platform.memory_budget_mb = 256;
            }
            PlatformTarget::Switch => {
                config.rendering.shadow_map_size = 1024; config.rendering.max_point_lights = 48;
                config.rendering.texture_budget_mb = 256; config.platform.memory_budget_mb = 1024;
            }
            _ => {}
        }
        config
    }
    pub fn set(&mut self, key: &str, value: ConfigValue) { self.custom.insert(key.to_string(), value); }
    pub fn get(&self, key: &str) -> Option<&ConfigValue> { self.custom.get(key) }
    pub fn get_bool(&self, key: &str) -> Option<bool> { match self.custom.get(key) { Some(ConfigValue::Bool(b)) => Some(*b), _ => None } }
    pub fn get_float(&self, key: &str) -> Option<f64> { match self.custom.get(key) { Some(ConfigValue::Float(f)) => Some(*f), _ => None } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_default_config() { let c = EngineConfig::default(); assert_eq!(c.physics.solver_iterations, 8); }
    #[test]
    fn test_mobile_config() { let c = EngineConfig::for_platform(PlatformTarget::Android); assert!(c.rendering.shadow_map_size < 2048); }
    #[test]
    fn test_custom_values() { let mut c = EngineConfig::default(); c.set("debug_mode", ConfigValue::Bool(true)); assert_eq!(c.get_bool("debug_mode"), Some(true)); }
}
