// engine/gameplay/src/settings_system.rs
// Game settings: graphics quality, audio, controls, accessibility, language.
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityPreset { VeryLow, Low, Medium, High, Ultra, Custom }

#[derive(Debug, Clone)]
pub struct GraphicsSettings {
    pub preset: QualityPreset, pub resolution: (u32, u32), pub fullscreen: bool,
    pub vsync: bool, pub render_scale: f32, pub texture_quality: u32,
    pub shadow_quality: u32, pub shadow_distance: f32,
    pub anti_aliasing: AAMode, pub ambient_occlusion: bool,
    pub bloom: bool, pub motion_blur: bool, pub volumetric_fog: bool,
    pub view_distance: f32, pub foliage_density: f32,
    pub gamma: f32, pub brightness: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AAMode { None, FXAA, TAA, MSAA2x, MSAA4x }

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self { preset: QualityPreset::High, resolution: (1920, 1080), fullscreen: false,
            vsync: true, render_scale: 1.0, texture_quality: 2, shadow_quality: 2,
            shadow_distance: 100.0, anti_aliasing: AAMode::TAA, ambient_occlusion: true,
            bloom: true, motion_blur: false, volumetric_fog: true, view_distance: 1000.0,
            foliage_density: 1.0, gamma: 2.2, brightness: 1.0 }
    }
}

impl GraphicsSettings {
    pub fn apply_preset(&mut self, preset: QualityPreset) {
        self.preset = preset;
        match preset {
            QualityPreset::VeryLow => { self.render_scale=0.5; self.texture_quality=0; self.shadow_quality=0;
                self.shadow_distance=30.0; self.anti_aliasing=AAMode::None; self.ambient_occlusion=false;
                self.bloom=false; self.volumetric_fog=false; self.view_distance=300.0; self.foliage_density=0.3; }
            QualityPreset::Low => { self.render_scale=0.75; self.texture_quality=1; self.shadow_quality=1;
                self.anti_aliasing=AAMode::FXAA; self.ambient_occlusion=false; self.view_distance=500.0; }
            QualityPreset::Medium => { *self = Self::default(); self.preset = QualityPreset::Medium; }
            QualityPreset::High => { *self = Self::default(); }
            QualityPreset::Ultra => { self.render_scale=1.0; self.texture_quality=3; self.shadow_quality=3;
                self.shadow_distance=200.0; self.motion_blur=true; self.view_distance=2000.0; }
            QualityPreset::Custom => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioSettings {
    pub master_volume: f32, pub music_volume: f32, pub sfx_volume: f32,
    pub voice_volume: f32, pub ambient_volume: f32, pub ui_volume: f32,
    pub mute_when_unfocused: bool, pub spatial_audio: bool,
    pub subtitles: bool, pub subtitle_size: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { master_volume: 0.8, music_volume: 0.6, sfx_volume: 0.8,
            voice_volume: 1.0, ambient_volume: 0.5, ui_volume: 0.7,
            mute_when_unfocused: true, spatial_audio: true,
            subtitles: false, subtitle_size: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct ControlSettings {
    pub mouse_sensitivity: f32, pub mouse_invert_y: bool,
    pub gamepad_sensitivity: f32, pub gamepad_invert_y: bool, pub gamepad_deadzone: f32,
    pub vibration: bool, pub toggle_crouch: bool, pub toggle_sprint: bool,
}

impl Default for ControlSettings {
    fn default() -> Self {
        Self { mouse_sensitivity: 1.0, mouse_invert_y: false,
            gamepad_sensitivity: 1.0, gamepad_invert_y: false, gamepad_deadzone: 0.15,
            vibration: true, toggle_crouch: false, toggle_sprint: true }
    }
}

#[derive(Debug, Clone)]
pub struct GameSettings {
    pub graphics: GraphicsSettings, pub audio: AudioSettings,
    pub controls: ControlSettings, pub language: String, pub version: u32,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self { graphics: GraphicsSettings::default(), audio: AudioSettings::default(),
            controls: ControlSettings::default(), language: "en".into(), version: 1 }
    }
}

impl GameSettings {
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("version={}\n", self.version));
        out.push_str(&format!("language={}\n", self.language));
        out.push_str(&format!("resolution={}x{}\n", self.graphics.resolution.0, self.graphics.resolution.1));
        out.push_str(&format!("master_volume={}\n", self.audio.master_volume));
        out.push_str(&format!("gamma={}\n", self.graphics.gamma));
        out
    }
    pub fn deserialize(data: &str) -> Self {
        let mut s = Self::default();
        for line in data.lines() {
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 { continue; }
            match parts[0].trim() {
                "language" => s.language = parts[1].trim().to_string(),
                "master_volume" => if let Ok(v) = parts[1].trim().parse() { s.audio.master_volume = v; },
                "gamma" => if let Ok(v) = parts[1].trim().parse() { s.graphics.gamma = v; },
                _ => {}
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_defaults() { let s = GameSettings::default(); assert_eq!(s.language, "en"); }
    #[test] fn test_serialize() { let s = GameSettings::default(); let d = s.serialize(); let s2 = GameSettings::deserialize(&d); assert_eq!(s2.language, s.language); }
    #[test] fn test_preset() { let mut g = GraphicsSettings::default(); g.apply_preset(QualityPreset::VeryLow); assert_eq!(g.shadow_quality, 0); }
}
