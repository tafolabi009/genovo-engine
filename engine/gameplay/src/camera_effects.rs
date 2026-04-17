// engine/gameplay/src/camera_effects.rs
//
// Camera gameplay effects for the Genovo engine.
//
// Provides gameplay-driven camera effects that enhance game feel:
//
// - **Screen shake** -- Camera shake from damage, explosions, impacts.
// - **Speed lines** -- Radial effect at high velocity.
// - **Scope zoom** -- Sniper/telescope zoom with scope overlay.
// - **Vision overlays** -- Thermal, night vision, underwater tint.
// - **Death grayscale** -- Desaturation on death.
// - **Hit indicators** -- Directional damage indicators on screen edges.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const MAX_HIT_INDICATORS: usize = 16;
const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Screen shake
// ---------------------------------------------------------------------------

/// A screen shake instance.
#[derive(Debug, Clone)]
pub struct ScreenShake {
    /// Shake intensity (amplitude in world units).
    pub intensity: f32,
    /// Shake frequency (shakes per second).
    pub frequency: f32,
    /// Remaining duration.
    pub duration: f32,
    /// Original duration.
    pub original_duration: f32,
    /// Shake decay curve (linear, ease-out, etc.).
    pub decay: ShakeDecay,
    /// Shake axes (which axes are affected).
    pub axes: ShakeAxes,
    /// Current accumulated offset.
    pub current_offset: [f32; 3],
    /// Current accumulated rotation.
    pub current_rotation: [f32; 3],
    /// Rotational shake intensity (radians).
    pub rotation_intensity: f32,
    /// Phase accumulator.
    phase: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShakeDecay {
    Linear,
    EaseOut,
    EaseInOut,
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct ShakeAxes {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl Default for ShakeAxes {
    fn default() -> Self {
        Self { x: true, y: true, z: false }
    }
}

impl ScreenShake {
    /// Create a new screen shake.
    pub fn new(intensity: f32, frequency: f32, duration: f32) -> Self {
        Self {
            intensity,
            frequency,
            duration,
            original_duration: duration,
            decay: ShakeDecay::EaseOut,
            axes: ShakeAxes::default(),
            current_offset: [0.0; 3],
            current_rotation: [0.0; 3],
            rotation_intensity: 0.0,
            phase: 0.0,
        }
    }

    /// Create a damage shake.
    pub fn damage(intensity: f32) -> Self {
        Self::new(intensity, 30.0, 0.3)
    }

    /// Create an explosion shake.
    pub fn explosion(intensity: f32) -> Self {
        let mut shake = Self::new(intensity, 15.0, 0.5);
        shake.rotation_intensity = intensity * 0.02;
        shake
    }

    /// Create a landing shake.
    pub fn landing(intensity: f32) -> Self {
        let mut shake = Self::new(intensity, 20.0, 0.15);
        shake.axes = ShakeAxes { x: false, y: true, z: false };
        shake
    }

    /// Update the shake (returns true if still active).
    pub fn update(&mut self, dt: f32) -> bool {
        if self.duration <= 0.0 {
            self.current_offset = [0.0; 3];
            self.current_rotation = [0.0; 3];
            return false;
        }

        self.duration -= dt;
        self.phase += dt * self.frequency * 2.0 * PI;

        let decay_factor = match self.decay {
            ShakeDecay::Linear => (self.duration / self.original_duration).max(0.0),
            ShakeDecay::EaseOut => {
                let t = 1.0 - (self.duration / self.original_duration).max(0.0);
                1.0 - t * t
            }
            ShakeDecay::EaseInOut => {
                let t = (self.duration / self.original_duration).max(0.0);
                t * t * (3.0 - 2.0 * t)
            }
            ShakeDecay::None => 1.0,
        };

        let amp = self.intensity * decay_factor;

        // Use different frequencies for each axis to avoid uniform motion.
        if self.axes.x {
            self.current_offset[0] = amp * self.phase.sin();
        }
        if self.axes.y {
            self.current_offset[1] = amp * (self.phase * 1.3).cos();
        }
        if self.axes.z {
            self.current_offset[2] = amp * (self.phase * 0.7).sin();
        }

        if self.rotation_intensity > EPSILON {
            let rot_amp = self.rotation_intensity * decay_factor;
            self.current_rotation[2] = rot_amp * (self.phase * 0.9).sin();
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Speed lines
// ---------------------------------------------------------------------------

/// Speed lines visual effect configuration.
#[derive(Debug, Clone)]
pub struct SpeedLinesEffect {
    /// Whether speed lines are active.
    pub active: bool,
    /// Speed threshold to start showing lines.
    pub speed_threshold: f32,
    /// Speed at which lines are at full intensity.
    pub full_speed: f32,
    /// Current intensity (0..1).
    pub intensity: f32,
    /// Number of speed lines.
    pub line_count: u32,
    /// Line length multiplier.
    pub length_scale: f32,
    /// Line color (RGBA).
    pub color: [f32; 4],
    /// Center offset (where lines converge; 0.5,0.5 = center).
    pub focus_point: [f32; 2],
    /// Fade-in time.
    pub fade_in_time: f32,
    /// Fade-out time.
    pub fade_out_time: f32,
    /// Current fade factor.
    fade_factor: f32,
}

impl Default for SpeedLinesEffect {
    fn default() -> Self {
        Self {
            active: true,
            speed_threshold: 15.0,
            full_speed: 30.0,
            intensity: 0.0,
            line_count: 64,
            length_scale: 1.0,
            color: [1.0, 1.0, 1.0, 0.3],
            focus_point: [0.5, 0.5],
            fade_in_time: 0.3,
            fade_out_time: 0.5,
            fade_factor: 0.0,
        }
    }
}

impl SpeedLinesEffect {
    /// Update based on current speed.
    pub fn update(&mut self, speed: f32, dt: f32) {
        if !self.active {
            self.intensity = 0.0;
            return;
        }

        let target = if speed > self.speed_threshold {
            ((speed - self.speed_threshold) / (self.full_speed - self.speed_threshold)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Smooth fade.
        if target > self.fade_factor {
            let rate = if self.fade_in_time > EPSILON { dt / self.fade_in_time } else { 1.0 };
            self.fade_factor = (self.fade_factor + rate).min(target);
        } else {
            let rate = if self.fade_out_time > EPSILON { dt / self.fade_out_time } else { 1.0 };
            self.fade_factor = (self.fade_factor - rate).max(target);
        }

        self.intensity = self.fade_factor;
    }
}

// ---------------------------------------------------------------------------
// Scope zoom
// ---------------------------------------------------------------------------

/// Scope/zoom effect for sniper rifles, telescopes, etc.
#[derive(Debug, Clone)]
pub struct ScopeEffect {
    /// Whether currently scoped in.
    pub scoped: bool,
    /// Target zoom level (1.0 = no zoom).
    pub target_zoom: f32,
    /// Current zoom level (smoothed).
    pub current_zoom: f32,
    /// Zoom speed.
    pub zoom_speed: f32,
    /// Scope overlay texture identifier.
    pub overlay_texture: String,
    /// Scope overlay opacity.
    pub overlay_opacity: f32,
    /// Vignette intensity while scoped.
    pub vignette_intensity: f32,
    /// Sensitivity multiplier while scoped.
    pub sensitivity_multiplier: f32,
    /// Scope sway amplitude.
    pub sway_amplitude: f32,
    /// Scope sway frequency.
    pub sway_frequency: f32,
    /// Current sway offset.
    pub sway_offset: [f32; 2],
    /// Sway phase accumulator.
    sway_phase: f32,
    /// Whether to hold breath (reduces sway).
    pub hold_breath: bool,
    /// Hold breath duration remaining.
    pub breath_remaining: f32,
    /// Max hold breath duration.
    pub max_breath_duration: f32,
}

impl Default for ScopeEffect {
    fn default() -> Self {
        Self {
            scoped: false,
            target_zoom: 1.0,
            current_zoom: 1.0,
            zoom_speed: 10.0,
            overlay_texture: String::new(),
            overlay_opacity: 0.0,
            vignette_intensity: 0.5,
            sensitivity_multiplier: 0.3,
            sway_amplitude: 0.01,
            sway_frequency: 1.5,
            sway_offset: [0.0; 2],
            sway_phase: 0.0,
            hold_breath: false,
            breath_remaining: 5.0,
            max_breath_duration: 5.0,
        }
    }
}

impl ScopeEffect {
    pub fn scope_in(&mut self, zoom: f32) {
        self.scoped = true;
        self.target_zoom = zoom;
    }

    pub fn scope_out(&mut self) {
        self.scoped = false;
        self.target_zoom = 1.0;
    }

    pub fn update(&mut self, dt: f32) {
        // Smooth zoom.
        let diff = self.target_zoom - self.current_zoom;
        self.current_zoom += diff * (1.0 - (-self.zoom_speed * dt).exp());

        // Overlay opacity.
        let target_opacity = if self.scoped { 1.0 } else { 0.0 };
        self.overlay_opacity += (target_opacity - self.overlay_opacity) * dt * self.zoom_speed;

        // Sway.
        if self.scoped {
            self.sway_phase += dt;
            let sway_mult = if self.hold_breath {
                self.breath_remaining -= dt;
                if self.breath_remaining <= 0.0 {
                    self.hold_breath = false;
                }
                0.1
            } else {
                self.breath_remaining = (self.breath_remaining + dt * 0.5).min(self.max_breath_duration);
                1.0
            };

            self.sway_offset[0] = self.sway_amplitude * sway_mult
                * (self.sway_phase * self.sway_frequency * 2.0 * PI).sin();
            self.sway_offset[1] = self.sway_amplitude * sway_mult
                * (self.sway_phase * self.sway_frequency * 1.3 * 2.0 * PI).cos();
        } else {
            self.sway_offset = [0.0; 2];
        }
    }
}

// ---------------------------------------------------------------------------
// Vision overlays
// ---------------------------------------------------------------------------

/// Type of vision overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionMode {
    Normal,
    NightVision,
    Thermal,
    Underwater,
    Xray,
    Detective,
}

/// Vision overlay effect.
#[derive(Debug, Clone)]
pub struct VisionOverlay {
    pub mode: VisionMode,
    pub intensity: f32,
    pub tint_color: [f32; 4],
    pub noise_intensity: f32,
    pub scanline_intensity: f32,
    pub active: bool,
    pub transition_speed: f32,
    current_intensity: f32,
}

impl VisionOverlay {
    pub fn night_vision() -> Self {
        Self {
            mode: VisionMode::NightVision,
            intensity: 1.0,
            tint_color: [0.0, 1.0, 0.0, 0.3],
            noise_intensity: 0.1,
            scanline_intensity: 0.05,
            active: false,
            transition_speed: 5.0,
            current_intensity: 0.0,
        }
    }

    pub fn thermal() -> Self {
        Self {
            mode: VisionMode::Thermal,
            intensity: 1.0,
            tint_color: [1.0, 0.2, 0.0, 0.2],
            noise_intensity: 0.05,
            scanline_intensity: 0.0,
            active: false,
            transition_speed: 3.0,
            current_intensity: 0.0,
        }
    }

    pub fn underwater() -> Self {
        Self {
            mode: VisionMode::Underwater,
            intensity: 1.0,
            tint_color: [0.0, 0.3, 0.6, 0.4],
            noise_intensity: 0.0,
            scanline_intensity: 0.0,
            active: false,
            transition_speed: 2.0,
            current_intensity: 0.0,
        }
    }

    pub fn toggle(&mut self) {
        self.active = !self.active;
    }

    pub fn update(&mut self, dt: f32) {
        let target = if self.active { self.intensity } else { 0.0 };
        let diff = target - self.current_intensity;
        self.current_intensity += diff * (1.0 - (-self.transition_speed * dt).exp());
    }

    pub fn current_intensity(&self) -> f32 {
        self.current_intensity
    }
}

// ---------------------------------------------------------------------------
// Hit indicator
// ---------------------------------------------------------------------------

/// A directional hit indicator on screen edges.
#[derive(Debug, Clone)]
pub struct HitIndicator {
    /// Direction the hit came from (radians, 0 = front).
    pub direction: f32,
    /// Remaining display time.
    pub remaining: f32,
    /// Total display time.
    pub total_time: f32,
    /// Intensity (based on damage amount).
    pub intensity: f32,
    /// Color (RGBA).
    pub color: [f32; 4],
    /// Whether this is a critical hit indicator.
    pub is_critical: bool,
}

impl HitIndicator {
    pub fn new(direction: f32, intensity: f32, duration: f32) -> Self {
        Self {
            direction,
            remaining: duration,
            total_time: duration,
            intensity: intensity.clamp(0.0, 2.0),
            color: [1.0, 0.0, 0.0, 0.8],
            is_critical: false,
        }
    }

    pub fn opacity(&self) -> f32 {
        if self.total_time <= EPSILON {
            return 0.0;
        }
        (self.remaining / self.total_time) * self.intensity * self.color[3]
    }

    pub fn update(&mut self, dt: f32) -> bool {
        self.remaining -= dt;
        self.remaining > 0.0
    }
}

// ---------------------------------------------------------------------------
// Death effect
// ---------------------------------------------------------------------------

/// Post-death visual effect.
#[derive(Debug, Clone)]
pub struct DeathEffect {
    pub active: bool,
    pub desaturation: f32,
    pub vignette_intensity: f32,
    pub vignette_color: [f32; 4],
    pub blur_amount: f32,
    pub fade_speed: f32,
    current_desaturation: f32,
    current_vignette: f32,
}

impl Default for DeathEffect {
    fn default() -> Self {
        Self {
            active: false,
            desaturation: 0.8,
            vignette_intensity: 0.7,
            vignette_color: [0.2, 0.0, 0.0, 1.0],
            blur_amount: 2.0,
            fade_speed: 2.0,
            current_desaturation: 0.0,
            current_vignette: 0.0,
        }
    }
}

impl DeathEffect {
    pub fn trigger(&mut self) {
        self.active = true;
    }

    pub fn reset(&mut self) {
        self.active = false;
        self.current_desaturation = 0.0;
        self.current_vignette = 0.0;
    }

    pub fn update(&mut self, dt: f32) {
        let target_desat = if self.active { self.desaturation } else { 0.0 };
        let target_vig = if self.active { self.vignette_intensity } else { 0.0 };

        self.current_desaturation += (target_desat - self.current_desaturation) * dt * self.fade_speed;
        self.current_vignette += (target_vig - self.current_vignette) * dt * self.fade_speed;
    }

    pub fn current_desaturation(&self) -> f32 {
        self.current_desaturation
    }

    pub fn current_vignette(&self) -> f32 {
        self.current_vignette
    }
}

// ---------------------------------------------------------------------------
// Camera effects manager
// ---------------------------------------------------------------------------

/// Manages all gameplay camera effects.
pub struct CameraEffectsManager {
    pub shakes: Vec<ScreenShake>,
    pub speed_lines: SpeedLinesEffect,
    pub scope: ScopeEffect,
    pub vision_overlays: Vec<VisionOverlay>,
    pub hit_indicators: VecDeque<HitIndicator>,
    pub death_effect: DeathEffect,
    /// Combined shake offset from all active shakes.
    pub combined_shake_offset: [f32; 3],
    /// Combined shake rotation.
    pub combined_shake_rotation: [f32; 3],
}

impl CameraEffectsManager {
    pub fn new() -> Self {
        Self {
            shakes: Vec::new(),
            speed_lines: SpeedLinesEffect::default(),
            scope: ScopeEffect::default(),
            vision_overlays: Vec::new(),
            hit_indicators: VecDeque::new(),
            death_effect: DeathEffect::default(),
            combined_shake_offset: [0.0; 3],
            combined_shake_rotation: [0.0; 3],
        }
    }

    pub fn add_shake(&mut self, shake: ScreenShake) {
        self.shakes.push(shake);
    }

    pub fn add_hit_indicator(&mut self, direction: f32, intensity: f32, duration: f32) {
        if self.hit_indicators.len() >= MAX_HIT_INDICATORS {
            self.hit_indicators.pop_front();
        }
        self.hit_indicators.push_back(HitIndicator::new(direction, intensity, duration));
    }

    pub fn update(&mut self, dt: f32, speed: f32) {
        // Update shakes.
        self.shakes.retain_mut(|s| s.update(dt));
        self.combined_shake_offset = [0.0; 3];
        self.combined_shake_rotation = [0.0; 3];
        for shake in &self.shakes {
            for i in 0..3 {
                self.combined_shake_offset[i] += shake.current_offset[i];
                self.combined_shake_rotation[i] += shake.current_rotation[i];
            }
        }

        // Update speed lines.
        self.speed_lines.update(speed, dt);

        // Update scope.
        self.scope.update(dt);

        // Update vision overlays.
        for overlay in &mut self.vision_overlays {
            overlay.update(dt);
        }

        // Update hit indicators.
        self.hit_indicators.retain_mut(|h| h.update(dt));

        // Update death effect.
        self.death_effect.update(dt);
    }

    pub fn clear_all(&mut self) {
        self.shakes.clear();
        self.hit_indicators.clear();
        self.death_effect.reset();
        self.scope.scope_out();
        self.combined_shake_offset = [0.0; 3];
        self.combined_shake_rotation = [0.0; 3];
    }
}

impl Default for CameraEffectsManager {
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
    fn test_screen_shake_decay() {
        let mut shake = ScreenShake::new(1.0, 30.0, 0.5);
        assert!(shake.update(0.1));
        assert!(shake.current_offset[0].abs() > 0.0 || shake.current_offset[1].abs() > 0.0);
        for _ in 0..10 {
            shake.update(0.1);
        }
        assert!(!shake.update(0.1));
    }

    #[test]
    fn test_speed_lines() {
        let mut sl = SpeedLinesEffect::default();
        sl.update(5.0, 0.016); // Below threshold.
        assert!(sl.intensity < EPSILON);
        sl.update(25.0, 1.0); // Above threshold.
        assert!(sl.intensity > 0.0);
    }

    #[test]
    fn test_scope() {
        let mut scope = ScopeEffect::default();
        scope.scope_in(4.0);
        scope.update(1.0);
        assert!(scope.current_zoom > 1.0);
        scope.scope_out();
        scope.update(1.0);
        assert!(scope.current_zoom < 4.0);
    }

    #[test]
    fn test_hit_indicator() {
        let mut indicator = HitIndicator::new(0.0, 1.0, 1.0);
        assert!(indicator.opacity() > 0.0);
        indicator.update(1.5);
        assert!(indicator.opacity() <= 0.0);
    }

    #[test]
    fn test_death_effect() {
        let mut effect = DeathEffect::default();
        effect.trigger();
        effect.update(1.0);
        assert!(effect.current_desaturation() > 0.0);
        effect.reset();
        assert!(effect.current_desaturation() < EPSILON);
    }
}
