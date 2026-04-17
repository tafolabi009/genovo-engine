// engine/gameplay/src/particle_effects.rs
//
// Gameplay particle effect presets for the Genovo engine.
// Provides ready-to-use particle configurations for common game effects:
// blood splatter, dust cloud, explosion VFX, magic spells, healing aura,
// shield bubble, footstep dust, water splash, fire/smoke, electricity.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticlePresetType {
    BloodSplatter, DustCloud, ExplosionFire, ExplosionSmoke, MagicSparkle,
    MagicOrb, HealingAura, ShieldBubble, FootstepDust, FootstepSnow,
    WaterSplash, WaterRipple, FireSmall, FireLarge, SmokeTrail,
    Electricity, LightningBolt, Embers, Sparks, LeafFall,
    SnowFall, RainDrop, MuzzleFlash, BulletImpact, LaserBeam,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterShape { Point, Sphere, Hemisphere, Cone, Box, Ring, Line, Mesh }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendModeParticle { Additive, AlphaBlend, Multiply, PremultipliedAlpha }

#[derive(Debug, Clone)]
pub struct ParticleColor {
    pub r: f32, pub g: f32, pub b: f32, pub a: f32,
}

impl ParticleColor {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
    pub fn white() -> Self { Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 } }
    pub fn red() -> Self { Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 } }
    pub fn fire() -> Self { Self { r: 1.0, g: 0.5, b: 0.1, a: 1.0 } }
    pub fn blood() -> Self { Self { r: 0.6, g: 0.0, b: 0.0, a: 1.0 } }
    pub fn dust() -> Self { Self { r: 0.7, g: 0.6, b: 0.4, a: 0.5 } }
    pub fn magic_blue() -> Self { Self { r: 0.2, g: 0.4, b: 1.0, a: 0.8 } }
    pub fn heal_green() -> Self { Self { r: 0.1, g: 1.0, b: 0.3, a: 0.7 } }
    pub fn electric() -> Self { Self { r: 0.5, g: 0.8, b: 1.0, a: 0.9 } }
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t, a: a.a + (b.a - a.a) * t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParticlePresetConfig {
    pub preset_type: ParticlePresetType,
    pub emitter_shape: EmitterShape,
    pub blend_mode: BlendModeParticle,
    pub max_particles: u32,
    pub emission_rate: f32,
    pub burst_count: u32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: ParticleColor,
    pub color_end: ParticleColor,
    pub gravity_multiplier: f32,
    pub drag: f32,
    pub rotation_speed: f32,
    pub emitter_radius: f32,
    pub cone_angle: f32,
    pub world_space: bool,
    pub looping: bool,
    pub duration: f32,
    pub sort_by_depth: bool,
    pub face_camera: bool,
    pub inherit_velocity: f32,
    pub noise_strength: f32,
    pub noise_frequency: f32,
    pub stretch_with_speed: bool,
    pub stretch_factor: f32,
    pub collision_enabled: bool,
    pub collision_bounce: f32,
    pub collision_lifetime_loss: f32,
    pub sub_emitter: Option<Box<ParticlePresetConfig>>,
}

impl Default for ParticlePresetConfig {
    fn default() -> Self {
        Self {
            preset_type: ParticlePresetType::DustCloud,
            emitter_shape: EmitterShape::Point, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 100, emission_rate: 10.0, burst_count: 0,
            lifetime_min: 1.0, lifetime_max: 2.0,
            speed_min: 1.0, speed_max: 3.0,
            size_start: 0.1, size_end: 0.5,
            color_start: ParticleColor::white(), color_end: ParticleColor::new(1.0, 1.0, 1.0, 0.0),
            gravity_multiplier: 0.0, drag: 0.0, rotation_speed: 0.0,
            emitter_radius: 0.0, cone_angle: 45.0,
            world_space: true, looping: false, duration: 1.0,
            sort_by_depth: false, face_camera: true,
            inherit_velocity: 0.0, noise_strength: 0.0, noise_frequency: 1.0,
            stretch_with_speed: false, stretch_factor: 1.0,
            collision_enabled: false, collision_bounce: 0.5, collision_lifetime_loss: 0.5,
            sub_emitter: None,
        }
    }
}

impl ParticlePresetConfig {
    pub fn blood_splatter() -> Self {
        Self {
            preset_type: ParticlePresetType::BloodSplatter,
            emitter_shape: EmitterShape::Cone, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 50, emission_rate: 0.0, burst_count: 30,
            lifetime_min: 0.3, lifetime_max: 0.8,
            speed_min: 3.0, speed_max: 8.0,
            size_start: 0.02, size_end: 0.08,
            color_start: ParticleColor::blood(), color_end: ParticleColor::new(0.3, 0.0, 0.0, 0.0),
            gravity_multiplier: 2.0, drag: 0.5, cone_angle: 60.0,
            collision_enabled: true, collision_bounce: 0.1, collision_lifetime_loss: 0.8,
            ..Default::default()
        }
    }

    pub fn dust_cloud() -> Self {
        Self {
            preset_type: ParticlePresetType::DustCloud,
            emitter_shape: EmitterShape::Sphere, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 30, emission_rate: 0.0, burst_count: 15,
            lifetime_min: 1.0, lifetime_max: 3.0,
            speed_min: 0.5, speed_max: 2.0,
            size_start: 0.2, size_end: 1.5,
            color_start: ParticleColor::dust(), color_end: ParticleColor::new(0.7, 0.6, 0.4, 0.0),
            gravity_multiplier: -0.1, drag: 1.0, emitter_radius: 0.3,
            ..Default::default()
        }
    }

    pub fn explosion_fire() -> Self {
        Self {
            preset_type: ParticlePresetType::ExplosionFire,
            emitter_shape: EmitterShape::Sphere, blend_mode: BlendModeParticle::Additive,
            max_particles: 60, emission_rate: 0.0, burst_count: 40,
            lifetime_min: 0.2, lifetime_max: 0.8,
            speed_min: 5.0, speed_max: 15.0,
            size_start: 0.5, size_end: 3.0,
            color_start: ParticleColor::fire(), color_end: ParticleColor::new(1.0, 0.1, 0.0, 0.0),
            gravity_multiplier: -0.5, drag: 2.0, emitter_radius: 0.5,
            sub_emitter: Some(Box::new(Self::explosion_smoke())),
            ..Default::default()
        }
    }

    pub fn explosion_smoke() -> Self {
        Self {
            preset_type: ParticlePresetType::ExplosionSmoke,
            emitter_shape: EmitterShape::Sphere, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 40, emission_rate: 0.0, burst_count: 20,
            lifetime_min: 1.0, lifetime_max: 4.0,
            speed_min: 1.0, speed_max: 5.0,
            size_start: 0.5, size_end: 5.0,
            color_start: ParticleColor::new(0.3, 0.3, 0.3, 0.6),
            color_end: ParticleColor::new(0.1, 0.1, 0.1, 0.0),
            gravity_multiplier: -0.3, drag: 1.5, emitter_radius: 1.0,
            ..Default::default()
        }
    }

    pub fn healing_aura() -> Self {
        Self {
            preset_type: ParticlePresetType::HealingAura,
            emitter_shape: EmitterShape::Ring, blend_mode: BlendModeParticle::Additive,
            max_particles: 50, emission_rate: 20.0, burst_count: 0,
            lifetime_min: 0.5, lifetime_max: 1.5,
            speed_min: 0.5, speed_max: 2.0,
            size_start: 0.1, size_end: 0.02,
            color_start: ParticleColor::heal_green(),
            color_end: ParticleColor::new(0.3, 1.0, 0.5, 0.0),
            gravity_multiplier: -1.0, drag: 0.0, emitter_radius: 1.0,
            looping: true, duration: 3.0,
            ..Default::default()
        }
    }

    pub fn electricity() -> Self {
        Self {
            preset_type: ParticlePresetType::Electricity,
            emitter_shape: EmitterShape::Sphere, blend_mode: BlendModeParticle::Additive,
            max_particles: 30, emission_rate: 30.0, burst_count: 0,
            lifetime_min: 0.05, lifetime_max: 0.15,
            speed_min: 5.0, speed_max: 15.0,
            size_start: 0.02, size_end: 0.01,
            color_start: ParticleColor::electric(),
            color_end: ParticleColor::new(0.8, 0.9, 1.0, 0.0),
            noise_strength: 5.0, noise_frequency: 20.0,
            stretch_with_speed: true, stretch_factor: 3.0,
            looping: true, emitter_radius: 0.3,
            ..Default::default()
        }
    }

    pub fn fire_small() -> Self {
        Self {
            preset_type: ParticlePresetType::FireSmall,
            emitter_shape: EmitterShape::Cone, blend_mode: BlendModeParticle::Additive,
            max_particles: 40, emission_rate: 20.0, burst_count: 0,
            lifetime_min: 0.3, lifetime_max: 0.8,
            speed_min: 1.0, speed_max: 3.0,
            size_start: 0.1, size_end: 0.4,
            color_start: ParticleColor::new(1.0, 0.8, 0.2, 0.9),
            color_end: ParticleColor::new(1.0, 0.2, 0.0, 0.0),
            gravity_multiplier: -2.0, cone_angle: 20.0,
            looping: true, noise_strength: 0.5,
            ..Default::default()
        }
    }

    pub fn water_splash() -> Self {
        Self {
            preset_type: ParticlePresetType::WaterSplash,
            emitter_shape: EmitterShape::Hemisphere, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 40, emission_rate: 0.0, burst_count: 25,
            lifetime_min: 0.3, lifetime_max: 1.0,
            speed_min: 2.0, speed_max: 6.0,
            size_start: 0.03, size_end: 0.08,
            color_start: ParticleColor::new(0.6, 0.8, 1.0, 0.7),
            color_end: ParticleColor::new(0.5, 0.7, 1.0, 0.0),
            gravity_multiplier: 1.5, drag: 0.3,
            collision_enabled: true, collision_bounce: 0.0, collision_lifetime_loss: 1.0,
            ..Default::default()
        }
    }

    pub fn muzzle_flash() -> Self {
        Self {
            preset_type: ParticlePresetType::MuzzleFlash,
            emitter_shape: EmitterShape::Cone, blend_mode: BlendModeParticle::Additive,
            max_particles: 10, emission_rate: 0.0, burst_count: 5,
            lifetime_min: 0.02, lifetime_max: 0.06,
            speed_min: 10.0, speed_max: 20.0,
            size_start: 0.05, size_end: 0.15,
            color_start: ParticleColor::new(1.0, 0.9, 0.5, 1.0),
            color_end: ParticleColor::new(1.0, 0.5, 0.1, 0.0),
            cone_angle: 15.0,
            ..Default::default()
        }
    }

    pub fn footstep_dust() -> Self {
        Self {
            preset_type: ParticlePresetType::FootstepDust,
            emitter_shape: EmitterShape::Hemisphere, blend_mode: BlendModeParticle::AlphaBlend,
            max_particles: 15, emission_rate: 0.0, burst_count: 8,
            lifetime_min: 0.5, lifetime_max: 1.5,
            speed_min: 0.3, speed_max: 1.0,
            size_start: 0.05, size_end: 0.3,
            color_start: ParticleColor::dust(),
            color_end: ParticleColor::new(0.7, 0.6, 0.4, 0.0),
            gravity_multiplier: -0.1, drag: 2.0,
            ..Default::default()
        }
    }

    pub fn shield_bubble() -> Self {
        Self {
            preset_type: ParticlePresetType::ShieldBubble,
            emitter_shape: EmitterShape::Sphere, blend_mode: BlendModeParticle::Additive,
            max_particles: 80, emission_rate: 30.0, burst_count: 0,
            lifetime_min: 0.3, lifetime_max: 0.8,
            speed_min: 0.0, speed_max: 0.5,
            size_start: 0.05, size_end: 0.02,
            color_start: ParticleColor::new(0.3, 0.5, 1.0, 0.5),
            color_end: ParticleColor::new(0.2, 0.3, 1.0, 0.0),
            emitter_radius: 2.0, looping: true, world_space: false,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct ParticlePresetLibrary {
    pub presets: HashMap<ParticlePresetType, ParticlePresetConfig>,
}

impl ParticlePresetLibrary {
    pub fn new() -> Self {
        let mut lib = Self { presets: HashMap::new() };
        lib.register(ParticlePresetConfig::blood_splatter());
        lib.register(ParticlePresetConfig::dust_cloud());
        lib.register(ParticlePresetConfig::explosion_fire());
        lib.register(ParticlePresetConfig::explosion_smoke());
        lib.register(ParticlePresetConfig::healing_aura());
        lib.register(ParticlePresetConfig::electricity());
        lib.register(ParticlePresetConfig::fire_small());
        lib.register(ParticlePresetConfig::water_splash());
        lib.register(ParticlePresetConfig::muzzle_flash());
        lib.register(ParticlePresetConfig::footstep_dust());
        lib.register(ParticlePresetConfig::shield_bubble());
        lib
    }

    pub fn register(&mut self, config: ParticlePresetConfig) {
        self.presets.insert(config.preset_type, config);
    }

    pub fn get(&self, preset: ParticlePresetType) -> Option<&ParticlePresetConfig> {
        self.presets.get(&preset)
    }

    pub fn preset_count(&self) -> usize { self.presets.len() }
}

impl Default for ParticlePresetLibrary { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_library() {
        let lib = ParticlePresetLibrary::new();
        assert!(lib.preset_count() >= 10);
        assert!(lib.get(ParticlePresetType::BloodSplatter).is_some());
        assert!(lib.get(ParticlePresetType::HealingAura).is_some());
    }

    #[test]
    fn test_color_lerp() {
        let a = ParticleColor::red();
        let b = ParticleColor::new(0.0, 0.0, 1.0, 1.0);
        let mid = ParticleColor::lerp(&a, &b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);
    }
}
