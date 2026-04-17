// engine/gameplay/src/physics_materials_gameplay.rs
// Gameplay physics materials: footstep sounds, bullet impacts, slide friction, bounce.
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceMaterial {
    Concrete, Wood, Metal, Grass, Sand, Snow, Water, Mud, Gravel, Glass, Carpet, Tile, Ice, Rubber,
}

#[derive(Debug, Clone)]
pub struct FootstepConfig {
    pub sound_ids: Vec<String>,
    pub volume: f32,
    pub pitch_variation: f32,
    pub particle_effect: Option<String>,
    pub decal: Option<String>,
    pub step_interval: f32,
}

impl Default for FootstepConfig {
    fn default() -> Self {
        Self { sound_ids: Vec::new(), volume: 0.5, pitch_variation: 0.1, particle_effect: None, decal: None, step_interval: 0.5 }
    }
}

#[derive(Debug, Clone)]
pub struct ImpactConfig {
    pub sound_ids: Vec<String>,
    pub particle_effect: String,
    pub decal: Option<String>,
    pub volume: f32,
    pub debris_count: u32,
    pub debris_material: Option<SurfaceMaterial>,
    pub sparks: bool,
    pub dust: bool,
    pub blood: bool,
}

impl Default for ImpactConfig {
    fn default() -> Self {
        Self { sound_ids: Vec::new(), particle_effect: String::new(), decal: None, volume: 0.7, debris_count: 0, debris_material: None, sparks: false, dust: false, blood: false }
    }
}

#[derive(Debug, Clone)]
pub struct SlideFrictionConfig {
    pub static_friction: f32,
    pub dynamic_friction: f32,
    pub slide_sound: Option<String>,
    pub slide_particles: Option<String>,
    pub speed_multiplier: f32,
}

#[derive(Debug, Clone)]
pub struct BounceConfig {
    pub restitution: f32,
    pub bounce_sound: Option<String>,
    pub min_velocity_for_sound: f32,
    pub volume_scale: f32,
}

#[derive(Debug, Clone)]
pub struct PhysicsMaterialGameplay {
    pub material: SurfaceMaterial,
    pub footsteps: FootstepConfig,
    pub bullet_impact: ImpactConfig,
    pub melee_impact: ImpactConfig,
    pub explosion_impact: ImpactConfig,
    pub slide_friction: SlideFrictionConfig,
    pub bounce: BounceConfig,
    pub is_penetrable: bool,
    pub penetration_depth: f32,
    pub noise_on_impact: f32,
}

pub struct PhysicsMaterialDatabase {
    materials: HashMap<SurfaceMaterial, PhysicsMaterialGameplay>,
}

impl PhysicsMaterialDatabase {
    pub fn new() -> Self {
        let mut db = Self { materials: HashMap::new() };
        db.register_defaults();
        db
    }

    fn register_defaults(&mut self) {
        self.register(SurfaceMaterial::Concrete, PhysicsMaterialGameplay {
            material: SurfaceMaterial::Concrete,
            footsteps: FootstepConfig { sound_ids: vec!["footstep_concrete_01".into(), "footstep_concrete_02".into(), "footstep_concrete_03".into()], volume: 0.5, pitch_variation: 0.1, particle_effect: Some("dust_puff_small".into()), decal: None, step_interval: 0.45 },
            bullet_impact: ImpactConfig { sound_ids: vec!["impact_concrete_01".into()], particle_effect: "concrete_chips".into(), decal: Some("bullet_hole_concrete".into()), volume: 0.8, debris_count: 5, debris_material: Some(SurfaceMaterial::Concrete), sparks: true, dust: true, blood: false },
            melee_impact: ImpactConfig { sound_ids: vec!["melee_concrete".into()], particle_effect: "concrete_dust".into(), ..Default::default() },
            explosion_impact: ImpactConfig { sound_ids: vec!["explosion_concrete".into()], particle_effect: "concrete_debris".into(), debris_count: 20, dust: true, ..Default::default() },
            slide_friction: SlideFrictionConfig { static_friction: 0.7, dynamic_friction: 0.5, slide_sound: Some("slide_concrete".into()), slide_particles: Some("concrete_scrape".into()), speed_multiplier: 1.0 },
            bounce: BounceConfig { restitution: 0.2, bounce_sound: Some("bounce_concrete".into()), min_velocity_for_sound: 1.0, volume_scale: 0.5 },
            is_penetrable: false, penetration_depth: 0.0, noise_on_impact: 0.8,
        });

        self.register(SurfaceMaterial::Metal, PhysicsMaterialGameplay {
            material: SurfaceMaterial::Metal,
            footsteps: FootstepConfig { sound_ids: vec!["footstep_metal_01".into(), "footstep_metal_02".into()], volume: 0.6, pitch_variation: 0.15, particle_effect: None, decal: None, step_interval: 0.4 },
            bullet_impact: ImpactConfig { sound_ids: vec!["impact_metal_01".into()], particle_effect: "sparks".into(), decal: Some("bullet_hole_metal".into()), volume: 0.9, debris_count: 0, debris_material: None, sparks: true, dust: false, blood: false },
            melee_impact: ImpactConfig { sound_ids: vec!["melee_metal".into()], particle_effect: "sparks_melee".into(), sparks: true, ..Default::default() },
            explosion_impact: ImpactConfig { sound_ids: vec!["explosion_metal".into()], particle_effect: "metal_debris".into(), debris_count: 10, sparks: true, ..Default::default() },
            slide_friction: SlideFrictionConfig { static_friction: 0.5, dynamic_friction: 0.3, slide_sound: Some("slide_metal".into()), slide_particles: Some("sparks_slide".into()), speed_multiplier: 1.2 },
            bounce: BounceConfig { restitution: 0.4, bounce_sound: Some("bounce_metal".into()), min_velocity_for_sound: 0.5, volume_scale: 0.7 },
            is_penetrable: false, penetration_depth: 0.0, noise_on_impact: 1.0,
        });

        self.register(SurfaceMaterial::Wood, PhysicsMaterialGameplay {
            material: SurfaceMaterial::Wood,
            footsteps: FootstepConfig { sound_ids: vec!["footstep_wood_01".into(), "footstep_wood_02".into()], volume: 0.4, pitch_variation: 0.1, particle_effect: None, decal: None, step_interval: 0.5 },
            bullet_impact: ImpactConfig { sound_ids: vec!["impact_wood_01".into()], particle_effect: "wood_splinters".into(), decal: Some("bullet_hole_wood".into()), volume: 0.6, debris_count: 8, debris_material: Some(SurfaceMaterial::Wood), sparks: false, dust: true, blood: false },
            melee_impact: ImpactConfig::default(),
            explosion_impact: ImpactConfig { particle_effect: "wood_debris".into(), debris_count: 15, ..Default::default() },
            slide_friction: SlideFrictionConfig { static_friction: 0.6, dynamic_friction: 0.4, slide_sound: None, slide_particles: None, speed_multiplier: 1.0 },
            bounce: BounceConfig { restitution: 0.3, bounce_sound: Some("bounce_wood".into()), min_velocity_for_sound: 1.0, volume_scale: 0.4 },
            is_penetrable: true, penetration_depth: 0.05, noise_on_impact: 0.6,
        });

        self.register(SurfaceMaterial::Grass, PhysicsMaterialGameplay {
            material: SurfaceMaterial::Grass,
            footsteps: FootstepConfig { sound_ids: vec!["footstep_grass_01".into(), "footstep_grass_02".into()], volume: 0.3, pitch_variation: 0.15, particle_effect: Some("grass_rustle".into()), decal: None, step_interval: 0.5 },
            bullet_impact: ImpactConfig { sound_ids: vec!["impact_dirt".into()], particle_effect: "dirt_puff".into(), decal: None, volume: 0.3, debris_count: 3, debris_material: None, sparks: false, dust: true, blood: false },
            melee_impact: ImpactConfig::default(),
            explosion_impact: ImpactConfig { particle_effect: "dirt_explosion".into(), debris_count: 10, dust: true, ..Default::default() },
            slide_friction: SlideFrictionConfig { static_friction: 0.8, dynamic_friction: 0.6, slide_sound: None, slide_particles: Some("grass_slide".into()), speed_multiplier: 0.8 },
            bounce: BounceConfig { restitution: 0.1, bounce_sound: None, min_velocity_for_sound: 2.0, volume_scale: 0.2 },
            is_penetrable: true, penetration_depth: 0.2, noise_on_impact: 0.3,
        });

        self.register(SurfaceMaterial::Ice, PhysicsMaterialGameplay {
            material: SurfaceMaterial::Ice,
            footsteps: FootstepConfig { sound_ids: vec!["footstep_ice_01".into()], volume: 0.4, pitch_variation: 0.2, particle_effect: Some("ice_crystals".into()), decal: None, step_interval: 0.55 },
            bullet_impact: ImpactConfig { sound_ids: vec!["impact_ice".into()], particle_effect: "ice_shatter".into(), decal: None, volume: 0.7, debris_count: 10, debris_material: Some(SurfaceMaterial::Ice), sparks: false, dust: false, blood: false },
            melee_impact: ImpactConfig::default(),
            explosion_impact: ImpactConfig::default(),
            slide_friction: SlideFrictionConfig { static_friction: 0.1, dynamic_friction: 0.05, slide_sound: Some("slide_ice".into()), slide_particles: Some("ice_scrape".into()), speed_multiplier: 2.0 },
            bounce: BounceConfig { restitution: 0.3, bounce_sound: Some("bounce_ice".into()), min_velocity_for_sound: 0.5, volume_scale: 0.5 },
            is_penetrable: true, penetration_depth: 0.1, noise_on_impact: 0.5,
        });
    }

    pub fn register(&mut self, material: SurfaceMaterial, config: PhysicsMaterialGameplay) {
        self.materials.insert(material, config);
    }

    pub fn get(&self, material: SurfaceMaterial) -> Option<&PhysicsMaterialGameplay> {
        self.materials.get(&material)
    }

    pub fn get_footstep_sound(&self, material: SurfaceMaterial, index: usize) -> Option<&str> {
        self.materials.get(&material).and_then(|m| m.footsteps.sound_ids.get(index % m.footsteps.sound_ids.len().max(1))).map(|s| s.as_str())
    }

    pub fn get_impact_effect(&self, material: SurfaceMaterial) -> Option<&str> {
        self.materials.get(&material).map(|m| m.bullet_impact.particle_effect.as_str())
    }

    pub fn get_slide_friction(&self, material: SurfaceMaterial) -> f32 {
        self.materials.get(&material).map(|m| m.slide_friction.dynamic_friction).unwrap_or(0.5)
    }

    pub fn get_bounce_restitution(&self, material: SurfaceMaterial) -> f32 {
        self.materials.get(&material).map(|m| m.bounce.restitution).unwrap_or(0.2)
    }

    pub fn combine_friction(a: f32, b: f32) -> f32 { (a * b).sqrt() }
    pub fn combine_restitution(a: f32, b: f32) -> f32 { a.max(b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_material_db() {
        let db = PhysicsMaterialDatabase::new();
        assert!(db.get(SurfaceMaterial::Concrete).is_some());
        assert!(db.get(SurfaceMaterial::Metal).is_some());
    }
    #[test]
    fn test_footstep() {
        let db = PhysicsMaterialDatabase::new();
        let sound = db.get_footstep_sound(SurfaceMaterial::Concrete, 0);
        assert!(sound.is_some());
    }
    #[test]
    fn test_friction_combine() {
        let f = PhysicsMaterialDatabase::combine_friction(0.5, 0.8);
        assert!(f > 0.0 && f < 1.0);
    }
}
