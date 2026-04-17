// engine/gameplay/src/environmental_hazards.rs
//
// Environmental hazard system for the Genovo engine.
// Provides lava, quicksand, toxic gas, radiation, extreme cold,
// underwater pressure, fall damage, electricity, fire spread,
// and hazard zones with damage-over-time.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HazardId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HazardType {
    Lava, Quicksand, ToxicGas, Radiation, ExtremeCold, ExtremeHeat,
    UnderwaterPressure, Electricity, FireSpread, AcidPool,
    Spikes, Vacuum, Poison, Magnetic, Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageChannel { Health, Shield, Armor, Stamina, Oxygen, Sanity }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardShape { Sphere, Box, Cylinder, Capsule, Plane }

#[derive(Debug, Clone)]
pub struct HazardDamageProfile {
    pub damage_per_second: f32,
    pub channel: DamageChannel,
    pub damage_type: String,
    pub armor_penetration: f32,
    pub tick_interval: f32,
    pub initial_delay: f32,
    pub ramp_up_time: f32,
    pub max_stacks: u32,
    pub stack_multiplier: f32,
    pub dot_duration: f32,
    pub dot_damage_per_tick: f32,
    pub dot_tick_interval: f32,
    pub apply_movement_slow: bool,
    pub slow_factor: f32,
    pub apply_visual_effect: bool,
    pub visual_effect_id: u32,
}

impl Default for HazardDamageProfile {
    fn default() -> Self {
        Self {
            damage_per_second: 10.0, channel: DamageChannel::Health,
            damage_type: "generic".to_string(), armor_penetration: 0.0,
            tick_interval: 0.5, initial_delay: 0.0, ramp_up_time: 0.0,
            max_stacks: 1, stack_multiplier: 1.0,
            dot_duration: 0.0, dot_damage_per_tick: 0.0, dot_tick_interval: 1.0,
            apply_movement_slow: false, slow_factor: 1.0,
            apply_visual_effect: false, visual_effect_id: 0,
        }
    }
}

impl HazardDamageProfile {
    pub fn lava() -> Self {
        Self {
            damage_per_second: 50.0, damage_type: "fire".to_string(),
            armor_penetration: 0.5, tick_interval: 0.25,
            dot_duration: 3.0, dot_damage_per_tick: 10.0, dot_tick_interval: 0.5,
            apply_visual_effect: true, ..Default::default()
        }
    }

    pub fn toxic_gas() -> Self {
        Self {
            damage_per_second: 8.0, damage_type: "poison".to_string(),
            tick_interval: 1.0, ramp_up_time: 2.0,
            dot_duration: 5.0, dot_damage_per_tick: 3.0,
            apply_movement_slow: true, slow_factor: 0.7,
            ..Default::default()
        }
    }

    pub fn radiation() -> Self {
        Self {
            damage_per_second: 5.0, damage_type: "radiation".to_string(),
            tick_interval: 1.0, max_stacks: 5, stack_multiplier: 1.5,
            dot_duration: 10.0, dot_damage_per_tick: 2.0,
            ..Default::default()
        }
    }

    pub fn extreme_cold() -> Self {
        Self {
            damage_per_second: 3.0, channel: DamageChannel::Stamina,
            damage_type: "cold".to_string(), tick_interval: 2.0,
            ramp_up_time: 5.0, apply_movement_slow: true, slow_factor: 0.5,
            ..Default::default()
        }
    }

    pub fn electricity() -> Self {
        Self {
            damage_per_second: 30.0, damage_type: "electric".to_string(),
            armor_penetration: 0.8, tick_interval: 0.1,
            apply_visual_effect: true, ..Default::default()
        }
    }

    pub fn quicksand() -> Self {
        Self {
            damage_per_second: 0.0, damage_type: "suffocation".to_string(),
            ramp_up_time: 10.0, apply_movement_slow: true, slow_factor: 0.2,
            ..Default::default()
        }
    }

    pub fn underwater_pressure() -> Self {
        Self {
            damage_per_second: 0.0, channel: DamageChannel::Oxygen,
            damage_type: "pressure".to_string(), tick_interval: 1.0,
            ramp_up_time: 3.0, ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct HazardZone {
    pub id: HazardId,
    pub hazard_type: HazardType,
    pub shape: HazardShape,
    pub position: [f32; 3],
    pub half_extents: [f32; 3],
    pub radius: f32,
    pub damage_profile: HazardDamageProfile,
    pub active: bool,
    pub intensity: f32,
    pub pulsating: bool,
    pub pulse_frequency: f32,
    pub pulse_min: f32,
    pub pulse_max: f32,
    pub affected_layers: u32,
    pub warning_radius: f32,
    pub visual_intensity: f32,
    pub sound_id: Option<u32>,
}

impl HazardZone {
    pub fn new(id: HazardId, hazard_type: HazardType, position: [f32; 3], radius: f32) -> Self {
        let profile = match hazard_type {
            HazardType::Lava => HazardDamageProfile::lava(),
            HazardType::ToxicGas => HazardDamageProfile::toxic_gas(),
            HazardType::Radiation => HazardDamageProfile::radiation(),
            HazardType::ExtremeCold => HazardDamageProfile::extreme_cold(),
            HazardType::Electricity => HazardDamageProfile::electricity(),
            HazardType::Quicksand => HazardDamageProfile::quicksand(),
            HazardType::UnderwaterPressure => HazardDamageProfile::underwater_pressure(),
            _ => HazardDamageProfile::default(),
        };
        Self {
            id, hazard_type, shape: HazardShape::Sphere,
            position, half_extents: [radius; 3], radius,
            damage_profile: profile, active: true, intensity: 1.0,
            pulsating: false, pulse_frequency: 1.0, pulse_min: 0.5, pulse_max: 1.0,
            affected_layers: 0xFFFFFFFF, warning_radius: radius * 1.5,
            visual_intensity: 1.0, sound_id: None,
        }
    }

    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        if !self.active { return false; }
        match self.shape {
            HazardShape::Sphere => {
                let dx = point[0] - self.position[0];
                let dy = point[1] - self.position[1];
                let dz = point[2] - self.position[2];
                (dx*dx + dy*dy + dz*dz) <= self.radius * self.radius
            }
            HazardShape::Box => {
                (point[0] - self.position[0]).abs() <= self.half_extents[0] &&
                (point[1] - self.position[1]).abs() <= self.half_extents[1] &&
                (point[2] - self.position[2]).abs() <= self.half_extents[2]
            }
            _ => {
                let dx = point[0] - self.position[0];
                let dy = point[1] - self.position[1];
                let dz = point[2] - self.position[2];
                (dx*dx + dy*dy + dz*dz) <= self.radius * self.radius
            }
        }
    }

    pub fn in_warning_zone(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        (dx*dx + dy*dy + dz*dz) <= self.warning_radius * self.warning_radius
    }

    pub fn current_intensity(&self, time: f32) -> f32 {
        if !self.active { return 0.0; }
        if self.pulsating {
            let t = (time * self.pulse_frequency * std::f32::consts::TAU).sin() * 0.5 + 0.5;
            (self.pulse_min + (self.pulse_max - self.pulse_min) * t) * self.intensity
        } else {
            self.intensity
        }
    }

    pub fn damage_at(&self, time: f32, exposure_time: f32) -> f32 {
        let intensity = self.current_intensity(time);
        let mut dps = self.damage_profile.damage_per_second * intensity;
        if self.damage_profile.ramp_up_time > 0.0 {
            let ramp = (exposure_time / self.damage_profile.ramp_up_time).min(1.0);
            dps *= ramp;
        }
        dps
    }
}

#[derive(Debug, Clone)]
pub struct HazardExposure {
    pub entity_id: u64,
    pub hazard_id: HazardId,
    pub exposure_time: f32,
    pub stacks: u32,
    pub last_tick_time: f32,
    pub total_damage: f32,
}

#[derive(Debug, Clone)]
pub struct FallDamageConfig {
    pub enabled: bool,
    pub safe_height: f32,
    pub lethal_height: f32,
    pub damage_per_unit: f32,
    pub min_damage: f32,
    pub max_damage: f32,
    pub landing_stun_duration: f32,
    pub reduce_with_roll: bool,
    pub roll_reduction: f32,
}

impl Default for FallDamageConfig {
    fn default() -> Self {
        Self {
            enabled: true, safe_height: 3.0, lethal_height: 30.0,
            damage_per_unit: 5.0, min_damage: 0.0, max_damage: 1000.0,
            landing_stun_duration: 0.5, reduce_with_roll: true, roll_reduction: 0.5,
        }
    }
}

impl FallDamageConfig {
    pub fn calculate_damage(&self, fall_height: f32, rolled: bool) -> f32 {
        if !self.enabled || fall_height <= self.safe_height { return 0.0; }
        let effective = fall_height - self.safe_height;
        let mut damage = effective * self.damage_per_unit;
        if rolled && self.reduce_with_roll { damage *= 1.0 - self.roll_reduction; }
        damage.clamp(self.min_damage, self.max_damage)
    }

    pub fn is_lethal(&self, fall_height: f32) -> bool {
        fall_height >= self.lethal_height
    }
}

#[derive(Debug)]
pub struct HazardSystem {
    pub zones: HashMap<HazardId, HazardZone>,
    pub exposures: Vec<HazardExposure>,
    pub fall_damage: FallDamageConfig,
    pub time: f32,
    pub active: bool,
    next_id: u32,
}

impl HazardSystem {
    pub fn new() -> Self {
        Self {
            zones: HashMap::new(), exposures: Vec::new(),
            fall_damage: FallDamageConfig::default(),
            time: 0.0, active: true, next_id: 0,
        }
    }

    pub fn add_zone(&mut self, hazard_type: HazardType, position: [f32; 3], radius: f32) -> HazardId {
        let id = HazardId(self.next_id);
        self.next_id += 1;
        self.zones.insert(id, HazardZone::new(id, hazard_type, position, radius));
        id
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        for exp in &mut self.exposures {
            exp.exposure_time += dt;
            exp.last_tick_time += dt;
        }
    }

    pub fn check_entity(&self, entity_id: u64, position: [f32; 3]) -> Vec<(HazardId, f32)> {
        let mut results = Vec::new();
        for (id, zone) in &self.zones {
            if zone.contains_point(position) {
                let exposure = self.exposures.iter()
                    .find(|e| e.entity_id == entity_id && e.hazard_id == *id)
                    .map(|e| e.exposure_time).unwrap_or(0.0);
                let damage = zone.damage_at(self.time, exposure);
                results.push((*id, damage));
            }
        }
        results
    }

    pub fn zone_count(&self) -> usize { self.zones.len() }
    pub fn active_zone_count(&self) -> usize { self.zones.values().filter(|z| z.active).count() }
}

impl Default for HazardSystem { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hazard_zone_contains() {
        let zone = HazardZone::new(HazardId(0), HazardType::Lava, [0.0, 0.0, 0.0], 5.0);
        assert!(zone.contains_point([1.0, 0.0, 1.0]));
        assert!(!zone.contains_point([10.0, 0.0, 10.0]));
    }

    #[test]
    fn test_fall_damage() {
        let config = FallDamageConfig::default();
        assert_eq!(config.calculate_damage(2.0, false), 0.0);
        assert!(config.calculate_damage(10.0, false) > 0.0);
        let no_roll = config.calculate_damage(10.0, false);
        let with_roll = config.calculate_damage(10.0, true);
        assert!(with_roll < no_roll);
    }

    #[test]
    fn test_hazard_system() {
        let mut sys = HazardSystem::new();
        sys.add_zone(HazardType::Lava, [0.0, 0.0, 0.0], 10.0);
        assert_eq!(sys.zone_count(), 1);
        let hits = sys.check_entity(1, [1.0, 0.0, 1.0]);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].1 > 0.0);
    }
}
