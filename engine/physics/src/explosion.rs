// engine/physics/src/explosion.rs
//
// Explosion physics for the Genovo engine.
//
// Features:
// - Radial impulse from a point with configurable falloff
// - Damage falloff with distance (linear, inverse square, custom curves)
// - Shrapnel generation (physics-simulated debris fragments)
// - Blast wave (expanding sphere of force with pressure)
// - Explosion queries (find all entities within blast radius)
// - Crater formation (deform terrain/surfaces at detonation point)
//
// Explosions produce both instantaneous impulses (for rigid bodies)
// and sustained blast waves (for ongoing force over time).

use glam::{Quat, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Default explosion radius.
pub const DEFAULT_EXPLOSION_RADIUS: f32 = 10.0;

/// Default explosion force.
pub const DEFAULT_EXPLOSION_FORCE: f32 = 5000.0;

/// Speed of sound in air (m/s) for blast wave propagation.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Maximum number of shrapnel pieces per explosion.
pub const MAX_SHRAPNEL: usize = 128;

/// Default shrapnel speed range.
pub const SHRAPNEL_SPEED_MIN: f32 = 10.0;
pub const SHRAPNEL_SPEED_MAX: f32 = 50.0;

/// Default crater depth.
pub const DEFAULT_CRATER_DEPTH: f32 = 0.5;

// ---------------------------------------------------------------------------
// Damage falloff
// ---------------------------------------------------------------------------

/// How damage/force decreases with distance from the explosion centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DamageFalloff {
    /// No falloff (constant within radius).
    None,
    /// Linear falloff (1 at centre, 0 at radius).
    Linear,
    /// Inverse square law (physically accurate).
    InverseSquare,
    /// Exponential falloff.
    Exponential(f32),
    /// Custom power curve.
    Power(f32),
    /// Step function (full damage within inner radius, zero outside).
    Step(f32),
}

impl DamageFalloff {
    /// Compute the falloff factor at a given distance.
    pub fn evaluate(&self, distance: f32, radius: f32) -> f32 {
        if distance >= radius || radius < EPSILON {
            return 0.0;
        }
        let t = distance / radius;
        match self {
            Self::None => 1.0,
            Self::Linear => 1.0 - t,
            Self::InverseSquare => {
                let d = (distance + 1.0).max(1.0);
                let r = (radius + 1.0).max(1.0);
                (1.0 / (d * d)) / (1.0 / (r * r + EPSILON)).max(EPSILON)
            }
            Self::Exponential(rate) => (-distance * rate).exp(),
            Self::Power(p) => (1.0 - t).powf(*p),
            Self::Step(inner) => {
                if t <= *inner {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Explosion configuration
// ---------------------------------------------------------------------------

/// Configuration for an explosion event.
#[derive(Debug, Clone)]
pub struct ExplosionConfig {
    /// Radius of the explosion effect.
    pub radius: f32,
    /// Maximum force applied at the centre.
    pub force: f32,
    /// Maximum damage at the centre.
    pub damage: f32,
    /// Damage falloff mode.
    pub falloff: DamageFalloff,
    /// Upward bias: adds an upward component to the impulse (for dramatic effect).
    pub upward_bias: f32,
    /// Whether to apply force to the source entity (self-damage).
    pub self_damage: bool,
    /// Whether to generate shrapnel.
    pub generate_shrapnel: bool,
    /// Number of shrapnel pieces.
    pub shrapnel_count: u32,
    /// Shrapnel damage per piece.
    pub shrapnel_damage: f32,
    /// Whether to create a blast wave.
    pub create_blast_wave: bool,
    /// Blast wave duration in seconds.
    pub blast_wave_duration: f32,
    /// Blast wave peak overpressure.
    pub blast_wave_pressure: f32,
    /// Whether to create a crater.
    pub create_crater: bool,
    /// Crater radius.
    pub crater_radius: f32,
    /// Crater depth.
    pub crater_depth: f32,
    /// Visual effect type hint.
    pub visual_type: ExplosionVisualType,
    /// Sound intensity.
    pub sound_intensity: f32,
    /// Camera shake intensity.
    pub camera_shake: f32,
    /// Camera shake radius.
    pub camera_shake_radius: f32,
    /// Layers affected by this explosion.
    pub affected_layers: u32,
}

/// Visual type hint for the explosion VFX system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplosionVisualType {
    /// Standard fiery explosion.
    Fire,
    /// Plasma/energy explosion.
    Plasma,
    /// Smoke/concussive explosion.
    Concussive,
    /// Chemical/toxic explosion.
    Chemical,
    /// Nuclear/massive explosion.
    Nuclear,
    /// Electrical discharge.
    Electric,
    /// Ice/frost explosion.
    Ice,
    /// Custom (handle elsewhere).
    Custom,
}

impl Default for ExplosionConfig {
    fn default() -> Self {
        Self {
            radius: DEFAULT_EXPLOSION_RADIUS,
            force: DEFAULT_EXPLOSION_FORCE,
            damage: 100.0,
            falloff: DamageFalloff::InverseSquare,
            upward_bias: 0.3,
            self_damage: false,
            generate_shrapnel: false,
            shrapnel_count: 16,
            shrapnel_damage: 10.0,
            create_blast_wave: false,
            blast_wave_duration: 0.5,
            blast_wave_pressure: 100.0,
            create_crater: false,
            crater_radius: 2.0,
            crater_depth: DEFAULT_CRATER_DEPTH,
            visual_type: ExplosionVisualType::Fire,
            sound_intensity: 1.0,
            camera_shake: 1.0,
            camera_shake_radius: 50.0,
            affected_layers: 0xFFFFFFFF,
        }
    }
}

impl ExplosionConfig {
    /// Create a small grenade-like explosion.
    pub fn grenade() -> Self {
        Self {
            radius: 5.0,
            force: 3000.0,
            damage: 80.0,
            falloff: DamageFalloff::Linear,
            upward_bias: 0.4,
            generate_shrapnel: true,
            shrapnel_count: 24,
            shrapnel_damage: 15.0,
            create_crater: true,
            crater_radius: 1.0,
            crater_depth: 0.3,
            ..Default::default()
        }
    }

    /// Create a large rocket explosion.
    pub fn rocket() -> Self {
        Self {
            radius: 8.0,
            force: 6000.0,
            damage: 150.0,
            falloff: DamageFalloff::Power(1.5),
            upward_bias: 0.2,
            generate_shrapnel: true,
            shrapnel_count: 32,
            create_blast_wave: true,
            blast_wave_duration: 0.3,
            create_crater: true,
            crater_radius: 2.5,
            crater_depth: 0.5,
            camera_shake: 2.0,
            ..Default::default()
        }
    }

    /// Create a massive C4/demolition explosion.
    pub fn demolition() -> Self {
        Self {
            radius: 15.0,
            force: 15000.0,
            damage: 300.0,
            falloff: DamageFalloff::InverseSquare,
            upward_bias: 0.5,
            generate_shrapnel: true,
            shrapnel_count: 64,
            shrapnel_damage: 25.0,
            create_blast_wave: true,
            blast_wave_duration: 0.8,
            blast_wave_pressure: 200.0,
            create_crater: true,
            crater_radius: 5.0,
            crater_depth: 1.5,
            camera_shake: 5.0,
            camera_shake_radius: 100.0,
            ..Default::default()
        }
    }

    /// Create an energy/plasma explosion (no shrapnel, pure force).
    pub fn energy_blast() -> Self {
        Self {
            radius: 12.0,
            force: 8000.0,
            damage: 120.0,
            falloff: DamageFalloff::Exponential(0.3),
            upward_bias: 0.0,
            visual_type: ExplosionVisualType::Plasma,
            create_blast_wave: true,
            blast_wave_duration: 0.6,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Explosion result
// ---------------------------------------------------------------------------

/// Result of an explosion affecting a single body/entity.
#[derive(Debug, Clone)]
pub struct ExplosionHit {
    /// Entity/body ID.
    pub entity_id: u64,
    /// World position of the entity.
    pub position: Vec3,
    /// Distance from explosion centre.
    pub distance: f32,
    /// Direction from explosion to entity (normalised).
    pub direction: Vec3,
    /// Impulse applied.
    pub impulse: Vec3,
    /// Damage dealt.
    pub damage: f32,
    /// Falloff factor at this distance.
    pub falloff_factor: f32,
    /// Whether the entity is blocked by an obstacle (line-of-sight check).
    pub blocked: bool,
}

/// A shrapnel piece generated by an explosion.
#[derive(Debug, Clone)]
pub struct ShrapnelPiece {
    /// World position (starts at explosion centre).
    pub position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Damage on impact.
    pub damage: f32,
    /// Mass of the piece.
    pub mass: f32,
    /// Size (radius for collision).
    pub size: f32,
    /// Remaining lifetime in seconds.
    pub lifetime: f32,
    /// Maximum lifetime.
    pub max_lifetime: f32,
    /// Whether this piece has hit something.
    pub hit: bool,
    /// Layer mask for collision.
    pub layer_mask: u32,
}

/// Complete result of detonating an explosion.
#[derive(Debug)]
pub struct ExplosionResult {
    /// All entities hit by the explosion.
    pub hits: Vec<ExplosionHit>,
    /// Generated shrapnel pieces.
    pub shrapnel: Vec<ShrapnelPiece>,
    /// Blast wave (if created).
    pub blast_wave: Option<BlastWave>,
    /// Crater info (if created).
    pub crater: Option<Crater>,
    /// Total damage dealt.
    pub total_damage: f32,
    /// Number of entities affected.
    pub entities_affected: u32,
}

// ---------------------------------------------------------------------------
// Explosion detonation
// ---------------------------------------------------------------------------

/// An explosion event that can be detonated.
#[derive(Debug, Clone)]
pub struct Explosion {
    /// Configuration.
    pub config: ExplosionConfig,
    /// Detonation position.
    pub position: Vec3,
    /// Source entity ID (for self-damage filtering).
    pub source_entity: Option<u64>,
    /// Time the explosion was created.
    pub creation_time: f64,
    /// Whether the explosion has been detonated.
    pub detonated: bool,
}

impl Explosion {
    /// Create a new explosion at a position.
    pub fn new(position: Vec3, config: ExplosionConfig) -> Self {
        Self {
            config,
            position,
            source_entity: None,
            creation_time: 0.0,
            detonated: false,
        }
    }

    /// Set the source entity.
    pub fn with_source(mut self, entity_id: u64) -> Self {
        self.source_entity = Some(entity_id);
        self
    }

    /// Compute the impulse for a body at a given position.
    pub fn compute_impulse(&self, target_pos: Vec3, target_mass: f32) -> Vec3 {
        let delta = target_pos - self.position;
        let distance = delta.length();

        if distance >= self.config.radius || distance < EPSILON {
            return Vec3::ZERO;
        }

        let direction = delta / distance;
        let falloff = self.config.falloff.evaluate(distance, self.config.radius);

        let mut impulse_dir = direction;
        // Add upward bias
        impulse_dir.y += self.config.upward_bias;
        impulse_dir = impulse_dir.normalize_or_zero();

        impulse_dir * self.config.force * falloff / target_mass.max(0.1)
    }

    /// Compute the damage at a given distance.
    pub fn compute_damage(&self, distance: f32) -> f32 {
        let falloff = self.config.falloff.evaluate(distance, self.config.radius);
        self.config.damage * falloff
    }

    /// Generate shrapnel pieces.
    pub fn generate_shrapnel(&self, rng: &mut SimpleRng) -> Vec<ShrapnelPiece> {
        if !self.config.generate_shrapnel {
            return Vec::new();
        }

        let count = (self.config.shrapnel_count as usize).min(MAX_SHRAPNEL);
        let mut pieces = Vec::with_capacity(count);

        for _ in 0..count {
            // Random direction on unit sphere
            let theta = rng.next_f32() * std::f32::consts::TAU;
            let phi = (rng.next_f32() * 2.0 - 1.0).acos();
            let dir = Vec3::new(
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            );

            let speed = SHRAPNEL_SPEED_MIN + rng.next_f32() * (SHRAPNEL_SPEED_MAX - SHRAPNEL_SPEED_MIN);
            let lifetime = 1.0 + rng.next_f32() * 3.0;

            pieces.push(ShrapnelPiece {
                position: self.position,
                velocity: dir * speed,
                damage: self.config.shrapnel_damage,
                mass: 0.05 + rng.next_f32() * 0.2,
                size: 0.02 + rng.next_f32() * 0.05,
                lifetime,
                max_lifetime: lifetime,
                hit: false,
                layer_mask: self.config.affected_layers,
            });
        }

        pieces
    }

    /// Query all entities within the blast radius.
    pub fn query_entities(
        &self,
        entities: &[(u64, Vec3, f32)], // (id, position, mass)
    ) -> Vec<ExplosionHit> {
        let mut hits = Vec::new();

        for &(id, pos, mass) in entities {
            if !self.config.self_damage {
                if let Some(source) = self.source_entity {
                    if id == source {
                        continue;
                    }
                }
            }

            let delta = pos - self.position;
            let distance = delta.length();

            if distance >= self.config.radius {
                continue;
            }

            let direction = if distance > EPSILON {
                delta / distance
            } else {
                Vec3::Y
            };

            let falloff = self.config.falloff.evaluate(distance, self.config.radius);
            let impulse = self.compute_impulse(pos, mass);
            let damage = self.compute_damage(distance);

            hits.push(ExplosionHit {
                entity_id: id,
                position: pos,
                distance,
                direction,
                impulse,
                damage,
                falloff_factor: falloff,
                blocked: false,
            });
        }

        // Sort by distance (closest first)
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }
}

// ---------------------------------------------------------------------------
// Blast wave
// ---------------------------------------------------------------------------

/// An expanding sphere of force that propagates outward over time.
#[derive(Debug, Clone)]
pub struct BlastWave {
    /// Centre of the blast wave.
    pub origin: Vec3,
    /// Current radius of the expanding wave front.
    pub current_radius: f32,
    /// Maximum radius the wave will reach.
    pub max_radius: f32,
    /// Expansion speed (typically speed of sound or faster).
    pub speed: f32,
    /// Current overpressure at the wave front.
    pub pressure: f32,
    /// Peak overpressure (at origin).
    pub peak_pressure: f32,
    /// Duration of the blast wave.
    pub duration: f32,
    /// Elapsed time since detonation.
    pub elapsed: f32,
    /// Wave front thickness.
    pub thickness: f32,
    /// Whether the blast wave is still active.
    pub active: bool,
}

impl BlastWave {
    /// Create a new blast wave.
    pub fn new(origin: Vec3, max_radius: f32, peak_pressure: f32, duration: f32) -> Self {
        Self {
            origin,
            current_radius: 0.0,
            max_radius,
            speed: SPEED_OF_SOUND * 2.0,
            pressure: peak_pressure,
            peak_pressure,
            duration,
            elapsed: 0.0,
            thickness: 1.0,
            active: true,
        }
    }

    /// Update the blast wave.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        self.elapsed += dt;
        self.current_radius += self.speed * dt;

        // Pressure decreases with distance and time
        let time_factor = 1.0 - (self.elapsed / self.duration).min(1.0);
        let distance_factor = if self.current_radius > EPSILON {
            1.0 / (self.current_radius * self.current_radius)
        } else {
            1.0
        };
        self.pressure = self.peak_pressure * time_factor * distance_factor * self.max_radius * self.max_radius;

        if self.elapsed >= self.duration || self.current_radius >= self.max_radius {
            self.active = false;
        }
    }

    /// Get the force applied to a body at a given position.
    pub fn force_at(&self, position: Vec3, mass: f32) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }

        let delta = position - self.origin;
        let distance = delta.length();

        // Only apply force at the wave front
        let dist_from_front = (distance - self.current_radius).abs();
        if dist_from_front > self.thickness {
            return Vec3::ZERO;
        }

        let front_factor = 1.0 - dist_from_front / self.thickness;
        let direction = if distance > EPSILON {
            delta / distance
        } else {
            Vec3::Y
        };

        direction * self.pressure * front_factor / mass.max(0.1)
    }

    /// Check if a position is within the wave front.
    pub fn is_at_front(&self, position: Vec3) -> bool {
        let distance = (position - self.origin).length();
        (distance - self.current_radius).abs() < self.thickness
    }
}

// ---------------------------------------------------------------------------
// Crater
// ---------------------------------------------------------------------------

/// Information about a crater formed by an explosion.
#[derive(Debug, Clone)]
pub struct Crater {
    /// Centre position.
    pub position: Vec3,
    /// Crater radius.
    pub radius: f32,
    /// Maximum depth at the centre.
    pub depth: f32,
    /// Rim height (raised earth around the crater).
    pub rim_height: f32,
    /// Rim width.
    pub rim_width: f32,
    /// Whether the crater has been applied to the terrain.
    pub applied: bool,
}

impl Crater {
    /// Create a new crater.
    pub fn new(position: Vec3, radius: f32, depth: f32) -> Self {
        Self {
            position,
            radius,
            depth,
            rim_height: depth * 0.3,
            rim_width: radius * 0.3,
            applied: false,
        }
    }

    /// Get the height displacement at a given distance from the centre.
    pub fn displacement_at(&self, distance: f32) -> f32 {
        if distance > self.radius + self.rim_width {
            return 0.0;
        }

        if distance <= self.radius {
            // Inside the crater: parabolic depression
            let t = distance / self.radius;
            -self.depth * (1.0 - t * t)
        } else {
            // Rim area
            let rim_t = (distance - self.radius) / self.rim_width;
            let rim = self.rim_height * (1.0 - rim_t) * (1.0 - rim_t);
            rim
        }
    }

    /// Get the total volume displaced by the crater (approximate).
    pub fn volume(&self) -> f32 {
        // Volume of paraboloid
        0.5 * std::f32::consts::PI * self.radius * self.radius * self.depth
    }

    /// Mark the crater as applied to the terrain.
    pub fn mark_applied(&mut self) {
        self.applied = true;
    }
}

// ---------------------------------------------------------------------------
// Simple RNG (for shrapnel generation)
// ---------------------------------------------------------------------------

/// Minimal pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
        }
    }

    /// Generate a random u32.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

// ---------------------------------------------------------------------------
// Explosion system (ECS)
// ---------------------------------------------------------------------------

/// The explosion system manages pending and active explosions.
#[derive(Debug)]
pub struct ExplosionSystem {
    /// Pending explosions to detonate.
    pub pending: Vec<Explosion>,
    /// Active blast waves.
    pub blast_waves: Vec<BlastWave>,
    /// Active shrapnel pieces.
    pub shrapnel: Vec<ShrapnelPiece>,
    /// Pending craters.
    pub pending_craters: Vec<Crater>,
    /// RNG for shrapnel generation.
    pub rng: SimpleRng,
    /// Gravity for shrapnel simulation.
    pub gravity: Vec3,
    /// Whether the system is active.
    pub active: bool,
    /// Statistics.
    pub stats: ExplosionStats,
}

/// Explosion system statistics.
#[derive(Debug, Clone, Default)]
pub struct ExplosionStats {
    /// Explosions detonated this frame.
    pub detonations: u32,
    /// Active blast waves.
    pub active_blast_waves: u32,
    /// Active shrapnel count.
    pub active_shrapnel: u32,
    /// Pending craters.
    pub pending_crater_count: u32,
    /// Total entities damaged this frame.
    pub entities_damaged: u32,
}

impl ExplosionSystem {
    /// Create a new explosion system.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            blast_waves: Vec::new(),
            shrapnel: Vec::new(),
            pending_craters: Vec::new(),
            rng: SimpleRng::new(12345),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            active: true,
            stats: ExplosionStats::default(),
        }
    }

    /// Queue an explosion for detonation.
    pub fn queue_explosion(&mut self, explosion: Explosion) {
        self.pending.push(explosion);
    }

    /// Convenience: create and queue an explosion at a position.
    pub fn explode(&mut self, position: Vec3, config: ExplosionConfig) {
        self.pending.push(Explosion::new(position, config));
    }

    /// Detonate all pending explosions against the given entity list.
    pub fn detonate_all(
        &mut self,
        entities: &[(u64, Vec3, f32)],
    ) -> Vec<ExplosionResult> {
        let mut results = Vec::new();
        self.stats = ExplosionStats::default();

        let pending = std::mem::take(&mut self.pending);
        for explosion in pending {
            let hits = explosion.query_entities(entities);
            let shrapnel = explosion.generate_shrapnel(&mut self.rng);

            let blast_wave = if explosion.config.create_blast_wave {
                let bw = BlastWave::new(
                    explosion.position,
                    explosion.config.radius * 2.0,
                    explosion.config.blast_wave_pressure,
                    explosion.config.blast_wave_duration,
                );
                self.blast_waves.push(bw.clone());
                Some(bw)
            } else {
                None
            };

            let crater = if explosion.config.create_crater {
                let c = Crater::new(
                    explosion.position,
                    explosion.config.crater_radius,
                    explosion.config.crater_depth,
                );
                self.pending_craters.push(c.clone());
                Some(c)
            } else {
                None
            };

            let total_damage: f32 = hits.iter().map(|h| h.damage).sum();
            let entities_affected = hits.len() as u32;

            self.shrapnel.extend(shrapnel.iter().cloned());

            results.push(ExplosionResult {
                hits,
                shrapnel,
                blast_wave,
                crater,
                total_damage,
                entities_affected,
            });

            self.stats.detonations += 1;
            self.stats.entities_damaged += entities_affected;
        }

        self.stats.active_blast_waves = self.blast_waves.len() as u32;
        self.stats.active_shrapnel = self.shrapnel.len() as u32;
        self.stats.pending_crater_count = self.pending_craters.len() as u32;

        results
    }

    /// Update active blast waves and shrapnel.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        // Update blast waves
        for wave in &mut self.blast_waves {
            wave.update(dt);
        }
        self.blast_waves.retain(|w| w.active);

        // Update shrapnel
        for piece in &mut self.shrapnel {
            if piece.hit {
                continue;
            }
            piece.velocity += self.gravity * dt;
            piece.position += piece.velocity * dt;
            piece.lifetime -= dt;
        }
        self.shrapnel.retain(|s| s.lifetime > 0.0 && !s.hit);

        // Update stats
        self.stats.active_blast_waves = self.blast_waves.len() as u32;
        self.stats.active_shrapnel = self.shrapnel.len() as u32;
    }

    /// Get the total blast wave force at a position.
    pub fn blast_force_at(&self, position: Vec3, mass: f32) -> Vec3 {
        let mut total = Vec3::ZERO;
        for wave in &self.blast_waves {
            total += wave.force_at(position, mass);
        }
        total
    }

    /// Clear all active effects.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.blast_waves.clear();
        self.shrapnel.clear();
        self.pending_craters.clear();
    }
}

impl Default for ExplosionSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// ECS component for an entity that can explode.
#[derive(Debug, Clone)]
pub struct ExplosionComponent {
    /// Explosion configuration.
    pub config: ExplosionConfig,
    /// Whether to auto-detonate on destruction.
    pub auto_detonate: bool,
    /// Delay before detonation (seconds).
    pub delay: f32,
    /// Whether the explosion has been triggered.
    pub triggered: bool,
    /// Countdown timer.
    pub timer: f32,
}

impl ExplosionComponent {
    /// Create a new explosion component.
    pub fn new(config: ExplosionConfig) -> Self {
        Self {
            config,
            auto_detonate: true,
            delay: 0.0,
            triggered: false,
            timer: 0.0,
        }
    }

    /// Trigger the explosion with an optional delay.
    pub fn trigger(&mut self, delay: f32) {
        self.triggered = true;
        self.delay = delay;
        self.timer = 0.0;
    }

    /// Update the component (check if delay has elapsed).
    pub fn update(&mut self, dt: f32) -> bool {
        if !self.triggered {
            return false;
        }
        self.timer += dt;
        self.timer >= self.delay
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_falloff() {
        assert!((DamageFalloff::None.evaluate(5.0, 10.0) - 1.0).abs() < EPSILON);
        assert!((DamageFalloff::Linear.evaluate(0.0, 10.0) - 1.0).abs() < EPSILON);
        assert!((DamageFalloff::Linear.evaluate(10.0, 10.0) - 0.0).abs() < EPSILON);
        assert!((DamageFalloff::Linear.evaluate(5.0, 10.0) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_explosion_impulse() {
        let explosion = Explosion::new(Vec3::ZERO, ExplosionConfig::default());
        let impulse = explosion.compute_impulse(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!(impulse.x > 0.0); // Should push away from centre
    }

    #[test]
    fn test_explosion_query() {
        let explosion = Explosion::new(Vec3::ZERO, ExplosionConfig {
            radius: 10.0,
            ..Default::default()
        });
        let entities = vec![
            (1, Vec3::new(3.0, 0.0, 0.0), 1.0),
            (2, Vec3::new(20.0, 0.0, 0.0), 1.0),
        ];
        let hits = explosion.query_entities(&entities);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entity_id, 1);
    }

    #[test]
    fn test_blast_wave() {
        let mut wave = BlastWave::new(Vec3::ZERO, 20.0, 100.0, 0.5);
        wave.update(0.1);
        assert!(wave.current_radius > 0.0);
        assert!(wave.active);
    }

    #[test]
    fn test_crater() {
        let crater = Crater::new(Vec3::ZERO, 5.0, 2.0);
        assert!((crater.displacement_at(0.0) - (-2.0)).abs() < EPSILON);
        assert!(crater.displacement_at(5.0).abs() < EPSILON);
        assert!(crater.displacement_at(5.5) > 0.0); // Rim
        assert!(crater.volume() > 0.0);
    }

    #[test]
    fn test_explosion_system() {
        let mut sys = ExplosionSystem::new();
        sys.explode(Vec3::ZERO, ExplosionConfig::grenade());
        let results = sys.detonate_all(&[(1, Vec3::new(2.0, 0.0, 0.0), 1.0)]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entities_affected, 1);
    }

    #[test]
    fn test_shrapnel_generation() {
        let explosion = Explosion::new(Vec3::ZERO, ExplosionConfig::grenade());
        let mut rng = SimpleRng::new(42);
        let shrapnel = explosion.generate_shrapnel(&mut rng);
        assert_eq!(shrapnel.len(), 24);
        for piece in &shrapnel {
            assert!(piece.velocity.length() > 0.0);
        }
    }
}
