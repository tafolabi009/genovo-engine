// engine/render/src/particles/system.rs
//
// High-level particle system component and manager. A `ParticleSystem` owns
// an emitter, particle pool, force fields, collision world, and renderer
// settings. The `ParticleSystemManager` manages all active particle systems
// and orchestrates updates and rendering.

use glam::Vec3;
use super::collision::{CollisionEvent, CollisionWorld};
use super::emitter::ParticleEmitter;
use super::forces::ForceFieldSet;
use super::particle::{ParticlePool, SortMode};
use super::renderer::ParticleRenderer;
use super::{ColorGradient, Curve};

// ---------------------------------------------------------------------------
// PlaybackState
// ---------------------------------------------------------------------------

/// The lifecycle state of a particle system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaybackState {
    /// System is actively emitting and simulating.
    Playing,
    /// System is paused (no simulation, no emission).
    Paused,
    /// System has stopped emitting but existing particles continue to live.
    Stopping,
    /// System is fully stopped (no particles alive, no emission).
    Stopped,
}

impl Default for PlaybackState {
    fn default() -> Self {
        PlaybackState::Stopped
    }
}

// ---------------------------------------------------------------------------
// SubEmitterTrigger
// ---------------------------------------------------------------------------

/// When a sub-emitter should fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubEmitterTrigger {
    /// Fire when a parent particle is born.
    Birth,
    /// Fire when a parent particle dies.
    Death,
    /// Fire when a parent particle collides.
    Collision,
}

/// A sub-emitter configuration: a reference to another emitter that spawns
/// particles in response to parent particle events.
#[derive(Debug, Clone)]
pub struct SubEmitterConfig {
    /// When this sub-emitter fires.
    pub trigger: SubEmitterTrigger,
    /// The sub-emitter template.
    pub emitter: ParticleEmitter,
    /// Number of particles to burst from the sub-emitter.
    pub burst_count: u32,
    /// Whether the sub-emitter inherits the parent particle's velocity.
    pub inherit_velocity: f32,
}

// ---------------------------------------------------------------------------
// LodSettings
// ---------------------------------------------------------------------------

/// Level-of-detail settings for particle systems.
#[derive(Debug, Clone, Copy)]
pub struct LodSettings {
    /// If `true`, LOD is enabled.
    pub enabled: bool,
    /// Distance at which particle count begins to be reduced.
    pub start_distance: f32,
    /// Distance at which particle count is at minimum.
    pub end_distance: f32,
    /// Minimum fraction of particles to keep (0.0 = can reduce to zero).
    pub min_fraction: f32,
}

impl LodSettings {
    /// Computes the fraction of particles to keep at a given distance.
    pub fn compute_fraction(&self, distance: f32) -> f32 {
        if !self.enabled || distance <= self.start_distance {
            return 1.0;
        }
        if distance >= self.end_distance {
            return self.min_fraction;
        }
        let t = (distance - self.start_distance)
            / (self.end_distance - self.start_distance);
        1.0 - t * (1.0 - self.min_fraction)
    }
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            start_distance: 50.0,
            end_distance: 200.0,
            min_fraction: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleSystem
// ---------------------------------------------------------------------------

/// A complete particle system: emitter + pool + forces + collision + renderer.
///
/// This is the primary user-facing type. Create one, configure it, play it,
/// and call `update` each frame.
pub struct ParticleSystem {
    /// A unique name for debugging.
    pub name: String,

    // -- Components --
    /// The particle emitter.
    pub emitter: ParticleEmitter,
    /// The particle pool.
    pub pool: ParticlePool,
    /// Active force fields.
    pub forces: ForceFieldSet,
    /// Collision world.
    pub collision: CollisionWorld,
    /// Renderer configuration.
    pub renderer: ParticleRenderer,
    /// Sub-emitter configurations.
    pub sub_emitters: Vec<SubEmitterConfig>,

    // -- Settings --
    /// Gravity vector for this system.
    pub gravity: Vec3,
    /// Size-over-lifetime curve.
    pub size_over_lifetime: Curve,
    /// Color-over-lifetime gradient.
    pub color_over_lifetime: ColorGradient,
    /// Speed-over-lifetime curve.
    pub speed_over_lifetime: Curve,
    /// Particle sort mode.
    pub sort_mode: SortMode,
    /// LOD settings.
    pub lod: LodSettings,
    /// Maximum duration of the system in seconds. 0 = infinite.
    pub duration: f32,
    /// If `true`, the system loops after `duration`.
    pub looping: bool,
    /// Warmup time: simulate this many seconds at startup.
    pub warmup_time: f32,
    /// Simulation time scale (1.0 = normal, 0.5 = half speed).
    pub time_scale: f32,

    // -- State --
    /// Current playback state.
    state: PlaybackState,
    /// Total elapsed time.
    elapsed: f32,
    /// Whether warmup has been performed.
    warmed_up: bool,
    /// Collision events from the last frame.
    collision_events: Vec<CollisionEvent>,
    /// Pending sub-emitter spawns from the last frame.
    pending_sub_spawns: Vec<(Vec3, Vec3)>, // (position, velocity)
}

impl ParticleSystem {
    /// Creates a new particle system with the given emitter.
    pub fn new(name: impl Into<String>, emitter: ParticleEmitter) -> Self {
        let capacity = emitter.max_particles as usize;
        Self {
            name: name.into(),
            emitter,
            pool: ParticlePool::new(capacity),
            forces: ForceFieldSet::new(),
            collision: CollisionWorld::new(),
            renderer: ParticleRenderer::default(),
            sub_emitters: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            size_over_lifetime: Curve::constant(1.0),
            color_over_lifetime: ColorGradient::default(),
            speed_over_lifetime: Curve::constant(1.0),
            sort_mode: SortMode::None,
            lod: LodSettings::default(),
            duration: 0.0,
            looping: true,
            warmup_time: 0.0,
            time_scale: 1.0,
            state: PlaybackState::Stopped,
            elapsed: 0.0,
            warmed_up: false,
            collision_events: Vec::new(),
            pending_sub_spawns: Vec::new(),
        }
    }

    /// Starts or resumes playback.
    pub fn play(&mut self) {
        if self.state == PlaybackState::Stopped {
            self.elapsed = 0.0;
            self.emitter.reset();
            self.pool.clear();
            self.warmed_up = false;
        }
        self.state = PlaybackState::Playing;

        // Perform warmup if needed.
        if !self.warmed_up && self.warmup_time > 0.0 {
            self.warmup();
            self.warmed_up = true;
        }
    }

    /// Pauses the system (no simulation or emission).
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
        }
    }

    /// Stops emission but lets existing particles finish their lifetime.
    pub fn stop(&mut self) {
        if self.state == PlaybackState::Playing || self.state == PlaybackState::Paused {
            self.state = PlaybackState::Stopping;
        }
    }

    /// Immediately stops and clears all particles.
    pub fn stop_and_clear(&mut self) {
        self.state = PlaybackState::Stopped;
        self.pool.clear();
        self.elapsed = 0.0;
    }

    /// Restarts the system from the beginning.
    pub fn restart(&mut self) {
        self.stop_and_clear();
        self.play();
    }

    /// Returns the current playback state.
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Returns `true` if the system is actively playing.
    pub fn is_playing(&self) -> bool {
        self.state == PlaybackState::Playing
    }

    /// Returns `true` if there are any alive particles.
    pub fn has_particles(&self) -> bool {
        self.pool.alive() > 0
    }

    /// Returns the number of alive particles.
    pub fn particle_count(&self) -> usize {
        self.pool.alive()
    }

    /// Returns the elapsed time.
    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    /// Returns collision events from the last frame.
    pub fn collision_events(&self) -> &[CollisionEvent] {
        &self.collision_events
    }

    /// Performs warmup by simulating the system at a fixed timestep for
    /// `self.warmup_time` seconds.
    fn warmup(&mut self) {
        let warmup_dt = 1.0 / 60.0; // 60 Hz warmup.
        let steps = (self.warmup_time / warmup_dt).ceil() as u32;
        for _ in 0..steps {
            self.step(warmup_dt);
        }
    }

    /// Updates the particle system for one frame.
    ///
    /// This is the main entry point called once per frame.
    ///
    /// # Arguments
    /// * `dt` - Frame delta time in seconds.
    /// * `camera_pos` - World-space camera position (for sorting and culling).
    /// * `frustum_check` - Optional closure that returns `true` if the system
    ///   is within the camera frustum.
    pub fn update(
        &mut self,
        dt: f32,
        camera_pos: Vec3,
        frustum_visible: bool,
    ) {
        if self.state == PlaybackState::Stopped || self.state == PlaybackState::Paused {
            return;
        }

        let dt = dt * self.time_scale;

        // Check duration.
        if self.duration > 0.0 && self.elapsed >= self.duration {
            if self.looping {
                self.elapsed = 0.0;
                self.emitter.reset();
            } else {
                self.state = PlaybackState::Stopping;
            }
        }

        // Culling: skip simulation if outside frustum (but still age
        // particles so they die correctly).
        if !frustum_visible && self.lod.enabled {
            // Just age particles, don't do full simulation.
            self.pool.update_simple(dt, Vec3::ZERO);
            self.elapsed += dt;

            if self.state == PlaybackState::Stopping && self.pool.is_empty() {
                self.state = PlaybackState::Stopped;
            }
            return;
        }

        // LOD: reduce emission rate based on distance.
        let distance = (self.emitter.position - camera_pos).length();
        let lod_fraction = self.lod.compute_fraction(distance);

        self.step_with_lod(dt, lod_fraction);
        self.elapsed += dt;

        // Sort particles for rendering.
        if self.sort_mode != SortMode::None {
            self.pool.sort(self.sort_mode, camera_pos);
        }

        // Transition to stopped if draining and empty.
        if self.state == PlaybackState::Stopping && self.pool.is_empty() {
            self.state = PlaybackState::Stopped;
        }
    }

    /// Performs one simulation step.
    fn step(&mut self, dt: f32) {
        self.step_with_lod(dt, 1.0);
    }

    /// Performs one simulation step with LOD-based emission reduction.
    fn step_with_lod(&mut self, dt: f32, lod_fraction: f32) {
        // Record particle count before spawn (for birth sub-emitters).
        let alive_before = self.pool.alive();

        // Emit new particles (only if playing, not stopping).
        if self.state == PlaybackState::Playing {
            let spawns = self.emitter.emit(dt, self.pool.alive());

            // Apply LOD: randomly skip some spawns.
            let spawn_count = if lod_fraction < 1.0 {
                ((spawns.len() as f32 * lod_fraction).round() as usize)
                    .min(spawns.len())
            } else {
                spawns.len()
            };

            for i in 0..spawn_count {
                self.pool.spawn(&spawns[i]);
            }
        }

        // Process birth sub-emitters.
        if !self.sub_emitters.is_empty() {
            let alive_after = self.pool.alive();
            for i in alive_before..alive_after {
                for sub in &self.sub_emitters {
                    if sub.trigger == SubEmitterTrigger::Birth {
                        self.pending_sub_spawns.push((
                            self.pool.positions[i],
                            self.pool.velocities[i] * sub.inherit_velocity,
                        ));
                    }
                }
            }
        }

        // Apply force fields.
        self.forces.apply_to_pool(&mut self.pool, dt, self.elapsed);

        // Integrate particles.
        let gravity = self.gravity * self.emitter.gravity_modifier;
        self.pool.update(
            dt,
            gravity,
            &self.size_over_lifetime,
            &self.color_over_lifetime,
            &self.speed_over_lifetime,
        );

        // Detect collisions.
        self.collision_events.clear();
        self.collision.resolve(
            &mut self.pool,
            dt,
            Some(&mut self.collision_events),
        );

        // Process collision sub-emitters.
        if !self.sub_emitters.is_empty() {
            for event in &self.collision_events {
                for sub in &self.sub_emitters {
                    if sub.trigger == SubEmitterTrigger::Collision {
                        self.pending_sub_spawns.push((
                            event.position,
                            event.velocity * sub.inherit_velocity,
                        ));
                    }
                }
            }
        }
    }

    /// Generates render data for this particle system.
    pub fn prepare_render(
        &mut self,
        camera_pos: Vec3,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_fwd: Vec3,
    ) {
        self.renderer.generate_billboards(
            &self.pool,
            camera_pos,
            camera_right,
            camera_up,
            camera_fwd,
        );
    }

    /// Sets the emitter position.
    pub fn set_position(&mut self, pos: Vec3) {
        self.emitter.set_position(pos);
    }

    /// Returns the bounding radius of the system.
    pub fn bounds_radius(&self) -> f32 {
        self.emitter.estimated_bounds_radius()
    }

    /// Takes pending sub-emitter spawns (drains the buffer).
    pub fn take_sub_spawns(&mut self) -> Vec<(Vec3, Vec3)> {
        std::mem::take(&mut self.pending_sub_spawns)
    }

    /// Sets the gravity vector.
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the duration.
    pub fn with_duration(mut self, duration: f32, looping: bool) -> Self {
        self.duration = duration;
        self.looping = looping;
        self
    }

    /// Sets warmup time.
    pub fn with_warmup(mut self, warmup: f32) -> Self {
        self.warmup_time = warmup;
        self
    }

    /// Sets LOD settings.
    pub fn with_lod(mut self, lod: LodSettings) -> Self {
        self.lod = lod;
        self
    }

    /// Sets the sort mode.
    pub fn with_sort_mode(mut self, mode: SortMode) -> Self {
        self.sort_mode = mode;
        self
    }

    /// Sets the time scale.
    pub fn with_time_scale(mut self, scale: f32) -> Self {
        self.time_scale = scale;
        self
    }

    /// Sets the size-over-lifetime curve.
    pub fn with_size_over_lifetime(mut self, curve: Curve) -> Self {
        self.size_over_lifetime = curve;
        self
    }

    /// Sets the color-over-lifetime gradient.
    pub fn with_color_over_lifetime(mut self, gradient: ColorGradient) -> Self {
        self.color_over_lifetime = gradient;
        self
    }

    /// Adds a sub-emitter.
    pub fn add_sub_emitter(&mut self, config: SubEmitterConfig) {
        self.sub_emitters.push(config);
    }
}

// ---------------------------------------------------------------------------
// ParticleSystemManager
// ---------------------------------------------------------------------------

/// Manages all active particle systems in the scene.
///
/// Provides batch update, frustum culling, and system lifecycle management.
pub struct ParticleSystemManager {
    /// All managed particle systems.
    systems: Vec<ParticleSystem>,
    /// Systems pending removal (indices).
    pending_removal: Vec<usize>,
    /// Global time elapsed.
    global_time: f32,
    /// Whether automatic cleanup of stopped systems is enabled.
    pub auto_cleanup: bool,
}

impl ParticleSystemManager {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            pending_removal: Vec::new(),
            global_time: 0.0,
            auto_cleanup: true,
        }
    }

    /// Adds a particle system and starts it playing.
    pub fn add(&mut self, mut system: ParticleSystem) -> usize {
        system.play();
        let id = self.systems.len();
        self.systems.push(system);
        id
    }

    /// Adds a system without starting it.
    pub fn add_stopped(&mut self, system: ParticleSystem) -> usize {
        let id = self.systems.len();
        self.systems.push(system);
        id
    }

    /// Returns a reference to a system by index.
    pub fn get(&self, index: usize) -> Option<&ParticleSystem> {
        self.systems.get(index)
    }

    /// Returns a mutable reference to a system by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ParticleSystem> {
        self.systems.get_mut(index)
    }

    /// Returns the number of managed systems.
    pub fn count(&self) -> usize {
        self.systems.len()
    }

    /// Returns the total number of alive particles across all systems.
    pub fn total_particles(&self) -> usize {
        self.systems.iter().map(|s| s.particle_count()).sum()
    }

    /// Updates all particle systems.
    ///
    /// # Arguments
    /// * `dt` - Frame delta time.
    /// * `camera_pos` - Camera position for sorting and culling.
    /// * `frustum` - Optional frustum for culling.
    pub fn update(&mut self, dt: f32, camera_pos: Vec3) {
        self.global_time += dt;
        self.pending_removal.clear();

        for (i, system) in self.systems.iter_mut().enumerate() {
            // Simple frustum check: compare distance to bounds.
            let visible = true; // Full frustum culling would go here.
            system.update(dt, camera_pos, visible);

            // Mark stopped non-looping systems for removal.
            if self.auto_cleanup
                && system.state() == PlaybackState::Stopped
                && !system.looping
            {
                self.pending_removal.push(i);
            }
        }

        // Remove stopped systems (in reverse order to maintain indices).
        for &i in self.pending_removal.iter().rev() {
            self.systems.swap_remove(i);
        }
    }

    /// Prepares render data for all visible systems.
    pub fn prepare_render(
        &mut self,
        camera_pos: Vec3,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_fwd: Vec3,
    ) {
        for system in &mut self.systems {
            if system.has_particles() {
                system.prepare_render(camera_pos, camera_right, camera_up, camera_fwd);
            }
        }
    }

    /// Stops and clears all particle systems.
    pub fn clear(&mut self) {
        for system in &mut self.systems {
            system.stop_and_clear();
        }
        self.systems.clear();
    }

    /// Returns an iterator over all systems.
    pub fn iter(&self) -> impl Iterator<Item = &ParticleSystem> {
        self.systems.iter()
    }

    /// Returns a mutable iterator over all systems.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ParticleSystem> {
        self.systems.iter_mut()
    }

    /// Returns the global elapsed time.
    pub fn global_time(&self) -> f32 {
        self.global_time
    }

    /// Finds a system by name.
    pub fn find_by_name(&self, name: &str) -> Option<usize> {
        self.systems.iter().position(|s| s.name == name)
    }

    /// Plays a system by name.
    pub fn play_by_name(&mut self, name: &str) {
        if let Some(idx) = self.find_by_name(name) {
            self.systems[idx].play();
        }
    }

    /// Stops a system by name.
    pub fn stop_by_name(&mut self, name: &str) {
        if let Some(idx) = self.find_by_name(name) {
            self.systems[idx].stop();
        }
    }
}

impl Default for ParticleSystemManager {
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
    fn system_lifecycle() {
        let emitter = ParticleEmitter::new().with_rate(100.0);
        let mut system = ParticleSystem::new("test", emitter);

        assert_eq!(system.state(), PlaybackState::Stopped);

        system.play();
        assert_eq!(system.state(), PlaybackState::Playing);

        // Simulate 1 second.
        for _ in 0..60 {
            system.update(1.0 / 60.0, Vec3::ZERO, true);
        }
        assert!(system.particle_count() > 0);

        system.pause();
        assert_eq!(system.state(), PlaybackState::Paused);

        system.play();
        assert_eq!(system.state(), PlaybackState::Playing);

        system.stop_and_clear();
        assert_eq!(system.state(), PlaybackState::Stopped);
        assert_eq!(system.particle_count(), 0);
    }

    #[test]
    fn system_with_gravity() {
        let emitter = ParticleEmitter::new()
            .with_rate(10.0)
            .with_speed(0.0, 0.0)
            .with_lifetime(5.0, 5.0);
        let mut system = ParticleSystem::new("gravity", emitter)
            .with_gravity(Vec3::new(0.0, -10.0, 0.0));

        system.play();
        for _ in 0..60 {
            system.update(1.0 / 60.0, Vec3::ZERO, true);
        }

        // Particles should have moved downward.
        if system.particle_count() > 0 {
            let positions = system.pool.alive_positions();
            for pos in positions {
                assert!(pos.y < 0.0, "Particles should fall with gravity");
            }
        }
    }

    #[test]
    fn manager_basic() {
        let mut manager = ParticleSystemManager::new();

        let emitter = ParticleEmitter::new().with_rate(50.0);
        let system = ParticleSystem::new("sys1", emitter);
        let id = manager.add(system);

        assert_eq!(manager.count(), 1);

        manager.update(0.1, Vec3::ZERO);
        assert!(manager.total_particles() > 0);
    }

    #[test]
    fn lod_reduces_particles() {
        let lod = LodSettings {
            enabled: true,
            start_distance: 10.0,
            end_distance: 100.0,
            min_fraction: 0.1,
        };

        assert!((lod.compute_fraction(5.0) - 1.0).abs() < 0.01);
        assert!(lod.compute_fraction(50.0) < 1.0);
        assert!((lod.compute_fraction(200.0) - 0.1).abs() < 0.01);
    }

    #[test]
    fn warmup_pre_simulates() {
        let emitter = ParticleEmitter::new()
            .with_rate(100.0)
            .with_lifetime(2.0, 2.0);
        let mut system = ParticleSystem::new("warmup", emitter)
            .with_warmup(1.0);

        system.play();
        // After warmup of 1 second at 100 pps, should have ~100 particles.
        assert!(
            system.particle_count() > 50,
            "Warmup should pre-fill particles, got {}",
            system.particle_count()
        );
    }
}
