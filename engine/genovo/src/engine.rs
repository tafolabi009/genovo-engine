//! Top-level engine orchestrator.
//!
//! The `Engine` struct owns all subsystems and drives the main loop.

use std::time::Duration;

use genovo_core::math::Vec3;

use genovo_audio::AudioMixer as _;

use crate::config::EngineConfig;

/// The top-level engine instance.
///
/// Owns all subsystems (ECS world, physics, audio, asset server) and drives
/// the main game loop with fixed-timestep physics and variable rendering.
pub struct Engine {
    /// User-supplied configuration.
    config: EngineConfig,

    /// Central time source for the engine. Tracks per-frame timing, drives
    /// the fixed-timestep accumulator, and provides delta/total times.
    pub clock: genovo_core::Clock,

    /// The ECS world holding all entities, components, and resources.
    world: genovo_ecs::World,

    /// Rigid-body physics simulation.
    pub physics_world: genovo_physics::PhysicsWorld,

    /// Software audio mixer (CPU-based PCM mixing, bus routing, voice
    /// management).
    audio_mixer: genovo_audio::SoftwareMixer,

    /// Centralised asset loading and caching service.
    asset_server: genovo_assets::AssetServer,

    /// Whether the main loop is currently running.
    running: bool,
}

impl Engine {
    /// Create a new engine instance with the given configuration.
    ///
    /// Initialisation order:
    /// 1. Clock (with fixed timestep from config)
    /// 2. ECS World
    /// 3. Physics world (with configured gravity)
    /// 4. Software audio mixer
    /// 5. Asset server (with configured root path + default loaders)
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        log::info!("Initializing Genovo Engine: {}", config.app_name);

        // 1. Clock
        let clock = genovo_core::Clock::new(
            Duration::from_secs_f64(config.fixed_timestep),
        );
        log::info!(
            "  Clock: fixed timestep = {:.4}s ({:.1} Hz)",
            config.fixed_timestep,
            1.0 / config.fixed_timestep,
        );

        // 2. ECS World
        let world = genovo_ecs::World::new();
        log::info!("  ECS World created");

        // 3. Physics
        let gravity = Vec3::new(
            config.gravity[0],
            config.gravity[1],
            config.gravity[2],
        );
        let physics_world = genovo_physics::PhysicsWorld::new(gravity);
        log::info!(
            "  Physics world created (gravity = [{:.2}, {:.2}, {:.2}])",
            gravity.x,
            gravity.y,
            gravity.z,
        );

        // 4. Audio
        let audio_mixer = genovo_audio::SoftwareMixer::new(
            config.audio_sample_rate,
            2, // stereo output
            config.audio_buffer_size,
            config.max_audio_voices,
        );
        log::info!(
            "  Audio mixer: {} Hz, {} max voices, buffer {}",
            config.audio_sample_rate,
            config.max_audio_voices,
            config.audio_buffer_size,
        );

        // 5. Asset server
        let asset_server = genovo_assets::AssetServer::new(&config.asset_root);

        // Register default loaders
        asset_server.register_loader(genovo_assets::TextureLoader);
        asset_server.register_loader(genovo_assets::ObjLoader);
        asset_server.register_loader(genovo_assets::WavLoader);
        asset_server.register_loader(genovo_assets::TextLoader);
        asset_server.register_loader(genovo_assets::BytesLoader);
        asset_server.register_loader(genovo_assets::GltfLoader);
        log::info!(
            "  Asset server at '{}' with default loaders (bmp, obj, wav, txt, bin, glTF)",
            config.asset_root,
        );

        log::info!("Genovo Engine '{}' initialized successfully", config.app_name);

        Ok(Self {
            config,
            clock,
            world,
            physics_world,
            audio_mixer,
            asset_server,
            running: false,
        })
    }

    // -----------------------------------------------------------------------
    // ECS World access
    // -----------------------------------------------------------------------

    /// Returns an immutable reference to the ECS world.
    #[inline]
    pub fn world(&self) -> &genovo_ecs::World {
        &self.world
    }

    /// Returns a mutable reference to the ECS world.
    #[inline]
    pub fn world_mut(&mut self) -> &mut genovo_ecs::World {
        &mut self.world
    }

    // -----------------------------------------------------------------------
    // Subsystem accessors
    // -----------------------------------------------------------------------

    /// Returns an immutable reference to the physics world.
    #[inline]
    pub fn physics(&self) -> &genovo_physics::PhysicsWorld {
        &self.physics_world
    }

    /// Returns a mutable reference to the physics world.
    #[inline]
    pub fn physics_mut(&mut self) -> &mut genovo_physics::PhysicsWorld {
        &mut self.physics_world
    }

    /// Returns an immutable reference to the software audio mixer.
    #[inline]
    pub fn audio(&self) -> &genovo_audio::SoftwareMixer {
        &self.audio_mixer
    }

    /// Returns a mutable reference to the software audio mixer.
    #[inline]
    pub fn audio_mut(&mut self) -> &mut genovo_audio::SoftwareMixer {
        &mut self.audio_mixer
    }

    /// Returns an immutable reference to the asset server.
    #[inline]
    pub fn assets(&self) -> &genovo_assets::AssetServer {
        &self.asset_server
    }

    /// Returns an immutable reference to the clock.
    #[inline]
    pub fn clock(&self) -> &genovo_core::Clock {
        &self.clock
    }

    /// Shorthand for `self.clock.delta_secs()`.
    #[inline]
    pub fn dt(&self) -> f32 {
        self.clock.delta_secs()
    }

    /// Returns a reference to the engine configuration.
    #[inline]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Returns `true` if the main loop is currently running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.running
    }

    // -----------------------------------------------------------------------
    // Physics-ECS synchronization
    // -----------------------------------------------------------------------

    /// Copy physics body transforms (position/rotation) to the ECS
    /// `TransformData` components for all entities that have an associated
    /// rigid body.
    ///
    /// Call this after `physics_world.step()` each fixed update to keep the
    /// ECS world in sync with the physics simulation.
    pub fn sync_physics_to_ecs(&mut self) {
        for body in self.physics_world.bodies() {
            let entity_id = body.handle.0;
            if let Some(entity) = self.world.entity_from_id(entity_id) {
                if let Some(transform) = self.world.get_component_mut::<genovo_ecs::TransformData>(entity) {
                    transform.position = [body.position.x, body.position.y, body.position.z];
                    transform.rotation = [body.rotation.x, body.rotation.y, body.rotation.z, body.rotation.w];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------

    /// Run the main engine loop.
    ///
    /// This implements a fixed-timestep game loop:
    /// - Physics updates run at a fixed rate (accumulator-based).
    /// - Audio mixing and asset loading happen every frame.
    /// - Frame counter is logged every 60 frames.
    ///
    /// The loop runs until [`stop`](Engine::stop) is called.
    pub fn run(&mut self) {
        self.running = true;
        log::info!("Starting main loop");

        while self.running {
            self.clock.tick();
            let dt = self.clock.delta_secs();

            // Fixed-timestep physics updates
            while self.clock.should_run_fixed_update() {
                let fixed_dt = self.clock.fixed_timestep().as_secs_f32();
                if let Err(e) = self.physics_world.step(fixed_dt) {
                    log::error!("Physics step error: {}", e);
                }
                self.sync_physics_to_ecs();
            }

            // Update audio mixer
            self.audio_mixer.update(dt);

            // Process completed asset loads
            self.asset_server.process_completed();

            // Periodic frame counter logging
            let frame = self.clock.frame_count();
            if frame % 60 == 0 {
                log::debug!(
                    "Frame {} | dt={:.4}s | physics bodies={} | audio voices={} | entities={}",
                    frame,
                    dt,
                    self.physics_world.body_count(),
                    self.audio_mixer.active_voice_count(),
                    self.world.entity_count(),
                );
            }
        }

        log::info!("Main loop ended after {} frames", self.clock.frame_count());
    }

    /// Signal the main loop to stop after the current frame.
    pub fn stop(&mut self) {
        log::info!("Engine stop requested");
        self.running = false;
    }

    /// Shut down the engine and release all resources.
    ///
    /// Stops the main loop if still running, then tears down subsystems in
    /// reverse initialisation order.
    pub fn shutdown(&mut self) {
        log::info!("Shutting down Genovo Engine");
        self.running = false;

        // Stop all audio playback
        self.audio_mixer.stop_all();

        // Collect unreferenced assets
        self.asset_server.collect_garbage();

        log::info!("Genovo Engine shut down");
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.running {
            self.shutdown();
        }
    }
}

/// Engine-level errors.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Platform initialization failed: {0}")]
    PlatformError(String),

    #[error("Render device creation failed: {0}")]
    RenderError(String),

    #[error("Asset system initialization failed: {0}")]
    AssetError(String),

    #[error("Audio system initialization failed: {0}")]
    AudioError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}
