//! # Genovo Engine
//!
//! A AAA-tier game engine built primarily in Rust with multi-backend rendering,
//! hybrid ECS+OOP architecture, and cross-platform support.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use genovo::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = EngineConfig {
//!         app_name: "My Game".to_string(),
//!         ..Default::default()
//!     };
//!
//!     let mut engine = Engine::new(config)?;
//!
//!     // Spawn entities, add physics bodies, load assets...
//!     let world = engine.world_mut();
//!     let entity = world.spawn_entity()
//!         .with(TransformComponent::from_position(Vec3::new(0.0, 5.0, 0.0)))
//!         .build();
//!
//!     println!("Entities: {}", engine.world().entity_count());
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod engine;
pub mod prelude;

// Re-export all engine subsystems for direct access.
pub use genovo_core as core;
pub use genovo_ecs as ecs;
pub use genovo_scene as scene;
pub use genovo_render as render;
pub use genovo_platform as platform;
pub use genovo_physics as physics;
pub use genovo_audio as audio;
pub use genovo_animation as animation;
pub use genovo_assets as assets;
pub use genovo_scripting as scripting;
pub use genovo_networking as networking;
pub use genovo_ai as ai;
pub use genovo_editor as editor;
pub use genovo_debug as debug;
pub use genovo_terrain as terrain;
pub use genovo_cinematics as cinematics;
pub use genovo_localization as localization;
pub use genovo_procgen as procgen;
pub use genovo_ui as ui;
pub use genovo_save as save;

// Re-export the top-level engine types at crate root for convenience.
pub use config::{EngineConfig, LogLevel, WindowConfig};
pub use engine::{Engine, EngineError};
