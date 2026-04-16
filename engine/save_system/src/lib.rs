//! # genovo-save
//!
//! Save/load system for the Genovo game engine.
//!
//! This crate provides:
//!
//! - **SaveGame** — complete save file with metadata and world state
//! - **SaveManager** — save slots, auto-save, quick-save/quick-load
//! - **WorldState** — serialized representation of all entities and components
//! - **Integrity** — checksum verification for save file corruption detection
//! - **Cloud compatibility** — platform-agnostic metadata for cloud save sync

pub mod save_game;

pub use save_game::{
    CloudSyncMetadata, ComponentData, EntityData, SaveError, SaveGame, SaveManager,
    SaveMetadata, SaveResult, SaveSlot, WorldState,
};
