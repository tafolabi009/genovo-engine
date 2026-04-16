//! # genovo-gameplay
//!
//! Comprehensive gameplay framework for the Genovo game engine.
//!
//! This crate provides ready-to-use gameplay systems that cover common game
//! mechanics. All systems are designed to work standalone or integrate with
//! the Genovo ECS.
//!
//! ## Modules
//!
//! | Module                 | Description                                        |
//! |------------------------|----------------------------------------------------|
//! | [`character_controller`] | Capsule-based physics character movement          |
//! | [`camera_controller`]    | Third-person, first-person, and top-down cameras  |
//! | [`inventory`]            | Items, stacking, equipment, loot, and crafting    |
//! | [`damage_system`]        | Health, damage types, DoTs, and status effects    |
//! | [`quest_system`]         | Quests, objectives, rewards, and journals         |
//! | [`dialogue`]             | Branching dialogue trees with conditions          |
//! | [`spawner`]              | Interval and wave-based entity spawning           |

pub mod camera_controller;
pub mod character_controller;
pub mod damage_system;
pub mod dialogue;
pub mod inventory;
pub mod quest_system;
pub mod spawner;

// ---------------------------------------------------------------------------
// Re-exports: character controller
// ---------------------------------------------------------------------------

pub use character_controller::{
    CharacterConfig, CharacterController, CharacterControllerComponent, CharacterInput,
    CharacterState, CollisionShape, CollisionWorld, GroundInfo, MovingPlatformState, OverlapResult,
    SweepHit,
};

// ---------------------------------------------------------------------------
// Re-exports: camera controller
// ---------------------------------------------------------------------------

pub use camera_controller::{
    CameraBehavior, CameraCollisionProvider, CameraRayHit, CameraRig, CameraShake,
    CameraTransform, FirstPersonCamera, FirstPersonConfig, FirstPersonInput, SmoothFollow,
    ThirdPersonCamera, ThirdPersonConfig, ThirdPersonInput, TopDownCamera, TopDownConfig,
    TopDownInput,
};

// ---------------------------------------------------------------------------
// Re-exports: inventory
// ---------------------------------------------------------------------------

pub use inventory::{
    CraftingBook, CraftingIngredient, CraftingOutput, CraftingRecipe, Equipment, EquipmentSlot,
    Inventory, InventoryError, InventoryResult, ItemCategory, ItemDefinition, ItemInstanceData,
    ItemProperty, ItemRarity, ItemRegistry, ItemStack, LootDrop, LootEntry, LootTable,
};

// ---------------------------------------------------------------------------
// Re-exports: damage system
// ---------------------------------------------------------------------------

pub use damage_system::{
    CritCalculator, DamageEvent, DamageNumber, DamageResistances, DamageResult, DamageType,
    DotDamage, Health, StatusEffect, StatusEffectType,
};

// ---------------------------------------------------------------------------
// Re-exports: quest system
// ---------------------------------------------------------------------------

pub use quest_system::{
    ObjectiveType, Quest, QuestError, QuestEvent, QuestJournal, QuestObjective, QuestReward,
    QuestState,
};

// ---------------------------------------------------------------------------
// Re-exports: dialogue
// ---------------------------------------------------------------------------

pub use dialogue::{
    ComparisonOp, DialogueChoice, DialogueCondition, DialogueConsequence, DialogueContext,
    DialogueNode, DialogueRunner, DialogueRunnerState, DialogueTree, SimpleDialogueContext,
    SkillCheck, SpeakerEmotion, evaluate_condition,
};

// ---------------------------------------------------------------------------
// Re-exports: spawner
// ---------------------------------------------------------------------------

pub use spawner::{
    EnemyGroup, SpawnPoint, SpawnProperty, SpawnRequest, Spawner, WaveDefinition, WaveSpawner,
    WaveState,
};
