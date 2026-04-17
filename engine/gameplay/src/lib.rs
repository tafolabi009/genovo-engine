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
//! | [`ai_director`]          | Dynamic difficulty and AI-driven pacing           |
//! | [`progression`]          | XP, leveling, stats, skill trees, achievements    |
//! | [`ability_system`]       | Abilities, cooldowns, combos, and effect dispatch |
//! | [`interaction`]          | World interaction: doors, pickups, switches       |
//! | [`movement`]             | Quake-style movement with multiple modes           |
//! | [`weapon_system`]        | Weapons, fire modes, projectiles, hitscan, melee  |
//! | [`vehicle_controller`]   | Vehicle gameplay: nitro, drifting, camera, audio  |
//! | [`weather_system`]       | Dynamic weather with transitions and time-of-day |
//! | [`economy`]              | Currencies, shops, auctions, supply/demand pricing |
//! | [`faction_system`]       | Factions, reputation, territory, inter-faction relations |
//! | [`building_system`]      | Placeable structures, grid snapping, structural integrity |
//! | [`crafting_v2`]           | Advanced crafting: recipes, quality tiers, enchanting |
//! | [`minimap`]              | Minimap with fog of war, markers, and zoom controls |
//! | [`stealth_system`]       | Visibility, noise, disguises, AI search patterns |
//! | [`scoring`]              | Points, combos, streaks, leaderboards, MVP calculation |

pub mod ability_system;
pub mod ai_director;
pub mod building_system;
pub mod camera_controller;
pub mod character_controller;
pub mod crafting_v2;
pub mod damage_system;
pub mod dialogue;
pub mod economy;
pub mod faction_system;
pub mod interaction;
pub mod inventory;
pub mod minimap;
pub mod movement;
pub mod progression;
pub mod quest_system;
pub mod scoring;
pub mod spawner;
pub mod state_machine;
pub mod stealth_system;
pub mod vehicle_controller;
pub mod weapon_system;
pub mod weather_system;

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

// ---------------------------------------------------------------------------
// Re-exports: ai_director
// ---------------------------------------------------------------------------

pub use ai_director::{
    AIDirector, AmbientEventKind, DesiredIntensity, DifficultyMultiplier, DirectorEvent,
    DirectorLog, IntensityState, PacingConfig, PacingCurve, PacingWaypoint, PlayerStats,
    SpawnDirective, StateDuration, TensionTracker,
};

// ---------------------------------------------------------------------------
// Re-exports: progression
// ---------------------------------------------------------------------------

pub use progression::{
    Achievement, AchievementCondition, AchievementReward, AchievementSystem,
    AchievementUnlockedEvent, CommonStat, ExperienceRecord, ExperienceSystem, LevelUpEvent,
    ModifierDuration, ModifierKind, PlayerProgressState, SkillEffect, SkillNode, SkillTree,
    SkillTreeError, Stat, StatBlock, StatId, StatModifier, XPCurveKind, XPTable,
};

// ---------------------------------------------------------------------------
// Re-exports: ability_system
// ---------------------------------------------------------------------------

pub use ability_system::{
    Ability, AbilityBar, AbilityComponent, AbilityDamageType, AbilityEffect, AbilityEvent,
    AbilityRegistry, AbilitySlot, AreaShape, CastState, ComboDefinition, ComboResult,
    ComboStep, ComboTracker, CrowdControlType, ResourcePool, ResourceType, TargetingMode,
    ABILITY_BAR_SIZE,
};

// ---------------------------------------------------------------------------
// Re-exports: interaction
// ---------------------------------------------------------------------------

pub use interaction::{
    DialogueTrigger, Door, Interactable, InteractionEvent, InteractionSystem, InteractionType,
    Pickup, Switch,
};

// ---------------------------------------------------------------------------
// Re-exports: state_machine
// ---------------------------------------------------------------------------

pub use state_machine::{
    AnyStateMachine, GameEvent, GameState, HistoryMode, State, StateMachine, StateMachineBuilder,
    StateMachineComponent, TimedTransition, Transition, TransitionResult,
    create_game_state_machine,
};

// ---------------------------------------------------------------------------
// Re-exports: movement
// ---------------------------------------------------------------------------

pub use movement::{
    AirMovement, GroundMovement, LadderMovement, ModeConfig, ModeConfigs, MovementComponent,
    MovementInput, MovementMode, MovementSystem, WaterMovement,
};

// ---------------------------------------------------------------------------
// Re-exports: weapon system
// ---------------------------------------------------------------------------

pub use weapon_system::{
    FireEvent, FireMode, HitscanResult, Projectile, ProjectileSystem, RecoilPattern, Weapon,
    WeaponInventory, WeaponState, WeaponSystem, WeaponType,
};

// ---------------------------------------------------------------------------
// Re-exports: vehicle controller
// ---------------------------------------------------------------------------

pub use vehicle_controller::{
    DriftState, DriftTracker, NitroBoost, SpeedUnit, VehicleAudio, VehicleCamera,
    VehicleController, VehicleHudData, VehicleInput,
};

// ---------------------------------------------------------------------------
// Re-exports: weather system
// ---------------------------------------------------------------------------

pub use weather_system::{
    TimeOfDay, WeatherEffects, WeatherManager, WeatherParameters, WeatherSchedulerConfig,
    WeatherSnapshot, WeatherState, WeatherTransition,
};

// ---------------------------------------------------------------------------
// Re-exports: economy
// ---------------------------------------------------------------------------

pub use economy::{
    AuctionError, AuctionHouse, AuctionListing, CraftingCost, CraftingCostCalculator,
    CurrencyType, EconomyTracker, FlowEntry, FlowTotals, PriceModifier, PriceModifierSource,
    Shop, ShopError, ShopItem, TransactionRecord, TransactionType, Wallet,
};

// ---------------------------------------------------------------------------
// Re-exports: faction system
// ---------------------------------------------------------------------------

pub use faction_system::{
    Faction, FactionPolicy, FactionRelationship, FactionRelationshipType, FactionSystem,
    FactionTerritory, ReputationEvent, ReputationReason, ReputationStanding,
};

// ---------------------------------------------------------------------------
// Re-exports: building system
// ---------------------------------------------------------------------------

pub use building_system::{
    Blueprint, BlueprintPiece, BuildPiece, BuildingEvent, BuildingManager, MaterialType,
    PieceType, PlacementResult, ResourceCost, SnapPoint, Structure,
};

// ---------------------------------------------------------------------------
// Re-exports: crafting v2
// ---------------------------------------------------------------------------

pub use crafting_v2::{
    AppliedEnchantment, CraftedItem, CraftingOutput as CraftingV2Output, CraftingResult,
    CraftingStation, CraftingSystem, EnchantResult, Enchantment, Ingredient, MaterialBonus,
    QualityTier, Recipe, RecipeBook, RecipeCategory, StatRoller,
};

// ---------------------------------------------------------------------------
// Re-exports: minimap
// ---------------------------------------------------------------------------

pub use minimap::{
    CustomIcon, FogOfWar, FogState, MarkerRenderData, MarkerType, MinimapConfig, MinimapMarker,
    MinimapRenderData, MinimapShape, MinimapSystem, RotationMode,
};

// ---------------------------------------------------------------------------
// Re-exports: stealth system
// ---------------------------------------------------------------------------

pub use stealth_system::{
    AlertState, Disguise, DisguiseBreakAction, LastKnownPositionEntry, NoiseEvent, NoiseType,
    Posture, SearchPattern, SearchPatternType, StealthComponent, StealthEvent, StealthSystem,
    SurfaceType,
};

// ---------------------------------------------------------------------------
// Re-exports: scoring
// ---------------------------------------------------------------------------

pub use scoring::{
    ComboTracker, HighlightType, KillStreak, Leaderboard, LeaderboardEntry, LeaderboardSort,
    MatchHighlight, MatchStats, PlayerMatchSummary, ScoreEvent, ScoreEventType, ScoreTracker,
    ScoringSystem, StatTracker,
};
