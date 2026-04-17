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
//! | [`crafting`]           | Advanced crafting: recipes, quality tiers, enchanting |
//! | [`minimap`]              | Minimap with fog of war, markers, and zoom controls |
//! | [`stealth_system`]       | Visibility, noise, disguises, AI search patterns |
//! | [`scoring`]              | Points, combos, streaks, leaderboards, MVP calculation |
//! | [`dialogue_ink`]          | Ink-style dialogue engine with variables and barks |
//! | [`cooking_system`]       | Cooking/alchemy with recipes, buffs, and discovery |
//! | [`mount_system`]         | Mountable creatures with stamina, taming, combat |
//! | [`day_night_cycle`]      | Day/night with sun, moon, sky, NPC schedules |
//! | [`trap_system`]          | Traps and hazards: spikes, fire, poison, tripwires |
//! | [`companion_ai`]         | AI companion follow, combat, commands, morale |
//! | [`procedural_animation`] | Procedural look-at, breathing, recoil, walk cycle |

pub mod ability_system;
pub mod ai_director;
pub mod building_system;
pub mod camera_controller;
pub mod character_controller;
pub mod companion_ai;
pub mod cooking_system;
pub mod crafting;
pub mod damage_system;
pub mod day_night_cycle;
pub mod dialogue;
pub mod economy;
pub mod faction_system;
pub mod interaction;
pub mod inventory;
pub mod minimap;
pub mod mount_system;
pub mod movement;
pub mod procedural_animation;
pub mod progression;
pub mod quest_system;
pub mod scoring;
pub mod spawner;
pub mod state_machine;
pub mod stealth_system;
pub mod trap_system;
pub mod vehicle_controller;
pub mod weapon_system;
pub mod weather_system;

// Tutorial/onboarding: tutorial steps with conditions/triggers, tooltip positioning,
// highlight UI elements, force player actions, tutorial progress tracking, skip option.
pub mod tutorial_system;

// Extended achievements: compound conditions (AND/OR/NOT), progress tracking, secret
// achievements, categories, rewards (items/titles/cosmetics), notification queue.
pub mod achievement;

// Advanced save: screenshot thumbnail, playtime tracking, difficulty level, save
// versioning, cloud save metadata, save slot management, save/load events.
pub mod save_game;

// Fighting game combos: input sequence detection, direction notation (236P = QCF+P),
// cancel windows, combo tree, frame-perfect inputs, input buffer, combo counter.
pub mod input_combo;

// Enhanced inventory: grid-based inventory (items occupy WxH cells like Diablo),
// auto-sort/compact, item categories, item comparison tooltips, equipment set bonuses,
// item durability repair.
// (merged into `inventory` module)

// NPC management: NPC schedules (daily routines), NPC spawning/despawning,
// NPC merchant with inventory, NPC dialogue triggers, NPC pathfinding integration,
// NPC animations, NPC barks, NPC relationships.
pub mod npc_system;

// Mission/objective system: main missions + side missions, mission chains,
// mission waypoints, mission timer, mission rating (bronze/silver/gold),
// mission replay, mission rewards scaling.
pub mod mission_system;

// Gameplay particle presets: blood splatter, dust cloud, explosion VFX,
// magic spell effects, healing aura, shield bubble, footstep dust,
// water splash, fire/smoke, electricity.
pub mod particle_effects;

// Environmental hazards: lava, quicksand, toxic gas, radiation, extreme cold,
// underwater pressure, fall damage, electricity, fire spread,
// hazard zones with damage-over-time.
pub mod environmental_hazards;

// Visual dialogue graph: nodes with text/choices/conditions, edges for flow,
// graph evaluation, variable substitution, localization keys.
pub mod dialogue_graph;

// RPG class/job system: class definitions, class abilities, class progression,
// multi-class, class switching, passive bonuses, stat scaling.
pub mod class_system;

// World events: random events, scheduled events, event conditions, event effects,
// event chains, event UI data, cooldowns, priority resolution.
pub mod world_event;

// Collectible system: collectible types, collection tracking, completion percentage,
// rewards on complete set, map markers, milestones, categories.
pub mod collectibles;

// Enhanced camera: multiple camera modes, smooth transitions, camera effects stack,
// cinematic camera with keyframed paths, split-screen support.
pub mod camera_system;

// Level/map progression: level sequence, level unlocking conditions, par time/score,
// star ratings, level statistics, level select data, world map nodes.
pub mod level_progression;

// Shop interface: shop inventory, buy/sell transactions, dynamic pricing, limited
// stock, restock timers, shop categories, currency display, purchase confirmation.
pub mod shop_system;

// Buff/debuff framework: buff stacking rules (stack count, refresh duration,
// strongest wins), buff categories, buff immunity, buff cleansing, buff icons.
pub mod buff_system;

// Death and respawn: respawn points, respawn timer, respawn invulnerability,
// death penalty (XP loss, item drop), spectator mode, respawn wave timing.
pub mod respawn_system;

// Object pool for gameplay entities: pre-allocate entities, acquire/release,
// auto-grow, warm pool on scene load, pool statistics, per-type pools.
pub mod object_pooling;

// Camera gameplay effects: screen shake from damage, speed lines at high velocity,
// scope zoom, thermal/night vision overlays, underwater tint, death grayscale,
// hit indicators on screen edges.
pub mod camera_effects;

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
    EquipSlotV2, EquipmentSetId, EquipmentSetV2, GridCell, GridInventory,
    GridItemPlacement, InventoryErrorV2, ItemCategory, ItemInstanceId, ItemInstanceV2,
    ItemRarityV2, ItemRegistryV2, ItemStatV2, ItemTypeDefV2, ItemTypeId,
    RepairCostV2, SetBonusV2, StatDiffV2,
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
    BarkManager, BarkTrigger, Choice as DialogueV2Choice, ContentLine,
    DialogueEvent as DialogueV2Event, DialogueQueue, DialogueV2Component,
    DialogueV2System, InkAction, InkCondition, Knot, NpcDialogueConfig,
    Story, StoryRunner, StoryRunnerState, StoryVariableValue, StoryVariables,
    Stitch,
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

pub use crafting::{
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
    ComboTracker as ScoringComboTracker, HighlightType, KillStreak, Leaderboard, LeaderboardEntry, LeaderboardSort,
    MatchHighlight, MatchStats, PlayerMatchSummary, ScoreEvent, ScoreEventType, ScoreTracker,
    ScoringSystem, StatTracker,
};

// (dialogue v2 re-exports merged into dialogue re-exports above)

// ---------------------------------------------------------------------------
// Re-exports: cooking system
// ---------------------------------------------------------------------------

pub use cooking_system::{
    CookedDish, CookingEngine, CookingError, CookingIngredient as CookIngredient,
    CookingRecipe, CookingStation, CookingStationType, FoodBuff,
    IngredientCategory, IngredientProperty, IngredientPropertyType, IngredientQuality,
    PropertyCombinationRules, PropertyInteraction,
};

// ---------------------------------------------------------------------------
// Re-exports: mount system
// ---------------------------------------------------------------------------

pub use mount_system::{
    Mount, MountAnimState, MountComponent, MountEvent, MountId, MountInput,
    MountPoint, MountStats, MountSystem, MountType, RiderComponent, RiderId,
    TamingAction, TamingResult, TamingState,
};

// ---------------------------------------------------------------------------
// Re-exports: day/night cycle
// ---------------------------------------------------------------------------

pub use day_night_cycle::{
    DayNightComponent, DayNightConfig, DayNightEvent, DayNightState,
    DayNightSystem, LightSchedule, MoonPhase, NpcScheduleEntry, NpcTimeSchedule,
    SkyColor, SkyPalette, TimePeriod,
};

// ---------------------------------------------------------------------------
// Re-exports: trap system
// ---------------------------------------------------------------------------

pub use trap_system::{
    Trap, TrapComponent, TrapDamageType, TrapEffect, TrapEvent, TrapId, TrapManager,
    TrapState, TrapTrigger, TrapType,
};

// ---------------------------------------------------------------------------
// Re-exports: companion AI
// ---------------------------------------------------------------------------

pub use companion_ai::{
    CombatBehavior, Companion, CompanionAbility, CompanionAbilityType,
    CompanionCommand, CompanionComponent as CompanionAIComponent, CompanionEvent,
    CompanionId, CompanionInventory, CompanionManager, CompanionMood,
    CompanionState, CompanionStats, FormationType, MoodType as CompanionMoodType,
};

// ---------------------------------------------------------------------------
// Re-exports: procedural animation
// ---------------------------------------------------------------------------

pub use procedural_animation::{
    BlinkLayer, BoneId, BoneTransform, BreathingLayer, IdleFidgetLayer,
    LeanLayer, LookAtConstraint, ProceduralAnimComponent, ProceduralAnimController,
    ProceduralWalkCycle, RecoilLayer, TailPhysics,
};

// Comprehensive input: action/axis bindings, dead zone processing, input
// smoothing, input recording/playback, virtual inputs (for AI), input device
// abstraction, input mapping serialization.
pub mod input_manager;

// Game mode framework: game mode state machine, game mode rules (team setup,
// scoring, win conditions), game mode events, player spawning rules, game
// mode transitions, match lifecycle.
pub mod game_mode;

// HUD framework: health bar, ammo display, crosshair, minimap integration,
// compass, notification toasts, damage indicators (screen edge), interaction
// prompts, score display, objective markers.
pub mod hud_system;
