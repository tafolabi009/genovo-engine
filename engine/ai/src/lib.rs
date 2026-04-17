//! Genovo Engine - AI Module
//!
//! Provides game AI infrastructure including A* and hierarchical pathfinding,
//! behavior tree execution, navigation mesh generation and queries,
//! crowd simulation with obstacle avoidance, utility AI, goal-oriented
//! action planning (GOAP), influence maps, steering behaviors,
//! perception/sensing systems, group formation management, knowledge
//! representation with inference, tactical combat AI, NPC dialogue
//! decision-making, decision trees, squad coordination, enemy spawning
//! with difficulty scaling, and extended sensory memory with attention.

pub mod behavior_trees;
pub mod decision_tree;
pub mod dialogue_ai;
pub mod enemy_spawning;
pub mod formation;
pub mod goap;
pub mod influence_maps;
pub mod knowledge;
pub mod navmesh;
pub mod pathfinding;
pub mod perception;
pub mod sensory_memory;
pub mod squad_ai;
pub mod steering;
pub mod tactical;
pub mod utility_ai;

// HTN (Hierarchical Task Network) planner: compound tasks decompose into primitives,
// methods with preconditions, plan search with backtracking, partial plan execution.
pub mod planner;

// AI emotions: PAD model (Pleasure/Arousal/Dominance), emotion decay, triggers,
// emotion-to-behavior mapping, facial expression output, mood persistence.
pub mod emotion_model;

// AI spatial awareness: visibility map, danger map, path safety evaluation,
// predictive position estimation, exposure analysis, defensive position scoring.
pub mod spatial_awareness;

// Enhanced behavior trees: utility-scored selectors (hybrid utility+BT),
// decorator: cooldown/limit/timeout/probability/force-success/force-fail,
// subtree references, shared blackboard across trees, runtime BT debugging data.
pub mod behavior_tree;

// AI world representation: simplified world state for planning, spatial occupancy
// grid, threat map, resource locations, ally positions, objective importance,
// environmental conditions.
pub mod world_model;

// AI conversation: multi-participant dialogue, turn-taking, interruption,
// topic switching, memory of previous conversations, relationship changes
// during dialogue, bartering/persuasion/intimidation skill checks.
pub mod conversation;

// AI controller framework: sense-think-act loop, blackboard per agent,
// behavior selection, action execution, perception integration, debug viz.
pub mod ai_controller;

// Combat AI: target selection (threat, distance, health), attack patterns,
// dodge/block decision, ability usage AI, retreat conditions, group tactics.
pub mod combat_ai;

// AI navigation: path request queue, async pathfinding, path smoothing,
// obstacle avoidance integration, stuck detection, teleport when stuck.
pub mod ai_navigation;

// Multi-agent behavior coordination: group goals, task assignment, role rotation,
// synchronized actions (coordinated breach), radio communication simulation.
pub mod behavior_coordinator;

// Simple AI learning: Q-table reinforcement learning, state-action pairs, reward
// tracking, exploration/exploitation balance (epsilon-greedy), policy serialization.
pub mod learning;

// AI debug visualization: draw perception cones, draw awareness levels, show
// current behavior tree node, show navigation path, show steering vectors,
// show influence map values, show blackboard contents.
pub mod ai_debug;

// Dialogue management: dialogue queue, simultaneous conversations, dialogue
// interruption priority, dialogue cooldown, subtitled/voiced/bark modes,
// dialogue events for gameplay triggers.
pub mod dialogue_manager;

// Re-exports for enhanced behavior trees.
pub use behavior_tree::{
    BehaviorTreeV2, BtDecorator, BtLibrary, BtNodeId, BtNodeType, BtNodeV2,
    BtStatus, DecoratorType, SharedBlackboard, UtilityConsideration,
    UtilityCurve, UtilityScore,
};

// Re-exports for world model.
pub use world_model::{
    AIWorldModel, CellOccupancy, EntityClass, EntityId as WorldEntityId,
    EnvironmentState, OccupancyGrid, ObjectiveInfo, ResourceLocation,
    ThreatLevel, ThreatMap, WeatherCondition, WorldEntity,
};

// Re-exports for conversation system.
pub use conversation::{
    Conversation, ConversationId, ConversationMemory, ConversationParticipant,
    ConversationState, ConversationSystem, ConversationTopic, DialogueResponse,
    ParticipantId, SkillCheck, SkillCheckResult, SkillCheckType, TopicId,
};

pub use behavior_trees::{
    ActionNode, BehaviorContext, BehaviorNode, BehaviorTree, BehaviorTreeAsset,
    BehaviorTreeBuilder, Blackboard, ConditionDecorator, ConditionNode, Cooldown, Inverter,
    LogLevel, LogNode, NodeStatus, Parallel, ParallelPolicy, RandomSelector, RepeatUntilFail,
    Repeater, Selector, Sequence, Timeout, WaitNode, WeightedSelector,
};
pub use formation::{
    Formation, FormationManager, FormationShape, FormationSlot,
};
pub use goap::{
    GOAPAction, GOAPAgent, GOAPGoal, GOAPPlan, GOAPPlanner, WorldState, WorldStateBuilder,
};
pub use influence_maps::{
    CombineOp, Falloff, InfluenceMap, InfluenceMapManager,
};
pub use navmesh::{
    CrowdManager, NavMesh, NavMeshAgent, NavMeshBuildConfig, NavMeshBuilder, NavMeshQuery,
    NavPoly, ObstacleAvoidance, ObstacleAvoidanceConfig,
};
pub use pathfinding::{
    AStarPathfinder, GridGraph, HierarchicalPathfinder, NavGraph, NodeId, Path, PathFinder,
    PathNode, PathRequest,
};
pub use perception::{
    AwarenessLevel, HearingSense, LastKnownPosition, OcclusionWall, PerceivedSound,
    PerceptionComponent, PerceptionConfig, PerceptionEvent, PerceptionMemory,
    PerceptionMemoryEntry, PerceptionSystem, RadarSense, SenseType, SightSense, SmellSense,
    SoundEvent, SoundType, StimulusSource, TouchSense,
};
pub use steering::{
    SteeringAgent, SteeringCombinator, CombineMethod, Obstacle, Wall, WaypointPath,
};
pub use utility_ai::{
    Action, Consideration, ResponseCurve, UtilityAI, UtilityAIBuilder, UtilityContext,
};
pub use knowledge::{
    ConclusionTemplate, ConclusionValue, ComputeOp, Fact, FactId, FactOrigin, FactPattern,
    FactValue, InferenceRule, KnowledgeBase, KnowledgeBaseStats, KnowledgeEvent, KnowledgeQuery,
    KnowledgeSystem, PoolId, QueryResult, RuleId, SharedKnowledge, SharedKnowledgeManager,
    SharedKnowledgeNotification, ValueConstraint, ValueType,
};
pub use tactical::{
    AmbushPosition, AmbushRole, AmbushSetup, AmbushTrigger, CoverPoint, CoverType,
    FiringPosition, PatrolMode, PatrolRoute, PatrolWaypoint, RetreatPath, TacticalCell,
    TacticalMap, ThreatInfo,
};
pub use dialogue_ai::{
    ConversationContext, ConversationEntry, ConversationTone,
    ConversationTopic as DialogueConversationTopic,
    DialogueDecisionMaker, DialogueStyle, DispositionLevel, MoodModel, MoodType, Personality,
    Relationship, RelationshipGraph, TopicCategory,
};
pub use decision_tree::{
    Action as DecisionAction, Condition, ConditionValue, DecisionContext,
    DecisionNode, DecisionTree, DecisionTreeBuilder, DecisionTreeComponent,
    DecisionTreeEvaluator, NodeId as DecisionNodeId, NodeStats, SimpleRng,
    TreeId, VisualizationNode, generate_visualization,
};
pub use squad_ai::{
    FireAndMoveState, MemberOrder, MemberState, RallyPoint, Squad, SquadAISystem,
    SquadComponent, SquadEvent, SquadFormation, SquadFormationShape,
    SquadHealthStatus, SquadId, SquadOrder, SquadRole, MemberId,
};
pub use enemy_spawning::{
    DifficultyScaler, DifficultyWaveScaling, EnemySpawnManager, EnemySpawnRequest,
    EnemyTier, EnemyTypeDefinition, EnemyTypeId, EncounterState, SpawnZone,
    SpawnWaveDefinition, SpawnerEvent, SpawnerStats, WaveComposer,
};
pub use sensory_memory::{
    AttentionFocus, DecayCurve, SensoryEvent, SensoryMemory, SensoryMemoryComponent,
    SensoryMemoryStats, StimulusCategory, StimulusId, StimulusMemory,
    SenseType as SensoryMemorySenseType,
};
