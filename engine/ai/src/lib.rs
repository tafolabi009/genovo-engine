//! Genovo Engine - AI Module
//!
//! Provides game AI infrastructure including A* and hierarchical pathfinding,
//! behavior tree execution, navigation mesh generation and queries,
//! crowd simulation with obstacle avoidance, utility AI, goal-oriented
//! action planning (GOAP), influence maps, steering behaviors,
//! perception/sensing systems, and group formation management.

pub mod behavior_trees;
pub mod formation;
pub mod goap;
pub mod influence_maps;
pub mod navmesh;
pub mod pathfinding;
pub mod perception;
pub mod steering;
pub mod utility_ai;

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
