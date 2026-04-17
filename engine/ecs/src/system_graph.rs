//! System dependency graph for automatic parallel execution scheduling.
//!
//! Analyzes component access patterns of registered systems to build a
//! dependency graph, detects data races, groups non-conflicting systems into
//! parallel batches, detects cycles, and supports conditional execution.
//!
//! # Architecture
//!
//! ```text
//! Systems: [Physics, Render, AI, Audio, Input]
//!
//! Access analysis:
//!   Physics: Write(Transform, Velocity)
//!   Render:  Read(Transform, Mesh)
//!   AI:      Write(AI), Read(Transform)
//!   Audio:   Read(Transform)
//!   Input:   Write(InputState)
//!
//! Dependency graph:
//!   Physics --[Write Transform]--> Render (Render reads Transform)
//!   Physics --[Write Transform]--> AI     (AI reads Transform)
//!   Physics --[Write Transform]--> Audio  (Audio reads Transform)
//!   AI      --[Write AI]---------> (no conflicts)
//!   Input   --[Write InputState]--> (no conflicts)
//!
//! Parallel batches:
//!   Batch 0: [Input]            (no dependencies)
//!   Batch 1: [Physics]          (depends on Input)
//!   Batch 2: [Render, AI, Audio] (all depend on Physics, no mutual conflicts)
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// ComponentAccessKind
// ---------------------------------------------------------------------------

/// The kind of access a system performs on a component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentAccessKind {
    /// Read-only access. Multiple systems can read simultaneously.
    Read,
    /// Write access. Exclusive; conflicts with both reads and writes.
    Write,
}

impl fmt::Display for ComponentAccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentAccessKind::Read => write!(f, "Read"),
            ComponentAccessKind::Write => write!(f, "Write"),
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentAccess
// ---------------------------------------------------------------------------

/// A single component access declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentAccess {
    /// Name of the component type.
    pub component: String,
    /// Kind of access.
    pub kind: ComponentAccessKind,
}

impl ComponentAccess {
    /// Create a read access.
    pub fn read(component: &str) -> Self {
        Self {
            component: component.to_string(),
            kind: ComponentAccessKind::Read,
        }
    }

    /// Create a write access.
    pub fn write(component: &str) -> Self {
        Self {
            component: component.to_string(),
            kind: ComponentAccessKind::Write,
        }
    }

    /// Check if this access conflicts with another.
    pub fn conflicts_with(&self, other: &ComponentAccess) -> bool {
        if self.component != other.component {
            return false;
        }
        // Two reads don't conflict; anything involving a write does.
        self.kind == ComponentAccessKind::Write || other.kind == ComponentAccessKind::Write
    }
}

impl fmt::Display for ComponentAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.kind, self.component)
    }
}

// ---------------------------------------------------------------------------
// SystemNode
// ---------------------------------------------------------------------------

/// Index type for systems in the graph.
pub type SystemIndex = usize;

/// A system node in the dependency graph.
#[derive(Debug, Clone)]
pub struct SystemNode {
    /// Unique index.
    pub index: SystemIndex,
    /// Name of the system.
    pub name: String,
    /// Component accesses declared by this system.
    pub accesses: Vec<ComponentAccess>,
    /// Explicit ordering constraints: this system must run after these.
    pub run_after: HashSet<SystemIndex>,
    /// Explicit ordering constraints: this system must run before these.
    pub run_before: HashSet<SystemIndex>,
    /// Whether the system is currently enabled.
    pub enabled: bool,
    /// Optional run condition: if set, the system only runs when this returns true.
    pub condition: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    /// Stage (grouping for high-level ordering).
    pub stage: String,
    /// Priority within a stage (higher = earlier).
    pub priority: i32,
}

impl SystemNode {
    /// Create a new system node.
    pub fn new(index: SystemIndex, name: &str) -> Self {
        Self {
            index,
            name: name.to_string(),
            accesses: Vec::new(),
            run_after: HashSet::new(),
            run_before: HashSet::new(),
            enabled: true,
            condition: None,
            stage: "Update".to_string(),
            priority: 0,
        }
    }

    /// Add a component access.
    pub fn with_access(mut self, access: ComponentAccess) -> Self {
        self.accesses.push(access);
        self
    }

    /// Add a read access.
    pub fn reads(mut self, component: &str) -> Self {
        self.accesses.push(ComponentAccess::read(component));
        self
    }

    /// Add a write access.
    pub fn writes(mut self, component: &str) -> Self {
        self.accesses.push(ComponentAccess::write(component));
        self
    }

    /// Set explicit "run after" dependency.
    pub fn after(mut self, system: SystemIndex) -> Self {
        self.run_after.insert(system);
        self
    }

    /// Set explicit "run before" dependency.
    pub fn before(mut self, system: SystemIndex) -> Self {
        self.run_before.insert(system);
        self
    }

    /// Set the stage.
    pub fn in_stage(mut self, stage: &str) -> Self {
        self.stage = stage.to_string();
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Disable this system.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Check if this system conflicts with another (based on component access).
    pub fn conflicts_with(&self, other: &SystemNode) -> bool {
        for a in &self.accesses {
            for b in &other.accesses {
                if a.conflicts_with(b) {
                    return true;
                }
            }
        }
        false
    }

    /// Get all component names this system reads.
    pub fn read_components(&self) -> Vec<&str> {
        self.accesses
            .iter()
            .filter(|a| a.kind == ComponentAccessKind::Read)
            .map(|a| a.component.as_str())
            .collect()
    }

    /// Get all component names this system writes.
    pub fn write_components(&self) -> Vec<&str> {
        self.accesses
            .iter()
            .filter(|a| a.kind == ComponentAccessKind::Write)
            .map(|a| a.component.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// An edge in the dependency graph.
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// Source system (must run first).
    pub from: SystemIndex,
    /// Target system (must run after).
    pub to: SystemIndex,
    /// Reason for the dependency.
    pub reason: DependencyReason,
}

/// Why a dependency exists.
#[derive(Debug, Clone)]
pub enum DependencyReason {
    /// Automatic: component access conflict.
    ComponentConflict(String),
    /// Explicit: user-specified ordering.
    ExplicitOrder,
    /// Stage ordering.
    StageBoundary,
}

impl fmt::Display for DependencyEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match &self.reason {
            DependencyReason::ComponentConflict(c) => format!("conflict on {}", c),
            DependencyReason::ExplicitOrder => "explicit order".to_string(),
            DependencyReason::StageBoundary => "stage boundary".to_string(),
        };
        write!(f, "{} -> {} ({})", self.from, self.to, reason)
    }
}

// ---------------------------------------------------------------------------
// ParallelBatch
// ---------------------------------------------------------------------------

/// A batch of systems that can run in parallel (no conflicts).
#[derive(Debug, Clone)]
pub struct ParallelBatch {
    /// Batch index.
    pub index: usize,
    /// Systems in this batch.
    pub systems: Vec<SystemIndex>,
}

impl fmt::Display for ParallelBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Batch {}: {:?}", self.index, self.systems)
    }
}

// ---------------------------------------------------------------------------
// GraphError
// ---------------------------------------------------------------------------

/// Errors that can occur during graph analysis.
#[derive(Debug, Clone)]
pub enum GraphError {
    /// A cycle was detected in the dependency graph.
    CycleDetected(Vec<SystemIndex>),
    /// Duplicate system name.
    DuplicateName(String),
    /// Invalid system index.
    InvalidIndex(SystemIndex),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::CycleDetected(cycle) => write!(f, "cycle detected: {:?}", cycle),
            GraphError::DuplicateName(name) => write!(f, "duplicate system name: {}", name),
            GraphError::InvalidIndex(idx) => write!(f, "invalid system index: {}", idx),
        }
    }
}

// ---------------------------------------------------------------------------
// SystemGraph
// ---------------------------------------------------------------------------

/// A dependency graph of ECS systems that supports automatic conflict detection,
/// topological sorting, and parallel batch scheduling.
pub struct SystemGraph {
    /// All registered systems.
    systems: Vec<SystemNode>,
    /// Adjacency list: edges[from] = vec of (to, reason).
    edges: Vec<Vec<(SystemIndex, DependencyReason)>>,
    /// Computed parallel batches (lazily updated).
    batches: Vec<ParallelBatch>,
    /// Whether the graph needs to be rebuilt.
    dirty: bool,
    /// Stage execution order.
    stage_order: Vec<String>,
    /// Name-to-index mapping.
    name_map: HashMap<String, SystemIndex>,
}

impl SystemGraph {
    /// Create a new, empty system graph.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            edges: Vec::new(),
            batches: Vec::new(),
            dirty: true,
            stage_order: vec![
                "PreUpdate".to_string(),
                "Update".to_string(),
                "PostUpdate".to_string(),
            ],
            name_map: HashMap::new(),
        }
    }

    /// Set the stage execution order.
    pub fn set_stage_order(&mut self, stages: Vec<String>) {
        self.stage_order = stages;
        self.dirty = true;
    }

    /// Add a system to the graph. Returns its index.
    pub fn add_system(&mut self, node: SystemNode) -> SystemIndex {
        let idx = self.systems.len();
        self.name_map.insert(node.name.clone(), idx);
        self.systems.push(node);
        self.edges.push(Vec::new());
        self.dirty = true;
        idx
    }

    /// Get a system by index.
    pub fn get_system(&self, index: SystemIndex) -> Option<&SystemNode> {
        self.systems.get(index)
    }

    /// Get a mutable reference to a system.
    pub fn get_system_mut(&mut self, index: SystemIndex) -> Option<&mut SystemNode> {
        self.dirty = true;
        self.systems.get_mut(index)
    }

    /// Look up a system index by name.
    pub fn find_by_name(&self, name: &str) -> Option<SystemIndex> {
        self.name_map.get(name).copied()
    }

    /// Enable or disable a system.
    pub fn set_enabled(&mut self, index: SystemIndex, enabled: bool) {
        if let Some(sys) = self.systems.get_mut(index) {
            sys.enabled = enabled;
            self.dirty = true;
        }
    }

    /// Add an explicit ordering constraint.
    pub fn add_ordering(&mut self, before: SystemIndex, after: SystemIndex) {
        if let Some(sys) = self.systems.get_mut(before) {
            sys.run_before.insert(after);
        }
        if let Some(sys) = self.systems.get_mut(after) {
            sys.run_after.insert(before);
        }
        self.dirty = true;
    }

    /// Number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Build edges based on component access conflicts and explicit orderings.
    fn build_edges(&mut self) {
        // Clear old edges.
        for edge_list in &mut self.edges {
            edge_list.clear();
        }

        let n = self.systems.len();

        // Explicit orderings.
        for i in 0..n {
            let run_before: Vec<SystemIndex> = self.systems[i].run_before.iter().copied().collect();
            for &target in &run_before {
                if target < n {
                    self.edges[i].push((target, DependencyReason::ExplicitOrder));
                }
            }
            let run_after: Vec<SystemIndex> = self.systems[i].run_after.iter().copied().collect();
            for &source in &run_after {
                if source < n {
                    self.edges[source].push((i, DependencyReason::ExplicitOrder));
                }
            }
        }

        // Automatic conflict detection.
        // For each pair, if they conflict on a component, add an edge based on
        // registration order (earlier system goes first).
        for i in 0..n {
            for j in (i + 1)..n {
                if !self.systems[i].enabled || !self.systems[j].enabled {
                    continue;
                }
                // Find conflicting component.
                for ai in &self.systems[i].accesses {
                    for aj in &self.systems[j].accesses {
                        if ai.conflicts_with(aj) {
                            // Add edge i -> j (i runs first since it was registered first).
                            self.edges[i].push((
                                j,
                                DependencyReason::ComponentConflict(ai.component.clone()),
                            ));
                            break;
                        }
                    }
                }
            }
        }

        // Stage boundaries.
        for stage_idx in 0..self.stage_order.len().saturating_sub(1) {
            let current_stage = &self.stage_order[stage_idx];
            let next_stage = &self.stage_order[stage_idx + 1];

            let current_systems: Vec<SystemIndex> = (0..n)
                .filter(|&i| self.systems[i].stage == *current_stage && self.systems[i].enabled)
                .collect();
            let next_systems: Vec<SystemIndex> = (0..n)
                .filter(|&i| self.systems[i].stage == *next_stage && self.systems[i].enabled)
                .collect();

            for &cs in &current_systems {
                for &ns in &next_systems {
                    self.edges[cs].push((ns, DependencyReason::StageBoundary));
                }
            }
        }
    }

    /// Perform topological sort, detecting cycles.
    fn topological_sort(&self) -> Result<Vec<SystemIndex>, GraphError> {
        let n = self.systems.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<SystemIndex>> = vec![Vec::new(); n];

        for (from, neighbors) in self.edges.iter().enumerate() {
            if !self.systems[from].enabled {
                continue;
            }
            for &(to, _) in neighbors {
                if self.systems[to].enabled {
                    adj[from].push(to);
                    in_degree[to] += 1;
                }
            }
        }

        let mut queue: VecDeque<SystemIndex> = VecDeque::new();
        for i in 0..n {
            if self.systems[i].enabled && in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            sorted.push(node);
            for &neighbor in &adj[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        let enabled_count = self.systems.iter().filter(|s| s.enabled).count();
        if sorted.len() != enabled_count {
            // Cycle detected; find cycle nodes.
            let cycle: Vec<SystemIndex> = (0..n)
                .filter(|&i| self.systems[i].enabled && in_degree[i] > 0)
                .collect();
            return Err(GraphError::CycleDetected(cycle));
        }

        Ok(sorted)
    }

    /// Build parallel batches from the topological order.
    fn build_batches(&mut self) -> Result<(), GraphError> {
        self.build_edges();
        let sorted = self.topological_sort()?;

        // Assign each system to the earliest batch it can join.
        let n = self.systems.len();
        let mut system_batch: Vec<usize> = vec![0; n];

        // Build reverse adjacency for computing earliest batch.
        for (from, neighbors) in self.edges.iter().enumerate() {
            for &(to, _) in neighbors {
                if self.systems[from].enabled && self.systems[to].enabled {
                    let new_batch = system_batch[from] + 1;
                    if new_batch > system_batch[to] {
                        system_batch[to] = new_batch;
                    }
                }
            }
        }

        // Group into batches.
        let max_batch = system_batch.iter().copied().max().unwrap_or(0);
        let mut batches: Vec<ParallelBatch> = (0..=max_batch)
            .map(|i| ParallelBatch {
                index: i,
                systems: Vec::new(),
            })
            .collect();

        for &sys_idx in &sorted {
            batches[system_batch[sys_idx]].systems.push(sys_idx);
        }

        // Remove empty batches.
        batches.retain(|b| !b.systems.is_empty());

        // Re-index.
        for (i, batch) in batches.iter_mut().enumerate() {
            batch.index = i;
        }

        self.batches = batches;
        self.dirty = false;
        Ok(())
    }

    /// Rebuild the graph if dirty, then return the parallel batches.
    pub fn get_batches(&mut self) -> Result<&[ParallelBatch], GraphError> {
        if self.dirty {
            self.build_batches()?;
        }
        Ok(&self.batches)
    }

    /// Get the linear execution order (topological sort).
    pub fn get_execution_order(&mut self) -> Result<Vec<SystemIndex>, GraphError> {
        if self.dirty {
            self.build_batches()?;
        }
        let mut order = Vec::new();
        for batch in &self.batches {
            order.extend(&batch.systems);
        }
        Ok(order)
    }

    /// Detect cycles in the graph.
    pub fn detect_cycles(&mut self) -> Option<Vec<SystemIndex>> {
        self.build_edges();
        match self.topological_sort() {
            Ok(_) => None,
            Err(GraphError::CycleDetected(cycle)) => Some(cycle),
            Err(_) => None,
        }
    }

    /// Get all edges in the graph.
    pub fn get_edges(&self) -> Vec<DependencyEdge> {
        let mut result = Vec::new();
        for (from, neighbors) in self.edges.iter().enumerate() {
            for (to, reason) in neighbors {
                result.push(DependencyEdge {
                    from,
                    to: *to,
                    reason: reason.clone(),
                });
            }
        }
        result
    }

    /// Generate DOT graph visualization data.
    pub fn to_dot(&mut self) -> String {
        self.build_edges();
        let mut dot = String::from("digraph SystemGraph {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        for sys in &self.systems {
            if sys.enabled {
                let color = if sys.accesses.iter().any(|a| a.kind == ComponentAccessKind::Write) {
                    "lightcoral"
                } else {
                    "lightblue"
                };
                dot.push_str(&format!(
                    "  {} [label=\"{}\\n{}\" style=filled fillcolor={}];\n",
                    sys.index, sys.name, sys.stage, color
                ));
            }
        }

        dot.push('\n');
        for (from, neighbors) in self.edges.iter().enumerate() {
            for (to, reason) in neighbors {
                let label = match reason {
                    DependencyReason::ComponentConflict(c) => c.clone(),
                    DependencyReason::ExplicitOrder => "order".to_string(),
                    DependencyReason::StageBoundary => "stage".to_string(),
                };
                dot.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    from, to, label
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Get a summary of the graph.
    pub fn summary(&self) -> GraphSummary {
        let enabled_count = self.systems.iter().filter(|s| s.enabled).count();
        let edge_count: usize = self.edges.iter().map(|e| e.len()).sum();
        GraphSummary {
            total_systems: self.systems.len(),
            enabled_systems: enabled_count,
            disabled_systems: self.systems.len() - enabled_count,
            edge_count,
            batch_count: self.batches.len(),
            stage_count: self.stage_order.len(),
        }
    }
}

impl Default for SystemGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SystemGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemGraph")
            .field("systems", &self.systems.len())
            .field("dirty", &self.dirty)
            .finish()
    }
}

/// Summary statistics for the system graph.
#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub total_systems: usize,
    pub enabled_systems: usize,
    pub disabled_systems: usize,
    pub edge_count: usize,
    pub batch_count: usize,
    pub stage_count: usize,
}

impl fmt::Display for GraphSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "System Graph Summary:")?;
        writeln!(f, "  total systems:  {}", self.total_systems)?;
        writeln!(f, "  enabled:        {}", self.enabled_systems)?;
        writeln!(f, "  disabled:       {}", self.disabled_systems)?;
        writeln!(f, "  edges:          {}", self.edge_count)?;
        writeln!(f, "  batches:        {}", self.batch_count)?;
        writeln!(f, "  stages:         {}", self.stage_count)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_conflicts() {
        let mut graph = SystemGraph::new();
        graph.add_system(SystemNode::new(0, "Input").reads("InputState"));
        graph.add_system(SystemNode::new(1, "Render").reads("Transform").reads("Mesh"));
        graph.add_system(SystemNode::new(2, "Audio").reads("Transform").reads("AudioSource"));

        let batches = graph.get_batches().unwrap();
        // All systems only read, so they can all be in one batch.
        assert!(batches.len() <= 2);
    }

    #[test]
    fn test_write_conflict() {
        let mut graph = SystemGraph::new();
        graph.add_system(SystemNode::new(0, "Physics").writes("Transform").writes("Velocity"));
        graph.add_system(SystemNode::new(1, "Render").reads("Transform"));

        let batches = graph.get_batches().unwrap();
        assert!(batches.len() >= 2);
        // Physics must be in an earlier batch than Render.
    }

    #[test]
    fn test_explicit_ordering() {
        let mut graph = SystemGraph::new();
        let a = graph.add_system(SystemNode::new(0, "A"));
        let b = graph.add_system(SystemNode::new(1, "B"));
        graph.add_ordering(a, b);

        let order = graph.get_execution_order().unwrap();
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        assert!(pos_a < pos_b);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = SystemGraph::new();
        let a = graph.add_system(SystemNode::new(0, "A"));
        let b = graph.add_system(SystemNode::new(1, "B"));
        graph.add_ordering(a, b);
        graph.add_ordering(b, a);

        let cycles = graph.detect_cycles();
        assert!(cycles.is_some());
    }

    #[test]
    fn test_disable_system() {
        let mut graph = SystemGraph::new();
        let a = graph.add_system(SystemNode::new(0, "A").writes("Transform"));
        let b = graph.add_system(SystemNode::new(1, "B").writes("Transform"));

        graph.set_enabled(b, false);
        let order = graph.get_execution_order().unwrap();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], a);
    }

    #[test]
    fn test_find_by_name() {
        let mut graph = SystemGraph::new();
        graph.add_system(SystemNode::new(0, "Physics"));
        graph.add_system(SystemNode::new(1, "Render"));

        assert_eq!(graph.find_by_name("Physics"), Some(0));
        assert_eq!(graph.find_by_name("Render"), Some(1));
        assert_eq!(graph.find_by_name("Missing"), None);
    }

    #[test]
    fn test_dot_output() {
        let mut graph = SystemGraph::new();
        graph.add_system(SystemNode::new(0, "A").writes("Transform"));
        graph.add_system(SystemNode::new(1, "B").reads("Transform"));

        let dot = graph.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("A"));
        assert!(dot.contains("B"));
    }
}
