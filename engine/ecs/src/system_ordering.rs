// engine/ecs/src/system_ordering.rs
//
// System ordering: before/after constraints, system sets, topological sort,
// cycle detection, and automatic ordering from resource access patterns.
// Produces a deterministic execution schedule for ECS systems.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A unique identifier for a system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemId(pub u32);

/// A unique identifier for a system set (group of systems).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemSetId(pub u32);

/// Access type for a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    Read,
    Write,
}

/// A resource access declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceAccess {
    pub resource_id: u64,
    pub resource_name: String,
    pub access_type: AccessType,
}

/// An ordering constraint between systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderingConstraint {
    /// System A must run before System B.
    Before { system: SystemId, before: SystemId },
    /// System A must run after System B.
    After { system: SystemId, after: SystemId },
    /// System must run within a specific set.
    InSet { system: SystemId, set: SystemSetId },
    /// Set A must run before Set B.
    SetBefore { set: SystemSetId, before: SystemSetId },
}

/// System registration data.
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub id: SystemId,
    pub name: String,
    pub set: Option<SystemSetId>,
    pub reads: Vec<ResourceAccess>,
    pub writes: Vec<ResourceAccess>,
    pub exclusive: bool,
    pub thread_local: bool,
    pub enabled: bool,
    pub priority: i32,
}

impl SystemInfo {
    pub fn new(id: SystemId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            set: None,
            reads: Vec::new(),
            writes: Vec::new(),
            exclusive: false,
            thread_local: false,
            enabled: true,
            priority: 0,
        }
    }

    pub fn with_read(mut self, resource_name: &str) -> Self {
        self.reads.push(ResourceAccess {
            resource_id: hash_resource_name(resource_name),
            resource_name: resource_name.to_string(),
            access_type: AccessType::Read,
        });
        self
    }

    pub fn with_write(mut self, resource_name: &str) -> Self {
        self.writes.push(ResourceAccess {
            resource_id: hash_resource_name(resource_name),
            resource_name: resource_name.to_string(),
            access_type: AccessType::Write,
        });
        self
    }

    pub fn with_set(mut self, set: SystemSetId) -> Self {
        self.set = Some(set);
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn exclusive(mut self) -> Self {
        self.exclusive = true;
        self
    }

    /// Check if this system conflicts with another (data race).
    pub fn conflicts_with(&self, other: &SystemInfo) -> bool {
        // Two systems conflict if one writes a resource the other reads or writes.
        for w in &self.writes {
            for r in &other.reads {
                if w.resource_id == r.resource_id {
                    return true;
                }
            }
            for w2 in &other.writes {
                if w.resource_id == w2.resource_id {
                    return true;
                }
            }
        }
        for w in &other.writes {
            for r in &self.reads {
                if w.resource_id == r.resource_id {
                    return true;
                }
            }
        }
        false
    }
}

fn hash_resource_name(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// System set data.
#[derive(Debug, Clone)]
pub struct SystemSetInfo {
    pub id: SystemSetId,
    pub name: String,
    pub systems: Vec<SystemId>,
}

impl SystemSetInfo {
    pub fn new(id: SystemSetId, name: &str) -> Self {
        Self { id, name: name.to_string(), systems: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Result of building a schedule.
#[derive(Debug, Clone)]
pub struct SystemSchedule {
    /// Systems in execution order. Systems within the same batch can run in parallel.
    pub batches: Vec<SystemBatch>,
    /// Total number of systems.
    pub total_systems: usize,
    /// Maximum parallelism (largest batch size).
    pub max_parallelism: usize,
    /// Whether automatic ordering from resource access was applied.
    pub auto_ordered: bool,
}

/// A batch of systems that can run in parallel.
#[derive(Debug, Clone)]
pub struct SystemBatch {
    pub systems: Vec<SystemId>,
    pub batch_index: usize,
}

/// Errors from the ordering system.
#[derive(Debug, Clone)]
pub enum OrderingError {
    /// A cycle was detected in the dependency graph.
    CycleDetected { cycle: Vec<SystemId> },
    /// A system was referenced but not registered.
    UnknownSystem(SystemId),
    /// A system set was referenced but not registered.
    UnknownSet(SystemSetId),
    /// Conflicting constraints.
    ConflictingConstraints(String),
}

impl std::fmt::Display for OrderingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected { cycle } => {
                write!(f, "Cycle detected: ")?;
                for (i, id) in cycle.iter().enumerate() {
                    if i > 0 { write!(f, " -> ")?; }
                    write!(f, "S{}", id.0)?;
                }
                Ok(())
            }
            Self::UnknownSystem(id) => write!(f, "Unknown system: {}", id.0),
            Self::UnknownSet(id) => write!(f, "Unknown system set: {}", id.0),
            Self::ConflictingConstraints(msg) => write!(f, "Conflicting constraints: {msg}"),
        }
    }
}

impl std::error::Error for OrderingError {}

// ---------------------------------------------------------------------------
// SystemOrderingBuilder
// ---------------------------------------------------------------------------

/// Builds and resolves system execution order.
pub struct SystemOrderingBuilder {
    systems: HashMap<u32, SystemInfo>,
    sets: HashMap<u32, SystemSetInfo>,
    constraints: Vec<OrderingConstraint>,
    next_system_id: u32,
    next_set_id: u32,
    auto_order: bool,
}

impl SystemOrderingBuilder {
    pub fn new() -> Self {
        Self {
            systems: HashMap::new(),
            sets: HashMap::new(),
            constraints: Vec::new(),
            next_system_id: 0,
            next_set_id: 0,
            auto_order: true,
        }
    }

    /// Whether to automatically infer ordering from resource access patterns.
    pub fn set_auto_order(&mut self, enabled: bool) {
        self.auto_order = enabled;
    }

    /// Register a new system.
    pub fn add_system(&mut self, info: SystemInfo) -> SystemId {
        let id = info.id;
        self.systems.insert(id.0, info);
        id
    }

    /// Allocate a new system ID.
    pub fn allocate_system_id(&mut self) -> SystemId {
        let id = SystemId(self.next_system_id);
        self.next_system_id += 1;
        id
    }

    /// Register a new system set.
    pub fn add_set(&mut self, info: SystemSetInfo) -> SystemSetId {
        let id = info.id;
        self.sets.insert(id.0, info);
        id
    }

    /// Allocate a new set ID.
    pub fn allocate_set_id(&mut self) -> SystemSetId {
        let id = SystemSetId(self.next_set_id);
        self.next_set_id += 1;
        id
    }

    /// Add a "before" constraint: `system` must run before `before`.
    pub fn add_before(&mut self, system: SystemId, before: SystemId) {
        self.constraints.push(OrderingConstraint::Before { system, before });
    }

    /// Add an "after" constraint: `system` must run after `after`.
    pub fn add_after(&mut self, system: SystemId, after: SystemId) {
        self.constraints.push(OrderingConstraint::After { system, after });
    }

    /// Add a set ordering constraint.
    pub fn add_set_before(&mut self, set: SystemSetId, before: SystemSetId) {
        self.constraints.push(OrderingConstraint::SetBefore { set, before });
    }

    /// Build the execution schedule.
    pub fn build(&self) -> Result<SystemSchedule, OrderingError> {
        // Build adjacency list (directed graph).
        let system_ids: Vec<SystemId> = self.systems.values()
            .filter(|s| s.enabled)
            .map(|s| s.id)
            .collect();

        let mut edges: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();

        for &id in &system_ids {
            edges.entry(id.0).or_insert_with(HashSet::new);
            in_degree.entry(id.0).or_insert(0);
        }

        // Process explicit constraints.
        for constraint in &self.constraints {
            match constraint {
                OrderingConstraint::Before { system, before } => {
                    if edges.entry(system.0).or_default().insert(before.0) {
                        *in_degree.entry(before.0).or_insert(0) += 1;
                    }
                }
                OrderingConstraint::After { system, after } => {
                    if edges.entry(after.0).or_default().insert(system.0) {
                        *in_degree.entry(system.0).or_insert(0) += 1;
                    }
                }
                OrderingConstraint::SetBefore { set, before } => {
                    // All systems in `set` must run before all systems in `before`.
                    let set_systems: Vec<SystemId> = self.systems.values()
                        .filter(|s| s.set == Some(*set))
                        .map(|s| s.id)
                        .collect();
                    let before_systems: Vec<SystemId> = self.systems.values()
                        .filter(|s| s.set == Some(*before))
                        .map(|s| s.id)
                        .collect();
                    for &from in &set_systems {
                        for &to in &before_systems {
                            if edges.entry(from.0).or_default().insert(to.0) {
                                *in_degree.entry(to.0).or_insert(0) += 1;
                            }
                        }
                    }
                }
                OrderingConstraint::InSet { system, set } => {
                    // Just for grouping, doesn't add edges by itself.
                    let _ = (system, set);
                }
            }
        }

        // Auto-order from resource access (writers before readers of the same resource).
        if self.auto_order {
            let sys_list: Vec<&SystemInfo> = self.systems.values()
                .filter(|s| s.enabled)
                .collect();

            for i in 0..sys_list.len() {
                for j in (i + 1)..sys_list.len() {
                    let a = sys_list[i];
                    let b = sys_list[j];

                    // If A writes something B reads, A should run before B.
                    let a_writes_b_reads = a.writes.iter().any(|w| {
                        b.reads.iter().any(|r| r.resource_id == w.resource_id)
                    });
                    if a_writes_b_reads {
                        if edges.entry(a.id.0).or_default().insert(b.id.0) {
                            *in_degree.entry(b.id.0).or_insert(0) += 1;
                        }
                    }

                    // If B writes something A reads, B should run before A.
                    let b_writes_a_reads = b.writes.iter().any(|w| {
                        a.reads.iter().any(|r| r.resource_id == w.resource_id)
                    });
                    if b_writes_a_reads {
                        if edges.entry(b.id.0).or_default().insert(a.id.0) {
                            *in_degree.entry(a.id.0).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Kahn's algorithm for topological sort with batching.
        let mut batches: Vec<SystemBatch> = Vec::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        // Find all nodes with in-degree 0.
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut processed = 0usize;
        let mut batch_index = 0;

        while !queue.is_empty() {
            // All systems in the current queue can run in parallel.
            let mut batch_systems: Vec<u32> = Vec::new();
            let batch_size = queue.len();

            for _ in 0..batch_size {
                let node = queue.pop_front().unwrap();
                batch_systems.push(node);
                processed += 1;
            }

            // Sort by priority within batch for determinism.
            batch_systems.sort_by(|a, b| {
                let pa = self.systems.get(a).map(|s| s.priority).unwrap_or(0);
                let pb = self.systems.get(b).map(|s| s.priority).unwrap_or(0);
                pb.cmp(&pa).then(a.cmp(b))
            });

            // Further split batch: exclusive systems must be alone.
            let (exclusive, parallel): (Vec<_>, Vec<_>) = batch_systems.iter()
                .partition(|&&id| {
                    self.systems.get(&id).map(|s| s.exclusive).unwrap_or(false)
                });

            if !parallel.is_empty() {
                // Check for conflicts within the parallel batch.
                let conflict_groups = self.partition_by_conflicts(&parallel);
                for group in conflict_groups {
                    batches.push(SystemBatch {
                        systems: group.into_iter().map(SystemId).collect(),
                        batch_index,
                    });
                    batch_index += 1;
                }
            }

            for &exc_id in &exclusive {
                batches.push(SystemBatch {
                    systems: vec![SystemId(*exc_id)],
                    batch_index,
                });
                batch_index += 1;
            }

            // Decrease in-degree of successors.
            for &node in &batch_systems {
                if let Some(successors) = edges.get(&node) {
                    for &succ in successors {
                        if let Some(deg) = in_degree.get_mut(&succ) {
                            *deg -= 1;
                            if *deg == 0 {
                                queue.push_back(succ);
                            }
                        }
                    }
                }
            }
        }

        // Check for cycles.
        if processed < system_ids.len() {
            let cycle = self.find_cycle(&edges, &in_degree);
            return Err(OrderingError::CycleDetected { cycle });
        }

        let max_parallelism = batches.iter().map(|b| b.systems.len()).max().unwrap_or(0);

        Ok(SystemSchedule {
            batches,
            total_systems: system_ids.len(),
            max_parallelism,
            auto_ordered: self.auto_order,
        })
    }

    /// Partition a set of system IDs into groups where systems within each
    /// group do not conflict (can run in parallel).
    fn partition_by_conflicts(&self, ids: &[&u32]) -> Vec<Vec<u32>> {
        let mut groups: Vec<Vec<u32>> = Vec::new();

        for &&id in ids {
            let sys = match self.systems.get(&id) {
                Some(s) => s,
                None => continue,
            };

            let mut placed = false;
            for group in &mut groups {
                let conflicts = group.iter().any(|&gid| {
                    self.systems.get(&gid)
                        .map(|gs| gs.conflicts_with(sys))
                        .unwrap_or(false)
                });
                if !conflicts {
                    group.push(id);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![id]);
            }
        }

        groups
    }

    /// Find a cycle in the graph (for error reporting).
    fn find_cycle(&self, edges: &HashMap<u32, HashSet<u32>>, in_degree: &HashMap<u32, usize>) -> Vec<SystemId> {
        // Find a node that still has in-degree > 0 (part of a cycle).
        let start = in_degree.iter()
            .find(|(_, &deg)| deg > 0)
            .map(|(&id, _)| id);

        let start = match start {
            Some(id) => id,
            None => return Vec::new(),
        };

        // DFS to find the cycle.
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut stack = vec![(start, false)];

        while let Some((node, processed)) = stack.pop() {
            if processed {
                path.pop();
                continue;
            }

            if visited.contains(&node) {
                // Found cycle.
                let cycle_start = path.iter().position(|&n| n == node).unwrap_or(0);
                return path[cycle_start..].iter().map(|&n| SystemId(n)).collect();
            }

            visited.insert(node);
            path.push(node);
            stack.push((node, true));

            if let Some(successors) = edges.get(&node) {
                for &succ in successors {
                    if in_degree.get(&succ).copied().unwrap_or(0) > 0 {
                        stack.push((succ, false));
                    }
                }
            }
        }

        path.iter().map(|&n| SystemId(n)).collect()
    }

    /// Get a system's info by ID.
    pub fn get_system(&self, id: SystemId) -> Option<&SystemInfo> {
        self.systems.get(&id.0)
    }

    /// Get all system infos.
    pub fn systems(&self) -> impl Iterator<Item = &SystemInfo> {
        self.systems.values()
    }
}

impl Default for SystemOrderingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

/// Generate a DOT graph representation for debugging.
pub fn schedule_to_dot(builder: &SystemOrderingBuilder, schedule: &SystemSchedule) -> String {
    let mut dot = String::from("digraph SystemSchedule {\n  rankdir=LR;\n");

    for batch in &schedule.batches {
        dot.push_str(&format!("  subgraph cluster_{} {{\n", batch.batch_index));
        dot.push_str(&format!("    label=\"Batch {}\";\n", batch.batch_index));
        for sys in &batch.systems {
            let name = builder.get_system(*sys)
                .map(|s| s.name.as_str())
                .unwrap_or("?");
            dot.push_str(&format!("    S{} [label=\"{}\"];\n", sys.0, name));
        }
        dot.push_str("  }\n");
    }

    dot.push_str("}\n");
    dot
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ordering() {
        let mut builder = SystemOrderingBuilder::new();
        builder.set_auto_order(false);

        let a = SystemId(0);
        let b = SystemId(1);
        let c = SystemId(2);

        builder.add_system(SystemInfo::new(a, "A"));
        builder.add_system(SystemInfo::new(b, "B"));
        builder.add_system(SystemInfo::new(c, "C"));

        builder.add_before(a, b); // A before B
        builder.add_before(b, c); // B before C

        let schedule = builder.build().unwrap();

        // A must come before B, B before C.
        let a_batch = schedule.batches.iter().position(|b| b.systems.contains(&a)).unwrap();
        let b_batch = schedule.batches.iter().position(|b| b.systems.contains(&b)).unwrap();
        let c_batch = schedule.batches.iter().position(|b| b.systems.contains(&c)).unwrap();

        assert!(a_batch <= b_batch);
        assert!(b_batch <= c_batch);
    }

    #[test]
    fn test_cycle_detection() {
        let mut builder = SystemOrderingBuilder::new();
        builder.set_auto_order(false);

        let a = SystemId(0);
        let b = SystemId(1);

        builder.add_system(SystemInfo::new(a, "A"));
        builder.add_system(SystemInfo::new(b, "B"));

        builder.add_before(a, b);
        builder.add_before(b, a); // cycle!

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_ordering() {
        let mut builder = SystemOrderingBuilder::new();

        let writer = SystemId(0);
        let reader = SystemId(1);

        builder.add_system(SystemInfo::new(writer, "Writer").with_write("transform"));
        builder.add_system(SystemInfo::new(reader, "Reader").with_read("transform"));

        let schedule = builder.build().unwrap();

        let w_batch = schedule.batches.iter().position(|b| b.systems.contains(&writer)).unwrap();
        let r_batch = schedule.batches.iter().position(|b| b.systems.contains(&reader)).unwrap();

        assert!(w_batch < r_batch, "Writer should be scheduled before Reader");
    }

    #[test]
    fn test_parallel_batching() {
        let mut builder = SystemOrderingBuilder::new();
        builder.set_auto_order(true);

        let a = SystemId(0);
        let b = SystemId(1);

        builder.add_system(SystemInfo::new(a, "A").with_read("x"));
        builder.add_system(SystemInfo::new(b, "B").with_read("y")); // no conflict

        let schedule = builder.build().unwrap();

        // A and B should be in the same batch since they don't conflict.
        assert!(schedule.batches.iter().any(|batch| {
            batch.systems.contains(&a) && batch.systems.contains(&b)
        }));
    }

    #[test]
    fn test_conflict_separation() {
        let mut builder = SystemOrderingBuilder::new();
        builder.set_auto_order(false);

        let a = SystemId(0);
        let b = SystemId(1);

        builder.add_system(SystemInfo::new(a, "A").with_write("transform"));
        builder.add_system(SystemInfo::new(b, "B").with_write("transform")); // conflict!

        let schedule = builder.build().unwrap();

        // A and B should NOT be in the same batch.
        assert!(!schedule.batches.iter().any(|batch| {
            batch.systems.contains(&a) && batch.systems.contains(&b)
        }));
    }

    #[test]
    fn test_exclusive_system() {
        let mut builder = SystemOrderingBuilder::new();
        builder.set_auto_order(false);

        let a = SystemId(0);
        let b = SystemId(1);

        builder.add_system(SystemInfo::new(a, "A").exclusive());
        builder.add_system(SystemInfo::new(b, "B"));

        let schedule = builder.build().unwrap();

        // Exclusive system should be alone in its batch.
        let a_batch = schedule.batches.iter().find(|batch| batch.systems.contains(&a)).unwrap();
        assert_eq!(a_batch.systems.len(), 1);
    }
}
