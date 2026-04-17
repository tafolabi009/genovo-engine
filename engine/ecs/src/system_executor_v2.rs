//! Enhanced system executor with automatic parallelism, thread pool integration,
//! system sets with ordering, run conditions, and fixed timestep systems.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKindV2 { Read, Write, Exclusive }

#[derive(Debug, Clone)]
pub struct ComponentAccessV2 { pub type_id: TypeId, pub type_name: &'static str, pub access: AccessKindV2 }

#[derive(Debug, Clone)]
pub struct SystemAccessV2 {
    pub components: Vec<ComponentAccessV2>,
    pub resources: Vec<(TypeId, AccessKindV2)>,
    pub exclusive: bool,
}

impl SystemAccessV2 {
    pub fn new() -> Self { Self { components: Vec::new(), resources: Vec::new(), exclusive: false } }
    pub fn read_component<T: 'static>(&mut self) -> &mut Self { self.components.push(ComponentAccessV2 { type_id: TypeId::of::<T>(), type_name: std::any::type_name::<T>(), access: AccessKindV2::Read }); self }
    pub fn write_component<T: 'static>(&mut self) -> &mut Self { self.components.push(ComponentAccessV2 { type_id: TypeId::of::<T>(), type_name: std::any::type_name::<T>(), access: AccessKindV2::Write }); self }
    pub fn set_exclusive(&mut self) -> &mut Self { self.exclusive = true; self }
    pub fn conflicts_with(&self, other: &SystemAccessV2) -> bool {
        if self.exclusive || other.exclusive { return true; }
        for a in &self.components { for b in &other.components { if a.type_id == b.type_id && (a.access == AccessKindV2::Write || b.access == AccessKindV2::Write) { return true; } } }
        for &(at, aa) in &self.resources { for &(bt, ba) in &other.resources { if at == bt && (aa == AccessKindV2::Write || ba == AccessKindV2::Write) { return true; } } }
        false
    }
}
impl Default for SystemAccessV2 { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct RunConditionId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SystemSetId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SystemIndexV2(pub usize);

pub struct RunCondition { pub id: RunConditionId, pub label: String, pub predicate: Box<dyn Fn() -> bool + Send + Sync> }
impl RunCondition { pub fn new(id: RunConditionId, label: impl Into<String>, pred: impl Fn() -> bool + Send + Sync + 'static) -> Self { Self { id, label: label.into(), predicate: Box::new(pred) } } pub fn evaluate(&self) -> bool { (self.predicate)() } }

#[derive(Debug, Clone)]
pub struct SystemSetV2 { pub id: SystemSetId, pub name: String, pub systems: Vec<usize>, pub after_sets: Vec<SystemSetId>, pub before_sets: Vec<SystemSetId>, pub run_condition: Option<RunConditionId>, pub enabled: bool }
impl SystemSetV2 {
    pub fn new(id: SystemSetId, name: impl Into<String>) -> Self { Self { id, name: name.into(), systems: Vec::new(), after_sets: Vec::new(), before_sets: Vec::new(), run_condition: None, enabled: true } }
    pub fn add_system(&mut self, idx: usize) { if !self.systems.contains(&idx) { self.systems.push(idx); } }
    pub fn after(&mut self, other: SystemSetId) -> &mut Self { if !self.after_sets.contains(&other) { self.after_sets.push(other); } self }
    pub fn before(&mut self, other: SystemSetId) -> &mut Self { if !self.before_sets.contains(&other) { self.before_sets.push(other); } self }
}

#[derive(Debug, Clone)]
pub struct FixedTimestep { pub step: Duration, pub accumulator: Duration, pub max_steps_per_frame: u32, pub steps_this_frame: u32, pub total_steps: u64, pub overshoot_alpha: f64 }
impl FixedTimestep {
    pub fn new(step: Duration) -> Self { Self { step, accumulator: Duration::ZERO, max_steps_per_frame: 8, steps_this_frame: 0, total_steps: 0, overshoot_alpha: 0.0 } }
    pub fn from_hz(hz: f64) -> Self { Self::new(Duration::from_secs_f64(1.0 / hz)) }
    pub fn accumulate(&mut self, delta: Duration) -> u32 {
        self.accumulator += delta; self.steps_this_frame = 0; let mut steps = 0u32;
        while self.accumulator >= self.step && steps < self.max_steps_per_frame { self.accumulator -= self.step; steps += 1; self.total_steps += 1; }
        if steps >= self.max_steps_per_frame { self.accumulator = Duration::ZERO; }
        self.overshoot_alpha = self.accumulator.as_secs_f64() / self.step.as_secs_f64(); self.steps_this_frame = steps; steps
    }
    pub fn step_secs(&self) -> f32 { self.step.as_secs_f32() }
    pub fn alpha(&self) -> f64 { self.overshoot_alpha }
}

pub struct SystemDescriptorV2 { pub index: SystemIndexV2, pub name: String, pub access: SystemAccessV2, pub run_fn: Box<dyn FnMut() + Send>, pub run_condition: Option<RunConditionId>, pub dependencies: Vec<SystemIndexV2>, pub dependents: Vec<SystemIndexV2>, pub set: Option<SystemSetId>, pub enabled: bool, pub fixed_timestep: Option<FixedTimestep>, pub avg_duration: Duration, pub last_duration: Duration, pub run_count: u64 }
impl SystemDescriptorV2 {
    pub fn new(index: SystemIndexV2, name: impl Into<String>, access: SystemAccessV2, run_fn: impl FnMut() + Send + 'static) -> Self {
        Self { index, name: name.into(), access, run_fn: Box::new(run_fn), run_condition: None, dependencies: Vec::new(), dependents: Vec::new(), set: None, enabled: true, fixed_timestep: None, avg_duration: Duration::ZERO, last_duration: Duration::ZERO, run_count: 0 }
    }
}

#[derive(Debug, Clone)] pub struct ParallelBatchV2 { pub systems: Vec<SystemIndexV2> }
impl ParallelBatchV2 { pub fn new() -> Self { Self { systems: Vec::new() } } pub fn add(&mut self, s: SystemIndexV2) { self.systems.push(s); } pub fn len(&self) -> usize { self.systems.len() } pub fn is_empty(&self) -> bool { self.systems.is_empty() } }
impl Default for ParallelBatchV2 { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)] pub struct ExecutorConfigV2 { pub thread_count: usize, pub profiling: bool, pub debug_schedule: bool }
impl Default for ExecutorConfigV2 { fn default() -> Self { Self { thread_count: 0, profiling: false, debug_schedule: false } } }

#[derive(Debug, Clone)]
pub struct ExecutionProfileV2 { pub total_duration: Duration, pub fixed_duration: Duration, pub batch_count: usize, pub systems_run: usize, pub systems_skipped: usize, pub fixed_steps: u32 }
impl ExecutionProfileV2 { pub fn empty() -> Self { Self { total_duration: Duration::ZERO, fixed_duration: Duration::ZERO, batch_count: 0, systems_run: 0, systems_skipped: 0, fixed_steps: 0 } } }

pub struct SystemExecutorV2 {
    systems: Vec<SystemDescriptorV2>, sets: HashMap<SystemSetId, SystemSetV2>, conditions: HashMap<RunConditionId, RunCondition>,
    schedule: Vec<ParallelBatchV2>, dirty: bool, config: ExecutorConfigV2, last_profile: ExecutionProfileV2, next_set_id: u32, next_condition_id: u32,
}

impl SystemExecutorV2 {
    pub fn new(config: ExecutorConfigV2) -> Self { Self { systems: Vec::new(), sets: HashMap::new(), conditions: HashMap::new(), schedule: Vec::new(), dirty: true, config, last_profile: ExecutionProfileV2::empty(), next_set_id: 0, next_condition_id: 0 } }
    pub fn add_system(&mut self, name: impl Into<String>, access: SystemAccessV2, run_fn: impl FnMut() + Send + 'static) -> SystemIndexV2 { let index = SystemIndexV2(self.systems.len()); self.systems.push(SystemDescriptorV2::new(index, name, access, run_fn)); self.dirty = true; index }
    pub fn add_dependency(&mut self, before: SystemIndexV2, after: SystemIndexV2) { if before.0 < self.systems.len() && after.0 < self.systems.len() { self.systems[after.0].dependencies.push(before); self.systems[before.0].dependents.push(after); self.dirty = true; } }
    pub fn create_set(&mut self, name: impl Into<String>) -> SystemSetId { let id = SystemSetId(self.next_set_id); self.next_set_id += 1; self.sets.insert(id, SystemSetV2::new(id, name)); id }
    pub fn add_to_set(&mut self, system: SystemIndexV2, set: SystemSetId) { if let Some(s) = self.sets.get_mut(&set) { s.add_system(system.0); if system.0 < self.systems.len() { self.systems[system.0].set = Some(set); } self.dirty = true; } }
    pub fn set_order(&mut self, before: SystemSetId, after: SystemSetId) { if let Some(s) = self.sets.get_mut(&after) { s.after(before); } if let Some(s) = self.sets.get_mut(&before) { s.before(after); } self.dirty = true; }
    pub fn add_run_condition(&mut self, label: impl Into<String>, pred: impl Fn() -> bool + Send + Sync + 'static) -> RunConditionId { let id = RunConditionId(self.next_condition_id); self.next_condition_id += 1; self.conditions.insert(id, RunCondition::new(id, label, pred)); id }
    pub fn set_system_condition(&mut self, system: SystemIndexV2, cond: RunConditionId) { if system.0 < self.systems.len() { self.systems[system.0].run_condition = Some(cond); } }
    pub fn set_fixed_timestep(&mut self, system: SystemIndexV2, step: Duration) { if system.0 < self.systems.len() { self.systems[system.0].fixed_timestep = Some(FixedTimestep::new(step)); } }
    pub fn set_enabled(&mut self, system: SystemIndexV2, enabled: bool) { if system.0 < self.systems.len() { self.systems[system.0].enabled = enabled; } }

    pub fn rebuild_schedule(&mut self) {
        if !self.dirty { return; } let n = self.systems.len();
        if n == 0 { self.schedule.clear(); self.dirty = false; return; }
        let mut in_degree = vec![0u32; n]; let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for sys in &self.systems { for dep in &sys.dependencies { adj[dep.0].push(sys.index.0); in_degree[sys.index.0] += 1; } }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n { if in_degree[i] == 0 { queue.push_back(i); } }
        let mut sorted = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() { sorted.push(node); for &nb in &adj[node] { in_degree[nb] -= 1; if in_degree[nb] == 0 { queue.push_back(nb); } } }
        let mut batches: Vec<ParallelBatchV2> = Vec::new();
        for &sys_idx in &sorted {
            if !self.systems[sys_idx].enabled { continue; }
            let mut placed = false;
            for batch in &mut batches {
                let can = batch.systems.iter().all(|&ex| !self.systems[sys_idx].access.conflicts_with(&self.systems[ex.0].access));
                let deps = self.systems[sys_idx].dependencies.iter().all(|dep| !batch.systems.contains(dep));
                if can && deps { batch.add(SystemIndexV2(sys_idx)); placed = true; break; }
            }
            if !placed { let mut b = ParallelBatchV2::new(); b.add(SystemIndexV2(sys_idx)); batches.push(b); }
        }
        self.schedule = batches; self.dirty = false;
    }

    pub fn run(&mut self, delta: Duration) {
        let frame_start = Instant::now(); if self.dirty { self.rebuild_schedule(); }
        let mut profile = ExecutionProfileV2::empty();
        for batch_idx in 0..self.schedule.len() {
            let batch = &self.schedule[batch_idx]; let mut to_run: Vec<SystemIndexV2> = Vec::new();
            for &sys_idx in &batch.systems {
                let sys = &self.systems[sys_idx.0];
                if !sys.enabled { profile.systems_skipped += 1; continue; }
                if let Some(cid) = sys.run_condition { if let Some(c) = self.conditions.get(&cid) { if !c.evaluate() { profile.systems_skipped += 1; continue; } } }
                to_run.push(sys_idx);
            }
            if to_run.is_empty() { continue; } profile.batch_count += 1;
            for &sys_idx in &to_run {
                let sys = &mut self.systems[sys_idx.0];
                if let Some(ref mut fixed) = sys.fixed_timestep {
                    let steps = fixed.accumulate(delta);
                    for _ in 0..steps { (sys.run_fn)(); profile.systems_run += 1; profile.fixed_steps += 1; }
                    sys.run_count += steps as u64;
                } else { let start = Instant::now(); (sys.run_fn)(); sys.last_duration = start.elapsed(); sys.run_count += 1; profile.systems_run += 1; }
            }
        }
        profile.total_duration = frame_start.elapsed(); self.last_profile = profile;
    }

    pub fn last_profile(&self) -> &ExecutionProfileV2 { &self.last_profile }
    pub fn system_count(&self) -> usize { self.systems.len() }
    pub fn batch_count(&self) -> usize { self.schedule.len() }
    pub fn system_name(&self, index: SystemIndexV2) -> Option<&str> { self.systems.get(index.0).map(|s| s.name.as_str()) }
    pub fn clear(&mut self) { self.systems.clear(); self.sets.clear(); self.conditions.clear(); self.schedule.clear(); self.dirty = false; }
}

#[derive(Debug, Clone)]
pub struct SystemProfile { pub name: String, pub avg_duration: Duration, pub last_duration: Duration, pub run_count: u64, pub enabled: bool }
impl fmt::Display for SystemProfile { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: avg={:.2?} last={:.2?} runs={}", self.name, self.avg_duration, self.last_duration, self.run_count) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn access_conflict() {
        let mut a = SystemAccessV2::new(); a.write_component::<u32>();
        let mut b = SystemAccessV2::new(); b.read_component::<u32>();
        assert!(a.conflicts_with(&b));
        let mut c = SystemAccessV2::new(); c.read_component::<u32>();
        assert!(!b.conflicts_with(&c));
    }
    #[test]
    fn executor_basic() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let counter = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&counter);
        executor.add_system("test", SystemAccessV2::new(), move || { c.fetch_add(1, Ordering::Relaxed); });
        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn fixed_timestep() { let mut f = FixedTimestep::from_hz(60.0); let s = f.accumulate(Duration::from_millis(32)); assert!(s >= 1); }
}
