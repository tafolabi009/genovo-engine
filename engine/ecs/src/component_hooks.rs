// engine/ecs/src/component_hooks.rs
//
// Component lifecycle hooks for the Genovo ECS.
//
// Provides on_insert, on_remove, on_modify callbacks per component type,
// hook registration, batch hook processing, and hook priority ordering.

use std::any::TypeId;
use std::collections::{HashMap, BTreeMap};
use std::fmt;

pub type Entity = u64;
pub type HookId = u64;
pub type ComponentTypeId = TypeId;

pub const MAX_HOOKS_PER_TYPE: usize = 64;
pub const DEFAULT_HOOK_PRIORITY: i32 = 0;
pub const HOOK_PRIORITY_FIRST: i32 = i32::MIN;
pub const HOOK_PRIORITY_LAST: i32 = i32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEvent { OnInsert, OnRemove, OnModify }

impl fmt::Display for HookEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::OnInsert => write!(f, "on_insert"), Self::OnRemove => write!(f, "on_remove"), Self::OnModify => write!(f, "on_modify") }
    }
}

#[derive(Debug, Clone)]
pub struct HookContext {
    pub entity: Entity,
    pub component_type: ComponentTypeId,
    pub event: HookEvent,
    pub frame: u64,
}

pub type HookCallback = Box<dyn Fn(&HookContext) + Send + Sync>;

struct RegisteredHook {
    id: HookId,
    priority: i32,
    event: HookEvent,
    component_type: ComponentTypeId,
    callback: HookCallback,
    enabled: bool,
    name: String,
    invocation_count: u64,
}

#[derive(Debug, Clone)]
struct PendingHookEvent {
    entity: Entity,
    component_type: ComponentTypeId,
    event: HookEvent,
    frame: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HookStats {
    pub total_hooks_registered: u32,
    pub total_hooks_invoked: u64,
    pub pending_events: usize,
    pub insert_hooks: u32,
    pub remove_hooks: u32,
    pub modify_hooks: u32,
    pub batch_events_processed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookProcessingMode {
    Immediate,
    Deferred,
    BatchEndOfFrame,
}

pub struct ComponentHookRegistry {
    hooks: Vec<RegisteredHook>,
    hook_index: HashMap<(ComponentTypeId, HookEvent), Vec<usize>>,
    pending_events: Vec<PendingHookEvent>,
    next_id: HookId,
    stats: HookStats,
    processing_mode: HookProcessingMode,
    current_frame: u64,
}

impl ComponentHookRegistry {
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(), hook_index: HashMap::new(),
            pending_events: Vec::new(), next_id: 1,
            stats: HookStats::default(),
            processing_mode: HookProcessingMode::Deferred,
            current_frame: 0,
        }
    }

    pub fn register<F>(&mut self, component_type: ComponentTypeId, event: HookEvent, name: &str, priority: i32, callback: F) -> HookId
    where F: Fn(&HookContext) + Send + Sync + 'static {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.hooks.len();
        self.hooks.push(RegisteredHook {
            id, priority, event, component_type, callback: Box::new(callback),
            enabled: true, name: name.to_string(), invocation_count: 0,
        });
        self.hook_index.entry((component_type, event)).or_default().push(idx);
        // Sort hooks by priority for this component/event combo.
        if let Some(indices) = self.hook_index.get_mut(&(component_type, event)) {
            indices.sort_by_key(|&i| self.hooks[i].priority);
        }
        match event {
            HookEvent::OnInsert => self.stats.insert_hooks += 1,
            HookEvent::OnRemove => self.stats.remove_hooks += 1,
            HookEvent::OnModify => self.stats.modify_hooks += 1,
        }
        self.stats.total_hooks_registered += 1;
        id
    }

    pub fn on_insert<C: 'static, F>(&mut self, name: &str, callback: F) -> HookId
    where F: Fn(&HookContext) + Send + Sync + 'static {
        self.register(TypeId::of::<C>(), HookEvent::OnInsert, name, DEFAULT_HOOK_PRIORITY, callback)
    }

    pub fn on_remove<C: 'static, F>(&mut self, name: &str, callback: F) -> HookId
    where F: Fn(&HookContext) + Send + Sync + 'static {
        self.register(TypeId::of::<C>(), HookEvent::OnRemove, name, DEFAULT_HOOK_PRIORITY, callback)
    }

    pub fn on_modify<C: 'static, F>(&mut self, name: &str, callback: F) -> HookId
    where F: Fn(&HookContext) + Send + Sync + 'static {
        self.register(TypeId::of::<C>(), HookEvent::OnModify, name, DEFAULT_HOOK_PRIORITY, callback)
    }

    pub fn on_insert_priority<C: 'static, F>(&mut self, name: &str, priority: i32, callback: F) -> HookId
    where F: Fn(&HookContext) + Send + Sync + 'static {
        self.register(TypeId::of::<C>(), HookEvent::OnInsert, name, priority, callback)
    }

    pub fn unregister(&mut self, hook_id: HookId) {
        if let Some(pos) = self.hooks.iter().position(|h| h.id == hook_id) {
            let hook = &self.hooks[pos];
            let key = (hook.component_type, hook.event);
            if let Some(indices) = self.hook_index.get_mut(&key) {
                indices.retain(|&i| i != pos);
            }
            self.hooks[pos].enabled = false;
        }
    }

    pub fn set_enabled(&mut self, hook_id: HookId, enabled: bool) {
        if let Some(hook) = self.hooks.iter_mut().find(|h| h.id == hook_id) { hook.enabled = enabled; }
    }

    pub fn notify(&mut self, entity: Entity, component_type: ComponentTypeId, event: HookEvent) {
        match self.processing_mode {
            HookProcessingMode::Immediate => {
                self.dispatch_event(entity, component_type, event);
            }
            HookProcessingMode::Deferred | HookProcessingMode::BatchEndOfFrame => {
                self.pending_events.push(PendingHookEvent { entity, component_type, event, frame: self.current_frame });
            }
        }
    }

    pub fn notify_insert<C: 'static>(&mut self, entity: Entity) {
        self.notify(entity, TypeId::of::<C>(), HookEvent::OnInsert);
    }

    pub fn notify_remove<C: 'static>(&mut self, entity: Entity) {
        self.notify(entity, TypeId::of::<C>(), HookEvent::OnRemove);
    }

    pub fn notify_modify<C: 'static>(&mut self, entity: Entity) {
        self.notify(entity, TypeId::of::<C>(), HookEvent::OnModify);
    }

    fn dispatch_event(&mut self, entity: Entity, component_type: ComponentTypeId, event: HookEvent) {
        let key = (component_type, event);
        if let Some(indices) = self.hook_index.get(&key).cloned() {
            let ctx = HookContext { entity, component_type, event, frame: self.current_frame };
            for &idx in &indices {
                if idx < self.hooks.len() && self.hooks[idx].enabled {
                    (self.hooks[idx].callback)(&ctx);
                    self.hooks[idx].invocation_count += 1;
                    self.stats.total_hooks_invoked += 1;
                }
            }
        }
    }

    pub fn flush(&mut self) {
        let events: Vec<PendingHookEvent> = self.pending_events.drain(..).collect();
        self.stats.batch_events_processed += events.len() as u64;
        for event in events {
            self.dispatch_event(event.entity, event.component_type, event.event);
        }
    }

    pub fn begin_frame(&mut self) { self.current_frame += 1; self.stats.pending_events = 0; }

    pub fn end_frame(&mut self) {
        if self.processing_mode == HookProcessingMode::BatchEndOfFrame { self.flush(); }
        self.stats.pending_events = self.pending_events.len();
    }

    pub fn set_processing_mode(&mut self, mode: HookProcessingMode) { self.processing_mode = mode; }
    pub fn processing_mode(&self) -> HookProcessingMode { self.processing_mode }
    pub fn pending_count(&self) -> usize { self.pending_events.len() }
    pub fn hook_count(&self) -> usize { self.hooks.iter().filter(|h| h.enabled).count() }
    pub fn stats(&self) -> &HookStats { &self.stats }

    pub fn hook_info(&self, id: HookId) -> Option<(String, HookEvent, i32, u64)> {
        self.hooks.iter().find(|h| h.id == id).map(|h| (h.name.clone(), h.event, h.priority, h.invocation_count))
    }

    pub fn clear_all(&mut self) {
        self.hooks.clear(); self.hook_index.clear();
        self.pending_events.clear(); self.stats = HookStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct Health(f32);
    struct Position(f32, f32, f32);

    #[test]
    fn test_register_and_notify() {
        let mut registry = ComponentHookRegistry::new();
        registry.set_processing_mode(HookProcessingMode::Immediate);
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        registry.on_insert::<Health, _>("health_added", move |ctx| { c.fetch_add(1, Ordering::Relaxed); });
        registry.notify_insert::<Health>(1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_deferred_processing() {
        let mut registry = ComponentHookRegistry::new();
        registry.set_processing_mode(HookProcessingMode::Deferred);
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        registry.on_insert::<Health, _>("h", move |_| { c.fetch_add(1, Ordering::Relaxed); });
        registry.notify_insert::<Health>(1);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        registry.flush();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut registry = ComponentHookRegistry::new();
        registry.set_processing_mode(HookProcessingMode::Immediate);
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let o1 = order.clone();
        registry.on_insert_priority::<Health, _>("second", 10, move |_| { o1.lock().unwrap().push(2); });
        let o2 = order.clone();
        registry.on_insert_priority::<Health, _>("first", -10, move |_| { o2.lock().unwrap().push(1); });
        registry.notify_insert::<Health>(1);
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_unregister() {
        let mut registry = ComponentHookRegistry::new();
        registry.set_processing_mode(HookProcessingMode::Immediate);
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let id = registry.on_insert::<Health, _>("h", move |_| { c.fetch_add(1, Ordering::Relaxed); });
        registry.notify_insert::<Health>(1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        registry.unregister(id);
        registry.notify_insert::<Health>(2);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
