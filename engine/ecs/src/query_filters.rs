// engine/ecs/src/query_filters.rs
//
// Query filter system for the Genovo ECS.
//
// Provides composable filter combinators for ECS queries:
//
// - With<T>: entity must have component T.
// - Without<T>: entity must NOT have component T.
// - Changed<T>: component T was modified since last query.
// - Added<T>: component T was added since last query.
// - Or<A, B>: either filter A or filter B matches.
// - And<A, B>: both filter A and filter B must match.
// - Not<T>: negation of filter T.
// - Optional<T>: component T is included if present but not required.
// - Composable combinations for complex queries.

use std::collections::HashSet;
use std::any::TypeId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum filter depth for nested combinations.
const MAX_FILTER_DEPTH: u32 = 16;

/// Tick value for change detection.
type Tick = u64;

/// Default tick window for change detection.
const DEFAULT_CHANGE_WINDOW: u64 = 1;

// ---------------------------------------------------------------------------
// Component ID (simplified for this module)
// ---------------------------------------------------------------------------

/// Simplified component identifier used by the filter system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FilterComponentId(pub u64);

impl FilterComponentId {
    /// Create from a type ID.
    pub fn of<T: 'static>() -> Self {
        let type_id = TypeId::of::<T>();
        let mut hasher_val: u64 = 0;
        let bytes: [u8; std::mem::size_of::<TypeId>()] = unsafe {
            std::mem::transmute_copy(&type_id)
        };
        for (i, &b) in bytes.iter().enumerate() {
            hasher_val ^= (b as u64) << ((i % 8) * 8);
        }
        Self(hasher_val)
    }

    /// Create from a raw ID.
    pub fn raw(id: u64) -> Self {
        Self(id)
    }
}

// ---------------------------------------------------------------------------
// Entity Component Metadata
// ---------------------------------------------------------------------------

/// Metadata about an entity's components for filter evaluation.
#[derive(Debug, Clone)]
pub struct EntityComponentMeta {
    /// Entity ID.
    pub entity_id: u64,
    /// Set of component IDs present on this entity.
    pub components: HashSet<FilterComponentId>,
    /// Last modification tick per component.
    pub modification_ticks: std::collections::HashMap<FilterComponentId, Tick>,
    /// Addition tick per component (when it was added).
    pub addition_ticks: std::collections::HashMap<FilterComponentId, Tick>,
}

impl EntityComponentMeta {
    /// Create new metadata for an entity.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            components: HashSet::new(),
            modification_ticks: std::collections::HashMap::new(),
            addition_ticks: std::collections::HashMap::new(),
        }
    }

    /// Add a component.
    pub fn add_component(&mut self, component_id: FilterComponentId, tick: Tick) {
        self.components.insert(component_id);
        self.addition_ticks.insert(component_id, tick);
        self.modification_ticks.insert(component_id, tick);
    }

    /// Remove a component.
    pub fn remove_component(&mut self, component_id: FilterComponentId) {
        self.components.remove(&component_id);
        self.addition_ticks.remove(&component_id);
        self.modification_ticks.remove(&component_id);
    }

    /// Mark a component as modified.
    pub fn mark_modified(&mut self, component_id: FilterComponentId, tick: Tick) {
        self.modification_ticks.insert(component_id, tick);
    }

    /// Check if entity has a component.
    pub fn has(&self, component_id: FilterComponentId) -> bool {
        self.components.contains(&component_id)
    }

    /// Check if a component was modified since the given tick.
    pub fn modified_since(&self, component_id: FilterComponentId, since_tick: Tick) -> bool {
        self.modification_ticks.get(&component_id).map_or(false, |&t| t >= since_tick)
    }

    /// Check if a component was added since the given tick.
    pub fn added_since(&self, component_id: FilterComponentId, since_tick: Tick) -> bool {
        self.addition_ticks.get(&component_id).map_or(false, |&t| t >= since_tick)
    }
}

// ---------------------------------------------------------------------------
// Filter Trait
// ---------------------------------------------------------------------------

/// A filter that can be evaluated against entity component metadata.
pub trait QueryFilterTrait: std::fmt::Debug + Send + Sync {
    /// Evaluate this filter against entity metadata.
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool;

    /// Get the component IDs this filter requires (for archetype matching).
    fn required_components(&self) -> Vec<FilterComponentId>;

    /// Get the component IDs this filter excludes.
    fn excluded_components(&self) -> Vec<FilterComponentId>;

    /// Get a human-readable description of this filter.
    fn describe(&self) -> String;

    /// Clone the filter into a boxed trait object.
    fn clone_box(&self) -> Box<dyn QueryFilterTrait>;
}

// ---------------------------------------------------------------------------
// With<T> Filter
// ---------------------------------------------------------------------------

/// Filter: entity must have component with this ID.
#[derive(Debug, Clone)]
pub struct WithFilter {
    /// Component ID to require.
    pub component_id: FilterComponentId,
    /// Component name (for debugging).
    pub component_name: String,
}

impl WithFilter {
    /// Create a new With filter.
    pub fn new(component_id: FilterComponentId, name: &str) -> Self {
        Self {
            component_id,
            component_name: name.to_string(),
        }
    }
}

impl QueryFilterTrait for WithFilter {
    fn matches(&self, meta: &EntityComponentMeta, _current_tick: Tick) -> bool {
        meta.has(self.component_id)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        vec![self.component_id]
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        vec![]
    }

    fn describe(&self) -> String {
        format!("With<{}>", self.component_name)
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Without<T> Filter
// ---------------------------------------------------------------------------

/// Filter: entity must NOT have component with this ID.
#[derive(Debug, Clone)]
pub struct WithoutFilter {
    /// Component ID to exclude.
    pub component_id: FilterComponentId,
    /// Component name (for debugging).
    pub component_name: String,
}

impl WithoutFilter {
    /// Create a new Without filter.
    pub fn new(component_id: FilterComponentId, name: &str) -> Self {
        Self {
            component_id,
            component_name: name.to_string(),
        }
    }
}

impl QueryFilterTrait for WithoutFilter {
    fn matches(&self, meta: &EntityComponentMeta, _current_tick: Tick) -> bool {
        !meta.has(self.component_id)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        vec![]
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        vec![self.component_id]
    }

    fn describe(&self) -> String {
        format!("Without<{}>", self.component_name)
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Changed<T> Filter
// ---------------------------------------------------------------------------

/// Filter: component was modified since the last system tick.
#[derive(Debug, Clone)]
pub struct ChangedFilter {
    /// Component ID.
    pub component_id: FilterComponentId,
    /// Component name.
    pub component_name: String,
    /// Tick window: how many ticks back to consider as "changed".
    pub window: u64,
}

impl ChangedFilter {
    /// Create a new Changed filter.
    pub fn new(component_id: FilterComponentId, name: &str) -> Self {
        Self {
            component_id,
            component_name: name.to_string(),
            window: DEFAULT_CHANGE_WINDOW,
        }
    }

    /// Create with a custom window.
    pub fn with_window(mut self, window: u64) -> Self {
        self.window = window;
        self
    }
}

impl QueryFilterTrait for ChangedFilter {
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        if !meta.has(self.component_id) {
            return false;
        }
        let since = current_tick.saturating_sub(self.window);
        meta.modified_since(self.component_id, since)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        vec![self.component_id]
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        vec![]
    }

    fn describe(&self) -> String {
        format!("Changed<{}>", self.component_name)
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Added<T> Filter
// ---------------------------------------------------------------------------

/// Filter: component was added since the last system tick.
#[derive(Debug, Clone)]
pub struct AddedFilter {
    /// Component ID.
    pub component_id: FilterComponentId,
    /// Component name.
    pub component_name: String,
    /// Tick window.
    pub window: u64,
}

impl AddedFilter {
    /// Create a new Added filter.
    pub fn new(component_id: FilterComponentId, name: &str) -> Self {
        Self {
            component_id,
            component_name: name.to_string(),
            window: DEFAULT_CHANGE_WINDOW,
        }
    }
}

impl QueryFilterTrait for AddedFilter {
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        let since = current_tick.saturating_sub(self.window);
        meta.added_since(self.component_id, since)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        vec![self.component_id]
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        vec![]
    }

    fn describe(&self) -> String {
        format!("Added<{}>", self.component_name)
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Or<A, B> Filter
// ---------------------------------------------------------------------------

/// Filter: either A or B matches.
#[derive(Debug)]
pub struct OrFilter {
    pub a: Box<dyn QueryFilterTrait>,
    pub b: Box<dyn QueryFilterTrait>,
}

impl OrFilter {
    /// Create a new Or filter.
    pub fn new(a: Box<dyn QueryFilterTrait>, b: Box<dyn QueryFilterTrait>) -> Self {
        Self { a, b }
    }
}

impl QueryFilterTrait for OrFilter {
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        self.a.matches(meta, current_tick) || self.b.matches(meta, current_tick)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        // Or doesn't strictly require either set.
        vec![]
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        // Only exclude if both filters exclude the same component.
        let a_excluded: HashSet<FilterComponentId> = self.a.excluded_components().into_iter().collect();
        let b_excluded: HashSet<FilterComponentId> = self.b.excluded_components().into_iter().collect();
        a_excluded.intersection(&b_excluded).copied().collect()
    }

    fn describe(&self) -> String {
        format!("Or({}, {})", self.a.describe(), self.b.describe())
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(OrFilter {
            a: self.a.clone_box(),
            b: self.b.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// And<A, B> Filter
// ---------------------------------------------------------------------------

/// Filter: both A and B must match.
#[derive(Debug)]
pub struct AndFilter {
    pub a: Box<dyn QueryFilterTrait>,
    pub b: Box<dyn QueryFilterTrait>,
}

impl AndFilter {
    /// Create a new And filter.
    pub fn new(a: Box<dyn QueryFilterTrait>, b: Box<dyn QueryFilterTrait>) -> Self {
        Self { a, b }
    }
}

impl QueryFilterTrait for AndFilter {
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        self.a.matches(meta, current_tick) && self.b.matches(meta, current_tick)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        let mut required = self.a.required_components();
        required.extend(self.b.required_components());
        required
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        let mut excluded = self.a.excluded_components();
        excluded.extend(self.b.excluded_components());
        excluded
    }

    fn describe(&self) -> String {
        format!("And({}, {})", self.a.describe(), self.b.describe())
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(AndFilter {
            a: self.a.clone_box(),
            b: self.b.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// Not<T> Filter
// ---------------------------------------------------------------------------

/// Filter: negation of inner filter.
#[derive(Debug)]
pub struct NotFilter {
    pub inner: Box<dyn QueryFilterTrait>,
}

impl NotFilter {
    /// Create a new Not filter.
    pub fn new(inner: Box<dyn QueryFilterTrait>) -> Self {
        Self { inner }
    }
}

impl QueryFilterTrait for NotFilter {
    fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        !self.inner.matches(meta, current_tick)
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        self.inner.excluded_components()
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        self.inner.required_components()
    }

    fn describe(&self) -> String {
        format!("Not({})", self.inner.describe())
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(NotFilter {
            inner: self.inner.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// Optional<T> Filter
// ---------------------------------------------------------------------------

/// Filter: always matches, but marks a component as optionally included.
#[derive(Debug, Clone)]
pub struct OptionalFilter {
    /// Component ID that is optional.
    pub component_id: FilterComponentId,
    /// Component name.
    pub component_name: String,
}

impl OptionalFilter {
    /// Create a new Optional filter.
    pub fn new(component_id: FilterComponentId, name: &str) -> Self {
        Self {
            component_id,
            component_name: name.to_string(),
        }
    }
}

impl QueryFilterTrait for OptionalFilter {
    fn matches(&self, _meta: &EntityComponentMeta, _current_tick: Tick) -> bool {
        // Optional always matches; the component may or may not be present.
        true
    }

    fn required_components(&self) -> Vec<FilterComponentId> {
        vec![] // Not required.
    }

    fn excluded_components(&self) -> Vec<FilterComponentId> {
        vec![]
    }

    fn describe(&self) -> String {
        format!("Optional<{}>", self.component_name)
    }

    fn clone_box(&self) -> Box<dyn QueryFilterTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Composite Filter Builder
// ---------------------------------------------------------------------------

/// Builder for constructing complex composite filters.
#[derive(Debug)]
pub struct FilterBuilder {
    /// Accumulated filters.
    filters: Vec<Box<dyn QueryFilterTrait>>,
}

impl FilterBuilder {
    /// Create a new filter builder.
    pub fn new() -> Self {
        Self { filters: Vec::new() }
    }

    /// Add a With filter.
    pub fn with(mut self, component_id: FilterComponentId, name: &str) -> Self {
        self.filters.push(Box::new(WithFilter::new(component_id, name)));
        self
    }

    /// Add a Without filter.
    pub fn without(mut self, component_id: FilterComponentId, name: &str) -> Self {
        self.filters.push(Box::new(WithoutFilter::new(component_id, name)));
        self
    }

    /// Add a Changed filter.
    pub fn changed(mut self, component_id: FilterComponentId, name: &str) -> Self {
        self.filters.push(Box::new(ChangedFilter::new(component_id, name)));
        self
    }

    /// Add an Added filter.
    pub fn added(mut self, component_id: FilterComponentId, name: &str) -> Self {
        self.filters.push(Box::new(AddedFilter::new(component_id, name)));
        self
    }

    /// Add an Optional filter.
    pub fn optional(mut self, component_id: FilterComponentId, name: &str) -> Self {
        self.filters.push(Box::new(OptionalFilter::new(component_id, name)));
        self
    }

    /// Build the composite filter (AND of all added filters).
    pub fn build(self) -> CompositeFilter {
        CompositeFilter { filters: self.filters }
    }
}

/// A composite filter that ANDs all contained filters.
#[derive(Debug)]
pub struct CompositeFilter {
    pub filters: Vec<Box<dyn QueryFilterTrait>>,
}

impl CompositeFilter {
    /// Evaluate the composite filter.
    pub fn matches(&self, meta: &EntityComponentMeta, current_tick: Tick) -> bool {
        self.filters.iter().all(|f| f.matches(meta, current_tick))
    }

    /// Get all required components.
    pub fn required_components(&self) -> Vec<FilterComponentId> {
        let mut required = Vec::new();
        for f in &self.filters {
            required.extend(f.required_components());
        }
        required.sort();
        required.dedup();
        required
    }

    /// Get all excluded components.
    pub fn excluded_components(&self) -> Vec<FilterComponentId> {
        let mut excluded = Vec::new();
        for f in &self.filters {
            excluded.extend(f.excluded_components());
        }
        excluded.sort();
        excluded.dedup();
        excluded
    }

    /// Get a description of the filter.
    pub fn describe(&self) -> String {
        let descriptions: Vec<String> = self.filters.iter().map(|f| f.describe()).collect();
        descriptions.join(" && ")
    }

    /// Number of sub-filters.
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }
}

// ---------------------------------------------------------------------------
// Query Filter Cache
// ---------------------------------------------------------------------------

/// Cache for precomputed filter results.
#[derive(Debug)]
pub struct FilterCache {
    /// Cached matching entity sets per filter hash.
    pub cache: std::collections::HashMap<u64, Vec<u64>>,
    /// Cache validity tick.
    pub valid_tick: Tick,
    /// Maximum cache entries.
    pub max_entries: usize,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
}

impl FilterCache {
    /// Create a new filter cache.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            valid_tick: 0,
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    /// Invalidate the cache.
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    /// Check if there is a cached result for a filter description.
    pub fn get(&mut self, filter_hash: u64, current_tick: Tick) -> Option<&Vec<u64>> {
        if current_tick > self.valid_tick {
            self.invalidate();
            self.valid_tick = current_tick;
            self.misses += 1;
            return None;
        }
        if let Some(result) = self.cache.get(&filter_hash) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a result in the cache.
    pub fn store(&mut self, filter_hash: u64, entities: Vec<u64>) {
        if self.cache.len() >= self.max_entries {
            // Simple eviction: clear the entire cache.
            self.cache.clear();
        }
        self.cache.insert(filter_hash, entities);
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f32 / total as f32
    }
}
